# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Bounded content-free telemetry and explicit incomplete usage accounting."""

import json
import math
import queue
import threading
import time
from collections import Counter
from pathlib import Path

OUTCOMES = {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}
REJECTIONS = {
    "capacity_exhausted",
    "session_busy",
    "workspace_busy",
    "admission_closed",
    "idempotency_conflict",
}
BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 300, 3600)


def normalize_usage(event, model, state, rates=None):
    raw = event.get("usage") or {}
    if not isinstance(raw, dict):
        raw = {}
    # Worker `tokens` means measured output tokens, never input or total tokens.
    values = {
        "input_tokens": raw.get("input_tokens"),
        "output_tokens": raw.get("output_tokens", raw.get("tokens")),
        "cache_tokens": raw.get("cache_tokens"),
    }
    values = {
        key: value if type(value) is int and 0 <= value < 10**12 else None
        for key, value in values.items()
    }
    known = any(value is not None for value in values.values())
    complete = (
        values["input_tokens"] is not None
        and values["output_tokens"] is not None
        and state == "succeeded"
    )
    result = {
        **values,
        "status": "reported" if complete else "incomplete" if known else "unavailable",
        "model": model,
        "source": "gaia_worker",
        "basis": "run_aggregate",
        "estimated_cost": None,
    }
    if rates is not None and complete:
        rate = rates["models"].get(model)
        if rate is not None:
            result["estimated_cost"] = {
                "status": "estimated",
                "currency": "USD",
                "rate_version": rates["version"],
                "basis": "reported_input_output_tokens_without_cache_discount",
                "amount": (
                    values["input_tokens"] * rate["input_per_million"]
                    + values["output_tokens"] * rate["output_per_million"]
                )
                / 1000000,
            }
    return result


def validate_rates(rates):
    if rates is None:
        return None
    if (
        not isinstance(rates.get("version"), str)
        or not 1 <= len(rates["version"]) <= 100
        or not isinstance(rates.get("models"), dict)
    ):
        raise ValueError("Rate table requires a version and model mapping")
    if len(rates["models"]) > 100:
        raise ValueError("Rate table exceeds model limit")
    for model, rate in rates["models"].items():
        if not isinstance(model, str) or not model or len(model) > 200:
            raise ValueError("Invalid rate model")
        for key in ("input_per_million", "output_per_million"):
            value = rate[key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1000000
            ):
                raise ValueError("Invalid USD per-million token rate")
    return rates


class Telemetry:
    """Fixed labels/buckets; optional bounded local JSONL export never gates runs."""

    def __init__(self, trace_directory=None):
        self.lock = threading.Lock()
        self.counts = Counter()
        self.durations = {
            kind: [0] * (len(BUCKETS) + 1)
            for kind in ("request", "startup", "run", "tool")
        }
        self.queue = queue.Queue(maxsize=128)
        self.done = threading.Event()
        self.thread = None
        self.root = Path(trace_directory) if trace_directory else None
        if self.root:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.root.stat().st_mode & 0o077:
                raise ValueError("Trace directory must be private")
            self.thread = threading.Thread(target=self._export, daemon=True)
            self.thread.start()

    def increment(self, kind, value=""):
        allowed = {
            "outcome": OUTCOMES,
            "rejection": REJECTIONS | {"other"},
            "usage": {"reported", "estimated", "incomplete", "unavailable"},
            "telemetry_dropped": {""},
            "telemetry_failures": {""},
            "reconciliation_failures": {""},
        }
        if kind not in allowed or value not in allowed[kind]:
            raise ValueError("Unbounded metric label refused")
        with self.lock:
            self.counts[(kind, value)] += 1

    def duration(self, kind, seconds):
        if kind not in self.durations or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid duration")
        with self.lock:
            index = next(
                (i for i, bound in enumerate(BUCKETS) if seconds <= bound), len(BUCKETS)
            )
            self.durations[kind][index] += 1

    def terminal(self, run):
        self.increment("outcome", run["state"])
        self.increment("usage", run["usage"]["status"])
        self.duration("run", max(0, run["ended_at"] - run["created_at"]))
        if self.root:
            record = {
                key: run[key]
                for key in (
                    "id",
                    "generation",
                    "state",
                    "execution_status",
                    "outcome_uncertain",
                )
            }
            record["timestamp"] = time.time()
            try:
                self.queue.put_nowait(record)
            except queue.Full:
                self.increment("telemetry_dropped")

    def _export(self):
        while not self.done.is_set() or not self.queue.empty():
            try:
                record = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                path = self.root / "events.jsonl"
                if path.exists() and path.stat().st_size >= 8 * 1024 * 1024:
                    path.replace(self.root / "events.previous.jsonl")
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            except OSError:
                self.increment("telemetry_failures")
                self.increment("telemetry_dropped")
            finally:
                self.queue.task_done()

    def snapshot(self):
        with self.lock:
            return {
                "counters": [
                    {"name": kind, "label": label, "value": value}
                    for (kind, label), value in sorted(self.counts.items())
                ],
                "duration_seconds": {
                    kind: {"bounds": list(BUCKETS) + [None], "counts": list(values)}
                    for kind, values in self.durations.items()
                },
                "trace_queue_items": self.queue.qsize(),
                "trace_export": self.root is not None,
            }

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.increment("telemetry_dropped")


class InferenceProbe:
    """Cached bounded model discovery, independent of request pressure."""

    def __init__(self, url, model, token_file=None):
        from urllib.parse import urlsplit

        import requests

        self.lock = threading.Lock()
        self.worker = None
        self.url, self.model = url, model
        self.token_file = Path(token_file) if token_file else None
        self.session = requests.Session()
        self.session.trust_env = False
        self.state = {"ready": False, "code": "inference_probe_unconfigured"}
        if url:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Use a credential-free inference probe URL")
            if parsed.scheme == "http" and parsed.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ValueError(
                    "Inference probe credentials require HTTPS outside loopback"
                )

    def check(self):
        if not self.url:
            return self.state
        with self.lock:
            if self.worker is not None and self.worker.is_alive():
                self.state = {"ready": False, "code": "inference_probe_timeout"}
                return self.state
            result = []
            self.worker = threading.Thread(
                target=lambda: result.append(self._perform()), daemon=True
            )
            self.worker.start()
            self.worker.join(timeout=3)
            self.state = (
                result[0]
                if result
                else {"ready": False, "code": "inference_probe_timeout"}
            )
            return self.state

    def _perform(self):
        import requests

        if not self.url:
            return self.state
        try:
            headers = {}
            if self.token_file:
                with self.token_file.open("r", encoding="utf-8") as handle:
                    token = handle.read(8193).strip()
                if not token or len(token) > 8192 or any(c.isspace() for c in token):
                    raise ValueError("Invalid probe token")
                headers["Authorization"] = "Bearer " + token
            with self.session.get(
                self.url.rstrip("/") + "/models",
                headers=headers,
                timeout=(2, 2),
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    return {"ready": False, "code": "inference_unavailable"}
                raw = bytearray()
                for chunk in response.iter_content(8192):
                    if len(raw) + len(chunk) > 1048576:
                        raise ValueError("Model catalog exceeded bound")
                    raw.extend(chunk)
                catalog = json.loads(raw)
                ready = any(row.get("id") == self.model for row in catalog["data"])
                result = {
                    "ready": ready,
                    "code": "ready" if ready else "model_not_advertised",
                }
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            requests.RequestException,
        ):
            result = {"ready": False, "code": "inference_unavailable"}
        return result

    def close(self):
        self.session.close()
