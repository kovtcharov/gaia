# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Durable orchestration over the existing GAIA worker and guardian contracts."""

import ipaddress
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from gaia.logger import get_logger

from .operations import REJECTIONS, InferenceProbe, Telemetry, validate_rates
from .store import Store, StoreError

LOGGER = get_logger(__name__)


@dataclass
class LiveRun:
    run: dict
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    endpoint: dict | None = None
    interactions: dict = field(default_factory=dict)
    secrets: list = field(default_factory=list)
    thread: threading.Thread | None = None
    epoch: str | None = None
    sensitive_input: bool = False


class Controller:
    def __init__(
        self,
        directory,
        guardian,
        secrets_to_redact=(),
        trace_directory=None,
        rates=None,
        content_days=7,
        metadata_days=30,
        inference_probe_url=None,
        inference_probe_token_file=None,
    ):
        if (
            type(content_days) is not int
            or not 0 <= content_days <= 365
            or type(metadata_days) is not int
            or not max(1, content_days) <= metadata_days <= 365
        ):
            raise ValueError("Invalid retention policy")
        self.content_days, self.metadata_days = content_days, metadata_days
        self.rates = validate_rates(rates)
        self.telemetry = Telemetry(trace_directory)
        self.guardian = guardian
        self.capabilities = guardian.call("capabilities")
        if self.capabilities["protocol"] < 2:
            raise StoreError("guardian_fencing_required", 503)
        self.store = Store(
            directory,
            self.capabilities["deployment"],
            self.capabilities["workspaces"],
            self.capabilities["slots"],
        )
        self.probe = InferenceProbe(
            inference_probe_url, self.capabilities["model"], inference_probe_token_file
        )
        self.probe.check()
        self.guardian_ready = True
        self.epoch = str(uuid4())
        self.healthy = False
        self.draining = False
        self.drain_lock = threading.Lock()
        self.drain_thread = None
        self.drain_errors = []
        self.store_close_thread = None
        self.store_close_error = False
        self.lock = threading.RLock()
        self.live = {}
        self.global_secrets = tuple(secrets_to_redact)
        # Fence delayed requests before interpreting a missing guardian run as never started.
        try:
            self._recover()
        except BaseException:
            self.telemetry.close()
            self.store.close()
            raise
        self.maintenance_done = threading.Event()
        self.maintenance_thread = threading.Thread(target=self._maintain, daemon=True)
        self.maintenance_thread.start()

    def _recover(self):
        self.guardian.call("fence", controller_epoch=self.epoch)
        for row in self.store.active():
            stopped = self.guardian.call(
                "cancel", run=row["id"], generation=row["generation"]
            )
            self.store.finish(
                row["id"],
                "interrupted",
                {
                    "type": "error",
                    "detail": "Controller restarted; external outcome may be uncertain.",
                    "status": 503,
                },
                stopped["state"] == "stopped",
                uncertain=row["execution_status"] != "not_started",
                rotate_generation=True,
            )
        self.store.maintain(
            content_days=self.content_days, metadata_days=self.metadata_days
        )
        self.healthy = True

    def _maintain(self):
        next_retention = time.monotonic() + 60
        while not self.maintenance_done.wait(5):
            self.probe.check()
            try:
                self.guardian_ready = self.guardian.call("health")["healthy"]
            except Exception:
                self.guardian_ready = False
                self.telemetry.increment("reconciliation_failures")
            if self.maintenance_done.is_set():
                return
            if time.monotonic() >= next_retention:
                try:
                    self.store.maintain(
                        content_days=self.content_days, metadata_days=self.metadata_days
                    )
                except Exception:
                    self.storage_failed()
                    LOGGER.error("Retention failed; admission is closed")
                    return
                next_retention = time.monotonic() + 60

    def storage_failed(self):
        # A fresh epoch also rejects delayed start requests from this controller.
        self.healthy = False
        for live in self.live.copy().values():
            live.done.set()
        try:
            self.guardian.call("fence", controller_epoch=str(uuid4()))
        except Exception:
            LOGGER.error("Storage failed and executor termination is unconfirmed")

    def submit(
        self, session_id, key, prompt, max_steps=20, acknowledge_history_loss=False
    ):
        # Replay must remain available even while the service is busy or draining.
        with self.lock:
            if (
                not self.healthy
                or self.draining
                or not self.guardian_ready
                or (self.probe.url and not self.probe.state["ready"])
            ):
                with self.store.lock:
                    existing = self.store.db.execute(
                        "SELECT id FROM runs WHERE key=? UNION ALL SELECT run_id AS id FROM tombstones WHERE key=?",
                        (key, key),
                    ).fetchone()
                if not existing:
                    raise StoreError("admission_closed", 503)
            try:
                run, created = self.store.admit(
                    session_id,
                    key,
                    prompt,
                    max_steps,
                    self.capabilities,
                    acknowledge_history_loss,
                )
            except StoreError as exc:
                self.telemetry.increment(
                    "rejection", exc.code if exc.code in REJECTIONS else "other"
                )
                raise
            if created:
                live = LiveRun(run, epoch=self.epoch)
                live.secrets.extend(self.global_secrets)
                self.live[run["id"]] = live
                live.thread = threading.Thread(
                    target=self._execute, args=(live,), daemon=True
                )
                try:
                    live.thread.start()
                except Exception:
                    self.cancel(run["id"])
                    raise
            return run, created

    @staticmethod
    def _session():
        session = requests.Session()
        session.trust_env = False
        return session

    def _worker(self, live, method, path, **kwargs):
        if live.endpoint is None:
            raise StoreError("executor_not_ready", 409)
        parsed = urlsplit(live.endpoint["url"])
        internal = live.endpoint.get("network_internal", False)
        allowed = parsed.hostname == "127.0.0.1"
        if internal:
            address = ipaddress.ip_address(parsed.hostname)
            allowed = (
                address.version == 4 and address.is_private and not address.is_loopback
            )
        if parsed.scheme != "http" or not allowed or parsed.username:
            raise StoreError("invalid_executor_endpoint", 503)
        session = self._session()
        try:
            response = session.request(
                method,
                live.endpoint["url"] + path,
                headers={
                    "Authorization": "Bearer " + live.endpoint["token"],
                    "Host": "127.0.0.1",
                },
                allow_redirects=False,
                timeout=(3, 15),
                **kwargs,
            )
            if response.status_code != 200:
                response.close()
                raise StoreError("executor_request_failed", 503)
            return session, response
        except BaseException:
            session.close()
            raise

    def _heartbeat(self, live):
        interval = max(0.1, self.capabilities["lease"] / 3)
        while not live.done.wait(interval):
            if not self.healthy or self.draining:
                return
            try:
                self.guardian.call(
                    "renew", run=live.run["id"], generation=live.run["generation"]
                )
            except Exception:
                self.healthy = False
                LOGGER.error("Guardian renewal failed; durable admission is closed")
                return

    def _redact(self, live, value):
        with live.lock:
            values = tuple(live.secrets)

        def scrub(item):
            if isinstance(item, str):
                for secret in values:
                    if secret:
                        item = item.replace(secret, "[redacted]")
                return item
            if isinstance(item, list):
                return [scrub(child) for child in item]
            if isinstance(item, dict):
                return {scrub(key): scrub(child) for key, child in item.items()}
            return item

        return scrub(value)

    def _event(self, live, event):
        kind = event.get("type")
        if kind in {"needs_input", "needs_confirmation"}:
            worker_id = event.get(
                "confirm_id" if kind == "needs_confirmation" else "request_id"
            )
            if not isinstance(worker_id, str) or not worker_id:
                raise StoreError("invalid_worker_interaction", 503)
            expires = min(
                time.time() + float(event.get("timeout_seconds", 300)),
                live.endpoint["deadline"],
            )
            with live.lock:
                row = self.store.create_interaction(
                    live.run["id"], kind, event, expires
                )
                live.interactions[row["id"]] = event
            return None
        sanitized = self._redact(live, event)
        if kind in {"final", "error"}:
            return sanitized
        with live.lock:
            if live.sensitive_input:
                sanitized = {"type": kind, "content_omitted": "sensitive_interaction"}
        self.store.append(live.run["id"], sanitized)
        return None

    def _execute(self, live):
        run = live.run
        result = {
            "type": "error",
            "detail": "Executor ended without a terminal result.",
            "status": 503,
        }
        state, uncertain = "failed", False
        heartbeat = None
        try:
            if not self.healthy or self.draining:
                raise StoreError("admission_closed", 503)
            if not self.store.dispatching(run["id"]):
                return
            live.endpoint = self.guardian.call(
                "start",
                run=run["id"],
                generation=run["generation"],
                workspace=run["workspace_id"],
                controller_epoch=live.epoch,
            )
            live.secrets.append(live.endpoint["token"])
            heartbeat = threading.Thread(
                target=self._heartbeat, args=(live,), daemon=True
            )
            heartbeat.start()
            self.store.started(run["id"])
            self.telemetry.duration("startup", max(0, time.time() - run["created_at"]))
            if self.store.run(run["id"])["state"] == "stopping":
                state = "cancelled"
                return
            expires = time.monotonic() + 60
            while True:
                try:
                    session, response = self._worker(live, "GET", "/ready")
                    response.close()
                    session.close()
                    break
                except (requests.RequestException, StoreError):
                    if time.monotonic() >= expires or live.done.wait(0.1):
                        raise StoreError("executor_startup_timeout", 503) from None
            context = self.store.context(run["session_id"], run["id"])
            payload = {
                "run_id": run["id"],
                "session_id": run["session_id"],
                "query": run["request"]["prompt"],
                "context": context,
                "max_steps": run["request"]["max_steps"],
                "can_answer_questions": True,
                "can_confirm_tools": True,
            }
            uncertain = True
            session, response = self._worker(
                live, "POST", "/v1/gaia/query", json=payload, stream=True
            )
            try:
                buffer, total, terminal = b"", 0, None
                for chunk in response.iter_content(chunk_size=8192):
                    total += len(chunk)
                    if total > 4 * 1024 * 1024 + 262144:
                        raise StoreError("executor_output_limit", 413)
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if len(line) > 262144:
                            raise StoreError("executor_event_limit", 413)
                        if line.startswith(b"data: "):
                            event = json.loads(line[6:])
                            if not isinstance(event, dict) or not isinstance(
                                event.get("type"), str
                            ):
                                raise StoreError("invalid_executor_event", 503)
                            terminal = self._event(live, event)
                            if terminal is not None:
                                break
                    if terminal is not None:
                        result = terminal
                        state = "succeeded" if terminal["type"] == "final" else "failed"
                        if terminal.get("status") == 504:
                            state = "timed_out"
                        break
                    if len(buffer) > 262144:
                        raise StoreError("executor_event_limit", 413)
            finally:
                response.close()
                session.close()
        except Exception:
            # Failure details may contain model/tool content; persist only a stable code.
            LOGGER.error("Durable execution failed; stopping its owned executor")
        finally:
            live.done.set()
            if heartbeat is not None:
                heartbeat.join(timeout=1)
            try:
                stopped = self.guardian.call(
                    "cancel", run=run["id"], generation=run["generation"]
                )
                terminal = self.store.finish(
                    run["id"],
                    state,
                    result,
                    stopped["state"] == "stopped",
                    uncertain=uncertain and state != "succeeded",
                    rates=self.rates,
                )
                self.telemetry.terminal(terminal)
                LOGGER.info(
                    "%s",
                    json.dumps(
                        {
                            "event": "run_terminal",
                            "run": run["id"],
                            "generation": run["generation"],
                            "state": terminal["state"],
                        }
                    ),
                )
            except Exception:
                self.healthy = False
                LOGGER.error("Durable termination is unconfirmed; admission is closed")
            with self.lock:
                self.live.pop(run["id"], None)

    def cancel(self, run_id):
        live = self.live.get(run_id)
        if live:
            live.done.set()
        try:
            run = self.store.stopping(run_id)
        except sqlite3.Error:
            self.storage_failed()
            # RAM identity remains usable when the database cannot even be read.
            if live:
                try:
                    self.guardian.call(
                        "cancel", run=run_id, generation=live.run["generation"]
                    )
                except Exception:
                    LOGGER.error("Cancellation termination is unconfirmed")
            raise StoreError("storage_unavailable", 503) from None
        if run["ended_at"] is not None:
            return run
        try:
            stopped = self.guardian.call(
                "cancel", run=run_id, generation=run["generation"]
            )
            return self.store.finish(
                run_id,
                "cancelled",
                {"type": "error", "detail": "Run cancelled.", "status": 499},
                stopped["state"] == "stopped",
                uncertain=run["execution_status"] != "not_started",
            )
        except Exception:
            self.healthy = False
            raise StoreError("termination_unconfirmed", 503) from None

    def pending_interaction(self, run_id):
        row = self.store.interaction(run_id)
        live = self.live.get(run_id)
        if live is None:
            raise StoreError("interaction_unavailable")
        with live.lock:
            raw = live.interactions.get(row["id"])
            if raw is None:
                raise StoreError("interaction_unavailable")
            return {
                "id": row["id"],
                "generation": row["generation"],
                "expires_at": row["expires_at"],
                "argument_digest": row["digest"],
                "request": raw,
            }

    def answer(
        self,
        run_id,
        interaction_id,
        generation,
        decision,
        response=None,
        submission_id=None,
    ):
        current = self.store.interaction(run_id, interaction_id)
        if (
            current["state"] == "pending"
            and current["kind"] == "needs_input"
            and (not isinstance(response, str) or len(response.encode()) > 4096)
        ):
            raise StoreError("invalid_answer", 422)
        try:
            row, deliver = self.store.receipt(
                run_id, interaction_id, generation, decision, submission_id
            )
        except sqlite3.Error:
            self.storage_failed()
            raise StoreError("storage_unavailable", 503) from None
        if not deliver:
            return {"accepted": True, "delivery": row["delivery"], "replayed": True}
        live = self.live.get(run_id)
        if live is None:
            self.cancel(run_id)
            raise StoreError("interaction_delivery_unknown", 503)
        try:
            with live.lock:
                raw = live.interactions.pop(interaction_id)
                if row["kind"] == "needs_confirmation":
                    path, payload = "confirm", {
                        "confirm_id": raw["confirm_id"],
                        "approved": decision == "approve",
                    }
                else:
                    if not isinstance(response, str) or len(response.encode()) > 4096:
                        raise StoreError("invalid_answer", 422)
                    if raw.get("sensitive"):
                        live.secrets.append(response)
                        live.sensitive_input = True
                    path, payload = "respond", {
                        "request_id": raw["request_id"],
                        "response": response,
                    }
            session, reply = self._worker(
                live, "POST", f"/v1/gaia/query/{run_id}/{path}", json=payload
            )
            try:
                if reply.json().get("delivered") is not True:
                    raise StoreError("interaction_delivery_unknown", 503)
            finally:
                reply.close()
                session.close()
            self.store.delivered(interaction_id)
            return {"accepted": True, "delivery": "acknowledged", "replayed": False}
        except Exception:
            self.cancel(run_id)
            raise StoreError("interaction_delivery_unknown", 503) from None

    def _drain_runs(self):
        try:
            self.guardian.call("fence", controller_epoch=str(uuid4()))
            for run in self.store.active():
                try:
                    self.cancel(run["id"])
                except Exception:
                    self.drain_errors.append("termination_unconfirmed")
        except Exception:
            self.healthy = False
            self.drain_errors.append("drain_unconfirmed")
            LOGGER.error("Drain could not confirm execution termination")

    def drain(self):
        self.draining = True
        with self.drain_lock:
            for live in self.live.copy().values():
                live.done.set()
            if self.drain_thread is None:
                self.drain_thread = threading.Thread(
                    target=self._drain_runs, daemon=True
                )
                self.drain_thread.start()
            return {
                "draining": True,
                "complete": not self.drain_thread.is_alive(),
                "errors": list(self.drain_errors),
            }

    def _close_store(self):
        try:
            self.store.close()
        except Exception:
            self.store_close_error = True
            LOGGER.error("Storage close failed; guardian owns recovery")

    def close(self, timeout=25):
        deadline = time.monotonic() + timeout
        self.maintenance_done.set()
        self.drain()
        self.maintenance_thread.join(timeout=max(0, deadline - time.monotonic()))
        self.drain_thread.join(timeout=max(0, deadline - time.monotonic()))
        pending = list(self.live.copy().values())
        for live in pending:
            live.done.set()
            live.thread.join(timeout=max(0, deadline - time.monotonic()))
        self.telemetry.close()
        self.probe.close()
        if (
            self.maintenance_thread.is_alive()
            or self.drain_thread.is_alive()
            or self.drain_errors
            or any(live.thread.is_alive() for live in pending)
        ):
            raise StoreError("drain_incomplete_guardian_owns_recovery", 503)
        if self.store_close_thread is None:
            self.store_close_thread = threading.Thread(
                target=self._close_store, daemon=True
            )
            self.store_close_thread.start()
        self.store_close_thread.join(timeout=max(0, deadline - time.monotonic()))
        if self.store_close_thread.is_alive() or self.store_close_error:
            raise StoreError("storage_close_incomplete_guardian_owns_recovery", 503)
