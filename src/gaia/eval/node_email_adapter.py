# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Drive the Node email-triage SDK from the committed Python eval harness.

The harness (``gaia.eval.benchmark.run_benchmark``) scores whatever object its
``agent_factory`` returns, and the only thing it asks of that object is
``process_query(prompt) -> {"conversation": [...]}``. :class:`NodeEmailAgent`
satisfies that contract by shelling out to
``hub/agents/email/node/eval/driver.js`` and reshaping its JSON into the
base-Agent transcript the harness already knows how to read.

Why a subprocess and not an HTTP call: the Node SDK is an in-process library
with no server. Adding one purely to score it would cost more than it saves.

**No scoring lives here.** Ground truth, confusion matrices, gate thresholds and
the scorecard are all upstream of this file and are shared verbatim with the
Python agent's run — that shared scoring is the whole point of the adapter.

Token accounting (deliberate): the per-call ``/stats`` readings this adapter
emits as ``stats`` conversation entries already carry every call's token counts,
so the triage envelope purposely omits ``data.usage``. Populating both would
double-count, since ``benchmark.build_result`` *adds* ``data.usage`` on top of
the step totals. ``tokens_per_triage`` is therefore not reported for a Node run
— the driver's ``llm_call_count`` is surfaced on the envelope instead.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
from email.utils import parseaddr
from pathlib import Path
from typing import Any, Callable, Iterable

from gaia.eval.fixture_paths import resolve_repo_path

# Where the SDK lives in the repo. Overridable per call for a checkout in a
# non-standard place (or a packaged copy of the SDK).
_NODE_DIR_PARTS = ("hub", "agents", "email", "node")

# The driver script, relative to the SDK directory.
_DRIVER_RELPATH = ("eval", "driver.js")


def default_node_dir() -> Path:
    """Repo path of the Node email SDK. Fails loud when the checkout is absent."""
    return resolve_repo_path(*_NODE_DIR_PARTS, must_be_dir=True)


# ---------------------------------------------------------------------------
# Gmail payload -> driver item (pure, offline-testable)
# ---------------------------------------------------------------------------


def _headers(payload: dict) -> dict[str, str]:
    """Case-folded header map from a Gmail ``payload`` dict."""
    out: dict[str, str] = {}
    for header in payload.get("headers") or []:
        name = str(header.get("name", "")).strip().lower()
        if name and name not in out:
            out[name] = str(header.get("value", ""))
    return out


def _decode_part_body(part: dict) -> str:
    """Decode one Gmail part's base64url body to text."""
    data = (part.get("body") or {}).get("data")
    if not data:
        return ""
    padded = data + "=" * (-len(data) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Gmail part body is not valid base64url: {exc}. "
            "The corpus fixture is malformed — regenerate it with "
            "`python tests/fixtures/email/generate_mbox.py`."
        ) from exc
    return raw.decode("utf-8", errors="replace")


def extract_body_text(payload: dict) -> str:
    """Best text rendering of a Gmail message payload.

    Prefers ``text/plain``; falls back to ``text/html`` only when no plain part
    exists (the committed corpus is single-part text/plain, so the fallback is
    for robustness, not the normal path).
    """
    plain: list[str] = []
    html: list[str] = []

    def walk(part: dict) -> None:
        mime = str(part.get("mimeType", ""))
        subparts = part.get("parts") or []
        if subparts:
            for sub in subparts:
                walk(sub)
            return
        if mime.startswith("text/plain"):
            plain.append(_decode_part_body(part))
        elif mime.startswith("text/html"):
            html.append(_decode_part_body(part))
        elif not mime and not plain:
            plain.append(_decode_part_body(part))

    walk(payload)
    chosen = plain or html
    return "\n".join(t for t in chosen if t).strip()


def _address(raw: str) -> dict[str, str]:
    """Parse one RFC-5322 address into the SDK's ``EmailAddress`` shape."""
    name, addr = parseaddr(raw or "")
    out: dict[str, str] = {"email": addr or (raw or "").strip()}
    if name:
        out["name"] = name
    return out


def _address_list(raw: str) -> list[dict[str, str]]:
    if not raw:
        return []
    return [_address(chunk) for chunk in raw.split(",") if chunk.strip()]


def gmail_message_to_driver_item(msg: dict) -> dict[str, Any]:
    """Convert one Gmail API v1 message dict into a driver request item.

    The ``message_id`` is the Gmail id the harness keys ground truth on — NOT
    the RFC ``Message-ID`` header — so predictions join to labels correctly.
    """
    payload = msg.get("payload") or {}
    headers = _headers(payload)
    return {
        "message_id": msg["id"],
        "thread_id": msg.get("threadId"),
        "from": _address(headers.get("from", "")),
        "to": _address_list(headers.get("to", "")),
        "cc": _address_list(headers.get("cc", "")),
        "subject": headers.get("subject", ""),
        "body": extract_body_text(payload),
        "date": headers.get("date", ""),
    }


# ---------------------------------------------------------------------------
# Driver output -> base-Agent conversation (pure, offline-testable)
# ---------------------------------------------------------------------------


def build_conversation(driver_output: dict) -> dict[str, Any]:
    """Reshape a driver response into a ``process_query``-compatible result.

    Emits, in order: one ``stats`` system entry per LLM call (what
    ``performance.extract_step_stats`` harvests TTFT / tok-per-s / peak memory
    from), then the assistant tool-call and the ``triage_inbox`` tool envelope
    (what ``benchmark._extract_triage_results`` scores).
    """
    if not isinstance(driver_output, dict):
        raise TypeError(
            f"driver output must be a JSON object, got "
            f"{type(driver_output).__name__}"
        )
    if "results" not in driver_output:
        raise ValueError(
            "driver output has no 'results' key — the Node driver failed before "
            f"triaging anything. Keys present: {sorted(driver_output)}"
        )

    conversation: list[dict[str, Any]] = []
    for entry in driver_output.get("stats") or []:
        if not isinstance(entry, dict) or entry.get("stats_error"):
            # A telemetry gap is reported, never silently rendered as a zero
            # reading that would drag the perf averages toward a fake pass.
            continue
        conversation.append(
            {"role": "system", "content": {"type": "stats", "performance_stats": entry}}
        )

    envelope = {
        "ok": True,
        "data": {
            "results": driver_output["results"],
            # Reported, not scored — see the module docstring on why the usage
            # block is deliberately absent.
            "llm_call_count": driver_output.get("llm_call_count", 0),
            "skipped": driver_output.get("skipped") or [],
            "errors": driver_output.get("errors") or [],
        },
    }
    conversation.append({"role": "assistant", "content": {"tool": "triage_inbox"}})
    conversation.append(
        {
            "role": "tool",
            "name": "triage_inbox",
            "content": json.dumps(envelope),
        }
    )
    return {"conversation": conversation}


# ---------------------------------------------------------------------------
# The agent stand-in
# ---------------------------------------------------------------------------


class NodeEmailAgent:
    """A ``process_query``-compatible facade over the Node email SDK.

    Args:
        gmail_backend: any object with ``list_messages`` / ``get_message`` in
            Gmail API v1 shape — in practice ``FakeGmailBackend`` over the
            committed corpus, the same instance kind the Python agent gets.
        model_id: model to pass through to the SDK.
        base_url: OpenAI-compatible endpoint (Lemonade). Required — the SDK has
            no default and guessing one hides a misconfigured run.
        limit: maximum messages to triage.
        node_dir: SDK directory; defaults to the repo checkout.
        ctx_size / force_llm_classify / now / user_context: passed through.
        timeout_s: wall-clock ceiling for the whole driver run.
        llm_timeout_ms: per-request ceiling handed to the SDK. Deliberately far
            above the SDK's 120s default: on a small local model a single
            action-extraction call can exceed it, and the SDK drops a timed-out
            phase-2 action with only a stderr warning — which would show up as a
            quietly worse action-item score rather than an error.
    """

    def __init__(
        self,
        *,
        gmail_backend: Any,
        model_id: str,
        base_url: str,
        limit: int = 50,
        node_dir: str | Path | None = None,
        ctx_size: int | None = None,
        force_llm_classify: bool = False,
        now: str | None = None,
        user_context: str | None = None,
        node_bin: str = "node",
        timeout_s: int = 14400,
        llm_timeout_ms: int = 600_000,
        debug: bool = False,
    ) -> None:
        if not base_url:
            raise ValueError(
                "NodeEmailAgent requires an explicit base_url for the "
                "OpenAI-compatible endpoint (e.g. http://localhost:8000/api/v1). "
                "Pass --backend, or set LEMONADE_BASE_URL."
            )
        self._backend = gmail_backend
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._limit = limit
        self._node_dir = Path(node_dir) if node_dir else default_node_dir()
        self._ctx_size = ctx_size
        self._force_llm_classify = force_llm_classify
        self._now = now
        self._user_context = user_context
        self._node_bin = node_bin
        self._timeout_s = timeout_s
        self._llm_timeout_ms = llm_timeout_ms
        self._debug = debug
        self.last_driver_output: dict[str, Any] | None = None

    # -- corpus ----------------------------------------------------------

    def _inbox_message_ids(self) -> list[str]:
        listing = self._backend.list_messages(
            label_ids=["INBOX"], max_results=self._limit
        )
        return [m["id"] for m in (listing.get("messages") or [])][: self._limit]

    def _driver_items(self, message_ids: Iterable[str]) -> list[dict[str, Any]]:
        return [
            gmail_message_to_driver_item(
                self._backend.get_message(mid, format="full")
            )
            for mid in message_ids
        ]

    # -- the harness contract --------------------------------------------

    def process_query(self, prompt: str) -> dict[str, Any]:  # noqa: ARG002
        """Triage the backend's inbox through the Node SDK.

        ``prompt`` is accepted for signature compatibility and ignored: the
        Python agent needs a natural-language nudge to pick its triage tool,
        while the Node SDK is a direct function call with no tool routing. The
        message count comes from ``limit``, which the harness set from the same
        ``--limit`` it steers the Python agent with.
        """
        driver = self._node_dir.joinpath(*_DRIVER_RELPATH)
        if not driver.is_file():
            raise FileNotFoundError(
                f"Node eval driver not found at {driver}. Point --node-dir at a "
                "checkout of hub/agents/email/node, and run `npm install` there."
            )
        if not (self._node_dir / "node_modules").is_dir():
            raise RuntimeError(
                f"{self._node_dir / 'node_modules'} is missing — the Node SDK's "
                "dependencies are not installed. Run `npm install` in "
                f"{self._node_dir}."
            )

        items = self._driver_items(self._inbox_message_ids())
        if not items:
            raise RuntimeError(
                "The mail backend returned no INBOX messages, so there is "
                "nothing to score. Build the corpus first: "
                "`python tests/fixtures/email/generate_mbox.py`."
            )

        request = {
            "base_url": self._base_url,
            "model": self._model_id,
            "ctx_size": self._ctx_size,
            "timeout_ms": self._llm_timeout_ms,
            "force_llm_classify": self._force_llm_classify,
            "collect_stats": True,
            "debug": self._debug,
            "principal": {"email": self._backend.get_user_email()},
            "items": items,
        }
        if self._now:
            request["now"] = self._now
        if self._user_context:
            request["user_context"] = self._user_context

        proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [self._node_bin, str(driver)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(self._node_dir),
            timeout=self._timeout_s,
            check=False,
            env={**os.environ, "NODE_NO_WARNINGS": "1"},
        )
        if proc.stderr:
            # The driver's progress log is the only visibility into a run that
            # can take many minutes; forward it rather than swallowing it.
            print(proc.stderr.rstrip())
        if proc.returncode != 0:
            raise RuntimeError(
                f"Node eval driver exited {proc.returncode}. "
                f"stderr tail: {proc.stderr[-2000:]!r}"
            )
        try:
            output = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Node eval driver did not emit valid JSON on stdout: {exc}. "
                f"stdout head: {proc.stdout[:500]!r}"
            ) from exc

        self.last_driver_output = output
        return build_conversation(output)

    def close_db(self) -> None:
        """No-op: the Node SDK is stateless and holds no database handle.

        Present because ``benchmark._close_agent_db`` looks for it.
        """


# ---------------------------------------------------------------------------
# Factory used by the CLI and the CI gate reader (one construction site)
# ---------------------------------------------------------------------------


def make_node_agent_factory(
    *,
    mbox_path: str,
    model_id: str,
    base_url: str,
    limit: int,
    node_dir: str | Path | None = None,
    ctx_size: int | None = None,
    force_llm_classify: bool = False,
    now: str | None = None,
    debug: bool = False,
) -> Callable[[], NodeEmailAgent]:
    """Build the zero-arg factory ``run_benchmark(agent_factory=...)`` wants.

    A fresh ``FakeGmailBackend`` per call, so repeated experiments cannot leak
    state between runs — matching the Python agent's per-experiment rebuild.
    """

    def factory() -> NodeEmailAgent:
        try:
            from tests.fixtures.email.fake_gmail import FakeGmailBackend
        except ImportError as exc:
            raise RuntimeError(
                "The Node email eval must run from a GAIA repo checkout — it "
                "drives the synthetic corpus in tests/fixtures/email and is not "
                f"available in a packaged install. Original import error: {exc}"
            ) from exc

        return NodeEmailAgent(
            gmail_backend=FakeGmailBackend(mbox_path),
            model_id=model_id,
            base_url=base_url,
            limit=limit,
            node_dir=node_dir,
            ctx_size=ctx_size,
            force_llm_classify=force_llm_classify,
            now=now,
            debug=debug,
        )

    return factory
