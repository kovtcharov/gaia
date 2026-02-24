# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
System test infrastructure for GAIA Code agent.

Provides:
- MockLLMHarness: scripted LLM response queue with call recording
- agent_workspace: creates a real GaiaCodeAgent backed by a real SharedAgentState
  in a tmpdir (only the LLM is mocked; all DBs are real SQLite)
- reset_gaia_code_state(): utility to wipe all DBs and output files

Design principle:
  The DB layer is REAL — not mocked.  Only chat.send_messages is stubbed.
  This means DB reads/writes during the agent loop are actual SQLite operations,
  so test failures will reliably catch the memory-underutilisation bug class.
"""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# pytest CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--real-llm",
        action="store_true",
        default=False,
        help="Run tests that require a live LLM (Lemonade or Claude).",
    )


# ---------------------------------------------------------------------------
# MockLLMHarness
# ---------------------------------------------------------------------------


class MockLLMHarness:
    """
    Scripted LLM response queue.

    Usage::

        harness = MockLLMHarness()
        responses = [
            harness.tool_call("write_file", {"file_path": "/tmp/x.py", "content": "x=1"}),
            harness.answer("done"),
        ]
        with harness.patch(agent, responses):
            agent.process_query("create x.py")

        harness.assert_prompt_contains("pathlib", call_index=0)
    """

    def __init__(self):
        self.call_log: List[tuple] = []  # (messages_sent, response_returned)

    # ------------------------------------------------------------------
    # Response builders
    # ------------------------------------------------------------------

    def tool_call(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        thought: str = "Working on the task.",
    ) -> Mock:
        """Return a mock LLM response that issues a tool call."""
        payload = json.dumps(
            {
                "thought": thought,
                "goal": "executing",
                "tool": tool_name,
                "tool_args": tool_args,
            }
        )
        mock_resp = Mock()
        mock_resp.text = payload
        mock_resp.stats = None  # None so agent skips stats processing (avoids Mock subscript error)
        return mock_resp

    def answer(self, text: str, thought: str = "Task complete.") -> Mock:
        """Return a mock LLM response that terminates the loop."""
        payload = json.dumps(
            {
                "thought": thought,
                "goal": "done",
                "answer": text,
            }
        )
        mock_resp = Mock()
        mock_resp.text = payload
        mock_resp.stats = None  # None so agent skips stats processing (avoids Mock subscript error)
        return mock_resp

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------

    @contextmanager
    def patch(self, agent, responses: List[Mock]):
        """
        Context manager: replaces agent.chat.send_messages with a scripted
        sequence and records every (messages, response) pair.
        """
        self.call_log.clear()
        responses_iter = list(responses)

        def _side_effect(messages, **kwargs):
            idx = len(self.call_log)
            if idx >= len(responses_iter):
                # Safety: return a final answer if the script runs out
                fallback = self.answer(f"[harness fallback answer at step {idx}]")
                self.call_log.append((messages, fallback))
                return fallback
            resp = responses_iter[idx]
            self.call_log.append((messages, resp))
            return resp

        with patch.object(agent.chat, "send_messages", side_effect=_side_effect):
            yield

    # ------------------------------------------------------------------
    # Assertion helpers
    # ------------------------------------------------------------------

    def prompt_at(self, call_index: int = 0) -> str:
        """Return the stringified messages sent on a given LLM call."""
        if call_index >= len(self.call_log):
            raise IndexError(
                f"call_index={call_index} but only {len(self.call_log)} calls recorded"
            )
        messages = self.call_log[call_index][0]
        return str(messages)

    def assert_prompt_contains(self, text: str, call_index: int = 0):
        """Assert that a given LLM call included *text* in the messages."""
        prompt = self.prompt_at(call_index)
        assert text in prompt, (
            f"Expected {text!r} in LLM prompt #{call_index}\n"
            f"Actual prompt (truncated):\n{prompt[:2000]}"
        )

    def call_count(self) -> int:
        """Number of LLM calls made so far."""
        return len(self.call_log)


# ---------------------------------------------------------------------------
# State reset utilities  (also used by the user in interactive sessions)
# ---------------------------------------------------------------------------


def reset_gaia_code_state(workspace_dir: Optional[Path] = None) -> None:
    """
    Reset all GaiaCode state so the next run starts completely fresh.

    Clears:
    - SharedAgentState singleton (in-memory)
    - All SQLite DB files in workspace_dir (if provided)
    - All output files in workspace_dir (if provided)

    Safe to call before/after test runs or from a REPL.
    """
    # 1. Kill the in-memory singleton
    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None

    # 2. Wipe DB files on disk
    if workspace_dir is not None:
        workspace_dir = Path(workspace_dir)
        for db_file in workspace_dir.glob("*.db"):
            try:
                db_file.unlink()
            except OSError:
                pass
        # Also remove checkpoint and any generated files
        for extra in ["checkpoint.json"]:
            p = workspace_dir / extra
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset SharedAgentState singleton before and after every system test."""
    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None
    yield
    SharedAgentState._instance = None


@pytest.fixture
def harness() -> MockLLMHarness:
    """Fresh MockLLMHarness for each test."""
    return MockLLMHarness()


@pytest.fixture
def agent_workspace(tmp_path):
    """
    Create a real GaiaCodeAgent with a real SharedAgentState backed by a
    fresh tmpdir workspace.  Only the LLM transport (chat.send_messages) is
    mocked — all DB operations are real SQLite.

    Yields:
        (agent, workspace_path)
    """
    ws = tmp_path / "workspace"
    ws.mkdir()

    _PATCH_CREDENTIALS = "gaia.agents.gaia_code.credentials.check_and_setup_credentials"
    _PATCH_CHAT_SDK = "gaia.agents.base.agent.ChatSDK"

    with patch(_PATCH_CREDENTIALS, return_value=(True, None)), patch(
        _PATCH_CHAT_SDK
    ) as mock_chat_cls:
        # ChatSDK() returns a MagicMock so agent.chat is a MagicMock —
        # harness.patch() will overlay send_messages on top of that.
        mock_chat_instance = MagicMock()
        mock_chat_instance.get_stats.return_value = None
        mock_chat_cls.return_value = mock_chat_instance

        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        agent = GaiaCodeAgent(
            workspace_dir=ws,
            silent_mode=True,
            tui_mode="off",
        )

    yield agent, ws

    # Cleanup: reset singleton so the tmpdir can be deleted safely
    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None
    # Close any open SQLite connections to allow Windows file deletion
    if hasattr(agent, "shared_state") and agent.shared_state:
        _close_shared_state_connections(agent.shared_state)


def _close_shared_state_connections(state) -> None:
    """Best-effort close all SQLite connections in a SharedAgentState."""
    for attr in ("memory", "knowledge", "tools", "skills", "agents", "logs"):
        db = getattr(state, attr, None)
        if db and hasattr(db, "conn"):
            try:
                db.conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Real-LLM fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def require_lemonade(request):
    """
    Skip the test unless --real-llm was passed on the command line.

    Also checks that the Lemonade server (or Claude API key) is reachable.
    """
    if not request.config.getoption("--real-llm"):
        pytest.skip("Pass --real-llm to run LLM integration tests")

    import requests as _req

    try:
        r = _req.get("http://localhost:8000/api/v1/health", timeout=5)
        if r.status_code != 200:
            pytest.skip("Lemonade server not healthy")
    except Exception:
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("Neither Lemonade nor ANTHROPIC_API_KEY is available")


@pytest.fixture
def real_agent_workspace(tmp_path, require_lemonade):
    """
    Create a real GaiaCodeAgent with a REAL LLM backend (Lemonade or Claude).

    Unlike agent_workspace, ChatSDK is NOT mocked — the agent talks to the
    real LLM.  Credentials check is still patched to avoid interactive prompts.

    Requires --real-llm flag (enforced via require_lemonade dependency).

    Yields:
        (agent, workspace_path)
    """
    ws = tmp_path / "workspace"
    ws.mkdir()

    _PATCH_CREDENTIALS = "gaia.agents.gaia_code.credentials.check_and_setup_credentials"

    with patch(_PATCH_CREDENTIALS, return_value=(True, None)):
        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        agent = GaiaCodeAgent(
            workspace_dir=ws,
            silent_mode=True,
            tui_mode="off",
        )

    yield agent, ws

    # Cleanup
    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None
    if hasattr(agent, "shared_state") and agent.shared_state:
        _close_shared_state_connections(agent.shared_state)
