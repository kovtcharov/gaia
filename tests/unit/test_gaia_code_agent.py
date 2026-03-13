# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit tests for GaiaCodeAgent.

Tests:
- Agent initialization
- Tool registration
- Quality gate integration
- Checkpoint/resume
- Audit logging
- Progress tracking
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaia.agents.gaia_code.agent import GaiaCodeAgent

# Correct patch targets:
# - get_shared_state lives in base/agent.py (imported there)
# - initialize_workspace is imported locally inside GaiaCodeAgent.__init__
_PATCH_SHARED_STATE = "gaia.agents.base.shared_state.get_shared_state"
_PATCH_CHAT_SDK = "gaia.agents.base.agent.ChatSDK"
_PATCH_INIT_WORKSPACE = "gaia.agents.gaia_code.integration.initialize_workspace"
_PATCH_CREDENTIALS = "gaia.agents.gaia_code.credentials.check_and_setup_credentials"


def _make_mock_state(tmpdir):
    """Helper: create a fully-mocked SharedAgentState."""
    mock_state = MagicMock()
    mock_state.workspace_dir = Path(tmpdir)
    # plan.get_all_tasks() returns empty list by default
    mock_state.plan.get_all_tasks.return_value = []
    return mock_state


def _make_agent(tmpdir, mock_get_shared_state, extra_kwargs=None):
    """Helper: instantiate GaiaCodeAgent with all external deps mocked."""
    mock_state = _make_mock_state(tmpdir)
    mock_get_shared_state.return_value = mock_state
    kwargs = dict(workspace_dir=Path(tmpdir), silent_mode=True, skip_lemonade=True)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    agent = GaiaCodeAgent(**kwargs)
    return agent, mock_state


class TestGaiaCodeAgent:
    """Test GaiaCodeAgent."""

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_initialization(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Test agent initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, _ = _make_agent(tmpdir, mock_get_shared_state)

            assert agent.shared_state is not None
            assert agent.quality_gates is not None
            assert agent.escalation_ladder is not None
            assert agent.session_start is not None

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_tool_registration(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Test that GAIA Code tools are registered."""
        from gaia.agents.base.tools import _TOOL_REGISTRY

        with tempfile.TemporaryDirectory() as tmpdir:
            agent, _ = _make_agent(tmpdir, mock_get_shared_state)

            # Verify key RAC tools are registered after instantiation
            assert "remember" in _TOOL_REGISTRY, "remember tool must be registered"
            assert "recall_memory" in _TOOL_REGISTRY, "recall_memory tool must be registered"
            assert "forget_memory" in _TOOL_REGISTRY, "forget_memory tool must be registered"
            assert "store_insight" in _TOOL_REGISTRY, "store_insight tool must be registered"
            assert "agent_query" in _TOOL_REGISTRY, "agent_query tool must be registered"
            assert "recall" in _TOOL_REGISTRY, "recall tool must be registered"
            assert "find_tool" in _TOOL_REGISTRY, "find_tool tool must be registered"

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_checkpoint_and_resume(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Test checkpoint and resume functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, _ = _make_agent(tmpdir, mock_get_shared_state)

            # Create checkpoint
            checkpoint_result = agent.checkpoint()
            assert checkpoint_result["success"] is True

            # Create new agent instance
            agent2, _ = _make_agent(tmpdir, mock_get_shared_state)

            # Resume from checkpoint
            resumed = agent2.resume_from_checkpoint()
            assert resumed is True

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_audit_logging(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Test audit logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, _ = _make_agent(tmpdir, mock_get_shared_state)

            # Log some actions
            agent._log_audit("TEST_ACTION", {"key": "value"})
            agent._log_audit("ANOTHER_ACTION", {"data": "test"})

            # Get audit log
            log = agent.get_audit_log()
            assert len(log) == 2
            assert log[0]["action_type"] == "TEST_ACTION"
            assert log[1]["action_type"] == "ANOTHER_ACTION"

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_progress_tracking(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Test progress tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_state = _make_mock_state(tmpdir)

            # Create mock plan with tasks
            mock_task1 = MagicMock()
            mock_task1.status = "completed"
            mock_task2 = MagicMock()
            mock_task2.status = "in_progress"
            mock_task3 = MagicMock()
            mock_task3.status = "pending"

            mock_state.plan.get_all_tasks.return_value = [mock_task1, mock_task2, mock_task3]
            mock_get_shared_state.return_value = mock_state

            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # Get progress
            progress = agent.get_progress()

            assert progress["total_tasks"] == 3
            assert progress["completed"] == 1
            assert progress["in_progress"] == 1
            assert progress["pending"] == 1
            assert progress["progress_percent"] == 33  # 1/3 = 33%

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_system_prompt(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Test system prompt generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, _ = _make_agent(tmpdir, mock_get_shared_state)

            prompt = agent._get_system_prompt()

            # Verify key concepts are in the prompt
            assert "RECURSIVE DECOMPOSITION" in prompt
            assert "QUALITY-FIRST" in prompt
            assert "agent_query()" in prompt
            assert "recall()" in prompt

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_plan_task_status_progresses(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Plan root task transitions pending→in_progress→completed during process_query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, mock_state = _make_agent(tmpdir, mock_get_shared_state)

            # Set up plan mock to return predictable IDs
            mock_state.plan.create_plan.return_value = "plan-1"
            mock_state.plan.create_task.return_value = "task-1"

            # Stub _execute_with_quality_gates so we never hit the LLM
            with patch.object(
                agent,
                "_execute_with_quality_gates",
                return_value={"success": True, "result": "done", "files": []},
            ):
                agent.process_query("do something", create_plan=True)

            # start_task called once with task-1 (pending → in_progress)
            mock_state.plan.start_task.assert_called_once_with("task-1")
            # complete_task called once with task-1 (in_progress → completed)
            mock_state.plan.complete_task.assert_called_once()
            args = mock_state.plan.complete_task.call_args[0]
            assert args[0] == "task-1"

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_plan_task_marked_failed_on_error(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """Plan root task is marked failed when execution returns success=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, mock_state = _make_agent(tmpdir, mock_get_shared_state)

            mock_state.plan.create_plan.return_value = "plan-x"
            mock_state.plan.create_task.return_value = "task-x"

            with patch.object(
                agent,
                "_execute_with_quality_gates",
                return_value={"success": False, "result": None, "error": "boom"},
            ):
                agent.process_query("fail please", create_plan=True)

            mock_state.plan.fail_task.assert_called_once()
            args = mock_state.plan.fail_task.call_args[0]
            assert args[0] == "task-x"

    @patch(_PATCH_INIT_WORKSPACE)
    @patch(_PATCH_CREDENTIALS, return_value=(True, None))
    @patch(_PATCH_CHAT_SDK)
    @patch(_PATCH_SHARED_STATE)
    def test_write_file_registers_in_manifest(
        self, mock_get_shared_state, mock_chat_sdk, mock_creds, mock_init_ws
    ):
        """write_file calls manifest.add_file() so quality gates see the output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent, mock_state = _make_agent(tmpdir, mock_get_shared_state)

            # Patch the base _execute_tool to return a successful write result
            with patch(
                "gaia.agents.base.agent.Agent._execute_tool",
                return_value={"status": "success", "written": True},
            ):
                agent._execute_tool(
                    "write_file",
                    {"file_path": "/tmp/test.cpp", "content": "int main() { return 0; }"},
                )

            # manifest.add_file should have been called with the written path
            mock_state.manifest.add_file.assert_called_once()
            call_args = mock_state.manifest.add_file.call_args[0]
            assert call_args[0] == "/tmp/test.cpp"
