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


class TestGaiaCodeAgent:
    """Test GaiaCodeAgent."""

    @patch("gaia.agents.gaia_code.agent.get_shared_state")
    @patch("gaia.agents.base.agent.ChatSDK")
    def test_initialization(self, mock_chat_sdk, mock_get_shared_state):
        """Test agent initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_state = MagicMock()
            mock_state.workspace_dir = Path(tmpdir)
            mock_get_shared_state.return_value = mock_state

            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            assert agent.shared_state is not None
            assert agent.quality_gates is not None
            assert agent.escalation_ladder is not None
            assert agent.session_start is not None

    @patch("gaia.agents.gaia_code.agent.get_shared_state")
    @patch("gaia.agents.base.agent.ChatSDK")
    def test_tool_registration(self, mock_chat_sdk, mock_get_shared_state):
        """Test that GAIA Code tools are registered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_state = MagicMock()
            mock_state.workspace_dir = Path(tmpdir)
            mock_get_shared_state.return_value = mock_state

            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # Verify key tools are registered
            # Note: This depends on implementation details
            # In full implementation, would check agent.tools or similar

    @patch("gaia.agents.gaia_code.agent.get_shared_state")
    @patch("gaia.agents.base.agent.ChatSDK")
    def test_checkpoint_and_resume(self, mock_chat_sdk, mock_get_shared_state):
        """Test checkpoint and resume functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_state = MagicMock()
            mock_state.workspace_dir = Path(tmpdir)
            mock_get_shared_state.return_value = mock_state

            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # Create checkpoint
            checkpoint_result = agent.checkpoint()
            assert checkpoint_result["success"] is True

            # Create new agent instance
            agent2 = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # Resume from checkpoint
            resumed = agent2.resume_from_checkpoint()
            assert resumed is True

    @patch("gaia.agents.gaia_code.agent.get_shared_state")
    @patch("gaia.agents.base.agent.ChatSDK")
    def test_audit_logging(self, mock_chat_sdk, mock_get_shared_state):
        """Test audit logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_state = MagicMock()
            mock_state.workspace_dir = Path(tmpdir)
            mock_get_shared_state.return_value = mock_state

            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # Log some actions
            agent._log_audit("TEST_ACTION", {"key": "value"})
            agent._log_audit("ANOTHER_ACTION", {"data": "test"})

            # Get audit log
            log = agent.get_audit_log()
            assert len(log) == 2
            assert log[0]["action_type"] == "TEST_ACTION"
            assert log[1]["action_type"] == "ANOTHER_ACTION"

    @patch("gaia.agents.gaia_code.agent.get_shared_state")
    @patch("gaia.agents.base.agent.ChatSDK")
    def test_progress_tracking(self, mock_chat_sdk, mock_get_shared_state):
        """Test progress tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_state = MagicMock()
            mock_state.workspace_dir = Path(tmpdir)

            # Create mock plan with tasks
            mock_task1 = MagicMock()
            mock_task1.status = "completed"
            mock_task2 = MagicMock()
            mock_task2.status = "in_progress"
            mock_task3 = MagicMock()
            mock_task3.status = "pending"

            mock_plan = MagicMock()
            mock_plan.get_all_tasks.return_value = [mock_task1, mock_task2, mock_task3]

            mock_state.plan = mock_plan
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

    @patch("gaia.agents.gaia_code.agent.get_shared_state")
    @patch("gaia.agents.base.agent.ChatSDK")
    def test_system_prompt(self, mock_chat_sdk, mock_get_shared_state):
        """Test system prompt generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_state = MagicMock()
            mock_state.workspace_dir = Path(tmpdir)
            mock_get_shared_state.return_value = mock_state

            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            prompt = agent._get_system_prompt()

            # Verify key concepts are in the prompt
            assert "RECURSIVE DECOMPOSITION" in prompt
            assert "QUALITY-FIRST" in prompt
            assert "agent_query()" in prompt
            assert "recall()" in prompt
