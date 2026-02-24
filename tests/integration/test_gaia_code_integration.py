# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Integration tests for GAIA Code.

These tests validate end-to-end functionality:
- Agent can execute tasks
- Quality gates work
- Checkpoint/resume works
- Specialists are invoked
- Memory persists across sessions

Requires: Lemonade server running with Qwen3-Coder-30B
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.integration
class TestGaiaCodeEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture(autouse=True)
    def reset_shared_state(self):
        """Reset SharedAgentState singleton between tests so each test gets a fresh instance."""
        from gaia.agents.base.shared_state import SharedAgentState

        SharedAgentState._instance = None
        yield
        SharedAgentState._instance = None

    @patch("gaia.agents.base.agent.ChatSDK")
    def test_simple_task_execution(self, mock_chat_sdk):
        """Test executing a simple coding task."""
        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock LLM responses
            mock_chat_sdk.return_value.chat.return_value = {
                "content": '{"thought": "Creating function", "tool": "write_file", "tool_args": {"path": "prime.py", "content": "def is_prime(n): return n > 1"}}'
            }

            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # This would execute the task
            # For now, just verify agent initializes
            assert agent is not None

    @patch("gaia.agents.base.agent.ChatSDK")
    def test_quality_gates_catch_errors(self, mock_chat_sdk):
        """Test that quality gates catch errors."""
        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # Create a file with syntax error
            test_file = Path(tmpdir) / "broken.py"
            test_file.write_text("def broken()\\n    return 'missing colon'")

            # Run quality gates using the correct API: run_all(paths, **kwargs)
            results = agent.quality_gates.run_all([str(test_file)], project_dir=tmpdir)
            all_passed = agent.quality_gates.all_passed(results)

            # Should fail syntax gate
            assert all_passed is False
            assert any(not r.passed and r.gate_name == "Syntax" for r in results.values())

    @patch("gaia.agents.base.agent.ChatSDK")
    def test_checkpoint_and_resume(self, mock_chat_sdk):
        """Test checkpoint/resume across agent instances."""
        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first agent
            agent1 = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            # Create checkpoint
            checkpoint_result = agent1.checkpoint()
            assert checkpoint_result["success"] is True

            # Create second agent and resume
            agent2 = GaiaCodeAgent(
                workspace_dir=Path(tmpdir),
                silent_mode=True,
                skip_lemonade=True,
            )

            resumed = agent2.resume_from_checkpoint()
            assert resumed is True

    def test_specialist_registration(self):
        """Test that specialists are registered."""
        from gaia.agents.gaia_code.integration import (
            initialize_workspace,
            list_available_specialists,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize workspace (registers specialists)
            workspace = initialize_workspace(Path(tmpdir))

            # List specialists
            specialists = list_available_specialists()

            # Should have 7 core specialists
            assert len(specialists) == 7

            specialist_names = [s["name"] for s in specialists]
            assert "DebuggerAgent" in specialist_names
            assert "SecurityAgent" in specialist_names
            assert "RefactoringAgent" in specialist_names
            assert "TestingAgent" in specialist_names

    def test_tool_registration(self):
        """Test that core tools are registered."""
        from gaia.agents.gaia_code.integration import initialize_workspace

        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize workspace (registers tools)
            workspace = initialize_workspace(Path(tmpdir))

            # Check tools were registered
            from gaia.agents.gaia_code.shared_state import get_shared_state

            state = get_shared_state(Path(tmpdir))

            cursor = state.tools.conn.execute("SELECT COUNT(*) FROM tools")
            tool_count = cursor.fetchone()[0]

            # Should have 20+ core tools
            assert tool_count >= 20


@pytest.mark.integration
class TestGaiaCodeValidation:
    """
    Validation tests from milestone requirements.

    These are the validation criteria from GAIA_CODE_MILESTONES.md
    """

    @pytest.mark.skip(reason="Requires LLM integration")
    def test_m0_simple_task(self):
        """
        M0 Validation Test 1: Simple task

        gaia code "Create a Python function that checks if a number is prime, with tests"
        ✅ Function is correct
        ✅ Tests exist and pass
        ✅ Code is clean
        """
        pass

    @pytest.mark.skip(reason="Requires LLM integration")
    def test_m0_multi_file_task(self):
        """
        M0 Validation Test 2: Multi-file task

        gaia code "Create a CLI calculator with add, subtract, multiply, divide. Include tests."
        ✅ Multiple files created (cli.py, calculator.py, test_calculator.py)
        ✅ CLI works
        ✅ Tests pass
        """
        pass

    @pytest.mark.skip(reason="Requires LLM integration")
    def test_m1_cross_session_memory(self):
        """
        M1 Validation: Cross-session memory

        Session 1: gaia code "Create a Python project using pathlib for all file operations"
        Session 2: gaia code "Add a file processing module to the project"
        ✅ Agent knows the project exists
        ✅ Agent uses pathlib (remembers from session 1)
        ✅ Agent doesn't re-read files it already cached
        """
        pass

    @pytest.mark.skip(reason="Requires LLM integration")
    def test_m2_quality_gates(self):
        """
        M2 Validation: Quality gates

        gaia code "Create a FastAPI API with user auth and tests"
        ✅ Agent runs until ALL tests pass (not until max_steps)
        ✅ If agent writes broken code, it detects and fixes it
        ✅ Agent produces a plan and tracks progress
        ✅ Final output: "All quality gates passed: syntax ✅, imports ✅, tests 12/12 ✅"
        """
        pass

    @pytest.mark.skip(reason="Requires LLM integration")
    def test_m3_crash_recovery(self):
        """
        M3 Validation: Crash recovery

        gaia code "Build a project with 20 files"
        # Kill the process at step 15
        gaia code --resume
        ✅ Agent resumes from step 15, not step 1
        ✅ Agent knows what it already completed
        ✅ Final output is complete and correct
        """
        pass


@pytest.mark.integration
class TestGaiaCodeSpecialists:
    """Test specialist functionality."""

    def test_specialist_metadata(self):
        """Test that specialists have correct metadata."""
        from gaia.agents.gaia_code.specialists import (
            DebuggerAgent,
            SecurityAgent,
            TestingAgent,
        )

        debugger = DebuggerAgent()
        metadata = debugger.get_metadata()

        assert metadata["name"] == "DebuggerAgent"
        assert len(metadata["capabilities"]) > 0
        assert len(metadata["tool_packs"]) > 0
        assert len(metadata["workflow"]) > 0

    def test_specialist_workflow(self):
        """Test specialist workflow execution."""
        from gaia.agents.gaia_code.specialists import DebuggerAgent

        debugger = DebuggerAgent()

        # Should have defined workflow
        assert len(debugger.workflow_steps) > 0

        # Should be able to get next step
        step1 = debugger.next_step()
        assert step1 is not None
        assert step1 == debugger.workflow_steps[0]

    def test_all_specialists_initialize(self):
        """Test that all specialists can be initialized."""
        from gaia.agents.gaia_code.specialists import (
            ArchitectureAgent,
            DebuggerAgent,
            DocumentationAgent,
            PerformanceAgent,
            RefactoringAgent,
            SecurityAgent,
            TestingAgent,
        )

        specialists = [
            DebuggerAgent(),
            SecurityAgent(),
            RefactoringAgent(),
            TestingAgent(),
            DocumentationAgent(),
            PerformanceAgent(),
            ArchitectureAgent(),
        ]

        # All should initialize without errors
        assert len(specialists) == 7

        # All should have workflows
        for spec in specialists:
            assert len(spec.workflow_steps) > 0
            assert spec.get_system_prompt() is not None
            assert len(spec.get_tool_packs()) > 0
