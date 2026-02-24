# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Tier 2: Coding Task Scenarios

Each scenario validates a concrete output on disk.  The mock LLM produces real
file content; tests then assert the content is correct Python.

Scenarios:
  1. Create a valid Python function  (fibonacci)
  2. Create a source file + matching test file
  3. Fix broken code  (quality gate recovery)
  4. Multi-file project creation
  5. Memory persistence across sessions
  6. Checkpoint mid-task, resume

Real-LLM tier (requires --real-llm flag):
  - test_fib_function_correct_output
  - test_calculator_tests_pass
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ===========================================================================
# Scenario 1: Create a valid Python function
# ===========================================================================


def test_create_valid_python_function(agent_workspace, harness):
    """
    Mock LLM writes fibonacci.py with a fib() function.
    Assert file exists and is valid Python.
    """
    agent, ws = agent_workspace
    fib_path = str(ws / "fibonacci.py")
    fib_content = (
        "def fib(n):\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fib(n - 1) + fib(n - 2)\n"
    )

    responses = [
        harness.tool_call(
            "write_file",
            {"file_path": fib_path, "content": fib_content},
        ),
        harness.answer("Created fibonacci.py with fib(n) function."),
    ]

    with harness.patch(agent, responses):
        result = agent.process_query(
            "Create fibonacci.py with a fib(n) function"
        )

    fib_file = Path(fib_path)
    assert fib_file.exists(), "fibonacci.py must be created on disk"

    source = fib_file.read_text()
    tree = ast.parse(source)  # raises SyntaxError if invalid

    func_names = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    ]
    assert "fib" in func_names, f"fibonacci.py must define \'fib\', got: {func_names}"

    assert result is not None, "process_query must return a result"


# ===========================================================================
# Scenario 2: Create source file + test file
# ===========================================================================


def test_create_file_and_matching_test_file(agent_workspace, harness):
    """
    Mock LLM writes calculator.py and test_calculator.py.
    Assert both exist, are valid Python, and the test file imports calculator.
    """
    agent, ws = agent_workspace
    calc_path = str(ws / "calculator.py")
    test_path = str(ws / "test_calculator.py")

    calc_content = (
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def subtract(a, b):\n"
        "    return a - b\n"
    )

    test_content = (
        "import sys, os\n"
        "sys.path.insert(0, os.path.dirname(__file__))\n"
        "from calculator import add, subtract\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_subtract():\n"
        "    assert subtract(5, 3) == 2\n"
    )

    responses = [
        harness.tool_call(
            "write_file",
            {"file_path": calc_path, "content": calc_content},
        ),
        harness.tool_call(
            "write_file",
            {"file_path": test_path, "content": test_content},
        ),
        harness.answer("Created calculator.py and test_calculator.py."),
    ]

    with harness.patch(agent, responses):
        agent.process_query(
            "Create calculator.py with add/subtract and test_calculator.py with pytest tests"
        )

    for path_str in (calc_path, test_path):
        p = Path(path_str)
        assert p.exists(), f"{p.name} must be created on disk"
        ast.parse(p.read_text())  # valid Python

    test_source = Path(test_path).read_text()
    assert "calculator" in test_source, "test file must import from calculator"

    assert harness.call_count() == 3, (
        f"Expected 3 LLM calls, got {harness.call_count()}"
    )


# ===========================================================================
# Scenario 3: Fix broken code (quality gate recovery)
# ===========================================================================


def test_broken_code_quality_gate_recovery(agent_workspace, harness):
    """
    Turn 1: LLM writes syntactically broken code -> SyntaxGate fails.
    Turn 2: LLM writes valid code -> gates pass.
    Assert:
      - escalation ladder was incremented (gate failure detected)
      - final file passes ast.parse
    """
    agent, ws = agent_workspace
    state = agent.shared_state
    file_path = str(ws / "broken.py")

    # 4-response sequence so the first loop ends with broken code:
    # Loop 1: write broken + answer → gates fail → escalation.increment()
    # Loop 2: write valid + answer → gates skip (no new files) → done
    responses = [
        harness.tool_call(
            "write_file",
            {"file_path": file_path, "content": "def broken(\n    pass\n"},
        ),
        harness.answer("wrote the file"),
        harness.tool_call(
            "write_file",
            {"file_path": file_path, "content": "def fixed():\n    pass\n"},
        ),
        harness.answer("Fixed the syntax error."),
    ]

    call_count = {"n": 0}

    def _patched_list_files():
        call_count["n"] += 1
        if call_count["n"] <= 1:
            return []  # before first loop: no files
        return [file_path]  # after first loop onward: file exists

    with patch.object(state.manifest, "list_files", side_effect=_patched_list_files):
        with patch.object(
            agent.escalation_ladder, "increment",
            wraps=agent.escalation_ladder.increment,
        ) as esc_spy:
            with harness.patch(agent, responses):
                agent.process_query("Create a fixed python file")

    assert esc_spy.called, "Escalation must be triggered on syntax gate failure"

    assert Path(file_path).exists(), "File must exist on disk"
    ast.parse(Path(file_path).read_text())


# ===========================================================================
# Scenario 4: Multi-file project creation
# ===========================================================================


def test_multi_file_project_creation(agent_workspace, harness):
    """
    Mock LLM creates main.py, utils.py, and README.md.
    Assert all 3 files exist, .py files are valid Python, README is non-empty.
    """
    agent, ws = agent_workspace
    main_path = str(ws / "main.py")
    utils_path = str(ws / "utils.py")
    readme_path = str(ws / "README.md")

    main_content = (
        "from utils import helper\n\n"
        "def main():\n"
        "    print(helper())\n\n"
        "if __name__ == \'__main__\':\n"
        "    main()\n"
    )
    utils_content = "def helper():\n    return \'Hello from utils\'\n"
    readme_content = "# My Project\n\nA simple Python project.\n"

    responses = [
        harness.tool_call(
            "write_file",
            {"file_path": main_path, "content": main_content},
        ),
        harness.tool_call(
            "write_file",
            {"file_path": utils_path, "content": utils_content},
        ),
        harness.tool_call(
            "write_file",
            {"file_path": readme_path, "content": readme_content},
        ),
        harness.answer("Created main.py, utils.py, and README.md."),
    ]

    with harness.patch(agent, responses):
        agent.process_query(
            "Create a project with main.py, utils.py, and README.md"
        )

    for py_path in (main_path, utils_path):
        p = Path(py_path)
        assert p.exists(), f"{p.name} must be created"
        ast.parse(p.read_text())

    readme = Path(readme_path)
    assert readme.exists(), "README.md must be created"
    assert len(readme.read_text()) > 0, "README.md must be non-empty"

    assert harness.call_count() == 4, (
        f"Expected 4 LLM calls (3 writes + 1 answer), got {harness.call_count()}"
    )


# ===========================================================================
# Scenario 5: Memory persistence across sessions
# ===========================================================================


def test_memory_persists_across_sessions(tmp_path):
    """
    Session 1: store a memory \'project uses SQLite for storage\'.
    Session 2 (new agent instance, same workspace): process a new query.
    Assert: second session\'s LLM prompt contains the stored memory.
    """
    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    from gaia.agents.base.shared_state import SharedAgentState
    from tests.system.conftest import MockLLMHarness

    ws = tmp_path / "workspace"
    ws.mkdir()

    _PATCH_CREDENTIALS = "gaia.agents.gaia_code.credentials.check_and_setup_credentials"
    _PATCH_CHAT_SDK = "gaia.agents.base.agent.ChatSDK"

    def _make_agent():
        with _patch(_PATCH_CREDENTIALS, return_value=(True, None)), _patch(
            _PATCH_CHAT_SDK
        ) as mock_chat_cls:
            mock_chat_instance = MagicMock()
            mock_chat_instance.get_stats.return_value = None
            mock_chat_cls.return_value = mock_chat_instance
            from gaia.agents.gaia_code.agent import GaiaCodeAgent

            ag = GaiaCodeAgent(workspace_dir=ws, silent_mode=True, tui_mode="off")
        return ag

    # --- Session 1: store a memory ---
    SharedAgentState._instance = None
    agent1 = _make_agent()
    agent1.shared_state.memory.store_memory(
        "db_tech", "project uses SQLite for storage"
    )
    harness1 = MockLLMHarness()
    with harness1.patch(agent1, [harness1.answer("noted")]):
        agent1.process_query("noted the storage tech")

    # --- Reset singleton to simulate a new process ---
    SharedAgentState._instance = None

    # --- Session 2: new agent on the same workspace ---
    agent2 = _make_agent()
    harness2 = MockLLMHarness()
    with harness2.patch(agent2, [harness2.answer("done")]):
        agent2.process_query("write a new feature")

    harness2.assert_prompt_contains("SQLite for storage", call_index=0)

    mem = agent2.shared_state.memory.get_memory("db_tech")
    assert mem is not None, "Memory must persist in the DB"
    assert "SQLite" in mem

    SharedAgentState._instance = None


# ===========================================================================
# Scenario 6: Checkpoint mid-task, resume
# ===========================================================================


def test_checkpoint_and_resume(agent_workspace, harness):
    """
    Create a checkpoint explicitly via agent.checkpoint().
    Create a new agent on the same workspace.
    Assert resume_from_checkpoint() returns True and the checkpoint file exists.
    """
    agent, ws = agent_workspace

    with harness.patch(agent, [harness.answer("mid-task")]):
        agent.process_query("start a long task")

    ckpt_result = agent.checkpoint()
    assert ckpt_result["success"] is True, "checkpoint() must succeed"
    ckpt_path = Path(ckpt_result["checkpoint_path"])
    assert ckpt_path.exists(), "checkpoint.json must exist on disk"

    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None

    from unittest.mock import MagicMock
    from unittest.mock import patch as _patch

    _PATCH_CREDENTIALS = "gaia.agents.gaia_code.credentials.check_and_setup_credentials"
    _PATCH_CHAT_SDK = "gaia.agents.base.agent.ChatSDK"

    with _patch(_PATCH_CREDENTIALS, return_value=(True, None)), _patch(
        _PATCH_CHAT_SDK
    ) as mock_chat_cls:
        mock_chat_instance = MagicMock()
        mock_chat_instance.get_stats.return_value = None
        mock_chat_cls.return_value = mock_chat_instance
        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        agent2 = GaiaCodeAgent(workspace_dir=ws, silent_mode=True, tui_mode="off")

    resumed = agent2.resume_from_checkpoint()
    assert resumed is True, "resume_from_checkpoint() must return True"

    SharedAgentState._instance = None


# ===========================================================================
# Tier 4: Real-LLM task validation  (requires --real-llm flag)
# ===========================================================================


@pytest.mark.requires_llm
def test_fib_function_correct_output(real_agent_workspace):
    """
    Real LLM generates fibonacci.py — validate it actually produces correct output.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        "Create fibonacci.py with a fib(n) function that returns the nth "
        "Fibonacci number.  0-indexed: fib(0)=0, fib(1)=1, fib(10)=55."
    )

    fib_file = ws / "fibonacci.py"
    assert fib_file.exists(), "fibonacci.py must be created"

    spec = importlib.util.spec_from_file_location("fibonacci", fib_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.fib(10) == 55, f"fib(10) should be 55, got {mod.fib(10)}"
    assert mod.fib(0) in (0, 1), f"fib(0) base case incorrect: {mod.fib(0)}"


@pytest.mark.requires_llm
def test_calculator_tests_pass(real_agent_workspace):
    """
    Real LLM generates calculator.py + test_calculator.py — run the generated tests.
    """
    agent, ws = real_agent_workspace

    agent.process_query(
        "Create calculator.py with add/subtract/multiply/divide functions "
        "and test_calculator.py with pytest tests for all four operations."
    )

    test_file = ws / "test_calculator.py"
    assert test_file.exists(), "test_calculator.py must be created"

    outcome = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        cwd=str(ws),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert outcome.returncode == 0, (
        f"Generated tests must pass.\nSTDOUT:\n{outcome.stdout}\n"
        f"STDERR:\n{outcome.stderr}"
    )
