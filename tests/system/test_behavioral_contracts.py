# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Tier 1: Behavioral Contract Tests

These tests verify system-level properties — the architectural "contracts" the
GaiaCodeAgent makes.  Each test uses spy wrappers (wraps=) to observe real DB
calls while letting the real code execute.

Contracts tested:
  1. Memory is read at query time (recall_memories called)
  2. Knowledge insights are recalled at query time
  3. Skills are surfaced at query time
  4. Tool usage is recorded in tools.db
  5. Tool results are stored in memory.db (tool_results table)
  6. Quality gates run when files are tracked in the manifest
  7. Quality gate failure triggers escalation increment
  8. Audit log captures TASK_START and TASK_COMPLETE
"""

import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.system.conftest import MockLLMHarness


# ---------------------------------------------------------------------------
# Contract 1: Working memory is read at query time
# ---------------------------------------------------------------------------


def test_memory_injected_into_llm_context(agent_workspace, harness):
    """recall_memories() must be called during process_query() and its result
    must appear in the first LLM prompt."""
    agent, ws = agent_workspace
    state = agent.shared_state

    state.memory.store_memory("coding_standard", "always use pathlib for file paths")

    with patch.object(
        state.memory, "recall_memories", wraps=state.memory.recall_memories
    ) as spy:
        with harness.patch(agent, [harness.answer("done")]):
            agent.process_query("write a file handler")

    assert spy.called, "recall_memories() must be called during process_query()"
    harness.assert_prompt_contains(
        "always use pathlib for file paths", call_index=0
    )


# ---------------------------------------------------------------------------
# Contract 2: Knowledge insights are recalled at query time
# ---------------------------------------------------------------------------


def test_knowledge_injected_into_llm_context(agent_workspace, harness):
    """knowledge.recall() must be called during process_query() and the
    stored insight must appear in the first LLM prompt."""
    agent, ws = agent_workspace
    state = agent.shared_state

    state.knowledge.store_insight(
        category="best_practice",
        content="always validate inputs before processing",
        domain="coding",
    )

    with patch.object(
        state.knowledge, "recall", wraps=state.knowledge.recall
    ) as spy:
        with harness.patch(agent, [harness.answer("done")]):
            # Use a query that overlaps with insight words ("validate", "inputs")
            # so FTS5 full-text search returns the stored insight
            agent.process_query("validate inputs before processing data")

    assert spy.called, "knowledge.recall() must be called during process_query()"
    harness.assert_prompt_contains(
        "always validate inputs before processing", call_index=0
    )


# ---------------------------------------------------------------------------
# Contract 3: Skills are surfaced at query time
# ---------------------------------------------------------------------------


def test_skills_injected_into_llm_context(agent_workspace, harness):
    """find_skills() must be called during process_query()."""
    agent, ws = agent_workspace

    with patch.object(
        agent.shared_state.skills, "find_skills",
        wraps=agent.shared_state.skills.find_skills,
    ) as spy:
        with harness.patch(agent, [harness.answer("done")]):
            agent.process_query("debug this error")

    assert spy.called, "find_skills() must be called during process_query()"


# ---------------------------------------------------------------------------
# Contract 4: Tool usage is recorded in tools.db
# ---------------------------------------------------------------------------


def test_tool_usage_recorded_in_db(agent_workspace, harness):
    """After write_file executes, tools.db must record at least one usage
    event for it (via the tool_usage table)."""
    agent, ws = agent_workspace
    state = agent.shared_state
    file_path = str(ws / "hello.py")

    responses = [
        harness.tool_call(
            "write_file",
            {"file_path": file_path, "content": "x = 1\n"},
        ),
        harness.answer("created hello.py"),
    ]
    with harness.patch(agent, responses):
        agent.process_query("create hello.py")

    stats = state.tools.get_tool_stats("write_file")
    assert stats["total"] >= 1, (
        f"tool_usage must have at least 1 row for write_file, got {stats}"
    )


# ---------------------------------------------------------------------------
# Contract 5: Tool results are stored in memory.db
# ---------------------------------------------------------------------------


def test_tool_result_stored_in_memory(agent_workspace, harness):
    """After write_file executes, memory.db must contain a tool_results row."""
    agent, ws = agent_workspace
    state = agent.shared_state
    file_path = str(ws / "hello.py")

    responses = [
        harness.tool_call(
            "write_file",
            {"file_path": file_path, "content": "x = 1\n"},
        ),
        harness.answer("done"),
    ]
    with harness.patch(agent, responses):
        agent.process_query("create hello.py")

    count = state.memory.conn.execute(
        "SELECT COUNT(*) FROM tool_results WHERE tool_name = \'write_file\'"
    ).fetchone()[0]
    assert count >= 1, (
        f"tool_results in memory.db must have at least 1 row for write_file, got {count}"
    )


# ---------------------------------------------------------------------------
# Contract 6: Quality gates run when files are tracked in the manifest
# ---------------------------------------------------------------------------


def test_quality_gates_run_on_manifest_tracked_file(agent_workspace, harness):
    """
    quality_gates.run_all() must be called when a file is present in
    result[\'files\'].

    NOTE: write_file does NOT currently add to the manifest automatically.
    This test exercises the quality gate mechanism by patching manifest
    list_files to include the file after execution, simulating what a
    manifest-aware write_file would do.
    """
    agent, ws = agent_workspace
    state = agent.shared_state
    file_path = str(ws / "hello.py")
    content = "def hello():\n    pass\n"

    # Patch manifest so the agent sees the file as "new" after write_file runs
    call_count = {"n": 0}

    def _patched_list_files():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return []  # "before" snapshot: no files yet
        return [file_path]  # "after" snapshot: file appeared

    responses = [
        harness.tool_call(
            "write_file",
            {"file_path": file_path, "content": content},
        ),
        harness.answer("done"),
    ]

    with patch.object(state.manifest, "list_files", side_effect=_patched_list_files):
        with patch.object(
            agent.quality_gates, "run_all", wraps=agent.quality_gates.run_all
        ) as gate_spy:
            with harness.patch(agent, responses):
                agent.process_query("create hello.py")

    assert gate_spy.called, (
        "quality_gates.run_all() must be called when result[\'files\'] is non-empty"
    )
    call_args = str(gate_spy.call_args)
    assert file_path in call_args, (
        f"quality_gates.run_all must receive the created file path.\n"
        f"Expected {file_path!r} in {call_args}"
    )


# ---------------------------------------------------------------------------
# Contract 7: Quality gate failure triggers escalation
# ---------------------------------------------------------------------------


def test_quality_gate_failure_triggers_escalation(agent_workspace, harness):
    """
    When the syntax gate fails (broken code), escalation_ladder.increment()
    must be called and the final file on disk must be syntactically valid.
    """
    agent, ws = agent_workspace
    state = agent.shared_state
    file_path = str(ws / "broken.py")

    # 4-response sequence:
    # Loop 1: write broken file + answer → _execute_task returns with broken file
    # → quality gate FAILS → escalation.increment() called → retry
    # Loop 2: write valid file + answer → _execute_task returns with valid file
    # → quality gates skip (no "new" files detected) → done
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
        harness.answer("fixed the syntax error"),
    ]

    # Patch manifest so quality gates see the file after the first write_file
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
                agent.process_query("create a fixed python file")

    assert esc_spy.called, (
        "escalation_ladder.increment() must be called when a quality gate fails"
    )

    assert Path(file_path).exists(), "Final file must exist on disk"
    ast.parse(Path(file_path).read_text())  # must be valid Python


# ---------------------------------------------------------------------------
# Contract 8: Audit log captures task lifecycle
# ---------------------------------------------------------------------------


def test_audit_log_records_task_start_and_complete(agent_workspace, harness):
    """TASK_START and TASK_COMPLETE must both appear in the audit log after
    a process_query() call completes."""
    agent, ws = agent_workspace

    with harness.patch(agent, [harness.answer("done")]):
        agent.process_query("do something simple")

    log = agent.get_audit_log()
    assert log, "Audit log must not be empty after process_query()"

    action_types = [e.get("action_type") or e.get("type") for e in log]
    assert "TASK_START" in action_types, (
        f"TASK_START must appear in audit log.  Got: {action_types}"
    )
    assert "TASK_COMPLETE" in action_types, (
        f"TASK_COMPLETE must appear in audit log.  Got: {action_types}"
    )
