# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Tier 3: Regression Tests

Specific regressions for bugs that have been found and fixed.
These tests must continue to pass to prevent regressions.

Bugs covered:
  R1. memory.db was written but never read  (recall_memories never called)
  R2. knowledge.db was written but never read  (recall never called)
  R3. skills.db was written but never read  (find_skills never called)
  R4. codebase_index flooded knowledge.db with >1000 high_complexity entries
  R5. _auto_add_query_paths was called twice in process_query  (duplicate call)
  R6. Timestamps used UTC (DEFAULT CURRENT_TIMESTAMP) instead of local time
  R7. EscalationLadder.get_action() returned "alternative" instead of "cloud"
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import call, patch

import pytest


# ---------------------------------------------------------------------------
# R1: recall_memories() must be called during process_query
# ---------------------------------------------------------------------------


def test_recall_memories_called_during_process_query(agent_workspace, harness):
    """
    Regression: memory.db was written to (tool results stored) but
    recall_memories() was never called to inject those memories into the
    LLM context at query time.
    """
    agent, ws = agent_workspace
    state = agent.shared_state

    state.memory.store_memory("test_key", "test_value_for_recall_regression")

    with patch.object(
        state.memory, "recall_memories", wraps=state.memory.recall_memories
    ) as spy:
        with harness.patch(agent, [harness.answer("done")]):
            agent.process_query("do something with memory")

    assert spy.called, (
        "REGRESSION R1: recall_memories() must be called during process_query(). "
        "memory.db was written but never read."
    )


# ---------------------------------------------------------------------------
# R2: knowledge.recall() must be called during process_query
# ---------------------------------------------------------------------------


def test_knowledge_recall_called_during_process_query(agent_workspace, harness):
    """
    Regression: knowledge.db was written to (insights stored) but
    knowledge.recall() was never called to surface those insights.
    """
    agent, ws = agent_workspace
    state = agent.shared_state

    state.knowledge.store_insight(
        category="regression",
        content="regression test insight content",
        domain="testing",
    )

    with patch.object(
        state.knowledge, "recall", wraps=state.knowledge.recall
    ) as spy:
        with harness.patch(agent, [harness.answer("done")]):
            agent.process_query("do something with knowledge")

    assert spy.called, (
        "REGRESSION R2: knowledge.recall() must be called during process_query(). "
        "knowledge.db was written but never read."
    )


# ---------------------------------------------------------------------------
# R3: skills.find_skills() must be called during process_query
# ---------------------------------------------------------------------------


def test_skills_find_called_during_process_query(agent_workspace, harness):
    """
    Regression: skills.db was written to (skills registered) but
    find_skills() was never called to surface them as context.
    """
    agent, ws = agent_workspace
    state = agent.shared_state

    with patch.object(
        state.skills, "find_skills", wraps=state.skills.find_skills
    ) as spy:
        with harness.patch(agent, [harness.answer("done")]):
            agent.process_query("run the debug workflow")

    assert spy.called, (
        "REGRESSION R3: find_skills() must be called during process_query(). "
        "skills.db was written but never read."
    )


# ---------------------------------------------------------------------------
# R4: knowledge.db must not be flooded with high_complexity entries
# ---------------------------------------------------------------------------


def test_knowledge_db_not_flooded_by_complexity_warnings(agent_workspace, harness):
    """
    Regression: codebase_index stored >1000 'high_complexity:...' entries in
    knowledge.db on every indexing run, filling it with noise.

    After a normal agent session (no indexing), the count of insights whose
    content starts with 'high_complexity:' must be <= 10.
    """
    agent, ws = agent_workspace
    state = agent.shared_state

    with harness.patch(agent, [harness.answer("done")]):
        agent.process_query("do a simple task")

    count = state.knowledge.conn.execute(
        "SELECT COUNT(*) FROM insights WHERE content LIKE 'high_complexity:%'"
    ).fetchone()[0]

    assert count <= 10, (
        f"REGRESSION R4: knowledge.db must not be flooded with high_complexity entries. "
        f"Found {count} entries (should be <= 10)."
    )


# ---------------------------------------------------------------------------
# R5: _auto_add_query_paths must be called exactly once per process_query
# ---------------------------------------------------------------------------


def test_auto_add_query_paths_called_once(agent_workspace, harness):
    """
    Regression: _auto_add_query_paths() was called twice per process_query()
    due to a duplicate line.  Each call scans the query for absolute paths
    and adds them to PathValidator — calling it twice causes redundant work
    and potential confusion.
    """
    agent, ws = agent_workspace

    with patch.object(
        agent, "_auto_add_query_paths", wraps=agent._auto_add_query_paths
    ) as spy:
        with harness.patch(agent, [harness.answer("done")]):
            agent.process_query("create /tmp/testfile.py")

    assert spy.call_count == 1, (
        f"REGRESSION R5: _auto_add_query_paths() must be called exactly once per "
        f"process_query(), but was called {spy.call_count} time(s)."
    )


# ---------------------------------------------------------------------------
# R6: Timestamps must use local time, not UTC
# ---------------------------------------------------------------------------


def test_timestamps_use_local_not_utc(agent_workspace, harness):
    """
    Regression: SQLite DEFAULT CURRENT_TIMESTAMP stores UTC, but the
    application expected local time.  All DB timestamps should be within
    60 seconds of datetime.now() (local time).

    We verify via the memory.db active_state table: store a memory, then
    read its stored_at timestamp and compare to the local clock.
    """
    agent, ws = agent_workspace
    state = agent.shared_state

    state.memory.store_memory("ts_test_key", "timestamp regression check")

    row = state.memory.conn.execute(
        "SELECT stored_at FROM active_state WHERE key = 'ts_test_key'"
    ).fetchone()

    assert row is not None, "The stored memory must be retrievable"

    stored_at_str = row[0]
    stored_at = datetime.strptime(stored_at_str, "%Y-%m-%d %H:%M:%S")
    local_now = datetime.now()
    local_now_str = local_now.strftime("%Y-%m-%d %H:%M:%S")

    diff_seconds = abs((local_now - stored_at).total_seconds())
    assert diff_seconds < 120, (
        f"REGRESSION R6: Timestamp must be local time (within 120 s of now). "
        f"Stored: {stored_at_str}, Local now: {local_now_str}, "
        f"Difference: {diff_seconds:.1f}s"
    )


# ---------------------------------------------------------------------------
# R7: EscalationLadder.get_action() must return "cloud" not "alternative"
# ---------------------------------------------------------------------------


def test_escalation_ladder_returns_cloud_not_alternative():
    """
    Regression: EscalationLadder.get_action() returned "alternative" at
    escalation level 2 instead of "cloud", causing the agent to loop
    indefinitely trying 'alternative' approaches instead of escalating.

    After 3 increments the action must be "cloud" (or "ask_user" if already
    past cloud level).
    """
    from gaia.agents.base.quality_gates import EscalationLadder

    ladder = EscalationLadder()
    assert ladder.get_action() != "alternative", (
        "REGRESSION R7: Initial action must not be 'alternative'."
    )

    ladder.increment()
    ladder.increment()
    ladder.increment()

    action = ladder.get_action()
    assert action in ("cloud", "ask_user"), (
        f"REGRESSION R7: After 3 increments, get_action() must return 'cloud' "
        f"or 'ask_user', got: {action!r}."
    )
    assert action != "alternative", (
        f"REGRESSION R7: get_action() must never return 'alternative', got: {action!r}."
    )
