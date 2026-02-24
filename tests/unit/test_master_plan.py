# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Comprehensive unit tests for MasterPlan.

Covers:
- Plan lifecycle (create, complete, abandon)
- Task creation (depth auto-compute, order_index, event recording)
- Status transitions (start, complete, fail, block, assign)
- Tree queries (get_task, get_plan_tasks, get_active_plan)
- get_summary() tree rendering (DFS, icons, indentation, agent labels)
- plan_task_events audit trail
- Multi-agent parallel ownership
- Backward compatibility (get_all_tasks → TaskNode)
- Edge cases (non-existent IDs, no-op calls, long titles)
"""

import tempfile
from pathlib import Path

import pytest

from gaia.agents.base.shared_state import MemoryDB, MasterPlan, TaskNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plan(tmp_path):
    """Fresh MasterPlan backed by a temp memory.db."""
    db = MemoryDB(tmp_path / "memory.db")
    return MasterPlan(db)


@pytest.fixture
def plan_id(plan):
    """A plan pre-created for tests that need one immediately."""
    return plan.create_plan("Test plan")


# ---------------------------------------------------------------------------
# Plan lifecycle
# ---------------------------------------------------------------------------


class TestPlanLifecycle:

    def test_create_plan_returns_uuid(self, plan):
        pid = plan.create_plan("Build something")
        assert isinstance(pid, str)
        assert len(pid) == 36  # UUID format
        assert pid.count("-") == 4

    def test_create_plan_stores_fields(self, plan):
        pid = plan.create_plan(
            "My project",
            project_dir="/code/my-project",
            target_dir="/output/my-project",
        )
        active = plan.get_active_plan()
        assert active is not None
        assert active["id"] == pid
        assert active["title"] == "My project"
        assert active["status"] == "active"
        assert active["project_dir"] == "/code/my-project"
        assert active["target_dir"] == "/output/my-project"
        assert active["completed_at"] is None

    def test_complete_plan(self, plan, plan_id):
        plan.complete_plan(plan_id)
        active = plan.get_active_plan()
        assert active["status"] == "completed"
        assert active["completed_at"] is not None

    def test_abandon_plan(self, plan, plan_id):
        plan.abandon_plan(plan_id)
        active = plan.get_active_plan()
        assert active["status"] == "abandoned"

    def test_get_active_plan_no_plans(self, plan):
        assert plan.get_active_plan() is None

    def test_get_active_plan_returns_most_recent(self, plan):
        plan.create_plan("First plan")
        pid2 = plan.create_plan("Second plan")
        active = plan.get_active_plan()
        assert active["id"] == pid2
        assert active["title"] == "Second plan"

    def test_title_truncated_to_500_chars(self, plan):
        long_title = "x" * 600
        pid = plan.create_plan(long_title)
        active = plan.get_active_plan()
        assert len(active["title"]) == 500


# ---------------------------------------------------------------------------
# Task creation
# ---------------------------------------------------------------------------


class TestTaskCreation:

    def test_create_task_returns_uuid(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Do something")
        assert isinstance(tid, str)
        assert len(tid) == 36

    def test_create_task_stored_with_correct_fields(self, plan, plan_id):
        tid = plan.create_task(
            plan_id, "Write tests",
            description="Cover all edge cases",
            priority=2,
            created_by="gaia-code",
        )
        task = plan.get_task(tid)
        assert task["title"] == "Write tests"
        assert task["description"] == "Cover all edge cases"
        assert task["priority"] == 2
        assert task["created_by"] == "gaia-code"
        assert task["status"] == "pending"
        assert task["plan_id"] == plan_id

    def test_root_task_defaults_to_depth_zero(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Milestone")
        task = plan.get_task(tid)
        assert task["depth"] == 0
        assert task["parent_id"] is None

    def test_child_task_depth_auto_computed(self, plan, plan_id):
        parent = plan.create_task(plan_id, "Milestone", depth=0)
        child = plan.create_task(plan_id, "Task", parent_id=parent)
        grandchild = plan.create_task(plan_id, "Subtask", parent_id=child)

        assert plan.get_task(child)["depth"] == 1
        assert plan.get_task(grandchild)["depth"] == 2

    def test_explicit_depth_overrides_auto(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Milestone", depth=0)
        task = plan.get_task(tid)
        assert task["depth"] == 0

    def test_order_index_increments_per_parent(self, plan, plan_id):
        # Root siblings
        t1 = plan.create_task(plan_id, "First")
        t2 = plan.create_task(plan_id, "Second")
        t3 = plan.create_task(plan_id, "Third")
        assert plan.get_task(t1)["order_index"] == 0
        assert plan.get_task(t2)["order_index"] == 1
        assert plan.get_task(t3)["order_index"] == 2

    def test_order_index_independent_per_parent(self, plan, plan_id):
        # Children of different parents each start at 0
        m1 = plan.create_task(plan_id, "Milestone A")
        m2 = plan.create_task(plan_id, "Milestone B")
        c1 = plan.create_task(plan_id, "Child of A", parent_id=m1)
        c2 = plan.create_task(plan_id, "Child of B", parent_id=m2)
        assert plan.get_task(c1)["order_index"] == 0
        assert plan.get_task(c2)["order_index"] == 0

    def test_create_task_records_created_event(self, plan, plan_id):
        plan.create_task(plan_id, "A task", created_by="test-agent")
        events = plan.memory_db.conn.execute(
            "SELECT event_type, agent_name FROM plan_task_events WHERE plan_id = ?",
            (plan_id,),
        ).fetchall()
        assert any(e[0] == "created" and e[1] == "test-agent" for e in events)

    def test_title_truncated_to_500_chars(self, plan, plan_id):
        long_title = "t" * 600
        tid = plan.create_task(plan_id, long_title)
        assert len(plan.get_task(tid)["title"]) == 500


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------


class TestStatusTransitions:

    def test_start_task(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Work item")
        plan.start_task(tid, owner="gaia-code")
        task = plan.get_task(tid)
        assert task["status"] == "in_progress"
        assert task["started_at"] is not None
        assert task["owner"] == "gaia-code"

    def test_complete_task(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Work item")
        plan.start_task(tid)
        plan.complete_task(tid, result="Done!", agent="gaia-code")
        task = plan.get_task(tid)
        assert task["status"] == "completed"
        assert task["result"] == "Done!"
        assert task["completed_at"] is not None
        assert task["owner"] == "gaia-code"

    def test_fail_task(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Risky item")
        plan.start_task(tid)
        plan.fail_task(tid, error="Syntax error on line 5", agent="gaia-code")
        task = plan.get_task(tid)
        assert task["status"] == "failed"
        assert task["error"] == "Syntax error on line 5"
        assert task["completed_at"] is not None

    def test_block_task(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Blocked item")
        plan.block_task(tid, reason="Waiting for dependency")
        task = plan.get_task(tid)
        assert task["status"] == "blocked"
        assert task["error"] == "Waiting for dependency"

    def test_assign_task_does_not_change_status(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Unassigned")
        plan.assign_task(tid, owner="specialist-agent")
        task = plan.get_task(tid)
        assert task["owner"] == "specialist-agent"
        assert task["status"] == "pending"  # unchanged

    def test_update_task_status_general(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Generic item")
        plan.update_task_status(tid, "in_progress", owner="agent-x")
        plan.update_task_status(tid, "completed", result="All good", owner="agent-x")
        task = plan.get_task(tid)
        assert task["status"] == "completed"
        assert task["result"] == "All good"

    def test_update_nonexistent_task_is_noop(self, plan):
        # Should not raise
        plan.update_task_status("nonexistent-id-12345", "completed")

    def test_start_task_sets_started_at_only_once(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Item")
        plan.start_task(tid)
        first_started = plan.get_task(tid)["started_at"]
        # start again (shouldn't overwrite started_at via direct update)
        plan.update_task_status(tid, "in_progress")
        # started_at should not change
        assert plan.get_task(tid)["started_at"] == first_started

    def test_multiple_status_transitions_all_record_events(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Eventful item")
        plan.start_task(tid, owner="agent-a")
        plan.fail_task(tid, error="oops")
        plan.start_task(tid, owner="agent-a")  # retry
        plan.complete_task(tid, result="fixed")

        events = plan.memory_db.conn.execute(
            "SELECT event_type FROM plan_task_events WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
        event_types = [e[0] for e in events]
        assert "created" in event_types
        assert "in_progress" in event_types
        assert "failed" in event_types
        assert "completed" in event_types


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestQueries:

    def test_get_task_returns_dict(self, plan, plan_id):
        tid = plan.create_task(plan_id, "A task")
        task = plan.get_task(tid)
        assert isinstance(task, dict)
        expected_keys = {
            "id", "plan_id", "parent_id", "title", "description", "status",
            "priority", "depth", "owner", "created_by", "result", "error",
            "dependencies", "order_index", "created_at", "updated_at",
            "started_at", "completed_at",
        }
        assert expected_keys.issubset(task.keys())

    def test_get_task_nonexistent_returns_none(self, plan):
        assert plan.get_task("does-not-exist") is None

    def test_get_plan_tasks_returns_all_for_plan(self, plan):
        pid1 = plan.create_plan("Plan A")
        pid2 = plan.create_plan("Plan B")
        plan.create_task(pid1, "A1")
        plan.create_task(pid1, "A2")
        plan.create_task(pid2, "B1")

        tasks_a = plan.get_plan_tasks(pid1)
        tasks_b = plan.get_plan_tasks(pid2)
        assert len(tasks_a) == 2
        assert len(tasks_b) == 1
        assert all(t["plan_id"] == pid1 for t in tasks_a)

    def test_get_plan_tasks_ordered_by_depth_then_order_index(self, plan, plan_id):
        m1 = plan.create_task(plan_id, "M1", depth=0)
        m2 = plan.create_task(plan_id, "M2", depth=0)
        plan.create_task(plan_id, "M1-child1", parent_id=m1)
        plan.create_task(plan_id, "M1-child2", parent_id=m1)

        tasks = plan.get_plan_tasks(plan_id)
        titles = [t["title"] for t in tasks]
        # Both milestones before subtasks
        assert titles.index("M1") < titles.index("M1-child1")
        assert titles.index("M2") < titles.index("M1-child1")

    def test_get_active_plan_includes_tasks(self, plan, plan_id):
        plan.create_task(plan_id, "Task 1")
        plan.create_task(plan_id, "Task 2")
        active = plan.get_active_plan()
        assert len(active["tasks"]) == 2

    def test_get_active_plan_tasks_are_dicts(self, plan, plan_id):
        plan.create_task(plan_id, "Task")
        tasks = plan.get_active_plan()["tasks"]
        assert all(isinstance(t, dict) for t in tasks)

    def test_dependencies_stored_and_retrieved_as_list(self, plan, plan_id):
        t1 = plan.create_task(plan_id, "First")
        t2 = plan.create_task(plan_id, "Second", dependencies=[t1])
        task = plan.get_task(t2)
        assert task["dependencies"] == [t1]

    def test_empty_dependencies_is_empty_list(self, plan, plan_id):
        tid = plan.create_task(plan_id, "No deps")
        assert plan.get_task(tid)["dependencies"] == []


# ---------------------------------------------------------------------------
# get_summary() tree rendering
# ---------------------------------------------------------------------------


class TestGetSummary:

    def test_empty_plan_returns_empty_string(self, plan):
        assert plan.get_summary() == ""

    def test_plan_with_no_tasks_returns_empty_string(self, plan, plan_id):
        assert plan.get_summary() == ""

    def test_summary_starts_with_plan_title(self, plan, plan_id):
        plan.create_task(plan_id, "Do something")
        summary = plan.get_summary()
        assert summary.startswith("## Active Plan: Test plan")

    def test_status_icons_correct(self, plan, plan_id):
        m = plan.create_task(plan_id, "Milestone")
        t_pending   = plan.create_task(plan_id, "Pending",   parent_id=m)
        t_running   = plan.create_task(plan_id, "Running",   parent_id=m)
        t_done      = plan.create_task(plan_id, "Done",      parent_id=m)
        t_failed    = plan.create_task(plan_id, "Failed",    parent_id=m)
        t_blocked   = plan.create_task(plan_id, "Blocked",   parent_id=m)

        plan.start_task(t_running)
        plan.complete_task(t_done)
        plan.fail_task(t_failed)
        plan.block_task(t_blocked)

        summary = plan.get_summary()
        assert "○ Pending" in summary
        assert "◉ Running" in summary
        assert "✓ Done" in summary
        assert "✗ Failed" in summary
        assert "⊘ Blocked" in summary

    def test_subtasks_appear_under_their_milestone(self, plan, plan_id):
        m1 = plan.create_task(plan_id, "Setup")
        m2 = plan.create_task(plan_id, "Testing")
        plan.create_task(plan_id, "Create dirs",    parent_id=m1)
        plan.create_task(plan_id, "Write fixtures", parent_id=m2)

        summary = plan.get_summary()
        lines = summary.splitlines()

        setup_idx    = next(i for i, l in enumerate(lines) if "Setup" in l)
        testing_idx  = next(i for i, l in enumerate(lines) if "Testing" in l)
        dirs_idx     = next(i for i, l in enumerate(lines) if "Create dirs" in l)
        fixture_idx  = next(i for i, l in enumerate(lines) if "Write fixtures" in l)

        # "Create dirs" must appear between "Setup" and "Testing"
        assert setup_idx < dirs_idx < testing_idx
        # "Write fixtures" must appear after "Testing"
        assert testing_idx < fixture_idx

    def test_indentation_reflects_depth(self, plan, plan_id):
        m   = plan.create_task(plan_id, "Milestone", depth=0)
        t   = plan.create_task(plan_id, "Task",      parent_id=m)
        st  = plan.create_task(plan_id, "Subtask",   parent_id=t)

        summary = plan.get_summary()
        lines = {l.lstrip(): l for l in summary.splitlines()}

        milestone_line = lines["○ Milestone"]
        task_line      = lines["○ Task"]
        subtask_line   = lines["○ Subtask"]

        assert not milestone_line.startswith(" ")     # depth 0: no indent
        assert task_line.startswith("  ")             # depth 1: 2 spaces
        assert subtask_line.startswith("    ")        # depth 2: 4 spaces

    def test_agent_owner_shown_in_brackets(self, plan, plan_id):
        tid = plan.create_task(plan_id, "My task")
        plan.start_task(tid, owner="security-specialist")
        summary = plan.get_summary()
        assert "[security-specialist]" in summary

    def test_no_owner_shows_no_brackets(self, plan, plan_id):
        plan.create_task(plan_id, "Unowned task")
        summary = plan.get_summary()
        assert "[" not in summary

    def test_sibling_order_preserved(self, plan, plan_id):
        m = plan.create_task(plan_id, "Milestone")
        plan.create_task(plan_id, "Alpha",   parent_id=m)
        plan.create_task(plan_id, "Beta",    parent_id=m)
        plan.create_task(plan_id, "Gamma",   parent_id=m)

        summary = plan.get_summary()
        lines = summary.splitlines()
        titles = [l.strip().split(" ", 1)[1].split(" [")[0] for l in lines if l.strip().startswith("○")]
        assert titles.index("Alpha") < titles.index("Beta") < titles.index("Gamma")

    def test_completed_plan_still_has_summary(self, plan, plan_id):
        plan.create_task(plan_id, "Final task")
        plan.complete_plan(plan_id)
        # get_active_plan returns most recent regardless of status
        assert plan.get_summary() != ""


# ---------------------------------------------------------------------------
# plan_task_events audit trail
# ---------------------------------------------------------------------------


class TestAuditTrail:

    def test_create_task_writes_created_event(self, plan, plan_id):
        tid = plan.create_task(plan_id, "A task", created_by="root-agent")
        events = plan.memory_db.conn.execute(
            "SELECT event_type, agent_name, details FROM plan_task_events WHERE task_id = ?",
            (tid,),
        ).fetchall()
        assert len(events) == 1
        assert events[0][0] == "created"
        assert events[0][1] == "root-agent"
        assert "A task" in events[0][2]

    def test_assign_writes_assigned_event(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Task")
        plan.assign_task(tid, owner="worker-agent")
        events = plan.memory_db.conn.execute(
            "SELECT event_type, agent_name FROM plan_task_events WHERE task_id = ? AND event_type = 'assigned'",
            (tid,),
        ).fetchall()
        assert len(events) == 1
        assert events[0][1] == "worker-agent"

    def test_each_status_transition_writes_event(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Task")
        plan.start_task(tid, owner="agent")
        plan.complete_task(tid, result="ok", agent="agent")

        event_types = [
            row[0] for row in plan.memory_db.conn.execute(
                "SELECT event_type FROM plan_task_events WHERE task_id = ? ORDER BY id",
                (tid,),
            ).fetchall()
        ]
        assert event_types == ["created", "in_progress", "completed"]

    def test_events_have_correct_plan_id(self, plan):
        pid1 = plan.create_plan("Plan A")
        pid2 = plan.create_plan("Plan B")
        tid1 = plan.create_task(pid1, "In A")
        tid2 = plan.create_task(pid2, "In B")
        plan.start_task(tid1)
        plan.start_task(tid2)

        events_a = plan.memory_db.conn.execute(
            "SELECT COUNT(*) FROM plan_task_events WHERE plan_id = ?", (pid1,)
        ).fetchone()[0]
        events_b = plan.memory_db.conn.execute(
            "SELECT COUNT(*) FROM plan_task_events WHERE plan_id = ?", (pid2,)
        ).fetchone()[0]
        assert events_a == 2  # created + in_progress
        assert events_b == 2


# ---------------------------------------------------------------------------
# Multi-agent scenarios
# ---------------------------------------------------------------------------


class TestMultiAgent:

    def test_parallel_tasks_different_owners(self, plan, plan_id):
        t1 = plan.create_task(plan_id, "API work")
        t2 = plan.create_task(plan_id, "Auth work")
        plan.start_task(t1, owner="api-specialist")
        plan.start_task(t2, owner="security-specialist")

        assert plan.get_task(t1)["owner"] == "api-specialist"
        assert plan.get_task(t2)["owner"] == "security-specialist"
        assert plan.get_task(t1)["status"] == "in_progress"
        assert plan.get_task(t2)["status"] == "in_progress"

    def test_retry_task_is_independent(self, plan, plan_id):
        m = plan.create_task(plan_id, "Test milestone")
        t_fail = plan.create_task(plan_id, "Integration tests", parent_id=m)
        plan.fail_task(t_fail, error="fixture missing")

        # Create a retry as a new sibling
        t_retry = plan.create_task(
            plan_id, "Integration tests (retry)",
            parent_id=m,
            description="Fixed fixture",
        )
        plan.complete_task(t_retry, result="23 tests passing")

        assert plan.get_task(t_fail)["status"] == "failed"
        assert plan.get_task(t_retry)["status"] == "completed"

        # Both appear as siblings in the milestone's children
        tasks = plan.get_plan_tasks(plan_id)
        children = [t for t in tasks if t["parent_id"] == m]
        assert len(children) == 2

    def test_owner_transfers_across_agents(self, plan, plan_id):
        tid = plan.create_task(plan_id, "Handoff task")
        plan.assign_task(tid, owner="agent-a")
        plan.start_task(tid, owner="agent-b")   # agent-b takes over
        assert plan.get_task(tid)["owner"] == "agent-b"

    def test_mixed_status_across_plan(self, plan, plan_id):
        t_done    = plan.create_task(plan_id, "Done")
        t_running = plan.create_task(plan_id, "Running")
        t_pending = plan.create_task(plan_id, "Pending")
        t_failed  = plan.create_task(plan_id, "Failed")

        plan.complete_task(t_done)
        plan.start_task(t_running)
        plan.fail_task(t_failed)

        tasks = {t["title"]: t["status"] for t in plan.get_plan_tasks(plan_id)}
        assert tasks["Done"] == "completed"
        assert tasks["Running"] == "in_progress"
        assert tasks["Pending"] == "pending"
        assert tasks["Failed"] == "failed"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:

    def test_get_all_tasks_returns_task_nodes(self, plan, plan_id):
        plan.create_task(plan_id, "Task A")
        plan.create_task(plan_id, "Task B")
        tasks = plan.get_all_tasks()
        assert all(isinstance(t, TaskNode) for t in tasks)

    def test_get_all_tasks_count_matches(self, plan, plan_id):
        for i in range(5):
            plan.create_task(plan_id, f"Task {i}")
        tasks = plan.get_all_tasks()
        assert len(tasks) == 5

    def test_task_node_has_required_attributes(self, plan, plan_id):
        tid = plan.create_task(plan_id, "A task")
        plan.start_task(tid, owner="some-agent")
        node = plan.get_all_tasks()[0]
        assert hasattr(node, "id")
        assert hasattr(node, "description")
        assert hasattr(node, "status")
        assert hasattr(node, "owner")
        assert hasattr(node, "parent_id")

    def test_task_node_description_maps_from_title(self, plan, plan_id):
        plan.create_task(plan_id, "My title here")
        node = plan.get_all_tasks()[0]
        assert node.description == "My title here"

    def test_task_node_to_dict(self, plan, plan_id):
        plan.create_task(plan_id, "Checkpointable task")
        node = plan.get_all_tasks()[0]
        d = node.to_dict()
        assert isinstance(d, dict)
        assert "id" in d
        assert "description" in d
        assert "status" in d

    def test_get_all_tasks_empty_when_no_plan(self, plan):
        assert plan.get_all_tasks() == []


# ---------------------------------------------------------------------------
# Isolation: separate plans don't bleed into each other
# ---------------------------------------------------------------------------


class TestIsolation:

    def test_tasks_isolated_per_plan(self, plan):
        pid1 = plan.create_plan("Plan A")
        pid2 = plan.create_plan("Plan B")
        plan.create_task(pid1, "Only in A")
        plan.create_task(pid2, "Only in B")

        tasks_a = plan.get_plan_tasks(pid1)
        tasks_b = plan.get_plan_tasks(pid2)
        assert all(t["title"] == "Only in A" for t in tasks_a)
        assert all(t["title"] == "Only in B" for t in tasks_b)

    def test_complete_task_in_one_plan_doesnt_affect_other(self, plan):
        pid1 = plan.create_plan("Plan 1")
        pid2 = plan.create_plan("Plan 2")
        t1 = plan.create_task(pid1, "Task in 1")
        t2 = plan.create_task(pid2, "Task in 2")

        plan.complete_task(t1)
        assert plan.get_task(t2)["status"] == "pending"

    def test_two_memory_db_instances_are_independent(self, tmp_path):
        db_a = MemoryDB(tmp_path / "a.db")
        db_b = MemoryDB(tmp_path / "b.db")
        plan_a = MasterPlan(db_a)
        plan_b = MasterPlan(db_b)

        pid_a = plan_a.create_plan("Plan in A")
        plan_a.create_task(pid_a, "Task in A")

        assert plan_b.get_active_plan() is None
