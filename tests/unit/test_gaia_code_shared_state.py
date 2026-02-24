# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit tests for SharedAgentState (RAC Foundation).

Tests:
- Singleton pattern
- Database initialization
- Memory operations
- Knowledge operations
- Master plan operations
- Call stack operations
- Message queue operations
"""

import tempfile
from pathlib import Path

import pytest

from gaia.agents.gaia_code.shared_state import (
    AgentCallStack,
    KnowledgeDB,
    MasterPlan,
    MemoryDB,
    MessageQueue,
    ProjectManifest,
    SharedAgentState,
    get_shared_state,
)


class TestMemoryDB:
    """Test MemoryDB (session-scoped cache)."""

    def test_file_cache(self):
        """Test file caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            memory = MemoryDB(db_path)

            # Cache a file
            memory.cache_file("test.py", "print('hello')")

            # Retrieve it
            content = memory.get_file("test.py")
            assert content == "print('hello')"

            # Non-existent file
            assert memory.get_file("nonexistent.py") is None

    def test_tool_results(self):
        """Test tool result storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "memory.db"
            memory = MemoryDB(db_path)

            # Store tool result
            memory.store_tool_result(
                "read_file", {"path": "test.py"}, "File contents here"
            )

            # Verify it was stored (would need to add a get_tool_results method)


class TestKnowledgeDB:
    """Test KnowledgeDB (cross-session learning)."""

    def test_store_insight(self):
        """Test insight storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge.db"
            knowledge = KnowledgeDB(db_path)

            # Store an insight
            insight_id = knowledge.store_insight(
                category="error_fix",
                content="Always check if variable is None before accessing attributes",
                domain="python",
                triggers=["AttributeError", "None"],
            )

            assert insight_id is not None

    def test_recall(self):
        """Test FTS5 search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge.db"
            knowledge = KnowledgeDB(db_path)

            # Store some insights
            knowledge.store_insight(
                category="error_fix",
                content="Check for None before accessing .attribute",
                triggers=["AttributeError", "None"],
            )

            knowledge.store_insight(
                category="pattern",
                content="Use list comprehension for filtering",
                triggers=["filter", "list"],
            )

            # Search
            results = knowledge.recall("AttributeError")
            assert len(results) >= 1
            assert "None" in results[0]["content"]

    def test_preferences(self):
        """Test preference storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge.db"
            knowledge = KnowledgeDB(db_path)

            # Store preference
            knowledge.store_preference(
                "code_style", "black", "Use Black for formatting"
            )

            # Retrieve preference
            value = knowledge.get_preference("code_style")
            assert value == "black"

            # Non-existent preference
            assert knowledge.get_preference("nonexistent") is None


class TestMasterPlan:
    """Test MasterPlan (hierarchical task tree, stored in memory.db)."""

    def _make_plan(self, tmpdir):
        """Helper: create MemoryDB + MasterPlan for a test."""
        from gaia.agents.base.shared_state import MemoryDB
        db_path = Path(tmpdir) / "memory.db"
        memory_db = MemoryDB(db_path)
        plan = MasterPlan(memory_db)
        return plan

    def test_create_task(self):
        """Test task and plan creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self._make_plan(tmpdir)

            # Create a plan first
            plan_id = plan.create_plan("Build REST API")
            assert plan_id is not None

            # Create root milestone
            root_id = plan.create_task(plan_id, "Build REST API")
            assert root_id is not None
            root = plan.get_task(root_id)
            assert root["title"] == "Build REST API"
            assert root["status"] == "pending"
            assert root["depth"] == 0

            # Create child task
            child_id = plan.create_task(plan_id, "Create auth endpoints", parent_id=root_id)
            child = plan.get_task(child_id)
            assert child["parent_id"] == root_id
            assert child["depth"] == 1

    def test_update_task_status(self):
        """Test task status updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self._make_plan(tmpdir)

            plan_id = plan.create_plan("Test")
            task_id = plan.create_task(plan_id, "Write tests")

            # Start task
            plan.update_task_status(task_id, "in_progress")
            task = plan.get_task(task_id)
            assert task["status"] == "in_progress"
            assert task["started_at"] is not None

            # Complete task
            plan.update_task_status(task_id, "completed", result="All tests passing")
            task = plan.get_task(task_id)
            assert task["status"] == "completed"
            assert task["result"] == "All tests passing"
            assert task["completed_at"] is not None

    def test_get_all_tasks(self):
        """Test getting all tasks via backward-compat get_all_tasks()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = self._make_plan(tmpdir)

            plan_id = plan.create_plan("Test")
            plan.create_task(plan_id, "Task 1")
            plan.create_task(plan_id, "Task 2")
            plan.create_task(plan_id, "Task 3")

            # get_all_tasks() returns TaskNode objects for backward compat
            tasks = plan.get_all_tasks()
            assert len(tasks) == 3


class TestAgentCallStack:
    """Test AgentCallStack (recursion tracking)."""

    def test_push_pop(self):
        """Test basic push/pop operations."""
        stack = AgentCallStack(max_depth=5)

        # Push first frame (root)
        frame1 = stack.push("Root task")
        assert frame1 is not None
        assert frame1.depth == 0
        assert frame1.parent_id is None

        # Push second frame (child)
        frame2 = stack.push("Child task")
        assert frame2 is not None
        assert frame2.depth == 1
        assert frame2.parent_id == frame1.agent_id

        # Current frame should be frame2
        current = stack.current()
        assert current.agent_id == frame2.agent_id

        # Pop frame2
        popped = stack.pop()
        assert popped.agent_id == frame2.agent_id

        # Current should now be frame1
        current = stack.current()
        assert current.agent_id == frame1.agent_id

    def test_max_depth(self):
        """Test max recursion depth limit."""
        stack = AgentCallStack(max_depth=3)

        # Push 3 frames (should succeed)
        frame1 = stack.push("Task 1")
        frame2 = stack.push("Task 2")
        frame3 = stack.push("Task 3")
        assert all([frame1, frame2, frame3])

        # Push 4th frame (should fail)
        frame4 = stack.push("Task 4")
        assert frame4 is None


class TestMessageQueue:
    """Test MessageQueue (async communication)."""

    def test_send_receive(self):
        """Test basic message sending and receiving."""
        queue = MessageQueue()

        # Send message
        msg_id = queue.send(
            content="Hello user",
            priority="FYI",
            sender="agent",
            recipient="user",
        )
        assert msg_id is not None

        # Receive messages
        messages = queue.receive(recipient="user")
        assert len(messages) == 1
        assert messages[0].content == "Hello user"
        assert messages[0].priority == "FYI"

        # Messages should be marked as read
        assert messages[0].read is True

        # Receiving again should return empty list
        messages = queue.receive(recipient="user")
        assert len(messages) == 0

    def test_priority_filter(self):
        """Test filtering by priority."""
        queue = MessageQueue()

        # Send messages with different priorities
        queue.send("FYI message", priority="FYI", recipient="user")
        queue.send("Question message", priority="Question", recipient="user")
        queue.send("Decision message", priority="Decision", recipient="user")

        # Receive only Questions
        messages = queue.receive(recipient="user", priority="Question")
        assert len(messages) == 1
        assert messages[0].content == "Question message"

    def test_respond(self):
        """Test responding to messages."""
        queue = MessageQueue()

        # Send message
        msg_id = queue.send("Need help", priority="Question", recipient="user")

        # Respond
        queue.respond(msg_id, "Here's the answer")

        # Verify response was stored
        messages = queue.receive(recipient="user")
        assert messages[0].response == "Here's the answer"


class TestProjectManifest:
    """Test ProjectManifest (live project state)."""

    def test_add_file(self):
        """Test file tracking."""
        manifest = ProjectManifest()

        # Add file
        manifest.add_file("main.py", "print('hello')")
        assert "main.py" in manifest.list_files()

        # Get file info
        info = manifest.get_file("main.py")
        assert info["content"] == "print('hello')"
        assert "created" in info
        assert "modified" in info

    def test_add_api(self):
        """Test API endpoint tracking."""
        manifest = ProjectManifest()

        # Add API endpoint
        manifest.add_api(
            endpoint="/users",
            method="GET",
            params={"page": "int"},
            response={"users": "list"},
        )

        assert "/users" in manifest.apis
        assert manifest.apis["/users"]["method"] == "GET"

    def test_add_decision(self):
        """Test architecture decision tracking."""
        manifest = ProjectManifest()

        # Add decision
        manifest.add_decision(
            decision="Use FastAPI for backend",
            rationale="Fast, modern, type-safe",
        )

        assert len(manifest.decisions) == 1
        assert "FastAPI" in manifest.decisions[0]["decision"]


class TestSharedAgentState:
    """Test SharedAgentState (complete RAC foundation)."""

    def test_singleton(self):
        """Test singleton pattern - should return same instance."""
        state1 = SharedAgentState()
        state2 = SharedAgentState()
        assert state1 is state2

    def test_initialization(self):
        """Test all components are initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = SharedAgentState(workspace_dir=Path(tmpdir))

            # Verify all databases exist
            assert state.memory is not None
            assert state.knowledge is not None
            assert state.tools is not None
            assert state.skills is not None
            assert state.agents is not None

            # Verify plan and manifest
            assert state.plan is not None
            assert state.manifest is not None

            # Verify call stack and message queue
            assert state.call_stack is not None
            assert state.message_queue is not None

    def test_reset_session(self):
        """Test session reset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state = SharedAgentState(workspace_dir=Path(tmpdir))

            # Add some session data
            state.memory.cache_file("test.py", "content")
            state.call_stack.push("task")
            state.message_queue.send("message")

            # Store some knowledge (should persist)
            state.knowledge.store_insight("category", "content")

            # Reset session
            state.reset_session()

            # Session data should be cleared
            assert state.memory.get_file("test.py") is None
            assert state.call_stack.current() is None
            assert len(state.message_queue.receive()) == 0

            # Knowledge should persist
            results = state.knowledge.recall("content")
            assert len(results) > 0


class TestGetSharedState:
    """Test get_shared_state() helper function."""

    def test_returns_singleton(self):
        """Test that get_shared_state() returns the singleton."""
        state1 = get_shared_state()
        state2 = get_shared_state()
        assert state1 is state2
