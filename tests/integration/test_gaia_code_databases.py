# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Integration tests for all GAIA Code databases.

Validates that each database is fully functional:
- memory.db   : working memory — store/recall/forget/clear
- knowledge.db: long-term insights, preferences, learnings
- tools.db    : tool registry, FTS5 search, usage tracking
- skills.db   : skill registry, usage tracking
- agents.db   : specialist registry, usage tracking
- plan.db     : hierarchical task tree

Also tests:
- initialize_workspace() populates tools, agents, skills correctly
- reset_session() clears working memory but keeps persistent DBs
- All DB operations are idempotent on repeated startup
"""

import json
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the SharedAgentState singleton between tests."""
    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None
    yield
    SharedAgentState._instance = None


@pytest.fixture()
def workspace(tmp_path):
    """Return a fresh temporary workspace path."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture()
def state(workspace):
    """Return a fresh SharedAgentState backed by a temp workspace."""
    from gaia.agents.base.shared_state import get_shared_state

    return get_shared_state(workspace)


@pytest.fixture()
def initialized_state(workspace):
    """SharedAgentState with full workspace init (tools + agents + skills)."""
    from gaia.agents.gaia_code.integration import initialize_workspace

    initialize_workspace(workspace)
    from gaia.agents.base.shared_state import get_shared_state

    return get_shared_state(workspace)


# ===========================================================================
# memory.db
# ===========================================================================


class TestMemoryDB:
    """memory.db — working memory: active_state, file_cache, tool_results."""

    # --- active_state (agent working memory) --------------------------------

    def test_store_and_recall_memory(self, state):
        mem = state.memory
        mem.store_memory("project_root", "/mnt/c/Users/14255/Work/gaia")
        mem.store_memory("auth_approach", "JWT with RS256", tags=["architecture"])
        mem.store_memory("active_branch", "gaia-v2")

        all_memories = mem.recall_memories()
        assert len(all_memories) == 3

        keys = {m["key"] for m in all_memories}
        assert keys == {"project_root", "auth_approach", "active_branch"}

    def test_recall_memory_with_query(self, state):
        mem = state.memory
        mem.store_memory("auth_approach", "JWT with RS256")
        mem.store_memory("db_schema", "users(id, email)")
        mem.store_memory("error_fix", "Use check_same_thread=False")

        hits = mem.recall_memories(query="auth")
        assert len(hits) == 1
        assert hits[0]["key"] == "auth_approach"

    def test_get_memory_exact_key(self, state):
        mem = state.memory
        mem.store_memory("project_root", "/mnt/c/Work/gaia")

        val = mem.get_memory("project_root")
        assert val == "/mnt/c/Work/gaia"

        assert mem.get_memory("nonexistent") is None

    def test_store_memory_overwrites_existing_key(self, state):
        mem = state.memory
        mem.store_memory("branch", "main")
        mem.store_memory("branch", "gaia-v2")

        assert mem.get_memory("branch") == "gaia-v2"
        assert len(mem.recall_memories()) == 1  # no duplicate rows

    def test_forget_memory(self, state):
        mem = state.memory
        mem.store_memory("temp_key", "temp_value")
        assert mem.get_memory("temp_key") is not None

        deleted = mem.forget_memory("temp_key")
        assert deleted is True
        assert mem.get_memory("temp_key") is None

    def test_forget_nonexistent_key(self, state):
        deleted = state.memory.forget_memory("no_such_key")
        assert deleted is False

    def test_memory_tags_stored_and_returned(self, state):
        mem = state.memory
        mem.store_memory("decision", "Use SQLite for storage", tags=["architecture", "db"])

        results = mem.recall_memories(query="decision")
        assert results[0]["tags"] == ["architecture", "db"]

    # --- file_cache ---------------------------------------------------------

    def test_cache_file_and_retrieve(self, state):
        mem = state.memory
        mem.cache_file("/project/agent.py", "class Agent: pass")

        content = mem.get_file("/project/agent.py")
        assert content == "class Agent: pass"

    def test_cache_file_miss(self, state):
        assert state.memory.get_file("/nonexistent.py") is None

    def test_cache_file_overwrite(self, state):
        mem = state.memory
        mem.cache_file("/file.py", "old content")
        mem.cache_file("/file.py", "new content")

        assert mem.get_file("/file.py") == "new content"

        # Only one row per path
        count = mem.conn.execute(
            "SELECT COUNT(*) FROM file_cache WHERE path = '/file.py'"
        ).fetchone()[0]
        assert count == 1

    # --- tool_results -------------------------------------------------------

    def test_store_tool_result(self, state):
        mem = state.memory
        mem.store_tool_result("read_file", {"file_path": "/foo.py"}, "x = 1")
        mem.store_tool_result("run_pytest", {"path": "."}, '{"passed": 5}')

        count = mem.conn.execute("SELECT COUNT(*) FROM tool_results").fetchone()[0]
        assert count == 2

    def test_tool_results_record_all_calls(self, state):
        mem = state.memory
        for i in range(5):
            mem.store_tool_result("read_file", {"file_path": f"/file{i}.py"}, f"content {i}")

        rows = mem.conn.execute(
            "SELECT tool_name FROM tool_results WHERE tool_name = 'read_file'"
        ).fetchall()
        assert len(rows) == 5

    # --- clear_working_memory / reset_session -------------------------------

    def test_clear_working_memory(self, state, workspace):
        mem = state.memory
        mem.store_memory("key1", "value1")
        mem.cache_file("/file.py", "content")
        mem.store_tool_result("read_file", {}, "result")

        mem.clear_working_memory()

        assert len(mem.recall_memories()) == 0
        assert mem.get_file("/file.py") is None
        count = mem.conn.execute("SELECT COUNT(*) FROM tool_results").fetchone()[0]
        assert count == 0

    def test_reset_session_keeps_db_file(self, state, workspace):
        """DB file must persist after reset — history stays browsable."""
        mem = state.memory
        mem.store_memory("key", "value")

        state.reset_session()

        # File still exists
        assert (workspace / "memory.db").exists()

        # Tables are empty
        assert len(mem.recall_memories()) == 0

    def test_reset_session_does_not_affect_knowledge(self, state, workspace):
        """Resetting session must not touch knowledge.db."""
        state.knowledge.store_insight("pattern", "Always use pathlib.Path for file ops")

        state.reset_session()

        # knowledge still has the insight
        results = state.knowledge.recall("pathlib")
        assert len(results) == 1

    def test_reset_session_does_not_affect_tools(self, state):
        """Resetting session must not touch tools.db."""
        state.tools.register_tool("my_tool", "testing", "A test tool", "core")
        state.reset_session()

        tool = state.tools.get_tool("my_tool")
        assert tool is not None

    def test_reset_session_does_not_affect_skills(self, state):
        """Resetting session must not touch skills.db."""
        state.skills.register_skill(
            "my_skill", "A test skill", "testing", [{"step": 1, "action": "test"}]
        )
        state.reset_session()

        skills = state.skills.find_skills(category="testing")
        assert len(skills) == 1


# ===========================================================================
# knowledge.db
# ===========================================================================


class TestKnowledgeDB:
    """knowledge.db — long-term persistent learning."""

    def test_store_and_recall_insight(self, state):
        kb = state.knowledge
        insight_id = kb.store_insight(
            category="error_fix",
            content="Always use check_same_thread=False for SQLite in threaded code",
            domain="python",
            triggers=["sqlite", "threading"],
        )
        assert insight_id is not None

        results = kb.recall("sqlite threading")
        assert len(results) >= 1
        assert any("check_same_thread" in r["content"] for r in results)

    def test_recall_empty_query_returns_nothing(self, state):
        state.knowledge.store_insight("pattern", "Use pathlib")
        results = state.knowledge.recall("")
        assert results == []

    def test_store_preference_and_retrieve(self, state):
        kb = state.knowledge
        kb.store_preference("code_style", "black", "Use Black for formatting")

        val = kb.get_preference("code_style")
        assert val == "black"

    def test_get_preference_missing_returns_none(self, state):
        assert state.knowledge.get_preference("nonexistent") is None

    def test_store_preference_overwrites(self, state):
        state.knowledge.store_preference("style", "pep8")
        state.knowledge.store_preference("style", "black")
        assert state.knowledge.get_preference("style") == "black"

    def test_fts5_search_across_domain_and_category(self, state):
        kb = state.knowledge
        kb.store_insight("convention", "Use dataclasses for DTOs", domain="python")
        kb.store_insight("pattern", "Prefer composition over inheritance", domain="design")

        # FTS5 does exact-word matching — search for the full word present in content
        results = kb.recall("dataclasses")
        assert len(results) >= 1
        assert results[0]["category"] == "convention"

    def test_multiple_insights_recall_top_k(self, state):
        kb = state.knowledge
        for i in range(10):
            kb.store_insight("pattern", f"Pattern number {i} about testing", domain="testing")

        results = kb.recall("pattern testing", top_k=5)
        assert len(results) <= 5

    def test_insights_persist_across_session_reset(self, state):
        state.knowledge.store_insight("learning", "Always write tests first")
        state.reset_session()

        results = state.knowledge.recall("tests")
        assert len(results) >= 1


# ===========================================================================
# tools.db
# ===========================================================================


class TestToolsDB:
    """tools.db — tool registry, FTS5 search, usage tracking."""

    def test_register_and_get_tool(self, state):
        state.tools.register_tool(
            name="my_tool",
            category="testing",
            description="A tool for testing things",
            source="core",
        )
        tool = state.tools.get_tool("my_tool")
        assert tool is not None
        assert tool["name"] == "my_tool"
        assert tool["category"] == "testing"

    def test_get_nonexistent_tool_returns_none(self, state):
        assert state.tools.get_tool("no_such_tool") is None

    def test_fts5_search_finds_tools_by_description(self, state):
        state.tools.register_tool("git_commit", "git", "Create a git commit", "core")
        state.tools.register_tool("run_tests", "testing", "Run pytest unit tests", "core")

        results = state.tools.find_tools("git")
        assert any(t["name"] == "git_commit" for t in results)

        results = state.tools.find_tools("pytest")
        assert any(t["name"] == "run_tests" for t in results)

    def test_find_tools_empty_query_returns_nothing(self, state):
        state.tools.register_tool("tool_a", "cat", "A tool", "core")
        results = state.tools.find_tools("")
        assert results == []

    def test_record_and_read_usage(self, state):
        state.tools.register_tool("read_file", "file_io", "Read a file", "core")

        state.tools.record_usage("read_file", success=True, duration_ms=12)
        state.tools.record_usage("read_file", success=True, duration_ms=8)
        state.tools.record_usage("read_file", success=False, duration_ms=5, error="File not found")

        stats = state.tools.get_tool_stats("read_file")
        assert stats["total"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1

    def test_record_usage_unknown_tool_does_not_crash(self, state):
        # Should silently store with null tool_id
        state.tools.record_usage("unknown_tool", success=True, duration_ms=1)

    def test_initialize_workspace_registers_core_tools(self, initialized_state):
        count = initialized_state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools"
        ).fetchone()[0]
        assert count >= 22, f"Expected at least 22 tools, got {count}"

    def test_tools_persist_across_session_reset(self, state):
        state.tools.register_tool("persistent_tool", "testing", "A tool", "core")
        state.reset_session()
        assert state.tools.get_tool("persistent_tool") is not None


# ===========================================================================
# skills.db
# ===========================================================================


class TestSkillsDB:
    """skills.db — learned workflow patterns."""

    def test_register_and_find_skill(self, state):
        state.skills.register_skill(
            name="debug_flow",
            description="Read file, fix error, verify",
            category="debugging",
            steps=[
                {"step": 1, "action": "read_file"},
                {"step": 2, "action": "edit_file"},
                {"step": 3, "action": "run_pytest"},
            ],
            domain="python",
            tools_used=["read_file", "edit_file", "run_pytest"],
        )

        skills = state.skills.find_skills(category="debugging")
        assert len(skills) == 1
        assert skills[0]["name"] == "debug_flow"
        assert len(skills[0]["steps"]) == 3

    def test_find_skills_by_domain(self, state):
        state.skills.register_skill(
            "py_skill", "A python skill", "coding",
            [{"step": 1, "action": "code"}], domain="python"
        )
        state.skills.register_skill(
            "js_skill", "A js skill", "coding",
            [{"step": 1, "action": "code"}], domain="javascript"
        )

        py_skills = state.skills.find_skills(domain="python")
        assert len(py_skills) == 1
        assert py_skills[0]["name"] == "py_skill"

    def test_record_usage_updates_confidence(self, state):
        state.skills.register_skill(
            "test_skill", "Test skill", "testing", [{"step": 1, "action": "test"}]
        )

        state.skills.record_usage("test_skill", success=True)
        state.skills.record_usage("test_skill", success=True)
        state.skills.record_usage("test_skill", success=False)

        skills = state.skills.find_skills(category="testing")
        skill = skills[0]
        # 2 successes out of 3 = 0.666...
        assert abs(skill["confidence"] - 2 / 3) < 0.01

    def test_record_usage_unknown_skill_does_not_crash(self, state):
        state.skills.record_usage("no_such_skill", success=True)

    def test_initialize_workspace_registers_8_skills(self, initialized_state):
        count = initialized_state.skills.conn.execute(
            "SELECT COUNT(*) FROM skills"
        ).fetchone()[0]
        assert count == 8

    def test_initial_skills_have_correct_categories(self, initialized_state):
        categories = {
            row[0]
            for row in initialized_state.skills.conn.execute(
                "SELECT DISTINCT category FROM skills"
            ).fetchall()
        }
        assert "debugging" in categories
        assert "testing" in categories
        assert "coding" in categories
        assert "documentation" in categories

    def test_initialize_workspace_idempotent(self, workspace):
        """Calling initialize_workspace twice must not duplicate skills."""
        from gaia.agents.gaia_code.integration import initialize_workspace

        initialize_workspace(workspace)

        # Reset singleton so second call sees same DB
        from gaia.agents.base.shared_state import SharedAgentState

        SharedAgentState._instance = None
        initialize_workspace(workspace)

        from gaia.agents.base.shared_state import get_shared_state

        count = get_shared_state(workspace).skills.conn.execute(
            "SELECT COUNT(*) FROM skills"
        ).fetchone()[0]
        assert count == 8  # not 16

    def test_skills_persist_across_session_reset(self, state):
        state.skills.register_skill(
            "persistent_skill", "Persists", "coding", [{"step": 1, "action": "x"}]
        )
        state.reset_session()
        skills = state.skills.find_skills(category="coding")
        assert len(skills) == 1


# ===========================================================================
# agents.db
# ===========================================================================


class TestAgentsDB:
    """agents.db — specialist agent registry."""

    def test_register_and_find_agent(self, state):
        state.agents.register_agent(
            name="DebuggerAgent",
            description="Finds and fixes bugs in Python code",
            capabilities=["debug", "trace", "fix"],
            system_prompt="You are an expert debugger.",
            tool_packs=["file_io", "execution"],
        )

        agent = state.agents.find_agent("DebuggerAgent")
        assert agent is not None
        assert agent["name"] == "DebuggerAgent"
        assert "debug" in agent["capabilities"]

    def test_find_nonexistent_agent_returns_none(self, state):
        assert state.agents.find_agent("NoSuchAgent") is None

    def test_list_all_agents(self, state):
        state.agents.register_agent("AgentA", "First", capabilities=["a"])
        state.agents.register_agent("AgentB", "Second", capabilities=["b"])

        agents = state.agents.list_agents()
        assert len(agents) == 2
        names = {a["name"] for a in agents}
        assert names == {"AgentA", "AgentB"}

    def test_record_usage_updates_confidence(self, state):
        state.agents.register_agent("TestAgent", "A test agent")

        state.agents.record_usage("TestAgent", success=True, task_type="debugging")
        state.agents.record_usage("TestAgent", success=True, task_type="testing")
        state.agents.record_usage("TestAgent", success=False, task_type="refactoring")

        agent = state.agents.find_agent("TestAgent")
        # 2 successes out of 3 = 0.666...
        assert abs(agent["confidence"] - 2 / 3) < 0.01

    def test_initialize_workspace_registers_7_agents(self, initialized_state):
        count = initialized_state.agents.conn.execute(
            "SELECT COUNT(*) FROM agents"
        ).fetchone()[0]
        assert count == 7

    def test_expected_specialists_registered(self, initialized_state):
        expected = {
            "DebuggerAgent",
            "SecurityAgent",
            "RefactoringAgent",
            "TestingAgent",
            "DocumentationAgent",
            "PerformanceAgent",
            "ArchitectureAgent",
        }
        registered = {
            row[0]
            for row in initialized_state.agents.conn.execute(
                "SELECT name FROM agents"
            ).fetchall()
        }
        assert registered == expected

    def test_agents_persist_across_session_reset(self, state):
        state.agents.register_agent("MyAgent", "Stays after reset")
        state.reset_session()
        assert state.agents.find_agent("MyAgent") is not None


# ===========================================================================
# plan.db
# ===========================================================================


class TestMasterPlan:
    """MasterPlan — hierarchical task tree stored in memory.db."""

    def test_create_and_get_task(self, state):
        plan_id = state.plan.create_plan("Test plan")
        task_id = state.plan.create_task(plan_id, "Implement authentication module")
        assert task_id is not None

        fetched = state.plan.get_task(task_id)
        assert fetched is not None
        assert fetched["title"] == "Implement authentication module"
        assert fetched["status"] == "pending"

    def test_create_subtask_links_parent(self, state):
        plan_id = state.plan.create_plan("Test plan")
        parent_id = state.plan.create_task(plan_id, "Build REST API")
        child_id = state.plan.create_task(plan_id, "Implement /users endpoint", parent_id=parent_id)

        child = state.plan.get_task(child_id)
        assert child["parent_id"] == parent_id

        # Verify by querying all plan tasks
        tasks = state.plan.get_plan_tasks(plan_id)
        children_of_parent = [t["id"] for t in tasks if t["parent_id"] == parent_id]
        assert child_id in children_of_parent

    def test_update_task_status(self, state):
        plan_id = state.plan.create_plan("Test plan")
        task_id = state.plan.create_task(plan_id, "Write tests")

        state.plan.update_task_status(task_id, "in_progress")
        assert state.plan.get_task(task_id)["status"] == "in_progress"

        state.plan.update_task_status(task_id, "completed", result="All tests pass")
        completed = state.plan.get_task(task_id)
        assert completed["status"] == "completed"
        assert completed["result"] == "All tests pass"
        assert completed["completed_at"] is not None

    def test_get_all_tasks(self, state):
        plan_id = state.plan.create_plan("Test plan")
        state.plan.create_task(plan_id, "Task A")
        state.plan.create_task(plan_id, "Task B")
        state.plan.create_task(plan_id, "Task C")

        tasks = state.plan.get_all_tasks()
        assert len(tasks) == 3

    def test_clear_all_tasks(self, state):
        plan_id = state.plan.create_plan("Test plan")
        state.plan.create_task(plan_id, "Task A")
        state.plan.create_task(plan_id, "Task B")
        state.plan.clear_all_tasks()
        assert len(state.plan.get_all_tasks()) == 0

    def test_get_nonexistent_task_returns_none(self, state):
        assert state.plan.get_task("nonexistent-id") is None


# ===========================================================================
# Cross-DB: initialize_workspace
# ===========================================================================


class TestInitializeWorkspace:
    """Validate initialize_workspace() bootstraps all DBs correctly."""

    def test_all_dbs_created(self, workspace):
        from gaia.agents.gaia_code.integration import initialize_workspace

        initialize_workspace(workspace)

        expected_dbs = [
            "memory.db",
            "knowledge.db",
            "tools.db",
            "skills.db",
            "agents.db",
            "logs.db",
        ]
        for db_name in expected_dbs:
            assert (workspace / db_name).exists(), f"{db_name} not created"

    def test_tools_agents_skills_counts(self, workspace):
        from gaia.agents.gaia_code.integration import initialize_workspace
        from gaia.agents.base.shared_state import get_shared_state

        initialize_workspace(workspace)
        s = get_shared_state(workspace)

        tools = s.tools.conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]
        agents = s.agents.conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
        skills = s.skills.conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

        assert tools >= 22, f"Expected at least 22 tools, got {tools}"
        assert agents == 7, f"Expected 7 agents, got {agents}"
        assert skills == 8, f"Expected 8 skills, got {skills}"

    def test_repeated_init_is_idempotent(self, workspace):
        """Calling initialize_workspace three times must not duplicate rows."""
        from gaia.agents.gaia_code.integration import initialize_workspace
        from gaia.agents.base.shared_state import SharedAgentState, get_shared_state

        for _ in range(3):
            SharedAgentState._instance = None
            initialize_workspace(workspace)

        SharedAgentState._instance = None
        s = get_shared_state(workspace)

        tools = s.tools.conn.execute("SELECT COUNT(*) FROM tools").fetchone()[0]
        agents = s.agents.conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
        skills = s.skills.conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

        assert tools >= 22
        assert agents == 7
        assert skills == 8


# ===========================================================================
# Data validation: check actual registered content
# ===========================================================================


class TestInitialDataContent:
    """Validate the actual data registered by initialize_workspace()."""

    def test_core_tool_names_present(self, initialized_state):
        """Every expected core tool must be in tools.db."""
        expected_tools = [
            "read_file", "write_file", "edit_file", "glob_search", "grep_content",
            "run_python", "run_shell_command",
            "run_pytest", "run_jest", "check_coverage",
            "git_status", "git_diff", "git_commit", "git_branch", "git_log",
            "check_syntax", "run_linter", "check_imports", "format_code",
            "gh_clone", "gh_pr_create", "gh_issue_list",
        ]
        for name in expected_tools:
            tool = initialized_state.tools.get_tool(name)
            assert tool is not None, f"Tool '{name}' missing from tools.db"
            assert tool["source"] == "core"
            assert tool["description"], f"Tool '{name}' has empty description"

    def test_core_tool_categories(self, initialized_state):
        """Tools must be assigned to the correct categories."""
        expected_categories = {
            "read_file": "file_io",
            "run_pytest": "testing",
            "git_commit": "git",
            "check_syntax": "quality",
            "gh_pr_create": "github",
            "run_python": "execution",
        }
        for name, expected_cat in expected_categories.items():
            tool = initialized_state.tools.get_tool(name)
            assert tool["category"] == expected_cat, (
                f"Tool '{name}': expected category '{expected_cat}', got '{tool['category']}'"
            )

    def test_specialist_agent_descriptions(self, initialized_state):
        """Every specialist must have a non-empty description."""
        agents = initialized_state.agents.list_agents()
        for agent in agents:
            assert agent["description"], f"Agent '{agent['name']}' has empty description"

    def test_specialist_agent_capabilities(self, initialized_state):
        """Agents must have at least one capability listed."""
        agents_db = initialized_state.agents
        rows = agents_db.conn.execute(
            "SELECT name, capabilities FROM agents"
        ).fetchall()
        for name, caps in rows:
            assert caps and caps.strip(), f"Agent '{name}' has no capabilities"

    def test_initial_skill_names(self, initialized_state):
        """All 8 initial skills must be present by name."""
        expected_skills = {
            "debug_error_fix",
            "create_feature",
            "refactor_code",
            "git_commit_workflow",
            "add_tests",
            "write_documentation",
            "security_audit",
            "codebase_exploration",
        }
        registered = {
            row[0]
            for row in initialized_state.skills.conn.execute(
                "SELECT name FROM skills"
            ).fetchall()
        }
        assert registered == expected_skills

    def test_initial_skill_steps_are_valid_json(self, initialized_state):
        """Every skill's steps column must be valid JSON with at least one step."""
        rows = initialized_state.skills.conn.execute(
            "SELECT name, steps FROM skills"
        ).fetchall()
        for name, steps_json in rows:
            steps = json.loads(steps_json)
            assert isinstance(steps, list), f"Skill '{name}' steps is not a list"
            assert len(steps) >= 1, f"Skill '{name}' has no steps"
            for step in steps:
                assert "step" in step, f"Skill '{name}' step missing 'step' key"
                assert "action" in step, f"Skill '{name}' step missing 'action' key"

    def test_initial_skill_tools_used_are_valid(self, initialized_state):
        """tools_used on each skill must be a valid JSON list."""
        rows = initialized_state.skills.conn.execute(
            "SELECT name, tools_used FROM skills WHERE tools_used IS NOT NULL"
        ).fetchall()
        for name, tools_json in rows:
            tools = json.loads(tools_json)
            assert isinstance(tools, list), f"Skill '{name}' tools_used is not a list"
            assert len(tools) >= 1, f"Skill '{name}' has empty tools_used"

    def test_memory_store_recall_preserves_tags(self, state):
        """Tags stored with a memory must be retrieved exactly."""
        state.memory.store_memory(
            "decision", "Use SQLite for all DBs", tags=["architecture", "db", "decision"]
        )
        results = state.memory.recall_memories(query="SQLite")
        assert len(results) == 1
        assert sorted(results[0]["tags"]) == ["architecture", "db", "decision"]

    def test_knowledge_insight_fields_complete(self, state):
        """Stored insight must have all expected fields on recall."""
        state.knowledge.store_insight(
            category="error_fix",
            content="Always use check_same_thread=False for SQLite in threads",
            domain="python",
            triggers=["sqlite", "threading"],
        )
        results = state.knowledge.recall("sqlite")
        assert len(results) == 1
        insight = results[0]
        assert insight["category"] == "error_fix"
        assert insight["domain"] == "python"
        assert "check_same_thread" in insight["content"]
        assert "confidence" in insight

    def test_tool_usage_stats_accurate(self, state):
        """Usage stats must reflect exact call counts."""
        state.tools.register_tool("file_reader", "file_io", "Reads files", "core")

        state.tools.record_usage("file_reader", success=True, duration_ms=10)
        state.tools.record_usage("file_reader", success=True, duration_ms=20)
        state.tools.record_usage("file_reader", success=False, duration_ms=5, error="not found")

        stats = state.tools.get_tool_stats("file_reader")
        assert stats["total"] == 3
        assert stats["successes"] == 2
        assert stats["failures"] == 1
        assert stats["avg_duration_ms"] == 12  # (10+20+5) / 3 = 11.67 → rounds to 12

    def test_skill_confidence_after_mixed_usage(self, state):
        """Confidence must be calculated as success_count / total."""
        state.skills.register_skill(
            "mixed_skill", "Has wins and losses", "coding",
            [{"step": 1, "action": "code"}]
        )
        # 3 successes, 1 failure → confidence = 0.75
        for _ in range(3):
            state.skills.record_usage("mixed_skill", success=True)
        state.skills.record_usage("mixed_skill", success=False)

        skill = state.skills.find_skills(category="coding")[0]
        assert abs(skill["confidence"] - 0.75) < 0.01

    def test_plan_task_hierarchy_stored_correctly(self, state):
        """Parent-child relationships must be reflected in DB."""
        plan_id = state.plan.create_plan("Test plan")
        parent_id = state.plan.create_task(plan_id, "Build API")
        child1_id = state.plan.create_task(plan_id, "Implement /users", parent_id=parent_id)
        child2_id = state.plan.create_task(plan_id, "Implement /auth", parent_id=parent_id)

        tasks = state.plan.get_plan_tasks(plan_id)
        children_of_parent = [t["id"] for t in tasks if t["parent_id"] == parent_id]
        assert child1_id in children_of_parent
        assert child2_id in children_of_parent
        assert len(children_of_parent) == 2

        assert state.plan.get_task(child1_id)["parent_id"] == parent_id
        assert state.plan.get_task(child2_id)["parent_id"] == parent_id


# ===========================================================================
# Cross-DB: session reset isolation
# ===========================================================================


class TestSessionResetIsolation:
    """Verify reset_session() only clears working memory, not persistent DBs."""

    def test_full_isolation(self, state):
        # Write to all DBs
        state.memory.store_memory("key", "value")
        state.memory.cache_file("/f.py", "x=1")
        state.knowledge.store_insight("pattern", "Always test")
        state.tools.register_tool("my_tool", "cat", "desc", "core")
        state.skills.register_skill("my_skill", "desc", "cat", [{"step": 1}])
        state.agents.register_agent("MyAgent", "desc")
        plan_id = state.plan.create_plan("Isolation test")
        state.plan.create_task(plan_id, "Do something")

        # Reset session
        state.reset_session()

        # Working memory cleared
        assert len(state.memory.recall_memories()) == 0
        assert state.memory.get_file("/f.py") is None

        # Persistent DBs untouched
        assert len(state.knowledge.recall("test")) >= 1
        assert state.tools.get_tool("my_tool") is not None
        assert len(state.skills.find_skills()) >= 1
        assert state.agents.find_agent("MyAgent") is not None
        assert len(state.plan.get_all_tasks()) >= 1
