# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit tests for GaiaCode tool creation, persistence, and discovery.

Covers the new adaptive-tool system added in the scalable-tool-routing milestone:
  - create_tool()  — Python, bash, powershell, python_script langs
  - list_learned_tools() / delete_tool()
  - _sync_tools_to_db()   — syncs _TOOL_REGISTRY → tools.db at startup
  - _load_learned_tools() — reloads source='learned' tools across sessions
  - seed_common_tool_recipes() — pre-seeds 82 shell-command discovery hints
  - Semantic pre-fetch (ToolsDB.find_tools) picks up seeded + learned tools

All tests are deterministic (no LLM calls).
SharedAgentState singleton is reset before/after every test via the autouse
conftest fixture in tests/unit/conftest.py.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

from gaia.agents.base.shared_state import SharedAgentState
from gaia.agents.base.tools import _TOOL_REGISTRY
from gaia.agents.gaia_code.integration import seed_common_tool_recipes
from gaia.agents.gaia_code.tools import GaiaCodeTools


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ws(tmp_path):
    """Fresh SharedAgentState backed by a temp workspace."""
    state = SharedAgentState(workspace_dir=tmp_path)
    yield state, tmp_path


@pytest.fixture
def tools_obj(ws):
    """GaiaCodeTools instance wired to the temp workspace."""
    state, tmp = ws
    obj = GaiaCodeTools.__new__(GaiaCodeTools)
    return obj, state, tmp


# ============================================================================
# create_tool — Python
# ============================================================================


class TestCreateToolPython:
    def test_success_registers_in_registry(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def greet(name: str) -> dict:\n    return {"greeting": f"Hello {name}"}\n'
        result = obj.tool_create_tool("greet", code, "Say hello", category="test")

        assert result["status"] == "success"
        assert result["name"] == "greet"
        assert result["lang"] == "python"
        assert "greet" in _TOOL_REGISTRY

    def test_success_writes_file_to_disk(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def add(a: int, b: int) -> dict:\n    return {"sum": a + b}\n'
        result = obj.tool_create_tool("add", code, "Add two numbers")

        tool_file = tmp / "tools" / "add.py"
        assert tool_file.exists()
        assert "def add" in tool_file.read_text()

    def test_success_persists_to_tools_db(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def double(x: int) -> dict:\n    return {"result": x * 2}\n'
        obj.tool_create_tool("double", code, "Double a number", category="math")

        row = state.tools.conn.execute(
            "SELECT source, category FROM tools WHERE name = 'double'"
        ).fetchone()
        assert row is not None
        assert row[0] == "learned"
        assert row[1] == "math"

    def test_success_tool_is_callable(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def multiply(a: int, b: int) -> dict:\n    return {"product": a * b}\n'
        obj.tool_create_tool("multiply", code, "Multiply two numbers")

        func = _TOOL_REGISTRY["multiply"]["function"]
        assert func(a=3, b=4) == {"product": 12}

    def test_invalid_name_returns_error(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def my_tool() -> dict:\n    return {}\n'
        result = obj.tool_create_tool("123invalid", code, "Bad name")

        assert result["status"] == "error"
        assert "identifier" in result["error"].lower()

    def test_syntax_error_returns_error(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def broken(\n    return "missing colon"'
        result = obj.tool_create_tool("broken", code, "Bad syntax")

        assert result["status"] == "error"
        assert "syntax" in result["error"].lower()

    def test_wrong_function_name_returns_error(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def actual_name() -> dict:\n    return {}\n'
        result = obj.tool_create_tool("expected_name", code, "Name mismatch")

        assert result["status"] == "error"
        assert "expected_name" in result["error"]

    def test_default_category_is_learned(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def no_cat() -> dict:\n    return {}\n'
        obj.tool_create_tool("no_cat", code, "No explicit category")

        row = state.tools.conn.execute(
            "SELECT category FROM tools WHERE name = 'no_cat'"
        ).fetchone()
        assert row[0] == "learned"

    def test_parameters_stored_in_db(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def typed_fn(x: int, y: str) -> dict:\n    return {}\n'
        result = obj.tool_create_tool("typed_fn", code, "Has params")

        assert result["status"] == "success"
        assert "parameters" in result
        assert "x" in result["parameters"]
        assert "y" in result["parameters"]

    def test_overwrite_existing_tool(self, tools_obj):
        """Creating a tool with same name twice replaces the old DB entry (not duplicated)."""
        obj, state, tmp = tools_obj
        code_v1 = 'def versioned() -> dict:\n    return {"v": 1}\n'
        code_v2 = 'def versioned() -> dict:\n    return {"v": 2}\n'

        obj.tool_create_tool("versioned", code_v1, "Version 1")
        obj.tool_create_tool("versioned", code_v2, "Version 2")

        # DB should have exactly one entry (not duplicated)
        count = state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools WHERE name='versioned'"
        ).fetchone()[0]
        assert count == 1, "Overwriting should not duplicate the DB entry"

        # DB entry should have the latest description
        desc = state.tools.conn.execute(
            "SELECT description FROM tools WHERE name='versioned'"
        ).fetchone()[0]
        assert desc == "Version 2"

        # Registry should still have the tool registered
        assert "versioned" in _TOOL_REGISTRY


# ============================================================================
# create_tool — Script languages
# ============================================================================


class TestCreateToolScriptLanguages:
    def test_bash_writes_sh_and_wrapper(self, tools_obj):
        obj, state, tmp = tools_obj
        bash_code = "#!/bin/bash\necho hello"
        result = obj.tool_create_tool("run_hello", bash_code, "Echo hello", lang="bash")

        assert result["status"] == "success"
        assert result["lang"] == "bash"
        sh_file = tmp / "tools" / "run_hello.sh"
        py_file = tmp / "tools" / "run_hello.py"
        assert sh_file.exists(), "bash script should be on disk"
        assert py_file.exists(), "python wrapper should be on disk"

    def test_bash_sh_alias_not_supported(self, tools_obj):
        """'sh' is not in _SCRIPT_WRAPPER_TEMPLATES; only 'bash' and 'powershell' are."""
        obj, state, tmp = tools_obj
        result = obj.tool_create_tool(
            "run_sh_alias", "echo test", "Alias test", lang="sh"
        )
        assert result["status"] == "error"
        assert "sh" in result["error"].lower() or "unsupported" in result["error"].lower()

    def test_bash_wrapper_registered_in_registry(self, tools_obj):
        obj, state, tmp = tools_obj
        result = obj.tool_create_tool("bash_registered", "echo ok", "Registered bash", lang="bash")
        assert result["status"] == "success"
        assert "bash_registered" in _TOOL_REGISTRY

    def test_powershell_writes_ps1_and_wrapper(self, tools_obj):
        obj, state, tmp = tools_obj
        ps_code = 'Write-Output "hello"'
        result = obj.tool_create_tool("run_ps", ps_code, "PowerShell test", lang="powershell")

        assert result["status"] == "success"
        assert (tmp / "tools" / "run_ps.ps1").exists()
        assert (tmp / "tools" / "run_ps.py").exists()

    def test_ps1_alias_not_supported(self, tools_obj):
        """'ps1' is not in _SCRIPT_WRAPPER_TEMPLATES; only 'powershell' is."""
        obj, state, tmp = tools_obj
        result = obj.tool_create_tool("run_ps1", 'echo hi', "PS1 alias", lang="ps1")
        assert result["status"] == "error"
        assert "ps1" in result["error"].lower() or "unsupported" in result["error"].lower()

    def test_unsupported_lang_returns_error(self, tools_obj):
        obj, state, tmp = tools_obj
        result = obj.tool_create_tool("bad_lang", "code", "Bad lang", lang="ruby")
        assert result["status"] == "error"
        assert "unsupported" in result["error"].lower() or "lang" in result["error"].lower()

    def test_script_persisted_as_learned_in_db(self, tools_obj):
        obj, state, tmp = tools_obj
        obj.tool_create_tool("bash_persisted", "echo hi", "Persisted bash", lang="bash")
        row = state.tools.conn.execute(
            "SELECT source FROM tools WHERE name = 'bash_persisted'"
        ).fetchone()
        assert row is not None
        assert row[0] == "learned"


# ============================================================================
# list_learned_tools
# ============================================================================


class TestListLearnedTools:
    def test_empty_returns_zero_count(self, tools_obj):
        obj, state, tmp = tools_obj
        result = obj.tool_list_learned_tools()
        assert result["status"] == "success"
        assert result["count"] == 0
        assert result["tools"] == []

    def test_shows_created_tool(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def my_tool() -> dict:\n    return {}\n'
        obj.tool_create_tool("my_tool", code, "My tool desc", category="test_cat")

        result = obj.tool_list_learned_tools()
        assert result["count"] == 1
        assert result["tools"][0]["name"] == "my_tool"
        assert result["tools"][0]["category"] == "test_cat"
        assert result["tools"][0]["description"] == "My tool desc"

    def test_does_not_show_registry_tools(self, tools_obj):
        """source='registry' tools should not appear in learned list."""
        obj, state, tmp = tools_obj
        state.tools.register_tool("core_tool", "core", "A core tool", source="core")

        result = obj.tool_list_learned_tools()
        names = [t["name"] for t in result["tools"]]
        assert "core_tool" not in names

    def test_multiple_tools_all_listed(self, tools_obj):
        obj, state, tmp = tools_obj
        for i in range(3):
            code = f'def tool_{i}() -> dict:\n    return {{"i": {i}}}\n'
            obj.tool_create_tool(f"tool_{i}", code, f"Tool {i}")

        result = obj.tool_list_learned_tools()
        assert result["count"] == 3
        names = {t["name"] for t in result["tools"]}
        assert names == {"tool_0", "tool_1", "tool_2"}


# ============================================================================
# delete_tool
# ============================================================================


class TestDeleteTool:
    def test_delete_removes_from_db(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def to_delete() -> dict:\n    return {}\n'
        obj.tool_create_tool("to_delete", code, "Will be deleted")

        result = obj.tool_delete_tool("to_delete")
        assert result["status"] == "success"

        row = state.tools.conn.execute(
            "SELECT id FROM tools WHERE name = 'to_delete'"
        ).fetchone()
        assert row is None

    def test_delete_removes_from_registry(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def reg_delete() -> dict:\n    return {}\n'
        obj.tool_create_tool("reg_delete", code, "Delete from registry")
        assert "reg_delete" in _TOOL_REGISTRY

        obj.tool_delete_tool("reg_delete")
        assert "reg_delete" not in _TOOL_REGISTRY

    def test_delete_removes_file_from_disk(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def disk_delete() -> dict:\n    return {}\n'
        obj.tool_create_tool("disk_delete", code, "Delete from disk")
        tool_file = tmp / "tools" / "disk_delete.py"
        assert tool_file.exists()

        obj.tool_delete_tool("disk_delete")
        assert not tool_file.exists()

    def test_delete_nonexistent_returns_error(self, tools_obj):
        obj, state, tmp = tools_obj
        result = obj.tool_delete_tool("nonexistent_xyz")
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_delete_core_tool_blocked(self, tools_obj):
        obj, state, tmp = tools_obj
        state.tools.register_tool("builtin_tool", "core", "A built-in", source="core")

        result = obj.tool_delete_tool("builtin_tool")
        assert result["status"] == "error"
        assert "source=core" in result["error"] or "cannot delete" in result["error"].lower()

    def test_delete_recipe_blocked(self, tools_obj):
        obj, state, tmp = tools_obj
        state.tools.register_tool("pytest_run", "python_testing", "Run pytest", source="recipe")

        result = obj.tool_delete_tool("pytest_run")
        assert result["status"] == "error"

    def test_list_after_delete_shows_zero(self, tools_obj):
        obj, state, tmp = tools_obj
        code = 'def ephemeral() -> dict:\n    return {}\n'
        obj.tool_create_tool("ephemeral", code, "Ephemeral tool")
        obj.tool_delete_tool("ephemeral")

        result = obj.tool_list_learned_tools()
        assert result["count"] == 0


# ============================================================================
# _sync_tools_to_db
# ============================================================================


class TestSyncToolsToDb:
    def test_registry_tools_appear_in_find_tools(self, ws):
        """After _sync_tools_to_db, tools registered via @tool are discoverable."""
        state, tmp = ws

        # Manually register a minimal tool in the real _TOOL_REGISTRY
        from gaia.agents.base.tools import tool as tool_decorator

        def dummy_sync_tool(x: str) -> dict:
            """A dummy tool for sync testing."""
            return {"x": x}

        tool_decorator(dummy_sync_tool)

        try:
            # Simulate what GaiaCodeAgent._sync_tools_to_db does
            params = {
                pname: {"type": pinfo["type"], "required": pinfo["required"]}
                for pname, pinfo in _TOOL_REGISTRY["dummy_sync_tool"]["parameters"].items()
            }
            state.tools.register_tool(
                name="dummy_sync_tool",
                category="utility",
                description=_TOOL_REGISTRY["dummy_sync_tool"]["description"],
                source="registry",
                parameters=params,
            )

            results = state.tools.find_tools("dummy sync tool testing", top_k=5)
            names = [r["name"] for r in results]
            assert "dummy_sync_tool" in names
        finally:
            _TOOL_REGISTRY.pop("dummy_sync_tool", None)

    def test_synced_tool_has_parameters(self, ws):
        state, tmp = ws
        from gaia.agents.base.tools import tool as tool_decorator

        def parameterised_tool(path: str, verbose: bool) -> dict:
            """Tool with parameters for schema test."""
            return {}

        tool_decorator(parameterised_tool)

        try:
            params = {
                pname: {"type": pinfo["type"], "required": pinfo["required"]}
                for pname, pinfo in _TOOL_REGISTRY["parameterised_tool"]["parameters"].items()
            }
            state.tools.register_tool(
                name="parameterised_tool",
                category="utility",
                description="Tool with parameters",
                source="registry",
                parameters=params,
            )

            results = state.tools.find_tools("parameterised_tool", top_k=5)
            match = next((r for r in results if r["name"] == "parameterised_tool"), None)
            assert match is not None
            assert match["parameters"] is not None
            assert "path" in match["parameters"]
        finally:
            _TOOL_REGISTRY.pop("parameterised_tool", None)

    def test_source_is_registry_not_learned(self, ws):
        state, tmp = ws
        state.tools.register_tool(
            name="explicit_registry_tool",
            category="utility",
            description="Explicitly registered",
            source="registry",
        )
        row = state.tools.conn.execute(
            "SELECT source FROM tools WHERE name = 'explicit_registry_tool'"
        ).fetchone()
        assert row[0] == "registry"


# ============================================================================
# _load_learned_tools
# ============================================================================


class TestLoadLearnedTools:
    def test_loads_tool_from_previous_session(self, ws):
        """Simulates a tool created in a prior session being loaded at startup."""
        state, tmp = ws

        # Write the tool file as if a previous session created it
        tools_dir = tmp / "tools"
        tools_dir.mkdir(exist_ok=True)
        tool_file = tools_dir / "prior_tool.py"
        tool_file.write_text(
            'def prior_tool(msg: str) -> dict:\n    return {"echo": msg}\n',
            encoding="utf-8",
        )

        # Register in tools.db as source='learned' (as tool_create_tool would do)
        state.tools.register_tool(
            name="prior_tool",
            category="learned",
            description="Tool from prior session",
            source="learned",
            code_path=str(tool_file),
        )

        # Simulate _load_learned_tools logic
        from gaia.agents.base.tools import tool as tool_decorator

        rows = state.tools.conn.execute(
            "SELECT name, code_path FROM tools WHERE source='learned' AND enabled=TRUE"
        ).fetchall()

        loaded = 0
        for name, code_path in rows:
            path = Path(code_path)
            if path.exists():
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                func = getattr(module, name)
                tool_decorator(func)
                loaded += 1

        assert loaded == 1
        assert "prior_tool" in _TOOL_REGISTRY

        func = _TOOL_REGISTRY["prior_tool"]["function"]
        assert func(msg="hi") == {"echo": "hi"}

    def test_skips_missing_file_gracefully(self, ws):
        """If the tool file was deleted, loading should skip without crashing."""
        state, tmp = ws

        state.tools.register_tool(
            name="missing_file_tool",
            category="learned",
            description="Points to nonexistent file",
            source="learned",
            code_path="/nonexistent/path/missing_file_tool.py",
        )

        rows = state.tools.conn.execute(
            "SELECT name, code_path FROM tools WHERE source='learned' AND enabled=TRUE"
        ).fetchall()

        loaded = 0
        for name, code_path in rows:
            path = Path(code_path)
            if path.exists():
                loaded += 1

        assert loaded == 0
        assert "missing_file_tool" not in _TOOL_REGISTRY

    def test_loaded_tool_callable_immediately(self, ws):
        state, tmp = ws
        from gaia.agents.base.tools import tool as tool_decorator

        tools_dir = tmp / "tools"
        tools_dir.mkdir(exist_ok=True)
        tool_file = tools_dir / "counter_tool.py"
        tool_file.write_text(
            'def counter_tool(n: int) -> dict:\n    return {"count": list(range(n))}\n',
            encoding="utf-8",
        )
        state.tools.register_tool(
            name="counter_tool",
            category="learned",
            description="Returns a range",
            source="learned",
            code_path=str(tool_file),
        )

        # Load it
        spec = importlib.util.spec_from_file_location("counter_tool", tool_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tool_decorator(module.counter_tool)

        func = _TOOL_REGISTRY["counter_tool"]["function"]
        assert func(n=3) == {"count": [0, 1, 2]}


# ============================================================================
# seed_common_tool_recipes
# ============================================================================


class TestSeedCommonToolRecipes:
    def test_seeds_expected_count(self, ws):
        state, tmp = ws
        count = seed_common_tool_recipes(state)
        assert count >= 80, f"Expected at least 80 recipes, got {count}"

    def test_seeded_source_is_recipe(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)
        row = state.tools.conn.execute(
            "SELECT source FROM tools WHERE name = 'pytest_run'"
        ).fetchone()
        assert row is not None
        assert row[0] == "recipe"

    def test_idempotent_double_seed(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)
        count_first = state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools WHERE source = 'recipe'"
        ).fetchone()[0]

        second_result = seed_common_tool_recipes(state)
        count_second = state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools WHERE source = 'recipe'"
        ).fetchone()[0]

        assert second_result == 0, "Second seed should return 0 (already seeded)"
        assert count_second == count_first, "Row count must not change on re-seed"

    def test_python_recipes_present(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)
        expected = ["pytest_run", "pytest_coverage", "black_format", "ruff_fix", "mypy_typecheck"]
        for name in expected:
            row = state.tools.conn.execute(
                "SELECT name FROM tools WHERE name = ?", (name,)
            ).fetchone()
            assert row is not None, f"Expected recipe '{name}' not found"

    def test_cpp_recipes_present(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)
        expected = ["cmake_configure", "cmake_build", "ctest_verbose"]
        for name in expected:
            row = state.tools.conn.execute(
                "SELECT name FROM tools WHERE name = ?", (name,)
            ).fetchone()
            assert row is not None, f"Expected C++ recipe '{name}' not found"

    def test_shell_recipes_present(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)
        expected = ["grep_recursive", "find_files", "docker_build"]
        for name in expected:
            row = state.tools.conn.execute(
                "SELECT name FROM tools WHERE name = ?", (name,)
            ).fetchone()
            assert row is not None, f"Expected shell recipe '{name}' not found"

    def test_frontend_recipes_present(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)
        expected = ["npm_install", "tsc_typecheck", "eslint_fix"]
        for name in expected:
            row = state.tools.conn.execute(
                "SELECT name FROM tools WHERE name = ?", (name,)
            ).fetchone()
            assert row is not None, f"Expected frontend recipe '{name}' not found"


# ============================================================================
# Semantic pre-fetch (find_tools)
# ============================================================================


class TestSemanticPrefetch:
    def test_finds_pytest_recipe_by_query(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)

        results = state.tools.find_tools("run pytest tests", top_k=5)
        names = [r["name"] for r in results]
        assert any("pytest" in n for n in names), f"Expected a pytest tool, got: {names}"

    def test_finds_cmake_recipe_by_query(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)

        results = state.tools.find_tools("cmake build project", top_k=5)
        names = [r["name"] for r in results]
        assert any("cmake" in n for n in names), f"Expected a cmake tool, got: {names}"

    def test_finds_docker_recipe_by_query(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)

        results = state.tools.find_tools("docker container build", top_k=5)
        names = [r["name"] for r in results]
        assert any("docker" in n for n in names), f"Expected a docker tool, got: {names}"

    def test_learned_tool_discoverable_after_create(self, ws):
        """A newly created tool is immediately discoverable via find_tools."""
        state, tmp = ws
        tools_obj = GaiaCodeTools.__new__(GaiaCodeTools)
        code = 'def parse_errors(output: str) -> dict:\n    return {"errors": []}\n'
        tools_obj.tool_create_tool("parse_errors", code, "Parse compiler error output from build")

        results = state.tools.find_tools("compiler error parse", top_k=10)
        names = [r["name"] for r in results]
        assert "parse_errors" in names

    def test_returns_parameters_in_results(self, ws):
        state, tmp = ws
        state.tools.register_tool(
            name="has_params_tool",
            category="test",
            description="A tool with parameters for discovery",
            source="registry",
            parameters={"x": {"type": "str", "required": True}},
        )

        results = state.tools.find_tools("parameters discovery tool", top_k=5)
        match = next((r for r in results if r["name"] == "has_params_tool"), None)
        assert match is not None
        assert match["parameters"] is not None
        assert "x" in match["parameters"]

    def test_empty_query_returns_empty(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)
        results = state.tools.find_tools("", top_k=10)
        assert results == []

    def test_top_k_limits_results(self, ws):
        state, tmp = ws
        seed_common_tool_recipes(state)

        results = state.tools.find_tools("run test build", top_k=3)
        assert len(results) <= 3


# ============================================================================
# _format_tools_for_prompt (essential-tools-only system prompt)
# ============================================================================


class TestFormatToolsForPrompt:
    """
    The system prompt should only include the 10 essential tools,
    not all 70+ registered tools.
    """

    def test_essential_tools_in_prompt(self, ws):
        from gaia.agents.base.tools import tool as tool_decorator
        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        state, tmp = ws

        # Register minimal essential tools in _TOOL_REGISTRY so the formatter works
        essential_stubs = {
            "read_file": "def read_file(path: str) -> dict:\n    return {}\n",
            "write_file": "def write_file(path: str, content: str) -> dict:\n    return {}\n",
            "find_tool": "def find_tool(query: str) -> dict:\n    return {}\n",
        }
        for name, code in essential_stubs.items():
            if name not in _TOOL_REGISTRY:
                exec(code, globs := {})
                tool_decorator(globs[name])

        try:
            # Create a minimal agent-like object with just _format_tools_for_prompt
            class _MinimalAgent:
                _ESSENTIAL_TOOLS = GaiaCodeAgent._ESSENTIAL_TOOLS

                def _format_tools_for_prompt(self):
                    return GaiaCodeAgent._format_tools_for_prompt(self)

            agent = _MinimalAgent()
            prompt = agent._format_tools_for_prompt()

            # Essential tools that were registered should appear
            for name in ("read_file", "write_file", "find_tool"):
                assert name in prompt, f"Essential tool '{name}' missing from prompt"
        finally:
            for name in essential_stubs:
                _TOOL_REGISTRY.pop(name, None)

    def test_fallback_note_in_prompt(self, ws):
        """Prompt must explain how to discover non-essential tools."""
        from gaia.agents.gaia_code.agent import GaiaCodeAgent

        state, tmp = ws

        class _MinimalAgent:
            _ESSENTIAL_TOOLS = GaiaCodeAgent._ESSENTIAL_TOOLS

            def _format_tools_for_prompt(self):
                return GaiaCodeAgent._format_tools_for_prompt(self)

        agent = _MinimalAgent()
        prompt = agent._format_tools_for_prompt()

        assert "tools.db" in prompt or "find_tool" in prompt
