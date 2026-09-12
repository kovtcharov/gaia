# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""File search must look where the user's work is, and say where it looked.

Ask the flagship how many Go files are under `tui/internal` and it answered
"Zero." There are 203. Two things compounded (#3576):

* the search was rooted at ``Path.cwd()``, which for a daemon-spawned sidecar is
  its own package directory — the user's project was not in the search set;
* nothing downstream could tell that apart from a genuine zero, because the
  result was ``status: "success"`` with no record of where it looked.

These tests build a fake project, point the agent's sandbox at it, and assert
the search reaches it — from a working directory that is somewhere else
entirely, which is the condition the bug needed.

No LLM or external service required.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from gaia.agents.base.tools import _TOOL_REGISTRY
from gaia.agents.tools.file_tools import FileSearchToolsMixin
from gaia.agents.tools.filesystem_tools import FileSystemToolsMixin


class _Sandbox:
    """Stand-in for PathValidator: the paths the operator declared.

    Implements the read gate the tools actually call — ``validate_read`` as
    well as ``is_path_allowed`` — so this double cannot pass while the real
    validator's interface moves underneath it.
    """

    def __init__(self, *paths):
        self.allowed_paths = {Path(p).resolve() for p in paths}

    def is_path_allowed(self, path, prompt_user=True):
        resolved = Path(path).resolve()
        return any(
            resolved == root or root in resolved.parents for root in self.allowed_paths
        )

    def validate_read(self, path, prompt_user=True):
        if not self.is_path_allowed(path, prompt_user=prompt_user):
            return False, f"Access denied: '{path}' is not in allowed paths"
        return True, ""


@pytest.fixture
def project(tmp_path):
    """A project with Go files, and a separate directory to run the agent from.

    ``elsewhere`` stands in for the sidecar's own package directory — the cwd
    the daemon actually spawns it in.
    """
    root = tmp_path / "myproject"
    (root / "tui" / "internal").mkdir(parents=True)
    for name in ("model.go", "view.go", "controller.go"):
        (root / "tui" / "internal" / name).write_text("package internal\n")
    (root / "README.md").write_text("# project\n")

    elsewhere = tmp_path / "sidecar-package-dir"
    elsewhere.mkdir()

    prev = Path.cwd()
    os.chdir(elsewhere)
    try:
        yield root, elsewhere
    finally:
        os.chdir(prev)


# ============================================================================
# 1. file_tools.search_file
# ============================================================================


@pytest.fixture
def search_file(project):
    root, _ = project
    mixin = FileSearchToolsMixin()
    mixin.path_validator = _Sandbox(root)

    saved = dict(_TOOL_REGISTRY)
    try:
        mixin.register_file_search_tools()
        entry = _TOOL_REGISTRY.get("search_file")
        assert entry is not None, "search_file was not registered"
        fn = entry["function"]
        fn.mixin = mixin
        yield fn
    finally:
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(saved)


class TestSearchFileLooksAtTheWorkspace:
    def test_it_finds_project_files_from_an_unrelated_cwd(self, search_file, project):
        """The condition the bug needed: cwd is not the project."""
        result = search_file("*.go")
        assert result["status"] == "success", result
        assert result["count"] == 3, result

    def test_a_named_directory_scopes_the_search(self, search_file, project):
        root, _ = project
        result = search_file("*.go", directory=str(root / "tui" / "internal"))
        assert result["count"] == 3
        assert all("tui/internal" in f for f in result["files"])

    def test_a_relative_directory_resolves_against_the_workspace(
        self, search_file, project
    ):
        """The user says "in tui/internal"; they do not mean the process cwd."""
        result = search_file("*.go", directory="tui/internal")
        assert result["count"] == 3

    def test_a_directory_that_does_not_exist_is_an_error_not_an_empty_result(
        self, search_file, project
    ):
        root, _ = project
        result = search_file("*.go", directory=str(root / "no" / "such" / "folder"))
        assert result["status"] == "error"
        assert "not an empty result" in result["error"]

    def test_an_unresolvable_relative_directory_is_refused_not_answered(
        self, search_file
    ):
        """It matches no workspace root, so it falls outside the sandbox.

        Refusing rather than reporting "not found" is deliberate: the sandbox
        is checked before the existence probe so an outside path cannot be
        used to ask whether a directory exists.
        """
        result = search_file("*.go", directory="no/such/folder")
        assert result["status"] == "error"
        assert "not in allowed paths" in result["error"]

    def test_a_directory_outside_the_sandbox_is_refused(self, search_file, tmp_path):
        outside = tmp_path / "not-my-project"
        outside.mkdir()
        result = search_file("*.go", directory=str(outside))
        assert result["status"] == "error"
        assert "not in allowed paths" in result["error"]

    def test_a_genuine_zero_says_where_it_looked(self, search_file, project):
        root, _ = project
        result = search_file("*.rs", directory=str(root))
        assert result["count"] == 0
        assert result["searched_paths"], result
        assert str(root) in result["display_message"]

    def test_a_genuine_zero_tells_the_model_not_to_generalise_it(
        self, search_file, project
    ):
        root, _ = project
        result = search_file("*.rs", directory=str(root))
        assert "not zero on the machine" in result["suggestion"]


# ============================================================================
# 2. filesystem_tools.find_files — the flagship's search tool
# ============================================================================


def _filesystem_agent(root):
    class MockAgent(FileSystemToolsMixin):
        def __init__(self):
            self._web_client = None
            self._path_validator = _Sandbox(root)
            self._fs_index = None
            self._tools = {}
            self._bookmarks = {}

    registered = {}

    def mock_tool(atomic=True):
        def decorator(func):
            registered[func.__name__] = func
            return func

        return decorator

    with patch("gaia.agents.base.tools.tool", mock_tool):
        agent = MockAgent()
        agent.register_filesystem_tools()
    return agent, registered


class TestFindFilesLooksAtTheWorkspace:
    def test_smart_scope_reaches_the_project_not_the_cwd(self, project):
        root, elsewhere = project
        agent, tools = _filesystem_agent(root)

        out = tools["find_files"]("*.go")

        assert "model.go" in out, out
        assert str(elsewhere) not in out

    def test_cwd_scope_means_the_workspace_not_the_process_directory(self, project):
        root, _ = project
        agent, tools = _filesystem_agent(root)

        assert "model.go" in tools["find_files"]("*.go", scope="cwd")

    def test_workspace_roots_prefers_the_sandbox_over_the_process_cwd(self, project):
        root, elsewhere = project
        agent, _ = _filesystem_agent(root)

        roots = agent.workspace_roots()

        assert roots == [str(root.resolve())]
        assert str(elsewhere) not in roots

    def test_workspace_roots_falls_back_to_cwd_without_a_sandbox(self, project):
        _, elsewhere = project

        class Bare(FileSystemToolsMixin):
            pass

        assert Bare().workspace_roots() == [str(elsewhere)]
