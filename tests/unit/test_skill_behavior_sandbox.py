# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
The harness must never touch the developer's real home.

Not a tidiness concern. ``MemoryStore`` resolves its path from ``Path.home()``
and ignores ``GAIA_CONFIG_DIR``, so a harness that redirects only the latter does
two harmful things at once: it writes the run's memory rows into the real
``~/.gaia/memory.db``, and its side-effect checks then search a sandbox those rows
never reached — reporting a working skill as failed.

These tests need no model: they stop before the agent is constructed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gaia.eval.skill_behavior import (
    FixtureServer,
    ScenarioContext,
    ScenarioUnavailable,
    build_gaia_agent,
)


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr("os.environ", dict(__import__("os").environ))
    home = tmp_path / "home"
    workspace = tmp_path / "work"
    home.mkdir()
    workspace.mkdir()
    server = FixtureServer()
    yield ScenarioContext(
        skill="fake",
        run=0,
        workspace=workspace,
        home=home,
        token="deadbeef",
        fixtures=server,
    )
    server.stop()


def test_the_run_home_is_redirected_before_the_agent_is_built(context, monkeypatch):
    """Every home-derived path — memory, index, config — lands in the sandbox."""
    seen = {}

    def _fail_after_redirect(**_kwargs):
        import os

        seen["HOME"] = os.environ.get("HOME")
        seen["USERPROFILE"] = os.environ.get("USERPROFILE")
        seen["GAIA_CONFIG_DIR"] = os.environ.get("GAIA_CONFIG_DIR")
        raise RuntimeError("stop before constructing a real agent")

    monkeypatch.setattr(
        "gaia_agent.agent.GaiaAgent.__init__",
        lambda self, *args, **kwargs: _fail_after_redirect(),
    )
    with pytest.raises(RuntimeError):
        build_gaia_agent(context=context, skill_roots=[], max_steps=5)

    assert seen["GAIA_CONFIG_DIR"] == str(context.home)
    assert seen["HOME"] == str(context.home)
    assert seen["USERPROFILE"] == str(context.home), (
        "Windows resolves Path.home() from USERPROFILE; redirecting only HOME "
        "leaves the real user profile exposed on the runner this actually runs on."
    )


def test_a_home_that_did_not_move_stops_the_run(context, monkeypatch):
    """Fail loudly rather than quietly writing into the real profile."""
    monkeypatch.setattr(
        Path, "home", classmethod(lambda cls: Path("C:/somewhere-else"))
    )
    with pytest.raises(ScenarioUnavailable) as excinfo:
        build_gaia_agent(context=context, skill_roots=[], max_steps=5)
    assert "Sandbox escape" in str(excinfo.value)


def test_a_missing_host_agent_is_blocked_not_crashed(context, monkeypatch):
    """An uninstalled agent must record `blocked` with the install command."""
    import builtins

    real_import = builtins.__import__

    def _no_gaia_agent(name, *args, **kwargs):
        if name == "gaia_agent.agent":
            raise ImportError("No module named 'gaia_agent'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_gaia_agent)
    with pytest.raises(ScenarioUnavailable) as excinfo:
        build_gaia_agent(context=context, skill_roots=[], max_steps=5)
    assert "pip install -e hub/agents/gaia/python" in str(excinfo.value)
