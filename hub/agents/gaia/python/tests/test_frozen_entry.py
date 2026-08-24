# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The frozen binary's transport dispatch, and the freeze wiring that carries it.

The bug this suite pins: the frozen executable is named ``gaia-agent``, and that
name already means ``gaia_agent.stdio:main`` -- it is the wheel's console script
and it is what the TUI resolves on PATH for its subprocess transport. Freezing
``packaging/server.py`` under that name shipped a binary that could only speak
HTTP, so a TUI launch got uvicorn's startup banner where it expected JSON lines
and every turn died with "unreadable event skipped".

Two directions matter and each breaks a different consumer:

1. **A bare launch must reach stdio.** Regress it and the TUI is dead again.
2. **``--host``/``--port`` alone must reach HTTP** -- with no ``--serve``,
   because that is the argv the daemon's sidecar spec and the npm client's
   ``spawnSidecar`` have always sent. Requiring ``--serve`` would break both
   without either of them changing a line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "packaging"


def _load(name: str):
    """Import a packaging module by path -- ``packaging/`` is not a package."""
    spec = importlib.util.spec_from_file_location(name, PACKAGING / f"{name}.py")
    assert spec and spec.loader, f"cannot load {name} from {PACKAGING}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def entry():
    return _load("entry")


# --- the dispatch rule ------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param([], id="bare"),
        pytest.param(["--dev"], id="dev"),
        pytest.param(["--json-events"], id="json-events"),
        pytest.param(["--use-claude", "--claude-model", "claude-sonnet-5"], id="claude"),
        pytest.param(["--model", "Gemma-4-E4B-it-GGUF"], id="model"),
        pytest.param(["--bypass-permissions"], id="bypass"),
    ],
)
def test_stdio_argv_does_not_select_http(entry, argv):
    """Every spelling the Go side emits literally must stay on stdio."""
    assert entry._wants_http(argv) is False


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--serve"], id="serve"),
        pytest.param(["--host", "127.0.0.1", "--port", "8141"], id="daemon-and-npm"),
        pytest.param(["--port", "8141"], id="port-only"),
        pytest.param(["--host=127.0.0.1", "--port=8141"], id="equals-form"),
    ],
)
def test_http_argv_selects_the_rest_sidecar(entry, argv):
    assert entry._wants_http(argv) is True


def test_serve_is_stripped_before_the_server_parser_sees_it(entry, monkeypatch):
    """``--serve`` is the dispatcher's flag; server.main's parser rejects it."""
    seen = {}
    monkeypatch.setitem(
        sys.modules,
        "gaia_agent.server",
        type(sys)("gaia_agent.server"),
    )
    def _fake_serve(argv):
        seen["argv"] = argv
        return 0

    sys.modules["gaia_agent.server"].main = _fake_serve

    assert entry.main(["--serve", "--host", "127.0.0.1", "--port", "8141"]) == 0
    assert seen["argv"] == ["--host", "127.0.0.1", "--port", "8141"]


def test_stdio_receives_its_argv_untouched(entry, monkeypatch):
    seen = {}
    monkeypatch.setitem(sys.modules, "gaia_agent.stdio", type(sys)("gaia_agent.stdio"))
    def _fake_stdio(argv):
        seen["argv"] = argv
        return 0

    sys.modules["gaia_agent.stdio"].main = _fake_stdio

    argv = ["--dev", "--use-claude", "--claude-model", "claude-sonnet-5"]
    assert entry.main(list(argv)) == 0
    assert seen["argv"] == argv


def test_usage_is_ascii_only(entry):
    """The frozen binary prints this on a cp1252 console, where non-ASCII
    punctuation comes out as a replacement character."""
    entry._USAGE.encode("ascii")


# --- the freeze wiring ------------------------------------------------------


def test_freeze_targets_the_dispatcher_not_the_rest_entry():
    freeze = _load("freeze")
    assert freeze.ENTRY.name == "entry.py", (
        "freeze.py points at %s; freezing server.py directly is exactly the "
        "one-name-two-protocols bug this dispatcher replaces" % freeze.ENTRY.name
    )
    assert freeze.NAME == "gaia-agent"


def test_freeze_hidden_imports_carry_both_transports():
    """The dispatcher imports each transport inside a branch, which PyInstaller's
    static analysis does not follow -- so both need an explicit hidden-import."""
    source = (PACKAGING / "freeze.py").read_text(encoding="utf-8")
    for module in ("gaia_agent.stdio", "gaia_agent.server"):
        assert f'"{module}"' in source, (
            f"{module} has no --hidden-import in freeze.py; the frozen binary "
            "would lose that transport"
        )
