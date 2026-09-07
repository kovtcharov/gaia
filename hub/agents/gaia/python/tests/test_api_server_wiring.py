# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The contract ``gaia api`` depends on: build the flagship the way the server does.

``AgentRegistry.get_agent`` is the only path to an agent behind
``/v1/chat/completions``. It resolves ``AGENT_MODELS["gaia"]["class_name"]`` by
string, copies ``init_params``, and adds ``output_handler=SSEOutputHandler(...)``
before calling the class. Three things in that sentence can drift without any
other test noticing, because every test near this area substitutes a stand-in
agent: the dotted class path, the init-param names, and the ``output_handler``
field on ``ChatAgentConfig`` — ``GaiaAgent`` funnels ``**kwargs`` straight into a
dataclass, so a rename there is a ``TypeError`` at the first real request while
CI stays green.

These build the real class through the real registry. No ``_load_agent_class``
patch — resolving the string IS half of what is under test.
"""

from __future__ import annotations

import contextlib

import pytest
from gaia_agent.agent import GaiaAgent

from gaia.agents.base.tools import _TOOL_REGISTRY
from gaia.api.agent_registry import AGENT_MODELS, AgentRegistry
from gaia.api.sse_handler import SSEOutputHandler


@contextlib.contextmanager
def _isolated_registry():
    """@tool writes into a process-global dict shared with every other agent."""
    saved = dict(_TOOL_REGISTRY)
    _TOOL_REGISTRY.clear()
    try:
        yield
    finally:
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(saved)


@pytest.fixture
def served_agent(monkeypatch):
    """The flagship, constructed exactly as ``gaia api`` constructs it."""
    monkeypatch.setenv("GAIA_MEMORY_DISABLED", "1")
    with _isolated_registry():
        yield AgentRegistry().get_agent("gaia")


def test_the_registry_builds_the_real_flagship(served_agent):
    """The dotted class path in AGENT_MODELS resolves, and the kwargs fit it.

    A rename of the module, the class, or any ``init_params`` key raises here
    instead of at a user's first request.
    """
    assert isinstance(served_agent, GaiaAgent)


def test_the_agent_streams_through_the_handler_the_server_gave_it(served_agent):
    """``silent_mode=True`` in AGENT_MODELS must not win over ``output_handler``.

    The base ``Agent`` picks a console one of two ways — the handler it was
    handed, or one built from ``silent_mode``. ``AGENT_MODELS["gaia"]`` sets
    ``silent_mode=True``, so if ``output_handler`` ever stops reaching the base
    constructor the agent silently falls back to a console that streams
    nothing, and every SSE response comes back empty.
    """
    assert isinstance(served_agent.console, SSEOutputHandler)


def test_only_the_flagship_is_served():
    """One model, and it is the flagship — the per-task agents are skills now."""
    assert list(AGENT_MODELS) == ["gaia"]


def test_an_unknown_model_names_what_is_available():
    """The 404 a client sees must say what to ask for instead."""
    with pytest.raises(ValueError) as excinfo:
        AgentRegistry().get_agent("gaia-code")
    message = str(excinfo.value)
    assert "gaia-code" in message
    assert "gaia" in message
