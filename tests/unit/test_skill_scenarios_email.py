# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for the email agent's skill behaviour scenarios.

No model, no network, no Lemonade: everything here is the scenario metadata,
the mbox seeding, and the verdict logic driven by a hand-built
:class:`~gaia.eval.skill_behavior.ToolLedger`.

The load-bearing test is :func:`test_one_scenario_per_bundled_skill` — the
expected set is globbed from the shipped ``SKILL.md`` files, so adding a skill
without a behaviour scenario turns red instead of shipping unvalidated.
"""

from __future__ import annotations

import secrets
import sys
import types
from pathlib import Path

import pytest

from gaia.eval.skill_behavior import (
    FixtureServer,
    ScenarioContext,
    ScenarioUnavailable,
    ToolCall,
    ToolLedger,
    Verdict,
)
from gaia.eval.skill_scenarios_email import (
    EMAIL_SKILL_SCENARIOS,
    build_email_agent,
    email_skill_roots,
)
from gaia.skills.format import parse_skill_file

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = (
    REPO_ROOT / "hub" / "agents" / "email" / "python" / "gaia_agent_email" / "skills"
)

#: Reply text that trips ``_success_markers``, so an absent side effect has to
#: classify as ``false_success`` rather than an honest failure.
CLAIMS_SUCCESS = "Done — everything is now sorted."

SCENARIOS_BY_SKILL = {s.skill: s for s in EMAIL_SKILL_SCENARIOS}


def bundled_skill_names() -> set[str]:
    """The skills the email agent actually ships, from disk."""
    return {p.parent.name for p in SKILLS_DIR.glob("*/SKILL.md")}


@pytest.fixture
def context(tmp_path: Path) -> ScenarioContext:
    """A repeat context with a real token and a never-started fixture server."""
    workspace = tmp_path / "work"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    return ScenarioContext(
        skill="unset",
        run=0,
        workspace=workspace,
        home=home,
        token=secrets.token_hex(4),
        fixtures=FixtureServer(),
    )


def prepared(scenario, context: ScenarioContext) -> ScenarioContext:
    """Run ``scenario.setup`` against the context and return it."""
    context.skill = scenario.skill
    scenario.setup(context)
    return context


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_one_scenario_per_bundled_skill():
    shipped = bundled_skill_names()
    assert shipped, f"no SKILL.md files found under {SKILLS_DIR}"
    covered = {s.skill for s in EMAIL_SKILL_SCENARIOS}
    assert covered == shipped, (
        f"missing scenarios for {sorted(shipped - covered)}; "
        f"scenarios for skills that do not ship: {sorted(covered - shipped)}"
    )
    assert len(EMAIL_SKILL_SCENARIOS) == len(shipped) == 6


def test_email_skill_roots_points_at_the_bundled_skills():
    roots = email_skill_roots()
    assert roots
    assert {p.parent.name for p in roots[0].glob("*/SKILL.md")} == bundled_skill_names()


# ---------------------------------------------------------------------------
# Scenario shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", EMAIL_SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_scenario_shape(scenario):
    assert scenario.agent == "email"
    assert scenario.expect_tools, "a skill with no expected tools proves nothing"
    assert callable(scenario.side_effect_check)
    assert callable(scenario.setup)
    assert scenario.blocked_reason is None, (
        "a blocked skill must never read as validated — if this is set on "
        "purpose, assert the reason here explicitly"
    )


@pytest.mark.parametrize("scenario", EMAIL_SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_expect_tools_are_declared_by_the_skill(scenario):
    skill = parse_skill_file(SKILLS_DIR / scenario.skill / "SKILL.md")
    declared = set(skill.gaia.tools_required)
    assert declared, f"{scenario.skill} declares no tools_required"
    undeclared = set(scenario.expect_tools) - declared
    assert not undeclared, (
        f"{scenario.skill} expects {sorted(undeclared)}, which its SKILL.md "
        f"does not declare in tools_required ({sorted(declared)})"
    )


@pytest.mark.parametrize("scenario", EMAIL_SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_prompt_embeds_the_planted_token(scenario, context):
    prompt = scenario.prompt_factory(prepared(scenario, context))
    assert context.token in prompt, (
        "without the token in the prompt a cached or hallucinated reply could "
        "satisfy the scenario"
    )


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", EMAIL_SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_setup_seeds_a_readable_mailbox(scenario, context):
    prepared(scenario, context)

    mbox_path = Path(context.state["mbox_path"])
    assert mbox_path.is_file()
    assert mbox_path.parent == context.workspace

    ids = context.state["message_ids"]
    assert len(ids) >= 2
    assert set(ids) == set(context.state["seed_labels"])

    backend = context.state["gmail_backend"]
    listed = {m["id"] for m in backend.list_messages(max_results=100)["messages"]}
    assert set(ids.values()) <= listed


def test_setup_plants_the_token_where_a_tool_will_read_it(context):
    scenario = SCENARIOS_BY_SKILL["travel-itinerary"]
    prepared(scenario, context)
    backend = context.state["gmail_backend"]
    flight = backend.get_message(context.state["message_ids"]["flight"])
    subjects = [
        h["value"] for h in flight["payload"]["headers"] if h["name"] == "Subject"
    ]
    assert any(context.token in s for s in subjects)
    assert context.token in flight["snippet"]


def test_meeting_scenario_seeds_a_calendar_carrying_the_token(context):
    scenario = SCENARIOS_BY_SKILL["meeting-scheduling"]
    prepared(scenario, context)
    calendar = context.state["calendar_backend"]
    summaries = [e["summary"] for e in calendar.events.values()]
    assert any(context.token in s for s in summaries)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

#: For a read skill, the tool whose RESULT must carry the token back.
READBACK_TOOL = {
    "action-item-extraction": "extract_action_items",
    "inbox-triage": "triage_inbox",
    "meeting-scheduling": "list_calendar_events",
    "travel-itinerary": "get_message",
}

#: For a write skill, the mutation on the fake backend and the message it lands
#: on, keyed by the ``setup`` stash key.
WRITE_EFFECT = {
    "escalation-routing": ("add_star", "escalation"),
    "newsletter-digest": ("archive_message", "planted-newsletter"),
}


def apply_side_effect(scenario, context: ScenarioContext) -> ToolLedger:
    """Make the scenario's side effect real; return the ledger to check with."""
    if scenario.skill in READBACK_TOOL:
        tool = READBACK_TOOL[scenario.skill]
        content = f'{{"ok": true, "subject": "ref GX-{context.token}"}}'
        return ToolLedger(calls=[ToolCall(name=tool, args={}, content=content)])

    mutation, key = WRITE_EFFECT[scenario.skill]
    backend = context.state["gmail_backend"]
    getattr(backend, mutation)(context.state["message_ids"][key])
    return ToolLedger()


@pytest.mark.parametrize("scenario", EMAIL_SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_absent_side_effect_with_success_claim_is_false_success(scenario, context):
    prepared(scenario, context)
    verdict = scenario.side_effect_check(context, CLAIMS_SUCCESS, ToolLedger())
    assert verdict is Verdict.false_success


@pytest.mark.parametrize("scenario", EMAIL_SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_absent_side_effect_without_a_claim_is_an_honest_failure(scenario, context):
    prepared(scenario, context)
    verdict = scenario.side_effect_check(
        context, "I could not find anything matching that.", ToolLedger()
    )
    assert verdict is Verdict.honest_failure


@pytest.mark.parametrize("scenario", EMAIL_SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_present_side_effect_is_a_true_success(scenario, context):
    prepared(scenario, context)
    ledger = apply_side_effect(scenario, context)
    verdict = scenario.side_effect_check(context, CLAIMS_SUCCESS, ledger)
    assert verdict is Verdict.true_success


def test_readback_verdict_rejects_a_token_the_agent_only_echoed(context):
    """A token in the agent's PROSE is not evidence — only a tool result is."""
    scenario = SCENARIOS_BY_SKILL["travel-itinerary"]
    prepared(scenario, context)
    reply = f"Done — your booking GX-{context.token} departs at 08:40."
    assert scenario.side_effect_check(context, reply, ToolLedger()) is (
        Verdict.false_success
    )


def test_readback_verdict_rejects_a_token_that_only_reached_tool_ARGS(context):
    """The model can put a token it was given into arguments; that proves
    nothing about whether the mailbox actually held it."""
    scenario = SCENARIOS_BY_SKILL["travel-itinerary"]
    prepared(scenario, context)
    ledger = ToolLedger(
        calls=[
            ToolCall(
                name="get_message",
                args={"message_id": f"GX-{context.token}"},
                content='{"ok": true, "subject": "unrelated"}',
            )
        ]
    )
    verdict = scenario.side_effect_check(context, CLAIMS_SUCCESS, ledger)
    assert verdict is Verdict.false_success


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def test_build_email_agent_blocks_when_the_package_is_missing(context, monkeypatch):
    for name in (
        "gaia_agent_email",
        "gaia_agent_email.agent",
        "gaia_agent_email.config",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    with pytest.raises(ScenarioUnavailable) as excinfo:
        build_email_agent(context=context, skill_roots=[], max_steps=5)

    message = str(excinfo.value)
    assert "gaia-agent-email" in message
    assert "pip install" in message
    assert "hub/agents/email/python" in message


def install_stub_email_package(monkeypatch) -> type:
    """Put an importable stand-in for ``gaia_agent_email`` on ``sys.modules``."""

    class StubEmailAgent:
        AUTOLOAD_DECLARED_SKILLS = True

        def __init__(self, config=None):
            self.config = config

    package = types.ModuleType("gaia_agent_email")
    package.__path__ = []
    agent_module = types.ModuleType("gaia_agent_email.agent")
    agent_module.EmailTriageAgent = StubEmailAgent
    config_module = types.ModuleType("gaia_agent_email.config")
    config_module.EmailAgentConfig = dict

    monkeypatch.setitem(sys.modules, "gaia_agent_email", package)
    monkeypatch.setitem(sys.modules, "gaia_agent_email.agent", agent_module)
    monkeypatch.setitem(sys.modules, "gaia_agent_email.config", config_module)
    return StubEmailAgent


def test_build_email_agent_blocks_when_setup_seeded_no_mailbox(context, monkeypatch):
    install_stub_email_package(monkeypatch)

    with pytest.raises(ScenarioUnavailable) as excinfo:
        build_email_agent(context=context, skill_roots=[], max_steps=5)

    assert "gmail_backend" in str(excinfo.value)


def test_build_email_agent_injects_the_fake_backend_and_disables_autoload(
    context, monkeypatch
):
    scenario = SCENARIOS_BY_SKILL["inbox-triage"]
    prepared(scenario, context)
    install_stub_email_package(monkeypatch)

    agent = build_email_agent(context=context, skill_roots=[], max_steps=7)

    assert agent is not None
    assert agent.AUTOLOAD_DECLARED_SKILLS is False
    assert agent.config["gmail_backend"] is context.state["gmail_backend"]
    assert agent.config["mail_provider"] == "google"
    assert agent.config["max_steps"] == 7
    assert agent.config["memory_enabled"] is False
    assert agent.config["start_scheduler"] is False
