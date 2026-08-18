# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
The behaviour suite's own guards — coverage, honesty, and classification.

Two failure modes these exist to stop, both silent:

1. **A skill arrives with no scenario.** The behaviour workflow still goes green
   because it validated everything it *knew about*. ``test_every_shipped_skill…``
   is what turns that into a red build.
2. **A scenario that cannot fail.** Empty ``expect_tools``, a check that ignores
   the planted token, a prompt that never plants one — each produces a passing
   run that proves nothing. The shape assertions below hold that line.

Nothing here needs a model, a network, or an agent: the classification rules are
pure functions, and the scenario definitions are data.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gaia.eval.behavior_harness import Verdict
from gaia.eval.skill_behavior import (
    FixtureServer,
    ScenarioContext,
    SkillResult,
    SkillScenario,
    SkillStatus,
    ToolCall,
    ToolLedger,
    classify,
    verdict_from_side_effect,
)
from gaia.eval.skill_scenarios import SKILL_SCENARIOS, scenario_for, shipped_skill_dirs
from gaia.skills.format import parse_skill_metadata

#: The one skill whose correct behaviour is to call no tool at all. Its contract
#: is the agent's honesty floor, not a tool sequence, so it is the sole entry
#: allowed an empty ``expect_tools``.
PROSE_ONLY_SKILLS = {"gaia-voice"}


# ----------------------------------------------------------------------
# Coverage — a skill with no scenario cannot be validated, so it must fail here
# ----------------------------------------------------------------------


def test_the_repo_actually_ships_skills_to_cover():
    """Guards the parametrized tests below from silently covering nothing."""
    assert len(shipped_skill_dirs()) >= 15, (
        "Expected the full shipped skill set; found "
        f"{[d.name for d in shipped_skill_dirs()]}. If the directories moved, "
        "retarget SHIPPED_SKILL_DIRS rather than deleting this test — it is the "
        "coverage guard."
    )


def test_every_shipped_skill_has_a_behaviour_scenario():
    """A new skill without a scenario fails the build, not the next release."""
    shipped = {d.name for d in shipped_skill_dirs()}
    covered = {s.skill for s in SKILL_SCENARIOS}
    missing = sorted(shipped - covered)
    assert not missing, (
        f"These shipped skills have no behaviour scenario: {missing}. GAIA "
        "publishes only validated skills, and a skill with no scenario can never "
        "be validated. Add one to gaia.eval.skill_scenarios.SKILL_SCENARIOS."
    )


def test_no_scenario_covers_a_skill_that_is_not_shipped():
    """A scenario for a deleted skill is dead weight that reads as coverage."""
    shipped = {d.name for d in shipped_skill_dirs()}
    covered = {s.skill for s in SKILL_SCENARIOS}
    assert not sorted(covered - shipped)


def test_scenario_names_are_unique():
    names = [s.skill for s in SKILL_SCENARIOS]
    assert len(names) == len(set(names)), f"duplicate scenarios: {names}"


# ----------------------------------------------------------------------
# Shape — a scenario that cannot fail is worse than no scenario
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_a_scenario_demands_that_tools_actually_ran(scenario: SkillScenario):
    """``expect_tools`` is the floor: without it a check can pass on prose."""
    if scenario.skill in PROSE_ONLY_SKILLS:
        assert scenario.expect_tools == (), (
            f"'{scenario.skill}' is listed as prose-only but declares "
            f"expect_tools={scenario.expect_tools}. Remove it from "
            "PROSE_ONLY_SKILLS or drop the tools."
        )
        return
    if scenario.blocked_reason:
        pytest.skip(f"blocked: {scenario.blocked_reason}")
    assert scenario.expect_tools, (
        f"'{scenario.skill}' names no expected tools, so its run would pass "
        "whenever the agent said something plausible. Name at least one tool "
        "from its tools_required."
    )


@pytest.mark.parametrize("scenario", SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_expected_tools_are_ones_the_skill_declares(scenario: SkillScenario):
    """Expecting a tool the manifest never mentions tests the agent, not the skill."""
    if not scenario.expect_tools:
        return
    directory = {d.name: d for d in shipped_skill_dirs()}[scenario.skill]
    skill = parse_skill_metadata(directory)
    declared = set(skill.gaia.tools_required) | {t.name for t in skill.gaia.tools}
    unknown = sorted(set(scenario.expect_tools) - declared)
    assert not unknown, (
        f"'{scenario.skill}' expects {unknown}, which its SKILL.md does not "
        f"declare (declared: {sorted(declared)}). Either the scenario is testing "
        "the wrong thing or the manifest is out of date."
    )


@pytest.mark.parametrize("scenario", SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_a_blocked_scenario_says_why(scenario: SkillScenario):
    """A blocked skill must never read as validated — the reason is the record."""
    if scenario.blocked_reason is None:
        return
    assert len(scenario.blocked_reason) > 40, (
        f"'{scenario.skill}' is blocked with a reason too short to act on: "
        f"{scenario.blocked_reason!r}. Name what is missing and what unblocks it."
    )


def _context(tmp_path: Path, skill: str) -> ScenarioContext:
    server = FixtureServer().start()
    workspace = tmp_path / skill / "work"
    home = tmp_path / skill / "home"
    workspace.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    return ScenarioContext(
        skill=skill,
        run=0,
        workspace=workspace,
        home=home,
        token="deadbeef",
        fixtures=server,
    )


@pytest.mark.parametrize("scenario", SKILL_SCENARIOS, ids=lambda s: s.skill)
def test_a_scenario_plants_its_unguessable_token_somewhere_reachable(
    scenario: SkillScenario, tmp_path, monkeypatch
):
    """A token the agent can never reach makes the run pass on plausible prose.

    Reachable means one of: the prompt, a fixture the server will serve, a file
    the setup wrote into the sandbox, or a value the setup stashed for its check.
    """
    if scenario.blocked_reason:
        pytest.skip(f"blocked: {scenario.blocked_reason}")
    # A setup may prepend to PATH (github-triage installs a `gh` shim); keep that
    # out of the rest of the unit run.
    monkeypatch.setattr("os.environ", dict(os.environ))
    context = _context(tmp_path, scenario.skill)
    try:
        if scenario.setup is not None:
            scenario.setup(context)
        prompt = scenario.prompt_factory(context)
        haystack = [prompt]
        haystack.extend(str(v) for v in context.state.values())
        haystack.extend(context.fixtures.route_bodies())
        for path in context.workspace.parent.rglob("*"):
            if path.is_file():
                haystack.append(path.read_text(encoding="utf-8", errors="replace"))
    finally:
        context.fixtures.stop()

    assert prompt.strip(), f"'{scenario.skill}' produced an empty prompt"
    assert any(context.token in text for text in haystack), (
        f"'{scenario.skill}' plants its token nowhere the agent can reach it — "
        "not in the prompt, not in a served fixture, not in a sandbox file. The "
        "run would then pass on a plausible answer."
    )


# ----------------------------------------------------------------------
# The ledger
# ----------------------------------------------------------------------


def test_the_ledger_reads_the_agents_own_transcript():
    result = {
        "conversation": [
            {"role": "assistant", "content": "thinking"},
            {
                "role": "tool",
                "name": "rss-digest/fetch_rss",
                "tool_args": {"url": "http://127.0.0.1/feed.xml"},
                "content": "Release cafebabe",
            },
        ]
    }
    ledger = ToolLedger.from_result(result)
    assert ledger.called("fetch_rss")
    assert ledger.called("rss-digest/fetch_rss")
    assert ledger.names() == {"fetch_rss"}
    assert ledger.args_mention("feed.xml", tool="fetch_rss")
    assert ledger.result_mentions("cafebabe", tool="fetch_rss")
    assert not ledger.result_mentions("cafebabe", tool="fetch_page")


def test_an_empty_result_yields_an_empty_ledger():
    assert ToolLedger.from_result(None).calls == []
    assert ToolLedger.from_result({}).calls == []


# ----------------------------------------------------------------------
# Classification — false success must dominate
# ----------------------------------------------------------------------


def _scenario(**kwargs) -> SkillScenario:
    defaults = dict(
        skill="fake",
        prompt_factory=lambda _c: "go",
        expect_tools=("write_file",),
        side_effect_check=lambda _c, _r, _l: Verdict.true_success,
    )
    defaults.update(kwargs)
    return SkillScenario(**defaults)


def test_claiming_success_with_no_tool_call_is_a_hard_fail(tmp_path):
    context = _context(tmp_path, "fake")
    context.fixtures.stop()
    verdict = classify(
        _scenario(), context, "Done — the report has been created.", ToolLedger()
    )
    assert verdict is Verdict.false_success


def test_no_tool_call_and_no_claim_is_an_honest_failure(tmp_path):
    context = _context(tmp_path, "fake")
    context.fixtures.stop()
    verdict = classify(
        _scenario(), context, "I could not reach the page.", ToolLedger()
    )
    assert verdict is Verdict.honest_failure


def test_the_side_effect_check_only_runs_once_the_tools_did(tmp_path):
    """Ordering matters: a loose check must not rescue a no-tool-call run."""
    context = _context(tmp_path, "fake")
    context.fixtures.stop()
    scenario = _scenario(side_effect_check=lambda _c, _r, _l: Verdict.true_success)
    assert (
        classify(scenario, context, "All done!", ToolLedger()) is Verdict.false_success
    )

    ledger = ToolLedger(calls=[ToolCall(name="write_file", args={}, content="ok")])
    assert classify(scenario, context, "All done!", ledger) is Verdict.true_success


def test_expect_tools_is_a_menu_by_default(tmp_path):
    """Several skills list alternatives — picking the other one is not a failure."""
    context = _context(tmp_path, "fake")
    context.fixtures.stop()
    scenario = _scenario(expect_tools=("fetch_page", "search_web"))
    ledger = ToolLedger(calls=[ToolCall(name="search_web", args={}, content="ok")])
    assert classify(scenario, context, "Done.", ledger) is Verdict.true_success


def test_require_all_tools_demands_the_whole_conjunction(tmp_path):
    """For a skill whose steps genuinely chain, half the chain is a failure."""
    context = _context(tmp_path, "fake")
    context.fixtures.stop()
    scenario = _scenario(
        expect_tools=("fetch_page", "remember"), require_all_tools=True
    )
    half = ToolLedger(calls=[ToolCall(name="fetch_page", args={}, content="ok")])
    assert classify(scenario, context, "Done — recorded it.", half) is (
        Verdict.false_success
    )

    whole = ToolLedger(
        calls=[
            ToolCall(name="fetch_page", args={}, content="ok"),
            ToolCall(name="remember", args={}, content="ok"),
        ]
    )
    assert classify(scenario, context, "Done.", whole) is Verdict.true_success


def test_a_prose_only_scenario_delegates_entirely_to_its_check(tmp_path):
    context = _context(tmp_path, "fake")
    context.fixtures.stop()
    scenario = _scenario(
        expect_tools=(), side_effect_check=lambda _c, _r, _l: Verdict.honest_failure
    )
    assert (
        classify(scenario, context, "All done!", ToolLedger()) is Verdict.honest_failure
    )


@pytest.mark.parametrize(
    "present,reply,expected",
    [
        (True, "anything", Verdict.true_success),
        (False, "Successfully created it.", Verdict.false_success),
        (False, "I was unable to.", Verdict.honest_failure),
    ],
)
def test_verdict_from_side_effect(present, reply, expected):
    assert verdict_from_side_effect(present, reply) is expected


# ----------------------------------------------------------------------
# Fixture server
# ----------------------------------------------------------------------


@pytest.mark.allow_network
def test_the_fixture_server_serves_and_logs():
    server = FixtureServer().start()
    try:
        url = server.add_route("/feed.xml", "application/rss+xml", "<rss>hi</rss>")
        assert not server.served("/feed.xml")
        with urllib.request.urlopen(url, timeout=5) as response:
            assert response.read().decode() == "<rss>hi</rss>"
        assert server.served("/feed.xml")
        assert not server.served("/missing")
    finally:
        server.stop()


@pytest.mark.allow_network
def test_the_fixture_server_404s_an_unknown_path():
    server = FixtureServer().start()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(server.url("/nope"), timeout=5)
        assert excinfo.value.code == 404
        # A miss is still logged — a scenario asserting on a typo'd path must not
        # look like the agent simply never fetched anything.
        assert server.served("/nope")
    finally:
        server.stop()


# ----------------------------------------------------------------------
# The record a run produces
# ----------------------------------------------------------------------


def test_only_a_pass_gets_a_validated_at_stamp():
    passed = SkillResult(skill="x", status=SkillStatus.validated).to_record(
        content_digest="sha256:abc", version="1.0.0"
    )
    failed = SkillResult(skill="x", status=SkillStatus.failed, reason="nope").to_record(
        content_digest="sha256:abc", version="1.0.0"
    )
    assert passed["validated_at"]
    assert not failed["validated_at"], (
        "A failed record with a validated_at reads as 'it worked at some point'. "
        "Keep the field empty unless the run passed."
    )
    assert failed["recorded_at"]


def test_scenario_for_names_the_fix_when_a_skill_is_uncovered():
    with pytest.raises(KeyError) as excinfo:
        scenario_for("no-such-skill")
    assert "SKILL_SCENARIOS" in str(excinfo.value)
