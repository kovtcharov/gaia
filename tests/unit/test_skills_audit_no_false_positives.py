# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
The gate must be quiet about real, honest skills (issue #2468).

A security gate that cries wolf gets ignored, and then it protects nothing. The
strongest evidence available in-repo is the set of skills already checked in:
``.claude/skills/`` holds a dozen substantial, human-written skill bodies full of
shell commands, imperative instructions ("never", "always", "do not"), and quoted
examples — exactly the material a naive injection scanner mangles.

These are also the skills GAIA's own CI would audit, since the skill-audit
workflow matches ``**/skills/**``. If this test ever fails, either a rule got too
greedy or a real problem landed in a repo skill; both are worth stopping for.

``.claude/skills/`` is GAIA's own *tooling*, though — skills the maintainers run,
not skills users install. The ones that actually ship are the starter pack under
``hub/skills/`` and the agent-bundled sets under ``hub/agents/*/python/*/skills/``,
and nothing was asserting those audit clean. They are covered below on the same
terms: ALLOW with zero findings, clearing the tier each one claims. A published
skill that trips the gate is worse than a repo skill that does — a contributor
copies it as the example of a correct skill.

Those skills ship no Python, so on their own they only exercise the instruction
scanner. ``tests/fixtures/skills/report-archive`` is the matching guard for the
**AST** analyzer — an honest tool skill that reads and writes files, reads one
named environment variable, and posts to one host, with ``permissions:`` that
say exactly that. Over-tighten a code rule and it fails there first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gaia.skills.audit import audit_skill
from gaia.skills.audit.code import analyze_code
from gaia.skills.format import SKILL_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAUDE_SKILLS = REPO_ROOT / ".claude" / "skills"
BENIGN_TOOL_SKILL = REPO_ROOT / "tests" / "fixtures" / "skills" / "report-archive"

#: The skills users actually install: the starter pack plus every agent-bundled set.
SHIPPED_SKILL_ROOTS = (
    REPO_ROOT / "hub" / "skills",
    REPO_ROOT / "hub" / "agents" / "gaia" / "python" / "gaia_agent" / "skills",
    REPO_ROOT / "hub" / "agents" / "email" / "python" / "gaia_agent_email" / "skills",
)


def _skill_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if (d / SKILL_FILENAME).is_file())


def _shipped_skill_dirs() -> list[Path]:
    found: list[Path] = []
    for root in SHIPPED_SKILL_ROOTS:
        found.extend(_skill_dirs(root))
    return sorted(found, key=lambda d: d.name)


REPO_SKILLS = _skill_dirs(CLAUDE_SKILLS)
SHIPPED_SKILLS = _shipped_skill_dirs()


def test_the_repo_actually_has_skills_to_check_against():
    """Guards the parametrized tests below from silently covering nothing."""
    assert len(REPO_SKILLS) >= 5, (
        f"Expected several skills under {CLAUDE_SKILLS}; found "
        f"{[d.name for d in REPO_SKILLS]}. If they moved, retarget this test "
        "rather than deleting it — it is the anti-false-positive guard."
    )


@pytest.mark.parametrize("directory", REPO_SKILLS, ids=lambda d: d.name)
def test_a_real_repo_skill_audits_clean(directory: Path):
    """Every checked-in skill must ALLOW with zero findings.

    Not 'not BLOCK' — zero findings. An advisory finding on an honest skill is
    still noise a reader has to dismiss, and the whole gate's credibility rests
    on it staying at zero here.
    """
    report = audit_skill(directory)
    assert report.verdict == "ALLOW", [
        f"{f.severity} {f.rule_id} {f.location}: {f.message}" for f in report.findings
    ]
    assert report.findings == (), [
        f"{f.severity} {f.rule_id} {f.location}: {f.message}" for f in report.findings
    ]


@pytest.mark.parametrize("directory", REPO_SKILLS, ids=lambda d: d.name)
def test_a_real_repo_skill_clears_the_community_tier(directory: Path):
    """The advisory tier passing proves little; community is the real bar."""
    from gaia.skills.audit import clears_tier

    report = audit_skill(directory)
    assert clears_tier(report.findings, "community")


# ----------------------------------------------------------------------
# The same bar for the skills that actually ship
# ----------------------------------------------------------------------


def test_there_are_shipped_skills_to_check_against():
    """Guards the parametrized tests below from silently covering nothing."""
    assert len(SHIPPED_SKILLS) >= 15, (
        f"Expected the full shipped skill set; found {[d.name for d in SHIPPED_SKILLS]}. "
        f"If the directories moved, retarget SHIPPED_SKILL_ROOTS rather than "
        "deleting this test."
    )


#: ``info`` rules the shipped pack legitimately trips today. Pinned so the noise
#: floor cannot creep: a new rule id here is a deliberate decision, not a drift.
#:
#: ``permission.unused`` is a KNOWN FALSE POSITIVE on instruction-only skills.
#: Nine of the eleven starter skills declare ``network:read`` and ship no
#: ``tools.py`` at all, because the network is reached by the *host agent's*
#: ``fetch_page`` — which is exactly what the permission is for. The analyzer
#: reads code only, sees no domain use, and calls the declaration unused. It is
#: ``info`` so nothing is gated, but the fix belongs in
#: ``gaia.skills.audit.permission_truth``: an instruction-only skill's
#: ``tools_required`` should count as domain evidence. Left flagged rather than
#: silently suppressed.
KNOWN_INFO_RULES = frozenset(
    {"permission.unused", "tier.human_audit_required", "code.suppression"}
)


@pytest.mark.parametrize("directory", SHIPPED_SKILLS, ids=lambda d: d.name)
def test_a_shipped_skill_has_no_finding_that_gates(directory: Path):
    """Nothing at ``low`` or above.

    A shipped skill is the worked example every contributor copies. One carrying
    a real finding teaches the wrong shape, and the gate's credibility rests on
    the project's own pack being clean.
    """
    report = audit_skill(directory)
    gating = [f for f in report.findings if f.severity != "info"]
    assert not gating, [
        f"{f.severity} {f.rule_id} {f.location}: {f.message}" for f in gating
    ]


@pytest.mark.parametrize("directory", SHIPPED_SKILLS, ids=lambda d: d.name)
def test_a_shipped_skill_trips_no_unexpected_advisory_rule(directory: Path):
    """The ``info`` noise floor is pinned, so it cannot creep upward unnoticed."""
    report = audit_skill(directory)
    unexpected = sorted(
        {f.rule_id for f in report.findings if f.severity == "info"} - KNOWN_INFO_RULES
    )
    assert not unexpected, (
        f"'{directory.name}' trips advisory rule(s) {unexpected} that the shipped "
        "pack has not accepted. Either the skill needs fixing or the rule is too "
        "greedy — decide, then update KNOWN_INFO_RULES with the reason."
    )


@pytest.mark.parametrize("directory", SHIPPED_SKILLS, ids=lambda d: d.name)
def test_a_shipped_skill_clears_the_tier_it_claims(directory: Path):
    """The claim in the front matter has to be one the findings actually earn.

    ``verified`` is deliberately excluded: :data:`TIERS_REQUIRING_HUMAN_AUDIT`
    means the automated gate never grants it — a clean scan there earns REVIEW,
    "cleared the robot, awaiting the human". ``hub/skills/coding`` and
    ``gaia-voice`` claim it, so they are asserted on the strongest thing the
    engine *can* grant, and their publish still waits on a human audit plus the
    ``skill-audit-reviewed`` label. Do not relax the tier rule to make this green.
    """
    from gaia.skills.audit import TIERS_REQUIRING_HUMAN_AUDIT, clears_tier

    report = audit_skill(directory)
    tier = report.security_tier
    if tier in TIERS_REQUIRING_HUMAN_AUDIT:
        assert clears_tier(report.findings, "community"), (
            f"'{directory.name}' claims '{tier}', which needs a human audit on "
            "top of a clean scan — but it does not even clear 'community'."
        )
        return
    assert clears_tier(report.findings, tier), (
        f"'{directory.name}' claims tier '{tier}' but its findings clear only "
        f"{report.cleared_tiers or 'nothing'}."
    )


# ----------------------------------------------------------------------
# The same bar for a skill that actually ships Python
# ----------------------------------------------------------------------


def test_the_benign_tool_skill_really_exercises_the_ast_analyzer():
    """Guards the test below from passing on a skill that touches nothing.

    Without this, deleting ``tools.py`` would leave a green 'no false positives'
    test that proves nothing about the code analyzer.
    """
    analysis = analyze_code(BENIGN_TOOL_SKILL)
    observed = {(use.domain, use.level) for use in analysis.domain_uses}
    assert ("filesystem", "read") in observed
    assert ("filesystem", "write") in observed
    assert ("network", "write") in observed
    assert ("env", "read") in observed


def test_an_honest_tool_skill_audits_clean():
    """File I/O + a named env var + one HTTP POST, all declared: zero findings.

    This is the shape of most real tool skills. If a code rule ever starts
    flagging it, that rule is too greedy — narrow the rule, do not relax this.
    """
    report = audit_skill(BENIGN_TOOL_SKILL)
    assert report.findings == (), [
        f"{f.severity} {f.rule_id} {f.location}: {f.message}" for f in report.findings
    ]
    assert report.verdict == "ALLOW"


def test_an_honest_tool_skill_clears_the_community_tier():
    from gaia.skills.audit import clears_tier

    report = audit_skill(BENIGN_TOOL_SKILL)
    assert clears_tier(report.findings, "community")
