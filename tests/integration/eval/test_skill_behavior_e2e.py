# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Live skill behaviour validation — every shipped skill, against a real model.

The sibling of ``test_behavior_e2e.py``. That one drives one agent over HTTP;
this one loads each skill into an in-process agent and watches whether its tools
actually run. In-process on purpose: the evidence is the agent's own tool
transcript, and there is no HTTP surface that loads an arbitrary skill into a
session.

What the run produces is not just a pass/fail — it is the **behaviour manifest**
that ``gaia skill publish`` reads. A skill absent from it, or stale in it, cannot
be published. So this file is the producer for
:mod:`gaia.skills.behavior_gate`, not merely a test of it.

Requires a live Lemonade server at ``LEMONADE_BASE_URL``. Gated by
``@pytest.mark.real_model``; it runs on ``[self-hosted, strix-halo]`` via
``.github/workflows/test_skill_behavior_e2e.yml``, and locally with::

    LEMONADE_BASE_URL=http://localhost:13305/api/v1 \\
    python -m pytest tests/integration/eval/test_skill_behavior_e2e.py \\
        -m real_model -v

**Do not run this concurrently with any other agent eval.** Every scenario forces
Lemonade to serve a model, and the backend is single-tenant per model slot — two
runs race-evict each other and produce failures that read like regressions.
"""

from __future__ import annotations

import json
import logging
import os

import pytest
import requests

from gaia.eval.skill_behavior import (
    SkillBehaviorHarness,
    SkillStatus,
)
from gaia.eval.skill_scenarios import SKILL_SCENARIOS, skill_roots
from gaia.skills.behavior_gate import write_manifest

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.real_model


def _lemonade_reachable() -> bool:
    base_url = os.environ.get("LEMONADE_BASE_URL", "http://localhost:13305/api/v1")
    health_url = base_url.removesuffix("/api/v1").rstrip("/") + "/api/v1/health"
    try:
        return requests.get(health_url, timeout=5).status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="module")
def require_real_model():
    """Skip the module when Lemonade is absent — never silently pass it."""
    if not _lemonade_reachable():
        pytest.skip(
            "Lemonade server not reachable — skipping real_model skill behaviour "
            "validation. Set LEMONADE_BASE_URL and ensure the server is running. "
            "NOTE: a skipped run validates nothing; the manifest is unchanged and "
            "every skill stays as unvalidated as it was."
        )


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    """Where the manifest and transcripts land for the workflow to upload."""
    directory = tmp_path_factory.mktemp("skill-behavior") / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


@pytest.fixture(scope="module")
def results(require_real_model, artifacts, tmp_path_factory):
    """Run every scenario once, serially, and record what happened.

    Module-scoped and serial by construction: the Lemonade backend is
    single-tenant per model slot, so parallel scenarios would evict each other's
    model and produce chaotic failures that read as skill regressions.
    """
    harness = SkillBehaviorHarness(
        skill_roots=skill_roots(),
        workspace_root=tmp_path_factory.mktemp("skill-workspaces"),
    )
    collected = [harness.run(scenario) for scenario in SKILL_SCENARIOS]

    (artifacts / "transcripts.json").write_text(
        json.dumps({r.skill: r.transcripts for r in collected}, indent=2, default=str),
        encoding="utf-8",
    )
    write_manifest(artifacts / "skill_behavior_validation.json", collected)

    for result in collected:
        logger.info(
            "%s -> %s %s", result.skill, result.status.value, result.reason or ""
        )
    return {r.skill: r for r in collected}


@pytest.mark.real_model
@pytest.mark.parametrize("skill", [s.skill for s in SKILL_SCENARIOS], ids=lambda s: s)
def test_skill_behaviour(results, skill, artifacts):
    """Each skill must run its declared tools and leave the expected side effect.

    A ``false_success`` — the agent reporting the work as done while its tools
    left no trace — is the #1428 regression class and is called out separately,
    because it is the failure that a human reading the reply would not notice.
    """
    result = results[skill]

    assert not result.hard_fail, (
        f"'{skill}' produced a FALSE SUCCESS: the agent claimed the work was done "
        f"while its tools left no side effect. Counts: {result.counts}. "
        f"Transcripts in {artifacts}."
    )
    assert result.status is not SkillStatus.blocked, (
        f"'{skill}' could not be validated: {result.reason} A skill whose "
        "validation was skipped must never ship as if it had passed — either "
        "resolve the blocker or stop shipping the skill."
    )
    assert result.status is SkillStatus.validated, (
        f"'{skill}' failed behaviour validation: {result.reason} Counts: "
        f"{result.counts}. Transcripts in {artifacts}."
    )


@pytest.mark.real_model
def test_the_run_covers_every_shipped_skill(results):
    """A manifest missing a skill would let that skill publish unnoticed."""
    from gaia.eval.skill_scenarios import shipped_skill_dirs

    missing = sorted({d.name for d in shipped_skill_dirs()} - set(results))
    assert not missing, (
        f"These shipped skills were not exercised: {missing}. The manifest this "
        "run writes is what the publish gate reads, so a gap here is a skill that "
        "could ship unvalidated."
    )
