# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
The behaviour workflow's load-bearing settings, asserted so an edit cannot undo them.

Two of these were learned the hard way on the Strix Halo runner and are invisible
when wrong — the workflow still goes green, it just stops validating anything:

- **One PowerShell session.** Lemonade is a detached ``Start-Process`` child and
  does not survive a step boundary. Split the start from the run and every
  scenario skips with "Lemonade server not reachable" while the job passes.
- **NO_PROXY on loopback.** Without it the runner proxy swallows requests to both
  the Lemonade server and the harness's own fixture HTTP server.

The rest are about coverage: a ``paths:`` trigger that misses a skill directory
means a changed skill never gets re-validated, and its stale record keeps
publishing it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test_skill_behavior_e2e.yml"

#: Directories a shipped skill can live in — each needs a paths: trigger.
SKILL_PATH_PREFIXES = (
    "hub/skills/",
    "hub/agents/gaia/python/gaia_agent/skills/",
    "hub/agents/email/python/gaia_agent_email/skills/",
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW.is_file(), f"{WORKFLOW} is missing"
    # PyYAML 1.1 parses a bare `on:` key as the boolean True, hence the lookup.
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def job(workflow: dict) -> dict:
    jobs = workflow["jobs"]
    assert len(jobs) == 1, f"expected one job, found {sorted(jobs)}"
    return next(iter(jobs.values()))


@pytest.fixture(scope="module")
def run_step(job: dict) -> dict:
    for step in job["steps"]:
        if "start-lemonade.ps1" in (step.get("run") or ""):
            return step
    pytest.fail("No step starts Lemonade — the harness would skip every scenario.")


def test_it_runs_on_the_strix_halo_runner(job: dict):
    """A hosted runner has no NPU/GPU and no model — every scenario would skip."""
    assert job["runs-on"] == ["self-hosted", "strix-halo"]


def test_lemonade_starts_and_the_tests_run_in_one_step(run_step: dict, job: dict):
    """The detached Lemonade child does not survive a step boundary."""
    assert "pytest" in run_step["run"], (
        "Lemonade is started in a step that does not also run pytest. The "
        "detached Start-Process child dies at the step boundary, so the harness "
        "skips with 'Lemonade server not reachable' — and the job still passes."
    )
    starters = [s for s in job["steps"] if "start-lemonade.ps1" in (s.get("run") or "")]
    assert len(starters) == 1


def test_the_lemonade_pid_is_stopped_in_a_finally(run_step: dict):
    """A leaked llama-server wedges the next run's model slot."""
    body = run_step["run"]
    assert "finally" in body and "Stop-Process" in body


def test_loopback_bypasses_the_runner_proxy(run_step: dict):
    """Covers both Lemonade and the harness's own fixture HTTP server."""
    no_proxy = run_step["env"]["NO_PROXY"]
    assert "127.0.0.1" in no_proxy and "localhost" in no_proxy


def test_the_failure_is_not_swallowed(run_step: dict):
    """PowerShell does not fail a step on a non-zero native exit code."""
    assert "$LASTEXITCODE" in run_step["run"], (
        "Without an explicit $LASTEXITCODE check, a failing pytest leaves the "
        "step green — PowerShell does not propagate a native exit code."
    )


def test_every_skill_directory_retriggers_the_run(workflow: dict):
    """A changed skill that does not retrigger keeps publishing on a stale record."""
    triggers = workflow.get("on") or workflow.get(True)
    paths = triggers["push"]["paths"]
    for prefix in SKILL_PATH_PREFIXES:
        assert any(p.startswith(prefix) for p in paths), (
            f"No paths: trigger covers {prefix}. A skill changed there would "
            "never be re-validated, and its stale record would keep it shippable."
        )


def test_the_harness_and_the_gate_retrigger_the_run(workflow: dict):
    """Changing the evidence rules must re-earn every verdict."""
    triggers = workflow.get("on") or workflow.get(True)
    paths = set(triggers["push"]["paths"])
    for module in (
        "src/gaia/eval/skill_behavior.py",
        "src/gaia/eval/skill_scenarios.py",
        "src/gaia/skills/behavior_gate.py",
    ):
        assert module in paths, f"{module} does not retrigger the workflow"


def test_the_host_agents_are_installed(job: dict):
    """Without them every scenario records `blocked` — honest, but validates nothing."""
    body = " ".join(s.get("run") or "" for s in job["steps"])
    for package in (
        "hub/agents/gaia/python",
        "hub/agents/email/python",
    ):
        assert package in body, f"{package} is never installed"


def test_artifacts_upload_on_success_as_well_as_failure(job: dict):
    """On success the manifest IS the deliverable — it is what gets committed."""
    uploads = [s for s in job["steps"] if "upload-artifact" in (s.get("uses") or "")]
    assert uploads, "nothing uploads the manifest or the transcripts"
    assert any(s.get("if") == "always()" for s in uploads), (
        "The manifest only uploads on failure. On a passing run it is the "
        "artifact to commit, so it has to survive success too."
    )


def test_runs_are_serialized(workflow: dict):
    """Two runs would race-evict each other's model in the single-tenant backend."""
    group = workflow["concurrency"]["group"]
    assert group == "skill-behavior-e2e", (
        "The concurrency group must be constant, not keyed on ref — two runs on "
        "different branches would still share the one Lemonade model slot."
    )
