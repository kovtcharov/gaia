# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Live skill-behavior eval — does a loaded skill make the agent *act*?

Every other skill suite in this repo stops at "the manifest is honest". This one
loads a skill into the real flagship agent, runs a real turn against a real
Lemonade model, and requires two independent signals to agree (see
:mod:`gaia.eval.skill_behavior`):

* the skill's tools were actually called — recorded outside the model, so it
  cannot be argued with; and
* a Claude judge accepts the answer against the scenario's rubric.

Each scenario plants an unguessable token in its fixture, so an answer that
merely *sounds* right cannot pass: the token exists nowhere in any training set.

Requires:
- Lemonade at ``LEMONADE_BASE_URL`` (bare host, e.g. ``http://127.0.0.1:13305``).
- ``ANTHROPIC_API_KEY`` for the judge. Absent ⇒ the module FAILS, never skips —
  an un-judged run is not evidence (CLAUDE.md fail-loudly).

Run locally (one at a time — the Lemonade slot is single-tenant)::

    LEMONADE_BASE_URL=http://127.0.0.1:13305 \\
    python -m pytest tests/integration/eval/test_skill_behavior_e2e.py \\
        -m real_model -v

CI: ``.github/workflows/test_skill_behavior_eval.yml`` on ``[self-hosted,
Windows, stx]``, serialized against the other evals by the ``lemonade-eval``
concurrency group.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import requests

from gaia.eval.skill_behavior import (
    SkillScenario,
    new_planted_token,
    run_skill_scenario,
)

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.real_model

#: Skills exercised here run entirely on local tools — no network, no OAuth
#: connector, no `gh` auth — so the scenario measures the skill rather than the
#: runner's credentials.
SKILLS_UNDER_TEST = ("data-explore", "document-brief")


def _lemonade_reachable() -> bool:
    base = os.environ.get("LEMONADE_BASE_URL", "http://127.0.0.1:13305")
    health = base.removesuffix("/api/v1").rstrip("/") + "/api/v1/health"
    try:
        return requests.get(health, timeout=5).status_code == 200
    except requests.RequestException:
        return False


@pytest.fixture(scope="module")
def require_real_model():
    """Skip the module when there is no backend to measure against."""
    if not _lemonade_reachable():
        pytest.skip(
            "Lemonade not reachable — set LEMONADE_BASE_URL and start the "
            "server (installer/scripts/ensure-lemonade-running.ps1)."
        )


@pytest.fixture(scope="module")
def judge():
    """The Claude judge. A missing credential FAILS the run, never skips it."""
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.fail(
            "ANTHROPIC_API_KEY is not set, so no scenario could be judged. "
            "This is a hard failure rather than a skip: a run that cannot "
            "judge produces no evidence, and a green skip would read as one."
        )
    from gaia.eval.claude import ClaudeClient

    return ClaudeClient(max_tokens=512, temperature=0)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """Fixtures carrying tokens that cannot be guessed or recalled."""
    root = tmp_path_factory.mktemp("skill_behavior")

    revenue_token = new_planted_token("rev")
    csv = root / "quarterly_revenue.csv"
    csv.write_text(
        "region,quarter,revenue\n"
        "North,Q1,120000\n"
        "South,Q1,95000\n"
        f"{revenue_token},Q1,125000\n",
        encoding="utf-8",
    )

    doc_token = new_planted_token("doc")
    memo = root / "policy_memo.txt"
    memo.write_text(
        "Internal policy memo.\n\n"
        f"The mandatory rollback codeword for a failed deploy is {doc_token}.\n"
        "All engineers must quote it when filing an incident.\n",
        encoding="utf-8",
    )

    return {
        "root": root,
        "csv": csv,
        "revenue_token": revenue_token,
        "memo": memo,
        "doc_token": doc_token,
    }


@pytest.fixture(scope="module")
def flagship(require_real_model, planted):
    """A real GaiaAgent scoped to the fixture directory."""
    from gaia_agent.agent import GaiaAgent, GaiaAgentConfig

    agent = GaiaAgent(
        GaiaAgentConfig(
            silent_mode=True,
            allowed_paths=[str(planted["root"])],
        )
    )
    return agent


def _scenarios(planted) -> dict:
    """Scenario table, built around the planted tokens."""
    return {
        "data-explore": SkillScenario(
            id="data_explore_totals",
            skill="data-explore",
            query=(
                f"Load {planted['csv']} into a table and tell me the total Q1 "
                "revenue across all regions, and which region row carries the "
                "identifier that is not a normal region name."
            ),
            must_call=("create_table", "query_data"),
            must_contain=("340000", planted["revenue_token"]),
            must_not_contain=("I would", "you could run"),
            rubric=(
                "The answer must report a total Q1 revenue of 340000 (commas or "
                "a currency symbol are fine) AND name the unusual identifier "
                f"{planted['revenue_token']}. It must report what it found, not "
                "describe what it would do."
            ),
        ),
        "document-brief": SkillScenario(
            id="document_brief_codeword",
            skill="document-brief",
            query=(
                f"Index {planted['memo']} and tell me the mandatory rollback "
                "codeword for a failed deploy."
            ),
            must_call=("index_document", "query_documents"),
            must_contain=(planted["doc_token"],),
            must_not_contain=("I would", "cannot access"),
            rubric=(
                "The answer must quote the rollback codeword "
                f"{planted['doc_token']} taken from the indexed document. An "
                "answer that guesses, or that says it cannot read the file, is "
                "a FAIL."
            ),
        ),
    }


@pytest.mark.parametrize("skill_name", SKILLS_UNDER_TEST)
def test_skill_actually_runs_its_procedure(skill_name, flagship, judge, planted):
    """A loaded skill must drive real tool calls, not a plausible paragraph."""
    scenario = _scenarios(planted)[skill_name]

    # The agent is module-scoped (construction is expensive), so clear the
    # transcript between scenarios: otherwise the second scenario can answer
    # out of the first one's context and score a pass without calling anything.
    if getattr(flagship, "conversation_history", None):
        flagship.conversation_history.clear()

    result = run_skill_scenario(scenario, flagship, judge=judge)

    logger.info(
        "scenario=%s tools=%s verdict=%s",
        result.scenario_id,
        result.tools_called,
        result.judge_verdict,
    )

    assert result.passed, (
        f"skill {skill_name!r} failed its behavioural check:\n  - "
        + "\n  - ".join(result.failures)
        + f"\n\ntools called: {result.tools_called or 'none'}"
        + f"\nanswer: {result.answer[:800]}"
    )


def test_every_skill_under_test_is_actually_loadable(flagship):
    """Guards the guard: an unloadable skill would error, not silently pass."""
    for name in SKILLS_UNDER_TEST:
        flagship.load_skill(name)
        assert name in flagship.loaded_skills


def test_the_planted_tokens_are_not_in_the_repo(planted):
    """A token that leaked into the tree could be answered without the tool."""
    repo_root = Path(__file__).resolve().parents[3]
    for key in ("revenue_token", "doc_token"):
        token = planted[key]
        assert not list(repo_root.glob(f"**/*{token}*")), (
            f"planted token {token} appears in the repo tree — it must exist "
            "only inside the tmp fixture, or the check proves nothing."
        )
