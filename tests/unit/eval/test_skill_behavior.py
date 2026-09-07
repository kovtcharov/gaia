# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Offline guards for the skill behavior harness.

The harness itself is what decides whether a skill "works", so it needs its own
tests: a scorer that silently passes a fabricated answer is worse than no scorer
at all. Everything here runs without Lemonade, a model, or a judge credential —
the agent and the judge are plain stubs.
"""

from __future__ import annotations

import pytest

from gaia.eval.skill_behavior import (
    SkillScenario,
    check_answer_content,
    check_tool_evidence,
    judge_answer,
    new_planted_token,
    record_tool_calls,
    run_skill_scenario,
)


class _StubAgent:
    """Enough agent surface for the harness, with a scriptable turn."""

    def __init__(self, answer="done", calls_to_make=(), tools=("query_data",)):
        self._instance_tools = {
            name: {"name": name, "function": lambda *a, **k: f"{name}-result"}
            for name in tools
        }
        self._answer = answer
        self._calls_to_make = list(calls_to_make)
        self.loaded = []

    def load_skill(self, name):
        self.loaded.append(name)

    def process_query(self, _query):
        for name in self._calls_to_make:
            self._instance_tools[name]["function"]()
        return {"final_answer": self._answer}


class _StubJudge:
    """Returns a scripted reply; records the prompt it was given."""

    def __init__(self, reply='{"verdict": "PASS", "reason": "ok"}'):
        self.reply = reply
        self.prompts = []

    def get_completion(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _scenario(**kw):
    base = {
        "id": "s1",
        "skill": "data-explore",
        "query": "totals please",
        "rubric": "Reports a real total.",
    }
    base.update(kw)
    return SkillScenario(**base)


# ----------------------------------------------------------------------
# Tool recording — the deterministic half
# ----------------------------------------------------------------------


def test_record_tool_calls_captures_names_in_order():
    agent = _StubAgent(tools=("query_data", "create_table"))
    with record_tool_calls(agent) as calls:
        agent._instance_tools["create_table"]["function"]()
        agent._instance_tools["query_data"]["function"]()
    assert calls == ["create_table", "query_data"]


def test_record_tool_calls_restores_the_registry():
    """A leaked wrapper would make every later run record into a dead list."""
    agent = _StubAgent(tools=("query_data",))
    original = agent._instance_tools["query_data"]["function"]
    with record_tool_calls(agent):
        pass
    assert agent._instance_tools["query_data"]["function"] is original


def test_record_tool_calls_restores_even_when_the_turn_raises():
    agent = _StubAgent(tools=("query_data",))
    original = agent._instance_tools["query_data"]["function"]
    with pytest.raises(ValueError):
        with record_tool_calls(agent):
            raise ValueError("turn blew up")
    assert agent._instance_tools["query_data"]["function"] is original


def test_record_tool_calls_passes_through_the_return_value():
    agent = _StubAgent(tools=("query_data",))
    with record_tool_calls(agent):
        result = agent._instance_tools["query_data"]["function"]()
    assert result == "query_data-result"


def test_record_tool_calls_refuses_an_agent_with_no_registry():
    """Silently recording nothing would make every scenario vacuously pass."""

    class _Bare:
        pass

    with pytest.raises(AttributeError, match="cannot record tool calls"):
        with record_tool_calls(_Bare()):
            pass


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def test_missing_required_tool_is_a_failure():
    failures = check_tool_evidence(_scenario(must_call=("query_data",)), [])
    assert failures and "never called query_data" in failures[0]


def test_called_required_tool_is_clean():
    assert (
        check_tool_evidence(_scenario(must_call=("query_data",)), ["query_data"]) == []
    )


def test_missing_planted_token_is_a_failure():
    failures = check_answer_content(
        _scenario(must_contain=("gaia-abc123",)), "no token"
    )
    assert failures and "gaia-abc123" in failures[0]


def test_content_matching_is_case_insensitive():
    assert check_answer_content(_scenario(must_contain=("GAIA-ABC",)), "gaia-abc") == []


def test_disallowed_content_is_a_failure():
    failures = check_answer_content(
        _scenario(must_not_contain=("I would",)), "I would run the query"
    )
    assert failures and "I would" in failures[0]


def test_planted_tokens_are_unguessable_and_unique():
    assert new_planted_token() != new_planted_token()


# ----------------------------------------------------------------------
# The judge — fail loudly, never an un-judged pass
# ----------------------------------------------------------------------


def test_judge_parses_a_clean_verdict():
    verdict, reason = judge_answer(
        _scenario(), "42", ["query_data"], judge=_StubJudge()
    )
    assert (verdict, reason) == ("PASS", "ok")


def test_judge_tolerates_prose_around_the_json():
    judge = _StubJudge('Sure!\n{"verdict": "FAIL", "reason": "made it up"}\nDone.')
    verdict, reason = judge_answer(_scenario(), "42", [], judge=judge)
    assert verdict == "FAIL" and reason == "made it up"


def test_the_judge_is_told_which_tools_actually_ran():
    """The judge must be able to catch an answer the tool calls do not support."""
    judge = _StubJudge()
    judge_answer(_scenario(), "42", ["query_data"], judge=judge)
    assert "query_data" in judge.prompts[0]


def test_no_tool_calls_are_reported_as_none_not_blank():
    judge = _StubJudge()
    judge_answer(_scenario(), "42", [], judge=judge)
    assert "(none)" in judge.prompts[0]


@pytest.mark.parametrize(
    "reply",
    ["", "   ", "no json here", '{"verdict": "MAYBE"}', "{not valid json}"],
    ids=["empty", "whitespace", "no-json", "bad-verdict", "malformed"],
)
def test_an_unusable_judge_reply_raises_rather_than_passing(reply):
    with pytest.raises(RuntimeError):
        judge_answer(_scenario(), "42", [], judge=_StubJudge(reply))


def test_a_judge_transport_error_raises():
    judge = _StubJudge(ConnectionError("no route to host"))
    with pytest.raises(RuntimeError, match="judge call failed"):
        judge_answer(_scenario(), "42", [], judge=judge)


# ----------------------------------------------------------------------
# End-to-end scoring against a stub agent
# ----------------------------------------------------------------------


def test_a_real_run_passes_on_both_signals():
    agent = _StubAgent(answer="The total is 340000", calls_to_make=["query_data"])
    result = run_skill_scenario(
        _scenario(must_call=("query_data",), must_contain=("340000",)),
        agent,
        judge=_StubJudge(),
    )
    assert result.passed
    assert result.tools_called == ["query_data"]
    assert agent.loaded == ["data-explore"]


def test_a_fabricated_answer_fails_even_when_the_judge_says_pass():
    """The whole point: fluent prose with no tool call is the failure mode."""
    agent = _StubAgent(answer="The total is 340000", calls_to_make=[])
    result = run_skill_scenario(
        _scenario(must_call=("query_data",)), agent, judge=_StubJudge()
    )
    assert not result.passed
    assert any("never called query_data" in f for f in result.failures)


def test_a_judge_fail_sinks_a_run_that_called_its_tools():
    agent = _StubAgent(answer="I could not tell", calls_to_make=["query_data"])
    result = run_skill_scenario(
        _scenario(must_call=("query_data",)),
        agent,
        judge=_StubJudge('{"verdict": "FAIL", "reason": "no total reported"}'),
    )
    assert not result.passed
    assert any("no total reported" in f for f in result.failures)


def test_a_skill_that_cannot_load_raises_rather_than_scoring_zero():
    class _Refusing(_StubAgent):
        def load_skill(self, name):
            raise RuntimeError("not installed")

    with pytest.raises(RuntimeError, match="could not load skill"):
        run_skill_scenario(_scenario(), _Refusing(), judge=_StubJudge())


def test_the_result_serializes_for_the_report():
    agent = _StubAgent(answer="ok", calls_to_make=["query_data"])
    payload = run_skill_scenario(
        _scenario(must_call=("query_data",)), agent, judge=_StubJudge()
    ).to_dict()
    assert payload["skill"] == "data-explore"
    assert payload["tools_called"] == ["query_data"]
    assert payload["judge_verdict"] == "PASS"
