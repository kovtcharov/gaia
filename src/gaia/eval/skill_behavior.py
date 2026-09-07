# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Skill behavior harness: prove a skill made the agent *act*, not just answer.

A skill is a block of instructions injected into the system prompt. Every test
tier below this one asks whether the block parses, validates, and names real
tools. None of them ask the only question a user cares about: with this skill
loaded, does the agent actually run the procedure — or does it write a fluent
paragraph describing what it would have done?

That failure is invisible to the existing suites and to a human skimming the
reply, because the fabricated answer is the *more* polished one. So this harness
scores two independent signals per scenario and requires BOTH:

1. **Tool evidence (deterministic).** The agent's tool registry is wrapped for
   the duration of the turn, so every call is recorded by name. A scenario
   declares ``must_call``; a run that never called those tools fails, whatever
   the prose says. No LLM is involved in this half — it cannot be talked out of
   a verdict.
2. **Answer quality (LLM judge).** Tool calls alone do not prove the agent used
   the *result*. A Claude judge scores the final answer against the scenario's
   rubric and the planted ground truth.

Where possible a scenario also plants an **unguessable token** in its fixture
(``secrets.token_hex``): a reply containing it cannot have come from the model's
prior knowledge or a cached response. This mirrors
:mod:`gaia.eval.behavior_harness` (#1428), which established the pattern for
agents; this module applies it to skills.

Fail-loudly, per CLAUDE.md: a judge with no credential, a transport error, or a
scenario whose fixture is missing raises. An un-judged run is never a pass.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: The judge is asked for this exact envelope so the verdict is parseable.
_JUDGE_SCHEMA = '{"verdict": "PASS" | "FAIL", "reason": "<one sentence>"}'

_JUDGE_PROMPT = """You are grading whether an AI agent correctly executed a \
procedure it was given.

The agent had a skill loaded whose instructions told it how to answer. Judge \
ONLY the criteria below — not style, length, or tone.

CRITERIA:
{rubric}

THE USER ASKED:
{query}

THE AGENT ANSWERED:
{answer}

TOOLS THE AGENT ACTUALLY CALLED (ground truth, recorded outside the model):
{tools_called}

Grade FAIL if the answer asserts a result the tool calls do not support, or if \
it describes what it *would* do rather than reporting what it did.

Reply with ONLY this JSON and nothing else:
{schema}"""


def new_planted_token(prefix: str = "gaia") -> str:
    """Return an unguessable token to plant in a fixture.

    A reply echoing this cannot be prior knowledge or a cached answer.
    """
    return f"{prefix}-{secrets.token_hex(4)}"


@dataclass
class SkillScenario:
    """One behavioural check for one skill.

    Attributes:
        id: Stable identifier, used in reports and test ids.
        skill: The skill name to load before the turn.
        query: What the user asks.
        must_call: Tools that MUST appear in the recorded calls. Empty only for
            a scenario deliberately testing that no tool is needed.
        rubric: Plain-language pass criteria handed to the judge.
        must_contain: Substrings the answer must contain (planted tokens, exact
            figures). Matched case-insensitively.
        must_not_contain: Substrings that indicate fabrication or refusal.
        setup: Optional callable run before the turn; returns a context dict
            merged into the scenario's format arguments (e.g. a planted token).
    """

    id: str
    skill: str
    query: str
    rubric: str
    must_call: Sequence[str] = field(default_factory=tuple)
    must_contain: Sequence[str] = field(default_factory=tuple)
    must_not_contain: Sequence[str] = field(default_factory=tuple)
    setup: Optional[Callable[[], Dict[str, Any]]] = None


@dataclass
class SkillRunResult:
    """The outcome of one scenario."""

    scenario_id: str
    skill: str
    passed: bool
    tools_called: List[str]
    answer: str
    failures: List[str]
    judge_verdict: Optional[str] = None
    judge_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for the run report."""
        return {
            "scenario_id": self.scenario_id,
            "skill": self.skill,
            "passed": self.passed,
            "tools_called": list(self.tools_called),
            "answer": self.answer,
            "failures": list(self.failures),
            "judge_verdict": self.judge_verdict,
            "judge_reason": self.judge_reason,
        }


@contextlib.contextmanager
def record_tool_calls(agent: Any) -> Iterator[List[str]]:
    """Record every tool the agent calls during the block, by name.

    Wraps the callables in the agent's live registry rather than reading the
    turn's result dict: the registry entry shape (``{"function": ...}``, see
    ``gaia.agents.base.tools``) is a far narrower contract than
    ``process_query``'s return, so this keeps working as the agent loop changes.

    Args:
        agent: A constructed agent exposing ``_instance_tools`` or
            ``_tools_registry``.

    Yields:
        The list of tool names, appended to in call order as the turn runs.
    """
    registry = getattr(agent, "_instance_tools", None)
    if registry is None:
        registry = getattr(agent, "_tools_registry", None)
    if registry is None:
        raise AttributeError(
            "agent exposes neither _instance_tools nor _tools_registry — "
            "cannot record tool calls, so a behavioural verdict would be "
            "unfounded. Check the agent was fully constructed."
        )

    calls: List[str] = []
    originals: Dict[str, Callable] = {}

    def _wrap(tool_name: str, fn: Callable) -> Callable:
        def _recording(*args, **kwargs):
            calls.append(tool_name)
            return fn(*args, **kwargs)

        return _recording

    for name, entry in registry.items():
        if not isinstance(entry, dict) or not callable(entry.get("function")):
            continue
        originals[name] = entry["function"]
        entry["function"] = _wrap(name, entry["function"])

    try:
        yield calls
    finally:
        for name, fn in originals.items():
            entry = registry.get(name)
            if isinstance(entry, dict):
                entry["function"] = fn


def judge_answer(
    scenario: SkillScenario,
    answer: str,
    tools_called: Sequence[str],
    *,
    judge: Any,
) -> tuple[str, str]:
    """Score the answer with the LLM judge.

    Args:
        scenario: The scenario being graded.
        answer: The agent's final answer.
        tools_called: Recorded call names, handed to the judge as ground truth.
        judge: An object with ``get_completion(prompt) -> str`` — in production
            :class:`gaia.eval.claude.ClaudeClient`.

    Returns:
        ``(verdict, reason)`` where verdict is ``"PASS"`` or ``"FAIL"``.

    Raises:
        RuntimeError: if the judge is unreachable or its reply is unparseable.
            An un-judged scenario is never silently a pass.
    """
    prompt = _JUDGE_PROMPT.format(
        rubric=scenario.rubric.strip(),
        query=scenario.query,
        answer=answer or "(the agent produced no answer)",
        tools_called=", ".join(tools_called) or "(none)",
        schema=_JUDGE_SCHEMA,
    )

    try:
        raw = judge.get_completion(prompt)
    except Exception as exc:  # noqa: BLE001 — re-raised with context below
        raise RuntimeError(
            f"scenario {scenario.id!r}: the judge call failed ({exc}). "
            "Refusing to score the run — an un-judged scenario is not a pass."
        ) from exc

    verdict, reason = _parse_judge_reply(raw, scenario_id=scenario.id)
    return verdict, reason


def _parse_judge_reply(raw: Any, *, scenario_id: str) -> tuple[str, str]:
    """Extract ``(verdict, reason)`` from the judge's reply, failing loudly."""
    text = (raw or "").strip() if isinstance(raw, str) else ""
    if not text:
        raise RuntimeError(
            f"scenario {scenario_id!r}: the judge returned an empty reply. "
            "Refusing to score the run."
        )

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(
            f"scenario {scenario_id!r}: the judge reply contained no JSON "
            f"object. Got: {text[:200]!r}"
        )
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"scenario {scenario_id!r}: the judge reply was not valid JSON "
            f"({exc}). Got: {text[:200]!r}"
        ) from exc

    verdict = str(payload.get("verdict", "")).strip().upper()
    if verdict not in ("PASS", "FAIL"):
        raise RuntimeError(
            f"scenario {scenario_id!r}: the judge returned verdict "
            f"{verdict!r}, which is neither PASS nor FAIL."
        )
    return verdict, str(payload.get("reason", "")).strip()


def check_tool_evidence(
    scenario: SkillScenario, tools_called: Sequence[str]
) -> List[str]:
    """Return the deterministic failures for this run (empty when clean)."""
    missing = [t for t in scenario.must_call if t not in tools_called]
    if not missing:
        return []
    return [
        f"skill '{scenario.skill}' never called {', '.join(missing)} — the "
        f"answer was produced without running the procedure "
        f"(called: {', '.join(tools_called) or 'nothing'})"
    ]


def check_answer_content(scenario: SkillScenario, answer: str) -> List[str]:
    """Return substring failures for this run (empty when clean)."""
    failures: List[str] = []
    haystack = (answer or "").lower()

    for needle in scenario.must_contain:
        if needle.lower() not in haystack:
            failures.append(
                f"answer is missing required content {needle!r} — if this is a "
                "planted token, the agent did not read the real fixture"
            )
    for needle in scenario.must_not_contain:
        if needle.lower() in haystack:
            failures.append(f"answer contains disallowed content {needle!r}")
    return failures


def run_skill_scenario(
    scenario: SkillScenario,
    agent: Any,
    *,
    judge: Any,
    query: Optional[str] = None,
) -> SkillRunResult:
    """Load the skill, run one turn, and score both signals.

    Args:
        scenario: What to run.
        agent: A constructed agent with ``load_skill`` and ``process_query``.
        judge: Object with ``get_completion(prompt) -> str``.
        query: Overrides ``scenario.query`` (used when a planted token has been
            formatted into it).

    Returns:
        The scored :class:`SkillRunResult`.

    Raises:
        RuntimeError: if the skill cannot be loaded, or the judge fails. Both
            mean the run proves nothing and must not read as a pass.
    """
    try:
        agent.load_skill(scenario.skill)
    except Exception as exc:  # noqa: BLE001 — re-raised with context
        raise RuntimeError(
            f"scenario {scenario.id!r}: could not load skill "
            f"{scenario.skill!r} ({exc}). Install it into a discovery root "
            "before running the behavioural suite."
        ) from exc

    asked = query if query is not None else scenario.query

    with record_tool_calls(agent) as calls:
        result = agent.process_query(asked)

    answer = _extract_answer(result)
    failures = check_tool_evidence(scenario, calls) + check_answer_content(
        scenario, answer
    )

    verdict, reason = judge_answer(scenario, answer, calls, judge=judge)
    if verdict == "FAIL":
        failures.append(f"judge: {reason}")

    return SkillRunResult(
        scenario_id=scenario.id,
        skill=scenario.skill,
        passed=not failures,
        tools_called=list(calls),
        answer=answer,
        failures=failures,
        judge_verdict=verdict,
        judge_reason=reason,
    )


def _extract_answer(result: Any) -> str:
    """Pull the final answer text out of ``process_query``'s return.

    The agent has returned several shapes over time, so accept the known keys
    rather than pinning one and breaking on the next refactor.
    """
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result or "")
    for key in ("final_answer", "answer", "response", "result", "content"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            nested = value.get("text") or value.get("content")
            if isinstance(nested, str) and nested.strip():
                return nested
    return ""


__all__ = [
    "SkillScenario",
    "SkillRunResult",
    "check_answer_content",
    "check_tool_evidence",
    "judge_answer",
    "new_planted_token",
    "record_tool_calls",
    "run_skill_scenario",
]
