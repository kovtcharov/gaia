# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Skill behaviour validation: prove a skill's tools actually ran.

The sibling of :mod:`gaia.eval.behavior_harness`, aimed at a different unit.
That module validates an **agent** end-to-end over HTTP; this one validates a
**skill** — the instruction body that gets injected into an agent's context and
is supposed to make it call particular tools in a particular order.

Why a separate module rather than more ``Scenario`` entries next door:

- A skill has to be *loaded* into an agent before it can be exercised, and the
  agent must load **only** that skill, or nothing distinguishes "the skill drove
  this" from "the agent would have done it anyway".
- The evidence is different. An agent scenario asks "did a file appear". A skill
  scenario also has to ask "did the tools the skill *declares* actually get
  called", because a skill whose body is ignored still produces plausible prose.
- Most skills reach a network or a connector. Those need hermetic fixtures
  (a localhost server, a fake mailbox) which the agent harness has no notion of.

What it borrows: :class:`~gaia.eval.behavior_harness.Verdict`, the
false-success-is-a-hard-fail rule, and the planted-unguessable-token discipline.
A cached or hallucinated reply cannot pass a check keyed on a token minted this
second.

Two kinds of evidence, both required for a pass:

1. **The tool ledger** — ``process_query`` returns the turn's conversation, and
   every executed tool leaves a ``{"role": "tool", "name": ..., "tool_args":
   ...}`` entry in it. That is the agent's own record, not instrumentation this
   module added, so it cannot be wrong in a way that flatters the result.
2. **The side effect** — a file on disk, a scratchpad row, a request the fixture
   server actually served. Asserted on the artifact, never on the prose.

Run it::

    python -m gaia.eval.skill_behavior --skill rss-digest --output run.json

or through pytest (``tests/integration/eval/test_skill_behavior_e2e.py``, marked
``real_model``). Both need a live Lemonade backend; neither is part of the normal
unit run.
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import secrets
import socket
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from gaia.eval.behavior_harness import Verdict, _aggregate_verdicts, _success_markers

logger = logging.getLogger(__name__)

#: Bumped when the evidence rules change, so an old record cannot be read as if
#: it had cleared the current bar.
HARNESS_VERSION = "gaia-skill-behavior/1"


# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------


class SkillStatus(str, Enum):
    """The recorded outcome for one skill. Only ``validated`` clears the gate."""

    validated = "validated"
    """Every repeat ran the skill's tools and left the expected side effect."""

    failed = "failed"
    """The harness ran and the skill did not clear the bar."""

    blocked = "blocked"
    """The scenario could not run — a prerequisite is absent. **Not a pass.**"""

    unvalidated = "unvalidated"
    """No harness run has produced a verdict for this skill yet."""


class ScenarioUnavailable(RuntimeError):
    """A prerequisite for this scenario does not exist in this environment.

    Raised by a scenario's ``setup``/agent factory, and recorded as
    :attr:`SkillStatus.blocked` with the message as the reason. Never swallowed:
    a blocked skill stays blocked, and the publish gate refuses it exactly as it
    refuses a failure.
    """


# ---------------------------------------------------------------------------
# The tool ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One executed tool, as the agent itself recorded it."""

    name: str
    args: Dict[str, Any]
    content: str

    @property
    def bare_name(self) -> str:
        """The tool name without its ``<skill>/`` namespace prefix."""
        return self.name.rsplit("/", 1)[-1]


@dataclass
class ToolLedger:
    """Every tool the agent executed during one turn.

    Built from ``process_query``'s returned conversation rather than by wrapping
    ``_execute_tool``: reading the agent's own transcript means the ledger cannot
    disagree with what the agent actually did.
    """

    calls: List[ToolCall] = field(default_factory=list)

    @classmethod
    def from_result(cls, result: Optional[Dict[str, Any]]) -> "ToolLedger":
        calls: List[ToolCall] = []
        for message in (result or {}).get("conversation", []) or []:
            if message.get("role") != "tool":
                continue
            calls.append(
                ToolCall(
                    name=str(message.get("name") or ""),
                    args=dict(message.get("tool_args") or {}),
                    content=str(message.get("content") or ""),
                )
            )
        return cls(calls=calls)

    def names(self) -> set:
        """Bare tool names that ran, namespace prefixes stripped."""
        return {c.bare_name for c in self.calls if c.bare_name}

    def called(self, tool: str) -> bool:
        return tool.rsplit("/", 1)[-1] in self.names()

    def called_any(self, tools: Iterable[str]) -> bool:
        wanted = {t.rsplit("/", 1)[-1] for t in tools}
        return bool(wanted & self.names())

    def called_all(self, tools: Iterable[str]) -> bool:
        wanted = {t.rsplit("/", 1)[-1] for t in tools}
        return wanted <= self.names()

    def _matching(self, tool: Optional[str]) -> List[ToolCall]:
        if tool is None:
            return list(self.calls)
        bare = tool.rsplit("/", 1)[-1]
        return [c for c in self.calls if c.bare_name == bare]

    def args_mention(self, token: str, *, tool: Optional[str] = None) -> bool:
        """True when ``token`` appears in the arguments of a matching call."""
        needle = token.lower()
        return any(
            needle in json.dumps(c.args, default=str).lower()
            for c in self._matching(tool)
        )

    def result_mentions(self, token: str, *, tool: Optional[str] = None) -> bool:
        """True when ``token`` appears in the *result* of a matching call.

        This is the strong form: the token came back from a real tool execution,
        so the model could not have produced it from context alone.
        """
        needle = token.lower()
        return any(needle in c.content.lower() for c in self._matching(tool))

    def summary(self) -> List[str]:
        return [c.name for c in self.calls]


# ---------------------------------------------------------------------------
# Hermetic fixture server
# ---------------------------------------------------------------------------


class FixtureServer:
    """A localhost HTTP server serving planted-token pages, and logging hits.

    Skills that declare ``network:read`` are validated against this rather than
    the open web: a live fetch cannot carry a token minted this second, cannot be
    made deterministic, and turns a skill regression into a flaky network test.
    The request log is the side effect — it proves the agent's ``fetch_page``
    reached a specific URL, which no amount of plausible prose can fake.
    """

    def __init__(self, routes: Optional[Dict[str, Tuple[str, str]]] = None):
        """Args:
        routes: ``{path: (content_type, body)}``. Paths include the leading
            slash and are matched exactly, query string stripped.
        """
        self._routes = dict(routes or {})
        self.requests: List[str] = []
        self._lock = threading.Lock()
        self._server: Optional[http.server.ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port = 0

    @property
    def base_url(self) -> str:
        if not self.port:
            raise RuntimeError("FixtureServer.start() has not run yet")
        return f"http://127.0.0.1:{self.port}"

    def url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def add_route(self, path: str, content_type: str, body: str) -> str:
        """Register ``path`` and return the absolute URL to hand the agent."""
        with self._lock:
            self._routes[path] = (content_type, body)
        return self.url(path)

    def route_bodies(self) -> List[str]:
        """Every body this server will serve — what a scenario planted in it."""
        with self._lock:
            return [body for _content_type, body in self._routes.values()]

    def served(self, path: str) -> bool:
        """True when a request for exactly ``path`` was handled."""
        with self._lock:
            return path in self.requests

    def start(self) -> "FixtureServer":
        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                path = self.path.split("?", 1)[0]
                with outer._lock:
                    outer.requests.append(path)
                    route = outer._routes.get(path)
                if route is None:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                content_type, body = route
                payload = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                """Silence the default stderr access log."""

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self.port), _Handler
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("fixture server listening on %s", self.base_url)
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


@dataclass
class ScenarioContext:
    """Everything one repeat of a scenario is handed.

    ``token`` is minted per repeat, so every assertion keyed on it is proof the
    tool ran *this time* — not that it ran once, or that the model remembered a
    plausible answer from its training data.
    """

    skill: str
    run: int
    workspace: Path
    home: Path
    token: str
    fixtures: FixtureServer
    #: Free-form slot for a scenario's ``setup`` to hand its check what it built.
    state: Dict[str, Any] = field(default_factory=dict)


#: ``(context, reply, ledger) -> Verdict``
SideEffectCheck = Callable[["ScenarioContext", str, "ToolLedger"], Verdict]


@dataclass
class SkillScenario:
    """One skill's behaviour test.

    Attributes:
        skill: The skill's ``name``, matching its directory.
        prompt_factory: ``(context) -> str``. Must embed ``context.token``
            wherever the answer could otherwise be guessed or cached.
        expect_tools: The tools-ran floor, enforced before
            ``side_effect_check`` is consulted so a scenario cannot forget that
            half. By default **at least one** must appear in the ledger, which
            is what most skills want — several list alternatives the agent may
            legitimately choose between (``fetch_page`` *or* ``search_web``).
            Empty only for a skill whose correct behaviour is to call nothing
            — see ``gaia-voice`` — and a unit test holds that line.
        require_all_tools: Demand every entry in ``expect_tools``, for a skill
            whose steps are a genuine conjunction (fetch *and* remember). Left
            off where the tuple is a menu, since a small model picking the other
            valid tool is not a regression.
        side_effect_check: The artifact proof, run only once ``expect_tools``
            is satisfied.
        setup: Builds this repeat's fixtures. Raise
            :class:`ScenarioUnavailable` to record ``blocked`` with a reason.
        after_agent: ``(context, agent) -> None``, run once the agent exists and
            the skill is loaded. This is where an egress seam gets replaced — a
            search backend, say — because those live on the agent instance and
            cannot be installed before it is built.
        agent: Which agent hosts the skill — ``"gaia"`` or ``"email"``.
        blocked_reason: Set to declare the scenario un-runnable up front. A
            blocked skill is never reported as validated.
        repeats: Independent runs. Three by default: enough for a
            non-deterministic failure to show, cheap enough for 18 skills.
        max_steps: Agent step ceiling for the turn.
    """

    skill: str
    prompt_factory: Callable[["ScenarioContext"], str]
    expect_tools: Tuple[str, ...]
    side_effect_check: SideEffectCheck
    require_all_tools: bool = False
    setup: Optional[Callable[["ScenarioContext"], None]] = None
    after_agent: Optional[Callable[["ScenarioContext", Any], None]] = None
    agent: str = "gaia"
    blocked_reason: Optional[str] = None
    repeats: int = 3
    max_steps: int = 15


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify(
    scenario: SkillScenario,
    context: ScenarioContext,
    reply: str,
    ledger: ToolLedger,
) -> Verdict:
    """Turn one repeat's evidence into a :class:`Verdict`.

    The ordering is the whole point: the tools-ran check comes first, because a
    skill that produced a confident answer without calling anything is the #1428
    failure this harness exists to catch, and a ``side_effect_check`` written
    loosely enough could otherwise pass it.
    """
    satisfied = (
        ledger.called_all(scenario.expect_tools)
        if scenario.require_all_tools
        else ledger.called_any(scenario.expect_tools)
    )
    if scenario.expect_tools and not satisfied:
        if _success_markers(reply):
            return Verdict.false_success
        return Verdict.honest_failure
    return scenario.side_effect_check(context, reply, ledger)


def verdict_from_side_effect(present: bool, reply: str) -> Verdict:
    """The standard tail of a ``side_effect_check``.

    Side effect present is a pass. Absent plus success-claiming prose is the hard
    fail; absent with no claim is an honest failure.
    """
    if present:
        return Verdict.true_success
    return Verdict.false_success if _success_markers(reply) else Verdict.honest_failure


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SkillResult:
    """The outcome of running one scenario, and the record the gate consumes."""

    skill: str
    status: SkillStatus
    reason: str = ""
    counts: Dict[str, int] = field(default_factory=dict)
    transcripts: List[Dict[str, Any]] = field(default_factory=list)
    hard_fail: bool = False

    def to_record(self, *, content_digest: str, version: str) -> Dict[str, Any]:
        """The manifest entry, bound to the bytes and version it was earned on."""
        stamped = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return {
            "skill": self.skill,
            "status": self.status.value,
            "reason": self.reason,
            "version": version,
            "content_digest": content_digest,
            "harness": HARNESS_VERSION,
            # Only a pass gets a validated_at — an empty one on a failed or
            # blocked record keeps "when did this last work" honest.
            "validated_at": stamped if self.status is SkillStatus.validated else "",
            "recorded_at": stamped,
            "counts": dict(self.counts),
            "hard_fail": self.hard_fail,
        }


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class SkillBehaviorHarness:
    """Runs a :class:`SkillScenario` against a real agent and a real model.

    One agent instance per repeat, built with skill autoload OFF and exactly the
    scenario's skill loaded, so the evidence is attributable to that skill.
    """

    def __init__(
        self,
        *,
        skill_roots: Sequence[Path],
        workspace_root: Path,
        agent_builders: Optional[Dict[str, Callable[..., Any]]] = None,
    ):
        """Args:
        skill_roots: Directories searched for the skill under test.
        workspace_root: Parent for the per-repeat temp workspaces.
        agent_builders: ``{scenario.agent: builder}`` override, for tests.
        """
        self._skill_roots = [Path(p) for p in skill_roots]
        self._workspace_root = Path(workspace_root)
        self._builders = dict(agent_builders or default_agent_builders())

    def run(self, scenario: SkillScenario) -> SkillResult:
        """Run every repeat of *scenario* and aggregate."""
        if scenario.blocked_reason:
            return SkillResult(
                skill=scenario.skill,
                status=SkillStatus.blocked,
                reason=scenario.blocked_reason,
            )

        verdicts: List[Verdict] = []
        transcripts: List[Dict[str, Any]] = []

        for run in range(scenario.repeats):
            fixtures = FixtureServer().start()
            workspace = self._workspace_root / f"{scenario.skill}-{run}"
            home = workspace / "home"
            (workspace / "work").mkdir(parents=True, exist_ok=True)
            home.mkdir(parents=True, exist_ok=True)

            context = ScenarioContext(
                skill=scenario.skill,
                run=run,
                workspace=workspace / "work",
                home=home,
                token=secrets.token_hex(4),
                fixtures=fixtures,
            )

            prompt = ""
            reply = ""
            ledger = ToolLedger()
            try:
                if scenario.setup is not None:
                    scenario.setup(context)
                prompt = scenario.prompt_factory(context)
                reply, ledger = self._drive(scenario, context, prompt)
                verdict = classify(scenario, context, reply, ledger)
            except ScenarioUnavailable as exc:
                fixtures.stop()
                return SkillResult(
                    skill=scenario.skill,
                    status=SkillStatus.blocked,
                    reason=str(exc),
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
                logger.error(
                    "%s run %d raised: %s", scenario.skill, run, exc, exc_info=True
                )
                verdict = Verdict.error
            finally:
                fixtures.stop()

            verdicts.append(verdict)
            transcripts.append(
                {
                    "run": run,
                    "token": context.token,
                    "prompt": prompt,
                    "reply": reply[:2000],
                    "tools": ledger.summary(),
                    "verdict": verdict.value,
                }
            )
            logger.info(
                "%s run %d/%d: %s (tools: %s)",
                scenario.skill,
                run + 1,
                scenario.repeats,
                verdict.value,
                ", ".join(ledger.summary()) or "none",
            )

        aggregate = _aggregate_verdicts(verdicts)
        counts = {v.value: n for v, n in aggregate["counts"].items()}
        status = SkillStatus.validated if aggregate["passed"] else SkillStatus.failed
        reason = "" if aggregate["passed"] else _failure_reason(aggregate, counts)
        return SkillResult(
            skill=scenario.skill,
            status=status,
            reason=reason,
            counts=counts,
            transcripts=transcripts,
            hard_fail=bool(aggregate["hard_fail"]),
        )

    def _drive(
        self, scenario: SkillScenario, context: ScenarioContext, prompt: str
    ) -> Tuple[str, ToolLedger]:
        """Build the agent, load only this skill, run one turn."""
        builder = self._builders.get(scenario.agent)
        if builder is None:
            raise ScenarioUnavailable(
                f"No agent builder registered for '{scenario.agent}'. Scenario "
                f"'{scenario.skill}' cannot run. Register one in "
                "gaia.eval.skill_behavior.default_agent_builders."
            )
        agent = builder(
            context=context,
            skill_roots=self._skill_roots,
            max_steps=scenario.max_steps,
        )
        try:
            load_only(agent, scenario.skill, self._skill_roots)
            if scenario.after_agent is not None:
                scenario.after_agent(context, agent)
            result = agent.process_query(prompt, max_steps=scenario.max_steps)
            reply = str((result or {}).get("result") or "")
            return reply, ToolLedger.from_result(result)
        finally:
            shutdown = getattr(agent, "shutdown", None)
            if callable(shutdown):
                shutdown()


def _failure_reason(aggregate: Dict[str, Any], counts: Dict[str, int]) -> str:
    if aggregate["hard_fail"]:
        return (
            f"FALSE SUCCESS in {counts.get('false_success', 0)} of "
            f"{sum(counts.values())} run(s): the agent reported success but the "
            "skill's tools left no side effect. This is the #1428 regression "
            "class and is never a soft failure."
        )
    return (
        f"Did not clear behaviour validation: {counts}. A skill passes only when "
        "every repeat both ran its declared tools and left the expected side "
        "effect."
    )


def load_only(agent: Any, skill: str, roots: Sequence[Path]) -> None:
    """Load exactly ``skill`` into ``agent``, unloading anything auto-loaded.

    Leaving another skill loaded would make the evidence unattributable — a tool
    call could have come from either body.
    """
    from gaia.skills.manager import SkillManager

    for name in list(getattr(agent, "loaded_skills", {}) or {}):
        if name != skill:
            agent.unload_skill(name)

    manager = SkillManager(
        agent_skill_dirs=[str(p) for p in roots],
        include_claude_roots=False,
    )
    agent.load_skill(skill, manager=manager)
    loaded = set(getattr(agent, "loaded_skills", {}) or {})
    if loaded != {skill}:
        raise ScenarioUnavailable(
            f"Expected only '{skill}' loaded, found {sorted(loaded)}. The "
            "evidence would not be attributable to the skill under test."
        )


def default_agent_builders() -> Dict[str, Callable[..., Any]]:
    """The agent factories scenarios name via ``SkillScenario.agent``."""
    return {"gaia": build_gaia_agent, "email": build_email_agent}


def sandbox_home(context: ScenarioContext) -> None:
    """Redirect this process's home at the run's sandbox, and verify it moved.

    ``GAIA_CONFIG_DIR`` alone is not enough: ``MemoryStore`` derives its path from
    ``Path.home()`` and ignores that variable, so a run would write memory rows
    into the developer's real ``~/.gaia`` and then hunt for the side effect in a
    sandbox they never reached — reporting a working skill as broken. Every agent
    builder calls this before constructing anything.

    Raises:
        ScenarioUnavailable: the redirection did not take, so the run would
            escape its sandbox.
    """
    os.environ["GAIA_CONFIG_DIR"] = str(context.home)
    os.environ["HOME"] = str(context.home)
    os.environ["USERPROFILE"] = str(context.home)

    resolved = Path.home().resolve()
    if resolved != context.home.resolve():
        raise ScenarioUnavailable(
            f"Sandbox escape: Path.home() resolves to {resolved}, not the "
            f"per-run home {context.home}. The run would write memory and index "
            "state into the real user profile and its side-effect checks would "
            "look in the wrong place. Fix the redirection in "
            "gaia.eval.skill_behavior.sandbox_home before running."
        )


def build_gaia_agent(*, context: ScenarioContext, skill_roots, max_steps: int):
    """The flagship agent, sandboxed to this repeat's workspace.

    Autoload is disabled on a throwaway subclass rather than globally: the class
    attribute is what ``Agent.__init__`` consults, and mutating the real class
    would leak into anything else built in this process.
    """
    try:
        from gaia_agent.agent import GaiaAgent, GaiaAgentConfig
    except ImportError as exc:
        raise ScenarioUnavailable(
            "gaia_agent is not installed, so the flagship agent cannot host the "
            "skill. Install it with 'pip install -e hub/agents/gaia/python' and "
            "re-run."
        ) from exc

    class _HarnessAgent(GaiaAgent):
        AUTOLOAD_DECLARED_SKILLS = False

    sandbox_home(context)
    # Per-turn semantic skill selection would hide the loaded body from the
    # model when the prompt scores below threshold — the harness needs it shown.
    os.environ["GAIA_DYNAMIC_SKILLS"] = "0"

    return _HarnessAgent(
        GaiaAgentConfig(
            silent_mode=True,
            max_steps=max_steps,
            allowed_paths=[str(context.workspace)],
            enable_filesystem=True,
            enable_scratchpad=True,
            enable_browser=True,
            scratchpad_db_path=str(context.home / "scratchpad.db"),
            filesystem_index_path=str(context.home / "file_index.db"),
            output_dir=str(context.workspace),
        )
    )


def build_email_agent(*, context: ScenarioContext, skill_roots, max_steps: int):
    """The email agent on its fake-mailbox eval seam.

    Implemented in :mod:`gaia.eval.skill_scenarios_email`, which owns the
    ``FakeGmailBackend`` wiring; imported lazily so the flagship scenarios do not
    depend on the email package being installed.
    """
    from gaia.eval.skill_scenarios_email import build_email_agent as _build

    return _build(context=context, skill_roots=skill_roots, max_steps=max_steps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m gaia.eval.skill_behavior`` — run scenarios, write records."""
    parser = argparse.ArgumentParser(
        description=(
            "Run skill behaviour validation and write the per-skill records the "
            "publish gate reads. Requires a live Lemonade backend."
        )
    )
    parser.add_argument(
        "--skill",
        action="append",
        default=None,
        help="Validate only this skill (repeatable). Default: every scenario.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Write the results manifest here (JSON).",
    )
    parser.add_argument(
        "--transcripts",
        default=None,
        help="Also write full per-run transcripts here, for post-mortem.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Parent directory for per-run sandboxes (default: a temp dir).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Override every scenario's repeat count.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from gaia.eval.skill_scenarios import SKILL_SCENARIOS
    from gaia.skills.behavior_gate import write_manifest

    scenarios = list(SKILL_SCENARIOS)
    if args.skill:
        wanted = set(args.skill)
        unknown = wanted - {s.skill for s in scenarios}
        if unknown:
            parser.error(
                f"No scenario for {sorted(unknown)}. Every shipped skill needs "
                "one — add it to gaia.eval.skill_scenarios.SKILL_SCENARIOS."
            )
        scenarios = [s for s in scenarios if s.skill in wanted]
    if args.repeats is not None:
        for scenario in scenarios:
            scenario.repeats = args.repeats

    if args.workspace:
        workspace_root = Path(args.workspace)
        workspace_root.mkdir(parents=True, exist_ok=True)
        results = run_all(scenarios, workspace_root)
    else:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="gaia-skill-behavior-") as tmp:
            results = run_all(scenarios, Path(tmp))

    write_manifest(Path(args.output), results)
    if args.transcripts:
        Path(args.transcripts).write_text(
            json.dumps(
                {r.skill: r.transcripts for r in results}, indent=2, default=str
            ),
            encoding="utf-8",
        )

    for result in results:
        print(f"{result.status.value.upper():<12} {result.skill} {result.reason}")
    failed = [r for r in results if r.status is not SkillStatus.validated]
    if failed:
        print(
            f"\n{len(failed)} of {len(results)} skill(s) are not validated. "
            "An unvalidated skill cannot be published — see "
            "gaia.skills.behavior_gate."
        )
        return 1
    print(f"\nAll {len(results)} skill(s) validated.")
    return 0


def run_all(
    scenarios: Sequence[SkillScenario], workspace_root: Path
) -> List[SkillResult]:
    """Run every scenario in order and return their results."""
    from gaia.eval.skill_scenarios import skill_roots

    harness = SkillBehaviorHarness(
        skill_roots=skill_roots(), workspace_root=workspace_root
    )
    return [harness.run(scenario) for scenario in scenarios]


if __name__ == "__main__":
    sys.exit(main())
