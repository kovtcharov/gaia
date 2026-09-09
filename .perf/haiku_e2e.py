"""End-to-end test CLI for dynamic tool loading on the flagship, run on Claude.

Backend: Claude Haiku 4.5 (Lemonade is out of service). This validates LOGIC —
does a trimmed ``tools=`` reach the model, does the model work from it, and does
the ``load_tools`` escape hatch recover a capability selection left out. It says
NOTHING about Gemma-4-E4B latency or eval scores.

The embedder normally comes from Lemonade. With Lemonade down these scenarios
inject a deterministic stub embedder instead, which is not a workaround but the
point: a stub lets the test *force* a semantic miss and prove the escape hatch
recovers from it. Real selection quality is the eval's job, on Gemma.

Usage:
    python .perf/haiku_e2e.py            # all scenarios
    python .perf/haiku_e2e.py failsafe   # one scenario by name
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys

import numpy as np
from gaia_agent.agent import GaiaAgent, GaiaAgentConfig

from gaia.agents.base.tools import _TOOL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
# The TOOL_LOADER json lines are the whole audit trail for this feature.
logging.getLogger("gaia.agents.base.tool_loader").setLevel(logging.INFO)

MODEL = "claude-haiku-4-5"
PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(scenario: str, name: str, ok: bool, detail: str = "") -> bool:
    results.append((scenario, name, PASS if ok else FAIL))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return ok


@contextlib.contextmanager
def isolated_registry():
    """Run one scenario against a clean global ``_TOOL_REGISTRY``.

    ``@tool`` writes into a process-global dict and ``_snapshot_tools`` copies
    the whole thing, so an agent built second inherits every tool the first
    registered. Without this, scenarios contaminate each other — see
    scenario_cross_agent_leak, which asserts that behaviour deliberately.
    """
    saved = dict(_TOOL_REGISTRY)
    _TOOL_REGISTRY.clear()
    try:
        yield
    finally:
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(saved)


def build(dynamic_tools: bool = True) -> GaiaAgent:
    """A live flagship on Haiku. use_claude=True skips LemonadeManager entirely."""
    return GaiaAgent(
        config=GaiaAgentConfig(
            use_claude=True,
            claude_model=MODEL,
            streaming=False,
            silent_mode=True,
            dynamic_tools=dynamic_tools,
        )
    )


def stub_embedder(agent: GaiaAgent, hot: set[str]) -> None:
    """Force selection to admit exactly *hot* (plus CORE) — no Lemonade.

    Tool docs whose name is in *hot* embed to [1, 0]; everything else to [0, 1].
    The query embeds to [1, 0], so hot tools score 1.0 and the rest 0.0 against
    a 0.20 threshold. Deterministic, and it lets a scenario engineer a miss.
    """
    loader = agent.tool_loader
    assert loader is not None, "loader not built — dynamic_tools off?"

    def embed(text: str) -> np.ndarray:
        name = text.split(":", 1)[0].strip()
        return np.array([1.0, 0.0] if name in hot else [0.0, 1.0], dtype=np.float32)

    loader._embed_fn = lambda text: np.array([1.0, 0.0], dtype=np.float32)
    loader._embed_batch_fn = lambda texts: np.stack([embed(t) for t in texts])
    loader._embed_cache.clear()
    # _dynamic_tools_active() also gates on a memory store; with the embedder
    # down MemoryMixin tore it out, and this scenario is not about memory.
    if getattr(agent, "_memory_store", None) is None:
        agent._memory_store = object()


def tools_sent(agent: GaiaAgent) -> list[str]:
    """The tool names actually going out in this turn's ``tools=`` payload."""
    schemas = agent._build_openai_tool_schemas(filter_to=agent._active_tool_filter)
    return sorted(s["function"]["name"] for s in schemas or [])


def called(result: dict) -> list[str]:
    """Tool names the model actually invoked, in order.

    process_query reports steps under "conversation"; a tool step is the entry
    carrying "tool_args", and the tool name is its "name". There is no
    top-level "tool_calls" key in the returned dict.
    """
    return [
        step["name"]
        for step in result.get("conversation") or []
        if isinstance(step, dict) and "tool_args" in step and step.get("name")
    ]


# ---------------------------------------------------------------------------


def scenario_failsafe() -> None:
    """Embedder unreachable => full registry, no crash. The Lemonade-down path."""
    print("\n=== failsafe: no embedder, agent must fall back to all tools ===")
    agent = build()
    check("failsafe", "loader was built", agent.tool_loader is not None)
    # No _memory_store (embedder down) => _dynamic_tools_active() is False.
    check(
        "failsafe",
        "selection inactive without an embedder",
        not agent._dynamic_tools_active(),
        f"_memory_store={getattr(agent, '_memory_store', None)!r}",
    )
    result = agent.process_query("Say the single word: ready.")
    sent = tools_sent(agent)
    check(
        "failsafe",
        "full registry sent (nothing trimmed)",
        len(sent) == len(agent._tools_registry),
        f"{len(sent)}/{len(agent._tools_registry)} tools",
    )
    answer = (result.get("result") or result.get("final_answer") or "").lower()
    check("failsafe", "Haiku answered", "ready" in answer, repr(answer[:80]))


def scenario_trimmed() -> None:
    """CORE-only selection: the model gets 10 tools and must work from them."""
    print("\n=== trimmed: CORE-only tools= reaches the model ===")
    agent = build()
    stub_embedder(agent, hot=set())  # nothing matches => CORE only
    check("trimmed", "selection active", agent._dynamic_tools_active())

    from gaia_agent_chat.tool_bundles import FULL_CORE_TOOLS

    # CORE minus whatever this environment could not register. Without an
    # embedder MemoryMixin skips its five tools, so the live CORE here is 5,
    # not 10 — the trim is still exact, just against a smaller registry.
    expected = sorted(n for n in FULL_CORE_TOOLS if n in agent._tools_registry)
    result = agent.process_query("What is 17 times 4? Answer with the number only.")
    sent = tools_sent(agent)
    check(
        "trimmed",
        "tools= trimmed to exactly CORE",
        sent == expected,
        f"{len(sent)} sent vs {len(expected)} expected",
    )
    check(
        "trimmed",
        "trim is a real reduction",
        len(sent) < len(agent._tools_registry),
        f"{len(sent)} of {len(agent._tools_registry)}",
    )
    check("trimmed", "load_tools present in tools=", "load_tools" in sent)
    check(
        "trimmed",
        "Haiku answered from the trimmed set",
        "68" in (result.get("result") or ""),
        repr((result.get("result") or "")[:80]),
    )


def scenario_escape_hatch() -> None:
    """The safety net: a tool the selection MISSED must still be reachable.

    browse_directory is deliberately not selected, so the only route to it is
    the model reading the bundle menu and calling load_tools("file_browse").
    """
    print("\n=== escape_hatch: model must recover a tool selection left out ===")
    agent = build()
    stub_embedder(agent, hot=set())  # force the miss

    prompt = agent.system_prompt
    check(
        "escape_hatch",
        "bundle menu renders in the system prompt",
        "==== LOADABLE TOOL BUNDLES ====" in prompt,
    )
    check(
        "escape_hatch",
        "menu lists file_browse",
        "- file_browse:" in prompt,
    )

    result = agent.process_query(
        "List the files in my home directory. If you do not have a tool for "
        "that, load the bundle that provides one first."
    )
    invoked = called(result)
    check(
        "escape_hatch",
        "Haiku called load_tools",
        "load_tools" in invoked,
        f"called={invoked}",
    )
    after = tools_sent(agent)
    check(
        "escape_hatch",
        "the missing tool became visible after load_tools",
        "browse_directory" in after,
        f"{len(after)} tools now sent",
    )
    check(
        "escape_hatch",
        "loaded set stayed under the cap",
        len(after) <= agent.tool_loader._max_tools,
        f"{len(after)} <= {agent.tool_loader._max_tools}",
    )


def scenario_unknown_bundle() -> None:
    """load_tools with a bad name fails loudly with the valid names listed."""
    print("\n=== unknown_bundle: actionable error, not a silent no-op ===")
    agent = build()
    stub_embedder(agent, hot=set())
    out = agent._tools_registry["load_tools"]["function"]("not_a_bundle")
    check("unknown_bundle", "status is error", out.get("status") == "error", str(out)[:90])
    check(
        "unknown_bundle",
        "error lists the valid bundle names",
        "file_browse" in out.get("error", ""),
    )
    # A bare tool name resolves to its owning bundle (the documented nicety).
    out2 = agent._tools_registry["load_tools"]["function"]("browse_directory")
    check(
        "unknown_bundle",
        "bare tool name resolves to its bundle",
        out2.get("status") == "success" and "browse_directory" in out2["loaded_tools"],
        str(out2.get("status")),
    )


def scenario_off_switch() -> None:
    """dynamic_tools=False must be byte-identical to pre-change behaviour."""
    print("\n=== off_switch: opting out restores the old prompt exactly ===")
    # Each build gets a clean registry — otherwise the loader-on agent's
    # load_tools leaks into the loader-off one and the comparison is meaningless.
    with isolated_registry():
        off = build(dynamic_tools=False)
        off_names = set(off._tools_registry)
        off_prompt = off.system_prompt
        off_loader = off.tool_loader
    with isolated_registry():
        on_names = set(build(dynamic_tools=True)._tools_registry)

    check("off_switch", "no loader built", off_loader is None)
    check("off_switch", "load_tools not registered", "load_tools" not in off_names)
    check(
        "off_switch",
        "no bundle menu in the prompt",
        "LOADABLE TOOL BUNDLES" not in off_prompt,
    )
    # Absolute counts move with the environment (no embedder => no memory
    # tools), so pin the relationship instead: turning the loader on adds
    # load_tools and nothing else.
    check(
        "off_switch",
        "enabling adds exactly load_tools",
        on_names - off_names == {"load_tools"},
        f"delta={sorted(on_names - off_names)}",
    )
    check("off_switch", "enabling removes nothing", not (off_names - on_names))


def scenario_cap_and_stability() -> None:
    """The loaded set grows monotonically, honours the cap, and stays stable.

    Stability matters as much as size: the loader sorts its output so a turn
    that admits nothing new serialises byte-identically, which is what keeps the
    backend's KV prefix cache warm across a conversation.
    """
    print("\n=== cap_and_stability: monotonic growth, cap held, stable bytes ===")
    agent = build()
    loader = agent.tool_loader
    cap = loader._max_tools

    # A realistic steady match: one bundle's worth of tools stays hot across
    # turns, the way a user working on files keeps matching file tools.
    stub_embedder(agent, hot={"browse_directory", "get_file_info", "list_recent_files"})

    sizes, payloads = [], []
    for turn in range(3):
        agent.process_query(f"Turn {turn}: reply with the word ok.")
        sent = tools_sent(agent)
        sizes.append(len(sent))
        payloads.append(json.dumps(sent))

    check(
        "cap_and_stability",
        "cap never exceeded",
        all(s <= cap for s in sizes),
        f"sizes={sizes} cap={cap}",
    )
    check(
        "cap_and_stability",
        "loaded set never shrinks",
        all(b >= a for a, b in zip(sizes, sizes[1:])),
        f"sizes={sizes}",
    )
    check(
        "cap_and_stability",
        "repeat turns serialise identically (KV prefix stays warm)",
        payloads[-1] == payloads[-2],
        f"{payloads[-2]} vs {payloads[-1]}",
    )
    check(
        "cap_and_stability",
        "CORE present throughout",
        all(n in tools_sent(agent) for n in loader._core if n in agent._tools_registry),
    )

    # Now the pathological input: every tool matches every turn, so there is
    # always a fresh candidate and the cap must still hold.
    hot_agent = build()
    stub_embedder(hot_agent, hot=set(hot_agent._tools_registry))
    hot_sizes = []
    for turn in range(3):
        hot_agent.process_query(f"Turn {turn}: reply with the word ok.")
        hot_sizes.append(len(tools_sent(hot_agent)))
    check(
        "cap_and_stability",
        "cap holds when everything matches",
        all(s <= cap for s in hot_sizes),
        f"sizes={hot_sizes} cap={cap}",
    )


def scenario_cross_agent_leak() -> None:
    """PRE-EXISTING framework issue, pinned here so it is tracked not hidden.

    ``@tool`` writes into a process-global registry and ``_snapshot_tools``
    copies all of it, so an agent built second inherits the first agent's tools
    — including a ``load_tools`` closure bound to the FIRST agent's loader.
    Not introduced by dynamic tool loading, but defaulting the flagship on makes
    it reachable: build a GaiaAgent then a plain ChatAgent in one process and the
    ChatAgent carries a load_tools that mutates the GaiaAgent's tool filter.

    The flagship sidecar builds exactly one agent per process, so the shipping
    path is unaffected. Fixing it means per-agent registration in
    ``src/gaia/agents/base/agent.py``, which this change does not touch.
    """
    print("\n=== cross_agent_leak: documents a known framework issue ===")
    with isolated_registry():
        first = build(dynamic_tools=True)
        check(
            "cross_agent_leak",
            "first agent registers load_tools",
            "load_tools" in first._tools_registry,
        )
        second = build(dynamic_tools=False)
        leaked = "load_tools" in second._tools_registry
        check(
            "cross_agent_leak",
            "KNOWN: loader-off agent inherits load_tools from the first",
            leaked,
            "pre-existing _snapshot_tools behaviour; sidecar is single-agent",
        )
        if leaked:
            check(
                "cross_agent_leak",
                "the inherited tool has no loader of its own",
                second.tool_loader is None,
            )


SCENARIOS = {
    "failsafe": scenario_failsafe,
    "cap_and_stability": scenario_cap_and_stability,
    "trimmed": scenario_trimmed,
    "escape_hatch": scenario_escape_hatch,
    "unknown_bundle": scenario_unknown_bundle,
    "off_switch": scenario_off_switch,
    "cross_agent_leak": scenario_cross_agent_leak,
}

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SCENARIOS)
    for name in wanted:
        try:
            # Scenarios that manage their own registry re-enter this harmlessly.
            with isolated_registry():
                SCENARIOS[name]()
        except Exception as exc:  # a crash IS the result — record, keep going
            import traceback

            traceback.print_exc()
            results.append((name, f"raised {type(exc).__name__}", FAIL))

    print("\n" + "=" * 72)
    failed = [r for r in results if r[2] == FAIL]
    for scenario, name, verdict in results:
        print(f"  {verdict:<5} {scenario:<15} {name}")
    print("=" * 72)
    print(f"backend: Claude {MODEL}   {len(results) - len(failed)}/{len(results)} passed")
    print(json.dumps({"passed": len(results) - len(failed), "failed": len(failed)}))
    sys.exit(1 if failed else 0)
