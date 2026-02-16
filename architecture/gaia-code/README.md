# GAIA Code: Recursive Agent Composition Architecture

**Status**: Strategic specification for the world's most autonomous coding agent
**Priority**: CRITICAL - This is the #1 priority for GAIA V2
**Paradigm**: Recursive Agent Composition (RAC)

---

## The Four Documents

| Document | Lines | Purpose |
|----------|-------|---------|
| **[GAIA_CODE_AUTONOMOUS_AGENT.md](./GAIA_CODE_AUTONOMOUS_AGENT.md)** | 5,045 | Complete specification - all capabilities, architecture, design |
| **[RECURSIVE_AGENT_COMPOSITION.md](./RECURSIVE_AGENT_COMPOSITION.md)** | 970 | The RAC paradigm explained - the novel contribution |
| **[GAIA_CODE_MILESTONES.md](./GAIA_CODE_MILESTONES.md)** | 637 | Implementation roadmap - M0 through M10, what to build when |
| **[GAIA_CODE_EVALUATION_PLAN.md](./GAIA_CODE_EVALUATION_PLAN.md)** | 928 | How to prove it works - benchmarks + real-world validation |

**Total:** 7,580 lines of technical specification

---

## Read This First

**Start here:** [RECURSIVE_AGENT_COMPOSITION.md](./RECURSIVE_AGENT_COMPOSITION.md)

This explains the paradigm shift - why RAC is fundamentally different from every other agent architecture.

**Then read:** [GAIA_CODE_MILESTONES.md](./GAIA_CODE_MILESTONES.md)

This tells you what to build, in what order, and why.

**Reference:** [GAIA_CODE_AUTONOMOUS_AGENT.md](./GAIA_CODE_AUTONOMOUS_AGENT.md)

The complete spec - all 10 core capabilities, full architecture, design philosophy.

**Validation:** [GAIA_CODE_EVALUATION_PLAN.md](./GAIA_CODE_EVALUATION_PLAN.md)

How to prove it works against Terminal-Bench, SWE-bench, and real-world scenarios.

---

## The Paradigm: Recursive Agent Composition (RAC)

### What It Is

**Agents recursively spawn specialized sub-agents to handle complex tasks, with all agents sharing global state.**

```
Main Agent: "Build full-stack app"
  ├─ agent_query(specialist="backend_agent")
  │    └─ agent_query(specialist="security_agent")
  │         └─ agent_query(specialist="refactoring_agent")
  │
  └─ agent_query(specialist="frontend_agent")

ALL agents share: memory.db, knowledge.db, manifest, plan
```

### Why It's Different

| Every Other Agent | GAIA Code (RAC) |
|------------------|-----------------|
| Fixed capabilities | Expanding (creates tools, skills, agents) |
| Monolithic | Recursive hierarchy |
| Flat state | Shared state across recursion |
| Cloud or local | Adaptive (local + cloud escalation) |
| Capability constant | Capability exponential (12x after 6 months) |

### The Foundation

**Extends:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) (RLMs)
- RLMs: `llm_query()` for recursive text processing
- RAC: `agent_query()` for recursive agent spawning

**Key insight:** Sub-agents aren't just LLM calls. They're full agents with tools, memory, verification, and the ability to spawn more agents.

---

## The 10 Core Capabilities

**C1-C9:** See main spec for details

**C10: Recursive Agent Composition**
- `agent_query()` tool spawns sub-agents
- 7 core specialists (Debugger, Security, Refactoring, Testing, Docs, Performance, Architecture)
- AgentFactory creates new specialists from patterns (after 3+ similar tasks)
- Three-level hierarchy: tools (78+) → skills (50+) → agents (7 core + learned)
- SharedAgentState: all agents share memory.db, knowledge.db, manifest, plan, call stack
- Single-threaded + async/await for simplicity
- Exponential capability growth (specialists create sub-specialists)

---

## The Seven Shared Databases

Every agent in the recursion tree shares these:

| Database | Scope | Purpose |
|----------|-------|---------|
| **memory.db** | Session | Working cache, active state |
| **knowledge.db** | Permanent | Insights, preferences, learnings |
| **tools.db** | Permanent | Tool registry, usage stats |
| **skills.db** | Permanent | Learned workflows |
| **agents.db** | Permanent | Specialist registry |
| **plans.db** | Session | Master plan, task tree |
| **manifests.db** | Session | Project state, decisions |

**Coordination mechanism:** All agents read/write to the same databases. Changes by one agent are visible to all others. Thread-safe via locks during async I/O.

---

## Implementation Timeline

```
PHASE 1: CORE AGENT FRAMEWORK (Days 1-19, 188h)
  See: ../features/CORE_AGENT_FRAMEWORK_MILESTONES.md
  Applies to: ALL agents (chat, code, CUA, workflow)

M0-M6: Build universal foundation
  ├─ M0: Prompting + RLM patterns (21h)
  ├─ M1: 7 databases + SharedAgentState + agent_query() (48h) ← RAC
  ├─ M2: Quality framework + plans (20h)
  ├─ M3: Checkpoint + audit (21h)
  ├─ M4: Registries + FAISS (18h)
  ├─ M5: Self-extension (tools, skills, agents) (41h)
  └─ M6: Vector search + defrag (19h)

PHASE 2: CODE AGENT CUSTOMIZATION (Days 19-32, 49h)
  See: GAIA_CODE_MILESTONES.md (code-specific milestones)
  Applies to: gaia code only

M7-M10: Customize for coding
  ├─ M7: Codebase indexing + code specialists (21h)
  ├─ M8: Code gates + config (12h)
  ├─ M9: Async I/O optimization (8h)
  └─ M10: Benchmarks (Terminal-Bench, SWE-bench) (8h)
```

**Key milestones:**
- **Day 5:** RAC foundation complete (agent_query works, shared state works)
- **Day 13:** 7 specialists deployed (recursive delegation works)
- **Day 16:** Agent auto-generation works (exponential growth begins)
- **Day 32:** Production-ready, benchmarked, battle-hardened

---

## Why This Matters

### For AMD
**"AMD has the only agent that builds agents. All on local hardware."**

### For Developers
**"Your coding agent gets 12x faster over 6 months. It learns YOUR codebase, creates YOUR specialists, optimizes for YOUR patterns."**

### For Research
**"We extended RLMs (recursive LLM calls) to RAC (recursive agent spawning). This enables exponential capability growth through self-specialization."**

---

## Quick Start (Implementation)

**Day 1-2:** Read [MILESTONES.md](./GAIA_CODE_MILESTONES.md), implement M0 (prompting)

**Day 3-5:** Implement M1 (SharedAgentState + agent_query())
- This is the hardest milestone (38 hours)
- This is the most important (everything builds on this)
- Deliverable: `agent_query()` works, recursion works, shared state works

**Day 5-10:** Implement M2-M3 (quality gates, checkpoint, audit)

**Day 10+:** Implement M4-M5 (specialists, auto-generation)

---

## Key Design Principles

1. **Composable** - Every feature opt-in, works across LLM tiers
2. **Recursive** - agent_query() enables unlimited decomposition
3. **Shared State** - Manifest, plan, knowledge DB across all agents
4. **Self-Extending** - Creates tools, skills, specialists
5. **Simple** - Single-threaded + async/await, not multi-threaded
6. **Quality-Focused** - Every sub-agent verifies via quality gates
7. **Time-Aware** - Everything timestamped, full audit trail
8. **Context-Lean** - RLMs + knowledge DB, never fills context
9. **Learning** - Insights, confidence tracking, memory defrag
10. **Adaptive** - Local → cloud escalation with cost ceiling

---

## The Research Contribution

**Novel contributions:**
1. Extending RLMs to full agents (agent_query vs llm_query)
2. SharedAgentState architecture (7 databases, thread-safe)
3. Three-level hierarchy with auto-generation (tools → skills → agents)
4. Recursive self-improvement (specialists creating specialists)
5. Smart escalation with learning (per-subtask routing, cost ceiling)

**Potential papers:**
- "Recursive Agent Composition: Extending RLMs to Self-Improving Agent Hierarchies"
- "Shared State Coordination in Recursive Multi-Agent Systems"
- "Agent Auto-Generation: Learning to Create Domain Specialists"

---

## FAQ

**Q: Is this actually Recursive Language Models?**
A: It's an extension. RLMs use `llm_query()` for recursive text processing. RAC uses `agent_query()` for recursive agent spawning with full agency (tools, verification, memory).

**Q: Why seven databases?**
A: Each has different scope and purpose. memory.db (session cache), knowledge.db (permanent learning), tools/skills/agents (capability registries), plans/manifests (session coordination). Separation of concerns.

**Q: Isn't this just multi-agent systems?**
A: No. Multi-agent systems have independent agents that struggle with coordination. RAC has a single main agent that recursively decomposes via agent_query(), with ALL sub-agents sharing state. One brain, recursive execution.

**Q: Why single-threaded?**
A: Simplicity. LLM calls are I/O-bound (async/await handles concurrency). True parallelism adds complexity (race conditions, non-determinism, debugging nightmares) for minimal benefit.

**Q: How do agents coordinate?**
A: Via SharedAgentState. Backend agent writes API schema to manifest. Frontend agent reads it. Security agent writes insights to knowledge.db. Debugger agent recalls them. No explicit coordination protocol needed - shared state IS the coordination.

**Q: What if two agents conflict?**
A: Manifest decisions are write-once. First agent to make a decision (e.g., "auth_method": "JWT") writes it. Other agents read and follow. If genuine conflict, escalate to user via message queue.

**Q: Why is this better than Claude Code?**
A: Ten differentiators (see main spec). Key ones: never compacts context, learns permanently, creates specialists, runs locally, verifies output, improves exponentially.

---

*Four documents. One paradigm. Recursive Agent Composition.*
*Start with RECURSIVE_AGENT_COMPOSITION.md. Build from MILESTONES.md. Reference AUTONOMOUS_AGENT.md. Validate with EVALUATION_PLAN.md.*
