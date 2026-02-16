# Core Agent Framework vs Code Agent Specific

**Date**: February 11, 2026
**Purpose**: Separate universal agent capabilities from code-agent-specific features

---

## The Critical Distinction

**CORE AGENT FRAMEWORK** = Works for ALL agents (ChatAgent, CodeAgent, CUAAgent, WorkflowAgent, etc.)
**CODE AGENT SPECIFIC** = Only needed for `gaia code` (coding tasks)

**Why this matters:**
- Core framework ships once, benefits all agents
- Code-specific features don't bloat chat agents
- Clear boundaries for implementation priorities

---

## CORE AGENT FRAMEWORK (Build First)

These capabilities apply to ANY agent, not just coding:

### Infrastructure (M1: Days 3-5)

| Feature | Why Universal | Used By |
|---------|--------------|---------|
| **memory.db** | All agents cache files, state | Chat (docs), Code (source), CUA (screenshots) |
| **knowledge.db** | All agents learn insights, preferences | Chat (topics), Code (patterns), CUA (UI workflows) |
| **tools.db** | All agents have tools | Chat (RAG tools), Code (git tools), CUA (mouse/keyboard) |
| **skills.db** | All agents learn workflows | Chat (summarization), Code (debugging), CUA (form filling) |
| **agents.db** | All agents can spawn specialists | Chat (ResearchAgent), Code (DebuggerAgent), CUA (BrowserAgent) |
| **ProjectManifest** | All agents track project state | Chat (docs read), Code (files created), CUA (apps used) |
| **MasterPlan** | All agents decompose tasks | Chat (multi-doc synthesis), Code (build app), CUA (multi-step automation) |
| **MessageQueue** | All agents interact async with user | Universal: any agent can ask questions, get answers |
| **AgentCallStack** | All agents can recurse | Universal: recursion isn't coding-specific |
| **SharedAgentState** | All agents share state when recursing | Universal: coordination mechanism |

**Verdict: ALL of M1 is CORE FRAMEWORK** ✅

### Recursive Decomposition (M0 + M1)

| Feature | Why Universal |
|---------|--------------|
| **agent_query()** | Chat agents decompose "analyze 50 PDFs", Code agents decompose "build app", CUA agents decompose "automate workflow" |
| **RLM patterns** | Any agent can recursively process large inputs (docs, code, UI sequences) |
| **Smart escalation** | All agents benefit from local→cloud routing with cost ceiling |

**Verdict: CORE FRAMEWORK** ✅

### Quality & Verification (M2-M3)

| Feature | Universal? | Notes |
|---------|:---:|-------|
| **Quality gates framework** | ✅ CORE | Framework is universal, specific gates are agent-dependent |
| **Syntax check gate** | ❌ CODE | Only for code agents |
| **Test runner gate** | ❌ CODE | Only for code agents |
| **Verification pattern** | ✅ CORE | All agents verify output (Chat: citations valid, CUA: action succeeded) |
| **Escalation ladder** | ✅ CORE | retry → decompose → cloud → ask user (universal pattern) |
| **Plan tracking** | ✅ CORE | All agents create plans and track progress |
| **Checkpoint/resume** | ✅ CORE | All agents benefit from crash recovery |
| **Audit log** | ✅ CORE | All agents track actions with timestamps |
| **Time awareness** | ✅ CORE | All agents benefit from knowing: session duration, task duration, file age |

**Verdict: Quality FRAMEWORK is core, specific GATES are agent-dependent**

### Learning & Self-Extension (M5)

| Feature | Universal? | Notes |
|---------|:---:|-------|
| **Insight generation** | ✅ CORE | Chat learns topics, Code learns patterns, CUA learns UI workflows |
| **Insight retrieval** | ✅ CORE | All agents recall relevant learnings |
| **Skill extraction** | ✅ CORE | All agents extract multi-step workflows |
| **Tool creation (ToolBuilder)** | ✅ CORE | Chat creates doc-processing tools, Code creates git tools, CUA creates automation tools |
| **AgentFactory** | ✅ CORE | All agents can create specialists from patterns |

**Verdict: ALL of self-extension is CORE FRAMEWORK** ✅

### Memory Management (M6)

| Feature | Universal? |
|---------|:---:|
| **Vector search (FAISS)** | ✅ CORE |
| **Memory defragmentation** | ✅ CORE |
| **Embedding engine** | ✅ CORE |

**Verdict: CORE FRAMEWORK** ✅

---

## CODE AGENT SPECIFIC (Build After Core)

These features ONLY apply to coding tasks:

### Code-Specific Tools

| Tool | Why Code-Only |
|------|--------------|
| **AST parsing** | Only for code analysis |
| **Syntax validator** | Only for code verification |
| **Test runner** | Only for code testing |
| **Git operations** | Primarily for code (though CUA might use) |
| **Linters (black, eslint)** | Only for code formatting |
| **Type checkers (mypy, tsc)** | Only for code type verification |
| **Build tools (docker, npm)** | Only for code building/deployment |

**Implementation:** These live in `tools/coding/` pack, loaded only by CodeAgent

### Code-Specific Quality Gates

| Gate | Why Code-Only |
|------|--------------|
| **Syntax check** | Code-specific |
| **Import check** | Code-specific |
| **Lint check** | Code-specific |
| **Type check** | Code-specific |
| **Test runner** | Code-specific |
| **Build check** | Code-specific |

**Implementation:** These implement the core QualityGate interface but are registered only for CodeAgent

### Code-Specific Specialists

| Specialist | Why Code-Only |
|-----------|--------------|
| **DebuggerAgent** | Debugging is code-specific |
| **RefactoringAgent** | Refactoring is code-specific |
| **TestingAgent** | Test generation is code-specific |
| **PerformanceAgent** | Code profiling is code-specific |
| **ArchitectureAgent** | Code architecture analysis is code-specific |

**But:** SecurityAgent, DocumentationAgent might be useful for other agents too

### Code-Specific Analysis (M7)

| Feature | Why Code-Only |
|---------|--------------|
| **Codebase indexing** | Only codebases need symbol extraction |
| **Dependency graph** | Only code has import dependencies |
| **Symbol resolution** | Only code has symbols |
| **Call graph** | Only code has function calls |
| **Circular dependency detection** | Only code has this issue |

**Verdict: M7 (codebase indexing) is CODE-SPECIFIC** ❌

---

## The Separation

### CORE AGENT FRAMEWORK (Days 1-19)

```
M0: Prompting + RLM patterns + smart escalation (21h)
M1: SharedAgentState (7 databases) + agent_query() (48h) ← ALL 7 DBs
M2: Quality framework + escalation ladder + plans (20h)
M3: Checkpoint/resume + audit + time awareness (21h)
M4: Agent registry + tool registry + FAISS (18h)
M5: Agent generation + insight generation + learning (41h)
M6: Vector search + defrag (19h)

Total: ~188 hours (Days 1-19)
```

**These 188 hours build the framework that ALL agents use.**

After M6, we have:
- Recursive agent composition working
- 7 databases with semantic search
- agent_query() spawning specialists
- Learning, insights, skills
- Works for Chat, Code, CUA, Workflow - any agent

### CODE AGENT SPECIFIC (Days 19-32)

```
M7: Codebase indexing (tree-sitter, dependency graphs, symbols) (21h)
M4*: 7 CODE-SPECIFIC specialists (Debugger, Refactoring, Testing, Performance, Architecture) (16h)
M2*: CODE-SPECIFIC quality gates (syntax, lint, test, build) (12h)

Total: ~49 hours (Days 19-25)
```

**These 49 hours customize the framework for coding tasks.**

### Other Agent Types (Days 25+)

```
CHAT AGENT SPECIFIC:
  - RAG tools (chunking, retrieval, citation)
  - DocumentSynthesisAgent specialist
  - Quality gate: citation_check

CUA AGENT SPECIFIC:
  - Browser tools (Playwright, screenshot, OCR)
  - BrowserAgent, VisionAgent specialists
  - Quality gate: action_verification

WORKFLOW AGENT SPECIFIC:
  - Email, calendar, Slack tools
  - EmailAgent, SchedulerAgent specialists
  - Quality gate: workflow_completion
```

---

## Updated Timeline

### Phase 1: CORE FRAMEWORK (Days 1-19)

**Build the universal foundation:**
- M0-M6: Everything above
- Works for ANY agent type
- 188 hours of engineering

**Deliverable:** A framework where you can implement ANY agent (chat, code, CUA, workflow) by just:
1. Adding domain-specific tools
2. Adding domain-specific specialists
3. Adding domain-specific quality gates

### Phase 2: CODE AGENT (Days 19-25)

**Customize for coding:**
- Add coding tools (git, pytest, docker, linters)
- Add coding specialists (Debugger, Refactoring, Testing, Performance, Architecture)
- Add coding gates (syntax, test, lint, build)
- Add codebase indexing

**Deliverable:** `gaia code` fully functional

### Phase 3: OTHER AGENTS (Days 25+)

**Parallelize:**
- `gaia chat` (add RAG tools + DocumentAgent)
- `gaia cua` (add browser tools + VisionAgent)
- `gaia workflow` (add email/slack tools + AutomationAgent)

**All three can be built in parallel because they share the core framework.**

---

## The Corrected Priorities

### Tier 1: CORE FRAMEWORK (Build First)

| Priority | Feature | Milestone | Applies To |
|:---:|---------|:---:|-----------|
| 1 | Prompting + RLM patterns | M0 | ALL agents |
| 2 | SharedAgentState (7 DBs) | M1 | ALL agents |
| 3 | agent_query() | M1 | ALL agents |
| 4 | Quality framework | M2 | ALL agents |
| 5 | Plan + manifest | M2 | ALL agents |
| 6 | Checkpoint/resume | M3 | ALL agents |
| 7 | Audit + time | M3 | ALL agents |
| 8 | Tool/agent registries | M4 | ALL agents |
| 9 | Agent generation | M5 | ALL agents |
| 10 | Vector search + defrag | M6 | ALL agents |

**After Tier 1 (Day 19): Universal agent framework complete. Can build ANY agent type on top.**

### Tier 2: CODE AGENT (Build Second)

| Priority | Feature | Type |
|:---:|---------|------|
| 11 | Code-specific tools (git, pytest, docker) | Code-only |
| 12 | Code-specific specialists (5) | Code-only |
| 13 | Code-specific gates (syntax, test, lint) | Code-only |
| 14 | Codebase indexing (M7) | Code-only |

**After Tier 2 (Day 25): `gaia code` complete**

### Tier 3: OTHER AGENTS (Build in Parallel)

Each agent type adds:
- Domain tools
- Domain specialists
- Domain quality gates

---

## The Realization

**You're right.** The current milestones conflate "core framework" with "code agent."

**Better structure:**

```
PHASE 1 (Days 1-19): Build CORE AGENT FRAMEWORK
  → M0-M6 as currently defined
  → Universal for all agents
  → 188 hours

PHASE 2 (Days 19-25): Customize for CODE AGENT
  → Code tools, code specialists, code gates, codebase indexing
  → 49 hours

PHASE 3 (Days 25+): Build OTHER AGENTS in parallel
  → Chat, CUA, Workflow
  → Each reuses the core framework
  → ~40 hours per agent type
```

**The framework comes first. Specialization comes after.**

---

## Should I Create a Separate Document?

**Yes.** I'll create:

**`CORE_AGENT_FRAMEWORK.md`**
- M0-M6 in detail (the universal foundation)
- What every agent gets: recursion, shared state, learning, memory
- ~2,000 lines

**Then update `GAIA_CODE_MILESTONES.md`** to focus on code-specific customization

Want me to create this split?