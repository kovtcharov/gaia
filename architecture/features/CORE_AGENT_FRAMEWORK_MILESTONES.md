# Core Agent Framework: Implementation Milestones

**Date**: February 11, 2026
**Version**: 1.0
**Scope**: Universal agent capabilities (ChatAgent, CodeAgent, CUAAgent, WorkflowAgent, etc.)
**Paradigm**: Recursive Agent Composition (RAC)

---

## Purpose

This document defines the **core agent framework** that ALL GAIA agents inherit. Build this once, all agent types benefit.

**NOT in this document:** Code-specific features (syntax checking, test running, git operations, codebase indexing)
**Those are in:** `gaia-code/GAIA_CODE_MILESTONES.md` (Days 19-32)

---

## The Core Framework (Days 1-19)

### Milestone 0: Prompting Foundation (Days 1-2, 21h)

| # | Feature | Hours | Applies To |
|---|---------|-------|------------|
| 0.1 | Base agent system prompt | 6h | ALL agents (teaches recursion, tools, planning) |
| 0.2 | RLM decomposition patterns | 4h | ALL agents (chunk → recurse → synthesize) |
| 0.3 | Tool description templates | 3h | ALL agents (clear tool descriptions for LLM) |
| 0.4 | Error recovery prompts | 4h | ALL agents (how to diagnose and fix failures) |
| 0.5 | Smart escalation | 3h | ALL agents (local → cloud routing with cost ceiling) |
| 0.6 | CLI entry point pattern | 1h | ALL agents (gaia chat, gaia code, gaia cua pattern) |

**Deliverable:** Well-prompted agent that can recursively decompose, use tools, handle errors.

---

### Milestone 1: SharedAgentState + Seven Databases (Days 3-5, 48h)

> **The RAC foundation. All agents share these seven databases.**

| # | Database | Hours | What It Stores | Used By |
|---|----------|-------|---------------|---------|
| 1.1 | **memory.db** | 3h | Session cache (files, state) | Chat (docs), Code (source), CUA (screenshots) |
| 1.2 | **knowledge.db** | 6h | Cross-session learning (insights, preferences) | Chat (topics), Code (patterns), CUA (workflows) |
| 1.3 | **tools.db** | 4h | Tool registry (metadata, usage stats) | ALL agents (78 core + domain tools) |
| 1.4 | **skills.db** | 3h | Learned multi-step workflows | ALL agents (extract patterns from experience) |
| 1.5 | **agents.db** | 3h | Specialist agent registry | ALL agents (spawn domain experts) |
| 1.6 | **ProjectManifest** | 4h | Current project state (files, artifacts, decisions) | ALL agents (track what's been created) |
| 1.7 | **MasterPlan** | 4h | Hierarchical task tree | ALL agents (decompose goal into tasks) |
| 1.8 | **MessageQueue** | 4h | Async agent ↔ user communication | ALL agents (FYI, Question, Decision priorities) |
| 1.9 | **AgentCallStack** | 3h | Recursion tracking (depth, lineage) | ALL agents (prevent infinite recursion) |
| 1.10 | **SharedAgentState** | 4h | Singleton integrating all 7 DBs + orchestration | ALL agents (THE SAME instance across recursion) |
| 1.11 | **agent_query() tool** | 8h | Core RAC mechanism (spawn sub-agents) | ALL agents (recursively delegate tasks) |
| 1.12 | **recall() tool** | 2h | Query knowledge.db for past context | ALL agents (remember past sessions) |

**Storage layout:**
```
~/.gaia/
├── memory/memory.db           # Session cache
├── knowledge/knowledge.db     # Cross-session learning
├── tools/tools.db             # Tool registry
├── skills/skills.db           # Learned workflows
└── agents/agents.db           # Specialist registry

.gaia/ (project-specific)
├── manifest.db                # Project state
└── plan.db                    # Active plan
```

**Why ALL 7 databases in M1:** They are foundational. agent_query() needs agents.db to find specialists. Tools need tools.db for dynamic loading. Learning needs knowledge.db. Can't build recursion without the foundation.

**Deliverable:** Complete RAC infrastructure. Any agent can now spawn specialists, share state, learn, and recurse.

---

### Milestone 2: Quality Framework + Planning (Days 5-8, 20h)

> **Universal verification and planning patterns.**

| # | Feature | Hours | Applies To |
|---|---------|-------|------------|
| 2.1 | **QualityGate base class** | 4h | ALL agents (interface: check(), passed, failures) |
| 2.2 | **Escalation ladder** | 4h | ALL agents (retry → decompose → cloud → ask) |
| 2.3 | **Plan creation** | 4h | ALL agents (decompose task into subtasks) |
| 2.4 | **Progress tracking** | 3h | ALL agents (N/M tasks, X% complete) |
| 2.5 | **Plan updates & replanning** | 3h | ALL agents (discover dependencies, adjust plan) |
| 2.6 | **Continuous execution** | 2h | ALL agents (run until verified complete, not max_steps) |

**Agent-specific gates implemented separately:**
- CodeAgent: syntax, lint, test, build gates
- ChatAgent: citation_valid, source_check gates
- CUAAgent: action_succeeded, screenshot_match gates

**Deliverable:** Quality framework + planning system. Each agent type implements its own gates.

---

### Milestone 3: Persistence & Transparency (Days 8-10, 21h)

> **Checkpoint, audit, time awareness.**

| # | Feature | Hours | Applies To |
|---|---------|-------|------------|
| 3.1 | **State-based checkpoint** | 6h | ALL agents (save plan + manifest + gates, not prose) |
| 3.2 | **Resume from checkpoint** | 6h | ALL agents (load state, inject context, continue) |
| 3.3 | **Audit log** | 4h | ALL agents (every action timestamped and tracked) |
| 3.4 | **Time awareness** | 3h | ALL agents (session duration, task duration, file age) |
| 3.5 | **gaia status / gaia audit** | 2h | ALL agents (CLI commands to check progress) |

**Deliverable:** Resilient agents that survive crashes, track everything, have sense of time.

---

### Milestone 4: Tool & Agent Registries (Days 10-13, 18h)

> **Organize capabilities, enable dynamic loading.**

| # | Feature | Hours | Applies To |
|---|---------|-------|------------|
| 4.1 | **FAISS embedding engine** | 4h | ALL agents (local embeddings on AMD NPU, <10ms) |
| 4.2 | **FAISS indices** | 4h | ALL agents (tools.db, skills.db, agents.db semantic search) |
| 4.3 | **Dynamic tool loading** | 4h | ALL agents (load only relevant 20-30 tools per task) |
| 4.4 | **Specialist auto-selection** | 4h | ALL agents (semantic search for best specialist) |
| 4.5 | **Usage tracking** | 2h | ALL agents (tools, skills, agents usage stats and confidence) |

**Note:** Actual tools/specialists are agent-specific, but the REGISTRY is universal.

**Deliverable:** Registries with semantic search. Agents load capabilities dynamically.

---

### Milestone 5: Self-Extension & Learning (Days 13-16, 41h)

> **Agents create their own capabilities.**

| # | Feature | Hours | Applies To |
|---|---------|-------|------------|
| 5.1 | **ToolBuilder** | 6h | ALL agents (write, validate, test, register tools) |
| 5.2 | **Skill extraction** | 6h | ALL agents (extract workflows from successful tasks) |
| 5.3 | **Skill recall** | 4h | ALL agents (find and apply similar past workflows) |
| 5.4 | **InsightEngine** | 6h | ALL agents (generate insights: error-fix, patterns, preferences) |
| 5.5 | **Insight retrieval** | 4h | ALL agents (semantic + trigger matching + domain search) |
| 5.6 | **AgentFactory** | 6h | ALL agents (detect patterns, generate specialists) |
| 5.7 | **Agent code generation** | 6h | ALL agents (generate specialist class with state machine, tools, prompt) |
| 5.8 | **Confidence tracking** | 3h | ALL agents (track specialist success rates, promote/deprecate) |

**Examples by agent type:**
- CodeAgent creates: FastAPICRUDAgent, AsyncDebuggerAgent
- ChatAgent creates: SummarizationAgent, CitationAgent
- CUAAgent creates: FormFillingAgent, BrowserNavigationAgent

**Deliverable:** Agents that improve themselves by creating specialists.

---

### Milestone 6: Memory Optimization (Days 16-19, 19h)

> **Keep quality high as data accumulates.**

| # | Feature | Hours | Applies To |
|---|---------|-------|------------|
| 6.1 | **Vector search** | 4h | ALL agents (semantic retrieval over all 7 DBs) |
| 6.2 | **Memory defragmentation** | 6h | ALL agents (dedup, reconcile, prune, consolidate) |
| 6.3 | **Defrag verification** | 2h | ALL agents (test retrieval quality before/after) |
| 6.4 | **Auto-defrag trigger** | 2h | ALL agents (trigger when quality drops below 90%) |
| 6.5 | **gaia memory commands** | 3h | ALL agents (status, defrag, archive commands) |
| 6.6 | **Archive management** | 2h | ALL agents (nothing deleted, everything archived) |

**Deliverable:** Agents stay fast and accurate even after 1000+ sessions.

---

## Composability: Simple → Complex, Small → Large

> **The framework must work with 0.6B models on simple tasks AND 200B models on multi-month projects.**

### Feature Degradation by LLM Tier

| Feature | 0.6B (basic) | 7B (standard) | 30B (advanced) | Cloud (opus) |
|---------|:---:|:---:|:---:|:---:|
| **Tools (read, write, terminal)** | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **Recursion (agent_query)** | ❌ Off | ✅ depth=2 | ✅ depth=10 | ✅ depth=10 |
| **memory.db (session cache)** | ✅ Full | ✅ Full | ✅ Full | ✅ Full |
| **knowledge.db (learning)** | ❌ Off | ✅ FTS5 only | ✅ FTS5 + FAISS | ✅ Full |
| **Quality gates** | ❌ Off | ✅ Basic | ✅ Full | ✅ Full |
| **Manifest** | ❌ Off | ✅ Full | ✅ Full | ✅ Full |
| **Plan tracking** | ❌ Off | ✅ Basic | ✅ Full | ✅ Full |
| **Checkpoint/resume** | ❌ Off | ✅ Full | ✅ Full | ✅ Full |
| **Specialists** | ❌ Off | ❌ Off | ✅ Full | ✅ Full |
| **Agent generation** | ❌ Off | ❌ Off | ✅ Full | ✅ Full |
| **Insights** | ❌ Off | ❌ Off | ✅ Full | ✅ Full |
| **Defrag** | ❌ Off | ❌ Off | ✅ Full | ✅ Full |

### Configuration Profiles

```python
# MINIMAL (0.6B model, single-turn tasks)
V2Config(
    llm_tier="basic",
    enable_recursion=False,     # Direct execution, no agent_query()
    enable_memory_db=True,      # Session cache only
    enable_knowledge_db=False,  # No cross-session learning
    enable_quality_gates=False, # No verification (fast)
    enable_specialists=False,   # No agent registry
    max_depth=0                 # No recursion
)

# Use case: "Summarize this doc" (ChatAgent), "Fix typo" (CodeAgent)
# Behavior: Direct execution, <5 seconds


# STANDARD (7B model, multi-turn sessions)
V2Config(
    llm_tier="standard",
    enable_recursion=True,      # Limited recursion
    enable_memory_db=True,
    enable_knowledge_db=True,   # FTS5 search only (no FAISS yet)
    enable_quality_gates=True,  # Basic verification
    enable_specialists=False,   # No specialists yet (7B can't handle)
    enable_manifest=True,
    enable_plan=True,
    max_depth=2                 # Shallow recursion
)

# Use case: "Analyze 5 docs and compare" (Chat), "Build CLI tool" (Code)
# Behavior: Recursive decomposition (depth=2), session memory, basic verification
# Time: 5-30 minutes


# ADVANCED (30B model, complex tasks)
V2Config(
    llm_tier="advanced",
    enable_recursion=True,      # Full recursion
    enable_memory_db=True,
    enable_knowledge_db=True,   # FTS5 + FAISS
    enable_quality_gates=True,
    enable_specialists=True,    # 7 core specialists available
    enable_agent_generation=True, # Can create new specialists
    enable_insights=True,
    enable_defrag=True,
    enable_smart_escalation=True, # Can escalate to cloud
    max_depth=10,
    max_cloud_cost=2.00         # $2/session ceiling
)

# Use case: "Build full-stack app" (Code), "Research 50 papers" (Chat),
#          "Automate complex workflow" (CUA)
# Behavior: Deep recursion, specialists, learning, cloud escalation for hard subtasks
# Time: 1-4 hours


# AUTONOMOUS (hybrid, multi-day/month)
V2Config(
    llm_tier="advanced",
    enable_all=True,            # Everything on
    max_depth=10,
    max_cloud_cost=50.00,       # Higher ceiling for complex reasoning
    enable_async_interaction=True, # User checks in periodically
    session_timeout=None,       # No timeout
    enable_continuous=True      # Runs indefinitely until goal complete
)

# Use case: "Build entire agent" (Code), "Create comprehensive knowledge base" (Chat)
# Behavior: Creates specialists, checkpoints, runs for weeks, user checks in weekly
# Time: Days to months
```

### Graceful Degradation Pattern

```python
class RecursionMixin:
    """All agents get this, degrades by tier."""

    def init(self, config: V2Config):
        if config.llm_tier == "basic":
            self.can_recurse = False
            self.agent_query = self._agent_query_disabled
        elif config.llm_tier == "standard":
            self.can_recurse = True
            self.max_depth = 2  # Shallow only
        else:
            self.can_recurse = True
            self.max_depth = 10  # Full depth

    def _agent_query_disabled(self, *args, **kwargs):
        """Fallback when recursion disabled."""
        raise FeatureDisabled(
            "agent_query() requires 'standard' tier or above. "
            "Current tier: basic. Upgrade to use recursive decomposition."
        )
```

**Every feature checks its flag:**
- Disabled = zero overhead (not even imported)
- Degraded = simpler version (depth=2 vs depth=10)
- Full = all capabilities

---

## Summary: Core Framework Complete

**After M0-M6 (188 hours over 19 days):**

```
You have a UNIVERSAL AGENT FRAMEWORK that provides:

✅ Recursive decomposition (agent_query, RLM patterns)
✅ Seven databases (memory, knowledge, tools, skills, agents, manifest, plan)
✅ Shared state across recursion tree
✅ Quality verification framework (each agent implements own gates)
✅ Planning and progress tracking
✅ Checkpoint/resume
✅ Audit log and time awareness
✅ Tool creation, skill extraction, agent generation
✅ Vector search and memory optimization
✅ Smart escalation (local → cloud with cost ceiling)
✅ Async interaction (message queue)

NOW you can build ANY agent type:
├─ gaia code   → Add coding tools, coding specialists, coding gates
├─ gaia chat   → Add RAG tools, doc specialists, citation gates
├─ gaia cua    → Add browser tools, vision specialists, action gates
└─ gaia workflow → Add email/slack tools, automation specialists
```

**The 188 hours build the foundation. Then each agent type is ~40-50 hours of customization.**

---

## What's Code-Specific (NOT in Core Framework)

**Move to `gaia-code/CODE_AGENT_CUSTOMIZATION.md` (separate doc):**

- M7: Codebase indexing (tree-sitter, dependency graphs)
- Code-specific tools: pytest, black, mypy, docker, git merge/rebase
- Code-specific specialists: DebuggerAgent, RefactoringAgent, TestingAgent, PerformanceAgent, ArchitectureAgent
- Code-specific quality gates: syntax, lint, test, type, build
- Code-specific skills: build_fastapi_app, debug_async, setup_docker

**These are customizations ON TOP of the core framework.**

---

---

## Evaluation Benchmarks Per Capability

> **For each core capability, define how to test and measure it.**

### M0: Prompting Foundation

**Eval:** Baseline Capability Test
```bash
# Test with 10 simple tasks across domains
tasks = [
    "Summarize this 10-page PDF",           # Chat
    "Fix syntax error in this Python file", # Code
    "Click the login button on this page",  # CUA
    "Send email to team about meeting"      # Workflow
]

# Metrics:
Success rate: X/10 tasks completed correctly
Avg time: Y seconds per task
Tool usage: Did agent pick correct tools?
```

**Benchmark:** Agent should succeed on 8/10 simple tasks with well-optimized prompts.

---

### M1: SharedAgentState + Recursion

**Eval:** Recursive Decomposition Test
```bash
# Test recursive agent spawning
gaia test-recursion "Build a 3-tier web app (frontend, backend, database)"

# Expected behavior:
# Main agent → agent_query("Build backend")
#            → agent_query("Build frontend")
#            → agent_query("Set up database")
# Sub-agents share manifest, plan, knowledge.db

# Metrics:
Recursion depth: Did agent decompose correctly? (target: depth=2-3)
State sharing: Did sub-agents see each other's work?
Manifest updates: Are all files tracked?
Plan coherence: Does master plan show all subtasks?
```

**Benchmark:** Successfully completes 3-tier decomposition, all agents coordinate via shared state.

---

### M1: Seven Databases

**Eval:** Database Integration Test
```bash
# Session 1:
gaia code "Create project using pathlib"
# Stores: preference in knowledge.db, files in memory.db

# Session 2 (new session):
gaia code "Add file processing to project"

# Metrics:
Cross-session memory: Did agent remember pathlib preference?
File cache hit rate: Did agent use memory.db vs re-reading?
Tool usage logged: Is tools.db tracking usage?
```

**Benchmark:**
- knowledge.db: 100% preference recall
- memory.db: >80% cache hit rate
- tools.db: All tool calls logged

---

### M2: Quality Framework

**Eval:** Quality Gate Test (agent-dependent gates)
```bash
# Code agent:
gaia code "Build API with intentional syntax error"
# Should: detect error, fix, retry, pass

# Chat agent:
gaia chat "Summarize with fake citation"
# Should: detect invalid citation, fix, pass

# Metrics:
Gate detection rate: Did gates catch the error?
Fix success rate: Did agent fix after detection?
Escalation: Did ladder trigger (retry → decompose)?
```

**Benchmark:** 100% error detection, >90% fix success rate

---

### M2: Escalation Ladder

**Eval:** Failure Handling Test
```bash
# Give agent a task that local LLM can't handle
gaia code "Implement OAuth2 PKCE flow with refresh token rotation"

# Expected escalation:
Attempt 1-2: Local LLM tries (likely fails on OAuth2 complexity)
Attempt 3: Decompose into smaller subtasks
Attempt 4: Escalate to cloud LLM for OAuth2 subtask specifically
Success: Cloud handles OAuth2, local handles rest

# Metrics:
Escalation triggered: Yes
Cloud cost: <$0.50 (only OAuth2 subtask)
Success rate: Task completes
Learning: Agent stores "OAuth2 needs cloud" insight
```

**Benchmark:** <10% of subtasks need cloud escalation, <$1/session avg cost

---

### M3: Checkpoint/Resume

**Eval:** Crash Recovery Test
```bash
# Start long task
gaia code "Build project with 30 files" &
PID=$!

# Kill at 50% complete
sleep 300 && kill $PID

# Resume
gaia code --resume

# Metrics:
State recovery: Did agent resume from 50%, not 0%?
Zero rework: Did agent skip completed files?
Plan continuity: Does plan show correct progress?
```

**Benchmark:** 100% state recovery, 0% rework

---

### M3: Audit Log

**Eval:** Transparency Test
```bash
gaia code "Build API with auth"
gaia code audit

# Check audit log contains:
✅ Every file write with timestamp
✅ Every test run with pass/fail
✅ Every error and fix
✅ Every plan update
✅ Every specialist delegation

# Metrics:
Coverage: 100% of actions logged
Queryable: Can filter by time, type, agent
```

**Benchmark:** Full audit trail, <100ms query time

---

### M4: Dynamic Tool Loading

**Eval:** Tool Relevance Test
```bash
# Task that needs docker tools
gaia code "Create Docker deployment"

# Check: tools loaded
Loaded tools should include: docker_build, docker_run, docker_compose
Loaded tools should NOT include: browser_click, send_email (irrelevant)

# Metrics:
Precision: % of loaded tools actually used
Recall: % of used tools that were loaded
```

**Benchmark:** >80% precision, 100% recall

---

### M4: Specialist Auto-Selection

**Eval:** Routing Test
```bash
# Tasks that should trigger specialists
gaia code "Debug this failing test"
# → Should auto-select debugger_agent

gaia code "Find security vulnerabilities"
# → Should auto-select security_agent

# Metrics:
Selection accuracy: Did it pick the right specialist?
Confidence correlation: Do high-confidence specialists succeed more?
```

**Benchmark:** >90% correct specialist selection

---

### M5: Agent Auto-Generation

**Eval:** Specialist Creation Test
```bash
# Session 1-3: Similar tasks
gaia code "Create CRUD for users"
gaia code "Create CRUD for posts"
gaia code "Create CRUD for comments"

# Session 4: Should trigger agent generation
gaia code "Create CRUD for products"

# Check:
ls ~/.gaia/agents/learned/
# Should contain: fastapi_crud_agent.py

# Metrics:
Pattern detection: Triggered after 3 similar tasks?
Specialist quality: Does generated specialist work?
Performance improvement: 4th task faster than 1st-3rd?
```

**Benchmark:**
- Pattern detected: Yes (after 3 tasks)
- Generated specialist works: Yes (>80% success rate)
- Speedup: 4th task is >2x faster

---

### M5: Insight Generation

**Eval:** Learning Test
```bash
# Session 1:
gaia code "Read CSV file"
# Agent uses pandas
# User corrects: "Use csv module instead"

# Session 2:
gaia code "Process TSV file"

# Check:
Agent should recall insight: "User prefers csv module"
Agent should use csv module without correction

# Metrics:
Insight generated: Yes (from user correction)
Insight retrieved: Yes (when relevant)
Behavior changed: Agent adapted
```

**Benchmark:** >90% insight recall when relevant, 100% behavior adaptation

---

### M6: Memory Defragmentation

**Eval:** Long-Term Reliability Test
```bash
# Simulate 100 sessions
for i in {1..100}; do
    gaia code "Build small project $i"
done

# After 100 sessions:
gaia memory status
# Check: duplicates, contradictions, stale entries

gaia memory defrag

# Measure:
Retrieval quality before: X%
Retrieval quality after: Y%
Improvement: Y - X (target: >5%)

# Metrics:
Entry reduction: How many merged/pruned?
Quality improvement: Retrieval accuracy improved?
Performance: Agent still fast after 100 sessions?
```

**Benchmark:**
- Retrieval quality: >90% after defrag
- Performance: No degradation after 100 sessions
- Duplicate removal: >50 duplicates merged

---

## Benchmark Summary Table

| Milestone | Capability | Test | Success Criteria |
|-----------|-----------|------|------------------|
| M0 | Prompting | 10 simple tasks | 8/10 success |
| M1 | Recursion | 3-tier decomposition | Depth 2-3, shared state works |
| M1 | 7 Databases | Cross-session memory | 100% preference recall, 80% cache hits |
| M2 | Quality gates | Error detection | 100% detection, 90% fix rate |
| M2 | Escalation | Failure handling | <10% cloud escalation, <$1/session |
| M3 | Checkpoint | Crash recovery | 100% state recovery, 0% rework |
| M3 | Audit | Transparency | 100% action coverage |
| M4 | Tool loading | Relevance | >80% precision, 100% recall |
| M4 | Specialist selection | Routing accuracy | >90% correct routing |
| M5 | Agent generation | Pattern detection | Specialist created after 3 tasks |
| M5 | Insights | Learning | >90% recall, 100% adaptation |
| M6 | Defrag | Long-term reliability | >90% quality, no degradation after 100 sessions |

**These benchmarks are AGENT-AGNOSTIC.** They work for chat, code, CUA, workflow agents.

---

*This document: 188 hours to build the universal agent framework.*
*All agents (chat, code, CUA, workflow) inherit this foundation.*
*Agent-specific features are 40-50h of customization each.*
*Each capability has measurable benchmarks for validation.*
