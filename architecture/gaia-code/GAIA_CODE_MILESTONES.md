# Gaia Code: Milestone Roadmap

**Date**: February 10, 2026
**Version**: 1.0
**Companion to**: `GAIA_CODE_AUTONOMOUS_AGENT.md` (full spec), `GAIA_CODE_EVALUATION_PLAN.md` (eval)

---

## How to Read This Document

`GAIA_CODE_AUTONOMOUS_AGENT.md` describes the full vision. **This document tells you what to build, in what order, and why.**

Each milestone is independently valuable -- if we stop after any milestone, we have a useful agent. Later milestones build on earlier ones but never invalidate them.

---

## The Honest Truth

Before planning milestones, we need to acknowledge reality:

1. **The LLM quality is the single biggest factor.** A great architecture with a mediocre LLM produces mediocre results. A simple architecture with a great LLM produces great results. Our infrastructure amplifies the LLM -- it doesn't replace it.

2. **Prompting matters more than infrastructure.** Getting the system prompt, tool descriptions, and error recovery prompts right will have more impact than any database or framework we build.

3. **Start simple, prove value, then add complexity.** Every piece of infrastructure we add before proving the basic agent works is a risk. Infrastructure that sits on top of a broken foundation is wasted effort.

4. **Test with real tasks constantly.** Don't build for 2 weeks then test. Build for 1 day, test with real coding tasks, learn, adjust, repeat.

---

## Milestone 0: The Prompting Foundation (Days 1-2)

> **The most important milestone. Everything else depends on this.**

### What

Get a basic `gaia code` command working with excellent prompting -- no new infrastructure, just a well-prompted agent that uses the existing CodeAgent's 70+ tools effectively.

### Why First

Claude Code's power comes from its prompts, not its infrastructure. Before building databases and frameworks, we need to prove that a well-prompted local LLM can perform useful coding tasks end-to-end. If it can't, no amount of infrastructure will fix it.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 0.1 | **System prompt engineering** | 8h | Craft the core system prompt for `gaia code`. This is the most important 8 hours of the entire project. Study Claude Code's prompting patterns. Test with Qwen3-Coder-30B. Iterate until the agent can reliably: read files, write code, run tests, fix errors, use git. |
| 0.2 | **Tool description optimization** | 4h | The existing 70+ tools have descriptions. Optimize them for the LLM -- clear, concise, with examples. Bad tool descriptions = LLM picks wrong tools. |
| 0.3 | **Error recovery prompts** | 4h | When code fails, the prompt that tells the agent HOW to debug matters enormously. Craft prompts for: syntax errors, test failures, import errors, runtime crashes. |
| 0.4 | **`gaia code` CLI entry point** | 2h | Wire up a `gaia code "task"` command that uses the optimized prompts. Minimal wrapper around existing CodeAgent. |
| 0.5 | **Smart escalation (local → cloud)** | 3h | Per-subtask routing: try local LLM first, escalate to cloud after 2 failures. Cost ceiling per session. Built into the agent from day 1. |
| 0.6 | **RLM-aware prompting** | 3h | Teach the agent recursive decomposition patterns via system prompt. When a task/file is too large for one pass, the agent should decompose → recurse → synthesize. Based on [Recursive Language Models](https://arxiv.org/abs/2512.24601) (RLM-Qwen3-8B outperforms base by 28.3%). |

### Validation

```bash
# Test 1: Simple task
gaia code "Create a Python function that checks if a number is prime, with tests"
# ✅ Function is correct
# ✅ Tests exist and pass
# ✅ Code is clean

# Test 2: Multi-file task
gaia code "Create a CLI calculator with add, subtract, multiply, divide. Include tests."
# ✅ Multiple files created (cli.py, calculator.py, test_calculator.py)
# ✅ CLI works
# ✅ Tests pass

# Test 3: Bug fixing
gaia code "Fix the bug in this file: [provide file with known bug]"
# ✅ Agent finds the bug
# ✅ Agent fixes it correctly
# ✅ Agent doesn't break other things
```

### What We Learn

- How capable is Qwen3-Coder-30B as a coding agent?
- What kinds of tasks does it handle well vs. poorly?
- Where does the LLM fail and what kind of infrastructure would actually help?
- Is the existing tool set sufficient or are there gaps?

**This milestone answers the most important question: Is this viable?**

### Exit Criteria

- `gaia code` command works end-to-end
- Agent can complete 5/10 simple coding tasks correctly
- We have a clear understanding of where the LLM struggles (this informs all future milestones)

---

## Milestone 1: SharedAgentState + Recursive Agent Foundation (Days 3-5)

> **The foundational infrastructure for Recursive Agent Composition (RAC).**

### What

Build the shared state infrastructure that ALL agents in the recursion tree access: memory.db (working cache), knowledge.db (cross-session learning), manifest (project state), plan (hierarchical tasks), call stack, message queue, and the three capability registries (tools, skills, agents).

**This is the RAC foundation.** Everything else builds on this.

### Why After M0

M0 tells us where the agent struggles. If it struggles with "forgetting what it did 10 steps ago" (likely), this milestone fixes that. If it struggles with "generating bad code" (also likely), we know to invest more in prompting before adding infrastructure.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 1.1 | **memory.db (working memory cache)** | 3h | Session-scoped cache: file contents, active state. Faster than re-reading from disk. Shared across all agents in recursion tree. |
| 1.2 | **knowledge.db (cross-session learning)** | 6h | Permanent storage: insights, preferences, learnings, conventions. Tables: insights, preferences, learnings, conventions. FTS5 full-text search. |
| 1.3 | **ProjectManifest class** | 4h | Live project state: files created, APIs defined, database schemas, architecture decisions. Thread-safe updates during async operations. |
| 1.4 | **MasterPlan class** | 4h | Hierarchical task tree: goal, tasks, subtasks, owners (which agent), progress. Each agent can add tasks, complete tasks, replan. |
| 1.5 | **AgentCallStack class** | 3h | Tracks recursion: who spawned whom, at what depth, with what task. Prevents infinite recursion (max_depth). Full audit trail. |
| 1.6 | **MessageQueue class** | 4h | Async communication: agent ↔ user, agent ↔ agent. Three priorities (FYI, Question, Decision). Non-blocking. |
| 1.7 | **SharedAgentState class** | 2h | Integrates all above into singleton. Every agent in recursion tree gets THE SAME instance. Thread locks for safety during async I/O. |
| 1.8 | **tools.db (tool registry)** | 4h | Tool registry database: name, description, category, usage stats. Start with 78 core tools registered. FTS5 search. Foundation for dynamic tool loading. |
| 1.9 | **skills.db (learned workflows)** | 4h | Skills database: name, steps, tools_used, success_count, confidence. Empty initially (populated in M5). Schema ready for skill extraction. |
| 1.10 | **tools.db (tool registry)** | 4h | Tool registry database with tables: tools, tool_usage, tool_tags. Register 78 existing core tools with metadata. FTS5 search. Ready for dynamic loading in M4. |
| 1.11 | **skills.db (learned workflows)** | 3h | Skills database with tables: skills, skill_usage. Schema ready. Empty initially - will be populated in M5 as agent learns. |
| 1.12 | **agents.db (specialist registry)** | 3h | Agent registry database with tables: agents, agent_usage, agent_capabilities. Schema ready. Empty initially - will be populated with 7 core specialists in M4. |
| 1.13 | **SharedAgentState class (integration)** | 4h | Integrates ALL 7 databases into singleton: memory, knowledge, tools, skills, agents, manifest, plan. Plus call_stack and message_queue. Thread locks for async safety. Every agent in recursion tree gets THE SAME instance. This is the complete RAC foundation. |
| 1.14 | **`agent_query()` tool (basic)** | 8h | Core RAC mechanism: spawn sub-agent with task, optional specialist, inherited context, tools, max_depth. Sub-agent has full agency (tools, quality gates, can recurse). Returns verified result. Searches agents.db for specialist (empty initially). |
| 1.15 | **`recall` tool** | 2h | Query knowledge.db for past context: `recall("what files did I create?")`. Uses FTS5. |
| 1.16 | **Automatic persistence** | 2h | Every tool call, file read, error automatically stored in memory.db and knowledge.db. Tool usage tracked in tools.db. |

### Validation

```bash
# Test: Cross-session memory
# Session 1:
gaia code "Create a Python project using pathlib for all file operations"
# Session 2 (new session):
gaia code "Add a file processing module to the project"
# ✅ Agent knows the project exists
# ✅ Agent uses pathlib (remembers from session 1)
# ✅ Agent doesn't re-read files it already cached

# Test: Context pressure
gaia code "Build a project with 20+ files"
# ✅ Agent completes without context compaction
# ✅ Context stays under 50%
# ✅ Agent can recall early files using `recall` tool
```

### What We Learn

- Does persistent context actually improve agent performance?
- Is FTS5 sufficient for retrieval or do we need vector search?
- How much latency does the DB add to each turn?

### Exit Criteria

- Agent remembers across sessions
- Context never hits compaction
- `recall` tool works and the agent uses it naturally

---

## Milestone 2: Quality Gates + Continuous Execution (Days 5-8)

> **The core differentiator: the agent verifies its own output and doesn't stop mid-task.**

### What

Add automated quality checks and remove the step limit. The agent runs until the task is verified complete (all quality gates pass), not until it hits max_steps.

### Why After M1

Quality gates need the knowledge DB to track which gates passed/failed across the session. Continuous execution needs context trimming (from M1) to prevent context overflow.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 2.1 | **Basic quality gates** | 6h | Three gates: syntax check (ast.parse), test runner (pytest/jest auto-detect), import check. These three catch 80% of issues. No lint, no type check, no security scan yet. |
| 2.2 | **Gate-driven completion + escalation ladder** | 6h | Agent cannot declare "done" until all enabled gates pass. If gates fail: retry (2x) → decompose into smaller subtasks → escalate to cloud LLM → ask user via message queue. Never loops infinitely. |
| 2.3 | **Remove max_steps** | 4h | Replace `max_steps=100` with quality-gate-driven completion. Add a generous safety timeout (60 minutes) instead of step count. |
| 2.4 | **Plan creation and tracking** | 6h | Before coding, agent creates a plan (list of tasks). Checks off tasks as it completes them. Plan persists in knowledge.db. |
| 2.5 | **Progress reporting** | 2h | Agent reports progress: "[Step 3/7] Implementing auth... Tests: 4/6 passing." |

### Validation

```bash
# Test: Quality gates
gaia code "Create a FastAPI API with user auth and tests"
# ✅ Agent runs until ALL tests pass (not until max_steps)
# ✅ If agent writes broken code, it detects and fixes it
# ✅ Agent produces a plan and tracks progress
# ✅ Final output: "All quality gates passed: syntax ✅, imports ✅, tests 12/12 ✅"

# Test: Continuous execution
gaia code "Build a project with 15+ files"
# ✅ Agent doesn't stop at step 100
# ✅ Agent completes the full task
# ✅ All files are syntactically valid
# ✅ All tests pass
```

### What We Learn

- Do quality gates actually improve output quality?
- How often does the agent enter fix-and-retry loops? (If constantly, the LLM is the bottleneck)
- Is plan-and-track effective or does the agent ignore its own plan?

### Exit Criteria

- Agent never stops mid-task due to step limit
- Quality gates catch real errors (not just false positives)
- Agent creates and follows plans

---

## Milestone 3: Checkpoint/Resume + Audit Log (Days 8-10)

> **The agent survives interruptions and shows its work.**

### What

Save full agent state to disk so it can resume after crashes, context limits, or intentional pauses. Every action is logged with timestamps.

### Why After M2

Checkpoint needs the plan (from M2) and knowledge DB (from M1) to know what to save. The audit log naturally extends the quality gate results tracking.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 3.1 | **Checkpoint serialization** | 6h | Save: current plan, completed tasks, quality gate results, active file list, conversation summary. Store in knowledge.db. |
| 3.2 | **Resume from checkpoint** | 6h | On startup, detect if there's an incomplete task. Load checkpoint, inject summary into context, continue from where we left off. |
| 3.3 | **Audit log** | 4h | Every action (file write, test run, plan update, error fix) is logged with timestamp, duration, and outcome. Stored in knowledge.db. Queryable. |
| 3.4 | **Time awareness** | 3h | Agent knows: session duration, task duration, time since last file access, estimated time remaining. Timestamps on everything. |
| 3.5 | **`gaia code status` / `gaia code audit`** | 2h | CLI commands to check progress and view audit log. |

### Validation

```bash
# Test: Crash recovery
gaia code "Build a project with 20 files"
# Kill the process at step 15
gaia code --resume
# ✅ Agent resumes from step 15, not step 1
# ✅ Agent knows what it already completed
# ✅ Final output is complete and correct

# Test: Audit log
gaia code audit
# ✅ Shows every action with timestamps
# ✅ Shows time per action, total time, errors encountered
```

### Exit Criteria

- Agent survives kill + resume without data loss
- Audit log captures all actions
- User can check progress at any time

---

## Milestone 4: Agent Registry + Core Specialists (Days 10-13)

> **The agent gets domain experts. Recursion becomes specialized.**

### What

Create agents.db registry and implement 7 core specialized agents (Debugger, Security, Refactoring, Testing, Documentation, Performance, Architecture). Update `agent_query()` to support specialist parameter. Organize tools into domain-specific packs.

### Why After M3

The foundation (prompting, knowledge, quality gates, checkpoint) is solid. Now we can improve the agent's capabilities by making tools more organized and discoverable.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 4.1 | **7 core specialized agents** | 16h | Implement: DebuggerAgent, SecurityAgent, RefactoringAgent, TestingAgent, DocumentationAgent, PerformanceAgent, ArchitectureAgent. Each has custom state machine, domain-specific tools, specialized system prompt. Register in agents.db (schema already exists from M1). |
| 4.2 | **Specialist auto-selection** | 4h | Implement find_specialist() with semantic search over agents.db. agent_query() auto-selects best specialist if not specified. Confidence-based ranking. |
| 4.3 | **Reorganize 78 tools into domain packs** | 6h | Organize into: core/, coding/, cua/, knowledge/, workflow/, analysis/. Update tools.db with categories. Specialists declare TOOL_PACKS they load. |
| 4.4 | **Dynamic tool loading** | 4h | Agent loads only relevant tools (top 20-30) per task via semantic search over tools.db (schema from M1). Reduces prompt size, improves LLM accuracy. |
| 4.5 | **FAISS indices for semantic search** | 4h | Build FAISS indices for tools.db, skills.db (empty), agents.db (7 specialists). Semantic search: "debug async code" → finds AsyncDebuggerAgent. |

### Validation

```bash
gaia code tools
# ✅ Shows organized tool list by category
# ✅ Shows usage stats

gaia code "Build a Docker-deployed FastAPI app"
# ✅ Agent automatically loads docker tools + python tools + http tools
# ✅ Agent doesn't get confused by irrelevant tools (e.g., browser tools)
```

### Exit Criteria

- Tools are organized into packs
- Agent loads relevant tools dynamically
- No tool is permanently unused (remove or improve underperforming tools)

---

## Milestone 5: Agent Auto-Generation + Self-Extension (Days 13-16)

> **The agent creates its own specialists. Recursive self-improvement begins.**

### What

The agent detects recurring patterns (3+ similar successful tasks) and automatically generates new specialized agents. Also creates tools and skills. The full three-level hierarchy comes alive: tools → skills → agents.

### Why After M4

The agent needs a working tools infrastructure (M4) before it can add to it. And we need enough usage data from M0-M4 to know what tools and skills are actually useful.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 5.1 | **ToolBuilder** | 6h | Agent can write a Python function, validate it (AST check, no dangerous ops), test it, and register it as a new tool. |
| 5.2 | **skills.db** | 4h | SQLite database for learned workflows. Schema: name, steps, tools_used, success_count, confidence. FTS5 search. |
| 5.3 | **Skill extraction** | 6h | After completing a task successfully, agent can extract the pattern as a reusable skill. "I built a FastAPI app with JWT auth. Steps: 1... 2... 3..." |
| 5.4 | **Skill recall** | 4h | When starting a new task, agent searches skills.db for relevant past workflows. "I've done something similar before. Let me follow that pattern." |
| 5.5 | **Insight generation engine** | 6h | After every completed task: extract structured insights (error-fix pairs, patterns, preferences) with category, domain, trigger keywords, confidence. Store in knowledge.db with full metadata for retrieval. |
| 5.6 | **Insight retrieval** | 4h | Before every turn: search insights by semantic similarity + trigger matching + domain. Inject relevant learnings into context. Track recall count and validate insights over time. |
| 5.7 | **AgentFactory (pattern detection)** | 6h | After task completion: find similar past tasks (min 3). Extract common tools, workflow, skills. If pattern strong (>80% similarity, >90% success), trigger agent generation. |
| 5.8 | **Agent code generation** | 6h | Generate specialized agent class: custom state machine, domain-specific tools, workflow steps, system prompt. Inherit from GaiaCodeAgent. Test on sample task before registration. |
| 5.9 | **Agent confidence tracking** | 3h | Track specialist usage: success rate, task types, performance. Confidence increases with successful use. Low-confidence agents get improved or deprecated. |

### Validation

```bash
# Test: Agent auto-generation
# Sessions 1-3: Create CRUD for users, posts, comments (similar tasks)
# Session 4:
gaia code "Create CRUD for products"
# ✅ Agent detects pattern (3+ similar successful tasks)
# ✅ Agent generates FastAPICRUDAgent specialist
# ✅ Agent delegates to specialist via agent_query(specialist="fastapi_crud_agent")
# ✅ Specialist completes in 4 min (was 15 min manual)
# ✅ Specialist confidence: 0.6 → 0.75 after success

# Test: Specialist usage
gaia code "Debug this failing async test"
# ✅ Main agent searches agents.db for "debug" + "async"
# ✅ Finds debugger_agent (confidence: 1.0) or async_debugger_agent (learned)
# ✅ Delegates via agent_query(specialist="debugger_agent")
# ✅ Debugger follows its custom state machine (identify → isolate → fix → verify)

# Test: Specialist recursion
gaia code "Build a production-ready API"
# ✅ Main agent → agent_query(specialist="backend_agent")
#    Backend → agent_query(specialist="security_agent") [finds vulnerability]
#    Security → agent_query(specialist="refactoring_agent") [fixes it]
# ✅ Three-level recursion, all coordinated via shared manifest
```

### Exit Criteria

- Agent has created ≥2 learned specialists from patterns
- Core specialists (7) are used automatically when relevant
- Specialist confidence scores accurately reflect success rates
- Recursive delegation (depth 2-3) works without conflicts

---

## Milestone 6: Memory Defragmentation + Vector Search (Days 16-19)

> **The agent stays reliable as it grows.**

### What

Add FAISS vector search for semantic retrieval. Add defragmentation to keep knowledge clean as it accumulates.

### Why After M5

By this point the agent has accumulated significant data across M1-M5 (hundreds of knowledge entries, dozens of tools, skills). If retrieval quality is degrading, we need defrag. If FTS5 search isn't finding the right results, we need vector search.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 6.1 | **Embedding engine** | 4h | Local embedding model (all-MiniLM-L6-v2 or similar). <10ms per embedding on AMD hardware. |
| 6.2 | **FAISS integration** | 4h | Add vector indices to knowledge.db, tools.db, skills.db. Semantic search in addition to FTS5. |
| 6.3 | **Defragmentation** | 6h | Deduplicate, reconcile contradictions, prune stale entries, consolidate patterns, re-index, verify. |
| 6.4 | **`gaia memory` commands** | 3h | `gaia memory status`, `gaia memory defrag`, `gaia memory archive`. |
| 6.5 | **Auto-defrag trigger** | 2h | When retrieval quality drops below 90% (measured by verification queries), trigger defrag automatically. |

### Validation

```bash
gaia memory status
# ✅ Shows entry counts, staleness, duplicate count, retrieval quality

gaia memory defrag
# ✅ Deduplicates, prunes, consolidates
# ✅ Retrieval quality improves (measured)

# Test: After 50+ sessions of usage
# ✅ Agent is still fast and accurate
# ✅ No contradictory knowledge surfacing
# ✅ Old patterns consolidated into high-confidence entries
```

### Exit Criteria

- Vector search finds more relevant results than FTS5 alone
- Defrag measurably improves retrieval quality
- Agent performance doesn't degrade over time

---

## Milestone 7: Codebase Indexing + Architecture Analysis (Days 19-23)

> **The agent can work with large, existing codebases.**

### What

Index a codebase (symbols, imports, dependencies) and analyze its architecture. This enables the agent to work on real-world projects, not just greenfield.

### Why After M6

Codebase indexing produces a lot of data that goes into the knowledge DB (from M1). It benefits from vector search (from M6) for finding relevant code. And the agent needs to be stable and reliable (M0-M6) before tackling complex existing codebases.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 7.1 | **File walker + symbol extractor** | 6h | Walk all source files, extract classes/functions/imports using tree-sitter or AST. Support Python, TypeScript, JavaScript initially. |
| 7.2 | **Dependency graph** | 4h | Build import graph. Detect circular dependencies. Know what files depend on what. |
| 7.3 | **Index storage** | 3h | Store index in knowledge.db. Incremental updates (only re-index changed files). |
| 7.4 | **`analyze_architecture` tool** | 4h | Agent can query: "What are the main modules?", "What depends on auth.py?", "Are there circular imports?" |
| 7.5 | **`find_issues` tool** | 4h | Detect: circular deps, unused imports, missing tests, inconsistent patterns. |
| 7.6 | **Natively recursive model exploration** | 8h | Explore post-training Qwen3-Coder-30B to be natively recursive (per [RLM paper](https://arxiv.org/abs/2512.24601)). The agent already uses RLM patterns via prompting (M0.6) and `llm_query()` (M1.6) -- this step makes recursion native to the model for even better decomposition quality. |

### Validation

```bash
# Test: Index the GAIA repo itself
gaia code index .
# ✅ Indexes ~100K LoC in <30 seconds
# ✅ Finds all classes, functions, imports
# ✅ Builds dependency graph

gaia code "What are the main modules in this project?"
# ✅ Agent answers from the index, doesn't read every file

gaia code "Find circular dependencies"
# ✅ Agent reports real circular deps (if any)
```

### Exit Criteria

- Agent can index a 100K LoC repo in <30 seconds
- Agent navigates the codebase by querying the index, not reading files randomly
- Architecture analysis produces useful, actionable results

---

## Milestone 8: V2Config + Composability (Days 23-25)

> **The agent becomes configurable across different LLMs and use cases.**

### What

Add V2Config with feature flags. Each capability from M1-M7 can be enabled/disabled. Add profile presets for different use cases and LLM tiers.

### Why After M7

We now have enough features that composability matters. Earlier, there's nothing to compose -- the agent is just one thing. Now we can offer: "coding" profile (all features), "light" profile (basic + quality gates), "minimal" profile (just tools).

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 8.1 | **V2Config dataclass** | 3h | Feature flags for every capability. Defaults based on LLM tier. |
| 8.2 | **Config loading** | 3h | CLI flags → env vars → project config → user config → defaults. |
| 8.3 | **Profile presets** | 2h | `--profile=coding`, `--profile=light`, `--profile=minimal`. |
| 8.4 | **Feature flag checks** | 4h | Each capability checks its flag. Disabled = zero overhead. |

### Validation

```bash
# Minimal profile (small LLM)
gaia code --profile=minimal "Fix the typo in README.md"
# ✅ Works fast, no memory/quality gates overhead

# Full profile (large LLM)
gaia code --profile=coding "Build a REST API with auth and tests"
# ✅ All features active: memory, quality gates, planning, tools
```

### Exit Criteria

- All features can be independently enabled/disabled
- Minimal profile works on 7B models
- Full profile uses all capabilities

---

## Milestone 9: Concurrent Task Threads (Days 25-28)

> **The agent does multiple things in parallel.**

### What

Single agent, multiple task threads. When tasks are independent (e.g., backend + frontend), work on them concurrently.

### Why After M8

Concurrent execution needs: checkpoint (M3) for each thread, knowledge DB (M1) for shared state, quality gates (M2) for each thread, and config (M8) to enable/disable.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 9.1 | **Task thread model** | 6h | Each thread has: its own plan, its own state, shared access to knowledge.db |
| 9.2 | **Thread coordination** | 6h | Main thread orchestrates: assigns tasks, detects conflicts, merges results |
| 9.3 | **Conflict resolution** | 4h | When two threads edit related files, detect and resolve conflicts |

### Validation

```bash
gaia code "Build a full-stack app: React frontend + FastAPI backend"
# ✅ Agent creates 2 threads (frontend, backend)
# ✅ Threads work concurrently
# ✅ Backend API schema shared with frontend thread
# ✅ Final result is coherent
```

---

## Milestone 10: External Benchmarks + Battle-Hardening (Days 28-32)

> **Prove it works. Measure against the best.**

### What

Run Terminal-Bench, SWE-bench, and GaiaCodeBench. Fix failures. Harden against edge cases.

### Why Last

You can only benchmark and harden a complete agent. Running benchmarks too early wastes time on issues that will be fixed by later milestones.

### Deliverables

| # | What | Hours | Details |
|---|------|-------|---------|
| 10.1 | **Terminal-Bench adapter** | 4h | Harbor adapter for Gaia Code |
| 10.2 | **SWE-bench adapter** | 4h | SWE-bench harness integration |
| 10.3 | **GaiaCodeBench tasks** | 8h | Write 100 task definitions (YAML) |
| 10.4 | **Run benchmarks** | 8h | Execute all benchmarks, analyze failures |
| 10.5 | **Fix top 10 failure patterns** | 12h | Address the most common failure modes |
| 10.6 | **Real-world validation** | 8h | RW-1 through RW-6 from eval plan |

### Targets

| Benchmark | Target (local LLM) | Stretch (cloud) |
|-----------|:---:|:---:|
| Terminal-Bench 2.0 | ≥50% | ≥65% |
| SWE-bench Verified | ≥55% | ≥70% |
| GaiaCodeBench | ≥70% | ≥80% |

---

## Future Outlook (Beyond Day 32)

These are capabilities we've designed but deliberately deferred:

| Capability | From Spec | When | Trigger |
|------------|-----------|------|---------|
| LLM Tier Router | F18 | When we have 2+ LLMs working | Currently single-tier is fine |
| Adaptive Prompts (6 layers) | F13 | After M10 benchmarks reveal prompt gaps | Start with 2 layers (base + memory) |
| State Machine (6 states) | F9 | When "plan + execute + review" proves insufficient | M2 has plan + quality gates |
| Full Learning Loop | F11 | After 50+ sessions of real usage | Need real usage patterns first |
| Self-Evaluation (7 dims) | F16 | After quality gates prove insufficient | May never need this |
| Proactive Analysis | F17 | After codebase indexing works well | Extension of M7 |
| 1M+ LoC support | C1 | After 100K LoC works reliably | Scale when needed |
| Multi-day execution | C3 | After checkpoint/resume is proven reliable | Extend gradually |
| CUA tool packs | Tool packs | When building gaia cua | Different project |
| Knowledge tool packs | Tool packs | When building gaia chat | Different project |
| Community tool sharing | Tools DB | When user base exists | Requires ecosystem |
| Full RLM REPL environment | C1, Context | After natively recursive model works (M7.6) | Agent gets full Python REPL where inputs are variables |
| RLM-based fine-tuning pipeline | LLM | After RLM patterns prove value via prompting | Post-train models to be natively recursive |

---

## Summary: The Honest Path

```
┌────────────────────────────────────────────────────────────────────┐
│                RECURSIVE AGENT COMPOSITION ROADMAP                  │
│                                                                     │
│   M0: Prompts + RLM patterns + escalation  (Days 1-2) ← START     │
│   M1: SharedAgentState + agent_query()     (Days 3-5) ← RAC CORE  │
│   M2: Quality gates + escalation ladder    (Days 5-8)             │
│   M3: Checkpoint/resume + audit            (Days 8-10)            │
│   ─── ABOVE = MVP with recursive decomposition ───                 │
│   M4: Agent registry + 7 specialists       (Days 10-13) ← EXPERTS │
│   M5: Agent auto-generation + learning     (Days 13-16) ← GROWTH  │
│   M6: Vector search + defrag               (Days 16-19)           │
│   ─── ABOVE = Self-improving agent. Benchmark. ───                 │
│   M7: Codebase indexing + native RLM       (Days 19-23)           │
│   M8: Composability + config               (Days 23-25)           │
│   M9: Async I/O for concurrent agents      (Days 25-28)           │
│   M10: Benchmarks + hardening              (Days 28-32)           │
│   ─── ABOVE = Production RAC system. Ship. ───                     │
│   Future: Specialists creating specialists, emergent coordination  │
│                                                                     │
│   Total: 32 days to production                                     │
│   RAC foundation at Day 5. Specialists at Day 13. Auto-gen at 16. │
│                                                                     │
│   KEY INNOVATION:                                                  │
│   M1 = SharedAgentState (7 databases, all agents share)           │
│   M1.8 = agent_query() (RLMs extended to full agents)             │
│   M4 = 7 core specialists (domain experts)                         │
│   M5 = AgentFactory (creates specialists from patterns)           │
│                                                                     │
│   This is Recursive Agent Composition (RAC).                      │
│   See: RECURSIVE_AGENT_COMPOSITION.md                             │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Key Principle: Prove Before Building

| Phase | Prove | Then Build |
|-------|-------|------------|
| M0 | "The LLM can code" | Knowledge DB |
| M1 | "Persistent context helps" | Quality gates |
| M2 | "Quality gates improve output" | Checkpoint/resume |
| M3 | "Checkpoint/resume works" | Tools DB |
| M4 | "Organized tools help" | Agent-created tools |
| M5 | "Agent creates useful tools" | Vector search + defrag |
| M6 | "Vector search improves retrieval" | Codebase indexing |

**Never build the next milestone until the current one is proven useful.**

---

*Build → Test → Learn → Repeat.*
*The best agent is the one that works, not the one with the most features.*
