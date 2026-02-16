# GAIA Code: The World's Most Autonomous Coding Agent

**Date**: February 10, 2026
**Version**: 1.2
**Status**: Strategic Specification
**Priority**: CRITICAL -- This is the #1 priority. Everything else depends on this.
**Goal**: Build an autonomous coding agent that exceeds Claude Code (Opus 4.6) in capability, autonomy, and reliability

### Core Design Principles

> **These principles override all feature decisions. When in doubt, choose simplicity.**

1. **Composable** -- Every feature is opt-in. Agent works with 0.6B or 200B. Features enable/disable independently.
2. **Reliable** -- Fewer features, working perfectly > many features, working sometimes. Each feature must have zero-overhead when disabled and verified correctness when enabled.
3. **Quality-focused** -- Agent never declares "done" without proof (tests pass, syntax valid). Quality of output is the north star metric.
4. **Simple** -- If a feature needs >3 integration points, it's too complex. Start with the simplest version that works. Iterate from usage, not speculation.
5. **Not brittle** -- No feature should break another. No cascading failures. Core capabilities (memory, knowledge DB, quality gates) are inherent parts of the agent, not optional add-ons.
6. **Context-lean** -- Never fill the context window. Store knowledge externally, keep only the active working set in context. No conversation compaction. No slowdowns. The agent should be equally fast on step 1 and step 1000.
7. **Time-aware** -- Every action is timestamped. The agent has a sense of time: how long tasks take, how long it's been working, when it last saw a file, how old a memory is. Time is a first-class dimension across the entire framework.

---

## The Context Problem (Why This Architecture Exists)

> **The single biggest usability problem with Claude Code is conversation compaction.** When the context window fills up, Claude Code pauses to compress the conversation. This is slow, lossy, and breaks the user's flow. Gaia Code eliminates this entirely.

### How Claude Code Fails

```
Claude Code's Architecture (context-heavy):

   Context Window (200K tokens)
   ┌──────────────────────────────────────────┐
   │ System prompt                              │
   │ CLAUDE.md instructions                     │
   │ Full file contents (read_file results)     │  ← Everything
   │ All tool call results                      │     lives here
   │ All conversation history                   │
   │ All error messages and debug output        │
   │ All git diffs and status output            │
   └──────────────────────────────────────────┘
              │
              ▼ Context fills up
   ┌──────────────────────────────────────────┐
   │ ⏳ COMPACTING CONVERSATION...             │  ← User waits
   │    Summarizing old messages...            │     Agent slows
   │    Losing details...                      │     Context lost
   │    Breaking flow...                       │     Quality drops
   └──────────────────────────────────────────┘
              │
              ▼ After compaction
   ┌──────────────────────────────────────────┐
   │ System prompt                              │
   │ [Summary of what happened before]          │  ← Lossy summary
   │ Recent conversation                        │     Details gone
   │ (room for ~50K more tokens)               │     Repeat mistakes
   └──────────────────────────────────────────┘
```

**Problems:**
1. **Slow** -- Compaction takes 10-30 seconds. Happens multiple times per complex task.
2. **Lossy** -- Summaries lose critical details (exact error messages, file paths, variable names).
3. **Breaks flow** -- User is interrupted mid-task. Momentum lost.
4. **Degrades quality** -- After compaction, agent may re-read files it already read, repeat mistakes it already fixed, or forget decisions it already made.
5. **Unpredictable** -- User doesn't know when compaction will happen or what will be lost.

### How Gaia Code Solves This

```
Gaia Code's Architecture (context-lean):

   Context Window (lean, never fills up)
   ┌──────────────────────────────────────────┐
   │ System prompt + active state              │
   │ Current task description                  │
   │ Active file snippet (only relevant lines) │  ← Only what's
   │ Last few tool results (summarized)        │     needed NOW
   │ Current conversation turn                 │
   │                                           │
   │          (always <50% full)               │
   └──────────────────────────────────────────┘
              │ queries ▲ stores
              ▼         │
   ┌──────────────────────────────────────────┐
   │         KNOWLEDGE DATABASE                 │
   │                                           │
   │  📁 File Cache                            │
   │     Full contents of every file read      │
   │     Indexed by path, searchable           │
   │                                           │
   │  🔧 Tool Results Store                    │
   │     Every tool call + result              │
   │     Indexed, searchable, never lost       │
   │                                           │
   │  💬 Conversation History                  │
   │     Full conversation (not summarized)    │
   │     Vector-indexed for semantic recall    │
   │                                           │
   │  🧠 Episodic Memory                       │
   │     Session summaries, decisions, errors  │
   │     Searchable across all sessions        │
   │                                           │
   │  📋 Task State                            │
   │     Current plan, completed steps         │
   │     Sub-task results, dependencies        │
   │                                           │
   │  🏗️ Project Manifest                      │
   │     File tree, dependency graph           │
   │     Symbols, imports, test mapping        │
   └──────────────────────────────────────────┘
```

**How it works:**
1. **Store everything externally** -- File contents, tool results, conversation history, decisions all go into the knowledge database (SQLite + FAISS).
2. **Keep context lean** -- Only the current task, active file snippet, and last few tool results stay in the context window.
3. **Query on demand** -- When the agent needs past context (a file it read earlier, a decision it made, an error it saw), it queries the knowledge DB instead of keeping it in context.
4. **Never compact** -- Context window stays under 50% capacity. No compaction ever needed.
5. **Never slow down** -- Agent is equally responsive on step 1 and step 500.

### The Context Management Strategy

```python
class ContextManager:
    """Keeps context window lean by offloading to knowledge DB."""

    MAX_CONTEXT_USAGE = 0.5  # Never exceed 50% of context window

    def before_tool_call(self, tool_name: str, args: dict):
        """Before each tool call, check if context needs trimming."""
        if self.context_usage > self.MAX_CONTEXT_USAGE:
            self._offload_to_knowledge_db()

    def after_tool_result(self, tool_name: str, result: str):
        """After each tool result, store full result in DB, keep summary in context."""
        # Store full result in knowledge DB
        self.knowledge_db.store_tool_result(tool_name, result)

        # Keep only a summary in context
        if len(result) > 500:
            return self._summarize_result(tool_name, result)
        return result

    def on_file_read(self, path: str, content: str):
        """When a file is read, cache it externally."""
        self.knowledge_db.cache_file(path, content)
        # Only return relevant lines to context, not full file
        return self._extract_relevant_lines(path, content)

    def recall(self, query: str) -> str:
        """Agent explicitly recalls past context from knowledge DB."""
        return self.knowledge_db.semantic_search(query, top_k=5)

    def _offload_to_knowledge_db(self):
        """Move old conversation turns to knowledge DB."""
        old_turns = self.conversation[:-3]  # Keep last 3 turns
        for turn in old_turns:
            self.knowledge_db.store_conversation_turn(turn)
        self.conversation = self.conversation[-3:]
```

### Recursive Language Models (RLMs): The Execution Paradigm

> **Reference:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab -- MIT CSAIL, 2025)

The context-lean architecture above uses a **store-and-recall** pattern: offload to knowledge DB, query on demand. This works. But a more powerful paradigm exists: **Recursive Language Models (RLMs)**.

**The core idea:** Instead of storing context externally and retrieving it, the model **treats the input as a variable in a Python REPL** and writes code to programmatically decompose, filter, and recursively call itself on smaller chunks.

```
TRADITIONAL LLM (Claude Code):
  Entire codebase → stuff into 200K context → pray it fits

STORE-AND-RECALL (our Knowledge DB approach):
  Entire codebase → store in SQLite/FAISS → recall relevant pieces on demand

RECURSIVE LM (RLM approach):
  Entire codebase → set as variable P in REPL → model writes code to:
    1. Filter P with regex to find relevant files
    2. Chunk relevant files into sections
    3. Recursively call llm_query() on each section
    4. Synthesize results across all recursive calls
    5. CONTEXT NEVER FILLS UP -- each recursive call has fresh context
```

**Why this is transformative for GAIA Code:**

RLM-Qwen3-8B outperforms base Qwen3-8B by **28.3%** and approaches GPT-5 on long-context tasks. Applied to Qwen3-Coder-30B, this could give us **frontier-level performance from a local model** -- drastically reducing the need for cloud escalation.

**How RLMs integrate with our architecture:**

```
┌────────────────────────────────────────────────────────────────┐
│                  GAIA CODE DUAL-LAYER ARCHITECTURE               │
│                                                                  │
│  LAYER 1: RLM (within-session execution)                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Agent receives task                                        │  │
│  │    → Decomposes into subtasks (writes Python code)         │  │
│  │    → Each subtask: llm_query() with ONLY relevant context  │  │
│  │    → Recursive: subtask can decompose further              │  │
│  │    → Results synthesized via code (Python variables)       │  │
│  │    → Context window NEVER fills up                         │  │
│  │                                                             │  │
│  │  Handles: decomposition, context, long inputs, synthesis   │  │
│  └────────────────────────────────────────────────────────────┘  │
│                            │                                     │
│                    persists to ▼ recalls from                    │
│                                                                  │
│  LAYER 2: Knowledge DB (cross-session persistence)              │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Insights, preferences, error-fix patterns                  │  │
│  │  Skills, tools, session summaries                           │  │
│  │  Project conventions, file cache                            │  │
│  │                                                             │  │
│  │  Handles: memory, learning, cross-session continuity       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  KEY: RLMs solve WITHIN-SESSION context. Knowledge DB solves    │
│       ACROSS-SESSION memory. Together = unlimited context +     │
│       permanent memory. Neither alone is sufficient.            │
└────────────────────────────────────────────────────────────────┘
```

**Concrete examples of RLM execution in GAIA Code:**

```python
# Example 1: Analyzing a large codebase (C1)
# Instead of building a tree-sitter index, the agent writes:

def analyze_codebase(path: str) -> str:
    """Agent writes this code in the REPL to recursively analyze."""
    files = list_files(path)  # P is the file listing

    # Chunk by directory (module-level analysis)
    modules = group_by_directory(files)

    module_analyses = []
    for module_name, module_files in modules.items():
        # RECURSIVE CALL: analyze each module with fresh context
        analysis = llm_query(
            f"Analyze this module's architecture:\n{read_files(module_files)}"
        )
        module_analyses.append(f"{module_name}: {analysis}")

    # RECURSIVE CALL: synthesize all module analyses
    return llm_query(
        f"Synthesize these module analyses into a full architecture report:\n"
        + "\n".join(module_analyses)
    )


# Example 2: Debugging across multiple files
def debug_test_failure(test_output: str) -> str:
    """Agent recursively traces the failure through the codebase."""
    # Extract failing test and error
    error_info = llm_query(f"Extract the failing test and error:\n{test_output}")

    # Find relevant source files
    relevant_files = grep(error_info.file_pattern)

    # RECURSIVE: analyze each relevant file for the bug
    file_analyses = []
    for f in relevant_files:
        analysis = llm_query(
            f"Given error '{error_info.message}', analyze this file for the bug:\n"
            f"{read_file(f)}"
        )
        file_analyses.append(analysis)

    # RECURSIVE: synthesize into root cause + fix
    return llm_query(
        f"Given these file analyses, identify root cause and write the fix:\n"
        + "\n".join(file_analyses)
    )
```

**Three RLM patterns the agent uses:**

| Pattern | When | How |
|---------|------|-----|
| **Filter** | Large input, only some parts relevant | Write regex/grep to extract relevant chunks, query only those |
| **Chunk & Recurse** | Input too large for one call | Split into N chunks, recursively analyze each, synthesize |
| **Variable Stitch** | Output too large for one response | Build output incrementally in Python variables, assemble at end |

**Impact on our architecture concerns:**

| Concern | Without RLMs | With RLMs |
|---------|-------------|-----------|
| LLM Quality (Concern 1) | Need cloud escalation for complex tasks | RLM boosts local model by ~28%+, reduces cloud need |
| Context overflow | Knowledge DB + recall tool | Recursive decomposition, each call has fresh context |
| Large codebase (C1) | Build tree-sitter index, query it | Recursively decompose and analyze, no index needed for basic analysis |
| Checkpoint = compaction (Concern 2) | State-based resume | Each recursive call is naturally isolated, no state to checkpoint within a recursion |

**Implementation plan:**

| When | What | Details |
|------|------|---------|
| **M0** | RLM-aware prompting | System prompt teaches the agent to use recursive decomposition patterns |
| **M1** | `llm_query()` tool | Tool that lets the agent recursively call itself with a sub-prompt and fresh context |
| **M7** | Natively recursive model | Fine-tune or post-train Qwen3-Coder-30B to be natively recursive (following the paper's approach) |
| **Future** | Full RLM REPL | Agent has full Python REPL where inputs are variables, outputs are code + recursive calls |

**Key insight:** We don't need to wait for M7 to get RLM benefits. At M0, we can teach the agent the recursive decomposition PATTERN through prompting. At M1, the `llm_query()` tool gives it the MECHANISM. The natively recursive model at M7 is optimization, not a prerequisite.

### Why This Is a 5th Pillar

The original 4 pillars assumed the context window is the primary working space. But **context management is foundational** -- it determines how well everything else works:

| Pillar | Without Context-Lean | With Context-Lean + RLMs |
|--------|---------------------|--------------------------|
| Continuous Execution | Slows down every ~30 steps for compaction | Runs at constant speed; recursive calls have fresh context |
| Quality Gates | After compaction, may forget which gates passed | Full gate history in knowledge DB; each recursive call checks independently |
| Checkpoint/Resume | Checkpoint needed mainly because context fills up | Checkpoints for crash recovery only; RLMs prevent context overflow |
| Persistent Memory | Memory injected into already-full context window | Memory in external DB; RLMs handle current-session context |

**Updated pillars:**

```
THE FIVE PILLARS (build these first, build them right)

1. CONTEXT-LEAN ARCHITECTURE -- Never fill the context window.
     Knowledge DB for cross-session persistence.
     RLMs for within-session recursive decomposition.
2. CONTINUOUS EXECUTION       -- Run until done (no step limit)
3. QUALITY GATES              -- Verify output works (tests pass)
4. CHECKPOINT/RESUME          -- Survive crashes and interruptions
5. PERSISTENT MEMORY          -- Learn across sessions
```

Context-lean is Pillar 1 because it enables all others to work efficiently. RLMs make it fundamentally more powerful.

## Table of Contents

1. [Priority Matrix](#priority-matrix-what-to-build-first)
2. [Strategic Thesis](#strategic-thesis)
3. [Core Capabilities (What the Agent MUST Do)](#core-capabilities-what-the-agent-must-do)
4. [The Context Problem (Why This Architecture Exists)](#the-context-problem-why-this-architecture-exists)
5. [What We Already Have](#what-we-already-have)
6. [Prerequisites: Core Infrastructure (Build FIRST)](#prerequisites-core-infrastructure-build-first)
7. [Composability Across LLM Tiers](#composability-across-llm-tiers)
8. [Competitive Landscape](#competitive-landscape)
9. [Claude Code: Capabilities & Limitations](#claude-code-capabilities--limitations)
10. [Core Features: What Gaia Code Must Have](#core-features-what-gaia-code-must-have)
11. [Differentiators: How Gaia Code Exceeds Claude Code](#differentiators-how-gaia-code-exceeds-claude-code)
12. [Architecture Overview](#architecture-overview)
13. [Feature Deep-Dives](#feature-deep-dives)
14. [Implementation Phases](#implementation-phases)
15. [The Bootstrap Effect](#the-bootstrap-effect)
16. [Validation Criteria](#validation-criteria)
17. [Design Philosophy: Simplicity Over Complexity](#design-philosophy-simplicity-over-complexity)
18. [Knowledge Retrieval Architecture](#knowledge-retrieval-architecture)
19. [Top Concerns & Novel Solutions](#top-concerns--novel-solutions)
20. [Summary](#summary)
21. [Risks & Mitigations](#risks--mitigations)

---

## Priority Matrix: What to Build First

> **This document is comprehensive by design. This section tells you what matters most given limited time.**

### Tier 1: Build Now (Days 1-10) -- The MVP

These features deliver 80% of the value. If we stop after Tier 1, we have a useful agent.

| Priority | Feature | Milestone | Impact | Why Now |
|:---:|---------|:---:|:---:|---------|
| **1** | **System prompt engineering + RLM patterns** | M0 | CRITICAL | Everything depends on prompt quality. Teach recursive decomposition from day 1. The most important 8 hours. |
| **2** | **SharedAgentState (7 databases)** | M1 | CRITICAL | Foundation for RAC. All agents share: memory.db, knowledge.db, manifest, plan, call stack, message queue, registries. |
| **3** | **agent_query() tool** | M1 | CRITICAL | The RAC mechanism. Spawn sub-agents recursively with tools, verification, shared state. This is RLMs extended to full agents. |
| **4** | **Quality gates + escalation ladder** | M2 | CRITICAL | Agent proves output works. Escalate: retry → decompose → cloud → ask user. Never infinite loops. |
| **5** | **Plan creation + progress tracking** | M2 | HIGH | MasterPlan shared across agents. Each agent sees overall goal, completed tasks, active tasks. |
| **6** | **Continuous execution (no step limit)** | M2 | HIGH | Agent runs until verified complete, not until max_steps. |
| **7** | **Checkpoint/resume (state-based)** | M3 | HIGH | Resume from plan + manifest + gates, not prose summary. Zero information loss. |
| **8** | **Audit log + time awareness** | M3 | MEDIUM | Full transparency. Every action timestamped and tracked. |

**After Tier 1: Recursive agent system with shared state, no context limits, verified output, cross-session memory. This is RAC working.**

### Tier 2: Build Next (Days 10-19) -- The Differentiator

These features separate us from every other agent.

| Priority | Feature | Milestone | Impact | Why Next |
|:---:|---------|:---:|:---:|---------|
| **9** | **Agent registry (agents.db)** | M4 | CRITICAL | Registry for specialists. Semantic search, confidence tracking. Foundation for recursive specialization. |
| **10** | **7 core specialized agents** | M4 | CRITICAL | Debugger, Security, Refactoring, Testing, Documentation, Performance, Architecture. Domain experts with custom state machines. |
| **11** | **Specialist selection + delegation** | M4 | HIGH | agent_query(specialist="debugger_agent"). Auto-select best specialist via semantic search. Track specialist usage. |
| **12** | **Tools DB + organized tool packs** | M4 | HIGH | 78 tools organized by domain. Specialists declare which packs they use. Dynamic loading. |
| **13** | **AgentFactory (pattern detection)** | M5 | HIGH | After 3+ similar successful tasks, generate new specialist. Example: FastAPICRUDAgent, AsyncDebuggerAgent. |
| **14** | **Agent code generation + testing** | M5 | HIGH | Generate specialist class: state machine, tools, workflow, system prompt. Test before registration. |
| **15** | **Insight generation + structured learning** | M5 | HIGH | Extract insights with category, domain, triggers, confidence. Retrieval by semantic + triggers + domain. |
| **16** | **Skills DB + ToolBuilder** | M5 | MEDIUM | Multi-step workflows and agent-created tools. Foundation for specialist capabilities. |
| **17** | **Vector search (FAISS) + defrag** | M6 | MEDIUM | Semantic retrieval for insights, tools, skills, agents. Memory defrag keeps quality high. |

**After Tier 2: Recursive agent system with specialists, auto-generation, compound learning. Specialists coordinate via shared state. This is RAC at full power.**

### Tier 3: Build Later (Days 19-32) -- The Superpower

These features make the agent truly exceptional but require the foundation to be solid.

| Priority | Feature | Milestone | Impact | Why Later |
|:---:|---------|:---:|:---:|---------|
| **14** | **Codebase indexing (tree-sitter)** | M7 | MEDIUM | Understand large repos. Needs vector search. |
| **15** | **V2Config + composability** | M8 | LOW | Feature flags. Useful when we have many features. |
| **16** | **Concurrent task threads** | M9 | MEDIUM | Parallel work. Needs everything above to be solid. |
| **17** | **External benchmarks** | M10 | HIGH | Prove it works. Meaningful only with a complete agent. |

### Tier 4: Future -- Build When Needed

These are designed but deliberately deferred. Build only when proven necessary.

| Feature | Trigger to Build |
|---------|-----------------|
| LLM Tier Router | When we have 2+ LLMs working well |
| Adaptive prompts (6 layers) | After benchmarks reveal prompt gaps (start with 2 layers) |
| Self-evaluation (7 dimensions) | If quality gates prove insufficient |
| Proactive analysis | After codebase indexing works well |
| 1M+ LoC support | After 100K LoC works reliably |
| Multi-day autonomous execution | After checkpoint/resume is battle-tested |
| Learning loop (full 6-step) | After 50+ sessions of real usage data |

### The 80/20 Rule (Updated for RAC)

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                    │
│   80% of agent value comes from 20% of features:                 │
│                                                                    │
│   1. Great prompts + RLM patterns    (M0)  ← Hours 1-21          │
│   2. SharedAgentState (7 databases)  (M1)  ← Hours 21-59         │
│   3. agent_query() recursive tool    (M1)  ← included above      │
│   4. Quality gates + escalation      (M2)  ← Hours 59-79         │
│   5. Checkpoint/resume (state-based) (M3)  ← Hours 79-100        │
│                                                                    │
│   The paradigm shift:                                             │
│   #2 + #3 = Recursive Agent Composition (RAC)                    │
│   Without RAC: monolithic agent with fixed capabilities          │
│   With RAC: recursive specialists with exponential growth        │
│                                                                    │
│   Everything else builds on RAC foundation.                      │
│   Ship RAC first (M0-M1). Ship it well.                          │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

**Cross-reference:** See `GAIA_CODE_MILESTONES.md` for detailed implementation plan with hours and validation criteria per milestone.

---

## Strategic Thesis

**If `gaia code` is a better autonomous coding agent than Claude Code, it can build all other GAIA features (CUA, Chat, etc.) faster and better than any human or existing tool.**

This is a bootstrapping strategy:

```
Phase 1: Build gaia code (manually, with Claude Code assist)
Phase 2: Use gaia code to build gaia cua (computer use agent)
Phase 3: Use gaia code to build gaia chat (knowledge assistant)
Phase 4: Use gaia code to build everything else
```

Each phase produces a better tool that accelerates the next phase. The compound effect means:
- Phase 1: 1x speed (human + Claude Code)
- Phase 2: 3-5x speed (gaia code is better than Claude Code)
- Phase 3: 5-10x speed (gaia code has learned from Phase 2)
- Phase 4: 10x+ speed (gaia code has learned patterns, has custom tools, has memory)

**This is why `gaia code` is the #1 priority. Everything else follows.**

---

## Core Capabilities (What the Agent MUST Do)

> **These are non-negotiable. If the agent can't do all of these, it's not state-of-the-art.**

### C1. Analyze Large Codebases (1M+ Lines of Code)

The agent must work with real-world codebases, not toy projects. Enterprise repos have millions of lines of code across thousands of files. The agent can't read them all into context -- it must **index, search, and understand at scale**.

**How it works:**

```
Step 1: INDEX the codebase (once, on first access)
  - Walk all files, extract symbols (classes, functions, imports)
  - Build dependency graph (who imports whom)
  - Build call graph (who calls whom)
  - Store in project manifest (SQLite + FAISS)
  - Time: ~2-5 minutes for 1M LoC (runs in background)

Step 2: NAVIGATE on demand
  - Agent never reads the whole codebase
  - Agent queries the index: "find all classes that implement AuthProvider"
  - Agent follows dependency chains: "what depends on database.py?"
  - Agent searches semantically: "find code related to payment processing"

Step 3: UNDERSTAND structure
  - Agent knows the architecture without reading every file
  - "This is a Django project with 12 apps, 45 models, 200 views"
  - "The auth system uses JWT with custom middleware in auth/middleware.py"
  - "There are 3 circular dependencies: A↔B, C↔D, E↔F"
```

```python
class CodebaseIndex:
    """Indexes a codebase for fast navigation and understanding."""

    def __init__(self, root: str):
        self.root = root
        self.files: dict[str, FileInfo] = {}
        self.symbols: dict[str, SymbolInfo] = {}  # name → definition location
        self.imports: dict[str, list[str]] = {}    # file → imported modules
        self.dependents: dict[str, list[str]] = {} # file → files that import it
        self.call_graph: dict[str, list[str]] = {} # function → functions it calls

    def index(self):
        """Build the full index. Runs once, updates incrementally."""
        for path in self._walk_source_files():
            self._index_file(path)
        self._build_dependency_graph()
        self._detect_circular_deps()
        self._persist()  # Save to knowledge DB

    def find_symbol(self, name: str) -> list[SymbolInfo]:
        """Find where a symbol is defined and used."""
        ...

    def get_dependents(self, path: str) -> list[str]:
        """What files would break if I change this file?"""
        ...

    def get_architecture_summary(self) -> str:
        """High-level summary of the project architecture."""
        ...

    def find_related_code(self, query: str) -> list[CodeSnippet]:
        """Semantic search across the codebase."""
        ...

    def detect_issues(self) -> list[Issue]:
        """Find architectural issues: circular deps, unused code, etc."""
        ...
```

**Key constraint:** The agent NEVER tries to read 1M lines into context. It reads the index (small), then reads specific files on demand. The index lives in the knowledge DB and persists across sessions.

**Scale targets:**

| Repo Size | Index Time | Query Time | Memory |
|-----------|-----------|-----------|--------|
| 10K LoC | <5 sec | <100ms | <50MB |
| 100K LoC | <30 sec | <100ms | <200MB |
| 1M LoC | <5 min | <200ms | <1GB |
| 10M LoC | <30 min | <500ms | <5GB |

### C2. Deep Dependency & Architecture Analysis

The agent must understand a codebase at every level -- from high-level architecture down to individual bugs. This is not just "grep for imports." This is understanding the **system**.

**Levels of analysis:**

```
LEVEL 1: Architecture
  - What are the major modules/services?
  - How do they communicate? (REST, gRPC, message queue, direct import)
  - What are the system boundaries?
  - Where are the anti-patterns? (god classes, circular deps, tight coupling)

LEVEL 2: Module Dependencies
  - Import graph: who depends on whom?
  - Dependency direction: are dependencies flowing the right way?
  - Circular dependencies: where are they and how to break them?
  - Unused dependencies: what's installed but never imported?

LEVEL 3: API Contracts
  - What APIs are exposed? (REST endpoints, function signatures, class interfaces)
  - Are contracts consistent? (same error format everywhere?)
  - Are there breaking changes between versions?
  - Are there undocumented APIs?

LEVEL 4: Data Flow
  - How does data flow through the system?
  - Where are the transformations?
  - Where is validation missing?
  - Where could data be corrupted?

LEVEL 5: Bug Patterns
  - Common error patterns (unchecked nulls, race conditions, resource leaks)
  - Security vulnerabilities (injection, auth bypass, data exposure)
  - Performance issues (N+1 queries, unnecessary allocations, blocking I/O)
  - Test coverage gaps (untested paths, missing edge cases)
```

```python
@tool
def analyze_architecture(scope: str = "full") -> str:
    """Analyze the codebase architecture at the requested level.

    Args:
        scope: "full" | "dependencies" | "apis" | "security" | "performance"
    """
    index = knowledge_db.get_codebase_index()

    if scope == "full":
        return "\n".join([
            index.get_architecture_summary(),
            index.get_dependency_report(),
            index.get_api_report(),
            index.get_issue_report(),
        ])
    elif scope == "dependencies":
        return index.get_dependency_report()
    # ... etc


@tool
def find_issues(category: str = None, severity: str = None) -> str:
    """Find issues in the codebase.

    Args:
        category: "circular_deps" | "unused_code" | "security" | "performance" | "test_gaps"
        severity: "critical" | "warning" | "info"
    """
    index = knowledge_db.get_codebase_index()
    issues = index.detect_issues(category=category, severity=severity)
    return format_issues(issues)
```

### C3. Autonomous Multi-Day Execution

The agent must be able to work for **hours or days** on complex tasks without supervision. This is the ultimate test of continuous execution, checkpoint/resume, and self-validation.

**What this looks like:**

```
Monday 9:00 AM:
  User: "Build a complete microservices platform with:
    - User service (auth, profiles)
    - Product service (catalog, inventory)
    - Order service (cart, checkout, payments)
    - API gateway
    - Message queue between services
    - Docker compose for local development
    - Kubernetes manifests for deployment
    - Full test suite for each service
    - CI/CD pipeline
    - API documentation"

Monday 9:01 AM:
  Agent creates plan (25 tasks, estimated 6 hours)
  Agent starts working

Monday 12:00 PM (3 hours in):
  Agent has completed: User service, Product service, API gateway
  Agent has checkpointed 15 times
  Agent has resumed from 2 context windows
  User checks progress dashboard: 12/25 tasks complete (48%)

Monday 3:00 PM (6 hours in):
  Agent has completed: All services, Docker compose, tests
  Agent is working on: Kubernetes manifests, CI/CD
  User checks: 20/25 tasks complete (80%)

Monday 5:00 PM (8 hours in):
  Agent: "All 25 tasks complete. Summary:
    - 6 services created (47 files, 8,200 LoC)
    - 312 tests, all passing
    - Docker compose working (verified with docker-compose up)
    - Kubernetes manifests validated (kubectl dry-run)
    - CI/CD pipeline: GitHub Actions with test → build → deploy
    - API docs generated (OpenAPI 3.0)

    Quality gates: ALL PASSED
    - Syntax: ✅
    - Lint: ✅
    - Tests: 312/312 ✅
    - Security: ✅ (no secrets, no injection)
    - Build: ✅ (all Docker images build)

    Ready for your review."
```

**What makes this work:**

| Requirement | How It's Achieved |
|-------------|-------------------|
| Runs for hours | Continuous execution engine (no step limit) |
| Doesn't lose context | Knowledge DB (never fills context window) |
| Survives crashes | Checkpoint/resume (every 10 steps) |
| Maintains quality | Quality gates run after every sub-task |
| Self-corrects | Debug loop: test fails → diagnose → fix → re-test |
| Stays on track | Plan with progress tracking, re-planning when needed |
| Doesn't drift | Original goal stays pinned, every action ties back to it |

**Self-validation loop (runs continuously):**

```python
class AutonomousExecutor:
    """Runs until task is verified complete. No supervision needed."""

    def execute(self, task: str):
        plan = self.create_plan(task)
        self.knowledge_db.store_plan(plan)

        while not plan.is_complete():
            subtask = plan.get_next_task()

            # Work on the subtask
            self.work_on(subtask)

            # Self-check: did the subtask actually succeed?
            validation = self.validate_subtask(subtask)
            if not validation.passed:
                # Don't just retry -- diagnose and fix
                diagnosis = self.diagnose_failure(validation)
                self.apply_fix(diagnosis)
                continue  # Re-validate

            plan.complete_task(subtask.id, result=validation.summary)

            # Periodic self-checks
            if plan.progress % 0.25 == 0:  # Every 25%
                self.run_full_quality_gates()
                self.verify_goal_alignment(task, plan)

            # Checkpoint
            self.checkpoint()

        # Final validation
        self.run_full_quality_gates()
        self.generate_summary(plan)
```

### C4. Dynamic Replanning & Transparency

The agent must adapt when reality diverges from the plan. And every action must be visible and auditable.

**Dynamic replanning:**

```
Original plan: 7 tasks
  [✅] 1. Create project structure
  [✅] 2. Define database models
  [🔄] 3. Implement auth endpoints
  [  ] 4. Implement CRUD endpoints
  [  ] 5. Write tests
  [  ] 6. Add Docker support
  [  ] 7. Final validation

During task 3, agent discovers: "OAuth2 library requires async.
Need to refactor the entire app to use async/await."

Agent replans:
  [✅] 1. Create project structure
  [✅] 2. Define database models
  [🔄] 3. Refactor to async/await (NEW -- discovered dependency)
  [  ] 4. Implement auth endpoints (was task 3, moved)
  [  ] 5. Implement CRUD endpoints
  [  ] 6. Write tests
  [  ] 7. Add Docker support
  [  ] 8. Final validation

Agent communicates: "I discovered that OAuth2 requires async.
I'm refactoring the project to use async/await first.
This adds ~30 minutes but ensures auth works correctly.
Updated plan: 8 tasks instead of 7. Progress: 2/8 (25%)."
```

**Audit log (every action recorded):**

```python
class AuditLog:
    """Complete record of every agent action. Persisted to disk."""

    def log_action(self, action: Action):
        entry = AuditEntry(
            timestamp=datetime.now(),
            action_type=action.type,        # "file_write", "test_run", "plan_update", etc.
            description=action.description,  # Human-readable description
            details=action.details,          # Full details (file content, test output, etc.)
            task_id=action.task_id,          # Which task this relates to
            plan_version=self.plan.version,  # Plan version at time of action
            context_usage=self.context_usage, # How full is the context window
            step_number=self.step_count,     # Global step counter
        )
        self.knowledge_db.store_audit_entry(entry)

    def get_log(self, since: datetime = None, task_id: str = None,
                action_type: str = None) -> list[AuditEntry]:
        """Query the audit log."""
        ...

    def generate_report(self) -> str:
        """Generate human-readable audit report."""
        ...
```

**Audit log format (stored in knowledge DB, viewable by user):**

```
═══════════════════════════════════════════════════════════
                    GAIA CODE AUDIT LOG
                    Session: 2026-02-10-001
                    Task: "Build REST API with auth"
═══════════════════════════════════════════════════════════

[09:00:01] PLAN_CREATE    Created plan with 7 tasks
[09:00:03] FILE_WRITE     Created src/main.py (FastAPI app skeleton)
[09:00:05] FILE_WRITE     Created src/models.py (User, Post models)
[09:00:08] QUALITY_GATE   Syntax check: PASS (2 files)
[09:00:10] FILE_WRITE     Created src/auth.py (JWT auth endpoints)
[09:00:15] TEST_RUN       pytest src/tests/ -- 3/5 PASS, 2 FAIL
[09:00:16] DIAGNOSIS      test_login_invalid: AssertionError, wrong status code
[09:00:18] FILE_EDIT      Fixed auth.py:45 -- return 401, not 400
[09:00:20] TEST_RUN       pytest src/tests/ -- 5/5 PASS ✅
[09:00:22] PLAN_UPDATE    Task 3 complete. Starting task 4.
[09:00:25] REPLAN         Discovered OAuth2 needs async. Added task 3.5.
[09:00:26] USER_UPDATE    "Refactoring to async/await. ETA +30 min."
...
[09:45:00] QUALITY_GATE   Final: Syntax ✅ Lint ✅ Tests ✅ Security ✅
[09:45:01] COMPLETE       All 8 tasks done. 47 files, 312 tests passing.

Summary: 45 minutes, 8 tasks, 47 files, 312 tests, 2 replans, 0 unresolved errors
═══════════════════════════════════════════════════════════
```

**User can check progress at any time:**

```bash
# Real-time progress
gaia code status
  Task: "Build REST API with auth"
  Progress: 5/8 tasks (62%)
  Current: Implementing CRUD endpoints
  Last action: 12 seconds ago
  Errors fixed: 3
  Tests passing: 187/187

# Full audit log
gaia code audit
  [shows full log]

# Specific query
gaia code audit --since "30 minutes ago" --type errors
  [shows only error-related entries from last 30 min]
```

### C5. Time Awareness

Code agents today have no sense of time. They don't know how long they've been working, how long a task took, or when they last touched a file. **Gaia Code treats time as a first-class dimension.**

This is a core framework capability -- it applies to all GAIA agents, not just the code agent.

**What time awareness enables:**

```
Agent knows:
  - "I've been working on this task for 47 minutes"
  - "I last modified auth.py 12 minutes ago"
  - "This file was read 3 hours ago -- it might have changed"
  - "The user hasn't responded in 5 minutes"
  - "My average time per subtask is 8 minutes; this one is taking 22 minutes (anomaly)"
  - "I learned this pattern 2 weeks ago (confidence still high)"
  - "This memory is 3 months old (confidence decaying)"
```

**Implementation:**

```python
class TimeAwareness:
    """Gives the agent a sense of time across all operations."""

    def __init__(self):
        self.session_start = datetime.now()
        self.task_start: datetime = None
        self.step_times: list[float] = []  # Duration of each step in seconds

    @property
    def session_duration(self) -> timedelta:
        """How long has this session been running?"""
        return datetime.now() - self.session_start

    @property
    def task_duration(self) -> timedelta:
        """How long has the current task been running?"""
        if self.task_start:
            return datetime.now() - self.task_start
        return timedelta(0)

    @property
    def avg_step_time(self) -> float:
        """Average seconds per step."""
        if self.step_times:
            return sum(self.step_times) / len(self.step_times)
        return 0

    def estimate_remaining(self, tasks_remaining: int) -> timedelta:
        """Estimate time to completion based on historical pace."""
        return timedelta(seconds=self.avg_step_time * tasks_remaining)

    def file_age(self, path: str) -> timedelta:
        """How long since this file was last read/modified by the agent?"""
        last_access = self.knowledge_db.get_file_last_access(path)
        if last_access:
            return datetime.now() - last_access
        return None  # Never accessed

    def is_stale(self, path: str, threshold_minutes: int = 30) -> bool:
        """Has enough time passed that this file should be re-read?"""
        age = self.file_age(path)
        return age and age > timedelta(minutes=threshold_minutes)
```

**Every audit log entry, every knowledge DB entry, every tool result is timestamped.** This allows the agent to:
- Estimate completion times ("Based on my pace, 5 tasks remaining = ~40 minutes")
- Detect anomalies ("This step is taking 3x longer than average -- might be stuck")
- Prioritize fresh knowledge over stale knowledge
- Report elapsed time to the user ("Completed in 47 minutes")

### C6. Self-Extending Tool Creation

The agent must be able to **build its own tools** by writing code. When the agent detects a repeated pattern or needs a capability it doesn't have, it writes a new tool, tests it, and adds it to its own tool registry.

**This is not academic. It's practical:** If the agent needs to parse a specific log format, it writes a parser tool. If it repeatedly creates FastAPI routes, it writes a route generator tool. The agent expands its own capabilities.

**How it works:**

```
Step 1: DETECT a repeated pattern or missing capability
  Agent notices: "I've created 5 FastAPI routes with the same pattern:
    - Model, schema, CRUD endpoints, tests"

Step 2: WRITE the tool as Python code
  Agent creates: create_fastapi_resource(name, fields, auth_required)
  The tool generates: model.py, schema.py, routes.py, test_routes.py

Step 3: VALIDATE the tool
  - Syntax check (ast.parse)
  - No dangerous operations (no eval, exec, subprocess with shell=True)
  - Run the tool on a test case
  - Verify output is correct

Step 4: REGISTER the tool
  - Add to the agent's tool registry
  - Store in skill library (~/.gaia/skills/)
  - Available in future sessions

Step 5: USE the tool
  - Next time: "Create a FastAPI resource for 'Comment' with fields..."
  - Agent uses its own tool instead of writing boilerplate
```

```python
class ToolBuilder:
    """Agent builds new tools by writing Python code."""

    def create_tool(self, name: str, description: str, code: str) -> bool:
        """Create a new tool from generated code.

        Args:
            name: Tool name (e.g., "create_fastapi_resource")
            description: What the tool does
            code: Python source code for the tool function
        """
        # 1. Validate the code is safe
        if not self._validate_safety(code):
            return False

        # 2. Syntax check
        try:
            ast.parse(code)
        except SyntaxError:
            return False

        # 3. Write to skills directory
        skill_path = Path.home() / ".gaia" / "skills" / f"{name}.py"
        skill_path.write_text(code)

        # 4. Test the tool
        test_result = self._run_tool_test(skill_path)
        if not test_result.passed:
            skill_path.unlink()  # Remove failed tool
            return False

        # 5. Register in tool registry
        self.tool_registry.register_from_file(skill_path, name, description)
        self.knowledge_db.store_skill(name, description, code)

        return True

    def _validate_safety(self, code: str) -> bool:
        """Ensure generated tool code is safe."""
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # No eval/exec
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("eval", "exec", "compile"):
                    return False
            # No subprocess with shell=True
            # No os.system
            # No open() without explicit mode
            # ... etc
        return True

    def list_skills(self) -> list[SkillInfo]:
        """List all custom tools the agent has built."""
        return self.knowledge_db.get_skills()
```

**User can see and manage custom tools:**

```bash
# List agent-created tools
gaia code skills
  create_fastapi_resource  -- Create FastAPI CRUD resource (created 3 days ago, used 12 times)
  parse_nginx_log          -- Parse nginx access log format (created 1 week ago, used 4 times)
  scaffold_pytest_suite    -- Generate pytest suite for a module (created 2 days ago, used 8 times)

# Remove a tool
gaia code skills --remove parse_nginx_log

# Export tools for sharing
gaia code skills --export skills.zip
```

**Tools DB: An Infinite, Searchable Repository**

The agent doesn't just keep tools in memory -- it stores them in a **searchable Tools DB** that can grow infinitely. The agent queries this database when it needs a capability, finding the right tool by semantic search.

```
~/.gaia/tools/
├── tools.db              # SQLite: tool metadata, usage stats, descriptions
├── tools_vectors.faiss   # FAISS: semantic search over tool descriptions
├── core/                 # Pre-installed core tools (ship with GAIA)
│   ├── github.py         # gh clone, gh pr create, gh issue, gh api
│   ├── git.py            # git add, commit, push, branch, merge, rebase, log
│   ├── bash.py           # Run commands, capture output, background processes
│   ├── powershell.py     # PowerShell commands (Windows-native)
│   ├── file_io.py        # read, write, edit, glob, grep, search
│   ├── search.py         # Semantic search, ripgrep, find symbols
│   ├── python_tools.py   # pytest, pip, venv, black, isort, mypy
│   ├── node_tools.py     # npm, npx, tsc, eslint, prettier
│   ├── docker_tools.py   # docker build, run, compose, logs
│   ├── http_tools.py     # curl, wget, API calls, JSON parsing
│   ├── database_tools.py # SQLite queries, PostgreSQL via psql
│   └── system_tools.py   # Process management, env vars, filesystem ops
├── learned/              # Tools the agent created itself
│   ├── create_fastapi_resource.py
│   ├── parse_nginx_log.py
│   └── scaffold_pytest_suite.py
└── community/            # Tools imported from shared library (future)
```

**Core tools (pre-installed, always available):**

| Category | Tools | Examples |
|----------|-------|---------|
| **GitHub** | 8+ | `gh_clone`, `gh_pr_create`, `gh_issue_list`, `gh_api_call` |
| **Git** | 12+ | `git_commit`, `git_branch`, `git_merge`, `git_log`, `git_diff` |
| **Bash/Shell** | 6+ | `run_command`, `run_background`, `capture_output`, `pipe_commands` |
| **PowerShell** | 6+ | `ps_run`, `ps_get_process`, `ps_file_ops`, `ps_registry` |
| **File I/O** | 10+ | `read_file`, `write_file`, `edit_file`, `glob_search`, `grep_content` |
| **Search** | 5+ | `semantic_search`, `find_symbol`, `find_references`, `ripgrep` |
| **Python** | 8+ | `run_pytest`, `pip_install`, `create_venv`, `run_black`, `run_mypy` |
| **Node.js** | 6+ | `npm_install`, `npm_run`, `run_tsc`, `run_eslint` |
| **Docker** | 5+ | `docker_build`, `docker_run`, `docker_compose_up`, `docker_logs` |
| **HTTP** | 4+ | `http_get`, `http_post`, `curl`, `parse_json_response` |
| **Database** | 4+ | `sqlite_query`, `pg_query`, `run_migration` |
| **System** | 4+ | `get_env`, `set_env`, `list_processes`, `disk_usage` |

**Total: 78+ pre-installed tools.** Agent-created tools add to this over time.

**Tool recall (how the agent finds the right tool):**

```python
class ToolsDB:
    """Searchable database of all tools (core + learned)."""

    def find_tool(self, query: str) -> list[ToolInfo]:
        """Semantic search: 'I need to create a GitHub PR' → gh_pr_create."""
        return self.vector_search(query, top_k=5)

    def find_by_category(self, category: str) -> list[ToolInfo]:
        """Get all tools in a category."""
        return self.db.query("SELECT * FROM tools WHERE category = ?", category)

    def get_usage_stats(self, tool_name: str) -> ToolStats:
        """How often has this tool been used? Success rate?"""
        return self.db.query("SELECT * FROM tool_usage WHERE name = ?", tool_name)

    def suggest_tool(self, task_description: str) -> ToolInfo | None:
        """Given a task, suggest the best tool."""
        candidates = self.find_tool(task_description)
        # Rank by relevance + success rate + recency
        return self._rank(candidates)[0] if candidates else None
```

**The agent doesn't need to remember all tools.** It queries the Tools DB when it needs a capability. The Tools DB scales to hundreds of tools effortlessly -- it grows with every tool the agent creates. Core tools are always available, learned tools persist across sessions, and semantic search ensures the right tool is found even with natural language queries like "I need to deploy this to Docker."

**Tools DB Schema (SQLite):**

```sql
-- Core table: every tool ever registered
CREATE TABLE tools (
    id          TEXT PRIMARY KEY,          -- Unique tool ID
    name        TEXT NOT NULL UNIQUE,      -- Function name (e.g., "gh_pr_create")
    category    TEXT NOT NULL,             -- "github", "git", "python", "learned", etc.
    source      TEXT NOT NULL,             -- "core" | "learned" | "community"
    description TEXT NOT NULL,             -- What the tool does (used for semantic search)
    parameters  TEXT,                      -- JSON: parameter names, types, descriptions
    code_path   TEXT,                      -- Path to Python source file
    created_at  TIMESTAMP DEFAULT NOW,
    updated_at  TIMESTAMP DEFAULT NOW,
    version     INTEGER DEFAULT 1,
    enabled     BOOLEAN DEFAULT TRUE
);

-- Usage tracking: how often each tool is used and how well it works
CREATE TABLE tool_usage (
    id          INTEGER PRIMARY KEY,
    tool_id     TEXT REFERENCES tools(id),
    timestamp   TIMESTAMP DEFAULT NOW,
    success     BOOLEAN,                   -- Did the tool call succeed?
    duration_ms INTEGER,                   -- How long did it take?
    context     TEXT,                      -- What task was the tool used for?
    error       TEXT                       -- Error message if failed
);

-- Tags for flexible categorization beyond the primary category
CREATE TABLE tool_tags (
    tool_id     TEXT REFERENCES tools(id),
    tag         TEXT,                      -- e.g., "async", "web", "testing", "deployment"
    PRIMARY KEY (tool_id, tag)
);

-- Indexes for fast lookup
CREATE INDEX idx_tools_category ON tools(category);
CREATE INDEX idx_tools_source ON tools(source);
CREATE INDEX idx_tool_usage_tool ON tool_usage(tool_id);
CREATE INDEX idx_tool_tags_tag ON tool_tags(tag);
```

**How the agent dynamically loads tools:**

```python
class ToolsDB:
    """Scalable tool repository. Core tools + agent-created tools."""

    def __init__(self, db_path: str = "~/.gaia/tools/tools.db"):
        self.db = sqlite3.connect(db_path)
        self.faiss_index = faiss.read_index("~/.gaia/tools/tools_vectors.faiss")
        self._load_core_tools()  # Always available

    def _load_core_tools(self):
        """Load pre-installed core tools on startup."""
        core_dir = Path(__file__).parent / "core_tools"
        for tool_file in core_dir.glob("*.py"):
            self._register_from_file(tool_file, source="core")

    def get_active_tools(self, task_context: str = None) -> list[ToolDef]:
        """Get tools relevant to the current task.

        Instead of loading ALL tools into the LLM context,
        load only the most relevant ones (top 20-30).
        This keeps the tool list manageable for any LLM.
        """
        if task_context:
            # Semantic search: find tools relevant to this task
            relevant = self.find_tool(task_context, top_k=20)
        else:
            # No context: return most-used tools
            relevant = self._get_most_used(limit=20)

        # Always include essential tools (file I/O, terminal, git)
        essentials = self._get_essentials()
        return self._deduplicate(essentials + relevant)

    def find_tool(self, query: str, top_k: int = 10) -> list[ToolInfo]:
        """Semantic search across all tools."""
        query_vec = self.embed(query)
        distances, indices = self.faiss_index.search(query_vec, top_k)
        return [self._load_tool_info(idx) for idx in indices[0]]

    def register_learned_tool(self, name, description, code, category="learned"):
        """Register a tool the agent created."""
        # Store in DB
        self.db.execute(
            "INSERT INTO tools (id, name, category, source, description, code_path) "
            "VALUES (?, ?, ?, 'learned', ?, ?)",
            (uuid4(), name, category, description, f"learned/{name}.py")
        )
        # Add to FAISS index
        vec = self.embed(description)
        self.faiss_index.add(vec)
        self.db.commit()

    @property
    def total_tools(self) -> int:
        """How many tools are in the vault."""
        return self.db.execute("SELECT COUNT(*) FROM tools").fetchone()[0]

    @property
    def tool_stats(self) -> dict:
        """Summary stats for the vault."""
        return {
            "total": self.total_tools,
            "core": self._count_by_source("core"),
            "learned": self._count_by_source("learned"),
            "most_used": self._get_most_used(limit=5),
            "recently_created": self._get_recent(limit=5),
        }
```

**Key design decisions for scalability:**
- **Lazy loading:** Only load tools relevant to the current task (top 20-30), not all hundreds
- **Semantic search:** Find tools by natural language, not just by name
- **Usage tracking:** Tools that are used more often rank higher in search results
- **Core + learned:** Core tools ship with GAIA; learned tools are created by the agent
- **SQLite + FAISS:** Scales to hundreds of tools without performance degradation

**Domain-Specific Tool Packs (checked into the repo):**

Tools are organized by application domain. Each agent type loads its relevant tool packs. Tool packs can be committed to git and shared across the team.

```
src/gaia/tools/                 # Checked into the repo, version-controlled
├── __init__.py
├── core/                       # Always loaded for ALL agents
│   ├── file_io.py              # read, write, edit, glob, grep
│   ├── terminal.py             # bash, powershell, background processes
│   ├── git.py                  # commit, branch, merge, diff, log
│   ├── github.py               # gh CLI wrapper: PRs, issues, API
│   ├── search.py               # semantic search, ripgrep, find symbols
│   ├── http.py                 # HTTP requests, API calls
│   └── system.py               # env vars, processes, disk ops
│
├── coding/                     # Loaded for `gaia code`
│   ├── python.py               # pytest, pip, venv, black, isort, mypy
│   ├── node.py                 # npm, npx, tsc, eslint, prettier
│   ├── docker.py               # docker build, run, compose
│   ├── database.py             # SQLite, PostgreSQL, migrations
│   ├── ast_tools.py            # AST parsing, symbol extraction
│   ├── refactoring.py          # rename, extract function, inline
│   └── quality_gates.py        # syntax check, lint, test runner, security scan
│
├── cua/                        # Loaded for `gaia cua` (Computer Use Agent)
│   ├── screenshot.py           # capture, analyze, OCR
│   ├── mouse.py                # click, drag, scroll
│   ├── keyboard.py             # type, hotkey, key combos
│   ├── browser.py              # Playwright: navigate, fill, click
│   ├── desktop.py              # UIAutomation, window management
│   └── vision.py               # VLM-based element detection
│
├── knowledge/                  # Loaded for `gaia chat` (Knowledge Assistant)
│   ├── rag.py                  # document retrieval, chunking, search
│   ├── pdf.py                  # PDF extraction, parsing
│   ├── summarize.py            # text summarization
│   ├── web_search.py           # web search, content extraction
│   └── citation.py             # source tracking, citation generation
│
├── workflow/                   # Loaded for workflow automation
│   ├── email.py                # IMAP/SMTP, Gmail API
│   ├── calendar.py             # Google Calendar, Outlook
│   ├── slack.py                # Slack API: messages, channels
│   ├── jira.py                 # Jira API: issues, sprints
│   └── scheduler.py            # cron, triggers, webhooks
│
└── analysis/                   # Cross-domain analysis tools
    ├── codebase_index.py       # Index 1M+ LoC repos
    ├── dependency_graph.py     # Build and query dependency graphs
    ├── architecture_report.py  # Generate architecture reports
    └── security_scan.py        # OWASP, secret detection, vulnerability scan
```

**How agents load tool packs:**

```python
class GaiaCodeAgent(CodeAgent):
    TOOL_PACKS = ["core", "coding", "analysis"]  # Loaded at init

class GaiaCUAAgent(Agent):
    TOOL_PACKS = ["core", "cua"]

class GaiaChatAgent(Agent):
    TOOL_PACKS = ["core", "knowledge"]

class GaiaWorkflowAgent(Agent):
    TOOL_PACKS = ["core", "workflow", "coding"]  # Can combine packs
```

**Adding new tools is as simple as:**

```python
# src/gaia/tools/coding/go.py  (new file, checked into git)

@tool
def go_build(path: str = ".") -> str:
    """Build a Go project."""
    return run_command(f"go build {path}")

@tool
def go_test(path: str = "./...", verbose: bool = True) -> str:
    """Run Go tests."""
    flags = "-v" if verbose else ""
    return run_command(f"go test {flags} {path}")

@tool
def go_mod_tidy() -> str:
    """Clean up Go module dependencies."""
    return run_command("go mod tidy")
```

The Tools DB automatically picks up new tool files from the tool packs. No registration boilerplate needed -- just write a `@tool`-decorated function in the right directory and it's available to the right agents.

**Tools vs Skills: Two Separate Databases**

Tools and skills are fundamentally different things. They have separate databases.

| | **Tools** (`tools.db`) | **Skills** (`skills.db`) |
|---|---|---|
| **What** | Concrete functions the agent calls | Learned workflows and patterns |
| **Granularity** | Single action (read file, run test) | Multi-step recipe (build FastAPI app, debug async code) |
| **Created by** | Developers (core) or agent (learned) | Agent (extracted from experience) |
| **Example** | `run_pytest(path, flags)` | "How to build a FastAPI app with JWT auth" |
| **Invocation** | Direct function call | Guides agent's planning and approach |
| **Storage** | `~/.gaia/tools/tools.db` | `~/.gaia/skills/skills.db` |

**Skills DB: Learned Workflows and Patterns**

```
~/.gaia/skills/
├── skills.db              # SQLite: skill metadata, steps, success rates
├── skills_vectors.faiss   # FAISS: semantic search over skill descriptions
└── templates/             # Skill templates (generated code patterns)
```

```sql
-- Skills are multi-step workflows the agent has learned
CREATE TABLE skills (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,              -- "build_fastapi_jwt_app"
    description     TEXT NOT NULL,              -- "Build a FastAPI app with JWT auth"
    category        TEXT NOT NULL,              -- "web_dev", "testing", "refactoring", etc.
    domain          TEXT,                       -- "coding", "cua", "knowledge", "workflow"
    steps           TEXT NOT NULL,              -- JSON: ordered list of steps
    prerequisites   TEXT,                       -- JSON: what must be true before using this skill
    tools_used      TEXT,                       -- JSON: list of tool IDs this skill uses
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    confidence      REAL DEFAULT 0.5,           -- 0.0 to 1.0, increases with successful use
    source          TEXT DEFAULT 'learned',     -- "learned" | "curated" | "imported"
    created_at      TIMESTAMP DEFAULT NOW,
    last_used_at    TIMESTAMP,
    version         INTEGER DEFAULT 1
);

-- Track each time a skill is used and the outcome
CREATE TABLE skill_usage (
    id          INTEGER PRIMARY KEY,
    skill_id    TEXT REFERENCES skills(id),
    timestamp   TIMESTAMP DEFAULT NOW,
    success     BOOLEAN,
    task_description TEXT,                      -- What task triggered this skill
    modifications TEXT,                        -- JSON: how the skill was adapted for this use
    feedback    TEXT                           -- User feedback if any
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_domain ON skills(domain);
CREATE INDEX idx_skills_confidence ON skills(confidence);
```

```python
class SkillsDB:
    """Learned workflows and patterns. Grows with experience."""

    def find_skill(self, task: str) -> list[SkillInfo]:
        """Find skills relevant to a task.
        'Build a REST API with auth' → skill: build_fastapi_jwt_app
        """
        return self.vector_search(task, top_k=5)

    def learn_skill(self, name: str, description: str, steps: list[str],
                    tools_used: list[str], category: str):
        """Extract a new skill from a successful task execution."""
        self.db.execute(
            "INSERT INTO skills (id, name, description, steps, tools_used, category) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (uuid4(), name, description, json.dumps(steps),
             json.dumps(tools_used), category)
        )
        # Add to FAISS index
        vec = self.embed(description)
        self.faiss_index.add(vec)
        self.db.commit()

    def record_usage(self, skill_id: str, success: bool, task: str):
        """Record skill usage and update confidence."""
        self.db.execute(
            "INSERT INTO skill_usage (skill_id, success, task_description) VALUES (?, ?, ?)",
            (skill_id, success, task)
        )
        # Update confidence based on success rate
        if success:
            self.db.execute(
                "UPDATE skills SET success_count = success_count + 1, "
                "last_used_at = CURRENT_TIMESTAMP WHERE id = ?", (skill_id,))
        else:
            self.db.execute(
                "UPDATE skills SET failure_count = failure_count + 1 WHERE id = ?",
                (skill_id,))
        self._recalculate_confidence(skill_id)
        self.db.commit()

    def get_by_domain(self, domain: str) -> list[SkillInfo]:
        """Get all skills for a specific domain (coding, cua, knowledge, workflow)."""
        return self.db.query("SELECT * FROM skills WHERE domain = ?", domain)
```

**Example skills the agent learns over time:**

```
Skill: "build_fastapi_jwt_app"
  Domain: coding
  Category: web_dev
  Confidence: 0.92 (used 12 times, succeeded 11)
  Steps:
    1. Create project structure (src/, tests/, requirements.txt)
    2. Define User model with SQLAlchemy
    3. Create auth module (register, login, refresh)
    4. Add JWT middleware
    5. Create CRUD endpoints
    6. Write unit tests
    7. Write integration tests
    8. Run quality gates
  Tools used: write_file, run_pytest, pip_install, run_black

Skill: "debug_async_python"
  Domain: coding
  Category: debugging
  Confidence: 0.85 (used 7 times, succeeded 6)
  Steps:
    1. Read the full error traceback
    2. Check for missing await keywords
    3. Check for blocking calls in async context
    4. Verify event loop is running
    5. Check for race conditions in shared state
    6. Add asyncio.gather exception handling
  Tools used: read_file, search, run_pytest

Skill: "fill_web_form"
  Domain: cua
  Category: browser_automation
  Confidence: 0.78 (used 5 times, succeeded 4)
  Steps:
    1. Screenshot the page
    2. Identify form fields with VLM
    3. Tab through fields in order
    4. Fill each field
    5. Screenshot to verify
    6. Click submit
  Tools used: screenshot, keyboard_type, mouse_click, vision_detect
```

### C7. Memory Self-Organization (Defragmentation)

> **As the agent accumulates hundreds of tools, skills, memories, and knowledge entries, the data can become fragmented, contradictory, and stale. Without self-organization, more knowledge = more noise = worse performance.** The agent must be able to clean up after itself.

**The problem:**

```
Week 1:   Agent has 50 memories, 80 tools, 10 skills → Fast, accurate retrieval
Week 4:   Agent has 200 memories, 120 tools, 40 skills → Some duplicates, some stale
Week 12:  Agent has 800 memories, 200 tools, 100 skills → Contradictions, noise, slow search
Week 24:  Agent has 2000+ entries → Retrieval quality degrades, wrong memories surface
```

Without defragmentation, the agent gets **worse** over time. The exact opposite of what we want.

**The solution: periodic self-organization**

```
┌──────────────────────────────────────────────────────────────────┐
│                  MEMORY SELF-ORGANIZATION                         │
│                                                                    │
│   Triggers:                                                        │
│   • Scheduled (e.g., every 100 sessions or weekly)                │
│   • On demand (user runs `gaia defrag`)                           │
│   • Automatic when retrieval quality drops                        │
│                                                                    │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│   │  DEDUPLICATE │──▶│  RECONCILE   │──▶│    PRUNE     │        │
│   │              │   │              │   │              │        │
│   │  Find near-  │   │  Resolve     │   │  Remove low- │        │
│   │  duplicate   │   │  contradic-  │   │  confidence  │        │
│   │  entries     │   │  tions       │   │  stale data  │        │
│   └──────────────┘   └──────────────┘   └──────────────┘        │
│          │                   │                   │                │
│          ▼                   ▼                   ▼                │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│   │  CONSOLIDATE │──▶│  RE-INDEX    │──▶│   VERIFY     │        │
│   │              │   │              │   │              │        │
│   │  Merge       │   │  Rebuild     │   │  Test that   │        │
│   │  related     │   │  FAISS       │   │  retrieval   │        │
│   │  entries     │   │  vectors     │   │  still works │        │
│   └──────────────┘   └──────────────┘   └──────────────┘        │
└──────────────────────────────────────────────────────────────────┘
```

**The 6 operations:**

```python
class MemoryDefragmenter:
    """Self-organizes the agent's knowledge stores."""

    def defrag(self, aggressive: bool = False):
        """Run full defragmentation. Safe by default, aggressive on request."""
        report = DefragReport()

        # 1. DEDUPLICATE: Find and merge near-duplicate entries
        duplicates = self._find_duplicates()
        for group in duplicates:
            merged = self._merge_entries(group)
            report.merged += len(group) - 1

        # 2. RECONCILE: Resolve contradictions
        contradictions = self._find_contradictions()
        for pair in contradictions:
            winner = self._resolve_contradiction(pair)
            report.reconciled += 1

        # 3. PRUNE: Remove low-value entries
        pruned = self._prune_stale(
            max_age_days=180 if not aggressive else 90,
            min_confidence=0.2 if not aggressive else 0.3,
            min_usage=0 if not aggressive else 1
        )
        report.pruned = len(pruned)

        # 4. CONSOLIDATE: Merge related entries into higher-level knowledge
        consolidated = self._consolidate_patterns()
        report.consolidated = len(consolidated)

        # 5. RE-INDEX: Rebuild vector indices
        self._rebuild_faiss_indices()
        report.reindexed = True

        # 6. VERIFY: Test retrieval quality
        report.retrieval_quality = self._verify_retrieval()

        return report

    def _find_duplicates(self) -> list[list[Entry]]:
        """Find entries that are semantically near-identical."""
        # Compare all pairs using cosine similarity
        # Entries with similarity > 0.95 are considered duplicates
        all_entries = self.knowledge_db.get_all_entries()
        all_vectors = self.knowledge_db.get_all_vectors()
        # Use FAISS to find near-neighbors efficiently
        # Group entries with similarity > threshold
        ...

    def _find_contradictions(self) -> list[tuple[Entry, Entry]]:
        """Find entries that contradict each other.
        e.g., 'User prefers pytest' vs 'User prefers unittest'
        """
        # Group by topic, then check for conflicting values
        # Use LLM to assess if two entries contradict
        ...

    def _resolve_contradiction(self, pair: tuple[Entry, Entry]) -> Entry:
        """Keep the more recent, higher-confidence entry."""
        a, b = pair
        # More recent wins (user preferences change)
        # Higher confidence wins (more validated)
        # Higher usage wins (more frequently retrieved)
        score_a = (0.4 * a.recency + 0.3 * a.confidence + 0.3 * a.usage)
        score_b = (0.4 * b.recency + 0.3 * b.confidence + 0.3 * b.usage)
        winner, loser = (a, b) if score_a >= score_b else (b, a)
        self.knowledge_db.archive(loser)  # Don't delete, archive
        return winner

    def _prune_stale(self, max_age_days, min_confidence, min_usage) -> list[Entry]:
        """Remove entries that are old, low-confidence, and unused."""
        stale = self.knowledge_db.query(
            "SELECT * FROM entries WHERE "
            "age_days > ? AND confidence < ? AND times_retrieved < ?",
            (max_age_days, min_confidence, min_usage)
        )
        for entry in stale:
            self.knowledge_db.archive(entry)  # Archive, don't delete
        return stale

    def _consolidate_patterns(self) -> list[Entry]:
        """Merge multiple specific memories into general patterns.

        Example:
          Memory 1: "Used pathlib in project A"
          Memory 2: "Used pathlib in project B"
          Memory 3: "Used pathlib in project C"
          → Consolidated: "User always prefers pathlib" (confidence: 0.95)
        """
        # Group entries by semantic similarity
        # If 3+ entries share a pattern, create a consolidated entry
        # Remove the individual entries, keep the consolidated one
        ...

    def _rebuild_faiss_indices(self):
        """Rebuild all FAISS vector indices from current data."""
        # Knowledge DB
        entries = self.knowledge_db.get_all_entries()
        vectors = self.embed_batch([e.content for e in entries])
        self.knowledge_db.rebuild_index(vectors)

        # Tools DB
        tools = self.tools_db.get_all_tools()
        vectors = self.embed_batch([t.description for t in tools])
        self.tools_db.rebuild_index(vectors)

        # Skills DB
        skills = self.skills_db.get_all_skills()
        vectors = self.embed_batch([s.description for s in skills])
        self.skills_db.rebuild_index(vectors)

    def _verify_retrieval(self) -> float:
        """Test that retrieval still works after defrag.
        Run a set of known queries and check if expected results appear.
        """
        test_queries = self._get_verification_queries()
        correct = 0
        for query, expected in test_queries:
            results = self.knowledge_db.semantic_search(query, top_k=5)
            if expected in [r.id for r in results]:
                correct += 1
        return correct / len(test_queries) if test_queries else 1.0
```

**User-facing commands:**

```bash
# Check memory health
gaia memory status
  Knowledge DB:  842 entries (37 stale, 12 duplicates detected)
  Tools DB:      156 tools (78 core, 68 learned, 10 unused)
  Skills DB:     47 skills (avg confidence: 0.82)
  Last defrag:   3 weeks ago
  Recommendation: Run defrag (12 duplicates, 37 stale entries)

# Run defragmentation
gaia memory defrag
  Deduplicating...   12 duplicates merged
  Reconciling...     3 contradictions resolved
  Pruning...         37 stale entries archived
  Consolidating...   8 patterns consolidated
  Re-indexing...     Done
  Verifying...       Retrieval quality: 97% (was 89%)
  ✅ Defrag complete. 842 → 782 entries. Retrieval improved 8%.

# Aggressive defrag (more pruning)
gaia memory defrag --aggressive
  ✅ Defrag complete. 782 → 614 entries.

# View what was archived (nothing is permanently deleted)
gaia memory archive
  [37 stale entries, 12 merged duplicates, 3 contradiction losers]

# Restore an archived entry if needed
gaia memory restore <entry-id>
```

**Key design decisions:**

| Decision | Rationale |
|----------|-----------|
| **Archive, never delete** | User can always restore something that was pruned by mistake |
| **Verify after defrag** | Run retrieval quality checks to ensure defrag didn't break anything |
| **Automatic triggers** | Don't rely on user remembering to defrag. Trigger when retrieval quality drops below 90% |
| **Consolidation** | The most important operation. Turns 10 specific memories into 1 general pattern with high confidence |
| **Separate operations** | Each step (dedup, reconcile, prune, consolidate, re-index, verify) runs independently. If one fails, others still complete |

**The three databases after defrag:**

```
~/.gaia/
├── knowledge/
│   ├── knowledge.db          # Memories, learnings, preferences, conventions
│   ├── knowledge_vectors.faiss
│   └── archive/              # Archived entries (from defrag)
├── tools/
│   ├── tools.db              # Tool registry, usage stats
│   ├── tools_vectors.faiss
│   ├── core/                 # Pre-installed tools (version-controlled)
│   └── learned/              # Agent-created tools
├── skills/
│   ├── skills.db             # Learned workflows, patterns
│   ├── skills_vectors.faiss
│   └── templates/            # Skill templates
└── defrag/
    ├── defrag_log.db         # History of all defrag runs
    └── verification_queries.json  # Test queries for retrieval verification
```

### C8. Queue-Based Asynchronous Interaction

> **The agent and user communicate like humans do -- via async messages. Neither blocks waiting for the other.**

Every current coding agent has the same interaction flaw: either the agent blocks waiting for user input (Claude Code's `AskUserQuestion`), or it runs completely headless with no interaction (Devin). Both are wrong.

**Humans don't work this way.** When you Slack a colleague a question, you don't sit idle waiting for their reply. You keep working on other things. When they answer, you incorporate it. The same should be true for agent-human interaction.

**How it works:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  QUEUE-BASED INTERACTION MODEL                    │
│                                                                   │
│   USER                           AGENT                           │
│   ┌──────────┐                   ┌──────────┐                    │
│   │  Outbox   │ ───messages────▶ │  Inbox    │                    │
│   │          │                   │          │                    │
│   │  "Change │                   │ (picks up │                    │
│   │  auth to │                   │  when     │                    │
│   │  OAuth2" │                   │  between  │                    │
│   │          │                   │  steps)   │                    │
│   └──────────┘                   └──────────┘                    │
│                                                                   │
│   ┌──────────┐                   ┌──────────┐                    │
│   │  Inbox   │ ◀───queries────── │  Outbox   │                    │
│   │          │                   │          │                    │
│   │ (answers │                   │ "Should I │                    │
│   │  when    │                   │  use JWT  │                    │
│   │  free)   │                   │  or       │                    │
│   │          │                   │  session  │                    │
│   └──────────┘                   │  auth?"   │                    │
│                                   └──────────┘                    │
│                                                                   │
│   KEY: Neither side ever blocks. Both keep working.              │
└─────────────────────────────────────────────────────────────────┘
```

**Three message priority levels:**

| Priority | Name | Behavior |
|----------|------|----------|
| **FYI** | Informational | Agent posts status updates. User reads when convenient. No response needed. |
| **Question** | Would-be-nice | Agent asks a question but continues with sensible defaults if no answer within N minutes. |
| **Decision** | Needs-answer | Agent needs user input for THIS task. Parks the task, works on other tasks. When user answers, resumes the parked task. |

**Critical design:** Even "Decision" priority never stops the agent entirely. It only parks the specific task that needs the answer. The agent continues working on other tasks or subtasks.

```python
class MessageQueue:
    """Bidirectional async message queue between agent and user."""

    def __init__(self, db_path: str = "~/.gaia/messages.db"):
        self.db = sqlite3.connect(db_path)
        self._create_tables()

    def send_to_user(self, message: str, priority: str = "fyi",
                     task_id: str = None, options: list[str] = None):
        """Agent sends message to user. Agent does NOT block."""
        self.db.execute(
            "INSERT INTO messages (direction, content, priority, task_id, "
            "options, status, created_at) VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            ("agent_to_user", message, priority, task_id,
             json.dumps(options) if options else None, datetime.now())
        )
        self.db.commit()

    def send_to_agent(self, message: str, reply_to: int = None):
        """User sends message to agent. User does NOT block."""
        self.db.execute(
            "INSERT INTO messages (direction, content, priority, reply_to, "
            "status, created_at) VALUES (?, ?, 'user_input', ?, 'pending', ?)",
            ("user_to_agent", message, reply_to, datetime.now())
        )
        self.db.commit()

    def check_inbox(self) -> list[Message]:
        """Agent checks for new user messages. Called between steps."""
        return self.db.execute(
            "SELECT * FROM messages WHERE direction = 'user_to_agent' "
            "AND status = 'pending' ORDER BY created_at"
        ).fetchall()

    def get_pending_questions(self) -> list[Message]:
        """Get unanswered agent questions (for user CLI/TUI)."""
        return self.db.execute(
            "SELECT * FROM messages WHERE direction = 'agent_to_user' "
            "AND priority IN ('question', 'decision') AND status = 'pending'"
        ).fetchall()

    def check_answer(self, question_id: int) -> Message | None:
        """Check if user answered a specific question."""
        return self.db.execute(
            "SELECT * FROM messages WHERE reply_to = ? AND status = 'pending'",
            (question_id,)
        ).fetchone()
```

**Agent-side behavior (runs every step):**

```python
class AsyncInteractionMixin:
    """Non-blocking interaction with the user."""

    def between_steps(self):
        """Called between every step. Process inbox, never block."""
        # 1. Check for user messages
        new_messages = self.message_queue.check_inbox()
        for msg in new_messages:
            self._process_user_message(msg)

        # 2. Check for answers to pending questions
        for question in self.pending_questions:
            answer = self.message_queue.check_answer(question.id)
            if answer:
                self._resume_parked_task(question.task_id, answer)
            elif question.priority == "question" and question.age > self.default_timeout:
                # Question timed out -- use default and continue
                self._use_default(question)

    def _process_user_message(self, msg: Message):
        """Handle incoming user message without interrupting current work."""
        if msg.is_task_modification:
            # "Change auth to OAuth2" -- update the plan
            self.plan.replan(reason=f"User requested: {msg.content}")
            self.audit_log.log("USER_INPUT", f"Plan updated: {msg.content}")
        elif msg.is_new_task:
            # "Also add rate limiting" -- add to task queue
            self.plan.add_task(msg.content)
        elif msg.is_preference:
            # "Use tabs not spaces" -- store preference
            self.knowledge_db.store_preference(msg.content)
        elif msg.is_answer:
            # Answer to a previous question -- resume parked task
            self._resume_parked_task(msg.reply_to, msg)

    def ask_user(self, question: str, priority: str = "question",
                 options: list[str] = None, default: str = None,
                 task_id: str = None):
        """Ask the user something. Never block."""
        self.message_queue.send_to_user(
            message=question, priority=priority,
            options=options, task_id=task_id
        )
        if priority == "decision":
            # Park this task, work on something else
            self.plan.park_task(task_id, reason=f"Waiting for user: {question}")
        # Agent continues working -- never waits
```

**User-side experience:**

```bash
# User checks agent's questions at any time
gaia code inbox
  [QUESTION] "Should I use JWT or session auth?" (asked 5 min ago, will use JWT in 10 min)
  [FYI] "Completed database models (3/7 tasks done)" (2 min ago)
  [DECISION] "Found 3 circular deps. Fix now or defer?" (waiting for answer)

# User answers a question
gaia code reply 1 "Use session auth"
  ✅ Answer sent. Agent will pick up on next step.

# User sends a new instruction without interrupting
gaia code tell "Also add rate limiting to the API endpoints"
  ✅ Message queued. Agent will incorporate this.

# User checks what the agent is working on
gaia code status
  Working on: Task 4/7 - CRUD endpoints
  Parked: Task 3 - Auth (waiting for your answer about auth method)
  Queued: "Add rate limiting" (will add to plan)
```

**Why this matters:**

| Current Model | Queue Model |
|---------------|-------------|
| Agent blocks on question → user waits → agent waits → time wasted | Agent asks, keeps working → user answers when free → zero downtime |
| User can't give feedback mid-task | User sends "change X" at any time → agent incorporates |
| Agent runs headless OR interactive (pick one) | Agent is always working AND always receptive |
| Missed context: agent doesn't know user changed their mind | User can redirect agent at any time via queued message |

**This is how human teams collaborate.** Slack messages, not phone calls. Async by default, sync when critical.

### C9. Insight Generation & Structured Learning

> **The agent doesn't just store data -- it generates insights, attaches metadata, and retrieves them when they matter.**

Raw data is useless without interpretation. Every coding session produces patterns, learnings, and insights that should be captured and recalled automatically in relevant future contexts.

**The problem with current agents:**

```
Session 1: Agent fixes a race condition in async code
Session 2: Agent hits the exact same race condition pattern
           → Doesn't remember Session 1's fix
           → Spends 20 minutes re-discovering the same solution
```

**What insight generation looks like:**

```
Session 1: Agent fixes a race condition in async code

  INSIGHT GENERATED:
  ┌──────────────────────────────────────────────────────────────┐
  │ Insight: "asyncio.gather() swallows exceptions by default.  │
  │          Always use return_exceptions=True or wrap each      │
  │          coroutine in try/except to prevent silent failures."│
  │                                                              │
  │ Category: debugging                                          │
  │ Domain: python/async                                         │
  │ Trigger: exception_handling, asyncio, race_condition         │
  │ Confidence: 0.7 (first occurrence)                           │
  │ Source: session_2026-02-10_003, task "fix user sync"         │
  │ Evidence: Fixed test_user_sync_parallel (was failing)        │
  │ Related files: src/services/user_sync.py                     │
  └──────────────────────────────────────────────────────────────┘

Session 5: Agent is writing async code with asyncio.gather()

  AUTO-RETRIEVER finds insight by semantic match:
  → "asyncio.gather() + exception handling" matches stored insight
  → Agent applies the pattern immediately
  → No re-discovery needed
```

**How insights are generated and stored:**

```python
class InsightEngine:
    """Generates structured insights from agent experience."""

    def after_task_complete(self, task: Task, plan: Plan, audit_log: AuditLog):
        """Extract insights from a completed task."""
        insights = []

        # 1. Error-fix insights: what errors did we hit and how did we fix them?
        errors_fixed = audit_log.get_entries(
            task_id=task.id, action_type="error_fix"
        )
        for error in errors_fixed:
            insights.append(Insight(
                content=f"Error: {error.error_message}\nFix: {error.fix_applied}",
                category="debugging",
                domain=self._detect_domain(error),  # "python/async", "react/hooks", etc.
                triggers=self._extract_triggers(error),  # keywords that should recall this
                confidence=0.7,  # First occurrence
                source_session=self.session_id,
                source_task=task.id,
                evidence=error.details,
            ))

        # 2. Pattern insights: what patterns did we use that worked?
        successful_patterns = self._detect_patterns(plan, audit_log)
        for pattern in successful_patterns:
            insights.append(Insight(
                content=pattern.description,
                category="pattern",
                domain=pattern.domain,
                triggers=pattern.keywords,
                confidence=0.6,
                source_session=self.session_id,
                source_task=task.id,
                evidence=f"Used in {pattern.occurrence_count} steps, all succeeded",
            ))

        # 3. Preference insights: what did the user correct?
        corrections = audit_log.get_entries(
            task_id=task.id, action_type="user_correction"
        )
        for correction in corrections:
            insights.append(Insight(
                content=f"User prefers: {correction.corrected_to} "
                        f"(instead of: {correction.original})",
                category="preference",
                domain="user",
                triggers=[correction.original, correction.corrected_to],
                confidence=0.8,  # User corrections are high-confidence
                source_session=self.session_id,
                source_task=task.id,
            ))

        # Store all insights with full metadata
        for insight in insights:
            self.knowledge_db.store_insight(insight)

    def on_session_end(self):
        """Generate session-level insights."""
        # What went well? What took too long? What should be different next time?
        session_summary = self._summarize_session()
        self.knowledge_db.store_insight(Insight(
            content=session_summary,
            category="session_learning",
            domain="meta",
            triggers=self._extract_session_topics(),
            confidence=0.5,
            source_session=self.session_id,
        ))


@dataclass
class Insight:
    """A structured learning with rich metadata for retrieval."""

    content: str                    # The actual insight (human-readable)
    category: str                   # "debugging", "pattern", "preference", "architecture", "performance"
    domain: str                     # "python/async", "react/hooks", "docker", "testing", "user"
    triggers: list[str]             # Keywords/concepts that should recall this insight
    confidence: float               # 0.0 to 1.0, increases with repeated validation
    source_session: str             # Which session generated this
    source_task: str = None         # Which task generated this
    evidence: str = None            # Supporting evidence (error messages, test results)
    created_at: datetime = None     # When it was generated
    last_recalled_at: datetime = None  # Last time it was retrieved
    recall_count: int = 0           # How many times it's been retrieved
    validated: bool = False         # Has this insight been validated by repeated use?
```

**Insight metadata schema (stored in knowledge.db):**

```sql
CREATE TABLE insights (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    category        TEXT NOT NULL,       -- debugging, pattern, preference, architecture, performance
    domain          TEXT NOT NULL,       -- python/async, react/hooks, docker, testing, user
    triggers        TEXT NOT NULL,       -- JSON array of trigger keywords
    confidence      REAL DEFAULT 0.5,
    source_session  TEXT NOT NULL,
    source_task     TEXT,
    evidence        TEXT,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_recalled_at TIMESTAMP,
    recall_count    INTEGER DEFAULT 0,
    validated       BOOLEAN DEFAULT FALSE,
    embedding       BLOB                 -- FAISS vector for semantic search
);

CREATE INDEX idx_insights_category ON insights(category);
CREATE INDEX idx_insights_domain ON insights(domain);
CREATE INDEX idx_insights_confidence ON insights(confidence);

-- Full-text search on content and triggers
CREATE VIRTUAL TABLE insights_fts USING fts5(content, triggers, domain);
```

**How insights are recalled (automatic, every turn):**

```python
class InsightRetriever:
    """Retrieves relevant insights before every agent turn."""

    def get_relevant_insights(self, current_context: str,
                               current_task: str) -> list[Insight]:
        """Find insights relevant to what the agent is about to do."""

        # 1. Semantic search: find insights similar to current context
        semantic_matches = self.knowledge_db.semantic_search_insights(
            query=current_task, top_k=10
        )

        # 2. Trigger search: find insights whose triggers match current context
        trigger_matches = self.knowledge_db.search_insights_by_triggers(
            context_keywords=self._extract_keywords(current_context)
        )

        # 3. Domain search: find insights in the same domain
        domain = self._detect_domain(current_context)
        domain_matches = self.knowledge_db.get_insights_by_domain(
            domain=domain, min_confidence=0.6
        )

        # 4. Merge, rank, deduplicate
        all_matches = self._merge_and_rank(
            semantic_matches, trigger_matches, domain_matches
        )

        # 5. Update recall metadata
        for insight in all_matches[:5]:
            self.knowledge_db.update_insight_recall(insight.id)

        return all_matches[:5]  # Top 5 most relevant
```

**Insight lifecycle:**

```
1. GENERATE  → Agent completes a task or fixes an error
                → InsightEngine extracts structured learnings
                → Stored with category, domain, triggers, confidence

2. VALIDATE  → Same insight applies successfully in a different context
                → Confidence increases (0.5 → 0.7 → 0.9)
                → validated = True after 3+ successful applications

3. RECALL    → Agent starts a new task
                → InsightRetriever finds relevant insights by:
                   semantic similarity + trigger matching + domain
                → Injected into context: "From past experience: ..."

4. EVOLVE    → Multiple related insights consolidate
                → "Used pathlib in project A, B, C"
                   → Consolidated: "Always use pathlib" (confidence: 0.95)
                → Part of memory defragmentation (C7)

5. DECAY     → Insight not recalled for 6+ months
                → Confidence decreases
                → Eventually pruned by defrag (archived, not deleted)
```

### C10. Recursive Agent Composition & Specialized Agents

> **The paradigm shift: The agent doesn't just use tools and skills -- it creates and orchestrates entire specialized agents recursively.**

This is the synthesis of everything above. Capabilities C1-C9 enable the agent to work effectively. **C10 enables the agent to build better versions of itself.**

**Foundation:** Extends [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab 2025) from stateless `llm_query()` to stateful `agent_query()`.

**The core mechanism: `agent_query()`**

```python
@tool
async def agent_query(
    task: str,
    specialist: str = None,
    context: dict = None,
    tools: list[str] = None,
    max_depth: int = 3,
    require_verification: bool = True
) -> AgentResult:
    """Recursively spawn a specialized agent to handle a task.

    This is RLMs extended to full agents:
    - Instead of llm_query() (text → text)
    - Use agent_query() (task → verified result)
    - Sub-agent has full tools, knowledge DB, quality gates
    - Sub-agent can recursively call agent_query() again
    - All agents share: manifest, plan, knowledge DB, memory DB

    Args:
        task: What the sub-agent should do
        specialist: Which specialized agent to use (e.g., "debugger_agent")
                   If None: auto-select from agents.db or use general agent
        context: Shared variables (inherited from parent + manifest)
        tools: Limit which tools sub-agent can use (default: all)
        max_depth: How many more recursion levels allowed
        require_verification: Sub-agent must pass quality gates

    Returns:
        AgentResult with success, result data, audit trail
    """
```

**What makes this different from RLMs:**

| | RLMs (llm_query) | RAC (agent_query) |
|---|-----------------|-------------------|
| Returns | Text | Verified result + audit trail |
| Has tools | No | Yes (full toolkit) |
| Has memory | No | Yes (shared knowledge DB + memory DB) |
| Can verify | No | Yes (quality gates) |
| State tracking | No | Yes (call stack, manifest, plan) |
| Can spawn specialists | No | Yes (agent registry) |

**The three-level capability hierarchy:**

```
LEVEL 1: Tools (78 core + learned)
  Single operations: read_file(), run_tests(), git_commit()
  Created by: Developers + agent (ToolBuilder)

     ↓ composed into

LEVEL 2: Skills (50+ learned)
  Multi-step workflows: "build FastAPI JWT auth"
  Created by: Agent (extracted from patterns)

     ↓ orchestrated by

LEVEL 3: Specialized Agents (7 core + learned)
  Domain experts with state machines: DebuggerAgent, SecurityAgent
  Created by: Developers + agent (AgentFactory)
```

**Agent Registry (agents.db):**

```
~/.gaia/agents/
├── agents.db                    # Registry: metadata, confidence, usage
├── agents_vectors.faiss         # Semantic search
├── core/                        # 7 pre-installed specialists
│   ├── debugger_agent.py        # Custom state machine for debugging
│   ├── security_agent.py        # OWASP scanning, vulnerability analysis
│   ├── refactoring_agent.py     # Extract, inline, rename, clean up
│   ├── testing_agent.py         # Test generation, coverage improvement
│   ├── documentation_agent.py   # Doc generation, comment updates
│   ├── performance_agent.py     # Profiling, optimization, benchmarking
│   └── architecture_agent.py    # Structure analysis, anti-pattern detection
└── learned/                     # Agent-created specialists
    ├── fastapi_crud_agent.py    # Created after 3 CRUD tasks
    ├── react_component_agent.py # Created after 3 React tasks
    └── docker_deploy_agent.py   # Created after 3 Docker tasks
```

**Core specialists (ship with GAIA):**

| Specialist | Domain | Custom State Machine | Key Tools |
|-----------|--------|---------------------|-----------|
| **DebuggerAgent** | debugging | identify → isolate → hypothesize → test → verify | read_file, edit_file, run_tests, trace |
| **SecurityAgent** | security | scan → analyze → fix → re-audit | security_scan, detect_secrets, check_deps |
| **RefactoringAgent** | refactoring | analyze → refactor → test → verify | edit_file, run_tests, analyze_ast |
| **TestingAgent** | testing | generate → run → improve_coverage → verify | write_file, run_tests, check_coverage |
| **DocumentationAgent** | docs | read_code → generate_docs → format → verify | read_file, write_file, generate_docstrings |
| **PerformanceAgent** | performance | profile → identify_bottleneck → optimize → benchmark | profile, benchmark, analyze_complexity |
| **ArchitectureAgent** | architecture | analyze_structure → detect_issues → suggest → validate | analyze_structure, detect_antipatterns |

**Agent auto-generation:**

```python
class AgentFactory:
    """Creates new specialized agents from observed patterns."""

    def after_task_complete(self, task: Task):
        """Check if this task reveals a recurring pattern."""

        # Find similar successful tasks (3+ required)
        similar = self.shared.knowledge_db.find_similar_tasks(
            task.description,
            min_similarity=0.8,
            min_success_rate=0.9
        )

        if len(similar) >= 3:
            # Extract common pattern
            pattern = self._analyze_pattern(similar)

            # Generate specialist agent code
            agent_code = self._generate_agent_class(
                name=pattern.suggested_name,
                domain=pattern.domain,
                tools=pattern.common_tools,
                skills=pattern.common_skills,
                workflow=pattern.workflow_steps,
                system_prompt=pattern.system_prompt
            )

            # Test on sample task
            if self._test_agent(agent_code, similar[0]):
                # Register in agents.db
                self.shared.agent_registry.register_learned_agent(
                    name=agent_code.class_name,
                    code=agent_code.source,
                    confidence=0.6
                )
```

**Example: FastAPICRUDAgent creation**

After completing tasks: "Create CRUD for users", "Create CRUD for posts", "Create CRUD for comments"

Agent generates:

```python
class FastAPICRUDAgent(GaiaCodeAgent):
    """Specialized in FastAPI CRUD endpoints.
    Generated by main agent on 2026-02-11.
    """

    DOMAIN = "web_dev"
    TOOLS = ["write_file", "edit_file", "run_tests", "run_black"]
    WORKFLOW = [
        "create_sqlalchemy_model",
        "create_pydantic_schemas",
        "create_crud_routes",
        "write_tests",
        "verify_tests_pass"
    ]

    SYSTEM_PROMPT = """
    You create FastAPI CRUD endpoints following this exact pattern:
    1. SQLAlchemy model in src/models/{entity}.py
    2. Pydantic schemas in src/schemas/{entity}.py
    3. Five routes (list, get, create, update, delete) in src/routes/{entity}.py
    4. Pytest tests in tests/test_{entity}.py
    5. Verify all tests pass

    Always follow project conventions from shared.manifest.decisions.
    """
```

**Compound learning effect:**

```
Month 1: Main agent + 7 core specialists
         Capability: 1x baseline

Month 2: Main agent + 7 core + 3 learned specialists
         Learned: FastAPICRUDAgent, ReactComponentAgent, DockerDeployAgent
         Capability: 1.8x (specialists handle recurring patterns faster)

Month 3: Main agent + 7 core + 8 learned specialists
         FastAPICRUDAgent created: WebSocketAgent (sub-specialist!)
         DebuggerAgent created: AsyncDebuggerAgent (sub-specialist!)
         Capability: 3.5x (specialists creating specialists)

Month 6: Main agent (now pure orchestrator)
         7 core + 25 learned specialists
         Specialists have 15 sub-specialists
         Emergent collaboration (specialists coordinate autonomously)
         Capability: 12x (exponential growth from recursive specialization)
```

**Why this is unprecedented:**

- **Claude Code:** Fixed 30 tools forever. Capability constant.
- **Devin:** Fixed workflow. Learns patterns but can't codify them as new agents.
- **GAIA Code:** Creates specialists → specialists create sub-specialists → exponential growth.

**See `RECURSIVE_AGENT_COMPOSITION.md` for complete RAC paradigm specification.**

### Why These 10 Capabilities Matter Together

```
C1 (Large Repo Analysis) + C2 (Deep Analysis)      = Agent understands ANY codebase
C3 (Multi-Day Execution) + C4 (Replanning)          = Agent can build ANYTHING
C5 (Time Awareness)      + C4 (Audit Log)            = Agent is fully transparent
C6 (Tool/Skill Building) + C3 (Multi-Day)            = Agent gets faster over time
C7 (Memory Defrag)       + C6 (Tool/Skill Building)  = Agent stays reliable as it grows
C8 (Async Interaction)   + C3 (Multi-Day)            = Agent collaborates like a human teammate
C9 (Insight Generation)  + C7 (Memory Defrag)        = Agent gets genuinely smarter over time
C10 (Recursive Agents)   + C6-C9 (Learning)          = Exponential capability growth

Together: An agent that understands million-line codebases,
builds complex features over days, adapts when things change,
creates its own tools, skills, AND specialists, tracks everything,
stays organized, collaborates asynchronously, generates insights,
recursively decomposes problems, spawns domain experts, and
improves exponentially over time.

No other agent does all ten. No other agent CAN.
This is Recursive Agent Composition (RAC).
```

```
C1 (Large Repo Analysis) + C2 (Deep Analysis)     = Agent understands ANY codebase
C3 (Multi-Day Execution) + C4 (Replanning)         = Agent can build ANYTHING
C5 (Time Awareness)      + C4 (Audit Log)           = Agent is fully transparent
C6 (Tool/Skill Building) + C3 (Multi-Day)           = Agent gets faster over time
C7 (Memory Defrag)       + C6 (Tool/Skill Building) = Agent stays reliable as it grows
C8 (Async Interaction)   + C3 (Multi-Day)           = Agent collaborates like a human teammate
C9 (Insight Generation)  + C7 (Memory Defrag)       = Agent gets genuinely smarter over time

Together: An agent that understands million-line codebases,
builds complex features over days, adapts when things change,
creates its own tools and skills, tracks everything with timestamps,
stays organized as it accumulates knowledge, collaborates
asynchronously with the user, generates and recalls structured
insights, and shows you everything it did.

No other agent does all nine.
```

---

## What We Already Have

**We are NOT starting from scratch.** The existing GAIA CodeAgent is already a substantial codebase with 70+ tools across 13 mixins. The V2 work builds ON TOP of this foundation.

### Existing CodeAgent Architecture

```
src/gaia/agents/code/
├── agent.py                    # CodeAgent class (13+ mixins)
├── orchestration/
│   ├── orchestrator.py         # LLM-driven checklist → deterministic execution
│   ├── checklist_generator.py  # LLM generates action checklists
│   └── checklist_executor.py   # Executes checklist steps
├── schema_inference.py         # AI-powered schema detection
├── validators/
│   ├── syntax_validator.py     # Python syntax checking
│   ├── antipattern_checker.py  # Code smell detection
│   ├── ast_analyzer.py         # AST-based analysis
│   └── requirements_validator.py  # Dependency hallucination detection
└── tools/                      # 13 tool mixins, 70+ @tool functions
    ├── code_tools.py           # 7 tools: generate_function/class/test, parse, validate
    ├── file_io.py              # 10 tools: read/write/edit files, search, diff
    ├── code_formatting.py      # 1 tool: format_with_black
    ├── project_management.py   # 3 tools: list_files, validate_project, create_project
    ├── testing.py              # 2 tools: execute_python_file, run_tests
    ├── error_fixing.py         # 1 tool: auto_fix_syntax_errors
    ├── typescript_tools.py     # 1 tool: validate_typescript
    ├── web_dev_tools.py        # 10 tools: React, Next.js, API endpoints
    ├── prisma_tools.py         # 2 tools: initialize_prisma, generate_prisma_model
    ├── cli_tools.py            # 5 tools: run_cli_command, process management
    ├── external_tools.py       # 2 tools: Context7, Perplexity search
    ├── validation_parsing.py   # Validation utilities
    └── validation_tools.py     # 1 tool: run_typescript_check
```

### Existing Capabilities to Build On

| What Exists | Where | V2 Evolution |
|-------------|-------|-------------|
| Orchestrator with checklist gen/exec | `orchestration/` | Evolves into State Machine (F9) |
| SyntaxValidator, ASTAnalyzer | `validators/` | Becomes Quality Gates (F8) |
| `auto_fix_syntax_errors` | `error_fixing.py` | Becomes Debug State in state machine |
| `run_tests` tool | `testing.py` | Becomes Test Quality Gate |
| `validate_project` | `project_management.py` | Becomes Build Quality Gate |
| `search_code` | `file_io.py` | Extends to semantic search (F4) |
| `list_symbols`, `parse_python_code` | `code_tools.py` | Extends to full Code Intelligence (F4) |
| `update_gaia_md` | `file_io.py` | Evolves into Project Manifest (F12) |
| Existing max_steps=100 | `agent.py` | Replaced by Continuous Execution (F7) |
| ChatSDK conversation history | `agent.py` | Extends to Episodic Memory (F10) |
| LLM factory (lemonade/claude/openai) | `src/gaia/llm/factory.py` | Extends to LLM Router (F18) |
| MCP integration | `src/gaia/mcp/` | Already exists, carried forward |
| @tool decorator + global registry | `src/gaia/agents/base/tools.py` | Must upgrade to instance-scoped (P0.2) |

### What This Means

**~60% of Foundation features (F1-F6) already exist.** The V2 work is primarily:
1. Adding the infrastructure prerequisites (config, serialization, lifecycle)
2. Adding the autonomous operation features (F7-F9) that don't exist
3. Adding the intelligence layer (F10-F13) that doesn't exist
4. Extending existing tools with new capabilities (AST, symbols, semantic search)

### Migration: Existing CodeAgent → V2 GaiaCodeAgent

```python
# BEFORE (V1 -- still works unchanged):
class CodeAgent(ApiAgent, Agent, CodeToolsMixin, ValidationAndParsingMixin, ...):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # max_steps=100, orchestrator-based execution

# AFTER (V2 -- extends V1 with opt-in features):
class GaiaCodeAgent(CodeAgent, ContinuousExecutionMixin, StateMachineMixin, ...):
    def __init__(self, config: V2Config = None, **kwargs):
        super().__init__(**kwargs)
        self.v2_config = config or V2Config()
        # Initialize V2 mixins (each checks its own flag)
        for mixin in self._v2_mixins:
            mixin.v2_init(self.v2_config)
```

**Key constraint:** `CodeAgent` (V1) continues to work unchanged. `GaiaCodeAgent` (V2) inherits from it and adds V2 capabilities. No breaking changes.

---

## Prerequisites: Core Infrastructure (Build FIRST)

**Before building ANY coding agent feature, these foundational infrastructure components must exist.** They are shared by ALL agents (`gaia code`, `gaia cua`, `gaia chat`) and cannot be skipped.

### Build Order: What Blocks Everything

```
LAYER 0: Must exist FIRST (blocks everything)
├── P0.1  V2Config + Feature Registry     (4h)   ← ALL features register here
├── P0.2  Enhanced Tool Registry           (4h)   ← Instance-scoped, not global
└── P0.3  Structured Logging               (2h)   ← Every component logs through this

LAYER 1: Core infrastructure (blocks all agent features)
├── P1.1  LLM Router + Tier System         (6h)   ← Selects right model per task
├── P1.2  Error Recovery (basic)            (4h)   ← Every LLM call needs retry
├── P1.3  Input Sanitization               (3h)   ← Security baseline
└── P1.4  Checkpoint/Serialization Base     (4h)   ← Save/restore any component state

LAYER 2: Shared agent infrastructure (blocks coding/CUA/chat features)
├── P2.1  Memory Infrastructure            (6h)   ← Storage layer for all memory tiers
├── P2.2  Task Infrastructure              (4h)   ← Task model + SQLite schema
└── P2.3  Mixin Lifecycle Protocol         (3h)   ← Standard init/enable/disable/cleanup

        ↓ ONLY NOW can coding agent features begin ↓
```

### P0.1: V2Config + Feature Registry (4 hours)

**Why FIRST:** Every single feature checks this to know if it's enabled. Without it, nothing can be opt-in.

```python
@dataclass
class V2Config:
    """Central configuration. All features default to OFF."""

    # LLM tier (determines which features are safe to enable)
    llm_tier: str = "basic"  # "basic" | "standard" | "advanced"

    # All feature flags (default: False)
    enable_memory: bool = False
    enable_task_queue: bool = False
    enable_continuous_execution: bool = False
    enable_state_machine: bool = False
    enable_manifest: bool = False
    enable_learning: bool = False
    enable_dynamic_tools: bool = False
    enable_adaptive_prompts: bool = False
    enable_self_eval: bool = False
    enable_proactive_analysis: bool = False
    enable_observability: bool = False
    enable_computer_use: bool = False
    enable_persona: bool = False
    enable_advanced_rag: bool = False
    enable_concurrent_threads: bool = False
    enable_voice: bool = False
    enable_tui: bool = False

    @classmethod
    def from_profile(cls, profile: str) -> "V2Config":
        """Create config from preset profile."""
        profiles = {
            "minimal": cls(),  # Everything off
            "standard": cls(llm_tier="standard", enable_memory=True, enable_task_queue=True),
            "coding": cls(llm_tier="advanced", enable_memory=True, enable_task_queue=True,
                         enable_continuous_execution=True, enable_state_machine=True,
                         enable_manifest=True, enable_adaptive_prompts=True,
                         enable_self_eval=True),
            "coding_light": cls(llm_tier="standard", enable_memory=True,
                               enable_task_queue=True, enable_state_machine=True),
            "chat": cls(llm_tier="standard", enable_memory=True, enable_persona=True,
                       enable_advanced_rag=True),
        }
        return profiles.get(profile, cls())

    def validate_tier(self) -> list[str]:
        """Warn about features above the LLM tier."""
        warnings = []
        advanced_only = ["enable_learning", "enable_dynamic_tools",
                        "enable_self_eval", "enable_concurrent_threads"]
        standard_plus = ["enable_continuous_execution", "enable_state_machine",
                        "enable_manifest", "enable_adaptive_prompts"]
        if self.llm_tier == "basic":
            for flag in standard_plus + advanced_only:
                if getattr(self, flag):
                    warnings.append(f"{flag} requires 'standard' tier or above")
        elif self.llm_tier == "standard":
            for flag in advanced_only:
                if getattr(self, flag):
                    warnings.append(f"{flag} requires 'advanced' tier")
        return warnings


class FeatureRegistry:
    """Tracks which features are registered, enabled, and healthy."""

    def register(self, name: str, mixin_class: type, depends_on: list[str] = None): ...
    def is_enabled(self, name: str) -> bool: ...
    def get_status(self) -> dict[str, str]: ...  # "enabled", "disabled", "degraded"
```

**Deliverable:** Config system that all features depend on. Ship this first, everything else uses it.

### P0.2: Enhanced Tool Registry (4 hours)

**Why FIRST:** Current `_TOOL_REGISTRY` is a global dict. Two agents running simultaneously overwrite each other's tools. Must fix before concurrent threads or even robust single-agent operation.

```python
class ToolRegistry:
    """Instance-scoped tool registry. Replaces global _TOOL_REGISTRY."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, func, name=None, description=None): ...
    def get_tools(self, filter_by_state=None) -> list[ToolDef]: ...
    def get_tool(self, name: str) -> ToolDef: ...

# Each agent gets its own registry
agent1 = CodeAgent(tool_registry=ToolRegistry())
agent2 = ChatAgent(tool_registry=ToolRegistry())
# No conflicts!
```

### P0.3: Structured Logging (2 hours)

**Why FIRST:** Every component needs to log. Structured logs are parseable by observability later.

```python
import structlog
logger = structlog.get_logger()
logger.info("tool_call", tool="read_file", path="/src/main.py", duration_ms=12)
```

### P1.1: LLM Router + Tier System (6 hours)

**Why before agent features:** The entire composability model depends on knowing which LLM tier is active, so features can gracefully degrade or disable themselves.

```python
class LLMRouter:
    """Route requests to the right LLM based on task complexity and available models."""

    def __init__(self, config: V2Config):
        self.tier = config.llm_tier
        self.providers = self._discover_providers()

    def route(self, task_complexity: str) -> LLMClient:
        """Select best available LLM for this task complexity."""
        if task_complexity == "simple" and "basic" in self.providers:
            return self.providers["basic"]      # 0.6B -- fast, cheap
        elif task_complexity == "standard" and "standard" in self.providers:
            return self.providers["standard"]   # 7B -- balanced
        elif "advanced" in self.providers:
            return self.providers["advanced"]   # 30B+ -- full reasoning
        else:
            return self.providers["cloud"]      # Claude fallback

    def _discover_providers(self) -> dict:
        """Check what's available: Lemonade models, cloud APIs."""
        # Auto-detect running Lemonade Server models
        # Check for cloud API keys (Claude, OpenAI)
        # Return available providers by tier
```

### P1.2: Error Recovery - Basic Retry (4 hours)

**Why before agent features:** Every LLM call can fail. Without retry, everything is fragile.

```python
class RetryEngine:
    """Wraps LLM calls with exponential backoff + jitter."""
    def __init__(self, max_retries=3, base_delay=1.0): ...
    async def execute(self, fn, *args, **kwargs): ...

class CircuitBreaker:
    """Prevents cascade failures when a provider is down."""
    def __init__(self, failure_threshold=5, recovery_timeout=30): ...
```

### P1.3: Input Sanitization (3 hours)

**Why before agent features:** Security baseline. All file paths, user inputs, tool arguments validated.

### P1.4: Checkpoint/Serialization Base (4 hours)

**Why before agent features:** Continuous execution, memory, and task queue ALL need the ability to serialize/deserialize state. Build the base layer once.

```python
class Serializable(Protocol):
    """Any component that can save/restore state implements this."""
    def to_checkpoint(self) -> dict: ...
    @classmethod
    def from_checkpoint(cls, data: dict) -> Self: ...

class CheckpointStore:
    """Persistent checkpoint storage (SQLite)."""
    def save(self, component_id: str, data: dict): ...
    def load(self, component_id: str) -> dict | None: ...
    def list_checkpoints(self) -> list[CheckpointInfo]: ...
```

### P2.1: Memory Infrastructure (6 hours)

**Why before memory features:** The storage layer (SQLite schemas, FAISS index management, directory conventions) must exist before any memory tier can be built on top.

```python
class MemoryStore:
    """Base storage for all memory tiers."""
    def __init__(self, db_path: str): ...
    # SQLite tables: interactions, tool_calls, file_operations, state_snapshots
    # FAISS index management for vector search
    # Directory conventions: ~/.gaia/memory/
```

### P2.2: Task Infrastructure (4 hours)

**Why before task features:** Task model, SQLite schema, and basic CRUD are needed by continuous execution, task queue, AND concurrent threads.

### P2.3: Mixin Lifecycle Protocol (3 hours)

**Why before agent features:** Every V2 mixin needs a standard way to initialize, enable/disable, and clean up. Without this, each mixin invents its own pattern.

```python
class V2Mixin:
    """Base protocol for all V2 feature mixins."""

    def v2_init(self, config: V2Config):
        """Called during agent init. Check feature flag, set up or no-op."""
        if not self._is_enabled(config):
            self._disabled = True
            return  # No-op: zero overhead
        self._setup()

    def _is_enabled(self, config: V2Config) -> bool:
        """Check if this feature should be active."""
        ...

    def v2_cleanup(self):
        """Called on agent shutdown. Save state, close connections."""
        ...
```

### Prerequisites Summary

| ID | Component | Hours | Why It's First | What It Unblocks |
|----|-----------|-------|----------------|-----------------|
| P0.1 | V2Config + Feature Registry | 4h | All features register flags here | Everything |
| P0.2 | Enhanced Tool Registry | 4h | Global registry breaks concurrent threads | All agents |
| P0.3 | Structured Logging | 2h | Every component needs logging | Debugging, observability |
| P1.1 | LLM Router + Tier System | 6h | Composability depends on tier awareness | All LLM-dependent features |
| P1.2 | Error Recovery (basic) | 4h | Every LLM call needs retry | All LLM calls |
| P1.3 | Input Sanitization | 3h | Security baseline | All user-facing features |
| P1.4 | Checkpoint/Serialization | 4h | State persistence for all components | Memory, continuous exec, tasks |
| P2.1 | Memory Infrastructure | 6h | Storage layer for all memory tiers | All memory features |
| P2.2 | Task Infrastructure | 4h | Task model used by queue, execution, concurrent threads | Task queue, continuous exec |
| P2.3 | Mixin Lifecycle Protocol | 3h | Standard init/cleanup for all mixins | All V2 mixins |

**Total prerequisite time: ~40 hours (~1.7 days)**

**Nothing else starts until Layer 0 is complete. Layer 1 and Layer 2 can partially overlap.**

---

## Composability Across LLM Tiers

**Critical design principle:** Gaia Code must work with LLMs ranging from 0.6B parameters to 200B+ parameters. Each tier gets a progressively capable agent. No feature should crash the agent -- if the LLM can't handle a feature, the feature degrades gracefully to a simpler behavior or disables itself.

### Three Tiers, Three Agents

```
┌─────────────────────────────────────────────────────────────────────┐
│ TIER 3: ADVANCED (30B+ or Cloud)                                    │
│ "Full autonomous agent -- exceeds Claude Code"                      │
│                                                                     │
│ Everything in Tier 2, PLUS:                                         │
│ ✅ Learning Loop (pattern extraction, adaptation)                   │
│ ✅ Dynamic Tool Creation (generates new tools from patterns)        │
│ ✅ Self-Evaluation (multi-dimensional quality scoring)              │
│ ✅ Proactive Analysis (security, complexity, dead code)             │
│ ✅ Concurrent Task Threads (parallel work, single brain)            │
│ ✅ Adaptive Prompts (learned conventions + memory context)          │
│ ✅ Semantic Memory (consolidated patterns with confidence)          │
│                                                                     │
│ LLMs: Qwen3-Coder-30B, Claude Opus 4.6, GPT-4+                   │
│ Profile: V2Config.from_profile("coding")                           │
├─────────────────────────────────────────────────────────────────────┤
│ TIER 2: STANDARD (7B-30B)                                           │
│ "Capable coding agent -- matches current gaia-code"                 │
│                                                                     │
│ Everything in Tier 1, PLUS:                                         │
│ ✅ Continuous Execution (run until done, no step limit)             │
│ ✅ State Machine (Requirements → Planning → Impl → Test → Debug)   │
│ ✅ Episodic Memory (session recall across conversations)            │
│ ✅ Task Queue (decompose, prioritize, track)                        │
│ ✅ Manifest (file tracking, dependency graph)                       │
│ ✅ Quality Gates (syntax, lint, test, build)                        │
│ ✅ Checkpoint/Resume (survive context limits)                       │
│                                                                     │
│ LLMs: Qwen3-7B, Llama-8B, Mistral-7B                              │
│ Profile: V2Config.from_profile("coding_light")                     │
├─────────────────────────────────────────────────────────────────────┤
│ TIER 1: BASIC (0.6B-3B)                                            │
│ "Simple tool executor -- code completion and formatting"             │
│                                                                     │
│ ✅ File Operations (read, write, edit, search)                      │
│ ✅ Terminal Execution (run commands, capture output)                 │
│ ✅ Git Operations (status, diff, commit)                            │
│ ✅ Basic Error Recovery (retry with backoff)                        │
│ ✅ Input Sanitization (security baseline)                           │
│ ✅ Structured Logging                                               │
│ ❌ No memory (can't summarize sessions reliably)                    │
│ ❌ No state machine (can't follow multi-step reasoning)             │
│ ❌ No continuous execution (needs step limits)                      │
│                                                                     │
│ LLMs: Qwen3-0.6B, Phi-3-mini, TinyLlama                          │
│ Profile: V2Config.from_profile("minimal")                          │
└─────────────────────────────────────────────────────────────────────┘
```

### How Features Degrade by Tier

Every feature has three modes: **full**, **degraded**, and **off**. The mixin checks the LLM tier at init and selects the appropriate mode.

| Feature | Tier 1 (Basic) | Tier 2 (Standard) | Tier 3 (Advanced) |
|---------|:-:|:-:|:-:|
| File Operations | **Full** | **Full** | **Full** |
| Terminal Execution | **Full** | **Full** | **Full** |
| Git Operations | **Full** | **Full** | **Full** |
| Error Recovery | Retry only | Retry + Circuit Breaker | Full (+ Fallback Chain) |
| Memory | **Off** | Episodic only (Tier 1-2) | Full 4-tier |
| Task Queue | **Off** | **Full** | **Full** |
| Continuous Execution | **Off** (use max_steps) | **Full** (run until done) | **Full** + self-healing |
| State Machine | **Off** | **Full** (6 states) | **Full** + learned transitions |
| Manifest | **Off** | **Full** | **Full** + architecture analysis |
| Quality Gates | Syntax only | Syntax + Lint + Test | All 8 gates |
| Adaptive Prompts | **Off** | **Off** | **Full** |
| Learning Loop | **Off** | **Off** | **Full** |
| Dynamic Tools | **Off** | **Off** | **Full** |
| Self-Evaluation | **Off** | **Off** | **Full** |
| Concurrent Threads | **Off** | **Off** | **Full** |

### Graceful Degradation Pattern

```python
class ContinuousExecutionMixin(V2Mixin):
    """Runs until task verified complete. Degrades to max_steps for basic tier."""

    def v2_init(self, config: V2Config):
        if config.llm_tier == "basic":
            # Basic LLMs can't assess completion reliably
            # Fall back to max_steps behavior
            self._mode = "max_steps"
            self._max_steps = 100
        elif config.llm_tier == "standard":
            # Standard LLMs can assess completion but may need help
            self._mode = "continuous"
            self._max_steps = 500  # Safety cap
            self._enable_quality_gates = True
        else:
            # Advanced LLMs get full autonomous mode
            self._mode = "continuous"
            self._max_steps = None  # No limit
            self._enable_quality_gates = True
            self._enable_self_healing = True

class MemoryMixin(V2Mixin):
    """4-tier memory. Degrades based on LLM capability."""

    def v2_init(self, config: V2Config):
        if config.llm_tier == "basic":
            self._disabled = True  # Can't summarize reliably
        elif config.llm_tier == "standard":
            # Episodic memory only (no consolidation to semantic)
            self._tiers = ["working", "episodic"]
        else:
            # Full 4-tier memory
            self._tiers = ["working", "episodic", "semantic", "universal"]
```

### Why Composability Matters for `gaia code`

1. **Development machines vary.** Not every developer has a 30B model running. A coding agent that only works with large LLMs is useless on a laptop.
2. **Tasks vary.** "Fix this typo" doesn't need a 30B model. "Build a full-stack app" does. The agent should pick the right tier.
3. **Cost optimization.** Using a 0.6B model for simple file operations saves 100x the cost/latency of using a cloud model.
4. **Incremental adoption.** Teams can start with Tier 1 (basic tools, small LLM) and upgrade to Tier 3 as they validate each feature.
5. **Offline fallback.** If cloud API is down, agent degrades to local LLM capabilities instead of stopping.

---

## Competitive Landscape

### Current State of the Art (February 2026)

| Agent | Strengths | Limitations |
|-------|-----------|-------------|
| **Claude Code** (Anthropic) | Best reasoning, 200K context, sub-agents, MCP, CLAUDE.md memory | Cloud-only, no persistent learning, no quality gates, context limits, no local LLM |
| **Devin** (Cognition) | Most autonomous, full dev environment, end-to-end tasks | Expensive, closed-source, slow, no local execution |
| **Codex** (OpenAI) | Deterministic multi-step, good test iteration | Limited context, less reasoning than Claude |
| **Cursor** (Anysphere) | Best IDE integration, Composer mode, fast iteration | IDE-only, limited autonomy, no CLI agent mode |
| **Cline** (Open Source) | Plan+Act modes, model-agnostic, open-source | Less capable reasoning, no persistent memory |
| **Aider** (Open Source) | Git-native, CLI-first, excellent refactoring | Limited to single-turn, no continuous execution |
| **Amp** (Sourcegraph) | Deep mode extended reasoning, repo-scale search | Limited tool execution, newer entry |

### What 4% of GitHub Commits Tells Us

Claude Code currently authors ~4% of all public GitHub commits, projected to reach 20%+ by end of 2026. This means:
1. Autonomous coding agents are already mainstream
2. The bar for "useful" is well-established
3. To win, we must exceed this bar significantly

### The Gap No One Has Filled

**No existing agent combines all of these:**
- Runs locally on user hardware (privacy, speed, cost)
- Learns permanently across sessions (never repeats mistakes)
- Executes continuously without limits (no step/token caps)
- Creates its own tools (dynamic capability expansion)
- Self-evaluates and iterates until quality gates pass
- Works offline with local LLMs on AMD hardware

**Gaia Code fills this gap.**

---

## Claude Code: Capabilities & Limitations

### What Claude Code Does Well (Must Match)

| Capability | How Claude Code Does It | Gaia Code Must Match |
|------------|------------------------|---------------------|
| **File Operations** | Read, Write, Edit (exact string replacement), Glob (pattern match), Grep (content search) | YES -- extend with AST-aware editing |
| **Terminal Execution** | Bash tool with timeout, background tasks | YES -- extend with sandboxing, output parsing |
| **Git Integration** | Full git workflow (status, diff, commit, push, PR via `gh`) | YES -- extend with semantic commit messages, auto-branching |
| **Code Search** | Glob for files, Grep (ripgrep) for content, both support regex | YES -- extend with semantic search, symbol resolution |
| **Context Management** | CLAUDE.md files, conversation compression at context limits | YES -- extend with unlimited context via checkpoint/resume |
| **Sub-Agents** | Task tool spawns parallel agents with isolated contexts | YES -- extend with persistent sub-agent results, shared state |
| **MCP Integration** | Connect to external tools (Jira, Slack, Google Drive, etc.) | YES -- already have MCP in GAIA |
| **Web Search/Fetch** | WebSearch + WebFetch tools for real-time information | YES |
| **Skills/Hooks** | Custom slash commands, pre/post action hooks | YES -- extend with learnable skills |
| **Multi-Surface** | Terminal, VS Code, JetBrains, Desktop, Web, iOS | LATER -- start with Terminal CLI |
| **Error Recovery** | When code fails, automatically debugs and iterates | YES -- extend with systematic debugging state machine |
| **Jupyter Notebooks** | Can read and edit notebook cells | YES |

### Where Claude Code Falls Short (Must Exceed)

| Limitation | Impact | Gaia Code Solution |
|------------|--------|-------------------|
| **No persistent memory** | CLAUDE.md is static, manually maintained. Agent forgets everything between sessions. | 4-tier memory: Working → Episodic → Semantic → Universal. Agent remembers patterns, preferences, project conventions permanently. |
| **Context window ceiling** | At ~200K tokens, conversation gets compressed. Complex multi-file tasks lose context. | Checkpoint/resume engine. Agent saves full state, resumes across unlimited context windows. Never loses context. |
| **No quality gates** | Agent declares "done" based on its own judgment. No automated verification. | Mandatory quality gates: syntax check, type check, lint, test runner, security scan. Agent cannot declare "done" until all gates pass. |
| **No state machine** | Jumps between tasks without structured workflow. Complex tasks become chaotic. | 6-state machine: Requirements → Planning → Implementation → Testing → Debug → Review. Each state has entry/exit criteria. |
| **No learning loop** | Same mistakes repeated across sessions. No pattern extraction. | 6-step learning loop: Execute → Record → Feedback → Extract → Consolidate → Adapt. Mistakes never repeated. |
| **No project manifest** | Doesn't maintain knowledge of project structure, dependencies, relationships. | Live manifest: every file tracked, dependency graph, import analysis. Agent knows what it built and why. |
| **No dynamic tools** | Fixed set of tools. Can't create new tools for project-specific patterns. | Tool builder: agent detects repeated patterns, generates reusable tools, stores in skill library. |
| **No self-evaluation** | Agent can't assess quality of its own output beyond "looks right." | Self-evaluation pass: agent rates output on correctness, completeness, style, security. Triggers additional passes when quality is low. |
| **Cloud-only execution** | Requires Anthropic API. No offline mode. No local hardware utilization. | Local-first: runs on AMD Ryzen AI with NPU acceleration via Lemonade Server. Works offline. Zero API cost for local LLM. |
| **No cost optimization** | Always uses full Opus model. No model selection based on task complexity. | Tiered LLM: simple tasks use 0.6B model, standard tasks use 7B, complex reasoning uses 30B+. 100x cost reduction for simple tasks. |
| **No proactive analysis** | Reactive only -- waits for errors. Doesn't proactively find issues. | Proactive scanning: AST analysis, type inference, security audit, performance profiling before user asks. |
| **Step/token limits** | Even with sub-agents, each sub-agent has context limits. Long tasks degrade. | Truly unlimited: checkpoint/resume means agent can work for hours/days on complex tasks without degradation. |
| **No task decomposition persistence** | Sub-agents are fire-and-forget. Results don't persist as structured data. | Persistent task queue: tasks decomposed, tracked, checkpointed. Sub-agent results stored and searchable. |

---

## Core Features: What Gaia Code Must Have

### Tier 1: Foundation (Must Ship Day 1)

These are table-stakes features that every coding agent needs:

#### F1. File System Operations
```
READ    - Read file with line numbers, offset, limit
WRITE   - Create/overwrite file
EDIT    - Exact string replacement (like Claude Code)
GLOB    - Pattern-based file search
GREP    - Content search with regex (ripgrep-based)
AST     - Parse code into AST for structural operations [NEW]
SYMBOL  - Find definition/references by symbol name [NEW]
```

#### F2. Terminal Execution
```
BASH    - Execute command with timeout, capture output
BG      - Background execution with async result retrieval
SANDBOX - Sandboxed execution (restricted filesystem, network) [NEW]
PARSE   - Structured output parsing (JSON, table, error) [NEW]
```

#### F3. Git Operations
```
STATUS  - Working tree status
DIFF    - Staged and unstaged changes
COMMIT  - Stage, commit with semantic message
BRANCH  - Create, switch, delete branches
PR      - Create pull request via gh CLI
LOG     - Recent commit history
MERGE   - Merge with conflict resolution [NEW]
```

#### F4. Code Intelligence
```
SEARCH   - Semantic code search (not just text matching) [NEW]
SYMBOLS  - List all symbols (classes, functions, variables) in file [NEW]
IMPORTS  - Analyze import graph [NEW]
TYPES    - Infer/check types without full type checker [NEW]
DEPS     - Analyze project dependencies (package.json, requirements.txt) [NEW]
```

#### F5. Context Management
```
CLAUDE_MD  - Read CLAUDE.md / GAIA.md project instructions
COMPRESS   - Summarize conversation when approaching context limit
CHECKPOINT - Save full agent state to disk [NEW]
RESUME     - Restore agent state from checkpoint [NEW]
```

#### F6. Error Handling
```
RETRY     - Exponential backoff with jitter for transient failures
CIRCUIT   - Circuit breaker for persistent failures
FALLBACK  - Fallback chain (try local LLM → cloud LLM → cached response)
RECOVER   - Parse error messages, identify root cause, suggest fix [NEW]
```

### Tier 2: Autonomous Operation (What Makes It Better Than Claude Code)

#### F7. Continuous Execution Engine
**The single most important differentiator.**

Claude Code runs until it hits context limits or max turns. Gaia Code runs until the task is **verified complete**.

```python
class ContinuousExecutionEngine:
    """Runs until task verified complete. No step limit. No context limit."""

    def execute(self, task: str):
        while not self.is_complete(task):
            # 1. Plan next action
            action = self.plan_next_action(task)

            # 2. Execute action
            result = self.execute_action(action)

            # 3. Evaluate result
            evaluation = self.evaluate_result(result, task)

            # 4. If approaching context limit, checkpoint and resume
            if self.context_usage > 0.8:
                self.checkpoint()  # Save full state
                self.compress()    # Summarize conversation
                self.resume()      # Continue with fresh context

            # 5. If quality gates fail, enter debug state
            if not evaluation.passes_quality_gates:
                self.enter_debug_state(evaluation.failures)
```

**Key properties:**
- No `max_steps` -- runs until verified complete
- Checkpoint/resume -- survives context window limits
- Quality gates -- cannot declare "done" prematurely
- Self-healing -- automatically debugs and retries on failure

#### F8. Quality Gates
**Automated validation at every stage.**

```python
class QualityGates:
    """Mandatory checks before task completion."""

    gates = [
        SyntaxCheckGate(),      # Parse all modified files (zero tolerance)
        TypeCheckGate(),         # Run mypy/pyright/tsc if configured
        LintGate(),             # Run project linter (black, eslint, etc.)
        TestGate(),             # Run affected tests (pytest, jest, etc.)
        SecurityGate(),         # Basic security scan (no secrets, no injection)
        ImportGate(),           # All imports resolve
        BuildGate(),            # Project builds successfully
        SelfReviewGate(),       # LLM reviews its own changes [NEW]
    ]

    def evaluate(self, changes: list[FileChange]) -> GateResult:
        for gate in self.gates:
            result = gate.check(changes)
            if not result.passed:
                return GateResult(passed=False, failures=[result])
        return GateResult(passed=True)
```

**Key insight:** Claude Code has no automated quality gates. It relies on its own judgment, which means it can (and does) produce code with lint errors, failing tests, or security issues. Gaia Code **cannot declare done until all gates pass**.

#### F9. State Machine
**Structured workflow for complex tasks.**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ Requirements │────▶│   Planning   │────▶│ Implementation   │
│  Gathering   │     │              │     │                  │
└──────────────┘     └──────────────┘     └──────────────────┘
       ▲                                          │
       │                                          ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│    Review    │◀────│    Debug     │◀────│    Testing       │
│              │     │              │     │                  │
└──────────────┘     └──────────────┘     └──────────────────┘
```

**Each state has:**
- Entry criteria (what must be true to enter)
- Exit criteria (what must be true to leave)
- State-specific tools (only relevant tools visible)
- State-specific prompts (focused instructions)

**Why this matters:** Claude Code jumps around. It might start implementing before understanding requirements, or skip testing entirely. The state machine enforces discipline.

#### F10. Persistent Memory (4 Tiers)

```
Tier 1: Working Memory (RAM)
├── Current conversation context
├── Active file contents
└── Tool results cache

Tier 2: Episodic Memory (FAISS + JSON)
├── Session summaries (what happened in each session)
├── Key decisions made
├── Files created/modified per session
└── Searchable via vector similarity

Tier 3: Semantic Memory (SQLite + confidence scores)
├── Extracted patterns ("this project uses pytest, not unittest")
├── User preferences ("prefer pathlib over os.path")
├── Project conventions ("all API routes in routes/ directory")
├── Confidence scores that decay over time
└── Contradiction detection

Tier 4: Universal Knowledge DB
├── Every interaction stored
├── Every tool call and result
├── Every file operation
├── Full audit trail
└── Cross-session search
```

**Why this matters:** Claude Code forgets everything between sessions. CLAUDE.md is static and manually maintained. Gaia Code **learns permanently** and **never forgets**.

#### F11. Learning Loop

```
Execute ──▶ Record ──▶ Feedback ──▶ Extract ──▶ Consolidate ──▶ Adapt
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

1. **Execute**: Agent performs action
2. **Record**: Full context captured (input, output, state, tools used)
3. **Feedback**: Collect explicit (user correction) + implicit (test pass/fail) + automated (quality gates)
4. **Extract**: Identify patterns from feedback ("every time I use subprocess, user corrects to pathlib")
5. **Consolidate**: Merge patterns into semantic memory with confidence scores
6. **Adapt**: Inject learned patterns into future prompts

**Concrete example:**
```
Session 1: Agent uses os.path.join() for file paths
           User corrects: "Use pathlib.Path instead"
           Pattern extracted: {os.path → pathlib, confidence: 0.3}

Session 2: Agent starts to use os.path.join()
           Memory injects: "User preference: use pathlib.Path, not os.path"
           Agent uses pathlib.Path instead
           Correction not needed → confidence increases to 0.6

Session 5: Pattern fully consolidated at confidence 0.9
           Agent always uses pathlib.Path without reminder
```

**Why this matters:** Claude Code makes the same mistakes across sessions. Gaia Code improves every session.

#### F12. Project Manifest

```python
class ProjectManifest:
    """Live knowledge graph of the entire project."""

    files: dict[str, FileInfo]          # Every file with metadata
    dependencies: DependencyGraph       # Import/require graph
    symbols: SymbolTable                # All classes, functions, variables
    test_map: dict[str, list[str]]      # Source file → test files
    change_history: list[ChangeEntry]   # What changed and why

    def get_affected_tests(self, changed_files: list[str]) -> list[str]:
        """Given changed files, return tests that need to run."""

    def get_dependency_chain(self, file: str) -> list[str]:
        """Given a file, return all files that depend on it."""

    def detect_circular_imports(self) -> list[Cycle]:
        """Find circular import chains."""
```

**Why this matters:** Claude Code re-discovers project structure every session. Gaia Code maintains a live manifest that grows across sessions.

#### F13. Adaptive Prompts

```python
class PromptComposer:
    """Builds dynamic prompts based on state + memory + task."""

    layers = [
        ImmutableCore(),          # "You are gaia-code, an autonomous coding agent"
        StateModule(),            # Different instructions per state (planning vs testing vs debugging)
        LearnedInstructions(),    # Injected from memory ("use pathlib", "prefer FastAPI", etc.)
        MemoryContext(),          # Relevant past sessions, decisions, patterns
        ProjectContext(),         # From manifest: files, dependencies, conventions
        TaskContext(),            # Current task requirements, progress, remaining work
    ]
```

**Why this matters:** Claude Code uses a static system prompt. Gaia Code's prompt dynamically adapts to the project, the user's preferences, and the current state of work.

### Tier 3: Superpowers (What Makes It Unprecedented)

#### F14. Dynamic Tool Creation

When Gaia Code detects a repeated pattern (e.g., "create FastAPI route with auth middleware"), it generates a reusable tool:

```python
# Agent detects pattern after 3 occurrences:
# "Create route → add middleware → add schema → add test"

# Agent generates:
@tool
def create_fastapi_route(name: str, method: str, auth_required: bool):
    """Create a complete FastAPI route with middleware, schema, and test."""
    # Generated code for route
    # Generated code for middleware
    # Generated code for Pydantic schema
    # Generated code for pytest test
```

**Security:** All generated tools pass AST validation. No `eval()`, no `exec()`, no arbitrary code execution. Sandboxed testing before promotion to tool registry.

#### F15. Concurrent Task Threads (NOT Multi-Agent)

> **Why not multi-agent?** Multi-agent systems are complex, brittle, and create coordination problems (merge conflicts, inconsistent style, debugging nightmares). Instead, Gaia Code is a **single agent** that can spawn **concurrent task threads** with isolated state. One brain, one memory, one set of preferences -- but multiple hands.

For complex tasks, Gaia Code runs concurrent threads:

```
User: "Build a full-stack social media app"

Single Agent (one brain, one knowledge base, one plan):
├── Thread 1: Backend (FastAPI endpoints, database models, auth)
│   └── State: implementing, progress: 60%
├── Thread 2: Frontend (React components, routing, state management)
│   └── State: implementing, progress: 40%
├── Thread 3: Tests (running in background, watching for new files)
│   └── State: testing, progress: ongoing
└── Main Thread: Orchestrating, checking progress, resolving conflicts

Key differences from multi-agent:
- ONE knowledge base (no sync issues between agents)
- ONE set of conventions (consistent style across all output)
- ONE plan (no merge conflicts between agent plans)
- Shared context (Thread 2 knows what Thread 1 built)
- Simple coordination (main thread resolves conflicts)
```

Think of it like a developer with multiple terminal tabs open -- one person, parallel execution, coherent output.

#### F16. Self-Evaluation Pass

After completing a task, agent reviews its own work:

```python
class SelfEvaluation:
    dimensions = [
        "correctness",      # Does the code do what was asked?
        "completeness",     # Are all requirements addressed?
        "style",            # Does it follow project conventions?
        "security",         # Any security concerns?
        "performance",      # Any obvious performance issues?
        "testability",      # Is the code testable?
        "maintainability",  # Will this be easy to maintain?
    ]

    def evaluate(self, changes, requirements) -> EvalResult:
        score = self.llm.evaluate(changes, requirements, self.dimensions)
        if score.overall < 0.7:
            return EvalResult(action="revise", feedback=score.feedback)
        return EvalResult(action="approve", score=score)
```

#### F17. Proactive Analysis

Don't wait for errors. Find them before the user does:

```python
class ProactiveAnalyzer:
    """Runs automatically after code changes."""

    analyzers = [
        DeadCodeDetector(),       # Find unused imports, variables, functions
        ComplexityAnalyzer(),     # Flag functions with cyclomatic complexity > 10
        SecurityScanner(),        # OWASP top 10 checks
        DependencyChecker(),      # Outdated or vulnerable dependencies
        TestCoverageAnalyzer(),   # Untested code paths
        TypeConsistencyChecker(), # Type mismatches across function boundaries
    ]
```

#### F18. LLM Tier Optimization

Use the right model for the right task:

```python
class LLMRouter:
    """Route tasks to appropriate LLM based on complexity."""

    tiers = {
        "basic": "Qwen3-0.6B",          # File ops, simple edits, formatting
        "standard": "Qwen3-7B",          # Code generation, test writing, debugging
        "advanced": "Qwen3-Coder-30B",   # Architecture, complex reasoning, multi-file
        "cloud": "claude-opus-4-6",      # Fallback for tasks beyond local LLM capability
    }

    def route(self, task: Task) -> str:
        complexity = self.assess_complexity(task)
        if complexity.reasoning_depth == "shallow":
            return "basic"
        elif complexity.multi_file and complexity.reasoning_depth == "deep":
            return "advanced"
        elif complexity.exceeds_local_capability:
            return "cloud"  # Only when local LLM can't handle it
        return "standard"
```

**Cost impact:**
- Claude Code: ~$0.15-0.75 per complex task (Opus pricing)
- Gaia Code (local): $0.00 per task (AMD hardware, zero API cost)
- Gaia Code (cloud fallback): ~$0.15 per complex task (only when needed)

---

## Differentiators: How Gaia Code Exceeds Claude Code

### Head-to-Head Comparison

| Dimension | Claude Code | Gaia Code | Advantage |
|-----------|-------------|-----------|-----------|
| **Execution duration** | Until context fills (~200K tokens) | Until task verified complete (unlimited) | **Gaia Code** -- never stops mid-task |
| **Memory** | CLAUDE.md (static, manual) | 4-tier persistent memory (automatic) | **Gaia Code** -- learns permanently |
| **Quality assurance** | Agent judgment only | Automated quality gates (syntax, type, lint, test, security) | **Gaia Code** -- verified correct |
| **Workflow structure** | Unstructured (agent decides) | 6-state machine with entry/exit criteria | **Gaia Code** -- disciplined process |
| **Learning** | None across sessions | 6-step learning loop | **Gaia Code** -- improves every session |
| **Tool creation** | Fixed tool set | Dynamic tool generation | **Gaia Code** -- expanding capabilities |
| **Project knowledge** | Re-discovers each session | Persistent manifest | **Gaia Code** -- instant project understanding |
| **Cost** | $0.15-0.75/task (cloud API) | $0.00/task (local) + cloud fallback | **Gaia Code** -- 100x cheaper |
| **Privacy** | All code sent to Anthropic | Local-first, code stays on device | **Gaia Code** -- zero data exposure |
| **Offline** | Requires internet | Works offline with local LLM | **Gaia Code** -- works anywhere |
| **Hardware optimization** | Generic cloud GPU | AMD Ryzen AI NPU acceleration | **Gaia Code** -- optimized for hardware |
| **Self-evaluation** | None | Multi-dimensional quality scoring | **Gaia Code** -- catches own mistakes |
| **Proactive analysis** | Reactive only | Automatic scanning after changes | **Gaia Code** -- prevents bugs |
| **Parallel execution** | Fire-and-forget sub-agents | Concurrent task threads with shared knowledge | **Gaia Code** -- coherent parallel work |
| **Reasoning model** | Single model (Opus) | Tiered (0.6B → 7B → 30B → cloud) | **Gaia Code** -- right model for right task |

### The Killer Feature: Compound Learning

Over time, Gaia Code gets dramatically better:

```
Week 1:  Agent learns project structure, conventions, test patterns
Week 2:  Agent has custom tools for common project patterns
Week 4:  Agent anticipates requirements, skips clarification for known patterns
Week 8:  Agent has deep knowledge of entire codebase, makes architecture-level suggestions
Week 12: Agent operates at 10x speed of Week 1, with near-zero errors
```

**No other coding agent does this.** Claude Code is equally capable in session 1 and session 1000.

---

## Architecture Overview

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GAIA CODE CLI                             │
│  $ gaia code "Build a REST API with auth and tests"             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  CONTINUOUS EXECUTION ENGINE                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Checkpoint│  │ Quality  │  │  Context  │  │   Task   │       │
│  │ Manager  │  │  Gates   │  │  Manager  │  │  Queue   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      STATE MACHINE                               │
│  Requirements → Planning → Implementation → Testing → Debug     │
│                                                     → Review    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    TOOL LAYER                                    │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │  File   │ │Terminal │ │   Git   │ │  Code   │ │ Dynamic │ │
│  │  Ops    │ │  Exec   │ │   Ops   │ │  Intel  │ │  Tools  │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    INTELLIGENCE LAYER                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│  │Adaptive │ │Learning │ │  Self   │ │Proactive│ │Manifest │ │
│  │Prompts  │ │  Loop   │ │  Eval   │ │Analysis │ │Tracker  │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    MEMORY LAYER                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Working  │  │ Episodic │  │ Semantic │  │Universal │       │
│  │ (RAM)    │  │ (FAISS)  │  │ (SQLite) │  │  (All)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    LLM LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Lemonade    │  │   Cloud      │  │   Router     │          │
│  │  Server      │  │   (Claude,   │  │   (picks     │          │
│  │  (Local AMD) │  │   OpenAI)    │  │   best LLM)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Shared Agent State: The Foundation for RAC

**Critical design:** All agents in a recursion tree share a single `SharedAgentState` instance. This enables coordination without complexity.

```python
@dataclass
class SharedAgentState:
    """Singleton shared by ALL agents in the recursion tree.

    When main agent spawns sub-agents via agent_query(), they all
    receive the SAME shared state instance. Changes by one agent
    are immediately visible to all others.
    """

    # DATABASE LAYER (persisted to disk)
    memory_db: MemoryDB              # Working memory: file cache, active state
    knowledge_db: KnowledgeDB        # Cross-session: insights, preferences, learnings

    # PROJECT STATE (session-scoped, persisted)
    manifest: ProjectManifest        # Files, APIs, schemas, decisions
    plan: MasterPlan                 # Hierarchical task tree with owners

    # ORCHESTRATION (runtime state)
    call_stack: AgentCallStack       # Who's doing what, at what depth
    message_queue: MessageQueue      # Async agent ↔ user, agent ↔ agent

    # CAPABILITY REGISTRIES
    tools_db: ToolsDB               # 78 core + learned tools
    skills_db: SkillsDB             # Learned workflows
    agents_db: AgentRegistry        # 7 core + learned specialists

    # THREAD SAFETY (single-threaded + async, locks for safety)
    _locks: dict[str, threading.Lock]

    # SESSION METADATA
    session_id: str
    session_start: datetime
    session_goal: str

    def __init__(self, session_id: str, goal: str):
        self.session_id = session_id
        self.session_goal = goal
        self.session_start = datetime.now()

        # Initialize databases (these persist to ~/.gaia/)
        self.memory_db = MemoryDB()
        self.knowledge_db = KnowledgeDB()
        self.manifest = ProjectManifest()
        self.plan = MasterPlan(goal=goal)
        self.call_stack = AgentCallStack()
        self.message_queue = MessageQueue()
        self.tools_db = ToolsDB()
        self.skills_db = SkillsDB()
        self.agents_db = AgentRegistry()

        # Thread locks (used during async I/O operations)
        self._locks = {
            "manifest": threading.Lock(),
            "plan": threading.Lock(),
            "knowledge_db": threading.Lock(),
            "memory_db": threading.Lock(),
            "call_stack": threading.Lock()
        }

    def lock(self, resource: str):
        """Context manager for thread-safe access during async operations."""
        return self._locks[resource]
```

**Storage layout:**

```
~/.gaia/
├── memory/
│   ├── memory.db            # Working memory: file cache, session state
│   └── cache/               # Cached file contents
│
├── knowledge/
│   ├── knowledge.db         # Cross-session: insights, preferences, learnings
│   ├── vectors.faiss        # Semantic search index
│   └── archive/             # Archived entries (from defrag)
│
├── tools/
│   ├── tools.db             # Tool registry
│   ├── vectors.faiss        # Tool semantic search
│   ├── core/*.py            # Pre-installed tools
│   └── learned/*.py         # Agent-created tools
│
├── skills/
│   ├── skills.db            # Learned workflows
│   ├── vectors.faiss        # Skill semantic search
│   └── templates/           # Skill templates
│
├── agents/
│   ├── agents.db            # Specialist agent registry
│   ├── vectors.faiss        # Agent semantic search
│   ├── core/*.py            # 7 core specialists
│   └── learned/*.py         # Agent-created specialists
│
└── sessions/
    ├── plans.db             # Master plans per session
    ├── manifests.db         # Project manifests per session
    └── {session_id}/        # Per-session data
```

**Why seven databases?**

| Database | Scope | Purpose | Shared Across |
|----------|-------|---------|---------------|
| memory.db | Session | Fast cache, working state | Current session only |
| knowledge.db | Permanent | Insights, preferences | All sessions, all agents |
| tools.db | Permanent | Tool registry | All sessions, all agents |
| skills.db | Permanent | Learned workflows | All sessions, all agents |
| agents.db | Permanent | Specialist registry | All sessions, all agents |
| plans.db | Session | Master plan, task tree | Current session, all agents |
| manifests.db | Session | Project state, decisions | Current session, all agents |

**Thread safety strategy:**

```python
# Single-threaded execution with async/await (preferred)
# Locks only needed during async I/O operations

# Example: Agent updates manifest during LLM call (async I/O)
async def some_agent_operation(self):
    # Yield during LLM call
    response = await self.llm.query_async("What should I do?")

    # Another agent might be updating manifest during this yield
    # Lock ensures atomic update
    with self.shared.lock("manifest"):
        self.shared.manifest.add_file("new_file.py", created_by=self.agent_id)
```

**Context injection (what sub-agents see):**

```python
def _prepare_context_for_subagent(self, task: str) -> dict:
    """Every sub-agent gets full context from shared state."""

    return {
        # HIGH-LEVEL CONTEXT
        "overall_goal": self.shared.plan.goal,
        "my_task": task,
        "my_depth": self.depth + 1,
        "parent_agent": self.agent_id,

        # FROM MASTER PLAN
        "completed_tasks": [t.description for t in self.shared.plan.completed],
        "active_tasks": [t.description for t in self.shared.plan.in_progress],
        "progress": f"{self.shared.plan.progress_pct}%",

        # FROM PROJECT MANIFEST
        "files_created": list(self.shared.manifest.files.keys()),
        "api_endpoints": self.shared.manifest.get_all_endpoints(),
        "database_tables": self.shared.manifest.database_schema,
        "architecture_decisions": self.shared.manifest.decisions,

        # FROM KNOWLEDGE DB
        "project_conventions": self.shared.knowledge_db.get_conventions(),
        "user_preferences": self.shared.knowledge_db.get_preferences(),
        "relevant_insights": self.shared.knowledge_db.get_relevant_insights(task),

        # FROM MEMORY DB
        "recently_accessed_files": self.shared.memory_db.get_recent_files(),

        # FROM CALL STACK
        "recursion_depth": self.depth,
        "sibling_agents": self.shared.call_stack.get_siblings(self.agent_id)
    }
```

**This context is injected into the sub-agent's system prompt automatically.**

**Why shared state matters:**

| Without Shared State | With Shared State |
|---------------------|-------------------|
| Backend uses JWT, Frontend uses sessions → incompatible | Both see decision in manifest → consistent |
| Agent A creates tool, Agent B doesn't know → duplication | Both access tools.db → reuse |
| Agent reads file, Agent B re-reads same file → slow | Agent A caches in memory.db, B reads cache → fast |
| User answers question, only one agent sees it → confusion | All agents see answer in message queue → coordinated |
| No visibility into what others are doing → chaos | All agents see plan + manifest → transparent |

### Mixin Composition

```python
class GaiaCodeAgent(
    # Base
    Agent,
    ApiAgent,

    # Foundation (always active)
    ErrorRecoveryMixin,
    SecurityMixin,

    # Coding Tools (always active for gaia-code)
    CodeToolsMixin,
    FileIOToolsMixin,
    CodeFormattingMixin,
    TestingMixin,
    ErrorFixingMixin,
    ValidationToolsMixin,

    # V2 Features (opt-in via config)
    MemoryMixin,                # enable_memory
    ContinuousExecutionMixin,   # enable_continuous_execution
    StateMachineMixin,          # enable_state_machine
    ManifestToolsMixin,         # enable_manifest
    AdaptivePromptsMixin,       # enable_adaptive_prompts
    LearningLoopMixin,          # enable_learning
    DynamicToolsMixin,          # enable_dynamic_tools
    ObservabilityMixin,         # enable_observability
    TaskToolsMixin,             # enable_task_queue
    SelfEvaluationMixin,        # enable_self_eval
    ProactiveAnalysisMixin,     # enable_proactive_analysis
):
    """The world's most autonomous coding agent."""
```

---

## Feature Deep-Dives

### Continuous Execution: The Core Innovation

The fundamental difference between Gaia Code and every other agent is **unlimited execution with guaranteed quality**.

**How it works:**

```
1. User submits task: "Build a REST API with user auth and tests"

2. REQUIREMENTS GATHERING (auto-skip for simple tasks)
   - "What framework? FastAPI or Flask?"
   - "What auth? JWT or session?"
   - "What database? PostgreSQL or SQLite?"
   → Stores answers in task metadata

3. PLANNING
   - Decompose into sub-tasks:
     a. Create project structure
     b. Implement database models
     c. Implement auth endpoints
     d. Implement CRUD endpoints
     e. Write unit tests
     f. Write integration tests
     g. Create Dockerfile (if requested)
   - Estimate complexity per sub-task
   - Identify dependencies between sub-tasks

4. IMPLEMENTATION (continuous loop)
   For each sub-task:
     a. Generate code
     b. Run quality gates (syntax, lint, import resolution)
     c. If gates fail → fix and retry
     d. If context approaching limit → checkpoint + resume
     e. Mark sub-task complete

5. TESTING (continuous loop)
   a. Run all tests
   b. If failures → enter DEBUG state
   c. Debug: identify root cause, fix, re-test
   d. Loop until all tests pass
   e. Run security scan
   f. Run lint check

6. REVIEW
   a. Self-evaluation pass (rate: correctness, completeness, style, security)
   b. If score < 0.7 → revise
   c. Generate summary of changes
   d. Present to user for approval

7. DONE
   - All quality gates pass
   - All tests pass
   - Self-evaluation score ≥ 0.7
   - Changes summarized for user
```

**Checkpoint/Resume in detail:**

```python
class CheckpointManager:
    def save_checkpoint(self):
        """Save everything needed to resume from this exact point."""
        checkpoint = {
            "task": self.current_task,
            "state": self.state_machine.current_state,
            "sub_tasks": self.task_queue.get_all(),
            "completed_sub_tasks": self.task_queue.get_completed(),
            "manifest": self.manifest.snapshot(),
            "memory_context": self.memory.get_relevant_context(),
            "conversation_summary": self.compress_conversation(),
            "files_modified": self.get_modified_files(),
            "quality_gate_results": self.quality_gates.last_results,
            "timestamp": datetime.now(),
        }
        self.storage.save(checkpoint)

    def resume_from_checkpoint(self):
        """Resume with full context restored."""
        checkpoint = self.storage.load_latest()
        self.state_machine.restore(checkpoint["state"])
        self.task_queue.restore(checkpoint["sub_tasks"])
        self.manifest.restore(checkpoint["manifest"])
        # Inject context into new conversation
        self.inject_context(checkpoint["conversation_summary"])
        self.inject_context(checkpoint["memory_context"])
        # Continue from where we left off
        self.continue_execution()
```

### Quality Gates: The Safety Net

```
┌──────────────────────────────────────────────────────────────┐
│                    QUALITY GATE PIPELINE                       │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Syntax   │──│  Import  │──│   Lint   │──│   Type   │    │
│  │ Check    │  │  Check   │  │  Check   │  │  Check   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│       │              │              │              │          │
│       ▼              ▼              ▼              ▼          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Test    │──│ Security │──│  Build   │──│  Self    │    │
│  │  Runner  │  │  Scan    │  │  Check   │  │  Review  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                               │
│  ALL GATES MUST PASS before task is declared complete         │
└──────────────────────────────────────────────────────────────┘
```

**Gate details:**

| Gate | What It Checks | Tool | Tolerance |
|------|---------------|------|-----------|
| Syntax | All modified files parse without errors | `ast.parse()` / language-specific | **Zero** -- must pass |
| Import | All imports resolve to existing modules | Custom import resolver | **Zero** -- must pass |
| Lint | Code formatting matches project standards | black/eslint/prettier (auto-detect) | Configurable |
| Type | Type annotations consistent (if project uses types) | mypy/pyright/tsc (if configured) | Configurable |
| Test | All affected tests pass | pytest/jest/go test (auto-detect) | **Zero** -- must pass |
| Security | No hardcoded secrets, no injection vulnerabilities | Custom scanner + bandit | **Zero** for critical |
| Build | Project builds/compiles successfully | Language-specific build tool | **Zero** -- must pass |
| Self-Review | LLM reviews changes for correctness/completeness | Self-evaluation pass | Score ≥ 0.7 |

---

## Implementation Phases

### Phase 0: Core Infrastructure Prerequisites (Days 1-2) -- BUILD FIRST

**Goal:** The foundational infrastructure that ALL agent features depend on. Nothing else starts until this is done.

**LAYER 0 (blocks everything):**

| Branch | Hours | What It Builds | What It Unblocks |
|--------|-------|----------------|------------------|
| `feature/v2-config-registry` | 4h | V2Config dataclass, FeatureRegistry, tier validation | Every feature flag |
| `feature/v2-tool-registry` | 4h | Instance-scoped ToolRegistry (replaces global dict) | Concurrent threads, robust single-agent |
| `feature/v2-structured-logging` | 2h | structlog integration, structured fields | Debugging, observability |

**LAYER 1 (blocks agent features):**

| Branch | Hours | What It Builds | What It Unblocks |
|--------|-------|----------------|------------------|
| `feature/v2-llm-router` | 6h | LLMRouter, tier detection, model selection | All LLM-dependent features |
| `feature/v2-error-recovery` | 4h | RetryEngine, CircuitBreaker | All LLM calls |
| `feature/v2-input-sanitization` | 3h | Path validation, injection prevention | All user-facing features |
| `feature/v2-checkpoint-base` | 4h | Serializable protocol, CheckpointStore (SQLite) | Memory, continuous exec, tasks |

**LAYER 2 (blocks domain features):**

| Branch | Hours | What It Builds | What It Unblocks |
|--------|-------|----------------|------------------|
| `feature/v2-memory-infra` | 6h | MemoryStore base, SQLite schemas, FAISS management | All memory tiers |
| `feature/v2-task-infra` | 4h | Task model, SQLite schema, basic CRUD | Task queue, continuous exec |
| `feature/v2-mixin-lifecycle` | 3h | V2Mixin base class, init/enable/disable/cleanup | All V2 mixins |

**Total: ~40 hours (~1.7 days)**

**Deliverable:** All infrastructure in place. Feature flags work. LLM routing works. State can be serialized. Existing agents unchanged.

```
Validation:
$ gaia code --v2-status
  V2Config:          ✅ loaded (all flags: false)
  FeatureRegistry:   ✅ 0 features registered (ready for registration)
  ToolRegistry:      ✅ instance-scoped (no global conflicts)
  LLMRouter:         ✅ detected: basic(Qwen3-0.6B), standard(Qwen3-7B)
  ErrorRecovery:     ✅ RetryEngine active (3 retries, backoff)
  CheckpointStore:   ✅ SQLite ready at ~/.gaia/checkpoints.db
  MemoryStore:       ✅ SQLite ready at ~/.gaia/memory/
  Existing agents:   ✅ ChatAgent, CodeAgent work unchanged
```

---

### Phase 1: Core Coding Tools + Basic Agent (Days 2-3)

**Goal:** Match Claude Code's basic capabilities on top of the infrastructure

| Branch | Hours | What It Builds |
|--------|-------|----------------|
| `feature/v2-security-vault` | 4h | SecretVault (AES-256), permissions model |
| `feature/v2-code-tools-enhanced` | 8h | AST editing, symbol resolution, import analysis |
| `feature/v2-git-enhanced` | 4h | Semantic commits, merge conflict resolution |
| `feature/v2-code-intelligence` | 6h | Semantic search, symbol table, dependency analysis |

**Deliverable:** Agent with Claude Code-equivalent tools + composable config system

---

### Phase 2: Continuous Execution + Quality Gates (Days 3-5)

**Goal:** The core differentiator -- unlimited execution with verified quality

| Branch | Hours | What It Builds |
|--------|-------|----------------|
| `feature/v2-continuous-execution` | 8h | ContinuousExecutionEngine, checkpoint/resume |
| `feature/v2-quality-gates` | 8h | All 8 quality gates, gate pipeline |
| `feature/v2-task-queue` | 6h | Task decomposition, priority queue, dependencies |
| `feature/v2-state-machine` | 8h | 6-state machine, requirements gathering, transitions |

**Deliverable:** Agent that runs until verified complete with quality gates. **This is Milestone 1 -- the MVP that proves the concept.**

---

### Phase 3: Intelligence Layer (Days 5-7)

**Goal:** Memory, learning, and adaptive behavior

| Branch | Hours | What It Builds |
|--------|-------|----------------|
| `feature/v2-memory-tier1-2` | 8h | Episodic memory (FAISS), working memory management |
| `feature/v2-memory-tier3-4` | 6h | Semantic memory (SQLite), universal knowledge DB |
| `feature/v2-manifest` | 6h | Project manifest, dependency graph, file tracking |
| `feature/v2-adaptive-prompts` | 4h | Dynamic prompt composition, state-aware context |
| `feature/v2-learning-loop` | 8h | Feedback collection, pattern extraction, adaptation |
| `feature/v2-self-eval` | 4h | Self-evaluation pass, multi-dimensional scoring |

**Deliverable:** Agent that learns from every session and improves over time. **This is Milestone 2 -- the learning agent.**

---

### Phase 4: Superpowers (Days 7-9)

**Goal:** Dynamic tools, concurrent threads, proactive analysis

| Branch | Hours | What It Builds |
|--------|-------|----------------|
| `feature/v2-dynamic-tools` | 8h | Tool generation, skill store, AST validation |
| `feature/v2-concurrent-threads` | 8h | Concurrent task threads, shared state, thread coordination |
| `feature/v2-proactive-analysis` | 6h | Dead code, complexity, security, coverage analysis |

**Deliverable:** Full-featured autonomous agent. **This is Milestone 3 -- the agent that exceeds Claude Code.**

### Phase 5: Polish + Integration (Days 8-9)

| Branch | Hours | What It Builds |
|--------|-------|----------------|
| Integration tests | 6h | Cross-feature tests, regression suite |
| CLI polish | 4h | `gaia code` CLI with all flags and profiles |
| Documentation | 4h | User guide, SDK docs |
| TUI integration | 6h | Real-time progress in terminal UI |

**Deliverable:** Production-ready `gaia code` command.

### Timeline Summary

```
Day 1-2:  Phase 0: PREREQUISITES       (40h)  → Infrastructure complete
Day 2-3:  Phase 1: Core Tools           (22h)  → Claude Code equivalent
Day 3-5:  Phase 2: Continuous + Quality (30h)  → EXCEEDS Claude Code (MVP)
Day 5-7:  Phase 3: Intelligence Layer   (36h)  → Learning agent
Day 7-9:  Phase 4: Superpowers          (24h)  → Unprecedented capabilities
Day 9-10: Phase 5: Polish + Integration (20h)  → Production ready

Total: ~172 autonomous-agent-hours → 7.5 days (at 24h/day)
```

### Critical Path (What Must Happen in Order)

```
Prerequisites (P0→P1→P2) → Core Tools → Continuous Execution → Quality Gates
                                                                     ↓
     Memory ← Task Queue ← State Machine ← (all need checkpoint base)
       ↓
     Adaptive Prompts ← Learning Loop ← Manifest
       ↓
     Dynamic Tools ← Concurrent Threads ← (needs everything above)
```

**The prerequisites are the critical path.** Cutting corners here slows everything downstream.

---

## The Bootstrap Effect

### How Gaia Code Accelerates Everything Else

Once Gaia Code is built (Phase 0-5, ~7.5 days), it becomes the primary tool for building all other GAIA features:

```
Week 1:     Build Gaia Code MVP (manually + Claude Code)
Week 1.5:   Use Gaia Code MVP to complete Gaia Code (self-improvement)
Week 2:     Use Gaia Code to build gaia cua (Computer Use Agent)
             - Gaia Code has learned GAIA's patterns
             - Gaia Code has custom tools for GAIA development
             - Gaia Code knows the manifest, dependencies, conventions
             → 3x faster than building manually
Week 3:     Use Gaia Code to build gaia chat (Knowledge Assistant)
             - Gaia Code has learned from building CUA
             - Cross-cutting features already built (memory, state machine)
             - Gaia Code has even more custom tools
             → 5x faster than building manually
Week 4+:    Use Gaia Code to build TUI, Voice, Workflows, etc.
             - Gaia Code is now a domain expert in GAIA development
             → 10x faster than building manually
```

### The Self-Improvement Cycle

```
Version 1.0: Gaia Code built manually (6 days)
             Can build code, run tests, use quality gates

Version 1.1: Gaia Code improves itself (1 day)
             Adds missing features it discovered while being used
             Fixes its own bugs
             Creates custom tools for common patterns

Version 1.2: Gaia Code improves itself again (0.5 days)
             Even faster because it has learned from v1.1
             More custom tools, better patterns

Version 2.0: Gaia Code at full capability
             Has built CUA, Chat, and other agents
             Deep knowledge of entire GAIA codebase
             Custom tools for every common development pattern
             Near-zero errors on GAIA-related tasks
```

### Quantified Impact

| Task | Without Gaia Code | With Gaia Code | Speedup |
|------|-------------------|----------------|---------|
| Build CUA (Computer Use Agent) | 3 days | 1 day | **3x** |
| Build Chat (Knowledge Assistant) | 2 days | 0.5 days | **4x** |
| Build TUI (Terminal UI) | 1.5 days | 0.3 days | **5x** |
| Build Voice Integration | 1 day | 0.2 days | **5x** |
| Build Workflow Engine | 1.5 days | 0.3 days | **5x** |
| Add new agent type | 2 days | 0.3 days | **7x** |
| Full V2 implementation | 9 days | 4 days | **2.3x** |

---

## Validation Criteria

### Milestone 1 Validation: "Coding Agent MVP" (Day 4)

```bash
# Test 1: Continuous execution (no step limit)
gaia code "Build a FastAPI REST API with:
  - User model with email, password, name
  - Auth endpoints (register, login, refresh)
  - CRUD endpoints for a 'posts' resource
  - JWT authentication middleware
  - SQLite database with SQLAlchemy
  - Full pytest test suite
  - Requirements.txt"

# Success criteria:
# ✅ Agent runs until ALL files created (no max_steps cutoff)
# ✅ All quality gates pass (syntax, import, lint, tests)
# ✅ All tests pass when run manually (pytest -xvs)
# ✅ Application starts and responds to curl requests
# ✅ JWT auth works end-to-end (register → login → access protected route)
# ✅ Agent produces summary of what it built
```

```bash
# Test 2: State machine (requirements gathering)
gaia code "Build an e-commerce platform"

# Success criteria:
# ✅ Agent asks clarifying questions (framework, DB, payment provider, etc.)
# ✅ Agent waits for answers before proceeding
# ✅ Planning state produces decomposed task list
# ✅ Each task tracked in queue
```

```bash
# Test 3: Context limits (checkpoint/resume)
gaia code "Build a project with 50+ files: full-stack app with
  React frontend, FastAPI backend, PostgreSQL database,
  Redis cache, WebSocket notifications, admin panel,
  comprehensive test suite, Docker deployment"

# Success criteria:
# ✅ Agent checkpoints when approaching context limit
# ✅ Agent resumes without losing progress
# ✅ Final result is complete despite multiple context windows
# ✅ Quality gates pass on final result
```

### Milestone 2 Validation: "Learning Agent" (Day 6)

```bash
# Test 4: Persistent memory
# Session 1:
gaia code "Create a Python project using pathlib for all file operations"
# Session 2 (new session):
gaia code "Add file processing to the project"

# Success criteria:
# ✅ Session 2 uses pathlib without being reminded
# ✅ Session 2 knows the project structure from Session 1
# ✅ Agent recalls decisions made in Session 1
```

```bash
# Test 5: Learning from correction
# Session 1:
gaia code "Write a function to read a CSV file"
# Agent uses pandas
# User: "Don't use pandas, use the csv module instead"

# Session 2 (new session):
gaia code "Write a function to process a TSV file"

# Success criteria:
# ✅ Agent uses csv module, not pandas
# ✅ Agent explains: "Based on your preference, using csv module instead of pandas"
```

### Milestone 3 Validation: "Exceeds Claude Code" (Day 8)

```bash
# Test 6: Head-to-head comparison
# Give BOTH Claude Code and Gaia Code the same complex task:
"Build a complete blog platform:
  - FastAPI backend with PostgreSQL
  - User registration and authentication
  - Blog post CRUD with rich text
  - Comment system with nested replies
  - Tag system with search
  - Pagination and filtering
  - Admin panel for content moderation
  - Full test suite (unit + integration)
  - Docker deployment
  - API documentation (OpenAPI/Swagger)"

# Comparison criteria:
# ✅ Gaia Code produces working code (all tests pass)
# ✅ Gaia Code runs without hitting context limits
# ✅ Gaia Code's code follows consistent patterns throughout
# ✅ Gaia Code catches and fixes its own errors automatically
# ✅ Gaia Code's test coverage ≥ 80%
# ✅ Gaia Code's security scan is clean
```

---

## Design Philosophy: Simplicity Over Complexity

**WARNING:** This document describes the *maximum* feature set. The actual implementation should follow a strict **minimum viable** approach. Over-engineering is the biggest risk -- a system with 18 features poorly integrated is worse than a system with 5 features done extremely well.

### The Minimum Viable Autonomous Agent

**5 core capabilities are needed to exceed Claude Code.** Everything else is enhancement.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   THE FIVE PILLARS (build these first, build them right)            │
│                                                                     │
│   1. CONTEXT-LEAN ARCHITECTURE -- Knowledge DB, not context window  │
│   2. CONTINUOUS EXECUTION      -- Run until done (no step limit)    │
│   3. QUALITY GATES             -- Verify output works (tests pass)  │
│   4. CHECKPOINT/RESUME         -- Survive crashes/interruptions     │
│   5. PERSISTENT MEMORY         -- Learn across sessions             │
│                                                                     │
│   Everything else is enhancement. Ship the pillars first.           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why These 5?

| Pillar | Why It's Essential | What It Replaces |
|--------|-------------------|-----------------|
| Context-Lean | Claude Code compacts conversation (slow, lossy). We never do. | Conversation compaction |
| Continuous Execution | Claude Code stops at context limits. We don't. | Fixed `max_steps=100` |
| Quality Gates | Claude Code guesses "I'm done." We verify. | No verification |
| Checkpoint/Resume | Claude Code loses state on crash. We recover. | Loss of progress |
| Persistent Memory | Claude Code forgets. We remember. | Static CLAUDE.md |

### Practical Features That Matter Most

> **Less academic, more practical.** These are the features that make an agent actually useful day-to-day. They are simple concepts but incredibly high-impact.

#### Planning & Task Decomposition

The most practical thing an agent can do is **build a good plan and keep it updated**.

```
User: "Build a REST API with auth and tests"

Agent builds plan:
┌─────────────────────────────────────────────────┐
│ PLAN: REST API with Auth                         │
│                                                  │
│ [✅] 1. Create project structure                 │
│ [✅] 2. Define database models (User, Post)      │
│ [🔄] 3. Implement auth endpoints                 │
│   [✅] 3a. Register endpoint                     │
│   [🔄] 3b. Login endpoint (JWT)                  │
│   [  ] 3c. Refresh token endpoint                │
│ [  ] 4. Implement CRUD endpoints                 │
│ [  ] 5. Write unit tests                         │
│ [  ] 6. Write integration tests                  │
│ [  ] 7. Final validation (all tests pass)        │
│                                                  │
│ Progress: 3/7 tasks (43%)                        │
│ Current: Step 3b - Login endpoint                │
│ Blocked: None                                    │
│ Updated: 2 minutes ago                           │
└─────────────────────────────────────────────────┘
```

**Key behaviors:**
- **Plan before coding** -- Always create a plan. Even for "simple" tasks.
- **Update the plan as you go** -- Check off completed steps. Add new steps discovered during implementation.
- **Replan when needed** -- If step 3 reveals that the data model needs changes, update step 2 and re-execute.
- **Show progress** -- User always knows what's done, what's in progress, what's remaining.
- **Break big tasks into subtasks** -- "Implement auth" becomes 3a, 3b, 3c. Each subtask is independently verifiable.

```python
class Plan:
    """A living plan that the agent creates, follows, and updates."""

    def __init__(self, goal: str):
        self.goal = goal
        self.tasks: list[Task] = []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def add_task(self, description: str, parent_id: str = None) -> Task:
        """Add a task or subtask to the plan."""
        task = Task(description=description, parent_id=parent_id)
        self.tasks.append(task)
        self._persist()
        return task

    def complete_task(self, task_id: str, result: str = None):
        """Mark task complete and persist to knowledge DB."""
        task = self.get_task(task_id)
        task.status = "completed"
        task.result = result
        task.completed_at = datetime.now()
        self._persist()

    def replan(self, reason: str, new_tasks: list[str]):
        """Update the plan based on new information discovered during execution."""
        self.revisions.append({"reason": reason, "timestamp": datetime.now()})
        for desc in new_tasks:
            self.add_task(desc)
        self._persist()

    def get_next_task(self) -> Task | None:
        """Get the next task to work on (respects dependencies)."""
        for task in self.tasks:
            if task.status == "pending" and self._dependencies_met(task):
                return task
        return None

    @property
    def progress(self) -> float:
        """Overall completion percentage."""
        if not self.tasks:
            return 0.0
        completed = sum(1 for t in self.tasks if t.status == "completed")
        return completed / len(self.tasks)

    def _persist(self):
        """Save plan to knowledge DB (survives context window, crashes, sessions)."""
        self.updated_at = datetime.now()
        self.knowledge_db.store_plan(self)
```

#### File Understanding Before Editing

Before editing any file, the agent should understand it:

```
Practical behavior:
1. Read the file
2. Understand its role (what does it do? what depends on it?)
3. Check the manifest (what other files import this?)
4. Check memory (have I edited this file before? any known issues?)
5. Make the edit
6. Verify the edit didn't break dependents
```

#### Iterative Debugging

When something fails, the agent should debug systematically, not guess:

```
Practical behavior:
1. Read the FULL error message
2. Identify the failing line and file
3. Read the relevant code
4. Form a hypothesis
5. Make ONE change to test the hypothesis
6. Run the test again
7. If still failing, form a new hypothesis (don't repeat the same fix)
8. Store the fix in knowledge DB (so it's never forgotten)
```

#### Progress Communication

The agent should always tell the user what it's doing:

```
[Step 3/7] Implementing login endpoint...
  - Writing src/auth/login.py
  - Adding JWT token generation
  - Running tests... 2/5 passing
  - Fixing test_login_invalid_password... ✅
  - Running tests... 5/5 passing ✅
[Step 3/7] Complete. Moving to step 4/7.
```

### What NOT to Build First

| Feature | Why It Can Wait |
|---------|----------------|
| Dynamic Tool Creation | Premature. Let the agent mature first. Add when patterns emerge naturally. |
| Multi-Agent Orchestration | Over-engineered. Single agent with concurrent task threads is simpler and more coherent. |
| Adaptive Prompts (6 layers) | Start with 2 layers: base + memory context. Add layers as needed. |
| Self-Evaluation (7 dimensions) | Quality gates already verify output. Self-eval is redundant for MVP. |
| Proactive Analysis | Nice-to-have. The quality gates already catch issues. |
| LLM Tier Routing | Start with single tier. Add routing when we have multiple models working. |

### Simplified Prerequisites

The 10-component prerequisite list can be trimmed to **4 essentials**:

| Must Have | Why | Skip For Now |
|-----------|-----|-------------|
| V2Config + Feature Flags | Composability depends on it | Enhanced Tool Registry (fix when concurrent threads needed) |
| Knowledge DB (SQLite + FAISS) | Context-lean architecture depends on it | Complex memory tiers (start with simple key-value + vector search) |
| Checkpoint/Serialization | Continuous execution needs it | Mixin Lifecycle Protocol (just use `__init__` checks) |
| Basic Error Recovery (retry) | Every LLM call needs it | LLM Router (start with single tier) |

**Simplified Phase 0: 4 components, ~18 hours instead of 40.**

### The Simplicity Test

Before adding ANY feature, ask:
1. **Does the agent work without it?** If yes, don't add it yet.
2. **Can a simpler version achieve 80% of the benefit?** If yes, build that.
3. **Will this feature interact with >2 other features?** If yes, think twice -- complexity compounds.
4. **Is this solving a problem we've actually hit?** If no, it's premature.

---

## Knowledge Retrieval Architecture

> **This is the brain of the agent.** Without effective retrieval, memory is useless -- the agent has knowledge it can't find. Retrieval quality directly determines agent quality.

### The Core Problem

An agent accumulates knowledge rapidly:
- Every file it reads (hundreds per session)
- Every tool result (dozens per task)
- Every error and fix (pattern learning)
- Every user correction (preference learning)
- Every plan and its outcome (process learning)
- Every session summary (project evolution)

**Without retrieval, this is just a database. With retrieval, it's intelligence.**

### Retrieval Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE RETRIEVAL SYSTEM                      │
│                                                                    │
│   Agent needs context ──▶ RETRIEVER ──▶ Relevant knowledge        │
│                              │                                     │
│                    ┌─────────┴─────────┐                          │
│                    │   QUERY PLANNER   │                          │
│                    │                   │                          │
│                    │  "What do I need  │                          │
│                    │   to know right   │                          │
│                    │   now?"           │                          │
│                    └─────────┬─────────┘                          │
│                              │                                     │
│              ┌───────────────┼───────────────┐                    │
│              ▼               ▼               ▼                    │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│   │   EXACT      │  │   SEMANTIC   │  │  RECENCY     │          │
│   │   LOOKUP     │  │   SEARCH     │  │  SEARCH      │          │
│   │              │  │              │  │              │          │
│   │  "Get file   │  │  "Find code  │  │  "What did  │          │
│   │   X at path  │  │   related    │  │   I just    │          │
│   │   Y"         │  │   to auth"   │  │   do?"      │          │
│   │              │  │              │  │              │          │
│   │  SQLite key  │  │  FAISS       │  │  SQLite      │          │
│   │  lookup      │  │  vectors     │  │  timestamp   │          │
│   └──────────────┘  └──────────────┘  └──────────────┘          │
│              │               │               │                    │
│              └───────────────┼───────────────┘                    │
│                              ▼                                     │
│                    ┌─────────────────┐                             │
│                    │   RANKER        │                             │
│                    │                 │                             │
│                    │  Score results  │                             │
│                    │  by relevance   │                             │
│                    │  to current     │                             │
│                    │  task context   │                             │
│                    └─────────┬───────┘                             │
│                              ▼                                     │
│                    ┌─────────────────┐                             │
│                    │   FORMATTER     │                             │
│                    │                 │                             │
│                    │  Compact the    │                             │
│                    │  results into   │                             │
│                    │  context-ready  │                             │
│                    │  snippets       │                             │
│                    └─────────────────┘                             │
└──────────────────────────────────────────────────────────────────┘
```

### Three Retrieval Modes

#### 1. Automatic Retrieval (happens without agent asking)

Before every agent turn, the system automatically retrieves:

```python
class AutoRetriever:
    """Automatically injects relevant context before each agent turn."""

    def before_turn(self, current_task: str, current_state: str) -> str:
        """Called before every LLM call. Returns context to inject."""
        context_parts = []

        # 1. Active plan status (always injected)
        plan = self.knowledge_db.get_active_plan()
        if plan:
            context_parts.append(f"Current plan: {plan.summary()}")
            context_parts.append(f"Next task: {plan.get_next_task()}")

        # 2. Relevant memories for this task
        memories = self.knowledge_db.semantic_search(
            query=current_task,
            categories=["learnings", "preferences", "patterns"],
            top_k=5
        )
        if memories:
            context_parts.append("Relevant knowledge:\n" + "\n".join(
                f"- {m.content} (confidence: {m.confidence})"
                for m in memories
            ))

        # 3. Recent errors (so agent doesn't repeat them)
        recent_errors = self.knowledge_db.get_recent(
            category="errors",
            limit=3,
            within_minutes=30
        )
        if recent_errors:
            context_parts.append("Recent errors (don't repeat):\n" + "\n".join(
                f"- {e.summary}" for e in recent_errors
            ))

        # 4. Project conventions (if working on code)
        conventions = self.knowledge_db.get_conventions(
            project_path=self.project_root
        )
        if conventions:
            context_parts.append("Project conventions:\n" + "\n".join(
                f"- {c}" for c in conventions
            ))

        return "\n\n".join(context_parts)
```

#### 2. Explicit Retrieval (agent queries the knowledge DB)

The agent has a `recall` tool to query the knowledge DB:

```python
@tool
def recall(query: str, category: str = None) -> str:
    """Search the knowledge database for relevant information.

    Use this when you need to:
    - Remember what you did in a previous session
    - Find a file you read earlier
    - Check if you've solved a similar problem before
    - Look up user preferences or project conventions

    Args:
        query: Natural language query (e.g., "how did I implement auth last time?")
        category: Optional filter - "files", "errors", "decisions", "preferences",
                  "patterns", "plans", "sessions"
    """
    results = knowledge_db.semantic_search(query, category=category, top_k=10)
    return format_results(results)


@tool
def recall_file(path: str) -> str:
    """Retrieve a previously-read file from the knowledge DB cache.

    Faster than re-reading from disk. Includes metadata about when
    the file was last read and any notes attached to it.
    """
    cached = knowledge_db.get_cached_file(path)
    if cached:
        return f"[Cached {cached.age_str} ago]\n{cached.content}"
    # Fall back to reading from disk
    return read_file(path)


@tool
def recall_error(error_type: str = None) -> str:
    """Find past errors and their fixes.

    Args:
        error_type: Optional filter (e.g., "ImportError", "TypeError", "test_failure")
    """
    errors = knowledge_db.search_errors(error_type=error_type, top_k=5)
    return "\n".join(
        f"Error: {e.message}\nFix: {e.fix}\nFile: {e.file}\n"
        for e in errors
    )
```

#### 3. Proactive Retrieval (system anticipates what agent will need)

```python
class ProactiveRetriever:
    """Predicts what the agent will need next and pre-fetches it."""

    def on_task_start(self, task: Task):
        """When a new task starts, pre-load relevant context."""
        # If task mentions a file, pre-cache it
        mentioned_files = self._extract_file_references(task.description)
        for f in mentioned_files:
            self.knowledge_db.cache_file(f, read_file(f))

        # If task is similar to a past task, load that session's summary
        similar_sessions = self.knowledge_db.find_similar_sessions(
            task.description, top_k=3
        )
        for session in similar_sessions:
            self.pre_loaded["past_sessions"].append(session.summary)

    def on_file_edit(self, path: str):
        """When agent edits a file, pre-load its dependents."""
        dependents = self.manifest.get_dependents(path)
        for dep in dependents:
            self.knowledge_db.cache_file(dep, read_file(dep))
```

### What Gets Stored (Knowledge Schema)

```python
# Everything the agent stores and can retrieve

class KnowledgeDB:
    """The agent's persistent brain. SQLite + FAISS."""

    # ── Files ──
    def cache_file(self, path: str, content: str, metadata: dict = None): ...
    def get_cached_file(self, path: str) -> CachedFile | None: ...

    # ── Tool Results ──
    def store_tool_result(self, tool: str, args: dict, result: str, duration_ms: int): ...
    def search_tool_results(self, query: str, tool: str = None) -> list[ToolResult]: ...

    # ── Conversation ──
    def store_turn(self, role: str, content: str, turn_number: int): ...
    def get_recent_turns(self, limit: int = 10) -> list[Turn]: ...
    def search_turns(self, query: str) -> list[Turn]: ...

    # ── Learnings (extracted from experience) ──
    def store_learning(self, content: str, category: str, confidence: float): ...
    def get_learnings(self, category: str = None, min_confidence: float = 0.5) -> list[Learning]: ...

    # ── Errors & Fixes ──
    def store_error(self, error: str, fix: str, file: str, category: str): ...
    def search_errors(self, error_type: str = None, file: str = None) -> list[ErrorFix]: ...

    # ── Plans ──
    def store_plan(self, plan: Plan): ...
    def get_active_plan(self) -> Plan | None: ...
    def get_past_plans(self, query: str = None) -> list[Plan]: ...

    # ── Sessions ──
    def store_session_summary(self, summary: str, files_modified: list[str]): ...
    def find_similar_sessions(self, query: str, top_k: int = 3) -> list[Session]: ...

    # ── User Preferences ──
    def store_preference(self, key: str, value: str, source: str): ...
    def get_preferences(self) -> dict[str, str]: ...

    # ── Project Conventions ──
    def store_convention(self, convention: str, project: str): ...
    def get_conventions(self, project: str) -> list[str]: ...

    # ── Semantic Search (across everything) ──
    def semantic_search(self, query: str, category: str = None,
                       top_k: int = 10) -> list[SearchResult]: ...
```

### Storage Backend

```
~/.gaia/knowledge/
├── knowledge.db          # SQLite: structured data (plans, errors, preferences, sessions)
├── vectors.faiss         # FAISS: vector embeddings for semantic search
├── vectors_meta.json     # FAISS metadata (maps vector IDs to knowledge entries)
├── file_cache/           # Cached file contents (avoid re-reading from disk)
│   └── {hash}.txt        # Content-addressed storage
└── embeddings_cache/     # Pre-computed embeddings (avoid re-embedding)
    └── {hash}.npy        # Numpy arrays

Per-project:
.gaia/
├── project_knowledge.db  # Project-specific learnings, conventions, manifest
└── plan.json             # Active plan (also in knowledge.db, but readable by user)
```

### Embedding Strategy

```python
class EmbeddingEngine:
    """Generates embeddings for semantic search. Runs locally on AMD hardware."""

    def __init__(self):
        # Use a small, fast embedding model (runs on NPU)
        # ~30M params, <10ms per embedding
        self.model = load_embedding_model("all-MiniLM-L6-v2")

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        return self.model.encode(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embedding for efficiency."""
        return self.model.encode(texts, batch_size=32)
```

**Why local embeddings matter:** Cloud embedding APIs add latency (100-500ms per call). Local embeddings on AMD NPU take <10ms. For an agent that queries knowledge DB 5-10 times per turn, this is the difference between snappy and sluggish.

### Retrieval Quality: How to Ensure the Right Knowledge Surfaces

```
Problem: Agent has 10,000 knowledge entries. How to find the 5 most relevant?

Solution: Multi-signal ranking

Score = 0.4 * semantic_similarity    # How related is the content?
      + 0.3 * recency_score          # How recent is it?
      + 0.2 * usage_frequency        # How often has this been useful?
      + 0.1 * confidence_score       # How confident are we in this knowledge?
```

```python
class Ranker:
    """Ranks retrieval results by multiple signals."""

    def rank(self, results: list[SearchResult], query: str,
             current_context: str) -> list[SearchResult]:
        for result in results:
            result.score = (
                0.4 * result.semantic_similarity +
                0.3 * self._recency_score(result.timestamp) +
                0.2 * self._usage_score(result.times_retrieved) +
                0.1 * result.confidence
            )
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _recency_score(self, timestamp: datetime) -> float:
        """More recent = higher score. Exponential decay."""
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        return math.exp(-age_hours / 168)  # Half-life: 1 week

    def _usage_score(self, times_retrieved: int) -> float:
        """Frequently retrieved = more useful. Log scale."""
        return min(1.0, math.log1p(times_retrieved) / 5)
```

### Practical Examples of Retrieval in Action

**Example 1: Agent remembers a past fix**
```
Turn 47: Agent sees "ModuleNotFoundError: No module named 'pydantic'"

AutoRetriever searches knowledge DB:
  → Found: Error "ModuleNotFoundError: pydantic" from 3 sessions ago
  → Fix: "pip install pydantic" and add to requirements.txt
  → Confidence: 0.95

Agent immediately applies the fix without trial-and-error.
```

**Example 2: Agent recalls user preference**
```
Turn 12: Agent is about to use os.path.join()

AutoRetriever injects:
  → Preference: "Use pathlib.Path, not os.path" (confidence: 0.9)

Agent uses pathlib.Path without being corrected.
```

**Example 3: Agent finds relevant past plan**
```
User: "Build a REST API with auth"

ProactiveRetriever finds:
  → Past session: Built FastAPI API with JWT auth (2 weeks ago)
  → Plan: 7 steps, all completed successfully
  → Files: src/auth/, src/models/, tests/

Agent: "I built a similar API 2 weeks ago. Want me to follow the same pattern?"
```

**Example 4: Agent avoids a known bad pattern**
```
Turn 28: Agent is writing a subprocess call

AutoRetriever injects:
  → Learning: "Always use subprocess.run() with check=True, not subprocess.call()"
  → Learning: "Always use shlex.quote() for user-provided arguments"
  → Error from past: "Command injection vulnerability in subprocess.call()"

Agent writes safe subprocess code on the first try.
```

---

## Top Concerns & Novel Solutions

> **Honest assessment of the hardest problems in this architecture, with novel approaches to each.**

### Concern 1: The LLM Quality Bottleneck

**The problem:** No amount of infrastructure compensates for a mediocre LLM. If Qwen3-Coder-30B can't reason through multi-step coding tasks, quality gates just reject bad output in a loop, the plan tracker tracks a failing plan, and the knowledge DB stores memories of failure.

**The novel solution: Smart Escalation with Cost Ceiling**

Don't treat this as "local vs. cloud." Treat it as **per-subtask routing with a cost ceiling.**

```
The agent starts every subtask with the local LLM.
If the subtask SUCCEEDS (quality gates pass): great, $0 cost.
If the subtask FAILS after 2 attempts: escalate to cloud LLM for JUST this subtask.

Cost ceiling: User sets max cloud spend per session (e.g., $2/session).
Agent optimizes: use cloud only when local can't handle it.
Over time, insight engine learns WHICH subtasks need cloud → routes smarter.
```

```python
class SmartEscalation:
    """Route subtasks to local or cloud LLM based on actual capability."""

    def execute_subtask(self, subtask: Task) -> Result:
        # Try local first (always)
        for attempt in range(self.max_local_attempts):  # Default: 2
            result = self.local_llm.execute(subtask)
            if self.quality_gates.check(result).passed:
                # Local LLM handled it. $0 cost.
                self.insight_engine.record("local_success", subtask.category)
                return result

        # Local failed. Check cost ceiling before escalating.
        if self.session_cost < self.cost_ceiling:
            result = self.cloud_llm.execute(subtask)
            self.session_cost += result.cost
            # Learn: this type of subtask needs cloud
            self.insight_engine.record("cloud_needed", subtask.category,
                                       reason="local_failed_2x")
            return result

        # Cost ceiling hit. Park the task, ask user.
        self.message_queue.send_to_user(
            f"Subtask '{subtask.description}' is too complex for the local LLM "
            f"and we've hit the ${self.cost_ceiling} cloud budget. Options?",
            priority="decision",
            options=["Increase budget", "Skip this subtask", "Simplify the task"],
            task_id=subtask.id
        )
        self.plan.park_task(subtask.id)
```

**Why this is novel:** No agent does per-subtask routing with cost ceilings and learning. The agent gets smarter about what needs cloud over time, eventually minimizing cloud usage.

**Expected distribution (based on Claude Code analysis):**
- ~70% of subtasks: file ops, simple edits, test running → local LLM handles fine
- ~20% of subtasks: standard code generation, debugging → local LLM handles with retries
- ~10% of subtasks: complex reasoning, architecture decisions → cloud escalation

**Result:** 90% of work is free (local). 10% uses cloud. Total cost: ~$0.20/session instead of ~$2.00.

### Concern 2: Checkpoint = Compaction Paradox

**The problem:** When the agent checkpoints and resumes in a new context window, it needs to "inject context" about what happened. If that injection is a summary, we're doing exactly what Claude Code does (compaction) -- just calling it "checkpoint/resume."

**The novel solution: State-Based Resume, Not Summary-Based Resume**

The key insight: **the agent doesn't need to know the conversation. It needs to know the state.**

```
WRONG approach (what Claude Code does):
  "Summarize the last 50 turns into 2000 tokens"
  → Lossy, misses details, agent re-reads files

RIGHT approach (what Gaia Code does):
  Load the PLAN (which tasks are done, which remain)
  Load the ACTIVE FILE LIST (which files were modified)
  Load the QUALITY GATE RESULTS (what passed, what failed)
  Load the LAST 3 ERRORS (so we don't repeat them)
  → Zero summary needed. State IS the context.
```

```python
class StateBasedResume:
    """Resume from checkpoint using structured state, not lossy summaries."""

    def resume(self) -> str:
        """Build context injection from structured state. No summarization."""
        checkpoint = self.checkpoint_store.load_latest()

        context = []

        # 1. Plan state (most important -- tells agent exactly where it is)
        plan = checkpoint["plan"]
        context.append(f"ACTIVE PLAN: {plan.goal}")
        context.append(f"  Completed: {[t.description for t in plan.completed_tasks]}")
        context.append(f"  Current:   {plan.current_task.description}")
        context.append(f"  Remaining: {[t.description for t in plan.remaining_tasks]}")
        context.append(f"  Progress:  {plan.progress_pct}%")

        # 2. Files modified (agent knows what it built)
        context.append(f"FILES MODIFIED THIS SESSION: {checkpoint['files_modified']}")

        # 3. Quality gate status (agent knows what works)
        gates = checkpoint["quality_gate_results"]
        context.append(f"QUALITY GATES: {gates}")

        # 4. Last errors (so agent doesn't repeat them)
        errors = checkpoint["recent_errors"][-3:]
        if errors:
            context.append(f"RECENT ERRORS (don't repeat): {errors}")

        # 5. User preferences (from knowledge DB, always available)
        prefs = self.knowledge_db.get_preferences()
        if prefs:
            context.append(f"USER PREFERENCES: {prefs}")

        return "\n".join(context)

        # Total: ~500-1000 tokens of structured state
        # vs. ~2000-5000 tokens of lossy summary
        # Zero information loss. Agent picks up exactly where it left off.
```

**Why this works:** The plan IS the summary. Completed tasks tell the agent what was done. Remaining tasks tell it what to do next. Quality gates tell it what works. Errors tell it what to avoid. No prose summary needed -- structured data is lossless and compact.

### Concern 3: Quality Gates Can't Fix Bad Code

**The problem:** Quality gates detect bad code. They don't fix it. If the LLM consistently generates bad code, quality gates just create an infinite reject-retry loop.

**The novel solution: The Escalation Ladder**

After N failures, don't just retry -- escalate through increasingly drastic interventions.

```
ESCALATION LADDER (4 rungs):

Rung 1: RETRY (same approach, fresh attempt)
  → Attempt 1-2: same LLM, same prompt
  → Works for: transient errors, typos, minor issues

Rung 2: DECOMPOSE (break the subtask into smaller pieces)
  → The LLM can't handle "implement auth system"
  → Break into: "create User model", "add password hashing", "create login endpoint"
  → Works for: tasks that are too complex for a single LLM call

Rung 3: ESCALATE (use a more capable LLM)
  → Send just THIS subtask to cloud LLM
  → Works for: tasks that require deeper reasoning than the local LLM can handle

Rung 4: ASK (send to user via message queue)
  → "I've tried 3 approaches to implement OAuth2 and none pass tests.
     Can you provide guidance or an example?"
  → Agent parks this task, continues working on other tasks
  → Works for: tasks that require human judgment or domain knowledge
```

```python
class EscalationLadder:
    """Progressively more drastic interventions when quality gates fail."""

    def handle_gate_failure(self, subtask: Task, failure: GateFailure,
                            attempt: int) -> Action:
        if attempt <= 2:
            # Rung 1: Simple retry
            return Action("retry", prompt_hint=f"Previous error: {failure.message}")

        elif attempt <= 4:
            # Rung 2: Decompose into smaller subtasks
            smaller_tasks = self.decompose(subtask)
            return Action("decompose", new_tasks=smaller_tasks)

        elif attempt <= 5 and self.cloud_available:
            # Rung 3: Escalate to cloud LLM
            return Action("escalate", llm="cloud", subtask=subtask)

        else:
            # Rung 4: Ask the user (async, don't block)
            self.message_queue.send_to_user(
                f"Stuck on: {subtask.description}\n"
                f"Failed {attempt} times. Last error: {failure.message}\n"
                f"Can you help?",
                priority="decision", task_id=subtask.id,
                options=["Provide example code", "Skip this", "Simplify requirements"]
            )
            return Action("park", task_id=subtask.id)
```

**Critical:** At every rung, the escalation ladder logs what happened and what worked. The insight engine captures this: "OAuth2 implementation always needs cloud LLM" → next time, skip local attempts for OAuth2.

### Concern 4: The Bootstrap Effect is Overstated

**The problem:** Claiming "Gaia Code will build all other features 10x faster" is aspirational. The agent needs to be genuinely good before it can improve anything, and a V1 agent won't be 10x better than Claude Code on day 1.

**The honest solution: Supervised Bootstrap**

Accept the reality: the bootstrap is a **gradient**, not a switch.

```
V1.0 (Days 1-10):
  Built manually with Claude Code assist.
  Capability: basic coding tasks.
  Bootstrap value: 1x (no faster than manual).

V1.1 (Days 10-15):
  Agent proposes improvements to itself.
  Human reviews every proposal.
  Capability: solid coding with quality gates.
  Bootstrap value: 1.5x (useful but needs supervision).

V1.2 (Days 15-20):
  Agent builds tools, learns patterns.
  Human reviews ~50% of output.
  Capability: knows GAIA codebase, has custom tools.
  Bootstrap value: 2-3x (genuinely faster for GAIA-related work).

V2.0 (Days 20-30):
  Agent builds new agents with minimal supervision.
  Human reviews architecture decisions.
  Capability: deep GAIA knowledge, 50+ custom tools, 100+ insights.
  Bootstrap value: 3-5x for GAIA-specific work.

NEVER claim 10x on day 1. Earn it through accumulated learning.
```

**The key metric:** Track bootstrap multiplier honestly. Measure actual time saved vs. manual development. If the multiplier isn't improving, focus on the foundation (prompts, quality gates) instead of adding features.

### Concern 5: 1M+ LoC is Aspirational

**The problem:** Most real-world use cases involve 10K-100K LoC repositories. Supporting 1M+ LoC requires significant infrastructure (tree-sitter, incremental indexing, streaming queries) that may not pay off for months.

**The pragmatic solution: Progressive Scale Targets**

```
M0-M3 (MVP):     No indexing needed. Agent reads files on demand.
                  Works fine for repos up to ~50K LoC.

M7 (Codebase):   Basic indexing with AST parsing.
                  10K LoC: <5 sec index, <100ms query
                  100K LoC: <30 sec index, <100ms query
                  This covers 95% of real-world use cases.

Future:           1M+ LoC support.
                  Build only when we have a customer/use case that needs it.
                  Don't optimize for a scale nobody is using yet.
```

**Design for 100K, test at 100K, aspire to 1M+.** The architecture supports 1M+ (incremental indexing, streaming queries) but we don't invest engineering time in it until the 100K use case is proven.

### Summary of Concerns vs. Solutions

| Concern | Solution | When |
|---------|----------|------|
| LLM Quality | Smart escalation: local first, cloud per-subtask, cost ceiling | M0 (built in from start) |
| Checkpoint = Compaction | State-based resume: plan + files + gates, not prose summary | M3 (checkpoint milestone) |
| Quality gates loop | Escalation ladder: retry → decompose → cloud → ask user | M2 (quality gates milestone) |
| Bootstrap overstated | Supervised bootstrap: measure multiplier honestly, earn trust | Ongoing |
| 1M+ LoC | Progressive targets: 100K first, 1M+ when needed | M7 (if proven useful) |

---

## Summary

### Why Gaia Code First

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│    Gaia Code is not just a feature.                      │
│    It is the tool that builds all other features.        │
│                                                          │
│    Build it first. Build it right. Build it best.        │
│    Everything else follows.                              │
│                                                          │
│    Simple > Complex. Working > Complete.                  │
│    5 pillars done well > 18 features done poorly.        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Minimum Viable Implementation

| Phase | What | Hours | Milestone |
|-------|------|-------|-----------|
| **Phase 0** | **4 Prerequisites** (Config + Feature Flags, Knowledge DB, Checkpoint Base, Basic Retry) | **18h** | Infrastructure ready |
| **Phase 1** | **Pillar 1+2**: Context-Lean Architecture + Continuous Execution | **16h** | **Agent never compacts, never stops mid-task** |
| **Phase 2** | **Pillar 3+4**: Quality Gates + Checkpoint/Resume | **12h** | **Output verified, survives crashes** |
| **Phase 3** | **Pillar 5**: Persistent Memory + Knowledge Retrieval | **12h** | **Agent learns across sessions** |
| Phase 4 | Practical: Planning, task decomposition, progress tracking | 12h | Structured workflow |
| Phase 5 | Enhance: Learning loop, adaptive prompts | 12h | Compound improvement |

**MVP (Phases 0-3): ~58 hours (~2.5 days). Exceeds Claude Code on the 5 pillars.**

### Full Feature Reference

The detailed feature list (F1-F18) in this document is the **roadmap**, not the **MVP**. Implementation order:

| Phase | What | Hours | Milestone |
|-------|------|-------|-----------|
| **Phase 0** | **PREREQUISITES** (Config, Tool Registry, LLM Router, Error Recovery, Checkpoint Base, Memory Infra, Task Infra, Mixin Lifecycle) | **40h** | Infrastructure complete |
| Phase 1 | Core Coding Tools (AST, symbols, git, intelligence) | 22h | Claude Code equivalent |
| Phase 2 | Continuous Execution + Quality Gates + State Machine + Task Queue | 30h | **Exceeds Claude Code (MVP)** |
| Phase 3 | Memory (4 tiers) + Manifest + Adaptive Prompts + Learning + Self-Eval | 36h | Learning agent |
| Phase 4 | Dynamic Tools + Concurrent Threads + Proactive Analysis | 22h | Unprecedented |
| Phase 5 | Polish, Integration tests, CLI, TUI, Docs | 20h | Production ready |

### Core Features Summary

| # | Feature | Category | Claude Code? | LLM Tier Required | Priority |
|---|---------|----------|:---:|:---:|:---:|
| P0 | **Prerequisites (Config, Registry, Router, etc.)** | **Infrastructure** | N/A | All | **P0** |
| F1 | File System Operations | Foundation | Has basic | All | P0 |
| F2 | Terminal Execution | Foundation | Has basic | All | P0 |
| F3 | Git Operations | Foundation | Has basic | All | P0 |
| F4 | Code Intelligence | Foundation | Partial | All | P0 |
| F5 | Context Management | Foundation | Has basic | All | P0 |
| F6 | Error Handling | Foundation | Has basic | All | P0 |
| F7 | **Continuous Execution** | Autonomous | **NO** | Standard+ | **P0** |
| F8 | **Quality Gates** | Autonomous | **NO** | Basic (partial) | **P0** |
| F9 | **State Machine** | Autonomous | **NO** | Standard+ | **P0** |
| F10 | **Persistent Memory** | Intelligence | **NO** | Standard+ | **P1** |
| F11 | **Learning Loop** | Intelligence | **NO** | Advanced | **P1** |
| F12 | **Project Manifest** | Intelligence | **NO** | Standard+ | **P1** |
| F13 | **Adaptive Prompts** | Intelligence | **NO** | Advanced | **P1** |
| F14 | **Dynamic Tool Creation** | Superpower | **NO** | Advanced | **P2** |
| F15 | **Concurrent Task Threads** | Superpower | **NO** | Advanced | **P2** |
| F16 | **Self-Evaluation** | Superpower | **NO** | Advanced | **P2** |
| F17 | **Proactive Analysis** | Superpower | **NO** | Advanced | **P2** |
| F18 | **LLM Tier Optimization** | Superpower | **NO** | All | **P2** |

### Timeline

```
Day 1-2:   Phase 0: PREREQUISITES (build first, blocks everything)
Day 2-3:   Phase 1: Core Tools (match Claude Code)
Day 3-5:   Phase 2: Continuous Execution + Quality Gates (exceed Claude Code)
Day 5-7:   Phase 3: Memory + Learning (unprecedented)
Day 7-9:   Phase 4: Superpowers (unmatched)
Day 9-10:  Phase 5: Polish + Ship

Day 11+:   Use Gaia Code to build everything else 10x faster
```

### Composability Quick Reference

```
Tier 1 (Basic 0.6B):     File ops + Terminal + Git + Syntax check
Tier 2 (Standard 7B):    + Continuous exec + State machine + Memory + Task queue + Quality gates
Tier 3 (Advanced 30B+):  + Learning + Dynamic tools + Self-eval + Concurrent threads + Adaptive prompts
```

### Configuration Loading

V2Config loads from multiple sources with this priority (highest wins):

```
1. CLI flags:       gaia code --enable-memory --llm-tier=advanced "task"
2. Environment:     GAIA_ENABLE_MEMORY=true GAIA_LLM_TIER=advanced
3. Project config:  ./gaia.yaml (or .gaia/config.yaml)
4. User config:     ~/.gaia/config.yaml
5. Profile preset:  gaia code --profile=coding "task"
6. Defaults:        All features off, tier=basic
```

```yaml
# Example gaia.yaml (project-level config)
v2:
  llm_tier: standard
  enable_memory: true
  enable_continuous_execution: true
  enable_state_machine: true
```

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|:---:|:---:|------------|
| **Over-engineering** -- Building 18 features before any work well | HIGH | HIGH | Ship 5 pillars first. Add features only when validated need exists. |
| **Quality gate false positives** -- Gates reject valid output | MEDIUM | HIGH | Make gates configurable. Start strict (syntax + tests only), loosen later. |
| **Checkpoint/resume fidelity** -- State corrupted on resume | MEDIUM | HIGH | Comprehensive serialization tests. Verify full round-trip before shipping. |
| **Memory pollution** -- Bad patterns stored, degrade future sessions | MEDIUM | MEDIUM | Confidence decay (old patterns lose weight). User can clear/edit memory. |
| **LLM capability mismatch** -- Features enabled that LLM can't handle | LOW | MEDIUM | Tier validation at startup warns/blocks. Graceful degradation to simpler mode. |
| **Performance overhead** -- Memory/manifest add latency | LOW | MEDIUM | All features must pass zero-overhead test when disabled. Profile before shipping. |
| **Breaking existing agents** -- V2 changes break V1 CodeAgent | LOW | HIGH | GaiaCodeAgent inherits from CodeAgent. V1 class untouched. Full regression suite. |

---

### The North Star

**Gaia Code should be the tool that every developer reaches for instead of Claude Code -- not because it's free (though it is), but because it's genuinely better: it learns, it never forgets, it never stops mid-task, and it gets better every time you use it.**

**Keep it simple. Ship the 5 pillars. Iterate from there.**

---

*Gaia Code: The autonomous coding agent that builds itself.*
*Priority #1. Ship first. Use it to build everything else.*
*Simple > Complex. Working > Complete. 5 pillars > 18 features.*
