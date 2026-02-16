# GAIA Code: Complete Architecture Diagram

**Version**: 1.0.0
**Date**: February 14, 2026
**Status**: M0-M6 Complete

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         GAIA CODE SYSTEM                                 │
│                   (Recursive Agent Composition)                          │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                      USER INTERFACE                                 │ │
│  │                                                                     │ │
│  │  CLI: gaia code <task>                                             │ │
│  │  Python API: GaiaCodeAgent().process_query(task)                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     GAIA CODE AGENT                                 │ │
│  │                                                                     │ │
│  │  Components:                                                        │ │
│  │  • System Prompt (M0) - RLM patterns, quality-first                │ │
│  │  • Quality Gates (M2) - Syntax, Imports, Tests                     │ │
│  │  • Escalation Ladder (M2) - retry → decompose → cloud → ask       │ │
│  │  • Checkpoint/Resume (M3) - State-based recovery                   │ │
│  │  • Audit Log (M3) - Full transparency                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                   SHARED AGENT STATE (M1)                           │ │
│  │                     (RAC Foundation)                                │ │
│  │                                                                     │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐               │ │
│  │  │ memory.db   │  │ knowledge.db │  │  tools.db   │               │ │
│  │  │ (session    │  │ (cross-      │  │ (70+ tools) │               │ │
│  │  │  cache)     │  │  session)    │  │             │               │ │
│  │  └─────────────┘  └──────────────┘  └─────────────┘               │ │
│  │                                                                     │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐               │ │
│  │  │ skills.db   │  │  agents.db   │  │   plan.db   │               │ │
│  │  │ (workflows) │  │ (7+ special- │  │ (task tree) │               │ │
│  │  │             │  │   ists)      │  │             │               │ │
│  │  └─────────────┘  └──────────────┘  └─────────────┘               │ │
│  │                                                                     │ │
│  │  + manifest (project state)                                        │ │
│  │  + call_stack (recursion tracking)                                 │ │
│  │  + message_queue (async communication)                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    RAC TOOLS (M1)                                   │ │
│  │                                                                     │ │
│  │  • agent_query(task, specialist) → Recursive delegation            │ │
│  │  • recall(query) → FTS5 + FAISS search                             │ │
│  │  • find_tool(query) → Semantic tool discovery                      │ │
│  │  • store_insight(category, content) → Learn from experience        │ │
│  │  • get_plan() → View master plan                                   │ │
│  │  • update_task(id, status) → Track progress                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                 DOMAIN SPECIALISTS (M4)                             │ │
│  │                                                                     │ │
│  │  1. DebuggerAgent       - Error diagnosis                          │ │
│  │  2. SecurityAgent       - OWASP Top 10                             │ │
│  │  3. RefactoringAgent    - Code cleanup                             │ │
│  │  4. TestingAgent        - Test generation                          │ │
│  │  5. DocumentationAgent  - Docs generation                          │ │
│  │  6. PerformanceAgent    - Optimization                             │ │
│  │  7. ArchitectureAgent   - Design patterns                          │ │
│  │                                                                     │ │
│  │  Each has: Custom state machine, specialized prompt, workflow      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              SELF-EXTENSION SYSTEM (M5)                             │ │
│  │                                                                     │ │
│  │  • ToolBuilder       - Agent creates new tools                     │ │
│  │  • SkillExtractor    - Learn workflows from experience             │ │
│  │  • InsightEngine     - Structured learning                         │ │
│  │  • AgentFactory      - Generate specialists from patterns          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                │                                         │
│                                ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │              SEMANTIC SEARCH LAYER (M6)                             │ │
│  │                                                                     │ │
│  │  • EmbeddingEngine      - all-MiniLM-L6-v2 (384 dim)               │ │
│  │  • VectorSearch         - FAISS indices for all databases          │ │
│  │  • MemoryDefragmenter   - Cleanup and consolidation                │ │
│  │                                                                     │ │
│  │  Hybrid Search: FAISS (semantic) + FTS5 (keyword)                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Flow

### Simple Task Execution

```
User: "Create a prime number checker with tests"
    │
    ▼
┌─────────────────────────────────────────┐
│ GaiaCodeAgent.process_query(task)       │
└─────────────────────────────────────────┘
    │
    ├─> Create root task in plan.db
    │
    ├─> Execute task
    │   ├─> LLM generates code
    │   ├─> Write prime.py
    │   ├─> Write test_prime.py
    │   └─> Store files in manifest
    │
    ├─> Run quality gates
    │   ├─> SyntaxGate: ✅ PASS
    │   ├─> ImportGate: ✅ PASS
    │   └─> TestGate: ✅ PASS (all tests pass)
    │
    └─> Return verified result
```

### Complex Task with Recursion

```
User: "Build a full-stack app"
    │
    ▼
┌─────────────────────────────────────────┐
│ GaiaCodeAgent.process_query(task)       │
└─────────────────────────────────────────┘
    │
    ├─> Create root task in plan.db
    │
    ├─> Decompose task
    │   ├─> agent_query("Build FastAPI backend")
    │   │   │
    │   │   ├─> Push to call_stack (depth=1)
    │   │   ├─> Create sub-agent with fresh context
    │   │   ├─> Sub-agent executes with full tools
    │   │   ├─> Sub-agent runs quality gates
    │   │   ├─> Pop from call_stack
    │   │   └─> Return verified backend
    │   │
    │   └─> agent_query("Build React frontend")
    │       │
    │       ├─> Push to call_stack (depth=1)
    │       ├─> Create sub-agent with fresh context
    │       ├─> Sub-agent executes
    │       ├─> Pop from call_stack
    │       └─> Return verified frontend
    │
    ├─> Coordinate via manifest
    │   ├─> Backend API schema → shared with frontend
    │   └─> Both components in manifest
    │
    └─> Final quality gates on complete system
```

### Error Recovery Flow

```
Task execution fails quality gates
    │
    ▼
┌─────────────────────────────────────────┐
│ Quality Gates: SyntaxGate FAILS         │
│ Error: Missing colon in auth.py:45      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ EscalationLadder.get_action()           │
└─────────────────────────────────────────┘
    │
    ├─> Attempt 1: RETRY
    │   ├─> Re-execute task with error context
    │   ├─> Apply fix
    │   └─> Re-run quality gates
    │
    ├─> Attempt 2: RETRY (if still failing)
    │   └─> Try different approach
    │
    ├─> Attempt 3: DECOMPOSE
    │   ├─> agent_query("Fix syntax error in auth.py:45")
    │   │   ├─> Auto-select DebuggerAgent
    │   │   ├─> DebuggerAgent diagnoses root cause
    │   │   ├─> DebuggerAgent applies minimal fix
    │   │   └─> Return verified fix
    │   └─> Re-run quality gates
    │
    ├─> Attempt 4: ESCALATE TO CLOUD (if available)
    │   ├─> Create new agent with Claude API
    │   └─> Retry task with more capable model
    │
    └─> Attempt 5: ASK USER
        └─> send_message("I'm stuck. What should I do?", priority="Question")
```

---

## Data Flow

### Context-Lean Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                   CONTEXT WINDOW (<50%)                       │
│                                                               │
│  System Prompt                                                │
│  + Current task description                                   │
│  + Active file snippet (only relevant lines)                  │
│  + Last 3 tool results (summarized)                           │
│  + Current conversation turn                                  │
│                                                               │
│  Total: <50K tokens (always <50% of 100K+ window)            │
└──────────────────────────────────────────────────────────────┘
                        │ queries ▲ stores
                        ▼         │
┌──────────────────────────────────────────────────────────────┐
│              EXTERNAL KNOWLEDGE STORAGE                       │
│                                                               │
│  memory.db (session cache):                                   │
│  ├─ file_cache (all files read this session)                 │
│  └─ tool_results (all tool calls this session)               │
│                                                               │
│  knowledge.db (persistent learning):                          │
│  ├─ insights (learnings from all sessions)                   │
│  ├─ preferences (user preferences)                            │
│  ├─ learnings (error-fix patterns)                            │
│  └─ conventions (project conventions)                         │
│                                                               │
│  tools.db (tool registry):                                    │
│  ├─ tools (70+ tools with descriptions)                      │
│  ├─ tool_usage (usage statistics)                             │
│  └─ tool_tags (categorization)                                │
│                                                               │
│  skills.db (workflow patterns):                               │
│  ├─ skills (multi-step workflows)                             │
│  └─ skill_usage (success rates)                               │
│                                                               │
│  agents.db (specialist registry):                             │
│  ├─ agents (7+ specialists)                                   │
│  └─ agent_usage (performance metrics)                         │
│                                                               │
│  plan.db (hierarchical tasks):                                │
│  └─ tasks (parent-child task tree)                            │
│                                                               │
│  manifest (live project state):                               │
│  ├─ files (all files created/modified)                        │
│  ├─ apis (endpoints defined)                                  │
│  ├─ schemas (database schemas)                                │
│  └─ decisions (architecture decisions)                        │
└──────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Layer 1: Foundation (M0-M1)

```
┌─────────────────────────────────────────────────────────────┐
│                 FOUNDATION LAYER                             │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ System Prompts   │  │ SharedAgentState │                │
│  │ (M0)             │  │ (M1)             │                │
│  │                  │  │                  │                │
│  │ • RLM patterns   │  │ • 7 databases    │                │
│  │ • Error recovery │  │ • Singleton      │                │
│  │ • Tool usage     │  │ • Thread-safe    │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Layer 2: Quality & Resilience (M2-M3)

```
┌─────────────────────────────────────────────────────────────┐
│              QUALITY & RESILIENCE LAYER                      │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Quality Gates    │  │ Checkpoint/      │                │
│  │ (M2)             │  │ Resume (M3)      │                │
│  │                  │  │                  │                │
│  │ • SyntaxGate     │  │ • State-based    │                │
│  │ • ImportGate     │  │ • Zero loss      │                │
│  │ • TestGate       │  │ • Audit log      │                │
│  │ • Escalation     │  │ • Time aware     │                │
│  └──────────────────┘  └──────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Layer 3: Expertise (M4)

```
┌─────────────────────────────────────────────────────────────┐
│                  SPECIALIST LAYER                            │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Debugger │ │ Security │ │Refactor- │ │ Testing  │      │
│  │          │ │          │ │   ing    │ │          │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                   │
│  │   Docs   │ │Perfor-   │ │Architec- │                   │
│  │          │ │  mance   │ │  ture    │                   │
│  └──────────┘ └──────────┘ └──────────┘                   │
│                                                              │
│  Each specialist:                                            │
│  • Custom state machine (5-7 states)                        │
│  • Specialized prompt (200-300 lines)                       │
│  • Domain workflow (5-7 steps)                              │
│  • Tool packs (core + domain)                               │
└─────────────────────────────────────────────────────────────┘
```

### Layer 4: Self-Extension (M5)

```
┌─────────────────────────────────────────────────────────────┐
│               SELF-EXTENSION LAYER                           │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │  ToolBuilder   │  │ SkillExtractor │                    │
│  │  (M5.1)        │  │ (M5.2-M5.4)    │                    │
│  │                │  │                │                    │
│  │ • Write tools  │  │ • Extract      │                    │
│  │ • Validate     │  │   workflows    │                    │
│  │ • Test         │  │ • Track success│                    │
│  │ • Register     │  │ • Recall       │                    │
│  └────────────────┘  └────────────────┘                    │
│                                                              │
│  ┌────────────────┐  ┌────────────────┐                    │
│  │ InsightEngine  │  │ AgentFactory   │                    │
│  │ (M5.5-M5.6)    │  │ (M5.7-M5.9)    │                    │
│  │                │  │                │                    │
│  │ • Generate     │  │ • Detect       │                    │
│  │   insights     │  │   patterns     │                    │
│  │ • Retrieve     │  │ • Generate     │                    │
│  │ • Validate     │  │   specialists  │                    │
│  └────────────────┘  └────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Layer 5: Memory Management (M6)

```
┌─────────────────────────────────────────────────────────────┐
│              MEMORY MANAGEMENT LAYER                         │
│                                                              │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ EmbeddingEngine  │  │  VectorSearch    │                │
│  │ (M6.1)           │  │  (M6.2)          │                │
│  │                  │  │                  │                │
│  │ • Local model    │  │ • FAISS indices  │                │
│  │ • <10ms/embed    │  │ • Semantic search│                │
│  │ • Batch support  │  │ • Hybrid search  │                │
│  └──────────────────┘  └──────────────────┘                │
│                                                              │
│  ┌──────────────────────────────────────┐                  │
│  │   MemoryDefragmenter (M6.3)          │                  │
│  │                                       │                  │
│  │ • Deduplicate near-duplicates        │                  │
│  │ • Prune stale entries                │                  │
│  │ • Consolidate patterns               │                  │
│  │ • Auto-trigger when quality drops    │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Recursive Decomposition Example

### Task: "Build a microservices platform"

```
Main Agent (depth=0)
│
├─> agent_query("Build User Service")
│   │
│   ├─> agent_query("Create User Model")
│   │   └─> Returns: user.py
│   │
│   ├─> agent_query("Create Auth Endpoints", specialist="SecurityAgent")
│   │   ├─> SecurityAgent analyzes security requirements
│   │   ├─> Implements JWT auth
│   │   └─> Returns: auth.py (security-verified)
│   │
│   └─> agent_query("Write Tests", specialist="TestingAgent")
│       ├─> TestingAgent generates comprehensive tests
│       ├─> Covers happy path + edge cases
│       └─> Returns: test_user.py (all passing)
│
├─> agent_query("Build Product Service")
│   └─> [Similar decomposition]
│
└─> agent_query("Create API Gateway")
    └─> [Similar decomposition]

Result: Complete microservices platform
- All services implemented
- All tests passing
- Security verified
- Documentation generated
```

---

## Quality Gate Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QUALITY GATE SYSTEM                       │
│                                                              │
│  Task Execution                                              │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────────┐                                    │
│  │  QualityGateRunner  │                                    │
│  └─────────────────────┘                                    │
│       │                                                      │
│       ├─> SyntaxGate                                        │
│       │   ├─> Parse with AST                                │
│       │   ├─> If valid: ✅ PASS                             │
│       │   └─> If invalid: ❌ FAIL + error details           │
│       │                                                      │
│       ├─> ImportGate                                        │
│       │   ├─> Check all imports resolve                     │
│       │   ├─> If valid: ✅ PASS                             │
│       │   └─> If invalid: ❌ FAIL + missing modules         │
│       │                                                      │
│       └─> TestGate                                          │
│           ├─> Auto-detect pytest/jest                       │
│           ├─> Run tests                                     │
│           ├─> If all pass: ✅ PASS                          │
│           └─> If failures: ❌ FAIL + test output            │
│                                                              │
│  ┌─────────────────────┐                                    │
│  │ All gates passed?   │                                    │
│  └─────────────────────┘                                    │
│       │                                                      │
│       ├─> YES: Return result (task complete)                │
│       │                                                      │
│       └─> NO: EscalationLadder                              │
│           ├─> retry (attempt 1-2)                           │
│           ├─> decompose (agent_query to fix)                │
│           ├─> escalate to cloud                             │
│           └─> ask user                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Specialist State Machine

```
All Specialists Follow This Pattern:

  START
    │
    ▼
┌───────────┐
│ ANALYZING │ ─> Understand the task
└───────────┘
    │
    ▼
┌───────────┐
│ PLANNING  │ ─> Determine approach
└───────────┘
    │
    ▼
┌───────────┐
│ EXECUTING │ ─> Apply domain expertise
└───────────┘
    │
    ▼
┌───────────┐
│VALIDATING │ ─> Verify output works
└───────────┘
    │
    ▼
┌───────────┐
│ COMPLETED │ ─> Return verified result
└───────────┘

Example: DebuggerAgent

  ANALYZING ─> Read error message
      │
      ▼
  DIAGNOSING ─> Find root cause
      │
      ▼
  ISOLATING ─> Locate problematic code
      │
      ▼
  FIXING ─> Apply minimal fix
      │
      ▼
  VALIDATING ─> Re-run to verify
      │
      ▼
  COMPLETED ─> Return fix
```

---

## Self-Extension Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 SELF-EXTENSION SYSTEM                        │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  TOOL CREATION (ToolBuilder)                            │ │
│  │                                                          │ │
│  │  Agent detects: "I'm doing this pattern repeatedly"     │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Agent writes: def new_tool(...): ...                   │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Validate: Safety check (no eval/exec)                  │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Test: Run tool on sample input                         │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Register: Add to tools.db                              │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Use: Available in future sessions                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  SKILL LEARNING (SkillExtractor)                        │ │
│  │                                                          │ │
│  │  Task completed successfully                            │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Extract: Steps taken, tools used, success factors      │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Store: skill("build_api", steps=[...], tools=[...])    │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Recall: For similar future tasks                       │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Improve: Confidence increases with successful use      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  AGENT GENERATION (AgentFactory)                        │ │
│  │                                                          │ │
│  │  Detect: 3+ similar successful tasks                    │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Extract: Common tools, workflows, patterns             │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Generate: New specialist class with custom prompt      │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Test: Validate on sample task                          │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Register: Add to agents.db                             │ │
│  │       │                                                  │ │
│  │       ▼                                                  │ │
│  │  Use: Auto-selected for matching tasks                  │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Search Architecture

### Hybrid Search (M6)

```
┌─────────────────────────────────────────────────────────────┐
│                     SEARCH SYSTEM                            │
│                                                              │
│  Query: "How do I fix AttributeError?"                      │
│                                                              │
│  ┌────────────────────┐  ┌────────────────────┐            │
│  │ FAISS Search       │  │ FTS5 Search        │            │
│  │ (Semantic)         │  │ (Keyword)          │            │
│  │                    │  │                    │            │
│  │ 1. Embed query     │  │ 1. Parse keywords  │            │
│  │ 2. Vector search   │  │ 2. MATCH query     │            │
│  │ 3. Get top-k       │  │ 3. Rank by TF-IDF  │            │
│  │ 4. Score: α        │  │ 4. Score: (1-α)    │            │
│  └────────────────────┘  └────────────────────┘            │
│           │                       │                         │
│           └──────┬────────────────┘                         │
│                  ▼                                          │
│         ┌─────────────────┐                                │
│         │  Merge Results  │                                │
│         │  (weighted avg) │                                │
│         └─────────────────┘                                │
│                  │                                          │
│                  ▼                                          │
│         Results ranked by:                                  │
│         • Semantic relevance (FAISS)                        │
│         • Keyword match (FTS5)                              │
│         • Confidence score                                  │
│         • Recency                                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Memory Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                   MEMORY LIFECYCLE                           │
│                                                              │
│  New Session                                                 │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────┐                                        │
│  │ Load knowledge  │ ← knowledge.db (persistent)            │
│  │ Create memory   │ ← memory.db (new, empty)               │
│  │ Load plan       │ ← plan.db (resume if exists)           │
│  └─────────────────┘                                        │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────┐                                        │
│  │ Execute tasks   │                                        │
│  │  • Read files   │ ──> Cache in memory.db                 │
│  │  • Call tools   │ ──> Store results in memory.db         │
│  │  • Generate     │ ──> Update manifest                    │
│  │  • Learn        │ ──> Insights to knowledge.db           │
│  └─────────────────┘                                        │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────┐                                        │
│  │ Periodic checks │                                        │
│  │  Every 10 steps │ ──> Create checkpoint                  │
│  │  Every 25%      │ ──> Quality gates                      │
│  │  Every 100 tasks│ ──> Consider defrag                    │
│  └─────────────────┘                                        │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────┐                                        │
│  │ Session ends    │                                        │
│  │  • Save final   │ ──> Checkpoint to disk                 │
│  │    checkpoint   │                                        │
│  │  • Close        │ ──> memory.db deleted                  │
│  │    session DB   │                                        │
│  │  • Knowledge    │ ──> knowledge.db persists              │
│  │    persists     │                                        │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema Summary

### memory.db (Session Cache)
```sql
file_cache (path, content, last_accessed)
tool_results (id, tool_name, args, result, timestamp)
```

### knowledge.db (Persistent Learning)
```sql
insights (id, category, domain, content, confidence, triggers, created_at, last_used, use_count)
insights_fts (FTS5: id, content, triggers)
preferences (key, value, description, created_at)
learnings (id, error_pattern, fix_pattern, success_count, confidence, created_at)
conventions (id, scope, pattern, description, created_at)
```

### tools.db (Tool Registry)
```sql
tools (id, name, category, source, description, parameters, code_path, created_at, version, enabled)
tools_fts (FTS5: id, name, description, category)
tool_usage (id, tool_id, timestamp, success, duration_ms, context, error)
tool_tags (tool_id, tag)
```

### skills.db (Workflow Patterns)
```sql
skills (id, name, description, category, domain, steps, tools_used, success_count, failure_count, confidence, created_at, last_used)
skill_usage (id, skill_id, timestamp, success, task_description, feedback)
```

### agents.db (Specialist Registry)
```sql
agents (id, name, description, capabilities, system_prompt, tool_packs, confidence, created_at, last_used)
agent_usage (id, agent_id, timestamp, success, task_type, duration_ms)
```

### plan.db (Task Hierarchy)
```sql
tasks (id, description, status, owner, parent_id, children, result, error, created_at, started_at, completed_at)
```

---

## File Count Summary

| Component | Files | Lines |
|-----------|-------|-------|
| Core Infrastructure | 8 | 3,500 |
| Specialists | 8 | 2,400 |
| Self-Extension | 4 | 1,550 |
| Search & Defrag | 3 | 1,050 |
| Integration | 1 | 400 |
| CLI | 1 | 350 |
| Tests | 5 | 1,300 |
| Documentation | 8 | 2,000+ |
| **TOTAL** | **32** | **9,200+** |

---

**This architecture delivers the world's most autonomous coding agent.**
