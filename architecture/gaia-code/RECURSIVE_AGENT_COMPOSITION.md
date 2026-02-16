# Recursive Agent Composition (RAC): A New Paradigm

**Date**: February 11, 2026
**Version**: 1.0
**Status**: Core Architecture
**Foundation**: Extends [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab - MIT CSAIL, Dec 2025)

---

## Executive Summary

**Recursive Agent Composition (RAC)** is a novel AI agent architecture where agents recursively spawn specialized sub-agents to handle complex tasks, with all agents sharing global state (knowledge, manifest, plan).

**Key innovation:** Extends Recursive Language Models (RLMs) from stateless `llm_query()` calls to stateful `agent_query()` calls that spawn full agent instances with tools, verification capabilities, and access to shared resources.

**Result:** Agents that build their own specialists, improve exponentially over time, coordinate autonomously via shared state, and handle unlimited task complexity through recursive decomposition.

**This document defines the RAC paradigm for implementation in GAIA Code.**

---

## Table of Contents

1. [The Problem: Fixed-Capability Agents](#the-problem-fixed-capability-agents)
2. [The RLM Foundation](#the-rlm-foundation)
3. [From RLMs to RAC: The Extension](#from-rlms-to-rac-the-extension)
4. [Shared Agent State Architecture](#shared-agent-state-architecture)
5. [The Three-Level Capability Hierarchy](#the-three-level-capability-hierarchy)
6. [Agent Registry & Specialized Agents](#agent-registry--specialized-agents)
7. [Recursive Decomposition Patterns](#recursive-decomposition-patterns)
8. [Agent Auto-Generation](#agent-auto-generation)
9. [Threading Model: Single-Threaded with Async I/O](#threading-model-single-threaded-with-async-io)
10. [Coordination via Shared State](#coordination-via-shared-state)
11. [Implementation Plan](#implementation-plan)
12. [Research Implications](#research-implications)
13. [Comparison to Existing Architectures](#comparison-to-existing-architectures)

---

## The Problem: Fixed-Capability Agents

Every AI agent today has **fixed capabilities**:

| Agent | Tools | Learning | Self-Extension |
|-------|-------|----------|----------------|
| Claude Code | 30 fixed tools | None | No |
| Devin | Fixed workflow | Pattern recognition | No |
| Cursor | IDE tools | None | No |
| Aider | Git tools | None | No |

**The limitation:** An agent is only as good as its initial design. After 100 sessions, it's no better than session 1.

**What we need:** Agents that **build their own capabilities** and **improve over time**.

---

## The RLM Foundation

### What RLMs Solve

**Paper:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab - MIT, Dec 2025)

**Core idea:** Treat long inputs as variables in a Python REPL. The model writes code to programmatically decompose, filter, and recursively call itself on smaller chunks.

**Traditional LLM:**
```python
# Input: 1M-line codebase
# ❌ Can't fit in context window
```

**RLM:**
```python
# Input: codebase = [1M lines] (stored as variable, not in context)

# Model writes code:
files = list_files(codebase)
relevant = [f for f in files if "auth" in f]

results = []
for file in relevant:
    analysis = llm_query(f"Analyze {file}")  # Fresh context per call
    results.append(analysis)

report = llm_query(f"Synthesize: {results}")
```

**Key mechanism:** `llm_query(prompt)` - recursive LLM call with fresh context.

**Limitation:** `llm_query()` is stateless. Returns text only. No tools. No verification.

---

## From RLMs to RAC: The Extension

### The Critical Insight

**RLMs use `llm_query()` - why not `agent_query()`?**

```python
# RLM approach (stateless):
analysis = llm_query("Analyze this code for bugs")
# Returns: Text analysis
# Can't: Read files, make changes, run tests, verify

# RAC approach (stateful):
fix = agent_query(
    task="Debug and fix this failing test",
    specialist="debugger_agent",
    tools=["read_file", "edit_file", "run_tests"],
    max_depth=3
)
# Returns: Verified fix + audit trail
# Can: Read files, make changes, run tests, recurse if needed
```

### What agent_query() Adds

| Feature | llm_query() (RLM) | agent_query() (RAC) |
|---------|------------------|---------------------|
| Fresh context | ✅ Yes | ✅ Yes |
| Recursive calls | ✅ Yes | ✅ Yes |
| Has tools | ❌ No | ✅ Yes (full toolkit) |
| Has memory | ❌ No | ✅ Yes (shared knowledge DB) |
| Can verify output | ❌ No | ✅ Yes (quality gates) |
| State tracking | ❌ No | ✅ Yes (call stack, audit log) |
| Returns structured data | ❌ Text only | ✅ Result + metadata |
| Can spawn specialists | ❌ No | ✅ Yes (agent registry) |
| Thread-safe | N/A | ✅ Yes (shared state with locks) |

**RAC = RLMs + Full Agency + Shared State + Specialization**

---

## Shared Agent State Architecture

### The Core Design

**Problem:** Recursive agents without shared state create chaos (incompatible solutions, duplicated work, no coordination).

**Solution:** All agents in the recursion tree share a single `SharedAgentState` instance.

```python
@dataclass
class SharedAgentState:
    """Singleton shared by ALL agents in the recursion tree."""

    # 1. MEMORY (cross-session persistence)
    memory_db: MemoryDB              # Working memory cache, session state
    knowledge_db: KnowledgeDB        # Insights, preferences, learnings

    # 2. PROJECT MANIFEST (current project state)
    manifest: ProjectManifest        # Files, APIs, schemas, decisions

    # 3. MASTER PLAN (hierarchical task tree)
    plan: MasterPlan                 # Goal, tasks, subtasks, progress

    # 4. CALL STACK (recursion tracking)
    call_stack: AgentCallStack       # Who's doing what, at what depth

    # 5. MESSAGE QUEUE (async communication)
    message_queue: MessageQueue      # Agent ↔ user, agent ↔ agent

    # 6. AGENT REGISTRY (available specialists)
    agent_registry: AgentRegistry    # Core + learned specialists

    # 7. TOOL & SKILL DATABASES
    tools_db: ToolsDB               # 78 core + learned tools
    skills_db: SkillsDB             # Learned workflows

    # Thread safety (for concurrent agent execution)
    _locks: dict[str, threading.Lock]

    # Session metadata
    session_id: str
    session_start: datetime
    session_goal: str
```

### Thread Safety Strategy

**Single-threaded execution with async I/O** (see Threading Model section).

Locks protect shared resources when we do need concurrent access:

```python
# Agent 1 updates manifest
with shared.lock("manifest"):
    shared.manifest.add_file("backend/api.py", created_by=agent1.id)

# Agent 2 updates plan (different lock, no conflict)
with shared.lock("plan"):
    shared.plan.complete_task(task_id, result)
```

**Read-heavy, write-light:** Most operations are reads (thread-safe by default). Writes are infrequent and serialized.

---

## The Three-Level Capability Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                LEVEL 3: SPECIALIZED AGENTS                   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │ Debugger   │ │ Security   │ │Refactoring │ ...          │
│  │ Agent      │ │ Agent      │ │ Agent      │              │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘              │
│        │              │              │                      │
│        └──────────────┴──────────────┘                      │
│                       │                                     │
│              orchestrate via agent_query()                  │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│                LEVEL 2: SKILLS                               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │build_fastapi│ │debug_async │ │setup_docker│ ...          │
│  │   _jwt_app │ │   _python  │ │  _compose  │              │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘              │
│        │              │              │                      │
│        └──────────────┴──────────────┘                      │
│                       │                                     │
│                  composed from                              │
└───────────────────────┼─────────────────────────────────────┘
                        │
┌───────────────────────┼─────────────────────────────────────┐
│                LEVEL 1: TOOLS                                │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │ read_file  │ │ run_tests  │ │git_commit  │ ... (78+)    │
│  └────────────┘ └────────────┘ └────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

**Each level is created and managed dynamically:**
- Tools: Developers write core tools; agent creates learned tools
- Skills: Agent extracts from successful task patterns
- Agents: Developers write core specialists; agent creates learned specialists

---

## Agent Registry & Specialized Agents

### Core Specialists (Ship with GAIA)

```python
# 7 core specialized agents (pre-installed)

class DebuggerAgent(GaiaCodeAgent):
    """Specialist in debugging with custom state machine."""
    DOMAIN = "debugging"
    STATE_MACHINE = ["identify_error", "isolate_cause", "form_hypothesis",
                     "test_hypothesis", "verify_fix", "check_regressions"]
    TOOLS = ["read_file", "edit_file", "run_tests", "trace_execution"]
    SYSTEM_PROMPT = "You are a debugging specialist. Always read full errors,
                     form hypotheses, test with minimal changes, verify with tests."

class SecurityAgent(GaiaCodeAgent):
    """Specialist in security analysis and hardening."""
    DOMAIN = "security"
    CAPABILITIES = ["detect_sql_injection", "detect_xss", "find_secrets",
                   "check_dependencies", "analyze_auth"]
    TOOLS = ["read_file", "run_security_scan", "detect_secrets", "check_deps"]

class RefactoringAgent(GaiaCodeAgent):
    """Specialist in code refactoring without behavior changes."""
    DOMAIN = "refactoring"
    SKILLS = ["extract_function", "inline_variable", "rename_symbol",
              "remove_dead_code"]
    TOOLS = ["read_file", "edit_file", "run_tests", "analyze_ast"]

class TestingAgent(GaiaCodeAgent):
    """Specialist in test generation and coverage improvement."""
    DOMAIN = "testing"
    TOOLS = ["write_file", "run_tests", "check_coverage", "mutation_test"]

class DocumentationAgent(GaiaCodeAgent):
    """Specialist in code documentation."""
    DOMAIN = "documentation"
    TOOLS = ["read_file", "write_file", "generate_docstrings", "create_readme"]

class PerformanceAgent(GaiaCodeAgent):
    """Specialist in performance optimization."""
    DOMAIN = "performance"
    TOOLS = ["profile_code", "benchmark", "analyze_complexity"]

class ArchitectureAgent(GaiaCodeAgent):
    """Specialist in architecture analysis and design."""
    DOMAIN = "architecture"
    TOOLS = ["analyze_structure", "detect_antipatterns", "suggest_refactoring"]
```

### Agent Registry

```python
class AgentRegistry:
    """Registry of all agents: core + learned."""

    def __init__(self, db_path: str = "~/.gaia/agents/agents.db"):
        self.db = sqlite3.connect(db_path)
        self.faiss_index = faiss.read_index("~/.gaia/agents/agents_vectors.faiss")
        self.core_agents = self._load_core_agents()

    def find_specialist(self, task: str) -> AgentInfo | None:
        """Semantic search for best specialist for this task."""
        query_vec = self.embed(task)
        distances, indices = self.faiss_index.search(query_vec, k=5)
        candidates = [self._load_agent_info(idx) for idx in indices[0]]

        # Rank by: semantic match + confidence + recent success
        return self._rank_agents(candidates, task)[0] if candidates else None

    def create_specialist(self, name: str, description: str, domain: str,
                         tools: list[str], skills: list[str],
                         state_machine: list[str], system_prompt: str,
                         code: str) -> bool:
        """Create a new learned specialist agent."""
        # Validate, test, register (full implementation in main spec)
        ...
```

---

## Recursive Decomposition Patterns

### Pattern 1: Sequential Decomposition

**Task:** "Build a full-stack app"

```python
# Main agent writes this code in its reasoning:

# Step 1: Backend first (defines API contract)
backend = agent_query(
    task="Build FastAPI backend with auth, products, orders",
    specialist="backend_agent",
    max_depth=3
)

# Backend agent internally decomposes:
#   auth = agent_query("Implement auth", specialist="auth_agent")
#   products = agent_query("Implement products CRUD")
#   orders = agent_query("Implement orders CRUD")

# Step 2: Frontend second (consumes API from manifest)
frontend = agent_query(
    task="Build React frontend",
    specialist="frontend_agent",
    context={"api_schema": backend.api_schema},  # From shared manifest
    max_depth=3
)

# Step 3: Integration tests
tests = agent_query(
    task="Write and run integration tests",
    specialist="testing_agent",
    max_depth=2
)
```

**Call tree:**
```
Main (d0)
├─ Backend (d1)
│  ├─ Auth (d2)
│  ├─ Products (d2)
│  └─ Orders (d2)
├─ Frontend (d1)
│  ├─ Components (d2)
│  └─ Routing (d2)
└─ Testing (d1)
   └─ Debugger (d2) [if tests fail]
```

### Pattern 2: Parallel Decomposition (Async I/O, Not Threads)

**Using async/await for concurrent execution:**

```python
import asyncio

# Spawn multiple specialists concurrently (async I/O)
results = await asyncio.gather(
    agent_query("Build backend", specialist="backend_agent"),
    agent_query("Build frontend", specialist="frontend_agent"),
    agent_query("Set up Docker", specialist="docker_agent")
)

# All three run concurrently via async I/O (event loop)
# NOT threads - single-threaded with cooperative multitasking
# Shared state accessed via locks during I/O waits
```

**Why async/await instead of threads:** Simpler, no true parallelism bugs, easier to debug. LLM calls are I/O-bound anyway (waiting for inference), so async is sufficient.

### Pattern 3: Recursive Debugging

**Task:** "Fix all failing tests"

```python
test_output = run_tests()
failures = parse_test_failures(test_output)

for failure in failures:
    fix = agent_query(
        task=f"Fix {failure.test_name}",
        specialist="debugger_agent",
        context={"error": failure.error, "file": failure.file},
        max_depth=2
    )

    # Debugger might recursively call:
    #   - Refactoring agent (if the fix needs cleanup)
    #   - Security agent (if the bug was security-related)

    if fix.success:
        verify = run_tests([failure.test_name])
        if not verify.passed:
            # Recurse deeper if fix didn't work
            deeper_fix = agent_query(
                task=f"Fix still failing: {failure.test_name}",
                context={"previous_fix": fix, "still_failing": verify.output},
                max_depth=1
            )
```

---

## Threading Model: Single-Threaded with Async I/O

### The Decision: Simplicity Over Parallelism

**Option A: Multi-threaded (complex)**
```python
# Spawn actual OS threads for parallel execution
thread1 = Thread(target=agent_query, args=("Build backend",))
thread2 = Thread(target=agent_query, args=("Build frontend",))

# Problems:
# - Race conditions on shared state
# - Debugging nightmares
# - Non-deterministic execution
# - Complex locking everywhere
```

**Option B: Single-threaded with async/await (simple) ← WE CHOOSE THIS**
```python
# Single thread, event loop, cooperative multitasking
results = await asyncio.gather(
    agent_query("Build backend"),
    agent_query("Build frontend")
)

# Benefits:
# - Deterministic execution
# - Easy to debug (single thread of execution)
# - Simple locking (only during I/O waits)
# - LLM calls are I/O-bound anyway (async is enough)
```

### How Async Works Here

```python
async def agent_query(task: str, specialist: str = None, ...) -> AgentResult:
    """Async agent spawning. Yields during LLM calls."""

    # 1. Create sub-agent (synchronous)
    sub_agent = self._create_sub_agent(specialist)

    # 2. Execute task (yields during LLM calls and tool I/O)
    result = await sub_agent.execute_async(task)  # Yields here

    # 3. Return (synchronous)
    return result


class GaiaCodeAgent:
    async def execute_async(self, task: str) -> Result:
        """Execute with async/await for concurrent decomposition."""

        # LLM calls yield to event loop
        action = await self.llm.query_async("What should I do first?")

        # Tool calls that do I/O yield
        if action.requires_file_read:
            content = await read_file_async(action.file_path)

        # Recursive agent calls yield
        if action.requires_specialist:
            result = await agent_query(
                task=action.subtask,
                specialist=action.specialist
            )
```

**Key points:**
- Single Python thread (simple)
- Event loop handles concurrency (async/await)
- Yields during: LLM calls, file I/O, subprocess execution
- Shared state accessed during yields (locks prevent conflicts)
- Deterministic execution order (for debugging)

### When Parallelism Isn't Needed

**Most agent operations are sequential by nature:**
```python
# This MUST be sequential:
design = agent_query("Design the database schema")
backend = agent_query("Build backend", context={"schema": design.schema})
tests = agent_query("Test backend", context={"api": backend.api})
```

**Async gives us concurrency where it matters (I/O-bound), without the complexity of true parallelism.**

### Optional: Thread Pool for CPU-Bound Work

**If needed later (probably not for MVP):**
```python
# For CPU-intensive operations only (tree-sitter parsing, embedding generation)
with ThreadPoolExecutor(max_workers=4) as pool:
    embeddings = pool.map(embed_function, large_text_list)

# But agent orchestration stays single-threaded + async
```

---

## Coordination via Shared State

### The Five Shared Resources

Every agent reads from and writes to:

**1. Memory DB (working memory cache)**
```python
shared.memory_db.cache_file("src/api.py", content)
shared.memory_db.get_cached_file("src/api.py")  # Faster than disk read
```

**2. Knowledge DB (cross-session learning)**
```python
shared.knowledge_db.store_insight(Insight(...))
insights = shared.knowledge_db.get_relevant_insights("debugging async")
```

**3. Project Manifest (current state)**
```python
shared.manifest.add_api_endpoint("/users", method="POST", schema={...})
endpoints = shared.manifest.get_all_endpoints()  # Other agents see this
```

**4. Master Plan (hierarchical tasks)**
```python
shared.plan.add_task("Implement auth", parent_id=task2_id, owner=backend_agent.id)
next_task = shared.plan.get_next_task()
```

**5. Message Queue (async communication)**
```python
shared.message_queue.send_to_user("Should I use Redis?", priority="question")
answer = shared.message_queue.check_answer(question_id)
```

### Context Injection

When spawning a sub-agent, inherit context from shared state:

```python
def _prepare_context_for_subagent(self) -> dict:
    """Build context snapshot for sub-agent."""
    return {
        "overall_goal": self.shared.plan.goal,
        "completed_tasks": [t.desc for t in self.shared.plan.completed],
        "api_endpoints": self.shared.manifest.get_all_endpoints(),
        "decisions": self.shared.manifest.decisions,
        "conventions": self.shared.knowledge_db.get_conventions(),
        "insights": self.shared.knowledge_db.get_relevant_insights(task),
    }
```

Sub-agent's prompt automatically includes:
```
OVERALL GOAL: Build full-stack e-commerce app
COMPLETED: Database schema, Auth endpoints
ACTIVE: Frontend (another agent is working on this)
DECISIONS: auth=JWT, database=PostgreSQL
CONVENTIONS: Use pathlib, run black before commit
INSIGHTS: JWT should use HS256, always validate tokens
```

---

## Agent Auto-Generation

### Pattern Detection

```python
class AgentFactory:
    """Detects patterns and creates specialized agents."""

    CREATION_THRESHOLD = 3  # Create specialist after 3 similar successful tasks

    def after_task_complete(self, task: Task):
        """Check if we should create a new specialist."""

        # Find similar past tasks
        similar = self.shared.knowledge_db.find_similar_tasks(
            task.description,
            min_similarity=0.8,
            min_success_rate=0.9
        )

        if len(similar) >= self.CREATION_THRESHOLD:
            # All used similar tools and workflow?
            common_tools = self._extract_common_tools(similar)
            common_workflow = self._extract_common_workflow(similar)

            if common_tools and common_workflow:
                # Generate specialist agent
                agent_code = self._generate_agent_code(
                    name=self._suggest_name(similar),
                    domain=self._detect_domain(similar),
                    tools=common_tools,
                    workflow=common_workflow,
                    skills=self._extract_skills(similar)
                )

                # Test on a sample task
                test_passed = self._test_agent(agent_code, similar[0])

                if test_passed:
                    # Register the new specialist
                    self.shared.agent_registry.create_specialist(
                        name=agent_code.class_name,
                        code=agent_code.source,
                        confidence=0.6  # Initial confidence
                    )

                    # Notify user
                    self.shared.message_queue.send_to_user(
                        f"Created new specialist: {agent_code.class_name}",
                        priority="fyi"
                    )
```

### Example: FastAPICRUDAgent Creation

**Trigger:** After completing 3 tasks that all followed the same pattern:
- Session 5: "Create CRUD for users"
- Session 8: "Create CRUD for posts"
- Session 12: "Create CRUD for comments"

**Agent generates:**

```python
class FastAPICRUDAgent(GaiaCodeAgent):
    """Generated by main agent on 2026-02-11.
    Specialized in creating FastAPI CRUD endpoints.
    """

    DOMAIN = "web_dev"
    TOOLS = ["write_file", "edit_file", "run_tests", "run_black"]

    WORKFLOW = [
        "create_sqlalchemy_model",
        "create_pydantic_schemas",
        "create_crud_routes",
        "write_tests",
        "verify_all_tests_pass"
    ]

    SYSTEM_PROMPT = """
    You create FastAPI CRUD endpoints following this pattern:
    1. SQLAlchemy model in src/models/{entity}.py
    2. Pydantic schemas (Create, Update, Response) in src/schemas/{entity}.py
    3. Five routes (list, get, create, update, delete) in src/routes/{entity}.py
    4. Pytest tests in tests/test_{entity}.py
    5. Verify all tests pass before returning

    Always: Follow project conventions, use existing patterns, run black.
    """

    async def execute_async(self, task: str) -> Result:
        # Parse entity name and fields from task
        entity = self._parse_entity(task)

        # Execute workflow
        model = await self.create_sqlalchemy_model(entity)
        schemas = await self.create_pydantic_schemas(entity)
        routes = await self.create_crud_routes(entity)
        tests = await self.write_tests(entity)

        # Verify
        verification = await self.quality_gates.verify_all()

        if not verification.passed:
            # Recursively fix issues
            fix = await agent_query(
                task=f"Fix: {verification.failures}",
                specialist="debugger_agent",
                max_depth=1
            )

        return Result(success=True, entity=entity)
```

**Saved to:** `~/.gaia/agents/learned/fastapi_crud_agent.py`
**Registered in:** `agents.db` with confidence=0.6

---

## Implementation Plan

### Milestone 1: Core Infrastructure (Days 3-5)

**Build SharedAgentState first (this is foundational):**

| # | What | Hours |
|---|------|-------|
| 1.1 | memory.db (working memory cache) | 3h |
| 1.2 | knowledge.db (insights, preferences) | 6h |
| 1.3 | ProjectManifest class | 4h |
| 1.4 | MasterPlan class | 4h |
| 1.5 | AgentCallStack class | 3h |
| 1.6 | MessageQueue class | 4h |
| 1.7 | SharedAgentState class (integrates above) | 2h |
| 1.8 | `agent_query()` tool (basic version) | 4h |

**Total: 30h**

This must be built BEFORE any specialized agents work.

### Milestone 4: Agent Registry + Core Specialists (Days 10-13)

| # | What | Hours |
|---|------|-------|
| 4.1 | agents.db schema + FAISS | 4h |
| 4.2 | AgentRegistry class | 4h |
| 4.3 | Build 7 core specialists | 12h |
| 4.4 | Update agent_query() to support specialists | 4h |
| 4.5 | Agent search and selection | 3h |

**Total: 27h**

### Milestone 5: Agent Auto-Generation (Days 13-16)

| # | What | Hours |
|---|------|-------|
| 5.1 | AgentFactory pattern detection | 6h |
| 5.2 | Agent code generation | 6h |
| 5.3 | Agent testing framework | 4h |
| 5.4 | Agent confidence tracking | 3h |

**Total: 19h**

---

## Research Implications

### Novel Contributions

**1. Extension of RLMs to full agents**
- RLMs: `llm_query()` (stateless text → text)
- RAC: `agent_query()` (stateful task → verified result)

**2. Shared state across recursion tree**
- Multi-agent systems: isolated agents with sync problems
- RAC: all agents share manifest, plan, knowledge DB

**3. Three-level capability hierarchy**
- Tools (functions) → Skills (workflows) → Agents (specialists)
- Each level created dynamically from the level below

**4. Agent auto-generation**
- Agents create specialists from patterns
- Specialists create sub-specialists
- Exponential capability growth

**5. Smart escalation with learning**
- Per-subtask LLM routing (local → cloud)
- Learn which tasks need cloud over time
- Cost ceiling enforcement

### Potential Papers

**Paper 1:** "Recursive Agent Composition: Extending RLMs to Self-Improving Agent Hierarchies"
- RAC paradigm definition
- Comparison to RLMs
- Benchmark results vs Claude Code, Devin
- Capability growth curves

**Paper 2:** "Shared State Coordination in Recursive Multi-Agent Systems"
- SharedAgentState architecture
- Thread-safety without threads (async/await model)
- Manifest as coordination mechanism
- Emergent specialist collaboration

**Paper 3:** "Agent Auto-Generation: Learning to Create Domain Specialists"
- Pattern detection from task similarity
- Agent code generation methodology
- Confidence tracking and promotion
- Long-term capability growth metrics

---

## Flexibility: From Single-Turn to Multi-Month Projects

### The Composability Design

**Critical requirement:** The same architecture must handle:
- Simple single-turn: "Fix this typo" (1 LLM call, 0.6B model, 5 seconds)
- Multi-turn chat: "Help me understand this code" (10 turns, 7B model, 2 minutes)
- Complex coding: "Build a REST API" (100+ turns, 30B model, 1 hour)
- Multi-day projects: "Build a microservices platform" (continuous, cloud escalation, 3 days)
- Multi-month autonomous: "Build GAIA's CUA agent" (specialists, learning, 2 months)

**The RAC architecture handles all five through progressive feature enablement:**

### Configuration Profiles

```python
# Profile 1: MINIMAL (single-turn, small LLM)
V2Config(
    llm_tier="basic",              # 0.6B model
    enable_recursion=False,        # No agent_query(), direct execution
    enable_memory=False,           # No cross-session persistence
    enable_quality_gates=False,    # No verification (fast response)
    enable_specialists=False       # No agent registry
)

# Use case: "Fix typo in README.md"
# Behavior: Read file, make edit, return. 5 seconds total.


# Profile 2: STANDARD (multi-turn, medium LLM)
V2Config(
    llm_tier="standard",           # 7B model
    enable_recursion=True,         # Can call agent_query() (depth=2)
    enable_memory=True,            # Session memory only (memory.db)
    enable_quality_gates=True,     # Syntax + test gates
    enable_specialists=False       # No specialists yet
)

# Use case: "Build a Python CLI tool with tests"
# Behavior: Creates plan, implements, runs tests, fixes errors.
#           Can recursively decompose (depth=2) if needed.
#           30 minutes. Remembers within session.


# Profile 3: CODING (complex tasks, large local LLM)
V2Config(
    llm_tier="advanced",           # 30B model (local)
    enable_recursion=True,         # Full recursion (depth=10)
    enable_memory=True,            # Full: memory.db + knowledge.db
    enable_quality_gates=True,     # All 8 gates
    enable_specialists=True,       # 7 core specialists available
    enable_learning=True,          # Insights, skills, tool creation
    enable_smart_escalation=True,  # Can escalate to cloud (cost ceiling)
    max_cloud_cost=2.00           # $2/session limit
)

# Use case: "Build a FastAPI app with auth, CRUD, tests, Docker"
# Behavior: Recursively decomposes, delegates to specialists,
#           learns patterns, creates tools/skills. If stuck, escalates
#           complex subtasks to cloud. 2-4 hours. Remembers across sessions.


# Profile 4: AUTONOMOUS (multi-day/month, cloud hybrid)
V2Config(
    llm_tier="advanced",
    enable_recursion=True,
    enable_memory=True,
    enable_quality_gates=True,
    enable_specialists=True,
    enable_learning=True,
    enable_agent_generation=True,  # Can create new specialists
    enable_smart_escalation=True,
    enable_checkpoint_resume=True,
    enable_async_interaction=True, # User can check in weekly
    max_cloud_cost=50.00,          # $50/session for complex reasoning
    session_timeout=None           # No timeout (runs for days/weeks)
)

# Use case: "Build GAIA's computer-use agent. Work for the next month."
# Behavior: Creates master plan (100+ tasks), spawns specialists,
#           creates new specialists from patterns, delegates recursively,
#           checkpoints daily, user checks in weekly via message queue.
#           Runs for weeks. Cloud escalation for hard subtasks.
```

### Feature Degradation by LLM Tier

| Feature | 0.6B (basic) | 7B (standard) | 30B (advanced) | Cloud (opus) |
|---------|:---:|:---:|:---:|:---:|
| Tools (read, write, edit) | ✅ | ✅ | ✅ | ✅ |
| Recursion (agent_query) | ❌ | ✅ depth=2 | ✅ depth=10 | ✅ depth=10 |
| Memory (session) | ❌ | ✅ | ✅ | ✅ |
| Memory (cross-session) | ❌ | ❌ | ✅ | ✅ |
| Quality gates (basic) | ❌ | ✅ syntax+test | ✅ all 8 | ✅ all 8 |
| Specialists | ❌ | ❌ | ✅ | ✅ |
| Agent generation | ❌ | ❌ | ✅ | ✅ |
| Smart escalation | N/A | To advanced | To cloud | N/A |

**The same codebase.** Feature flags control what's active.

### Recursion Depth by Use Case

| Use Case | Recursion Depth | Why |
|----------|:---:|-----|
| Fix typo | 0 (no recursion) | Direct execution faster |
| Build CLI tool | 1-2 | Plan → implement → test |
| Build REST API | 2-4 | Main → specialists → sub-specialists |
| Build full-stack app | 3-6 | Main → backend/frontend → auth/CRUD/tests → debuggers |
| Multi-month project | 4-10 | Deep hierarchies, specialists creating specialists |

**Bounded recursion:** max_depth prevents infinite loops while allowing complex decomposition.

### Memory Tiers by Use Case

| Use Case | Memory Needed | Configuration |
|----------|---------------|---------------|
| Single-turn | None | memory=False |
| Multi-turn session | Session cache (memory.db) | memory=True, knowledge=False |
| Cross-session coding | Insights, preferences (knowledge.db) | memory=True, knowledge=True |
| Long-running autonomous | Full learning, defrag | memory=True, knowledge=True, defrag=True |

**Graceful degradation:** If memory.db isn't available, agent still works (just doesn't cache). If knowledge.db isn't available, agent doesn't learn across sessions but completes current tasks.

### Quality Gates by Complexity

| Task Complexity | Gates Enabled | Rationale |
|----------------|---------------|-----------|
| Simple edits | None | Fast response more important than verification |
| Standard coding | Syntax + Tests | Catch 90% of issues |
| Production code | All 8 gates | Full verification before declaring done |

**User-configurable:** `--strict` enables all gates, `--fast` disables verification.

---

## Comparison to Existing Architectures

### vs. RLMs (our foundation)

| | RLMs | RAC |
|---|------|-----|
| Recursive calls | ✅ llm_query() | ✅ agent_query() |
| Fresh context per call | ✅ Yes | ✅ Yes |
| Stateful | ❌ No | ✅ Yes (shared state) |
| Has tools | ❌ No | ✅ Yes |
| Can verify | ❌ No | ✅ Yes (quality gates) |
| Can create new capabilities | ❌ No | ✅ Yes (tools, skills, agents) |

**RAC is RLMs + agency + memory + self-extension**

### vs. Multi-Agent Systems (prior art)

| | Traditional Multi-Agent | RAC |
|---|------------------------|-----|
| Multiple agents | ✅ Yes | ✅ Yes (recursive) |
| Shared memory | ❌ Manual sync | ✅ SharedAgentState |
| Coordination | ❌ Complex protocols | ✅ Via manifest |
| Agent creation | ❌ Fixed at design time | ✅ Dynamic from patterns |
| State space | Finite | Infinite (compositional) |
| Hierarchy | Flat | Recursive tree |

**RAC solves the coordination problem that plagues multi-agent systems**

### vs. Hierarchical RL (inspiration)

| | Hierarchical RL (Options) | RAC |
|---|--------------------------|-----|
| Hierarchy | ✅ Actions → Options → Policies | ✅ Tools → Skills → Agents |
| Temporal abstraction | ✅ Yes | ✅ Yes |
| Learning | ✅ Reinforcement learning | ✅ Pattern extraction + confidence |
| Recursion | ❌ Fixed depth | ✅ Dynamic (bounded by max_depth) |
| Shared state | ❌ Markov property | ✅ Full shared memory |

**RAC applies hierarchical RL concepts to LLM agents**

---

## The Paradigm in One Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│          RECURSIVE AGENT COMPOSITION (RAC) PARADIGM              │
│                                                                  │
│  Main Agent (orchestrator)                                      │
│       │                                                          │
│       ├─ agent_query(specialist="backend") ────────┐            │
│       │                                             ↓            │
│       │                                    Backend Agent        │
│       │                                             │            │
│       │                                    agent_query("auth")  │
│       │                                             ↓            │
│       │                                      Auth Agent         │
│       │                                        (depth 2)        │
│       │                                                          │
│       ├─ agent_query(specialist="frontend") ───────┐            │
│       │                                             ↓            │
│       │                                   Frontend Agent        │
│       │                                             │            │
│       │                                    agent_query("UI")    │
│       │                                             ↓            │
│       │                                     UI Agent           │
│       │                                       (depth 2)         │
│       │                                                          │
│       └─ agent_query(specialist="testing") ────────┐            │
│                                                     ↓            │
│                                            Testing Agent        │
│                                                     │            │
│                                           agent_query("debug") │
│                                                     ↓            │
│                                             Debugger Agent      │
│                                                (depth 2)        │
│                                                                  │
│  ALL AGENTS SHARE:                                              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  SharedAgentState                                       │    │
│  │  • memory.db (cache)                                    │    │
│  │  • knowledge.db (insights, learnings)                   │    │
│  │  • manifest (files, APIs, decisions)                    │    │
│  │  • plan (hierarchical tasks)                            │    │
│  │  • call_stack (recursion tree)                          │    │
│  │  • message_queue (async communication)                  │    │
│  │  • tools.db, skills.db, agents.db                       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  EXECUTION: Single-threaded + async/await                       │
│  COORDINATION: Via shared manifest + knowledge DB               │
│  LEARNING: Creates tools → skills → agents                      │
│  IMPROVEMENT: Exponential (specialists create specialists)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why This Changes Everything

**Traditional agent paradigm:**
```
Build the best agent you can
→ Ship it
→ Hope it's good enough
→ Capability is fixed
```

**RAC paradigm:**
```
Build a simple agent with the ability to extend itself
→ Ship it
→ It creates specialists for recurring patterns
→ Specialists create sub-specialists
→ Capabilities compound exponentially
→ After 6 months, it's unrecognizable (in a good way)
```

**The agent becomes a platform for building agents.**

---

## Implementation Priority

**CRITICAL UPDATE based on user feedback:**

> "This is a core fundamental capability that should be implemented before code specialization."

**New priority:** Recursive agent composition is **Milestone 1**, not Milestone 4-5.

### Updated Milestone Order:

```
M0: Prompting + RLM patterns               (Days 1-2)
M1: SharedAgentState + agent_query()       (Days 3-5)  ← RAC FOUNDATION
M2: Quality gates + plans                  (Days 5-8)
M3: Checkpoint/resume                      (Days 8-10)
M4: Agent registry + 7 core specialists    (Days 10-13) ← SPECIALISTS
M5: Agent auto-generation                  (Days 13-16) ← SELF-EXTENSION
...
```

**Why M1 (not M4):** Recursive decomposition is HOW the agent works, not an enhancement. Build the recursion foundation first, everything else uses it.

---

## Summary

**Recursive Agent Composition (RAC) is:**
- RLMs extended to full agents (not just LLM calls)
- Recursive `agent_query()` with specialists
- Shared state across the recursion tree (manifest, plan, knowledge, memory)
- Three-level hierarchy (tools → skills → agents)
- Agent auto-generation from patterns
- Single-threaded + async/await for simplicity
- Exponential capability growth over time

**This is fundamentally different from every other agent architecture.**

**Implementation:** Start at M1 (foundational), extend at M4-M5 (specialists + auto-generation).

---

*RAC: The paradigm for agents that build agents.*
*Foundation: RLMs. Extension: Full agency + shared state + self-improvement.*
*Result: Exponential capability growth on local hardware.*
