# Missing Architectures for State-of-the-Art AI Agents

**Date**: February 6, 2026
**Scope**: Critical architectures needed to make Gaia agents competitive with frontier AI coding/knowledge/computer-use assistants
**Context**: Assessment of gaps beyond the 4 core frameworks (Persistent Memory, Adaptive Prompts, Dynamic Tools, Learning Loops)

---

## Executive Summary

The 4 framework documents (Persistent Memory, Adaptive Prompts, Dynamic Tools, Learning & Adaptation) provide foundational capabilities for self-improving agents. However, **state-of-the-art coding assistants** (Cursor, GitHub Copilot, Devin, Claude Code, etc.) and **advanced agentic systems** (AutoGPT, BabyAGI, MetaGPT) have additional architectural components that are critical for production use.

This document identifies **12 missing architectures**, prioritizes them by impact and urgency, and recommends which to implement now vs. later.

---

## Assessment Framework

Each architecture is evaluated on:

| Dimension | Criteria |
|-----------|----------|
| **Impact** | How much does this improve agent capability? (Critical / High / Medium / Low) |
| **Urgency** | Does the base system need this to function? (Immediate / Phase 1 / Phase 2 / Future) |
| **Complexity** | Engineering effort required (Low: <2 weeks / Medium: 2-4 weeks / High: 4-8 weeks / Very High: 8+ weeks) |
| **Dependencies** | What must exist first? |
| **Code Assistant Relevance** | Essential for coding agents specifically? (Yes / Partial / No) |

---

## Missing Architectures (Prioritized)

### 1. Continuous Execution Engine ★★★★★

**Impact**: CRITICAL
**Urgency**: Immediate (required for your use case)
**Complexity**: Medium (2-3 weeks)
**Dependencies**: None
**Code Assistant**: Yes — essential for "build a complete web app" tasks

**Problem**: Gaia's `process_query()` enforces `max_steps` (default: 20 steps). For complex tasks like "build a full-stack app with auth, database, tests, and deployment", 20 steps is insufficient. The agent stops mid-task.

**What's needed**:

A **Task Completion Engine** that:
- Runs until task verification succeeds (not step limit)
- Quality gates at each phase (syntax check, tests pass, human review)
- Checkpoint/resume for multi-hour tasks
- Token budget tracking but no hard cutoff
- Progress tracking and estimation ("35% complete, 12 files done, 8 to go")

**Architecture components**:

```python
class ContinuousExecutionEngine:
    """Manages unbounded task execution until completion."""

    def execute_until_complete(
        self,
        task_description: str,
        completion_criteria: Dict[str, Any],
        max_iterations: int = 1000,  # Safety limit (not step limit)
        checkpoint_interval: int = 50,
    ) -> ExecutionResult:
        """
        Run agent until task meets completion criteria.

        completion_criteria:
            {
                "all_tests_pass": True,
                "all_files_created": True,
                "human_approval": False,  # Optional gate
                "quality_score": 0.9,
            }
        """
```

**Key features**:
- **Phase-based execution**: Task broken into phases (planning → implementation → testing → refinement). Each phase has its own completion criteria.
- **Iterative refinement**: After each phase, agent self-evaluates. If quality < threshold, re-do phase.
- **Checkpointing**: Every N steps, save full agent state (conversation, memory, file changes) to disk. Resume from checkpoint on crash/restart.
- **Token accountability**: Track cumulative tokens used, but don't stop — log when budget exceeded for cost transparency.
- **Progress visualization**: Real-time status ("Currently: implementing auth system, 3/8 API endpoints done")

**Integration with Gaia**:
- Replaces `max_steps` enforcement with `completion_criteria` evaluation
- Wraps `process_query()` in a loop that continues until criteria met
- Adds `_evaluate_completion()` hook after each iteration

**Priority**: **IMPLEMENT IMMEDIATELY** — This is the foundation for "run until done" behavior.

---

### 2. Architecture Manifest System ★★★★★

**Impact**: CRITICAL
**Urgency**: Immediate (needed for continuous execution)
**Complexity**: High (4-5 weeks)
**Dependencies**: Persistent Memory
**Code Assistant**: Yes — essential for multi-file projects

**Problem**: For tasks like "build a React app with backend API", the agent needs to track:
- What files exist and their purpose
- Dependencies between files (main.py imports db.py)
- Project structure and conventions
- What's been implemented vs. what's planned
- Architecture decisions made during development

Without this, the agent loses coherence after ~30 files. It forgets what it already built, creates duplicate components, violates its own architecture decisions.

**What's needed**:

A **Project Manifest** that acts as the agent's understanding of the project:

```json
{
    "manifest_version": "1.0",
    "project_id": "web-app-abc123",
    "created_at": "2026-02-06T10:00:00Z",
    "updated_at": "2026-02-06T15:30:00Z",
    "task_description": "Build full-stack e-commerce web app with user auth, product catalog, cart, checkout, admin dashboard",

    "architecture": {
        "pattern": "monorepo with separate frontend/backend",
        "frontend": {"framework": "React", "styling": "TailwindCSS"},
        "backend": {"framework": "FastAPI", "database": "PostgreSQL", "orm": "SQLAlchemy"},
        "decisions": [
            {"decision": "Use JWT for auth (not sessions)", "rationale": "Stateless for horizontal scaling", "timestamp": "2026-02-06T10:15:00Z"},
            {"decision": "Stripe for payments (not manual)", "rationale": "PCI compliance", "timestamp": "2026-02-06T11:00:00Z"}
        ]
    },

    "goals": {
        "primary": ["User can create account", "User can browse products", "User can checkout"],
        "completed": ["User can create account", "User can browse products"],
        "in_progress": ["Implement cart system"],
        "pending": ["Checkout flow", "Admin dashboard", "Deployment"]
    },

    "files": [
        {
            "path": "frontend/src/App.tsx",
            "type": "component",
            "purpose": "Main React app component with routing",
            "dependencies": ["./pages/Home", "./pages/Products", "./pages/Cart"],
            "status": "complete",
            "last_modified": "2026-02-06T12:00:00Z",
            "quality_check": {"syntax": "pass", "tests": "pass", "review": "pending"}
        },
        {
            "path": "backend/models/user.py",
            "type": "model",
            "purpose": "SQLAlchemy User model with password hashing",
            "dependencies": ["database.py", "auth_utils.py"],
            "status": "complete",
            "relations": ["has_many orders", "has_one cart"],
            "last_modified": "2026-02-06T10:30:00Z",
            "quality_check": {"syntax": "pass", "tests": "pass"}
        },
        {
            "path": "backend/api/cart.py",
            "type": "api_endpoint",
            "purpose": "Cart CRUD operations",
            "dependencies": ["models/cart.py", "models/product.py"],
            "status": "in_progress",
            "endpoints": ["/cart", "/cart/add", "/cart/remove"],
            "implemented_endpoints": ["/cart"],
            "quality_check": {"syntax": "pass", "tests": "failing"}
        }
    ],

    "dependencies": {
        "frontend": ["react@18", "react-router-dom@6", "tailwindcss@3", "axios@1"],
        "backend": ["fastapi@0.110", "sqlalchemy@2.0", "pydantic@2.0", "stripe@7"],
        "testing": ["pytest@8", "jest@29"],
        "deployment": ["docker", "nginx"]
    },

    "progress": {
        "overall_percent": 45,
        "files_total": 32,
        "files_complete": 18,
        "files_in_progress": 3,
        "files_pending": 11,
        "tests_total": 45,
        "tests_passing": 28,
        "tests_failing": 4,
        "tests_pending": 13
    }
}
```

**Key capabilities**:
- **Automatic manifest updates**: Every file creation/edit triggers manifest update
- **Dependency tracking**: Agent knows "if I change user.py, cart.py and order.py are affected"
- **Architecture consistency**: Before making a decision, agent checks manifest for existing patterns
- **Progress estimation**: "Project is 45% complete. Estimated 8 hours remaining based on current pace."
- **Resume from checkpoint**: If agent restarts mid-task, manifest provides full context

**Integration**:
- `ManifestToolsMixin` with tools: `update_manifest`, `get_project_status`, `check_dependencies`, `verify_architecture_consistency`
- Manifest auto-updated in `_post_process_tool_result()` after file operations
- Manifest injected into system prompt for large projects

**Priority**: **IMPLEMENT IN PHASE 1** (alongside continuous execution)

---

### 3. Verification & Quality Gate System ★★★★★

**Impact**: CRITICAL
**Urgency**: Immediate (required for continuous execution)
**Complexity**: Medium (3-4 weeks)
**Dependencies**: Continuous Execution Engine
**Code Assistant**: Yes

**Problem**: Without automated quality gates, continuous execution produces low-quality output. The agent needs to verify its own work before marking a task complete.

**What's needed**:

```python
class QualityGateSystem:
    """Multi-level verification before task completion."""

    def verify_task_completion(self, task: Task, outputs: List[str]) -> VerificationResult:
        """
        Run all verification gates.

        Gates (configurable per task type):
        - Syntax validation (all code compiles)
        - Type checking (mypy, TypeScript, etc.)
        - Linting (project style rules)
        - Unit tests (all pass)
        - Integration tests (if applicable)
        - Security scan (no vulnerabilities)
        - Performance benchmarks (if applicable)
        - Human review (optional final gate)
        """
```

**Verification levels** (cascading):
1. **L0 — Syntax**: Files parse without errors
2. **L1 — Semantics**: Type checking passes, no undefined references
3. **L2 — Behavior**: Tests pass
4. **L3 — Integration**: Multi-component tests pass
5. **L4 — Quality**: Linting, security, performance benchmarks
6. **L5 — Human**: Optional human approval gate

**Iterative refinement loop**:
```
Agent completes task
    ↓
Run verification gates
    ↓
Gate fails? → Agent receives error → fixes → re-verify
    ↓
All gates pass → Task marked complete
```

**Priority**: **IMPLEMENT IN PHASE 1**

---

### 4. Multi-Agent Orchestration Framework ★★★★☆

**Impact**: HIGH
**Urgency**: Phase 1
**Complexity**: High (5-6 weeks)
**Dependencies**: Continuous Execution, Manifest System
**Code Assistant**: Yes — essential for complex tasks

**Problem**: A single agent can't be expert in everything. Building a full-stack app needs:
- Frontend expert (React/TypeScript)
- Backend expert (Python/FastAPI)
- Database expert (SQL/migrations)
- DevOps expert (Docker/deployment)
- Testing expert (pytest/Jest)

**What's needed**:

A **coordinating agent** that delegates to specialist sub-agents:

```python
class OrchestratorAgent:
    """Coordinates multiple specialist agents for complex tasks."""

    def __init__(self):
        self.specialists = {
            "frontend": FrontendAgent(...),
            "backend": BackendAgent(...),
            "database": DatabaseAgent(...),
            "devops": DevOpsAgent(...),
            "testing": TestingAgent(...),
            "tool_builder": ToolBuilderAgent(...),  # Builds custom tools
        }
        self.shared_manifest = ProjectManifest()  # All agents read/write
        self.coordination_db = CoordinationDB()   # Track agent interactions
```

**Coordination patterns**:
- **Sequential**: Frontend agent finishes → Backend agent starts
- **Parallel**: Frontend + Backend work concurrently on independent features
- **Collaborative**: Backend agent creates API → notifies Frontend agent → Frontend consumes API
- **Review**: Testing agent reviews all code before merge

**Communication**:
- Shared manifest (single source of truth)
- Message bus for agent-to-agent communication
- Conflict resolution (two agents edit same file)

**Priority**: **IMPLEMENT IN PHASE 1-2** (critical for code assistants)

---

### 5. Skill System (Agent-Created Skills) ★★★★☆

**Impact**: HIGH
**Urgency**: Phase 2
**Complexity**: Medium (3-4 weeks)
**Dependencies**: Dynamic Tools, Persistent Memory
**Code Assistant**: Yes

**Problem**: Users invoke skills via `/command` syntax (e.g., `/commit`, `/review-pr`). Today these are static markdown files. Agents should be able to **create new skills dynamically** and store them in vectorized memory.

**What's needed**:

**Skill definition** (extends dynamic tools):
```python
@dataclass
class Skill:
    name: str                    # "optimize-bundle-size"
    description: str             # What it does
    trigger_patterns: List[str]  # ["optimize bundle", "reduce bundle size"]
    prompt_template: str         # LLM instructions for this skill
    tools_required: List[str]    # ["analyze_bundle", "run_webpack"]
    success_criteria: Dict       # How to verify success
    examples: List[str]          # Example usages
    created_at: str              # ISO timestamp
    created_by: str              # "agent" | "user"
    usage_count: int
    success_rate: float
    embedding: np.ndarray        # For semantic search
```

**Skill creation**:
- Agent detects recurring multi-step workflows
- Generates skill definition with prompt template
- Stores in `.gaia/skills/` as `.py` file + metadata
- Embeds trigger patterns in FAISS for retrieval

**Skill invocation**:
- User: "/optimize-bundle-size" (or natural language that matches trigger patterns)
- Agent searches skill database (semantic + exact match)
- Loads skill prompt template into current context
- Executes with skill-specific tools

**Storage**:
- File: `.gaia/skills/optimize_bundle_size.py` (executable code)
- Metadata: `.gaia/skills/registry.json`
- Vector index: FAISS embeddings of trigger patterns + descriptions

**Priority**: **IMPLEMENT IN PHASE 2** (after dynamic tools)

---

### 6. State Machine for Multi-Mode Execution ★★★★☆

**Impact**: HIGH
**Urgency**: Phase 1
**Complexity**: Medium-High (4-5 weeks)
**Dependencies**: Adaptive Prompts
**Code Assistant**: Yes

**Problem**: Complex tasks require switching between different "modes" of operation:
- Planning mode (high-level architecture)
- Implementation mode (writing code)
- Testing mode (running tests, fixing failures)
- Review mode (checking quality)
- Debug mode (investigating failures)

Each mode needs different system prompts, different available tools, different behavior.

**What's needed**:

**State machine architecture**:
```python
@dataclass
class AgentState:
    state_id: str                # "implementation_mode"
    system_prompt: str           # State-specific prompt
    available_tools: List[str]   # Tool subset for this state
    entry_conditions: Dict       # When to enter this state
    exit_conditions: Dict        # When to exit this state
    max_steps_in_state: int      # Limit per state (not global)
    metadata: Dict

class StateMachine:
    def enter_state(self, state_id: str) -> None:
        """Switch to a different execution mode."""

    def execute_in_state(self, user_input: str) -> Any:
        """Run agent loop with state-specific prompt+tools."""

    def exit_state(self, return_value: Any = None) -> None:
        """Return to previous state."""
```

**Example states for code assistant**:
- **Planning State**: Only planning tools (no code execution). Prompt emphasizes architecture.
- **Implementation State**: Code writing tools enabled. Prompt emphasizes best practices.
- **Testing State**: Test execution tools. Prompt emphasizes failure diagnosis.
- **Debug State**: Debugging tools (inspect variables, add logging). Prompt emphasizes hypothesis generation.
- **Review State**: Read-only tools. Prompt emphasizes code quality and security.

**State persistence**:
- States stored in memory database with creation timestamp
- Agent can return to previously used states
- States versioned (can update a state based on learnings)

**Priority**: **IMPLEMENT IN PHASE 1** (needed for multi-phase continuous execution)

---

### 7. Code Intelligence Backend ★★★★☆

**Impact**: HIGH
**Urgency**: Phase 1
**Complexity**: Very High (8-10 weeks)
**Dependencies**: None (can integrate with manifest system)
**Code Assistant**: Yes — essential

**Problem**: LLMs don't have IDE-level code intelligence:
- No "go to definition" (agent can't find where a function is defined across files)
- No "find all references" (agent doesn't know what breaks when it changes an API)
- No type inference (agent can't determine what type a variable has in complex codebases)
- No semantic code search (can't find "functions that handle user authentication")

**What's needed**:

Integration with a **Language Server Protocol (LSP)** backend:

```python
class CodeIntelligenceBackend:
    """LSP-powered code analysis for agents."""

    def __init__(self, project_root: str, language: str = "python"):
        # Start LSP server for the language
        self.lsp_client = LSPClient(language=language, root=project_root)

    def find_definition(self, symbol: str, file: str, line: int) -> Location:
        """Jump to definition of symbol."""

    def find_references(self, symbol: str, file: str, line: int) -> List[Location]:
        """Find all usages of symbol."""

    def get_type_at_position(self, file: str, line: int, col: int) -> str:
        """Infer type at cursor position."""

    def get_completions(self, file: str, line: int, col: int) -> List[str]:
        """Get valid completions at position."""

    def find_implementations(self, interface: str) -> List[Location]:
        """Find all classes implementing interface."""

    def semantic_search(self, query: str) -> List[CodeLocation]:
        """Search codebase semantically (embedding-based)."""
```

**Exposed as agent tools**:
- `find_definition(symbol, context)` → returns file:line
- `find_usages(symbol, context)` → returns all locations
- `semantic_code_search(query)` → returns relevant code snippets

**Integration**:
- LSP server runs alongside agent
- Agent tools call LSP via JSON-RPC
- Results injected into agent context

**Supported languages** (via LSP):
- Python (`pylsp`, `pyright`)
- TypeScript/JavaScript (`typescript-language-server`)
- Go (`gopls`)
- Rust (`rust-analyzer`)
- Java (`jdtls`)

**Priority**: **IMPLEMENT IN PHASE 1-2** (massive improvement for code quality)

---

### 8. Tool-Building Sub-Agent ★★★★☆

**Impact**: HIGH
**Urgency**: Phase 2
**Complexity**: Medium (3-4 weeks)
**Dependencies**: Dynamic Tools, Multi-Agent Orchestration
**Code Assistant**: Yes

**Problem**: The main agent shouldn't build its own tools — that's a meta-cognitive task that distracts from the primary task. Tool building should be delegated to a specialist.

**What's needed**:

A **dedicated ToolBuilderAgent** (subclass of `CodeAgent`):

```python
class ToolBuilderAgent(CodeAgent):
    """Specialist agent that builds tools for other agents."""

    def build_tool_from_spec(
        self,
        name: str,
        description: str,
        example_usage: str,
        required_capabilities: List[str],
    ) -> Tool:
        """
        Build a complete, tested tool from specification.

        Process:
        1. Generate tool code based on spec
        2. Write unit tests
        3. Run tests in sandbox
        4. Debug failures
        5. Add docstrings and type hints
        6. Return validated tool
        """
```

**Interaction pattern**:
```
Main agent: "I keep doing this 5-step analysis pattern repeatedly"
    ↓
Main agent calls ToolBuilderAgent.build_tool_from_spec(...)
    ↓
ToolBuilderAgent generates code, tests, validates
    ↓
Returns validated .py file
    ↓
Main agent installs tool into its registry
```

**Priority**: **IMPLEMENT IN PHASE 2** (after dynamic tools framework)

---

### 9. Context Window Management System ★★★★☆

**Impact**: HIGH
**Urgency**: Phase 1
**Complexity**: Medium (3 weeks)
**Dependencies**: Persistent Memory
**Code Assistant**: Yes

**Problem**: Long conversations or large codebases exceed LLM context windows. The agent needs intelligent context management.

**What's needed**:

```python
class ContextWindowManager:
    """Intelligently manage what fits in LLM context."""

    def __init__(self, max_tokens: int = 200_000):  # Claude Opus 4.6 limit
        self.max_tokens = max_tokens
        self.token_tracker = TokenTracker()

    def prioritize_context(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Given all available context, select what to include.

        Priority order:
        1. Immutable core prompt (always included)
        2. Current task description (always included)
        3. Recent conversation (last N turns, configurable)
        4. Relevant memory (highest confidence knowledge)
        5. Tool results (most recent)
        6. Older conversation (summarized)
        7. Additional context (lowest priority)
        """
```

**Strategies**:
- **Hierarchical summarization**: Older conversation turns summarized by lightweight LLM
- **Relevance filtering**: Only include memory entries with relevance score > threshold
- **Tool result compression**: Large DataFrames → summaries
- **File content sampling**: For large files, include function signatures only
- **Dynamic chunk sizing**: Adjust chunk size based on remaining budget

**Priority**: **IMPLEMENT IN PHASE 1**

---

### 10. Execution Provenance & Debugging ★★★☆☆

**Impact**: MEDIUM-HIGH
**Urgency**: Phase 2
**Complexity**: Medium (3 weeks)
**Dependencies**: Continuous Execution
**Code Assistant**: Yes

**Problem**: When a 100-step task fails at step 87, the agent and user need to understand why. Current Gaia only tracks `error_history` (list of strings).

**What's needed**:

**Execution trace database**:
```sql
CREATE TABLE execution_trace (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT,
    step_number INTEGER,
    timestamp TEXT,
    state TEXT,                -- Planning, Executing, Testing, etc.
    thought TEXT,              -- Agent reasoning
    tool_called TEXT,
    tool_args TEXT,            -- JSON
    tool_result TEXT,          -- JSON
    success BOOLEAN,
    error TEXT,
    token_count INTEGER,
    duration_ms INTEGER,
    files_modified TEXT        -- JSON array
);
```

**Debug capabilities**:
- **Replay**: Re-run execution from step N
- **Inspect**: See full tool args/results at any step
- **Why-provenance**: "Why did you call tool X at step 45?"
- **What-if**: "What would have happened if tool Y returned Z instead?"

**Priority**: **IMPLEMENT IN PHASE 2**

---

### 11. Diff & Merge System for Concurrent Edits ★★★☆☆

**Impact**: MEDIUM-HIGH
**Urgency**: Phase 2
**Complexity**: High (5-6 weeks)
**Dependencies**: Multi-Agent Orchestration
**Code Assistant**: Yes

**Problem**: When multiple specialist agents work on the same codebase concurrently, they can create conflicting edits.

**What's needed**:

- **3-way merge** for concurrent file edits
- **Conflict detection** before writing
- **Atomic file operations** (optimistic locking)
- **Change review system** (one agent's edits reviewed by another)

**Priority**: **IMPLEMENT IN PHASE 2** (only needed if multi-agent parallelism is used)

---

### 12. Retrieval-Augmented Code Generation ★★★☆☆

**Impact**: MEDIUM-HIGH
**Urgency**: Phase 2
**Complexity**: Medium (3-4 weeks)
**Dependencies**: Code Intelligence, Persistent Memory
**Code Assistant**: Yes

**Problem**: When generating code, the agent should reference existing codebase patterns, not generate from scratch.

**What's needed**:

- **Codebase embedding index**: All functions/classes embedded for semantic search
- **Pattern library**: Common code patterns extracted and indexed
- **Example retrieval**: "Find examples of database transaction handling in this codebase"

**Priority**: **IMPLEMENT IN PHASE 2**

---

### 13. Streaming & Real-Time Feedback ★★★☆☆

**Impact**: MEDIUM
**Urgency**: Phase 2
**Complexity**: Low (1-2 weeks)
**Dependencies**: None
**Code Assistant**: Partial — improves UX

**Problem**: Users wait for agent to complete entire response. For long-running tasks, they want real-time progress.

**What's needed**:

Gaia already supports `streaming=True`. Extend with:
- **Thought streaming**: Show agent reasoning in real-time
- **Progress streaming**: "Currently implementing user authentication..." (live updates)
- **Incremental results**: Show partial outputs as they're generated

**Priority**: **IMPLEMENT IN PHASE 2** (UX improvement, not critical)

---

### 14. Cost & Resource Management ★★☆☆☆

**Impact**: MEDIUM
**Urgency**: Phase 2
**Complexity**: Low (1 week)
**Dependencies**: Continuous Execution
**Code Assistant**: No

**Problem**: Continuous execution without token limits can rack up costs. Enterprises need budgets and controls.

**What's needed**:

```python
class ResourceManager:
    """Track and limit resource usage."""

    def __init__(self, daily_token_budget: int = 10_000_000):
        self.budget = daily_token_budget
        self.usage = self._load_usage_from_db()

    def check_budget(self, estimated_tokens: int) -> bool:
        """Return False if operation would exceed budget."""

    def track_usage(self, tokens: int, cost_usd: float) -> None:
        """Record usage in database."""
```

**Features**:
- Per-agent budgets
- Per-user budgets (for multi-tenant)
- Cost alerts ("50% of daily budget used")
- Optimization recommendations ("switch to Haiku for simple queries")

**Priority**: **IMPLEMENT IN PHASE 2-3** (not critical for single-user agents)

---

### 15. Security Sandbox for Tool Execution ★★★☆☆

**Impact**: MEDIUM-HIGH
**Urgency**: Phase 1
**Complexity**: Medium-High (4-5 weeks)
**Dependencies**: Dynamic Tools
**Code Assistant**: Yes (if executing user code)

**Problem**: Dynamic tools execute arbitrary Python code. Even with validation, there are risks.

**What's needed**:

- **Containerized execution**: Docker/Podman sandbox for tool execution
- **Resource limits**: CPU, memory, disk, network quotas
- **Filesystem isolation**: Restricted to specific directories
- **Network isolation**: No internet access (or allowlist)
- **Timeout enforcement**: Hard kill after timeout

**Implementation**:
```python
class SandboxedExecutor:
    """Execute code in isolated container."""

    def execute_tool_in_sandbox(
        self,
        tool_code: str,
        tool_args: Dict,
        timeout_sec: int = 60,
        memory_limit_mb: int = 512,
    ) -> ToolResult:
        """Run tool in Docker container with resource limits."""
```

**Priority**: **IMPLEMENT IN PHASE 1** (critical if dynamic tools are used)

---

### 16. Codebase Indexing & Understanding ★★★☆☆

**Impact**: MEDIUM-HIGH
**Urgency**: Phase 2
**Complexity**: High (5-6 weeks)
**Dependencies**: Code Intelligence, Persistent Memory
**Code Assistant**: Yes

**Problem**: When agent enters a large codebase (100K+ LOC), it needs to build understanding before making changes.

**What's needed**:

- **Automatic codebase analysis**: Run on project open, extract architecture
- **Dependency graph**: Visualize imports, function calls, data flow
- **Entry point detection**: Identify main(), CLI commands, API routes
- **Module clustering**: Group related files by semantic similarity
- **Change impact analysis**: "If I modify function X, what breaks?"

**Priority**: **IMPLEMENT IN PHASE 2** (significant for large projects)

---

### 17. Test Generation & Validation ★★★☆☆

**Impact**: MEDIUM-HIGH
**Urgency**: Phase 1
**Complexity**: Medium-High (4-5 weeks)
**Dependencies**: Code Intelligence, Quality Gates
**Code Assistant**: Yes

**Problem**: Code without tests is not production-ready. The agent should automatically generate tests for every function it writes.

**What's needed**:

```python
class TestGenerator:
    """Generate comprehensive tests for code."""

    def generate_tests(self, function_code: str, context: str) -> List[Test]:
        """
        Generate unit tests covering:
        - Happy path
        - Edge cases (empty input, None, large values)
        - Error cases (invalid input)
        - Integration cases (if applicable)
        """
```

**Test strategies**:
- **Property-based testing**: Generate tests from function signature
- **Mutation testing**: Verify tests catch bugs
- **Coverage tracking**: Ensure > 80% line coverage

**Priority**: **IMPLEMENT IN PHASE 1-2** (critical for quality)

---

### 18. Human-in-the-Loop Review System ★★☆☆☆

**Impact**: MEDIUM
**Urgency**: Phase 2
**Complexity**: Medium (2-3 weeks)
**Dependencies**: Quality Gates
**Code Assistant**: Yes

**Problem**: Some decisions need human approval (architecture choices, security-sensitive code, destructive operations).

**What's needed**:

- **Review queue**: Agent pauses and requests human review
- **Review interface**: Show proposed changes with context
- **Approval tracking**: Store approved/rejected decisions in memory
- **Learn from reviews**: If human consistently rejects pattern X, agent learns to avoid it

**Priority**: **IMPLEMENT IN PHASE 2**

---

### 19. Incremental Learning from Execution ★★☆☆☆

**Impact**: MEDIUM
**Urgency**: Phase 3
**Complexity**: Medium-High (4-5 weeks)
**Dependencies**: Learning Loop, Execution Provenance
**Code Assistant**: Partial

**Problem**: Learning currently happens from explicit feedback. The agent should also learn from its own execution patterns.

**What's needed**:

- **Success pattern extraction**: "When I do X then Y, success rate is 95%"
- **Failure pattern extraction**: "Tool Z fails 80% of the time when condition C is true"
- **Efficiency analysis**: "Using tool A is 3x faster than tool B for task T"
- **Automated knowledge updates**: No user feedback required

**Priority**: **IMPLEMENT IN PHASE 3** (enhancement, not critical)

---

### 20. Semantic Code Search ★★☆☆☆

**Impact**: MEDIUM
**Urgency**: Phase 2
**Complexity**: Medium (3 weeks)
**Dependencies**: Code Intelligence
**Code Assistant**: Yes

**Problem**: Agent needs to find code by intent, not just by name. "Find all functions that validate email addresses" should work even if they're named `check_email_format`, `validate_user_email`, `email_regex_match`, etc.

**What's needed**:

- Embed all functions/classes using code-specific model (CodeBERT, StarCoder)
- Index in FAISS
- Semantic search tool

**Priority**: **IMPLEMENT IN PHASE 2**

---

### 21. Automated Documentation Generation ★★☆☆☆

**Impact**: MEDIUM
**Urgency**: Phase 3
**Complexity**: Low-Medium (2 weeks)
**Dependencies**: Code Intelligence
**Code Assistant**: Partial

**Problem**: Agent generates code but not documentation.

**What's needed**:

- Auto-generate docstrings for all functions
- Auto-generate README for projects
- Auto-generate API docs for web services
- Update docs when code changes

**Priority**: **IMPLEMENT IN PHASE 3** (nice-to-have)

---

### 22. Version Control Integration ★★☆☆☆

**Impact**: MEDIUM
**Urgency**: Phase 2
**Complexity**: Low (1-2 weeks)
**Dependencies**: Manifest System
**Code Assistant**: Yes

**Problem**: Agent makes file changes but doesn't commit them properly.

**What's needed**:

- **Auto-commit**: Commit after each logical unit of work with descriptive message
- **Branch management**: Create feature branches for each task
- **PR creation**: Auto-generate PR description from task + changes
- **Conflict resolution**: Handle merge conflicts

**Gaia already supports** git via bash tools. Just needs structured workflow.

**Priority**: **IMPLEMENT IN PHASE 2** (important for team environments)

---

## Summary Table

| # | Architecture | Impact | Urgency | Complexity | Code Assistant | Implement When |
|---|--------------|--------|---------|------------|---------------|----------------|
| 1 | **Continuous Execution Engine** | CRITICAL | Immediate | Medium | Yes | **Phase 0 (Now)** |
| 2 | **Architecture Manifest System** | CRITICAL | Immediate | High | Yes | **Phase 1** |
| 3 | **Verification & Quality Gates** | CRITICAL | Immediate | Medium | Yes | **Phase 1** |
| 4 | **Multi-Agent Orchestration** | HIGH | Phase 1 | High | Yes | **Phase 1-2** |
| 5 | **Skill System** | HIGH | Phase 2 | Medium | Yes | **Phase 2** |
| 6 | **State Machine (Multi-Mode)** | HIGH | Phase 1 | Med-High | Yes | **Phase 1** |
| 7 | **Code Intelligence (LSP)** | HIGH | Phase 1 | Very High | Yes | **Phase 1-2** |
| 8 | **Tool-Building Sub-Agent** | HIGH | Phase 2 | Medium | Yes | **Phase 2** |
| 9 | **Context Window Management** | HIGH | Phase 1 | Medium | Yes | **Phase 1** |
| 10 | **Execution Provenance** | MED-HIGH | Phase 2 | Medium | Yes | **Phase 2** |
| 11 | **Diff & Merge (Concurrent Edits)** | MED-HIGH | Phase 2 | High | Yes | **Phase 2** |
| 12 | **Retrieval-Augmented Codegen** | MED-HIGH | Phase 2 | Medium | Yes | **Phase 2** |
| 13 | **Streaming & Real-Time** | MEDIUM | Phase 2 | Low | Partial | **Phase 2** |
| 14 | **Cost & Resource Management** | MEDIUM | Phase 2 | Low | No | **Phase 2-3** |
| 15 | **Security Sandbox** | MED-HIGH | Phase 1 | Med-High | Yes | **Phase 1** |
| 16 | **Codebase Indexing** | MED-HIGH | Phase 2 | High | Yes | **Phase 2** |
| 17 | **Test Generation** | MED-HIGH | Phase 1-2 | Med-High | Yes | **Phase 1-2** |
| 18 | **Human-in-the-Loop** | MEDIUM | Phase 2 | Medium | Yes | **Phase 2** |
| 19 | **Incremental Learning** | MEDIUM | Phase 3 | Med-High | Partial | **Phase 3** |
| 20 | **Semantic Code Search** | MEDIUM | Phase 2 | Medium | Yes | **Phase 2** |
| 21 | **Auto Documentation** | MEDIUM | Phase 3 | Low-Med | Partial | **Phase 3** |
| 22 | **Version Control Integration** | MEDIUM | Phase 2 | Low | Yes | **Phase 2** |

---

## Implementation Strategy

### Phase 0: Foundation (Week 1-2) — IMMEDIATE

Must implement before anything else:

1. **Continuous Execution Engine** — Enables unbounded task execution
2. **Quality Gates (basic)** — Syntax + test validation

**Rationale**: Without these, the agent can't "run until complete" as required.

---

### Phase 1: Core Systems (Week 3-8)

Build the essential architectures:

3. **Architecture Manifest** — Track multi-file projects
4. **State Machine** — Multi-mode execution
5. **Context Window Management** — Handle large projects
6. **Security Sandbox** — Safe dynamic tool execution
7. **Code Intelligence (LSP integration)** — IDE-level code understanding
8. **Test Generation (basic)** — Auto-test generation

**Rationale**: These are table-stakes for a production code assistant.

---

### Phase 2: Advanced Capabilities (Week 9-16)

Extend with specialist systems:

9. **Multi-Agent Orchestration** — Specialist sub-agents
10. **Skill System** — Agent-created skills
11. **Tool-Building Sub-Agent** — Automated tool creation
12. **Execution Provenance** — Debugging and replay
13. **Retrieval-Augmented Codegen** — Codebase-aware generation
14. **Semantic Code Search** — Intent-based code finding
15. **Human-in-the-Loop** — Review system
16. **Codebase Indexing** — Automatic architecture extraction
17. **Version Control Workflows** — Git automation
18. **Streaming** — Real-time feedback

**Rationale**: These significantly improve quality and UX.

---

### Phase 3: Polish (Week 17+)

19. **Incremental Learning** — Learn from execution without feedback
20. **Cost Management** — Enterprise budgeting
21. **Auto Documentation** — Generate docs
22. **Diff & Merge** — Concurrent edit resolution

**Rationale**: Nice-to-have enhancements.

---

## Critical Path for Code Assistants

For a **state-of-the-art coding agent**, implement in this order:

**Must-have (before agent is useful)**:
1. Continuous Execution Engine
2. Quality Gates
3. Architecture Manifest
4. State Machine
5. Test Generation

**Should-have (before production)**:
6. Code Intelligence (LSP)
7. Context Window Management
8. Security Sandbox
9. Multi-Agent Orchestration

**Nice-to-have (iterative improvements)**:
- Everything else in Phase 2-3

---

## Integration with 4 Core Frameworks

The 4 framework docs (Persistent Memory, Adaptive Prompts, Dynamic Tools, Learning) are **horizontal capabilities** that apply to all agents. The architectures in this document are:

- **Some are vertical (code assistant specific)**: Code Intelligence, LSP, Test Generation, Codebase Indexing
- **Some are horizontal (all agents)**: Continuous Execution, Quality Gates, State Machine, Multi-Agent, Context Window Management

**Recommended approach**:

1. **Implement the 4 core frameworks first** (Memory, Prompts, Tools, Learning) — these are the foundation
2. **Add continuous execution + quality gates** (this assessment, #1-3) — enables "run until done"
3. **Add code-specific systems** (this assessment, #6, 7, 17) — makes it a great code assistant
4. **Iteratively add the rest** based on user feedback and needs

---

## Open Questions

1. **Do all 22 architectures need to be in Gaia SDK core, or can some be agent-specific extensions?**
   - Recommendation: #1, 3, 6, 9, 15 belong in Gaia core. The rest are agent-level or vertical.

2. **Should the Architecture Manifest be framework-agnostic (works for React, Python, Go) or specialized per language?**
   - Recommendation: Start framework-agnostic (generic file tracking), add language-specific extensions later.

3. **Does continuous execution need infinite budget, or should there be a safety cutoff (e.g., 10M tokens)?**
   - Recommendation: Configurable safety limit with human override. Default: 10M tokens, then pause for approval.

4. **Which architectures have the highest ROI for InferenceMAX/Jarvis specifically?**
   - Continuous Execution (run nightly analysis until all regressions diagnosed)
   - Quality Gates (verify all recommendations are data-backed)
   - Architecture Manifest (track benchmark configurations across runs)
   - Execution Provenance (debug why a recommendation was wrong)

---

*Assessment of missing architectures for state-of-the-art Gaia-based AI agents.*
*Use this to prioritize development beyond the 4 core frameworks.*
