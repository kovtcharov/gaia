# GAIA V2: Composable Feature Branch Plan

**Date**: February 10, 2026
**Version**: 1.0
**Scope**: Sequenced feature branches with composable architecture for GAIA V2
**Estimation Model**: Autonomous coding agent (Claude Code + Opus 4.6)
**Key Capabilities**: Coding Agent + Computer Use Agent

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Always-On Components](#always-on-components)
3. [Two Capability Tracks](#two-capability-tracks)
4. [Estimation Basis](#estimation-basis)
5. [Shared Foundation (Day 1)](#shared-foundation-day-1)
6. [Track A: Coding Agent (Days 2-7)](#track-a-coding-agent-days-2-7)
7. [Track B: Computer Use Agent (Days 2-7)](#track-b-computer-use-agent-days-2-7)
8. [Convergence (Days 7-9)](#convergence-days-7-9)
9. [Release Milestones](#release-milestones)
10. [LLM Tier Compatibility Matrix](#llm-tier-compatibility-matrix)
11. [Composability Matrix by Milestone](#composability-matrix-by-milestone)
12. [Computer Use Sub-Branch Breakdown](#computer-use-sub-branch-breakdown)
13. [Coding Agent Sub-Branch Breakdown](#coding-agent-sub-branch-breakdown)
14. [Dependency Graph](#dependency-graph)
15. [Composability Profiles](#composability-profiles)
16. [Key Architectural Rules](#key-architectural-rules)
17. [Summary](#summary)

---

## Design Philosophy

### The Feature Registry

Every V2 feature follows the **Mixin + FeatureFlag** pattern. The framework is composable such that simple agents with smaller LLMs run lean, while complex agents with larger LLMs can enable advanced capabilities.

```
+--------------------------------------------------------------+
|                    ALWAYS-ON (CORE)                           |
|  Agent base class, @tool decorator, LLM factory, CLI,        |
|  logging, config system                                      |
+--------------------------------------------------------------+
|                ALWAYS-ON IF INSTALLED                         |
|  Error Recovery (RetryEngine, CircuitBreaker)                |
|  Security Foundations (input sanitization, path validation)   |
+--------------------------------------------------------------+
|              OPT-IN VIA CONFIG (default: OFF)                |
|  Everything else -- each feature is a Mixin + feature flag   |
+--------------------------------------------------------------+
```

### Mixin + FeatureFlag Pattern

```python
class MyAgent(Agent, MemoryMixin, ObservabilityMixin):  # opt-in composition
    def __init__(self, config):
        super().__init__(**config)
        # Each mixin checks its own feature flag
        # MemoryMixin: config.enable_memory (default: False)
        # ObservabilityMixin: config.enable_observability (default: False)
```

### Existing Codebase Alignment

The current GAIA codebase already demonstrates this composability pattern:

```python
# ChatAgent uses 5 mixins
class ChatAgent(Agent, RAGToolsMixin, FileToolsMixin, ShellToolsMixin, FileSearchToolsMixin):

# CodeAgent uses 13+ mixins
class CodeAgent(
    ApiAgent, Agent, CodeToolsMixin, ValidationAndParsingMixin,
    FileIOToolsMixin, CodeFormattingMixin, ProjectManagementMixin,
    TestingMixin, ErrorFixingMixin, TypeScriptToolsMixin,
    WebToolsMixin, PrismaToolsMixin, CLIToolsMixin,
    ExternalToolsMixin, ValidationToolsMixin,
):
```

V2 features extend this established pattern. No breaking changes to the base `Agent` class.

---

## Always-On Components

These are structural changes that improve the framework without adding complexity or LLM cost. They are **non-optional**:

| Component | Reason | Impact |
|-----------|--------|--------|
| **Cross-Cutting Config System** | Unified config is needed for feature flags themselves | Zero overhead |
| **Error Recovery (basic)** | RetryEngine for LLM calls already needed -- every agent benefits | Minimal overhead |
| **Input Sanitization** | Security baseline -- path traversal, injection prevention | Zero LLM cost |
| **Structured Logging** | Foundation for all debugging -- adds structured fields to existing logger | Zero overhead |
| **Feature Registry** | The system that manages enable/disable of all other features | Zero overhead |

**Everything else is disabled by default.**

---

## Two Capability Tracks

Instead of a single linear sequence, run **two parallel tracks** that converge:

```
TRACK A: CODING AGENT                    TRACK B: COMPUTER USE AGENT
(Make agents that write code better)      (Make agents that control computers)

Day 1: ---------- Shared Foundation ------------- Day 1:
                  (Config + Feature Flags)
                         |
              +----------+----------+
              v                     v
Day 1-2: Error Recovery      Security Foundations     :Day 1-2
Day 2:   Memory Tier 1-2     Observability            :Day 2
Day 2-3: Task Queue          -- (merge point) --      :Day 2-3
Day 2-3: Continuous Exec
Day 3-4: State Machine       Computer Use (vision)    :Day 3-4
Day 4:   Manifest            Computer Use (input)     :Day 4
Day 4-5: Memory Tier 3-4     Computer Use (browser)   :Day 4-5
Day 5:   Adaptive Prompts    Computer Use (safety)    :Day 5
Day 5-6: Learning Loop       Computer Use (desktop)   :Day 5-6
Day 6-7: Dynamic Tools       Email Integration        :Day 6-7
                              Workflow Orchestration   :Day 7
              +----------+----------+
                         v
Day 7-8: ---- Convergence ----------------------- Day 7-8:
              Multi-Agent Orchestration
              Advanced RAG + Multi-Doc
              TUI (full)

Day 8-9: ---- Polish + Integration --------------- Day 8-9:
              Cross-track integration tests
              Composability profiles
              Release packaging
```

**Total: 9 working days**

---

## Estimation Basis

### Autonomous Agent vs Human + AI Assist

| Factor | Human + AI Assist | Fully Autonomous Agent |
|--------|-------------------|----------------------|
| Working hours/day | 8h | 24h |
| Context switching | Frequent | Zero |
| Boilerplate generation | Fast | Instant |
| Test writing | Manual review | Auto-generated + auto-run |
| Parallelism | 1-2 streams | 4-6 concurrent branches |
| Code review | Per-PR | Batch review checkpoints |
| **Effective multiplier** | **1x** | **6-10x** |

**Rule of thumb:** 1 human-week = 1 autonomous-agent-day

---

## Shared Foundation (Day 1)

### Branch 1: `feature/v2-foundation-config` (4 hours)

**Priority:** P0 -- Blocks everything else

**What it builds:**
- `src/gaia/config/` -- Unified configuration system (YAML + env vars + overlays)
- `src/gaia/config/feature_flags.py` -- `FeatureRegistry` that all V2 features register with
- `src/gaia/config/v2_config.py` -- `V2Config` dataclass with all feature flags (all defaulting to `False`)

**Key class:**

```python
@dataclass
class V2Config:
    # Foundation (always on, no flag needed)
    # -- config system itself, structured logging

    # Opt-in features (all default False)
    enable_memory: bool = False
    enable_observability: bool = False
    enable_task_queue: bool = False
    enable_continuous_execution: bool = False
    enable_state_machine: bool = False
    enable_manifest: bool = False
    enable_learning: bool = False
    enable_dynamic_tools: bool = False
    enable_voice: bool = False
    enable_computer_use: bool = False
    enable_email: bool = False
    enable_workflow: bool = False
    enable_multi_doc: bool = False
    enable_advanced_rag: bool = False
    enable_multi_agent: bool = False
    enable_tui: bool = False

    # LLM complexity tier (determines which features are safe)
    llm_tier: str = "basic"  # "basic", "standard", "advanced"
```

- **Always on:** Yes -- this is the scaffolding
- **Breaking changes:** None -- purely additive to existing Agent class
- **Validation criteria:** Existing agents continue working with zero changes

---

### Branch 2: `feature/v2-error-recovery` (6 hours)

**Priority:** P0 -- Blocks all external integrations

**What it builds:**
- `src/gaia/error_recovery/` -- RetryEngine, CircuitBreaker, FallbackChain
- `src/gaia/error_recovery/retry.py` -- Exponential backoff, jitter, configurable strategies
- `src/gaia/error_recovery/circuit_breaker.py` -- Prevents cascade failures
- `src/gaia/error_recovery/mixin.py` -- `ErrorRecoveryMixin`

- **Always on:** **Basic retry for LLM calls -- YES.** The `RetryEngine` wraps the existing LLM client factory. CircuitBreaker and FallbackChain are opt-in via `enable_error_recovery_advanced=True`.
- **Why basic is always-on:** Every agent already retries LLM calls ad-hoc. This standardizes it. Zero additional LLM cost.
- **Validation criteria:**
  - LLM call with intentional failure retries 3x with backoff
  - CircuitBreaker opens after 5 failures, half-opens after 30s
  - Existing agents work unchanged

---

### Branch 3: `feature/v2-security-foundations` (6 hours, parallel with Branch 2)

**Priority:** P0 -- Blocks email, computer use, workflows

**What it builds:**
- `src/gaia/security/` -- SecretVault, PermissionEnforcer, AuditLogger
- `src/gaia/security/vault.py` -- AES-256 encrypted credential storage
- `src/gaia/security/permissions.py` -- Permission model with 20+ granular permissions
- `src/gaia/security/audit.py` -- Tamper-proof audit logging
- `src/gaia/security/sanitize.py` -- Input sanitization utilities

- **Always on:** **Input sanitization only** (`sanitize.py`) -- path traversal checks, basic injection prevention. Everything else opt-in via `enable_security_vault=True`.
- **Why sanitization is always-on:** Prevents OWASP Top 10 issues without any user-visible behavior change.
- **Validation criteria:**
  - Path traversal blocked (`../../etc/passwd`)
  - SecretVault encrypts/decrypts credentials
  - PermissionEnforcer blocks unauthorized tool calls
  - Existing agents unaffected

**Day 1 deliverable:** Foundation merged to `main`, all feature flags in place.

---

## Track A: Coding Agent (Days 2-7)

### Branch 4 (Track A): `feature/v2-memory-tier1-2` (8 hours)

**Priority:** P0 -- Foundation for learning, manifest, adaptive prompts
**Depends on:** Branch 1 (config)

**What it builds:**
- `src/gaia/memory/` -- EpisodicMemory (FAISS + JSON), session summarization
- `src/gaia/memory/episodic.py` -- Session-level memory with vector search
- `src/gaia/memory/working.py` -- Enhanced working memory (context window management)
- `src/gaia/memory/mixin.py` -- `MemoryMixin` with `recall` tool
- `src/gaia/memory/tools.py` -- `MemoryToolsMixin`

- **Always on:** NO -- opt-in via `enable_memory=True`
- **Default off because:** Requires FAISS dependency, adds storage overhead, adds LLM calls for summarization. A simple `gaia llm "hello"` shouldn't need memory.
- **Validation criteria:**
  - 10 sessions saved and searchable via `recall` tool
  - Session summaries generated automatically at session end
  - Memory disabled by default -- zero overhead when off
  - Graceful degradation if FAISS not installed

---

### Branch 5 (Track A): `feature/v2-task-queue` (6 hours)

**Priority:** P0
**Depends on:** Branch 1 (config), Branch 2 (error recovery)

**What it builds:**
- `src/gaia/tasks/` -- Task dataclass, TaskManager, TaskQueue
- `src/gaia/tasks/models.py` -- Task dataclass + SQLite schema
- `src/gaia/tasks/manager.py` -- Create/start/pause/resume tasks
- `src/gaia/tasks/queue.py` -- Priority scheduling, dependency resolution
- `src/gaia/tasks/mixin.py` -- `TaskToolsMixin`

- **Always on:** NO -- opt-in via `enable_task_queue=True`
- **Default off because:** Chat-only agents don't need task management. Adds database overhead.
- **Validation criteria:**
  - Create 3 tasks with dependencies, verify execution order
  - Pause and resume a task mid-execution
  - Task state persists across restarts (SQLite)
  - Existing `process_query()` unchanged

---

### Branch 6 (Track A): `feature/v2-continuous-execution` (6 hours, parallel with Branch 5)

**Priority:** P0
**Depends on:** Branch 1 (config), Branch 2 (error recovery)

**What it builds:**
- `src/gaia/execution/` -- ContinuousExecutionEngine, QualityGates, Checkpoints
- `src/gaia/execution/continuous.py` -- Replace `max_steps` with completion criteria
- `src/gaia/execution/quality_gates.py` -- Syntax check, test runner, lint
- `src/gaia/execution/checkpoints.py` -- Save/restore execution state
- `src/gaia/execution/mixin.py` -- `ContinuousExecutionMixin`

- **Always on:** NO -- opt-in via `enable_continuous_execution=True`
- **Default off because:** Existing `max_steps` behavior is simpler and safer for small LLMs. Continuous execution needs larger, more capable models.
- **LLM tier requirement:** `standard` or `advanced` (not `basic`)
- **Validation criteria:**
  - Agent runs 100+ steps until task verified complete
  - Checkpoint created every 10 steps, restorable on crash
  - Quality gates block completion if tests fail
  - `max_steps` still works when continuous execution is disabled

---

### Branch 7 (Track A): `feature/v2-state-machine` (8 hours)

**Priority:** P1
**Depends on:** Branch 4 (memory for state persistence)

**What it builds:**
- `src/gaia/state_machine/` -- AgentState, StateMachine, predefined states
- `src/gaia/state_machine/models.py` -- State dataclass + transitions
- `src/gaia/state_machine/machine.py` -- State machine with enter/execute/return
- `src/gaia/state_machine/states.py` -- 6 predefined states (Requirements, Planning, Implementation, Testing, Debug, Review)
- `src/gaia/state_machine/requirements.py` -- Requirements Gathering auto-skip logic
- `src/gaia/state_machine/mixin.py` -- `StateMachineMixin`

- **Always on:** NO -- opt-in via `enable_state_machine=True`
- **Default off because:** Adds prompt overhead (state context injection), requires capable LLM to follow state transitions.
- **LLM tier requirement:** `standard` or `advanced`
- **Validation criteria:**
  - Agent transitions: Requirements -> Planning -> Implementation -> Testing
  - Requirements Gathering asks 3-7 questions for complex tasks
  - Simple tasks auto-skip to Planning
  - State persists in memory/SQLite

---

### Branch 8 (Track A): `feature/v2-manifest` (6 hours)

**Priority:** P1
**Depends on:** Branch 4 (memory), Branch 5 (tasks)

**What it builds:**
- `src/gaia/manifest/` -- ProjectManifest, ManifestTracker, DependencyGraph
- `src/gaia/manifest/models.py` -- File tracking, dependency edges
- `src/gaia/manifest/tracker.py` -- Auto-update on file operations
- `src/gaia/manifest/graph.py` -- Import analysis, dependency visualization
- `src/gaia/manifest/mixin.py` -- `ManifestToolsMixin`

- **Always on:** NO -- opt-in via `enable_manifest=True`
- **Default off because:** Only useful for code agents working on multi-file projects. Chat agents don't need file tracking.
- **LLM tier requirement:** `standard` or `advanced`
- **Validation criteria:**
  - 30-file project fully tracked
  - Dependency graph detects circular imports
  - File changes auto-update manifest

---

### Branch 9 (Track A): `feature/v2-memory-tier3-4` (6 hours, parallel with Branch 8)

**Priority:** P1
**Depends on:** Branch 4 (memory tier 1-2)

**What it builds:**
- `src/gaia/memory/semantic.py` -- SQLite knowledge base with confidence scores
- `src/gaia/memory/universal.py` -- UniversalKnowledgeDB storing all interactions
- `src/gaia/memory/consolidation.py` -- Consolidation pipeline (episodic -> semantic)

- **Always on:** NO -- opt-in via `enable_memory=True` AND `enable_semantic_memory=True`
- **Default off because:** Requires LLM calls for consolidation. More storage. Only valuable for long-lived agents.
- **Validation criteria:**
  - 20 sessions consolidated into semantic patterns
  - Confidence scores decay over time
  - Knowledge conflicts detected and flagged

---

### Branch 10 (Track A): `feature/v2-adaptive-prompts` (4 hours)

**Priority:** P1
**Depends on:** Branch 4 (memory), Branch 7 (state machine)

**What it builds:**
- `src/gaia/prompts/` -- PromptBuilder, ContextWindowManager, state-specific prompts
- `src/gaia/prompts/adaptive.py` -- Dynamic prompt construction based on state + memory
- `src/gaia/prompts/context_manager.py` -- Smart context window allocation
- `src/gaia/prompts/mixin.py` -- `AdaptivePromptsMixin`

- **Always on:** NO -- opt-in via `enable_adaptive_prompts=True`
- **Default off because:** Adds prompt overhead, requires memory + state machine to be useful.
- **Validation criteria:**
  - Prompt includes relevant learnings from memory
  - Different states produce different system prompts
  - Context window managed within token limits

---

### Branch 11 (Track A): `feature/v2-learning-loop` (8 hours)

**Priority:** P2
**Depends on:** Branch 9 (semantic memory), Branch 8 (manifest)

**What it builds:**
- `src/gaia/learning/` -- FeedbackCollector, OutcomeTracker, PatternExtractor
- `src/gaia/learning/feedback.py` -- Explicit + implicit + automated feedback
- `src/gaia/learning/outcomes.py` -- Track recommendation outcomes
- `src/gaia/learning/patterns.py` -- Extract success/failure patterns
- `src/gaia/learning/mixin.py` -- `LearningLoopMixin`

- **Always on:** NO -- opt-in via `enable_learning=True`
- **Default off because:** Requires semantic memory, adds LLM calls for evaluation/consolidation. Advanced feature.
- **LLM tier requirement:** `advanced`
- **Validation criteria:**
  - Agent learns from correction ("use pathlib not subprocess")
  - Pattern extracted after 3+ occurrences
  - Confidence scores adjust based on outcomes
  - Learning disabled -> zero overhead

---

### Branch 12 (Track A): `feature/v2-dynamic-tools` (8 hours)

**Priority:** P2
**Depends on:** Branch 11 (learning loop for quality tracking)

**What it builds:**
- `src/gaia/tools/dynamic/` -- ToolBuilderAgent, DynamicToolManager
- `src/gaia/skills/` -- SkillStore, vector search for skills
- `src/gaia/tools/dynamic/builder.py` -- LLM generates new tools from patterns
- `src/gaia/tools/dynamic/security.py` -- AST-based validation of generated code
- `src/gaia/skills/store.py` -- Persistent skill storage with FAISS

- **Always on:** NO -- opt-in via `enable_dynamic_tools=True`
- **Default off because:** Generated code execution is a security surface. Requires capable LLM. Advanced use case.
- **LLM tier requirement:** `advanced`
- **Validation criteria:**
  - Agent creates a tool after detecting repeated pattern
  - Generated tool passes AST security validation
  - Skill stored and retrievable via vector search
  - Security: no `eval()`, no `exec()`, no file system access without permission

**Track A total: ~66 hours -> ~3 days at 24h/day**

---

## Track B: Computer Use Agent (Days 2-7)

### Branch 13 (Track B): `feature/v2-observability` (6 hours)

**Priority:** P0 -- Needed for debugging all subsequent features
**Depends on:** Branch 1 (config)

**What it builds:**
- `src/gaia/observability/` -- TracerProvider, MeterProvider, CostTracker
- `src/gaia/observability/tracing.py` -- OpenTelemetry-compatible distributed tracing
- `src/gaia/observability/metrics.py` -- Prometheus-compatible metrics (token counts, latency)
- `src/gaia/observability/cost.py` -- Per-task cost tracking
- `src/gaia/observability/mixin.py` -- `ObservabilityMixin`

- **Always on:** NO -- completely opt-in via `enable_observability=True`
- **Default off because:** Adds overhead (tracing context propagation), requires understanding of observability concepts. Small LLM users won't need this.
- **Validation criteria:**
  - `enable_observability=True` produces trace spans for each tool call
  - Metrics include: tokens_used, latency_ms, tool_call_count
  - Cost tracker reports total cost per session
  - `enable_observability=False` -- zero overhead

---

### Branch 14 (Track B): `feature/v2-computer-use-vision` (10 hours)

**Priority:** P1
**Depends on:** Branch 3 (security), Branch 13 (observability)

**What it builds:**
- `src/gaia/computer_use/vision.py` -- Screenshot capture, OCR (Tesseract), VLM element detection
- `src/gaia/computer_use/coordinates.py` -- Coordinate extraction from VLM output
- `src/gaia/computer_use/screen.py` -- Multi-monitor support, region capture

- **Always on:** NO -- opt-in via `enable_computer_use=True`
- **LLM tier requirement:** `advanced` (needs VLM)
- **Validation criteria:**
  - Screenshot captured and analyzed by VLM
  - UI elements identified with bounding boxes
  - Coordinates extracted for click targets

---

### Branch 15 (Track B): `feature/v2-computer-use-input` (6 hours)

**Priority:** P1
**Depends on:** Branch 14 (vision -- needs target coordinates)

**What it builds:**
- `src/gaia/computer_use/input.py` -- Mouse (click, drag, scroll), Keyboard (type, hotkeys)
- `src/gaia/computer_use/timing.py` -- Human-like timing and jitter

- **Validation criteria:**
  - Click on VLM-identified target coordinates
  - Type text with human-like speed variation
  - Keyboard shortcuts work (Ctrl+C, Alt+Tab)

---

### Branch 16 (Track B): `feature/v2-computer-use-browser` (8 hours)

**Priority:** P1
**Depends on:** Branch 14 (vision) + Branch 15 (input)

**What it builds:**
- `src/gaia/computer_use/browser.py` -- Playwright integration, page navigation
- `src/gaia/computer_use/forms.py` -- Form detection, filling, submission
- `src/gaia/computer_use/tabs.py` -- Tab management, URL navigation

- **Validation criteria:**
  - Navigate to URL, fill form, submit
  - Tab management (open, switch, close)
  - Form fields identified and filled correctly

---

### Branch 17 (Track B): `feature/v2-computer-use-desktop` (8 hours)

**Priority:** P1
**Depends on:** Branch 14 (vision) + Branch 15 (input)

**What it builds:**
- `src/gaia/computer_use/desktop.py` -- Windows accessibility API (UIAutomation)
- `src/gaia/computer_use/apps.py` -- Application launching, window management
- `src/gaia/computer_use/system.py` -- System tray, notifications, file dialogs

- **Validation criteria:**
  - Launch application by name
  - Switch between windows
  - Interact with native UI controls via accessibility API

---

### Branch 18 (Track B): `feature/v2-computer-use-safety` (6 hours)

**Priority:** P0 (safety is non-negotiable for computer use)
**Depends on:** All computer use branches above

**What it builds:**
- `src/gaia/computer_use/safety.py` -- SafetyGuard (forbidden actions, URL allowlist, confirmation prompts)
- `src/gaia/computer_use/panic.py` -- Panic button (Ctrl+Shift+Esc) halts all actions
- `src/gaia/computer_use/rollback.py` -- Undo last N actions where possible
- `src/gaia/computer_use/audit.py` -- Full audit trail of every action

- **Always on when computer use enabled:** YES -- SafetyGuard cannot be disabled if computer use is active
- **Validation criteria:**
  - Forbidden actions blocked (format disk, delete system files, admin commands)
  - Panic button halts all actions immediately
  - Every action logged to audit trail
  - User confirmation required for destructive actions

---

### Branch 19 (Track B): `feature/v2-computer-use-controller` (6 hours)

**Priority:** P1
**Depends on:** All computer use branches above

**What it builds:**
- `src/gaia/computer_use/controller.py` -- ComputerUseController (intent -> plan -> execute -> verify)
- `src/gaia/computer_use/replanner.py` -- Action replanning on failure
- `src/gaia/computer_use/mixin.py` -- `ComputerUseMixin`

- **Validation criteria:**
  - Full loop: screenshot -> plan -> execute -> verify result
  - Replanning when action fails (element moved, page changed)
  - Integration with SafetyGuard at every step

---

### Branch 20 (Track B): `feature/v2-email-integration` (8 hours)

**Priority:** P2
**Depends on:** Branch 3 (security for OAuth), Branch 2 (error recovery)

**What it builds:**
- `src/gaia/email/` -- GmailProvider, IMAP/SMTP, EmailAgent
- `src/gaia/email/providers/gmail.py` -- Gmail API integration
- `src/gaia/email/providers/imap.py` -- Generic IMAP provider
- `src/gaia/email/classify.py` -- LLM-based email classification
- `src/gaia/email/mixin.py` -- `EmailToolsMixin`

- **Always on:** NO -- opt-in via `enable_email=True`
- **Default off because:** Requires OAuth credentials, sends real emails, privacy-sensitive.
- **Security requirement:** `enable_security_vault=True` MUST also be set
- **Validation criteria:**
  - Fetch unread emails via Gmail API
  - Classify emails into categories
  - Draft and send responses (with user confirmation)
  - OAuth tokens stored in SecretVault

---

### Branch 21 (Track B): `feature/v2-workflow-orchestration` (10 hours)

**Priority:** P2
**Depends on:** Branch 5 (tasks), Branch 20 (email), Branch 3 (security), Branch 2 (error recovery)

**What it builds:**
- `src/gaia/workflow/` -- WorkflowEngine, triggers (cron, webhook), DAG execution
- `src/gaia/workflow/engine.py` -- Workflow execution with 7 step types
- `src/gaia/workflow/triggers.py` -- Cron, webhook, event triggers
- `src/gaia/workflow/dag.py` -- DAG-based step sequencing
- `src/gaia/workflow/mixin.py` -- `WorkflowMixin`

- **Always on:** NO -- opt-in via `enable_workflow=True`
- **Default off because:** Long-running background workflows need careful resource management. Advanced use case.
- **LLM tier requirement:** `advanced`
- **Validation criteria:**
  - Cron trigger fires on schedule
  - DAG with 5 steps executes in correct order
  - Failed step retries with backoff
  - Workflow state persists across restarts

**Track B total: ~62 hours -> ~3 days at 24h/day**

---

## Convergence (Days 7-9)

### Branch 22: `feature/v2-multi-agent` (10 hours)

**Priority:** P2
**Depends on:** All core features (Branches 1-12)

**What it builds:**
- `src/gaia/agents/orchestration/` -- OrchestratorAgent, MessageBus, SharedState
- `src/gaia/agents/orchestration/orchestrator.py` -- Task decomposition + specialist routing
- `src/gaia/agents/orchestration/message_bus.py` -- Async inter-agent communication
- `src/gaia/agents/orchestration/shared_state.py` -- CRDT conflict resolution

- **Always on:** NO -- opt-in via `enable_multi_agent=True`
- **Default off because:** Complex, resource-intensive. Multiple LLM instances.
- **LLM tier requirement:** `advanced`
- **Validation criteria:**
  - Orchestrator decomposes "build full-stack app" into 3 specialist tasks
  - Agents communicate via message bus
  - Shared state merges without conflicts

---

### Branch 23: `feature/v2-advanced-rag` (8 hours)

**Priority:** P2
**Depends on:** Branch 4 (memory)

**What it builds:**
- Enhancement to existing `src/gaia/rag/` -- Hybrid search, re-ranking, semantic chunking
- `src/gaia/rag/hybrid.py` -- BM25 + vector search fusion
- `src/gaia/rag/reranker.py` -- Cross-encoder re-ranking
- `src/gaia/rag/chunking.py` -- Semantic-aware chunking
- `src/gaia/rag/knowledge_graph.py` -- Entity extraction + graph building

- **Always on:** NO -- opt-in via `enable_advanced_rag=True`
- **Default off because:** Current RAG works fine for basic use. Advanced RAG adds latency and complexity.
- **Validation criteria:**
  - Hybrid search improves retrieval relevance by 20%+
  - Re-ranking filters irrelevant chunks
  - Knowledge graph links entities across documents

---

### Branch 24: `feature/v2-multi-doc-synthesis` (8 hours)

**Priority:** P2
**Depends on:** Branch 23 (advanced RAG), Branch 4 (memory)

**What it builds:**
- `src/gaia/multi_doc/` -- DocumentStore, EntityResolver, ContradictionDetector
- `src/gaia/multi_doc/store.py` -- Multi-document management
- `src/gaia/multi_doc/entity.py` -- Entity resolution across documents
- `src/gaia/multi_doc/contradiction.py` -- Contradiction detection
- `src/gaia/multi_doc/citations.py` -- Citation tracking

- **Always on:** NO -- opt-in via `enable_multi_doc=True`
- **Default off because:** Specialized for knowledge assistant use case. Heavy LLM usage.
- **LLM tier requirement:** `advanced`
- **Validation criteria:**
  - 10 documents ingested and cross-referenced
  - Entities linked across documents
  - Contradictions detected and reported with citations

---

### Branch 25: `feature/v2-tui-full` (10 hours)

**Priority:** P2
**Depends on:** Branch 7 (state machine), Branch 4 (memory), Branch 8 (manifest)

**What it builds:**
- `src/gaia/tui/` -- Textual-based TUI with all panels
- `src/gaia/tui/app.py` -- Main TUI application
- `src/gaia/tui/panels/chat.py` -- Chat view with streaming
- `src/gaia/tui/panels/state.py` -- State machine monitor
- `src/gaia/tui/panels/memory.py` -- Memory browser
- `src/gaia/tui/panels/manifest.py` -- File/dependency viewer
- `src/gaia/tui/panels/progress.py` -- Task progress
- `src/gaia/tui/panels/observability.py` -- Tracing overlays
- `src/gaia/tui/animations.py` -- Processing animations
- `src/gaia/tui/app.css` -- Styling

- **Always on:** NO -- opt-in via `gaia-tui` command (separate entry point)
- **Default off because:** Users who prefer CLI shouldn't be forced into TUI. Requires `textual` dependency.
- **Validation criteria:**
  - `gaia-tui` launches with all panels
  - Real-time streaming of agent output
  - State transitions visible in monitor panel
  - Memory searchable from browser panel
  - Keyboard shortcuts functional
  - Graceful fallback if `textual` not installed

---

### Branch 26: `feature/v2-voice` (6 hours)

**Priority:** P3
**Depends on:** Branch 5 (tasks)

**What it builds:**
- `src/gaia/voice/` -- Enhanced STT/TTS, wake word, voice commands
- Integration with existing `src/gaia/audio/` (Whisper ASR, Kokoro TTS)
- `src/gaia/voice/commands.py` -- Voice command parser
- `src/gaia/voice/notifications.py` -- Voice notification system

- **Always on:** NO -- opt-in via `enable_voice=True`
- **Validation criteria:**
  - Voice command creates a task
  - Status query returns spoken response
  - Wake word detection (optional)

---

### Integration + Polish (12 hours)

- Cross-track integration tests (coding agent + computer use together)
- Composability profile presets (`minimal`, `standard`, `advanced`)
- LLM tier validation at startup (warn if features above tier)
- Documentation updates
- Release packaging

**Convergence total: ~54 hours -> ~2.5 days**

---

## Release Milestones

### Milestone 1: "Coding Agent MVP" (Day 3)

**Branch cutoff:** Foundation + Track A through State Machine (Branches 1-7)

**What works:**
- Agent runs continuously until task verified complete (no step limit)
- Task queue with pause/resume/dependencies
- Episodic memory across sessions
- State machine: Requirements -> Planning -> Implementation -> Testing -> Debug -> Review
- Requirements gathering asks clarifying questions before building
- All features opt-in, existing agents unchanged

**Demo:**
```bash
gaia-code --enable-v2 "Build a REST API with auth, tests, and docs"
# Agent asks: "FastAPI or Flask? JWT or session? PostgreSQL or SQLite?"
# Runs until all tests pass, not until max_steps
```

**LLM requirement:** `standard` (7B+)

---

### Milestone 2: "Computer Use MVP" (Day 5)

**Branch cutoff:** Foundation + Track B through Computer Use Safety (Branches 1-3, 13-19)

**What works:**
- Screenshot -> VLM analysis -> action planning -> execution -> verification
- Mouse/keyboard control with human-like timing
- Browser automation via Playwright
- Safety guard: forbidden action list, confirmation prompts, panic button (Ctrl+Shift+Esc)
- Full audit trail of every action
- Observability tracing for all computer use actions

**Demo:**
```bash
gaia computer-use --enable-v2 "Fill out the expense report in the open browser tab"
# Screenshots page, identifies form fields via VLM
# Types values, clicks submit
# Verifies result, reports completion
```

**LLM requirement:** `advanced` (30B+ with VLM)
**Security requirement:** SecretVault enabled, explicit user consent

---

### Milestone 3: "Learning Agent" (Day 7)

**Branch cutoff:** Full Track A + Track B Email/Workflow (Branches 1-21)

**What works (in addition to M1 + M2):**
- Manifest tracks all files and dependencies in project
- Semantic memory consolidates patterns across sessions
- Learning loop adjusts from feedback and outcomes
- Dynamic tool creation when patterns detected
- Adaptive prompts inject learned conventions
- Email triage and workflow orchestration
- All features composable and independently toggleable

**Demo:**
```bash
gaia-code --profile=advanced "Build the same API pattern you built last week"
# Agent recalls learned patterns, skips requirements gathering
# Uses previously created custom tools
# Follows conventions learned from prior sessions
```

---

### Milestone 4: "Production Release" (Day 9)

**Branch cutoff:** All branches merged (Branches 1-26 + integration)

**What works (in addition to M1 + M2 + M3):**
- Multi-agent orchestration (decompose -> delegate -> merge)
- Advanced RAG (hybrid search, re-ranking, knowledge graphs)
- Multi-document synthesis (entity resolution, contradictions, citations)
- Full TUI with all panels, animations, observability overlays
- Voice commands
- Composability profiles (`minimal` / `standard` / `advanced`)
- Cross-track integration (coding agent that can also control browser)

**Demo:**
```bash
# Coding agent that also browses docs
gaia-code --profile=advanced \
  --enable-computer-use \
  "Build a Stripe integration -- look up the latest API docs in the browser"

# Multi-agent decomposition
gaia-code --enable-multi-agent \
  "Build a full-stack social media app with React, FastAPI, and PostgreSQL"
# Orchestrator spawns: FrontendAgent, BackendAgent, TestAgent
```

---

## LLM Tier Compatibility Matrix

| Feature | `basic` (0.6B) | `standard` (7B) | `advanced` (30B+) |
|---------|:-:|:-:|:-:|
| Config System | **YES** | **YES** | **YES** |
| Error Recovery (basic) | **YES** | **YES** | **YES** |
| Input Sanitization | **YES** | **YES** | **YES** |
| Structured Logging | **YES** | **YES** | **YES** |
| Feature Registry | **YES** | **YES** | **YES** |
| Memory Tier 1-2 | - | **YES** | **YES** |
| Task Queue | - | **YES** | **YES** |
| Observability | - | **YES** | **YES** |
| Continuous Execution | - | **YES** | **YES** |
| State Machine | - | **YES** | **YES** |
| Manifest | - | **YES** | **YES** |
| Memory Tier 3-4 | - | - | **YES** |
| Adaptive Prompts | - | - | **YES** |
| Learning Loop | - | - | **YES** |
| Dynamic Tools | - | - | **YES** |
| Advanced RAG | - | **YES** | **YES** |
| Multi-Doc Synthesis | - | - | **YES** |
| Computer Use | - | - | **YES** (VLM) |
| Email Integration | - | **YES** | **YES** |
| Workflow Orchestration | - | - | **YES** |
| Multi-Agent | - | - | **YES** |
| TUI | **YES** | **YES** | **YES** |
| Voice | **YES** | **YES** | **YES** |

---

## Composability Matrix by Milestone

```
                            M1(Day3)  M2(Day5)  M3(Day7)  M4(Day9)
                            --------  --------  --------  --------
Config + Feature Flags         Y         Y         Y         Y     ALWAYS ON
Error Recovery (basic)         Y         Y         Y         Y     ALWAYS ON
Input Sanitization             Y         Y         Y         Y     ALWAYS ON
Structured Logging             Y         Y         Y         Y     ALWAYS ON

Memory Tier 1-2                Y         Y         Y         Y     opt-in
Task Queue                     Y         Y         Y         Y     opt-in
Continuous Execution           Y         Y         Y         Y     opt-in
State Machine                  Y         Y         Y         Y     opt-in

Observability                  -         Y         Y         Y     opt-in
Security Vault                 -         Y         Y         Y     opt-in
Computer Use (vision)          -         Y         Y         Y     opt-in
Computer Use (input)           -         Y         Y         Y     opt-in
Computer Use (browser)         -         Y         Y         Y     opt-in
Computer Use (safety)          -         Y         Y         Y     opt-in
Computer Use (desktop)         -         Y         Y         Y     opt-in

Manifest                       -         -         Y         Y     opt-in
Memory Tier 3-4                -         -         Y         Y     opt-in
Adaptive Prompts               -         -         Y         Y     opt-in
Learning Loop                  -         -         Y         Y     opt-in
Dynamic Tools                  -         -         Y         Y     opt-in
Email Integration              -         -         Y         Y     opt-in
Workflow Orchestration         -         -         Y         Y     opt-in

Multi-Agent                    -         -         -         Y     opt-in
Advanced RAG                   -         -         -         Y     opt-in
Multi-Doc Synthesis            -         -         -         Y     opt-in
Full TUI                       -         -         -         Y     opt-in
Voice                          -         -         -         Y     opt-in
```

---

## Computer Use Sub-Branch Breakdown

Since Computer Use is a flagship capability, here is the detailed sub-branch sequence:

| Sub-Branch | Hours | What It Builds | Depends On |
|------------|-------|----------------|------------|
| `v2-computer-use/vision` | 10h | Screenshot capture, OCR (Tesseract), VLM element detection, coordinate extraction | Security, Observability |
| `v2-computer-use/input` | 6h | Mouse (click, drag, scroll), Keyboard (type, hotkeys), human-like timing/jitter | Vision (needs target coords) |
| `v2-computer-use/browser` | 8h | Playwright integration, page navigation, form filling, tab management | Vision + Input |
| `v2-computer-use/desktop` | 8h | Windows accessibility API (UIAutomation), app launching, window management | Vision + Input |
| `v2-computer-use/safety` | 6h | SafetyGuard (forbidden actions, URL allowlist, confirmation prompts), panic button, rollback, audit trail | All above |
| `v2-computer-use/controller` | 6h | ComputerUseController (intent -> plan -> execute -> verify loop), action replanning on failure | All above |

**Total: 44 hours -> ~2 days autonomous**

---

## Coding Agent Sub-Branch Breakdown

| Sub-Branch | Hours | What It Builds | Key Capability |
|------------|-------|----------------|----------------|
| `v2-memory-tier1-2` | 8h | Episodic memory, session recall | "Remember what we discussed yesterday" |
| `v2-task-queue` | 6h | Task CRUD, priority scheduling | "Queue up: build API, then write tests, then deploy" |
| `v2-continuous-execution` | 6h | Run until done, quality gates, checkpoints | "Build this entire app" (no step limit) |
| `v2-state-machine` | 8h | 6 states + requirements gathering | "What framework? What DB?" before building |
| `v2-manifest` | 6h | File tracking, dependency graph | Agent knows every file it created and why |
| `v2-memory-tier3-4` | 6h | Semantic memory, consolidation | "This project uses pytest fixtures, not setUp" |
| `v2-adaptive-prompts` | 4h | State-aware prompts, learned conventions | Prompt automatically includes project patterns |
| `v2-learning-loop` | 8h | Feedback, outcomes, pattern extraction | "Don't use subprocess, use pathlib" -> permanent |
| `v2-dynamic-tools` | 8h | Tool generation, skill store | Agent creates reusable tools from repeated patterns |

**Total: 60 hours -> ~2.5 days autonomous**

---

## Dependency Graph

```
Day 1:
  +-- Branch 1: v2-foundation-config -----------------+
  +-- Branch 2: v2-error-recovery ---------------------+
  +-- Branch 3: v2-security-foundations ---------------+
                                                       |
Day 2-3:                                               |
  +-- Branch 4:  v2-memory-tier1-2 -------------------+ (needs 1)
  +-- Branch 5:  v2-task-queue -----------------------+ (needs 1,2)
  +-- Branch 6:  v2-continuous-execution -------------+ (needs 1,2)
  +-- Branch 13: v2-observability --------------------+ (needs 1)
                                                       |
Day 3-5:                                               |
  +-- Branch 7:  v2-state-machine --------------------+ (needs 4)
  +-- Branch 8:  v2-manifest -------------------------+ (needs 4,5)
  +-- Branch 9:  v2-memory-tier3-4 -------------------+ (needs 4)
  +-- Branch 14: v2-computer-use-vision --------------+ (needs 3,13)
  +-- Branch 15: v2-computer-use-input ---------------+ (needs 14)
                                                       |
Day 4-6:                                               |
  +-- Branch 10: v2-adaptive-prompts -----------------+ (needs 4,7)
  +-- Branch 11: v2-learning-loop --------------------+ (needs 8,9)
  +-- Branch 16: v2-computer-use-browser -------------+ (needs 14,15)
  +-- Branch 17: v2-computer-use-desktop -------------+ (needs 14,15)
  +-- Branch 18: v2-computer-use-safety --------------+ (needs 14-17)
                                                       |
Day 5-7:                                               |
  +-- Branch 12: v2-dynamic-tools --------------------+ (needs 11)
  +-- Branch 19: v2-computer-use-controller ----------+ (needs 14-18)
  +-- Branch 20: v2-email-integration ----------------+ (needs 3,2)
  +-- Branch 21: v2-workflow-orchestration ------------+ (needs 5,20,3)
  +-- Branch 25: v2-tui-full -------------------------+ (needs 7,4,8)
                                                       |
Day 7-9:                                               |
  +-- Branch 22: v2-multi-agent ----------------------+ (needs all core)
  +-- Branch 23: v2-advanced-rag ---------------------+ (needs 4)
  +-- Branch 24: v2-multi-doc-synthesis --------------+ (needs 23)
  +-- Branch 26: v2-voice ----------------------------+ (needs 5)
  +-- Integration + Polish ---------------------------+
```

---

## Composability Profiles

For ease of use, offer preset profiles that bundle feature flags:

```python
# Simple chat agent with small LLM
config = V2Config.from_profile("minimal")
# Enables: nothing extra (basic retry only)

# Standard agent with medium LLM
config = V2Config.from_profile("standard")
# Enables: memory, task_queue, observability, state_machine

# Full-featured agent with large LLM
config = V2Config.from_profile("advanced")
# Enables: everything

# Coding agent profile
config = V2Config.from_profile("coding")
# Enables: memory, task_queue, continuous_execution, state_machine, manifest,
#          adaptive_prompts, learning, dynamic_tools

# Computer use profile
config = V2Config.from_profile("computer_use")
# Enables: security_vault, observability, computer_use (all sub-features)

# Custom composition
config = V2Config(
    enable_memory=True,
    enable_state_machine=True,
    enable_manifest=True,
    # everything else stays off
)
```

---

## Key Architectural Rules

1. **Every feature is a Mixin** -- never modify the base `Agent` class to require V2 features
2. **Feature flags checked at init** -- if a feature is disabled, its mixin's `__init__` becomes a no-op
3. **No cross-feature hard dependencies at runtime** -- if memory is disabled, learning should gracefully degrade (not crash)
4. **LLM tier validation at startup** -- warn if user enables features above their LLM tier
5. **Each branch has its own test suite** -- tests for disabled features should verify zero overhead
6. **Each branch is independently mergeable** -- no branch should break `main` if merged alone
7. **Safety features are non-optional within their domain** -- if computer use is enabled, SafetyGuard is always active (cannot be disabled separately)
8. **Backward compatibility guarantee** -- existing V1 code works unchanged:

```python
# This still works in GAIA V2 with zero changes
from gaia.agents.chat.agent import ChatAgent
agent = ChatAgent()
agent.process_query("Analyze this document", max_steps=10)
```

---

## Summary

| Dimension | Original Plan (Human + AI) | Updated Plan (Autonomous Agent) |
|-----------|---------------------------|-------------------------------|
| **Timeline** | 12-16 weeks | **9 days** |
| **Milestone structure** | Alpha -> Beta -> RC -> GA | **Coding MVP -> Computer Use MVP -> Learning -> Production** |
| **Computer Use priority** | Branch 17, Week 8-10 | **Track B co-primary, Day 3-5** |
| **Coding Agent priority** | Spread across all phases | **Track A co-primary, Day 2-3 MVP** |
| **Parallelism** | 1-2 streams | **2 full tracks + convergence** |
| **First usable release** | Week 4 (Alpha) | **Day 3 (Coding MVP)** |
| **Everything working** | Week 12+ | **Day 9** |
| **Total branches** | 22 | **26 (Computer Use split into 6 sub-branches)** |
| **Total estimated hours** | ~76 engineer-weeks | **~182 autonomous-agent-hours** |
| **Always-on components** | 5 | **5 (unchanged)** |
| **Opt-in features** | 17 | **21 (more granular)** |

---

*V2 Feature Branch Plan for GAIA.*
*Composable architecture with two parallel tracks: Coding Agent + Computer Use Agent.*
*All advanced features disabled by default, enabled via feature flags.*
*Estimated 9 days with autonomous coding agents.*
