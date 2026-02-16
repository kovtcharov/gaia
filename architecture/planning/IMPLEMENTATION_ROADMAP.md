# Gaia V2: Implementation Roadmap & Migration Strategy

**Date**: February 6, 2026
**Version**: 1.0
**Scope**: Phased roadmap for transitioning from Gaia 0.15.3 to Gaia V2 with all advanced architectures
**Goal**: Clean migration path with backward compatibility and incremental value delivery

---

## Executive Summary

**What is Gaia V2**: A complete reimagining of the Gaia Agent SDK with:
- Continuous execution (no token limits)
- 4-tier persistent memory
- State machine for multi-modal execution
- Dynamic tool/skill creation
- Task-centric interface (not chat)
- Voice integration
- Project manifest tracking
- Learning loops with outcome tracking
- Enhanced TUI with accomplishment-focused UX

**Migration Strategy**: Incremental enhancement, not rewrite. Each phase adds new capabilities while maintaining backward compatibility with Gaia 0.15.3 agents.

---

## Dependency Graph

```
                    ┌─────────────────────┐
                    │  GAIA 0.15.3 BASE   │
                    │  (Current)          │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
   ┌────────▼────────┐ ┌──────▼───────┐ ┌───────▼────────┐
   │  CONTINUOUS     │ │    TASK      │ │   MEMORY       │
   │  EXECUTION      │ │  INTERFACE   │ │   TIER 1-2     │
   │  (Phase 0)      │ │  (Phase 0)   │ │   (Phase 0)    │
   └────────┬────────┘ └──────┬───────┘ └───────┬────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    MANIFEST         │
                    │    FRAMEWORK        │
                    │    (Phase 1)        │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
   ┌────────▼────────┐ ┌──────▼───────┐ ┌───────▼────────┐
   │  STATE          │ │  MEMORY      │ │   BASIC TUI    │
   │  MACHINE        │ │  TIER 3-4    │ │   (Phase 1)    │
   │  (Phase 1)      │ │  (Phase 1)   │ │                │
   └────────┬────────┘ └──────┬───────┘ └───────┬────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   LEARNING LOOP     │
                    │   (Phase 2)         │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┬──────────────┐
            │                  │                  │              │
   ┌────────▼────────┐ ┌──────▼───────┐ ┌───────▼────────┐ ┌──▼────┐
   │  DYNAMIC        │ │   VOICE      │ │   ENHANCED     │ │DASH   │
   │  TOOLS/SKILLS   │ │   (STT/TTS)  │ │   TUI          │ │BOARD  │
   │  (Phase 2)      │ │  (Phase 2)   │ │   (Phase 2)    │ │(Ph 2) │
   └─────────────────┘ └──────────────┘ └────────────────┘ └───────┘
```

**Critical Path** (must be built in order):
1. Continuous Execution + Task Interface + Memory Tier 1-2 (can parallelize)
2. Manifest Framework (depends on Memory)
3. State Machine + Memory Tier 3-4 + Basic TUI (can parallelize)
4. Everything else builds on top

---

## Phase 0: Foundation (Week 1-3) 🚀 START HERE

**Goal**: Enable continuous execution with basic task management

### Week 1: Continuous Execution Engine

**Priority**: P0 (blocking for everything else)
**Dependencies**: None
**Effort**: 1 week, 1 engineer

**Deliverables**:
- `ContinuousExecutionEngine` class
- Replace `max_steps` with completion criteria
- Quality gate system (syntax, tests, lint)
- Checkpoint/resume basic implementation

**Code location**: `gaia/execution/continuous.py`

**API**:
```python
agent.continuous_execute_until_complete(
    task_description="Build REST API",
    completion_criteria={"all_tests_pass": True},
)
```

**Test**: Build a 10-file project, verify agent doesn't stop at step 20

**Backward Compatibility**: Existing `process_query()` unchanged, continuous execution is opt-in

---

### Week 2: Task Interface & Queue

**Priority**: P0
**Dependencies**: Continuous Execution
**Effort**: 1 week, 1 engineer

**Deliverables**:
- `Task` dataclass and database schema
- `TaskManager` class with create/start/pause/resume
- `TaskQueue` with priority and dependency scheduling
- `TaskToolsMixin` (create_task, pause_task, resume_task, list_tasks)

**Code location**: `gaia/tasks/`

**API**:
```python
task = task_manager.create_task(
    title="Build API",
    description="...",
    completion_criteria={},
)
task_manager.start_task(task.task_id, agent)
```

**Test**: Queue 3 tasks, verify concurrent execution up to `max_concurrent`

**Backward Compatibility**: Tasks are optional — agents can still use chat-only mode

---

### Week 3: Memory Tier 1-2 (Working + Episodic)

**Priority**: P0
**Dependencies**: None (can parallelize with Week 1-2)
**Effort**: 1 week, 1 engineer

**Deliverables**:
- `EpisodicMemory` class with FAISS + JSON storage
- Session summarization logic
- `recall` tool for memory search
- Conversation persistence

**Code location**: `gaia/memory/episodic.py`

**API**:
```python
agent.episodic_memory.save_session(session_summary)
results = agent.episodic_memory.search("find analyses about allreduce")
```

**Test**: Run 10 sessions, verify all are searchable, resume from session 5

**Backward Compatibility**: Memory is opt-in via config flag `enable_episodic_memory=True`

---

**Phase 0 Exit Criteria**:
- ✅ Agent can run a task until verified complete (no step limit)
- ✅ Tasks can be queued, paused, resumed
- ✅ Past sessions are searchable
- ✅ Checkpoints enable recovery from crash

**Deliverable**: Gaia V2 Alpha — continuous execution + tasks + episodic memory

---

## Phase 1: Core Infrastructure (Week 4-10)

**Goal**: Add manifest, state machine, semantic memory, basic TUI

### Week 4-5: Architecture Manifest

**Priority**: P1
**Dependencies**: Memory Tier 1-2
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- `ProjectManifest` data model + SQLite schema
- `ManifestTracker` with auto-update on file operations
- `DependencyGraph` with import analysis
- `ManifestToolsMixin`

**Code location**: `gaia/manifest/`

**Test**: Build 30-file web app, verify all files/dependencies tracked

---

### Week 6-7: State Machine

**Priority**: P1
**Dependencies**: Memory (for state persistence)
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- `AgentState` dataclass + database schema
- `StateMachine` class with enter/execute/return
- 5 predefined states (Planning, Implementation, Testing, Debug, Review)
- `StateToolsMixin`

**Code location**: `gaia/state_machine/`

**Test**: Execute task with automatic state transitions (Planning → Implementation → Debug → Implementation → Testing)

---

### Week 8: Memory Tier 3-4 (Semantic + Universal)

**Priority**: P1
**Dependencies**: Episodic Memory (for consolidation)
**Effort**: 1 week, 1 engineer

**Deliverables**:
- `SemanticMemory` with SQLite knowledge base
- `UniversalKnowledgeDB` storing all interactions
- Memory consolidation pipeline
- Full CRUD operations

**Code location**: `gaia/memory/semantic.py`, `gaia/memory/universal.py`

**Test**: Run consolidation on 20 sessions, verify patterns extracted with confidence scores

---

### Week 9-10: Basic TUI

**Priority**: P1
**Dependencies**: Task Interface, Memory, Manifest
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- Main TUI app with Textual
- Chat panel + State/Progress panel
- Task list view
- Memory browser (basic)
- Keyboard shortcuts

**Code location**: `gaia/tui/`

**Test**: Run TUI, create task, monitor real-time execution, search memory

---

**Phase 1 Exit Criteria**:
- ✅ Agent tracks project architecture in manifest
- ✅ Agent switches between execution states intelligently
- ✅ All interactions stored in universal DB
- ✅ TUI provides real-time monitoring
- ✅ Knowledge accumulates in semantic memory

**Deliverable**: Gaia V2 Beta — full memory + manifest + states + TUI

---

## Phase 2: Advanced Capabilities (Week 11-18)

**Goal**: Learning, tool creation, voice, enhanced UX

### Week 11-12: Learning Loop

**Priority**: P2
**Dependencies**: Memory Tier 3 (semantic), Manifest
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- `FeedbackCollector` (explicit + implicit + automated)
- `OutcomeTracker` with confidence adjustment
- `PatternExtractor` for success/failure patterns
- `KnowledgeConsolidator`

**Code location**: `gaia/learning/`

---

### Week 13-14: Dynamic Tools & SKILLS

**Priority**: P2
**Dependencies**: Learning Loop (for quality tracking)
**Effort**: 2 weeks, 1-2 engineers

**Deliverables**:
- `ToolBuilderAgent` (CodeAgent specialist)
- `DynamicToolManager` with security validation
- `SkillStore` with FAISS vector search
- `SkillToolsMixin`

**Code location**: `gaia/tools/dynamic/`, `gaia/skills/`

---

### Week 15-16: Voice Interface

**Priority**: P2
**Dependencies**: Task Interface
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- `VoiceInterface` with STT (Whisper) + TTS (Coqui)
- Voice command parser
- Wake word detection (optional)
- Voice notification system

**Code location**: `gaia/voice/`

---

### Week 17-18: Enhanced TUI

**Priority**: P2
**Dependencies**: All Phase 1 + Learning
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- Accomplishment-focused main view
- Rich syntax highlighting
- Custom animations (processing, writing, debugging)
- Observability overlays (Ctrl+O/M/T/S)
- Settings panel with customization
- Voice integration in TUI

**Code location**: `gaia/tui/` (enhancement)

---

**Phase 2 Exit Criteria**:
- ✅ Agent learns from feedback and outcomes
- ✅ Agent creates tools/skills when patterns detected
- ✅ Voice commands work (task creation, status queries)
- ✅ TUI is beautiful, fast, and customizable

**Deliverable**: Gaia V2 RC — learning + tools + voice + polished UX

---

## Phase 3: Production & Scale (Week 19-24)

**Goal**: Multi-agent, web dashboard, code intelligence, security

### Week 19-20: Web Dashboard

**Priority**: P3
**Dependencies**: All Phase 1-2
**Effort**: 2 weeks, 1 frontend + 1 backend engineer

**Deliverables**:
- React frontend with 8 dashboard sections
- FastAPI backend with WebSocket
- Team access with authentication (optional)

**Code location**: `gaia/dashboard/`

---

### Week 21-22: Code Intelligence (LSP)

**Priority**: P3 (high value for code assistants)
**Dependencies**: Manifest
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- LSP client integration (Python, TypeScript, Go, Rust)
- Code intelligence tools (find_definition, find_references, semantic_search)

**Code location**: `gaia/code_intelligence/`

---

### Week 23-24: Security & Production Hardening

**Priority**: P3
**Dependencies**: Dynamic Tools
**Effort**: 2 weeks, 1 engineer

**Deliverables**:
- Sandbox execution (Docker containers)
- Security validator enhancements
- Audit logging
- Cost management
- Multi-tenant support (optional)

**Code location**: `gaia/security/`

---

**Phase 3 Exit Criteria**:
- ✅ Web dashboard for team access
- ✅ LSP-powered code intelligence
- ✅ Production security hardening

**Deliverable**: Gaia V2.0 — Production Ready

---

## Migration Strategy: Gaia 0.15.3 → V2

### Backward Compatibility Approach

**Principle**: V2 features are **additive, not breaking**

| V2 Feature | V1 Behavior | Compatibility |
|------------|-------------|---------------|
| Continuous execution | Uses `max_steps` | Opt-in via `continuous=True` flag |
| Task interface | Conversation-only | Tasks auto-created from conversations |
| Memory (Tier 3-4) | In-memory only | Opt-in via `enable_memory=True` |
| State machine | Single mode | Opt-in, default state = "general" |
| Dynamic tools | Static only | Opt-in via `enable_dynamic_tools=True` |
| Manifest | No tracking | Opt-in via `enable_manifest=True` |
| Voice | No voice | Opt-in via `enable_voice=True` |
| Enhanced TUI | Basic CLI | Launch with `gaia-tui` vs `gaia` |

### Migration Path for Existing Agents

#### Example: Migrating JarvisAgent

**V1 (Current)**:
```python
from gaia.agents.base.agent import Agent

class JarvisAgent(Agent):
    def __init__(self):
        super().__init__(max_steps=20)

    def _register_tools(self):
        # Register tools...
```

**V2 (Incremental)**:

**Step 1** (Week 3): Add episodic memory
```python
from gaia.agents.base.agent import Agent
from gaia.memory.episodic import EpisodicMemory  # NEW

class JarvisAgent(Agent):
    def __init__(self):
        super().__init__(max_steps=20)
        self.episodic_memory = EpisodicMemory()  # NEW

    # Everything else unchanged
```

**Step 2** (Week 6): Add continuous execution + tasks
```python
from gaia.agents.base.agent import Agent
from gaia.memory.episodic import EpisodicMemory
from gaia.execution.continuous import ContinuousMixin  # NEW
from gaia.tasks.manager import TaskManager  # NEW

class JarvisAgent(Agent, ContinuousMixin):  # NEW mixin
    def __init__(self):
        super().__init__(max_steps=20)  # Still works
        self.episodic_memory = EpisodicMemory()
        self.task_manager = TaskManager()  # NEW
```

**Step 3** (Week 10): Add manifest + states
```python
from gaia.agents.base.agent import Agent
from gaia.memory import EpisodicMemory, SemanticMemory  # NEW
from gaia.execution.continuous import ContinuousMixin
from gaia.tasks.manager import TaskManager
from gaia.manifest.tracker import ManifestTracker  # NEW
from gaia.state_machine import StateMachine  # NEW

class JarvisAgent(Agent, ContinuousMixin):
    def __init__(self, config):
        super().__init__(**config)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()  # NEW
        self.task_manager = TaskManager()
        self.manifest_tracker = ManifestTracker()  # NEW
        self.state_machine = StateMachine()  # NEW

        # Load predefined states
        self.state_machine.load_predefined_states([
            "performance_analysis", "latency_mode", "throughput_mode"
        ])
```

**Step 4** (Week 16): Add learning + tools
```python
# Full V2 agent
from gaia.agents.v2.agent import AgentV2  # NEW base class with all features

class JarvisAgent(AgentV2):  # Simplified
    def __init__(self, config):
        super().__init__(config)  # Gets all V2 features

    # Agent-specific customization only
    def _get_custom_states(self):
        return [...performance-specific states...]
```

---

## Detailed Phase Breakdown (AI-Assisted Development)

**Assumption**: Using Claude Code with Opus 4.6 for implementation
**Speedup Factor**: 3-5x faster than human-only development
**Team Size**: Smaller (AI handles boilerplate, testing, documentation)

### PHASE 0: Foundation (Week 1-2) — Compressed from 3 weeks

#### Week 1, Days 1-3: Continuous Execution + Quality Gates

**Team**: 1 engineer + Claude Code
**Human effort**: Design decisions, code review, integration testing
**Claude Code handles**: Implementation, tests, documentation, boilerplate

**Day 1** (8 hours human + Claude):
- Design `ContinuousExecutionEngine` API (human: 2h)
- Claude implements: `continuous.py`, `quality_gates.py`, `checkpoints.py` (6h with human review)
- Tests generated by Claude

**Day 2** (6 hours):
- Integrate with `Agent` base class (human: design 1h, Claude: implement 3h)
- Claude writes comprehensive test suite (2h)

**Day 3** (4 hours):
- Human testing and refinement
- Claude fixes issues, updates docs

**Files created** (by Claude):
- `gaia/execution/continuous.py` — `ContinuousExecutionEngine`
- `gaia/execution/quality_gates.py` — `QualityGateSystem`
- `gaia/execution/checkpoints.py` — `CheckpointManager`
- `tests/test_continuous_execution.py` (comprehensive)
- `docs/continuous_execution.md`

---

#### Week 1, Days 4-5: Task Interface

**Team**: 1 engineer + Claude Code
**Time**: 2 days (vs. 1 week human-only)

**Day 4** (6 hours):
- Design task data model (human: 1h)
- Claude implements: `Task` dataclass, `TaskManager`, `TaskQueue`, SQL schema (4h)
- Claude generates tests (1h)

**Day 5** (4 hours):
- Human reviews and tests
- Claude implements `TaskToolsMixin` (2h)
- Claude fixes integration issues (2h)

**Files created** (by Claude):
- `gaia/tasks/models.py`, `manager.py`, `queue.py`, `tools.py`, `schema.sql`
- `tests/test_task_manager.py`, `tests/test_task_queue.py`

---

#### Week 2: Memory Tier 1-2 + Gaia4 Integrations

**Team**: 1 engineer + Claude Code
**Time**: 1 week (vs. 2 weeks)

**Days 1-2** (Episodic Memory):
- Claude implements `EpisodicMemory` class with FAISS integration
- Claude writes `MemoryToolsMixin`
- Tests generated automatically

**Days 3-4** (Gaia4 Patterns):
- Claude implements `ProvenPatternsLibrary` (from Gaia4 insights)
- Claude implements `MultiTurnDiffFixer`
- Integration tests

**Day 5** (Integration):
- Human testing end-to-end
- Claude fixes bugs, polishes documentation

---

**Phase 0 Deliverable**: `gaia==2.0.0-alpha` (2 weeks, not 3)
- Continuous execution ✓
- Task interface ✓
- Episodic memory ✓
- Proven patterns ✓
- Multi-turn fixing ✓

---

### PHASE 1: Core Infrastructure (Week 3-5) — Compressed from 7 weeks

#### Week 3: Manifest + State Machine (Parallel)

**Team**: 1 engineer + Claude Code (doing 2 weeks of work in 1)
**Parallel workstreams**:

**Stream A** (Manifest):
- Claude implements: models, tracker, graph, detector, tools, schema (2 days)
- Human reviews and tests (1 day)

**Stream B** (State Machine):
- Claude implements: models, machine, predefined states, tools, schema (2 days)
- Human reviews, tests state transitions (1 day)

**Output**: Both frameworks complete in 1 week (vs. 4 weeks human-only)

---

#### Week 4: Memory Tier 3-4 + Gaia4 Enhancements (Parallel)

**Team**: 1 engineer + Claude Code

**Stream A** (Memory):
- Claude implements: `SemanticMemory`, `UniversalKnowledgeDB`, consolidation (2 days)
- Enhanced manifest with exports/imports tracking (from Gaia4) (1 day)

**Stream B** (Gaia4 Patterns):
- Claude implements: `IterativeRefinementEngine` (orchestrator pattern) (1 day)
- Claude implements: `RuntimeValidator` (boots dev server) (1 day)
- Claude implements: `StructuredErrorRecovery` with error categorization (1 day)

**Output**: Complete memory system + Gaia4 innovations

---

#### Week 5: Basic TUI

**Team**: 1 engineer + Claude Code
**Time**: 1 week (vs. 2 weeks)

**Days 1-3**:
- Claude scaffolds Textual app (app.py, base panels, CSS) (1 day)
- Claude implements ChatPanel, StateMonitorPanel (1 day)
- Claude implements MemoryBrowser, task list view (1 day)

**Days 4-5**:
- Human UX testing and refinement
- Claude iterates on design, fixes issues
- Claude adds keyboard shortcuts, animations

**Output**: Working TUI with real-time monitoring

---

**Phase 1 Deliverable**: `gaia==2.0.0-beta` (5 weeks total, not 10)
- All core infrastructure ✓
- Gaia4 patterns integrated ✓
- TUI available ✓

---

### PHASE 2: Advanced Features (Week 6-9) — Compressed from 8 weeks

#### Week 6: Learning Loop + Checklist Model (Parallel)

**Team**: 1 engineer + Claude Code

**Stream A** (Learning - 3 days):
- Claude implements: feedback collector, outcome tracker, pattern extractor (2 days)
- Human tests learning loop end-to-end (1 day)

**Stream B** (Checklist Model from Gaia4 - 2 days):
- Claude implements: `ChecklistGenerator`, `ChecklistExecutor`, template catalog (2 days)

**Output**: Learning + checklist execution model

---

#### Week 7: Dynamic Tools & SKILLS (Parallel)

**Team**: 1 engineer + Claude Code (doing 2 engineers' work)

**Stream A** (ToolBuilder - 3 days):
- Claude implements: `ToolBuilderAgent` (CodeAgent specialist) (2 days)
- Claude implements: `DynamicToolManager`, `SecurityValidator` (1 day)

**Stream B** (SKILLS - 2 days):
- Claude implements: `Skill` models, `SkillStore`, `SkillVectorStore` (1 day)
- Claude implements: `SkillToolsMixin`, pattern detection (1 day)

**Output**: Full dynamic tool + SKILLS system

---

#### Week 8: Voice Interface

**Team**: 1 engineer + Claude Code
**Time**: 1 week (vs. 2 weeks)

**Days 1-2**: STT/TTS integration
- Claude implements: Whisper wrapper, Coqui TTS wrapper (1 day)
- Claude implements: voice command parser (1 day)

**Days 3-4**: TUI integration
- Claude adds voice panel to TUI (1 day)
- Claude implements: wake word detection, notifications (1 day)

**Day 5**: Testing and polish
- Human tests voice accuracy
- Claude iterates on command parsing

---

#### Week 9: Enhanced TUI + Task-Specific Guidance (from Gaia4)

**Team**: 1 engineer + Claude Code
**Time**: 1 week (vs. 2 weeks)

**Days 1-2**: Visual enhancements
- Claude implements: accomplishment-focused layout, animations (1 day)
- Claude adds: syntax highlighting, emoji system, color palette (1 day)

**Days 3-4**: Observability panels
- Claude implements: deep dive overlays (tool trace, memory browser, state viz) (2 days)

**Day 5**: Settings + polish
- Claude implements: settings panel, user preferences persistence (0.5 day)
- Claude implements: task-specific guidance system (from Gaia4) (0.5 day)

---

**Phase 2 Deliverable**: `gaia==2.0.0` (9 weeks total, not 18)
- Full feature set ✓
- Gaia4 patterns integrated ✓
- Production ready ✓

---

### PHASE 3: Ecosystem (Week 19-24+)

#### Week 19-20: Web Dashboard

**Team**: 2 engineers (1 React, 1 FastAPI)
**Deliverables**: Full web dashboard from AGENT_DASHBOARD_DESIGN.md

---

#### Week 21-22: Code Intelligence

**Team**: 1 engineer
**Deliverables**: LSP integration, semantic code search

---

#### Week 23-24: Multi-Agent Orchestration

**Team**: 1 engineer
**Deliverables**: OrchestratorAgent, specialist sub-agents, message bus

---

**Phase 3 Deliverable**: `gaia==2.1.0`
- Enterprise features
- Multi-agent support
- Advanced code intelligence

---

## Team Structure

### Core Team (Week 1-18)

| Role | Count | Responsibilities |
|------|-------|------------------|
| **Backend Engineers** | 2-3 | Memory, execution, state machine, manifest, learning |
| **Frontend Engineer** | 1 | TUI (Textual), later dashboard (React) |
| **ML Engineer** | 1 | Learning loop, pattern extraction, consolidation |
| **Tech Lead** | 1 | Architecture, code review, integration |

**Total**: 4-5 engineers for 18 weeks

### Extended Team (Week 19-24)

Add 1-2 more engineers for dashboard, LSP, multi-agent

---

## Code Organization

```
gaia/
├── __init__.py (exports V2 classes)
│
├── agents/
│   ├── base/
│   │   ├── agent.py (enhanced with V2 hooks)
│   │   └── tools.py (unchanged)
│   ├── v2/
│   │   └── agent.py (NEW: AgentV2 base class with all features)
│   └── [existing agents: chat, code, etc.]
│
├── execution/          # NEW
│   ├── continuous.py
│   ├── quality_gates.py
│   └── checkpoints.py
│
├── tasks/              # NEW
│   ├── models.py
│   ├── manager.py
│   ├── queue.py
│   └── tools.py
│
├── memory/             # NEW
│   ├── episodic.py
│   ├── semantic.py
│   ├── universal.py
│   ├── consolidation.py
│   └── tools.py
│
├── manifest/           # NEW
│   ├── models.py
│   ├── tracker.py
│   ├── graph.py
│   └── tools.py
│
├── state_machine/      # NEW
│   ├── models.py
│   ├── machine.py
│   ├── states.py
│   └── tools.py
│
├── learning/           # NEW
│   ├── feedback.py
│   ├── outcomes.py
│   ├── patterns.py
│   └── tools.py
│
├── tools/              # NEW
│   ├── dynamic_manager.py
│   ├── builder_agent.py
│   └── security.py
│
├── skills/             # NEW
│   ├── models.py
│   ├── store.py
│   ├── vector.py
│   └── tools.py
│
├── voice/              # NEW
│   ├── stt.py
│   ├── tts.py
│   ├── commands.py
│   └── notifications.py
│
├── tui/                # NEW
│   ├── app.py
│   ├── panels/
│   │   ├── chat.py
│   │   ├── state.py
│   │   ├── memory.py
│   │   ├── accomplishments.py
│   │   └── observability.py
│   ├── animations.py
│   └── app.css
│
├── dashboard/          # NEW (Phase 3)
│   ├── backend/ (FastAPI)
│   └── frontend/ (React)
│
└── code_intelligence/  # NEW (Phase 3)
    └── lsp_client.py
```

---

## Migration Checklist

### For Existing Gaia Agents

- [ ] Update `gaia` to 2.0.0-alpha
- [ ] Add `enable_continuous=True` to config (opt-in)
- [ ] Add `EpisodicMemory` to `__init__`
- [ ] Test with existing workflows (should work unchanged)
- [ ] Gradually enable: manifest, states, learning
- [ ] Launch with `gaia-tui` instead of `gaia` CLI
- [ ] Add voice config if desired

### For New Agents

```python
# Start with AgentV2 base class (all features included)
from gaia.agents.v2.agent import AgentV2
from gaia.config import AgentV2Config

config = AgentV2Config(
    enable_all=True,  # All V2 features
    max_concurrent_tasks=3,
    enable_voice=True,
)

class MyAgent(AgentV2):
    # Minimal code — everything inherited
    pass
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation | Timeline |
|------|--------|------------|----------|
| **Breaking changes** to existing agents | HIGH | Strict backward compatibility, V2 as opt-in | Week 1 |
| **Scope creep** — trying to build everything | HIGH | Phased approach, MVP in Phase 0 | Ongoing |
| **Performance regression** from added features | MEDIUM | Benchmarking, lazy loading, optimization | Week 10, 18 |
| **Database migration** complexity | MEDIUM | Schema versioning, migration scripts | Week 4 |
| **TUI adoption** — users prefer CLI | LOW | Both TUI and CLI supported | Week 9 |
| **Voice quality** issues | MEDIUM | Local Whisper (proven), fallback to text | Week 15 |

---

## Success Metrics

### Phase 0 Success

- ✅ 3+ existing agents migrated without breaking
- ✅ Continuous execution works for 100+ step tasks
- ✅ Task pause/resume works
- ✅ Episodic memory searchable

### Phase 1 Success

- ✅ 30-file project tracked in manifest
- ✅ State transitions work automatically
- ✅ TUI used by 5+ developers
- ✅ Knowledge consolidation extracts patterns

### Phase 2 Success

- ✅ Agent creates 3+ custom tools/skills
- ✅ Voice commands work 95% accuracy
- ✅ Learning loop improves recommendations (measured via outcome tracking)
- ✅ TUI rated 8+/10 for UX

---

## Critical Path Summary

**Must build first** (blocking dependencies):
1. Continuous Execution (Week 1)
2. Task Interface (Week 2)
3. Memory Tier 1-2 (Week 3)

**Then can parallelize**:
- Manifest (Week 4-5) || State Machine (Week 6-7) || Memory Tier 3-4 (Week 8)
- All require Phase 0 but independent of each other

**Then build on top**:
- TUI (Week 9-10) — requires all Phase 1
- Learning (Week 11-12) — requires Memory Tier 3
- Tools/Skills (Week 13-14) — requires Learning
- Voice (Week 15-16) — requires Task Interface
- Enhanced TUI (Week 17-18) — requires everything

---

## Development Workflow

### Week 1-3 (Phase 0)

**Daily standup**: Continuous execution, tasks, memory teams sync

**Milestones**:
- Week 1 Friday: Continuous execution working, demo with 100-step task
- Week 2 Friday: Task queue working, demo with 3 concurrent tasks
- Week 3 Friday: Memory search working, demo searching 20 sessions

**Deliverable**: Alpha release, call for testers

### Week 4-10 (Phase 1)

**Sprint structure**: 2-week sprints
- Sprint 1 (Week 4-5): Manifest
- Sprint 2 (Week 6-7): State machine
- Sprint 3 (Week 8-9): Memory Tier 3-4 + TUI start
- Sprint 4 (Week 10): TUI finish

**Deliverable**: Beta release, production pilots

### Week 11-18 (Phase 2)

**Sprint structure**: 2-week sprints
- Sprint 5 (Week 11-12): Learning loop
- Sprint 6 (Week 13-14): Tools + SKILLS
- Sprint 7 (Week 15-16): Voice
- Sprint 8 (Week 17-18): Enhanced TUI + polish

**Deliverable**: V2.0 GA release

---

## Recommended First Steps (This Week)

**Day 1-2**: Set up architecture
- Create `gaia/execution/`, `gaia/tasks/`, `gaia/memory/` directories
- Define interfaces (`ContinuousExecutionEngine`, `TaskManager`, `EpisodicMemory`)
- Write schemas (SQL for tasks, memory)

**Day 3-4**: Implement continuous execution
- Build `ContinuousExecutionEngine.execute_until_complete()`
- Add quality gates (syntax check, test runner)
- Test with 100-step dummy task

**Day 5**: Implement task basics
- `Task` dataclass
- `TaskManager.create_task()`, `.start_task()`, `.pause_task()`
- Simple CLI: `gaia-task create "Build API"`

**Week 2**: Task queue + episodic memory (parallel)
- Team A: Task queue with priorities
- Team B: Episodic memory with FAISS

**Week 3**: Integration + testing
- Wire everything together
- Migrate JarvisAgent as reference
- Publish alpha

---

## Backward Compatibility Guarantee

**V1 code continues to work**:

```python
# This still works in Gaia V2
from gaia.agents.chat.agent import ChatAgent

agent = ChatAgent()
agent.process_query("Analyze this document", max_steps=10)
```

**V2 features are opt-in**:

```python
# Enable V2 features incrementally
from gaia.agents.chat.agent import ChatAgent
from gaia.memory import EpisodicMemory

agent = ChatAgent()
agent.episodic_memory = EpisodicMemory()  # Add memory

# Or use full V2 base
from gaia.agents.v2.agent import AgentV2

agent = AgentV2(AgentV2Config(enable_all=True))
```

---

## Summary: Clean Transition Path

1. **Week 1-3**: Build Phase 0 in parallel with current Gaia (no disruption)
2. **Week 3**: Publish `gaia==2.0.0-alpha` with opt-in features
3. **Week 4-10**: Migrate existing agents incrementally while building Phase 1
4. **Week 10**: Publish `gaia==2.0.0-beta`, deprecation notices for V1-only features
5. **Week 11-18**: Build Phase 2 on top of stable Phase 1
6. **Week 18**: Publish `gaia==2.0.0`, V1 enters maintenance mode
7. **Week 19+**: V2 becomes default, V1 supported for 6 months

**No big-bang rewrite. Incremental enhancement. Always backward compatible.**

---

*Implementation Roadmap for Gaia V2.*
*Phased approach with clear dependencies, backward compatibility, and incremental value delivery.*
*Start with Phase 0 (3 weeks) for immediate continuous execution capability.*
