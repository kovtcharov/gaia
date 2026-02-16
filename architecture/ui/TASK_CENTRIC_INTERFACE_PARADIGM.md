# Task-Centric Interface Paradigm for AI Agents

**Date**: February 6, 2026
**Version**: 1.0
**Scope**: Rethinking agent interaction from "chat messages" to "task management"
**Problem**: Chat is the wrong abstraction for complex, long-running, multi-phase work

---

## 1. The Problem with Chat-Based Interfaces

### Why Chat Fails for Complex Work

**Current paradigm**: User sends messages, agent responds with messages

```
User: "Build a REST API"
Agent: "I'll create a FastAPI application..."
[Agent works...]
Agent: "Created users endpoint"
Agent: "Created auth endpoint"
Agent: "Running tests"
Agent: "2 tests failed"
Agent: "Fixed auth bug"
Agent: "All tests passing"
Agent: "REST API complete"
```

**Problems**:

1. **No persistent work context** — The "task" exists only as messages in a conversation
2. **No state separation** — Can't pause a build task and ask an unrelated question
3. **No task hierarchy** — Can't see that "Build API" contains sub-tasks like "Implement auth", "Write tests"
4. **No progress tracking** — Have to read the entire conversation to understand where we are
5. **No task reuse** — Can't say "Run that API build task again with different parameters"
6. **No concurrent tasks** — Can't work on frontend while backend builds
7. **Conversation pollution** — Task execution details (tool calls, errors) clutter the chat
8. **Poor resumability** — After restart, just a wall of text to parse
9. **No accountability** — Can't easily audit what the agent did for a specific deliverable

---

## 2. Task-Centric Paradigm

### The Core Shift

**From**: Conversation as the unit of work
**To**: Task as the unit of work, with conversations as communication within tasks

```
Task: "Build E-Commerce Application"
├─ Status: In Progress (67% complete)
├─ Created: Feb 6, 12:00
├─ Started: Feb 6, 12:05
├─ ETA: Feb 6, 16:30
├─ Assigned to: CodeAgent
├─ State: Implementation Mode
├─ Files: 21 created, 11 pending
├─ Tests: 32/45 passing
│
├─ Sub-tasks:
│  ├─ [✓] Design API Architecture (complete)
│  ├─ [✓] Implement User Auth (complete)
│  ├─ [⟳] Implement Cart System (in progress, 70%)
│  ├─ [⧗] Implement Checkout (blocked: needs Stripe key)
│  └─ [⧗] Write Documentation (pending)
│
├─ Conversations (3):
│  ├─ Main (23 messages) — last: "Cart system 70% done"
│  ├─ Debug Session (5 messages) — closed
│  └─ Architecture Review (8 messages) — closed
│
├─ Artifacts (21 files):
│  ├─ backend/api/users.py (✓ complete, 243 LOC)
│  ├─ backend/api/auth.py (✓ complete, 189 LOC)
│  ├─ backend/api/cart.py (⟳ in progress, 156 LOC)
│  └─ ...
│
├─ Decisions (5):
│  ├─ Use JWT for auth (not sessions)
│  ├─ PostgreSQL database (not MongoDB)
│  └─ ...
│
└─ Quality Gates:
   ├─ [✓] Syntax check: All files pass
   ├─ [⚠] Tests: 32/45 passing (71%)
   └─ [⧗] Security scan: Not run yet
```

---

## 3. Task Data Model

### 3.1 Task Structure

```python
@dataclass
class Task:
    """A unit of work with clear deliverables."""

    # Identity
    task_id: str                       # UUID
    title: str                         # "Build E-Commerce Application"
    description: str                   # Full requirements
    type: str                          # "code_generation", "analysis", "refactoring", "research"

    # Lifecycle
    status: str                        # "created", "planned", "in_progress", "paused", "completed", "failed", "cancelled"
    created_at: str                    # ISO timestamp
    started_at: Optional[str]
    completed_at: Optional[str]
    paused_at: Optional[str]

    # Assignment
    assigned_agent: str                # "CodeAgent", "JarvisAgent", etc.
    created_by: str                    # "user", "parent_task", "agent"

    # Progress
    progress_percent: float            # 0.0 to 100.0
    current_phase: str                 # "Planning", "Implementation", "Testing", etc.
    estimated_completion: Optional[str]  # ISO timestamp

    # Hierarchy
    parent_task: Optional[str]         # Task ID
    sub_tasks: List[str]               # Task IDs of children
    dependencies: List[str]            # Task IDs that must complete first

    # Deliverables
    artifacts: List[Artifact]          # Files, reports, analyses produced
    goals: List[Goal]                  # What this task aims to achieve
    completion_criteria: Dict[str, bool]  # {"all_tests_pass": True, ...}

    # Communication
    conversations: List[str]           # Conversation IDs within this task
    active_conversation: Optional[str] # Currently active conversation

    # Context
    manifest_id: Optional[str]         # Project manifest (for code tasks)
    memory_snapshot_id: Optional[str]  # Checkpoint for resume
    state_stack: List[str]             # State machine stack at pause

    # Quality
    quality_score: Optional[float]     # 0.0 to 1.0
    verification_status: Dict[str, str]  # {"tests": "pass", "lint": "3 warnings"}

    # Resources
    tokens_used: int
    cost_usd: float
    duration_minutes: float

    # Metadata
    tags: List[str]
    metadata: Dict[str, Any]
```

### 3.2 Conversation (Within Task)

```python
@dataclass
class Conversation:
    """A focused conversation within a task context."""

    conversation_id: str
    task_id: str                       # Parent task
    title: str                         # "Main discussion", "Debug session", "Architecture review"
    created_at: str
    closed_at: Optional[str]
    status: str                        # "active", "archived"

    messages: List[Message]            # Message history
    purpose: str                       # "task_execution", "debugging", "clarification", "review"

    # Scoping
    focus: Optional[str]               # What this conversation is about (file, component, goal)
```

### 3.3 Artifact

```python
@dataclass
class Artifact:
    """Something produced by the agent."""

    artifact_id: str
    task_id: str
    type: str                          # "file", "report", "analysis", "diagram"
    path: str                          # File path or identifier
    created_at: str
    size_bytes: int
    status: str                        # "draft", "complete", "superseded"
    version: int                       # Versioning for iterative refinement
```

---

## 4. New Interface Model

### 4.1 TUI Layout (Task-Centric)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent                                                           [?][X] │
├────────────────────────────┬───────────────────────────────────────────────┤
│ TASKS                      │ ACTIVE TASK: Build E-Commerce App            │
│                            │ Status: In Progress (67%)  ETA: 45 minutes   │
│ ▶ Build E-Commerce App     ├───────────────────────────────────────────────┤
│   67% │ 1h 30m │ 21/32   │ PHASE: Implementation                         │
│                            │ Current: Writing cart API (3/8 endpoints)     │
│ ▼ Sub-tasks:               │ State: Implementation Mode                    │
│   ✓ Design API (15m)       │ Last action: write_file(api/cart.py) - 0.4s  │
│   ✓ User Auth (45m)        ├───────────────────────────────────────────────┤
│   ⟳ Cart System (30m, 70%) │ DELIVERABLES (21 files)                       │
│   ⧗ Checkout (blocked)     │ ├─ api/users.py ✓ 243 LOC (tests pass)       │
│   ⧗ Documentation          │ ├─ api/auth.py ✓ 189 LOC (tests pass)        │
│                            │ ├─ api/cart.py ⟳ 156 LOC (2 tests failing)   │
│ + Create New Task          │ ├─ models/user.py ✓ 89 LOC                   │
│                            │ └─ ... (17 more files)                        │
│ RECENT TASKS               ├───────────────────────────────────────────────┤
│ ✓ Analyze GPU Trace (2d)   │ CONVERSATIONS (2 active, 1 archived)          │
│ ✓ Debug Auth Bug (3d)      │ [Main] 12 messages - last: "Cart 70% done"   │
│ ✓ Refactor Models (1w)     │ [Debug] 4 messages - "Fixing cart test fail" │
│                            │ [Architecture] Archived - 8 messages          │
│ [Show All Tasks...]        │                                               │
│                            │ [Open Chat][View All Files][Check Quality]    │
├────────────────────────────┴───────────────────────────────────────────────┤
│ [Tab] Switch Pane [Enter] Open Task [+] New [Esc] Menu [Ctrl+Q] Quit      │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Task Detail View

When you select a task:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ TASK: Build E-Commerce Application                              [Close][X] │
├────────────────────────────────────────────────────────────────────────────┤
│ [Overview] [Conversations] [Files] [Quality] [Decisions] [Timeline]        │
├────────────────────────────────────────────────────────────────────────────┤
│ OVERVIEW                                                                   │
│                                                                            │
│ Status: In Progress (67% complete)                                         │
│ Created: Feb 6, 12:00  │  Started: Feb 6, 12:05  │  ETA: Feb 6, 16:30     │
│ Time: 1h 30m elapsed, 45m remaining                                        │
│                                                                            │
│ Assigned Agent: CodeAgent (state: Implementation)                          │
│ Tokens Used: 2.4M  │  Cost: $12.00  │  Quality Score: 8.5/10              │
│                                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                            │
│ SUB-TASKS (4 total, 2 complete, 1 in progress, 1 blocked)                 │
│ ✓ Design API Architecture (15m) - Completed Feb 6, 12:15                  │
│ ✓ Implement User Authentication (45m) - Completed Feb 6, 13:00            │
│ ⟳ Implement Cart System (30m so far, 70% done)                            │
│   └─ Writing API endpoints (3/8 complete)                                 │
│   └─ 2 test failures being debugged                                       │
│ ⧗ Implement Checkout (blocked: waiting for Stripe API key)                │
│                                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                            │
│ COMPLETION CRITERIA                                                        │
│ ✓ All planned files created (21/32 done - 65%)                            │
│ ⚠ All tests passing (32/45 - 71%)                                         │
│ ⧗ Security scan passes (not run yet)                                      │
│ ⧗ Human review approved (pending)                                         │
│                                                                            │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                            │
│ QUICK ACTIONS                                                              │
│ [Resume Task] [Pause Task] [View All Files] [Open Main Chat]              │
│ [Review Quality] [Export Report] [Create Sub-Task] [Edit Criteria]        │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Conversation View (Within Task Context)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ TASK: Build E-Commerce App > CONVERSATION: Main                           │
├────────────────────────────────────────────────────────────────────────────┤
│ Conversation: Main (12 messages)                                           │
│ Purpose: Task execution                                                    │
│ Active since: Feb 6, 12:05 (1h 30m ago)                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│ You (12:05): Build a REST API with user auth, cart, and checkout          │
│                                                                            │
│ Agent (12:06): I'll create a FastAPI application. Starting with planning  │
│ mode to design the architecture.                                          │
│                                                                            │
│ Agent (12:15): Architecture complete. Entering implementation mode.       │
│ [Created file: backend/api/users.py]                                       │
│                                                                            │
│ Agent (12:30): User authentication complete (5 endpoints, 8 tests).       │
│ Moving to cart system.                                                     │
│                                                                            │
│ Agent (14:00): Cart API in progress. 2 tests failing, entering debug mode.│
│ [View debug conversation]                                                  │
│                                                                            │
│ Agent (14:10): Tests fixed. Resuming cart implementation.                 │
│                                                                            │
│ You: How's progress?                                                       │
│                                                                            │
│ Agent (14:15): Cart system is 70% complete (3/8 endpoints done).          │
│ Overall project: 67% complete, ETA 45 minutes.                            │
│ [View detailed progress in Overview tab]                                  │
│                                                                            │
│ You: _                                                                     │
│                                                                            │
├────────────────────────────────────────────────────────────────────────────┤
│ Context: Task "Build E-Commerce App" | Sub-task: "Cart System"            │
│ [Send][Attach File][Switch to Debug Conversation][Close Conversation]     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Benefits of Task-Centric Interface

| Aspect | Chat-Based | Task-Based | Improvement |
|--------|-----------|------------|-------------|
| **Work organization** | Linear message stream | Hierarchical task tree | Clear structure |
| **Progress visibility** | Read all messages | Glance at progress % | Instant understanding |
| **Resumability** | Parse conversation | Load task state | One-click resume |
| **Concurrent work** | Single conversation | Multiple active tasks | Parallel execution |
| **Context switching** | All in one chat | Separate conversations per focus | No pollution |
| **Accountability** | Scattered in messages | Task-specific audit trail | Clear deliverables |
| **Reusability** | Re-describe task | Clone/template task | Efficient repetition |
| **Collaboration** | Hard to hand off | Task metadata explicit | Easy delegation |

---

## 6. Task Management in TUI

### 6.1 Updated TUI Layout

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent                                                           [?][X] │
├────────────────────────────┬───────────────────────────────────────────────┤
│ TASKS (3 active, 12 total) │ TASK DETAIL: Build E-Commerce App             │
│                            │                                               │
│ ACTIVE                     │ Progress: ████████████░░░░ 67%  ETA: 45m     │
│ ▶ Build E-Commerce App     │ Phase: Implementation > Cart System           │
│   67% │ 1h 30m │ ⟳        │ Agent: CodeAgent (running)                    │
│   ├─ Sub-tasks: 2/4 ✓      ├───────────────────────────────────────────────┤
│   └─ Files: 21/32          │ [Artifacts][Conversations][Quality][Timeline] │
│                            ├───────────────────────────────────────────────┤
│ ▶ Analyze Benchmark Data   │ RECENT ACTIVITY (last 5 min)                  │
│   35% │ 45m │ ⟳           │ 14:30:15 write_file api/cart.py          ✓    │
│                            │ 14:30:18 run_tests test_cart.py          ✓    │
│ ▶ Write Documentation      │ 14:30:25 run_tests test_integration      ✗    │
│   10% │ 10m │ ⟳           │ 14:30:26 enter_state debug               ✓    │
│                            │ 14:30:35 edit_file api/cart.py           ✓    │
│ RECENT (completed)         │ 14:30:40 return_from_state               ✓    │
│ ✓ Debug Auth Bug (3h ago)  │ 14:30:45 write_file api/checkout.py      ✓    │
│ ✓ Refactor Models (1d ago) ├───────────────────────────────────────────────┤
│                            │ SUB-TASKS                                     │
│ ARCHIVED (8 tasks)         │ ✓ Design API Architecture                     │
│ [View All...]              │ ✓ Implement User Auth                         │
│                            │ ⟳ Implement Cart System (70%)                 │
│ [+ New Task]               │   └─ 2 test failures in debug                │
│                            │ ⧗ Implement Checkout (blocked)                │
│                            │                                               │
│                            │ [Open Main Chat] [View Files] [Pause Task]    │
├────────────────────────────┴───────────────────────────────────────────────┤
│ [Tab] Next [Enter] Open Task [+] New [/] Search [Esc] Menu [Ctrl+Q] Quit  │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key differences from chat-first TUI**:
- **Left panel = Task List** (not chat history)
- **Right panel = Task Detail** (not just monitoring)
- **Chat is secondary** (accessible via "Open Chat" button within task)
- **Focus on deliverables** (files, goals, quality) not messages

---

## 7. Task Operations

### 7.1 Create Task

```
┌──────────────────────────────────────────────────┐
│ CREATE NEW TASK                                  │
├──────────────────────────────────────────────────┤
│ Title: _                                         │
│                                                  │
│ Description (detailed requirements):             │
│ ┌──────────────────────────────────────────────┐│
│ │                                              ││
│ │                                              ││
│ └──────────────────────────────────────────────┘│
│                                                  │
│ Type: [Code Generation ▼]                       │
│ Agent: [CodeAgent ▼]                             │
│                                                  │
│ Completion Criteria:                             │
│ ☑ All tests pass                                │
│ ☑ All goals complete                            │
│ ☐ Human review required                         │
│ ☑ Quality score ≥ 8.0                           │
│                                                  │
│ [Advanced Options ▼]                             │
│   Parent Task: [None ▼]                         │
│   Dependencies: [Select tasks...]               │
│   Tags: design, api, backend                    │
│                                                  │
│ [Create and Start] [Create Only] [Cancel]       │
└──────────────────────────────────────────────────┘
```

### 7.2 Task Actions

```
Right-click task or press 'a' for actions:

┌─────────────────────────┐
│ TASK ACTIONS            │
├─────────────────────────┤
│ ▶ Resume/Start          │
│ ⏸ Pause                 │
│ ⏹ Cancel                │
│ 📋 Clone Task            │
│ 🔄 Restart from Checkpoint│
│ ✏ Edit Details          │
│ 📊 View Quality Report   │
│ 💬 Open Chat             │
│ 📁 View All Files        │
│ 🏗 View Manifest         │
│ 🔗 Add Dependency        │
│ 🎯 Add Sub-Task          │
│ 🗑 Delete Task           │
└─────────────────────────┘
```

### 7.3 Multi-Task View

See all active tasks at once:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ ALL TASKS                                           3 active, 12 total     │
├────────────────────────────────────────────────────────────────────────────┤
│ Task │ Status │ Progress │ Time │ Agent │ Last Update                      │
│──────┼────────┼──────────┼──────┼───────┼──────────────────────────────────│
│ Build E-Commerce │ Running │ 67% │ 1h 30m │ CodeAgent │ 30s ago: Writing cart API│
│ Analyze Benchmark│ Running │ 35% │ 45m │ JarvisAgent│ 2m ago: Comparing traces │
│ Write Docs │ Paused │ 10% │ 10m │ CodeAgent │ 20m ago: User paused          │
│ Debug Auth │ Complete│ 100%│ 35m │ CodeAgent │ 3h ago: All tests pass       │
│ Refactor Models│ Complete│ 100%│ 2h 15m│ CodeAgent │ 1d ago: Refactor done    │
│                                                                            │
│ [Filter: All ▼] [Sort: Recent ▼] [Group by: Status ▼]                     │
│ [Select task to view details]                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Changes

### 8.1 TaskManager Class

```python
class TaskManager:
    """Manage task lifecycle."""

    def __init__(self, db_path: str = ".gaia/tasks/tasks.db"):
        self.db = sqlite3.connect(db_path)
        self._init_schema()
        self.active_tasks: Dict[str, Task] = {}

    def create_task(
        self,
        title: str,
        description: str,
        task_type: str,
        assigned_agent: str,
        completion_criteria: dict,
        parent_task: Optional[str] = None,
    ) -> Task:
        """Create a new task."""
        timestamp = datetime.now().isoformat()
        task_id = str(uuid.uuid4())[:8]

        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            type=task_type,
            status="created",
            created_at=timestamp,
            assigned_agent=assigned_agent,
            created_by="user",
            progress_percent=0.0,
            current_phase="Not started",
            parent_task=parent_task,
            sub_tasks=[],
            dependencies=[],
            artifacts=[],
            goals=[],
            completion_criteria=completion_criteria,
            conversations=[],
            active_conversation=None,
            tokens_used=0,
            cost_usd=0.0,
            duration_minutes=0.0,
            tags=[],
            metadata={},
        )

        # Persist
        self._save_task(task)

        return task

    def start_task(self, task_id: str, agent: Agent) -> None:
        """Start executing a task."""
        timestamp = datetime.now().isoformat()

        # Update status
        task = self.get_task(task_id)
        task.status = "in_progress"
        task.started_at = timestamp
        self._save_task(task)

        # Create main conversation
        conv_id = self._create_conversation(task_id, "Main", "task_execution")
        task.active_conversation = conv_id
        task.conversations.append(conv_id)

        # Start agent execution
        self.active_tasks[task_id] = task
        agent.continuous_execute_until_complete_for_task(task)

    def pause_task(self, task_id: str) -> None:
        """Pause a running task."""
        timestamp = datetime.now().isoformat()

        task = self.get_task(task_id)
        task.status = "paused"
        task.paused_at = timestamp

        # Snapshot state for resume
        task.memory_snapshot_id = self._create_checkpoint(task)
        self._save_task(task)

    def resume_task(self, task_id: str, agent: Agent) -> None:
        """Resume a paused task."""
        task = self.get_task(task_id)

        # Restore from checkpoint
        if task.memory_snapshot_id:
            agent.resume_from_checkpoint(task.memory_snapshot_id)

        task.status = "in_progress"
        task.paused_at = None
        self._save_task(task)

        agent.continuous_execute_until_complete_for_task(task)

    def complete_task(self, task_id: str) -> None:
        """Mark task as complete."""
        timestamp = datetime.now().isoformat()

        task = self.get_task(task_id)
        task.status = "completed"
        task.completed_at = timestamp
        task.progress_percent = 100.0

        self._save_task(task)

        # Check if completing this task unblocks others
        self._check_unblocked_tasks(task_id)
```

### 8.2 Agent Integration

```python
class TaskAwareAgent(Agent):
    """Agent that executes tasks (not just queries)."""

    def __init__(self, config):
        super().__init__(config)
        self.task_manager = TaskManager()
        self.current_task: Optional[Task] = None

    def continuous_execute_until_complete_for_task(self, task: Task) -> dict:
        """Execute a task to completion."""
        self.current_task = task
        session_id = task.task_id  # Use task ID as session ID

        # Inject task context into prompt
        self._task_context = f"""ACTIVE TASK: {task.title}
Description: {task.description}
Current Phase: {task.current_phase}
Progress: {task.progress_percent:.0f}%
Completion Criteria: {task.completion_criteria}"""

        # Run continuous execution
        result = self.continuous_execute_until_complete(
            task_description=task.description,
            completion_criteria=task.completion_criteria,
        )

        # Mark task complete
        self.task_manager.complete_task(task.task_id)

        return result

    def _compose_system_prompt(self) -> str:
        """Override to include task context."""
        parts = [
            super()._compose_system_prompt(),
        ]

        if self.current_task:
            parts.append("--- CURRENT TASK CONTEXT ---")
            parts.append(self._task_context)

        return "\n\n".join(parts)

    def create_sub_task(
        self,
        title: str,
        description: str,
        parent_task_id: str,
    ) -> Task:
        """Agent can create sub-tasks during execution."""
        return self.task_manager.create_task(
            title=title,
            description=description,
            task_type=self.current_task.type,
            assigned_agent=self.__class__.__name__,
            completion_criteria={},
            parent_task=parent_task_id,
        )
```

---

## 9. Database Schema for Tasks

```sql
-- Tasks table
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    type TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    paused_at TEXT,
    assigned_agent TEXT,
    created_by TEXT,
    progress_percent REAL DEFAULT 0.0,
    current_phase TEXT,
    estimated_completion TEXT,
    parent_task TEXT,
    memory_snapshot_id TEXT,
    tokens_used INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0.0,
    duration_minutes REAL DEFAULT 0.0,
    quality_score REAL,
    metadata_json TEXT,

    FOREIGN KEY (parent_task) REFERENCES tasks(task_id)
);

CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_created ON tasks(created_at DESC);
CREATE INDEX idx_tasks_parent ON tasks(parent_task);

-- Task dependencies
CREATE TABLE task_dependencies (
    task_id TEXT NOT NULL,
    depends_on_task_id TEXT NOT NULL,

    PRIMARY KEY (task_id, depends_on_task_id),
    FOREIGN KEY (task_id) REFERENCES tasks(task_id),
    FOREIGN KEY (depends_on_task_id) REFERENCES tasks(task_id)
);

-- Task conversations
CREATE TABLE task_conversations (
    conversation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    title TEXT,
    purpose TEXT,
    created_at TEXT NOT NULL,
    closed_at TEXT,
    status TEXT,
    message_count INTEGER DEFAULT 0,

    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

-- Conversation messages (linked to task)
CREATE TABLE conversation_messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT,
    content TEXT,
    metadata_json TEXT,

    FOREIGN KEY (conversation_id) REFERENCES task_conversations(conversation_id)
);

-- Task artifacts
CREATE TABLE task_artifacts (
    artifact_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    type TEXT,
    path TEXT,
    created_at TEXT NOT NULL,
    size_bytes INTEGER,
    status TEXT,
    version INTEGER DEFAULT 1,
    metadata_json TEXT,

    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

-- Task goals
CREATE TABLE task_goals (
    goal_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,
    priority INTEGER,
    progress_percent REAL,
    created_at TEXT NOT NULL,
    completed_at TEXT,

    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

-- Task events (audit trail)
CREATE TABLE task_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT,
    details_json TEXT,

    FOREIGN KEY (task_id) REFERENCES tasks(task_id)
);

CREATE INDEX idx_task_events_timestamp ON task_events(timestamp DESC);
```

---

## 10. Task Templates

Reusable task templates:

```python
TASK_TEMPLATES = {
    "web_app_fullstack": {
        "title": "Build Full-Stack Web Application",
        "type": "code_generation",
        "sub_tasks": [
            "Design System Architecture",
            "Setup Project Structure",
            "Implement Backend API",
            "Implement Frontend UI",
            "Write Tests",
            "Setup Deployment",
        ],
        "completion_criteria": {
            "all_tests_pass": True,
            "test_coverage": 0.8,
            "security_scan_pass": True,
            "human_review": True,
        },
    },

    "performance_analysis": {
        "title": "Analyze Performance Trace",
        "type": "analysis",
        "sub_tasks": [
            "Load and validate trace",
            "Compute metrics",
            "Identify bottlenecks",
            "Generate recommendations",
            "Create report",
        ],
        "completion_criteria": {
            "report_generated": True,
            "recommendations_actionable": True,
        },
    },

    "code_refactoring": {
        "title": "Refactor Codebase",
        "type": "refactoring",
        "sub_tasks": [
            "Analyze current code",
            "Identify refactoring targets",
            "Apply refactorings",
            "Run tests",
            "Update documentation",
        ],
        "completion_criteria": {
            "all_tests_pass": True,
            "no_breaking_changes": True,
        },
    },
}

# Usage:
task = task_manager.create_from_template(
    template="web_app_fullstack",
    custom_description="Build e-commerce app with Stripe payments",
)
```

---

## 11. Comparison: Chat vs. Task Interface

### Same Scenario, Different Paradigms

**Chat-based** (current):
```
User: "Build a REST API"
[200 messages of back-and-forth]
[Scroll up to see what's done]
[Search for "test" to find test results]
[Can't pause and ask unrelated question without losing context]
```

**Task-based** (proposed):
```
[Create task "Build REST API"]
[Task runs in background]
[Dashboard shows: 67% complete, 21/32 files, ETA 45 min]
[Click "Open Chat" if you need to communicate]
[Create separate task "Analyze trace" without disrupting first task]
[Both tasks run concurrently with separate contexts]
[Click task to see deliverables, not messages]
```

---

## 12. Integration Across Architecture Documents

This paradigm affects:

### PERSISTENT_MEMORY_FRAMEWORK.md
- **Change**: Sessions → Tasks
- **Add**: Task-scoped memory (episodic memory organized by task, not session)
- **Add**: Cross-task knowledge sharing (patterns from task A inform task B)

### ADAPTIVE_PROMPTS_FRAMEWORK.md
- **Change**: Prompt includes task context (current task, completion criteria, progress)
- **Add**: Task-specific learned instructions

### ARCHITECTURE_MANIFEST_FRAMEWORK.md
- **Change**: Manifest linked to task (one manifest per code generation task)
- **Add**: Task → Manifest → Files relationship

### TUI_DESIGN_SPECIFICATION.md
- **Change**: Primary interface is task list, not chat
- **Change**: Chat becomes secondary (accessible within task context)
- **Add**: Task creation UI, task detail view, multi-task dashboard

### AGENT_DASHBOARD_DESIGN.md
- **Change**: Home page = task dashboard (not chat)
- **Add**: Task management views, task templates, task analytics

---

## 13. Task Queue System

**Your requirement**: "We should be able to queue tasks as the agent is progressing through."

### 13.1 Concurrent Task Queue

```python
class TaskQueue:
    """Manage multiple tasks with priority and dependency scheduling."""

    def __init__(self):
        self.queued: List[Task] = []
        self.active: Dict[str, Task] = {}  # {task_id: task}
        self.max_concurrent: int = 3       # Configurable

    def enqueue(
        self,
        task: Task,
        priority: int = 5,
        start_immediately: bool = False,
    ) -> str:
        """Add task to queue.

        Args:
            task: Task to queue
            priority: 0 (highest) to 10 (lowest)
            start_immediately: Start now if slots available
        """
        task.metadata["priority"] = priority
        task.metadata["queued_at"] = datetime.now().isoformat()

        self.queued.append(task)
        self._sort_queue()  # Re-sort by priority

        if start_immediately and len(self.active) < self.max_concurrent:
            return self._start_next_task()

        return task.task_id

    def _sort_queue(self):
        """Sort queue by priority, then dependencies, then creation time."""
        def sort_key(t: Task):
            priority = t.metadata.get("priority", 5)
            has_deps = len(t.dependencies) > 0
            created = t.created_at
            return (priority, has_deps, created)

        self.queued.sort(key=sort_key)

    def _start_next_task(self) -> Optional[str]:
        """Start the highest-priority task with satisfied dependencies."""
        for task in self.queued:
            # Check dependencies
            if task.dependencies:
                deps_complete = all(
                    self.task_manager.get_task(dep_id).status == "completed"
                    for dep_id in task.dependencies
                )
                if not deps_complete:
                    continue  # Skip, dependencies not ready

            # Start task
            self.queued.remove(task)
            self.active[task.task_id] = task

            # Assign to agent and start
            agent = self._get_agent_for_task(task)
            threading.Thread(
                target=agent.continuous_execute_until_complete_for_task,
                args=(task,),
                daemon=True
            ).start()

            return task.task_id

        return None

    def on_task_complete(self, task_id: str):
        """Called when a task finishes."""
        if task_id in self.active:
            del self.active[task_id]

        # Check if this unblocks any queued tasks
        for queued_task in self.queued:
            if task_id in queued_task.dependencies:
                # Dependency satisfied, check if can start
                if len(self.active) < self.max_concurrent:
                    self._start_next_task()
```

### 13.2 TUI with Task Queue

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent                           3 Active, 2 Queued               [?][X]│
├────────────────────────────┬───────────────────────────────────────────────┤
│ ACTIVE TASKS (3/3 slots)   │ SELECTED: Build E-Commerce App                │
│                            │                                               │
│ ▶ Build E-Commerce App     │ [Details shown in previous examples]          │
│   67% │ 1h 30m │ CodeAgent│                                               │
│                            │                                               │
│ ▶ Analyze Benchmark        │                                               │
│   35% │ 45m │ JarvisAgent │                                               │
│                            │                                               │
│ ▶ Write Documentation      │                                               │
│   10% │ 10m │ CodeAgent   │                                               │
├────────────────────────────┤                                               │
│ QUEUED (2 tasks)           │                                               │
│                            │                                               │
│ [P1] Refactor Database     │                                               │
│      Waiting: E-Commerce   │                                               │
│      complete              │                                               │
│                            │                                               │
│ [P5] Generate API Docs     │                                               │
│      Waiting: slot         │                                               │
│      available             │                                               │
│                            │                                               │
│ [+ Queue New Task]         │                                               │
├────────────────────────────┴───────────────────────────────────────────────┤
│ [Enter] View Task [+] Queue New [p] Pause [r] Resume [d] Dependencies     │
└────────────────────────────────────────────────────────────────────────────┘
```

### 13.3 Quick Queue from Anywhere

```
[Press '+' anywhere in TUI]

┌──────────────────────────────────────────┐
│ QUEUE NEW TASK                           │
├──────────────────────────────────────────┤
│ Title: Refactor database layer_          │
│                                          │
│ Quick mode:                              │
│ ○ Start now (pause current task)        │
│ ● Queue for later (wait for slot)       │
│ ○ Queue with dependency (after task...) │
│                                          │
│ Priority: [Medium ▼]                     │
│ Agent: [CodeAgent ▼]                     │
│                                          │
│ [Queue Task] [Full Form] [Cancel]       │
└──────────────────────────────────────────┘
```

---

## 14. Voice Interface Integration

**Your requirement**: "Incorporate voice (STT and TTS) into this interface."

### 14.1 Voice Interaction Model

**Use cases**:
1. **Hands-free monitoring**: "Jarvis, what's the status of the API build?"
2. **Task creation while coding**: Voice-queue a task without leaving IDE
3. **Progress updates**: Agent speaks status updates while you work on something else
4. **Error notifications**: "Alert: Task 'Build Frontend' failed with test errors"

### 14.2 Voice Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    User (speaking)                          │
└────────────────┬───────────────────────────────────────────┘
                 │ Speech
                 ▼
┌────────────────────────────────────────────────────────────┐
│              STT Engine (Speech-to-Text)                    │
│  Options: Whisper (local), Google STT, Azure STT           │
└────────────────┬───────────────────────────────────────────┘
                 │ Text command
                 ▼
┌────────────────────────────────────────────────────────────┐
│            Voice Command Interpreter                        │
│  Classify: Task creation, status query, control command    │
└────────────────┬───────────────────────────────────────────┘
                 │ Structured command
                 ▼
┌────────────────────────────────────────────────────────────┐
│                  Task Manager / Agent                       │
│  Execute command (create task, query status, pause, etc.)  │
└────────────────┬───────────────────────────────────────────┘
                 │ Response
                 ▼
┌────────────────────────────────────────────────────────────┐
│              TTS Engine (Text-to-Speech)                    │
│  Options: Coqui TTS (local), Google TTS, ElevenLabs        │
└────────────────┬───────────────────────────────────────────┘
                 │ Speech
                 ▼
┌────────────────────────────────────────────────────────────┐
│                    User (hearing)                           │
└────────────────────────────────────────────────────────────┘
```

### 14.3 Voice Command Types

```python
VOICE_COMMANDS = {
    # Task management
    "create_task": {
        "triggers": ["create task", "new task", "queue task", "add task"],
        "parameters": ["title", "description", "priority"],
        "examples": [
            "Create task: Build a REST API with FastAPI",
            "Queue task: Analyze that GPU trace, priority high",
        ],
    },

    # Status queries
    "query_status": {
        "triggers": ["status", "progress", "how's it going", "what's the status"],
        "parameters": ["task_id (optional)"],
        "examples": [
            "What's the status of the API build?",
            "How's progress on all tasks?",
            "Status of task abc123",
        ],
    },

    # Control commands
    "pause_task": {
        "triggers": ["pause", "stop", "hold on"],
        "parameters": ["task_id (optional)"],
        "examples": [
            "Pause the API build task",
            "Pause all tasks",
        ],
    },

    "resume_task": {
        "triggers": ["resume", "continue", "keep going"],
        "parameters": ["task_id (optional)"],
        "examples": [
            "Resume the API build",
            "Continue all paused tasks",
        ],
    },

    # Information queries
    "list_tasks": {
        "triggers": ["list tasks", "show tasks", "what tasks"],
        "examples": ["What tasks are running?", "Show me all tasks"],
    },

    "get_details": {
        "triggers": ["details", "tell me about", "what's in"],
        "parameters": ["task_id or task_name"],
        "examples": [
            "Tell me about the API build task",
            "What's in the cart system sub-task?",
        ],
    },
}
```

### 14.4 Voice-Enabled TUI

Add voice panel to TUI:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent                [🎤 Voice: ON]                             [?][X] │
├────────────────────────────┬───────────────────────────────────────────────┤
│ TASKS                      │ ACTIVE TASK: Build E-Commerce App             │
│                            │                                               │
│ [... task list ...]        │ [... task details ...]                        │
│                            │                                               │
├────────────────────────────┼───────────────────────────────────────────────┤
│ VOICE INTERACTION          │                                               │
│ 🎤 Listening...            │                                               │
│                            │                                               │
│ You (voice): "What's the status of the API build?"                        │
│                                                                            │
│ Agent (voice): "The API build is 67% complete. Cart system is in progress,│
│ 3 out of 8 endpoints done. 2 tests are currently failing but being        │
│ debugged. Estimated 45 minutes until completion."                         │
│                                                                            │
│ [🎤 Push to Talk] [Toggle Auto-Listen] [🔇 Mute Agent] [Voice Settings]    │
├────────────────────────────────────────────────────────────────────────────┤
│ [V] Voice Command [Tab] Next Panel [Esc] Menu                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### 14.5 Voice Integration Code

```python
class VoiceInterface:
    """Voice interface for task-based agent interaction."""

    def __init__(self, config: VoiceConfig):
        # STT (Speech-to-Text)
        if config.stt_engine == "whisper":
            from whisper import load_model
            self.stt_model = load_model(config.whisper_model)
        else:
            self.stt_client = STTClient(config.stt_api_key)

        # TTS (Text-to-Speech)
        if config.tts_engine == "coqui":
            from TTS.api import TTS
            self.tts_model = TTS(config.tts_model)
        else:
            self.tts_client = TTSClient(config.tts_api_key)

        # Audio capture
        import sounddevice as sd
        self.sample_rate = 16000
        self.is_listening = False

    def listen(self, duration_sec: int = 5) -> str:
        """Capture audio and convert to text."""
        import numpy as np

        # Record audio
        audio = sd.rec(
            int(duration_sec * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1
        )
        sd.wait()

        # Convert to text
        if hasattr(self, 'stt_model'):
            # Local Whisper
            result = self.stt_model.transcribe(audio)
            return result["text"]
        else:
            # Cloud STT
            return self.stt_client.transcribe(audio)

    def speak(self, text: str) -> None:
        """Convert text to speech and play."""
        if hasattr(self, 'tts_model'):
            # Local Coqui TTS
            wav = self.tts_model.tts(text)
            sd.play(wav, self.sample_rate)
            sd.wait()
        else:
            # Cloud TTS
            audio = self.tts_client.synthesize(text)
            sd.play(audio, self.sample_rate)
            sd.wait()

    def parse_voice_command(self, text: str) -> dict:
        """Parse natural language voice command into structured action."""
        text_lower = text.lower()

        # Task creation
        if any(trigger in text_lower for trigger in ["create task", "new task", "queue task"]):
            # Extract title (everything after trigger)
            for trigger in ["create task", "new task", "queue task"]:
                if trigger in text_lower:
                    title = text[text_lower.index(trigger) + len(trigger):].strip()
                    return {
                        "action": "create_task",
                        "title": title.rstrip(".,"),
                        "priority": self._extract_priority(text_lower),
                    }

        # Status query
        if any(trigger in text_lower for trigger in ["status", "progress", "how's it going"]):
            task_name = self._extract_task_reference(text)
            return {
                "action": "query_status",
                "task": task_name,
            }

        # Control
        if "pause" in text_lower:
            return {"action": "pause_task", "task": self._extract_task_reference(text)}
        if "resume" in text_lower or "continue" in text_lower:
            return {"action": "resume_task", "task": self._extract_task_reference(text)}

        # Fallback: general query
        return {"action": "query", "text": text}

    def _extract_priority(self, text: str) -> int:
        """Extract priority from voice command."""
        if "urgent" in text or "high priority" in text:
            return 0
        elif "low priority" in text:
            return 8
        else:
            return 5  # Default
```

### 14.6 Voice in TUI (Textual Integration)

```python
class VoicePanel(Container):
    """Voice interaction panel in TUI."""

    def __init__(self, voice_interface: VoiceInterface, task_manager: TaskManager):
        super().__init__()
        self.voice = voice_interface
        self.task_manager = task_manager
        self.listening = False

    def compose(self) -> ComposeResult:
        yield Static("VOICE INTERACTION", classes="panel-title")
        yield Label("🎤 Press Space to talk, or say wake word", id="voice-status")
        yield RichLog(id="voice-transcript", max_lines=10)
        yield Button("🎤 Push to Talk", id="ptt-button")
        yield Button("🔇 Mute Agent Voice", id="mute-button")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "ptt-button":
            self.start_listening()

    def start_listening(self):
        """Start voice capture."""
        self.listening = True
        self.query_one("#voice-status", Label).update("🎤 Listening...")

        # Capture in background thread
        self.run_worker(self._listen_and_process, exclusive=False)

    @work(thread=True)
    async def _listen_and_process(self):
        """Listen for voice command and process."""
        # Record audio and transcribe
        text = self.voice.listen(duration_sec=5)

        # Show transcription
        self.call_from_thread(
            self.query_one("#voice-transcript", RichLog).write,
            f"You: {text}"
        )

        # Parse command
        command = self.voice.parse_voice_command(text)

        # Execute
        if command["action"] == "create_task":
            task = self.task_manager.create_task(
                title=command["title"],
                description=command["title"],  # Use LLM to expand later
                task_type="general",
                assigned_agent="CodeAgent",
                completion_criteria={},
            )
            response = f"Task '{command['title']}' queued with priority {command.get('priority', 5)}"

        elif command["action"] == "query_status":
            task = self.task_manager.find_task_by_name(command.get("task", ""))
            if task:
                response = f"Task '{task.title}' is {task.progress_percent:.0f}% complete, currently in {task.current_phase} phase, estimated {task.estimated_completion or 'unknown'} time remaining"
            else:
                # All tasks
                active = self.task_manager.get_active_tasks()
                response = f"You have {len(active)} active tasks. Overall progress: {self._compute_avg_progress(active):.0f}%"

        else:
            response = "Command not recognized"

        # Speak response
        self.call_from_thread(
            self.query_one("#voice-transcript", RichLog).write,
            f"Agent: {response}"
        )

        self.voice.speak(response)

        # Reset listening state
        self.call_from_thread(
            self.query_one("#voice-status", Label).update,
            "🎤 Ready (press Space)"
        )
        self.listening = False
```

### 14.7 Voice Notifications

Agent proactively speaks important updates:

```python
class VoiceNotificationSystem:
    """Proactive voice notifications during task execution."""

    def __init__(self, voice: VoiceInterface, config: VoiceNotificationConfig):
        self.voice = voice
        self.config = config
        self.notification_queue = []

    def on_agent_event(self, event: dict):
        """React to agent events, speak if important."""
        if not self.config.enable_notifications:
            return

        event_type = event["type"]
        data = event["data"]

        # Task complete
        if event_type == "task_complete":
            if self.config.notify_on_task_complete:
                self.speak(f"Task '{data['task_title']}' complete")

        # Task error
        elif event_type == "task_error":
            if self.config.notify_on_errors:
                self.speak(f"Error in task '{data['task_title']}': {data['error'][:50]}")

        # Quality gate failure
        elif event_type == "quality_gate_failed":
            if self.config.notify_on_quality_issues:
                self.speak(f"Quality gate failed: {data['gate']} - {data['issue']}")

        # Progress milestones
        elif event_type == "progress_update":
            if self.config.notify_on_milestones:
                percent = data["percent"]
                # Speak at 25%, 50%, 75%, 100%
                if percent in [25, 50, 75, 100]:
                    self.speak(f"Task {data['task_title']} is {percent}% complete")

    def speak(self, text: str):
        """Queue and speak notification (non-blocking)."""
        self.notification_queue.append(text)
        threading.Thread(target=self._speak_queued, daemon=True).start()

    def _speak_queued(self):
        """Speak all queued notifications."""
        while self.notification_queue:
            text = self.notification_queue.pop(0)
            self.voice.speak(text)
```

### 14.8 Wake Word Detection

Hands-free activation:

```python
class WakeWordDetector:
    """Detect wake word for hands-free activation."""

    def __init__(self, wake_word: str = "hey gaia"):
        self.wake_word = wake_word.lower()
        self.is_active = False

    def start_listening_for_wake_word(self):
        """Continuously listen for wake word in background."""
        import pvporcupine  # Porcupine wake word detection

        porcupine = pvporcupine.create(keywords=[self.wake_word])

        while self.is_active:
            # Capture short audio chunk
            audio_chunk = self._capture_audio_chunk()

            # Detect wake word
            keyword_index = porcupine.process(audio_chunk)

            if keyword_index >= 0:
                # Wake word detected!
                self.on_wake_word_detected()

    def on_wake_word_detected(self):
        """Wake word heard, start full voice capture."""
        # Play confirmation sound (beep)
        self._play_beep()

        # Trigger voice interface to listen for command
        # (integrated with TUI voice panel)
```

---

## 15. Voice-Enabled Task Queue Workflow

Complete example:

```
[Agent is building REST API task, 50% complete]

You (voice): "Hey Gaia, queue a new task"
Agent (voice + beep): "Listening for new task description"

You (voice): "Analyze the GPU trace from last night's benchmark, high priority"
Agent (voice): "Task 'Analyze GPU trace from last night's benchmark' queued with high priority. Starting as soon as a slot is available."

[TUI updates: Task appears in queue]

[API build task continues...]

[10 minutes later]

Agent (voice, unprompted): "Task 'Build REST API' complete. All tests passing. Starting queued task 'Analyze GPU trace'."

You (voice): "What's the status?"
Agent (voice): "GPU trace analysis is 15% complete. Currently loading trace data. One active task, no queued tasks."

You (voice): "Pause that and build the frontend instead"
Agent (voice): "GPU trace analysis paused. Would you like to create a new task for building the frontend?"

You (voice): "Yes, build a React frontend with user auth and cart views"
Agent (voice): "Task 'Build React frontend with user auth and cart views' created and starting now. GPU trace analysis remains paused in queue."
```

---

## 16. Configuration

```python
@dataclass
class VoiceConfig:
    """Voice interface configuration."""

    # STT (Speech-to-Text)
    enable_stt: bool = True
    stt_engine: str = "whisper"        # "whisper", "google", "azure"
    whisper_model: str = "base"        # "tiny", "base", "small", "medium", "large"
    stt_language: str = "en"
    stt_api_key: Optional[str] = None

    # TTS (Text-to-Speech)
    enable_tts: bool = True
    tts_engine: str = "coqui"          # "coqui", "google", "elevenlabs"
    tts_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    tts_voice: str = "default"
    tts_api_key: Optional[str] = None

    # Wake word
    enable_wake_word: bool = False
    wake_word: str = "hey gaia"

    # Notifications
    enable_notifications: bool = True
    notify_on_task_complete: bool = True
    notify_on_errors: bool = True
    notify_on_quality_issues: bool = True
    notify_on_milestones: bool = True  # 25%, 50%, 75%, 100%

    # Audio
    sample_rate: int = 16000
    voice_activation_threshold: float = 0.5
    silence_timeout_sec: int = 2

    # Behavior
    auto_listen: bool = False          # Continuous listening vs push-to-talk
    confirm_commands: bool = True      # Speak back command before executing
```

---

## 17. Updated TUI with Task Queue + Voice

Full TUI layout with all features:

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent              [🎤 Listening] [3 Active, 2 Queued]          [?][X] │
├────────────────────────────┬───────────────────────────────────────────────┤
│ ACTIVE TASKS (3/3 slots)   │ TASK: Build E-Commerce App                    │
│                            │ ████████████░░░░ 67%  ETA: 45m  [🔊Details]   │
│ ▶ Build E-Commerce App     ├───────────────────────────────────────────────┤
│   67% │ 1h 30m │ ⟳         │ Phase: Implementation > Cart System           │
│   [View]                   │ Current: Writing cart endpoints (3/8)         │
│                            │                                               │
│ ▶ Analyze GPU Trace        │ Sub-tasks: 2/4 complete                       │
│   35% │ 45m │ ⟳            │ ✓ Design Architecture                         │
│   [View]                   │ ✓ User Authentication                         │
│                            │ ⟳ Cart System (70% - debugging 2 test fails)  │
│ ▶ Write API Docs           │ ⧗ Checkout (blocked: Stripe key)              │
│   10% │ 10m │ ⟳            │                                               │
│   [View]                   │ Files: 21 created (18 ✓, 1 ⟳, 2 ⧗)           │
├────────────────────────────┤ Tests: 32/45 passing (71%)                    │
│ QUEUED (2 tasks)           │                                               │
│                            │ [Open Chat][View Files][Quality Report]       │
│ [P0] Refactor DB Layer     ├───────────────────────────────────────────────┤
│      Depends: E-Commerce ✓ │ 🎤 VOICE                                      │
│      [Start Now][Edit]     │ You: "What's the status?"                     │
│                            │ Agent: "API build is 67% complete, cart       │
│ [P5] Security Scan         │ system in progress, 45 minutes remaining."    │
│      Waiting: slot         │                                               │
│      [Edit][Remove]        │ [🎤 Tap Space] [Toggle Voice] [Settings]      │
│                            │                                               │
│ [+ Queue Task (or say it)] │                                               │
├────────────────────────────┴───────────────────────────────────────────────┤
│ [Space] Voice [Tab] Next [Enter] View Task [+] Queue [Ctrl+Q] Quit        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 18. Migration Path

**Phase 1**: Add task layer **on top of** existing chat interface
- Tasks are created automatically from conversations
- Each conversation = one task
- Backward compatible

**Phase 2**: Make tasks explicit
- Users create tasks before chatting
- Chat becomes sub-feature of task
- Add task queue

**Phase 3**: Add voice
- STT for task creation and queries
- TTS for status updates
- Wake word detection (optional)

**Phase 4**: Advanced task features
- Templates, dependencies, concurrent tasks
- Task analytics, task history

---

*Task-Centric Interface Paradigm for Gaia Agent SDK.*
*Shifts from "conversation as work unit" to "task as work unit" with queue management and voice interface integration for hands-free operation during long-running continuous execution.*

