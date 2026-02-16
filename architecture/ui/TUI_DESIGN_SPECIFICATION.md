# Terminal User Interface (TUI) Design Specification for Gaia CLI

**Date**: February 6, 2026
**Version**: 1.0
**Scope**: Interactive terminal interface for Gaia agents with real-time monitoring, memory browsing, and state visualization
**Foundation**: AMD Gaia Agent SDK 0.15.3+ with all 7 architectural frameworks
**Technology**: Textual framework (Python TUI library)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Layout Design](#3-layout-design)
4. [Core Panels](#4-core-panels)
5. [Real-Time Data Binding](#5-real-time-data-binding)
6. [Navigation & Keyboard Shortcuts](#6-navigation--keyboard-shortcuts)
7. [Visual Design](#7-visual-design)
8. [Integration with Agent Frameworks](#8-integration-with-agent-frameworks)
9. [Performance Optimization](#9-performance-optimization)
10. [Implementation with Textual](#10-implementation-with-textual)
11. [Configuration](#11-configuration)
12. [Appendix: Complete Code](#12-appendix-complete-code)

---

## 1. Problem Statement

### Current Gaia CLI Experience

```
$ gaia-agent query "Build a REST API"
[Agent thinking...]
[Agent working...]
[Many lines of output...]
[2 hours later...]
Done.
```

**Problems**:
- **No visibility** during execution (what is it doing RIGHT NOW?)
- **No progress indication** for continuous execution (0%? 50%? 90%?)
- **Can't inspect state** (which state? what tools available? what's in memory?)
- **Can't browse memory** without stopping and running separate commands
- **No real-time debugging** (which tool just failed? what was the error?)
- **Lost in output** (hundreds of lines scroll by, can't find relevant info)
- **Can't pause/resume** interactive control during long runs
- **No project overview** (what files exist? what's pending? dependencies?)

### What a TUI Provides

```
┌──────────────────────────────────────────────────────────────────────┐
│ ✓ Live progress bar                                                  │
│ ✓ Real-time tool call log                                           │
│ ✓ Current state visualization (Planning → Implementation → Debug)   │
│ ✓ Memory browser (navigate episodic/semantic/universal DB)          │
│ ✓ Manifest tree view (files, dependencies, goals)                   │
│ ✓ Interactive controls (pause, resume, enter state, create skill)   │
│ ✓ Split-pane: chat on left, monitoring on right                     │
│ ✓ Searchable logs, filterable views                                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Architecture Overview

### High-Level Design

```
┌──────────────────────────────────────────────────────────────┐
│                     Textual TUI Application                   │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ ChatPanel  │  │ StatePanel │  │MemoryPanel │            │
│  │ (Messages) │  │ (Live Info)│  │ (Browser)  │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
│         │                │                │                   │
│         └────────────────┴────────────────┘                   │
│                          │                                    │
│                    ┌─────▼─────┐                             │
│                    │  AppState  │                             │
│                    │ (Reactive) │                             │
│                    └─────┬─────┘                             │
└──────────────────────────┼──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │AgentConnector│ (Background thread)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Gaia Agent  │
                    │ + All       │
                    │ Frameworks  │
                    └─────────────┘
```

**Key principles**:
1. **Non-blocking UI**: Agent runs in background thread, UI updates via reactive state
2. **Event-driven**: Agent emits events (tool_call, state_change, progress), TUI consumes
3. **Modular panels**: Each panel is independent Textual widget
4. **Keyboard-first**: All features accessible via keyboard (mouse optional)
5. **Responsive**: Layout adapts to terminal size

---

## 3. Layout Design

### 3.1 Default Layout (Wide Terminal ≥120 cols)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent                    [Planning Mode]        Session: abc123  [?][X] │
├───────────────────────────────┬────────────────────────────────────────────┤
│                               │ STATE & PROGRESS                           │
│  CHAT                         │ State: Implementation (145 steps)          │
│                               │ Progress: ████████████░░░░ 67% (21/32)    │
│                               │ Time: 2h 15m  Tokens: 2.4M  Cost: $12.20  │
│  User: Build a REST API       ├────────────────────────────────────────────┤
│                               │ RECENT TOOLS (last 10)                     │
│  Agent: I'll create a         │ 14:30:15 write_file api/users.py    ✓ 450ms│
│  FastAPI application with...  │ 14:30:18 run_tests test_users.py    ✓ 890ms│
│                               │ 14:30:25 run_tests test_auth.py     ✗ 1.2s │
│  [Status: Writing auth        │ 14:30:26 enter_state debug          ✓ 10ms │
│  endpoints, 3/8 complete]     │ 14:30:30 inspect_error ...          ✓ 120ms│
│                               │ 14:30:35 edit_file api/auth.py      ✓ 380ms│
│  Agent: Tests passing for     ├────────────────────────────────────────────┤
│  users and auth. Moving to    │ MANIFEST TREE         Goals: 5/8 complete │
│  cart system...               │ ├─ backend/                                │
│                               │ │  ├─ api/                                 │
│  User: _                      │ │  │  ├─ users.py ✓ (243 LOC)              │
│                               │ │  │  ├─ auth.py ⟳ (testing)               │
│                               │ │  │  ├─ cart.py ⧗ (pending)               │
│                               │ │  │  └─ checkout.py ⧗                     │
│                               │ │  ├─ models/                              │
│                               │ │  │  ├─ user.py ✓                         │
│                               │ │  │  ├─ cart.py ✓                         │
│                               │ │  └─ database.py ✓                        │
│                               │ └─ tests/ (28/45 passing ⚠)               │
│                               │                                            │
├───────────────────────────────┴────────────────────────────────────────────┤
│ [Tab] Next Panel  [F1] Memory  [F2] Tools  [F3] States  [F4] Logs  [Esc] Menu│
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Narrow Terminal (<120 cols)

Auto-collapses to single-panel with tab switching:

```
┌──────────────────────────────────────────────────────┐
│ Gaia Agent  [Chat][State][Memory][Tools]     [?][X]  │
├──────────────────────────────────────────────────────┤
│  Active Tab: CHAT                                    │
│                                                      │
│  User: Build a REST API                             │
│                                                      │
│  Agent: I'll create a FastAPI application...        │
│                                                      │
│  [Status: Writing auth endpoints, 3/8 complete]     │
│                                                      │
│                                                      │
│  User: _                                             │
│                                                      │
├──────────────────────────────────────────────────────┤
│ Progress: 67% │ State: Implementation │ Step: 145   │
├──────────────────────────────────────────────────────┤
│ [Tab] Switch  [Enter] Send  [Ctrl+C] Pause  [Esc] Menu│
└──────────────────────────────────────────────────────┘
```

---

## 4. Core Panels

### 4.1 Chat Panel (Primary)

**Purpose**: Main conversation interface

**Features**:
- Message list (scrollable, auto-scroll to bottom)
- Input field with multi-line support
- Syntax highlighting for code in messages
- Tool call indicators (show which tools were invoked)
- Thinking indicators (show agent reasoning if available)
- Message timestamps
- Copy message (keyboard shortcut)
- Search messages (Ctrl+F)

**Textual widgets**:
```python
class ChatPanel(ScrollableContainer):
    def __init__(self):
        super().__init__()
        self.messages: List[Message] = []
        self.input_field = TextArea(placeholder="Ask me anything...")
        self.auto_scroll = True

    def add_message(self, role: str, content: str, timestamp: str):
        """Add message to chat (reactive update)."""
        msg = Message(role, content, timestamp)
        self.messages.append(msg)

        # Render with syntax highlighting if code block
        if "```" in content:
            msg_widget = CodeMessage(msg)
        else:
            msg_widget = TextMessage(msg)

        self.mount(msg_widget)

        if self.auto_scroll:
            self.scroll_end(animate=False)

    def add_tool_call_indicator(self, tool_name: str, status: str):
        """Show tool invocation inline."""
        indicator = ToolCallIndicator(tool_name, status)
        self.mount(indicator)
```

### 4.2 State & Progress Panel

**Purpose**: Real-time execution monitoring

**Sections**:
1. **Current State**: Which state agent is in, stack depth, steps in state
2. **Progress Bar**: Overall task progress with percentage
3. **Metrics**: Time elapsed, tokens used, estimated time remaining
4. **Recent Tools**: Last 10 tool calls with status (✓/✗) and duration
5. **Active Goals**: Which goals are in-progress

**Textual widgets**:
```python
class StateProgressPanel(Container):
    def __init__(self):
        super().__init__()

        # State indicator
        self.state_label = Label("STATE: --")
        self.state_stack_label = Label("Stack: []")

        # Progress
        self.progress_bar = ProgressBar(total=100)
        self.progress_label = Label("Progress: 0%")

        # Metrics
        self.metrics_table = DataTable()
        self.metrics_table.add_columns("Metric", "Value")

        # Recent tools (live log)
        self.tool_log = RichLog(max_lines=10, highlight=True)

    def update_state(self, state_id: str, stack: List[str]):
        """Update state display (reactive)."""
        self.state_label.update(f"STATE: {state_id}")
        self.state_stack_label.update(f"Stack: {' ← '.join(stack)}")

    def update_progress(self, percent: float, completed: int, total: int):
        """Update progress bar."""
        self.progress_bar.update(progress=percent)
        self.progress_label.update(f"Progress: {percent:.0f}% ({completed}/{total})")

    def add_tool_call(self, tool: str, status: str, duration_ms: int, timestamp: str):
        """Add tool call to log (auto-scrolls)."""
        icon = "✓" if status == "success" else "✗"
        self.tool_log.write(f"[{timestamp}] {icon} {tool} - {duration_ms}ms")
```

### 4.3 Memory Browser Panel

**Purpose**: Explore episodic, semantic, and universal memory

**Tabs within panel**:
- **Episodic** — Past sessions timeline
- **Semantic** — Knowledge base table
- **Universal** — All interactions log

**Episodic tab**:
```
┌──────────────────────────────────────────────────────────┐
│ MEMORY > Episodic                            [Search: __] │
├──────────────────────────────────────────────────────────┤
│ Timeline (last 30 days)                                  │
│                                                          │
│ Feb 6, 14:30 │ Session abc123 (2h 15m)                  │
│              │ "Build REST API"                         │
│              │ 21 files created, 28 tests passing       │
│              │ [View Details]                           │
│                                                          │
│ Feb 5, 09:15 │ Session def456 (45m)                     │
│              │ "Debug authentication issue"             │
│              │ 3 files modified, issue resolved         │
│              │ [View Details]                           │
│                                                          │
│ Feb 4, 16:00 │ Session ghi789 (1h 30m)                  │
│              │ "Analyze GPU trace"                      │
│              │ MI355X, DeepSeek, Expert imbalance found │
│              │ [View Details]                           │
│                                                          │
│ [Load More...]                                           │
└──────────────────────────────────────────────────────────┘
```

**Semantic tab** (knowledge table):
```
┌──────────────────────────────────────────────────────────────────────┐
│ MEMORY > Semantic                Filter: [All▼] Confidence: [>0.5▼]  │
├──────────────────────────────────────────────────────────────────────┤
│ ID │ Category │ Pattern │ Confidence │ Evidence │ Last Confirmed    │
│────┼──────────┼─────────┼────────────┼──────────┼──────────────────│
│ hw │ Hardware │ MI355X allreduce < 50% │ 0.92 │ 23 │ Feb 6, 14:30   │
│ opt│ Optimize │ Expert imbalance fix   │ 0.85 │  8 │ Feb 5, 11:20   │
│ cor│ Correct  │ Flash attn uses hipCK  │ 1.00 │  1 │ Feb 4, 09:00   │
│                                                                      │
│ [Selected: hw-mi355x-allreduce]                                      │
│ Pattern: MI355X allreduce consistently achieves 40-50% bandwidth     │
│          for messages < 256KB                                        │
│ Recommendation: Normal behavior, don't flag as anomaly              │
│ Evidence: 23 sessions from Jan 15 - Feb 6                           │
│ [Edit] [Delete] [View Sources]                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Textual implementation**:
```python
class MemoryBrowser(TabbedContent):
    def __init__(self, memory_db):
        super().__init__()
        self.memory_db = memory_db

        # Episodic tab
        self.episodic_tab = EpisodicMemoryView(memory_db.episodic)
        self.add_pane("episodic", "Episodic", self.episodic_tab)

        # Semantic tab
        self.semantic_tab = SemanticMemoryView(memory_db.semantic)
        self.add_pane("semantic", "Semantic", self.semantic_tab)

        # Universal tab
        self.universal_tab = UniversalDBView(memory_db.universal)
        self.add_pane("universal", "Universal", self.universal_tab)

class SemanticMemoryView(Container):
    def __init__(self, semantic_memory):
        super().__init__()
        self.memory = semantic_memory

        # Table
        self.table = DataTable()
        self.table.add_columns("ID", "Category", "Pattern", "Confidence", "Evidence", "Last")

        # Search/filter
        self.search_input = Input(placeholder="Search knowledge...")
        self.category_filter = Select(options=["All", "Hardware", "Optimization", "Correction"])

        # Detail view
        self.detail_view = Static("")

    def on_mount(self):
        """Load initial data when panel mounted."""
        self.refresh_table()

    def refresh_table(self):
        """Query memory and update table."""
        knowledge = self.memory.search(
            category=self.category_filter.value if self.category_filter.value != "All" else None,
            min_confidence=0.5,
        )

        self.table.clear()
        for k in knowledge:
            self.table.add_row(
                k["id"][:8],
                k["category"],
                k["pattern"][:40],
                f"{k['confidence']:.2f}",
                str(k["evidence_count"]),
                k["last_confirmed"][:10],
            )
```

### 4.4 Manifest Tree Panel

**Purpose**: Visual project structure with real-time updates

```
┌──────────────────────────────────────────────────────────┐
│ MANIFEST                        Progress: 67% (21/32)    │
├──────────────────────────────────────────────────────────┤
│ Project: REST API                                        │
│ Created: Feb 6, 12:00  │  Updated: Feb 6, 14:45 (live)  │
│                                                          │
│ [Tree View] [Dependency Graph] [Goals]                  │
│                                                          │
│ ▼ backend/                                              │
│   ▼ api/                                                │
│     ├─ users.py ✓ 243 LOC (2m ago)                      │
│     ├─ auth.py ⟳ Testing (30s ago)                      │
│     ├─ cart.py ⧗ Pending                                │
│     └─ checkout.py ⧗ Pending                            │
│   ▼ models/                                             │
│     ├─ user.py ✓ 89 LOC                                 │
│     ├─ cart.py ✓ 65 LOC                                 │
│     └─ product.py ✓ 102 LOC                             │
│   └─ database.py ✓ 45 LOC                               │
│ ▼ tests/                                                │
│   ├─ test_users.py ✓ 5/5 passing                       │
│   ├─ test_auth.py ⚠ 3/5 passing (2 failing)            │
│   └─ test_cart.py ⧗ Not run                            │
│ ▼ config/                                               │
│   └─ settings.py ✓                                      │
│                                                          │
│ [Selected: api/auth.py]                                 │
│ Status: Testing (entered debug mode 30s ago)           │
│ Dependencies: models/user.py, database.py, auth_utils.py│
│ Dependents: test_auth.py                               │
└──────────────────────────────────────────────────────────┘
```

**Textual implementation**:
```python
class ManifestTreePanel(Container):
    def __init__(self, manifest_tracker):
        super().__init__()
        self.manifest = manifest_tracker

        # Tree widget
        self.tree = Tree("Project Root")
        self.tree.show_root = True

        # Detail view
        self.detail_label = Label("")

        # Auto-refresh every 2 seconds
        self.set_interval(2.0, self.refresh_tree)

    def refresh_tree(self):
        """Update tree from manifest (reactive)."""
        # Clear and rebuild
        self.tree.clear()

        files = self.manifest.get_all_files()

        # Build tree structure
        file_tree = {}
        for f in files:
            parts = f.path.split("/")
            current = file_tree
            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        # Render tree
        self._render_tree_recursive(self.tree.root, file_tree, files)

    def _render_tree_recursive(self, parent_node, tree_dict, files):
        """Recursively render tree structure."""
        for name, subtree in tree_dict.items():
            # Find file object
            file_obj = next((f for f in files if f.path.endswith(name)), None)

            if file_obj:
                # Leaf node (file)
                icon = self._get_status_icon(file_obj.status)
                label = f"{icon} {name}"
                if file_obj.lines_of_code:
                    label += f" {file_obj.lines_of_code} LOC"
                node = parent_node.add_leaf(label)
            else:
                # Directory node
                node = parent_node.add(name)
                if subtree:
                    self._render_tree_recursive(node, subtree, files)

    def _get_status_icon(self, status: str) -> str:
        return {
            "complete": "✓",
            "in_progress": "⟳",
            "failing": "✗",
            "planned": "⧗",
        }.get(status, "·")
```

### 4.5 Tools Panel

**Purpose**: Inspect tool registry, test tools, view metrics

```
┌──────────────────────────────────────────────────────────────┐
│ TOOLS                        [Static][Dynamic][Skills]        │
├──────────────────────────────────────────────────────────────┤
│ Active Tools: 23 (15 static, 5 dynamic, 3 skills)            │
│                                                              │
│ Name │ Type │ Success Rate │ Avg Time │ Last Used │ Actions │
│──────┼──────┼──────────────┼──────────┼───────────┼─────────│
│ write_file │ Static │ 98% │ 0.3s │ 30s ago │ [Test][Docs]│
│ run_tests  │ Static │ 95% │ 1.2s │ 45s ago │ [Test][Docs]│
│ analyze_bundle │ Dynamic │ 87% │ 2.1s │ 2d ago │ [Edit][Delete]│
│ optimize-bundle│ Skill │ 100% │ 8.5s │ 1w ago │ [View][Invoke]│
│                                                              │
│ [Selected: optimize-bundle (Skill)]                          │
│ Description: Analyze bundle size and suggest optimizations  │
│ Trigger Patterns: "optimize bundle", "reduce size"          │
│ Workflow: analyze_bundle → find_large_deps → suggest_alts  │
│ Created: Jan 28 by agent │ Used: 5 times │ Success: 100%   │
│ [Invoke Now] [View Code] [Delete Skill]                     │
└──────────────────────────────────────────────────────────────┘
```

### 4.6 Logs Panel

**Purpose**: Detailed execution trace

**Features**:
- Searchable log (filter by level, tool name, timestamp)
- Copy log lines
- Export to file
- Auto-scroll toggle
- Color-coded by level (error=red, warning=yellow, info=white, debug=dim)

```
┌──────────────────────────────────────────────────────────────┐
│ LOGS                    Filter: [All Levels▼] Search: [__]   │
├──────────────────────────────────────────────────────────────┤
│ 14:30:15.234 INFO  Entering state: implementation           │
│ 14:30:16.102 DEBUG Tool call: write_file(api/users.py)      │
│ 14:30:16.552 INFO  File created: api/users.py (243 lines)   │
│ 14:30:18.023 DEBUG Tool call: run_tests(test_users.py)      │
│ 14:30:18.913 INFO  Tests passed: 5/5                        │
│ 14:30:25.441 DEBUG Tool call: run_tests(test_auth.py)       │
│ 14:30:26.632 ERROR Tests failed: 2/5                        │
│ 14:30:26.633 WARN  Entering debug mode (reason: test_fail)  │
│ 14:30:27.104 DEBUG Tool call: inspect_error(...)            │
│ 14:30:27.224 INFO  Error cause: Missing user validation     │
│ 14:30:30.558 DEBUG Tool call: edit_file(api/auth.py)        │
│ 14:30:30.938 INFO  Fix applied, re-running tests...         │
│                                                              │
│ [Auto-scroll: ON]  [Copy Selection]  [Export Logs]          │
└──────────────────────────────────────────────────────────────┘
```

### 4.7 State Visualizer Panel

**Purpose**: Understand state machine status

```
┌──────────────────────────────────────────────────────────────┐
│ STATE MACHINE                                                │
├──────────────────────────────────────────────────────────────┤
│ Current State: IMPLEMENTATION                                │
│ Stack Depth: 2 (Planning ← Implementation)                   │
│ Steps in State: 85 / unlimited                               │
│                                                              │
│ Available Tools in This State (15):                          │
│ ✓ write_file      ✓ edit_file       ✓ run_tests            │
│ ✓ update_manifest ✓ enter_state     ✓ return_from_state    │
│ ✗ delete_database (disabled in implementation mode)         │
│                                                              │
│ System Prompt Preview (2,450 tokens):                        │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ [Immutable Core] You are an AI agent...                │  │
│ │ [Implementation Mode] Write clean, tested code...      │  │
│ │ [Learned Instructions] - Use snake_case for funcs...   │  │
│ │ [Memory Context] Relevant knowledge: MI355X...         │  │
│ │ [Tools] 15 tools available...                          │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ State Transition History:                                    │
│ 12:00:00 START → Planning                                    │
│ 12:15:30 Planning → Implementation (plan approved)           │
│                                                              │
│ [Enter Different State] [Return to Previous] [View All States]│
└──────────────────────────────────────────────────────────────┘
```

---

## 5. Real-Time Data Binding

### 5.1 Event Stream from Agent

Agent emits events that TUI consumes:

```python
class AgentEventEmitter:
    """Emit events from agent to TUI."""

    def __init__(self):
        self.subscribers: List[Callable] = []

    def subscribe(self, callback: Callable):
        """Subscribe to agent events."""
        self.subscribers.append(callback)

    def emit(self, event_type: str, data: dict):
        """Emit event to all subscribers."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }
        for callback in self.subscribers:
            callback(event)

# Event types:
EVENTS = {
    "tool_call_start": {"tool": str, "args": dict},
    "tool_call_end": {"tool": str, "result": dict, "duration_ms": int, "success": bool},
    "state_changed": {"from": str, "to": str, "reason": str},
    "progress_update": {"percent": float, "completed": int, "total": int},
    "message": {"role": str, "content": str},
    "goal_completed": {"goal_id": str, "title": str},
    "file_created": {"path": str, "type": str, "size": int},
    "error": {"error_type": str, "message": str, "recoverable": bool},
    "checkpoint": {"snapshot_id": str, "reason": str},
}
```

### 5.2 Reactive UI Updates

```python
class GaiaTUI(App):
    """Main TUI application."""

    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.agent_thread = None

        # Reactive state
        self.app_state = reactive({
            "current_state": None,
            "state_stack": [],
            "progress": 0.0,
            "messages": [],
            "recent_tools": [],
            "manifest": None,
        })

        # Subscribe to agent events
        self.agent.event_emitter.subscribe(self.on_agent_event)

    def on_agent_event(self, event: dict):
        """Handle agent event (called from background thread)."""
        event_type = event["type"]
        data = event["data"]

        # Use call_from_thread to update UI safely from background thread
        if event_type == "tool_call_end":
            self.call_from_thread(
                self.handle_tool_call,
                data["tool"],
                data["success"],
                data["duration_ms"],
                event["timestamp"],
            )
        elif event_type == "state_changed":
            self.call_from_thread(
                self.handle_state_change,
                data["from"],
                data["to"],
            )
        elif event_type == "progress_update":
            self.call_from_thread(
                self.handle_progress_update,
                data["percent"],
            )
        elif event_type == "message":
            self.call_from_thread(
                self.handle_message,
                data["role"],
                data["content"],
                event["timestamp"],
            )

    def handle_tool_call(self, tool: str, success: bool, duration_ms: int, timestamp: str):
        """Update UI with tool call result."""
        # Update state
        self.app_state["recent_tools"].append({
            "tool": tool,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": timestamp,
        })

        # Keep only last 20
        if len(self.app_state["recent_tools"]) > 20:
            self.app_state["recent_tools"].pop(0)

        # UI widgets auto-update via reactivity

    def handle_progress_update(self, percent: float):
        """Update progress bar."""
        self.app_state["progress"] = percent
```

---

## 6. Navigation & Keyboard Shortcuts

### 6.1 Global Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Pause agent execution (can resume) |
| `Ctrl+Q` | Quit TUI (prompts to save if agent running) |
| `Ctrl+R` | Resume agent execution |
| `Ctrl+K` | Kill agent (emergency stop) |
| `Tab` | Cycle through panels |
| `Shift+Tab` | Cycle backwards |
| `Esc` | Open command menu |
| `?` or `F1` | Help overlay |

### 6.2 Panel-Specific Shortcuts

**Chat Panel**:
| Shortcut | Action |
|----------|--------|
| `Enter` | Send message (Shift+Enter for newline) |
| `Ctrl+F` | Search messages |
| `Ctrl+L` | Clear chat (keeps session) |
| `Ctrl+E` | Export conversation |
| `↑/↓` | Navigate message history (when input empty) |

**Memory Browser**:
| Shortcut | Action |
|----------|--------|
| `/` | Focus search |
| `j/k` | Navigate table rows |
| `Enter` | View selected entry details |
| `e` | Edit selected entry |
| `d` | Delete selected entry |
| `n` | Next page |
| `p` | Previous page |

**Manifest Tree**:
| Shortcut | Action |
|----------|--------|
| `j/k` | Navigate tree |
| `Space` | Expand/collapse node |
| `Enter` | View file details |
| `g` | Switch to Goals view |
| `d` | Switch to Dependency Graph view |
| `r` | Refresh manifest |

**Tools Panel**:
| Shortcut | Action |
|----------|--------|
| `t` | Test selected tool |
| `i` | Invoke selected skill |
| `c` | Create new tool |
| `e` | Edit dynamic tool |
| `d` | Delete dynamic tool |

### 6.3 Command Menu (Esc)

Vim-style command palette:

```
┌──────────────────────────────────────────┐
│ Command Menu                        [Esc] │
├──────────────────────────────────────────┤
│ > _                                      │
│                                          │
│ Suggestions:                             │
│ enter state <state_id>                   │
│ create skill <name>                      │
│ checkpoint now                           │
│ consolidate memory                       │
│ export session                           │
│ load manifest <project_id>               │
│ switch panel <name>                      │
│ toggle auto-scroll                       │
└──────────────────────────────────────────┘
```

---

## 7. Visual Design

### 7.1 Color Scheme

**Dark theme** (default):
```python
DARK_THEME = {
    "background": "#1e1e1e",          # VS Code dark
    "surface": "#252526",
    "primary": "#007acc",             # Blue (state indicators)
    "success": "#4ec9b0",             # Teal (completed items)
    "warning": "#dcdcaa",             # Yellow (in-progress)
    "error": "#f48771",               # Red (failures)
    "text": "#d4d4d4",
    "text_dim": "#808080",
    "border": "#3e3e3e",
    "accent": "#c586c0",              # Purple (skills)
}
```

**Light theme** (optional):
```python
LIGHT_THEME = {
    "background": "#ffffff",
    "surface": "#f3f3f3",
    "primary": "#0066cc",
    "success": "#00aa66",
    "warning": "#cc8800",
    "error": "#cc3333",
    "text": "#000000",
    "text_dim": "#666666",
    "border": "#cccccc",
}
```

### 7.2 Status Icons

```python
STATUS_ICONS = {
    "complete": "✓",        # Checkmark (green)
    "in_progress": "⟳",     # Rotating arrow (yellow)
    "pending": "⧗",         # Hourglass (dim)
    "failing": "✗",         # X mark (red)
    "blocked": "⊘",         # Prohibited (orange)
    "success": "✓",         # Tool success (green)
    "error": "✗",          # Tool error (red)
    "running": "⟳",        # Agent running (blue, animated)
    "paused": "⏸",         # Agent paused (yellow)
}
```

### 7.3 Live Indicators

Animated spinners for active processes:

```python
class LiveSpinner(Static):
    """Animated spinner for active tasks."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def on_mount(self):
        self.frame_index = 0
        self.set_interval(0.1, self.animate)  # 100ms per frame

    def animate(self):
        self.update(self.FRAMES[self.frame_index])
        self.frame_index = (self.frame_index + 1) % len(self.FRAMES)
```

---

## 8. Integration with Agent Frameworks

### 8.1 Wiring to Persistent Memory

```python
class MemoryBrowserPanel(Container):
    """Memory browser integrated with all memory tiers."""

    def __init__(self, agent):
        super().__init__()
        self.episodic = agent.episodic_memory
        self.semantic = agent.semantic_memory
        self.universal = agent.universal_db

        # Subscribe to memory changes
        agent.event_emitter.subscribe(self.on_memory_event)

    def on_memory_event(self, event):
        """React to memory updates."""
        if event["type"] == "knowledge_added":
            self.refresh_semantic_table()
        elif event["type"] == "session_saved":
            self.refresh_episodic_timeline()
```

### 8.2 Wiring to State Machine

```python
class StateMachinePanel(Container):
    """State machine visualization."""

    def __init__(self, agent):
        super().__init__()
        self.state_machine = agent.state_machine

        # Subscribe to state transitions
        agent.event_emitter.subscribe(self.on_state_event)

    def on_state_event(self, event):
        """React to state changes."""
        if event["type"] == "state_changed":
            self.update_current_state(event["data"]["to"])
            self.update_stack(event["data"].get("stack", []))
            self.refresh_available_tools()
```

### 8.3 Wiring to Manifest

```python
class ManifestPanel(Container):
    """Manifest tree with live updates."""

    def __init__(self, agent):
        super().__init__()
        self.manifest = agent.manifest_tracker

        # Poll manifest every 2 seconds (or subscribe to file events)
        self.set_interval(2.0, self.refresh_from_manifest)

    def refresh_from_manifest(self):
        """Pull latest manifest state."""
        progress = self.manifest.compute_progress()
        files = self.manifest.get_all_files()

        self.update_progress_bar(progress.overall_percent)
        self.update_tree(files)
```

---

## 9. Performance Optimization

### 9.1 Handling High-Frequency Updates

For continuous execution with 1000+ steps:

```python
class ThrottledUpdater:
    """Prevent UI thrashing from high-frequency events."""

    def __init__(self, min_interval_ms: int = 100):
        self.min_interval = min_interval_ms / 1000
        self.last_update = 0
        self.pending_update = None

    def throttle_update(self, update_fn: Callable, *args):
        """Only update UI at most every min_interval_ms."""
        now = time.time()

        if now - self.last_update >= self.min_interval:
            # Execute immediately
            update_fn(*args)
            self.last_update = now
            self.pending_update = None
        else:
            # Queue for later
            self.pending_update = (update_fn, args)

    def flush_pending(self):
        """Execute pending update if any."""
        if self.pending_update:
            fn, args = self.pending_update
            fn(*args)
            self.last_update = time.time()
            self.pending_update = None
```

### 9.2 Lazy Loading

Don't load all memory at startup:

```python
class LazyMemoryView(Container):
    """Load memory data on-demand."""

    def __init__(self, memory):
        super().__init__()
        self.memory = memory
        self.loaded = False

    def on_show(self):
        """Only load data when panel becomes visible."""
        if not self.loaded:
            self.load_data()
            self.loaded = True

    def load_data(self):
        """Query memory database."""
        # Load only recent data initially
        recent = self.memory.search(
            time_range=(
                (datetime.now() - timedelta(days=7)).isoformat(),
                datetime.now().isoformat()
            )
        )
        self.populate_table(recent)
```

### 9.3 Virtual Scrolling

For large datasets (1000+ knowledge entries, 500+ sessions):

```python
# Use Textual's DataTable with virtual scrolling (built-in)
table = DataTable(cursor_type="row", zebra_stripes=True)

# DataTable automatically virtualizes (only renders visible rows)
# Can handle 100K+ rows without performance issues
```

---

## 10. Implementation with Textual

### 10.1 Application Structure

```python
# tui/app.py
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Button, DataTable, Input, Log, ProgressBar

from tui.panels.chat import ChatPanel
from tui.panels.state import StateProgressPanel
from tui.panels.memory import MemoryBrowser
from tui.panels.manifest import ManifestTreePanel
from tui.panels.tools import ToolsPanel
from tui.panels.logs import LogsPanel

class GaiaTUI(App):
    """Gaia Agent Terminal User Interface."""

    CSS_PATH = "app.css"
    TITLE = "Gaia Agent"
    SUB_TITLE = "Continuous Execution Mode"

    BINDINGS = [
        ("ctrl+c", "pause_agent", "Pause"),
        ("ctrl+r", "resume_agent", "Resume"),
        ("ctrl+q", "quit", "Quit"),
        ("f1", "show_memory", "Memory"),
        ("f2", "show_tools", "Tools"),
        ("f3", "show_states", "States"),
        ("f4", "show_logs", "Logs"),
        ("tab", "next_panel", "Next Panel"),
        ("escape", "command_menu", "Menu"),
    ]

    def __init__(self, agent, config: TUIConfig):
        super().__init__()
        self.agent = agent
        self.config = config

        # Start agent in background thread
        self.agent_thread = threading.Thread(
            target=self._run_agent,
            daemon=True
        )

    def compose(self) -> ComposeResult:
        """Create UI layout."""
        yield Header()

        # Main content area
        with Horizontal(id="main-container"):
            # Left side: Chat (60% width)
            with Vertical(id="chat-container", classes="panel"):
                yield ChatPanel(self.agent, id="chat-panel")

            # Right side: Monitoring (40% width)
            with Vertical(id="monitor-container", classes="panel"):
                yield StateProgressPanel(self.agent, id="state-panel")
                yield ManifestTreePanel(self.agent.manifest_tracker, id="manifest-panel")

        yield Footer()

    def on_mount(self):
        """Called when TUI is ready."""
        # Start agent
        self.agent_thread.start()

        # Focus chat input
        self.query_one("#chat-panel").focus_input()

    def action_pause_agent(self):
        """Pause button pressed."""
        self.agent.pause()
        self.notify("Agent paused. Press Ctrl+R to resume.")

    def action_resume_agent(self):
        """Resume button pressed."""
        self.agent.resume()
        self.notify("Agent resumed.")

    def action_show_memory(self):
        """F1: Open memory browser."""
        # Switch right panel to memory browser
        monitor = self.query_one("#monitor-container")
        monitor.remove_children()
        monitor.mount(MemoryBrowser(self.agent))

    def action_command_menu(self):
        """Esc: Open command palette."""
        self.push_screen(CommandMenu(self.agent))

    def _run_agent(self):
        """Background thread: run agent continuously."""
        try:
            result = self.agent.continuous_execute_until_complete(
                task_description=self.config.initial_task,
                completion_criteria=self.config.completion_criteria,
            )

            # Emit completion event
            self.agent.event_emitter.emit("task_complete", result)

        except Exception as e:
            self.agent.event_emitter.emit("error", {
                "error_type": "agent_crash",
                "message": str(e),
                "recoverable": False,
            })
```

### 10.2 CSS Styling (Textual CSS)

```css
/* app.css */

#main-container {
    layout: horizontal;
    height: 100%;
}

#chat-container {
    width: 60%;
    border-right: solid $primary;
}

#monitor-container {
    width: 40%;
}

.panel {
    padding: 1;
    border: solid $border;
}

/* Chat panel */
#chat-panel {
    height: 100%;
}

.message-user {
    background: $surface;
    color: $accent;
    padding: 1;
    margin: 1 0;
}

.message-assistant {
    background: transparent;
    color: $text;
    padding: 1;
    margin: 1 0;
}

.tool-call-indicator {
    color: $text_dim;
    padding: 0 2;
}

.tool-call-success {
    color: $success;
}

.tool-call-error {
    color: $error;
}

/* State panel */
#state-panel {
    height: 40%;
    border-bottom: solid $border;
}

.progress-bar {
    background: $surface;
    color: $success;
}

.metrics-table {
    height: auto;
}

/* Manifest tree */
#manifest-panel {
    height: 60%;
}

.tree-node-complete {
    color: $success;
}

.tree-node-in-progress {
    color: $warning;
}

.tree-node-failing {
    color: $error;
}

/* Tables */
DataTable {
    height: 100%;
}

DataTable > .datatable--header {
    background: $primary;
    color: $background;
    text-style: bold;
}

DataTable > .datatable--cursor {
    background: $accent 20%;
}
```

---

## 11. Advanced Features

### 11.1 Diff Viewer

When agent modifies files, show diff inline:

```
┌──────────────────────────────────────────────────────────┐
│ FILE CHANGE: api/auth.py                                 │
├──────────────────────────────────────────────────────────┤
│ @@ line 45 @@                                            │
│                                                          │
│   def authenticate(user: str, password: str):            │
│ -     # TODO: Add validation                            │ (red)
│ +     if not user or not password:                      │ (green)
│ +         raise ValueError("Missing credentials")        │ (green)
│       hashed = bcrypt.hash(password)                     │
│       ...                                                │
│                                                          │
│ [Accept] [Reject] [Edit] [View Full File]               │
└──────────────────────────────────────────────────────────┘
```

### 11.2 Execution Timeline Visualizer

Gantt-chart style view of execution:

```
┌──────────────────────────────────────────────────────────────┐
│ EXECUTION TIMELINE                        Total: 2h 15m      │
├──────────────────────────────────────────────────────────────┤
│ Time │ State          │ Activity                             │
│──────┼────────────────┼──────────────────────────────────────│
│ 12:00│ Planning       │ ████████░░░░░ (15 min)               │
│ 12:15│ Implementation │ ████████████████████████░░░ (1h 45m) │
│ 13:30│  └─ Debug      │   ████░ (15 min) [nested]            │
│ 13:45│ Implementation │ ████░ (continued, 15 min)            │
│ 14:00│ Testing        │ ████░ (15 min)                       │
│ 14:15│ Review         │ ██ (5 min, current)                  │
│                                                              │
│ Click on bar to see details for that phase                  │
└──────────────────────────────────────────────────────────────┘
```

### 11.3 Live Dependency Graph

Interactive graph visualization (using ASCII art):

```
┌──────────────────────────────────────────────────────────┐
│ DEPENDENCY GRAPH                           [Expand All]   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│     main.py ──────────> models.py                        │
│        │                   │                             │
│        │                   └────> database.py            │
│        │                            │                    │
│        └────> api/                  │                    │
│               ├─ users.py ──────────┘                    │
│               ├─ auth.py ───> utils/auth.py              │
│               └─ cart.py ───> models.py                  │
│                                                          │
│ [Selected: api/auth.py]                                  │
│ Depends on: models.py, database.py, utils/auth.py       │
│ Depended on by: test_auth.py                            │
│ Impact if changed: 1 direct, 2 indirect                  │
└──────────────────────────────────────────────────────────┘
```

---

## 12. Complete Implementation Example

```python
# tui/main.py
"""
Gaia Agent TUI - Terminal User Interface

Usage:
    python -m tui.main --agent jarvis --task "Build REST API"
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Label, Input, Button,
    DataTable, Tree, ProgressBar, TabbedContent, TabPane,
    RichLog, Sparkline
)
from textual.reactive import reactive
from textual import on, work

import asyncio
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

from gaia.agents.base.agent import Agent


class GaiaTUI(App):
    """Gaia Agent Terminal User Interface."""

    CSS_PATH = "app.css"

    BINDINGS = [
        Binding("ctrl+c", "pause_agent", "Pause", priority=True),
        Binding("ctrl+r", "resume_agent", "Resume"),
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("f1", "toggle_memory", "Memory"),
        Binding("f2", "toggle_tools", "Tools"),
        Binding("f3", "toggle_states", "States"),
        Binding("f4", "toggle_logs", "Logs"),
        Binding("tab", "focus_next", "Next"),
        Binding("escape", "command_menu", "Menu"),
    ]

    # Reactive state
    agent_status = reactive("idle")  # "idle", "running", "paused", "error", "complete"
    current_state = reactive("--")
    progress_percent = reactive(0.0)

    def __init__(self, agent: Agent, initial_task: str):
        super().__init__()
        self.agent = agent
        self.initial_task = initial_task

        # Subscribe to agent events
        self.agent.event_emitter.subscribe(self.on_agent_event)

        # Layout state
        self.right_panel_view = "state"  # "state", "memory", "tools", "logs"

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header(show_clock=True)

        with Horizontal(id="main-container"):
            # Left: Chat (60%)
            with Vertical(id="chat-container", classes="panel-border"):
                yield Static("CHAT", id="chat-title", classes="panel-title")
                yield ChatMessages(id="chat-messages")
                yield Input(placeholder="Ask me anything...", id="chat-input")

            # Right: Monitoring (40%)
            with Vertical(id="right-container", classes="panel-border"):
                yield self._create_right_panel()

        yield Footer()

    def _create_right_panel(self) -> Container:
        """Create the right monitoring panel (switches based on view)."""
        if self.right_panel_view == "state":
            return StateMonitorPanel(self.agent, id="right-panel")
        elif self.right_panel_view == "memory":
            return MemoryBrowserPanel(self.agent, id="right-panel")
        elif self.right_panel_view == "tools":
            return ToolsPanel(self.agent, id="right-panel")
        elif self.right_panel_view == "logs":
            return LogsPanel(id="right-panel")
        else:
            return StateMonitorPanel(self.agent, id="right-panel")

    def on_mount(self):
        """Initialize after mounting."""
        # Start agent in background
        self.run_worker(self._run_agent_async(), exclusive=True)

        # Focus chat input
        self.query_one("#chat-input").focus()

    @work(exclusive=True, thread=True)
    async def _run_agent_async(self):
        """Run agent in worker thread."""
        try:
            self.agent_status = "running"

            result = self.agent.continuous_execute_until_complete(
                task_description=self.initial_task,
                completion_criteria={
                    "all_tests_pass": True,
                    "all_goals_complete": True,
                }
            )

            self.agent_status = "complete"
            self.notify("Task complete!", severity="information")

        except Exception as e:
            self.agent_status = "error"
            self.notify(f"Error: {e}", severity="error")

    def on_agent_event(self, event: dict):
        """Handle events from agent (called from agent thread)."""
        event_type = event["type"]
        data = event["data"]
        timestamp = event["timestamp"]

        # Use call_from_thread to safely update UI from background thread
        if event_type == "message":
            self.call_from_thread(
                self.add_chat_message,
                data["role"],
                data["content"],
                timestamp
            )

        elif event_type == "tool_call_end":
            self.call_from_thread(
                self.add_tool_to_log,
                data["tool"],
                data["success"],
                data["duration_ms"],
                timestamp
            )

        elif event_type == "state_changed":
            self.call_from_thread(
                self.update_state_display,
                data["to"],
                data.get("stack", [])
            )

        elif event_type == "progress_update":
            self.call_from_thread(
                self.update_progress,
                data["percent"],
                data["completed"],
                data["total"]
            )

        elif event_type == "file_created":
            self.call_from_thread(
                self.refresh_manifest_tree
            )

    def add_chat_message(self, role: str, content: str, timestamp: str):
        """Add message to chat panel."""
        chat_messages = self.query_one("#chat-messages", ChatMessages)
        chat_messages.add_message(role, content, timestamp)

    def update_state_display(self, state: str, stack: List[str]):
        """Update state indicator."""
        self.current_state = state

        state_panel = self.query_one("#right-panel", StateMonitorPanel)
        state_panel.update_state(state, stack)

    def update_progress(self, percent: float, completed: int, total: int):
        """Update progress bar."""
        self.progress_percent = percent

        state_panel = self.query_one("#right-panel", StateMonitorPanel)
        state_panel.update_progress(percent, completed, total)

    def action_toggle_memory(self):
        """F1: Switch right panel to memory browser."""
        self.right_panel_view = "memory"
        self._refresh_right_panel()

    def action_pause_agent(self):
        """Ctrl+C: Pause agent execution."""
        if self.agent_status == "running":
            self.agent.pause()
            self.agent_status = "paused"
            self.notify("Agent paused", severity="warning")

    def action_resume_agent(self):
        """Ctrl+R: Resume agent."""
        if self.agent_status == "paused":
            self.agent.resume()
            self.agent_status = "running"
            self.notify("Agent resumed", severity="information")

    def _refresh_right_panel(self):
        """Swap out right panel content."""
        container = self.query_one("#right-container")
        container.remove_children()
        container.mount(self._create_right_panel())
```

### 10.2 Chat Messages Widget

```python
class ChatMessages(ScrollableContainer):
    """Scrollable chat message list."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.messages = []

    def add_message(self, role: str, content: str, timestamp: str):
        """Add a message (reactive update)."""
        msg = MessageWidget(role, content, timestamp)
        self.mount(msg)
        self.messages.append(msg)

        # Auto-scroll to bottom
        self.scroll_end(animate=False)

class MessageWidget(Static):
    """Single message bubble."""

    def __init__(self, role: str, content: str, timestamp: str):
        time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")

        # Format content
        if role == "user":
            formatted = f"[bold cyan]You[/] ({time_str}):\n{content}"
            classes = "message-user"
        else:
            formatted = f"[bold green]Agent[/] ({time_str}):\n{content}"
            classes = "message-assistant"

        # Syntax highlighting for code blocks
        if "```" in content:
            formatted = self._highlight_code(formatted)

        super().__init__(formatted, classes=classes)

    def _highlight_code(self, content: str) -> str:
        """Add syntax highlighting to code blocks."""
        from rich.syntax import Syntax
        # Textual's Static widget supports Rich renderables
        # (implementation depends on language detection)
        return content  # Simplified
```

### 10.3 State Monitor Panel

```python
class StateMonitorPanel(Container):
    """Right panel showing agent state and progress."""

    def __init__(self, agent: Agent):
        super().__init__()
        self.agent = agent

    def compose(self) -> ComposeResult:
        """Build state monitor UI."""
        # State indicator
        yield Static("STATE & PROGRESS", classes="section-title")
        yield Label(f"State: {self.agent.state_machine.current_state or '--'}", id="state-label")
        yield Label(f"Stack: {self.agent.state_machine.state_stack}", id="stack-label")

        # Progress
        yield ProgressBar(total=100, id="progress-bar")
        yield Label("Progress: 0%", id="progress-label")

        # Metrics table
        yield Static("METRICS", classes="section-title")
        metrics_table = DataTable(id="metrics-table", zebra_stripes=True)
        metrics_table.add_columns("Metric", "Value")
        metrics_table.add_row("Time Elapsed", "0m")
        metrics_table.add_row("Tokens Used", "0")
        metrics_table.add_row("Cost", "$0.00")
        metrics_table.add_row("Est. Remaining", "--")
        yield metrics_table

        # Recent tools log
        yield Static("RECENT TOOLS", classes="section-title")
        yield RichLog(id="tool-log", max_lines=10, highlight=True, markup=True)

        # Manifest summary
        yield Static("MANIFEST", classes="section-title")
        yield Tree("Project", id="mini-manifest-tree")

        # Auto-refresh
        self.set_interval(1.0, self.refresh_metrics)

    def update_state(self, state: str, stack: List[str]):
        """Update state labels."""
        self.query_one("#state-label", Label).update(f"State: {state}")
        self.query_one("#stack-label", Label).update(f"Stack: {' ← '.join(stack)}")

    def update_progress(self, percent: float, completed: int, total: int):
        """Update progress bar."""
        self.query_one("#progress-bar", ProgressBar).update(progress=percent)
        self.query_one("#progress-label", Label).update(
            f"Progress: {percent:.0f}% ({completed}/{total})"
        )

    def add_tool_call(self, tool: str, success: bool, duration_ms: int, timestamp: str):
        """Add tool to log."""
        time_str = datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
        icon = "[green]✓[/]" if success else "[red]✗[/]"
        log = self.query_one("#tool-log", RichLog)
        log.write(f"{time_str} {icon} {tool} - {duration_ms}ms")

    def refresh_metrics(self):
        """Update metrics table (polled every 1s)."""
        progress = self.agent.manifest_tracker.compute_progress()

        table = self.query_one("#metrics-table", DataTable)
        table.update_cell_at((0, 1), f"{progress.time_elapsed_minutes}m")
        table.update_cell_at((1, 1), f"{progress.tokens_used / 1_000_000:.1f}M")
        table.update_cell_at((2, 1), f"${progress.tokens_used * 0.000005:.2f}")

        if progress.estimated_time_remaining_minutes:
            table.update_cell_at((3, 1), f"{progress.estimated_time_remaining_minutes}m")
```

---

## 13. Configuration

```python
@dataclass
class TUIConfig:
    """Configuration for TUI."""

    # Layout
    chat_width_percent: int = 60
    right_panel_width_percent: int = 40
    default_right_view: str = "state"  # "state", "memory", "tools", "logs"

    # Behavior
    auto_scroll_chat: bool = True
    auto_refresh_interval_sec: float = 1.0
    max_messages_displayed: int = 1000
    max_log_lines: int = 500

    # Visual
    theme: str = "dark"  # "dark" or "light"
    show_timestamps: bool = True
    syntax_highlighting: bool = True
    animate_progress: bool = True

    # Performance
    throttle_updates_ms: int = 100
    lazy_load_memory: bool = True
    virtual_scroll_threshold: int = 100

    # Features
    enable_diff_viewer: bool = True
    enable_graph_view: bool = True
    enable_search: bool = True
    enable_command_menu: bool = True

    # Agent
    initial_task: str = ""
    completion_criteria: Dict[str, bool] = field(default_factory=dict)
```

---

## 14. Running the TUI

### Installation

```bash
pip install textual rich
pip install amd-gaia[all]
```

### Launch

```bash
# Option 1: Direct launch with task
python -m tui.main \
    --agent jarvis \
    --task "Build a REST API with FastAPI" \
    --continuous

# Option 2: Interactive mode (enter task in TUI)
gaia-tui

# Option 3: Resume from checkpoint
gaia-tui --resume abc123
```

### Entry Point

```python
# tui/__main__.py
import argparse
from tui.main import GaiaTUI
from jarvis.agent import JarvisAgent, JarvisConfig

def main():
    parser = argparse.ArgumentParser(description="Gaia Agent TUI")
    parser.add_argument("--agent", default="jarvis", help="Agent type")
    parser.add_argument("--task", help="Initial task")
    parser.add_argument("--continuous", action="store_true", help="Continuous execution mode")
    parser.add_argument("--resume", help="Resume from session ID")

    args = parser.parse_args()

    # Initialize agent
    if args.agent == "jarvis":
        config = JarvisConfig(
            use_claude=True,
            enable_manifest=True,
            enable_states=True,
            enable_skills=True,
            memory_dir="./.gaia/memory",
        )
        agent = JarvisAgent(config)
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # Resume if requested
    if args.resume:
        agent.resume_from_checkpoint(args.resume)

    # Launch TUI
    app = GaiaTUI(agent, args.task or "")
    app.run()

if __name__ == "__main__":
    main()
```

---

## 15. Benefits Over CLI + Web Dashboard

| Feature | CLI Only | Web Dashboard | TUI | Winner |
|---------|----------|---------------|-----|--------|
| **No browser needed** | ✓ | ✗ | ✓ | CLI/TUI |
| **Keyboard-only workflow** | ✓ | Partial | ✓ | CLI/TUI |
| **Real-time monitoring** | ✗ | ✓ | ✓ | Dashboard/TUI |
| **Memory browsing** | ✗ (separate commands) | ✓ | ✓ | Dashboard/TUI |
| **Split-pane simultaneous view** | ✗ | ✓ | ✓ | Dashboard/TUI |
| **Works over SSH** | ✓ | ✗ (needs port forward) | ✓ | CLI/TUI |
| **Resource footprint** | Minimal | High (browser) | Low | CLI/TUI |
| **Mouse support** | ✗ | ✓ | ✓ | Dashboard/TUI |
| **Copy/paste** | ✓ | ✓ | ✓ | All |
| **Multi-user access** | ✗ | ✓ | ✗ | Dashboard |

**Recommendation**: Build **both TUI and Dashboard**:
- **TUI** for individual developers (local, SSH, terminal-first workflow)
- **Dashboard** for teams (multi-user, broader access, richer visualizations)

---

## 16. Implementation Roadmap

### Phase 1: Core TUI (Week 1-2)

- Chat panel with message list + input
- State monitor with progress bar
- Basic layout (split-pane)
- Keyboard shortcuts (Ctrl+C pause, Enter send)
- Agent event integration

**Exit criteria**: Can chat with agent in TUI, see real-time progress

### Phase 2: Advanced Panels (Week 3)

- Memory browser (episodic + semantic tabs)
- Manifest tree view
- Tools panel
- Logs panel
- Panel switching (F1-F4)

### Phase 3: Polish (Week 4)

- Diff viewer
- Command menu (Esc)
- Search functionality
- Export features
- Theme customization
- Performance optimization

---

## 17. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **UI flickering** during rapid updates | Throttle updates (100ms min), batch events |
| **Memory leaks** from unbounded logs | Max line limits, circular buffer for logs |
| **Thread safety** issues | Use Textual's `call_from_thread` for all UI updates from agent thread |
| **Complex terminal detection** | Graceful degradation for limited terminals, fallback to simple layout |
| **Unicode rendering** issues | Fallback ASCII icons if Unicode unsupported |

---

## 18. Textual-Specific Best Practices

### 18.1 Reactive State Pattern

```python
class MyWidget(Static):
    # Reactive variables auto-trigger UI updates
    count = reactive(0)

    def watch_count(self, new_value: int):
        """Called automatically when count changes."""
        self.update(f"Count: {new_value}")
```

### 18.2 Worker Threads

```python
@work(exclusive=True, thread=True)
async def long_running_task(self):
    """Run in background thread, don't block UI."""
    result = await some_async_operation()

    # Update UI from worker
    self.call_from_thread(self.update_result, result)
```

### 18.3 CSS for Layout

Use Textual CSS (not inline styles):

```css
#chat-container {
    width: 60%;
    height: 100%;
}

#right-container {
    width: 40%;
    height: 100%;
    border-left: solid cyan;
}

.panel-title {
    background: $primary;
    color: $background;
    padding: 0 1;
    text-align: center;
    text-style: bold;
}
```

---

## 19. Screenshots (ASCII Art Examples)

### Initial Launch

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent                    [Ready]               Session: --      [?][X] │
├───────────────────────────────┬────────────────────────────────────────────┤
│ CHAT                          │ STATE & PROGRESS                           │
│                               │ State: -- (not started)                    │
│                               │ Progress: ░░░░░░░░░░░░░░░░ 0%              │
│                               ├────────────────────────────────────────────┤
│ Welcome to Gaia Agent TUI     │ TOOLS                                      │
│                               │ No tools called yet                        │
│ Enter your task or question:  │                                            │
│                               │                                            │
│ > Build a REST API with       ├────────────────────────────────────────────┤
│   authentication_             │ MANIFEST                                   │
│                               │ No active project                          │
│                               │                                            │
│                               │ [F1] Memory [F2] Tools [F3] States         │
│                               │                                            │
├───────────────────────────────┴────────────────────────────────────────────┤
│ [Enter] Send  [Tab] Next Panel  [F1-F4] Switch View  [Ctrl+Q] Quit         │
└────────────────────────────────────────────────────────────────────────────┘
```

### Mid-Execution

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Gaia Agent        [Implementation Mode]  Session: abc123  Token: 2.4M [?][X]│
├───────────────────────────────────┬────────────────────────────────────────┤
│ CHAT                              │ STATE & PROGRESS                       │
│                                   │ ⟳ Implementation (step 145, 1h 30m)    │
│ You: Build a REST API             │ ████████████░░░░ 67% (21/32 files)    │
│                                   │ Est. remaining: 45 minutes             │
│ Agent: I'll create FastAPI app... ├────────────────────────────────────────┤
│                                   │ RECENT TOOLS                           │
│ [⟳ Status: Writing cart API]      │ 14:30:15 ✓ write_file api/cart.py 420ms│
│                                   │ 14:30:18 ✓ run_tests test_cart.py 890ms│
│ Agent: User auth complete ✓       │ 14:30:25 ✓ update_manifest ...     45ms│
│ Agent: Cart system in progress... │ 14:30:30 ✗ run_tests test_cart 1.2s   │
│                                   │ 14:30:31 ⟳ enter_state debug       10ms│
│ You: [Watching agent work...]    │ 14:30:35 ✓ inspect_error ...      120ms│
│                                   │ 14:30:40 ✓ edit_file api/cart  380ms│
│                                   │ 14:30:45 ✓ run_tests test_cart 950ms│
│                                   │ 14:30:46 ✓ return_from_state ...   5ms│
│                                   │ 14:30:47 ✓ write_file api/chk  520ms│
│                                   ├────────────────────────────────────────┤
│                                   │ MANIFEST (live)    Goals: 5/8 ✓        │
│                                   │ ├─ api/                                │
│                                   │ │  ├─ users.py ✓                       │
│                                   │ │  ├─ auth.py ✓                        │
│                                   │ │  ├─ cart.py ✓ (just fixed)           │
│                                   │ │  └─ checkout.py ⟳ (writing now)      │
│                                   │ └─ tests/ (32/45 passing)              │
├───────────────────────────────────┴────────────────────────────────────────┤
│ [Ctrl+C] Pause  [F1] Memory  [F4] Full Logs  [Esc] Menu  [Auto-scroll: ON] │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 20. Why Textual is the Right Choice

1. **Modern & Maintained**: Active development, Python 3.7+, 1.0+ stable
2. **Rich Integration**: Built on Rich library (beautiful formatting, syntax highlighting)
3. **Reactive**: Automatic UI updates when state changes
4. **Widget Library**: DataTable, Tree, ProgressBar, TabbedContent, Input, etc.
5. **CSS Styling**: Familiar styling with CSS-like syntax
6. **Async Support**: Works with asyncio, worker threads
7. **Cross-Platform**: Linux, macOS, Windows (WSL)
8. **Mouse + Keyboard**: Both interaction modes supported
9. **Documentation**: Excellent docs and examples
10. **Production-Ready**: Used by projects like Rich CLI, PyPI TUI tools

---

## Recommendation

**Yes, build the TUI** — it's the ideal interface for:
- Developers who live in the terminal
- SSH/remote work (dashboard needs port forwarding)
- Continuous execution monitoring (see real-time progress without browser)
- Low-resource environments (TUI uses <10MB RAM vs. browser's 500MB+)

Implement in **Phase 1-2 alongside the core frameworks** (Persistent Memory, State Machine, Manifest). The TUI becomes the primary interface for agent developers and power users, while the dashboard serves teams and non-technical stakeholders.

---

*Terminal User Interface Design Specification for Gaia Agent SDK.*
*Built on Textual framework, integrates with all 7 architectural frameworks, optimized for continuous execution and real-time monitoring.*
