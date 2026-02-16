# Agent Management Dashboard: Architecture & Design

**Date**: February 6, 2026
**Scope**: Web-based dashboard for managing, monitoring, and interacting with Gaia-based AI agents
**Purpose**: Provide observability into agent memory, knowledge, tools, execution, and state

---

## 1. Overview

### The Need

Advanced agents with persistent memory, adaptive prompts, dynamic tools, and learning loops accumulate significant state:
- Thousands of knowledge entries in semantic memory
- Hundreds of past sessions in episodic memory
- Dozens of custom tools with quality metrics
- Multiple prompt versions with performance data
- Execution traces across long-running tasks
- Real-time state during active execution

Without a dashboard, this state is opaque. Users can't:
- See what the agent knows
- Understand why it made a recommendation
- Review and clean up accumulated knowledge
- Monitor resource usage
- Debug failed executions
- Manage custom tools and skills

### What the Dashboard Provides

A **web-based interface** that exposes all agent subsystems:

1. **Agent Interaction** — Chat interface for querying the agent
2. **Memory Browser** — Explore episodic and semantic memory
3. **Knowledge Database Viewer** — Query, filter, edit knowledge entries
4. **Tool Registry** — Inspect static and dynamic tools with quality metrics
5. **State Visualizer** — View current agent state, prompt composition, available tools
6. **Execution History** — Replay past executions, inspect tool calls, debug failures
7. **Real-Time Monitoring** — Live progress during continuous execution
8. **Configuration** — Manage agent settings, budgets, permissions

---

## 2. Technology Stack

### Backend

**Framework**: FastAPI (Python) — same as Gaia's API server
**Database**: SQLite (agent's existing knowledge.db) + PostgreSQL (if multi-agent/multi-tenant)
**Vector Store**: FAISS (agent's existing embeddings)
**WebSocket**: For real-time updates during continuous execution

### Frontend

**Framework**: React + TypeScript
**Styling**: TailwindCSS
**State Management**: React Query (for server state) + Zustand (for UI state)
**Data Visualization**: Recharts (for metrics), D3.js (for graph visualizations)
**Code Editor**: Monaco Editor (VS Code's editor) for viewing/editing tool code
**Markdown Rendering**: react-markdown

### Deployment

**Development**: `npm run dev` (Vite) + `uvicorn app:app --reload`
**Production**: Docker Compose (frontend + backend + agent)
**Access**: `http://localhost:3000` (dev) or nginx reverse proxy (prod)

---

## 3. Dashboard Sections

### 3.1 Agent Interaction (Chat Interface)

**Route**: `/`

**Features**:
- **Chat window**: Send queries, receive responses (same as CLI but with UI)
- **Streaming responses**: Real-time token streaming during generation
- **Tool call visualization**: Show which tools were called, arguments, results
- **Thought visualization**: Display agent reasoning (if available)
- **Session management**: Save/load/resume sessions
- **Multi-turn context**: Full conversation history with scroll
- **Attachments**: Upload files (traces, logs, code) directly in chat
- **Export**: Download conversation as markdown or JSON

**API Endpoints**:
```python
POST /api/chat/query
  Request: {"message": str, "session_id": str, "stream": bool}
  Response: {"response": str, "tools_called": [...], "session_id": str}

GET /api/chat/sessions
  Response: [{"session_id": str, "created_at": str, "last_message": str}, ...]

GET /api/chat/session/{session_id}
  Response: {"conversation_history": [...], "metadata": {...}}
```

**UI Components**:
- `ChatInterface` (main container)
- `MessageList` (conversation display)
- `MessageInput` (text area + file upload + submit)
- `ToolCallCard` (shows tool invocation details)
- `StreamingMessage` (updates in real-time)

---

### 3.2 Memory Browser

**Route**: `/memory`

**Sub-tabs**:

#### 3.2.1 Episodic Memory (Past Sessions)

**Features**:
- **Timeline view**: Sessions sorted by date
- **Search**: Semantic search + structured filters (date range, hardware, model, tags)
- **Session details**: Expand to see full conversation, findings, recommendations, outcomes
- **Session comparison**: Select 2+ sessions, compare findings side-by-side
- **Export**: Download sessions as JSON or markdown

**UI**:
```
+----------------------------------+
| Search: [find allreduce issues ] | Filter: Hardware [MI355X▼] Model [All▼] Date [Last 30 days▼]
+----------------------------------+
| Timeline
|
| Feb 6, 2026  [Session abc123] MI355X DeepSeek - Expert imbalance found  [View] [Compare]
| Feb 5, 2026  [Session def456] MI300X LLaMA - Allreduce bottleneck        [View] [Compare]
| Feb 4, 2026  [Session ghi789] B200 GPT-OSS - Flash attention regression   [View] [Compare]
|
| [Load More...]
+----------------------------------+
```

**API Endpoints**:
```python
GET /api/memory/episodic/search?q=...&hardware=...&model=...&from=...&to=...
  Response: [{"session_id": str, "timestamp": str, "summary": str, ...}, ...]

GET /api/memory/episodic/session/{session_id}
  Response: {full session data}
```

#### 3.2.2 Semantic Memory (Knowledge Base)

**Features**:
- **Knowledge table**: Searchable, sortable table of all knowledge entries
- **Filters**: By category, hardware, model, confidence range, evidence count
- **Edit**: Click to edit knowledge entry (updates confidence, adds counterexamples, marks as invalid)
- **Add**: Manually add new knowledge
- **Delete**: Remove stale/incorrect knowledge
- **Confidence graph**: Visualize knowledge confidence distribution
- **Evidence tracking**: See which sessions contributed to each knowledge entry

**UI**:
```
+------------------------------------------------------------------+
| Knowledge Base                                    [+ Add Knowledge]
+------------------------------------------------------------------+
| Search: [flash attention] | Category [All▼] Hardware [All▼] Confidence [> 0.5]
+------------------------------------------------------------------+
| ID | Category | Pattern | Confidence | Evidence | Last Confirmed | Actions
|----+----------+---------+------------+----------+----------------+---------
| hw-mi355x-allreduce | Hardware | MI355X allreduce < 50% for small msgs | 0.92 | 23 | 2026-02-06 | [Edit][Delete]
| opt-expert-rebalance | Optimization | Expert imbalance > 150% → rebalance | 0.85 | 8 | 2026-02-05 | [Edit][Delete]
|
+------------------------------------------------------------------+
```

**API Endpoints**:
```python
GET /api/memory/semantic/search?q=...&category=...&min_confidence=...
POST /api/memory/semantic/entry (create)
PUT /api/memory/semantic/entry/{id} (update)
DELETE /api/memory/semantic/entry/{id}
GET /api/memory/semantic/stats (confidence distribution, category breakdown)
```

---

### 3.3 Tool Registry

**Route**: `/tools`

**Features**:
- **Tool list**: All static + dynamic tools
- **Tool details**: Description, parameters, code (for dynamic tools), usage stats
- **Quality metrics**: Success rate, avg execution time, last used
- **Test tool**: Run tool with sample arguments directly from UI
- **Create tool**: Web-based tool builder with code editor
- **Edit tool**: Update dynamic tool code
- **Delete tool**: Remove dynamic tool

**UI**:
```
+------------------------------------------------------------------+
| Tool Registry                                     [+ Create Tool]
+------------------------------------------------------------------+
| Filter: Type [All▼] Quality [All▼] Sort [Usage▼]
+------------------------------------------------------------------+
| Tool Name | Type | Success Rate | Avg Time | Last Used | Actions
|-----------+------+--------------+----------+-----------+---------
| analyze_trace | Static | 98% | 2.3s | 10 min ago | [View][Test]
| compare_traces | Static | 95% | 4.1s | 1 hour ago | [View][Test]
| classify_gemm_bottlenecks | Dynamic | 87% | 1.2s | 2 days ago | [View][Edit][Test][Delete]
|
+------------------------------------------------------------------+
```

**Tool Details Modal**:
- **Code view**: Monaco editor showing tool implementation
- **Parameter schema**: Interactive form
- **Test interface**: Run with sample args, see results
- **Usage history**: Recent invocations with args + results
- **Quality trend**: Graph of success rate over time

**API Endpoints**:
```python
GET /api/tools/list?type=...&sort=...
GET /api/tools/{name}
POST /api/tools/test (execute tool with args)
POST /api/tools/create
PUT /api/tools/{name}
DELETE /api/tools/{name}
```

---

### 3.4 State Visualizer

**Route**: `/state`

**Features**:
- **Current state**: Show active state (planning, implementation, testing, etc.)
- **System prompt**: View full composed prompt (all layers)
- **Available tools**: List of tools in current state
- **Memory context**: What memory was injected into prompt
- **State history**: Timeline of state transitions during execution
- **State manager**: Create, save, load, delete custom states

**UI**:
```
+------------------------------------------------------------------+
| Current State: IMPLEMENTATION_MODE                    [Change State▼]
+------------------------------------------------------------------+
| System Prompt Preview (2,450 tokens)
|   [Immutable Core] You are a code assistant...
|   [Task Module: Code Implementation] Focus on writing clean, testable code...
|   [Learned Instructions] - When writing React components, use functional style...
|   [Memory Context] Relevant knowledge: MI355X flash attention uses hipCK...
|   [Tools] 15 tools available in this state
|
| Available Tools in This State:
|   ✓ write_file    ✓ edit_file     ✓ run_tests
|   ✓ analyze_code  ✓ check_syntax  ✓ lint_code
|   ✗ delete_database (disabled in this state)
|
| State Transition History (last 10):
|   14:30:00 - PLANNING_MODE → IMPLEMENTATION_MODE (reason: plan approved)
|   14:45:00 - IMPLEMENTATION_MODE → TESTING_MODE (reason: code complete)
|   14:50:00 - TESTING_MODE → IMPLEMENTATION_MODE (reason: test failure, fixing)
+------------------------------------------------------------------+
```

**API Endpoints**:
```python
GET /api/state/current
GET /api/state/prompt (full composed prompt)
POST /api/state/enter/{state_id}
POST /api/state/exit
GET /api/state/list (all saved states)
POST /api/state/create (define new state)
```

---

### 3.5 Execution History & Debugging

**Route**: `/executions`

**Features**:
- **Execution list**: All past executions (long-running tasks)
- **Execution timeline**: Step-by-step trace of tool calls, thoughts, state transitions
- **Replay**: Re-run execution from specific step
- **Inspect**: Drill into any step to see full tool args, results, files modified
- **Diff view**: See file changes made at each step
- **Error analysis**: Filter to failed executions, see error patterns
- **Performance**: Token usage, latency per step

**UI** (Execution Timeline):
```
+------------------------------------------------------------------+
| Execution: Build E-Commerce App (Session xyz789)    Status: Complete
| Started: Feb 6, 14:00  |  Ended: Feb 6, 18:30  |  Duration: 4.5 hours  |  Steps: 342
+------------------------------------------------------------------+
| Step | Time | State | Tool | Args | Result | Files | Tokens
|------+------+-------+------+------+--------+-------+--------
| 1 | 14:00:05 | PLANNING | create_plan | {task: "..."} | Success | - | 1,200
| 2 | 14:01:12 | IMPLEMENTATION | write_file | {path: "backend/main.py"} | Success | 1 created | 850
| 3 | 14:02:30 | IMPLEMENTATION | write_file | {path: "backend/models.py"} | Success | 1 created | 920
| ... | ... | ... | ... | ... | ... | ... | ...
| 87 | 15:45:00 | TESTING | run_tests | {path: "tests/"} | FAIL | - | 450
| 88 | 15:45:30 | DEBUG | inspect_error | {error_msg: "..."} | Success | - | 300
| 89 | 15:46:15 | IMPLEMENTATION | edit_file | {path: "backend/api.py"} | Success | 1 modified | 600
| 90 | 15:47:00 | TESTING | run_tests | {path: "tests/"} | Success | - | 420
|
| [Show All 342 Steps] [Export CSV] [Replay from Step 87]
+------------------------------------------------------------------+
```

**API Endpoints**:
```python
GET /api/executions/list?status=...&from=...&to=...
GET /api/executions/{exec_id}/trace
GET /api/executions/{exec_id}/step/{step_num}
POST /api/executions/{exec_id}/replay?from_step=...
```

---

### 3.6 Manifest Viewer

**Route**: `/manifest`

**Features**:
- **Project overview**: High-level architecture, tech stack, goals
- **File explorer**: Tree view of project files with metadata
- **Dependency graph**: Interactive visualization of file dependencies
- **Progress tracker**: Goals completed vs. pending
- **Architecture decisions**: Timeline of decisions with rationale
- **Quality dashboard**: Test coverage, lint status, per-file quality scores

**UI** (Dependency Graph):
```
+------------------------------------------------------------------+
| Project: E-Commerce App                         Progress: 45% ███████░░░░░
+------------------------------------------------------------------+
| Architecture: Monorepo (React + FastAPI)
| Files: 32 total (18 complete, 3 in-progress, 11 pending)
| Tests: 28/45 passing
|
| Dependency Graph                                    [Switch to Tree View]
|
|     main.py ──────> models.py ──────> database.py
|        │               │                   │
|        │               └──> cart.py ───────┘
|        │                       │
|        └──────────────────> auth.py
|                                │
|                              user.py
|
| Click node to see details. Red = failing tests. Yellow = in-progress.
+------------------------------------------------------------------+
```

**API Endpoints**:
```python
GET /api/manifest/current
GET /api/manifest/files
GET /api/manifest/dependencies
GET /api/manifest/progress
PUT /api/manifest/update
```

---

### 3.7 Real-Time Monitoring

**Route**: `/monitor`

**Features** (for active/continuous execution):
- **Current status**: What the agent is doing right now
- **Progress**: Percentage complete, estimated time remaining
- **Token usage**: Cumulative tokens, cost (if using paid API)
- **Recent tools**: Last 10 tool calls with status
- **File changes**: Files created/modified in current execution
- **Quality metrics**: Tests passing, lint warnings, errors
- **Pause/Resume**: Controls for long-running tasks
- **Kill switch**: Emergency stop

**UI** (Real-Time):
```
+------------------------------------------------------------------+
| AGENT ACTIVE: Building E-Commerce App              [Pause] [Stop]
+------------------------------------------------------------------+
| Status: Implementing cart API endpoints
| Progress: 45% ███████░░░░░  (18/32 files complete)
| Running: 2h 30m  |  Est. Remaining: 3h 15m  |  Tokens: 8.5M  |  Cost: $42.50
|
| Current Action:
|   [14:30:15] STATE: Implementation Mode
|   [14:30:16] TOOL: write_file(path="backend/api/cart.py", ...)
|   [14:30:18] STATUS: Success - File created
|   [14:30:19] TOOL: run_tests(path="tests/test_cart.py")
|   [14:30:21] STATUS: FAILING - 2/5 tests fail
|   [14:30:22] ENTERING: Debug Mode
|   [14:30:23] TOOL: inspect_test_failure(test="test_add_to_cart")
|   [14:30:25] THOUGHT: "Test fails because cart.add() doesn't validate product exists..."
|   [14:30:26] TOOL: edit_file(path="backend/api/cart.py", ...)
|   [Currently executing...]
|
| Recent Files Modified:
|   backend/api/cart.py (2m ago)
|   backend/models/cart.py (5m ago)
|   tests/test_cart.py (8m ago)
|
| Quality Status:
|   Tests: 28/45 passing (62%)  ⚠ 4 failing
|   Lint: 3 warnings, 0 errors  ✓
|   Type Check: All files pass  ✓
+------------------------------------------------------------------+
```

**WebSocket Events** (real-time push):
```javascript
ws.on('agent_status', (data) => {
    // {status: "executing", current_action: "writing file", ...}
})
ws.on('tool_call', (data) => {
    // {tool: "write_file", args: {...}, result: {...}}
})
ws.on('progress_update', (data) => {
    // {percent: 47, files_complete: 19, ...}
})
ws.on('state_change', (data) => {
    // {from: "IMPLEMENTATION", to: "TESTING", reason: "..."}
})
```

---

### 3.8 Configuration Panel

**Route**: `/config`

**Features**:
- **Agent settings**: LLM provider, model, max_steps (or continuous mode)
- **Memory settings**: Retention policy, consolidation frequency, confidence thresholds
- **Tool settings**: Max custom tools, auto-deprecation threshold, security allowlist
- **Quality gates**: Test coverage requirement, lint rules
- **Budget limits**: Daily token budget, cost alerts
- **Permissions**: Allowed file paths, network access, executable commands

**API Endpoints**:
```python
GET /api/config
PUT /api/config
POST /api/config/reset (restore defaults)
```

---

## 4. Data Model

### Database Schema

**Primary database**: `dashboard.db` (SQLite for single-agent, PostgreSQL for multi-agent)

```sql
-- Dashboard sessions (separate from agent sessions)
CREATE TABLE dashboard_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TEXT,
    last_active TEXT,
    ip_address TEXT,
    user_agent TEXT
);

-- Dashboard activity log
CREATE TABLE activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    session_id TEXT,
    action TEXT,           -- 'view_memory', 'edit_knowledge', 'create_tool', etc.
    target TEXT,           -- Entity ID (knowledge_id, tool_name, etc.)
    details TEXT           -- JSON blob
);

-- Agent configuration snapshots
CREATE TABLE config_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    config_json TEXT,      -- Full agent config
    changed_by TEXT,       -- User ID or 'system'
    reason TEXT
);

-- Tool usage metrics (aggregated from agent's knowledge.db)
CREATE VIEW tool_metrics AS
SELECT
    tool_name,
    COUNT(*) as usage_count,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
    AVG(duration_ms) as avg_duration,
    MAX(timestamp) as last_used
FROM execution_trace
GROUP BY tool_name;
```

**Agent database** (reused): `.gaia/knowledge.db` contains:
- `knowledge` table (semantic memory)
- `recommendations` table (outcome tracking)
- `execution_trace` table (provenance)
- `sessions` table (episodic memory metadata)

---

## 5. API Design

### REST API Structure

**Base URL**: `http://localhost:8080/api/v1`

**Endpoint groups**:

| Group | Prefix | Purpose |
|-------|--------|---------|
| Chat | `/chat` | Agent interaction |
| Memory | `/memory` | Episodic + semantic memory |
| Tools | `/tools` | Tool registry management |
| State | `/state` | State machine control |
| Executions | `/executions` | Execution history + debugging |
| Manifest | `/manifest` | Project manifest CRUD |
| Config | `/config` | Agent configuration |
| Monitor | `/monitor` | Real-time metrics |

**Authentication**: JWT tokens (if multi-user) or none (if single-user local deployment)

**Rate Limiting**: None (local deployment) or per-user limits (multi-tenant)

---

## 6. Real-Time Updates (WebSocket)

**WebSocket endpoint**: `ws://localhost:8080/ws/monitor`

**Event types**:

```javascript
// Agent events
{type: 'agent_status', data: {status: 'executing', task: '...'}}
{type: 'tool_call', data: {tool: 'write_file', args: {...}, result: {...}}}
{type: 'state_change', data: {from: 'PLANNING', to: 'IMPLEMENTATION'}}
{type: 'progress', data: {percent: 47, eta_minutes: 195}}
{type: 'error', data: {error: '...', step: 87, recovery_plan: '...'}}
{type: 'quality_gate', data: {gate: 'tests', status: 'fail', details: '...'}}

// Memory events
{type: 'knowledge_added', data: {id: '...', category: '...', confidence: 0.8}}
{type: 'session_saved', data: {session_id: '...', findings_count: 5}}
{type: 'consolidation_complete', data: {patterns_extracted: 8}}

// Tool events
{type: 'tool_created', data: {name: '...', created_by: 'agent'}}
{type: 'tool_deprecated', data: {name: '...', reason: 'high failure rate'}}
```

---

## 7. Implementation Phases

### Phase 1: Core Dashboard (Week 1-2)

**Features**:
- Chat interface (basic, no streaming)
- Memory browser (episodic + semantic, read-only)
- Tool registry (list + view)
- Basic monitoring (token usage, status)

**Stack**: React frontend + FastAPI backend + SQLite

---

### Phase 2: Continuous Execution Integration (Week 3)

**Features**:
- Real-time monitoring (WebSocket)
- Progress tracking
- Pause/Resume controls
- Execution history viewer

---

### Phase 3: Advanced Features (Week 4-6)

**Features**:
- State visualizer
- Manifest viewer with dependency graph
- Tool creation UI
- Knowledge editing
- Execution replay
- Quality metrics visualization

---

## 8. Security Considerations

| Risk | Mitigation |
|------|------------|
| **Unauthorized access** | JWT auth for multi-user, firewall for local |
| **Code injection via dashboard** | Tool code editor validates before save |
| **Prompt injection** | Dashboard inputs sanitized, no direct prompt editing |
| **Memory tampering** | Knowledge edits logged in activity_log |
| **Resource exhaustion** | Rate limits on API, kill switch on monitor page |

---

## 9. Dashboard Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Browser                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │   Chat   │ │  Memory  │ │  Tools   │ │ Monitor  │      │
│  │Interface │ │ Browser  │ │ Registry │ │ Real-Time│      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
│       │            │            │            │              │
└───────┼────────────┼────────────┼────────────┼──────────────┘
        │            │            │            │
        │ HTTP/REST  │            │            │ WebSocket
        │            │            │            │
┌───────┼────────────┼────────────┼────────────┼──────────────┐
│       │            │            │            │               │
│  ┌────▼────────────▼────────────▼────────────▼─────┐       │
│  │          FastAPI Backend (Dashboard Server)      │       │
│  │  - /api/chat/* endpoints                         │       │
│  │  - /api/memory/* endpoints                       │       │
│  │  - /api/tools/* endpoints                        │       │
│  │  - /ws/monitor (WebSocket)                       │       │
│  └────┬─────────────┬─────────────┬─────────────┬──┘       │
│       │             │             │             │           │
│  ┌────▼─────┐  ┌────▼────┐  ┌─────▼────┐  ┌────▼────┐    │
│  │Dashboard │  │ SQLite  │  │  FAISS   │  │ Agent   │    │
│  │  DB      │  │knowledge│  │Embeddings│  │Instance │    │
│  │(sessions,│  │  .db    │  │          │  │(Jarvis) │    │
│  │ activity)│  │(memory) │  │(sessions)│  │         │    │
│  └──────────┘  └─────────┘  └──────────┘  └────┬────┘    │
│                                                  │          │
│                                             ┌────▼────┐    │
│                                             │TraceLens│    │
│                                             │   API   │    │
│                                             └─────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Key Technologies

| Component | Technology | Why |
|-----------|-----------|-----|
| **Frontend** | React + TypeScript | Industry standard, rich ecosystem |
| **Styling** | TailwindCSS | Rapid UI development |
| **Charts** | Recharts + D3.js | Metrics visualization |
| **Code Editor** | Monaco Editor | Same as VS Code |
| **Backend** | FastAPI | Same as Gaia API server, async support |
| **Real-Time** | WebSockets | Low-latency bidirectional communication |
| **Database** | SQLite (local) / PostgreSQL (cloud) | Lightweight for single-agent, scalable for multi-agent |
| **Vector Search** | FAISS | Agent already uses this |

---

## 11. Deployment Options

### Option 1: Local (Development & Single-User)

```bash
# Terminal 1: Start agent backend
cd jarvis/
python -m uvicorn dashboard.app:app --reload --port 8080

# Terminal 2: Start React frontend
cd dashboard-ui/
npm run dev

# Open browser
http://localhost:3000
```

**Pros**: Simple, no deployment complexity
**Cons**: Not accessible to team

---

### Option 2: Docker Compose (Team Deployment)

```yaml
version: '3.8'
services:
  backend:
    build: ./jarvis
    ports:
      - "8080:8080"
    volumes:
      - ./.gaia:/app/.gaia
      - ./traces:/app/traces
    environment:
      - GAIA_MEMORY_DIR=/app/.gaia
      - GAIA_USE_CLAUDE=true

  frontend:
    build: ./dashboard-ui
    ports:
      - "3000:3000"
    depends_on:
      - backend

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - frontend
```

**Access**: `http://team-server/` (via nginx reverse proxy with auth)

---

### Option 3: Cloud Deployment (Enterprise)

**Stack**: Kubernetes + Helm charts
**Components**:
- **Frontend**: Static files in nginx pods
- **Backend**: FastAPI pods (autoscaling)
- **Database**: PostgreSQL (managed service)
- **Vector Store**: Pinecone or Weaviate (managed)
- **Auth**: OAuth2 + RBAC

---

## 12. Integration with Agent

### Backend Implementation

```python
# dashboard/app.py
from fastapi import FastAPI, WebSocket
from jarvis.agent import JarvisAgent, JarvisConfig

app = FastAPI(title="Jarvis Dashboard API")

# Single agent instance (for local deployment)
agent = JarvisAgent(JarvisConfig(
    use_claude=True,
    docs_path="./docs",
    memory_dir="./.gaia",
))

@app.post("/api/chat/query")
async def chat_query(request: ChatRequest):
    result = agent.process_query(request.message)
    return {
        "response": result["final_answer"],
        "tools_called": result.get("tools_used", []),
        "session_id": result.get("session_id"),
    }

@app.get("/api/memory/semantic/search")
async def search_knowledge(q: str, min_confidence: float = 0.5):
    knowledge = agent.semantic_memory.search(q, min_confidence)
    return {"results": knowledge}

@app.websocket("/ws/monitor")
async def monitor_websocket(websocket: WebSocket):
    await websocket.accept()
    # Stream agent status updates in real-time
    while True:
        status = agent.get_current_status()
        await websocket.send_json(status)
        await asyncio.sleep(1)  # Update every second
```

---

## 13. Open Questions

1. **Should the dashboard run as a separate service or embedded in the agent?**
   - Recommendation: Separate service (cleaner separation, can manage multiple agents)

2. **Should knowledge editing be allowed, or is it read-only?**
   - Recommendation: Allow editing with activity logging for accountability

3. **How to handle multi-agent dashboards (managing multiple specialist agents)?**
   - Recommendation: Agent selector dropdown, shared manifest view

4. **Should the dashboard require authentication for local single-user deployments?**
   - Recommendation: Optional (disable for local, enable for team)

---

*Dashboard design for Gaia-based agent management and observability.*
*Integrates with Persistent Memory, Adaptive Prompts, Dynamic Tools, and Learning frameworks.*
