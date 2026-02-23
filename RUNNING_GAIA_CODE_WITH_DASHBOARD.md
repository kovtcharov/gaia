# Running GAIA Code with Real-Time Dashboard Monitoring

**Date:** 2026-02-23
**Status:** ✅ All improvements committed, ready to run

---

## What Was Implemented

**Total changes:**
- 4 core files modified (~550 lines)
- 27 new files (11 analysis docs + 16 dashboard files)
- 13,162 lines added

**Key improvements:**
1. ✅ Hard 32K context limit with monitoring
2. ✅ logs.db for agent introspection
3. ✅ RAC-first planning guidance
4. ✅ Electron dashboard for real-time monitoring
5. ✅ Default model: claude-sonnet-4-6

---

## Step 1: Install Dashboard (One-Time Setup)

```bash
cd src/gaia/electron/db-dashboard
npm install
```

**Takes:** 2-3 minutes
**Installs:** Electron, better-sqlite3, chokidar
**Size:** ~312 MB

---

## Step 2: Start Dashboard

**Terminal 1:**
```bash
cd src/gaia/electron/db-dashboard
npm start
```

**What opens:**
- Electron window with dark theme
- Dashboard tab showing workspace: `~/.gaia/workspace/`
- 6 existing databases visible (agents, knowledge, memory, plan, skills, tools)
- Auto-refresh: ON, 2s interval
- Read-only: ON

**Verify:**
- See "Dashboard" tab is active
- See database tabs across the top
- See "Auto ⟳ 2s" in top right (green)

---

## Step 3: Run GAIA Code

**Terminal 2:**
```bash
cd <your-gaia-repo>

# Run C++ port benchmark with all fixes
gaia-code "Implement comprehensive C++17 port of GAIA Python agent framework.

CRITICAL: This is a COMPLEX task (19 files, ~3500 LOC). You MUST use agent_query() decomposition.

Source: ~/Work/gaia3/src/gaia/agents/base/{agent.py,tools.py,console.py} and ~/Work/gaia3/src/gaia/mcp/mcp_client.py

Target: ~/Work/Projects/GaiaCodeExperiments/gaiacpp_gaia_round5/

STRATEGY: Decompose into 4-6 agent_query calls (headers, MCP client, Agent class, tests, build files). DO NOT write 19 files sequentially." \
  --allowed-paths "~/Work/Projects/GaiaCodeExperiments/gaiacpp_gaia_round5,~/Work/gaia3" \
  --tui simple
```

---

## Step 4: Watch Dashboard in Real-Time

### In Dashboard Tab (Overview):

**Watch for:**
- **Context Token Usage** sparkline updates every 2s
  - Blue bars = healthy (<24K)
  - Yellow bars = warning (>24K)
  - Red bars = critical (>30K)
- **Error count** should stay low
- **Last Activity** timestamp updates

### In logs.db Tab:

**Switch to logs.db:**
1. Click "logs.db" tab (if it doesn't exist yet, agent hasn't started logging)
2. Click "runtime_logs" table in sidebar
3. See empty table or "No data"

**As GAIA Code runs:**
- Rows appear every 2 seconds (auto-refresh)
- New rows flash green for 2 seconds
- Toast: "New data detected in logs.db"
- Filter by level: Click "level" column → choose ERROR to see only errors

**What to look for:**
- ✅ INFO: "Plan created with 4 steps"
- ✅ INFO: "Tool execution: agent_query"
- ⚠️  WARNING: "Context at 75% of limit" (agent should adapt)
- ❌ ERROR: "Context limit exceeded" (should NEVER happen)

### In plan.db Tab:

**Switch to plan.db:**
1. Click "plan.db" tab
2. Click "tasks" table

**Watch for:**
- ✅ Good: 4-6 tasks with agent_query in descriptions
- ❌ Bad: 19 tasks with write_file in descriptions
- Status changes: pending → in_progress → completed

---

## Expected Behavior (Success)

### Step 1-2: Agent analyzes task, creates decomposed plan

**Dashboard:**
- Context: ~3K tokens (9% of limit)

**plan.db:**
```
Task 1: "Generate headers via agent_query" | pending
Task 2: "Generate MCP client via agent_query" | pending
Task 3: "Generate Agent class via agent_query" | pending
Task 4: "Generate tests via agent_query" | pending
```

**logs.db:**
```
[INFO] Plan created with 4 steps using agent_query decomposition
```

### Step 3-6: Execute agent_query calls

**Dashboard:**
- Context: ~5K-8K tokens (15-24% of limit, stays blue)
- No warnings

**plan.db:**
```
Task 1: "Generate headers..." | completed ✓
Task 2: "Generate MCP client..." | in_progress
```

**logs.db:**
- Lots of INFO/DEBUG from sub-agents
- No WARNING or ERROR entries

### Step 7-8: Build and test

**Dashboard:**
- Peak context: ~10K tokens (30% of limit)
- Error count: 0
- Status: Success

**Result:** All 19 files generated, 82/82 tests pass

---

## What Could Go Wrong

### Problem: Context warnings appear

**Dashboard shows:**
- Yellow bars in context graph
- Warning count > 0

**logs.db shows:**
```
[WARNING] Context at 75% of limit (24,576 tokens)
```

**Interpretation:** Agent created sequential plan instead of using agent_query

**What agent should do:**
1. Call `get_logs(search="context", limit=10)`
2. See the warning
3. Adapt: Use `agent_query` to decompose remaining work

---

### Problem: Repeated errors

**logs.db shows:**
```
[ERROR] write_file failed: Missing required arguments (Step 13)
[ERROR] write_file failed: Missing required arguments (Step 14)
[ERROR] write_file failed: Missing required arguments (Step 15)
```

**Interpretation:** Output token truncation (file too large for single response)

**What agent should do:**
1. Call `get_logs(level="ERROR", limit=5)`
2. See repeated pattern
3. Adapt: Use `agent_query` to delegate file generation to sub-agent

---

## Dashboard Keyboard Shortcuts

| Key | Action | Use When |
|-----|--------|----------|
| `Ctrl+D` | Dashboard tab | Quick overview |
| `Ctrl+F` | Search | Find specific log messages |
| `Ctrl+E` | SQL console | Run custom queries |
| `Ctrl+R` | Refresh | Force immediate update |
| `Escape` | Close modal | Cancel operation |

---

## Files to Monitor

### 1. logs.db (Most Important)

**Table:** `runtime_logs`

**Columns:**
- `timestamp` - When logged
- `level` - DEBUG/INFO/WARNING/ERROR/CRITICAL
- `step_number` - Which agent step
- `message` - Log message
- `context_tokens` - Context size when logged
- `logger_name` - Which module logged

**Filter examples:**
- Errors only: Sort by `level`, scroll to ERROR
- Context logs: Search "context"
- Specific step: Filter `step_number = 15`

---

### 2. plan.db

**Table:** `tasks`

**What to check:**
- `description` - Should mention "agent_query" for complex tasks
- `status` - Track progress (pending → in_progress → completed)
- Count of tasks - Should be 4-6 (not 19)

---

### 3. knowledge.db

**Tables:** `insights`, `preferences`

**You can edit:**
- Add debugging hints
- Store error patterns
- Add preferences for agent behavior

**Agent can retrieve via:** `recall("your query")`

---

## Summary

**Before running GAIA Code:**

1. ✅ Commit completed (f08b6c9)
2. ✅ Dashboard installed: `cd src/gaia/electron/db-dashboard && npm install`
3. ✅ Dashboard started: `npm start`
4. ✅ Verified: Dashboard shows workspace and databases

**To run GAIA Code:**

```bash
# In terminal 2
gaia-code "Your task" --tui simple
```

**Database path:** `~/.gaia/workspace/`
- knowledge.db: `~/.gaia/workspace/knowledge.db` (56 KB, exists)
- logs.db: `~/.gaia/workspace/logs.db` (will be created on first run)

**Watch in dashboard:**
- Dashboard tab for overview
- logs.db tab for real-time logs
- plan.db tab for task decomposition

Ready to run! 🚀
