# GAIA DB Dashboard - Complete Setup and Usage Guide

**Version:** 1.0.0
**Date:** 2026-02-23
**Purpose:** Real-time monitoring and debugging of GAIA agent databases

---

## What This Dashboard Does

**Monitor GAIA Code agent execution in real-time:**
- Watch logs appear as agent executes (logs.db)
- Track context token usage to see if agent decomposes properly
- See errors and warnings as they occur
- Monitor task progress (plan.db)
- View tool usage patterns (tools.db)
- Edit knowledge and insights (knowledge.db)

**Perfect for debugging:**
- "Why did the agent fail?"  → Check logs.db for ERROR entries
- "Is the agent using agent_query?" → Check plan.db for task decomposition
- "Is context growing too fast?" → Check Dashboard context usage graph

---

## Database Locations

### Your Setup (WSL + Windows)

**Linux/WSL path:**
```
/home/user/.gaia/workspace/
```

**Windows File Explorer path:**
```
\\wsl.localhost\Ubuntu-24.04\home\user\.gaia\workspace
```

**To access in Windows:**
1. Open File Explorer (Win+E)
2. Paste in address bar: `\\wsl.localhost\Ubuntu-24.04\home\user\.gaia\workspace`
3. You'll see: `agents.db`, `knowledge.db`, `memory.db`, `plan.db`, `skills.db`, `tools.db`
4. After first GAIA Code run with new code: `logs.db` will appear

**The dashboard automatically detects and opens this path.**

---

## Installation

### Step 1: Navigate to Dashboard Directory

```bash
cd /mnt/c/Users/14255/Work/gaia/src/gaia/electron/db-dashboard
```

### Step 2: Install Dependencies

```bash
npm install
```

**What gets installed:**
- `electron@31.0.0` (~300MB) - Desktop app framework
- `better-sqlite3@11.7.0` (~10MB) - Fast SQLite driver
- `chokidar@3.6.0` (~2MB) - File watching for live updates

**Total:** ~312MB, takes about 2-3 minutes on first install.

**Note:** better-sqlite3 compiles native bindings during install. This is normal.

---

## Running the Dashboard

### Mode 1: Electron (Full Features - Recommended)

```bash
npm start
```

**Opens:** Desktop window at 1400x900px

**Features available:**
- ✅ All 7 databases
- ✅ Dashboard overview with stats
- ✅ Live auto-refresh (2s default)
- ✅ Edit cells, add/delete rows
- ✅ SQL console with write queries
- ✅ Export to JSON/CSV
- ✅ Database backup
- ✅ File watching for real-time updates

**Use this mode for:** Production debugging and monitoring

---

### Mode 2: Browser (Read-Only Testing)

```bash
npm run dev:browser
```

**Opens:** http://localhost:3847 in your default browser

**Features available:**
- ✅ All 7 databases
- ✅ Dashboard overview
- ✅ Live auto-refresh
- ❌ No editing (read-only)
- ❌ No file operations

**Use this mode for:** Quick inspection without Electron overhead

---

## Workflow: Monitoring GAIA Code

### Step-by-Step

**1. Start dashboard FIRST (in terminal 1):**
```bash
cd /mnt/c/Users/14255/Work/gaia/src/gaia/electron/db-dashboard
npm start
```

**2. Configure auto-refresh:**
- Ensure "Auto" checkbox is ON ✓
- Set interval to 2s (or 1s for faster updates)
- Dashboard should show "Auto-refresh: ON ⟳ 2s"

**3. Open logs.db tab:**
- Click "logs.db" tab at the top
- Click "runtime_logs" table in sidebar
- You should see: "No data" or old logs if databases exist

**4. Start GAIA Code (in terminal 2):**
```bash
cd /mnt/c/Users/14255/Work/gaia

# Run the C++ port benchmark
gaia-code "Implement comprehensive C++17 port of GAIA framework..." --tui simple
```

**5. Watch the dashboard update:**
- **Logs tab:** New rows appear every 2 seconds as agent logs
- **Dashboard tab:** Context usage graph updates, error count increases
- **Plan tab:** Tasks appear and change status
- **Changed rows:** Flash green briefly when they first appear

**6. Debug issues:**
- If you see context warnings → Check Dashboard context graph
- If agent fails → Check logs.db ERROR entries
- If plan isn't decomposed → Check plan.db tasks (should see agent_query calls)

---

## Dashboard Overview Explained

When you open the dashboard, the first tab shows:

### Top Section: Workspace Statistics

```
╔══ Overview ═══════════════════════════════════════╗
║ Workspace: /home/user/.gaia/workspace            ║
║ Databases: 7 found                               ║
║ Total Size: 213 KB                               ║
║ Last Activity: 2 seconds ago (logs.db)           ║
║ Total Logs: 1,247 | Errors: 12 | Warnings: 3    ║
╚═══════════════════════════════════════════════════╝
```

**Updates every 2 seconds** - Shows live stats across all databases

---

### Middle Section: Recent Activity

**Recent Errors (logs.db):**
```
[ERROR] 2s ago | Step 13
Missing required arguments for write_file: file_path, content

[ERROR] 45s ago | Step 12
Context at 75% of limit (24,576 tokens)
```

**Active Tasks (plan.db):**
```
⏳ pending | Generate C++ headers
⚙️  in_progress | Generate MCP client
✓ completed | Read Python source
```

**Recent Insights (knowledge.db):**
```
💡 debugging | Step 15
Repeated write_file failures due to output token truncation.
Solution: Use agent_query decomposition.
```

---

### Bottom Section: Analytics

**Top Tools (tools.db):**
```
write_file     ████████████████░░░░ 156 calls (98% success)
read_file      ██████████████░░░░░░ 89 calls (100% success)
run_cli_command████████░░░░░░░░░░░░ 45 calls (91% success)
agent_query    ████░░░░░░░░░░░░░░░░ 12 calls (100% success)
get_logs       ██░░░░░░░░░░░░░░░░░░ 8 calls (100% success)
```

**Context Token Usage (logs.db):**
```
[Sparkline chart - vertical bars]
Step:  5    10   15   20   25   30
Tokens: ▂▃▄▅▇█▇▅
       5K  10K  18K  24K 🟡28K 🔴31K (exceeded!)
```

Colors: Blue (OK), Yellow (>24K warning), Red (>30K critical)

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Ctrl+D** | Switch to Dashboard tab |
| **Ctrl+F** | Focus search/filter box |
| **Ctrl+E** | Switch to SQL Console tab |
| **Ctrl+Enter** | Execute SQL query |
| **Ctrl+R** | Manual refresh (even with auto-refresh on) |
| **Ctrl+O** | Open custom workspace path |
| **Escape** | Close any modal |

---

## Common Tasks

### Task 1: Monitor Context Usage

**Goal:** Watch if agent stays under 32K token limit

**Steps:**
1. Open Dashboard tab
2. Scroll to "Context Token Usage" section
3. As agent runs, bars appear showing token count per step
4. Yellow bars (>24K) = Warning threshold reached
5. Red bars (>30K) = Critical threshold reached

**What to look for:**
- ✅ Good: Bars stay blue (< 24K tokens)
- ⚠️  Warning: Yellow bars appear → agent should use agent_query soon
- ❌ Bad: Red bars appear → agent didn't decompose, emergency compaction triggered

---

### Task 2: Debug Agent Errors

**Goal:** Understand why agent failed

**Steps:**
1. Click "logs.db" tab
2. Click "runtime_logs" table
3. Click "level" column header to sort
4. Scroll to ERROR entries
5. Read the "message" column for error details
6. Check "step_number" to see when it occurred
7. Check "context_tokens" to see if context was high

**Example:**
```
Row 145 | ERROR | Step 13 | 28,500 tokens
Message: "Missing required arguments for write_file: file_path, content"

→ Interpretation: Output token truncation at high context usage
→ Solution: Agent should have used agent_query decomposition
```

---

### Task 3: Verify RAC Decomposition

**Goal:** Check if agent used agent_query

**Steps:**
1. Click "plan.db" tab
2. Click "tasks" table
3. Look for task descriptions
4. Check if descriptions mention "agent_query" or show decomposition
5. Count tasks: Should be 4-5 high-level tasks, not 19 granular tasks

**Good pattern (RAC):**
```
Task 1: "Generate type system headers via agent_query"
Task 2: "Generate MCP client via agent_query"
Task 3: "Generate Agent class via agent_query"
Task 4: "Generate tests via agent_query"
```

**Bad pattern (Sequential):**
```
Task 1: "write_file types.h"
Task 2: "write_file json_utils.h"
Task 3: "write_file json_utils.cpp"
... 16 more tasks
```

---

### Task 4: Edit Knowledge

**Goal:** Add debugging insights to knowledge.db

**Steps:**
1. Click "knowledge.db" tab
2. Turn off "Read-Only" toggle (top right)
3. Click "insights" table
4. Click "Add Row" button
5. Fill in form:
   - category: "debugging"
   - content: "Context warnings at 75% indicate need for agent_query decomposition"
   - domain: "gaia-code"
   - triggers: "context,decomposition,agent_query"
6. Click "Add"
7. Row appears in table (flash green)

**The agent can now retrieve this via `recall("context warnings")`**

---

### Task 5: Export Error Logs

**Goal:** Save error logs for analysis

**Steps:**
1. Click "logs.db" tab
2. Click SQL Console tab
3. Enter query:
   ```sql
   SELECT timestamp, level, step_number, message, context_tokens
   FROM runtime_logs
   WHERE level IN ('ERROR', 'CRITICAL')
   ORDER BY timestamp DESC
   LIMIT 100
   ```
4. Click "Run" (or press Ctrl+Enter)
5. Click "Export Results" → "JSON"
6. Save to `gaia_errors_2026-02-23.json`

---

## Troubleshooting

### Issue: Dashboard shows "No databases found"

**Solution 1:** Check workspace path
- Click "Change" button next to workspace path
- Browse to `\\wsl.localhost\Ubuntu-24.04\home\user\.gaia\workspace`
- Click "Select Folder"

**Solution 2:** Databases don't exist yet
- Run GAIA Code once to create databases:
  ```bash
  gaia-code "Hello" --tui simple
  ```
- This will create the databases in `~/.gaia/workspace/`

---

### Issue: Auto-refresh not working

**Check:**
1. "Auto" checkbox is ON (green)
2. Status badge shows "ON" (not "OFF" or "PAUSED")
3. "Updated: Xs ago" counter is incrementing
4. No modal is open (auto-refresh pauses during editing)

**Fix:**
- Click "Auto" checkbox off then on
- Close any open modals
- Check browser console (F12) for errors

---

### Issue: logs.db doesn't exist

**Cause:** The logs.db database is only created when GAIA Code runs with `enable_shared_state=True`.

**Solution:** Run GAIA Code once:
```bash
gaia-code "Test task" --tui simple
```

After this, `logs.db` will appear in the workspace.

---

### Issue: Can't edit cells

**Check:**
1. "Read-Only" toggle (top right) is OFF
2. You're in Electron mode (not browser mode)
3. Database file has write permissions

**Fix:**
- Turn off read-only mode
- Ensure you ran `npm start` (not `npm run dev:browser`)

---

## Performance Notes

**Auto-refresh interval recommendations:**

| Use Case | Recommended Interval |
|----------|---------------------|
| **Active debugging** (watching agent run) | 1s or 2s |
| **Background monitoring** | 5s or 10s |
| **Large databases** (>100K rows) | 5s or 10s |
| **Editing mode** | Turn OFF (auto-pauses anyway) |

**Database size limits:**
- Tested with databases up to 10MB
- Pagination keeps UI responsive even with 100K+ rows
- FTS5 search handles large datasets efficiently

---

## How to Run Before GAIA Code Benchmark

### Complete Workflow

**Terminal 1 - Start Dashboard:**
```bash
cd /mnt/c/Users/14255/Work/gaia/src/gaia/electron/db-dashboard
npm start

# Dashboard opens
# 1. Verify workspace path shows: /home/user/.gaia/workspace
# 2. See existing databases (agents.db, knowledge.db, etc.)
# 3. Ensure "Auto" is ON with 2s interval
# 4. Click "logs.db" tab (might be empty if no runs yet)
# 5. Leave dashboard open
```

**Terminal 2 - Run GAIA Code:**
```bash
cd /mnt/c/Users/14255/Work/gaia

# Commit the fixes first
git add src/gaia/agents/base/agent.py \
        src/gaia/agents/base/shared_state.py \
        src/gaia/agents/gaia_code/agent.py \
        src/gaia/agents/gaia_code/tools.py

git commit -m "Add 32K context limit, logs.db introspection, and RAC-first planning"

# Run Round 5 benchmark
gaia-code "Implement comprehensive C++17 port of GAIA framework. CRITICAL: Use agent_query() decomposition - DO NOT generate files sequentially. Decompose into 4-5 component-based subtasks. Target: /mnt/c/Users/14255/Work/Projects/GaiaCodeExperiments/gaiacpp_gaia_round5/" \
  --allowed-paths "/mnt/c/Users/14255/Work/Projects/GaiaCodeExperiments/gaiacpp_gaia_round5,/mnt/c/Users/14255/Work/gaia3" \
  --tui simple
```

**Watch the dashboard:**
- **Dashboard tab:** Context usage updates, error count increases
- **logs.db tab:** Rows appear in real-time (DEBUG, INFO, WARNING, ERROR)
- **plan.db tab:** Tasks appear showing decomposition strategy

---

## What to Watch For (Success Indicators)

### ✅ Good Signs

**In Dashboard tab:**
- Context usage stays under 24K tokens (blue bars)
- No red critical warnings
- Error count stays low (< 5)

**In plan.db tab:**
- Tasks show "agent_query" in descriptions
- 4-6 high-level tasks (not 19 granular tasks)
- Tasks transition: pending → in_progress → completed

**In logs.db tab:**
- INFO level: "Plan created with agent_query decomposition"
- No WARNING: "Context approaching limit"
- No ERROR: "Context limit exceeded"

---

### ⚠️ Warning Signs

**In Dashboard tab:**
- Yellow bars in context graph (>24K tokens)
- Warning count > 0
- Context usage trending upward rapidly

**In logs.db tab:**
- WARNING: "Context at 75% of limit"
- Message: "Consider using agent_query()"

**Action:** Agent should adapt by querying logs and decomposing

---

### ❌ Failure Signs

**In Dashboard tab:**
- Red bars in context graph (>30K tokens)
- Critical count > 0
- Error count > 10

**In logs.db tab:**
- CRITICAL: "Context limit exceeded"
- ERROR: "Emergency compaction triggered"
- ERROR: "Missing required arguments" (repeated)

**Action:** Stop run, analyze logs, adjust approach

---

## Advanced Features

### Custom Database Path

**To open databases from other locations:**

1. Click "Open File" button (top bar)
2. Browse to any `.db` file
3. Database opens in a new tab
4. Can have workspace + custom databases open simultaneously

**Use cases:**
- Compare production vs test databases
- Analyze archived agent runs
- Debug specific session databases

---

### SQL Console Power Features

**Example queries:**

**Find all context warnings:**
```sql
SELECT timestamp, step_number, message, context_tokens
FROM runtime_logs
WHERE message LIKE '%context%'
  AND level IN ('WARNING', 'ERROR', 'CRITICAL')
ORDER BY timestamp DESC;
```

**Analyze tool failure rate:**
```sql
SELECT
  tool_name,
  COUNT(*) as total_calls,
  SUM(CASE WHEN message LIKE '%error%' THEN 1 ELSE 0 END) as failures,
  ROUND(100.0 * SUM(CASE WHEN message LIKE '%success%' THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
FROM runtime_logs
WHERE logger_name LIKE '%tool%'
GROUP BY tool_name
ORDER BY total_calls DESC;
```

**Track context growth:**
```sql
SELECT step_number, context_tokens, level, message
FROM runtime_logs
WHERE context_tokens IS NOT NULL
ORDER BY step_number;
```

---

## Files Created

**Total:** 16 files, ~5,400 lines

```
db-dashboard/
├── package.json          # Dependencies
├── main.js               # Electron main process (717 lines)
├── preload.js            # IPC bridge (98 lines)
├── dev-server.js         # Browser mode server (344 lines)
├── README.md             # Architecture docs
├── SETUP_AND_USAGE.md    # This file
└── renderer/
    ├── index.html        # App shell (271 lines)
    ├── css/
    │   └── styles.css    # Dark theme (1,519 lines)
    └── js/
        ├── state.js      # App state (92 lines)
        ├── utils.js      # Helpers (197 lines)
        ├── dashboard.js  # Overview page (441 lines) ⭐ NEW
        ├── auto-refresh.js # Live updates (348 lines) ⭐ NEW
        ├── data-grid.js  # Table viewer (387 lines)
        ├── sql-console.js # SQL editor (142 lines)
        ├── fts-search.js # FTS5 search (129 lines)
        ├── modals.js     # Dialogs (171 lines)
        ├── table-list.js # Sidebar (114 lines)
        └── app.js        # Main app (457 lines)
```

---

## Implementation Review - Gaps Check

### ✅ Completed Features

- [x] Dashboard overview page with cross-database stats
- [x] Live auto-refresh (0.5s to 10s configurable)
- [x] File modification watching
- [x] Row change detection with green flash
- [x] Toast notifications for new data
- [x] WSL path auto-detection (\\wsl.localhost\...)
- [x] Custom database file picker
- [x] 7 database tabs (memory, knowledge, tools, skills, agents, plan, logs)
- [x] Context usage sparkline chart
- [x] Error/warning summaries
- [x] Recent activity panels
- [x] Read-only mode toggle
- [x] Edit/add/delete rows
- [x] SQL console
- [x] FTS5 search
- [x] Export to JSON/CSV
- [x] Dark theme

### ⚠️ Known Limitations

1. **logs.db won't exist until first run** - Normal, will be created when GAIA Code runs
2. **Change detection is polling-based** - Checks file mtime every 1.5s (not inotify)
3. **Large result sets** (>10K rows) may be slow - Use SQL filters or pagination
4. **No undo** - Edits are immediate (backup before bulk changes)

### 🔄 Future Enhancements (Not Blocking)

- Query history in SQL console
- Syntax highlighting for SQL
- Visual query builder
- Real-time charts (line graphs for trends)
- Database diff view (compare before/after)
- Import from CSV/JSON

---

## Ready to Run

**Installation command:**
```bash
cd /mnt/c/Users/14255/Work/gaia/src/gaia/electron/db-dashboard && npm install
```

**Start command:**
```bash
npm start
```

**Expected startup time:** 3-5 seconds

**Database paths:**
- **Full Linux path:** `/home/user/.gaia/workspace/`
- **Windows path:** `\\wsl.localhost\Ubuntu-24.04\home\user\.gaia\workspace`
- **knowledge.db full path:** `/home/user/.gaia/workspace/knowledge.db`

**All 6 existing databases** (agents, knowledge, memory, plan, skills, tools) should appear immediately.

**logs.db will appear** after running GAIA Code with the new code (with shared_state enabled).

The dashboard is production-ready! 🚀
