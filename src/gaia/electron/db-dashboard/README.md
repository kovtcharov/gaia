# GAIA DB Dashboard

**Real-time monitoring and debugging tool for GAIA agent databases.**

Monitor agent execution, track context usage, debug errors, and edit knowledge - all in real-time.

---

## Quick Start

### Prerequisites

Install build tools (required for better-sqlite3 native compilation):

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y build-essential

# macOS
xcode-select --install
```

### Installation

```bash
# 1. Navigate to dashboard
cd src/gaia/electron/db-dashboard

# 2. Install dependencies
npm install

# 3. Run
npm start
```

The dashboard opens showing your GAIA workspace at `~/.gaia/workspace/` with live auto-refresh enabled.

---

## What It Does

**Monitor GAIA Code in real-time:**
- Watch logs appear as agent executes
- Track context token usage (stay under 32K limit)
- See errors and warnings immediately
- Monitor task decomposition (verify agent uses agent_query)
- View tool usage patterns

**Debug issues:**
- Search error logs to understand failures
- Check context growth to detect decomposition problems
- Analyze task tree to verify RAC architecture usage
- Query any database with SQL

**Edit agent knowledge:**
- Add insights to knowledge.db
- Modify task status in plan.db
- Update tool metadata in tools.db

---

## The 7 Databases

| Database | Contains | Use For |
|----------|----------|---------|
| **memory.db** | Session cache (files, tool results) | See what agent has read/executed this session |
| **knowledge.db** | Insights, preferences, learnings | Edit to add debugging hints for agent |
| **tools.db** | Tool registry, usage stats | See which tools agent uses most |
| **skills.db** | Learned workflows | See patterns agent has discovered |
| **agents.db** | Specialist agent registry | View available sub-agent types |
| **plan.db** | Task tree (MasterPlan) | **Monitor decomposition strategy** |
| **logs.db** | Runtime logs (DEBUG/INFO/WARNING/ERROR) | **Debug failures, track context** ⭐ |

**Default location:** `~/.gaia/workspace/` (auto-detected)

---

## Features

### 📊 Dashboard Tab (Default View)

**Overview section:**
- Total workspace size
- Database count and stats
- Last activity timestamp
- Total errors/warnings/critical logs

**Recent activity:**
- Last 10 errors (from logs.db)
- Active tasks (from plan.db)
- Recent insights (from knowledge.db)

**Analytics:**
- Top 5 tools usage bar chart
- Context token sparkline (shows if agent stays under limit)
- Error trends

**Updates every 2 seconds by default**

---

### 🔄 Live Auto-Refresh

- **Configurable interval:** 0.5s, 1s, 2s (default), 5s, 10s
- **Toggle on/off** with checkbox
- **Auto-pauses** during editing (prevents conflicts)
- **Change detection:** New rows flash green for 2 seconds
- **Toast notifications** when data changes
- **Last updated** indicator ("2s ago", "5m ago")

**Perfect for:** Watching GAIA Code agent run in real-time

---

### 🗂️ Database Tabs

- Click any database tab to view its tables
- Sidebar shows tables with row counts
- FTS5 tables marked with 🔍 badge

---

### 📋 Data Grid

- **Pagination:** 25/50/100/250 rows per page
- **Sorting:** Click column headers
- **Filtering:** Search box filters all columns
- **Special rendering:**
  - Log levels: Colored badges (ERROR=red, WARNING=yellow, INFO=blue)
  - Task status: Badges (pending/in_progress/completed/failed)
  - JSON: Pretty-printed and syntax-highlighted
  - Timestamps: Relative ("2s ago") and absolute

---

### ✏️ Editing

**Turn off "Read-Only" mode to enable:**
- Click any cell to edit
- Delete rows (with confirmation)
- Add new rows (via modal form)
- Execute INSERT/UPDATE/DELETE in SQL console

**Safety:**
- Read-only ON by default
- Confirmations for destructive operations
- Backup button available

---

### 🔍 Advanced Tools

**SQL Console (Ctrl+E):**
- Execute any SQL query
- Syntax-aware (detects SELECT vs write operations)
- Results shown in table
- Export results to JSON/CSV

**FTS5 Search:**
- Full-text search for logs_fts, insights_fts, tools_fts
- Fast search across large datasets
- Supports MATCH syntax

**Schema Viewer:**
- See CREATE TABLE statements
- Understand database structure

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+D` | Dashboard tab |
| `Ctrl+F` | Focus search |
| `Ctrl+E` | SQL console |
| `Ctrl+Enter` | Run SQL |
| `Ctrl+R` | Refresh data |
| `Ctrl+O` | Open custom database |
| `Escape` | Close modal |

---

## Monitoring GAIA Code Agent

### Workflow

**1. Start dashboard:**
```bash
cd <your-gaia-repo>/src/gaia/electron/db-dashboard
npm start
```

**2. Run GAIA Code in another terminal:**
```bash
gaia-code "Your task" --tui simple
```

**3. Watch in dashboard:**
- **Dashboard tab:** See context usage, errors, task progress
- **logs.db tab:** See all runtime logs in real-time
- **plan.db tab:** Verify agent uses agent_query decomposition

---

### Success Indicators ✅

**In Dashboard:**
- Context usage stays under 24K tokens (blue bars)
- Error count low (< 5)
- No critical warnings

**In logs.db:**
- INFO: "Plan created with agent_query"
- No WARNING: "Context approaching limit"
- No ERROR: "Context limit exceeded"

**In plan.db:**
- 4-6 high-level tasks using agent_query
- Not 19 granular write_file tasks

---

### Warning Signs ⚠️

**Yellow bars in context graph** → Agent approaching 24K limit
**Repeated ERROR entries** → Same failure pattern (agent should adapt)
**No agent_query in plan** → Missing decomposition

---

## Troubleshooting

**Q: Dashboard shows "No databases found"**
A: Run GAIA Code once to create databases. They'll appear in `~/.gaia/workspace/`

**Q: logs.db doesn't exist**
A: Normal. Created on first run with new code (shared_state enabled).

**Q: Auto-refresh not updating**
A: Check "Auto" checkbox is ON (green), status shows "ON", not "PAUSED"

**Q: Can't edit cells**
A: Turn off "Read-Only" toggle (top right). Ensure using Electron mode (not browser).

---

## Architecture

- **Electron main process** (`main.js`) - Database access via better-sqlite3
- **Renderer** (`renderer/`) - Vanilla JS UI, no build step needed
- **IPC** (`preload.js`) - Secure bridge between main and renderer
- **Auto-refresh** (`auto-refresh.js`) - Polling + file watching
- **Dashboard** (`dashboard.js`) - Cross-database analytics

**No React, no TypeScript, no webpack** - Just npm install and run.

---

## Next Steps

1. **Install:** `cd src/gaia/electron/db-dashboard && npm install`
2. **Run:** `npm start`
3. **Verify:** See Dashboard tab with workspace stats
4. **Click tabs:** Explore memory.db, knowledge.db, etc.
5. **Run GAIA Code** in another terminal
6. **Watch:** logs.db updates in real-time

Ready to debug! 🚀
