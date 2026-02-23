# GAIA DB Dashboard - Quick Start

**Real-time monitoring for GAIA Code agent execution**

---

## Prerequisites

Install build tools (required for better-sqlite3):

```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

## Installation (One-Time)

```bash
cd src/gaia/electron/db-dashboard
npm install
```

Takes 2-3 minutes. Installs Electron, better-sqlite3, and chokidar.

---

## Running

### Start the Dashboard

```bash
cd /mnt/c/Users/14255/Work/gaia/src/gaia/electron/db-dashboard
npm start
```

**Dashboard opens automatically showing:**
- Default workspace: `/home/user/.gaia/workspace/`
- All 7 databases (6 existing + logs.db when created)
- Dashboard overview with stats
- Auto-refresh: ON, every 2 seconds

---

## Database Paths

**All databases live in:**
```
/home/user/.gaia/workspace/
```

**Current databases:**
- `agents.db` (24 KB)
- `knowledge.db` (56 KB) ← You can edit insights here
- `memory.db` (20 KB)
- `plan.db` (12 KB)
- `skills.db` (20 KB)
- `tools.db` (52 KB)
- `logs.db` (created on first GAIA Code run with new code)

---

## How to Use with GAIA Code

### Terminal 1: Dashboard (start first)
```bash
cd /mnt/c/Users/14255/Work/gaia/src/gaia/electron/db-dashboard
npm start
```

**Configure:**
- Ensure "Auto" is ON ✓
- Set refresh to 2s
- Click "logs.db" tab to prepare for monitoring

### Terminal 2: GAIA Code
```bash
cd /mnt/c/Users/14255/Work/gaia

# Run your task
gaia-code "Your task here" --tui simple
```

**Watch live in dashboard:**
- logs.db updates every 2s with new log entries
- Dashboard shows context usage, error count
- Changed rows flash green

---

## Key Features

**Dashboard (Overview) Tab:**
- Cross-database statistics
- Recent errors from logs.db
- Active tasks from plan.db
- Context usage sparkline chart
- Top tools usage

**logs.db Tab:**
- See all runtime logs (DEBUG, INFO, WARNING, ERROR)
- Filter by level: Click column header → filter ERROR
- Search messages: Type in search box
- Watch context warnings appear in real-time

**Auto-Refresh:**
- Configurable: 0.5s, 1s, 2s, 5s, 10s
- Toggle on/off
- Auto-pauses during editing
- "Updated: Xs ago" indicator

**Edit Mode:**
- Turn off "Read-Only" toggle
- Click any cell to edit
- Add/delete rows
- Changes save immediately

---

## What to Watch For

### ✅ Agent Using RAC Properly
- Dashboard context graph stays under 24K tokens (blue bars)
- plan.db shows 4-5 tasks with "agent_query" in descriptions
- No context warnings in logs.db

### ⚠️ Agent Needs to Adapt
- Dashboard shows yellow bars (>24K tokens)
- logs.db shows: "Context at 75% of limit"
- Agent should query get_logs() and decompose

### ❌ Agent Failing
- Red bars in context graph (>30K tokens)
- logs.db shows: "Context limit exceeded", "Emergency compaction"
- plan.db shows 19 granular tasks (not decomposed)

---

## Next Steps

1. **Install:** `npm install` (one-time, 2-3 min)
2. **Start dashboard:** `npm start`
3. **Verify:** See 6 databases in tabs (logs.db appears after first run)
4. **Run GAIA Code** in another terminal
5. **Watch:** logs.db updates in real-time

Ready to monitor! 🚀
