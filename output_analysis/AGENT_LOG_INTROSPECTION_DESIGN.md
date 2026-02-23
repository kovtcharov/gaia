# Agent Log Introspection Design

**Date:** 2026-02-23
**Question:** Should runtime logs (DEBUG, INFO, WARNING, ERROR) be stored in a database for agent introspection?
**Answer:** Yes! This enables the LLM to understand what's happening "under its hood"

---

## Current State

### What Exists

**6 Databases in SharedAgentState:**
1. `memory.db` - Session-scoped cache (file contents, tool results)
2. `knowledge.db` - Cross-session insights and learnings
3. `tools.db` - Tool registry with semantic search
4. `skills.db` - Learned workflows
5. `agents.db` - Specialist agent registry
6. `plan.db` - Hierarchical task tree

**Audit log (in-memory only):**
- `self.audit_log = []` - List of action records (tool calls, state transitions)
- Accessible via `get_audit_log()` method
- NOT persisted to database
- NOT searchable
- Limited to session memory

**Python logging (not accessible to agent):**
```python
logger.debug("[CONTEXT] At 75% of limit")
logger.warning("Tool execution failed")
logger.error("[CONTEXT] LIMIT EXCEEDED")
```

These logs go to:
- Console (if configured)
- Log file (if configured)
- **NOT accessible to the LLM** - no tool to query them

---

## The Problem

When the agent encounters issues, it's blind to its own runtime state:

**Example scenario:**
```
Step 15: write_file fails with "Missing required arguments"
Step 16: Agent retries same tool
Step 17: Fails again
Step 18: Agent doesn't know WHY it's failing repeatedly
```

**What the agent CAN'T currently do:**
- Query its own error logs: `recall("what errors have I encountered?")`
- See context warnings: `get_context_metrics()` (added today, but incomplete)
- Understand tool execution patterns: "Have I called this tool before? What happened?"
- Debug itself: "Why am I in ERROR_RECOVERY state?"

**What would help:**
```python
# Agent introspects its own logs
logs = get_recent_logs(level="ERROR", limit=10)
# Returns: [
#   {"timestamp": "...", "level": "ERROR", "module": "agent._execute_tool",
#    "message": "Missing required arguments for write_file: file_path, content"},
#   {"timestamp": "...", "level": "ERROR", "module": "chat.sdk.send_messages",
#    "message": "Error code: 400 - prompt is too long: 200817 tokens"}
# ]

# Agent sees the pattern: output token limit exceeded
# Agent adjusts: splits large file into chunks or uses agent_query
```

---

## Proposed Solution: logs.db

### Option 1: New Database (Recommended)

Create **logs.db** as the 7th SharedAgentState database.

**Why separate DB:**
- Clean separation: logs vs operational data
- Can be truncated independently (logs can be huge)
- Different retention policy (rotate old logs)
- Query performance (logs table can grow large)

**Schema:**

```sql
CREATE TABLE runtime_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,              -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    logger_name TEXT NOT NULL,        -- e.g., "gaia.agents.base.agent"
    module TEXT NOT NULL,             -- e.g., "process_query"
    line_number INTEGER,
    message TEXT NOT NULL,
    context_tokens INTEGER,           -- If log is context-related
    step_number INTEGER,              -- Which agent step
    session_id TEXT,
    agent_name TEXT,
    extras TEXT                       -- JSON for additional structured data
);

CREATE INDEX idx_logs_timestamp ON runtime_logs(timestamp DESC);
CREATE INDEX idx_logs_level ON runtime_logs(level);
CREATE INDEX idx_logs_session ON runtime_logs(session_id);
CREATE INDEX idx_logs_step ON runtime_logs(step_number);

-- FTS5 for full-text search on messages
CREATE VIRTUAL TABLE logs_fts USING fts5(
    message,
    logger_name,
    module,
    content=runtime_logs,
    content_rowid=id
);
```

**Integration:**

```python
# shared_state.py

class LogsDB:
    """Runtime logs database for agent introspection."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            # Create schema (see above)
            self.conn.execute("""CREATE TABLE IF NOT EXISTS runtime_logs ...""")
            # ... FTS5 setup

    def log(self, level: str, logger_name: str, module: str, message: str,
            line_number: int = None, step_number: int = None,
            context_tokens: int = None, extras: Dict = None):
        """Store a log entry."""
        with self.lock:
            self.conn.execute("""
                INSERT INTO runtime_logs
                (timestamp, level, logger_name, module, line_number, message,
                 context_tokens, step_number, session_id, extras)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                level,
                logger_name,
                module,
                line_number,
                message,
                context_tokens,
                step_number,
                self.session_id,
                json.dumps(extras) if extras else None
            ))
            self.conn.commit()

    def query_logs(self, level: str = None, limit: int = 100,
                   search: str = None, since: str = None) -> List[Dict]:
        """Query logs with filters."""
        query = "SELECT * FROM runtime_logs WHERE 1=1"
        params = []

        if level:
            query += " AND level = ?"
            params.append(level)

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        if search:
            # Use FTS5 for full-text search
            query = """
                SELECT l.* FROM runtime_logs l
                JOIN logs_fts f ON l.id = f.rowid
                WHERE logs_fts MATCH ?
            """
            params = [search]

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.lock:
            cursor = self.conn.execute(query, params)
            rows = cursor.fetchall()

        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "level": row[2],
                "logger_name": row[3],
                "module": row[4],
                "line_number": row[5],
                "message": row[6],
                "context_tokens": row[7],
                "step_number": row[8],
                "extras": json.loads(row[11]) if row[11] else None
            }
            for row in rows
        ]

# In SharedAgentState.__init__:
self.logs = LogsDB(workspace_dir / "logs.db")
```

---

### Option 2: Use memory.db (Not Recommended)

Add a `logs` table to `memory.db`:
- ❌ Mixes concerns (cache vs logs)
- ❌ Session-scoped DB might be cleared, losing logs
- ❌ Harder to manage retention (logs grow faster than cache)

---

### Option 3: Use knowledge.db (Hybrid Approach)

Store **important logs** (ERROR, CRITICAL, context warnings) in `knowledge.db`:
- ✅ Cross-session persistence (learn from errors)
- ✅ Already has insights table (similar structure)
- ❌ All verbosity levels would clutter knowledge DB
- ❌ High-frequency DEBUG logs don't belong in "knowledge"

**Recommendation:** Use hybrid:
- All logs → `logs.db` (queryable, truncatable)
- Important insights from logs → `knowledge.db` (persistent learning)

---

## Custom Logging Handler

To capture ALL Python logging to the database:

```python
# shared_state.py

import logging

class DatabaseLogHandler(logging.Handler):
    """Custom logging handler that writes to logs.db."""

    def __init__(self, logs_db: LogsDB, session_id: str, agent_name: str):
        super().__init__()
        self.logs_db = logs_db
        self.session_id = session_id
        self.agent_name = agent_name
        self.step_number = 0  # Updated by agent

    def emit(self, record: logging.LogRecord):
        """Called for every log message."""
        try:
            # Extract structured data
            extras = {}
            if hasattr(record, 'context_tokens'):
                extras['context_tokens'] = record.context_tokens
            if hasattr(record, 'step_number'):
                extras['step_number'] = record.step_number

            self.logs_db.log(
                level=record.levelname,
                logger_name=record.name,
                module=record.funcName,
                message=record.getMessage(),
                line_number=record.lineno,
                step_number=getattr(record, 'step_number', self.step_number),
                context_tokens=getattr(record, 'context_tokens', None),
                extras=extras
            )
        except Exception:
            # Don't let logging failures break the agent
            pass

    def set_step(self, step: int):
        """Update current step number for log tagging."""
        self.step_number = step
```

**Usage in Agent:**

```python
# agent.py - in __init__ when shared_state is enabled

if self.shared_state:
    # Add database log handler
    self.db_log_handler = DatabaseLogHandler(
        logs_db=self.shared_state.logs,
        session_id=str(uuid4()),
        agent_name=self.__class__.__name__
    )
    # Attach to all gaia.* loggers
    for logger_name in ["gaia", "gaia.agents", "gaia.chat", "gaia.llm"]:
        logging.getLogger(logger_name).addHandler(self.db_log_handler)

# In process_query loop:
if self.shared_state:
    self.db_log_handler.set_step(steps_taken)
```

---

## Agent-Accessible Tools

Add tools for the agent to query its own logs:

```python
# gaia_code/tools.py

@tool
def get_recent_logs(
    level: Optional[str] = None,
    limit: int = 50,
    search: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve recent runtime logs for self-introspection.

    The agent can query its own logs to understand:
    - What errors occurred and when
    - Which operations succeeded/failed
    - Context usage warnings
    - Tool execution patterns

    **Use this when:**
    - Debugging repeated failures ("Why does write_file keep failing?")
    - Understanding context warnings ("Why am I seeing context alerts?")
    - Analyzing performance ("How long did compilation take?")
    - Reviewing error history ("What errors happened in the last 10 steps?")

    **Examples:**

    # Get recent errors
    get_recent_logs(level="ERROR", limit=10)

    # Search for context-related logs
    get_recent_logs(search="context limit", limit=20)

    # Get all recent activity
    get_recent_logs(limit=50)

    Args:
        level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        limit: Maximum number of log entries to return (default: 50)
        search: Full-text search query (searches message, logger_name, module)

    Returns:
        Dict with 'logs' array containing matching log entries
    """
    if not self.shared_state:
        return {"status": "error", "error": "Shared state not enabled"}

    logs = self.shared_state.logs.query_logs(
        level=level,
        limit=limit,
        search=search
    )

    return {
        "status": "success",
        "count": len(logs),
        "logs": logs,
        "summary": f"Retrieved {len(logs)} log entries" +
                   (f" with level={level}" if level else "")
    }

@tool
def analyze_errors(since_step: Optional[int] = None, limit: int = 20) -> Dict[str, Any]:
    """
    Analyze recent errors to identify patterns and root causes.

    Aggregates ERROR and CRITICAL logs, groups by error type,
    and provides actionable insights.

    **Use this when:**
    - Multiple operations are failing
    - Stuck in error recovery loop
    - Need to understand root cause of failures

    **Example:**
    analyze_errors(since_step=10)  # Errors in last 10-20 steps

    Returns:
        Dict with error_count, unique_errors, patterns, recommendations
    """
    if not self.shared_state:
        return {"status": "error", "error": "Shared state not enabled"}

    # Get ERROR and CRITICAL logs
    error_logs = self.shared_state.logs.query_logs(level="ERROR", limit=limit)
    critical_logs = self.shared_state.logs.query_logs(level="CRITICAL", limit=limit)

    all_errors = error_logs + critical_logs

    # Filter by step if requested
    if since_step is not None:
        all_errors = [
            e for e in all_errors
            if e.get("step_number", 0) >= since_step
        ]

    # Group by error message patterns
    error_patterns = {}
    for error in all_errors:
        msg = error["message"]
        # Extract error type (first line or key phrase)
        error_type = msg.split("\n")[0][:100]

        if error_type not in error_patterns:
            error_patterns[error_type] = []
        error_patterns[error_type].append(error)

    # Generate recommendations
    recommendations = []
    for pattern, occurrences in error_patterns.items():
        if len(occurrences) >= 3:
            recommendations.append(
                f"⚠️  Repeated error ({len(occurrences)}x): {pattern}\n"
                f"   Consider: decompose task with agent_query() or change approach"
            )

    return {
        "status": "success",
        "error_count": len(all_errors),
        "unique_error_types": len(error_patterns),
        "patterns": [
            {"error_type": k, "count": len(v), "first_occurrence": v[0]["timestamp"]}
            for k, v in error_patterns.items()
        ],
        "recommendations": recommendations,
        "recent_errors": all_errors[:10]  # Most recent 10
    }

@tool
def get_context_history(limit: int = 20) -> Dict[str, Any]:
    """
    Retrieve context usage history showing how context grew over time.

    Shows token counts at each step, when warnings were triggered,
    and whether compaction occurred.

    **Use this to:**
    - Understand why context warnings appeared
    - See if task needs decomposition
    - Analyze context growth patterns

    Returns:
        Dict with context_timeline showing token usage per step
    """
    if not self.shared_state:
        return {"status": "error", "error": "Shared state not enabled"}

    # Query all context-related logs
    context_logs = self.shared_state.logs.query_logs(
        search="CONTEXT",
        limit=limit
    )

    # Build timeline
    timeline = []
    for log in context_logs:
        if log.get("context_tokens"):
            timeline.append({
                "step": log.get("step_number"),
                "tokens": log["context_tokens"],
                "level": log["level"],
                "message": log["message"]
            })

    # Calculate peak usage
    peak = max((t["tokens"] for t in timeline), default=0)

    return {
        "status": "success",
        "timeline": timeline,
        "peak_tokens": peak,
        "max_allowed": self.max_input_tokens,
        "peak_percent": round((peak / self.max_input_tokens) * 100, 1),
        "recommendation": (
            "Context usage is healthy" if peak < self.WARNING_INPUT_TOKENS
            else "⚠️  Context exceeded 75%. Use agent_query() decomposition."
        )
    }
```

---

## Integration Points

### 1. Logger Setup (When SharedState Enabled)

```python
# agent.py - in __init__ after shared_state initialization

if enable_shared_state:
    from gaia.agents.base.shared_state import get_shared_state, DatabaseLogHandler

    ws_dir = Path(workspace_dir) if workspace_dir else None
    self.shared_state = get_shared_state(ws_dir)

    # Add database log handler to capture all logs
    self.session_id = str(uuid4())
    self.db_log_handler = DatabaseLogHandler(
        logs_db=self.shared_state.logs,
        session_id=self.session_id,
        agent_name=self.__class__.__name__
    )

    # Attach to gaia.* loggers
    for logger_name in ["gaia", "gaia.agents", "gaia.chat", "gaia.llm", "gaia.rag"]:
        lg = logging.getLogger(logger_name)
        lg.addHandler(self.db_log_handler)
        # Set level to capture everything (DEBUG and up)
        lg.setLevel(logging.DEBUG)
```

### 2. Update Context Logging

All context-related logs should include `context_tokens`:

```python
# agent.py - in _check_context_size()

logger.warning(
    "[CONTEXT] At 75%% of limit",
    extra={
        "context_tokens": total_tokens,
        "step_number": step_num
    }
)
# DatabaseLogHandler will capture this in logs.db
```

### 3. Log Rotation (Prevent Infinite Growth)

```python
# logs_db.py

def rotate_old_logs(self, keep_days: int = 7):
    """Remove logs older than keep_days."""
    cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()

    with self.lock:
        deleted = self.conn.execute(
            "DELETE FROM runtime_logs WHERE timestamp < ?",
            (cutoff,)
        ).rowcount
        self.conn.commit()

    logger.info(f"[LogsDB] Rotated {deleted} old log entries")
    return deleted

# Called automatically on agent initialization:
if self.shared_state:
    self.shared_state.logs.rotate_old_logs(keep_days=7)
```

---

## Database Size Considerations

### Estimated Growth

**Per agent step:**
- ~10-20 log messages (INFO, WARNING, ERROR, DEBUG)
- ~200 bytes per message
- = ~4KB per step

**100-step run:**
- 100 steps × 4KB = 400KB
- Reasonable for SQLite (can handle GB easily)

**Retention:**
- 7 days of logs at 10 runs/day × 100 steps = ~28MB
- FTS5 index adds ~30% overhead = ~36MB total
- Negligible storage cost

**Rotation:**
- Auto-delete logs >7 days old
- Keeps db size under 50MB
- Agent can still query recent week of activity

---

## Verbosity Levels in logs.db

**What gets logged:**

| Level | Frequency | Examples | Agent Should See? |
|-------|-----------|----------|-------------------|
| **DEBUG** | High (10-50/step) | "Parsing response", "Tool found", "State transition" | Yes (for debugging) |
| **INFO** | Medium (5-10/step) | "Step 5/20", "Tool executed successfully" | Yes (for progress tracking) |
| **WARNING** | Low (0-2/step) | "Context at 75%", "Tool result truncated" | **YES (critical feedback)** |
| **ERROR** | Rare (0-1/step) | "Tool failed", "Context limit exceeded", "JSON parse failed" | **YES (must see errors)** |
| **CRITICAL** | Very rare | System failures | **YES (immediate attention)** |

**All levels stored**, but agent typically queries:
- `get_recent_logs(level="ERROR")` - See what's broken
- `get_recent_logs(level="WARNING")` - See what's concerning
- `get_recent_logs(search="context")` - Debug context issues
- `analyze_errors()` - Pattern detection

---

## Example Agent Self-Introspection

### Scenario: Agent Stuck in Error Loop

```
Step 15: write_file fails
Step 16: Agent retries, fails again
Step 17: Agent becomes self-aware 👀

Agent queries its own logs:
```

```python
{
  "thought": "I've failed twice. Let me check my error logs to understand why.",
  "goal": "Debug repeated write_file failures",
  "tool": "get_recent_logs",
  "tool_args": {"level": "ERROR", "limit": 10}
}

# Returns:
{
  "logs": [
    {
      "timestamp": "2026-02-23T01:34:00",
      "level": "ERROR",
      "module": "_execute_tool",
      "message": "Missing required arguments for write_file: file_path, content",
      "step_number": 15
    },
    {
      "timestamp": "2026-02-23T01:34:41",
      "level": "ERROR",
      "module": "_execute_tool",
      "message": "Missing required arguments for write_file: file_path, content",
      "step_number": 16
    }
  ]
}

# Agent analyzes:
{
  "thought": "The error is consistent: 'Missing required arguments'. My tool calls are being truncated before completing. This is likely due to output token limit. I should split the file into chunks or use agent_query to delegate this file generation to a sub-agent.",
  "goal": "Fix truncation by decomposing",
  "tool": "agent_query",
  "tool_args": {
    "task": "Generate mcp_client.cpp with full cross-platform subprocess management. ~500 lines.",
    "specialist": "cpp-developer"
  }
}
```

**The agent figured out the problem by introspecting its own logs!**

---

### Scenario: Understanding Context Warnings

```python
{
  "thought": "I'm seeing context warnings. Let me check the history to understand growth pattern.",
  "tool": "get_context_history",
  "tool_args": {"limit": 20}
}

# Returns:
{
  "timeline": [
    {"step": 5, "tokens": 8500, "level": "DEBUG", "message": "At 25% of limit"},
    {"step": 10, "tokens": 17000, "level": "DEBUG", "message": "At 52% of limit"},
    {"step": 12, "tokens": 24600, "level": "WARNING", "message": "At 75% of limit"}
  ],
  "peak_tokens": 24600,
  "peak_percent": 75.1,
  "recommendation": "⚠️  Context exceeded 75%. Use agent_query() decomposition."
}

# Agent responds:
{
  "thought": "I'm at 75% context. I have 7 more files to generate. At current growth rate (2K tokens/file), I'll hit 100% by step 15. I should decompose the remaining files via agent_query.",
  "goal": "Prevent context exhaustion via decomposition",
  "tool": "agent_query",
  "tool_args": {
    "task": "Generate remaining 7 C++ files (console.h/cpp, json_utils.h/cpp, test_console.cpp, test_json_utils.cpp, simple_agent.cpp)"
  }
}
```

**The agent predicted context exhaustion and proactively decomposed!**

---

## Benefits of This Design

### 1. Self-Awareness
Agent can query `get_recent_logs()` to understand its own state:
- "What errors have I encountered?"
- "Why am I in ERROR_RECOVERY state?"
- "What did I do in the last 10 steps?"

### 2. Pattern Detection
Agent can call `analyze_errors()` to find repeated failures:
- "I've failed write_file 5 times with the same error"
- "All my git commits are failing with 'not a repository'"
- "Context warnings correlate with large read_file calls"

### 3. Proactive Decomposition
Agent sees context warnings in logs and reacts:
- At 75%: "I should wrap up soon or decompose"
- At 93%: "I must use agent_query NOW"

### 4. Debugging Assistance
When user asks "Why did you fail?", agent can:
```python
logs = get_recent_logs(level="ERROR", limit=5)
# Provide actual error messages, not hallucinations
```

### 5. Learning from Logs
Agent can store insights derived from log patterns:
```python
errors = analyze_errors(since_step=10)
if errors["error_count"] >= 5:
    store_insight(
        category="debugging",
        content=f"Repeated {errors['patterns'][0]['error_type']} errors. Root cause: ...",
        triggers=["write_file", "large files"]
    )
```

---

## Implementation Plan

### Phase 1: Database Setup (2 hours)
1. Create `LogsDB` class in `shared_state.py`
2. Add `logs.db` to SharedAgentState initialization
3. Implement schema with FTS5 for search
4. Add `query_logs()` method with filters

### Phase 2: Log Handler (1 hour)
1. Create `DatabaseLogHandler` class
2. Attach to gaia.* loggers when shared_state enabled
3. Implement step number tracking
4. Test that logs are captured

### Phase 3: Agent Tools (1 hour)
1. Add `get_recent_logs()` tool
2. Add `analyze_errors()` tool
3. Add `get_context_history()` tool
4. Update tool descriptions with examples

### Phase 4: Enhanced Context Logging (30 mins)
1. Add `extra={"context_tokens": ...}` to all context logs
2. Add `extra={"step_number": ...}` to tool execution logs
3. Verify logs.db contains structured data

### Phase 5: Testing (1 hour)
1. Run agent with deliberate errors
2. Call `get_recent_logs(level="ERROR")` to verify retrieval
3. Call `analyze_errors()` to verify pattern detection
4. Verify log rotation works

**Total effort:** ~5.5 hours

---

## Alternative: Lightweight Approach

If 5.5 hours is too much, do a **minimal version**:

### Just Capture to memory.db

Add a simple `logs` table to existing `memory.db`:

```python
# shared_state.py - in MemoryDB class

def _init_db(self):
    # ... existing tables ...

    # Add logs table (session-scoped, cleared on reset)
    self.conn.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            message TEXT,
            step INTEGER
        )
    """)

def store_log(self, level: str, message: str, step: int = None):
    """Store a log entry."""
    with self.lock:
        self.conn.execute(
            "INSERT INTO logs (timestamp, level, message, step) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), level, message, step)
        )
        self.conn.commit()

def get_logs(self, level: str = None, limit: int = 50) -> List[Dict]:
    """Retrieve logs."""
    query = "SELECT * FROM logs"
    params = []

    if level:
        query += " WHERE level = ?"
        params.append(level)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with self.lock:
        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()

    return [
        {"id": r[0], "timestamp": r[1], "level": r[2], "message": r[3], "step": r[4]}
        for r in rows
    ]
```

**Manually log important events:**

```python
# agent.py - after context warnings

if self.shared_state:
    self.shared_state.memory.store_log(
        level="WARNING",
        message=f"Context at {percent}% of limit ({total_tokens} tokens)",
        step=step_num
    )
```

**Add simple tool:**

```python
@tool
def get_logs(level: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
    """Get recent runtime logs."""
    if not self.shared_state:
        return {"status": "error", "error": "Shared state not enabled"}

    logs = self.shared_state.memory.get_logs(level=level, limit=limit)
    return {"status": "success", "logs": logs}
```

**Effort:** ~1 hour

**Trade-off:** Less comprehensive (manual logging) but gets 80% of the benefit.

---

## Recommendation

**Start with lightweight approach** (1 hour):
1. Add `logs` table to `memory.db`
2. Manually log key events (context warnings, errors, tool failures)
3. Add `get_logs()` tool for agent introspection
4. Test with a failing scenario

**Later upgrade to full DatabaseLogHandler** (4.5 more hours) when:
- Manual logging proves useful
- Need automatic capture of ALL logs
- Need cross-session log persistence
- Need advanced pattern detection

---

## Quick Win: Log Context Warnings NOW

Since you just added context monitoring, let's make those logs queryable immediately:

```python
# agent.py - in _check_context_size(), after each warning/error

# Add to audit log (already in memory)
self._log_audit("context_warning", {
    "level": "warning" / "emergency" / "critical",
    "tokens": total_tokens,
    "percent": percent,
    "step": step_num
})

# Also store in memory.db if shared_state enabled
if self.shared_state:
    self.shared_state.memory.store_log(
        level=level,
        message=f"Context at {percent:.1f}% ({total_tokens:,} tokens)",
        step=step_num
    )
```

**Agent can then query:**
```python
recall("context warning")  # Uses knowledge DB search
get_audit_log(limit=20)    # Gets audit entries
get_logs(level="WARNING")  # Gets memory.db logs (NEW)
```

---

## Answer to Your Question

**Should we store runtime logs for agent introspection?**
**YES!** And here's how:

**Minimal (1 hour):**
- Add `logs` table to `memory.db`
- Manually log important events
- Add `get_logs()` tool

**Complete (5.5 hours):**
- Create dedicated `logs.db`
- Auto-capture ALL Python logs via DatabaseLogHandler
- Add 3 introspection tools
- Enable full self-debugging

**Start with minimal, upgrade if useful.**

Which approach do you prefer?