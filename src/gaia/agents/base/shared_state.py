# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
SharedAgentState: The RAC Foundation

This is the core of Recursive Agent Composition (RAC). Every agent in the
recursion tree shares THE SAME instance of this state, enabling:
- Persistent memory across recursive calls
- Shared knowledge and learnings
- Coordinated planning via MasterPlan
- Tool/skill/agent discovery via registries
- Message passing between agents
- Full audit trail

The 7 databases that make up SharedAgentState:
1. memory.db - Session-scoped working memory cache
2. knowledge.db - Cross-session learning and insights
3. tools.db - Tool registry with semantic search
4. skills.db - Learned workflows and patterns
5. agents.db - Specialist agent registry
Plus:
6. manifest - Live project state (files, APIs, schemas)
7. plan - Hierarchical task tree (MasterPlan)
8. call_stack - Recursion tracking
9. message_queue - Async communication
"""

import json
import logging
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import re

logger = logging.getLogger(__name__)


def _sanitize_fts5_query(query: str) -> Optional[str]:
    """Sanitize a query string for FTS5 MATCH.

    FTS5 treats characters like . : - @ as special syntax.
    Replace them with spaces so the query works as plain word search.
    Uses OR semantics so partial matches still return results.

    Returns None if query is empty or contains no searchable words.
    """
    if not query or not query.strip():
        logger.debug("[FTS5] query empty/invalid, returning None")
        return None
    # Replace FTS5 special chars with spaces, keep alphanumeric and underscores
    sanitized = re.sub(r'[^\w\s]', ' ', query)
    # Collapse multiple spaces
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    if not sanitized:
        logger.debug("[FTS5] query empty/invalid, returning None")
        return None
    # Join words with OR for fuzzy matching (AND is too strict for recall)
    words = sanitized.split()
    if len(words) > 1:
        result = ' OR '.join(words)
        logger.debug("[FTS5] sanitized %r -> %r", query, result)
        return result
    logger.debug("[FTS5] sanitized %r -> %r", query, sanitized)
    return sanitized


# ============================================================================
# Data Classes for Structured State
# ============================================================================


@dataclass
class TaskNode:
    """A node in the hierarchical task tree (MasterPlan)."""

    id: str  # Unique task ID
    description: str  # What this task does
    status: str = "pending"  # pending | in_progress | completed | failed
    owner: Optional[str] = None  # Which agent is working on this
    parent_id: Optional[str] = None  # Parent task (None for root)
    children: List[str] = field(default_factory=list)  # Child task IDs
    result: Optional[str] = None  # Result when completed
    error: Optional[str] = None  # Error if failed
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "owner": self.owner,
            "parent_id": self.parent_id,
            "children": json.dumps(self.children),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
        }

    @staticmethod
    def from_dict(data: Dict) -> "TaskNode":
        """Create from dictionary."""
        return TaskNode(
            id=data["id"],
            description=data["description"],
            status=data["status"],
            owner=data["owner"],
            parent_id=data["parent_id"],
            children=json.loads(data["children"]) if data["children"] else [],
            result=data["result"],
            error=data["error"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data["completed_at"]
                else None
            ),
        )


@dataclass
class AgentCallFrame:
    """A frame in the agent call stack (tracks recursion)."""

    agent_id: str  # Unique ID for this agent instance
    parent_id: Optional[str]  # Parent agent ID (None for root)
    depth: int  # Recursion depth (0 for root)
    task: str  # Task assigned to this agent
    specialist: Optional[str]  # Specialist type (if any)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    status: str = "running"  # running | completed | failed


@dataclass
class Message:
    """A message in the agent-user message queue."""

    id: str  # Unique message ID
    priority: str  # FYI | Question | Decision
    sender: str  # Agent ID or "user"
    recipient: str  # Agent ID or "user"
    content: str  # Message content
    timestamp: datetime = field(default_factory=datetime.now)
    read: bool = False
    response: Optional[str] = None


# ============================================================================
# Database Classes
# ============================================================================


class MemoryDB:
    """
    Session-scoped working memory cache.

    Stores:
    - File contents read during this session
    - Active state and variables
    - Recent tool results

    Shared across all agents in recursion tree for fast access.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        """Create memory cache tables."""
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_cache (
                    path TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    last_accessed TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    args TEXT,
                    result TEXT,
                    timestamp TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
            """
            )
            # Working memory: arbitrary key/value facts the agent stores explicitly.
            # Persists for the session; cleared on reset_session().
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS active_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    tags TEXT,
                    stored_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    last_accessed TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    source_dir TEXT,
                    query_context TEXT
                )
            """
            )
            # Migrate existing databases that predate source_dir / query_context columns
            for col in ("source_dir", "query_context"):
                try:
                    self.conn.execute(f"ALTER TABLE active_state ADD COLUMN {col} TEXT")
                    self.conn.commit()
                except Exception:
                    pass  # Column already exists
            # Persistent conversation history across sessions.
            # Stores every user/assistant turn so sessions can be restored and
            # searched.  Only the final user+assistant exchange per query is
            # stored (not intermediate tool-call turns).
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_history (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role      TEXT NOT NULL,
                    content   TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversation_history(session_id)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conv_timestamp
                ON conversation_history(timestamp DESC)
                """
            )
            # FTS5 for searching past conversations
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS conversation_fts
                USING fts5(content, content=conversation_history, content_rowid=id)
                """
            )
            # Keep FTS in sync via triggers
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS conv_ai
                AFTER INSERT ON conversation_history BEGIN
                    INSERT INTO conversation_fts(rowid, content) VALUES (new.id, new.content);
                END
                """
            )
            self.conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS conv_ad
                AFTER DELETE ON conversation_history BEGIN
                    INSERT INTO conversation_fts(conversation_fts, rowid, content)
                    VALUES ('delete', old.id, old.content);
                END
                """
            )
            # ----------------------------------------------------------------
            # Plan tables: plans, plan_tasks, plan_task_events
            # These live in memory.db so plans are co-located with working
            # memory and searchable alongside other session data.
            # ----------------------------------------------------------------
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id           TEXT PRIMARY KEY,
                    title        TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'active',
                    project_dir  TEXT,
                    target_dir   TEXT,
                    created_at   TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    completed_at TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_tasks (
                    id           TEXT PRIMARY KEY,
                    plan_id      TEXT NOT NULL REFERENCES plans(id),
                    parent_id    TEXT,
                    title        TEXT NOT NULL,
                    description  TEXT,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    priority     INTEGER NOT NULL DEFAULT 5,
                    depth        INTEGER NOT NULL DEFAULT 0,
                    owner        TEXT,
                    created_by   TEXT,
                    result       TEXT,
                    error        TEXT,
                    dependencies TEXT,
                    order_index  INTEGER NOT NULL DEFAULT 0,
                    created_at   TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    updated_at   TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    started_at   TIMESTAMP,
                    completed_at TIMESTAMP
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plan_tasks_plan_id
                ON plan_tasks(plan_id)
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_task_events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id    TEXT NOT NULL,
                    plan_id    TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    agent_name TEXT,
                    details    TEXT,
                    timestamp  TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_plan_events_task_id
                ON plan_task_events(task_id)
                """
            )
            self.conn.commit()

    def cache_file(self, path: str, content: str):
        """Cache a file's contents."""
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO file_cache (path, content, last_accessed)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (path, content),
            )
            self.conn.commit()
        logger.debug("[MemoryDB] cached path=%s size=%d", path, len(content))

    def get_file(self, path: str) -> Optional[str]:
        """Get cached file contents."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT content FROM file_cache WHERE path = ?", (path,)
            )
            row = cursor.fetchone()
            content = row[0] if row else None
        if content is not None:
            logger.debug("[MemoryDB] cache hit path=%s", path)
        else:
            logger.debug("[MemoryDB] cache miss path=%s", path)
        return content

    def store_tool_result(self, tool_name: str, args: Dict, result: str):
        """Store a tool call result."""
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO tool_results (tool_name, args, result)
                VALUES (?, ?, ?)
            """,
                (tool_name, json.dumps(args), result),
            )
            self.conn.commit()
        logger.debug("[MemoryDB] tool result stored tool=%s", tool_name)

    def store_memory(
        self,
        key: str,
        value: str,
        tags: Optional[List[str]] = None,
        source_dir: Optional[str] = None,
        query_context: Optional[str] = None,
    ):
        """
        Store an arbitrary fact or context value under a key.

        This is the agent's working memory — used to persist important context
        across tool calls and sub-tasks within a session. Examples:
            store_memory("current_project", "~/Work/gaia")
            store_memory("auth_approach", "JWT with RS256", tags=["architecture"])
            store_memory("failing_test", "test_agent.py::test_tool_registry")

        Args:
            source_dir: The project directory gaia-code was run from.
                        Used to filter facts when recalling — cross-project facts
                        are annotated so the LLM knows they may not be relevant.
            query_context: The user query being processed when this was stored.
                           Gives the LLM context for why this fact was recorded.
        """
        tags_json = json.dumps(tags) if tags else None
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO active_state
                    (key, value, tags, stored_at, last_accessed, source_dir, query_context)
                VALUES (?, ?, ?, strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'),
                        strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'), ?, ?)
                """,
                (key, value, tags_json, source_dir, query_context),
            )
            self.conn.commit()
        logger.debug("[MemoryDB] stored key=%s source_dir=%s", key, source_dir)

    def recall_memories(
        self,
        query: Optional[str] = None,
        limit: int = 20,
        source_dir: Optional[str] = None,
    ) -> List[Dict]:
        """
        Recall memories from active_state.

        If source_dir is provided, results are split into two groups:
          - same_project: facts stored from the same directory (prioritised)
          - other_project: facts from different directories (included but annotated)

        If query is given, does a LIKE search on key and value.
        Otherwise returns the most recently stored entries.
        """
        with self.lock:
            base_select = "SELECT key, value, tags, stored_at, source_dir, query_context FROM active_state"
            if query:
                pattern = f"%{query}%"
                cursor = self.conn.execute(
                    f"""
                    {base_select}
                    WHERE key LIKE ? OR value LIKE ?
                    ORDER BY last_accessed DESC
                    LIMIT ?
                    """,
                    (pattern, pattern, limit),
                )
            else:
                cursor = self.conn.execute(
                    f"""
                    {base_select}
                    ORDER BY last_accessed DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()

        results = []
        for r in rows:
            entry = {
                "key": r[0],
                "value": r[1],
                "tags": json.loads(r[2]) if r[2] else [],
                "stored_at": r[3],
                "source_dir": r[4],
                "query_context": r[5],
                "same_project": source_dir is None or r[4] is None or r[4] == source_dir,
            }
            results.append(entry)

        logger.debug("[MemoryDB] recall query=%r source_dir=%r results=%d", query, source_dir, len(results))
        return results

    def get_memory(self, key: str) -> Optional[str]:
        """Get a specific memory by exact key."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT value FROM active_state WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if row:
                self.conn.execute(
                    "UPDATE active_state SET last_accessed = CURRENT_TIMESTAMP WHERE key = ?",
                    (key,),
                )
                self.conn.commit()
        value = row[0] if row else None
        logger.debug("[MemoryDB] get_memory key=%s found=%s", key, value is not None)
        return value

    def forget_memory(self, key: str) -> bool:
        """Remove a specific memory entry."""
        with self.lock:
            rowcount = self.conn.execute(
                "DELETE FROM active_state WHERE key = ?", (key,)
            ).rowcount
            self.conn.commit()
        logger.debug("[MemoryDB] forget key=%s deleted=%s", key, rowcount > 0)
        return rowcount > 0

    def store_conversation_turn(self, session_id: str, role: str, content: str):
        """
        Persist one conversation turn (role='user' or 'assistant') to the DB.

        Called after each process_query() completes so the exchange survives
        process restarts and is searchable across sessions.
        """
        with self.lock:
            self.conn.execute(
                "INSERT INTO conversation_history (session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            self.conn.commit()
        logger.debug("[MemoryDB] stored conversation turn session=%s role=%s", session_id, role)

    def get_conversation_history(self, session_id: str = None, limit: int = 20) -> List[Dict]:
        """
        Retrieve recent conversation turns, optionally filtered by session.

        Args:
            session_id: If given, return only turns from that session.
                        If None, return the most recent turns across all sessions.
            limit:      Maximum number of turns to return (default 20).

        Returns:
            List of dicts with keys: id, session_id, role, content, timestamp.
            Ordered oldest-first so they can be passed directly to the LLM as
            a messages array.
        """
        with self.lock:
            if session_id:
                cursor = self.conn.execute(
                    """
                    SELECT id, session_id, role, content, timestamp
                    FROM conversation_history
                    WHERE session_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (session_id, limit),
                )
            else:
                # Most recent `limit` turns across all sessions, oldest-first
                cursor = self.conn.execute(
                    """
                    SELECT id, session_id, role, content, timestamp
                    FROM (
                        SELECT id, session_id, role, content, timestamp
                        FROM conversation_history
                        ORDER BY id DESC
                        LIMIT ?
                    ) ORDER BY id ASC
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
        return [
            {"id": r[0], "session_id": r[1], "role": r[2], "content": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def search_conversations(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Full-text search across all stored conversation turns.

        Uses FTS5 so results are ranked by relevance.  Useful for the agent
        to recall what was discussed in previous sessions without loading the
        full history into context.

        Args:
            query: Search terms (FTS5 syntax supported, e.g. "authentication jwt").
            limit: Maximum results to return.

        Returns:
            List of dicts with keys: id, session_id, role, content, timestamp.
        """
        with self.lock:
            try:
                cursor = self.conn.execute(
                    """
                    SELECT c.id, c.session_id, c.role, c.content, c.timestamp
                    FROM conversation_history c
                    JOIN conversation_fts f ON c.id = f.rowid
                    WHERE conversation_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                )
                rows = cursor.fetchall()
            except Exception:
                rows = []
        logger.debug("[MemoryDB] conversation search query=%r results=%d", query, len(rows))
        return [
            {"id": r[0], "session_id": r[1], "role": r[2], "content": r[3], "timestamp": r[4]}
            for r in rows
        ]

    def clear_working_memory(self):
        """
        Clear all working/session-scoped tables while keeping the DB file.

        This preserves the DB history (browsable in the dashboard) but starts
        the agent fresh. Called by SharedAgentState.reset_session().
        Note: conversation_history is intentionally NOT cleared here — it is
        persistent across sessions by design.
        """
        with self.lock:
            self.conn.execute("DELETE FROM active_state")
            self.conn.execute("DELETE FROM file_cache")
            self.conn.execute("DELETE FROM tool_results")
            self.conn.commit()
        logger.info("[MemoryDB] working memory cleared (active_state, file_cache, tool_results)")


class LogsDB:
    """
    Runtime logs database for agent introspection.

    Stores ALL runtime logs (DEBUG, INFO, WARNING, ERROR, CRITICAL) to enable:
    - Self-debugging: Agent can query "what errors occurred?"
    - Pattern detection: Identify repeated failures
    - Context monitoring: Track token usage over time
    - Performance analysis: Measure tool execution times

    Future extensibility:
    - Code introspection: Link logs to source code locations
    - Trace visualization: Show execution flow
    - Performance profiling: Identify bottlenecks
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()
        logger.debug("[LogsDB] initialized at %s", db_path)

    def _init_db(self):
        """Initialize database schema with extensibility for future introspection."""
        with self.lock:
            # Core logs table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS runtime_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    logger_name TEXT NOT NULL,
                    module TEXT,
                    function_name TEXT,
                    line_number INTEGER,
                    message TEXT NOT NULL,
                    step_number INTEGER,
                    context_tokens INTEGER,
                    session_id TEXT,
                    agent_name TEXT,
                    extras TEXT,
                    -- Future extensibility columns:
                    source_file TEXT,        -- Link to source code (future)
                    source_line INTEGER,     -- Line in source (future)
                    execution_time_ms REAL,  -- Performance tracking (future)
                    parent_log_id INTEGER,   -- Causality chain (future)
                    trace_id TEXT            -- Distributed tracing (future)
                )
            """)

            # Indexes for fast queries
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_timestamp
                ON runtime_logs(timestamp DESC)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_level
                ON runtime_logs(level)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_session
                ON runtime_logs(session_id)
            """)

            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_logs_step
                ON runtime_logs(step_number)
            """)

            # FTS5 virtual table for full-text search
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(
                    message,
                    logger_name,
                    module,
                    function_name,
                    content=runtime_logs,
                    content_rowid=id
                )
            """)

            # Triggers to keep FTS5 in sync
            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS logs_ai AFTER INSERT ON runtime_logs BEGIN
                    INSERT INTO logs_fts(rowid, message, logger_name, module, function_name)
                    VALUES (new.id, new.message, new.logger_name, new.module, new.function_name);
                END
            """)

            self.conn.execute("""
                CREATE TRIGGER IF NOT EXISTS logs_ad AFTER DELETE ON runtime_logs BEGIN
                    DELETE FROM logs_fts WHERE rowid = old.id;
                END
            """)

            # Conversation turns table — stores full LLM inputs/outputs per step
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT NOT NULL,
                    session_id  TEXT,
                    agent_name  TEXT,
                    step_number INTEGER,
                    role        TEXT NOT NULL,  -- 'user' | 'assistant' | 'tool_call' | 'tool_result'
                    content     TEXT NOT NULL,
                    token_count INTEGER,
                    model_id    TEXT
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_conv_session
                ON conversation_turns(session_id, step_number)
            """)

            self.conn.commit()

    def log_conversation_turn(
        self,
        role: str,
        content: str,
        step_number: int = None,
        session_id: str = None,
        agent_name: str = None,
        token_count: int = None,
        model_id: str = None,
    ):
        """Store one LLM conversation turn (user prompt or assistant response)."""
        with self.lock:
            self.conn.execute(
                """INSERT INTO conversation_turns
                   (timestamp, session_id, agent_name, step_number, role, content, token_count, model_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    session_id,
                    agent_name,
                    step_number,
                    role,
                    content,
                    token_count,
                    model_id,
                ),
            )
            self.conn.commit()

    def log(
        self,
        level: str,
        logger_name: str,
        message: str,
        module: str = None,
        function_name: str = None,
        line_number: int = None,
        step_number: int = None,
        context_tokens: int = None,
        session_id: str = None,
        agent_name: str = None,
        extras: Dict = None,
    ):
        """Store a log entry with full context."""
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO runtime_logs
                (timestamp, level, logger_name, module, function_name, line_number,
                 message, step_number, context_tokens, session_id, agent_name, extras)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    level,
                    logger_name,
                    module,
                    function_name,
                    line_number,
                    message,
                    step_number,
                    context_tokens,
                    session_id,
                    agent_name,
                    json.dumps(extras) if extras else None,
                ),
            )
            self.conn.commit()

    def query_logs(
        self,
        level: str = None,
        search: str = None,
        since: str = None,
        since_step: int = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Query logs with filters and full-text search."""

        if search:
            # Use FTS5 for full-text search
            sanitized = _sanitize_fts5_query(search)
            if not sanitized:
                logger.debug("[LogsDB] empty search query")
                return []

            query = """
                SELECT l.id, l.timestamp, l.level, l.logger_name, l.module,
                       l.function_name, l.line_number, l.message, l.step_number,
                       l.context_tokens, l.session_id, l.agent_name, l.extras
                FROM runtime_logs l
                JOIN logs_fts f ON l.id = f.rowid
                WHERE logs_fts MATCH ?
            """
            params = [sanitized]

            # Add additional filters
            if level:
                query += " AND l.level = ?"
                params.append(level)

            if since:
                query += " AND l.timestamp >= ?"
                params.append(since)

            if since_step is not None:
                query += " AND l.step_number >= ?"
                params.append(since_step)

            query += " ORDER BY l.timestamp DESC LIMIT ?"
            params.append(limit)

        else:
            # Standard query without FTS
            query = "SELECT * FROM runtime_logs WHERE 1=1"
            params = []

            if level:
                query += " AND level = ?"
                params.append(level)

            if since:
                query += " AND timestamp >= ?"
                params.append(since)

            if since_step is not None:
                query += " AND step_number >= ?"
                params.append(since_step)

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
                "function_name": row[5],
                "line_number": row[6],
                "message": row[7],
                "step_number": row[8],
                "context_tokens": row[9],
                "session_id": row[10],
                "agent_name": row[11],
                "extras": json.loads(row[12]) if row[12] else None,
            }
            for row in rows
        ]

    def rotate_old_logs(self, keep_days: int = 7) -> int:
        """Remove logs older than keep_days to prevent unbounded growth."""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()

        with self.lock:
            deleted = self.conn.execute(
                "DELETE FROM runtime_logs WHERE timestamp < ?", (cutoff,)
            ).rowcount
            self.conn.commit()

        if deleted > 0:
            logger.info("[LogsDB] Rotated %d old log entries", deleted)

        return deleted

    def get_summary(self) -> Dict[str, Any]:
        """Get database summary statistics."""
        with self.lock:
            cursor = self.conn.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN level='ERROR' THEN 1 END) as errors,
                    COUNT(CASE WHEN level='WARNING' THEN 1 END) as warnings,
                    COUNT(CASE WHEN level='CRITICAL' THEN 1 END) as critical,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM runtime_logs
            """)
            row = cursor.fetchone()

        return {
            "total_logs": row[0],
            "errors": row[1],
            "warnings": row[2],
            "critical": row[3],
            "oldest_log": row[4],
            "newest_log": row[5],
        }


class DatabaseLogHandler(logging.Handler):
    """
    Custom logging handler that writes to logs.db for agent introspection.

    Captures ALL Python logging and stores in SQLite. The agent can then
    query these logs via tools to understand its runtime state and adapt strategy.
    """

    def __init__(self, logs_db: LogsDB, session_id: str, agent_name: str):
        super().__init__()
        self.logs_db = logs_db
        self.session_id = session_id
        self.agent_name = agent_name
        self.step_number = 0  # Updated by agent during execution

    def emit(self, record: logging.LogRecord):
        """Called for every log message - stores to database."""
        try:
            # Extract extras if provided via logging.log(..., extra={...})
            context_tokens = getattr(record, "context_tokens", None)
            step = getattr(record, "step_number", self.step_number)

            self.logs_db.log(
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                module=record.module,
                function_name=record.funcName,
                line_number=record.lineno,
                step_number=step,
                context_tokens=context_tokens,
                session_id=self.session_id,
                agent_name=self.agent_name,
                extras=getattr(record, "extras", None),
            )
        except Exception:
            # Don't let logging failures break the agent
            pass

    def set_step(self, step: int):
        """Update current step number for automatic log tagging."""
        self.step_number = step


class KnowledgeDB:
    """
    Cross-session persistent knowledge database.

    Stores:
    - Insights and learnings
    - User preferences
    - Error-fix patterns
    - Project conventions
    - Session summaries

    This is the agent's long-term memory.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self):
        """Create knowledge tables with FTS5 search."""
        with self.lock:
            # Insights table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    domain TEXT,
                    content TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    triggers TEXT,
                    created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0
                )
            """
            )

            # Preferences table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
            """
            )

            # Learnings table (error-fix patterns)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS learnings (
                    id TEXT PRIMARY KEY,
                    error_pattern TEXT NOT NULL,
                    fix_pattern TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
            """
            )

            # Conventions table (project-specific patterns)
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conventions (
                    id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'))
                )
            """
            )

            # FTS5 virtual table for full-text search
            # Check if FTS5 table exists with old schema (missing domain/category)
            cursor = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='insights_fts'"
            )
            if cursor.fetchone():
                # Check column count to detect old schema
                try:
                    test = self.conn.execute("SELECT id, content, triggers, domain, category FROM insights_fts LIMIT 0")
                except sqlite3.OperationalError:
                    # Old schema without domain/category - migrate
                    logger.info("[KnowledgeDB] FTS5 schema migrated (added domain/category)")
                    self.conn.execute("DROP TABLE insights_fts")
                    self.conn.execute(
                        """
                        CREATE VIRTUAL TABLE insights_fts USING fts5(
                            id, content, triggers, domain, category
                        )
                    """
                    )
                    # Re-populate FTS index from insights table
                    rows = self.conn.execute(
                        "SELECT id, content, triggers, domain, category FROM insights"
                    ).fetchall()
                    for row in rows:
                        self.conn.execute(
                            "INSERT INTO insights_fts (id, content, triggers, domain, category) VALUES (?, ?, ?, ?, ?)",
                            (row[0], row[1], row[2] or "", row[3] or "", row[4] or ""),
                        )
            else:
                self.conn.execute(
                    """
                    CREATE VIRTUAL TABLE insights_fts USING fts5(
                        id, content, triggers, domain, category
                    )
                """
                )

            self.conn.commit()

    def store_insight(
        self,
        category: str,
        content: str,
        domain: Optional[str] = None,
        triggers: Optional[List[str]] = None,
    ) -> str:
        """Store a new insight."""
        insight_id = str(uuid4())
        triggers_json = json.dumps(triggers) if triggers else None

        with self.lock:
            self.conn.execute(
                """
                INSERT INTO insights (id, category, domain, content, triggers)
                VALUES (?, ?, ?, ?, ?)
            """,
                (insight_id, category, domain, content, triggers_json),
            )

            # Add to FTS index (includes domain and category for better recall)
            self.conn.execute(
                """
                INSERT INTO insights_fts (id, content, triggers, domain, category)
                VALUES (?, ?, ?, ?, ?)
            """,
                (insight_id, content, triggers_json or "", domain or "", category or ""),
            )

            self.conn.commit()

        logger.info("[KnowledgeDB] insight stored id=%s category=%s domain=%s", insight_id, category, domain)
        return insight_id

    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search insights using FTS5 full-text search."""
        safe_query = _sanitize_fts5_query(query)
        if safe_query is None:
            logger.debug("[KnowledgeDB] recall skipped, empty/invalid query")
            return []  # Empty/invalid query returns no results
        with self.lock:
            cursor = self.conn.execute(
                """
                SELECT i.id, i.category, i.domain, i.content, i.confidence
                FROM insights i
                JOIN insights_fts fts ON i.id = fts.id
                WHERE insights_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """,
                (safe_query, top_k),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row[0],
                        "category": row[1],
                        "domain": row[2],
                        "content": row[3],
                        "confidence": row[4],
                    }
                )
            result_count = len(results)
        logger.debug("[KnowledgeDB] recall query=%r sanitized=%r results=%d", query, safe_query, result_count)
        return results

    def store_preference(self, key: str, value: str, description: Optional[str] = None):
        """Store a user preference."""
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO preferences (key, value, description)
                VALUES (?, ?, ?)
            """,
                (key, value, description),
            )
            self.conn.commit()
        logger.info("[KnowledgeDB] preference stored key=%s", key)

    def get_preference(self, key: str) -> Optional[str]:
        """Get a user preference."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            value = row[0] if row else None
        if value is not None:
            logger.debug("[KnowledgeDB] preference hit key=%s", key)
        else:
            logger.debug("[KnowledgeDB] preference miss key=%s", key)
        return value


class ToolsDB:
    """
    Tool registry with semantic search.

    Stores:
    - Tool metadata (name, description, parameters)
    - Usage statistics
    - Tool categories and tags

    Enables dynamic tool loading and semantic tool discovery.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()
        self._migrate_tables()

    def _migrate_tables(self):
        """Add new observability columns to existing tables if missing."""
        migrations = [
            ("tools", "last_used", "TIMESTAMP"),
            ("tools", "use_count", "INTEGER DEFAULT 0"),
            ("tools", "error_count", "INTEGER DEFAULT 0"),
            ("tools", "avg_duration_ms", "REAL DEFAULT 0"),
        ]
        with self.lock:
            for table, col, defn in migrations:
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
                except Exception:
                    pass  # Column already exists
            self.conn.commit()

    def _create_tables(self):
        """Create tool registry tables."""
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tools (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT NOT NULL,
                    parameters TEXT,
                    code_path TEXT,
                    created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    version INTEGER DEFAULT 1,
                    enabled BOOLEAN DEFAULT TRUE,
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    avg_duration_ms REAL DEFAULT 0
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT REFERENCES tools(id),
                    timestamp TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    success BOOLEAN,
                    duration_ms INTEGER,
                    context TEXT,
                    error TEXT
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_tags (
                    tool_id TEXT REFERENCES tools(id),
                    tag TEXT,
                    PRIMARY KEY (tool_id, tag)
                )
            """
            )

            # FTS5 for tool search
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS tools_fts USING fts5(
                    id, name, description, category
                )
            """
            )

            self.conn.commit()

    def register_tool(
        self,
        name: str,
        category: str,
        description: str,
        source: str = "core",
        parameters: Optional[Dict] = None,
        code_path: Optional[str] = None,
    ) -> str:
        """Register a new tool (idempotent: updates existing entry if name already exists)."""
        params_json = json.dumps(parameters) if parameters else None

        with self.lock:
            existing = self.conn.execute(
                "SELECT id FROM tools WHERE name = ?", (name,)
            ).fetchone()

            if existing:
                tool_id = existing[0]
                self.conn.execute(
                    """
                    UPDATE tools SET category=?, source=?, description=?, parameters=?, code_path=?
                    WHERE name=?
                    """,
                    (category, source, description, params_json, code_path, name),
                )
            else:
                tool_id = str(uuid4())
                self.conn.execute(
                    """
                    INSERT INTO tools (id, name, category, source, description, parameters, code_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tool_id, name, category, source, description, params_json, code_path),
                )
                # Add to FTS index (only for new entries)
                self.conn.execute(
                    """
                    INSERT INTO tools_fts (id, name, description, category)
                    VALUES (?, ?, ?, ?)
                    """,
                    (tool_id, name, description, category),
                )

            self.conn.commit()

        logger.info("[ToolsDB] registered name=%s category=%s", name, category)
        return tool_id

    def find_tools(self, query: str, top_k: int = 10) -> List[Dict]:
        """Find tools using FTS5 search. Returns name, category, description, and parameters."""
        safe_query = _sanitize_fts5_query(query)
        if safe_query is None:
            logger.debug("[ToolsDB] find skipped, empty/invalid query")
            return []  # Empty/invalid query returns no results
        with self.lock:
            cursor = self.conn.execute(
                """
                SELECT t.id, t.name, t.category, t.description, t.parameters
                FROM tools t
                JOIN tools_fts fts ON t.id = fts.id
                WHERE tools_fts MATCH ? AND t.enabled = TRUE
                ORDER BY rank
                LIMIT ?
            """,
                (safe_query, top_k),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "category": row[2],
                        "description": row[3],
                        "parameters": json.loads(row[4]) if row[4] else None,
                    }
                )
            result_count = len(results)
        logger.debug("[ToolsDB] find query=%r results=%d", query, result_count)
        return results

    def get_tool(self, name: str) -> Optional[Dict]:
        """Get a tool by name."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT id, name, category, source, description, parameters, enabled FROM tools WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                result = {
                    "id": row[0],
                    "name": row[1],
                    "category": row[2],
                    "source": row[3],
                    "description": row[4],
                    "parameters": json.loads(row[5]) if row[5] else None,
                    "enabled": bool(row[6]),
                }
            else:
                result = None
        logger.debug("[ToolsDB] get_tool name=%s found=%s", name, result is not None)
        return result

    def record_usage(
        self,
        tool_name: str,
        success: bool,
        duration_ms: int = 0,
        context: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Record a tool usage event."""
        with self.lock:
            # Look up tool_id from name
            cursor = self.conn.execute(
                "SELECT id FROM tools WHERE name = ?", (tool_name,)
            )
            row = cursor.fetchone()
            tool_id = row[0] if row else None

            self.conn.execute(
                """
                INSERT INTO tool_usage (tool_id, success, duration_ms, context, error)
                VALUES (?, ?, ?, ?, ?)
            """,
                (tool_id, success, duration_ms, context, error),
            )

            # Update aggregated stats on the tools row for quick dashboard access
            if tool_id:
                self.conn.execute(
                    """
                    UPDATE tools SET
                        last_used = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'),
                        use_count = use_count + 1,
                        error_count = error_count + (CASE WHEN ? THEN 0 ELSE 1 END),
                        avg_duration_ms = (avg_duration_ms * use_count + ?) / (use_count + 1)
                    WHERE id = ?
                """,
                    (success, duration_ms, tool_id),
                )

            self.conn.commit()
        logger.debug("[ToolsDB] usage tool=%s success=%s duration=%dms", tool_name, success, duration_ms)

    def get_tool_stats(self, tool_name: str) -> Dict:
        """Get usage statistics for a tool."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT id FROM tools WHERE name = ?", (tool_name,)
            )
            row = cursor.fetchone()
            if not row:
                return {"total": 0, "successes": 0, "failures": 0, "avg_duration_ms": 0}

            tool_id = row[0]
            cursor = self.conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures,
                    COALESCE(AVG(duration_ms), 0) as avg_duration
                FROM tool_usage WHERE tool_id = ?
            """,
                (tool_id,),
            )
            row = cursor.fetchone()
            stats = {
                "total": row[0],
                "successes": row[1],
                "failures": row[2],
                "avg_duration_ms": round(row[3]),
            }
        logger.debug("[ToolsDB] stats tool=%s total=%d successes=%d", tool_name, stats["total"], stats["successes"])
        return stats


class SkillsDB:
    """
    Skills database for learned workflows.

    Stores:
    - Multi-step workflow patterns
    - Success/failure rates
    - Confidence scores

    Skills are higher-level than tools - they're learned patterns of tool usage.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()
        self._migrate_tables()

    def _migrate_tables(self):
        """Add new observability columns to existing tables if missing."""
        migrations = [
            ("skills", "use_count", "INTEGER DEFAULT 0"),
        ]
        with self.lock:
            for table, col, defn in migrations:
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
                except Exception:
                    pass
            self.conn.commit()

    def _create_tables(self):
        """Create skills tables."""
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    domain TEXT,
                    steps TEXT NOT NULL,
                    tools_used TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    use_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    last_used TIMESTAMP
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT REFERENCES skills(id),
                    timestamp TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    success BOOLEAN,
                    task_description TEXT,
                    feedback TEXT
                )
            """
            )

            self.conn.commit()


    def register_skill(
        self,
        name: str,
        description: str,
        category: str,
        steps: List[Dict],
        domain: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
    ) -> str:
        """Register a new skill (learned workflow pattern)."""
        skill_id = str(uuid4())
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO skills (id, name, description, category, domain, steps, tools_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    skill_id,
                    name,
                    description,
                    category,
                    domain,
                    json.dumps(steps),
                    json.dumps(tools_used) if tools_used else None,
                ),
            )
            self.conn.commit()
        logger.info("[SkillsDB] registered name=%s category=%s steps=%d", name, category, len(steps))
        return skill_id

    def find_skills(self, category: Optional[str] = None, domain: Optional[str] = None) -> List[Dict]:
        """Find skills by category and/or domain."""
        with self.lock:
            query = "SELECT id, name, description, category, domain, steps, confidence FROM skills WHERE 1=1"
            params = []
            if category:
                query += " AND category = ?"
                params.append(category)
            if domain:
                query += " AND domain = ?"
                params.append(domain)
            query += " ORDER BY confidence DESC"

            cursor = self.conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "category": row[3],
                    "domain": row[4],
                    "steps": json.loads(row[5]),
                    "confidence": row[6],
                })
            result_count = len(results)
        logger.debug("[SkillsDB] find category=%s domain=%s results=%d", category, domain, result_count)
        return results

    def record_usage(self, skill_name: str, success: bool, task_description: Optional[str] = None, feedback: Optional[str] = None):
        """Record skill usage and update confidence."""
        with self.lock:
            cursor = self.conn.execute("SELECT id, success_count, failure_count FROM skills WHERE name = ?", (skill_name,))
            row = cursor.fetchone()
            if not row:
                return
            skill_id, successes, failures = row[0], row[1], row[2]

            self.conn.execute(
                "INSERT INTO skill_usage (skill_id, success, task_description, feedback) VALUES (?, ?, ?, ?)",
                (skill_id, success, task_description, feedback),
            )

            # Update counts and confidence
            if success:
                successes += 1
            else:
                failures += 1
            total = successes + failures
            confidence = successes / total if total > 0 else 0.5

            self.conn.execute(
                """UPDATE skills SET success_count = ?, failure_count = ?, confidence = ?,
                   use_count = use_count + 1,
                   last_used = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime') WHERE id = ?""",
                (successes, failures, confidence, skill_id),
            )
            self.conn.commit()
        logger.debug("[SkillsDB] usage skill=%s success=%s confidence=%.2f", skill_name, success, confidence)


# ============================================================================
# AgentsDB: Specialist Agent Registry
# ============================================================================


class AgentsDB:
    """
    Specialist agent registry.

    Stores:
    - Agent metadata (name, capabilities, system prompt)
    - Usage statistics
    - Confidence scores

    Enables semantic agent discovery and auto-selection.
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._create_tables()
        self._migrate_tables()

    def _migrate_tables(self):
        """Add new observability columns to existing tables if missing."""
        migrations = [
            ("agents", "use_count", "INTEGER DEFAULT 0"),
            ("agents", "success_count", "INTEGER DEFAULT 0"),
            ("agents", "failure_count", "INTEGER DEFAULT 0"),
        ]
        with self.lock:
            for table, col, defn in migrations:
                try:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")
                except Exception:
                    pass
            self.conn.commit()

    def _create_tables(self):
        """Create agent registry tables."""
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL,
                    capabilities TEXT,
                    system_prompt TEXT,
                    tool_packs TEXT,
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT REFERENCES agents(id),
                    timestamp TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')),
                    success BOOLEAN,
                    task_type TEXT,
                    duration_ms INTEGER
                )
            """
            )

            self.conn.commit()

    def register_agent(
        self,
        name: str,
        description: str,
        capabilities: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        tool_packs: Optional[List[str]] = None,
    ) -> str:
        """Register a specialist agent."""
        agent_id = str(uuid4())
        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO agents (id, name, description, capabilities, system_prompt, tool_packs)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    agent_id,
                    name,
                    description,
                    json.dumps(capabilities) if capabilities else None,
                    system_prompt,
                    json.dumps(tool_packs) if tool_packs else None,
                ),
            )
            self.conn.commit()
        logger.info("[AgentsDB] registered name=%s", name)
        return agent_id

    def find_agent(self, name: str) -> Optional[Dict]:
        """Find an agent by name."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT id, name, description, capabilities, system_prompt, tool_packs, confidence FROM agents WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if row:
                result = {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "capabilities": json.loads(row[3]) if row[3] else None,
                    "system_prompt": row[4],
                    "tool_packs": json.loads(row[5]) if row[5] else None,
                    "confidence": row[6],
                }
            else:
                result = None
        logger.debug("[AgentsDB] find_agent name=%s found=%s", name, result is not None)
        return result

    def list_agents(self) -> List[Dict]:
        """List all registered agents."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT id, name, description, capabilities, confidence FROM agents ORDER BY confidence DESC"
            )
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "capabilities": json.loads(row[3]) if row[3] else None,
                    "confidence": row[4],
                })
            result_count = len(results)
        logger.debug("[AgentsDB] list_agents count=%d", result_count)
        return results

    def record_usage(self, agent_name: str, success: bool, task_type: Optional[str] = None, duration_ms: int = 0):
        """Record agent usage and update confidence."""
        with self.lock:
            cursor = self.conn.execute("SELECT id FROM agents WHERE name = ?", (agent_name,))
            row = cursor.fetchone()
            if not row:
                return
            agent_id = row[0]

            self.conn.execute(
                "INSERT INTO agent_usage (agent_id, success, task_type, duration_ms) VALUES (?, ?, ?, ?)",
                (agent_id, success, task_type, duration_ms),
            )

            cursor = self.conn.execute(
                """
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
                FROM agent_usage WHERE agent_id = ?
            """,
                (agent_id,),
            )
            stats = cursor.fetchone()
            confidence = stats[1] / stats[0] if stats[0] > 0 else 0.5

            self.conn.execute(
                """UPDATE agents SET confidence = ?,
                   last_used = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'),
                   use_count = use_count + 1,
                   success_count = success_count + ?,
                   failure_count = failure_count + ?
                   WHERE id = ?""",
                (confidence, 1 if success else 0, 0 if success else 1, agent_id),
            )
            self.conn.commit()
        logger.debug("[AgentsDB] usage agent=%s success=%s confidence=%.2f", agent_name, success, confidence)


# ============================================================================
# MasterPlan: Hierarchical Task Tree (stored in memory.db)
# ============================================================================


class MasterPlan:
    """
    Hierarchical task tree shared across all agents.

    Plans and tasks are stored in memory.db (plans, plan_tasks, plan_task_events
    tables) rather than a separate plan.db.  This co-locates plan data with
    other working memory and keeps the file count low.

    Features:
    - plans table: top-level goal records with status tracking
    - plan_tasks table: rich task tree (milestones → tasks → subtasks)
    - plan_task_events table: full audit trail of all status transitions
    - Multi-agent: owner/created_by columns track which agent worked on what
    - Dynamic replanning: any task can gain subtasks at any time
    """

    def __init__(self, memory_db: "MemoryDB"):
        self.memory_db = memory_db
        # Reuse MemoryDB's lock — all plan ops share the same connection+lock
        self.lock = memory_db.lock

    # ------------------------------------------------------------------
    # Plan lifecycle
    # ------------------------------------------------------------------

    def create_plan(
        self,
        title: str,
        project_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
    ) -> str:
        """Create a new top-level plan.  Returns the plan_id."""
        plan_id = str(uuid4())
        with self.lock:
            self.memory_db.conn.execute(
                """
                INSERT INTO plans (id, title, status, project_dir, target_dir)
                VALUES (?, ?, 'active', ?, ?)
                """,
                (plan_id, title[:500], project_dir, target_dir),
            )
            self.memory_db.conn.commit()
        logger.info("[MasterPlan] plan created id=%s title=%.80s", plan_id, title)
        return plan_id

    def complete_plan(self, plan_id: str):
        """Mark a plan as completed."""
        with self.lock:
            self.memory_db.conn.execute(
                """
                UPDATE plans SET status = 'completed',
                    completed_at = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')
                WHERE id = ?
                """,
                (plan_id,),
            )
            self.memory_db.conn.commit()
        logger.info("[MasterPlan] plan completed id=%s", plan_id)

    def abandon_plan(self, plan_id: str):
        """Mark a plan as abandoned."""
        with self.lock:
            self.memory_db.conn.execute(
                "UPDATE plans SET status = 'abandoned' WHERE id = ?",
                (plan_id,),
            )
            self.memory_db.conn.commit()
        logger.info("[MasterPlan] plan abandoned id=%s", plan_id)

    # ------------------------------------------------------------------
    # Task creation
    # ------------------------------------------------------------------

    def create_task(
        self,
        plan_id: str,
        title: str,
        description: Optional[str] = None,
        parent_id: Optional[str] = None,
        depth: Optional[int] = None,
        priority: int = 5,
        created_by: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """Create a task in the plan tree.  Returns the task_id.

        Args:
            plan_id:     The plan this task belongs to.
            title:       Short imperative description (shown in dashboard).
            description: Optional detailed description.
            parent_id:   Parent task ID (None = top-level milestone).
            depth:       Tree depth; auto-computed from parent if omitted.
            priority:    1 (highest) to 10 (lowest).
            created_by:  Agent name that created this task.
            dependencies: Task IDs that must complete before this one.
        """
        task_id = str(uuid4())

        # Auto-compute depth from parent
        if depth is None:
            if parent_id:
                parent = self.get_task(parent_id)
                depth = (parent["depth"] + 1) if parent else 1
            else:
                depth = 0

        with self.lock:
            # Compute position among siblings
            order_index = self.memory_db.conn.execute(
                "SELECT COUNT(*) FROM plan_tasks WHERE plan_id = ? AND parent_id IS ?",
                (plan_id, parent_id),
            ).fetchone()[0]

            self.memory_db.conn.execute(
                """
                INSERT INTO plan_tasks
                    (id, plan_id, parent_id, title, description, status, priority,
                     depth, created_by, dependencies, order_index)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    plan_id,
                    parent_id,
                    title[:500],
                    description,
                    priority,
                    depth,
                    created_by,
                    json.dumps(dependencies) if dependencies else None,
                    order_index,
                ),
            )
            self.memory_db.conn.execute(
                """
                INSERT INTO plan_task_events (task_id, plan_id, event_type, agent_name, details)
                VALUES (?, ?, 'created', ?, ?)
                """,
                (task_id, plan_id, created_by, f"Task created: {title[:200]}"),
            )
            self.memory_db.conn.commit()

        logger.info(
            "[MasterPlan] task created id=%s plan=%s depth=%d title=%.80s",
            task_id, plan_id, depth, title,
        )
        return task_id

    # ------------------------------------------------------------------
    # Task status transitions
    # ------------------------------------------------------------------

    def assign_task(self, task_id: str, owner: str):
        """Assign a task to an agent (status stays pending)."""
        with self.lock:
            row = self.memory_db.conn.execute(
                "SELECT plan_id FROM plan_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if not row:
                return
            plan_id = row[0]
            self.memory_db.conn.execute(
                """
                UPDATE plan_tasks SET owner = ?,
                    updated_at = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')
                WHERE id = ?
                """,
                (owner, task_id),
            )
            self.memory_db.conn.execute(
                "INSERT INTO plan_task_events (task_id, plan_id, event_type, agent_name) VALUES (?, ?, 'assigned', ?)",
                (task_id, plan_id, owner),
            )
            self.memory_db.conn.commit()
        logger.debug("[MasterPlan] task assigned id=%s owner=%s", task_id, owner)

    def start_task(self, task_id: str, owner: Optional[str] = None):
        """Mark a task as in_progress."""
        self.update_task_status(task_id, "in_progress", owner=owner)

    def complete_task(self, task_id: str, result: Optional[str] = None, agent: Optional[str] = None):
        """Mark a task as completed."""
        self.update_task_status(task_id, "completed", result=result, owner=agent)

    def fail_task(self, task_id: str, error: Optional[str] = None, agent: Optional[str] = None):
        """Mark a task as failed."""
        self.update_task_status(task_id, "failed", error=error, owner=agent)

    def block_task(self, task_id: str, reason: Optional[str] = None):
        """Mark a task as blocked."""
        self.update_task_status(task_id, "blocked", error=reason)

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
        owner: Optional[str] = None,
    ):
        """General-purpose task status update with event recording."""
        with self.lock:
            row = self.memory_db.conn.execute(
                "SELECT plan_id, status FROM plan_tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if not row:
                return
            plan_id, old_status = row[0], row[1]

            # Build started_at / completed_at expressions
            started_at_sql = (
                "strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')"
                if status == "in_progress" else "started_at"
            )
            completed_at_sql = (
                "strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime')"
                if status in ("completed", "failed", "cancelled") else "completed_at"
            )

            self.memory_db.conn.execute(
                f"""
                UPDATE plan_tasks SET
                    status = ?,
                    result = COALESCE(?, result),
                    error  = COALESCE(?, error),
                    owner  = COALESCE(?, owner),
                    updated_at   = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime'),
                    started_at   = {started_at_sql},
                    completed_at = {completed_at_sql}
                WHERE id = ?
                """,
                (status, result, error, owner, task_id),
            )
            self.memory_db.conn.execute(
                """
                INSERT INTO plan_task_events
                    (task_id, plan_id, event_type, agent_name, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (task_id, plan_id, status, owner, result or error),
            )
            self.memory_db.conn.commit()
        logger.info("[MasterPlan] task %s: %s -> %s", task_id, old_status, status)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get a single task by ID."""
        with self.lock:
            row = self.memory_db.conn.execute(
                """
                SELECT id, plan_id, parent_id, title, description, status, priority,
                       depth, owner, created_by, result, error, dependencies,
                       order_index, created_at, updated_at, started_at, completed_at
                FROM plan_tasks WHERE id = ?
                """,
                (task_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_plan_tasks(self, plan_id: str) -> List[Dict]:
        """Return all tasks for a plan, ordered by depth then position."""
        with self.lock:
            rows = self.memory_db.conn.execute(
                """
                SELECT id, plan_id, parent_id, title, description, status, priority,
                       depth, owner, created_by, result, error, dependencies,
                       order_index, created_at, updated_at, started_at, completed_at
                FROM plan_tasks WHERE plan_id = ?
                ORDER BY depth ASC, order_index ASC, created_at ASC
                """,
                (plan_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_active_plan(self) -> Optional[Dict]:
        """Return the most recent active plan with all its tasks."""
        with self.lock:
            row = self.memory_db.conn.execute(
                """
                SELECT id, title, status, project_dir, target_dir, created_at, completed_at
                FROM plans ORDER BY created_at DESC, rowid DESC LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        plan_id = row[0]
        tasks = self.get_plan_tasks(plan_id)
        return {
            "id": row[0],
            "title": row[1],
            "status": row[2],
            "project_dir": row[3],
            "target_dir": row[4],
            "created_at": row[5],
            "completed_at": row[6],
            "tasks": tasks,
        }

    def get_summary(self) -> str:
        """Return a compact text summary of the active plan for LLM context injection.

        Renders in DFS order (parent then its children) so the tree structure
        is preserved in the text — subtasks appear directly under their milestone.
        """
        plan = self.get_active_plan()
        if not plan:
            return ""
        tasks = plan["tasks"]
        if not tasks:
            return ""

        status_icons = {
            "pending": "○", "in_progress": "◉", "completed": "✓",
            "failed": "✗", "blocked": "⊘", "cancelled": "⊝",
        }

        # Build parent→children map for DFS traversal
        children_map: Dict[Optional[str], List[Dict]] = {}
        for t in tasks:
            pid = t["parent_id"]
            children_map.setdefault(pid, []).append(t)
        # Sort each sibling group by order_index then created_at
        for pid in children_map:
            children_map[pid].sort(key=lambda t: (t["order_index"], t["created_at"] or ""))

        lines = [f"## Active Plan: {plan['title'][:120]}"]

        def render(task_id: Optional[str], depth: int):
            for t in children_map.get(task_id, []):
                indent = "  " * depth
                icon = status_icons.get(t["status"], "○")
                agent_note = f" [{t['owner']}]" if t["owner"] else ""
                lines.append(f"{indent}{icon} {t['title']}{agent_note}")
                render(t["id"], depth + 1)

        render(None, 0)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Backward compatibility (used by agent.py checkpoint + TUI)
    # ------------------------------------------------------------------

    def clear_all_tasks(self):
        """Delete all plans and tasks (used in tests / session reset)."""
        with self.lock:
            self.memory_db.conn.execute("DELETE FROM plan_task_events")
            self.memory_db.conn.execute("DELETE FROM plan_tasks")
            self.memory_db.conn.execute("DELETE FROM plans")
            self.memory_db.conn.commit()
        logger.debug("[MasterPlan] all tasks cleared")

    def get_all_tasks(self) -> List["TaskNode"]:
        """Return tasks from the active plan as TaskNode objects (backward compat)."""
        plan = self.get_active_plan()
        if not plan:
            return []
        result = []
        for t in plan["tasks"]:
            node = TaskNode(
                id=t["id"],
                description=t["title"],
                status=t["status"],
                owner=t["owner"],
                parent_id=t["parent_id"],
            )
            result.append(node)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row) -> Dict:
        return {
            "id":           row[0],
            "plan_id":      row[1],
            "parent_id":    row[2],
            "title":        row[3],
            "description":  row[4],
            "status":       row[5],
            "priority":     row[6],
            "depth":        row[7],
            "owner":        row[8],
            "created_by":   row[9],
            "result":       row[10],
            "error":        row[11],
            "dependencies": json.loads(row[12]) if row[12] else [],
            "order_index":  row[13],
            "created_at":   row[14],
            "updated_at":   row[15],
            "started_at":   row[16],
            "completed_at": row[17],
        }


# ============================================================================
# AgentCallStack: Recursion Tracking
# ============================================================================


class AgentCallStack:
    """
    Tracks the agent recursion call stack.

    Features:
    - Track who spawned whom
    - Prevent infinite recursion (max_depth)
    - Full audit trail
    """

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.stack: List[AgentCallFrame] = []
        self.lock = threading.Lock()

    def push(
        self, task: str, specialist: Optional[str] = None
    ) -> Optional[AgentCallFrame]:
        """Push a new agent call frame onto the stack."""
        with self.lock:
            depth = len(self.stack)

            if depth >= self.max_depth:
                logger.warning("[CallStack] max depth %d reached, push rejected", self.max_depth)
                return None  # Max recursion depth reached

            parent_id = self.stack[-1].agent_id if self.stack else None
            frame = AgentCallFrame(
                agent_id=str(uuid4()),
                parent_id=parent_id,
                depth=depth,
                task=task,
                specialist=specialist,
            )

            self.stack.append(frame)
        logger.info("[CallStack] push depth=%d task=%.80s specialist=%s", depth, task, specialist)
        return frame

    def pop(self) -> Optional[AgentCallFrame]:
        """Pop the top agent call frame."""
        with self.lock:
            if self.stack:
                frame = self.stack.pop()
                frame.completed_at = datetime.now()
                frame.status = "completed"
                depth = frame.depth
            else:
                frame = None
                depth = -1
        if frame:
            logger.info("[CallStack] pop depth=%d status=%s", depth, frame.status)
        return frame

    def current(self) -> Optional[AgentCallFrame]:
        """Get the current (top) call frame."""
        with self.lock:
            frame = self.stack[-1] if self.stack else None
        if frame:
            logger.debug("[CallStack] current depth=%d task=%.80s", frame.depth, frame.task)
        else:
            logger.debug("[CallStack] current: stack empty")
        return frame


# ============================================================================
# MessageQueue: Async Communication
# ============================================================================


class MessageQueue:
    """
    Non-blocking message queue for agent <-> user and agent <-> agent communication.

    Features:
    - Three priority levels (FYI, Question, Decision)
    - Non-blocking send/receive
    - Message history
    """

    def __init__(self):
        self.messages: deque = deque()
        self.lock = threading.Lock()

    def send(
        self,
        content: str,
        priority: str = "FYI",
        sender: str = "agent",
        recipient: str = "user",
    ) -> str:
        """Send a message."""
        msg = Message(
            id=str(uuid4()),
            priority=priority,
            sender=sender,
            recipient=recipient,
            content=content,
        )

        with self.lock:
            self.messages.append(msg)

        logger.info("[MessageQueue] sent id=%s priority=%s %s -> %s", msg.id, priority, sender, recipient)
        return msg.id

    def receive(
        self, recipient: str = "user", priority: Optional[str] = None
    ) -> List[Message]:
        """Receive messages for a recipient."""
        with self.lock:
            msgs = []
            for msg in self.messages:
                if msg.recipient == recipient and not msg.read:
                    if priority is None or msg.priority == priority:
                        msg.read = True
                        msgs.append(msg)
            msg_count = len(msgs)
        logger.debug("[MessageQueue] receive recipient=%s messages=%d", recipient, msg_count)
        return msgs

    def respond(self, message_id: str, response: str):
        """Respond to a message."""
        with self.lock:
            for msg in self.messages:
                if msg.id == message_id:
                    msg.response = response
                    break
        logger.info("[MessageQueue] response to id=%s", message_id)


# ============================================================================
# ProjectManifest: Live Project State
# ============================================================================


class ProjectManifest:
    """
    Live project state shared across all agents.

    Tracks:
    - Files created/modified
    - API endpoints defined
    - Database schemas
    - Architecture decisions
    - Dependencies
    """

    def __init__(self):
        self.files: Dict[str, Dict] = {}  # path -> {content, created, modified}
        self.apis: Dict[str, Dict] = {}  # endpoint -> {method, params, response}
        self.schemas: Dict[str, Dict] = {}  # table -> {columns, types}
        self.decisions: List[Dict] = []  # [{decision, rationale, timestamp}]
        self.dependencies: List[str] = []  # List of dependencies
        self.lock = threading.Lock()

    def add_file(self, path: str, content: str):
        """Register a file in the manifest."""
        with self.lock:
            is_update = path in self.files
            if is_update:
                self.files[path]["modified"] = datetime.now()
                self.files[path]["content"] = content
            else:
                self.files[path] = {
                    "content": content,
                    "created": datetime.now(),
                    "modified": datetime.now(),
                }
        if is_update:
            logger.debug("[Manifest] file updated path=%s", path)
        else:
            logger.info("[Manifest] file added path=%s", path)

    def add_api(self, endpoint: str, method: str, params: Dict, response: Dict):
        """Register an API endpoint."""
        with self.lock:
            self.apis[endpoint] = {
                "method": method,
                "params": params,
                "response": response,
                "created": datetime.now(),
            }
        logger.info("[Manifest] api added %s %s", method, endpoint)

    def add_decision(self, decision: str, rationale: str):
        """Record an architecture decision."""
        with self.lock:
            self.decisions.append(
                {
                    "decision": decision,
                    "rationale": rationale,
                    "timestamp": datetime.now(),
                }
            )
        logger.info("[Manifest] decision: %.80s", decision)

    def get_file(self, path: str) -> Optional[Dict]:
        """Get file info from manifest."""
        with self.lock:
            result = self.files.get(path)
        logger.debug("[Manifest] get_file path=%s found=%s", path, result is not None)
        return result

    def list_files(self) -> List[str]:
        """List all files in manifest."""
        with self.lock:
            files = list(self.files.keys())
        logger.debug("[Manifest] list_files count=%d", len(files))
        return files


# ============================================================================
# SharedAgentState: The Complete RAC Foundation
# ============================================================================


class SharedAgentState:
    """
    The complete RAC foundation: 7 databases + manifest + plan + call stack + message queue.

    Every agent in the recursion tree gets THE SAME instance.
    Thread-safe for async operations.

    This is the heart of Recursive Agent Composition.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - only one SharedAgentState per session."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    logger.info("[SharedState] creating singleton instance")
        else:
            logger.debug("[SharedState] returning existing singleton")
        return cls._instance

    def __init__(self, workspace_dir: Optional[Path] = None):
        """Initialize SharedAgentState (only runs once due to singleton)."""
        # Avoid re-initialization
        if hasattr(self, "_initialized"):
            logger.debug("[SharedState] already initialized, skipping")
            return

        # Set up workspace directory
        if workspace_dir is None:
            workspace_dir = Path.home() / ".gaia" / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir = workspace_dir

        # Initialize databases
        self.memory = MemoryDB(workspace_dir / "memory.db")
        self.knowledge = KnowledgeDB(workspace_dir / "knowledge.db")
        self.tools = ToolsDB(workspace_dir / "tools.db")
        self.skills = SkillsDB(workspace_dir / "skills.db")
        self.agents = AgentsDB(workspace_dir / "agents.db")
        self.logs = LogsDB(workspace_dir / "logs.db")  # Runtime logs for introspection

        # Initialize plan (shares memory.db connection — no separate plan.db file)
        self.plan = MasterPlan(self.memory)
        self.manifest = ProjectManifest()

        # Initialize call stack and message queue
        self.call_stack = AgentCallStack(max_depth=10)
        self.message_queue = MessageQueue()

        # Rotate old logs to prevent unbounded growth
        try:
            deleted = self.logs.rotate_old_logs(keep_days=7)
            if deleted > 0:
                logger.debug("[SharedState] rotated %d old log entries", deleted)
        except Exception as e:
            logger.warning("[SharedState] log rotation failed: %s", e)

        # Mark as initialized
        self._initialized = True
        logger.info("[SharedState] initialized workspace=%s", workspace_dir)

    def reset_session(self):
        """
        Reset working memory for a new session while keeping all persistent knowledge.

        Clears:
        - active_state (agent's working memory notes)
        - file_cache (cached file contents)
        - tool_results (tool call history)
        - call stack and message queue

        Keeps (persistent across sessions):
        - knowledge.db  (insights, preferences, learnings)
        - tools.db      (registry + usage history)
        - skills.db     (learned workflows + usage history)
        - agents.db     (specialist registry + usage history)
        - plans/plan_tasks in memory.db (task history persists intentionally)
        """
        self.memory.clear_working_memory()

        # Clear in-process state
        self.call_stack = AgentCallStack()
        self.message_queue = MessageQueue()

        logger.info("[SharedState] session reset — working memory cleared, knowledge retained")


def get_shared_state(workspace_dir: Optional[Path] = None) -> SharedAgentState:
    """
    Get the singleton SharedAgentState instance.

    This ensures all agents in the recursion tree share the same state.
    """
    return SharedAgentState(workspace_dir)
