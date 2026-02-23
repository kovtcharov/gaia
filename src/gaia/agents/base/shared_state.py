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
from datetime import datetime
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
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version INTEGER DEFAULT 1,
                    enabled BOOLEAN DEFAULT TRUE
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_id TEXT REFERENCES tools(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        """Register a new tool."""
        tool_id = str(uuid4())
        params_json = json.dumps(parameters) if parameters else None

        with self.lock:
            self.conn.execute(
                """
                INSERT INTO tools (id, name, category, source, description, parameters, code_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (tool_id, name, category, source, description, params_json, code_path),
            )

            # Add to FTS index
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
        """Find tools using FTS5 search."""
        safe_query = _sanitize_fts5_query(query)
        if safe_query is None:
            logger.debug("[ToolsDB] find skipped, empty/invalid query")
            return []  # Empty/invalid query returns no results
        with self.lock:
            cursor = self.conn.execute(
                """
                SELECT t.id, t.name, t.category, t.description
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
                    confidence REAL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id TEXT REFERENCES skills(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
                "UPDATE skills SET success_count = ?, failure_count = ?, confidence = ?, last_used = CURRENT_TIMESTAMP WHERE id = ?",
                (successes, failures, confidence, skill_id),
            )
            self.conn.commit()
        logger.debug("[SkillsDB] usage skill=%s success=%s confidence=%.2f", skill_name, success, confidence)


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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                )
            """
            )

            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT REFERENCES agents(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

            # Update confidence based on usage history
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
                "UPDATE agents SET confidence = ?, last_used = CURRENT_TIMESTAMP WHERE id = ?",
                (confidence, agent_id),
            )
            self.conn.commit()
        logger.debug("[AgentsDB] usage agent=%s success=%s confidence=%.2f", agent_name, success, confidence)


# ============================================================================
# MasterPlan: Hierarchical Task Tree
# ============================================================================


class MasterPlan:
    """
    Hierarchical task tree shared across all agents.

    Features:
    - Parent-child task relationships
    - Task ownership (which agent is working on what)
    - Progress tracking
    - Dynamic replanning
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.RLock()  # Re-entrant: create_task calls get_task/update_task while holding lock
        self._create_tables()

    def _create_tables(self):
        """Create plan tables."""
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    owner TEXT,
                    parent_id TEXT,
                    children TEXT,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                )
            """
            )
            self.conn.commit()

    def create_task(
        self, description: str, parent_id: Optional[str] = None
    ) -> TaskNode:
        """Create a new task."""
        task = TaskNode(id=str(uuid4()), description=description, parent_id=parent_id)

        with self.lock:
            self.conn.execute(
                """
                INSERT INTO tasks (id, description, status, owner, parent_id, children, result, error, created_at, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                tuple(task.to_dict().values()),
            )

            # Update parent's children list
            if parent_id:
                parent = self.get_task(parent_id)
                if parent:
                    parent.children.append(task.id)
                    self._update_task(parent)

            self.conn.commit()

        logger.info("[MasterPlan] task created id=%s parent=%s desc=%.80s", task.id, parent_id, description)
        return task

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Get a task by ID."""
        with self.lock:
            cursor = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))
                task = TaskNode.from_dict(data)
            else:
                task = None
        logger.debug("[MasterPlan] get_task id=%s found=%s", task_id, task is not None)
        return task

    def update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Update task status."""
        task = self.get_task(task_id)
        if not task:
            return

        old_status = task.status
        task.status = status
        if result:
            task.result = result
        if error:
            task.error = error

        if status == "in_progress" and not task.started_at:
            task.started_at = datetime.now()
        elif status in ("completed", "failed"):
            task.completed_at = datetime.now()

        self._update_task(task)
        logger.info("[MasterPlan] task %s: %s -> %s", task_id, old_status, status)

    def _update_task(self, task: TaskNode):
        """Update task in database."""
        with self.lock:
            self.conn.execute(
                """
                UPDATE tasks
                SET description = ?, status = ?, owner = ?, parent_id = ?, children = ?,
                    result = ?, error = ?, created_at = ?, started_at = ?, completed_at = ?
                WHERE id = ?
            """,
                tuple(list(task.to_dict().values())[1:] + [task.id]),
            )
            self.conn.commit()

    def get_all_tasks(self) -> List[TaskNode]:
        """Get all tasks."""
        with self.lock:
            cursor = self.conn.execute("SELECT * FROM tasks")
            columns = [desc[0] for desc in cursor.description]
            tasks = []
            for row in cursor.fetchall():
                data = dict(zip(columns, row))
                tasks.append(TaskNode.from_dict(data))
            task_count = len(tasks)
        logger.debug("[MasterPlan] get_all_tasks count=%d", task_count)
        return tasks

    def clear_all_tasks(self):
        """Clear all tasks from the plan. Use at session start to avoid stale tasks."""
        with self.lock:
            self.conn.execute("DELETE FROM tasks")
            self.conn.commit()
        logger.info("[MasterPlan] all tasks cleared")


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

        # Initialize plan and manifest
        self.plan = MasterPlan(workspace_dir / "plan.db")
        self.manifest = ProjectManifest()

        # Initialize call stack and message queue
        self.call_stack = AgentCallStack(max_depth=10)
        self.message_queue = MessageQueue()

        # Mark as initialized
        self._initialized = True
        logger.info("[SharedState] initialized workspace=%s", workspace_dir)

    def reset_session(self):
        """Reset session-scoped state (memory cache) while keeping knowledge."""
        # Close and recreate memory DB
        self.memory.conn.close()
        memory_path = self.workspace_dir / "memory.db"
        if memory_path.exists():
            memory_path.unlink()
        self.memory = MemoryDB(memory_path)

        # Clear call stack and message queue
        self.call_stack = AgentCallStack()
        self.message_queue = MessageQueue()

        # Keep knowledge, tools, skills, agents, plan, and manifest
        # (these persist across sessions)
        logger.info("[SharedState] session reset")


def get_shared_state(workspace_dir: Optional[Path] = None) -> SharedAgentState:
    """
    Get the singleton SharedAgentState instance.

    This ensures all agents in the recursion tree share the same state.
    """
    return SharedAgentState(workspace_dir)
