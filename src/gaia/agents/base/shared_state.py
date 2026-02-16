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
import sqlite3
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


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

    def get_file(self, path: str) -> Optional[str]:
        """Get cached file contents."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT content FROM file_cache WHERE path = ?", (path,)
            )
            row = cursor.fetchone()
            return row[0] if row else None

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
            self.conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS insights_fts USING fts5(
                    id, content, triggers
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

            # Add to FTS index
            self.conn.execute(
                """
                INSERT INTO insights_fts (id, content, triggers)
                VALUES (?, ?, ?)
            """,
                (insight_id, content, triggers_json or ""),
            )

            self.conn.commit()

        return insight_id

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize a query string for FTS5 MATCH.

        FTS5 treats characters like . : - @ as special syntax.
        Replace them with spaces so the query works as plain word search.
        """
        import re
        # Replace FTS5 special chars with spaces, keep alphanumeric and underscores
        sanitized = re.sub(r'[^\w\s]', ' ', query)
        # Collapse multiple spaces
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized if sanitized else query

    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search insights using FTS5 full-text search."""
        safe_query = self._sanitize_fts5_query(query)
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

    def get_preference(self, key: str) -> Optional[str]:
        """Get a user preference."""
        with self.lock:
            cursor = self.conn.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else None


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

        return tool_id

    def find_tools(self, query: str, top_k: int = 10) -> List[Dict]:
        """Find tools using FTS5 search."""
        safe_query = self._sanitize_fts5_query(query)
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

            return results


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
        self.lock = threading.Lock()
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

        return task

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Get a task by ID."""
        with self.lock:
            cursor = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))
                return TaskNode.from_dict(data)
            return None

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
            return tasks

    def clear_all_tasks(self):
        """Clear all tasks from the plan. Use at session start to avoid stale tasks."""
        with self.lock:
            self.conn.execute("DELETE FROM tasks")
            self.conn.commit()


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
            return frame

    def pop(self) -> Optional[AgentCallFrame]:
        """Pop the top agent call frame."""
        with self.lock:
            if self.stack:
                frame = self.stack.pop()
                frame.completed_at = datetime.now()
                frame.status = "completed"
                return frame
            return None

    def current(self) -> Optional[AgentCallFrame]:
        """Get the current (top) call frame."""
        with self.lock:
            return self.stack[-1] if self.stack else None


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
            return msgs

    def respond(self, message_id: str, response: str):
        """Respond to a message."""
        with self.lock:
            for msg in self.messages:
                if msg.id == message_id:
                    msg.response = response
                    break


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
            if path in self.files:
                self.files[path]["modified"] = datetime.now()
                self.files[path]["content"] = content
            else:
                self.files[path] = {
                    "content": content,
                    "created": datetime.now(),
                    "modified": datetime.now(),
                }

    def add_api(self, endpoint: str, method: str, params: Dict, response: Dict):
        """Register an API endpoint."""
        with self.lock:
            self.apis[endpoint] = {
                "method": method,
                "params": params,
                "response": response,
                "created": datetime.now(),
            }

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

    def get_file(self, path: str) -> Optional[Dict]:
        """Get file info from manifest."""
        with self.lock:
            return self.files.get(path)

    def list_files(self) -> List[str]:
        """List all files in manifest."""
        with self.lock:
            return list(self.files.keys())


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
        return cls._instance

    def __init__(self, workspace_dir: Optional[Path] = None):
        """Initialize SharedAgentState (only runs once due to singleton)."""
        # Avoid re-initialization
        if hasattr(self, "_initialized"):
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


def get_shared_state(workspace_dir: Optional[Path] = None) -> SharedAgentState:
    """
    Get the singleton SharedAgentState instance.

    This ensures all agents in the recursion tree share the same state.
    """
    return SharedAgentState(workspace_dir)
