# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code IPC Server

HTTP-based IPC server that bridges the Electron renderer process
to the GAIA Code Python backend (SharedAgentState, QualityGateRunner, etc.).

Usage:
    python gaia_code_ipc.py --port 9720

Endpoints:
    POST /api  -  JSON RPC-style endpoint for all operations
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)

# ============================================================================
# Lazy imports (agent modules may not be available in all environments)
# ============================================================================

_shared_state = None
_agent = None


def get_shared_state():
    """Get or create the SharedAgentState singleton."""
    global _shared_state
    if _shared_state is None:
        try:
            from gaia.agents.gaia_code.shared_state import get_shared_state as _get_state

            _shared_state = _get_state()
        except ImportError:
            logger.warning("SharedAgentState not available, using mock state")
            _shared_state = MockSharedState()
    return _shared_state


def get_agent():
    """Get or create the GaiaCodeAgent instance."""
    global _agent
    if _agent is None:
        try:
            from gaia.agents.gaia_code.agent import GaiaCodeAgent

            _agent = GaiaCodeAgent(tui_mode="off", silent_mode=True)
        except Exception as e:
            logger.warning(f"GaiaCodeAgent not available: {e}")
            _agent = MockAgent()
    return _agent


# ============================================================================
# Mock classes for when full agent is not available
# ============================================================================


class MockSharedState:
    """Mock shared state for development/testing."""

    def __init__(self):
        self.workspace_dir = Path.home() / ".gaia" / "workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._audit_log = []
        self._tasks = []
        self._checkpoints = []

    class _Plan:
        def get_all_tasks(self):
            return []

    class _MessageQueue:
        def receive(self, *args, **kwargs):
            return []

    class _CallStack:
        def current(self):
            return None

        @property
        def stack(self):
            return []

    plan = _Plan()
    message_queue = _MessageQueue()
    call_stack = _CallStack()


class MockAgent:
    """Mock agent for development/testing."""

    def __init__(self):
        self.session_start = datetime.now()
        self.task_start = None
        self.audit_log = []
        self.shared_state = get_shared_state()

    def process_query(self, query):
        return {
            "success": True,
            "result": f"Mock response: {query}",
        }

    def get_audit_log(self):
        return self.audit_log

    def get_progress(self):
        return {
            "total_tasks": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "failed": 0,
            "progress_percent": 0,
            "elapsed_seconds": (
                datetime.now() - self.session_start
            ).total_seconds(),
        }

    def checkpoint(self):
        return {
            "success": True,
            "checkpoint_path": str(
                self.shared_state.workspace_dir / "checkpoint.json"
            ),
            "timestamp": datetime.now().isoformat(),
        }


# ============================================================================
# Database Query Helpers
# ============================================================================

DATABASE_MAP = {
    "memory": "memory.db",
    "knowledge": "knowledge.db",
    "tools": "tools.db",
    "skills": "skills.db",
    "agents": "agents.db",
    "plan": "plan.db",
}

SPECIALIST_INFO = [
    {
        "name": "Debugger",
        "description": "Debugging and error diagnosis specialist",
        "active": False,
        "tasks_completed": 0,
        "last_used": None,
    },
    {
        "name": "Security",
        "description": "Security scanning and vulnerability detection",
        "active": False,
        "tasks_completed": 0,
        "last_used": None,
    },
    {
        "name": "Refactoring",
        "description": "Code refactoring and cleanup specialist",
        "active": False,
        "tasks_completed": 0,
        "last_used": None,
    },
    {
        "name": "Testing",
        "description": "Test generation and validation specialist",
        "active": False,
        "tasks_completed": 0,
        "last_used": None,
    },
    {
        "name": "Documentation",
        "description": "Documentation generation specialist",
        "active": False,
        "tasks_completed": 0,
        "last_used": None,
    },
    {
        "name": "Performance",
        "description": "Performance analysis and optimization",
        "active": False,
        "tasks_completed": 0,
        "last_used": None,
    },
    {
        "name": "Architecture",
        "description": "Architecture analysis and design specialist",
        "active": False,
        "tasks_completed": 0,
        "last_used": None,
    },
]


def get_db_path(db_name: str) -> Optional[Path]:
    """Get the full path to a database file."""
    state = get_shared_state()
    filename = DATABASE_MAP.get(db_name)
    if not filename:
        return None
    db_path = state.workspace_dir / filename
    return db_path if db_path.exists() else None


def execute_db_query(
    db_name: str, query: str
) -> Dict[str, Any]:
    """Execute a SQL query against a database."""
    db_path = get_db_path(db_name)
    if not db_path:
        return {"error": f"Database '{db_name}' not found at workspace"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(query)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        conn.close()

        return {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "row_count": len(rows),
        }
    except Exception as e:
        return {"error": str(e)}


def get_db_tables(db_name: str) -> Dict[str, Any]:
    """Get list of tables in a database."""
    db_path = get_db_path(db_name)
    if not db_path:
        return {"tables": [], "message": f"Database '{db_name}' not found"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return {"tables": tables}
    except Exception as e:
        return {"tables": [], "error": str(e)}


def get_db_schema(db_name: str, table: str) -> Dict[str, Any]:
    """Get schema for a table."""
    db_path = get_db_path(db_name)
    if not db_path:
        return {"error": f"Database '{db_name}' not found"}

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(f"PRAGMA table_info({table})")
        columns = []
        for row in cursor.fetchall():
            columns.append(
                {
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "not_null": bool(row[3]),
                    "default": row[4],
                    "pk": bool(row[5]),
                }
            )
        conn.close()
        return {"table": table, "columns": columns}
    except Exception as e:
        return {"error": str(e)}


def browse_db_table(
    db_name: str, table: str, limit: int = 50, offset: int = 0
) -> Dict[str, Any]:
    """Browse rows from a table."""
    query = f"SELECT * FROM {table} LIMIT {limit} OFFSET {offset}"
    return execute_db_query(db_name, query)


# ============================================================================
# Request Handler
# ============================================================================


class GaiaCodeIPCHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the GAIA Code IPC server."""

    def log_message(self, format, *args):
        """Override to use Python logging."""
        logger.info(format % args)

    def do_POST(self):
        """Handle POST requests to /api."""
        if self.path != "/api":
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            request = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            self.send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        action = request.get("action", "")
        response = self.dispatch(action, request)
        self.send_json(response)

    def do_GET(self):
        """Health check endpoint."""
        if self.path == "/health":
            self.send_json({"status": "ok", "service": "gaia-code-ipc"})
        else:
            self.send_error(404, "Not Found")

    def dispatch(self, action: str, request: Dict) -> Dict[str, Any]:
        """Route request to appropriate handler."""
        handlers = {
            "status": self.handle_status,
            "chat": self.handle_chat,
            "plan": self.handle_plan,
            "quality_gates": self.handle_quality_gates,
            "db_query": self.handle_db_query,
            "db_tables": self.handle_db_tables,
            "db_schema": self.handle_db_schema,
            "db_browse": self.handle_db_browse,
            "audit_log": self.handle_audit_log,
            "codebase_index": self.handle_codebase_index,
            "specialists": self.handle_specialists,
            "metrics": self.handle_metrics,
            "checkpoint_list": self.handle_checkpoint_list,
            "checkpoint_create": self.handle_checkpoint_create,
            "checkpoint_restore": self.handle_checkpoint_restore,
            "call_stack": self.handle_call_stack,
            "messages": self.handle_messages,
        }

        handler = handlers.get(action)
        if not handler:
            return {"error": f"Unknown action: {action}"}

        try:
            return handler(request)
        except Exception as e:
            logger.error(f"Error handling {action}: {e}", exc_info=True)
            return {"error": str(e)}

    # ========================================================================
    # Action Handlers
    # ========================================================================

    def handle_status(self, request: Dict) -> Dict:
        """Get overall agent status."""
        agent = get_agent()
        state = get_shared_state()

        progress = agent.get_progress() if hasattr(agent, "get_progress") else {}

        # Get tasks
        tasks = []
        try:
            all_tasks = state.plan.get_all_tasks()
            tasks = [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "owner": t.owner,
                    "created_at": t.created_at.isoformat()
                    if hasattr(t.created_at, "isoformat")
                    else str(t.created_at),
                }
                for t in all_tasks
            ]
        except Exception:
            pass

        # Get call stack info
        recursion_depth = 0
        try:
            if hasattr(state, "call_stack") and hasattr(state.call_stack, "stack"):
                recursion_depth = len(state.call_stack.stack)
        except Exception:
            pass

        progress["recursion_depth"] = recursion_depth
        progress["current_step"] = progress.get("completed", 0) + progress.get(
            "in_progress", 0
        )

        return {
            "connected": True,
            "progress": progress,
            "tasks": tasks,
            "quality_gates": {},
            "specialists": SPECIALIST_INFO,
        }

    def handle_chat(self, request: Dict) -> Dict:
        """Process a chat message."""
        message = request.get("message", "")
        if not message:
            return {"error": "No message provided"}

        agent = get_agent()

        # Run in a thread to avoid blocking
        result = {"reply": None, "error": None}

        def run_task():
            try:
                task_result = agent.process_query(message)
                result["reply"] = task_result.get("result", "Task completed.")
            except Exception as e:
                result["error"] = str(e)

        thread = threading.Thread(target=run_task, daemon=True)
        thread.start()
        thread.join(timeout=60)  # Wait up to 60s

        if thread.is_alive():
            return {
                "reply": "Task started. It is running in the background. Check status for updates.",
            }

        if result["error"]:
            return {"error": result["error"]}

        return {"reply": result["reply"]}

    def handle_plan(self, request: Dict) -> Dict:
        """Get the master plan."""
        state = get_shared_state()

        try:
            tasks = state.plan.get_all_tasks()
            return {
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "status": t.status,
                        "owner": t.owner,
                        "parent_id": t.parent_id,
                        "created_at": t.created_at.isoformat()
                        if hasattr(t.created_at, "isoformat")
                        else str(t.created_at),
                    }
                    for t in tasks
                ]
            }
        except Exception as e:
            return {"tasks": [], "error": str(e)}

    def handle_quality_gates(self, request: Dict) -> Dict:
        """Get quality gate results."""
        agent = get_agent()
        try:
            if hasattr(agent, "quality_gates"):
                gates = {}
                for name, gate in agent.quality_gates.gates.items():
                    gates[name] = "pending"
                return {"quality_gates": gates}
        except Exception:
            pass
        return {"quality_gates": {"syntax": "pending", "imports": "pending", "tests": "pending"}}

    def handle_db_query(self, request: Dict) -> Dict:
        """Execute a database query."""
        db_name = request.get("database", "memory")
        query = request.get("query", "")

        if db_name == "manifest":
            return self._query_manifest(query)

        return execute_db_query(db_name, query)

    def handle_db_tables(self, request: Dict) -> Dict:
        """Get database tables."""
        db_name = request.get("database", "memory")

        if db_name == "manifest":
            return {"tables": ["files", "apis", "schemas", "decisions", "dependencies"]}

        return get_db_tables(db_name)

    def handle_db_schema(self, request: Dict) -> Dict:
        """Get table schema."""
        db_name = request.get("database", "memory")
        table = request.get("table", "")
        return get_db_schema(db_name, table)

    def handle_db_browse(self, request: Dict) -> Dict:
        """Browse database table."""
        db_name = request.get("database", "memory")
        table = request.get("table", "")
        limit = request.get("limit", 50)
        offset = request.get("offset", 0)
        return browse_db_table(db_name, table, limit, offset)

    def handle_audit_log(self, request: Dict) -> Dict:
        """Get audit log entries."""
        agent = get_agent()
        limit = request.get("limit", 100)
        offset = request.get("offset", 0)
        filter_type = request.get("filter")

        entries = agent.get_audit_log() if hasattr(agent, "get_audit_log") else []

        # Apply filter
        if filter_type and filter_type != "all":
            entries = [
                e
                for e in entries
                if filter_type.upper() in (e.get("action_type", "") or "").upper()
            ]

        # Apply pagination
        entries = entries[offset : offset + limit]

        return {"entries": entries, "total": len(entries)}

    def handle_codebase_index(self, request: Dict) -> Dict:
        """Get codebase index data."""
        return {
            "stats": {
                "files_indexed": 0,
                "symbols_found": 0,
                "dependencies": 0,
                "issues_found": 0,
            },
            "tree": [],
            "issues": [],
            "symbols": [],
            "message": "Run 'index_codebase' tool to populate the codebase index.",
        }

    def handle_specialists(self, request: Dict) -> Dict:
        """Get specialist agent information."""
        return {"specialists": SPECIALIST_INFO}

    def handle_metrics(self, request: Dict) -> Dict:
        """Get performance metrics."""
        agent = get_agent()

        session_time = 0
        task_time = 0

        if hasattr(agent, "session_start") and agent.session_start:
            session_time = (datetime.now() - agent.session_start).total_seconds()

        if hasattr(agent, "task_start") and agent.task_start:
            task_time = (datetime.now() - agent.task_start).total_seconds()

        audit_log = agent.get_audit_log() if hasattr(agent, "get_audit_log") else []

        # Count audit events by type
        retries = sum(
            1
            for e in audit_log
            if "RETRY" in (e.get("action_type", "") or "").upper()
        )
        escalations = sum(
            1
            for e in audit_log
            if "ESCALAT" in (e.get("action_type", "") or "").upper()
        )
        gate_runs = sum(
            1
            for e in audit_log
            if "QUALITY" in (e.get("action_type", "") or "").upper()
        )

        return {
            "session_time": session_time,
            "task_time": task_time,
            "total_steps": len(audit_log),
            "gate_runs": gate_runs,
            "retries": retries,
            "escalations": escalations,
            "agent_calls": 0,
            "insights_stored": 0,
        }

    def handle_checkpoint_list(self, request: Dict) -> Dict:
        """List all checkpoints."""
        state = get_shared_state()
        checkpoint_path = state.workspace_dir / "checkpoint.json"

        checkpoints = []
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    data = json.load(f)
                checkpoints.append(
                    {
                        "id": data.get("timestamp", "unknown"),
                        "timestamp": data.get("timestamp"),
                        "task_start": data.get("task_start"),
                        "task_count": len(data.get("plan_tasks", [])),
                        "audit_count": len(data.get("audit_log", [])),
                    }
                )
            except Exception:
                pass

        return {"checkpoints": checkpoints}

    def handle_checkpoint_create(self, request: Dict) -> Dict:
        """Create a new checkpoint."""
        agent = get_agent()
        if hasattr(agent, "checkpoint"):
            return agent.checkpoint()
        return {"error": "Checkpoint not available"}

    def handle_checkpoint_restore(self, request: Dict) -> Dict:
        """Restore from a checkpoint."""
        agent = get_agent()
        if hasattr(agent, "resume_from_checkpoint"):
            success = agent.resume_from_checkpoint()
            return {"success": success}
        return {"error": "Checkpoint restore not available"}

    def handle_call_stack(self, request: Dict) -> Dict:
        """Get current call stack."""
        state = get_shared_state()
        try:
            stack = state.call_stack.stack if hasattr(state.call_stack, "stack") else []
            return {
                "stack": [
                    {
                        "agent_id": f.agent_id,
                        "depth": f.depth,
                        "task": f.task,
                        "specialist": f.specialist,
                        "status": f.status,
                    }
                    for f in stack
                ],
                "depth": len(stack),
            }
        except Exception:
            return {"stack": [], "depth": 0}

    def handle_messages(self, request: Dict) -> Dict:
        """Get pending messages."""
        state = get_shared_state()
        try:
            messages = state.message_queue.receive("user")
            return {
                "messages": [
                    {
                        "id": m.id,
                        "priority": m.priority,
                        "content": m.content,
                        "sender": m.sender,
                        "timestamp": m.timestamp.isoformat()
                        if hasattr(m.timestamp, "isoformat")
                        else str(m.timestamp),
                    }
                    for m in messages
                ]
            }
        except Exception:
            return {"messages": []}

    # ========================================================================
    # Manifest queries (not a real SQLite DB)
    # ========================================================================

    def _query_manifest(self, query: str) -> Dict:
        """Handle queries against the in-memory manifest."""
        state = get_shared_state()
        if not hasattr(state, "manifest"):
            return {"columns": [], "rows": [], "error": "Manifest not available"}

        manifest = state.manifest

        if "files" in query.lower():
            files = manifest.list_files() if hasattr(manifest, "list_files") else []
            return {
                "columns": ["path"],
                "rows": [[f] for f in files],
                "row_count": len(files),
            }
        elif "decisions" in query.lower():
            decisions = manifest.decisions if hasattr(manifest, "decisions") else []
            return {
                "columns": ["decision", "rationale", "timestamp"],
                "rows": [
                    [
                        d.get("decision", ""),
                        d.get("rationale", ""),
                        str(d.get("timestamp", "")),
                    ]
                    for d in decisions
                ],
                "row_count": len(decisions),
            }
        elif "dependencies" in query.lower():
            deps = manifest.dependencies if hasattr(manifest, "dependencies") else []
            return {
                "columns": ["dependency"],
                "rows": [[d] for d in deps],
                "row_count": len(deps),
            }
        else:
            return {
                "columns": ["info"],
                "rows": [
                    ["Manifest is an in-memory store. Query 'files', 'decisions', or 'dependencies'."]
                ],
                "row_count": 1,
            }

    # ========================================================================
    # Response Helpers
    # ========================================================================

    def send_json(self, data: Dict, status: int = 200):
        """Send a JSON response."""
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ============================================================================
# Server Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="GAIA Code IPC Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GAIA_CODE_IPC_PORT", "9720")),
        help="Port to listen on (default: 9720)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Start server
    server = HTTPServer((args.host, args.port), GaiaCodeIPCHandler)
    logger.info(f"GAIA Code IPC Server starting on {args.host}:{args.port}")
    logger.info(f"Workspace: {get_shared_state().workspace_dir}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
