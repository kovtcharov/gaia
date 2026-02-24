# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code Tools: The RAC Mechanism

Key tools:
- agent_query(): The core RAC mechanism - spawn sub-agents recursively
- recall(): Query knowledge DB for past context
- find_tool(): Semantic tool search
- store_insight(): Persist learnings to knowledge DB

These tools enable recursive decomposition and persistent memory.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from gaia.agents.base.tools import tool

from .shared_state import get_shared_state

logger = logging.getLogger(__name__)


class GaiaCodeTools:
    """Mixin providing GAIA Code-specific tools."""

    def register_gaia_code_tools(self):
        """Register GAIA Code tools using @tool decorator."""
        # Tools are registered via @tool decorator
        # This method defines them in the agent's scope

        @tool
        def agent_query(task: str, specialist: Optional[str] = None, max_depth: Optional[int] = None) -> Dict[str, Any]:
            """
            Delegate a subtask to a sub-agent with FRESH CONTEXT (200K tokens).

            **USE THIS FOR:**
            - Generating files >500 lines
            - Multi-file tasks (>3 related files)
            - Complex components requiring deep focus
            - When approaching context limits

            **BENEFITS:**
            - Sub-agent gets dedicated 200K token context
            - Parent stays context-lean (only coordinates)
            - Better code quality (sub-agent focuses on one thing)
            - Prevents context exhaustion

            **EXAMPLES:**

            # Generate large implementation file
            agent_query(task="Generate agent.cpp with processQuery() loop, 5-state machine, plan execution. ~600 lines. Match Python agent.py architecture.", specialist="cpp-developer")

            # Generate multiple test files
            agent_query(task="Generate all unit test files (test_agent.cpp, test_tool_registry.cpp, test_mcp_client.cpp). Use GoogleTest. Aim for 80+ tests total.", specialist="test-engineer")

            # Generate related headers
            agent_query(task="Generate C++ type system: types.h (enums, structs), json_utils.h (parsing), tool_registry.h (registration). 3 files, ~400 lines total.", specialist="cpp-developer")

            **Args:**
            - task: Detailed description of what the sub-agent should do
            - specialist: Optional specialist agent type (e.g., "cpp-developer", "test-engineer")
            - max_depth: Maximum recursion depth (default: 3)
            """
            return self.tool_agent_query(task, specialist, max_depth)

        @tool
        def recall(query: str, top_k: int = 5) -> Dict[str, Any]:
            """Query knowledge DB for past context using full-text search."""
            return self.tool_recall(query, top_k)

        @tool
        def find_tool(query: str, top_k: int = 10) -> Dict[str, Any]:
            """Search for tools by natural language query."""
            return self.tool_find_tool(query, top_k)

        @tool
        def store_insight(category: str, content: str, domain: Optional[str] = None, triggers: Optional[List[str]] = None) -> Dict[str, Any]:
            """Store a learning or insight to knowledge DB."""
            return self.tool_store_insight(category, content, domain, triggers)

        @tool
        def remember(key: str, value: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
            """
            Store an important fact or context value in working memory (memory.db).

            Use this to persist any piece of context you'll need later in this session
            or that a sub-agent should be able to retrieve. Memory survives across
            tool calls and recursive agent_query() calls.

            **EXAMPLES — store anything important:**

            # Project context
            remember(key="project_root", value="/mnt/c/Users/14255/Work/gaia")
            remember(key="active_branch", value="gaia-v2")

            # Architecture decisions
            remember(key="auth_approach", value="JWT with RS256 signed tokens", tags=["architecture"])
            remember(key="db_schema", value="users(id, email, hashed_pw), sessions(token, user_id)")

            # Current task state
            remember(key="files_created", value="agent.py, test_agent.py, types.ts")
            remember(key="next_step", value="run pytest and fix any failures")

            # Error patterns learned this session
            remember(key="error_sqlite_lock", value="Use check_same_thread=False in connect()")

            Args:
                key: Unique identifier for this memory (use descriptive names)
                value: The content to store (facts, decisions, paths, code snippets, etc.)
                tags: Optional category tags for easier recall (e.g., ["architecture", "error"])
            """
            return self.tool_remember(key, value, tags)

        @tool
        def recall_memory(query: Optional[str] = None, key: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
            """
            Retrieve stored memories from working memory (memory.db).

            Use this at the START of tasks to orient yourself, or any time you need
            to recall context stored earlier in the session.

            **EXAMPLES:**

            # Recall everything (orient at task start)
            recall_memory()

            # Search by keyword
            recall_memory(query="auth")
            recall_memory(query="error")

            # Get a specific memory by key
            recall_memory(key="project_root")
            recall_memory(key="auth_approach")

            Args:
                query: Keyword to search for in keys and values (optional)
                key: Exact key to retrieve (optional, takes priority over query)
                limit: Maximum number of results (default: 10)
            """
            return self.tool_recall_memory(query, key, limit)

        @tool
        def forget_memory(key: str) -> Dict[str, Any]:
            """
            Remove an entry from working memory.

            Use when a stored fact is no longer accurate or relevant.

            Args:
                key: Exact key to remove
            """
            return self.tool_forget_memory(key)

        @tool
        def search_conversations(query: str, limit: int = 10) -> Dict[str, Any]:
            """
            Search past conversation history stored in memory.db.

            Uses full-text search (FTS5) across all previous sessions so you
            can recall what was discussed, decided, or built in prior runs.
            Conversations are stored automatically — no explicit save needed.

            **EXAMPLES:**

            # Recall decisions about authentication
            search_conversations("authentication JWT approach")

            # Find when a bug was discussed
            search_conversations("sqlite lock error")

            # Remember what files were created in a past session
            search_conversations("created files hello.py")

            Args:
                query: Search terms (FTS5 syntax: AND/OR/phrase supported)
                limit: Maximum number of matching turns to return (default: 10)
            """
            return self.tool_search_conversations(query, limit)

        @tool
        def get_plan() -> Dict[str, Any]:
            """Get the current master plan showing all tasks."""
            return self.tool_get_plan()

        @tool
        def update_task(task_id: str, status: str, result: Optional[str] = None, error: Optional[str] = None) -> Dict[str, Any]:
            """Update a task's status in the master plan."""
            return self.tool_update_task(task_id, status, result, error)

        @tool
        def send_message(content: str, priority: str = "FYI") -> Dict[str, Any]:
            """Send a message to the user."""
            return self.tool_send_message(content, priority)

        @tool
        def get_audit_log(limit: int = 20) -> Dict[str, Any]:
            """
            Retrieve recent audit log entries for introspection.

            The audit log contains timestamped records of all agent actions:
            tool executions, state transitions, errors, context warnings, etc.

            Useful for:
            - Debugging why something went wrong
            - Understanding what actions were taken
            - Analyzing context usage patterns
            - Reviewing error history

            Args:
                limit: Maximum number of recent entries to return (default: 20)

            Returns:
                Dict with 'entries' list containing recent audit log records
            """
            log = self.get_audit_log()
            recent = log[-limit:] if len(log) > limit else log
            return {
                "status": "success",
                "total_entries": len(log),
                "returned": len(recent),
                "entries": recent
            }

        @tool
        def get_context_metrics() -> Dict[str, Any]:
            """
            Get current input context usage metrics.

            Returns token counts, percentage of limit, and warnings if approaching limit.

            Useful for:
            - Detecting when to use agent_query() decomposition
            - Understanding why context warnings appeared
            - Diagnosing context exhaustion issues

            Returns:
                Dict with current_tokens, max_tokens, percentage, warnings_triggered
            """
            # Query context warnings from logs
            context_warnings = []
            if self.shared_state:
                context_warnings = self.shared_state.logs.query_logs(
                    search="context",
                    limit=20
                )

            return {
                "status": "success",
                "max_input_tokens": self.max_input_tokens,
                "warning_threshold": self.WARNING_INPUT_TOKENS,
                "emergency_threshold": self.EMERGENCY_INPUT_TOKENS,
                "warnings_triggered": list(self._context_warnings_shown),
                "warning_count": len(context_warnings),
                "recent_warnings": context_warnings[-5:] if context_warnings else [],
                "message": (
                    "No context warnings. Task is properly decomposed." if not context_warnings
                    else f"⚠️  {len(context_warnings)} context warning(s) triggered. Consider using agent_query()."
                )
            }

        @tool
        def get_logs(
            level: Optional[str] = None,
            search: Optional[str] = None,
            limit: int = 50,
            since_step: Optional[int] = None
        ) -> Dict[str, Any]:
            """
            Query runtime logs for self-introspection and adaptive strategy.

            The agent can query ALL runtime logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            to understand what's happening under its hood and adapt its approach.

            **Use this to:**
            - Debug failures: get_logs(level="ERROR", limit=10)
            - Understand warnings: get_logs(level="WARNING")
            - Track context growth: get_logs(search="context", limit=20)
            - Review recent activity: get_logs(limit=50)
            - Analyze specific steps: get_logs(since_step=10)

            **Adaptive strategy examples:**

            # Detect repeated failures
            errors = get_logs(level="ERROR", limit=5)
            # If same error 3+ times → change approach (use agent_query, split file, etc.)

            # Monitor context usage
            context_logs = get_logs(search="context limit")
            # If warnings present → decompose remaining work via agent_query

            # Understand tool patterns
            tool_logs = get_logs(search="Tool execution", limit=20)
            # See which tools succeed/fail, adapt tool selection

            Args:
                level: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                search: Full-text search in log messages (e.g., "context", "failed", "truncated")
                limit: Maximum entries to return (default: 50)
                since_step: Only show logs from this step onwards

            Returns:
                Dict with logs array, summary stats, and adaptive insights
            """
            if not self.shared_state:
                return {
                    "status": "error",
                    "error": "Shared state not enabled. Logs not available."
                }

            logs = self.shared_state.logs.query_logs(
                level=level,
                search=search,
                limit=limit,
                since_step=since_step
            )

            # Analyze for patterns
            error_count = sum(1 for log in logs if log["level"] == "ERROR")
            warning_count = sum(1 for log in logs if log["level"] == "WARNING")
            critical_count = sum(1 for log in logs if log["level"] == "CRITICAL")
            context_mentions = sum(1 for log in logs if "context" in log["message"].lower())

            # Generate adaptive insights
            insights = []
            if error_count >= 3:
                insights.append(f"⚠️  {error_count} errors detected - consider changing approach or using agent_query()")
            if warning_count >= 2:
                insights.append(f"⚠️  {warning_count} warnings - review and adapt strategy")
            if context_mentions >= 1:
                insights.append(f"⚠️  Context warnings detected - use agent_query() to decompose remaining work")

            return {
                "status": "success",
                "count": len(logs),
                "logs": logs,
                "summary": {
                    "errors": error_count,
                    "warnings": warning_count,
                    "critical": critical_count,
                    "context_warnings": context_mentions
                },
                "insights": insights if insights else ["No issues detected. Proceeding normally."],
                "recommendation": (
                    "Continue current approach" if not insights
                    else "Adapt strategy based on insights above"
                )
            }

        # M7: Codebase analysis tools
        @tool
        def index_codebase(root_path: str = ".", include_tests: bool = True) -> Dict[str, Any]:
            """Index a codebase to extract symbols, dependencies, and architecture."""
            return self.tool_index_codebase(root_path, include_tests)

        @tool
        def analyze_architecture(root_path: str = ".", scope: str = "full") -> Dict[str, Any]:
            """Analyze codebase architecture."""
            return self.tool_analyze_architecture(root_path, scope)

        @tool
        def find_symbol(name: str, root_path: str = ".") -> Dict[str, Any]:
            """Find where a symbol is defined."""
            return self.tool_find_symbol(name, root_path)

        @tool
        def get_dependents(file_path: str, root_path: str = ".") -> Dict[str, Any]:
            """Find files that depend on a given file."""
            return self.tool_get_dependents(file_path, root_path)

        @tool
        def detect_issues(root_path: str = ".", severity: Optional[str] = None) -> Dict[str, Any]:
            """Detect code issues and antipatterns."""
            return self.tool_detect_issues(root_path, severity)

        @tool
        def search_codebase(query: str, root_path: str = ".", top_k: int = 10) -> Dict[str, Any]:
            """Semantic search across codebase using hybrid FAISS + FTS5 retrieval."""
            return self.tool_search_codebase(query, root_path, top_k)

        @tool
        def search_code_structure(
            symbol_pattern: Optional[str] = None,
            chunk_type: Optional[str] = None,
            module_pattern: Optional[str] = None,
            root_path: str = ".",
            top_k: int = 50,
        ) -> Dict[str, Any]:
            """Structural search across codebase using LIKE/GLOB patterns on symbols, types, and modules."""
            return self.tool_search_code_structure(symbol_pattern, chunk_type, module_pattern, root_path, top_k)

        @tool
        def impact_analysis(file_path: str, symbol_name: Optional[str] = None, root_path: str = ".") -> Dict[str, Any]:
            """Analyze what breaks if you change a file or symbol. Returns direct dependents, symbol references, transitive impact, and risk level."""
            return self.tool_impact_analysis(file_path, symbol_name, root_path)

        # Execution and Observation Tools
        @tool
        def run_and_observe(project_type: str, entry_point: str) -> Dict[str, Any]:
            """Run code and observe behavior. Critical for verifying code actually works."""
            return self.tool_run_and_observe(project_type, entry_point)

        @tool
        def run_until_functional(project_type: str, entry_point: str, max_iterations: int = 5) -> Dict[str, Any]:
            """Run code, observe, debug, fix until fully functional."""
            return self.tool_run_until_functional(project_type, entry_point, max_iterations)

        # Interactive CLI Tools
        @tool
        def run_interactive_cli(command: str, interactions: List[Dict], timeout: int = 300) -> Dict[str, Any]:
            """Run an interactive CLI tool and respond to prompts."""
            return self.tool_run_interactive_cli(command, interactions, timeout)

        # ── Learned Tools ────────────────────────────────────────────────────

        @tool
        def create_tool(
            name: str,
            code: str,
            description: str,
            category: Optional[str] = None,
            lang: Optional[str] = None,
        ) -> Dict[str, Any]:
            """
            Write a Python function as a reusable tool and store it in tools.db.

            The tool is immediately callable this session AND auto-loaded in all
            future sessions. Use this when you find yourself writing the same logic
            more than once, or when a task-specific helper would save repeated work.

            **Requirements for `code`:**
            - Must define a function with exactly the name given in `name`
            - Must include type annotations on all parameters (needed for registry)
            - Should return a dict with at least {"status": "success"} or {"status": "error"}
            - May import standard library modules at the top of the code string

            **Example — create a C++ error parser:**
            create_tool(
                name="parse_cmake_errors",
                description="Extract error lines from cmake build output",
                category="cpp_utility",
                code='''
def parse_cmake_errors(output: str) -> dict:
    lines = output.splitlines()
    errors = [l for l in lines if "error:" in l.lower()]
    warnings = [l for l in lines if "warning:" in l.lower()]
    return {"errors": errors, "warnings": warnings, "error_count": len(errors)}
'''
            )

            **Example — project-specific file validator:**
            create_tool(
                name="validate_header_guard",
                description="Check C++ header has correct include guard",
                code='''
import re
def validate_header_guard(path: str) -> dict:
    import pathlib
    content = pathlib.Path(path).read_text()
    name = pathlib.Path(path).stem.upper()
    guard = f"_{name}_HPP_"
    has_guard = f"#ifndef {guard}" in content and f"#define {guard}" in content
    return {"valid": has_guard, "expected_guard": guard, "path": path}
'''
            )

            Args:
                name: Function name (valid Python identifier, e.g. "parse_cmake_errors")
                code: Full source code for the tool (function body or script)
                description: What the tool does — used by find_tool() for discovery
                category: Optional category for grouping (default: "learned")
                lang: "python" (default) | "bash" | "sh" | "powershell" | "python_script"
                      bash/powershell/python_script: saved as script, auto-wrapped in Python shim
            """
            return self.tool_create_tool(name, code, description, category, lang or "python")

        @tool
        def list_learned_tools() -> Dict[str, Any]:
            """
            List all tools created by the agent in previous and current sessions.

            Shows name, description, category, use count, and when last used.
            Useful for checking what custom tools are available before writing new ones.
            """
            return self.tool_list_learned_tools()

        @tool
        def delete_tool(name: str) -> Dict[str, Any]:
            """
            Delete a learned tool from tools.db and disk.

            Only works on tools with source='learned' (agent-created tools).
            Cannot delete built-in core or registry tools.

            Args:
                name: Name of the learned tool to delete
            """
            return self.tool_delete_tool(name)

    def tool_agent_query(
        self, task: str, specialist: Optional[str] = None, max_depth: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Delegate a subtask to a sub-agent with fresh context.

        This is the core RAC mechanism. The sub-agent:
        - Gets its own fresh context window
        - Has access to all tools
        - Can recursively spawn more sub-agents
        - Runs quality gates on its output
        - Returns verified result

        Auto-selection pipeline (if specialist not specified):
        1. Query agents.db for best capability match
        2. If no match, check if recurring pattern warrants new specialist creation

        Args:
            task: The subtask to delegate (be specific and complete)
            specialist: Optional specialist agent type (debugger, security, refactoring, etc.)
            max_depth: Optional max recursion depth for this call (default: 3)

        Returns:
            Dict with 'success', 'result', 'errors' keys
        """
        state = get_shared_state()

        # 1. Auto-select specialist from agents.db if not provided
        if not specialist:
            from gaia.agents.gaia_code.integration import find_specialist_for_task
            specialist = find_specialist_for_task(task)

        # 2. Auto-create specialist if recurring pattern has no handler
        if not specialist:
            try:
                from gaia.agents.gaia_code.agent_factory import maybe_create_specialist
                ws = Path(state.workspace_dir) if state.workspace_dir else None
                specialist = maybe_create_specialist(task, ws)
            except Exception:
                pass

        # Check recursion depth
        current_frame = state.call_stack.current()
        current_depth = current_frame.depth if current_frame else 0

        max_allowed_depth = max_depth if max_depth is not None else 3
        if current_depth >= max_allowed_depth:
            return {
                "success": False,
                "result": None,
                "errors": [f"Max recursion depth ({max_allowed_depth}) reached"],
            }

        # Push new call frame
        frame = state.call_stack.push(task, specialist)
        if not frame:
            return {
                "success": False,
                "result": None,
                "errors": ["Failed to push call frame (max depth exceeded)"],
            }

        try:
            result = self._execute_subtask(task, specialist)

            # Pop call frame
            state.call_stack.pop()

            success = not result.startswith("[FAILED]")
            return {"success": success, "result": result, "errors": []}

        except Exception as e:
            state.call_stack.pop()
            return {
                "success": False,
                "result": None,
                "errors": [str(e)],
            }

    def _execute_subtask(self, task: str, specialist: Optional[str]) -> str:
        """
        Execute a subtask by spawning a fresh GaiaCodeAgent sub-agent.

        The sub-agent:
        - Gets its own fresh context window
        - Optionally runs with a specialist system prompt overlay
        - Shares the same SharedAgentState singleton (DBs, call stack, plan)
        - Runs with create_plan=False (operates within parent's scope)

        Args:
            task: The subtask to execute
            specialist: Optional specialist name for system prompt injection

        Returns:
            Result string, prefixed with "[FAILED] " on failure
        """
        import time

        state = get_shared_state()
        start_ms = int(time.time() * 1000)

        try:
            # Lazy import to avoid circular: tools.py is imported by agent.py
            from gaia.agents.gaia_code.agent import GaiaCodeAgent

            workspace_dir = Path(state.workspace_dir) if state.workspace_dir else None
            sub_agent = GaiaCodeAgent(
                workspace_dir=workspace_dir,
                specialist_name=specialist,
                silent_mode=True,
                tui_mode="off",
            )
            # Inherit project_dir from parent so paths stay consistent
            project_dir = getattr(self, "project_dir", None)
            if project_dir:
                sub_agent.project_dir = project_dir

            result = sub_agent.process_query(task, create_plan=False)
            success = result.get("success", False)
            output = result.get("result") or ""

        except Exception as e:
            success = False
            output = f"Sub-agent error: {e}"
            logger.error("[RAC] _execute_subtask failed specialist=%s err=%s", specialist, e)

        # Record usage in agents.db
        if specialist:
            try:
                duration_ms = int(time.time() * 1000) - start_ms
                state.agents.record_usage(
                    specialist,
                    success=success,
                    task_type=task[:100],
                    duration_ms=duration_ms,
                )
            except Exception:
                pass

        return output if success else f"[FAILED] {output}"

    def tool_remember(self, key: str, value: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Store a fact in working memory (memory.db active_state)."""
        import os
        state = get_shared_state()
        agent = getattr(self, "_agent", None)
        source_dir = getattr(agent, "project_dir", None) or os.getcwd()
        query_context = getattr(agent, "_current_query", None)
        state.memory.store_memory(
            key=key, value=value, tags=tags,
            source_dir=source_dir,
            query_context=query_context,
        )
        return {"success": True, "key": key, "stored": True}

    def tool_recall_memory(
        self,
        query: Optional[str] = None,
        key: Optional[str] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Retrieve memories from working memory (memory.db active_state)."""
        import os
        state = get_shared_state()

        if key:
            value = state.memory.get_memory(key)
            if value is not None:
                return {"success": True, "count": 1, "memories": [{"key": key, "value": value}]}
            return {"success": True, "count": 0, "memories": [], "message": f"No memory found for key '{key}'"}

        agent = getattr(self, "_agent", None)
        source_dir = getattr(agent, "project_dir", None) or os.getcwd()
        memories = state.memory.recall_memories(query=query, limit=limit, source_dir=source_dir)
        return {
            "success": True,
            "query": query,
            "count": len(memories),
            "memories": memories,
        }

    def tool_forget_memory(self, key: str) -> Dict[str, Any]:
        """Remove a memory entry from working memory."""
        state = get_shared_state()
        deleted = state.memory.forget_memory(key)
        return {"success": True, "key": key, "deleted": deleted}

    def tool_search_conversations(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search past conversation history in memory.db using FTS5."""
        state = get_shared_state()
        results = state.memory.search_conversations(query=query, limit=limit)
        return {
            "success": True,
            "query": query,
            "count": len(results),
            "turns": results,
        }

    def tool_recall(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query knowledge DB for past context.

        Uses FTS5 full-text search to find relevant insights, learnings,
        and past decisions.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            Dict with 'results' list of insights
        """
        state = get_shared_state()
        insights = state.knowledge.recall(query, top_k=top_k)

        return {
            "query": query,
            "count": len(insights),
            "results": insights,
        }

    def tool_find_tool(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Find tools by natural language query.

        Uses FTS5 search over tool descriptions to find relevant tools.

        Args:
            query: Natural language query (e.g., "create GitHub PR")
            top_k: Number of tools to return

        Returns:
            Dict with 'tools' list
        """
        state = get_shared_state()
        tools = state.tools.find_tools(query, top_k=top_k)

        return {
            "query": query,
            "count": len(tools),
            "tools": tools,
        }

    def tool_store_insight(
        self,
        category: str,
        content: str,
        domain: Optional[str] = None,
        triggers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Store an insight to knowledge DB.

        Categories:
        - error_fix: Error pattern and fix
        - pattern: Reusable code/workflow pattern
        - preference: User preference
        - convention: Project convention

        Args:
            category: Insight category
            content: Insight content
            domain: Optional domain (coding, testing, etc.)
            triggers: Optional trigger keywords for retrieval

        Returns:
            Dict with insight ID
        """
        state = get_shared_state()
        insight_id = state.knowledge.store_insight(
            category=category,
            content=content,
            domain=domain,
            triggers=triggers,
        )

        return {
            "success": True,
            "insight_id": insight_id,
            "category": category,
        }

    def tool_get_plan(self) -> Dict[str, Any]:
        """
        Get the current master plan.

        Returns all tasks with their status, owner, and progress.
        """
        state = get_shared_state()
        tasks = state.plan.get_all_tasks()

        return {
            "total_tasks": len(tasks),
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status,
                    "owner": t.owner,
                    "parent_id": t.parent_id,
                }
                for t in tasks
            ],
        }

    def tool_update_task(
        self,
        task_id: str,
        status: str,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update a task's status in the master plan.

        Args:
            task_id: Task ID
            status: New status (pending | in_progress | completed | failed)
            result: Optional result if completed
            error: Optional error if failed

        Returns:
            Dict with success status
        """
        state = get_shared_state()
        state.plan.update_task_status(task_id, status, result, error)

        return {
            "success": True,
            "task_id": task_id,
            "status": status,
        }

    def tool_send_message(
        self, content: str, priority: str = "FYI"
    ) -> Dict[str, Any]:
        """
        Send a message to the user.

        Priority levels:
        - FYI: Informational message
        - Question: Needs user response
        - Decision: Needs user approval

        Args:
            content: Message content
            priority: Message priority

        Returns:
            Dict with message ID
        """
        state = get_shared_state()
        msg_id = state.message_queue.send(
            content=content,
            priority=priority,
            sender="agent",
            recipient="user",
        )

        return {
            "success": True,
            "message_id": msg_id,
            "priority": priority,
        }

    # ========================================================================
    # M7: Codebase Analysis Tools (using CodeRetrievalPipeline)
    # ========================================================================

    # Cached pipeline instance per root_path
    _retrieval_pipelines: Dict[str, Any] = {}

    def _get_retrieval_pipeline(self, root_path: str = ".") -> "CodeRetrievalPipeline":
        """
        Get or create a cached CodeRetrievalPipeline for the given root path.

        The pipeline is cached on the class so all tool calls share the same
        indexed state, avoiding re-indexing on every call.
        """
        from .code_retrieval import CodeRetrievalPipeline

        resolved_root = str(Path(root_path).resolve())

        if resolved_root not in GaiaCodeTools._retrieval_pipelines:
            # Determine LLM client for annotations
            llm_client = None
            llm_provider = "cloud"  # Default to cloud (matches gaia-code agent)

            # Try to get the agent's LLM client
            if hasattr(self, "llm_client") and self.llm_client:
                llm_client = self.llm_client
            elif hasattr(self, "client") and self.client:
                llm_client = self.client

            if llm_client is None:
                llm_provider = "none"  # No LLM available, use AST-only

            workspace_dir = None
            if hasattr(self, "shared_state") and self.shared_state:
                workspace_dir = self.shared_state.workspace_dir

            pipeline = CodeRetrievalPipeline(
                root_path=resolved_root,
                workspace_dir=workspace_dir,
                llm_provider=llm_provider,
                llm_client=llm_client,
                enable_watcher=True,
            )

            GaiaCodeTools._retrieval_pipelines[resolved_root] = pipeline
            logger.info(f"[CodeRetrieval] Created pipeline for {resolved_root} (llm={llm_provider})")

        return GaiaCodeTools._retrieval_pipelines[resolved_root]

    def tool_index_codebase(
        self, root_path: str = ".", include_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Index a codebase using the CodeRetrievalPipeline.

        Incrementally indexes files, builds dependency graph, creates
        semantic annotations, and builds FAISS + FTS5 search indices.

        Args:
            root_path: Root directory of codebase (default: current dir)
            include_tests: Include test files in index (default: True)

        Returns:
            Dict with index statistics
        """
        try:
            pipeline = self._get_retrieval_pipeline(root_path)
            stats = pipeline.index_repository(include_tests=include_tests)

            # Also generate summaries
            pipeline.generate_summaries()

            return {
                "success": True,
                "stats": stats,
                "message": (
                    f"Indexed {stats['files_indexed']} files "
                    f"({stats['chunks_created']} chunks) in {stats['index_time_seconds']}s. "
                    f"Total: {stats['total_files']} files, {stats['total_chunks']} chunks."
                ),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_analyze_architecture(
        self, root_path: str = ".", scope: str = "full"
    ) -> Dict[str, Any]:
        """
        Analyze codebase architecture using the retrieval pipeline.

        Uses cached index for fast analysis. Also uses legacy CodebaseIndex
        for reports that the pipeline doesn't generate.

        Args:
            root_path: Root directory of codebase
            scope: Analysis scope - "full", "dependencies", "issues", "summary"

        Returns:
            Dict with architecture analysis
        """
        try:
            pipeline = self._get_retrieval_pipeline(root_path)

            # Ensure index is up to date (incremental)
            stats = pipeline.index_repository()

            if scope in ("full", "summary"):
                # Use pipeline summaries
                summary_result = pipeline.query("architecture overview")
                report = summary_result.get("context", "")

            if scope == "full":
                # Also run legacy index for detailed reports
                from .codebase_index import CodebaseIndex
                index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
                index.index_repository()
                report = "\n\n".join([
                    report or index.get_architecture_summary(),
                    index.get_dependency_report(),
                    index.get_issues_report(),
                ])
            elif scope == "dependencies":
                from .codebase_index import CodebaseIndex
                index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
                index.index_repository()
                report = index.get_dependency_report()
            elif scope == "issues":
                from .codebase_index import CodebaseIndex
                index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
                index.index_repository()
                report = index.get_issues_report()

            return {
                "success": True,
                "report": report,
                "stats": stats,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_find_symbol(self, name: str, root_path: str = ".") -> Dict[str, Any]:
        """
        Find where a symbol is defined using the retrieval pipeline.

        Uses the structural search on the code_index.db for fast lookup.

        Args:
            name: Symbol name (class, function, variable)
            root_path: Root directory of codebase

        Returns:
            Dict with symbol locations
        """
        try:
            pipeline = self._get_retrieval_pipeline(root_path)

            # Ensure index is fresh
            pipeline.index_repository()

            # Structural search for exact symbol name
            results = pipeline.search_structure(
                symbol_pattern=name,
                top_k=50,
            )

            if not results:
                # Try fuzzy match with LIKE
                results = pipeline.search_structure(
                    symbol_pattern=f"%{name}%",
                    top_k=20,
                )

            if not results:
                return {
                    "success": False,
                    "message": f"Symbol '{name}' not found in codebase",
                }

            return {
                "success": True,
                "symbol_name": name,
                "locations": [
                    {
                        "file": r.file_path,
                        "line": r.start_line,
                        "type": r.chunk_type,
                        "module": r.module_path,
                        "summary": r.summary,
                    }
                    for r in results
                ],
                "count": len(results),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_get_dependents(
        self, file_path: str, root_path: str = "."
    ) -> Dict[str, Any]:
        """
        Find files that depend on a given file using the retrieval pipeline.

        Uses impact analysis for comprehensive dependency information.

        Args:
            file_path: Path to file
            root_path: Root directory of codebase

        Returns:
            Dict with dependent files
        """
        try:
            pipeline = self._get_retrieval_pipeline(root_path)
            pipeline.index_repository()

            impact = pipeline.impact_analysis(file_path)

            return {
                "success": True,
                "file": file_path,
                "dependents": [d["file"] for d in impact.direct_dependents],
                "transitive_dependents": impact.transitive_impact,
                "count": len(impact.direct_dependents),
                "risk_level": impact.risk_level,
                "message": f"Found {len(impact.direct_dependents)} direct dependents "
                           f"({len(impact.transitive_impact)} transitive) for {file_path}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_detect_issues(
        self, root_path: str = ".", severity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect code issues and antipatterns.

        Uses the legacy CodebaseIndex for issue detection, but ensures
        the retrieval pipeline is also indexed for other queries.

        Args:
            root_path: Root directory of codebase
            severity: Filter by severity - "critical", "warning", "info" (default: all)

        Returns:
            Dict with detected issues
        """
        from .codebase_index import CodebaseIndex

        try:
            # Ensure pipeline is indexed
            pipeline = self._get_retrieval_pipeline(root_path)
            pipeline.index_repository()

            # Use legacy index for issue detection
            index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
            index.index_repository()

            issues = index.issues

            if severity:
                issues = [i for i in issues if i["severity"] == severity]

            return {
                "success": True,
                "issues": issues,
                "count": len(issues),
                "report": index.get_issues_report(),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_search_codebase(
        self, query: str, root_path: str = ".", top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Semantic search across codebase using hybrid FAISS + FTS5 retrieval.

        Combines vector similarity (FAISS) with keyword matching (FTS5)
        for comprehensive code search. Automatically classifies queries
        and assembles context within a token budget.

        Args:
            query: Natural language query (e.g., "authentication logic")
            root_path: Root directory of codebase
            top_k: Number of results to return

        Returns:
            Dict with relevant code chunks, context, and metadata
        """
        try:
            pipeline = self._get_retrieval_pipeline(root_path)

            # Ensure index is fresh (incremental — fast if nothing changed)
            pipeline.index_repository()

            # Use the full query pipeline (classification + hybrid search + context assembly)
            result = pipeline.query(query, top_k=top_k)

            return {
                "success": True,
                "query": query,
                "query_type": result.get("query_type", "semantic"),
                "results": result.get("results", []),
                "context": result.get("context", ""),
                "count": result.get("result_count", 0),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_search_code_structure(
        self,
        symbol_pattern: Optional[str] = None,
        chunk_type: Optional[str] = None,
        module_pattern: Optional[str] = None,
        root_path: str = ".",
        top_k: int = 50,
    ) -> Dict[str, Any]:
        """
        Structural search across codebase using LIKE/GLOB patterns.

        Searches directly on the code index database for exact or
        pattern-matched results on symbol names, chunk types, and module paths.

        Args:
            symbol_pattern: Pattern to match symbol names (SQL LIKE syntax, e.g., "%Agent%")
            chunk_type: Filter by chunk type (class, function, method, etc.)
            module_pattern: Pattern to match module paths (e.g., "gaia.agents.%")
            root_path: Root directory of codebase
            top_k: Max results

        Returns:
            Dict with matched code structures
        """
        try:
            pipeline = self._get_retrieval_pipeline(root_path)
            pipeline.index_repository()

            results = pipeline.search_structure(
                symbol_pattern=symbol_pattern,
                chunk_type=chunk_type,
                module_pattern=module_pattern,
                top_k=top_k,
            )

            return {
                "success": True,
                "results": [
                    {
                        "name": r.symbol_name,
                        "type": r.chunk_type,
                        "file": r.relative_path,
                        "line": r.start_line,
                        "module": r.module_path,
                        "summary": r.summary,
                    }
                    for r in results
                ],
                "count": len(results),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_impact_analysis(
        self,
        file_path: str,
        symbol_name: Optional[str] = None,
        root_path: str = ".",
    ) -> Dict[str, Any]:
        """
        Analyze what breaks if you change a file or symbol.

        Performs BFS transitive dependency traversal to find all
        direct dependents, symbol references, and transitive impact.

        Args:
            file_path: Path to file to analyze
            symbol_name: Specific symbol to analyze (optional)
            root_path: Root directory of codebase

        Returns:
            Dict with impact analysis including risk level
        """
        try:
            pipeline = self._get_retrieval_pipeline(root_path)
            pipeline.index_repository()

            impact = pipeline.impact_analysis(file_path, symbol_name)

            return {
                "success": True,
                "target_file": impact.target_file,
                "target_symbol": impact.target_symbol,
                "risk_level": impact.risk_level,
                "direct_dependents": impact.direct_dependents,
                "symbol_references": impact.symbol_references,
                "transitive_impact": impact.transitive_impact,
                "total_impact": len(impact.direct_dependents) + len(impact.transitive_impact),
                "context": pipeline._format_impact_context(impact),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_run_and_observe(
        self, project_type: str, entry_point: str
    ) -> Dict[str, Any]:
        """Run code and observe behavior."""
        from .execution_observer import ExecutionObserver

        try:
            observer = ExecutionObserver(Path("."))
            result = observer.run_and_observe(project_type, entry_point)
            observer.cleanup()
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def tool_run_until_functional(
        self, project_type: str, entry_point: str, max_iterations: int = 5
    ) -> Dict[str, Any]:
        """Run code, observe, debug, fix until fully functional."""
        from .execution_observer import ExecutionObserver

        try:
            observer = ExecutionObserver(Path("."))
            observations_history = []

            for iteration in range(max_iterations):
                result = observer.run_and_observe(project_type, entry_point)
                observations_history.append(result)

                if result["success"]:
                    observer.cleanup()
                    return {
                        "success": True,
                        "iterations": iteration + 1,
                        "final_result": result,
                        "history": observations_history,
                        "message": f"Fully functional after {iteration + 1} iteration(s)",
                    }

                logger.info(f"Iteration {iteration + 1}: {result.get('error', 'Unknown error')}")

            observer.cleanup()
            return {
                "success": False,
                "iterations": max_iterations,
                "final_result": observations_history[-1] if observations_history else None,
                "history": observations_history,
                "message": f"Not fully functional after {max_iterations} iterations",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def tool_run_interactive_cli(
        self, command: str, interactions: List[Dict], timeout: int = 300
    ) -> Dict[str, Any]:
        """Run an interactive CLI tool and respond to prompts."""
        from .execution_observer import InteractiveCLIExecutor

        try:
            executor = InteractiveCLIExecutor()
            result = executor.run_interactive(command, interactions, timeout)
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Learned Tools implementation ─────────────────────────────────────────

    # Wrapper templates for non-Python script languages.
    # The generated wrapper is stored as {name}.py and loaded into _TOOL_REGISTRY.
    _SCRIPT_WRAPPER_TEMPLATES = {
        "bash": '''\
import subprocess
from pathlib import Path

def {name}(cwd: str = ".") -> dict:
    """{description}"""
    result = subprocess.run(
        ["bash", r"{script_path}"],
        cwd=cwd, capture_output=True, text=True
    )
    return {{
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout[-4000:] if result.stdout else "",
        "error": result.stderr[-1000:] if result.stderr else "",
        "returncode": result.returncode,
    }}
''',
        "powershell": '''\
import subprocess
from pathlib import Path

def {name}(cwd: str = ".") -> dict:
    """{description}"""
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", r"{script_path}"],
        cwd=cwd, capture_output=True, text=True
    )
    return {{
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout[-4000:] if result.stdout else "",
        "error": result.stderr[-1000:] if result.stderr else "",
        "returncode": result.returncode,
    }}
''',
        "python_script": '''\
import subprocess
import sys
from pathlib import Path

def {name}(cwd: str = ".") -> dict:
    """{description}"""
    result = subprocess.run(
        [sys.executable, r"{script_path}"],
        cwd=cwd, capture_output=True, text=True
    )
    return {{
        "status": "success" if result.returncode == 0 else "error",
        "output": result.stdout[-4000:] if result.stdout else "",
        "error": result.stderr[-1000:] if result.stderr else "",
        "returncode": result.returncode,
    }}
''',
    }

    # File extensions for each script language
    _SCRIPT_EXTENSIONS = {
        "bash": ".sh",
        "sh": ".sh",
        "powershell": ".ps1",
        "ps1": ".ps1",
        "python_script": ".py",
        "python": ".py",
    }

    def tool_create_tool(
        self,
        name: str,
        code: str,
        description: str,
        category: Optional[str] = None,
        lang: str = "python",
    ) -> Dict[str, Any]:
        """
        Write a script as a persistent callable tool stored in tools.db.

        Supports Python functions (loaded directly into registry) and shell scripts
        (bash, PowerShell, python_script — wrapped in a thin Python shim).
        """
        import ast
        import importlib.util
        from gaia.agents.base.tools import tool as tool_decorator, _TOOL_REGISTRY

        # Normalise
        name = name.strip()
        lang = lang.lower().strip()
        effective_category = category or "learned"

        # Validate name
        if not name.isidentifier():
            return {"status": "error", "error": f"Invalid tool name '{name}': must be a valid Python identifier"}

        # Resolve workspace tools dir
        state = get_shared_state()
        if not state or not state.workspace_dir:
            return {"status": "error", "error": "No workspace directory configured"}

        tools_dir = Path(state.workspace_dir) / "tools"
        tools_dir.mkdir(exist_ok=True)

        ext = self._SCRIPT_EXTENSIONS.get(lang, ".py")

        if lang in ("python",):
            # ── Python function: validate, save, load directly ───────────────
            try:
                ast.parse(code)
            except SyntaxError as e:
                return {"status": "error", "error": f"Syntax error in Python code: {e}"}

            tree = ast.parse(code)
            func_defs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            if name not in func_defs:
                return {"status": "error", "error": f"Code must define a function named '{name}'"}

            tool_file = tools_dir / f"{name}.py"
            tool_file.write_text(code, encoding="utf-8")
            load_file = tool_file

        else:
            # ── Script language: save script, generate Python wrapper ────────
            template = self._SCRIPT_WRAPPER_TEMPLATES.get(lang)
            if template is None:
                supported = list(self._SCRIPT_WRAPPER_TEMPLATES.keys())
                return {"status": "error", "error": f"Unsupported lang '{lang}'. Supported: python, {', '.join(supported)}"}

            script_file = tools_dir / f"{name}{ext}"
            script_file.write_text(code, encoding="utf-8")
            if ext == ".sh":
                script_file.chmod(0o755)

            wrapper_code = template.format(
                name=name,
                description=description.replace('"', '\\"'),
                script_path=str(script_file).replace("\\", "/"),
            )

            wrapper_file = tools_dir / f"{name}.py"
            wrapper_file.write_text(wrapper_code, encoding="utf-8")
            load_file = wrapper_file

        # Load and register in _TOOL_REGISTRY
        try:
            spec = importlib.util.spec_from_file_location(name, load_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, name)
            tool_decorator(func)
        except Exception as e:
            return {"status": "error", "error": f"Failed to load tool '{name}': {e}"}

        # Extract parameters from the now-registered tool
        registered = _TOOL_REGISTRY.get(name, {})
        params = {
            pname: {"type": pinfo["type"], "required": pinfo["required"]}
            for pname, pinfo in registered.get("parameters", {}).items()
        }

        # Persist in tools.db
        state.tools.register_tool(
            name=name,
            category=effective_category,
            description=description,
            source="learned",
            parameters=params,
            code_path=str(load_file),
        )

        logger.info(f"[GaiaCode] created learned tool: {name} (lang={lang})")

        return {
            "status": "success",
            "name": name,
            "lang": lang,
            "file": str(load_file),
            "parameters": params,
            "message": (
                f"Tool '{name}' created and registered. "
                "Callable this session and auto-loaded in all future sessions."
            ),
        }

    def tool_list_learned_tools(self) -> Dict[str, Any]:
        """List all tools created by the agent (source='learned')."""
        state = get_shared_state()
        if not state:
            return {"status": "error", "error": "No shared state"}

        rows = state.tools.conn.execute(
            """
            SELECT name, category, description, use_count, last_used, code_path
            FROM tools
            WHERE source = 'learned' AND enabled = TRUE
            ORDER BY use_count DESC, name
            """
        ).fetchall()

        tools = [
            {
                "name": r[0],
                "category": r[1],
                "description": r[2],
                "use_count": r[3],
                "last_used": r[4],
                "file": r[5],
            }
            for r in rows
        ]

        return {
            "status": "success",
            "count": len(tools),
            "tools": tools,
        }

    def tool_delete_tool(self, name: str) -> Dict[str, Any]:
        """Delete a learned tool from tools.db and disk."""
        import os
        from gaia.agents.base.tools import _TOOL_REGISTRY

        state = get_shared_state()
        if not state:
            return {"status": "error", "error": "No shared state"}

        row = state.tools.conn.execute(
            "SELECT source, code_path FROM tools WHERE name = ?", (name,)
        ).fetchone()

        if not row:
            return {"status": "error", "error": f"Tool '{name}' not found"}
        if row[0] != "learned":
            return {"status": "error", "error": f"Cannot delete built-in tool '{name}' (source={row[0]})"}

        code_path = row[1]

        # Remove from DB
        state.tools.conn.execute("DELETE FROM tools WHERE name = ?", (name,))
        state.tools.conn.execute("DELETE FROM tools_fts WHERE name = ?", (name,))
        state.tools.conn.commit()

        # Remove from _TOOL_REGISTRY
        _TOOL_REGISTRY.pop(name, None)

        # Remove files from disk
        deleted_files = []
        for path_str in [code_path]:
            if path_str and Path(path_str).exists():
                try:
                    os.remove(path_str)
                    deleted_files.append(path_str)
                except Exception:
                    pass

        logger.info(f"[GaiaCode] deleted learned tool: {name}")
        return {"status": "success", "name": name, "deleted_files": deleted_files}
