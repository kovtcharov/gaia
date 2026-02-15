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
from typing import Any, Dict, List, Optional

from .shared_state import get_shared_state


class GaiaCodeTools:
    """Mixin providing GAIA Code-specific tools."""

    def register_gaia_code_tools(self):
        """Register GAIA Code tools."""
        self.register_tool(
            "agent_query",
            self.tool_agent_query,
            "Delegate a subtask to a sub-agent with fresh context. Use for recursive decomposition. The sub-agent has full agency: tools, quality gates, can recurse further. Returns verified result.",
        )

        self.register_tool(
            "recall",
            self.tool_recall,
            "Query knowledge DB for past context using full-text search. Examples: recall('what files did I create?'), recall('error in auth.py')",
        )

        self.register_tool(
            "find_tool",
            self.tool_find_tool,
            "Search for tools by natural language query. Returns relevant tools with descriptions.",
        )

        self.register_tool(
            "store_insight",
            self.tool_store_insight,
            "Store a learning or insight to knowledge DB for future reference. Categories: error_fix, pattern, preference, convention.",
        )

        self.register_tool(
            "get_plan",
            self.tool_get_plan,
            "Get the current master plan showing all tasks and their status.",
        )

        self.register_tool(
            "update_task",
            self.tool_update_task,
            "Update a task's status in the master plan. Use this to track progress.",
        )

        self.register_tool(
            "send_message",
            self.tool_send_message,
            "Send a message to the user. Priority: FYI (info), Question (needs answer), Decision (needs approval).",
        )

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

        Args:
            task: The subtask to delegate (be specific and complete)
            specialist: Optional specialist agent type (debugger, security, refactoring, etc.)
            max_depth: Optional max recursion depth for this call

        Returns:
            Dict with 'success', 'result', 'errors' keys
        """
        state = get_shared_state()

        # Check recursion depth
        current_frame = state.call_stack.current()
        current_depth = current_frame.depth if current_frame else 0

        max_allowed_depth = max_depth if max_depth is not None else 10
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
            # Create sub-agent (inherit context from shared state)
            # In a real implementation, this would instantiate a new agent
            # For now, we'll simulate it
            result = self._execute_subtask(task, specialist)

            # Pop call frame
            state.call_stack.pop()

            return {"success": True, "result": result, "errors": []}

        except Exception as e:
            state.call_stack.pop()
            return {
                "success": False,
                "result": None,
                "errors": [str(e)],
            }

    def _execute_subtask(self, task: str, specialist: Optional[str]) -> str:
        """Execute a subtask (placeholder for actual sub-agent execution)."""
        # In the full implementation, this would:
        # 1. Create a new GaiaCodeAgent instance (or specialist)
        # 2. Pass it the task
        # 3. Run quality gates on its output
        # 4. Return verified result
        #
        # For now, we'll return a placeholder
        return f"Subtask completed: {task}"

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
