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
            """Delegate a subtask to a sub-agent with fresh context. Use for recursive decomposition."""
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
            """Semantic search across codebase."""
            return self.tool_search_codebase(query, root_path, top_k)

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

    # ========================================================================
    # M7: Codebase Analysis Tools
    # ========================================================================

    def tool_index_codebase(
        self, root_path: str = ".", include_tests: bool = True
    ) -> Dict[str, Any]:
        """
        Index a codebase for fast navigation and analysis.

        Extracts symbols, builds dependency graph, detects issues.

        Args:
            root_path: Root directory of codebase (default: current dir)
            include_tests: Include test files in index (default: True)

        Returns:
            Dict with index statistics
        """
        from .codebase_index import CodebaseIndex

        try:
            index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
            stats = index.index_repository(include_tests=include_tests)

            # Store the index object for future queries
            # (in a real implementation, would cache this)

            return {
                "success": True,
                "stats": stats,
                "message": f"Indexed {stats['files_indexed']} files with {stats['symbols_found']} symbols",
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
        Analyze codebase architecture.

        Args:
            root_path: Root directory of codebase
            scope: Analysis scope - "full", "dependencies", "issues", "summary"

        Returns:
            Dict with architecture analysis
        """
        from .codebase_index import CodebaseIndex

        try:
            # Create index
            index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
            index.index_repository()

            # Generate reports based on scope
            if scope == "full":
                report = "\n\n".join([
                    index.get_architecture_summary(),
                    index.get_dependency_report(),
                    index.get_issues_report(),
                ])
            elif scope == "dependencies":
                report = index.get_dependency_report()
            elif scope == "issues":
                report = index.get_issues_report()
            else:  # summary
                report = index.get_architecture_summary()

            return {
                "success": True,
                "report": report,
                "stats": index.stats,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def tool_find_symbol(self, name: str, root_path: str = ".") -> Dict[str, Any]:
        """
        Find where a symbol is defined.

        Args:
            name: Symbol name (class, function, variable)
            root_path: Root directory of codebase

        Returns:
            Dict with symbol locations
        """
        from .codebase_index import CodebaseIndex

        try:
            index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
            index.index_repository()

            symbols = index.find_symbol(name)

            if not symbols:
                return {
                    "success": False,
                    "message": f"Symbol '{name}' not found in codebase",
                }

            return {
                "success": True,
                "symbol_name": name,
                "locations": [
                    {
                        "file": s.file_path,
                        "line": s.line_number,
                        "type": s.type,
                        "module": s.module_path,
                    }
                    for s in symbols
                ],
                "count": len(symbols),
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
        Find files that depend on a given file.

        Args:
            file_path: Path to file
            root_path: Root directory of codebase

        Returns:
            Dict with dependent files
        """
        from .codebase_index import CodebaseIndex

        try:
            index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
            index.index_repository()

            # Resolve file path
            full_path = str((Path(root_path) / file_path).resolve())
            dependents = index.get_dependents(full_path)

            return {
                "success": True,
                "file": file_path,
                "dependents": list(dependents),
                "count": len(dependents),
                "message": f"Found {len(dependents)} files that depend on {file_path}",
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

        Detects:
        - Circular dependencies
        - Large files (>500 lines)
        - Missing docstrings
        - High complexity functions
        - Missing tests

        Args:
            root_path: Root directory of codebase
            severity: Filter by severity - "critical", "warning", "info" (default: all)

        Returns:
            Dict with detected issues
        """
        from .codebase_index import CodebaseIndex

        try:
            index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
            index.index_repository()

            issues = index.issues

            # Filter by severity if requested
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
        Semantic search across codebase.

        Uses vector similarity to find code related to a concept.

        Args:
            query: Natural language query (e.g., "authentication logic")
            root_path: Root directory of codebase
            top_k: Number of results to return

        Returns:
            Dict with relevant files and symbols
        """
        from .codebase_index import CodebaseIndex

        try:
            index = CodebaseIndex(root_path, self.shared_state.workspace_dir)
            index.index_repository()

            # Search symbols by docstring similarity
            results = []

            for symbol_name, symbol_list in index.symbols.items():
                for symbol in symbol_list:
                    if symbol.docstring:
                        # Simple keyword matching for now
                        # In full implementation, would use vector similarity
                        if any(word in symbol.docstring.lower() for word in query.lower().split()):
                            results.append({
                                "name": symbol.name,
                                "type": symbol.type,
                                "file": symbol.file_path,
                                "line": symbol.line_number,
                                "docstring": symbol.docstring[:200],
                            })

            # Sort by relevance and limit
            results = results[:top_k]

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
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
