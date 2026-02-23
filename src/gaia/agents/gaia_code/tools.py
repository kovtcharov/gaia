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
