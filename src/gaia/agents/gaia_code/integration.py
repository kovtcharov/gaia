# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Integration utilities for GAIA Code.

This module handles:
- Tool registration from existing CodeAgent tools
- Specialist registration in agents.db
- LLM client integration
- Initial workspace setup
"""

import logging
from pathlib import Path
from typing import List, Optional

from .shared_state import get_shared_state
from .specialists import (
    ArchitectureAgent,
    DebuggerAgent,
    DocumentationAgent,
    PerformanceAgent,
    RefactoringAgent,
    SecurityAgent,
    TestingAgent,
)

logger = logging.getLogger(__name__)


def register_core_tools(state) -> int:
    """
    Register core tools in tools.db.

    Returns:
        Number of tools registered
    """
    core_tools = [
        # File I/O tools
        {
            "name": "read_file",
            "category": "file_io",
            "description": "Read contents of a file",
            "source": "core",
        },
        {
            "name": "write_file",
            "category": "file_io",
            "description": "Write contents to a file",
            "source": "core",
        },
        {
            "name": "edit_file",
            "category": "file_io",
            "description": "Edit a file by replacing old text with new text",
            "source": "core",
        },
        {
            "name": "glob_search",
            "category": "file_io",
            "description": "Search for files matching a pattern",
            "source": "core",
        },
        {
            "name": "grep_content",
            "category": "file_io",
            "description": "Search for content within files",
            "source": "core",
        },
        # Code execution tools
        {
            "name": "run_python",
            "category": "execution",
            "description": "Run Python code",
            "source": "core",
        },
        {
            "name": "run_shell_command",
            "category": "execution",
            "description": "Run a shell command",
            "source": "core",
        },
        # Testing tools
        {
            "name": "run_pytest",
            "category": "testing",
            "description": "Run pytest tests",
            "source": "core",
        },
        {
            "name": "run_jest",
            "category": "testing",
            "description": "Run Jest tests",
            "source": "core",
        },
        {
            "name": "check_coverage",
            "category": "testing",
            "description": "Check test coverage",
            "source": "core",
        },
        # Git tools
        {
            "name": "git_status",
            "category": "git",
            "description": "Show git status",
            "source": "core",
        },
        {
            "name": "git_diff",
            "category": "git",
            "description": "Show git diff",
            "source": "core",
        },
        {
            "name": "git_commit",
            "category": "git",
            "description": "Create git commit",
            "source": "core",
        },
        {
            "name": "git_branch",
            "category": "git",
            "description": "Git branch operations",
            "source": "core",
        },
        {
            "name": "git_log",
            "category": "git",
            "description": "Show git log",
            "source": "core",
        },
        # Quality tools
        {
            "name": "check_syntax",
            "category": "quality",
            "description": "Check Python syntax",
            "source": "core",
        },
        {
            "name": "run_linter",
            "category": "quality",
            "description": "Run code linter",
            "source": "core",
        },
        {
            "name": "check_imports",
            "category": "quality",
            "description": "Check if all imports resolve",
            "source": "core",
        },
        {
            "name": "format_code",
            "category": "quality",
            "description": "Format code with Black",
            "source": "core",
        },
        # GitHub tools
        {
            "name": "gh_clone",
            "category": "github",
            "description": "Clone GitHub repository",
            "source": "core",
        },
        {
            "name": "gh_pr_create",
            "category": "github",
            "description": "Create GitHub pull request",
            "source": "core",
        },
        {
            "name": "gh_issue_list",
            "category": "github",
            "description": "List GitHub issues",
            "source": "core",
        },
    ]

    count = 0
    for tool in core_tools:
        try:
            state.tools.register_tool(
                name=tool["name"],
                category=tool["category"],
                description=tool["description"],
                source=tool["source"],
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to register tool {tool['name']}: {e}")

    logger.info(f"Registered {count} core tools")
    return count


def register_specialists(state) -> int:
    """
    Register the 7 core specialist agents in agents.db.

    Returns:
        Number of specialists registered
    """
    specialists = [
        DebuggerAgent(),
        SecurityAgent(),
        RefactoringAgent(),
        TestingAgent(),
        DocumentationAgent(),
        PerformanceAgent(),
        ArchitectureAgent(),
    ]

    count = 0
    for specialist in specialists:
        try:
            metadata = specialist.get_metadata()

            # Register in agents.db
            state.agents.conn.execute(
                """
                INSERT OR REPLACE INTO agents (id, name, description, capabilities,
                                              system_prompt, tool_packs, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metadata["name"],
                    metadata["name"],
                    metadata["description"],
                    ",".join(metadata["capabilities"]),
                    specialist.get_system_prompt(),
                    ",".join(metadata["tool_packs"]),
                    1.0,  # Initial confidence
                ),
            )
            state.agents.conn.commit()
            count += 1

            logger.info(f"Registered specialist: {metadata['name']}")

        except Exception as e:
            logger.warning(f"Failed to register specialist {specialist.__class__.__name__}: {e}")

    logger.info(f"Registered {count} specialists")
    return count


def initialize_workspace(workspace_dir: Optional[Path] = None) -> Path:
    """
    Initialize GAIA Code workspace.

    Creates directory structure and initializes databases.

    Args:
        workspace_dir: Optional workspace directory (default: ~/.gaia/workspace)

    Returns:
        Path to workspace directory
    """
    if workspace_dir is None:
        workspace_dir = Path.home() / ".gaia" / "workspace"

    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Get shared state (creates databases)
    state = get_shared_state(workspace_dir)

    # Register core tools
    tools_count = register_core_tools(state)

    # Register specialists
    specialists_count = register_specialists(state)

    logger.info(f"Workspace initialized at {workspace_dir}")
    logger.info(f"  - {tools_count} tools registered")
    logger.info(f"  - {specialists_count} specialists registered")

    return workspace_dir


def find_specialist_for_task(task: str) -> Optional[str]:
    """
    Find the best specialist for a task using semantic search.

    Args:
        task: Task description

    Returns:
        Specialist name or None if no good match
    """
    state = get_shared_state()

    # Simple keyword matching for now
    # In full implementation, would use FAISS semantic search
    task_lower = task.lower()

    keywords_to_specialist = {
        "debug": "DebuggerAgent",
        "error": "DebuggerAgent",
        "fix": "DebuggerAgent",
        "security": "SecurityAgent",
        "vulnerability": "SecurityAgent",
        "injection": "SecurityAgent",
        "refactor": "RefactoringAgent",
        "cleanup": "RefactoringAgent",
        "test": "TestingAgent",
        "coverage": "TestingAgent",
        "documentation": "DocumentationAgent",
        "docstring": "DocumentationAgent",
        "readme": "DocumentationAgent",
        "performance": "PerformanceAgent",
        "optimize": "PerformanceAgent",
        "slow": "PerformanceAgent",
        "architecture": "ArchitectureAgent",
        "design": "ArchitectureAgent",
        "pattern": "ArchitectureAgent",
    }

    for keyword, specialist in keywords_to_specialist.items():
        if keyword in task_lower:
            logger.info(f"Auto-selected specialist: {specialist} (keyword: {keyword})")
            return specialist

    return None


def get_tool_packs_for_specialist(specialist_name: str) -> List[str]:
    """
    Get tool packs for a specialist.

    Args:
        specialist_name: Name of specialist

    Returns:
        List of tool pack names
    """
    state = get_shared_state()

    try:
        cursor = state.agents.conn.execute(
            "SELECT tool_packs FROM agents WHERE name = ?", (specialist_name,)
        )
        row = cursor.fetchone()

        if row and row[0]:
            return row[0].split(",")

    except Exception as e:
        logger.warning(f"Failed to get tool packs for {specialist_name}: {e}")

    # Default tool packs if query fails
    return ["core", "coding"]


def get_specialist_system_prompt(specialist_name: str) -> Optional[str]:
    """
    Get system prompt for a specialist.

    Args:
        specialist_name: Name of specialist

    Returns:
        System prompt or None
    """
    state = get_shared_state()

    try:
        cursor = state.agents.conn.execute(
            "SELECT system_prompt FROM agents WHERE name = ?", (specialist_name,)
        )
        row = cursor.fetchone()

        if row:
            return row[0]

    except Exception as e:
        logger.warning(f"Failed to get system prompt for {specialist_name}: {e}")

    return None


def list_available_specialists() -> List[dict]:
    """
    List all registered specialists.

    Returns:
        List of specialist info dicts
    """
    state = get_shared_state()

    try:
        cursor = state.agents.conn.execute(
            "SELECT name, description, capabilities, confidence FROM agents"
        )

        specialists = []
        for row in cursor.fetchall():
            specialists.append(
                {
                    "name": row[0],
                    "description": row[1],
                    "capabilities": row[2].split(",") if row[2] else [],
                    "confidence": row[3],
                }
            )

        return specialists

    except Exception as e:
        logger.warning(f"Failed to list specialists: {e}")
        return []


def get_workspace_stats() -> dict:
    """
    Get statistics about the workspace.

    Returns:
        Dict with statistics
    """
    state = get_shared_state()

    stats = {
        "tools": {
            "total": 0,
            "core": 0,
            "learned": 0,
        },
        "skills": {
            "total": 0,
        },
        "specialists": {
            "total": 0,
        },
        "knowledge": {
            "insights": 0,
            "preferences": 0,
            "learnings": 0,
        },
        "plan": {
            "total_tasks": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
        },
    }

    try:
        # Tool stats
        cursor = state.tools.conn.execute("SELECT COUNT(*) FROM tools")
        stats["tools"]["total"] = cursor.fetchone()[0]

        cursor = state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools WHERE source = 'core'"
        )
        stats["tools"]["core"] = cursor.fetchone()[0]

        cursor = state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools WHERE source = 'learned'"
        )
        stats["tools"]["learned"] = cursor.fetchone()[0]

        # Skills stats
        cursor = state.skills.conn.execute("SELECT COUNT(*) FROM skills")
        stats["skills"]["total"] = cursor.fetchone()[0]

        # Specialists stats
        cursor = state.agents.conn.execute("SELECT COUNT(*) FROM agents")
        stats["specialists"]["total"] = cursor.fetchone()[0]

        # Knowledge stats
        cursor = state.knowledge.conn.execute("SELECT COUNT(*) FROM insights")
        stats["knowledge"]["insights"] = cursor.fetchone()[0]

        cursor = state.knowledge.conn.execute("SELECT COUNT(*) FROM preferences")
        stats["knowledge"]["preferences"] = cursor.fetchone()[0]

        cursor = state.knowledge.conn.execute("SELECT COUNT(*) FROM learnings")
        stats["knowledge"]["learnings"] = cursor.fetchone()[0]

        # Plan stats
        tasks = state.plan.get_all_tasks()
        stats["plan"]["total_tasks"] = len(tasks)
        stats["plan"]["completed"] = len([t for t in tasks if t.status == "completed"])
        stats["plan"]["in_progress"] = len(
            [t for t in tasks if t.status == "in_progress"]
        )
        stats["plan"]["pending"] = len([t for t in tasks if t.status == "pending"])

    except Exception as e:
        logger.warning(f"Failed to get workspace stats: {e}")

    return stats
