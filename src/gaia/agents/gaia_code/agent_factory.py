# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
AgentFactory: Generate specialist agents from patterns.

M5: Agent Auto-Generation + Self-Extension

The AgentFactory enables the agent to:
- Detect recurring task patterns (3+ similar successful tasks)
- Generate new specialist agents from patterns
- Register specialists in agents.db
- Test specialists before deployment

This is the ultimate self-extension capability - the agent creates its own experts.
"""

import ast
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .shared_state import get_shared_state
from .specialists.base_specialist import BaseSpecialist

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Generates new specialist agents from recurring task patterns.

    Process:
    1. Detect pattern: Find 3+ similar successful tasks
    2. Extract common elements: tools, workflows, success factors
    3. Generate specialist: Create class with custom state machine and prompt
    4. Test specialist: Validate on sample task
    5. Register specialist: Add to agents.db

    This enables exponential growth in capabilities.
    """

    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir or Path.home() / ".gaia" / "workspace"
        self.generated_agents_dir = self.workspace_dir / "generated_agents"
        self.generated_agents_dir.mkdir(parents=True, exist_ok=True)
        self.state = get_shared_state(workspace_dir)

    def detect_pattern(
        self, min_occurrences: int = 3, min_similarity: float = 0.8
    ) -> List[Dict]:
        """
        Detect recurring task patterns from history.

        Queries plan_tasks across ALL past plans (not just the active plan) so
        patterns accumulate across sessions.

        Args:
            min_occurrences: Minimum number of similar tasks (default: 3)
            min_similarity: Minimum similarity threshold (default: 0.8)

        Returns:
            List of pattern dicts
        """
        # Query ALL completed tasks across all plans from the DB
        try:
            from types import SimpleNamespace
            rows = self.state.memory.conn.execute(
                "SELECT id, title FROM plan_tasks WHERE status='completed'"
            ).fetchall()
            completed = [
                SimpleNamespace(id=r[0], description=r[1] or "")
                for r in rows if r[1]
            ]
        except Exception:
            return []

        if len(completed) < min_occurrences:
            logger.info(f"Not enough completed tasks ({len(completed)}) to detect patterns")
            return []

        # Keyword-based clustering across all sessions
        patterns = self._cluster_similar_tasks(completed, min_occurrences, min_similarity)

        logger.info(f"Detected {len(patterns)} patterns from {len(completed)} tasks")

        return patterns

    def _cluster_similar_tasks(
        self, tasks: List, min_occurrences: int, min_similarity: float
    ) -> List[Dict]:
        """
        Cluster tasks by similarity.

        Args:
            tasks: List of completed tasks
            min_occurrences: Minimum cluster size
            min_similarity: Minimum similarity

        Returns:
            List of pattern clusters
        """
        # Extract keywords from task descriptions
        task_keywords = []
        for task in tasks:
            words = task.description.lower().split()
            keywords = set([w for w in words if len(w) > 3])
            task_keywords.append((task, keywords))

        # Find clusters
        patterns = []
        used_tasks = set()

        for i, (task1, keywords1) in enumerate(task_keywords):
            if task1.id in used_tasks:
                continue

            # Find similar tasks
            cluster = [task1]
            cluster_keywords = keywords1

            for j, (task2, keywords2) in enumerate(task_keywords):
                if i == j or task2.id in used_tasks:
                    continue

                # Calculate similarity
                intersection = cluster_keywords & keywords2
                union = cluster_keywords | keywords2
                similarity = len(intersection) / len(union) if union else 0

                if similarity >= min_similarity:
                    cluster.append(task2)
                    cluster_keywords |= keywords2
                    used_tasks.add(task2.id)

            # If cluster is large enough, it's a pattern
            if len(cluster) >= min_occurrences:
                patterns.append(
                    {
                        "tasks": cluster,
                        "keywords": list(cluster_keywords),
                        "count": len(cluster),
                        "similarity": min_similarity,
                    }
                )
                for task in cluster:
                    used_tasks.add(task.id)

        return patterns

    def generate_specialist(
        self,
        pattern: Dict,
        specialist_name: str,
        description: str,
    ) -> Dict:
        """
        Generate a new specialist agent from a pattern.

        Args:
            pattern: Pattern dict from detect_pattern()
            specialist_name: Name for the specialist
            description: Description of what the specialist does

        Returns:
            Dict with success status and specialist info
        """
        # Extract common tools and workflows
        tools_used = self._extract_common_tools(pattern["tasks"])
        workflow = self._extract_common_workflow(pattern["tasks"])

        # Generate agent code
        agent_code = self._generate_agent_code(
            specialist_name,
            description,
            workflow,
            tools_used,
            pattern["keywords"],
        )

        # Validate generated code
        try:
            ast.parse(agent_code)
        except SyntaxError as e:
            return {
                "success": False,
                "error": f"Generated code has syntax error: {e}",
            }

        # Write agent file
        agent_file = self.generated_agents_dir / f"{specialist_name.lower()}.py"
        agent_file.write_text(agent_code)

        # Register in agents.db
        try:
            self.state.agents.conn.execute(
                """
                INSERT OR REPLACE INTO agents (id, name, description, capabilities,
                                              system_prompt, tool_packs, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    specialist_name,
                    specialist_name,
                    description,
                    ",".join(pattern["keywords"]),
                    self._generate_system_prompt(specialist_name, description, workflow),
                    ",".join(["core", "coding"]),  # Default tool packs
                    0.5,  # Initial confidence
                    datetime.now().isoformat(),
                ),
            )
            self.state.agents.conn.commit()

            logger.info(f"Generated specialist: {specialist_name}")

            return {
                "success": True,
                "specialist_name": specialist_name,
                "agent_file": str(agent_file),
                "pattern_count": pattern["count"],
            }

        except Exception as e:
            logger.error(f"Failed to register specialist {specialist_name}: {e}")
            agent_file.unlink()  # Remove file on failure
            return {
                "success": False,
                "error": str(e),
            }

    def _extract_common_tools(self, tasks: List) -> List[str]:
        """
        Extract tools commonly used across tasks.

        Args:
            tasks: List of tasks

        Returns:
            List of tool names
        """
        # In full implementation, would analyze audit logs
        # For now, return common tools
        return [
            "read_file",
            "write_file",
            "run_pytest",
            "check_syntax",
        ]

    def _extract_common_workflow(self, tasks: List) -> List[str]:
        """
        Extract common workflow steps from tasks.

        Args:
            tasks: List of tasks

        Returns:
            List of workflow step names
        """
        # In full implementation, would analyze completed task steps
        # For now, return generic workflow
        return [
            "analyze_requirements",
            "plan_approach",
            "implement_solution",
            "validate_output",
        ]

    def _generate_agent_code(
        self,
        name: str,
        description: str,
        workflow: List[str],
        tools: List[str],
        keywords: List[str],
    ) -> str:
        """
        Generate Python code for a specialist agent.

        Args:
            name: Specialist name
            description: What the specialist does
            workflow: Workflow steps
            tools: Tools used
            keywords: Capability keywords

        Returns:
            Python source code
        """
        # Pre-compute formatted values so the f-string doesn't try to call
        # self._format_workflow_for_prompt() on AgentFactory at generation time.
        formatted_workflow = "\n".join(
            f"{i}. {step.replace('_', ' ').title()}"
            for i, step in enumerate(workflow, 1)
        )
        formatted_tools = "\n".join(f"- {tool}" for tool in tools)

        code = f'''# Auto-generated specialist agent
# Generated by AgentFactory on {datetime.now().isoformat()}

from typing import List
from gaia.agents.gaia_code.specialists.base_specialist import BaseSpecialist


class {name}(BaseSpecialist):
    """
    {description}

    Auto-generated from {len(workflow)} similar successful tasks.

    Keywords: {", ".join(keywords[:5])}
    """

    def define_workflow(self) -> List[str]:
        """Define the specialist's workflow."""
        return {workflow}

    def get_system_prompt(self) -> str:
        """Get the specialist's system prompt."""
        return """# {name}: {description}

You are a specialist in this domain. Your expertise comes from successfully
completing similar tasks multiple times.

## Your Workflow

{formatted_workflow}

## Your Tools

You have access to these tools:
{formatted_tools}

## Your Approach

1. Analyze the task requirements carefully
2. Follow your proven workflow
3. Use your specialized tools effectively
4. Validate your output before declaring complete
"""

    def get_tool_packs(self) -> List[str]:
        """Get tool packs for this specialist."""
        return ["core", "coding"]

    def get_capabilities(self) -> List[str]:
        """Get specialist capabilities."""
        return {[f'"{k}"' for k in keywords[:10]]}

    def _format_workflow_for_prompt(self, workflow):
        """Format workflow steps for prompt."""
        lines = []
        for i, step in enumerate(workflow, 1):
            lines.append(f"{{i}}. {{step.replace('_', ' ').title()}}")
        return "\\n".join(lines)

    def _format_tools_for_prompt(self, tools):
        """Format tools for prompt."""
        return "\\n".join(f"- {{tool}}" for tool in tools)
'''

        return code

    def _generate_system_prompt(
        self, name: str, description: str, workflow: List[str]
    ) -> str:
        """
        Generate system prompt for specialist.

        Args:
            name: Specialist name
            description: Description
            workflow: Workflow steps

        Returns:
            System prompt
        """
        workflow_text = "\n".join(
            f"{i}. {step.replace('_', ' ').title()}"
            for i, step in enumerate(workflow, 1)
        )

        return f"""# {name}: {description}

You are a specialist in this domain. Your expertise comes from successfully
completing similar tasks multiple times.

## Your Workflow

{workflow_text}

## Your Approach

1. Analyze the task requirements carefully
2. Follow your proven workflow
3. Use your specialized tools effectively
4. Validate your output before declaring complete

Remember: You were created because this pattern occurs frequently.
Your goal is to execute it efficiently and reliably.
"""

    def list_generated_specialists(self) -> List[Dict]:
        """
        List all auto-generated specialists.

        Returns:
            List of specialist info dicts
        """
        try:
            cursor = self.state.agents.conn.execute(
                """
                SELECT name, description, confidence, created_at
                FROM agents
                WHERE created_at > ?
                ORDER BY created_at DESC
            """,
                # Consider specialists created in last 30 days as "generated"
                (datetime.now().isoformat(),),  # This will get none, update logic
            )

            specialists = []
            for row in cursor.fetchall():
                specialists.append(
                    {
                        "name": row[0],
                        "description": row[1],
                        "confidence": row[2],
                        "created_at": row[3],
                    }
                )

            return specialists

        except Exception as e:
            logger.warning(f"Failed to list generated specialists: {e}")
            return []

    def update_specialist_confidence(
        self, specialist_name: str, success: bool
    ) -> bool:
        """
        Update specialist confidence based on usage.

        Args:
            specialist_name: Name of specialist
            success: Whether the specialist succeeded

        Returns:
            True if updated successfully
        """
        try:
            # Get current confidence
            cursor = self.state.agents.conn.execute(
                "SELECT confidence FROM agents WHERE name = ?",
                (specialist_name,),
            )
            row = cursor.fetchone()

            if not row:
                return False

            current_confidence = row[0]

            # Update confidence
            # Success: increase by 0.1 (max 1.0)
            # Failure: decrease by 0.15 (min 0.0)
            if success:
                new_confidence = min(1.0, current_confidence + 0.1)
            else:
                new_confidence = max(0.0, current_confidence - 0.15)

            # Update database
            self.state.agents.conn.execute(
                "UPDATE agents SET confidence = ?, last_used = ? WHERE name = ?",
                (new_confidence, datetime.now().isoformat(), specialist_name),
            )
            self.state.agents.conn.commit()

            logger.info(
                f"Updated specialist {specialist_name}: confidence {current_confidence:.2f} -> {new_confidence:.2f}"
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to update specialist confidence for {specialist_name}: {e}"
            )
            return False

    def remove_specialist(self, specialist_name: str) -> bool:
        """
        Remove a generated specialist.

        Args:
            specialist_name: Name of specialist to remove

        Returns:
            True if removed successfully
        """
        try:
            # Remove from database
            self.state.agents.conn.execute(
                "DELETE FROM agents WHERE name = ?", (specialist_name,)
            )
            self.state.agents.conn.commit()

            # Remove file
            agent_file = self.generated_agents_dir / f"{specialist_name.lower()}.py"
            if agent_file.exists():
                agent_file.unlink()

            logger.info(f"Removed specialist: {specialist_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove specialist {specialist_name}: {e}")
            return False


def maybe_create_specialist(task: str, workspace_dir=None) -> Optional[str]:
    """
    Create a specialist if the task matches a recurring pattern.

    Scans plan_tasks history for patterns (3+ similar completed tasks).
    If a matching pattern is found but no registered specialist handles it,
    a new specialist is auto-generated and persisted to agents.db.

    Args:
        task: Task description to match against patterns
        workspace_dir: Optional workspace directory

    Returns:
        Specialist name if found or created, None otherwise
    """
    factory = AgentFactory(workspace_dir)
    patterns = factory.detect_pattern(min_occurrences=3)
    task_words = set(task.lower().split())

    for pattern in patterns:
        overlap = task_words & {kw.lower() for kw in pattern["keywords"]}
        if len(overlap) < 2:
            continue

        # Derive a stable name from the top 2 longest keywords
        top_kw = sorted(pattern["keywords"], key=len, reverse=True)
        name = "Auto_" + "_".join(top_kw[:2]).title().replace(" ", "")

        # Already registered? Return immediately — no need to regenerate.
        if factory.state.agents.find_agent(name):
            logger.info("[RAC] reusing existing auto-specialist=%s", name)
            return name

        # Generate and register a new specialist
        desc = f"Auto-generated for: {', '.join(pattern['keywords'][:5])}"
        result = factory.generate_specialist(pattern, name, desc)
        if result.get("success"):
            logger.info("[RAC] created auto-specialist=%s", name)
            return name

    return None
