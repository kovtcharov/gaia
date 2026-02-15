# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
SkillExtractor: Learn workflows from successful task executions.

M5: Agent Auto-Generation + Self-Extension

The SkillExtractor enables the agent to:
- Extract multi-step workflows from completed tasks
- Store skills in skills.db
- Recall skills for similar future tasks
- Track skill success rates

Skills are higher-level than tools - they're patterns of tool usage.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from .shared_state import get_shared_state

logger = logging.getLogger(__name__)


class SkillExtractor:
    """
    Extracts reusable skills from successful task executions.

    A skill is a multi-step workflow that can be reused.
    Example: "Build FastAPI app with JWT auth" is a skill that uses
    multiple tools in a specific sequence.
    """

    def __init__(self, workspace_dir=None):
        self.state = get_shared_state(workspace_dir)

    def extract_skill(
        self,
        name: str,
        description: str,
        task_description: str,
        steps: List[str],
        tools_used: List[str],
        category: str = "workflow",
        domain: str = "coding",
    ) -> Dict:
        """
        Extract a skill from a successful task execution.

        Args:
            name: Skill name (e.g., "build_fastapi_jwt_app")
            description: What the skill does
            task_description: Original task that led to this skill
            steps: List of step descriptions
            tools_used: List of tool names used
            category: Skill category
            domain: Domain (coding, testing, etc.)

        Returns:
            Dict with success status and skill ID
        """
        # Validate inputs
        if not name or not description or not steps:
            return {
                "success": False,
                "error": "Missing required fields (name, description, or steps)",
            }

        # Check if skill already exists
        existing = self._find_similar_skill(name, description)
        if existing:
            # Update existing skill instead of creating duplicate
            return self._update_skill(existing["id"], steps, tools_used)

        # Create skill ID
        skill_id = str(uuid4())

        # Store in skills.db
        try:
            self.state.skills.conn.execute(
                """
                INSERT INTO skills (id, name, description, category, domain,
                                   steps, tools_used, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    skill_id,
                    name,
                    description,
                    category,
                    domain,
                    json.dumps(steps),
                    json.dumps(tools_used),
                    0.5,  # Initial confidence
                    datetime.now().isoformat(),
                ),
            )
            self.state.skills.conn.commit()

            logger.info(f"Extracted skill: {name} (ID: {skill_id})")

            return {
                "success": True,
                "skill_id": skill_id,
                "skill_name": name,
            }

        except Exception as e:
            logger.error(f"Failed to extract skill {name}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _find_similar_skill(
        self, name: str, description: str
    ) -> Optional[Dict]:
        """
        Find a similar existing skill.

        Args:
            name: Skill name
            description: Skill description

        Returns:
            Skill info dict or None
        """
        try:
            # Simple name match for now
            # In full implementation, would use semantic similarity
            cursor = self.state.skills.conn.execute(
                "SELECT id, name, description, confidence FROM skills WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "confidence": row[3],
                }

            return None

        except Exception as e:
            logger.warning(f"Failed to find similar skill: {e}")
            return None

    def _update_skill(
        self, skill_id: str, steps: List[str], tools_used: List[str]
    ) -> Dict:
        """
        Update an existing skill.

        Args:
            skill_id: ID of skill to update
            steps: Updated steps
            tools_used: Updated tools

        Returns:
            Dict with success status
        """
        try:
            self.state.skills.conn.execute(
                """
                UPDATE skills
                SET steps = ?, tools_used = ?, last_used = ?
                WHERE id = ?
            """,
                (
                    json.dumps(steps),
                    json.dumps(tools_used),
                    datetime.now().isoformat(),
                    skill_id,
                ),
            )
            self.state.skills.conn.commit()

            logger.info(f"Updated skill: {skill_id}")

            return {
                "success": True,
                "skill_id": skill_id,
            }

        except Exception as e:
            logger.error(f"Failed to update skill {skill_id}: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def recall_skill(self, task_description: str, top_k: int = 3) -> List[Dict]:
        """
        Find skills relevant to a task.

        Args:
            task_description: Description of the task
            top_k: Number of skills to return

        Returns:
            List of skill info dicts
        """
        try:
            # Simple keyword matching for now
            # In full implementation, would use FAISS semantic search
            cursor = self.state.skills.conn.execute(
                """
                SELECT id, name, description, steps, tools_used, confidence
                FROM skills
                ORDER BY confidence DESC, last_used DESC
                LIMIT ?
            """,
                (top_k,),
            )

            skills = []
            for row in cursor.fetchall():
                skills.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "steps": json.loads(row[3]) if row[3] else [],
                        "tools_used": json.loads(row[4]) if row[4] else [],
                        "confidence": row[5],
                    }
                )

            return skills

        except Exception as e:
            logger.warning(f"Failed to recall skills: {e}")
            return []

    def record_skill_usage(
        self, skill_id: str, success: bool, task_description: str
    ) -> bool:
        """
        Record that a skill was used.

        Updates:
        - Usage count
        - Success rate
        - Confidence score

        Args:
            skill_id: ID of skill used
            success: Whether the skill succeeded
            task_description: Description of the task

        Returns:
            True if recorded successfully
        """
        try:
            # Record usage
            self.state.skills.conn.execute(
                """
                INSERT INTO skill_usage (skill_id, success, task_description, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (skill_id, success, task_description, datetime.now().isoformat()),
            )

            # Update skill stats
            if success:
                self.state.skills.conn.execute(
                    """
                    UPDATE skills
                    SET success_count = success_count + 1,
                        last_used = ?
                    WHERE id = ?
                """,
                    (datetime.now().isoformat(), skill_id),
                )
            else:
                self.state.skills.conn.execute(
                    """
                    UPDATE skills
                    SET failure_count = failure_count + 1
                    WHERE id = ?
                """,
                    (skill_id,),
                )

            # Recalculate confidence
            self._recalculate_confidence(skill_id)

            self.state.skills.conn.commit()

            logger.info(f"Recorded skill usage: {skill_id} (success={success})")
            return True

        except Exception as e:
            logger.error(f"Failed to record skill usage: {e}")
            return False

    def _recalculate_confidence(self, skill_id: str):
        """
        Recalculate confidence score for a skill.

        Confidence = success_rate * sqrt(total_uses) / 10

        Args:
            skill_id: ID of skill
        """
        try:
            cursor = self.state.skills.conn.execute(
                """
                SELECT success_count, failure_count
                FROM skills
                WHERE id = ?
            """,
                (skill_id,),
            )
            row = cursor.fetchone()

            if row:
                success_count = row[0]
                failure_count = row[1]
                total_uses = success_count + failure_count

                if total_uses > 0:
                    success_rate = success_count / total_uses
                    # Confidence increases with both success rate and usage
                    confidence = min(1.0, success_rate * (total_uses**0.5) / 10)

                    self.state.skills.conn.execute(
                        "UPDATE skills SET confidence = ? WHERE id = ?",
                        (confidence, skill_id),
                    )

        except Exception as e:
            logger.warning(f"Failed to recalculate confidence for {skill_id}: {e}")

    def list_skills(
        self, category: Optional[str] = None, domain: Optional[str] = None
    ) -> List[Dict]:
        """
        List all skills.

        Args:
            category: Optional category filter
            domain: Optional domain filter

        Returns:
            List of skill info dicts
        """
        try:
            query = "SELECT id, name, description, category, domain, confidence, success_count, failure_count FROM skills"
            params = []

            if category and domain:
                query += " WHERE category = ? AND domain = ?"
                params = [category, domain]
            elif category:
                query += " WHERE category = ?"
                params = [category]
            elif domain:
                query += " WHERE domain = ?"
                params = [domain]

            query += " ORDER BY confidence DESC"

            cursor = self.state.skills.conn.execute(query, params)

            skills = []
            for row in cursor.fetchall():
                total_uses = row[6] + row[7]
                success_rate = (row[6] / total_uses * 100) if total_uses > 0 else 0

                skills.append(
                    {
                        "id": row[0],
                        "name": row[1],
                        "description": row[2],
                        "category": row[3],
                        "domain": row[4],
                        "confidence": row[5],
                        "total_uses": total_uses,
                        "success_rate": success_rate,
                    }
                )

            return skills

        except Exception as e:
            logger.warning(f"Failed to list skills: {e}")
            return []

    def remove_skill(self, skill_id: str) -> bool:
        """
        Remove a skill.

        Args:
            skill_id: ID of skill to remove

        Returns:
            True if removed successfully
        """
        try:
            self.state.skills.conn.execute(
                "DELETE FROM skills WHERE id = ?", (skill_id,)
            )
            self.state.skills.conn.commit()

            logger.info(f"Removed skill: {skill_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove skill {skill_id}: {e}")
            return False

    def get_skill_stats(self) -> Dict:
        """
        Get statistics about all skills.

        Returns:
            Dict with statistics
        """
        try:
            cursor = self.state.skills.conn.execute(
                """
                SELECT
                    COUNT(*) as total_skills,
                    AVG(confidence) as avg_confidence,
                    SUM(success_count) as total_successes,
                    SUM(failure_count) as total_failures
                FROM skills
            """
            )
            row = cursor.fetchone()

            total_uses = (row[2] or 0) + (row[3] or 0)
            success_rate = (
                ((row[2] or 0) / total_uses * 100) if total_uses > 0 else 0
            )

            return {
                "total_skills": row[0] or 0,
                "avg_confidence": row[1] or 0,
                "total_uses": total_uses,
                "success_rate": success_rate,
            }

        except Exception as e:
            logger.warning(f"Failed to get skill stats: {e}")
            return {}
