# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
InsightEngine: Structured learning and insight generation.

M5: Agent Auto-Generation + Self-Extension

The InsightEngine enables the agent to:
- Extract insights from completed tasks
- Store insights with metadata for retrieval
- Retrieve insights by semantic similarity + triggers + domain
- Track insight confidence and validation

Insights are learnings that improve future performance.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .shared_state import get_shared_state

logger = logging.getLogger(__name__)


class InsightEngine:
    """
    Generates and manages insights from agent experience.

    Insight categories:
    - error_fix: Error patterns and their solutions
    - pattern: Reusable code/workflow patterns
    - preference: User preferences learned over time
    - convention: Project-specific conventions
    - best_practice: General best practices learned
    """

    def __init__(self, workspace_dir=None):
        self.state = get_shared_state(workspace_dir)

    def generate_insight(
        self,
        category: str,
        content: str,
        domain: Optional[str] = None,
        triggers: Optional[List[str]] = None,
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate and store a new insight.

        Args:
            category: Insight category (error_fix, pattern, preference, convention, best_practice)
            content: The insight content
            domain: Optional domain (coding, testing, etc.)
            triggers: Optional trigger keywords for retrieval
            context: Optional context information

        Returns:
            Dict with success status and insight ID
        """
        # Validate category
        valid_categories = [
            "error_fix",
            "pattern",
            "preference",
            "convention",
            "best_practice",
        ]
        if category not in valid_categories:
            return {
                "success": False,
                "error": f"Invalid category. Must be one of: {valid_categories}",
            }

        # Store insight
        try:
            insight_id = self.state.knowledge.store_insight(
                category=category,
                content=content,
                domain=domain,
                triggers=triggers,
            )

            logger.info(
                f"Generated insight: {category} - {content[:50]}... (ID: {insight_id})"
            )

            return {
                "success": True,
                "insight_id": insight_id,
                "category": category,
            }

        except Exception as e:
            logger.error(f"Failed to generate insight: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def generate_error_fix_insight(
        self, error_type: str, error_message: str, fix_description: str
    ) -> Dict:
        """
        Generate an insight from an error-fix pair.

        Args:
            error_type: Type of error (SyntaxError, AttributeError, etc.)
            error_message: The error message
            fix_description: How the error was fixed

        Returns:
            Dict with success status
        """
        content = f"Error: {error_type} - {error_message}\nFix: {fix_description}"

        return self.generate_insight(
            category="error_fix",
            content=content,
            domain="coding",
            triggers=[error_type, "error", "fix"],
        )

    def generate_pattern_insight(
        self, pattern_name: str, pattern_description: str, use_cases: List[str]
    ) -> Dict:
        """
        Generate an insight from a reusable pattern.

        Args:
            pattern_name: Name of the pattern
            pattern_description: What the pattern does
            use_cases: When to use this pattern

        Returns:
            Dict with success status
        """
        content = f"Pattern: {pattern_name}\nDescription: {pattern_description}\nUse cases: {', '.join(use_cases)}"

        return self.generate_insight(
            category="pattern",
            content=content,
            domain="coding",
            triggers=[pattern_name, "pattern"],
        )

    def retrieve_insights(
        self,
        query: str,
        category: Optional[str] = None,
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve insights relevant to a query.

        Uses:
        - FTS5 full-text search
        - Trigger keyword matching
        - Domain filtering
        - Confidence scoring

        Args:
            query: Search query
            category: Optional category filter
            domain: Optional domain filter
            top_k: Number of insights to return

        Returns:
            List of insight dicts
        """
        # Use knowledge DB's recall method for FTS5 search
        insights = self.state.knowledge.recall(query, top_k=top_k * 2)

        # Filter by category and domain if specified
        filtered = []
        for insight in insights:
            if category and insight.get("category") != category:
                continue
            if domain and insight.get("domain") != domain:
                continue
            filtered.append(insight)

        # Sort by confidence and return top_k
        filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        return filtered[:top_k]

    def validate_insight(self, insight_id: str, validated: bool) -> bool:
        """
        Validate an insight (increases or decreases confidence).

        Args:
            insight_id: ID of insight to validate
            validated: Whether the insight proved useful

        Returns:
            True if updated successfully
        """
        try:
            # Get current confidence
            cursor = self.state.knowledge.conn.execute(
                "SELECT confidence, use_count FROM insights WHERE id = ?",
                (insight_id,),
            )
            row = cursor.fetchone()

            if not row:
                return False

            current_confidence = row[0]
            use_count = row[1] or 0

            # Update confidence
            # Validated: increase by 0.1 (max 1.0)
            # Not validated: decrease by 0.05 (min 0.0)
            if validated:
                new_confidence = min(1.0, current_confidence + 0.1)
            else:
                new_confidence = max(0.0, current_confidence - 0.05)

            # Update database
            self.state.knowledge.conn.execute(
                """
                UPDATE insights
                SET confidence = ?, use_count = ?, last_used = ?
                WHERE id = ?
            """,
                (
                    new_confidence,
                    use_count + 1,
                    datetime.now().isoformat(),
                    insight_id,
                ),
            )
            self.state.knowledge.conn.commit()

            logger.info(
                f"Validated insight {insight_id}: confidence {current_confidence:.2f} -> {new_confidence:.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to validate insight {insight_id}: {e}")
            return False

    def get_insight_stats(self) -> Dict:
        """
        Get statistics about all insights.

        Returns:
            Dict with statistics
        """
        try:
            cursor = self.state.knowledge.conn.execute(
                """
                SELECT
                    COUNT(*) as total_insights,
                    AVG(confidence) as avg_confidence,
                    SUM(use_count) as total_uses
                FROM insights
            """
            )
            row = cursor.fetchone()

            # Count by category
            cursor = self.state.knowledge.conn.execute(
                """
                SELECT category, COUNT(*) as count
                FROM insights
                GROUP BY category
            """
            )
            by_category = {row[0]: row[1] for row in cursor.fetchall()}

            # Count by domain
            cursor = self.state.knowledge.conn.execute(
                """
                SELECT domain, COUNT(*) as count
                FROM insights
                GROUP BY domain
            """
            )
            by_domain = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_insights": row[0] or 0,
                "avg_confidence": row[1] or 0,
                "total_uses": row[2] or 0,
                "by_category": by_category,
                "by_domain": by_domain,
            }

        except Exception as e:
            logger.warning(f"Failed to get insight stats: {e}")
            return {}

    def prune_low_confidence_insights(self, threshold: float = 0.2) -> int:
        """
        Remove insights with very low confidence.

        Args:
            threshold: Minimum confidence to keep (default: 0.2)

        Returns:
            Number of insights removed
        """
        try:
            cursor = self.state.knowledge.conn.execute(
                "DELETE FROM insights WHERE confidence < ? RETURNING id",
                (threshold,),
            )
            removed = len(cursor.fetchall())
            self.state.knowledge.conn.commit()

            logger.info(f"Pruned {removed} low-confidence insights (threshold: {threshold})")

            return removed

        except Exception as e:
            logger.error(f"Failed to prune insights: {e}")
            return 0

    def export_insights(self, output_file: str) -> bool:
        """
        Export all insights to a file.

        Args:
            output_file: Path to output file

        Returns:
            True if exported successfully
        """
        try:
            import json

            cursor = self.state.knowledge.conn.execute(
                """
                SELECT id, category, domain, content, confidence, triggers,
                       created_at, last_used, use_count
                FROM insights
                ORDER BY category, confidence DESC
            """
            )

            insights = []
            for row in cursor.fetchall():
                insights.append(
                    {
                        "id": row[0],
                        "category": row[1],
                        "domain": row[2],
                        "content": row[3],
                        "confidence": row[4],
                        "triggers": row[5],
                        "created_at": row[6],
                        "last_used": row[7],
                        "use_count": row[8],
                    }
                )

            with open(output_file, "w") as f:
                json.dump(insights, f, indent=2)

            logger.info(f"Exported {len(insights)} insights to {output_file}")

            return True

        except Exception as e:
            logger.error(f"Failed to export insights: {e}")
            return False
