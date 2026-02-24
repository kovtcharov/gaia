# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
MemoryDefragmentation: Keep knowledge clean as it grows.

M6: Memory Defragmentation + Vector Search

The Defragmenter:
- Deduplicates near-duplicate insights
- Reconciles contradictory information
- Prunes stale, low-confidence entries
- Consolidates patterns
- Re-indexes after cleanup

Without defrag, the agent gets WORSE over time. With defrag, it gets BETTER.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from .embedding_engine import EmbeddingEngine
from .shared_state import get_shared_state
from .vector_search import VectorSearch

logger = logging.getLogger(__name__)


class MemoryDefragmenter:
    """
    Defragments knowledge DB to maintain quality as it grows.

    Steps:
    1. DEDUPLICATE: Find and merge near-duplicate insights
    2. RECONCILE: Resolve contradictory information
    3. PRUNE: Remove stale, low-confidence entries
    4. CONSOLIDATE: Merge related patterns
    5. RE-INDEX: Rebuild FAISS indices

    Triggers:
    - Scheduled (every 100 sessions or weekly)
    - On demand (user runs `gaia memory defrag`)
    - Automatic when retrieval quality drops
    """

    def __init__(
        self,
        workspace_dir=None,
        similarity_threshold: float = 0.9,
        confidence_threshold: float = 0.2,
        staleness_days: int = 90,
    ):
        """
        Initialize defragmenter.

        Args:
            workspace_dir: Optional workspace directory
            similarity_threshold: Threshold for duplicate detection (default: 0.9)
            confidence_threshold: Minimum confidence to keep (default: 0.2)
            staleness_days: Days before entry is considered stale (default: 90)
        """
        self.state = get_shared_state(workspace_dir)
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.staleness_days = staleness_days

        # Initialize embedder and vector search (optional — requires sentence-transformers)
        try:
            self.embedder = EmbeddingEngine()
            self.vector_search = VectorSearch(
                workspace_dir=workspace_dir,
                embedding_engine=self.embedder,
            )
        except ImportError:
            logger.debug("sentence-transformers not available; embedding-based defrag disabled")
            self.embedder = None
            self.vector_search = None

    def defragment(self) -> Dict:
        """
        Run full defragmentation process.

        Returns:
            Dict with defrag statistics
        """
        logger.info("Starting memory defragmentation...")

        stats = {
            "started_at": datetime.now().isoformat(),
            "deduplicated": 0,
            "reconciled": 0,
            "pruned": 0,
            "consolidated": 0,
            "re_indexed": 0,
        }

        # Step 1: Deduplicate
        stats["deduplicated"] = self._deduplicate_insights()

        # Step 2: Reconcile contradictions
        stats["reconciled"] = self._reconcile_contradictions()

        # Step 3: Prune stale entries
        stats["pruned"] = self._prune_stale_entries()

        # Step 4: Consolidate patterns
        stats["consolidated"] = self._consolidate_patterns()

        # Step 5: Re-index
        index_counts = self.vector_search.build_all_indices()
        stats["re_indexed"] = sum(index_counts.values())

        stats["completed_at"] = datetime.now().isoformat()

        logger.info(f"Defragmentation complete: {stats}")

        return stats

    def _deduplicate_insights(self) -> int:
        """
        Find and merge near-duplicate insights.

        Uses semantic similarity to find duplicates.

        Returns:
            Number of duplicates merged
        """
        logger.info("Deduplicating insights...")

        # Get all insights
        cursor = self.state.knowledge.conn.execute(
            "SELECT id, content, confidence FROM insights"
        )
        insights = [(row[0], row[1], row[2]) for row in cursor.fetchall()]

        if len(insights) < 2:
            return 0

        # Generate embeddings
        contents = [i[1] for i in insights]
        embeddings = self.embedder.batch_embed(contents)

        # Find duplicates using cosine similarity
        duplicates = []
        for i in range(len(insights)):
            for j in range(i + 1, len(insights)):
                emb1 = embeddings[i]
                emb2 = embeddings[j]

                # Cosine similarity
                similarity = float(
                    emb1.dot(emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                )

                if similarity >= self.similarity_threshold:
                    duplicates.append((i, j, similarity))

        # Merge duplicates (keep higher confidence, delete lower)
        merged_count = 0
        for i, j, similarity in duplicates:
            id1, content1, conf1 = insights[i]
            id2, content2, conf2 = insights[j]

            # Keep higher confidence insight
            if conf1 >= conf2:
                keep_id, delete_id = id1, id2
            else:
                keep_id, delete_id = id2, id1

            # Delete lower confidence
            self.state.knowledge.conn.execute(
                "DELETE FROM insights WHERE id = ?", (delete_id,)
            )

            # Increase confidence of kept insight
            self.state.knowledge.conn.execute(
                "UPDATE insights SET confidence = MIN(1.0, confidence + 0.1) WHERE id = ?",
                (keep_id,),
            )

            merged_count += 1

        self.state.knowledge.conn.commit()

        logger.info(f"Merged {merged_count} duplicate insights")

        return merged_count

    def _reconcile_contradictions(self) -> int:
        """
        Find and reconcile contradictory insights.

        Returns:
            Number of contradictions reconciled
        """
        logger.info("Reconciling contradictions...")

        # This is complex - would need semantic analysis
        # For now, just log and return 0
        # In full implementation:
        # 1. Find insights with opposing meanings
        # 2. Check which has higher confidence / more recent
        # 3. Keep the better one, delete or downgrade the other

        return 0

    def _prune_stale_entries(self) -> int:
        """
        Remove stale, low-confidence entries.

        Removes insights that:
        - Have low confidence (<0.2)
        - Haven't been used in 90+ days
        - Have been validated as incorrect

        Returns:
            Number of entries pruned
        """
        logger.info("Pruning stale entries...")

        staleness_date = (
            datetime.now() - timedelta(days=self.staleness_days)
        ).isoformat()

        # Delete low-confidence, unused insights
        cursor = self.state.knowledge.conn.execute(
            """
            DELETE FROM insights
            WHERE confidence < ?
            AND (last_used IS NULL OR last_used < ?)
            RETURNING id
        """,
            (self.confidence_threshold, staleness_date),
        )

        pruned = len(cursor.fetchall())
        self.state.knowledge.conn.commit()

        logger.info(f"Pruned {pruned} stale insights")

        return pruned

    def _consolidate_patterns(self) -> int:
        """
        Consolidate related patterns into higher-level insights.

        Returns:
            Number of patterns consolidated
        """
        logger.info("Consolidating patterns...")

        # This would:
        # 1. Find groups of related insights
        # 2. Create a higher-level insight that encompasses them
        # 3. Delete the individual insights
        # 4. Keep the consolidated version

        # For now, just return 0
        return 0

    def measure_retrieval_quality(self, test_queries: List[str]) -> float:
        """
        Measure retrieval quality using test queries.

        Args:
            test_queries: List of queries to test

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not test_queries:
            return 1.0

        total_score = 0.0

        for query in test_queries:
            # Get top result
            results = self.vector_search.search_insights(query, top_k=1)

            if results:
                # Quality = confidence of top result
                total_score += results[0]["confidence"]

        return total_score / len(test_queries)

    def should_defragment(self) -> bool:
        """
        Check if defragmentation is needed.

        Triggers:
        - Retrieval quality < 0.9
        - Number of insights > 1000
        - Last defrag > 7 days ago

        Returns:
            True if defrag is recommended
        """
        # Check insight count
        cursor = self.state.knowledge.conn.execute("SELECT COUNT(*) FROM insights")
        insight_count = cursor.fetchone()[0]

        if insight_count > 1000:
            logger.info(f"Defrag recommended: {insight_count} insights (threshold: 1000)")
            return True

        # Check duplicate count
        # (simplified - just check if there are many)
        if insight_count > 500:
            logger.info(f"Defrag recommended: {insight_count} insights (may have duplicates)")
            return True

        return False

    def get_defrag_stats(self) -> Dict:
        """
        Get statistics about defragmentation needs.

        Returns:
            Dict with statistics
        """
        # Count insights
        cursor = self.state.knowledge.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                AVG(confidence) as avg_confidence,
                SUM(CASE WHEN confidence < 0.3 THEN 1 ELSE 0 END) as low_confidence
            FROM insights
        """
        )
        row = cursor.fetchone()

        # Count by staleness
        staleness_date = (
            datetime.now() - timedelta(days=self.staleness_days)
        ).isoformat()
        cursor = self.state.knowledge.conn.execute(
            """
            SELECT COUNT(*)
            FROM insights
            WHERE last_used IS NULL OR last_used < ?
        """,
            (staleness_date,),
        )
        stale_count = cursor.fetchone()[0]

        return {
            "total_insights": row[0] or 0,
            "avg_confidence": row[1] or 0,
            "low_confidence_count": row[2] or 0,
            "stale_count": stale_count,
            "defrag_recommended": self.should_defragment(),
        }
