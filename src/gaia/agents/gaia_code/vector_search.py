# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
VectorSearch: FAISS-based semantic search for knowledge, tools, skills, and agents.

M6: Memory Defragmentation + Vector Search

Provides:
- FAISS indices for each database
- Semantic search (find by meaning, not just keywords)
- Hybrid search (FAISS + FTS5 combined)
- Index management (build, update, persist)
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embedding_engine import EmbeddingEngine
from .shared_state import get_shared_state

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")


class VectorSearch:
    """
    FAISS-based semantic search for knowledge, tools, skills, and agents.

    Features:
    - Fast approximate nearest neighbor search
    - Semantic similarity (not just keyword matching)
    - Hybrid search (vector + FTS5)
    - Incremental index updates
    """

    def __init__(
        self,
        workspace_dir: Optional[Path] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
    ):
        """
        Initialize vector search.

        Args:
            workspace_dir: Optional workspace directory
            embedding_engine: Optional embedding engine (creates if not provided)
        """
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

        self.workspace_dir = workspace_dir or Path.home() / ".gaia" / "workspace"
        self.indices_dir = self.workspace_dir / "faiss_indices"
        self.indices_dir.mkdir(parents=True, exist_ok=True)

        self.state = get_shared_state(workspace_dir)

        # Initialize embedding engine
        if embedding_engine:
            self.embedder = embedding_engine
        else:
            self.embedder = EmbeddingEngine()

        # FAISS indices (lazy loaded)
        self.indices = {
            "insights": None,
            "tools": None,
            "skills": None,
            "agents": None,
        }

        # ID mappings (FAISS index position → database ID)
        self.id_mappings = {
            "insights": [],
            "tools": [],
            "skills": [],
            "agents": [],
        }

    def build_insights_index(self) -> int:
        """
        Build FAISS index for insights.

        Returns:
            Number of insights indexed
        """
        logger.info("Building insights index...")

        # Get all insights
        cursor = self.state.knowledge.conn.execute(
            "SELECT id, content FROM insights"
        )
        insights = cursor.fetchall()

        if not insights:
            logger.info("No insights to index")
            return 0

        # Generate embeddings
        ids = [row[0] for row in insights]
        contents = [row[1] for row in insights]
        embeddings = self.embedder.batch_embed(contents)

        # Build FAISS index
        dimension = self.embedder.get_dimension()
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save index and mappings
        self.indices["insights"] = index
        self.id_mappings["insights"] = ids

        self._save_index("insights", index, ids)

        logger.info(f"Built insights index: {len(ids)} entries")

        return len(ids)

    def build_tools_index(self) -> int:
        """
        Build FAISS index for tools.

        Returns:
            Number of tools indexed
        """
        logger.info("Building tools index...")

        # Get all tools
        cursor = self.state.tools.conn.execute(
            "SELECT id, description FROM tools WHERE enabled = TRUE"
        )
        tools = cursor.fetchall()

        if not tools:
            logger.info("No tools to index")
            return 0

        # Generate embeddings
        ids = [row[0] for row in tools]
        descriptions = [row[1] for row in tools]
        embeddings = self.embedder.batch_embed(descriptions)

        # Build FAISS index
        dimension = self.embedder.get_dimension()
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save index and mappings
        self.indices["tools"] = index
        self.id_mappings["tools"] = ids

        self._save_index("tools", index, ids)

        logger.info(f"Built tools index: {len(ids)} entries")

        return len(ids)

    def build_skills_index(self) -> int:
        """
        Build FAISS index for skills.

        Returns:
            Number of skills indexed
        """
        logger.info("Building skills index...")

        # Get all skills
        cursor = self.state.skills.conn.execute(
            "SELECT id, description FROM skills"
        )
        skills = cursor.fetchall()

        if not skills:
            logger.info("No skills to index")
            return 0

        # Generate embeddings
        ids = [row[0] for row in skills]
        descriptions = [row[1] for row in skills]
        embeddings = self.embedder.batch_embed(descriptions)

        # Build FAISS index
        dimension = self.embedder.get_dimension()
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save index and mappings
        self.indices["skills"] = index
        self.id_mappings["skills"] = ids

        self._save_index("skills", index, ids)

        logger.info(f"Built skills index: {len(ids)} entries")

        return len(ids)

    def build_agents_index(self) -> int:
        """
        Build FAISS index for agents.

        Returns:
            Number of agents indexed
        """
        logger.info("Building agents index...")

        # Get all agents
        cursor = self.state.agents.conn.execute(
            "SELECT id, description FROM agents"
        )
        agents = cursor.fetchall()

        if not agents:
            logger.info("No agents to index")
            return 0

        # Generate embeddings
        ids = [row[0] for row in agents]
        descriptions = [row[1] for row in agents]
        embeddings = self.embedder.batch_embed(descriptions)

        # Build FAISS index
        dimension = self.embedder.get_dimension()
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # Save index and mappings
        self.indices["agents"] = index
        self.id_mappings["agents"] = ids

        self._save_index("agents", index, ids)

        logger.info(f"Built agents index: {len(ids)} entries")

        return len(ids)

    def build_all_indices(self) -> Dict[str, int]:
        """
        Build all FAISS indices.

        Returns:
            Dict with counts for each index
        """
        counts = {
            "insights": self.build_insights_index(),
            "tools": self.build_tools_index(),
            "skills": self.build_skills_index(),
            "agents": self.build_agents_index(),
        }

        logger.info(f"Built all indices: {counts}")

        return counts

    def search_insights(
        self, query: str, top_k: int = 5, min_confidence: float = 0.3
    ) -> List[Dict]:
        """
        Search insights using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results
            min_confidence: Minimum confidence threshold

        Returns:
            List of insight dicts
        """
        # Load index if not loaded
        if self.indices["insights"] is None:
            self._load_index("insights")

        if self.indices["insights"] is None:
            # Fall back to FTS5
            return self.state.knowledge.recall(query, top_k=top_k)

        # Generate query embedding
        query_emb = self.embedder.embed(query).reshape(1, -1)

        # Search FAISS index
        distances, indices = self.indices["insights"].search(query_emb, top_k)

        # Get insight IDs
        results = []
        for idx in indices[0]:
            if idx < len(self.id_mappings["insights"]):
                insight_id = self.id_mappings["insights"][idx]

                # Get insight from database
                cursor = self.state.knowledge.conn.execute(
                    """
                    SELECT id, category, domain, content, confidence
                    FROM insights
                    WHERE id = ? AND confidence >= ?
                """,
                    (insight_id, min_confidence),
                )
                row = cursor.fetchone()

                if row:
                    results.append(
                        {
                            "id": row[0],
                            "category": row[1],
                            "domain": row[2],
                            "content": row[3],
                            "confidence": row[4],
                        }
                    )

        return results

    def search_tools(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search tools using semantic similarity.

        Args:
            query: Search query (e.g., "create GitHub PR")
            top_k: Number of results

        Returns:
            List of tool dicts
        """
        # Load index if not loaded
        if self.indices["tools"] is None:
            self._load_index("tools")

        if self.indices["tools"] is None:
            # Fall back to FTS5
            return self.state.tools.find_tools(query, top_k=top_k)

        # Generate query embedding
        query_emb = self.embedder.embed(query).reshape(1, -1)

        # Search FAISS index
        distances, indices = self.indices["tools"].search(query_emb, top_k)

        # Get tool IDs
        results = []
        for idx in indices[0]:
            if idx < len(self.id_mappings["tools"]):
                tool_id = self.id_mappings["tools"][idx]

                # Get tool from database
                cursor = self.state.tools.conn.execute(
                    """
                    SELECT id, name, category, description
                    FROM tools
                    WHERE id = ? AND enabled = TRUE
                """,
                    (tool_id,),
                )
                row = cursor.fetchone()

                if row:
                    results.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "category": row[2],
                            "description": row[3],
                        }
                    )

        return results

    def search_agents(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search specialists using semantic similarity.

        Args:
            query: Search query (e.g., "debug async code")
            top_k: Number of results

        Returns:
            List of agent dicts
        """
        # Load index if not loaded
        if self.indices["agents"] is None:
            self._load_index("agents")

        if self.indices["agents"] is None:
            logger.warning("No agents index available")
            return []

        # Generate query embedding
        query_emb = self.embedder.embed(query).reshape(1, -1)

        # Search FAISS index
        distances, indices = self.indices["agents"].search(query_emb, top_k)

        # Get agent IDs
        results = []
        for idx in indices[0]:
            if idx < len(self.id_mappings["agents"]):
                agent_id = self.id_mappings["agents"][idx]

                # Get agent from database
                cursor = self.state.agents.conn.execute(
                    """
                    SELECT id, name, description, confidence
                    FROM agents
                    WHERE id = ?
                """,
                    (agent_id,),
                )
                row = cursor.fetchone()

                if row:
                    results.append(
                        {
                            "id": row[0],
                            "name": row[1],
                            "description": row[2],
                            "confidence": row[3],
                        }
                    )

        return results

    def _save_index(self, index_name: str, index: "faiss.Index", id_mapping: List[str]):
        """
        Save FAISS index to disk.

        Args:
            index_name: Name of index
            index: FAISS index
            id_mapping: ID mapping list
        """
        index_file = self.indices_dir / f"{index_name}.faiss"
        mapping_file = self.indices_dir / f"{index_name}_mapping.pkl"

        # Save FAISS index
        faiss.write_index(index, str(index_file))

        # Save ID mapping
        with open(mapping_file, "wb") as f:
            pickle.dump(id_mapping, f)

        logger.debug(f"Saved {index_name} index to {index_file}")

    def _load_index(self, index_name: str) -> bool:
        """
        Load FAISS index from disk.

        Args:
            index_name: Name of index

        Returns:
            True if loaded successfully
        """
        index_file = self.indices_dir / f"{index_name}.faiss"
        mapping_file = self.indices_dir / f"{index_name}_mapping.pkl"

        if not index_file.exists() or not mapping_file.exists():
            logger.debug(f"Index {index_name} not found on disk")
            return False

        try:
            # Load FAISS index
            index = faiss.read_index(str(index_file))
            self.indices[index_name] = index

            # Load ID mapping
            with open(mapping_file, "rb") as f:
                id_mapping = pickle.load(f)
            self.id_mappings[index_name] = id_mapping

            logger.debug(f"Loaded {index_name} index: {len(id_mapping)} entries")

            return True

        except Exception as e:
            logger.warning(f"Failed to load {index_name} index: {e}")
            return False

    def load_all_indices(self) -> Dict[str, bool]:
        """
        Load all FAISS indices from disk.

        Returns:
            Dict with load status for each index
        """
        results = {}
        for index_name in ["insights", "tools", "skills", "agents"]:
            results[index_name] = self._load_index(index_name)

        return results

    def hybrid_search_insights(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> List[Dict]:
        """
        Hybrid search combining FAISS (semantic) and FTS5 (keyword).

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for FAISS (0.0 = FTS5 only, 1.0 = FAISS only)

        Returns:
            List of insight dicts
        """
        # Get FAISS results
        faiss_results = self.search_insights(query, top_k=top_k * 2)

        # Get FTS5 results
        fts5_results = self.state.knowledge.recall(query, top_k=top_k * 2)

        # Combine and score
        combined = {}

        # Add FAISS results with weight alpha
        for i, result in enumerate(faiss_results):
            score = (1.0 - i / len(faiss_results)) * alpha
            combined[result["id"]] = {
                "result": result,
                "score": score,
            }

        # Add FTS5 results with weight (1 - alpha)
        for i, result in enumerate(fts5_results):
            score = (1.0 - i / len(fts5_results)) * (1 - alpha)
            if result["id"] in combined:
                combined[result["id"]]["score"] += score
            else:
                combined[result["id"]] = {
                    "result": result,
                    "score": score,
                }

        # Sort by combined score
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return [r["result"] for r in sorted_results[:top_k]]

    def get_index_stats(self) -> Dict:
        """
        Get statistics about FAISS indices.

        Returns:
            Dict with statistics
        """
        stats = {}

        for index_name in ["insights", "tools", "skills", "agents"]:
            if self.indices[index_name] is not None:
                stats[index_name] = {
                    "total": self.indices[index_name].ntotal,
                    "dimension": self.indices[index_name].d,
                }
            else:
                stats[index_name] = {
                    "total": 0,
                    "dimension": 0,
                }

        return stats
