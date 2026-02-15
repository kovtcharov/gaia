# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
EmbeddingEngine: Local semantic embeddings for vector search.

M6: Memory Defragmentation + Vector Search

The EmbeddingEngine provides:
- Fast local embeddings (all-MiniLM-L6-v2)
- <10ms per embedding on AMD hardware
- Batch embedding support
- Embedding cache

This enables FAISS semantic search over insights, tools, skills, and agents.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning(
        "sentence-transformers not available. Install with: pip install sentence-transformers"
    )


class EmbeddingEngine:
    """
    Local embedding engine for semantic search.

    Uses all-MiniLM-L6-v2 model:
    - 384 dimensions
    - Fast inference (~10ms per text)
    - Good quality for semantic similarity
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: HuggingFace model name
            cache_dir: Optional cache directory
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.cache_dir = cache_dir or Path.home() / ".gaia" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Embedding cache (in-memory)
        self.cache = {}

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.

        Args:
            text: Single text string or list of strings

        Returns:
            numpy array of shape (n, 384) where n is number of texts
        """
        if isinstance(text, str):
            # Single text
            if text in self.cache:
                return self.cache[text]

            embedding = self.model.encode(text, convert_to_numpy=True)
            self.cache[text] = embedding
            return embedding

        else:
            # Batch of texts
            embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, t in enumerate(text):
                if t in self.cache:
                    embeddings.append(self.cache[t])
                else:
                    uncached_texts.append(t)
                    uncached_indices.append(i)

            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(
                    uncached_texts, convert_to_numpy=True
                )

                # Cache new embeddings
                for t, emb in zip(uncached_texts, new_embeddings):
                    self.cache[t] = emb

                # Insert into results
                for idx, emb in zip(uncached_indices, new_embeddings):
                    embeddings.insert(idx, emb)

            return np.array(embeddings)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)

        # Cosine similarity
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def save_cache(self):
        """Save embedding cache to disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        with open(cache_file, "wb") as f:
            pickle.dump(self.cache, f)

        logger.info(f"Saved {len(self.cache)} embeddings to cache")

    def load_cache(self):
        """Load embedding cache from disk."""
        cache_file = self.cache_dir / "embedding_cache.pkl"

        if not cache_file.exists():
            return

        try:
            with open(cache_file, "rb") as f:
                self.cache = pickle.load(f)

            logger.info(f"Loaded {len(self.cache)} embeddings from cache")

        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.cache = {}

    def clear_cache(self):
        """Clear embedding cache."""
        self.cache = {}
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()

        logger.info("Cleared embedding cache")

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed a large batch of texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (n, 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

        # Cache embeddings
        for text, emb in zip(texts, embeddings):
            self.cache[text] = emb

        return embeddings
