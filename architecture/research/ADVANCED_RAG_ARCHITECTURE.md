# ADVANCED RAG ARCHITECTURE

**Version:** 1.0
**Status:** Proposed
**Last Updated:** 2026-02-07

---

## Executive Summary

### Current GAIA RAG Capabilities

**Existing Implementation** (`src/gaia/rag/sdk.py`):
- Basic PDF text extraction (PyPDF2/pypdf)
- Simple fixed-size chunking (500 chars, 100 overlap)
- FAISS vector store with sentence-transformers embeddings
- Top-K retrieval (default: 5 chunks)
- Direct LLM integration via ChatSDK
- Basic caching and index persistence

**Limitations:**
1. **No hybrid search** - Dense vectors only, missing keyword/sparse signals
2. **Naive chunking** - Fixed character boundaries, ignores semantic structure
3. **Single-hop retrieval** - No multi-step reasoning or query decomposition
4. **No re-ranking** - Top-K results returned without refinement
5. **Limited query understanding** - No entity extraction, intent classification
6. **Flat document model** - No hierarchical structure or metadata relationships
7. **No compression** - Full chunks passed to LLM, wasting context window
8. **Missing observability** - Minimal metrics on retrieval quality

### Gap Analysis: Production RAG Requirements

| Capability | Current State | Production Need | Priority |
|------------|---------------|-----------------|----------|
| **Hybrid Search** | Dense only | Dense + BM25 + RRF | CRITICAL |
| **Semantic Chunking** | Fixed 500 chars | LLM-aware boundaries | HIGH |
| **Re-ranking** | None | Cross-encoder refinement | CRITICAL |
| **Query Understanding** | Pass-through | Classification, entities, expansion | MEDIUM |
| **Multi-hop Reasoning** | Single retrieval | Iterative decomposition | HIGH |
| **Context Compression** | Full chunks | LLMLingua/summarization | MEDIUM |
| **Knowledge Graphs** | None | Entity extraction + graph traversal | LOW |
| **Hierarchical Indexing** | Flat chunks | Document → Section → Chunk | MEDIUM |
| **Metadata Filtering** | Basic | Rich filtering (date, type, source) | MEDIUM |
| **Retrieval Metrics** | None | Precision@K, MRR, NDCG | MEDIUM |

### Architecture Principles

1. **Extend, Don't Replace** - Build on existing `RAGSDK` class
2. **AMD NPU Optimization** - Leverage local embeddings and LLM inference
3. **Backward Compatible** - Existing code continues to work
4. **Observable** - Rich metrics for debugging and optimization
5. **Modular** - Mix-and-match retrieval strategies
6. **Production-Ready** - Caching, async, error handling

---

## 1. Hybrid Search Architecture

### 1.1 Dense + Sparse Retrieval

**Problem:** Dense vectors miss exact keyword matches. Queries like "NPU batch size 32" need BM25.

**Solution:** Combine dense (semantic) + sparse (keyword) retrieval with Reciprocal Rank Fusion.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi

@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search."""
    dense_weight: float = 0.7  # Weight for dense retrieval
    sparse_weight: float = 0.3  # Weight for BM25
    dense_top_k: int = 20  # Retrieve more for re-ranking
    sparse_top_k: int = 20
    final_top_k: int = 5
    rrf_k: int = 60  # RRF constant


class HybridRetriever:
    """Hybrid dense + sparse retrieval with RRF fusion."""

    def __init__(self, config: HybridSearchConfig):
        self.config = config
        self.bm25_index = None
        self.corpus_tokens = []

    def build_bm25_index(self, documents: List[str]):
        """Build BM25 index from document chunks."""
        self.corpus_tokens = [doc.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(self.corpus_tokens)

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 keyword search."""
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top-K indices and scores
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices]
        return results

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """Combine rankings using RRF (Reciprocal Rank Fusion)."""
        rrf_scores = {}
        k = self.config.rrf_k

        # Dense results
        for rank, (doc_id, _score) in enumerate(dense_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Sparse results
        for rank, (doc_id, _score) in enumerate(sparse_results, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Sort by RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:self.config.final_top_k]
```

### 1.2 Integration with RAGSDK

```python
class AdvancedRAGSDK(RAGSDK):
    """Extended RAG with hybrid search."""

    def __init__(self, config: RAGConfig, hybrid_config: HybridSearchConfig = None):
        super().__init__(config)
        self.hybrid_config = hybrid_config or HybridSearchConfig()
        self.hybrid_retriever = HybridRetriever(self.hybrid_config)
        self.use_hybrid = False

    def enable_hybrid_search(self):
        """Enable hybrid dense+sparse retrieval."""
        if not self.chunks:
            raise ValueError("No documents indexed. Call index_document() first.")

        # Build BM25 index from existing chunks
        self.hybrid_retriever.build_bm25_index(self.chunks)
        self.use_hybrid = True
        self.log.info(f"Hybrid search enabled for {len(self.chunks)} chunks")

    def _hybrid_retrieve(self, query: str, top_k: int) -> List[int]:
        """Retrieve using hybrid search."""
        # Dense retrieval (existing FAISS)
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            self.hybrid_config.dense_top_k
        )
        dense_results = [(idx, 1.0 - dist) for idx, dist in zip(indices[0], distances[0])]

        # Sparse retrieval (BM25)
        sparse_results = self.hybrid_retriever.sparse_search(
            query, self.hybrid_config.sparse_top_k
        )

        # Fuse with RRF
        fused_results = self.hybrid_retriever.reciprocal_rank_fusion(
            dense_results, sparse_results
        )

        return [doc_id for doc_id, _score in fused_results]
```

**Performance Impact:**
- Dense: ~50ms (NPU-accelerated embeddings)
- BM25: ~5ms (CPU, no ML)
- RRF: ~1ms
- **Total: 56ms vs 50ms (12% overhead for 20-30% better recall)**

---

## 2. Advanced Chunking Strategies

### 2.1 Semantic Chunking with LLM

**Problem:** Fixed 500-char chunks split mid-sentence, lose context.

**Solution:** Use local LLM to identify semantic boundaries (paragraphs, topics).

```python
class SemanticChunker:
    """LLM-based semantic chunking."""

    def __init__(self, llm_client, target_chunk_size: int = 500):
        self.llm_client = llm_client
        self.target_chunk_size = target_chunk_size

    def chunk_document(self, text: str) -> List[str]:
        """Split text at semantic boundaries."""
        # Step 1: Split into candidate chunks (paragraphs)
        paragraphs = text.split('\n\n')

        # Step 2: Merge small paragraphs, split large ones
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            if current_size + para_size < self.target_chunk_size * 1.5:
                current_chunk.append(para)
                current_size += para_size
            else:
                # Flush current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))

                # Large paragraph - needs splitting
                if para_size > self.target_chunk_size * 2:
                    sub_chunks = self._split_large_paragraph(para)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_size = 0
                else:
                    current_chunk = [para]
                    current_size = para_size

        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split large paragraph at sentence boundaries."""
        # Use LLM to identify split points
        prompt = f"""Split this paragraph into 2-3 coherent chunks.
Return JSON with split indices.

Paragraph:
{paragraph}

Format: {{"splits": [index1, index2, ...]}}"""

        try:
            response = self.llm_client.generate(
                prompt, max_tokens=100, temperature=0.0
            )
            splits = json.loads(response)['splits']

            chunks = []
            prev = 0
            for split_idx in splits:
                chunks.append(paragraph[prev:split_idx].strip())
                prev = split_idx
            chunks.append(paragraph[prev:].strip())

            return chunks
        except Exception as e:
            # Fallback to sentence splitting
            sentences = paragraph.split('. ')
            mid = len(sentences) // 2
            return [
                '. '.join(sentences[:mid]) + '.',
                '. '.join(sentences[mid:])
            ]
```

### 2.2 Hierarchical Chunking

**Problem:** Lost document structure (sections, subsections).

**Solution:** Maintain hierarchy for better context.

```python
@dataclass
class HierarchicalChunk:
    """Chunk with structural metadata."""
    text: str
    level: int  # 0=document, 1=section, 2=subsection, 3=paragraph
    parent_id: Optional[str]
    chunk_id: str
    metadata: Dict[str, Any]


class HierarchicalChunker:
    """Maintain document structure during chunking."""

    def chunk_with_hierarchy(self, document: Dict) -> List[HierarchicalChunk]:
        """
        Extract hierarchical chunks from structured document.

        Args:
            document: {
                'title': str,
                'sections': [
                    {'heading': str, 'content': str, 'subsections': [...]},
                    ...
                ]
            }
        """
        chunks = []
        doc_id = hashlib.md5(document['title'].encode()).hexdigest()[:8]

        # Level 0: Document summary
        doc_summary = f"{document['title']}\n\nSections: " + \
                     ', '.join(s['heading'] for s in document['sections'])
        chunks.append(HierarchicalChunk(
            text=doc_summary,
            level=0,
            parent_id=None,
            chunk_id=f"{doc_id}_doc",
            metadata={'type': 'document', 'title': document['title']}
        ))

        # Level 1-2: Sections and subsections
        for sect_idx, section in enumerate(document['sections']):
            sect_id = f"{doc_id}_s{sect_idx}"

            # Section chunk
            sect_text = f"# {section['heading']}\n\n{section['content']}"
            chunks.append(HierarchicalChunk(
                text=sect_text,
                level=1,
                parent_id=f"{doc_id}_doc",
                chunk_id=sect_id,
                metadata={'type': 'section', 'heading': section['heading']}
            ))

            # Subsections
            for sub_idx, subsection in enumerate(section.get('subsections', [])):
                sub_id = f"{sect_id}_ss{sub_idx}"
                sub_text = f"## {subsection['heading']}\n\n{subsection['content']}"
                chunks.append(HierarchicalChunk(
                    text=sub_text,
                    level=2,
                    parent_id=sect_id,
                    chunk_id=sub_id,
                    metadata={'type': 'subsection', 'heading': subsection['heading']}
                ))

        return chunks

    def retrieve_with_context(
        self,
        chunk_ids: List[str],
        chunks_map: Dict[str, HierarchicalChunk]
    ) -> List[str]:
        """Retrieve chunks with parent context."""
        enriched = []

        for chunk_id in chunk_ids:
            chunk = chunks_map[chunk_id]

            # Build context from parents
            context_parts = [chunk.text]
            parent_id = chunk.parent_id

            while parent_id:
                parent = chunks_map.get(parent_id)
                if parent:
                    context_parts.insert(0, f"[Parent Context: {parent.metadata.get('heading', 'Document')}]")
                    parent_id = parent.parent_id
                else:
                    break

            enriched.append('\n'.join(context_parts))

        return enriched
```

### 2.3 Metadata-Aware Chunking

**Problem:** Lose document metadata (source, date, author).

**Solution:** Propagate metadata to chunks for filtering.

```python
@dataclass
class ChunkMetadata:
    """Rich metadata for chunks."""
    source_file: str
    source_type: str  # 'pdf', 'markdown', 'code'
    page_number: Optional[int]
    section_title: Optional[str]
    timestamp: datetime
    tags: List[str]
    custom: Dict[str, Any]


class MetadataChunker:
    """Chunking with metadata propagation."""

    def chunk_with_metadata(
        self,
        text: str,
        metadata: ChunkMetadata
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Chunk and attach metadata."""
        base_chunks = self._basic_chunk(text)

        chunks_with_meta = []
        for idx, chunk in enumerate(base_chunks):
            chunk_meta = ChunkMetadata(
                source_file=metadata.source_file,
                source_type=metadata.source_type,
                page_number=metadata.page_number,
                section_title=metadata.section_title,
                timestamp=metadata.timestamp,
                tags=metadata.tags,
                custom={
                    **metadata.custom,
                    'chunk_index': idx,
                    'total_chunks': len(base_chunks)
                }
            )
            chunks_with_meta.append((chunk, chunk_meta))

        return chunks_with_meta

    def filter_by_metadata(
        self,
        query_filters: Dict[str, Any],
        chunks: List[Tuple[str, ChunkMetadata]]
    ) -> List[Tuple[str, ChunkMetadata]]:
        """Filter chunks by metadata criteria."""
        filtered = []

        for chunk, meta in chunks:
            match = True

            if 'source_type' in query_filters:
                if meta.source_type != query_filters['source_type']:
                    match = False

            if 'tags' in query_filters:
                required_tags = set(query_filters['tags'])
                if not required_tags.issubset(set(meta.tags)):
                    match = False

            if 'date_after' in query_filters:
                if meta.timestamp < query_filters['date_after']:
                    match = False

            if match:
                filtered.append((chunk, meta))

        return filtered
```

---

## 3. Query Understanding Pipeline

### 3.1 Intent Classification

**Problem:** Treat "Define X" and "Compare X vs Y" the same.

**Solution:** Classify query intent to route to specialized retrieval.

```python
from enum import Enum

class QueryIntent(Enum):
    FACTUAL = "factual"  # "What is X?"
    COMPARISON = "comparison"  # "X vs Y"
    PROCEDURAL = "procedural"  # "How to X?"
    DIAGNOSTIC = "diagnostic"  # "Why doesn't X work?"
    EXPLORATORY = "exploratory"  # "Tell me about X"


class QueryClassifier:
    """Classify query intent for specialized retrieval."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def classify(self, query: str) -> QueryIntent:
        """Classify query intent."""
        # Fast heuristic classification
        query_lower = query.lower()

        if any(q in query_lower for q in ['what is', 'define', 'meaning of']):
            return QueryIntent.FACTUAL

        if ' vs ' in query_lower or ' versus ' in query_lower or 'compare' in query_lower:
            return QueryIntent.COMPARISON

        if any(q in query_lower for q in ['how to', 'steps to', 'tutorial', 'guide']):
            return QueryIntent.PROCEDURAL

        if any(q in query_lower for q in ['why', 'error', 'not working', 'issue']):
            return QueryIntent.DIAGNOSTIC

        return QueryIntent.EXPLORATORY

    def get_retrieval_strategy(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get retrieval parameters based on intent."""
        strategies = {
            QueryIntent.FACTUAL: {
                'top_k': 3,
                'prefer_definitions': True,
                'boost_glossary': True
            },
            QueryIntent.COMPARISON: {
                'top_k': 8,
                'retrieve_both_terms': True,
                'prefer_tables': True
            },
            QueryIntent.PROCEDURAL: {
                'top_k': 5,
                'prefer_sequential': True,
                'boost_numbered_lists': True
            },
            QueryIntent.DIAGNOSTIC: {
                'top_k': 10,
                'prefer_error_logs': True,
                'boost_troubleshooting': True
            },
            QueryIntent.EXPLORATORY: {
                'top_k': 5,
                'diverse_results': True
            }
        }
        return strategies.get(intent, strategies[QueryIntent.EXPLORATORY])
```

### 3.2 Entity Extraction

**Problem:** Miss key entities in query ("AMD Ryzen AI 9 HX 370").

**Solution:** Extract and expand entities for better retrieval.

```python
import spacy

class EntityExtractor:
    """Extract named entities from queries."""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.custom_entities = {
            'AMD': 'ORG',
            'NPU': 'TECH',
            'Ryzen': 'PRODUCT',
            'GAIA': 'PRODUCT'
        }

    def extract(self, query: str) -> Dict[str, List[str]]:
        """Extract entities by type."""
        doc = self.nlp(query)
        entities = {}

        # spaCy entities
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)

        # Custom entities
        for entity, label in self.custom_entities.items():
            if entity.lower() in query.lower():
                entities.setdefault(label, []).append(entity)

        return entities

    def expand_query_with_entities(self, query: str, entities: Dict) -> str:
        """Add entity context to query."""
        expansions = []

        if 'PRODUCT' in entities:
            expansions.append(f"Product: {', '.join(entities['PRODUCT'])}")

        if 'TECH' in entities:
            expansions.append(f"Technology: {', '.join(entities['TECH'])}")

        if expansions:
            return f"{query}\n\nEntities: {'; '.join(expansions)}"

        return query
```

### 3.3 Query Expansion

**Problem:** Query "NPU optimization" misses "neural processing unit performance tuning".

**Solution:** Generate synonym/related queries.

```python
class QueryExpander:
    """Expand queries with synonyms and related terms."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        # Pre-built synonym map for common terms
        self.synonyms = {
            'npu': ['neural processing unit', 'ai accelerator', 'xdna'],
            'optimization': ['tuning', 'performance', 'speedup'],
            'inference': ['prediction', 'generation'],
        }

    def expand(self, query: str, max_expansions: int = 2) -> List[str]:
        """Generate expanded queries."""
        queries = [query]  # Original

        # Synonym-based expansion
        query_lower = query.lower()
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                for syn in synonyms[:max_expansions]:
                    expanded = query_lower.replace(term, syn)
                    queries.append(expanded)

        # LLM-based expansion (slower, better)
        if len(queries) < 3:
            llm_expansions = self._llm_expand(query, max_expansions)
            queries.extend(llm_expansions)

        return queries[:max_expansions + 1]

    def _llm_expand(self, query: str, count: int) -> List[str]:
        """Use LLM to generate semantically similar queries."""
        prompt = f"""Generate {count} alternative phrasings of this query:
"{query}"

Requirements:
- Same information need
- Different wording
- Natural language

Return JSON: {{"queries": ["...", "..."]}}"""

        try:
            response = self.llm_client.generate(
                prompt, max_tokens=150, temperature=0.3
            )
            result = json.loads(response)
            return result.get('queries', [])
        except Exception:
            return []
```

---

## 4. Re-ranking and Compression

### 4.1 Cross-Encoder Re-ranking

**Problem:** Top-K from embeddings has false positives (semantic drift).

**Solution:** Re-rank with cross-encoder (query + doc jointly scored).

```python
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """Re-rank retrieved chunks with cross-encoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Re-rank chunks and return top-K."""
        # Create query-chunk pairs
        pairs = [[query, chunk] for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]


class AdvancedRAGSDKWithReranking(AdvancedRAGSDK):
    """RAG with cross-encoder re-ranking."""

    def __init__(self, *args, use_reranking: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_reranking = use_reranking
        if use_reranking:
            self.reranker = CrossEncoderReranker()

    def query(self, question: str, max_chunks: int = 5) -> RAGResponse:
        """Query with optional re-ranking."""
        # Step 1: Initial retrieval (over-fetch for re-ranking)
        initial_k = max_chunks * 4 if self.use_reranking else max_chunks

        if self.use_hybrid:
            chunk_indices = self._hybrid_retrieve(question, initial_k)
        else:
            chunk_indices = self._dense_retrieve(question, initial_k)

        initial_chunks = [self.chunks[i] for i in chunk_indices]

        # Step 2: Re-rank
        if self.use_reranking:
            reranked = self.reranker.rerank(question, initial_chunks, max_chunks)
            final_chunks = [chunk for chunk, _score in reranked]
            chunk_scores = [score for _chunk, score in reranked]
        else:
            final_chunks = initial_chunks[:max_chunks]
            chunk_scores = [1.0] * len(final_chunks)

        # Step 3: Generate answer
        return self._generate_answer(question, final_chunks, chunk_scores)
```

**Performance:**
- Cross-encoder: ~100ms for 20 candidates (CPU)
- **Impact: 15-25% improvement in relevance at 100ms cost**

### 4.2 Context Compression

**Problem:** Waste context window on irrelevant sentences within chunks.

**Solution:** Compress chunks to essential information.

```python
class ContextCompressor:
    """Compress retrieved context to essential information."""

    def __init__(self, llm_client, compression_ratio: float = 0.5):
        self.llm_client = llm_client
        self.compression_ratio = compression_ratio

    def compress_chunks(self, query: str, chunks: List[str]) -> List[str]:
        """Compress chunks while preserving query-relevant information."""
        compressed = []

        for chunk in chunks:
            # Skip compression for short chunks
            if len(chunk) < 200:
                compressed.append(chunk)
                continue

            # Compress long chunks
            compressed_chunk = self._compress_single(query, chunk)
            compressed.append(compressed_chunk)

        return compressed

    def _compress_single(self, query: str, chunk: str) -> str:
        """Compress a single chunk."""
        target_length = int(len(chunk) * self.compression_ratio)

        prompt = f"""Compress this text to ~{target_length} characters while preserving information relevant to: "{query}"

Text:
{chunk}

Compressed:"""

        compressed = self.llm_client.generate(
            prompt,
            max_tokens=target_length // 4,
            temperature=0.0
        )

        return compressed.strip()

    def extractive_compression(self, query: str, chunk: str) -> str:
        """Fast extractive compression (sentence selection)."""
        sentences = chunk.split('. ')

        # Score sentences by query relevance
        query_terms = set(query.lower().split())
        scored_sentences = []

        for sent in sentences:
            sent_terms = set(sent.lower().split())
            overlap = len(query_terms & sent_terms)
            scored_sentences.append((sent, overlap))

        # Keep top 50% of sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        keep_count = max(1, len(scored_sentences) // 2)

        compressed = '. '.join(s for s, _score in scored_sentences[:keep_count])
        return compressed + '.'
```

---

## 5. Knowledge Graph Integration

### 5.1 Entity Extraction and Linking

**Problem:** Documents have implicit relationships (Product X uses Technology Y).

**Solution:** Extract entities and relationships, build knowledge graph.

```python
from dataclasses import dataclass
from typing import Set

@dataclass
class Entity:
    """Knowledge graph entity."""
    id: str
    type: str  # PRODUCT, TECHNOLOGY, FEATURE, etc.
    name: str
    aliases: Set[str]
    properties: Dict[str, Any]


@dataclass
class Relationship:
    """Knowledge graph relationship."""
    source_id: str
    target_id: str
    relation_type: str  # USES, REQUIRES, IMPLEMENTS, etc.
    properties: Dict[str, Any]


class KnowledgeGraphBuilder:
    """Extract entities and relationships from documents."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []

    def extract_from_chunk(self, chunk: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract structured knowledge from text."""
        prompt = f"""Extract entities and relationships from this text.

Text:
{chunk}

Return JSON:
{{
  "entities": [
    {{"name": "...", "type": "PRODUCT|TECHNOLOGY|FEATURE|CONCEPT", "properties": {{}}}}
  ],
  "relationships": [
    {{"source": "...", "target": "...", "type": "USES|REQUIRES|IMPLEMENTS|ENABLES"}}
  ]
}}"""

        try:
            response = self.llm_client.generate(
                prompt, max_tokens=500, temperature=0.0
            )
            data = json.loads(response)

            entities = [
                Entity(
                    id=self._make_id(e['name']),
                    type=e['type'],
                    name=e['name'],
                    aliases=set(),
                    properties=e.get('properties', {})
                )
                for e in data.get('entities', [])
            ]

            relationships = [
                Relationship(
                    source_id=self._make_id(r['source']),
                    target_id=self._make_id(r['target']),
                    relation_type=r['type'],
                    properties={}
                )
                for r in data.get('relationships', [])
            ]

            return entities, relationships

        except Exception as e:
            self.log.warning(f"Failed to extract knowledge: {e}")
            return [], []

    def _make_id(self, name: str) -> str:
        """Generate entity ID from name."""
        return name.lower().replace(' ', '_')

    def build_graph(self, chunks: List[str]):
        """Build knowledge graph from all chunks."""
        for chunk in chunks:
            entities, relationships = self.extract_from_chunk(chunk)

            # Add entities
            for entity in entities:
                if entity.id not in self.entities:
                    self.entities[entity.id] = entity
                else:
                    # Merge properties
                    self.entities[entity.id].properties.update(entity.properties)

            # Add relationships
            self.relationships.extend(relationships)
```

### 5.2 Graph-Based Retrieval

**Problem:** Query about "AMD NPU optimization" misses chunks about "Ryzen AI" features.

**Solution:** Traverse graph to find related entities.

```python
import networkx as nx

class GraphRetriever:
    """Retrieve using knowledge graph traversal."""

    def __init__(self, kg_builder: KnowledgeGraphBuilder):
        self.kg = kg_builder
        self.graph = nx.DiGraph()
        self._build_networkx()

    def _build_networkx(self):
        """Convert to NetworkX graph."""
        for entity_id, entity in self.kg.entities.items():
            self.graph.add_node(entity_id, **entity.properties)

        for rel in self.kg.relationships:
            self.graph.add_edge(
                rel.source_id,
                rel.target_id,
                relation=rel.relation_type,
                **rel.properties
            )

    def find_related_entities(
        self,
        entity_id: str,
        max_hops: int = 2
    ) -> List[str]:
        """Find entities within N hops."""
        if entity_id not in self.graph:
            return []

        related = set()
        visited = set()
        queue = [(entity_id, 0)]

        while queue:
            current, hops = queue.pop(0)

            if current in visited or hops > max_hops:
                continue

            visited.add(current)
            related.add(current)

            # Add neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, hops + 1))

        return list(related)

    def graph_augmented_retrieval(
        self,
        query: str,
        base_chunks: List[int],
        chunks_map: Dict[int, str],
        max_additional: int = 3
    ) -> List[int]:
        """Augment retrieval with graph-related chunks."""
        # Extract entities from query
        extractor = EntityExtractor()
        query_entities = extractor.extract(query)

        # Find related entities
        related_entity_ids = set()
        for entity_type, entities in query_entities.items():
            for entity_name in entities:
                entity_id = self.kg._make_id(entity_name)
                related = self.find_related_entities(entity_id, max_hops=2)
                related_entity_ids.update(related)

        # Find chunks mentioning related entities
        additional_chunks = []
        for chunk_id, chunk_text in chunks_map.items():
            if chunk_id in base_chunks:
                continue

            # Check if chunk mentions related entities
            chunk_lower = chunk_text.lower()
            for entity_id in related_entity_ids:
                entity = self.kg.entities.get(entity_id)
                if entity and entity.name.lower() in chunk_lower:
                    additional_chunks.append(chunk_id)
                    break

        return base_chunks + additional_chunks[:max_additional]
```

---

## 6. Multi-Hop Reasoning

### 6.1 Query Decomposition

**Problem:** Complex queries need multiple retrieval steps ("Compare AMD NPU vs NVIDIA GPU for LLM inference").

**Solution:** Decompose into sub-queries, retrieve separately, synthesize.

```python
@dataclass
class SubQuery:
    """Decomposed sub-query."""
    query: str
    dependencies: List[int]  # Indices of prerequisite sub-queries
    retrieval_strategy: str


class QueryDecomposer:
    """Decompose complex queries into retrievable sub-queries."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def decompose(self, complex_query: str) -> List[SubQuery]:
        """Break down query into sub-queries."""
        prompt = f"""Decompose this complex query into 2-4 simpler sub-queries that can be answered independently.

Query: {complex_query}

Return JSON:
{{
  "sub_queries": [
    {{"query": "...", "dependencies": [], "strategy": "factual|comparison|procedural"}},
    ...
  ],
  "reasoning": "..."
}}"""

        try:
            response = self.llm_client.generate(
                prompt, max_tokens=300, temperature=0.0
            )
            data = json.loads(response)

            sub_queries = [
                SubQuery(
                    query=sq['query'],
                    dependencies=sq.get('dependencies', []),
                    retrieval_strategy=sq.get('strategy', 'factual')
                )
                for sq in data.get('sub_queries', [])
            ]

            return sub_queries

        except Exception as e:
            # Fallback: treat as single query
            return [SubQuery(
                query=complex_query,
                dependencies=[],
                retrieval_strategy='factual'
            )]


class MultiHopRetriever:
    """Multi-hop retrieval with query decomposition."""

    def __init__(self, rag_sdk: AdvancedRAGSDK, llm_client):
        self.rag = rag_sdk
        self.llm_client = llm_client
        self.decomposer = QueryDecomposer(llm_client)

    def retrieve_multi_hop(
        self,
        query: str,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Iterative multi-hop retrieval."""
        # Step 1: Decompose query
        sub_queries = self.decomposer.decompose(query)

        # Step 2: Retrieve for each sub-query
        sub_results = {}
        for idx, sub_q in enumerate(sub_queries):
            # Wait for dependencies
            dep_contexts = [
                sub_results[dep]['answer']
                for dep in sub_q.dependencies
                if dep in sub_results
            ]

            # Enhance sub-query with dependency context
            enhanced_query = sub_q.query
            if dep_contexts:
                enhanced_query = f"{sub_q.query}\n\nContext from previous steps:\n" + \
                               '\n'.join(dep_contexts)

            # Retrieve
            response = self.rag.query(enhanced_query, max_chunks=5)
            sub_results[idx] = {
                'query': sub_q.query,
                'chunks': response.chunks,
                'answer': response.text
            }

        # Step 3: Synthesize final answer
        final_answer = self._synthesize_answers(query, sub_results)

        return {
            'query': query,
            'sub_queries': [sq.query for sq in sub_queries],
            'sub_results': sub_results,
            'final_answer': final_answer
        }

    def _synthesize_answers(
        self,
        original_query: str,
        sub_results: Dict[int, Dict]
    ) -> str:
        """Synthesize sub-answers into final response."""
        sub_answers = [
            f"Q: {result['query']}\nA: {result['answer']}"
            for result in sub_results.values()
        ]

        synthesis_prompt = f"""Original question: {original_query}

Sub-question answers:
{chr(10).join(sub_answers)}

Synthesize a comprehensive answer to the original question using the sub-answers above."""

        final_answer = self.llm_client.generate(
            synthesis_prompt,
            max_tokens=500,
            temperature=0.3
        )

        return final_answer
```

### 6.2 Iterative Refinement

**Problem:** First retrieval misses information, need follow-up.

**Solution:** Iteratively refine query based on gaps.

```python
class IterativeRefiner:
    """Iteratively refine retrieval based on answer quality."""

    def __init__(self, rag_sdk: AdvancedRAGSDK, llm_client):
        self.rag = rag_sdk
        self.llm_client = llm_client

    def retrieve_with_refinement(
        self,
        query: str,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Iteratively retrieve and refine."""
        iteration = 0
        accumulated_chunks = []
        refinement_history = []

        current_query = query

        while iteration < max_iterations:
            # Retrieve
            response = self.rag.query(current_query, max_chunks=5)
            accumulated_chunks.extend(response.chunks)

            # Assess answer quality
            confidence = self._assess_confidence(query, response.text)
            refinement_history.append({
                'iteration': iteration,
                'query': current_query,
                'answer': response.text,
                'confidence': confidence
            })

            if confidence >= confidence_threshold:
                break

            # Generate refinement query
            current_query = self._generate_refinement_query(
                query, response.text, accumulated_chunks
            )
            iteration += 1

        # Final synthesis
        final_answer = self._synthesize_iterative_results(
            query, refinement_history
        )

        return {
            'query': query,
            'iterations': iteration + 1,
            'final_answer': final_answer,
            'history': refinement_history
        }

    def _assess_confidence(self, query: str, answer: str) -> float:
        """Assess answer confidence (0-1)."""
        # Heuristic: check for uncertainty phrases
        uncertainty_phrases = [
            'not sure', 'unclear', 'insufficient information',
            'cannot determine', 'no mention', 'not found'
        ]

        answer_lower = answer.lower()
        for phrase in uncertainty_phrases:
            if phrase in answer_lower:
                return 0.4

        # Check answer length (too short = low confidence)
        if len(answer.split()) < 20:
            return 0.5

        # Default: medium-high confidence
        return 0.75

    def _generate_refinement_query(
        self,
        original_query: str,
        partial_answer: str,
        retrieved_chunks: List[str]
    ) -> str:
        """Generate refined query to fill gaps."""
        prompt = f"""The user asked: "{original_query}"

Current partial answer:
{partial_answer}

What additional information is needed? Generate a refined query to retrieve missing details.

Refined query:"""

        refined = self.llm_client.generate(
            prompt, max_tokens=100, temperature=0.3
        )

        return refined.strip()

    def _synthesize_iterative_results(
        self,
        query: str,
        history: List[Dict]
    ) -> str:
        """Synthesize answers from iterative refinement."""
        answers = [h['answer'] for h in history]

        synthesis_prompt = f"""Original question: {query}

Answers from iterative retrieval:
{chr(10).join(f"{i+1}. {ans}" for i, ans in enumerate(answers))}

Synthesize a comprehensive final answer:"""

        final = self.llm_client.generate(
            synthesis_prompt, max_tokens=500, temperature=0.3
        )

        return final
```

---

## 7. Production Optimizations

### 7.1 Async and Caching

**Problem:** Synchronous retrieval blocks, no result caching.

**Solution:** Async retrieval, LRU cache for queries.

```python
import asyncio
from functools import lru_cache
import hashlib

class AsyncAdvancedRAG(AdvancedRAGSDK):
    """Async RAG with caching."""

    def __init__(self, *args, enable_cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_cache = enable_cache
        self.query_cache = {}
        self.max_cache_size = 1000

    @lru_cache(maxsize=1000)
    def _cached_embedding(self, text: str) -> np.ndarray:
        """Cache embeddings for repeated queries."""
        return self.embedding_model.encode([text])[0]

    async def query_async(
        self,
        question: str,
        max_chunks: int = 5
    ) -> RAGResponse:
        """Async query with caching."""
        # Check cache
        cache_key = self._make_cache_key(question, max_chunks)

        if self.enable_cache and cache_key in self.query_cache:
            self.log.debug("Cache hit for query")
            return self.query_cache[cache_key]

        # Parallel retrieval steps
        tasks = [
            self._dense_retrieve_async(question, max_chunks),
            self._sparse_search_async(question, max_chunks) if self.use_hybrid else asyncio.sleep(0)
        ]

        dense_results, sparse_results = await asyncio.gather(*tasks)

        # Fuse and re-rank
        if self.use_hybrid:
            fused = self.hybrid_retriever.reciprocal_rank_fusion(
                dense_results, sparse_results
            )
            chunk_indices = [idx for idx, _score in fused]
        else:
            chunk_indices = dense_results

        # Generate answer (async LLM call)
        response = await self._generate_answer_async(
            question, chunk_indices
        )

        # Cache result
        if self.enable_cache:
            self._update_cache(cache_key, response)

        return response

    async def _dense_retrieve_async(
        self,
        query: str,
        top_k: int
    ) -> List[int]:
        """Async dense retrieval."""
        # Embedding can run in thread pool
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self._cached_embedding, query
        )

        # FAISS search
        distances, indices = self.index.search(
            np.array([embedding], dtype=np.float32), top_k
        )

        return indices[0].tolist()

    async def _sparse_search_async(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Async BM25 search."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.hybrid_retriever.sparse_search,
            query,
            top_k
        )

    async def _generate_answer_async(
        self,
        question: str,
        chunk_indices: List[int]
    ) -> RAGResponse:
        """Async answer generation."""
        chunks = [self.chunks[i] for i in chunk_indices]

        # Build prompt
        context = '\n\n'.join(chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        # Async LLM call
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            self.llm_client.generate,
            prompt
        )

        return RAGResponse(
            text=answer,
            chunks=chunks,
            source_files=self._get_source_files(chunk_indices),
            chunk_scores=[1.0] * len(chunks)
        )

    def _make_cache_key(self, query: str, max_chunks: int) -> str:
        """Generate cache key."""
        content = f"{query}_{max_chunks}_{self.use_hybrid}_{self.use_reranking}"
        return hashlib.md5(content.encode()).hexdigest()

    def _update_cache(self, key: str, value: RAGResponse):
        """Update cache with LRU eviction."""
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[key] = value
```

### 7.2 Index Optimization

**Problem:** FAISS FlatL2 is slow for large document sets.

**Solution:** Use IVF (Inverted File) index for sub-linear search.

```python
class OptimizedIndexManager:
    """FAISS index optimization for large-scale retrieval."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = None
        self.use_gpu = False

    def build_optimized_index(
        self,
        embeddings: np.ndarray,
        index_type: str = "IVF",
        nlist: int = 100
    ):
        """Build optimized FAISS index."""
        n_vectors = embeddings.shape[0]

        if index_type == "Flat":
            # Exact search (small datasets)
            self.index = faiss.IndexFlatL2(self.dimension)

        elif index_type == "IVF":
            # Inverted file index (medium datasets: 10K-1M)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist
            )
            # Train clustering
            self.index.train(embeddings)

        elif index_type == "IVFPQ":
            # IVF with Product Quantization (large datasets: 1M+)
            quantizer = faiss.IndexFlatL2(self.dimension)
            m = 8  # Number of subquantizers
            bits = 8  # Bits per subquantizer
            self.index = faiss.IndexIVFPQ(
                quantizer, self.dimension, nlist, m, bits
            )
            self.index.train(embeddings)

        elif index_type == "HNSW":
            # Hierarchical NSW (best quality/speed tradeoff)
            M = 32  # Number of connections
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16

        # Add vectors
        self.index.add(embeddings)

        self.log.info(
            f"Built {index_type} index with {n_vectors} vectors"
        )

    def optimize_search_params(self, recall_target: float = 0.95):
        """Tune search parameters for recall/speed tradeoff."""
        if isinstance(self.index, faiss.IndexIVFFlat):
            # More probes = better recall, slower search
            nprobe_values = [1, 5, 10, 20, 50, 100]

            # Binary search for nprobe that meets recall target
            # (requires ground truth, simplified here)
            self.index.nprobe = 10  # Default

        elif isinstance(self.index, faiss.IndexHNSWFlat):
            # Tune efSearch
            self.index.hnsw.efSearch = 32 if recall_target > 0.95 else 16

    def enable_gpu_acceleration(self):
        """Move index to GPU for faster search."""
        try:
            import faiss.contrib.torch_utils

            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                self.use_gpu = True
                self.log.info("FAISS index moved to GPU")
        except Exception as e:
            self.log.warning(f"GPU acceleration failed: {e}")
```

### 7.3 Batch Processing

**Problem:** Indexing PDFs one-by-one is slow.

**Solution:** Batch embedding generation and parallel processing.

```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchIndexer:
    """Batch processing for efficient indexing."""

    def __init__(self, rag_sdk: AdvancedRAGSDK, n_workers: int = 4):
        self.rag = rag_sdk
        self.n_workers = n_workers

    def batch_index_documents(
        self,
        document_paths: List[str],
        batch_size: int = 32
    ):
        """Index multiple documents in parallel."""
        # Step 1: Extract text (parallel)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._extract_text, path): path
                for path in document_paths
            }

            all_texts = {}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    text = future.result()
                    all_texts[path] = text
                except Exception as e:
                    self.rag.log.error(f"Failed to extract {path}: {e}")

        # Step 2: Chunk documents
        all_chunks = []
        chunk_metadata = []

        for path, text in all_texts.items():
            chunks = self.rag._chunk_text(text)
            all_chunks.extend(chunks)
            chunk_metadata.extend([{'source': path}] * len(chunks))

        # Step 3: Batch embedding generation
        embeddings = self._batch_embed(all_chunks, batch_size)

        # Step 4: Add to index
        self.rag.chunks.extend(all_chunks)
        self.rag.chunk_metadata.extend(chunk_metadata)
        self.rag.index.add(embeddings)

        self.rag.log.info(
            f"Indexed {len(document_paths)} documents "
            f"({len(all_chunks)} chunks) in batches"
        )

    def _extract_text(self, path: str) -> str:
        """Extract text from single document."""
        # Use existing extraction logic
        return self.rag._extract_text_from_pdf(path)

    def _batch_embed(
        self,
        texts: List[str],
        batch_size: int
    ) -> np.ndarray:
        """Generate embeddings in batches."""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.rag.embedding_model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=True
            )
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)
```

---

## 8. Integration with Multi-Document Synthesis

### 8.1 Cross-Document Retrieval

**Problem:** Query spans multiple documents (compare specifications).

**Solution:** Retrieve from multiple documents, synthesize cross-document answer.

```python
class CrossDocumentRetriever:
    """Retrieve and synthesize across multiple documents."""

    def __init__(self, rag_sdk: AdvancedRAGSDK):
        self.rag = rag_sdk

    def retrieve_cross_document(
        self,
        query: str,
        document_filters: Optional[List[str]] = None,
        chunks_per_doc: int = 3
    ) -> Dict[str, List[str]]:
        """Retrieve from multiple documents."""
        # Retrieve top chunks
        response = self.rag.query(query, max_chunks=chunks_per_doc * 10)

        # Group by source document
        doc_chunks = {}
        for chunk, metadata in zip(response.chunks, response.chunk_metadata):
            source = metadata.get('source_file', 'unknown')

            # Apply filters
            if document_filters and source not in document_filters:
                continue

            if source not in doc_chunks:
                doc_chunks[source] = []

            doc_chunks[source].append(chunk)

        # Limit chunks per document
        for source in doc_chunks:
            doc_chunks[source] = doc_chunks[source][:chunks_per_doc]

        return doc_chunks

    def synthesize_cross_document(
        self,
        query: str,
        doc_chunks: Dict[str, List[str]]
    ) -> str:
        """Synthesize answer from multiple documents."""
        # Build structured context
        context_parts = []
        for source, chunks in doc_chunks.items():
            doc_name = Path(source).stem
            context_parts.append(
                f"=== {doc_name} ===\n" + '\n\n'.join(chunks)
            )

        context = '\n\n'.join(context_parts)

        prompt = f"""Answer this question using information from multiple documents.
Cite which document each fact comes from.

Question: {query}

Documents:
{context}

Answer:"""

        answer = self.rag.llm_client.generate(
            prompt, max_tokens=1000, temperature=0.3
        )

        return answer
```

### 8.2 Document Comparison Agent

**Problem:** "Compare document A vs document B" needs structured comparison.

**Solution:** Specialized comparison retrieval and synthesis.

```python
class DocumentComparator:
    """Compare content across documents."""

    def __init__(self, rag_sdk: AdvancedRAGSDK):
        self.rag = rag_sdk

    def compare_documents(
        self,
        doc_a: str,
        doc_b: str,
        comparison_aspects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Structured comparison of two documents."""
        if not comparison_aspects:
            comparison_aspects = ['features', 'performance', 'requirements']

        comparisons = {}

        for aspect in comparison_aspects:
            query = f"What are the {aspect} mentioned?"

            # Retrieve from doc A
            chunks_a = self._retrieve_from_document(query, doc_a, max_chunks=3)

            # Retrieve from doc B
            chunks_b = self._retrieve_from_document(query, doc_b, max_chunks=3)

            # Generate comparison
            comparison = self._generate_comparison(
                aspect, chunks_a, chunks_b
            )

            comparisons[aspect] = comparison

        # Synthesize overall comparison
        overall = self._synthesize_overall_comparison(comparisons)

        return {
            'doc_a': doc_a,
            'doc_b': doc_b,
            'aspects': comparisons,
            'overall': overall
        }

    def _retrieve_from_document(
        self,
        query: str,
        document: str,
        max_chunks: int
    ) -> List[str]:
        """Retrieve chunks from specific document."""
        # Filter to specific document
        response = self.rag.query(query, max_chunks=max_chunks * 3)

        filtered_chunks = [
            chunk for chunk, meta in zip(response.chunks, response.chunk_metadata)
            if meta.get('source_file') == document
        ]

        return filtered_chunks[:max_chunks]

    def _generate_comparison(
        self,
        aspect: str,
        chunks_a: List[str],
        chunks_b: List[str]
    ) -> str:
        """Generate comparison for one aspect."""
        prompt = f"""Compare {aspect} between two documents.

Document A:
{chr(10).join(chunks_a)}

Document B:
{chr(10).join(chunks_b)}

Comparison:"""

        comparison = self.rag.llm_client.generate(
            prompt, max_tokens=300, temperature=0.2
        )

        return comparison

    def _synthesize_overall_comparison(
        self,
        aspect_comparisons: Dict[str, str]
    ) -> str:
        """Synthesize overall comparison summary."""
        comparisons_text = '\n\n'.join(
            f"{aspect.title()}:\n{comp}"
            for aspect, comp in aspect_comparisons.items()
        )

        prompt = f"""Synthesize an overall comparison summary:

{comparisons_text}

Overall Summary:"""

        summary = self.rag.llm_client.generate(
            prompt, max_tokens=400, temperature=0.3
        )

        return summary
```

---

## 9. Observability and Metrics

### 9.1 Retrieval Quality Metrics

**Problem:** No visibility into retrieval quality (precision, recall).

**Solution:** Track standard IR metrics.

```python
from dataclasses import dataclass
from typing import Set

@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    precision_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    latency_ms: float


class RetrievalMetricsTracker:
    """Track and compute retrieval metrics."""

    def __init__(self):
        self.query_log = []

    def compute_metrics(
        self,
        retrieved_docs: List[int],
        relevant_docs: Set[int],
        k: int = 5
    ) -> RetrievalMetrics:
        """Compute standard IR metrics."""
        retrieved_k = retrieved_docs[:k]

        # Precision@K
        relevant_retrieved = len(set(retrieved_k) & relevant_docs)
        precision = relevant_retrieved / k if k > 0 else 0.0

        # Recall@K
        recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0.0

        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for idx, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                mrr = 1.0 / idx
                break

        # NDCG@K (simplified binary relevance)
        dcg = sum(
            1 / np.log2(idx + 2)
            for idx, doc_id in enumerate(retrieved_k)
            if doc_id in relevant_docs
        )

        ideal_dcg = sum(
            1 / np.log2(idx + 2)
            for idx in range(min(k, len(relevant_docs)))
        )

        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mean_reciprocal_rank=mrr,
            ndcg_at_k=ndcg,
            latency_ms=0.0  # Set externally
        )

    def log_query(
        self,
        query: str,
        retrieved_docs: List[int],
        relevant_docs: Set[int],
        latency_ms: float
    ):
        """Log query for offline analysis."""
        metrics = self.compute_metrics(retrieved_docs, relevant_docs)
        metrics.latency_ms = latency_ms

        self.query_log.append({
            'query': query,
            'retrieved': retrieved_docs,
            'relevant': list(relevant_docs),
            'metrics': metrics
        })

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics across queries."""
        if not self.query_log:
            return {}

        return {
            'avg_precision': np.mean([q['metrics'].precision_at_k for q in self.query_log]),
            'avg_recall': np.mean([q['metrics'].recall_at_k for q in self.query_log]),
            'avg_mrr': np.mean([q['metrics'].mean_reciprocal_rank for q in self.query_log]),
            'avg_ndcg': np.mean([q['metrics'].ndcg_at_k for q in self.query_log]),
            'avg_latency_ms': np.mean([q['metrics'].latency_ms for q in self.query_log]),
            'p95_latency_ms': np.percentile([q['metrics'].latency_ms for q in self.query_log], 95)
        }
```

### 9.2 Observability Dashboard

**Problem:** Hard to debug retrieval failures.

**Solution:** Structured logging and tracing.

```python
import time
import uuid
from contextlib import contextmanager

class RAGObservability:
    """Observability and tracing for RAG pipeline."""

    def __init__(self):
        self.traces = []
        self.current_trace = None

    @contextmanager
    def trace_query(self, query: str):
        """Context manager for tracing a query."""
        trace_id = str(uuid.uuid4())
        start_time = time.time()

        trace = {
            'trace_id': trace_id,
            'query': query,
            'start_time': start_time,
            'steps': [],
            'metadata': {}
        }

        self.current_trace = trace

        try:
            yield trace
        finally:
            trace['duration_ms'] = (time.time() - start_time) * 1000
            self.traces.append(trace)
            self.current_trace = None

    def log_step(
        self,
        step_name: str,
        metadata: Dict[str, Any],
        duration_ms: Optional[float] = None
    ):
        """Log a step in the current trace."""
        if not self.current_trace:
            return

        step = {
            'step': step_name,
            'metadata': metadata,
            'timestamp': time.time()
        }

        if duration_ms:
            step['duration_ms'] = duration_ms

        self.current_trace['steps'].append(step)

    def get_trace(self, trace_id: str) -> Optional[Dict]:
        """Get trace by ID."""
        for trace in self.traces:
            if trace['trace_id'] == trace_id:
                return trace
        return None

    def get_slow_queries(self, threshold_ms: float = 1000) -> List[Dict]:
        """Get queries exceeding latency threshold."""
        return [
            trace for trace in self.traces
            if trace['duration_ms'] > threshold_ms
        ]

    def export_traces(self, format: str = 'json') -> str:
        """Export traces for analysis."""
        if format == 'json':
            return json.dumps(self.traces, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Usage
class ObservableRAG(AdvancedRAGSDK):
    """RAG with observability."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observability = RAGObservability()

    def query(self, question: str, max_chunks: int = 5) -> RAGResponse:
        """Query with full observability."""
        with self.observability.trace_query(question) as trace:
            # Dense retrieval
            start = time.time()
            if self.use_hybrid:
                chunk_indices = self._hybrid_retrieve(question, max_chunks * 4)
                retrieval_method = 'hybrid'
            else:
                chunk_indices = self._dense_retrieve(question, max_chunks * 4)
                retrieval_method = 'dense'

            self.observability.log_step(
                'retrieval',
                {
                    'method': retrieval_method,
                    'candidates': len(chunk_indices)
                },
                duration_ms=(time.time() - start) * 1000
            )

            # Re-ranking
            if self.use_reranking:
                start = time.time()
                chunks = [self.chunks[i] for i in chunk_indices]
                reranked = self.reranker.rerank(question, chunks, max_chunks)
                final_chunks = [c for c, _s in reranked]

                self.observability.log_step(
                    'reranking',
                    {
                        'input_count': len(chunks),
                        'output_count': len(final_chunks)
                    },
                    duration_ms=(time.time() - start) * 1000
                )
            else:
                final_chunks = [self.chunks[i] for i in chunk_indices[:max_chunks]]

            # Answer generation
            start = time.time()
            response = self._generate_answer(question, final_chunks, [])

            self.observability.log_step(
                'generation',
                {
                    'prompt_length': len(question) + sum(len(c) for c in final_chunks),
                    'response_length': len(response.text)
                },
                duration_ms=(time.time() - start) * 1000
            )

            # Add trace ID to response
            trace['metadata']['response_length'] = len(response.text)
            response.trace_id = trace['trace_id']

            return response
```

---

## 10. AMD NPU Optimization

### 10.1 NPU-Accelerated Embeddings

**Problem:** CPU embeddings are slow (100ms for large documents).

**Solution:** Offload embedding generation to AMD NPU via Lemonade.

```python
class NPUAcceleratedEmbedder:
    """NPU-accelerated embedding generation."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text-v2-moe-GGUF",
        lemonade_url: str = "http://localhost:8000/api/v1"
    ):
        self.model_name = model_name
        self.lemonade_url = lemonade_url
        self.client = None

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings using NPU."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # Call Lemonade embedding endpoint
            response = requests.post(
                f"{self.lemonade_url}/embeddings",
                json={
                    'model': self.model_name,
                    'input': batch
                }
            )

            batch_embeddings = response.json()['data']
            embeddings.extend([e['embedding'] for e in batch_embeddings])

        return np.array(embeddings, dtype=np.float32)


class NPUOptimizedRAG(AdvancedRAGSDK):
    """RAG optimized for AMD NPU."""

    def __init__(self, config: RAGConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Replace embedding model with NPU-accelerated version
        if config.use_local_llm:
            self.embedding_model = NPUAcceleratedEmbedder(
                model_name=config.embedding_model,
                lemonade_url=config.base_url
            )
```

**Performance Impact:**
- CPU (sentence-transformers): ~150ms for 10 texts
- NPU (Lemonade): ~30ms for 10 texts
- **5x speedup for embedding generation**

### 10.2 NPU-Aware Batching

**Problem:** Small batches underutilize NPU.

**Solution:** Dynamic batching to maximize NPU throughput.

```python
class NPUBatchOptimizer:
    """Optimize batch sizes for NPU utilization."""

    def __init__(self, target_batch_size: int = 64):
        self.target_batch_size = target_batch_size
        self.pending_requests = []
        self.lock = asyncio.Lock()

    async def add_request(
        self,
        text: str,
        callback: Callable
    ):
        """Add embedding request to batch."""
        async with self.lock:
            self.pending_requests.append((text, callback))

            if len(self.pending_requests) >= self.target_batch_size:
                await self._flush_batch()

    async def _flush_batch(self):
        """Process accumulated batch."""
        if not self.pending_requests:
            return

        texts, callbacks = zip(*self.pending_requests)
        self.pending_requests.clear()

        # Generate embeddings in batch
        embeddings = await self._generate_batch(list(texts))

        # Invoke callbacks
        for embedding, callback in zip(embeddings, callbacks):
            callback(embedding)

    async def _generate_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch."""
        # Call NPU-accelerated embedder
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.embedder.encode,
            texts
        )
```

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Extend RAGSDK with advanced capabilities**

1. **Hybrid Search** (Priority: CRITICAL)
   - Implement `HybridRetriever` class
   - Add BM25 indexing to `AdvancedRAGSDK`
   - Implement RRF fusion
   - Test on 1000+ document corpus

2. **Cross-Encoder Re-ranking** (Priority: CRITICAL)
   - Integrate sentence-transformers cross-encoder
   - Add `rerank()` method to query pipeline
   - Benchmark precision improvement

3. **Observability** (Priority: HIGH)
   - Add `RAGObservability` class
   - Implement trace logging
   - Export metrics to JSON

**Success Metrics:**
- Hybrid search improves recall by 20%+
- Re-ranking improves precision@5 by 15%+
- All queries traced and logged

### Phase 2: Advanced Retrieval (Weeks 3-4)

4. **Semantic Chunking** (Priority: HIGH)
   - Implement `SemanticChunker` with LLM boundary detection
   - Test on technical documents
   - Compare to fixed chunking

5. **Multi-Hop Reasoning** (Priority: HIGH)
   - Implement `QueryDecomposer`
   - Build `MultiHopRetriever`
   - Test on complex queries

6. **Context Compression** (Priority: MEDIUM)
   - Implement `ContextCompressor`
   - Add extractive compression
   - Benchmark context window savings

**Success Metrics:**
- Semantic chunking reduces boundary splits by 50%+
- Multi-hop handles 80%+ of complex queries
- Compression saves 30%+ context tokens

### Phase 3: Knowledge Integration (Weeks 5-6)

7. **Knowledge Graph** (Priority: LOW)
   - Build `KnowledgeGraphBuilder`
   - Extract entities and relationships
   - Implement graph traversal retrieval

8. **Hierarchical Chunking** (Priority: MEDIUM)
   - Implement `HierarchicalChunker`
   - Maintain document structure
   - Test context enrichment

9. **Query Understanding** (Priority: MEDIUM)
   - Implement `QueryClassifier`
   - Add entity extraction
   - Build query expansion

**Success Metrics:**
- Knowledge graph improves recall by 10%+
- Hierarchical chunking preserves context
- Query understanding routes 90%+ correctly

### Phase 4: Production Hardening (Weeks 7-8)

10. **Async and Caching** (Priority: HIGH)
    - Implement `AsyncAdvancedRAG`
    - Add LRU query cache
    - Benchmark async speedup

11. **Index Optimization** (Priority: MEDIUM)
    - Implement IVF and HNSW indexes
    - Tune search parameters
    - Benchmark latency at scale

12. **NPU Optimization** (Priority: HIGH - AMD specific)
    - Integrate NPU-accelerated embeddings
    - Implement dynamic batching
    - Measure NPU utilization

**Success Metrics:**
- Async reduces p95 latency by 40%+
- Cache hit rate >60% for repeated queries
- NPU acceleration achieves 5x speedup

### Phase 5: Integration & Testing (Week 9)

13. **Multi-Document Synthesis**
    - Integrate with `MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md`
    - Implement `CrossDocumentRetriever`
    - Build `DocumentComparator`

14. **End-to-End Testing**
    - Create evaluation dataset with ground truth
    - Measure precision, recall, NDCG
    - Benchmark latency at scale (10K+ docs)

15. **Documentation**
    - Update `docs/sdk/sdks/rag.mdx`
    - Write migration guide
    - Create performance tuning guide

**Success Metrics:**
- Cross-document synthesis works for 95%+ queries
- All metrics tracked and exportable
- Documentation complete

---

## 12. Configuration API

### 12.1 Unified Configuration

```python
@dataclass
class AdvancedRAGConfig:
    """Unified configuration for advanced RAG."""

    # Base RAG config
    model: str = "Qwen3-Coder-30B-A3B-Instruct-GGUF"
    embedding_model: str = "nomic-embed-text-v2-moe-GGUF"
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Hybrid search
    use_hybrid_search: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60

    # Re-ranking
    use_reranking: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 5
    initial_retrieval_k: int = 20

    # Chunking
    chunking_strategy: str = "semantic"  # "fixed", "semantic", "hierarchical"
    use_llm_chunking: bool = False

    # Query processing
    enable_query_expansion: bool = True
    enable_entity_extraction: bool = True
    max_query_expansions: int = 2

    # Multi-hop
    enable_multi_hop: bool = False
    max_reasoning_iterations: int = 3

    # Context compression
    enable_compression: bool = True
    compression_ratio: float = 0.5
    compression_method: str = "extractive"  # "extractive", "llm"

    # Knowledge graph
    enable_knowledge_graph: bool = False
    kg_max_hops: int = 2

    # Observability
    enable_tracing: bool = True
    enable_metrics: bool = True
    log_retrieved_chunks: bool = False

    # Performance
    use_async: bool = True
    enable_cache: bool = True
    max_cache_size: int = 1000
    batch_size: int = 32

    # Index optimization
    index_type: str = "IVF"  # "Flat", "IVF", "IVFPQ", "HNSW"
    index_nlist: int = 100

    # AMD NPU
    use_npu_embeddings: bool = True
    npu_batch_size: int = 64


# Factory function
def create_advanced_rag(
    config: Optional[AdvancedRAGConfig] = None,
    **kwargs
) -> "AdvancedRAGSDK":
    """Create AdvancedRAGSDK with configuration."""
    config = config or AdvancedRAGConfig(**kwargs)

    # Build RAG instance
    rag = AdvancedRAGSDK(
        rag_config=RAGConfig(
            model=config.model,
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        ),
        hybrid_config=HybridSearchConfig(
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
            rrf_k=config.rrf_k
        )
    )

    # Enable features
    if config.use_hybrid_search:
        rag.enable_hybrid_search()

    if config.use_reranking:
        rag.use_reranking = True
        rag.reranker = CrossEncoderReranker(config.rerank_model)

    if config.enable_tracing:
        rag.observability = RAGObservability()

    return rag
```

### 12.2 Usage Examples

```python
# Example 1: Quick setup with defaults
rag = create_advanced_rag()
response = rag.query("What is AMD NPU?")

# Example 2: High-precision configuration
config = AdvancedRAGConfig(
    use_hybrid_search=True,
    use_reranking=True,
    enable_query_expansion=True,
    rerank_top_k=10,
    initial_retrieval_k=40
)
rag = create_advanced_rag(config)

# Example 3: Speed-optimized configuration
config = AdvancedRAGConfig(
    use_hybrid_search=False,  # Dense only
    use_reranking=False,
    enable_compression=False,
    use_async=True,
    enable_cache=True,
    use_npu_embeddings=True
)
rag = create_advanced_rag(config)

# Example 4: Complex reasoning configuration
config = AdvancedRAGConfig(
    enable_multi_hop=True,
    max_reasoning_iterations=5,
    enable_knowledge_graph=True,
    enable_query_expansion=True,
    use_reranking=True
)
rag = create_advanced_rag(config)
```

---

## 13. Backward Compatibility

### 13.1 Migration Path

**Existing code continues to work:**

```python
# Old code (still works)
from gaia.rag.sdk import RAGSDK, RAGConfig

rag = RAGSDK(RAGConfig())
rag.index_document("manual.pdf")
response = rag.query("What is X?")
```

**Opt-in to advanced features:**

```python
# New code (advanced features)
from gaia.rag.sdk import AdvancedRAGSDK, AdvancedRAGConfig

rag = AdvancedRAGSDK(AdvancedRAGConfig(
    use_hybrid_search=True,
    use_reranking=True
))
rag.index_document("manual.pdf")
rag.enable_hybrid_search()  # Must build BM25 index
response = rag.query("What is X?")
```

### 13.2 Feature Flags

**Gradual adoption:**

```python
# Start with base RAG
rag = RAGSDK(RAGConfig())

# Enable features incrementally
rag.enable_hybrid_search()  # Add BM25
rag.enable_reranking()      # Add cross-encoder
rag.enable_observability()  # Add tracing
```

---

## 14. Testing Strategy

### 14.1 Unit Tests

```python
# tests/unit/test_advanced_rag.py

def test_hybrid_search():
    """Test hybrid dense + sparse retrieval."""
    config = HybridSearchConfig()
    retriever = HybridRetriever(config)

    # Build index
    docs = ["NPU optimization guide", "GPU performance tuning"]
    retriever.build_bm25_index(docs)

    # Search
    results = retriever.sparse_search("NPU", top_k=1)
    assert len(results) == 1
    assert results[0][0] == 0  # First doc


def test_reciprocal_rank_fusion():
    """Test RRF fusion."""
    config = HybridSearchConfig(rrf_k=60)
    retriever = HybridRetriever(config)

    dense_results = [(0, 0.9), (1, 0.8)]
    sparse_results = [(1, 10.0), (0, 5.0)]

    fused = retriever.reciprocal_rank_fusion(dense_results, sparse_results)

    # Both methods ranked doc 1 higher -> should be first
    assert fused[0][0] == 1


def test_semantic_chunking():
    """Test LLM-based semantic chunking."""
    llm = MockLLMClient()
    chunker = SemanticChunker(llm, target_chunk_size=500)

    text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    chunks = chunker.chunk_document(text)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) > 0
```

### 14.2 Integration Tests

```python
# tests/integration/test_rag_pipeline.py

@pytest.mark.integration
def test_end_to_end_hybrid_rag():
    """Test full pipeline with hybrid search."""
    config = AdvancedRAGConfig(
        use_hybrid_search=True,
        use_reranking=True
    )
    rag = create_advanced_rag(config)

    # Index document
    rag.index_document("tests/fixtures/sample.pdf")
    rag.enable_hybrid_search()

    # Query
    response = rag.query("What is NPU?", max_chunks=3)

    assert response.text is not None
    assert len(response.chunks) == 3
    assert response.trace_id is not None


@pytest.mark.integration
def test_multi_hop_reasoning():
    """Test multi-hop query decomposition."""
    rag = create_advanced_rag(AdvancedRAGConfig())
    multi_hop = MultiHopRetriever(rag, rag.llm_client)

    result = multi_hop.retrieve_multi_hop(
        "Compare AMD NPU vs NVIDIA GPU for LLM inference",
        max_iterations=3
    )

    assert len(result['sub_queries']) >= 2
    assert result['final_answer'] is not None
```

### 14.3 Performance Benchmarks

```python
# tests/benchmarks/test_rag_performance.py

@pytest.mark.benchmark
def test_retrieval_latency(benchmark):
    """Benchmark retrieval latency."""
    rag = create_advanced_rag()
    rag.index_document("tests/fixtures/large_doc.pdf")

    result = benchmark(rag.query, "test query")
    assert result is not None


@pytest.mark.benchmark
def test_npu_vs_cpu_embeddings():
    """Compare NPU vs CPU embedding speed."""
    texts = ["sample text"] * 100

    # CPU
    cpu_embedder = SentenceTransformer("nomic-embed-text")
    cpu_time = timeit.timeit(lambda: cpu_embedder.encode(texts), number=10)

    # NPU
    npu_embedder = NPUAcceleratedEmbedder()
    npu_time = timeit.timeit(lambda: npu_embedder.encode(texts), number=10)

    speedup = cpu_time / npu_time
    assert speedup > 3.0  # Expect 3x+ speedup
```

---

## 15. Summary

### Key Enhancements

| Enhancement | Current | Advanced | Impact |
|-------------|---------|----------|--------|
| **Retrieval** | Dense only | Dense + BM25 + RRF | +25% recall |
| **Re-ranking** | None | Cross-encoder | +20% precision@5 |
| **Chunking** | Fixed 500 chars | Semantic LLM-based | +30% context quality |
| **Multi-hop** | Single step | Decomposition + iteration | Handle complex queries |
| **Compression** | None | Extractive + LLM | -40% context tokens |
| **Observability** | None | Traces + metrics | Full debugging |
| **NPU** | CPU only | NPU-accelerated | 5x embedding speed |
| **Caching** | None | LRU cache | 60% cache hit |
| **Index** | FlatL2 | IVF/HNSW | 10x search speed |

### Architecture Principles

1. **Backward Compatible** - Existing `RAGSDK` code works unchanged
2. **Opt-in Features** - Enable advanced features selectively
3. **AMD Optimized** - Leverage NPU for embeddings and inference
4. **Observable** - Full tracing and metrics
5. **Production Ready** - Async, caching, error handling

### Next Steps

1. Implement Phase 1 (hybrid search + re-ranking)
2. Benchmark on GAIA documentation corpus
3. Integrate with Multi-Document Synthesis
4. Roll out to production incrementally

---

## Appendix A: Dependencies

```toml
# pyproject.toml additions

[project.optional-dependencies]
rag-advanced = [
    "rank-bm25>=0.2.2",          # BM25 sparse retrieval
    "sentence-transformers>=2.2.2",  # Cross-encoder re-ranking
    "spacy>=3.7.0",              # Entity extraction
    "networkx>=3.0",             # Knowledge graph
    "faiss-cpu>=1.7.4",          # Vector indexing (use faiss-gpu on GPU systems)
]
```

Install:
```bash
uv pip install -e ".[rag-advanced]"
```

## Appendix B: Performance Targets

**Latency (p95):**
- Dense retrieval: <50ms
- Hybrid retrieval: <100ms
- Re-ranking (20 candidates): <150ms
- End-to-end query: <500ms

**Quality (on eval dataset):**
- Precision@5: >0.75
- Recall@10: >0.85
- NDCG@10: >0.80
- MRR: >0.70

**Scalability:**
- 100K documents: <1s query latency
- 1M documents: <5s query latency (with IVF index)
- Indexing throughput: >1000 docs/min

## Appendix C: References

- **Hybrid Search:** "Reciprocal Rank Fusion" (Cormack et al., 2009)
- **Re-ranking:** "Cross-Encoders for Semantic Search" (Reimers & Gurevych, 2020)
- **Context Compression:** "LLMLingua" (Jiang et al., 2023)
- **Multi-hop RAG:** "ReAct" (Yao et al., 2023), "Self-RAG" (Asai et al., 2024)
- **FAISS Optimization:** "Billion-Scale Similarity Search" (Johnson et al., 2019)

---

**End of Document**
