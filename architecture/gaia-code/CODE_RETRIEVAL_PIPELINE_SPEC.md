# Code Retrieval Pipeline Specification (v1)

## Context

The GAIA Code agent's current `search_codebase` tool uses naive keyword matching on docstrings only (`tools.py:573-626`). Every tool call (`search_codebase`, `find_symbol`, `get_dependents`, `detect_issues`, `analyze_architecture`) creates a brand new `CodebaseIndex` and re-indexes the entire repo from scratch. There is no semantic code search, no incremental indexing, no persistent code index, and no code-aware chunking.

This spec defines a **multi-layer code retrieval pipeline** that gives the agent:
1. **Bug fixing** — Find relevant code given a bug description
2. **Codebase Q&A** — Answer "what is the architecture?" AND "what does line 47 do?"
3. **Summarization** — Summaries at any granularity (codebase → module → class → function)
4. **Impact analysis** — "What breaks if I change this function?"
5. **Structural queries** — "Find all classes that inherit from Agent"

**Priority:** Retrieval quality over indexing speed. Indexing can take minutes if it means better answers.

**Decisions:**
- LLM enrichment defaults to cloud (Claude API) matching gaia-code agent defaults; local LLM (Lemonade/Qwen3-Coder) as opt-in
- Build all 4 layers in this pass (full pipeline)
- File watcher for auto-updating index as files change
- Incremental indexing for large codebase support

---

## Architecture: 4 Layers

```
Query
  │
  ▼
[Layer 4: Context Assembler]  — Classifies query, assembles right context for LLM
  │
  ▼
[Layer 3: Retrieval Index]    — FAISS embeddings + FTS5 keyword search
  │
  ▼
[Layer 2: Semantic Annotations] — Summaries/concepts per chunk (AST + optional LLM)
  │
  ▼
[Layer 1: Structural Index]   — AST-parsed symbols, deps, file hashes, code chunks
  │
  ▼
[Source Files on Disk]
```

**New primary file:** `src/gaia/agents/gaia_code/code_retrieval.py` — `CodeRetrievalPipeline` class

---

## Layer 1: Structural Index

**Extends:** `codebase_index.py` (755 lines, already has AST parsing, dependency graphs, Tarjan's SCC)

### What Changes

**1. File Change Detection** — SHA-256 hashing per file for incremental re-indexing

**2. Code Chunk Extraction** — New `CodeChunk` dataclass, chunk at natural code boundaries:

| Chunk Type | Rule |
|-----------|------|
| `module_docstring` | File-level docstring |
| `import_block` | All imports grouped as one chunk |
| `class` | Entire class (for classes ≤200 lines) |
| `function` | Each top-level function |
| `method` | Each method in large classes (>200 lines), with class signature as context |
| `standalone_block` | Module-level code (assignments, if-name-main) |

**CodeChunk fields:** chunk_id, file_path, relative_path, module_path, chunk_type, symbol_name, parent_symbol, start_line, end_line, source_code, docstring, decorators, signature, complexity, dependencies, content_hash

**3. Non-Python Files** — Chunk by structure where possible:
- `.js/.ts/.tsx` → blank-line-separated blocks (future: tree-sitter)
- `.yaml/.json/.toml` → one chunk per file
- `.md/.rst` → chunk by header sections
- Other → fixed-size blocks (500 lines)

**4. Persistence** — New `code_index.db` SQLite database (separate from knowledge.db):

```sql
CREATE TABLE repositories (id INTEGER PRIMARY KEY, root_path TEXT, last_full_index TEXT, last_incremental TEXT, total_files INTEGER, total_chunks INTEGER);
CREATE TABLE files (file_path TEXT PRIMARY KEY, repo_id INTEGER, relative_path TEXT, module_path TEXT, content_hash TEXT, indexed_at TEXT, line_count INTEGER, size_bytes INTEGER, language TEXT, docstring TEXT);
CREATE TABLE code_chunks (chunk_id TEXT PRIMARY KEY, file_path TEXT, chunk_type TEXT, symbol_name TEXT, parent_symbol TEXT, start_line INTEGER, end_line INTEGER, source_code TEXT, docstring TEXT, signature TEXT, decorators TEXT, complexity INTEGER, dependencies TEXT, content_hash TEXT);
CREATE TABLE dependencies (source_file TEXT, target_file TEXT, import_type TEXT, symbols_imported TEXT);
CREATE TABLE annotations (chunk_id TEXT PRIMARY KEY, summary TEXT, purpose TEXT, concepts TEXT, relationships TEXT, search_text TEXT, annotation_source TEXT, generated_at TEXT);
CREATE TABLE summaries (id INTEGER PRIMARY KEY AUTOINCREMENT, level TEXT, scope TEXT, summary TEXT, generated_at TEXT);
```

**5. Incremental Indexing Algorithm:**
```
1. Walk source files, compute SHA-256 per file
2. Compare against stored hashes in code_index.db
3. Classify: UNCHANGED (skip) | MODIFIED (re-index) | NEW (index) | DELETED (remove)
4. Only process changed files
5. Rebuild dependency graph for affected files
6. Rebuild FAISS index (full — FAISS doesn't support deletion)
```

**6. File Watcher (Auto-Update):**
- Background thread watches source directories for file changes
- Debounces changes (500ms) to batch rapid edits
- Triggers incremental re-index automatically
- Updates FAISS index after batch completes

---

## Layer 2: Semantic Annotations

Each code chunk gets a natural-language annotation enabling semantic search.

### CodeAnnotation Fields
summary, purpose, concepts (list), relationships (list), search_text, annotation_source, generated_at

### Two Annotation Modes

**Mode 1: AST-Only (fallback, no LLM required)**
```
summary = "{chunk_type} '{symbol_name}' in {module_path}"
purpose = docstring if present, else ""
concepts = nouns extracted from docstring + decorator names
relationships = ["inherits from X", "imports Y"]
search_text = "{summary} | {purpose} | {concepts joined}"
```

**Mode 2: LLM-Enriched (default for gaia-code, higher quality)**
```
LLM prompt per chunk:
  "Summarize this code in 1-2 sentences. What does it do?"
  + source code
  → Returns SUMMARY, PURPOSE, CONCEPTS
```
Batched: 10 chunks per LLM call. ~200 calls for 100K LoC. ~3-4 min with local LLM.

**LLM source is configurable:**
- `llm_provider="cloud"` (default): Uses Claude API. Higher quality. Reports estimated cost before proceeding.
- `llm_provider="local"`: Uses Lemonade/Qwen3-Coder. Free, private, no API costs.
- `llm_provider="none"`: AST-only annotations. No LLM calls.

### Higher-Level Summaries

| Level | Content | How |
|-------|---------|-----|
| File | What this file does, key exports | Concatenate chunk summaries |
| Module | What this package does | Concatenate file summaries |
| Architecture | Full codebase overview | Concatenate module summaries |

---

## Layer 3: Retrieval Index

### What Gets Embedded
The `search_text` field from CodeAnnotation — natural language, not raw code.

### FAISS Index
- Model: all-MiniLM-L6-v2 (384d) via existing EmbeddingEngine
- Index type: `IndexFlatIP` (inner product, better for normalized embeddings)
- For >50K chunks: switch to `IndexIVFFlat`
- Files: `code_chunks.faiss` + `code_chunks_mapping.pkl`

### FTS5 Index
```sql
CREATE VIRTUAL TABLE code_chunks_fts USING fts5(chunk_id, symbol_name, module_path, source_code, summary, concepts);
```

### Hybrid Search
Combine FAISS (semantic) + FTS5 (keyword) with configurable alpha weight.

---

## Layer 4: Context Assembler

### Query Classification (rule-based, no LLM)

| Type | Heuristic | Strategy |
|------|-----------|----------|
| **Structural** | "find all", "list", "which classes" | FTS5-heavy (alpha=0.2), filter by chunk_type |
| **Semantic** | Natural language about behavior | FAISS-heavy (alpha=0.8) |
| **Locational** | Contains file paths, symbol names | Direct SQLite lookup |
| **Impact** | "what breaks", "what depends" | Dependency graph traversal |
| **Summary** | "summarize", "overview", "architecture" | Hierarchical summaries |

### Context Budget
Default 8000 tokens. Add results by score until budget exhausted.

### Impact Analysis
```
impact_analysis(file_path, symbol_name):
1. Find chunk for this symbol
2. Get files that import from this file (reverse deps)
3. For each dependent: find chunks that reference target symbol
4. BFS transitive dependents (depth 3)
5. Return: direct dependents, symbol references, transitive impact, risk level
```

---

## Performance Targets

| Operation | 100K LoC | 1M+ LoC |
|-----------|----------|---------|
| Full index (L1+L2 AST-only) | <40s | <7 min |
| LLM enrichment (L2) | <5 min | <30 min |
| FAISS build (L3) | <5s | <30s |
| Incremental update (10 files) | <3s | <5s |
| Search query | <100ms | <200ms |
| Full pipeline query (cached) | <500ms | <1s |

---

## Key Files

| File | Action |
|------|--------|
| `src/gaia/agents/gaia_code/code_retrieval.py` | **CREATE** — Main pipeline class |
| `src/gaia/agents/gaia_code/codebase_index.py` | **MODIFY** — Add CodeChunk, extract_code_chunks |
| `src/gaia/agents/gaia_code/tools.py` | **MODIFY** — Rewire tools to use pipeline |
| `src/gaia/agents/gaia_code/embedding_engine.py` | **REUSE AS-IS** |
| `src/gaia/agents/gaia_code/vector_search.py` | **REUSE AS-IS** |
