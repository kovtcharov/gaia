# Persistent Memory Framework for Gaia Agent SDK

**Date**: February 6, 2026
**Version**: 2.0
**Scope**: Generalized persistent memory architecture for any Gaia agent
**Foundation**: AMD Gaia Agent SDK 0.15.3+

---

## 1. Problem Statement

Gaia's `Agent` base class stores conversation in `self.conversation_history` — a Python list in RAM that disappears on restart. Agents are **stateless between sessions**:

- A GPU performance agent forgets all past benchmark analyses when it restarts
- A code assistant forgets the project conventions it learned from reviewing 50 files
- A medical assistant forgets patient interaction history
- A research agent forgets which sources it already consulted

**Every agent restarts from zero knowledge.** There's no compounding expertise, no learning, no memory of what worked or what failed.

For agents running **continuously until task completion** (your requirement), this is catastrophic. A 10-hour coding task that crashes at hour 9 loses 9 hours of context.

---

## 2. Three-Tier Memory Model

```
┌─────────────────────────────────────────────────────────────┐
│          TIER 1: WORKING MEMORY (Current Session)            │
│  conversation_history + task state + caches                  │
│  Lifetime: Until agent stops     Storage: RAM                │
│  Purpose: Active context for current execution               │
└──────────────────────┬──────────────────────────────────────┘
                       │ Consolidation (on session end)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│       TIER 2: EPISODIC MEMORY (Past Sessions)                │
│  Session summaries + full conversation archives              │
│  Lifetime: Configurable (30-365 days)  Storage: FAISS + JSON │
│  Purpose: "What did I do last week?" retrieval               │
└──────────────────────┬──────────────────────────────────────┘
                       │ Distillation (periodic, every N sessions)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│        TIER 3: SEMANTIC MEMORY (Distilled Knowledge)         │
│  Patterns, facts, rules, learned from experience             │
│  Lifetime: Permanent (with confidence decay)  Storage: SQLite│
│  Purpose: "MI355X allreduce is slow for small messages"     │
└─────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│   TIER 4: UNIVERSAL KNOWLEDGE DATABASE (Everything)          │
│  All tool outputs, files, reports, analyses, errors          │
│  Lifetime: Permanent      Storage: SQLite + Blob + Embeddings│
│  Purpose: Complete audit trail + deep context retrieval      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Tier 4: Universal Knowledge Database (NEW)

**Your requirement**: "Anything that agent interacts with should ultimately be stored in this database."

### 3.1 Comprehensive Storage

The Universal Knowledge DB stores:
- **Tool interactions**: Every tool call (args, result, timestamp, duration, success/fail)
- **Files**: Every file read/written (content, diffs, timestamps)
- **Documents**: All documents indexed (PDFs, code, markdown)
- **Analysis results**: All structured outputs (DataFrames, JSON reports)
- **Errors**: All errors encountered (with stack traces, recovery attempts)
- **Conversations**: Full message history (not just summaries)
- **Agent states**: State transitions, prompt changes, tool registry snapshots
- **External data**: Anything fetched (web pages, API responses, database queries)

### 3.2 Database Schema

```sql
-- Master index of all interactions
CREATE TABLE interactions (
    interaction_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,           -- ISO 8601 with milliseconds
    session_id TEXT,
    interaction_type TEXT NOT NULL,    -- 'tool_call', 'file_read', 'file_write', 'analysis', 'error', 'state_change'
    entity_type TEXT,                  -- 'file', 'tool', 'report', 'document'
    entity_id TEXT,
    success BOOLEAN,
    duration_ms INTEGER,
    metadata_json TEXT
);

CREATE INDEX idx_interactions_timestamp ON interactions(timestamp DESC);
CREATE INDEX idx_interactions_type ON interactions(interaction_type);
CREATE INDEX idx_interactions_session ON interactions(session_id);

-- Tool calls (detailed)
CREATE TABLE tool_calls (
    call_id TEXT PRIMARY KEY,
    interaction_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    args_json TEXT,
    result_json TEXT,
    success BOOLEAN,
    error TEXT,
    duration_ms INTEGER,
    state TEXT,                        -- Agent state when tool was called
    triggered_by TEXT,                 -- 'user_query', 'agent_plan', 'error_recovery'

    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
);

CREATE INDEX idx_tool_calls_timestamp ON tool_calls(timestamp DESC);
CREATE INDEX idx_tool_calls_tool ON tool_calls(tool_name);
CREATE INDEX idx_tool_calls_success ON tool_calls(success);

-- File operations
CREATE TABLE file_operations (
    operation_id TEXT PRIMARY KEY,
    interaction_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    operation_type TEXT NOT NULL,      -- 'read', 'write', 'edit', 'delete'
    file_path TEXT NOT NULL,
    content_before TEXT,               -- For edits
    content_after TEXT,
    diff TEXT,                         -- Unified diff
    success BOOLEAN,
    error TEXT,
    tool_name TEXT,

    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
);

CREATE INDEX idx_file_ops_timestamp ON file_operations(timestamp DESC);
CREATE INDEX idx_file_ops_path ON file_operations(file_path);
CREATE INDEX idx_file_ops_type ON file_operations(operation_type);

-- File content storage (with versioning)
CREATE TABLE file_versions (
    version_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    content BLOB,                      -- Full file content
    content_hash TEXT,                 -- SHA-256
    size_bytes INTEGER,
    encoding TEXT DEFAULT 'utf-8',
    created_by_tool TEXT,
    metadata_json TEXT
);

CREATE INDEX idx_file_versions_path ON file_versions(file_path);
CREATE INDEX idx_file_versions_timestamp ON file_versions(timestamp DESC);

-- Reports and analysis outputs
CREATE TABLE analysis_results (
    result_id TEXT PRIMARY KEY,
    interaction_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    analysis_type TEXT NOT NULL,       -- 'performance_analysis', 'code_review', 'test_results'
    input_files TEXT,                  -- JSON array of input file paths
    output_format TEXT,                -- 'dataframe', 'json', 'text', 'binary'
    output_data BLOB,                  -- Pickled DataFrame, JSON, or text
    output_summary TEXT,               -- Human-readable summary
    metadata_json TEXT,

    FOREIGN KEY (interaction_id) REFERENCES interactions(interaction_id)
);

CREATE INDEX idx_analysis_results_timestamp ON analysis_results(timestamp DESC);
CREATE INDEX idx_analysis_results_type ON analysis_results(analysis_type);

-- Documents indexed (for RAG)
CREATE TABLE indexed_documents (
    document_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    last_accessed_at TEXT,             -- For LRU eviction
    document_type TEXT,                -- 'pdf', 'code', 'markdown', 'webpage'
    num_chunks INTEGER,
    embedding_model TEXT,
    metadata_json TEXT
);

CREATE INDEX idx_indexed_docs_accessed ON indexed_documents(last_accessed_at DESC);

-- Document chunks (for detailed retrieval)
CREATE TABLE document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER,
    chunk_text TEXT NOT NULL,
    embedding BLOB,                    -- Numpy array pickled
    start_pos INTEGER,
    end_pos INTEGER,
    metadata_json TEXT,

    FOREIGN KEY (document_id) REFERENCES indexed_documents(document_id)
);

-- Errors and exceptions
CREATE TABLE errors (
    error_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    error_type TEXT,                   -- 'tool_error', 'llm_error', 'validation_error'
    error_message TEXT,
    stack_trace TEXT,
    context_json TEXT,                 -- State when error occurred
    recovery_attempted BOOLEAN,
    recovery_successful BOOLEAN,
    recovery_method TEXT
);

CREATE INDEX idx_errors_timestamp ON errors(timestamp DESC);
CREATE INDEX idx_errors_type ON errors(error_type);

-- Agent state snapshots (for resume)
CREATE TABLE state_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    agent_state TEXT,                  -- Current execution state
    conversation_history TEXT,         -- JSON
    working_memory TEXT,               -- JSON (caches, temp data)
    active_goals TEXT,                 -- JSON
    progress_percent REAL,
    checkpoint_reason TEXT             -- 'periodic', 'goal_complete', 'error', 'manual'
);

CREATE INDEX idx_snapshots_timestamp ON state_snapshots(timestamp DESC);
CREATE INDEX idx_snapshots_session ON state_snapshots(session_id);
```

### 3.3 Universal Storage API

```python
class UniversalKnowledgeDB:
    """Stores everything the agent interacts with."""

    def __init__(self, db_path: str = ".gaia/knowledge/universal.db"):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def record_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: dict,
        timestamp: str,
        duration_ms: int,
        session_id: str,
        state: str,
    ) -> str:
        """Record a tool invocation."""
        call_id = str(uuid.uuid4())[:12]
        interaction_id = f"tool-{call_id}"
        success = result.get("status") == "success"

        # Master interactions table
        self.db.execute("""
            INSERT INTO interactions VALUES (?, ?, ?, 'tool_call', 'tool', ?, ?, ?, ?)
        """, (interaction_id, timestamp, session_id, tool_name, success, duration_ms, json.dumps({})))

        # Detailed tool_calls table
        self.db.execute("""
            INSERT INTO tool_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'agent_plan')
        """, (call_id, interaction_id, timestamp, tool_name,
              json.dumps(args), json.dumps(result), success,
              result.get("error"), duration_ms, state))

        self.db.commit()
        return call_id

    def record_file_operation(
        self,
        operation_type: str,
        file_path: str,
        content_before: Optional[str],
        content_after: Optional[str],
        timestamp: str,
        tool_name: str,
    ) -> str:
        """Record a file read/write/edit."""
        op_id = str(uuid.uuid4())[:12]
        interaction_id = f"file-{op_id}"

        # Compute diff
        diff = None
        if content_before and content_after:
            import difflib
            diff = "\n".join(difflib.unified_diff(
                content_before.splitlines(),
                content_after.splitlines(),
                lineterm=""
            ))

        self.db.execute("""
            INSERT INTO file_operations VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?)
        """, (op_id, interaction_id, timestamp, operation_type, file_path,
              content_before, content_after, diff, tool_name))

        # Store version if write/edit
        if operation_type in ["write", "edit"] and content_after:
            self.store_file_version(file_path, content_after, timestamp, tool_name)

        # Log interaction
        self.db.execute("""
            INSERT INTO interactions VALUES (?, ?, NULL, 'file_operation', 'file', ?, 1, 0, ?)
        """, (interaction_id, timestamp, file_path, json.dumps({"operation": operation_type})))

        self.db.commit()
        return op_id

    def store_file_version(
        self,
        file_path: str,
        content: str,
        timestamp: str,
        created_by_tool: str
    ) -> str:
        """Store full file content with version."""
        import hashlib
        content_bytes = content.encode('utf-8')
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        version_id = f"{file_path}@{timestamp}"

        self.db.execute("""
            INSERT INTO file_versions VALUES (?, ?, ?, ?, ?, ?, 'utf-8', ?, ?)
        """, (version_id, file_path, timestamp, content_bytes,
              content_hash, len(content_bytes), created_by_tool, json.dumps({})))

        self.db.commit()
        return version_id

    def record_analysis(
        self,
        analysis_type: str,
        input_files: List[str],
        output_data: Any,
        summary: str,
        timestamp: str,
    ) -> str:
        """Store an analysis result (DataFrame, JSON report, etc.)."""
        result_id = str(uuid.uuid4())[:12]
        interaction_id = f"analysis-{result_id}"

        # Serialize output
        output_format = type(output_data).__name__
        if hasattr(output_data, 'to_pickle'):  # pandas DataFrame
            output_bytes = pickle.dumps(output_data)
            output_format = "dataframe"
        elif isinstance(output_data, dict):
            output_bytes = json.dumps(output_data).encode('utf-8')
            output_format = "json"
        else:
            output_bytes = str(output_data).encode('utf-8')
            output_format = "text"

        self.db.execute("""
            INSERT INTO analysis_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (result_id, interaction_id, timestamp, analysis_type,
              json.dumps(input_files), output_format, output_bytes,
              summary, json.dumps({})))

        self.db.execute("""
            INSERT INTO interactions VALUES (?, ?, NULL, 'analysis', 'report', ?, 1, 0, ?)
        """, (interaction_id, timestamp, result_id, json.dumps({"type": analysis_type})))

        self.db.commit()
        return result_id

    def record_error(
        self,
        error_type: str,
        error_message: str,
        stack_trace: str,
        timestamp: str,
        session_id: str,
        context: dict,
    ) -> str:
        """Store error details for analysis."""
        error_id = str(uuid.uuid4())[:12]

        self.db.execute("""
            INSERT INTO errors VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, NULL)
        """, (error_id, timestamp, session_id, error_type, error_message,
              stack_trace, json.dumps(context)))

        self.db.commit()
        return error_id

    def store_snapshot(
        self,
        session_id: str,
        agent_state: str,
        conversation_history: list,
        working_memory: dict,
        active_goals: list,
        progress: float,
        timestamp: str,
        reason: str,
    ) -> str:
        """Store complete agent state snapshot (for resume)."""
        snapshot_id = f"snap-{timestamp}"

        self.db.execute("""
            INSERT INTO state_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (snapshot_id, timestamp, session_id, agent_state,
              json.dumps(conversation_history), json.dumps(working_memory),
              json.dumps(active_goals), progress, reason))

        self.db.commit()
        return snapshot_id
```

---

## 4. Time-Aware Memory System

### 4.1 Temporal Queries

All memory retrieval is time-aware:

```python
class TemporalMemoryQuery:
    """Query memory with time-based filters and recency weighting."""

    def search_by_time_range(
        self,
        query: str,
        start_time: str,
        end_time: str,
        memory_tier: str = "all",
    ) -> List[dict]:
        """Search within specific time range."""
        results = []

        if memory_tier in ["all", "episodic"]:
            # Search episodic (sessions)
            sessions = self.db.execute("""
                SELECT * FROM sessions
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_time, end_time)).fetchall()
            results.extend(sessions)

        if memory_tier in ["all", "semantic"]:
            # Search semantic (knowledge confirmed in time range)
            knowledge = self.db.execute("""
                SELECT * FROM knowledge
                WHERE last_confirmed >= ? AND last_confirmed <= ?
            """, (start_time, end_time)).fetchall()
            results.extend(knowledge)

        if memory_tier in ["all", "universal"]:
            # Search universal DB
            interactions = self.db.execute("""
                SELECT * FROM interactions
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (start_time, end_time)).fetchall()
            results.extend(interactions)

        return results

    def search_recent(
        self,
        query: str,
        hours: int = 24,
        limit: int = 20,
    ) -> List[dict]:
        """Search recent memory (last N hours)."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        return self.search_by_time_range(query, cutoff, datetime.now().isoformat())

    def search_with_recency_boost(
        self,
        query: str,
        recency_weight: float = 0.3,
    ) -> List[Tuple[dict, float]]:
        """Search with scores boosted by recency.

        Score = base_relevance * (1 - recency_weight) + recency_score * recency_weight

        where recency_score = 1.0 for today, decaying to 0.0 for old entries.
        """
        # Semantic search (base relevance)
        chunks, scores = self.episodic_memory.rag._retrieve_chunks(query)

        # Load timestamps
        results_with_time = []
        for chunk, score in zip(chunks, scores):
            # Extract timestamp from chunk metadata
            timestamp = self._get_chunk_timestamp(chunk)
            recency = self._compute_recency_score(timestamp)

            # Combine scores
            final_score = (score * (1 - recency_weight)) + (recency * recency_weight)
            results_with_time.append((chunk, final_score, timestamp))

        # Re-sort by final score
        results_with_time.sort(key=lambda x: x[1], reverse=True)
        return [(r[0], r[1]) for r in results_with_time]

    def _compute_recency_score(self, timestamp: str, half_life_days: int = 30) -> float:
        """Exponential decay: 1.0 today → 0.5 at half_life → 0.0 at infinity."""
        ts = datetime.fromisoformat(timestamp)
        age_days = (datetime.now() - ts).days
        return 2 ** (-age_days / half_life_days)
```

### 4.2 Time-Based Memory Consolidation

New knowledge **updates** old knowledge based on timestamps:

```python
def consolidate_with_time_awareness(self, new_observation: dict, timestamp: str) -> str:
    """Add knowledge entry, handling temporal conflicts."""

    # Find existing knowledge about same pattern
    existing = self.db.execute("""
        SELECT * FROM knowledge
        WHERE category = ? AND pattern LIKE ?
    """, (new_observation["category"], f"%{new_observation['pattern'][:50]}%")).fetchall()

    if not existing:
        # New knowledge
        return self._insert_knowledge(new_observation, timestamp)

    # Exists — check timestamps
    existing_entry = dict(existing[0])
    existing_time = datetime.fromisoformat(existing_entry["last_confirmed"])
    new_time = datetime.fromisoformat(timestamp)

    if (new_time - existing_time).days < 7:
        # Recent confirmation → boost confidence
        new_confidence = min(existing_entry["confidence"] + 0.05, 1.0)
        new_evidence = existing_entry["evidence_count"] + 1
        self.db.execute("""
            UPDATE knowledge
            SET confidence = ?, evidence_count = ?, last_confirmed = ?
            WHERE id = ?
        """, (new_confidence, new_evidence, timestamp, existing_entry["id"]))
        return "updated"

    elif new_observation.get("contradicts", False):
        # Contradiction — create conflict entry
        self.db.execute("""
            INSERT INTO knowledge_conflicts (timestamp, existing_id, new_entry_json, resolution)
            VALUES (?, ?, ?, 'pending')
        """, (timestamp, existing_entry["id"], json.dumps(new_observation)))
        return "conflict"

    else:
        # Different time context — both valid
        return self._insert_knowledge(new_observation, timestamp)
```

---

## 5. Memory CRUD Operations

Full create, read, update, delete for all memory tiers:

### 5.1 Episodic Memory CRUD

```python
class EpisodicMemory:
    """Persistent memory of past sessions."""

    # CREATE
    def save_session(self, session_summary: dict, timestamp: str) -> str:
        """Create new session entry."""
        session_id = session_summary.get("session_id") or str(uuid.uuid4())[:8]
        session_summary["timestamp"] = timestamp

        # Save to JSON
        json_path = os.path.join(self.sessions_dir, f"{session_id}.json")
        with open(json_path, "w") as f:
            json.dump(session_summary, f, indent=2)

        # Index in FAISS
        text_doc = self._session_to_text(session_summary)
        text_path = os.path.join(self.sessions_dir, f"{session_id}.md")
        with open(text_path, "w") as f:
            f.write(text_doc)
        self.rag.index_document(text_path)

        return session_id

    # READ
    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve full session by ID."""
        json_path = os.path.join(self.sessions_dir, f"{session_id}.json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                return json.load(f)
        return None

    def search(
        self,
        query: str,
        time_range: Optional[Tuple[str, str]] = None,
        filters: Optional[dict] = None,
    ) -> List[Tuple[dict, float]]:
        """Semantic search with optional time filter."""
        # FAISS semantic search
        chunks, scores = self.rag._retrieve_chunks(query)

        # Load full sessions
        results = []
        for chunk, score in zip(chunks, scores):
            session_id = self._extract_session_id(chunk)
            session = self.get_session(session_id)
            if session:
                # Apply time filter
                if time_range:
                    ts = session["timestamp"]
                    if not (time_range[0] <= ts <= time_range[1]):
                        continue
                # Apply structured filters
                if filters and not self._matches_filters(session, filters):
                    continue
                results.append((session, score))

        return results

    # UPDATE
    def update_session(self, session_id: str, updates: dict, timestamp: str) -> bool:
        """Update session data (e.g., add outcome feedback)."""
        session = self.get_session(session_id)
        if not session:
            return False

        session.update(updates)
        session["updated_at"] = timestamp

        # Re-save
        json_path = os.path.join(self.sessions_dir, f"{session_id}.json")
        with open(json_path, "w") as f:
            json.dump(session, f, indent=2)

        # Re-index (FAISS handles updates via content hash)
        text_doc = self._session_to_text(session)
        text_path = os.path.join(self.sessions_dir, f"{session_id}.md")
        with open(text_path, "w") as f:
            f.write(text_doc)
        self.rag.index_document(text_path)

        return True

    # DELETE
    def delete_session(self, session_id: str) -> bool:
        """Remove session from memory (soft delete)."""
        # Mark as deleted (don't actually delete — preserve for audit)
        session = self.get_session(session_id)
        if session:
            session["deleted"] = True
            session["deleted_at"] = datetime.now().isoformat()
            json_path = os.path.join(self.sessions_dir, f"{session_id}.json")
            with open(json_path, "w") as f:
                json.dump(session, f)
            return True
        return False
```

### 5.2 Semantic Memory CRUD

```python
class SemanticMemory:
    """Distilled knowledge with full CRUD."""

    # CREATE
    def insert(self, entry: dict, timestamp: str) -> str:
        """Add new knowledge entry."""
        entry_id = entry.get("id") or self._generate_id(entry)
        entry["created_at"] = timestamp
        entry["last_confirmed"] = timestamp

        self.db.execute("""
            INSERT INTO knowledge VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (entry_id, entry["category"], entry.get("hardware"),
              entry.get("model"), entry.get("operation"), entry["pattern"],
              entry.get("recommendation"), entry.get("confidence", 0.5),
              entry.get("evidence_count", 1), timestamp, timestamp,
              json.dumps(entry.get("source_sessions", [])),
              json.dumps(entry.get("conditions", [])), 0))
        self.db.commit()
        return entry_id

    # READ
    def get(self, entry_id: str) -> Optional[dict]:
        """Retrieve knowledge entry by ID."""
        row = self.db.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,)).fetchone()
        return dict(row) if row else None

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        hardware: Optional[str] = None,
        min_confidence: float = 0.0,
        min_evidence: int = 0,
        time_range: Optional[Tuple[str, str]] = None,
    ) -> List[dict]:
        """Structured search with multiple filters."""
        sql = "SELECT * FROM knowledge WHERE confidence >= ? AND evidence_count >= ?"
        params = [min_confidence, min_evidence]

        if category:
            sql += " AND category = ?"
            params.append(category)
        if hardware:
            sql += " AND hardware = ?"
            params.append(hardware)
        if time_range:
            sql += " AND last_confirmed >= ? AND last_confirmed <= ?"
            params.extend(time_range)

        sql += " ORDER BY confidence DESC, evidence_count DESC LIMIT 50"

        rows = self.db.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # UPDATE
    def update(
        self,
        entry_id: str,
        confidence: Optional[float] = None,
        evidence_count: Optional[int] = None,
        last_confirmed: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Update knowledge entry."""
        updates = []
        params = []

        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if evidence_count is not None:
            updates.append("evidence_count = ?")
            params.append(evidence_count)
        if last_confirmed:
            updates.append("last_confirmed = ?")
            params.append(last_confirmed)

        if not updates:
            return False

        sql = f"UPDATE knowledge SET {', '.join(updates)} WHERE id = ?"
        params.append(entry_id)
        self.db.execute(sql, params)
        self.db.commit()
        return True

    # DELETE
    def delete(self, entry_id: str, reason: str = "") -> bool:
        """Mark knowledge as invalid (soft delete)."""
        timestamp = datetime.now().isoformat()

        # Don't actually delete — mark as deprecated
        self.db.execute("""
            UPDATE knowledge
            SET metadata = json_set(metadata, '$.deprecated', true),
                metadata = json_set(metadata, '$.deprecated_at', ?),
                metadata = json_set(metadata, '$.deprecated_reason', ?)
            WHERE id = ?
        """, (timestamp, reason, entry_id))

        self.db.commit()
        return True
```

---

## 6. Recency Prioritization

**Your requirement**: "Memories that are more recent may be prioritized or used over older memories."

### 6.1 Recency-Weighted Retrieval

```python
def retrieve_with_recency_priority(
    self,
    query: str,
    max_results: int = 10,
    recency_weight: float = 0.4,  # 40% weight on recency
    time_decay_days: int = 90,
) -> List[Tuple[dict, float, str]]:
    """
    Retrieve knowledge with both relevance and recency considered.

    Score = (relevance * (1 - recency_weight)) + (recency * recency_weight)

    Returns: List of (entry, final_score, reason)
    """
    # Semantic search for base relevance
    all_knowledge = self.search(min_confidence=0.3)

    results = []
    for entry in all_knowledge:
        # Relevance score (keyword matching + semantic)
        relevance = self._compute_relevance(query, entry)

        # Recency score (exponential decay)
        last_confirmed = datetime.fromisoformat(entry["last_confirmed"])
        age_days = (datetime.now() - last_confirmed).days
        recency = 2 ** (-age_days / time_decay_days)

        # Combined score
        final_score = (relevance * (1 - recency_weight)) + (recency * recency_weight)

        results.append((entry, final_score, f"relevance={relevance:.2f}, recency={recency:.2f}"))

    # Sort by final score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_results]
```

---

## 7. Continuous Execution Integration

For agents that "run until task complete":

### 7.1 Checkpoint/Resume System

```python
class ContinuousExecutionMemory:
    """Memory management for long-running tasks."""

    def __init__(self, universal_db: UniversalKnowledgeDB):
        self.db = universal_db
        self.checkpoint_interval_steps = 50

    def checkpoint(
        self,
        agent: Agent,
        session_id: str,
        step_number: int,
        reason: str = "periodic",
    ) -> str:
        """Save complete agent state for resume."""
        timestamp = datetime.now().isoformat()

        snapshot_id = self.db.store_snapshot(
            session_id=session_id,
            agent_state=agent.execution_state,
            conversation_history=agent.conversation_history,
            working_memory={
                "trace_cache": {k: "TreePerfAnalyzer" for k in agent._trace_cache.keys()},
                "session_context": agent._session_context,
            },
            active_goals=agent.manifest_tracker.get_active_goals(),
            progress=agent.manifest_tracker.compute_progress().overall_percent,
            timestamp=timestamp,
            reason=reason,
        )

        logging.info(f"Checkpoint {snapshot_id} saved at step {step_number}")
        return snapshot_id

    def resume(self, snapshot_id: str) -> dict:
        """Restore agent state from checkpoint."""
        row = self.db.db.execute(
            "SELECT * FROM state_snapshots WHERE snapshot_id = ?",
            (snapshot_id,)
        ).fetchone()

        if not row:
            raise ValueError(f"Snapshot {snapshot_id} not found")

        return {
            "session_id": row["session_id"],
            "agent_state": row["agent_state"],
            "conversation_history": json.loads(row["conversation_history"]),
            "working_memory": json.loads(row["working_memory"]),
            "active_goals": json.loads(row["active_goals"]),
            "progress": row["progress_percent"],
            "checkpoint_timestamp": row["timestamp"],
        }

    def auto_checkpoint_if_due(
        self,
        agent: Agent,
        session_id: str,
        step_number: int,
    ) -> Optional[str]:
        """Automatically checkpoint at intervals."""
        if step_number % self.checkpoint_interval_steps == 0:
            return self.checkpoint(agent, session_id, step_number, reason="periodic")
        return None
```

---

## 8. Gaia SDK Integration

### 8.1 Extension Points

| Feature | Gaia Hook | Implementation |
|---------|-----------|----------------|
| Tool call storage | `_execute_tool()` override | Record to universal DB before/after execution |
| File operation tracking | `_post_process_tool_result()` | Detect file tools, store content + diffs |
| State snapshot | `process_query()` wrapper | Checkpoint every N steps |
| Memory injection | `_compose_system_prompt()` | Retrieve recent + relevant knowledge |
| Session save | End of `process_query()` | Generate summary, save to episodic memory |
| Error logging | `_execute_tool()` exception handling | Store error in universal DB |

### 8.2 Complete Integration Example

```python
from gaia.agents.base.agent import Agent

class MemoryAwareAgent(Agent, MemoryToolsMixin):
    """Agent with full persistent memory."""

    def __init__(self, config):
        self.universal_db = UniversalKnowledgeDB(
            db_path=os.path.join(config.memory_dir, "universal.db")
        )
        self.episodic_memory = EpisodicMemory(
            memory_dir=os.path.join(config.memory_dir, "episodic")
        )
        self.semantic_memory = SemanticMemory(
            db_path=os.path.join(config.memory_dir, "knowledge.db")
        )
        self.checkpoint_manager = ContinuousExecutionMemory(self.universal_db)

        super().__init__(config)

    def _execute_tool(self, tool_name: str, tool_args: dict) -> Any:
        """Override to record all tool calls."""
        start_time = datetime.now()
        timestamp = start_time.isoformat()

        try:
            result = super()._execute_tool(tool_name, tool_args)
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            # Store in universal DB
            self.universal_db.record_tool_call(
                tool_name=tool_name,
                args=tool_args,
                result=result,
                timestamp=timestamp,
                duration_ms=duration_ms,
                session_id=self._session_id,
                state=self.execution_state,
            )

            return result
        except Exception as e:
            # Log error
            self.universal_db.record_error(
                error_type="tool_error",
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                timestamp=datetime.now().isoformat(),
                session_id=self._session_id,
                context={"tool": tool_name, "args": tool_args},
            )
            raise

    def process_query(self, user_input: str, **kwargs):
        """Override to add checkpointing and session save."""
        self._session_id = str(uuid.uuid4())[:8]
        self._step_count = 0

        result = super().process_query(user_input, **kwargs)

        # Generate and save session summary
        summary = self._generate_session_summary(result)
        self.episodic_memory.save_session(summary, timestamp=datetime.now().isoformat())

        return result

    def _post_process_tool_result(self, tool_name, tool_args, tool_result):
        """Override to checkpoint and track file ops."""
        super()._post_process_tool_result(tool_name, tool_args, tool_result)

        self._step_count += 1

        # Auto-checkpoint
        self.checkpoint_manager.auto_checkpoint_if_due(
            agent=self,
            session_id=self._session_id,
            step_number=self._step_count,
        )

        # Track file operations
        if tool_name in ["write_file", "edit_file", "read_file"]:
            self._record_file_operation(tool_name, tool_args, tool_result)
```

---

## 9. MemoryToolsMixin (Complete)

```python
class MemoryToolsMixin:
    """Mixin providing memory management tools for any agent."""

    def register_memory_tools(self) -> None:
        """Register all memory tools."""

        @tool(atomic=True, name="recall")
        def recall(
            query: str,
            time_range: str = "",
            tier: str = "all",
        ) -> Dict[str, Any]:
            """Search all memory tiers for relevant information.

            Args:
                query: What to search for
                time_range: Optional, format "2026-01-01:2026-02-01" or "last_7_days"
                tier: 'working', 'episodic', 'semantic', 'universal', 'all'
            """
            # Parse time range
            start_time, end_time = self._parse_time_range(time_range)

            results = {"working": [], "episodic": [], "semantic": [], "universal": []}

            if tier in ["all", "working"]:
                # Search current conversation
                results["working"] = self._search_conversation(query)

            if tier in ["all", "episodic"]:
                # Search past sessions
                sessions = self.episodic_memory.search(
                    query, time_range=(start_time, end_time) if start_time else None
                )
                results["episodic"] = [{"session": s, "score": score} for s, score in sessions[:5]]

            if tier in ["all", "semantic"]:
                # Search knowledge base
                knowledge = self.semantic_memory.search_with_recency(
                    query, time_range=(start_time, end_time) if start_time else None
                )
                results["semantic"] = knowledge[:5]

            if tier in ["all", "universal"]:
                # Search all interactions
                interactions = self.universal_db.search_interactions(
                    query, start_time, end_time
                )
                results["universal"] = interactions[:10]

            return {"status": "success", "results": results}

        @tool(name="remember")
        def remember(
            content: str,
            category: str = "general",
            confidence: float = 0.7,
        ) -> Dict[str, Any]:
            """Explicitly store information in semantic memory.

            Use when the user provides important context or corrections.
            """
            timestamp = datetime.now().isoformat()
            entry_id = self.semantic_memory.insert(
                entry={
                    "category": category,
                    "pattern": content,
                    "confidence": confidence,
                    "evidence_count": 1,
                    "source": "explicit_user_input",
                },
                timestamp=timestamp,
            )

            return {
                "status": "success",
                "entry_id": entry_id,
                "message": f"Stored in semantic memory with confidence {confidence}",
            }

        @tool(name="update_memory")
        def update_memory(
            entry_id: str,
            field: str,
            new_value: Any,
        ) -> Dict[str, Any]:
            """Update a specific memory entry.

            Use when correcting or refining stored knowledge.
            """
            timestamp = datetime.now().isoformat()
            success = self.semantic_memory.update(
                entry_id=entry_id,
                **{field: new_value, "last_confirmed": timestamp}
            )

            return {
                "status": "success" if success else "error",
                "entry_id": entry_id,
                "updated_field": field,
            }

        @tool(name="forget")
        def forget(entry_id: str, reason: str = "") -> Dict[str, Any]:
            """Remove or deprecate a memory entry.

            Use when knowledge is incorrect or no longer relevant.
            """
            success = self.semantic_memory.delete(entry_id, reason)

            return {
                "status": "success" if success else "error",
                "entry_id": entry_id,
                "message": "Entry marked as deprecated (soft delete)",
            }

        @tool(atomic=True, name="list_recent_memories")
        def list_recent_memories(hours: int = 24, limit: int = 20) -> Dict[str, Any]:
            """Show memories created/updated in the last N hours.

            Use to review recent learning.
            """
            cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

            # Recent sessions
            recent_sessions = self.episodic_memory.search(
                query="",
                time_range=(cutoff, datetime.now().isoformat())
            )

            # Recent knowledge
            recent_knowledge = self.semantic_memory.search(
                time_range=(cutoff, datetime.now().isoformat())
            )

            return {
                "status": "success",
                "time_range": f"Last {hours} hours",
                "sessions": [s[0] for s in recent_sessions[:limit]],
                "knowledge": recent_knowledge[:limit],
            }
```

---

## 10. Configuration

```python
@dataclass
class MemoryConfig:
    """Configuration for persistent memory system."""

    # Directories
    memory_dir: str = ".gaia/memory"
    universal_db_path: str = ".gaia/memory/universal.db"
    episodic_dir: str = ".gaia/memory/episodic"
    semantic_db_path: str = ".gaia/memory/knowledge.db"

    # Episodic memory (sessions)
    episodic_retention_days: int = 365
    max_episodic_sessions: int = 1000
    episodic_chunk_size: int = 300
    episodic_max_chunks: int = 10

    # Semantic memory (knowledge)
    knowledge_confidence_threshold: float = 0.3  # Min confidence to retrieve
    knowledge_evidence_threshold: int = 2        # Min evidence to trust
    knowledge_decay_half_life_days: int = 90
    max_knowledge_entries: int = 10000

    # Universal DB
    universal_store_file_content: bool = True    # Store full file versions
    universal_store_tool_results: bool = True
    universal_retention_days: int = 180          # Archive after 6 months

    # Checkpointing
    checkpoint_enabled: bool = True
    checkpoint_interval_steps: int = 50
    checkpoint_on_error: bool = True
    max_checkpoints_per_session: int = 50

    # Consolidation
    consolidation_enabled: bool = True
    consolidation_frequency_sessions: int = 10  # Every 10 sessions
    consolidation_lookback: int = 20

    # Context injection
    inject_recent_memory: bool = True
    inject_relevant_memory: bool = True
    max_memory_context_tokens: int = 1000
    recency_weight: float = 0.4

    # Time awareness
    enable_time_decay: bool = True
    prioritize_recent: bool = True
```

---

## 11. Examples Across Agent Types

### Example 1: Code Assistant

```python
# Agent builds a web app over 8 hours
agent = CodeAgent(MemoryConfig(
    checkpoint_interval_steps=50,
    inject_recent_memory=True,
))

# Hour 1: Creates 15 files
# Memory stores: all file versions, imports detected, decisions made

# Hour 4: Crash
# Memory has: 4 checkpoints, 15 file versions, all tool calls

# Resume:
last_checkpoint = agent.checkpoint_manager.get_latest_checkpoint(session_id)
agent.resume_from_checkpoint(last_checkpoint)
# → Agent restores conversation, knows what files exist, continues

# Hour 8: Task complete
# Memory consolidates: "This project uses FastAPI + React + PostgreSQL,
# JWT auth pattern, RESTful API design, pytest for testing"

# Next project (week later):
# Agent recalls: "You previously used FastAPI for backend. Continue this pattern?"
```

### Example 2: Performance Analyst

```python
# Nightly benchmark analysis for 30 days
agent = PerformanceAgent(MemoryConfig(
    consolidation_frequency_sessions=5,  # Every 5 nights
    knowledge_decay_half_life_days=60,
))

# Night 1: Analyzes MI355X trace
# Memory stores: trace analysis results, findings, recommendations

# Night 5: Consolidation triggered
# Pattern extracted: "MI355X allreduce consistently at 45% bandwidth for EP-8"
# Confidence: 0.6 (5 sessions evidence)

# Night 15: Pattern reconfirmed
# Confidence updated: 0.85 (15 sessions evidence)

# Night 20: User provides feedback
agent.remember("allreduce < 50% is normal for MI300X", category="correction", confidence=1.0)

# Night 21: Agent analyzes MI300X trace
# Retrieves recent memory: "allreduce < 50% is normal for MI300X (confidence: 1.0)"
# → Doesn't flag as anomaly
```

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Stale knowledge** | Time decay, last_confirmed tracking, periodic re-validation |
| **Contradictory entries** | Conflict detection on insert, conflict resolution workflow |
| **Memory bloat** | Retention policies, LRU eviction, archival to cold storage |
| **Slow queries** | Indexes on all time fields, lazy loading, caching |
| **Corruption** | Atomic transactions, WAL mode, backup on checkpoint |
| **Privacy** | Local storage only, encryption at rest (optional), audit log |

---

*Persistent Memory Framework for Gaia Agent SDK.*
*Provides 4-tier memory (working, episodic, semantic, universal) with full CRUD, timestamps, recency prioritization, and checkpoint/resume for continuous execution.*
