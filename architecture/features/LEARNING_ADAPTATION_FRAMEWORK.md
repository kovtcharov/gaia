# Learning & Adaptation Framework for Gaia Agent SDK

**Date**: February 6, 2026
**Version**: 2.0
**Scope**: Feedback collection, outcome tracking, knowledge distillation, and continuous improvement for any Gaia agent
**Foundation**: AMD Gaia Agent SDK 0.15.3+
**Optimized For**: Code assistants and continuous execution workflows

---

## 1. Problem Statement

Static agents deliver the same quality on day 1 and day 100. They don't learn from:
- User corrections ("That's wrong, actually X uses Y not Z")
- Outcomes ("Your recommendation worked, we got 15% improvement")
- Failures ("That approach didn't work, try differently")
- Patterns ("Every time I ask about X, you should check Y first")

For **code assistants running continuously until task complete**, this is expensive:
- Agent repeats the same mistakes across a 10-hour build session
- Agent doesn't learn project conventions after seeing 50 files
- Agent suggests patterns that failed 3 steps ago
- Each new task starts from zero knowledge

### The Solution

A **closed-loop learning system** where:
1. Agent executes task
2. Records execution trace (what worked, what failed)
3. Collects feedback (explicit + implicit + automated)
4. Extracts patterns ("When X happens, Y strategy works 90% of time")
5. Consolidates into knowledge ("Flash attention on AMD uses hipCK kernels")
6. Adapts behavior (next execution uses updated knowledge)

---

## 2. The Learning Loop (6 Steps)

```
┌──────────────────────────────────────────────────────────┐
│  1. EXECUTE                                               │
│  Agent performs task, calls tools, generates outputs      │
│  All actions logged with timestamps in universal DB       │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  2. RECORD                                                │
│  Session summary generated, stored in episodic memory     │
│  Tool calls, results, decisions stored in universal DB    │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  3. FEEDBACK                                              │
│  Explicit: User tells agent (correction, outcome)         │
│  Implicit: Behavioral signals (re-ask, detail request)   │
│  Automated: Outcome measurement (tests pass/fail)         │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  4. EXTRACT                                               │
│  Process feedback into structured learnings              │
│  - Corrections → update semantic memory                  │
│  - Positive outcomes → boost confidence                  │
│  - Negative outcomes → reduce confidence, add counterex  │
│  - Patterns → learned instructions                       │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  5. CONSOLIDATE (periodic, every N sessions)             │
│  LLM reviews recent sessions, extracts recurring patterns │
│  Updates semantic memory with high-confidence insights   │
│  Confidence decay for unconfirmed knowledge              │
└────────────────┬─────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────┐
│  6. ADAPT                                                 │
│  Next execution uses updated knowledge:                   │
│  - Learned instructions in prompt                        │
│  - Memory context injection                              │
│  - Confidence-weighted recommendations                   │
│  - Optimized tool selection                              │
└────────────────┬─────────────────────────────────────────┘
                 │
                 └───────> (Back to step 1)
```

---

## 3. Feedback Collection

### 3.1 Explicit Feedback

```python
class FeedbackCollector:
    """Collect and categorize user feedback."""

    FEEDBACK_TYPES = {
        "correction": "User corrects a factual error",
        "positive_outcome": "User confirms recommendation worked",
        "negative_outcome": "User reports recommendation failed",
        "missing_capability": "User points out missing analysis",
        "preference": "User indicates preference for style/format",
        "project_convention": "User teaches project-specific pattern (code assistants)",
    }

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def record_feedback(
        self,
        feedback_text: str,
        feedback_type: str,
        session_id: str,
        timestamp: str,
        context: Optional[dict] = None,
    ) -> str:
        """Store user feedback with full context."""
        feedback_id = str(uuid.uuid4())[:12]

        self.db.execute("""
            INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """, (feedback_id, timestamp, session_id, feedback_type,
              feedback_text, json.dumps(context or {})))

        self.db.commit()

        # Process immediately
        self._process_feedback(feedback_id, feedback_type, feedback_text, timestamp)

        return feedback_id

    def _process_feedback(
        self,
        feedback_id: str,
        feedback_type: str,
        feedback_text: str,
        timestamp: str,
    ) -> None:
        """Convert feedback into actionable knowledge updates."""

        if feedback_type == "correction":
            # Extract factual correction
            self._add_to_semantic_memory(
                category="factual_correction",
                pattern=feedback_text,
                confidence=1.0,
                source="user_correction",
                timestamp=timestamp,
            )

            # Add to learned instructions
            self._add_learned_instruction(
                instruction=f"[CORRECTION] {feedback_text}",
                category="correction",
                confidence=1.0,
                source="user_feedback",
                timestamp=timestamp,
            )

        elif feedback_type == "positive_outcome":
            # Boost confidence of recent recommendations
            self._boost_recommendation_confidence(session_id, feedback_text, timestamp)

        elif feedback_type == "negative_outcome":
            # Reduce confidence, add counterexample
            self._reduce_recommendation_confidence(session_id, feedback_text, timestamp)

        elif feedback_type == "project_convention":
            # Code assistant specific — learn project patterns
            self._add_learned_instruction(
                instruction=f"[PROJECT CONVENTION] {feedback_text}",
                category="domain_knowledge",
                confidence=0.9,
                source="user_feedback",
                timestamp=timestamp,
            )

        # Mark feedback as processed
        self.db.execute(
            "UPDATE feedback SET status = 'processed' WHERE id = ?",
            (feedback_id,)
        )
        self.db.commit()
```

### 3.2 Implicit Feedback Detection

Behavioral signals indicate feedback:

```python
class ImplicitFeedbackDetector:
    """Detect feedback from user behavior."""

    def detect(self, user_input: str, conversation_history: list, timestamp: str) -> Optional[dict]:
        """Analyze user input for implicit feedback signals."""

        input_lower = user_input.lower()

        # Signal 1: Re-phrased question (answer was unclear)
        if self._is_rephrased_question(user_input, conversation_history):
            return {
                "type": "unclear_response",
                "signal": "User re-asked the same question",
                "suggested_action": "Improve clarity in future similar queries",
                "timestamp": timestamp,
            }

        # Signal 2: Request for more detail (analysis too shallow)
        detail_keywords = ["more detail", "specifically", "drill down", "elaborate",
                          "break down", "which exact", "show me the", "what about"]
        if any(kw in input_lower for kw in detail_keywords):
            return {
                "type": "needs_more_detail",
                "signal": f"User asked for more detail: '{user_input[:50]}...'",
                "suggested_action": "Increase depth of analysis for this topic",
                "timestamp": timestamp,
            }

        # Signal 3: Correction indicator
        correction_keywords = ["actually", "no that's wrong", "incorrect", "you're mistaken",
                              "that's not right", "correction:"]
        if any(kw in input_lower for kw in correction_keywords):
            return {
                "type": "correction",
                "signal": f"User is correcting a mistake: '{user_input}'",
                "suggested_action": "Extract factual correction and store in memory",
                "timestamp": timestamp,
            }

        # Signal 4: Positive confirmation (CODE ASSISTANT SPECIFIC)
        positive_keywords = ["works", "perfect", "exactly", "that's right", "correct",
                           "tests pass", "builds successfully", "deployed"]
        if any(kw in input_lower for kw in positive_keywords):
            return {
                "type": "positive_confirmation",
                "signal": "User confirmed success",
                "suggested_action": "Boost confidence of recent recommendations",
                "timestamp": timestamp,
            }

        # Signal 5: Missing capability (agent should have done something)
        missing_keywords = ["you didn't", "you forgot", "also check", "what about",
                           "don't forget", "you should have"]
        if any(kw in input_lower for kw in missing_keywords):
            return {
                "type": "missing_capability",
                "signal": f"User pointed out missing analysis: '{user_input}'",
                "suggested_action": "Add to learned instructions for future",
                "timestamp": timestamp,
            }

        return None

    def _is_rephrased_question(self, current: str, history: list) -> bool:
        """Check if current input is a rephrasing of a recent question."""
        if len(history) < 2:
            return False

        # Get last user message
        last_user_msgs = [m for m in history[-5:] if m.get("role") == "user"]
        if not last_user_msgs:
            return False

        last_question = last_user_msgs[-1]["content"]

        # Simple similarity check (could use embeddings for better detection)
        current_words = set(current.lower().split())
        last_words = set(last_question.lower().split())

        overlap = current_words & last_words
        similarity = len(overlap) / max(len(current_words), len(last_words))

        return similarity > 0.5  # >50% word overlap = likely rephrase
```

### 3.3 Automated Feedback (for CI/Pipeline Agents)

```python
class AutomatedFeedbackCollector:
    """Collect feedback from measurable outcomes (no human required)."""

    def collect_from_test_results(
        self,
        recommendation: str,
        test_results_before: dict,
        test_results_after: dict,
        timestamp: str,
    ) -> dict:
        """Generate feedback from test outcome comparison.

        For code assistants: Did the recommended change make tests pass?
        """
        tests_passed_before = test_results_before.get("passed", 0)
        tests_passed_after = test_results_after.get("passed", 0)

        improvement = tests_passed_after - tests_passed_before

        if improvement > 0:
            return {
                "type": "positive_outcome",
                "feedback": f"Recommendation '{recommendation}' improved test results: +{improvement} tests passing",
                "timestamp": timestamp,
                "measurable_impact": improvement,
            }
        elif improvement < 0:
            return {
                "type": "negative_outcome",
                "feedback": f"Recommendation '{recommendation}' broke tests: {-improvement} tests now failing",
                "timestamp": timestamp,
                "measurable_impact": improvement,
            }
        else:
            return {
                "type": "neutral_outcome",
                "feedback": f"Recommendation '{recommendation}' had no impact on tests",
                "timestamp": timestamp,
                "measurable_impact": 0,
            }

    def collect_from_performance_change(
        self,
        recommendation: str,
        metric_before: float,
        metric_after: float,
        metric_name: str,
        higher_is_better: bool,
        timestamp: str,
    ) -> dict:
        """Generate feedback from performance metric change.

        For performance agents: Did the optimization improve the metric?
        """
        change_percent = ((metric_after - metric_before) / metric_before) * 100

        improved = (
            (higher_is_better and metric_after > metric_before) or
            (not higher_is_better and metric_after < metric_before)
        )

        if improved and abs(change_percent) > 5:  # Significant improvement
            return {
                "type": "positive_outcome",
                "feedback": f"{recommendation} → {metric_name} improved {abs(change_percent):.1f}%",
                "timestamp": timestamp,
                "measurable_impact": change_percent,
            }
        elif not improved and abs(change_percent) > 5:  # Regression
            return {
                "type": "negative_outcome",
                "feedback": f"{recommendation} → {metric_name} regressed {abs(change_percent):.1f}%",
                "timestamp": timestamp,
                "measurable_impact": change_percent,
            }
        else:
            return {
                "type": "neutral_outcome",
                "feedback": f"{recommendation} → {metric_name} unchanged (Δ {change_percent:.1f}%)",
                "timestamp": timestamp,
                "measurable_impact": change_percent,
            }
```

---

## 4. Outcome Tracking

Track every recommendation with measurable results:

```python
class OutcomeTracker:
    """Track recommendation outcomes with timestamps."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def track_recommendation(
        self,
        recommendation: str,
        session_id: str,
        context: dict,
        expected_impact: str,
        timestamp: str,
    ) -> str:
        """Track a new recommendation for outcome monitoring."""
        rec_id = str(uuid.uuid4())[:12]

        self.db.execute("""
            INSERT INTO recommendations VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'unknown', 0.5, NULL)
        """, (rec_id, session_id, timestamp, recommendation,
              json.dumps(context), expected_impact, "", None))

        self.db.commit()
        return rec_id

    def record_outcome(
        self,
        rec_id: str,
        actual_outcome: str,
        outcome_type: str,
        timestamp: str,
        measurable_delta: Optional[float] = None,
    ) -> None:
        """Record actual outcome of a recommendation."""

        # Confidence adjustment (asymmetric — negatives weigh more)
        confidence_delta = 0.0
        if outcome_type == "positive":
            confidence_delta = 0.1
        elif outcome_type == "negative":
            confidence_delta = -0.2  # Negative evidence weighs 2x
        elif outcome_type == "neutral":
            confidence_delta = -0.05

        self.db.execute("""
            UPDATE recommendations
            SET actual_outcome = ?,
                outcome_type = ?,
                confidence = MAX(0.0, MIN(1.0, confidence + ?)),
                outcome_recorded_at = ?
            WHERE id = ?
        """, (actual_outcome, outcome_type, confidence_delta, timestamp, rec_id))

        self.db.commit()

        # If measurable delta provided, store for analysis
        if measurable_delta is not None:
            self.db.execute("""
                UPDATE recommendations
                SET metadata = json_set(metadata, '$.measurable_delta', ?)
                WHERE id = ?
            """, (measurable_delta, rec_id))
            self.db.commit()

    def get_accuracy_by_category(
        self,
        category: str,
        time_range: Optional[Tuple[str, str]] = None,
    ) -> dict:
        """Compute accuracy for a specific recommendation category."""
        query = """
            SELECT outcome_type, COUNT(*)
            FROM recommendations
            WHERE category = ? AND outcome_type != 'unknown'
        """
        params = [category]

        if time_range:
            query += " AND timestamp >= ? AND timestamp <= ?"
            params.extend(time_range)

        query += " GROUP BY outcome_type"

        results = dict(self.db.execute(query, params).fetchall())
        total = sum(results.values())

        return {
            "category": category,
            "total_tracked": total,
            "positive_rate": results.get("positive", 0) / max(total, 1),
            "negative_rate": results.get("negative", 0) / max(total, 1),
            "neutral_rate": results.get("neutral", 0) / max(total, 1),
            "breakdown": results,
        }

    def get_top_recommendations(
        self,
        limit: int = 10,
        min_confidence: float = 0.7,
    ) -> List[dict]:
        """Get highest-confidence recommendations based on outcomes."""
        cursor = self.db.execute("""
            SELECT recommendation, category, confidence, outcome_type, COUNT(*) as evidence
            FROM recommendations
            WHERE confidence >= ? AND outcome_type = 'positive'
            GROUP BY recommendation
            ORDER BY confidence DESC, evidence DESC
            LIMIT ?
        """, (min_confidence, limit))

        return [dict(row) for row in cursor.fetchall()]
```

---

## 5. Pattern Extraction

### 5.1 From Execution Traces

Extract patterns from what worked:

```python
class PatternExtractor:
    """Extract reusable patterns from execution history."""

    def __init__(self, universal_db: UniversalKnowledgeDB):
        self.universal_db = universal_db

    def extract_success_patterns(
        self,
        lookback_hours: int = 24,
        min_occurrences: int = 3,
    ) -> List[dict]:
        """Find patterns in successful executions.

        For code assistants: "When implementing auth, agent always uses bcrypt for hashing"
        For performance agents: "When analyzing latency, agent always checks prefill/decode split first"
        """
        cutoff = (datetime.now() - timedelta(hours=lookback_hours)).isoformat()

        # Get successful tool sequences
        cursor = self.universal_db.db.execute("""
            SELECT tool_name, args_json, result_json, timestamp
            FROM tool_calls
            WHERE success = 1 AND timestamp > ?
            ORDER BY timestamp ASC
        """, (cutoff,))

        tool_sequences = []
        current_sequence = []

        for row in cursor:
            current_sequence.append({
                "tool": row["tool_name"],
                "args": json.loads(row["args_json"]),
                "result": json.loads(row["result_json"]),
                "timestamp": row["timestamp"],
            })

            # Sequence boundary: gap > 5 minutes or state change
            if len(current_sequence) > 1:
                time_gap = (
                    datetime.fromisoformat(current_sequence[-1]["timestamp"]) -
                    datetime.fromisoformat(current_sequence[-2]["timestamp"])
                ).total_seconds()

                if time_gap > 300:  # 5 minutes
                    tool_sequences.append(current_sequence[:-1])
                    current_sequence = [current_sequence[-1]]

        # Find recurring sequences
        patterns = {}
        for seq in tool_sequences:
            seq_signature = tuple(s["tool"] for s in seq)
            if seq_signature not in patterns:
                patterns[seq_signature] = []
            patterns[seq_signature].append(seq)

        # Filter to patterns with min_occurrences
        recurring = [
            {
                "pattern": list(sig),
                "occurrences": len(seqs),
                "examples": seqs[:3],  # Sample executions
            }
            for sig, seqs in patterns.items()
            if len(seqs) >= min_occurrences
        ]

        return recurring

    def extract_code_conventions(
        self,
        project_files: List[str],
        language: str = "python",
    ) -> List[dict]:
        """Extract coding conventions from project files (code assistant specific).

        Analyzes existing code to learn:
        - Naming conventions (camelCase vs snake_case, class naming)
        - Import patterns (absolute vs relative, order)
        - Docstring style (Google vs NumPy)
        - Test patterns (pytest fixtures, naming)
        - Type hint usage
        """
        conventions = []

        if language == "python":
            # Analyze Python files
            import ast

            naming_patterns = {"functions": [], "classes": [], "variables": []}
            import_styles = []
            has_type_hints = 0
            total_functions = 0

            for file_path in project_files:
                try:
                    with open(file_path) as f:
                        tree = ast.parse(f.read())

                    for node in ast.walk(tree):
                        # Function naming
                        if isinstance(node, ast.FunctionDef):
                            naming_patterns["functions"].append(node.name)
                            total_functions += 1
                            # Check type hints
                            if node.returns or any(arg.annotation for arg in node.args.args):
                                has_type_hints += 1

                        # Class naming
                        elif isinstance(node, ast.ClassDef):
                            naming_patterns["classes"].append(node.name)

                        # Import style
                        elif isinstance(node, ast.ImportFrom):
                            import_styles.append("from X import Y")

                except:
                    pass

            # Extract conventions
            if naming_patterns["functions"]:
                snake_case_funcs = sum(1 for f in naming_patterns["functions"] if "_" in f)
                if snake_case_funcs / len(naming_patterns["functions"]) > 0.8:
                    conventions.append({
                        "pattern": "Use snake_case for function names",
                        "confidence": 0.9,
                        "evidence_count": len(naming_patterns["functions"]),
                        "category": "naming_convention",
                    })

            if total_functions > 0 and has_type_hints / total_functions > 0.7:
                conventions.append({
                    "pattern": "Add type hints to functions (>70% of codebase uses them)",
                    "confidence": 0.85,
                    "evidence_count": total_functions,
                    "category": "type_usage",
                })

        return conventions
```

---

## 6. Knowledge Consolidation

### 6.1 LLM-Powered Consolidation

```python
class KnowledgeConsolidator:
    """Distill knowledge from episodic memory."""

    def __init__(self, episodic: EpisodicMemory, semantic: SemanticMemory):
        self.episodic = episodic
        self.semantic = semantic
        self.llm_client = None  # LLM for consolidation

    def consolidate(
        self,
        lookback_sessions: int = 20,
        timestamp: str = None,
    ) -> List[dict]:
        """
        Extract patterns from recent sessions.

        Process:
        1. Load last N sessions from episodic memory
        2. Load existing semantic knowledge
        3. Send to LLM with consolidation prompt
        4. Parse extracted patterns
        5. For each pattern:
           - If new: INSERT into semantic memory
           - If exists: UPDATE confidence/evidence
           - If contradicts: CREATE conflict entry
        6. Return summary of changes
        """
        timestamp = timestamp or datetime.now().isoformat()

        # Load recent sessions (with timestamps)
        recent_sessions = self.episodic.search_structured()[:lookback_sessions]

        # Sort by recency
        recent_sessions.sort(key=lambda s: s["timestamp"], reverse=True)

        # Load existing knowledge
        existing = self.semantic.get_all(min_confidence=0.2)

        # Consolidation prompt
        consolidation_prompt = f"""Analyze these {len(recent_sessions)} recent sessions (most recent first)
and extract recurring patterns, facts, and recommendations.

EXISTING KNOWLEDGE (check each against new sessions):
{json.dumps(existing, indent=2)}

RECENT SESSIONS (newest first):
{json.dumps(recent_sessions, indent=2)}

For each pattern found:
1. Check if it updates existing knowledge (increase confidence if confirmed, decrease if contradicted)
2. If new, provide:
   - category: factual | causal | correlational | preference | project_convention
   - pattern: Description
   - recommendation: Suggested action (if applicable)
   - confidence: 0.0-1.0 based on consistency across sessions
   - evidence_count: Number of sessions supporting this
   - first_observed: Timestamp of earliest supporting session
   - last_confirmed: Timestamp of most recent confirmation

IMPORTANT: Prioritize recent sessions higher when conflicts exist.

Return as JSON array."""

        response = self.llm_client.generate(consolidation_prompt, max_tokens=4000)
        patterns = json.loads(response)

        # Process patterns
        updates = []
        for pattern in patterns:
            # Check for conflicts with existing knowledge
            conflicts = self._find_conflicts(pattern, existing)

            if conflicts:
                # Resolve based on recency and confidence
                resolution = self._resolve_conflict(pattern, conflicts[0], timestamp)
                updates.append(resolution)
            elif pattern.get("updates_existing"):
                # Update existing entry
                existing_id = pattern["updates_existing"]
                self.semantic.update(
                    entry_id=existing_id,
                    confidence=pattern["confidence"],
                    evidence_count=pattern["evidence_count"],
                    last_confirmed=pattern["last_confirmed"],
                )
                updates.append({"action": "updated", "id": existing_id})
            else:
                # New knowledge
                new_id = self.semantic.insert(pattern, timestamp)
                updates.append({"action": "inserted", "id": new_id})

        return updates

    def _resolve_conflict(
        self,
        new_pattern: dict,
        existing_pattern: dict,
        timestamp: str,
    ) -> dict:
        """Resolve conflict between new and existing knowledge.

        Strategy:
        - If new has higher confidence: supersede old
        - If old has more evidence: keep old, add new as alternative
        - If recency matters: prefer newer (last 30 days)
        """
        new_confidence = new_pattern.get("confidence", 0.5)
        old_confidence = existing_pattern.get("confidence", 0.5)

        new_time = datetime.fromisoformat(new_pattern.get("last_confirmed", timestamp))
        old_time = datetime.fromisoformat(existing_pattern.get("last_confirmed", "2020-01-01"))

        recency_diff_days = (new_time - old_time).days

        # Resolution logic
        if new_confidence > 0.9 and recency_diff_days < 30:
            # New knowledge is high-confidence and recent → supersede old
            self.semantic.delete(existing_pattern["id"], reason=f"Superseded by newer pattern (confidence {new_confidence})")
            new_id = self.semantic.insert(new_pattern, timestamp)
            return {"action": "superseded", "old_id": existing_pattern["id"], "new_id": new_id}

        elif old_confidence > 0.9 and new_confidence < 0.7:
            # Old knowledge is high-confidence → keep, ignore new
            return {"action": "kept_existing", "reason": "Old knowledge has high confidence"}

        else:
            # Both valid — create conflict for human review
            conflict_id = self.semantic.create_conflict(existing_pattern["id"], new_pattern, timestamp)
            return {"action": "conflict_created", "conflict_id": conflict_id}
```

### 6.2 Confidence Decay

Old knowledge loses confidence if not reconfirmed:

```python
def apply_confidence_decay(
    self,
    half_life_days: int = 60,
    timestamp: str = None,
) -> int:
    """
    Apply exponential decay to knowledge confidence.

    Formula: new_confidence = old_confidence * 2^(-age_days / half_life_days)

    Returns: Number of entries updated
    """
    timestamp = timestamp or datetime.now().isoformat()
    now = datetime.fromisoformat(timestamp)

    # Get all knowledge
    cursor = self.db.execute("SELECT id, last_confirmed, confidence FROM knowledge")

    updated_count = 0
    for row in cursor:
        entry_id, last_confirmed, confidence = row
        last_time = datetime.fromisoformat(last_confirmed)
        age_days = (now - last_time).days

        if age_days > 0:
            decay_factor = 2 ** (-age_days / half_life_days)
            new_confidence = confidence * decay_factor

            # Only update if significant change
            if abs(new_confidence - confidence) > 0.01:
                self.db.execute("""
                    UPDATE knowledge
                    SET confidence = ?
                    WHERE id = ?
                """, (new_confidence, entry_id))
                updated_count += 1

    self.db.commit()
    return updated_count
```

---

## 7. Self-Evaluation

Agents evaluate their own responses:

```python
class SelfEvaluator:
    """Agent self-evaluation using multiple signals."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def evaluate_response(
        self,
        user_query: str,
        agent_response: str,
        tools_used: List[str],
        expected_quality: float = 0.8,
        timestamp: str = None,
    ) -> dict:
        """
        Evaluate response quality.

        Returns scores for:
        - tool_selection_accuracy: Right tools used?
        - response_completeness: All aspects addressed?
        - actionability: Recommendations specific?
        - factual_grounding: Claims supported by tool outputs?
        - code_quality: (Code assistants) Code follows best practices?
        """
        timestamp = timestamp or datetime.now().isoformat()

        eval_prompt = f"""Evaluate this agent response (timestamp: {timestamp}).

USER QUERY: {user_query}
TOOLS USED: {tools_used}
AGENT RESPONSE: {agent_response}

Score 0.0-1.0 on:
1. tool_selection_accuracy: Were the right tools used?
2. response_completeness: All aspects of query addressed?
3. actionability: Recommendations specific and actionable?
4. factual_grounding: Claims supported by tool output data?

For code assistants, also score:
5. code_quality: Code follows best practices (type hints, tests, docstrings)?
6. architecture_consistency: Code matches project patterns?

Also provide:
- overall_quality: Weighted average (correctness 40%, completeness 30%, actionability 20%, grounding 10%)
- improvement_suggestion: One specific way to improve next time

Return as JSON."""

        eval_result = self.llm_client.generate(eval_prompt, max_tokens=500)
        scores = json.loads(eval_result)

        # Store evaluation
        self._store_evaluation(user_query, scores, timestamp)

        return scores

    def _store_evaluation(self, query: str, scores: dict, timestamp: str) -> None:
        """Store evaluation in database."""
        eval_id = str(uuid.uuid4())[:12]

        self.db.execute("""
            INSERT INTO self_evaluations VALUES (?, ?, ?, ?, ?)
        """, (eval_id, timestamp, query, json.dumps(scores),
              scores.get("improvement_suggestion", "")))

        self.db.commit()
```

---

## 8. Adaptation Mechanisms

How learning influences future behavior:

### 8.1 Prompt Adaptation

```python
def inject_learnings_into_prompt(self, context: dict, timestamp: str) -> str:
    """Retrieve and format learnings for prompt injection."""

    sections = []

    # 1. Recent corrections (last 7 days) — highest priority
    recent_cutoff = (datetime.now() - timedelta(days=7)).isoformat()
    corrections = self.semantic_memory.search(
        category="factual_correction",
        time_range=(recent_cutoff, timestamp),
    )

    if corrections:
        sections.append("RECENT CORRECTIONS:")
        for c in corrections[:5]:
            sections.append(f"  - {c['pattern']} (confirmed {c['last_confirmed']})")

    # 2. Project conventions (for code assistants)
    if context.get("task_type") == "coding":
        conventions = self.semantic_memory.search(
            category="project_convention",
            min_confidence=0.7,
        )
        if conventions:
            sections.append("\nPROJECT CONVENTIONS:")
            for conv in conventions[:5]:
                sections.append(f"  - {conv['pattern']}")

    # 3. High-confidence recommendations
    top_recs = self.outcome_tracker.get_top_recommendations(limit=3)
    if top_recs:
        sections.append("\nPROVEN RECOMMENDATIONS:")
        for rec in top_recs:
            sections.append(f"  - [{rec['confidence']:.0%}] {rec['recommendation']}")

    return "\n".join(sections)
```

### 8.2 Tool Selection Optimization

Learn which tools work best for which queries:

```python
class ToolSelectionOptimizer:
    """Learn optimal tool selection from outcomes."""

    def __init__(self, universal_db: UniversalKnowledgeDB):
        self.db = universal_db

    def get_tool_effectiveness(
        self,
        query_type: str,
        time_range: Optional[Tuple[str, str]] = None,
    ) -> Dict[str, float]:
        """
        Compute effectiveness score for each tool on a query type.

        Effectiveness = (success_count / total_uses) * usage_frequency
        """
        # Get tool usage for this query type
        query = """
            SELECT tool_name,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                   COUNT(*) as total
            FROM tool_calls
            WHERE context LIKE ?
        """
        params = [f"%{query_type}%"]

        if time_range:
            query += " AND timestamp >= ? AND timestamp <= ?"
            params.extend(time_range)

        query += " GROUP BY tool_name"

        cursor = self.db.db.execute(query, params)

        effectiveness = {}
        for row in cursor:
            tool_name, successes, total = row
            success_rate = successes / total
            usage_freq = total / 100  # Normalize
            effectiveness[tool_name] = success_rate * usage_freq

        return effectiveness

    def suggest_tools_for_query(
        self,
        user_query: str,
        top_k: int = 3,
    ) -> List[str]:
        """Suggest which tools to use based on learned effectiveness."""
        query_type = self._classify_query_type(user_query)
        effectiveness = self.get_tool_effectiveness(query_type)

        # Sort by effectiveness
        sorted_tools = sorted(
            effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [tool for tool, score in sorted_tools[:top_k]]
```

---

## 9. Code Assistant Specific Learnings

### 9.1 Learning Project Architecture

```python
class ProjectArchitectureLearner:
    """Learn project patterns for code assistants."""

    def learn_from_codebase(
        self,
        project_files: List[str],
        timestamp: str,
    ) -> List[dict]:
        """
        Analyze existing codebase to extract patterns.

        Learns:
        - File organization (where models go, where tests go)
        - Import patterns (absolute vs relative)
        - Naming conventions
        - Framework usage (FastAPI patterns, React patterns)
        - Testing patterns
        - Architecture patterns (MVC, microservices, etc.)
        """
        learnings = []

        # File organization
        org_pattern = self._detect_organization_pattern(project_files)
        if org_pattern:
            learnings.append({
                "category": "file_organization",
                "pattern": org_pattern,
                "confidence": 0.9,
                "timestamp": timestamp,
            })

        # Framework detection
        frameworks = self._detect_frameworks(project_files)
        for fw in frameworks:
            learnings.append({
                "category": "framework_usage",
                "pattern": f"Project uses {fw['name']} framework with {fw['pattern']} pattern",
                "confidence": 0.85,
                "timestamp": timestamp,
            })

        # Naming conventions
        conventions = self._extract_naming_conventions(project_files)
        learnings.extend(conventions)

        return learnings

    def _detect_organization_pattern(self, files: List[str]) -> Optional[str]:
        """Detect project structure pattern."""
        # Check common patterns
        has_src = any("src/" in f for f in files)
        has_tests_separate = any("tests/" in f for f in files)
        has_models_dir = any("/models/" in f for f in files)
        has_api_dir = any("/api/" in f for f in files)

        if has_src and has_tests_separate:
            if has_models_dir and has_api_dir:
                return "Layered architecture: src/ (models, api, services), tests/ separate"
            return "Source-test separation: src/ for code, tests/ for tests"

        return None
```

---

## 10. LearningLoopMixin

Complete mixin for any agent:

```python
class LearningLoopMixin:
    """Mixin providing learning and adaptation capabilities."""

    def register_learning_tools(self) -> None:
        """Register all learning tools."""

        @tool(name="record_feedback")
        def record_feedback(
            feedback: str,
            feedback_type: str = "general",
        ) -> Dict[str, Any]:
            """Record user feedback to improve future responses.

            Args:
                feedback: The feedback content
                feedback_type: 'correction', 'positive_outcome', 'negative_outcome',
                              'missing_capability', 'preference', 'project_convention'
            """
            timestamp = datetime.now().isoformat()

            feedback_id = self.feedback_collector.record_feedback(
                feedback_text=feedback,
                feedback_type=feedback_type,
                session_id=self._session_id,
                timestamp=timestamp,
                context={"recent_tools": self._get_recent_tool_calls(5)},
            )

            return {
                "status": "success",
                "feedback_id": feedback_id,
                "message": f"Feedback recorded. This will improve future {feedback_type} handling.",
            }

        @tool(atomic=True, name="get_learning_stats")
        def get_learning_stats(time_range: str = "last_30_days") -> Dict[str, Any]:
            """View learning progress and accuracy metrics.

            Shows how the agent has improved over time.
            """
            # Parse time range
            start_time, end_time = self._parse_time_range(time_range)

            # Get recommendation accuracy by category
            categories = ["code_suggestion", "architecture_decision", "debugging", "optimization"]
            accuracy_by_category = {}

            for cat in categories:
                accuracy = self.outcome_tracker.get_accuracy_by_category(
                    category=cat,
                    time_range=(start_time, end_time),
                )
                accuracy_by_category[cat] = accuracy

            # Get knowledge growth
            knowledge_growth = self.semantic_memory.get_growth_over_time(start_time, end_time)

            # Get pattern extraction stats
            patterns_extracted = len(self.pattern_extractor.extract_success_patterns(
                lookback_hours=(datetime.now() - datetime.fromisoformat(start_time)).total_seconds() / 3600
            ))

            return {
                "status": "success",
                "time_range": time_range,
                "recommendation_accuracy": accuracy_by_category,
                "knowledge_entries_added": knowledge_growth,
                "success_patterns_extracted": patterns_extracted,
                "overall_learning_trend": "improving" if knowledge_growth > 0 else "stable",
            }

        @tool(name="consolidate_knowledge")
        def consolidate_knowledge(lookback_sessions: int = 20) -> Dict[str, Any]:
            """Trigger knowledge consolidation manually.

            Normally runs automatically every N sessions, but can be triggered
            when you want to distill learnings immediately.
            """
            timestamp = datetime.now().isoformat()

            updates = self.knowledge_consolidator.consolidate(
                lookback_sessions=lookback_sessions,
                timestamp=timestamp,
            )

            # Apply confidence decay
            decayed = self.semantic_memory.apply_confidence_decay(
                half_life_days=60,
                timestamp=timestamp,
            )

            return {
                "status": "success",
                "sessions_analyzed": lookback_sessions,
                "knowledge_updates": len(updates),
                "entries_decayed": decayed,
                "updates": updates[:5],  # Sample
            }
```

---

## 11. Continuous Execution Optimization

For agents running until task complete:

### 11.1 In-Session Learning

Learn and adapt **during** execution, not just between sessions:

```python
class InSessionLearner:
    """Learn from execution in real-time."""

    def __init__(self, agent):
        self.agent = agent
        self.session_learnings = []

    def learn_from_test_failure(
        self,
        test_name: str,
        error_message: str,
        fix_applied: str,
        timestamp: str,
    ) -> None:
        """Learn from test failure → fix → success pattern."""

        # Extract pattern
        pattern = {
            "category": "debugging_pattern",
            "pattern": f"Test '{test_name}' fails with '{error_message[:50]}...' → fix: {fix_applied}",
            "confidence": 0.6,  # Initial confidence
            "evidence_count": 1,
            "timestamp": timestamp,
        }

        self.session_learnings.append(pattern)

        # If seen again in same session, boost confidence
        similar = [p for p in self.session_learnings
                   if p.get("pattern", "").startswith(f"Test '{test_name}'")]

        if len(similar) > 1:
            # Same test failed multiple times — high-confidence pattern
            pattern["confidence"] = 0.85
            pattern["evidence_count"] = len(similar)

            # Store immediately (don't wait for session end)
            self.agent.semantic_memory.insert(pattern, timestamp)

    def learn_from_code_pattern(
        self,
        file_path: str,
        code_snippet: str,
        pattern_type: str,
        timestamp: str,
    ) -> None:
        """Learn coding patterns from files being written.

        Example: After writing 5 API endpoints, learn "API endpoints follow this structure"
        """
        self.session_learnings.append({
            "category": "code_pattern",
            "pattern": f"{pattern_type} pattern: {code_snippet[:100]}",
            "example_file": file_path,
            "confidence": 0.5,
            "timestamp": timestamp,
        })

    def finalize_session_learnings(self, timestamp: str) -> None:
        """At session end, consolidate in-session learnings."""
        # Group by pattern type
        grouped = {}
        for learning in self.session_learnings:
            key = (learning["category"], learning["pattern"][:50])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(learning)

        # Save patterns that occurred multiple times
        for (cat, pat), instances in grouped.items():
            if len(instances) >= 2:  # Seen 2+ times in session → significant
                self.agent.semantic_memory.insert({
                    "category": cat,
                    "pattern": instances[0]["pattern"],
                    "confidence": min(0.5 + (len(instances) * 0.1), 0.9),
                    "evidence_count": len(instances),
                    "source": "in_session_learning",
                }, timestamp)
```

---

## 12. Code Assistant Learning Examples

### Example 1: Learning Test Patterns

```python
# Over 3 sessions, agent writes 50 test files
# Pattern extractor notices:

patterns = [
    {
        "category": "test_pattern",
        "pattern": "Tests use @pytest.fixture for test data setup",
        "confidence": 0.95,
        "evidence_count": 48,  # 48/50 files use this
        "recommendation": "Always use fixtures for test data, avoid setUp() methods",
    },
    {
        "category": "test_naming",
        "pattern": "Test files named test_*.py, test functions named test_*",
        "confidence": 1.0,
        "evidence_count": 50,
    },
    {
        "category": "test_organization",
        "pattern": "Test files mirror source structure (tests/api/test_users.py for src/api/users.py)",
        "confidence": 0.92,
        "evidence_count": 45,
    },
]

# Next test: Agent automatically follows learned patterns
```

### Example 2: Learning From Failures

```python
# Session 1: Agent suggests using subprocess for file operations
# → User: "Don't use subprocess, use pathlib"
# → Feedback: correction

# Session 2: Agent about to use subprocess
# → Memory retrieves: "Don't use subprocess, use pathlib (confidence: 1.0)"
# → Agent uses pathlib instead
# → Outcome: Success

# Session 5: Consolidation
# → Pattern: "For file operations, prefer pathlib over subprocess"
# → Confidence: 1.0 (never contradicted)
# → Added to learned instructions (permanent)
```

---

## 13. Integration with All Frameworks

Learning loop depends on and enhances other frameworks:

| Framework | Integration Point |
|-----------|------------------|
| **Persistent Memory** | Stores feedback, outcomes, patterns with timestamps |
| **Adaptive Prompts** | Learned instructions injected into prompt, state-specific learning |
| **Dynamic Tools** | Pattern detection triggers tool/skill creation |
| **Manifest** | Track which recommendations affected which files/goals |
| **Continuous Execution** | In-session learning, checkpoint includes learnings |

---

## 14. Configuration

```python
@dataclass
class LearningConfig:
    """Configuration for learning system."""

    # Feedback
    enable_explicit_feedback: bool = True
    enable_implicit_feedback: bool = True
    enable_automated_feedback: bool = True

    # Outcome tracking
    track_recommendations: bool = True
    confidence_boost_positive: float = 0.1
    confidence_penalty_negative: float = 0.2  # Asymmetric

    # Consolidation
    consolidation_frequency_sessions: int = 10
    consolidation_lookback: int = 20
    enable_auto_consolidation: bool = True

    # Pattern extraction
    enable_pattern_detection: bool = True
    min_pattern_occurrences: int = 3
    pattern_extraction_interval_hours: int = 24

    # Knowledge decay
    enable_confidence_decay: bool = True
    knowledge_half_life_days: int = 60

    # Self-evaluation
    enable_self_evaluation: bool = True
    self_eval_frequency: str = "every_session"  # or "every_N_sessions"

    # Code assistant specific
    learn_project_conventions: bool = True
    learn_from_test_outcomes: bool = True
    in_session_learning: bool = True
```

---

## 15. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Hallucinated patterns** | Require evidence_count >= 3, human review queue |
| **Feedback gaming** | Cap confidence at 0.95, require diverse evidence |
| **Over-adaptation** | Immutable core prompt, periodic generality check |
| **Catastrophic forgetting** | Confidence decay (not deletion), conflict detection |
| **Cold start** | Bootstrap from documentation, expert-seeded knowledge |
| **Evaluation bias** | Cross-validate with user ratings, track calibration |
| **Expensive consolidation** | Triggered only every N sessions, use lightweight LLM |

---

*Learning & Adaptation Framework for Gaia Agent SDK.*
*Enables continuous improvement through feedback collection, outcome tracking, pattern extraction, and knowledge consolidation — optimized for code assistants and continuous execution.*
