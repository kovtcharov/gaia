# Dynamic Tools & Skills Framework for Gaia Agent SDK

**Date**: February 6, 2026
**Version**: 2.0
**Scope**: Runtime tool creation, validation, SKILLS system, and tool-building sub-agent
**Foundation**: AMD Gaia Agent SDK 0.15.3+

---

## 1. Problem Statement

Gaia agents have a **fixed set of tools** defined at build time. But agents encounter patterns that no single tool covers:

**Code Agent**: "Run pytest, collect failures, analyze each failure, generate fixes, re-run tests, repeat until all pass" — this is 5-10 tool calls per test failure. The agent does this hundreds of times.

**Performance Agent**: "For each GEMM kernel, compute arithmetic intensity, classify as compute/memory bound, compare to baseline" — recurring compound analysis.

**Research Agent**: "Search Google Scholar, filter to papers after 2024, extract abstracts, summarize key findings" — multi-step retrieval pattern.

### Your Requirements

1. **Tool-Building Sub-Agent**: A dedicated CodeAgent that builds tools for the main agent
2. **SKILLS System**: Agent-created "skills" (complex multi-step workflows) stored as vectorized, retrievable entities (NOT markdown files)
3. **Continuous execution**: Tool building shouldn't interrupt the main task — delegate to specialist

---

## 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│               Main Agent (e.g., CodeAgent)                │
│                                                           │
│  Detects recurring pattern: "I've done this 5 times"    │
└────────────────────┬─────────────────────────────────────┘
                     │ Delegates
                     ▼
┌──────────────────────────────────────────────────────────┐
│            ToolBuilderAgent (CodeAgent specialist)        │
│                                                           │
│  1. Generate tool code from pattern description          │
│  2. Write unit tests for the tool                        │
│  3. Run tests in sandbox                                 │
│  4. Debug failures                                       │
│  5. Add docstrings, type hints                           │
│  6. Validate security (import allowlist)                 │
│  7. Return validated .py file                            │
└────────────────────┬─────────────────────────────────────┘
                     │ Returns tool
                     ▼
┌──────────────────────────────────────────────────────────┐
│          DynamicToolManager (Tool Registry)               │
│                                                           │
│  1. Store tool file in .gaia/custom_tools/               │
│  2. Register in _TOOL_REGISTRY                           │
│  3. Track quality metrics (usage, success rate)          │
│  4. Embed tool description in FAISS (for SKILLS search)  │
└────────────────────┬─────────────────────────────────────┘
                     │ Tool now available
                     ▼
┌──────────────────────────────────────────────────────────┐
│               Main Agent (next invocation)                │
│                                                           │
│  Custom tool auto-loaded at startup                      │
│  Agent can now use tool in single call                   │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Tool-Building Sub-Agent

### 3.1 ToolBuilderAgent Design

Specialized CodeAgent for generating tools:

```python
class ToolBuilderAgent(CodeAgent):
    """Specialist agent that builds tools for other agents.

    Inherits from CodeAgent (has all code generation capabilities),
    specialized for tool creation with validation and testing workflows.
    """

    def __init__(self, config: Optional[ToolBuilderConfig] = None):
        if config is None:
            config = ToolBuilderConfig()

        # CodeAgent config optimized for tool building
        code_config = {
            "max_steps": 100,  # Tool building may need many iterations
            "language": "python",
            "enable_tests": True,
            "enable_lint": True,
            "enable_type_check": True,
        }

        super().__init__(**code_config)

        self.security_validator = SecurityValidator(config.import_allowlist)
        self.sandbox_executor = SandboxExecutor(config.sandbox_timeout)

    def build_tool_from_spec(
        self,
        name: str,
        description: str,
        example_usage: str,
        required_capabilities: List[str],
        target_agent_type: str = "generic",
    ) -> ToolBuildResult:
        """
        Build a complete, tested tool from specification.

        Process:
        1. Generate tool code based on spec (using CodeAgent's code generation)
        2. Generate unit tests
        3. Run tests in sandbox
        4. If tests fail: analyze failures, fix code, re-run (continuous loop)
        5. Validate security (import allowlist, no blocked patterns)
        6. Add comprehensive docstrings
        7. Add type hints
        8. Format with black
        9. Return validated tool

        Args:
            name: Tool name (snake_case)
            description: What the tool does
            example_usage: Example of how to call the tool
            required_capabilities: What the tool needs (e.g., ["pandas", "numpy"])
            target_agent_type: What agent will use this (for context)

        Returns:
            ToolBuildResult with code, tests, validation status
        """
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # PHASE 1: Code Generation
        spec_prompt = f"""Generate a Python tool function for a Gaia agent.

TOOL NAME: {name}
DESCRIPTION: {description}
EXAMPLE USAGE: {example_usage}
REQUIRED CAPABILITIES: {required_capabilities}
TARGET AGENT: {target_agent_type}

Requirements:
- Must use @tool decorator from gaia.agents.base.tools
- Must return Dict[str, Any] with at minimum a 'status' key
- Must handle exceptions internally (return error dict, don't raise)
- Parameters must have type hints
- Must have comprehensive docstring
- If imports are needed, only use: {self.security_validator.allowlist}

Generate the complete tool code."""

        code_result = self.process_query(spec_prompt, max_steps=10)
        tool_code = self._extract_code_from_response(code_result["final_answer"])

        # PHASE 2: Test Generation
        test_prompt = f"""Generate comprehensive unit tests for this tool.

TOOL CODE:
```python
{tool_code}
```

Requirements:
- Use pytest
- Cover happy path, edge cases, error cases
- Use parametrize for multiple test cases
- Mock any external dependencies

Generate complete test file."""

        test_result = self.process_query(test_prompt, max_steps=10)
        test_code = self._extract_code_from_response(test_result["final_answer"])

        # PHASE 3: Validation Loop (continuous until tests pass)
        iteration = 0
        max_iterations = 10

        while iteration < max_iterations:
            iteration += 1

            # Security validation
            security_check = self.security_validator.validate(tool_code)
            if not security_check["safe"]:
                # Fix security issues
                fix_prompt = f"""This tool has security issues: {security_check['reason']}

TOOL CODE:
```python
{tool_code}
```

Fix the security issues without changing functionality."""

                fix_result = self.process_query(fix_prompt, max_steps=5)
                tool_code = self._extract_code_from_response(fix_result["final_answer"])
                continue

            # Run tests in sandbox
            test_result = self.sandbox_executor.run_tests(
                tool_code=tool_code,
                test_code=test_code,
                timeout_sec=60,
            )

            if test_result["all_passed"]:
                # Success!
                break

            # Tests failed — debug and fix
            debug_prompt = f"""Tests failed. Analyze and fix.

TOOL CODE:
```python
{tool_code}
```

TEST CODE:
```python
{test_code}
```

TEST RESULTS:
{test_result['output']}

Fix the tool code to pass all tests."""

            fix_result = self.process_query(debug_prompt, max_steps=10)
            tool_code = self._extract_code_from_response(fix_result["final_answer"])

        # PHASE 4: Polish
        if iteration >= max_iterations:
            return ToolBuildResult(
                success=False,
                error=f"Failed to pass tests after {max_iterations} iterations",
            )

        # Format and finalize
        formatted_code = self._format_code(tool_code)

        return ToolBuildResult(
            success=True,
            tool_name=name,
            tool_code=formatted_code,
            test_code=test_code,
            validation_passed=True,
            timestamp=timestamp,
            iterations=iteration,
        )

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code block from agent response."""
        import re
        # Match ```python ... ``` blocks
        match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if match:
            return match.group(1)
        return response  # Fallback
```

---

## 4. SKILLS System (Vectorized)

**Your requirement**: "Agent should be able to create its own skills. Instead of storing them as markdown, all skills, similar to system prompts above, are stored in a vectorized memory for retrieval in the future."

### 4.1 Skill vs. Tool

**Tool** = Single-step operation (e.g., `analyze_trace`, `write_file`)
**Skill** = Multi-step workflow (orchestrated sequence of tool calls + reasoning)

Example skills:
- `/optimize-bundle` — Analyze bundle size → identify large deps → suggest alternatives → apply changes → verify reduction
- `/debug-test-failure` — Run test → analyze failure → inspect code → form hypothesis → fix → re-run
- `/refactor-for-type-safety` — Analyze code → add type hints → run type checker → fix errors → verify

### 4.2 Skill Data Model

```python
@dataclass
class Skill:
    """Agent-created reusable skill."""

    skill_id: str                      # "optimize-bundle-size"
    name: str                          # "Bundle Size Optimizer"
    description: str                   # What it does
    category: str                      # "optimization", "debugging", "analysis", "refactoring"

    # Invocation
    trigger_patterns: List[str]        # ["optimize bundle", "reduce bundle size", "bundle too large"]
    trigger_embedding: np.ndarray      # For semantic matching

    # Implementation
    workflow_steps: List[WorkflowStep]  # Sequence of tool calls + reasoning
    required_tools: List[str]          # Tools this skill needs
    parameters: Dict[str, Any]         # User-providable parameters

    # Prompt
    skill_prompt_addition: str         # Additional context added to prompt when skill active

    # Quality
    success_rate: float                # % of successful executions
    avg_duration_minutes: float
    usage_count: int
    last_used_at: str

    # Metadata
    created_at: str
    created_by: str                    # "agent" or "user"
    example_invocations: List[str]     # Example usages
    tags: List[str]

    # Persistence
    skill_version: int
    embedding_model: str               # Model used for trigger_embedding
```

```python
@dataclass
class WorkflowStep:
    """One step in a skill workflow."""

    step_number: int
    description: str                   # What this step does
    tool_call: str                     # Tool name
    tool_args_template: Dict[str, Any]  # Args with placeholders like ${user_input}, ${prev_result.field}
    success_condition: str             # How to know step succeeded
    failure_recovery: str              # What to do if step fails
```

### 4.3 Skill Storage (Database + Vector Store)

```sql
-- Skills table
CREATE TABLE skills (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    category TEXT,
    workflow_steps TEXT NOT NULL,     -- JSON array of WorkflowStep
    required_tools TEXT,              -- JSON array
    parameters_schema TEXT,           -- JSON schema
    skill_prompt_addition TEXT,
    success_rate REAL DEFAULT 0.0,
    avg_duration_minutes REAL,
    usage_count INTEGER DEFAULT 0,
    last_used_at TEXT,
    created_at TEXT NOT NULL,
    created_by TEXT,
    example_invocations TEXT,         -- JSON array
    tags TEXT,                        -- JSON array
    version INTEGER DEFAULT 1,
    embedding_model TEXT,
    metadata_json TEXT
);

CREATE INDEX idx_skills_category ON skills(category);
CREATE INDEX idx_skills_success ON skills(success_rate DESC);
CREATE INDEX idx_skills_usage ON skills(usage_count DESC);

-- Skill trigger patterns (for exact matching)
CREATE TABLE skill_triggers (
    skill_id TEXT NOT NULL,
    trigger_pattern TEXT NOT NULL,
    embedding BLOB,                   -- Numpy array pickled

    PRIMARY KEY (skill_id, trigger_pattern),
    FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
);

-- Skill execution history
CREATE TABLE skill_executions (
    execution_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    user_input TEXT,
    parameters_json TEXT,
    steps_completed INTEGER,
    success BOOLEAN,
    duration_minutes REAL,
    error TEXT,

    FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
);

CREATE INDEX idx_skill_exec_timestamp ON skill_executions(timestamp DESC);
```

**Vector index** for semantic matching:

```python
class SkillVectorStore:
    """FAISS-based semantic search over skills."""

    def __init__(self, embedding_model: str = "nomic-embed-text-v2-moe-GGUF"):
        self.embedding_model = embedding_model
        self.embedder = SentenceTransformer(embedding_model)  # Or Lemonade
        self.index: Optional[faiss.Index] = None
        self.skill_ids: List[str] = []

    def index_skill(self, skill: Skill) -> None:
        """Add skill to vector index."""
        # Combine triggers + description for embedding
        text = f"{skill.description} {' '.join(skill.trigger_patterns)}"
        embedding = self.embedder.encode([text])[0]

        # Add to FAISS
        if self.index is None:
            dimension = len(embedding)
            self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embedding.reshape(1, -1))
        self.skill_ids.append(skill.skill_id)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Semantic search for matching skills."""
        query_embedding = self.embedder.encode([query])[0]

        distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            skill_id = self.skill_ids[idx]
            similarity = 1 / (1 + dist)  # Convert distance to similarity
            results.append((skill_id, similarity))

        return results
```

---

## 3. Skill Creation Workflow

### 3.1 Pattern Detection

The main agent detects when to create a skill:

```python
class SkillPatternDetector:
    """Detect recurring patterns that should become skills."""

    def __init__(self):
        self.tool_call_history: List[List[str]] = []  # Recent tool sequences

    def analyze_tool_sequence(self, recent_calls: List[str]) -> Optional[dict]:
        """Detect if recent tool calls match a recurring pattern.

        Returns pattern if found, else None.
        """
        # Last 5 tool calls
        current_sequence = recent_calls[-5:]

        # Check history for similar sequences
        similar_count = 0
        for past_sequence in self.tool_call_history[-20:]:  # Last 20 sequences
            if self._sequences_similar(current_sequence, past_sequence):
                similar_count += 1

        # If seen 3+ times, it's a pattern
        if similar_count >= 3:
            return {
                "pattern": current_sequence,
                "frequency": similar_count,
                "suggest_skill_creation": True,
            }

        return None

    def _sequences_similar(self, seq1: List[str], seq2: List[str], threshold: float = 0.6) -> bool:
        """Check if two tool sequences are similar (allow some variation)."""
        if not seq1 or not seq2:
            return False

        # Longest common subsequence
        lcs_length = self._lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))

        return (lcs_length / max_len) >= threshold

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Longest common subsequence length (dynamic programming)."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]
```

### 3.2 Skill Creation via Sub-Agent

When pattern detected:

```python
class SkillCreationOrchestrator:
    """Coordinates skill creation via ToolBuilderAgent."""

    def __init__(self):
        self.builder_agent = ToolBuilderAgent()
        self.skill_store = SkillStore()
        self.vector_store = SkillVectorStore()

    def create_skill_from_pattern(
        self,
        pattern: List[str],  # Sequence of tool names
        user_query: str,     # Original query that triggered pattern
        context: dict,       # Conversation context
    ) -> Skill:
        """
        Create a skill from a detected pattern.

        Steps:
        1. Analyze pattern to understand what it does
        2. Generate skill specification
        3. Delegate to ToolBuilderAgent to implement
        4. Validate and test
        5. Store skill in database + vector index
        6. Return skill definition
        """
        timestamp = datetime.now().isoformat()

        # Step 1: Understand pattern
        analysis_prompt = f"""Analyze this recurring tool call pattern:

PATTERN (last 5 tool calls):
{pattern}

ORIGINAL USER QUERY:
{user_query}

CONTEXT:
{json.dumps(context, indent=2)}

Describe:
1. What is this pattern accomplishing?
2. What should the skill be called?
3. What parameters should it take?
4. What are the workflow steps?

Return as JSON."""

        analysis = self.builder_agent.process_query(analysis_prompt, max_steps=5)
        spec = json.loads(analysis["final_answer"])

        # Step 2: Build the skill implementation
        skill_code_result = self.builder_agent.build_tool_from_spec(
            name=spec["skill_name"],
            description=spec["description"],
            example_usage=user_query,
            required_capabilities=spec.get("required_tools", []),
        )

        if not skill_code_result.success:
            raise ValueError(f"Skill creation failed: {skill_code_result.error}")

        # Step 3: Convert to Skill object
        skill = Skill(
            skill_id=spec["skill_name"],
            name=spec.get("display_name", spec["skill_name"]),
            description=spec["description"],
            category=spec.get("category", "custom"),
            trigger_patterns=spec.get("trigger_patterns", [user_query]),
            trigger_embedding=np.array([]),  # Will be computed below
            workflow_steps=[],  # Encapsulated in the generated tool
            required_tools=pattern,  # Original tools used
            parameters=spec.get("parameters", {}),
            skill_prompt_addition="",
            success_rate=0.0,
            avg_duration_minutes=0.0,
            usage_count=0,
            last_used_at=timestamp,
            created_at=timestamp,
            created_by="agent",
            example_invocations=[user_query],
            tags=spec.get("tags", []),
            skill_version=1,
            embedding_model="nomic-embed-text-v2-moe-GGUF",
        )

        # Step 4: Store in database
        self.skill_store.save_skill(skill, skill_code_result.tool_code)

        # Step 5: Index in vector store
        self.vector_store.index_skill(skill)

        logging.info(f"Created skill: {skill.skill_id} (from pattern: {pattern})")
        return skill
```

---

## 4. Skill Invocation

### 4.1 Semantic Skill Matching

User query → semantic search → find matching skill → execute:

```python
class SkillMatcher:
    """Match user queries to skills."""

    def __init__(self, skill_store: SkillStore, vector_store: SkillVectorStore):
        self.skill_store = skill_store
        self.vector_store = vector_store

    def find_matching_skill(
        self,
        user_query: str,
        min_similarity: float = 0.7,
    ) -> Optional[Skill]:
        """Find skill that matches user query."""

        # 1. Exact trigger match (fastest)
        all_skills = self.skill_store.get_all_skills()
        for skill in all_skills:
            for trigger in skill.trigger_patterns:
                if trigger.lower() in user_query.lower():
                    return skill

        # 2. Semantic search (FAISS)
        matches = self.vector_store.search(user_query, top_k=3)

        if matches and matches[0][1] >= min_similarity:
            skill_id = matches[0][0]
            return self.skill_store.get_skill(skill_id)

        return None

    def execute_skill(
        self,
        skill: Skill,
        user_input: str,
        agent: Agent,
        parameters: Optional[dict] = None,
    ) -> dict:
        """Execute a skill's workflow."""
        session_id = str(uuid.uuid4())[:8]
        start_time = datetime.now()
        timestamp = start_time.isoformat()

        # Load skill code
        skill_tool_code = self.skill_store.get_skill_code(skill.skill_id)

        # Register as temporary tool
        exec(skill_tool_code, {"__name__": f"skill_{skill.skill_id}"})

        # Execute
        try:
            result = agent._execute_tool(
                tool_name=skill.skill_id,
                tool_args=parameters or {},
            )

            duration = (datetime.now() - start_time).total_seconds() / 60
            success = result.get("status") == "success"

            # Record execution
            self.skill_store.record_execution(
                skill_id=skill.skill_id,
                timestamp=timestamp,
                user_input=user_input,
                parameters=parameters,
                success=success,
                duration=duration,
            )

            return result

        except Exception as e:
            # Record failure
            self.skill_store.record_execution(
                skill_id=skill.skill_id,
                timestamp=timestamp,
                user_input=user_input,
                parameters=parameters,
                success=False,
                duration=0,
                error=str(e),
            )
            raise
```

### 4.2 Skill Storage

```python
class SkillStore:
    """Persistent storage for skills."""

    def __init__(self, db_path: str = ".gaia/skills/skills.db",
                 code_dir: str = ".gaia/skills/code"):
        self.db = sqlite3.connect(db_path)
        self.code_dir = code_dir
        os.makedirs(code_dir, exist_ok=True)

    def save_skill(self, skill: Skill, skill_code: str) -> None:
        """Save skill to database + filesystem."""
        # Save code to file
        code_path = os.path.join(self.code_dir, f"{skill.skill_id}.py")
        with open(code_path, "w") as f:
            f.write(skill_code)

        # Save metadata to DB
        self.db.execute("""
            INSERT OR REPLACE INTO skills VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (skill.skill_id, skill.name, skill.description, skill.category,
              json.dumps([s.__dict__ for s in skill.workflow_steps]),
              json.dumps(skill.required_tools), json.dumps(skill.parameters),
              skill.skill_prompt_addition, skill.success_rate,
              skill.avg_duration_minutes, skill.usage_count, skill.last_used_at,
              skill.created_at, skill.created_by, json.dumps(skill.example_invocations),
              json.dumps(skill.tags), skill.skill_version, skill.embedding_model,
              json.dumps({})))

        # Save trigger embeddings
        for trigger in skill.trigger_patterns:
            embedding_bytes = pickle.dumps(skill.trigger_embedding)
            self.db.execute("""
                INSERT OR REPLACE INTO skill_triggers VALUES (?, ?, ?)
            """, (skill.skill_id, trigger, embedding_bytes))

        self.db.commit()

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Retrieve skill by ID."""
        row = self.db.execute("SELECT * FROM skills WHERE skill_id = ?", (skill_id,)).fetchone()
        if not row:
            return None
        return self._row_to_skill(row)

    def get_skill_code(self, skill_id: str) -> str:
        """Load skill implementation code."""
        code_path = os.path.join(self.code_dir, f"{skill_id}.py")
        with open(code_path) as f:
            return f.read()

    def record_execution(
        self,
        skill_id: str,
        timestamp: str,
        user_input: str,
        parameters: Optional[dict],
        success: bool,
        duration: float,
        error: Optional[str] = None,
    ) -> None:
        """Record skill execution for quality tracking."""
        exec_id = str(uuid.uuid4())[:12]

        self.db.execute("""
            INSERT INTO skill_executions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (exec_id, skill_id, timestamp, user_input, json.dumps(parameters),
              0, success, duration, error))

        # Update skill stats
        cursor = self.db.execute(
            "SELECT success_rate, usage_count, avg_duration_minutes FROM skills WHERE skill_id = ?",
            (skill_id,)
        )
        row = cursor.fetchone()
        old_rate, old_count, old_dur = row

        new_count = old_count + 1
        new_rate = ((old_rate * old_count) + (1 if success else 0)) / new_count
        new_dur = ((old_dur * old_count) + duration) / new_count

        self.db.execute("""
            UPDATE skills
            SET success_rate = ?, usage_count = ?, avg_duration_minutes = ?, last_used_at = ?
            WHERE skill_id = ?
        """, (new_rate, new_count, new_dur, timestamp, skill_id))

        self.db.commit()
```

---

## 5. SkillToolsMixin

Agent-facing tools for skill management:

```python
class SkillToolsMixin:
    """Mixin providing skill creation and execution tools."""

    def register_skill_tools(self) -> None:
        """Register skill management tools."""

        @tool(name="create_skill")
        def create_skill(
            name: str,
            description: str,
            trigger_patterns: str,  # Comma-separated
        ) -> Dict[str, Any]:
            """Create a reusable skill from a recurring workflow.

            Use when you've performed the same multi-step analysis/task
            repeatedly and want to codify it as a single invocable skill.

            The skill will be built by a specialist ToolBuilderAgent and stored
            for future use.

            Args:
                name: Skill name (e.g., "optimize-bundle-size")
                description: What the skill does
                trigger_patterns: Phrases that invoke this (e.g., "optimize bundle, reduce bundle")
            """
            # Detect the pattern from recent tool calls
            recent_calls = self._get_recent_tool_calls(limit=10)
            pattern_info = self.pattern_detector.analyze_tool_sequence(recent_calls)

            if not pattern_info:
                return {
                    "status": "error",
                    "error": "No clear pattern detected in recent tool calls. Need at least 3 similar sequences.",
                }

            # Delegate to ToolBuilderAgent
            skill = self.skill_orchestrator.create_skill_from_pattern(
                pattern=pattern_info["pattern"],
                user_query=description,
                context={
                    "recent_conversation": self.conversation_history[-5:],
                    "trigger_patterns": [t.strip() for t in trigger_patterns.split(",")],
                },
            )

            return {
                "status": "success",
                "skill_id": skill.skill_id,
                "skill_name": skill.name,
                "trigger_patterns": skill.trigger_patterns,
                "message": f"Skill '{skill.name}' created. Invoke with any of: {trigger_patterns}",
            }

        @tool(atomic=True, name="list_skills")
        def list_skills(category: str = "") -> Dict[str, Any]:
            """List all available skills."""
            if category:
                skills = self.skill_store.get_skills_by_category(category)
            else:
                skills = self.skill_store.get_all_skills()

            return {
                "status": "success",
                "skills": [
                    {
                        "skill_id": s.skill_id,
                        "name": s.name,
                        "description": s.description,
                        "category": s.category,
                        "trigger_patterns": s.trigger_patterns,
                        "usage_count": s.usage_count,
                        "success_rate": s.success_rate,
                    }
                    for s in skills
                ],
            }

        @tool(name="invoke_skill")
        def invoke_skill(skill_id: str, parameters: str = "{}") -> Dict[str, Any]:
            """Explicitly invoke a skill by ID.

            Alternatively, skills are auto-invoked when user query matches trigger patterns.

            Args:
                skill_id: Skill identifier
                parameters: JSON string of parameters
            """
            skill = self.skill_store.get_skill(skill_id)
            if not skill:
                # Suggest similar skills
                suggestions = self.vector_store.search(skill_id, top_k=3)
                return {
                    "status": "error",
                    "error": f"Skill '{skill_id}' not found",
                    "suggestions": [s[0] for s in suggestions],
                }

            # Execute skill
            params = json.loads(parameters) if parameters else {}
            result = self.skill_matcher.execute_skill(
                skill=skill,
                user_input=f"Invoking {skill_id}",
                agent=self,
                parameters=params,
            )

            return result

        @tool(name="delete_skill")
        def delete_skill(skill_id: str, reason: str = "") -> Dict[str, Any]:
            """Remove a skill from the library.

            Use when a skill is broken, outdated, or no longer needed.
            """
            success = self.skill_store.delete_skill(skill_id, reason)

            return {
                "status": "success" if success else "error",
                "skill_id": skill_id,
                "message": "Skill deleted" if success else "Skill not found",
            }
```

---

## 6. Skill Auto-Invocation

Skills can be automatically invoked when user query matches:

```python
class SkillAwareAgent(Agent):
    """Agent that automatically uses skills when appropriate."""

    def process_query(self, user_input: str, **kwargs):
        """Override to check for skill matches before normal processing."""

        # Check if query matches a skill
        matching_skill = self.skill_matcher.find_matching_skill(user_input)

        if matching_skill:
            logging.info(f"Auto-invoking skill: {matching_skill.skill_id}")

            # Execute skill instead of normal agent loop
            result = self.skill_matcher.execute_skill(
                skill=matching_skill,
                user_input=user_input,
                agent=self,
                parameters=self._extract_parameters(user_input, matching_skill.parameters),
            )

            # Return as if normal query
            return {
                "success": True,
                "final_answer": result.get("output", str(result)),
                "skill_used": matching_skill.skill_id,
                "steps_taken": 1,  # Skill = single abstracted step
            }

        # No skill match — normal processing
        return super().process_query(user_input, **kwargs)
```

---

## 7. Security for Dynamic Tools & Skills

### 7.1 Import Allowlist

```python
class SecurityValidator:
    """Validate tool code for security."""

    # Configurable per agent type
    DEFAULT_ALLOWLIST = {
        "builtins": ["abs", "all", "any", "dict", "enumerate", "filter", "int", "len", "list", "map", "max", "min", "range", "sorted", "str", "sum", "zip"],
        "modules": ["json", "math", "statistics", "datetime", "collections", "itertools", "functools", "typing", "dataclasses"],
    }

    CODE_ASSISTANT_ALLOWLIST = {
        "modules": DEFAULT_ALLOWLIST["modules"] + ["ast", "re", "pathlib", "os.path", "difflib"],
    }

    PERFORMANCE_AGENT_ALLOWLIST = {
        "modules": DEFAULT_ALLOWLIST["modules"] + ["pandas", "numpy", "pickle"],
        "custom": ["TraceLens"],  # Agent-specific libraries
    }

    def __init__(self, allowlist: Optional[dict] = None):
        self.allowlist = allowlist or self.DEFAULT_ALLOWLIST

    def validate(self, code: str) -> dict:
        """Validate tool code safety."""
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"safe": False, "reason": f"Syntax error: {e}"}

        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                root_module = module_name.split(".")[0]

                allowed_modules = self.allowlist.get("modules", []) + self.allowlist.get("custom", [])
                if root_module not in allowed_modules:
                    return {"safe": False, "reason": f"Disallowed import: {module_name}"}

            # Check for dangerous calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec", "__import__", "compile"]:
                        return {"safe": False, "reason": f"Blocked function: {node.func.id}"}

        # Check for blocked patterns (string-based)
        blocked = ["subprocess", "os.system", "os.popen", "socket", "requests.get", "urllib.request"]
        for pattern in blocked:
            if pattern in code:
                return {"safe": False, "reason": f"Blocked pattern: {pattern}"}

        return {"safe": True}
```

---

## 8. Integration Example: Code Assistant

```python
class AdvancedCodeAgent(
    Agent,
    CodeToolsMixin,
    FileIOToolsMixin,
    StateToolsMixin,
    SkillToolsMixin,
    MemoryToolsMixin,
):
    """Code agent with states, skills, and continuous execution."""

    def __init__(self, config):
        super().__init__(config)

        # State machine
        self.state_machine = StateMachine()
        self._load_predefined_states()  # Planning, Implementation, Debug, Testing, Review

        # Skill system
        self.skill_store = SkillStore()
        self.skill_vector_store = SkillVectorStore()
        self.skill_matcher = SkillMatcher(self.skill_store, self.skill_vector_store)
        self.skill_orchestrator = SkillCreationOrchestrator()
        self.pattern_detector = SkillPatternDetector()

        # Tool builder
        self.tool_builder_agent = ToolBuilderAgent()

        # Load existing skills at startup
        self._load_skills()

    def _load_skills(self) -> None:
        """Load all persisted skills and register as tools."""
        skills = self.skill_store.get_all_skills()
        for skill in skills:
            code = self.skill_store.get_skill_code(skill.skill_id)
            # Execute to register via @tool decorator
            exec(code, {"__name__": f"skill_{skill.skill_id}"})
            logging.info(f"Loaded skill: {skill.skill_id}")

    def process_query(self, user_input: str, **kwargs):
        """Override to add skill matching + pattern detection."""

        # 1. Check for skill match
        matching_skill = self.skill_matcher.find_matching_skill(user_input)
        if matching_skill:
            return self._execute_via_skill(matching_skill, user_input)

        # 2. Normal execution
        result = super().process_query(user_input, **kwargs)

        # 3. Pattern detection (post-execution)
        recent_tools = self._get_recent_tool_calls(limit=10)
        pattern = self.pattern_detector.analyze_tool_sequence(recent_tools)

        if pattern and pattern.get("suggest_skill_creation"):
            # Suggest creating a skill
            self.conversation_history.append({
                "role": "assistant",
                "content": f"[SUGGESTION] I've detected a recurring pattern ({pattern['frequency']} times). Should I create a reusable skill for this? Use create_skill() to do so.",
            })

        return result
```

---

## 9. Configuration

```python
@dataclass
class DynamicToolConfig:
    """Configuration for dynamic tools and skills."""

    # Storage
    tools_dir: str = ".gaia/custom_tools"
    skills_dir: str = ".gaia/skills"
    skills_db_path: str = ".gaia/skills/skills.db"

    # Tool Builder
    enable_tool_builder_agent: bool = True
    tool_builder_max_iterations: int = 10
    tool_builder_timeout_minutes: int = 30

    # Security
    import_allowlist: Dict[str, List[str]] = field(default_factory=lambda: SecurityValidator.DEFAULT_ALLOWLIST)
    enable_sandbox: bool = True
    sandbox_timeout_sec: int = 60

    # Quality
    min_success_rate: float = 0.5      # Auto-deprecate below this
    max_dynamic_tools: int = 50
    max_skills: int = 100

    # Skills
    enable_skill_auto_invocation: bool = True
    skill_match_threshold: float = 0.7
    enable_pattern_detection: bool = True
    pattern_detection_frequency: int = 3  # Detect after 3 occurrences

    # Embeddings
    embedding_model: str = "nomic-embed-text-v2-moe-GGUF"
```

---

## 10. Complete Example: Building a Skill

```
User: "Analyze this React component for bundle size impact"

Agent: [Uses: analyze_bundle, find_large_deps, suggest_alternatives, estimate_reduction]
        (4-step process, takes 2 minutes)

--- User asks same type of question 2 more times ---

Agent: [Pattern detected after 3rd occurrence]
       "I've noticed I'm repeatedly analyzing bundle sizes. Should I create a
        reusable skill for this?"

User: "Yes, create it"

Agent: [Calls create_skill(
           name="analyze-bundle-impact",
           description="Analyze React component bundle size impact and suggest optimizations",
           trigger_patterns="analyze bundle, bundle size, bundle impact"
       )]

       → ToolBuilderAgent starts (in background or foreground)
       → Generates Python code that encapsulates the 4-step workflow
       → Writes tests
       → Validates in sandbox
       → Returns validated tool

Agent: "Skill 'analyze-bundle-impact' created and ready to use."

--- Next week ---

User: "Check bundle impact of this component"

Agent: [Auto-matches to analyze-bundle-impact skill via semantic search]
       [Executes skill in single step]
       "Bundle size: 145KB. Largest deps: lodash (87KB), moment (43KB).
        Recommendation: Replace with date-fns (12KB)."
```

---

*Dynamic Tools & Skills Framework for Gaia Agent SDK.*
*Enables runtime tool creation via specialist sub-agent, SKILLS system with vectorized storage, and pattern-based workflow automation.*
