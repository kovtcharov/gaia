# Adaptive Prompts & State Machine Framework for Gaia Agent SDK

**Date**: February 6, 2026
**Version**: 2.0
**Scope**: Dynamic prompt composition, execution states, and multi-mode agent operation
**Foundation**: AMD Gaia Agent SDK 0.15.3+

---

## 1. Problem Statement

Gaia's `Agent` base class builds the system prompt once via `_compose_system_prompt()` and caches it. The prompt is static for the agent's lifetime. But different tasks need different expertise:

- A code agent writing Python needs different guidance than writing TypeScript
- A performance agent analyzing latency needs different tools than analyzing throughput
- An agent in "planning mode" shouldn't have file-writing tools enabled
- An agent in "debug mode" needs debugging-specific instructions

**Your requirement**: "The agent should be able to create a specific system prompt that it can then enter as a state, execute (sometimes in multiple steps), then return from that state. The state may be retained in memory for future use."

This requires a **state machine architecture** where each state has:
- Its own system prompt
- Its own subset of tools
- Entry/exit conditions
- Ability to nest (enter sub-state, return to parent)

---

## 2. State Machine Architecture

### 2.1 Core Concepts

**State** = A named configuration of (prompt, tools, behavior)

**State transition** = Changing from one state to another

**State stack** = Nested states (e.g., Planning → Implementation → Debug → return to Implementation → return to Planning)

```
┌────────────────────────────────────────────────────────────┐
│                   Agent State Machine                       │
│                                                             │
│  State Stack: [Planning] ← [Implementation] ← [Debug]     │
│  Current State: Debug                                      │
│                                                             │
│  ┌─────────────┐  enter_state  ┌──────────────┐          │
│  │   Planning  │───────────────>│Implementation│          │
│  │   State     │<───────────────│   State      │          │
│  └─────────────┘  return_state  └──────┬───────┘          │
│                                         │ enter_state       │
│                                         ▼                    │
│                                  ┌──────────────┐          │
│                                  │    Debug     │          │
│                                  │    State     │          │
│                                  └──────────────┘          │
└────────────────────────────────────────────────────────────┘
```

### 2.2 State Definition

```python
@dataclass
class AgentState:
    """Definition of an execution state."""

    state_id: str                      # "debug_mode", "implementation_mode", etc.
    name: str                          # Human-readable name
    description: str                   # What this state is for

    # Prompt
    system_prompt_module: str          # Prompt text specific to this state
    immutable_core_preserved: bool     # Include base agent prompt (usually True)

    # Tools
    enabled_tools: Optional[List[str]] # Tool whitelist (None = all tools)
    disabled_tools: List[str]          # Tool blacklist
    tool_config: Dict[str, Any]        # State-specific tool parameters

    # Behavior
    max_steps_in_state: Optional[int]  # Step limit for this state
    auto_exit_conditions: List[str]    # Conditions that trigger exit
    entry_hook: Optional[Callable]     # Called when entering state
    exit_hook: Optional[Callable]      # Called when exiting state

    # Metadata
    created_at: str                    # ISO timestamp
    created_by: str                    # "agent", "user", "system"
    parent_state: Optional[str]        # For nested states
    usage_count: int                   # How many times entered
    avg_duration_minutes: float        # Average time spent in state

    # Persistence
    persist_to_memory: bool            # Save this state definition for reuse
    state_version: int                 # For versioning state definitions
```

### 2.3 State Storage (Database)

```sql
-- State definitions
CREATE TABLE states (
    state_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    system_prompt_module TEXT NOT NULL,
    enabled_tools TEXT,                -- JSON array
    disabled_tools TEXT,               -- JSON array
    tool_config TEXT,                  -- JSON
    max_steps_in_state INTEGER,
    auto_exit_conditions TEXT,         -- JSON array
    created_at TEXT NOT NULL,
    created_by TEXT,
    parent_state TEXT,
    usage_count INTEGER DEFAULT 0,
    avg_duration_minutes REAL,
    persist BOOLEAN DEFAULT 1,
    version INTEGER DEFAULT 1,
    metadata_json TEXT,

    FOREIGN KEY (parent_state) REFERENCES states(state_id)
);

-- State transition history
CREATE TABLE state_transitions (
    transition_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    session_id TEXT,
    from_state TEXT,
    to_state TEXT NOT NULL,
    reason TEXT,                       -- Why the transition happened
    triggered_by TEXT,                 -- 'agent_decision', 'auto_condition', 'user_command'
    stack_depth INTEGER,               -- Nesting level
    metadata_json TEXT,

    FOREIGN KEY (from_state) REFERENCES states(state_id),
    FOREIGN KEY (to_state) REFERENCES states(state_id)
);

CREATE INDEX idx_transitions_timestamp ON state_transitions(timestamp DESC);
CREATE INDEX idx_transitions_session ON state_transitions(session_id);

-- State execution log
CREATE TABLE state_executions (
    execution_id TEXT PRIMARY KEY,
    state_id TEXT NOT NULL,
    session_id TEXT,
    entered_at TEXT NOT NULL,
    exited_at TEXT,
    duration_minutes REAL,
    steps_taken INTEGER,
    tools_called TEXT,                 -- JSON array of tool names
    outcome TEXT,                      -- 'completed', 'interrupted', 'error', 'manual_exit'
    metadata_json TEXT,

    FOREIGN KEY (state_id) REFERENCES states(state_id)
);

CREATE INDEX idx_executions_state ON state_executions(state_id);
CREATE INDEX idx_executions_timestamp ON state_executions(entered_at DESC);
```

---

## 3. State Machine Implementation

```python
class StateMachine:
    """Manages agent state transitions."""

    def __init__(self, db_path: str = ".gaia/states/states.db"):
        self.db = sqlite3.connect(db_path)
        self._init_schema()

        self.current_state: Optional[str] = None
        self.state_stack: List[str] = []  # For nested states
        self.state_start_time: Optional[datetime] = None
        self.steps_in_state: int = 0

    def enter_state(
        self,
        state_id: str,
        session_id: str,
        reason: str = "",
        push_to_stack: bool = True,
    ) -> AgentState:
        """Enter a new state (optionally nested)."""
        timestamp = datetime.now().isoformat()

        # Load state definition
        state = self._load_state(state_id)
        if not state:
            raise ValueError(f"State {state_id} not found")

        # Save transition
        transition_id = str(uuid.uuid4())[:12]
        self.db.execute("""
            INSERT INTO state_transitions VALUES (?, ?, ?, ?, ?, ?, 'agent_decision', ?, ?)
        """, (transition_id, timestamp, session_id, self.current_state,
              state_id, reason, len(self.state_stack), json.dumps({})))

        # Update state stack
        if push_to_stack and self.current_state:
            self.state_stack.append(self.current_state)

        self.current_state = state_id
        self.state_start_time = datetime.now()
        self.steps_in_state = 0

        # Start execution record
        self.current_execution_id = str(uuid.uuid4())[:12]
        self.db.execute("""
            INSERT INTO state_executions (execution_id, state_id, session_id, entered_at, steps_taken)
            VALUES (?, ?, ?, ?, 0)
        """, (self.current_execution_id, state_id, session_id, timestamp))

        self.db.commit()

        # Call entry hook if defined
        if state.entry_hook:
            state.entry_hook()

        logging.info(f"Entered state: {state_id} (stack depth: {len(self.state_stack)})")
        return state

    def exit_state(
        self,
        session_id: str,
        outcome: str = "completed",
        return_value: Any = None,
    ) -> Optional[str]:
        """Exit current state, return to previous."""
        if not self.current_state:
            return None

        timestamp = datetime.now().isoformat()
        duration = (datetime.now() - self.state_start_time).total_seconds() / 60

        # Update execution record
        self.db.execute("""
            UPDATE state_executions
            SET exited_at = ?, duration_minutes = ?, steps_taken = ?, outcome = ?
            WHERE execution_id = ?
        """, (timestamp, duration, self.steps_in_state, outcome, self.current_execution_id))

        # Update state usage stats
        self._update_state_usage(self.current_state, duration)

        # Get state definition for exit hook
        state = self._load_state(self.current_state)
        if state and state.exit_hook:
            state.exit_hook(return_value)

        # Pop state stack
        previous_state = None
        if self.state_stack:
            previous_state = self.state_stack.pop()
            self.current_state = previous_state
        else:
            self.current_state = None

        # Log transition
        self.db.execute("""
            INSERT INTO state_transitions VALUES (?, ?, ?, ?, ?, 'state_exit', 'auto', ?, ?)
        """, (str(uuid.uuid4())[:12], timestamp, session_id, state.state_id,
              previous_state, len(self.state_stack), json.dumps({"outcome": outcome})))

        self.db.commit()

        logging.info(f"Exited state: {state.state_id} → {previous_state or 'None'}")
        return previous_state

    def execute_in_state(
        self,
        state_id: str,
        user_input: str,
        session_id: str,
        agent: Agent,
    ) -> dict:
        """Enter state, execute agent query, exit state."""
        # Enter state
        state = self.enter_state(state_id, session_id, reason=f"execute: {user_input[:50]}")

        # Temporarily modify agent prompt and tools
        original_prompt = agent.system_prompt
        original_tools = agent.get_available_tools()

        agent._override_system_prompt(state.system_prompt_module)
        agent._override_available_tools(state.enabled_tools)

        try:
            # Execute query in this state
            result = agent.process_query(
                user_input,
                max_steps=state.max_steps_in_state or agent.max_steps,
            )

            # Check auto-exit conditions
            if self._check_exit_conditions(state, result):
                self.exit_state(session_id, outcome="auto_exit")
            else:
                # Manual exit
                self.exit_state(session_id, outcome="completed")

            return result

        finally:
            # Restore original configuration
            agent._override_system_prompt(original_prompt)
            agent._override_available_tools(original_tools)

    def create_state(
        self,
        state_id: str,
        name: str,
        prompt_module: str,
        enabled_tools: Optional[List[str]] = None,
        created_by: str = "agent",
    ) -> AgentState:
        """Create and persist a new state definition."""
        timestamp = datetime.now().isoformat()

        state = AgentState(
            state_id=state_id,
            name=name,
            description="",
            system_prompt_module=prompt_module,
            immutable_core_preserved=True,
            enabled_tools=enabled_tools,
            disabled_tools=[],
            tool_config={},
            max_steps_in_state=None,
            auto_exit_conditions=[],
            entry_hook=None,
            exit_hook=None,
            created_at=timestamp,
            created_by=created_by,
            parent_state=None,
            usage_count=0,
            avg_duration_minutes=0.0,
            persist_to_memory=True,
            state_version=1,
        )

        # Persist to database
        self.db.execute("""
            INSERT INTO states VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0.0, 1, 1, ?)
        """, (state_id, name, "", prompt_module,
              json.dumps(enabled_tools), json.dumps([]),
              json.dumps({}), None, json.dumps([]), timestamp,
              created_by, None, json.dumps({})))

        self.db.commit()
        return state
```

---

## 4. Predefined States for Code Assistants

```python
# Built-in states for code agents
PREDEFINED_STATES = {
    "planning": AgentState(
        state_id="planning",
        name="Planning Mode",
        system_prompt_module="""You are in PLANNING mode. Your job is to create a detailed
implementation plan. You have NO file-writing tools — only analysis and reading tools.

Focus on:
- Understanding requirements
- Designing architecture
- Breaking down into tasks
- Identifying risks

Output a structured plan with phases, tasks, dependencies.""",
        enabled_tools=["read_file", "analyze_codebase", "search_files", "get_manifest"],
        disabled_tools=["write_file", "edit_file", "delete_file", "run_command"],
        max_steps_in_state=20,
        auto_exit_conditions=["plan_approved"],
    ),

    "implementation": AgentState(
        state_id="implementation",
        name="Implementation Mode",
        system_prompt_module="""You are in IMPLEMENTATION mode. Write clean, tested code.

Guidelines:
- Follow project conventions from manifest
- Write tests alongside code (TDD when possible)
- Add type hints and docstrings
- Run tests after each file
- Update manifest after file creation

If tests fail, enter DEBUG mode. If uncertain about approach, enter PLANNING mode.""",
        enabled_tools=["write_file", "edit_file", "run_tests", "update_manifest", "get_file_dependencies"],
        disabled_tools=[],
        auto_exit_conditions=["all_tests_pass"],
    ),

    "testing": AgentState(
        state_id="testing",
        name="Testing Mode",
        system_prompt_module="""You are in TESTING mode. Run all tests and verify quality.

Process:
1. Run full test suite
2. If failures: enter DEBUG mode for each failure
3. Check test coverage (must be >= 80%)
4. If coverage low: write additional tests
5. Run lint + type check
6. Exit when all quality gates pass""",
        enabled_tools=["run_tests", "check_coverage", "lint", "type_check", "enter_state"],
        disabled_tools=["write_file"],  # Can't write non-test code in testing mode
        auto_exit_conditions=["all_quality_gates_pass"],
    ),

    "debug": AgentState(
        state_id="debug",
        name="Debug Mode",
        system_prompt_module="""You are in DEBUG mode. Diagnose and fix a specific failure.

Workflow:
1. Read error message and stack trace
2. Inspect relevant code
3. Form hypothesis about root cause
4. Add logging/prints if needed
5. Make targeted fix
6. Re-run failing test
7. If pass: exit to previous state. If fail: repeat.

IMPORTANT: Make minimal changes. Don't refactor while debugging.""",
        enabled_tools=["read_file", "edit_file", "run_tests", "inspect_variables", "add_logging"],
        max_steps_in_state=10,  # Prevent infinite debug loops
        auto_exit_conditions=["test_passes"],
    ),

    "review": AgentState(
        state_id="review",
        name="Review Mode",
        system_prompt_module="""You are in REVIEW mode. Check code quality without modifying.

Checklist:
- Code follows project conventions
- No security vulnerabilities
- No performance anti-patterns
- Proper error handling
- Adequate test coverage
- Documentation is clear

Output: List of issues with severity (critical/major/minor).""",
        enabled_tools=["read_file", "analyze_code", "check_security", "get_manifest"],
        disabled_tools=["write_file", "edit_file", "delete_file"],
    ),
}
```

---

## 5. Dynamic Prompt Composition

### 5.1 Four-Layer Model

```python
class PromptComposer:
    """Composes system prompt from multiple layers."""

    def compose(
        self,
        current_state: Optional[AgentState],
        learned_instructions: str,
        memory_context: str,
        tools_description: str,
    ) -> str:
        """
        Compose full system prompt from layers.

        Layer order:
        1. Immutable Core (identity, safety, output format)
        2. State Module (task-specific expertise)
        3. Learned Instructions (accumulated wisdom)
        4. Memory Context (relevant past knowledge)
        5. Tool Descriptions (available tools)
        """
        parts = []

        # Layer 1: Immutable core (always included)
        parts.append(self.IMMUTABLE_CORE_PROMPT)

        # Layer 2: State-specific prompt
        if current_state:
            parts.append(f"--- CURRENT MODE: {current_state.name.upper()} ---")
            parts.append(current_state.system_prompt_module)

        # Layer 3: Learned instructions
        if learned_instructions:
            parts.append("--- LEARNED FROM EXPERIENCE ---")
            parts.append(learned_instructions)

        # Layer 4: Memory context
        if memory_context:
            parts.append("--- RELEVANT PAST KNOWLEDGE ---")
            parts.append(memory_context)

        # Layer 5: Tools
        parts.append("--- AVAILABLE TOOLS ---")
        parts.append(tools_description)

        return "\n\n".join(parts)

    IMMUTABLE_CORE_PROMPT = """You are an AI agent powered by the Gaia Agent SDK.

IDENTITY: You are helpful, precise, and transparent about your capabilities and limitations.

SAFETY:
- Never execute destructive operations without explicit user confirmation
- Never access files outside allowed directories
- Never make network requests to untrusted sources
- Always validate inputs before processing

OUTPUT FORMAT:
- Provide actionable, specific recommendations
- Support claims with data from tool outputs
- Explain your reasoning clearly
- If data is truncated, tell the user

MULTI-TURN: You maintain conversation history. Reference previous context when relevant."""
```

### 5.2 Gaia Integration

Override Gaia's system prompt caching:

```python
class StateAwareAgent(Agent):
    """Agent with state machine for prompt/tool switching."""

    def __init__(self, config):
        super().__init__(config)
        self.state_machine = StateMachine()
        self.prompt_composer = PromptComposer()

    @property
    def system_prompt(self) -> str:
        """Override to make prompt dynamic based on current state."""
        current_state = self.state_machine.get_current_state()
        learned = self._load_learned_instructions()
        memory = self._retrieve_memory_context()
        tools = self._format_tools_for_prompt()

        return self.prompt_composer.compose(
            current_state=current_state,
            learned_instructions=learned,
            memory_context=memory,
            tools_description=tools,
        )

    def get_available_tools(self) -> List[str]:
        """Filter tools based on current state."""
        current_state = self.state_machine.get_current_state()
        all_tools = list(_TOOL_REGISTRY.keys())

        if not current_state:
            return all_tools

        # Apply state's tool filter
        if current_state.enabled_tools is not None:
            # Whitelist mode
            return [t for t in all_tools if t in current_state.enabled_tools]
        else:
            # Blacklist mode
            return [t for t in all_tools if t not in current_state.disabled_tools]

    def _format_tools_for_prompt(self) -> str:
        """Override to only show tools available in current state."""
        available_tools = self.get_available_tools()
        tool_descriptions = []
        for tool_name in available_tools:
            tool_meta = _TOOL_REGISTRY.get(tool_name)
            if tool_meta:
                tool_descriptions.append(
                    f"- {tool_name}(...): {tool_meta['description']}"
                )
        return "\n".join(tool_descriptions)
```

---

## 6. State Tools (StateToolsMixin)

```python
class StateToolsMixin:
    """Mixin providing state management tools."""

    def register_state_tools(self) -> None:
        """Register all state management tools."""

        @tool(name="enter_state")
        def enter_state(state_id: str, reason: str = "") -> Dict[str, Any]:
            """Enter a different execution mode/state.

            Use when switching between planning, implementation, testing, debugging.

            Args:
                state_id: ID of state to enter ('planning', 'implementation', 'debug', etc.)
                reason: Why entering this state
            """
            state = self.state_machine.enter_state(
                state_id=state_id,
                session_id=self._session_id,
                reason=reason,
            )

            return {
                "status": "success",
                "current_state": state.name,
                "state_description": state.description,
                "enabled_tools": state.enabled_tools or "all",
                "max_steps": state.max_steps_in_state or "unlimited",
                "message": f"Now in {state.name}. System prompt and tools updated.",
            }

        @tool(name="return_from_state")
        def return_from_state(return_value: str = "") -> Dict[str, Any]:
            """Exit current state and return to previous state.

            Use when finished with a sub-task (e.g., debugging complete, return to implementation).

            Args:
                return_value: Optional result to pass back to parent state
            """
            previous = self.state_machine.exit_state(
                session_id=self._session_id,
                outcome="completed",
                return_value=return_value,
            )

            return {
                "status": "success",
                "previous_state": previous or "None (base state)",
                "return_value": return_value,
            }

        @tool(name="create_state")
        def create_state(
            state_id: str,
            name: str,
            prompt_module: str,
            enabled_tools: str = "",  # Comma-separated
        ) -> Dict[str, Any]:
            """Create a new custom state for future use.

            Use when you discover a recurring execution pattern that would benefit
            from a dedicated state (e.g., "database_migration_mode", "performance_optimization_mode").

            Args:
                state_id: Unique ID (snake_case)
                name: Human-readable name
                prompt_module: System prompt text for this state
                enabled_tools: Comma-separated tool names (empty = all tools)
            """
            tools = [t.strip() for t in enabled_tools.split(",")] if enabled_tools else None

            state = self.state_machine.create_state(
                state_id=state_id,
                name=name,
                prompt_module=prompt_module,
                enabled_tools=tools,
                created_by="agent",
            )

            return {
                "status": "success",
                "state_id": state_id,
                "message": f"State '{name}' created and persisted. Use enter_state('{state_id}') to activate it.",
            }

        @tool(atomic=True, name="get_current_state")
        def get_current_state() -> Dict[str, Any]:
            """Get information about the current execution state."""
            current = self.state_machine.get_current_state()
            if not current:
                return {"status": "no_state", "message": "In base state (no specific mode)"}

            return {
                "status": "success",
                "state_id": current.state_id,
                "state_name": current.name,
                "description": current.description,
                "enabled_tools": current.enabled_tools or "all",
                "disabled_tools": current.disabled_tools,
                "steps_in_state": self.state_machine.steps_in_state,
                "max_steps": current.max_steps_in_state,
                "state_stack": [s for s in self.state_machine.state_stack],
            }

        @tool(atomic=True, name="list_available_states")
        def list_available_states() -> Dict[str, Any]:
            """List all defined states."""
            states = self.state_machine.list_all_states()

            return {
                "status": "success",
                "states": [
                    {
                        "state_id": s.state_id,
                        "name": s.name,
                        "description": s.description,
                        "usage_count": s.usage_count,
                        "created_by": s.created_by,
                    }
                    for s in states
                ],
            }
```

---

## 7. State-Aware Execution Loop

Integration with continuous execution:

```python
class StateAwareContinuousAgent(Agent):
    """Agent that uses states + runs until complete."""

    def continuous_execute_until_complete(
        self,
        task_description: str,
        completion_criteria: dict,
        initial_state: str = "planning",
    ) -> dict:
        """Execute task through multiple states until completion."""
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()

        # Start in initial state
        self.state_machine.enter_state(initial_state, session_id, "task_start")

        step = 0
        max_iterations = 1000  # Safety limit

        while step < max_iterations:
            step += 1

            # Check completion
            if self._check_completion(completion_criteria):
                break

            # Execute one step
            current_state = self.state_machine.get_current_state()
            next_action = self._determine_next_action(current_state)

            # State transition if needed
            if next_action.get("transition_to"):
                self.state_machine.exit_state(session_id, "transition")
                self.state_machine.enter_state(
                    next_action["transition_to"],
                    session_id,
                    reason=next_action.get("reason", ""),
                )

            # Execute in current state
            result = self.process_query(
                next_action["query"],
                max_steps=1,  # Single step at a time
            )

            # Auto-checkpoint
            if step % 50 == 0:
                self.checkpoint_manager.checkpoint(self, session_id, step, "periodic")

        # Final state exit
        self.state_machine.exit_state(session_id, "task_complete")

        return {
            "success": True,
            "steps_taken": step,
            "final_state": self.state_machine.current_state,
            "completion_verified": self._check_completion(completion_criteria),
        }

    def _determine_next_action(self, state: Optional[AgentState]) -> dict:
        """Decide what to do next based on current state and progress."""
        if not state or state.state_id == "planning":
            # In planning: check if plan is done
            if self.manifest_tracker.has_complete_plan():
                return {"query": "Begin implementation", "transition_to": "implementation"}
            else:
                return {"query": "Continue planning"}

        elif state.state_id == "implementation":
            # In implementation: check for test failures
            failing_tests = self.manifest_tracker.get_failing_tests()
            if failing_tests:
                return {"query": f"Debug {failing_tests[0]}", "transition_to": "debug", "reason": "test_failure"}
            else:
                next_goal = self.manifest_tracker.get_next_task()
                if next_goal:
                    return {"query": f"Implement {next_goal['title']}"}
                else:
                    return {"query": "Run final tests", "transition_to": "testing"}

        elif state.state_id == "debug":
            # In debug: check if fix worked
            if self.manifest_tracker.test_now_passes():
                return {"query": "Continue implementation", "transition_to": "implementation", "reason": "debug_complete"}
            else:
                return {"query": "Continue debugging"}

        elif state.state_id == "testing":
            # In testing: check all quality gates
            if self._all_quality_gates_pass():
                return {"query": "Task complete", "transition_to": None}
            else:
                return {"query": "Fix quality issues"}

        # Fallback
        return {"query": "Continue current task"}
```

---

## 8. State Persistence & Retrieval

States stored in database are retrievable:

```python
class StateLibrary:
    """Manage persistent state definitions."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)

    def save_state_for_reuse(self, state: AgentState) -> None:
        """Persist a state definition for future sessions."""
        # Already saved in states table during create_state()
        # This marks it as a reusable template
        self.db.execute("""
            UPDATE states
            SET metadata_json = json_set(metadata_json, '$.reusable', true)
            WHERE state_id = ?
        """, (state.state_id,))
        self.db.commit()

    def load_state_by_purpose(self, purpose: str) -> Optional[AgentState]:
        """Find state by semantic search over descriptions."""
        # Fuzzy match on description
        cursor = self.db.execute("""
            SELECT * FROM states
            WHERE description LIKE ? OR name LIKE ?
            ORDER BY usage_count DESC
            LIMIT 1
        """, (f"%{purpose}%", f"%{purpose}%"))

        row = cursor.fetchone()
        if row:
            return self._row_to_state(row)
        return None

    def get_most_used_states(self, limit: int = 10) -> List[AgentState]:
        """Get states by usage frequency."""
        cursor = self.db.execute("""
            SELECT * FROM states
            ORDER BY usage_count DESC
            LIMIT ?
        """, (limit,))
        return [self._row_to_state(row) for row in cursor.fetchall()]

    def search_states_by_tools(self, required_tools: List[str]) -> List[AgentState]:
        """Find states that have specific tools enabled."""
        results = []
        cursor = self.db.execute("SELECT * FROM states")

        for row in cursor.fetchall():
            state = self._row_to_state(row)
            if state.enabled_tools and all(t in state.enabled_tools for t in required_tools):
                results.append(state)

        return results
```

---

## 9. Learned Instructions (Append-Only)

Accumulated behavioral refinements:

```python
class LearnedInstructionsStore:
    """Manage learned instructions with timestamps."""

    def __init__(self, file_path: str = ".gaia/prompts/learned_instructions.jsonl"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def add_instruction(
        self,
        instruction: str,
        category: str,
        confidence: float,
        source: str,
        timestamp: str,
    ) -> None:
        """Append a new learned instruction (append-only log)."""
        entry = {
            "instruction": instruction,
            "category": category,  # 'correction', 'preference', 'domain_knowledge', 'behavioral'
            "confidence": confidence,
            "source": source,      # 'user_feedback', 'self_evaluation', 'outcome_tracking'
            "timestamp": timestamp,
            "active": True,
        }

        with open(self.file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_active_instructions(
        self,
        max_tokens: int = 500,
        min_confidence: float = 0.5,
        recency_hours: Optional[int] = None,
    ) -> str:
        """Retrieve instructions for prompt injection.

        Prioritizes:
        1. High confidence (>= 0.8)
        2. Recent (last 30 days weighted higher)
        3. Specific categories (corrections > preferences)
        """
        instructions = []

        # Load all
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry.get("active", True) and entry["confidence"] >= min_confidence:
                        # Time filter
                        if recency_hours:
                            ts = datetime.fromisoformat(entry["timestamp"])
                            if (datetime.now() - ts).total_seconds() / 3600 > recency_hours:
                                continue
                        instructions.append(entry)

        # Prioritize
        category_priority = {"correction": 4, "domain_knowledge": 3, "behavioral": 2, "preference": 1}
        instructions.sort(
            key=lambda x: (
                category_priority.get(x["category"], 0),
                x["confidence"],
                x["timestamp"]  # Most recent first within category
            ),
            reverse=True
        )

        # Format for prompt (respecting token budget)
        formatted = []
        token_count = 0
        for entry in instructions:
            text = f"- [{entry['category'].upper()}] {entry['instruction']}"
            entry_tokens = len(text.split())  # Rough estimate

            if token_count + entry_tokens > max_tokens:
                break

            formatted.append(text)
            token_count += entry_tokens

        return "\n".join(formatted)

    def deprecate_instruction(self, instruction_text: str, reason: str, timestamp: str) -> None:
        """Mark an instruction as no longer active."""
        # Append deprecation entry
        self.add_instruction(
            instruction=f"DEPRECATED: {instruction_text} (Reason: {reason})",
            category="deprecation",
            confidence=1.0,
            source="agent_review",
            timestamp=timestamp,
        )
```

---

## 10. State Composition (Nested States)

States can be composed and nested:

```python
def create_composite_state(
    self,
    state_id: str,
    name: str,
    sub_states: List[str],
    orchestration_logic: str,
) -> AgentState:
    """Create a state that sequences through multiple sub-states.

    Example: "full_implementation_cycle" = Planning → Implementation → Testing → Review
    """
    composite_prompt = f"""You are in {name} — a multi-phase workflow.

Sub-states in order: {' → '.join(sub_states)}

Orchestration logic:
{orchestration_logic}

Current phase: {sub_states[0]}

Use enter_state() to transition between phases."""

    return self.state_machine.create_state(
        state_id=state_id,
        name=name,
        prompt_module=composite_prompt,
        created_by="agent",
    )
```

---

## 11. State Versioning

As states evolve with learnings:

```python
def update_state_prompt(
    self,
    state_id: str,
    new_prompt: str,
    reason: str,
    timestamp: str,
) -> int:
    """Update a state's prompt, creating a new version."""
    # Load current version
    current = self.db.execute(
        "SELECT version FROM states WHERE state_id = ?", (state_id,)
    ).fetchone()

    new_version = current[0] + 1 if current else 1

    # Archive old version
    self.db.execute("""
        INSERT INTO state_versions (state_id, version, prompt_module, timestamp, reason)
        VALUES (?, ?, (SELECT system_prompt_module FROM states WHERE state_id = ?), ?, ?)
    """, (state_id, current[0], state_id, timestamp, "version_update"))

    # Update current
    self.db.execute("""
        UPDATE states
        SET system_prompt_module = ?, version = ?
        WHERE state_id = ?
    """, (new_prompt, new_version, state_id))

    self.db.commit()
    return new_version
```

---

## 12. Complete Example: Code Assistant with States

```python
# Initialize agent with state machine
agent = CodeAgent(StateConfig(enable_states=True))

# Task: "Build a REST API"
agent.execute_until_complete(
    task="Build RESTful API with FastAPI",
    initial_state="planning",
)

# Execution flow:
# Step 1-15: PLANNING state
#   - Reads requirements
#   - Designs API structure
#   - Creates manifest with goals
#   - Plan complete → auto-transitions to IMPLEMENTATION

# Step 16-180: IMPLEMENTATION state
#   - Writes endpoints, models, utils
#   - Runs tests after each file
#   - Test fails at step 87 → auto-transitions to DEBUG

# Step 87-95: DEBUG state (nested under IMPLEMENTATION)
#   - Analyzes test failure
#   - Adds logging
#   - Fixes bug
#   - Re-runs test → passes
#   - Exits back to IMPLEMENTATION

# Step 96-200: IMPLEMENTATION state (resumed)
#   - Continues writing remaining endpoints
#   - All files complete → auto-transitions to TESTING

# Step 201-220: TESTING state
#   - Runs full test suite
#   - Checks coverage (85% — pass)
#   - Runs lint (3 warnings — acceptable)
#   - All gates pass → auto-transitions to REVIEW

# Step 221-230: REVIEW state
#   - Reviews all code
#   - Checks security (no issues)
#   - Verifies architecture consistency
#   - Approves quality → Task complete

# All state transitions stored in database with timestamps
# Agent can resume from any checkpoint
```

---

*Adaptive Prompts & State Machine Framework for Gaia Agent SDK.*
*Enables multi-mode execution with state persistence, nested states, and dynamic prompt composition.*
