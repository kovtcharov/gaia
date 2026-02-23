# Why RAC (agent_query) Wasn't Used in Benchmark Runs

**Date:** 2026-02-23
**Issue:** GAIA Code agent has `agent_query()` tool for recursive decomposition, but didn't use it
**Impact:** Agent attempted to generate 19 files sequentially instead of recursively, causing context exhaustion

---

## The Evidence

### Step 7 of Round 2 Retry (bf493b0)

**LLM's stated intention:**
```
🧠 Thought: I now have a thorough understanding of the Python framework
architecture. Let me now implement the C++ port. This is a massive task, so I'll
use agent_query to decompose it into manageable chunks. Let me first create the
project structure and CMakeLists.txt, then implement each component.
🎯 Goal: Create the complete C++ port of the GAIA agent framework
```

**What the LLM actually did:**
```
🔧 Executing operation
  Tool: run_cli_command  ← NOT agent_query!
  Arguments: {"command": "mkdir -p ..."}
```

**Contradiction:** The LLM said it would use `agent_query` but then called `run_cli_command` instead.

---

## Why This Happened

### Reason 1: Weak Guidance in System Prompt

The system prompt mentions `agent_query` exists, but doesn't **strongly guide** when to use it. Looking at the tool description:

```python
@tool
def agent_query(task: str, specialist: Optional[str] = None, max_depth: Optional[int] = None):
    """Delegate a subtask to a sub-agent with fresh context. Use for recursive decomposition."""
```

**Problems with this description:**
1. Too generic - doesn't explain WHEN to use it
2. No examples of what constitutes a "subtask"
3. No guidance on task granularity (how big should subtasks be?)
4. Doesn't explain the BENEFIT (fresh context, context defragmentation)

### Reason 2: Sequential Thinking Bias

LLMs naturally prefer **sequential execution** over **recursive delegation**:
- Sequential: "I'll do step 1, then 2, then 3..." (familiar, simple mental model)
- Recursive: "I'll delegate step 1 to a sub-agent while I..." (requires meta-reasoning)

The prompt for "generate 19 files" triggered sequential thinking:
```
🧠 Thought: Let me write them all systematically - tool_registry.h, then
tool_registry.cpp, output_handler.h, output_handler.cpp, agent.h, agent.cpp...
```

The LLM saw this as a **linear task list**, not as **independent subtasks** that could be delegated.

### Reason 3: No Task Complexity Threshold

The system prompt doesn't tell the agent:
- "If a task requires generating >5 files, use agent_query"
- "If a single file exceeds 500 lines, use agent_query"
- "If you're in PLANNING state with 10+ steps, delegate chunks via agent_query"

Without clear thresholds, the LLM defaults to sequential execution.

### Reason 4: agent_query Not in the Plan

When the LLM creates a plan, it lists concrete tools like `write_file`, `run_cli_command`. But `agent_query` is meta-level — it doesn't appear in plans because plans are for direct actions, not delegation.

This means **RAC decomposition must happen BEFORE planning**, but the current prompt flow encourages planning first:

```
Current flow:
1. Receive task
2. Create plan with 19 steps (all write_file calls)
3. Execute plan sequentially
4. [CRASH at step 41 from context exhaustion]

RAC flow (should be):
1. Receive task
2. Recognize task is complex (19 files)
3. Use agent_query to delegate file generation to sub-agents:
   - agent_query("Generate types.h and tool_registry.h")
   - agent_query("Generate agent.h and agent.cpp")
   - agent_query("Generate all test files")
4. Each sub-agent works in fresh context
5. Parent agent collects results
```

---

## Code Generation and JSON Structure

### Your Question: Does forcing JSON during code generation hurt quality?

**Current approach:**
```json
{
  "thought": "Generate agent.cpp",
  "goal": "Create C++ implementation",
  "tool": "write_file",
  "tool_args": {
    "file_path": "/path/to/agent.cpp",
    "content": "#include <iostream>\nvoid Agent::processQuery() {\n  // 600 lines of C++ code...\n}\n"
  }
}
```

**The problem:** The `content` field is embedded in JSON, which requires:
1. Escaping all quotes (`"` → `\"`)
2. Escaping all backslashes (`\` → `\\`)
3. Escaping newlines (`\n` → `\\n`)
4. Fitting everything in a single max_tokens response

**Does this hurt quality?** YES, potentially:
- Escaped strings are harder for LLMs to generate correctly
- Large escaped content may exceed output token limit (16384)
- Truncation mid-escape creates invalid JSON

### Alternative Approach: Two-Phase Generation

**Phase 1: Generate code in markdown block (no JSON)**
```markdown
```cpp
#include <iostream>

void Agent::processQuery() {
    // Clean, unescaped C++ code
    // LLM generates naturally without escaping
}
```
```

**Phase 2: Wrap in tool call (after generation)**

Extract the code block and wrap it in the tool call:
```json
{
  "thought": "Code generated, writing to file",
  "tool": "write_file",
  "tool_args": {
    "file_path": "/path/to/agent.cpp",
    "content": "<extracted code from markdown block>"
  }
}
```

**Benefits:**
- Code generation is natural (no escaping)
- Better quality (LLM doesn't fight JSON constraints)
- Can handle larger files (code block isn't counted against JSON structure)

**But this breaks the structured output requirement** — the LLM would need to generate markdown THEN JSON, which is two responses.

---

## The Real Solution: RAC Decomposition

Instead of generating a 600-line file in one shot, use `agent_query`:

```python
# Parent agent recognizes large file generation task
{
  "thought": "agent.cpp is complex (~600 lines). Decompose into logical chunks.",
  "goal": "Generate agent.cpp via recursive delegation",
  "tool": "agent_query",
  "tool_args": {
    "task": "Generate the Agent class implementation in C++ with: constructor, processQuery() loop (state machine with 5 states), executeTool() helper, resolvePlanParameters() for $PREV/$STEP_N placeholders. Target: src/agent.cpp, ~600 lines. Use the architecture from the Python agent.py.",
    "specialist": "cpp-developer",
    "max_depth": 1
  }
}
```

The **sub-agent** then operates in fresh context:
- Has 200K tokens for just THIS file
- Can generate the full 600 lines without truncation
- Returns the completed file path
- Parent agent continues with next component

**Each sub-agent gets a clean slate — no context exhaustion.**

---

## Why agent_query Wasn't Used

Looking at step 7's thought: "This is a massive task, so I'll use agent_query to decompose it into manageable chunks."

But then the tool call was `run_cli_command` (mkdir).

**Why the disconnect?**

### Hypothesis 1: LLM Didn't Follow Through
The thought mentioned `agent_query`, but when generating the actual tool call, the LLM defaulted to a simpler action (mkdir). This is "saying vs doing" mismatch.

### Hypothesis 2: No Examples in System Prompt
The system prompt describes `agent_query` but doesn't show EXAMPLES of when/how to use it. Without examples, LLMs often ignore meta-tools.

### Hypothesis 3: Plan Creation Blocked RAC
The prompt instructed:
> "IMPORTANT: ALWAYS BEGIN WITH A PLAN before executing any tools."

This forced the LLM into planning mode, and plans list direct actions (write_file), not meta-actions (agent_query). The decomposition step was skipped.

---

## The Correct RAC Flow

### Current Benchmark Prompt (Problematic)
```
Analyze the GAIA Python agent framework and implement a comprehensive C++17 port.

Required components (all 4 must be implemented):
1. Base Agent class with 5-state machine...
2. Tool Registry with registration...
[... details for all 19 files ...]

Write all output to /path/to/gaiacpp_gaia/.
```

**Problem:** This creates a mental model of "19 sequential write operations" rather than "4 recursive subtasks".

### RAC-Optimized Prompt (Better)
```
You are generating a C++ port of the GAIA Python agent framework.

This is a LARGE task (19 files, ~3500 LOC). You MUST use recursive decomposition:

**STEP 1: Break down into components**
Use agent_query() to delegate these 4 subtasks to sub-agents:

1. agent_query(task="Generate C++ type definitions (types.h with AgentState enum, Message struct, ToolInfo struct, ParsedResponse struct). Target: include/gaia/types.h (~150 lines)", specialist="cpp-developer")

2. agent_query(task="Generate Tool Registry (tool_registry.h/cpp with registration, name resolution via suffix matching, execution with error handling). Target: include/gaia/tool_registry.h + src/tool_registry.cpp (~250 lines total)", specialist="cpp-developer")

3. agent_query(task="Generate MCP Client (mcp_client.h/cpp with cross-platform subprocess, JSON-RPC 2.0, connect/disconnect/listTools/callTool). Target: include/gaia/mcp_client.h + src/mcp_client.cpp (~600 lines total)", specialist="cpp-developer")

4. agent_query(task="Generate Agent base class (agent.h/cpp with processQuery loop, 5-state machine, plan execution with $PREV/$STEP_N resolution, LLM HTTP calls via cpp-httplib). Target: include/gaia/agent.h + src/agent.cpp (~800 lines total)", specialist="cpp-developer")

**STEP 2: After sub-agents complete, use agent_query for tests**
5. agent_query(task="Generate comprehensive unit tests for all components (test_types.cpp, test_tool_registry.cpp, test_mcp_client.cpp, test_agent.cpp, test_console.cpp). Target: tests/*.cpp, 80+ tests", specialist="cpp-developer")

**STEP 3: Build and verify**
6. run_cli_command: cmake -B build && cmake --build build
7. run_cli_command: ctest --test-dir build

Do NOT try to generate all files yourself. DELEGATE to sub-agents via agent_query.
```

**Why this works:**
- Explicit instruction to use `agent_query`
- Clear subtask boundaries
- Each subtask is small enough for one sub-agent
- Pattern is obvious: component decomposition, not file-by-file

---

## The JSON/Code Quality Question

**Your concern:** Does forcing JSON structure during code generation hurt quality?

**Answer:** Yes, when the code is embedded directly in the JSON:

```json
{
  "tool": "write_file",
  "tool_args": {
    "content": "#include <iostream>\\nvoid func() {\\n  std::cout << \\"hello\\";\\n}\\n"
  }
}
```

Problems:
- Excessive escaping (`\\n`, `\\"`)
- Hard for LLM to generate correctly
- Truncation risk at max_tokens

**But with RAC, this isn't an issue** because:
1. Sub-agent generates ONLY the code (in its own dedicated response)
2. Code is small enough (<1000 lines per sub-agent) to fit in output
3. Parent agent wraps result, not generate-and-wrap simultaneously

---

## Recommended Fix: Make RAC Decomposition Automatic

### Approach 1: Agent-Driven (Current, Requires Strong Prompting)

Rely on the LLM to recognize when to use `agent_query`. Requires:
- Examples in system prompt
- Clear thresholds ("If task requires >5 files, use agent_query")
- Reinforcement: "DECOMPOSE complex tasks. Do NOT try to do everything yourself."

**Problem:** Unreliable. LLMs often ignore meta-instructions.

### Approach 2: Framework-Driven (Automatic, Reliable)

Detect task complexity and **auto-decompose** before the LLM sees it:

```python
# src/gaia/agents/gaia_code/agent.py

def process_query(self, user_input: str, **kwargs):
    """Process a query with automatic RAC decomposition for complex tasks."""

    # Analyze task complexity
    complexity = self._analyze_task_complexity(user_input)

    if complexity["should_decompose"]:
        # Auto-decompose into subtasks
        subtasks = self._decompose_task(user_input, complexity)

        results = []
        for subtask in subtasks:
            # Each subtask gets fresh context via agent_query
            result = self.tool_agent_query(
                task=subtask["description"],
                specialist=subtask["specialist"]
            )
            results.append(result)

        # Synthesize results
        return self._synthesize_results(user_input, results)

    else:
        # Simple task - direct execution
        return super().process_query(user_input, **kwargs)

def _analyze_task_complexity(self, task: str) -> Dict[str, Any]:
    """Determine if task should be decomposed."""

    # Heuristics for complexity:
    file_count = task.count("file") + task.count("component") + task.count("class")
    has_multiple_files = file_count > 5
    has_large_scope = any(kw in task.lower() for kw in ["framework", "system", "architecture", "port", "implement all"])
    has_explicit_count = any(str(n) in task for n in range(10, 100))  # "19 files", "50 tests"

    should_decompose = has_multiple_files or has_large_scope or has_explicit_count

    return {
        "should_decompose": should_decompose,
        "estimated_files": file_count,
        "scope": "large" if has_large_scope else "medium" if has_multiple_files else "small"
    }

def _decompose_task(self, task: str, complexity: Dict) -> List[Dict[str, str]]:
    """Break task into subtasks for agent_query."""

    # Use LLM to decompose (meta-planning)
    decomposition_prompt = f"""
    Task: {task}

    This is a complex task (estimated {complexity['estimated_files']} files).
    Break it down into 3-5 independent subtasks that can be delegated to specialist agents.

    Each subtask should:
    - Generate 1-3 related files
    - Be self-contained (not depend on other subtasks completing first)
    - Take <30 minutes for a sub-agent to complete

    Return JSON array: [{{"description": "...", "specialist": "cpp-developer"}}, ...]
    """

    response = self.chat.send_messages(
        messages=[{"role": "user", "content": decomposition_prompt}],
        system_prompt="You decompose complex coding tasks into independent subtasks."
    )

    # Parse subtasks
    import json
    subtasks = json.loads(response.text)
    return subtasks
```

**Impact:** Framework automatically uses RAC when appropriate, regardless of whether the LLM "remembers" to.

---

## Approach Comparison

| Approach | Reliability | Effort | Flexibility |
|----------|-------------|--------|-------------|
| **Prompt-driven** ("use agent_query for complex tasks") | Low (LLM often ignores) | Low (just add examples) | High (LLM decides) |
| **Framework-driven** (auto-decompose by heuristics) | Medium (heuristics may mis-classify) | Medium (implement analysis logic) | Medium (fixed rules) |
| **Hybrid** (framework suggests, LLM confirms) | High | High (both systems) | High |

---

## Recommended Solution: Hybrid Approach

### Step 1: Add Strong Guidance to System Prompt

```python
# system_prompt.py

RAC_DECOMPOSITION_GUIDANCE = """
==== RECURSIVE DECOMPOSITION (agent_query) ====

For COMPLEX TASKS, you MUST use agent_query() to delegate subtasks to sub-agents.

**When to use agent_query:**
- Task requires generating >5 files
- Single file exceeds 500 lines
- Task involves multiple independent components
- You're creating a "framework", "system", or "port"

**How to use agent_query:**

WRONG (sequential generation):
{
  "plan": [
    {"tool": "write_file", "tool_args": {"file_path": "a.cpp", "content": "..."}},
    {"tool": "write_file", "tool_args": {"file_path": "b.cpp", "content": "..."}},
    {"tool": "write_file", "tool_args": {"file_path": "c.cpp", "content": "..."}}
  ]
}

RIGHT (recursive delegation):
{
  "plan": [
    {"tool": "agent_query", "tool_args": {"task": "Generate types.h and tool_registry.h with..."}},
    {"tool": "agent_query", "tool_args": {"task": "Generate agent.h and agent.cpp with..."}},
    {"tool": "agent_query", "tool_args": {"task": "Generate all test files (5 files) with..."}}
  ]
}

**Why agent_query is better:**
- Sub-agents work in FRESH CONTEXT (no 200K token crashes)
- Each subtask completes before next starts (better quality)
- Parallel delegation possible (faster completion)
- Parent agent stays context-lean (just coordinates results)

**Example:**
Task: "Port GAIA framework to C++ (19 files)"
Plan:
1. agent_query("Generate C++ type system headers: types.h, json_utils.h, tool_registry.h")
2. agent_query("Generate MCP client: mcp_client.h/cpp with subprocess + JSON-RPC")
3. agent_query("Generate Agent base class: agent.h/cpp with state machine")
4. agent_query("Generate tests: test_*.cpp files, aim for 80+ tests")
5. run_cli_command: build and test the generated code
"""
```

### Step 2: Add Framework Detection

```python
# agent.py - after parsing plan

if "plan" in parsed and len(parsed["plan"]) > 10:
    # Large plan - suggest decomposition
    uses_agent_query = any(
        step.get("tool") == "agent_query"
        for step in parsed["plan"]
    )

    if not uses_agent_query:
        # Suggest RAC decomposition
        suggestion = (
            f"⚠️  You created a plan with {len(parsed['plan'])} steps. "
            "Consider using agent_query() to delegate groups of related steps to sub-agents. "
            "This prevents context exhaustion and improves quality.\n\n"
            "Would you like to:\n"
            "A) Proceed with this plan (may hit context limits)\n"
            "B) Decompose into agent_query subtasks (recommended for >10 steps)\n"
        )

        # In interactive mode, prompt user
        # In autonomous mode, auto-decompose
        if self.tui_mode != "off":
            self.console.print_warning(suggestion)
            # For now, proceed - but we've warned
```

### Step 3: Add agent_query Examples to Tool Description

```python
@tool
def agent_query(task: str, specialist: Optional[str] = None, max_depth: Optional[int] = None):
    """
    Delegate a subtask to a sub-agent with FRESH CONTEXT.

    Use agent_query for:
    - Generating large files (>500 lines)
    - Multi-file tasks (>5 related files)
    - Complex logic that requires deep reasoning
    - When approaching context limits

    **Benefits:**
    - Sub-agent gets 200K tokens of fresh context
    - Parent agent stays context-lean
    - Better code quality (sub-agent focuses on one thing)
    - Parallel execution possible

    **Examples:**

    # Generate a large C++ file
    agent_query(
        task="Generate agent.cpp with Agent class implementation: processQuery() loop, 5-state machine (PLANNING, EXECUTING_PLAN, ERROR_RECOVERY, DIRECT_EXECUTION, COMPLETION), executeTool(), resolvePlanParameters() for $PREV/$STEP_N placeholders. ~600 lines. Use architecture from Python agent.py.",
        specialist="cpp-developer"
    )

    # Generate related test files
    agent_query(
        task="Generate unit tests for the Agent class: test state machine transitions, test processQuery with mock LLM, test plan execution, test error recovery. Target: tests/test_agent.cpp, ~200 lines, use GoogleTest.",
        specialist="cpp-developer"
    )

    # Generate multiple headers
    agent_query(
        task="Generate C++ headers for core types: types.h (AgentState enum, Message struct, ToolInfo), json_utils.h (extraction strategies), tool_registry.h (ToolRegistry class). 3 files, ~400 lines total.",
        specialist="cpp-developer"
    )
    """
    return self.tool_agent_query(task, specialist, max_depth)
```

---

## Expected Behavior After Fixes

### Scenario: "Port GAIA framework to C++ (19 files)"

**With strong RAC guidance:**

```
📝 Step 1: Analyzing task complexity...
🧠 Thought: This task requires 19 files (~3500 LOC). I'll decompose into 4 component-based subtasks using agent_query.
🎯 Goal: Coordinate recursive decomposition for C++ port

╭─── 📋 Execution Plan (5 steps) ───────────────────────╮
│   Step 1/5: agent_query
│     💡 Generate type system headers (types.h, json_utils.h, tool_registry.h)
│     📝 task="Generate C++ headers...", specialist="cpp-developer"
│ ▶ Step 2/5: agent_query
│     💡 Generate MCP client implementation
│     📝 task="Generate mcp_client.h/cpp...", specialist="cpp-developer"
│   Step 3/5: agent_query
│     💡 Generate Agent base class
│     📝 task="Generate agent.h/cpp...", specialist="cpp-developer"
│   Step 4/5: agent_query
│     💡 Generate all unit tests
│     📝 task="Generate test_*.cpp files...", specialist="cpp-developer"
│   Step 5/5: run_cli_command
│     💡 Build and test the complete implementation
│     📝 command="cd ... && cmake ... && ctest ..."
╰───────────────────────────────────────────────────────╯

[Each agent_query spawns a sub-agent with 200K fresh tokens]
[Parent agent stays at ~10K tokens throughout]
[No context exhaustion possible]
```

---

## Implementation Priority

1. **HIGHEST:** Add RAC guidance to system prompt with examples (1 hour)
2. **HIGH:** Enhance `agent_query` tool description with use cases (30 minutes)
3. **MEDIUM:** Add task complexity detection with warnings (2 hours)
4. **LOW:** Automatic decomposition (research needed - may be too aggressive)

---

## Key Insight

**The problem wasn't that RAC doesn't exist** — it's that the LLM wasn't properly guided to use it.

With strong prompting showing WHEN and HOW to use `agent_query`, plus examples, the agent should naturally decompose complex tasks. This solves both:
- Context exhaustion (each sub-agent is fresh)
- Code quality (sub-agents focus on one file, no JSON escaping issues)

**The answer to your question:** Yes, forcing JSON during large code generation hurts quality. **But with proper RAC decomposition, you never generate large code in JSON** — you generate it in a sub-agent's dedicated response, then the framework wraps it.
