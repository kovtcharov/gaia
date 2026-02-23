# Thought, Goal, and Plan Display Bugs

**Date:** 2026-02-23
**Issue:** Missing thought/goal display in PLANNING state and missing formatted plan output

---

## Bug 1: Empty Thought/Goal Suppression

### Symptom

Steps 6-7 in Round 2 showed:
```
📝 Step 6: Thinking...
🔄 PLANNING: Creating or refining plan

[BLANK - no thought displayed]
[BLANK - no goal displayed]

📝 Step 7: Thinking...
🔄 PLANNING: Creating or refining plan

[BLANK - no thought displayed]
[BLANK - no goal displayed]

╭── ✅ Final Answer ──╮
[Answer collapse - code dump]
```

### Root Cause

The display logic at `agent.py:2271-2278` suppresses thought/goal if empty or placeholder:

```python
thought = parsed.get("thought", "").strip()
goal = parsed.get("goal", "").strip()

if thought and thought != "No explicit reasoning provided":
    self.console.print_thought(thought)

if goal and goal != "No explicit goal provided":
    self.console.print_goal(goal)
```

When the LLM returns empty strings or the default placeholder, **nothing is printed**.

### Why This is a Problem

1. **Debugging:** Can't tell if the LLM is malfunctioning (returning empty thoughts) or actively planning
2. **User feedback:** Silent steps look like the agent is stuck
3. **Answer collapse detection:** Empty thought/goal often precedes answer collapse — it's a warning sign

### Proposed Fix

**Always display thought/goal, even if empty, to show the agent is alive:**

```python
# src/gaia/agents/base/agent.py line 2271-2278

thought = parsed.get("thought", "").strip()
goal = parsed.get("goal", "").strip()

# Always display thought (even if empty/placeholder)
if thought:
    self.console.print_thought(thought)
elif parsed.get("answer"):  # Answering without thought
    self.console.print_thought("[No reasoning provided]")
else:  # Planning/acting without thought
    self.console.print_thought("[Empty thought - possible LLM issue]")

# Always display goal (even if empty/placeholder)
if goal:
    self.console.print_goal(goal)
elif parsed.get("answer"):  # Answering without goal
    self.console.print_goal("[Task completion]")
else:  # Planning/acting without goal
    self.console.print_goal("[No goal specified - possible LLM issue]")
```

**Impact:** Users can now see when the LLM is returning degenerate responses, enabling earlier intervention.

---

## Bug 2: Plan Not Displayed in PLANNING State

### Symptom

When the agent creates a plan, the plan is:
1. Stored internally (`self.current_plan = parsed["plan"]`)
2. Execution state transitions to `EXECUTING_PLAN`
3. **But the plan is never printed to the console**

The only place `console.print_plan()` is called is at line 2383:
```python
if self.current_plan:
    self.console.print_plan(self.current_plan, self.current_step)
```

But this is **inside** the `if "tool" in parsed` block, so it only prints when a tool is about to execute, not when the plan is first created.

### Why This is a Problem

Users don't see:
- What the agent is planning to do
- How many steps are in the plan
- Whether the plan makes sense before execution starts

This makes debugging and understanding agent behavior much harder.

### Proposed Fix

**Display the plan immediately when it's created:**

```python
# src/gaia/agents/base/agent.py after line 2376

# Plan is valid - proceed with execution
self.current_plan = parsed["plan"]
self.current_step = 0
self.total_plan_steps = len(self.current_plan)
self.execution_state = self.STATE_EXECUTING_PLAN
logger.debug(f"New plan created with {self.total_plan_steps} steps: {self.current_plan}")

# ✨ NEW: Display the plan to the user in a nicely formatted way
self.console.print_plan(self.current_plan, current_step=-1)  # -1 = no step highlighted
self.console.print_info(f"Plan created with {self.total_plan_steps} steps. Starting execution...")
```

---

## Bug 3: Plan Display Format is Not User-Friendly

### Current Implementation

The `print_plan()` method in `console.py` shows minimal information:

```
📋 Plan:
  1. read_file
  2. write_file
  3. run_cli_command
```

This doesn't show:
- What arguments each step uses
- What each step is trying to accomplish
- Whether the plan makes logical sense

### Proposed Fix

**Enhanced plan display with arguments and descriptions:**

```python
# src/gaia/agents/base/console.py

def print_plan(self, plan: List[Dict[str, Any]], current_step: int = -1):
    """
    Display the execution plan in a nicely formatted, human-readable format.

    Args:
        plan: List of plan steps (dicts with tool, tool_args, optional description)
        current_step: Which step is currently executing (-1 = none, show full plan)
    """
    self._print(f"\n{COLORS['blue']}╭─── 📋 Execution Plan ({len(plan)} steps) ───╮{COLORS['reset']}")

    for i, step in enumerate(plan):
        tool_name = step.get("tool", "unknown")
        tool_args = step.get("tool_args", {})
        description = step.get("description", "")

        # Highlight current step
        if i == current_step:
            prefix = f"{COLORS['green']}▶ {i+1}.{COLORS['reset']}"
        else:
            prefix = f"  {i+1}."

        # Format the step
        self._print(f"{prefix} {COLORS['yellow']}{tool_name}{COLORS['reset']}")

        # Show description if available
        if description:
            self._print(f"     💡 {COLORS['dim']}{description}{COLORS['reset']}")

        # Show key arguments (truncate if too long)
        if tool_args:
            args_str = self._format_args_summary(tool_args, max_len=60)
            self._print(f"     📝 {COLORS['dim']}{args_str}{COLORS['reset']}")

    self._print(f"{COLORS['blue']}╰───────────────────────────────────────────╯{COLORS['reset']}\n")

def _format_args_summary(self, args: Dict[str, Any], max_len: int = 60) -> str:
    """Format tool arguments as a compact summary."""
    if not args:
        return "(no args)"

    items = []
    for key, value in args.items():
        # Truncate long values
        if isinstance(value, str) and len(value) > 40:
            value_str = value[:37] + "..."
        elif isinstance(value, (dict, list)):
            value_str = type(value).__name__ + f"({len(value)})"
        else:
            value_str = str(value)

        items.append(f"{key}={value_str}")

    summary = ", ".join(items)
    if len(summary) > max_len:
        return summary[:max_len-3] + "..."
    return summary
```

**Example output:**

```
╭─── 📋 Execution Plan (5 steps) ───╮
  1. read_file
     💡 Read the Python agent.py source code
     📝 file_path=/mnt/c/.../agent.py
  2. write_file
     💡 Create C++ agent.h header
     📝 file_path=.../agent.h, content=str(641)
  3. write_file
     💡 Create C++ agent.cpp implementation
     📝 file_path=.../agent.cpp, content=str(1205)
  4. run_cli_command
     💡 Build the C++ project with cmake
     📝 command=cmake --build build, working_dir=...
  5. run_cli_command
     💡 Run all unit tests
     📝 command=ctest --test-dir build --output-on-failure
╰───────────────────────────────────────────╯
```

---

## Bug 4: No Plan Refinement Display

### Symptom

When the agent creates a NEW plan after completing a previous plan (plan iteration), the new plan replaces `self.current_plan` but there's no visual indication that the plan changed.

### Current Code

```python
# Line 2370-2376
self.current_plan = parsed["plan"]
self.current_step = 0
self.total_plan_steps = len(self.current_plan)
self.execution_state = self.STATE_EXECUTING_PLAN
logger.debug(f"New plan created with {self.total_plan_steps} steps: {self.current_plan}")
# No console output!
```

### Proposed Fix

```python
# After line 2376
logger.debug(f"New plan created with {self.total_plan_steps} steps")

# Display plan iteration info
if self.plan_iterations > 0:
    self.console.print_header(f"📋 Refined Plan (Iteration {self.plan_iterations + 1})")
else:
    self.console.print_header("📋 Execution Plan")

# Display the full plan
self.console.print_plan(self.current_plan, current_step=-1)
self.console.print_info(
    f"Plan has {self.total_plan_steps} steps. Starting execution..."
)
```

---

## Bug 5: Plan Completion Not Announced

### Symptom

When all plan steps complete (`self.current_step >= self.total_plan_steps`), the agent transitions to `COMPLETION` state but doesn't announce what was accomplished.

### Current Code (Line 1830-1841)

```python
if self.current_step >= self.total_plan_steps:
    logger.debug("Plan execution completed")
    self.execution_state = self.STATE_COMPLETION
    self.console.print_state_info("COMPLETION: Plan fully executed")
    # No summary of what was done!
```

### Proposed Fix

```python
if self.current_step >= self.total_plan_steps:
    logger.debug("Plan execution completed")
    self.execution_state = self.STATE_COMPLETION

    # Show completion summary
    self.console.print_separator()
    self.console.print_success(f"✅ Plan completed ({self.total_plan_steps} steps executed)")

    # Summarize what was done
    self.console.print_header("Plan Summary")
    for i, step in enumerate(self.current_plan):
        tool_name = step.get("tool", "unknown")
        desc = step.get("description", "")
        status = "✓"  # All completed if we reached here
        self.console.print_info(f"{status} Step {i+1}: {tool_name} - {desc or '(no description)'}")

    self.console.print_separator()
    self.console.print_state_info("COMPLETION: Requesting final answer...")
```

---

## Summary of Proposed Changes

| Bug | File | Line | Change | Impact |
|-----|------|------|--------|--------|
| **Empty thought/goal suppression** | `agent.py` | 2271-2278 | Always display, show placeholder when empty | Better debugging |
| **Plan not displayed on creation** | `agent.py` | ~2376 | Add `print_plan()` after plan validation | User sees plan before execution |
| **Poor plan format** | `console.py` | `print_plan()` | Enhanced formatting with args/descriptions | Better plan readability |
| **No plan refinement indicator** | `agent.py` | ~2376 | Add header showing iteration number | User sees plan evolution |
| **No plan completion summary** | `agent.py` | 1830-1841 | Add completion summary with checkmarks | User sees what was accomplished |

---

## Implementation Priority

1. **HIGH:** Bug 2 (display plan on creation) — Critical for user understanding
2. **HIGH:** Bug 3 (enhance plan format) — Makes plans actually readable
3. **MEDIUM:** Bug 1 (show empty thoughts) — Helps debugging
4. **LOW:** Bug 4 (plan refinement indicator) — Nice-to-have
5. **LOW:** Bug 5 (completion summary) — Nice-to-have

---

## Testing

### Test Case 1: Empty Thought/Goal

```python
def test_empty_thought_goal_displayed():
    """Verify that empty thoughts/goals are displayed with placeholders."""

    # Mock LLM returns empty thought/goal
    response = '{"thought": "", "goal": "", "tool": "echo", "tool_args": {}}'

    # Should print placeholder, not be silent
    # Expected output: "🧠 Thought: [Empty thought - possible LLM issue]"
```

### Test Case 2: Plan Display on Creation

```python
def test_plan_displayed_on_creation():
    """Verify plan is displayed immediately when created."""

    response = '''{
        "thought": "Need multiple steps",
        "goal": "Complete the task",
        "plan": [
            {"tool": "read_file", "tool_args": {"file_path": "a.py"}},
            {"tool": "write_file", "tool_args": {"file_path": "b.cpp", "content": "..."}}
        ],
        "tool": "read_file",
        "tool_args": {"file_path": "a.py"}
    }'''

    # Should call print_plan() before executing first tool
    # Expected: "╭─── 📋 Execution Plan (2 steps) ───╮"
```

### Test Case 3: Enhanced Plan Format

```python
def test_enhanced_plan_format_shows_args():
    """Verify plan display includes argument summaries."""

    plan = [
        {"tool": "read_file", "tool_args": {"file_path": "/long/path/to/file.py"},
         "description": "Read Python source"},
        {"tool": "write_file", "tool_args": {"file_path": "out.cpp", "content": "x" * 1000},
         "description": "Write C++ output"}
    ]

    # Expected output includes truncated args:
    # "📝 file_path=/long/path/to/file.py"
    # "📝 file_path=out.cpp, content=str(1000)"
```

---

## Recommended Changes

### File: `src/gaia/agents/base/agent.py`

**Change 1: Always display thought/goal (lines 2271-2278)**

```python
# OLD:
thought = parsed.get("thought", "").strip()
goal = parsed.get("goal", "").strip()

if thought and thought != "No explicit reasoning provided":
    self.console.print_thought(thought)

if goal and goal != "No explicit goal provided":
    self.console.print_goal(goal)

# NEW:
thought = parsed.get("thought", "").strip()
goal = parsed.get("goal", "").strip()

# Always display thought (with placeholder if empty)
if thought and thought != "No explicit reasoning provided":
    self.console.print_thought(thought)
elif not parsed.get("answer"):  # Not an answer response
    self.console.print_thought("[⚠️  Empty thought]")

# Always display goal (with placeholder if empty)
if goal and goal != "No explicit goal provided":
    self.console.print_goal(goal)
elif not parsed.get("answer"):  # Not an answer response
    self.console.print_goal("[⚠️  No goal specified]")
```

**Change 2: Display plan on creation (after line 2376)**

```python
# After creating the plan (line 2376):
self.execution_state = self.STATE_EXECUTING_PLAN
logger.debug(f"New plan created with {self.total_plan_steps} steps")

# ✨ NEW: Display the plan nicely
if self.plan_iterations > 0:
    self.console.print_info(f"📋 Refined Plan (Iteration {self.plan_iterations + 1}/{self.max_plan_iterations})")
else:
    self.console.print_info("📋 Execution Plan Created")

# Show the full plan with enhanced formatting
self.console.print_plan(self.current_plan, current_step=-1)
```

**Change 3: Plan completion announcement (line 1830-1841)**

```python
if self.current_step >= self.total_plan_steps:
    logger.debug("Plan execution completed")
    self.execution_state = self.STATE_COMPLETION

    # ✨ NEW: Show what was accomplished
    self.console.print_success(
        f"✅ Plan completed ({self.total_plan_steps}/{self.total_plan_steps} steps)"
    )
    self.console.print_state_info("COMPLETION: Requesting final answer from LLM...")
```

---

### File: `src/gaia/agents/base/console.py`

**Change: Enhanced print_plan() formatting**

Replace the current `print_plan()` implementation with:

```python
def print_plan(self, plan: List[Dict[str, Any]], current_step: int = -1):
    """
    Display an execution plan in a nicely formatted, readable format.

    Args:
        plan: List of plan steps, each a dict with 'tool', 'tool_args', optional 'description'
        current_step: Index of currently executing step (-1 = no step active, show full plan)
    """
    if not plan:
        return

    # Header
    total_steps = len(plan)
    self._print(f"\n{ANSI_BLUE}╭─── 📋 Plan ({total_steps} step{'s' if total_steps > 1 else ''}) ───", end="")
    self._print("─" * (60 - len(f"Plan ({total_steps} steps)") - 10) + "╮")

    # Steps
    for i, step in enumerate(plan):
        tool_name = step.get("tool", "unknown_tool")
        tool_args = step.get("tool_args", {})
        description = step.get("description", "")

        # Current step indicator
        if i == current_step:
            prefix = f"│ {ANSI_GREEN}▶ Step {i+1}/{total_steps}{ANSI_RESET}"
        elif i < current_step:
            prefix = f"│ {ANSI_DIM}✓ Step {i+1}/{total_steps}{ANSI_RESET}"
        else:
            prefix = f"│   Step {i+1}/{total_steps}"

        # Tool name
        self._print(f"{prefix}: {ANSI_YELLOW}{tool_name}{ANSI_RESET}")

        # Description (if provided)
        if description:
            self._print(f"│     {ANSI_DIM}💡 {description}{ANSI_RESET}")

        # Key arguments (show first 3 most important)
        if tool_args:
            args_summary = self._summarize_args(tool_args, max_items=3, max_value_len=50)
            if args_summary:
                self._print(f"│     {ANSI_DIM}📝 {args_summary}{ANSI_RESET}")

    # Footer
    self._print(f"{ANSI_BLUE}╰─────────────────────────────────────────────────────────────╯{ANSI_RESET}\n")

def _summarize_args(self, args: Dict[str, Any], max_items: int = 3, max_value_len: int = 50) -> str:
    """Create a compact summary of tool arguments."""
    if not args:
        return ""

    items = []
    for i, (key, value) in enumerate(args.items()):
        if i >= max_items:
            items.append(f"... (+{len(args) - max_items} more)")
            break

        # Format value based on type
        if isinstance(value, str):
            if len(value) > max_value_len:
                value_str = f'"{value[:max_value_len-3]}..."'
            else:
                value_str = f'"{value}"'
        elif isinstance(value, (dict, list)):
            value_str = f"{type(value).__name__}[{len(value)}]"
        else:
            value_str = str(value)

        items.append(f"{key}={value_str}")

    return ", ".join(items)
```

**Example output:**

```
╭─── 📋 Plan (5 steps) ───────────────────────────────────╮
│   Step 1/5: read_file
│     💡 Read the Python agent.py source code
│     📝 file_path="/mnt/c/Users/14255/Work/gaia3/src/gaia/agents/base/agent.py"
│   Step 2/5: write_file
│     💡 Create C++ types.h header
│     📝 file_path=".../types.h", content=str[1250]
│ ▶ Step 3/5: write_file
│     💡 Create C++ agent.cpp implementation
│     📝 file_path=".../agent.cpp", content=str[2104]
│   Step 4/5: run_cli_command
│     💡 Build with CMake
│     📝 command="cmake --build build -j4"
│   Step 5/5: run_cli_command
│     💡 Run unit tests
│     📝 command="ctest --test-dir build --output-on-failure"
╰─────────────────────────────────────────────────────────╯
```

---

## Rollout Plan

### Phase 1: Display Fixes (1 hour)
1. Implement Change 1: Always display thought/goal with placeholders
2. Implement Change 2: Display plan on creation
3. Implement Change 3: Plan completion announcement
4. Test with synthetic agent run

### Phase 2: Enhanced Formatting (2 hours)
1. Implement enhanced `print_plan()` with args/descriptions
2. Add `_summarize_args()` helper
3. Test with real GAIA Code benchmark
4. Verify output is readable and informative

### Phase 3: Validation (30 minutes)
1. Run a test query that creates a multi-step plan
2. Verify plan is displayed before execution starts
3. Verify current step is highlighted during execution
4. Verify completion announcement shows what was done

---

## Expected User Experience (After Fixes)

```
🤖 Processing: 'Port GAIA agent to C++'

📝 Step 1: Thinking...
🔄 PLANNING: Creating or refining plan

🧠 Thought: I need to read the Python source first to understand the architecture
🎯 Goal: Analyze Python agent framework

🔧 Executing operation: read_file
[... tool executes ...]

📝 Step 2: Thinking...
🔄 PLANNING: Creating detailed execution plan

🧠 Thought: Based on the Python code, I'll create a 5-step plan to generate all C++ files
🎯 Goal: Create comprehensive C++17 port with all components

📋 Execution Plan Created

╭─── 📋 Plan (5 steps) ───────────────────────────────────╮
│   Step 1/5: write_file
│     💡 Create types.h with core enums and structs
│     📝 file_path=".../types.h", content=str[450]
│   Step 2/5: write_file
│     💡 Create agent.h with base Agent class
│     📝 file_path=".../agent.h", content=str[320]
│   [... 3 more steps ...]
╰─────────────────────────────────────────────────────────╯

ℹ️  Plan has 5 steps. Starting execution...

📝 Step 3: Thinking...
🔄 EXECUTING PLAN: Step 1/5

╭─── 📋 Plan (5 steps) ───────────────────────────────────╮
│ ▶ Step 1/5: write_file  [← highlighted]
│     💡 Create types.h with core enums and structs
│     📝 file_path=".../types.h", content=str[450]
│   [... remaining steps ...]
╰─────────────────────────────────────────────────────────╯

[... execution continues ...]

📝 Step 7: Thinking...
✅ Plan completed (5/5 steps)

╭─── Plan Summary ───╮
│ ✓ Step 1: write_file - Create types.h
│ ✓ Step 2: write_file - Create agent.h
│ ✓ Step 3: write_file - Create agent.cpp
│ ✓ Step 4: run_cli_command - Build with CMake
│ ✓ Step 5: run_cli_command - Run unit tests
╰────────────────────╯

🔄 COMPLETION: Requesting final answer from LLM...
```

**Much better user experience** — users can see what the agent is planning before it acts, track progress through the plan, and see a summary of accomplishments.
