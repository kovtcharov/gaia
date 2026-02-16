# GAIA Code: Autonomous Self-Review Capability

**Question**: Does GAIA Code autonomously review and fix its own code, or does it require manual prompting like Claude Code?

**Answer**: ✅ **YES - GAIA Code has AUTONOMOUS self-review built-in**

---

## How It Works

### 1. Quality Gates (Automatic Review)

**Every time code is written**, quality gates automatically run:

```python
# Agent writes code
agent.process_query("Create a REST API")

# Automatic quality gates (no manual prompting needed):
✓ SyntaxGate: Check all files parse without errors
✓ ImportGate: Check all imports resolve
✓ TestGate: Run tests and verify they pass
```

**If ANY gate fails**, agent automatically:
1. Detects the error
2. Diagnoses root cause
3. Applies fix
4. Re-runs gates
5. Repeats until all gates pass

**No manual prompting needed!**

### 2. Escalation Ladder (Automatic Recovery)

**When quality gates fail**, automatic escalation:

```
Attempt 1-2: RETRY
  → Agent analyzes error
  → Applies targeted fix
  → Re-runs quality gates

Attempt 3: DECOMPOSE
  → Agent breaks task into smaller pieces
  → Uses agent_query() to delegate each piece
  → Each piece independently verified

Attempt 4: ALTERNATIVE (if still failing)
  → Agent tries completely different approach
  → Re-runs quality gates

Attempt 5: ASK_USER (last resort)
  → Only if all automatic attempts fail
```

**No manual review prompting - it's all automatic!**

### 3. DebuggerAgent (Autonomous Debugging)

**When tests fail or errors occur**:

```
Agent detects: Test failure in test_auth.py
  ↓
Auto-selects: DebuggerAgent specialist
  ↓
DebuggerAgent workflow (autonomous):
  1. ANALYZE: Read error message
  2. DIAGNOSE: Find root cause (not just symptom)
  3. ISOLATE: Locate problematic code
  4. FIX: Apply minimal fix
  5. VALIDATE: Re-run test to verify
  ↓
Returns: Verified fix
```

**No manual prompting - all automatic!**

### 4. Continuous Execution (Keeps Going)

**Unlike Claude Code** (stops and asks):
```
Claude Code:
  Writes code → Stops → Waits for user to say "test it"
  Tests fail → Stops → Waits for user to say "fix it"
```

**GAIA Code** (continuous):
```
GAIA Code:
  Writes code → Auto-runs tests → Detects failures → Auto-fixes → Re-tests
  ↓
  Only stops when ALL quality gates pass (or max attempts reached)
```

**Runs autonomously until verified complete!**

---

## Comparison: Claude Code vs GAIA Code

### Claude Code (Manual Review)

**Workflow**:
```
User: "Build a REST API"
Claude: [Writes code]
User: "Test it"  ← Manual prompt needed
Claude: [Tests fail]
User: "Fix the errors"  ← Manual prompt needed
Claude: [Fixes]
User: "Test again"  ← Manual prompt needed
Claude: [Tests pass]
```

**Problems**:
- ❌ Requires manual prompting at each step
- ❌ User must notice errors
- ❌ User must ask for fixes
- ❌ Slow, tedious, error-prone

### GAIA Code (Autonomous Review)

**Workflow**:
```
User: "Build a REST API"
  ↓
GAIA Code:
  1. Writes code
  2. Auto-runs quality gates ← AUTONOMOUS
     - Syntax check
     - Import check
     - Test check
  3. Detects test failures ← AUTONOMOUS
  4. Auto-selects DebuggerAgent ← AUTONOMOUS
  5. Diagnoses root cause ← AUTONOMOUS
  6. Applies fix ← AUTONOMOUS
  7. Re-runs gates ← AUTONOMOUS
  8. Repeats until all gates pass ← AUTONOMOUS
  9. Returns verified result
```

**Benefits**:
- ✅ No manual prompting needed
- ✅ Automatic error detection
- ✅ Automatic fixing
- ✅ Continuous execution
- ✅ Verified quality

---

## Self-Review Mechanisms

### Mechanism 1: Quality Gates (M2)

**File**: `quality_gates.py`

**Automatic checks**:
1. **SyntaxGate**: Parses all Python files with AST
   - Detects: Missing colons, unmatched parens, indentation errors
   - Action: Automatically triggers error recovery

2. **ImportGate**: Validates all imports
   - Detects: Missing packages, typos, wrong paths
   - Action: Automatically triggers fix

3. **TestGate**: Runs pytest/jest
   - Detects: Failing tests, assertion errors
   - Action: Automatically triggers DebuggerAgent

**Triggered**: After EVERY code change

### Mechanism 2: EscalationLadder (M2)

**File**: `quality_gates.py`

**Automatic progression**:
```python
class EscalationLadder:
    """
    Automatic recovery strategy when code has errors.

    No manual prompting needed - escalates automatically.
    """

    def get_action(self) -> str:
        if self.should_retry():
            return "retry"  # Fix and try again
        elif self.should_decompose():
            return "decompose"  # Break into smaller pieces
        elif self.should_escalate_to_cloud():
            return "alternative"  # Try different approach
        else:
            return "ask_user"  # Last resort only
```

**Triggered**: Automatically when quality gates fail

### Mechanism 3: DebuggerAgent Specialist (M4)

**File**: `specialists/debugger_agent.py`

**Autonomous workflow**:
```
ERROR DETECTED
  ↓ (automatic)
DebuggerAgent selected
  ↓ (automatic)
Workflow executes:
  1. analyze_error (reads error message)
  2. diagnose_root_cause (finds real problem)
  3. isolate_problem (locates exact line)
  4. apply_fix (minimal targeted fix)
  5. validate_fix (re-run to verify)
  ↓ (automatic)
Returns verified fix
```

**Triggered**: Automatically when errors detected

### Mechanism 4: InsightEngine (M5)

**File**: `insight_engine.py`

**Learns from errors**:
```python
# After fixing an error, automatically stores:
insight = InsightEngine.generate_error_fix_insight(
    error_type="AttributeError",
    error_message="'NoneType' has no attribute 'name'",
    fix_description="Added None check before accessing attribute"
)
# Stored in knowledge.db

# Next time similar error occurs:
# → Agent recalls this insight automatically
# → Applies same fix faster
```

**Triggered**: Automatically after every error fix

### Mechanism 5: Self-Validation Loop (M2)

**File**: `agent.py:204-282`

**Continuous self-checking**:
```python
while not all_quality_gates_pass():
    execute_task()
    run_quality_gates()  # AUTONOMOUS CHECK

    if gates_fail:
        diagnose_and_fix()  # AUTONOMOUS FIX
        re_run_gates()      # AUTONOMOUS VERIFY
```

**No user intervention needed!**

---

## Example: Autonomous Self-Review in Action

### Scenario: User asks for REST API

```
User: "Build a REST API with auth"
```

**GAIA Code's autonomous process**:

```
[Step 1] Write initial code
  ✓ Created auth.py, models.py, main.py

[Step 2] Run quality gates (AUTONOMOUS)
  ✓ Syntax: PASS
  ✓ Imports: PASS
  ✗ Tests: FAIL (3/5 tests failing)

[Step 3] Auto-select DebuggerAgent (AUTONOMOUS)
  Specialist: DebuggerAgent
  Task: Fix failing tests

[Step 4] DebuggerAgent analyzes (AUTONOMOUS)
  Error: AssertionError in test_login
  Root cause: Wrong status code (400 vs 401)
  Location: auth.py:45

[Step 5] DebuggerAgent fixes (AUTONOMOUS)
  Applied: Changed return status from 400 to 401

[Step 6] Re-run quality gates (AUTONOMOUS)
  ✓ Syntax: PASS
  ✓ Imports: PASS
  ✓ Tests: PASS (5/5 passing)

[Step 7] Store insight (AUTONOMOUS)
  Learned: "For unauthorized access, use 401 not 400"
  Category: error_fix
  Triggers: ["auth", "status code", "401"]

[Step 8] Return verified result
  ✓ All quality gates passed
  ✓ Code is correct
  ✓ Tests are passing
```

**Total user prompts needed**: 1 (the initial request)
**Everything else**: AUTONOMOUS

---

## vs Claude Code

### What Claude Code Requires

```
# You (reviewing Claude's code):
"Test the code"
"There are syntax errors - fix them"
"The tests are failing - debug them"
"Check for import errors"
"Verify all tests pass"
"Review the code for issues"
```

**6 manual prompts for basic quality checks!**

### What GAIA Code Does

```
# You:
"Build a REST API"

# GAIA Code (autonomously):
- Writes code
- Tests code
- Finds errors
- Fixes errors
- Verifies fixes
- Returns when verified
```

**1 prompt total - everything else is autonomous!**

---

## Autonomous Capabilities Summary

### ✅ GAIA Code Autonomously:

1. **Reviews code** - Quality gates after every change
2. **Detects errors** - Syntax, import, test failures
3. **Diagnoses issues** - DebuggerAgent finds root causes
4. **Fixes problems** - Applies targeted fixes
5. **Verifies fixes** - Re-runs gates to confirm
6. **Learns from errors** - Stores insights for future
7. **Escalates smartly** - retry → decompose → alternative → ask
8. **Runs continuously** - Until verified complete

### ❌ Claude Code Requires:

1. **Manual review** - User must prompt "test it"
2. **Manual error detection** - User must notice failures
3. **Manual debugging** - User must prompt "fix this error"
4. **Manual verification** - User must prompt "test again"
5. **No learning** - Forgets errors across sessions
6. **Stops frequently** - Waits for user prompts
7. **No autonomous recovery** - User drives every step

---

## Architecture Design for Autonomy

### Quality-Driven Completion (M2)

**Core principle**: Agent CANNOT say "done" without proof

```python
def process_query(task):
    while not verified_complete:
        execute(task)

        # AUTONOMOUS REVIEW
        gates = run_quality_gates()

        if not gates.all_passed:
            # AUTONOMOUS FIX
            auto_fix_failures(gates)
        else:
            return verified_result
```

**This is the core difference from Claude Code!**

### Continuous Execution (M2)

**Core principle**: Run until verified, not until step limit

```python
# Claude Code:
for step in range(max_steps):
    do_something()
    if step == max_steps:
        stop()  # Might not be done!

# GAIA Code:
while not verified_complete():  # Quality-driven
    do_something()
    verify()
    if verified:
        break  # Actually done!
```

### Specialist Auto-Selection (M4)

**Core principle**: Right expert for right problem

```python
# Error detected:
if "test failure" in error:
    specialist = auto_select("TestingAgent")
elif "security" in error:
    specialist = auto_select("SecurityAgent")
elif "error" in error:
    specialist = auto_select("DebuggerAgent")

# Specialist handles autonomously
```

---

## Confirmation

### Question: Does GAIA Code autonomously review and fix its own code?

**Answer**: ✅ **YES - This is the core design principle!**

**Evidence**:
1. Quality gates run automatically (M2)
2. Escalation ladder recovers automatically (M2)
3. DebuggerAgent fixes errors automatically (M4)
4. InsightEngine learns automatically (M5)
5. Continuous execution until verified (M2)

**Key Difference**:
- Claude Code: **Manual** review (user prompts each step)
- GAIA Code: **Autonomous** review (built into architecture)

---

## How to Verify

### Test 1: Write Buggy Code

```bash
gaia code "Create a function with a deliberate syntax error"
```

**GAIA Code will**:
1. Write code (with syntax error)
2. Run SyntaxGate (AUTONOMOUS)
3. Detect syntax error (AUTONOMOUS)
4. Fix syntax error (AUTONOMOUS)
5. Re-run SyntaxGate (AUTONOMOUS)
6. Return corrected code

**No manual "fix it" prompt needed!**

### Test 2: Failing Tests

```bash
gaia code "Create a function that returns wrong value, with tests"
```

**GAIA Code will**:
1. Write function (with bug)
2. Write tests
3. Run TestGate (AUTONOMOUS)
4. Tests fail (AUTONOMOUS DETECTION)
5. DebuggerAgent diagnoses (AUTONOMOUS)
6. Fix applied (AUTONOMOUS)
7. Tests re-run (AUTONOMOUS)
8. Return when tests pass

**No manual debugging needed!**

### Test 3: Complex Error

```bash
gaia code "Build a web scraper"
```

**GAIA Code will**:
1. Write scraper
2. Quality gates detect issues (AUTONOMOUS)
3. Escalation ladder activates (AUTONOMOUS)
   - Retry 1: Fix import errors
   - Retry 2: Fix test failures
   - If still failing: Decompose into subtasks
4. Each subtask verified independently
5. Return complete, verified scraper

**Autonomous from start to finish!**

---

## Implementation References

### Code Locations

**Quality Gates**: `src/gaia/agents/gaia_code/quality_gates.py:308-367`
```python
class QualityGateRunner:
    def run_all(self, paths):
        # Automatically runs all gates
        # No manual triggering needed
```

**Escalation**: `src/gaia/agents/gaia_code/quality_gates.py:369-414`
```python
class EscalationLadder:
    def get_action(self):
        # Automatically determines next action
        # retry → decompose → alternative → ask_user
```

**Auto-Fix**: `src/gaia/agents/gaia_code/agent.py:204-282`
```python
def _execute_with_quality_gates(self, query, root_task):
    while not all_gates_pass:
        execute()
        check()  # AUTONOMOUS
        if fail:
            auto_fix()  # AUTONOMOUS
```

**Autonomous Debugging**: `src/gaia/agents/gaia_code/specialists/debugger_agent.py`
```python
class DebuggerAgent:
    workflow = [
        "analyze_error",      # AUTONOMOUS
        "diagnose_root_cause", # AUTONOMOUS
        "isolate_problem",     # AUTONOMOUS
        "apply_fix",           # AUTONOMOUS
        "validate_fix",        # AUTONOMOUS
    ]
```

---

## Summary

### GAIA Code Design Principle

**"Autonomous verification and recovery until proven correct"**

NOT:
- ❌ Write code → Stop → Wait for user
- ❌ User reviews → User tests → User debugs

BUT:
- ✅ Write code → Auto-test → Auto-fix → Auto-verify → Done

### Key Difference

**Claude Code**: Reactive (waits for user prompts)
**GAIA Code**: Proactive (automatically reviews and fixes)

### Why This Matters

**Claude Code workflow** (5+ prompts):
1. "Build API" → Writes code
2. "Test it" → Runs tests
3. "Fix errors" → Applies fixes
4. "Test again" → Verifies
5. "Check syntax" → Reviews

**GAIA Code workflow** (1 prompt):
1. "Build API" → Does everything autonomously

**Result**: 5x fewer manual interventions!

---

## Confirmation

✅ **YES - GAIA Code has autonomous self-review**

**Built-in mechanisms**:
1. Quality gates (automatic code review)
2. Escalation ladder (automatic recovery)
3. Specialist agents (automatic expert selection)
4. Continuous execution (runs until verified)
5. Insight learning (improves over time)

**No manual prompting needed for**:
- Testing code
- Finding errors
- Fixing bugs
- Verifying fixes
- Reviewing quality

**This is the core architectural difference that makes GAIA Code autonomous!**
