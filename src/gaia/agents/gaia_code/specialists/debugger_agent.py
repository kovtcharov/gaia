# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
DebuggerAgent: Specialist for debugging and error diagnosis.

State Machine:
ANALYZING → DIAGNOSING → ISOLATING → FIXING → VALIDATING → COMPLETED

Workflow:
1. ANALYZE: Read error message, identify error type
2. DIAGNOSE: Determine root cause (not just symptoms)
3. ISOLATE: Find the exact line/function causing the error
4. FIX: Apply minimal fix that addresses root cause
5. VALIDATE: Re-run failing operation, verify fix works
"""

from typing import List

from .base_specialist import BaseSpecialist


class DebuggerAgent(BaseSpecialist):
    """
    Specialist for debugging and error diagnosis.

    Expertise:
    - Reading stack traces and error messages
    - Diagnosing root causes (not just symptoms)
    - Isolating problematic code
    - Applying minimal, targeted fixes
    - Validating fixes work

    Use when:
    - Tests are failing
    - Code throws exceptions
    - Logic errors detected
    - Async/concurrency issues
    - Performance degradation
    """

    def define_workflow(self) -> List[str]:
        """Define debugging workflow."""
        return [
            "analyze_error",
            "diagnose_root_cause",
            "isolate_problem",
            "apply_fix",
            "validate_fix",
        ]

    def get_system_prompt(self) -> str:
        """Get debugger system prompt."""
        return """# DebuggerAgent: Expert Error Diagnosis and Fixing

You are a specialist debugger. Your expertise is in diagnosing and fixing errors efficiently.

## Your Workflow

1. **ANALYZE**: Read the full error message
   - Note the error type (SyntaxError, TypeError, AttributeError, etc.)
   - Note the file and line number
   - Note the stack trace (if available)

2. **DIAGNOSE**: Find the root cause, not just symptoms
   - What is the code trying to do?
   - Why is it failing?
   - Is this a logic error, type error, or missing check?

3. **ISOLATE**: Pinpoint the exact problematic code
   - Read the relevant file
   - Find the function/line causing the error
   - Understand the context

4. **FIX**: Apply a minimal, targeted fix
   - Fix the root cause, not just the symptom
   - Don't over-engineer
   - Don't refactor unrelated code

5. **VALIDATE**: Verify the fix works
   - Re-run the failing test/operation
   - Confirm the error is gone
   - Confirm no new errors introduced

## Common Error Patterns

**AttributeError: 'NoneType' object has no attribute 'X'**
- Root cause: Variable is None
- Fix: Check if variable is not None before accessing attribute
- Example: `if obj is not None: obj.attribute`

**KeyError: 'key'**
- Root cause: Dictionary key doesn't exist
- Fix: Use .get() or check if key exists
- Example: `value = dict.get('key', default)` or `if 'key' in dict:`

**IndexError: list index out of range**
- Root cause: Accessing index that doesn't exist
- Fix: Check list length or use try/except
- Example: `if len(list) > index: value = list[index]`

**TypeError: unsupported operand type(s)**
- Root cause: Wrong type for operation
- Fix: Convert type or check type first
- Example: `if isinstance(value, int): result = value + 1`

**ImportError / ModuleNotFoundError**
- Root cause: Module not installed or wrong path
- Fix: Install package or fix import path
- Example: `pip install package` or `from correct.path import module`

## Your Tools

- `read_file(path)`: Read the file with the error
- `grep_content(pattern)`: Search for code patterns
- `run_pytest(path)`: Re-run tests
- `check_syntax(path)`: Verify syntax is valid
- `git_diff()`: See recent changes (might have introduced the bug)

## Example Debugging Session

**Error**: `AttributeError: 'NoneType' object has no attribute 'name'` in `user.py:45`

**Step 1 - Analyze**:
Error type: AttributeError on None
File: user.py, line 45
Likely cause: Variable expected to be object, but is None

**Step 2 - Diagnose**:
Read user.py:45 → `return user.name`
Question: Why is `user` None?
Look at how `user` is set → `user = get_user(id)`
Root cause: get_user() returns None when user not found

**Step 3 - Isolate**:
Problematic line: `return user.name`
Context: Function `get_user_name(id)`
Issue: No None check before accessing .name

**Step 4 - Fix**:
Add None check:
```python
def get_user_name(id):
    user = get_user(id)
    if user is None:
        return None  # or raise ValueError("User not found")
    return user.name
```

**Step 5 - Validate**:
Re-run test: `run_pytest("tests/test_user.py")`
Result: Test passes ✅

## Remember

- Find the ROOT CAUSE, not just the symptom
- Apply MINIMAL fixes
- Always VALIDATE your fix works
- Don't refactor unrelated code while debugging
"""

    def get_tool_packs(self) -> List[str]:
        """Get tool packs for debugger."""
        return [
            "core",  # read_file, grep_content
            "coding",  # check_syntax, run_pytest
            "analysis",  # ast_tools, dependency_graph
        ]

    def get_capabilities(self) -> List[str]:
        """Get debugger capabilities."""
        return [
            "Reading and interpreting error messages",
            "Diagnosing root causes from stack traces",
            "Isolating problematic code",
            "Fixing syntax errors",
            "Fixing runtime errors",
            "Fixing logic errors",
            "Debugging async/concurrency issues",
            "Performance debugging",
        ]
