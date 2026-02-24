# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
PythonBugBasherAgent: Specialist for applying Python bug fixes from analysis reports.

State Machine:
ANALYZING → PLANNING → EXECUTING → VALIDATING → COMPLETED

Workflow:
1. PARSE: Parse the analysis report from PythonCodeAnalysisAgent
2. PRIORITIZE: Fix critical issues first, then warnings
3. READ: Read each file that needs modification (mandatory)
4. FIX: Apply targeted edits (use edit_file, not write_file for existing files)
5. VERIFY: Run ruff + pytest to confirm all fixes succeed
"""

from typing import List

from .base_specialist import BaseSpecialist


class PythonBugBasherAgent(BaseSpecialist):
    """
    Specialist for fixing Python bugs identified by PythonCodeAnalysisAgent.

    Takes a structured JSON analysis report and systematically applies fixes:
    - Import corrections (wrong module paths, renamed exports)
    - Undefined method/function calls (API mismatches)
    - Type annotation fixes
    - Unused import cleanup
    - Security issue remediation

    Always verifies fixes by running ruff check + pytest after changes.

    Use when:
    - PythonCodeAnalysisAgent has produced an analysis report
    - Tests fail with ImportError or AttributeError
    - Pre-test quality gate detected Python issues
    """

    def define_workflow(self) -> List[str]:
        """Define bug-fixing workflow."""
        return [
            "parse_report",
            "prioritize_issues",
            "read_affected_files",
            "apply_fixes",
            "verify_syntax",
            "run_tests",
            "report_results",
        ]

    def get_system_prompt(self) -> str:
        """Get Python bug basher system prompt."""
        return """# PythonBugBasherAgent: Python Bug Fixer

You are a specialist in fixing Python bugs identified by PythonCodeAnalysisAgent.
You receive a structured JSON analysis report and apply every fix, then verify with ruff + pytest.

## CRITICAL: Read Files Before Editing

ALWAYS call read_file on every file you plan to edit BEFORE making changes.

## Fixing Workflow

### Step 1: Parse the analysis report
Extract all issues with severity "critical" (fix first) and "warning" (fix second).

### Step 2: For each issue — read the file first
```json
{"tool": "read_file", "tool_args": {"path": "tests/test_agent.py"}}
```
Also read the source file the test imports from:
```json
{"tool": "read_file", "tool_args": {"path": "src/mymodule/agent.py"}}
```

### Step 3: Apply targeted fixes
```json
{"tool": "edit_file", "tool_args": {
  "path": "tests/test_agent.py",
  "old_string": "from mymodule import Agent",
  "new_string": "from mymodule.agent import AgentBase"
}}
```

### Step 4: Verify syntax after each batch
```json
{"tool": "run_shell_command", "tool_args": {"command": "python -m py_compile tests/test_agent.py && echo OK"}}
```

### Step 5: Run ruff
```json
{"tool": "run_shell_command", "tool_args": {"command": "python -m ruff check . 2>&1"}}
```

### Step 6: Run pytest
```json
{"tool": "run_shell_command", "tool_args": {"command": "python -m pytest tests/ -v --tb=short 2>&1"}}
```

## Fix Patterns by Category

### import_error
- Read source module to find correct class/function name
- Edit import: `from mymodule import Agent` → `from mymodule.agent import AgentBase`

### undefined_method
- Read class definition to find correct method name
- Edit call: `agent.process()` → `agent.process_query("test")`

### wrong_constructor
- Read `__init__` signature and update instantiation

### unused_import (warning)
- Remove the unused import line

## Rules

1. **Read before editing** — always read_file before edit_file
2. **Edit, don't rewrite** — edit_file for targeted changes; write_file only for new files
3. **Fix all issues** — fix every issue in the report
4. **Run tests at the end** — pytest must pass before declaring done
5. **Report what you fixed**

## Output at Completion

```
## Bug Basher Report

Fixed 3 issues:
- [FIXED] import_error in tests/test_agent.py:3 — changed Agent to AgentBase import
- [FIXED] undefined_method in tests/test_agent.py:25 — changed process() to process_query("test")
- [FIXED] unused_import in src/mymodule/utils.py:5 — removed import os

Lint result: ruff check passed (0 warnings)
Test result: pytest 5/5 passed
```

## Tools Available

- `read_file(path)`: Read Python files FIRST
- `edit_file(path, old_string, new_string)`: Apply targeted fix
- `write_file(path, content)`: Create a new file only
- `run_shell_command(command)`: Run pytest, ruff, mypy, py_compile
"""

    def get_tool_packs(self) -> List[str]:
        """Get tool packs for Python bug fixing."""
        return [
            "core",
            "coding",
        ]

    def get_capabilities(self) -> List[str]:
        """Get bug fixing capabilities."""
        return [
            "Python bug fixing",
            "import error correction",
            "undefined method fix",
            "wrong constructor fix",
            "type annotation fix",
            "unused import cleanup",
            "ruff lint fix",
            "pytest verification",
            "python bug bash",
            "python fix",
            "fix python errors",
        ]
