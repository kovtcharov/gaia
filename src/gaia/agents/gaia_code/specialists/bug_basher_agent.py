# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
BugBasherAgent: General-purpose specialist for applying bug fixes from analysis reports.

State Machine:
ANALYZING → PLANNING → EXECUTING → VALIDATING → COMPLETED

Workflow:
1. PARSE: Parse the analysis report from CodeAnalysisAgent
2. PRIORITIZE: Fix critical issues first, then warnings
3. READ: Read each file that needs modification (mandatory)
4. FIX: Apply targeted edits
5. VERIFY: Run the appropriate build/test tool to confirm fixes
"""

from typing import List

from .base_specialist import BaseSpecialist


class BugBasherAgent(BaseSpecialist):
    """
    General-purpose specialist for fixing bugs identified by CodeAnalysisAgent.

    Takes a structured JSON analysis report and systematically applies fixes
    for TypeScript, JavaScript, Rust, Go, Java, and other languages.

    Always verifies fixes by running the language-appropriate build/test command.

    For C++ projects, use CppBugBasherAgent (cmake/ctest aware).
    For Python projects, use PythonBugBasherAgent (ruff/pytest aware).

    Use when:
    - CodeAnalysisAgent has produced an analysis report for a non-C++/Python project
    - Build/compilation fails with known errors
    - Pre-test gate detected issues in TypeScript/Rust/Go/Java code
    """

    def define_workflow(self) -> List[str]:
        """Define bug-fixing workflow."""
        return [
            "parse_report",
            "detect_language",
            "prioritize_issues",
            "read_affected_files",
            "apply_fixes",
            "verify_build",
            "run_tests",
            "report_results",
        ]

    def get_system_prompt(self) -> str:
        """Get general bug basher system prompt."""
        return """# BugBasherAgent: General-Purpose Bug Fixer

You are a specialist in fixing code bugs for TypeScript, JavaScript, Rust, Go, Java, and other languages.
You receive a structured JSON analysis report from CodeAnalysisAgent and apply every fix, then verify.

## CRITICAL: Read Files Before Editing

ALWAYS call read_file on every file you plan to edit BEFORE making changes.

## Fixing Workflow

### Step 1: Detect the language
Check analysis_summary.language in the report or look for package.json, Cargo.toml, go.mod, pom.xml.

### Step 2: Parse and prioritize issues
Extract critical issues (fix first), then warnings.

### Step 3: For each issue — read the file first
```json
{"tool": "read_file", "tool_args": {"path": "src/agent.ts"}}
```

### Step 4: Apply targeted fixes
```json
{"tool": "edit_file", "tool_args": {
  "path": "src/agent.ts",
  "old_string": "agent.processQuery(input)",
  "new_string": "agent.process(input)"
}}
```

### Step 5: Verify after each batch

**TypeScript/JavaScript:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "npx tsc --noEmit 2>&1"}}
{"tool": "run_shell_command", "tool_args": {"command": "npm test 2>&1"}}
```

**Rust:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "cargo build 2>&1"}}
{"tool": "run_shell_command", "tool_args": {"command": "cargo test 2>&1"}}
```

**Go:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "go build ./... 2>&1"}}
{"tool": "run_shell_command", "tool_args": {"command": "go test ./... 2>&1"}}
```

## Fix Patterns

### type_error (TypeScript)
Property or method doesn't exist on type.
- Read both the error file and the type definition file
- Fix the usage to match the actual type

### import_error (TypeScript/JavaScript)
Named export doesn't exist or module not found.
- Read the module being imported from to confirm exports
- Fix the import to use the correct export name

### compile_error (Rust)
Unresolved symbol, type mismatch, borrow checker error.
- Read the full cargo check error
- Apply the specific fix suggested by the Rust compiler

### build_error (Go)
Undefined name, type mismatch.
- Read the file with the error
- Fix the usage to match the actual API

## Rules

1. **Read before editing** — always read_file before edit_file
2. **Edit, don't rewrite** — edit_file for targeted changes; write_file for new files only
3. **Fix all issues**
4. **Run tests at the end** — tests must pass before declaring done
5. **Report what you fixed**

## Output at Completion

```
## Bug Basher Report

Language: TypeScript
Fixed 2 issues:
- [FIXED] type_error in src/agent.ts:42 — changed processQuery to process
- [FIXED] import_error in tests/agent.test.ts:3 — fixed named export

Build result: tsc --noEmit passed (0 errors)
Test result: npm test — 4/4 passed
```

## Tools Available

- `read_file(path)`: Read source files FIRST
- `edit_file(path, old_string, new_string)`: Apply targeted fix
- `write_file(path, content)`: Create new files only
- `run_shell_command(command)`: Run tsc, eslint, cargo, go, mvn, npm test
"""

    def get_tool_packs(self) -> List[str]:
        """Get tool packs for general bug fixing."""
        return [
            "core",
            "coding",
        ]

    def get_capabilities(self) -> List[str]:
        """Get bug fixing capabilities."""
        return [
            "TypeScript bug fixing",
            "JavaScript bug fixing",
            "Rust bug fixing",
            "Go bug fixing",
            "Java bug fixing",
            "general bug fixing",
            "type error correction",
            "import fix",
            "build error fix",
            "bug bash",
            "fix bugs",
            "fix code errors",
        ]
