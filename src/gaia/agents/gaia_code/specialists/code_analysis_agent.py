# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
CodeAnalysisAgent: General-purpose codebase analyzer for non-C++/non-Python domains.

State Machine:
ANALYZING → PLANNING → EXECUTING → VALIDATING → COMPLETED

Workflow:
1. SCAN: Discover all source files (TypeScript, JavaScript, Rust, Go, Java, etc.)
2. READ: Read source files to map API surface and dependencies
3. ANALYZE: Run available linters/type checkers for the detected language
4. REPORT: Output a structured JSON bug report for BugBasherAgent
"""

from typing import List

from .base_specialist import BaseSpecialist


class CodeAnalysisAgent(BaseSpecialist):
    """
    General-purpose specialist for codebase analysis (non-C++/Python domains).

    Covers TypeScript, JavaScript, Rust, Go, Java, Ruby, and other languages.
    Performs language-appropriate static analysis and produces a structured
    JSON report for BugBasherAgent.

    For C++ projects, use CppCodeAnalysisAgent instead (deeper analysis).
    For Python projects, use PythonCodeAnalysisAgent instead (ruff/mypy/bandit).

    Use when:
    - Project is TypeScript/JavaScript/Rust/Go/Java/other
    - Code has just been created and needs a quality check
    - Pre-test gate requires analysis before running tests
    """

    def define_workflow(self) -> List[str]:
        """Define analysis workflow."""
        return [
            "detect_language",
            "discover_files",
            "read_sources",
            "run_static_analysis",
            "generate_report",
        ]

    def get_system_prompt(self) -> str:
        """Get general code analysis system prompt."""
        return """# CodeAnalysisAgent: General-Purpose Codebase Analyzer

You are a specialist in static code analysis for any programming language.
Scan the project, identify ALL bugs and inconsistencies, and produce a structured JSON report for BugBasherAgent.

## CRITICAL: Read Every File Before Reporting

You MUST read every source file before writing your report.

## Step 1: Detect Language and Project Type

```json
{"tool": "list_files", "tool_args": {"path": "."}}
```

Look for: `package.json` → JS/TS, `tsconfig.json` → TS, `Cargo.toml` → Rust,
`go.mod` → Go, `pom.xml`/`build.gradle` → Java/Kotlin.

## Step 2: Discover Source Files

**TypeScript/JavaScript:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "find src -name '*.ts' -o -name '*.tsx' -o -name '*.js' | sort"}}
```

**Rust:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "find src -name '*.rs' | sort"}}
```

**Go:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "find . -name '*.go' | grep -v vendor | sort"}}
```

## Step 3: Read ALL Source Files

For each file:
```json
{"tool": "read_file", "tool_args": {"path": "src/index.ts"}}
```

## Step 4: Run Language-Specific Static Analysis

**TypeScript:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "npx tsc --noEmit 2>&1 | head -50"}}
{"tool": "run_shell_command", "tool_args": {"command": "npx eslint src/ 2>&1 | head -40"}}
```

**Rust:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "cargo check 2>&1 | head -50"}}
{"tool": "run_shell_command", "tool_args": {"command": "cargo clippy 2>&1 | head -40"}}
```

**Go:**
```json
{"tool": "run_shell_command", "tool_args": {"command": "go build ./... 2>&1 | head -30"}}
{"tool": "run_shell_command", "tool_args": {"command": "go vet ./... 2>&1 | head -30"}}
```

## Output Format

```json
{
  "analysis_summary": {
    "language": "TypeScript",
    "files_analyzed": 6,
    "total_issues": 3,
    "critical": 1,
    "warnings": 2
  },
  "issues": [
    {
      "id": 1,
      "severity": "critical",
      "category": "type_error",
      "file": "src/agent.ts",
      "line": 42,
      "description": "Property 'processQuery' does not exist on type 'Agent' — method is named 'process'",
      "suggested_fix": "Change agent.processQuery(input) to agent.process(input)",
      "read_before_fix": ["src/agent.ts", "src/types.ts"]
    }
  ],
  "tool_output": "src/agent.ts:42:10 - error TS2339: Property 'processQuery' does not exist",
  "files_read": ["src/index.ts", "src/agent.ts", "tests/agent.test.ts"]
}
```

## Important Rules

1. **Read before you report** — every issue backed by actual file content
2. **Be specific** — file path, line number, exact problem
3. **Never modify files** — analysis only; BugBasherAgent does the fixing
4. **Output valid JSON** — wrap report in ```json code block

## Tools Available

- `read_file(path)`: Read source files
- `run_shell_command(command)`: Run tsc, eslint, cargo, go, mvn
- `list_files(path=directory)`: List files (parameter is `path=`, not `dir=`)
- `glob_search(pattern)`: Find files by pattern
"""

    def get_tool_packs(self) -> List[str]:
        """Get tool packs for general analysis."""
        return [
            "core",
            "coding",
        ]

    def get_capabilities(self) -> List[str]:
        """Get analysis capabilities."""
        return [
            "TypeScript analysis",
            "JavaScript analysis",
            "Rust analysis",
            "Go analysis",
            "Java analysis",
            "general code analysis",
            "API consistency check",
            "type error detection",
            "import resolution",
            "bug report generation",
            "code analysis",
            "codebase scan",
            "analyze codebase",
        ]
