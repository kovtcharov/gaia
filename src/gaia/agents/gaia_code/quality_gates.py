# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Quality Gates: The Core Differentiator

Quality gates are automated checks that verify code works BEFORE the agent
declares "done". This is what separates GAIA Code from other agents.

The agent CANNOT complete a task until all enabled quality gates pass.

M2: Quality Gates + Continuous Execution
- Basic gates: syntax, imports, tests
- Gate-driven completion (not step-driven)
- Escalation ladder: retry → decompose → cloud → ask user
"""

import ast
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GateResult:
    """Result of a quality gate check."""

    gate_name: str
    passed: bool
    message: str
    details: Optional[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class QualityGate:
    """Base class for quality gates."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, context: Dict) -> GateResult:
        """Run the quality gate check."""
        raise NotImplementedError


class SyntaxGate(QualityGate):
    """
    Syntax Check Gate: Verify all code files parse without errors.

    Uses AST parsing to check Python syntax.
    For other languages, uses appropriate parsers.
    """

    def check(self, context: Dict) -> GateResult:
        """Check syntax of all code files in context."""
        files = context.get("files", [])
        if not files:
            return GateResult(
                gate_name="Syntax",
                passed=True,
                message="No files to check",
            )

        errors = []
        for file_path in files:
            if not file_path.endswith(".py"):
                continue  # Skip non-Python files for now

            try:
                with open(file_path, "r") as f:
                    code = f.read()
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")

        if errors:
            return GateResult(
                gate_name="Syntax",
                passed=False,
                message=f"Syntax errors in {len(errors)} file(s)",
                errors=errors,
            )

        return GateResult(
            gate_name="Syntax",
            passed=True,
            message=f"All {len(files)} file(s) syntax valid",
        )


class ImportGate(QualityGate):
    """
    Import Check Gate: Verify all imports resolve successfully.

    Checks that imported modules exist and are accessible.
    """

    def check(self, context: Dict) -> GateResult:
        """Check imports in all code files."""
        files = context.get("files", [])
        if not files:
            return GateResult(
                gate_name="Imports",
                passed=True,
                message="No files to check",
            )

        errors = []
        for file_path in files:
            if not file_path.endswith(".py"):
                continue

            try:
                with open(file_path, "r") as f:
                    code = f.read()

                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            self._check_import(alias.name, file_path, errors)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            self._check_import(node.module, file_path, errors)

            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")

        if errors:
            return GateResult(
                gate_name="Imports",
                passed=False,
                message=f"Import errors in {len(errors)} case(s)",
                errors=errors,
            )

        return GateResult(
            gate_name="Imports",
            passed=True,
            message="All imports resolve successfully",
        )

    def _check_import(self, module_name: str, file_path: str, errors: List[str]):
        """Check if a module can be imported."""
        try:
            # Try to find the module spec
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                errors.append(f"{file_path}: Cannot import '{module_name}'")
        except (ImportError, ModuleNotFoundError, ValueError):
            errors.append(f"{file_path}: Cannot import '{module_name}'")


class TestGate(QualityGate):
    """
    Test Check Gate: Run tests and verify they pass.

    Auto-detects test framework (pytest, jest, etc.) and runs tests.
    """

    def check(self, context: Dict) -> GateResult:
        """Run tests in the project."""
        project_dir = context.get("project_dir", ".")

        # Try to find test files
        test_files = self._find_test_files(project_dir)

        if not test_files:
            return GateResult(
                gate_name="Tests",
                passed=True,
                message="No tests found (skipping)",
            )

        # Detect test framework and run
        if self._has_pytest(test_files):
            return self._run_pytest(project_dir, test_files)
        elif self._has_jest(project_dir):
            return self._run_jest(project_dir)
        else:
            return GateResult(
                gate_name="Tests",
                passed=True,
                message="No test framework detected (skipping)",
            )

    def _find_test_files(self, project_dir: str) -> List[str]:
        """Find test files in project."""
        test_files = []
        project_path = Path(project_dir)

        # Python tests
        for pattern in ["test_*.py", "*_test.py"]:
            test_files.extend([str(p) for p in project_path.rglob(pattern)])

        # JavaScript tests
        for pattern in ["*.test.js", "*.spec.js", "*.test.ts", "*.spec.ts"]:
            test_files.extend([str(p) for p in project_path.rglob(pattern)])

        return test_files

    def _has_pytest(self, test_files: List[str]) -> bool:
        """Check if project uses pytest."""
        return any(f.endswith(".py") for f in test_files)

    def _has_jest(self, project_dir: str) -> bool:
        """Check if project uses jest."""
        package_json = Path(project_dir) / "package.json"
        return package_json.exists()

    def _run_pytest(self, project_dir: str, test_files: List[str]) -> GateResult:
        """Run pytest tests."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "-v", "--tb=short"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return GateResult(
                    gate_name="Tests",
                    passed=True,
                    message=f"All tests passed ({len(test_files)} file(s))",
                    details=result.stdout,
                )
            else:
                errors = self._parse_pytest_errors(result.stdout)
                return GateResult(
                    gate_name="Tests",
                    passed=False,
                    message=f"Tests failed",
                    details=result.stdout,
                    errors=errors,
                )

        except subprocess.TimeoutExpired:
            return GateResult(
                gate_name="Tests",
                passed=False,
                message="Tests timed out (>60s)",
            )
        except Exception as e:
            return GateResult(
                gate_name="Tests",
                passed=False,
                message=f"Error running tests: {str(e)}",
            )

    def _run_jest(self, project_dir: str) -> GateResult:
        """Run jest tests."""
        try:
            result = subprocess.run(
                ["npm", "test"],
                cwd=project_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                return GateResult(
                    gate_name="Tests",
                    passed=True,
                    message="All tests passed",
                    details=result.stdout,
                )
            else:
                return GateResult(
                    gate_name="Tests",
                    passed=False,
                    message="Tests failed",
                    details=result.stdout,
                )

        except subprocess.TimeoutExpired:
            return GateResult(
                gate_name="Tests",
                passed=False,
                message="Tests timed out (>60s)",
            )
        except Exception as e:
            return GateResult(
                gate_name="Tests",
                passed=False,
                message=f"Error running tests: {str(e)}",
            )

    def _parse_pytest_errors(self, output: str) -> List[str]:
        """Parse pytest output to extract error messages."""
        errors = []
        for line in output.split("\n"):
            if "FAILED" in line or "ERROR" in line:
                errors.append(line.strip())
        return errors


class QualityGateRunner:
    """
    Runs all quality gates and enforces gate-driven completion.

    The agent CANNOT declare "done" until all enabled gates pass.
    """

    def __init__(self):
        self.gates = {
            "syntax": SyntaxGate(),
            "imports": ImportGate(),
            "tests": TestGate(),
        }

    def run_all(self, paths: List[str], **kwargs) -> Dict[str, "GateResult"]:
        """
        Run all enabled quality gates.

        Args:
            paths: List of file or directory paths to check
            **kwargs: Additional arguments for gates

        Returns:
            Dict mapping gate name to GateResult
        """
        results = {}

        for gate_name, gate in self.gates.items():
            if not gate.enabled:
                continue

            result = gate.check(paths, **kwargs)
            results[gate_name] = result

        return results

    def all_passed(self, results: Dict[str, "GateResult"]) -> bool:
        """
        Check if all gates passed.

        Args:
            results: Dict of gate results from run_all()

        Returns:
            True if all gates passed
        """
        return all(r.passed for r in results.values())

    def enable_gate(self, gate_name: str):
        """Enable a quality gate."""
        if gate_name in self.gates:
            self.gates[gate_name].enabled = True

    def disable_gate(self, gate_name: str):
        """Disable a quality gate."""
        if gate_name in self.gates:
            self.gates[gate_name].enabled = False

    def format_results(self, results: List[GateResult]) -> str:
        """Format gate results for display."""
        lines = ["Quality Gate Results:", "=" * 50]

        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"{status} {result.gate_name}: {result.message}")

            if result.errors:
                for error in result.errors[:5]:  # Show first 5 errors
                    lines.append(f"  - {error}")
                if len(result.errors) > 5:
                    lines.append(f"  ... and {len(result.errors) - 5} more errors")

        lines.append("=" * 50)
        return "\n".join(lines)


class EscalationLadder:
    """
    Handles the escalation ladder when quality gates fail.

    Ladder: retry (2x) → decompose → cloud → ask user
    """

    def __init__(self):
        self.retry_count = 0
        self.max_retries = 2

    def should_retry(self) -> bool:
        """Check if we should retry."""
        return self.retry_count < self.max_retries

    def should_decompose(self) -> bool:
        """Check if we should decompose into smaller subtasks."""
        return self.retry_count >= self.max_retries

    def should_escalate_to_cloud(self) -> bool:
        """Check if we should escalate to cloud LLM."""
        return self.retry_count >= self.max_retries + 1

    def should_ask_user(self) -> bool:
        """Check if we should ask user for help."""
        return self.retry_count >= self.max_retries + 2

    def increment(self):
        """Increment retry count."""
        self.retry_count += 1

    def reset(self):
        """Reset retry count."""
        self.retry_count = 0

    def get_action(self) -> str:
        """Get the current escalation action."""
        if self.should_retry():
            return "retry"
        elif self.should_decompose():
            return "decompose"
        elif self.should_escalate_to_cloud():
            return "cloud"
        else:
            return "ask_user"
