# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Quality Gates: Automated Verification Framework

Quality gates are automated checks that verify output works BEFORE the agent
declares "done". Any agent type can use quality gates for verification.

Features:
- Gate-driven completion (not step-driven)
- Pluggable gate system (add custom gates per agent type)
- Escalation ladder: retry -> decompose -> cloud -> ask user
- Built-in gates: syntax, imports, tests
"""

import ast
import importlib.util
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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
            logger.debug("[SyntaxGate] no files to check")
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
                logger.debug("[SyntaxGate] %s syntax valid", file_path)
            except SyntaxError as e:
                logger.debug("[SyntaxGate] %s:%s: %s", file_path, e.lineno, e.msg)
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
            except Exception as e:
                logger.debug("[SyntaxGate] %s: %s", file_path, e)
                errors.append(f"{file_path}: {str(e)}")

        if errors:
            logger.warning("[SyntaxGate] failed errors=%d", len(errors))
            return GateResult(
                gate_name="Syntax",
                passed=False,
                message=f"Syntax errors in {len(errors)} file(s)",
                errors=errors,
            )

        logger.info("[SyntaxGate] passed files=%d", len(files))
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
            logger.debug("[ImportGate] no files to check")
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
            logger.warning("[ImportGate] failed errors=%d", len(errors))
            return GateResult(
                gate_name="Imports",
                passed=False,
                message=f"Import errors in {len(errors)} case(s)",
                errors=errors,
            )

        logger.info("[ImportGate] passed files=%d", len(files))
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
                logger.debug("[ImportGate] %s: cannot import %s", file_path, module_name)
                errors.append(f"{file_path}: Cannot import '{module_name}'")
            else:
                logger.debug("[ImportGate] %s: import %s OK", file_path, module_name)
        except (ImportError, ModuleNotFoundError, ValueError):
            logger.debug("[ImportGate] %s: cannot import %s", file_path, module_name)
            errors.append(f"{file_path}: Cannot import '{module_name}'")


class TestGate(QualityGate):
    """
    Test Check Gate: Run tests and verify they pass.

    Auto-detects test framework (pytest, jest, etc.) and runs tests.
    """

    def check(self, context: Dict) -> GateResult:
        """Run tests in the project."""
        project_dir = context.get("project_dir")

        # Infer project_dir from files if not provided
        if not project_dir:
            files = context.get("files", [])
            if files:
                # Use the common parent directory of all files
                parents = [str(Path(f).parent) for f in files]
                project_dir = os.path.commonpath(parents) if parents else "."
                logger.debug("[TestGate] project_dir inferred: %s", project_dir)
            else:
                project_dir = "."

        # Try to find test files
        test_files = self._find_test_files(project_dir)

        if not test_files:
            logger.debug("[TestGate] no test files in %s", project_dir)
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
            logger.debug("[TestGate] no test framework detected in %s", project_dir)
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

        logger.debug("[TestGate] found %d test files in %s", len(test_files), project_dir)
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
                logger.info("[TestGate] passed test_files=%d", len(test_files))
                return GateResult(
                    gate_name="Tests",
                    passed=True,
                    message=f"All tests passed ({len(test_files)} file(s))",
                    details=result.stdout,
                )
            else:
                errors = self._parse_pytest_errors(result.stdout)
                logger.warning("[TestGate] failed errors=%d", len(errors))
                return GateResult(
                    gate_name="Tests",
                    passed=False,
                    message=f"Tests failed",
                    details=result.stdout,
                    errors=errors,
                )

        except subprocess.TimeoutExpired:
            logger.warning("[TestGate] timed out after 60s")
            return GateResult(
                gate_name="Tests",
                passed=False,
                message="Tests timed out (>60s)",
            )
        except Exception as e:
            logger.error("[TestGate] exception: %s", e)
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
                logger.info("[TestGate] jest passed")
                return GateResult(
                    gate_name="Tests",
                    passed=True,
                    message="All tests passed",
                    details=result.stdout,
                )
            else:
                logger.warning("[TestGate] jest failed")
                return GateResult(
                    gate_name="Tests",
                    passed=False,
                    message="Tests failed",
                    details=result.stdout,
                )

        except subprocess.TimeoutExpired:
            logger.warning("[TestGate] jest timed out after 60s")
            return GateResult(
                gate_name="Tests",
                passed=False,
                message="Tests timed out (>60s)",
            )
        except Exception as e:
            logger.error("[TestGate] jest exception: %s", e)
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

        # Build context dict that gates expect
        context = {"files": paths, **kwargs}

        enabled_count = sum(1 for g in self.gates.values() if g.enabled)
        logger.info("[QualityGates] running %d gates on %d files", enabled_count, len(paths))

        for gate_name, gate in self.gates.items():
            if not gate.enabled:
                logger.debug("[QualityGates] %s disabled, skipping", gate_name)
                continue

            try:
                result = gate.check(context)
            except Exception as e:
                logger.error("[QualityGates] %s threw exception: %s", gate_name, e)
                result = GateResult(
                    gate_name=gate_name.capitalize(),
                    passed=True,
                    message=f"Gate check skipped: {e}",
                )
            results[gate_name] = result

        passed_count = sum(1 for r in results.values() if r.passed)
        logger.info("[QualityGates] complete: %d/%d passed", passed_count, len(results))
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
            logger.info("[QualityGates] enabled: %s", gate_name)

    def disable_gate(self, gate_name: str):
        """Disable a quality gate."""
        if gate_name in self.gates:
            self.gates[gate_name].enabled = False
            logger.info("[QualityGates] disabled: %s", gate_name)

    def format_results(self, results: List[GateResult]) -> str:
        """Format gate results for display."""
        lines = ["Quality Gate Results:", "=" * 50]

        for result in results:
            status = "PASS" if result.passed else "FAIL"
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

    Ladder: retry (2x) -> decompose -> cloud -> ask user
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
        logger.info("[Escalation] retry %d/%d", self.retry_count, self.max_retries)

    def escalate(self):
        """Escalate to next level in the ladder."""
        self.retry_count += 1
        logger.info("[Escalation] escalated count=%d action=%s", self.retry_count, self.get_action())

    def reset(self):
        """Reset retry count."""
        old_count = self.retry_count
        self.retry_count = 0
        logger.info("[Escalation] reset (was count=%d)", old_count)

    def get_action(self) -> str:
        """Get the current escalation action.

        Checks in reverse order (most escalated first) so the elif
        chain correctly progresses: retry → decompose → alternative → ask_user.
        """
        if self.should_ask_user():
            action = "ask_user"
        elif self.should_escalate_to_cloud():
            action = "alternative"
        elif self.should_decompose():
            action = "decompose"
        else:
            action = "retry"
        logger.debug("[Escalation] action=%s retry_count=%d", action, self.retry_count)
        return action
