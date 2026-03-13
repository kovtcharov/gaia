# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit tests for Quality Gates.

Tests:
- Syntax gate
- Import gate
- Test gate
- Quality gate runner
- Escalation ladder
"""

import tempfile
from pathlib import Path

import pytest

from gaia.agents.gaia_code.quality_gates import (
    EscalationLadder,
    FileCompletenessGate,
    ImportGate,
    QualityGateRunner,
    SyntaxGate,
    TestGate,
)


class TestSyntaxGate:
    """Test SyntaxGate."""

    def test_valid_syntax(self):
        """Test with valid Python syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid Python file
            file_path = Path(tmpdir) / "valid.py"
            file_path.write_text("def hello():\n    print('hello')")

            gate = SyntaxGate()
            result = gate.check({"files": [str(file_path)]})

            assert result.passed is True
            assert "valid" in result.message.lower()

    def test_invalid_syntax(self):
        """Test with invalid Python syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid Python file (missing colon)
            file_path = Path(tmpdir) / "invalid.py"
            file_path.write_text("def hello()\n    print('hello')")

            gate = SyntaxGate()
            result = gate.check({"files": [str(file_path)]})

            assert result.passed is False
            assert len(result.errors) > 0
            assert "invalid.py" in result.errors[0]

    def test_no_files(self):
        """Test with no files."""
        gate = SyntaxGate()
        result = gate.check({"files": []})

        assert result.passed is True
        assert "no files" in result.message.lower()


class TestImportGate:
    """Test ImportGate."""

    def test_valid_imports(self):
        """Test with valid imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with valid imports
            file_path = Path(tmpdir) / "valid.py"
            file_path.write_text("import os\nimport sys\nfrom pathlib import Path")

            gate = ImportGate()
            result = gate.check({"files": [str(file_path)]})

            assert result.passed is True

    def test_invalid_imports(self):
        """Test with invalid imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with invalid import
            file_path = Path(tmpdir) / "invalid.py"
            file_path.write_text(
                "import os\nimport nonexistent_module_12345\nfrom pathlib import Path"
            )

            gate = ImportGate()
            result = gate.check({"files": [str(file_path)]})

            assert result.passed is False
            assert len(result.errors) > 0
            assert "nonexistent_module_12345" in result.errors[0]


class TestTestGate:
    """Test TestGate."""

    def test_no_tests(self):
        """Test with no test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gate = TestGate()
            result = gate.check({"project_dir": tmpdir})

            assert result.passed is True
            assert "no tests" in result.message.lower()

    def test_pytest_found(self):
        """Test detection of pytest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test_example.py"
            test_file.write_text(
                """
import pytest

def test_example():
    assert True
"""
            )

            gate = TestGate()
            test_files = gate._find_test_files(tmpdir)

            assert len(test_files) > 0
            assert gate._has_pytest(test_files) is True


class TestFileCompletenessGate:
    """Test FileCompletenessGate."""

    def test_all_files_present_and_non_empty(self):
        """Gate passes when all files exist and are larger than MIN_SIZE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "file.cpp"
            f1.write_text("#include <iostream>\nint main() { return 0; }")
            f2 = Path(tmpdir) / "header.hpp"
            f2.write_text("#pragma once\nclass Foo {};\n")

            gate = FileCompletenessGate()
            result = gate.check({"files": [str(f1), str(f2)]})

            assert result.passed is True
            assert "non-empty" in result.message

    def test_missing_file_fails(self):
        """Gate fails when a tracked file does not exist on disk."""
        gate = FileCompletenessGate()
        result = gate.check({"files": ["/nonexistent/path/file.cpp"]})

        assert result.passed is False
        assert any("Missing" in e for e in result.errors)

    def test_stub_file_fails(self):
        """Gate fails when a file is smaller than MIN_SIZE bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stub = Path(tmpdir) / "stub.cpp"
            stub.write_text("// TODO")  # 7 bytes — well below MIN_SIZE

            gate = FileCompletenessGate()
            result = gate.check({"files": [str(stub)]})

            assert result.passed is False
            assert any("Stub" in e or "empty" in e for e in result.errors)

    def test_no_files_passes(self):
        """Gate passes trivially when no files are provided."""
        gate = FileCompletenessGate()
        result = gate.check({"files": []})

        assert result.passed is True
        assert "no files" in result.message.lower()

    def test_cpp_files_checked(self):
        """Gate checks non-Python files (C++, TypeScript, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a real C++ file (not Python)
            cpp = Path(tmpdir) / "agent.cpp"
            cpp.write_text(
                '#include "agent.hpp"\nstd::string Agent::run(std::string q) { return q; }\n'
            )

            gate = FileCompletenessGate()
            result = gate.check({"files": [str(cpp)]})

            assert result.passed is True


class TestQualityGateRunner:
    """Test QualityGateRunner."""

    def test_all_gates_pass(self):
        """Test when all gates pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid file — content must be > MIN_SIZE (20 bytes)
            file_path = Path(tmpdir) / "valid.py"
            file_path.write_text("import os\n\ndef hello():\n    print('hello')")

            runner = QualityGateRunner()
            results = runner.run_all([str(file_path)], project_dir=tmpdir)
            all_passed = runner.all_passed(results)

            assert all_passed is True
            assert len(results) >= 3  # completeness, syntax, imports

    def test_some_gates_fail(self):
        """Test when some gates fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with syntax error
            file_path = Path(tmpdir) / "invalid.py"
            file_path.write_text("def hello()\n    print('hello')")

            runner = QualityGateRunner()
            results = runner.run_all([str(file_path)], project_dir=tmpdir)
            all_passed = runner.all_passed(results)

            assert all_passed is False
            assert any(not r.passed for r in results.values())

    def test_disable_gate(self):
        """Test disabling a gate."""
        runner = QualityGateRunner()

        # Disable syntax gate
        runner.disable_gate("syntax")
        assert runner.gates["syntax"].enabled is False

        # Enable it back
        runner.enable_gate("syntax")
        assert runner.gates["syntax"].enabled is True

    def test_format_results(self):
        """Test result formatting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "valid.py"
            file_path.write_text("import os\n\ndef hello():\n    print('hello')")

            runner = QualityGateRunner()
            results = runner.run_all([str(file_path)], project_dir=tmpdir)

            formatted = runner.format_results(list(results.values()))

            assert "Quality Gate Results" in formatted
            assert "✅" in formatted or "PASS" in formatted


class TestEscalationLadder:
    """Test EscalationLadder."""

    def test_initial_state(self):
        """Test initial state is retry."""
        ladder = EscalationLadder()

        assert ladder.should_retry() is True
        assert ladder.get_action() == "retry"

    def test_progression(self):
        """Test escalation progression."""
        ladder = EscalationLadder()

        # Start with retry
        assert ladder.get_action() == "retry"

        ladder.increment()
        assert ladder.should_retry() is True

        ladder.increment()
        assert ladder.should_retry() is False
        assert ladder.should_decompose() is True
        assert ladder.get_action() == "decompose"

        ladder.increment()
        assert ladder.should_escalate_to_cloud() is True
        assert ladder.get_action() == "cloud"

        ladder.increment()
        assert ladder.should_ask_user() is True
        assert ladder.get_action() == "ask_user"

    def test_reset(self):
        """Test reset functionality."""
        ladder = EscalationLadder()

        # Increment several times
        ladder.increment()
        ladder.increment()
        ladder.increment()

        assert ladder.get_action() == "cloud"

        # Reset
        ladder.reset()

        assert ladder.retry_count == 0
        assert ladder.get_action() == "retry"
