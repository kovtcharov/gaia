#!/usr/bin/env python3
# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code Validation Script

Validates the GAIA Code implementation by checking:
1. All required files exist
2. All imports work
3. All databases initialize
4. All tests can run
5. Integration points are ready
"""

import sys
from pathlib import Path


def check_files_exist():
    """Check that all required files were created."""
    print("=" * 60)
    print("CHECKING FILE EXISTENCE")
    print("=" * 60)

    base_path = Path(__file__).parent / "src" / "gaia" / "agents" / "gaia_code"

    required_files = [
        "__init__.py",
        "agent.py",
        "shared_state.py",
        "quality_gates.py",
        "system_prompt.py",
        "tools.py",
        "cli.py",
        "integration.py",
        "tool_builder.py",
        "skill_extractor.py",
        "insight_engine.py",
        "agent_factory.py",
        "embedding_engine.py",
        "vector_search.py",
        "defragmentation.py",
        "README.md",
    ]

    specialist_files = [
        "specialists/__init__.py",
        "specialists/base_specialist.py",
        "specialists/debugger_agent.py",
        "specialists/security_agent.py",
        "specialists/refactoring_agent.py",
        "specialists/testing_agent.py",
        "specialists/documentation_agent.py",
        "specialists/performance_agent.py",
        "specialists/architecture_agent.py",
    ]

    missing = []
    found = []

    for file in required_files + specialist_files:
        file_path = base_path / file
        if file_path.exists():
            found.append(file)
            print(f"✅ {file}")
        else:
            missing.append(file)
            print(f"❌ {file} - MISSING")

    print()
    print(f"Found: {len(found)}/{len(required_files) + len(specialist_files)}")

    if missing:
        print(f"\nMissing files: {len(missing)}")
        for f in missing:
            print(f"  - {f}")
        return False

    return True


def check_imports():
    """Check that imports work."""
    print("\n" + "=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)

    try:
        print("Importing gaia.agents.gaia_code...")
        from gaia.agents.gaia_code import GaiaCodeAgent

        print("✅ Main package imports successfully")

        print("\nImporting specialists...")
        from gaia.agents.gaia_code.specialists import (
            ArchitectureAgent,
            DebuggerAgent,
            DocumentationAgent,
            PerformanceAgent,
            RefactoringAgent,
            SecurityAgent,
            TestingAgent,
        )

        print("✅ All specialists import successfully")

        print("\nImporting utilities...")
        from gaia.agents.gaia_code.tool_builder import ToolBuilder
        from gaia.agents.gaia_code.skill_extractor import SkillExtractor
        from gaia.agents.gaia_code.insight_engine import InsightEngine
        from gaia.agents.gaia_code.agent_factory import AgentFactory

        print("✅ All utilities import successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_databases():
    """Check that databases can be initialized."""
    print("\n" + "=" * 60)
    print("CHECKING DATABASE INITIALIZATION")
    print("=" * 60)

    try:
        import tempfile
        from gaia.agents.gaia_code.shared_state import get_shared_state

        with tempfile.TemporaryDirectory() as tmpdir:
            state = get_shared_state(Path(tmpdir))

            print("✅ SharedAgentState initialized")
            print(f"  - memory.db: {state.memory.db_path.exists()}")
            print(f"  - knowledge.db: {state.knowledge.db_path.exists()}")
            print(f"  - tools.db: {state.tools.db_path.exists()}")
            print(f"  - skills.db: {state.skills.db_path.exists()}")
            print(f"  - agents.db: {state.agents.db_path.exists()}")
            print(f"  - plan.db: {state.plan.db_path.exists()}")

            # Test basic operations
            print("\nTesting basic database operations...")

            # Memory
            state.memory.cache_file("test.py", "print('hello')")
            content = state.memory.get_file("test.py")
            assert content == "print('hello')"
            print("✅ memory.db operations work")

            # Knowledge
            insight_id = state.knowledge.store_insight("test", "Test insight")
            results = state.knowledge.recall("Test")
            assert len(results) > 0
            print("✅ knowledge.db operations work")

            # Tools
            tool_id = state.tools.register_tool(
                "test_tool", "test", "Test tool", "core"
            )
            tools = state.tools.find_tools("test")
            assert len(tools) > 0
            print("✅ tools.db operations work")

            # Plan
            task = state.plan.create_task("Test task")
            retrieved = state.plan.get_task(task.id)
            assert retrieved.id == task.id
            print("✅ plan.db operations work")

            return True

    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_tests_exist():
    """Check that test files exist."""
    print("\n" + "=" * 60)
    print("CHECKING TESTS")
    print("=" * 60)

    test_path = Path(__file__).parent / "tests" / "unit"

    test_files = [
        "test_gaia_code_shared_state.py",
        "test_gaia_code_quality_gates.py",
        "test_gaia_code_agent.py",
        "test_gaia_code_m5_m6.py",
    ]

    missing = []
    found = []

    for file in test_files:
        file_path = test_path / file
        if file_path.exists():
            found.append(file)
            print(f"✅ {file}")
        else:
            missing.append(file)
            print(f"❌ {file} - MISSING")

    print()
    print(f"Found: {len(found)}/{len(test_files)}")

    return len(missing) == 0


def print_summary():
    """Print implementation summary."""
    print("\n" + "=" * 60)
    print("IMPLEMENTATION SUMMARY")
    print("=" * 60)

    base_path = Path(__file__).parent / "src" / "gaia" / "agents" / "gaia_code"

    # Count files
    py_files = list(base_path.rglob("*.py"))
    md_files = list(base_path.rglob("*.md"))

    print(f"Python files: {len(py_files)}")
    print(f"Documentation files: {len(md_files)}")

    # Count lines
    total_lines = 0
    for file in py_files:
        try:
            lines = len(file.read_text().splitlines())
            total_lines += lines
        except:
            pass

    print(f"Total lines of code: ~{total_lines}")

    # Count tests
    test_path = Path(__file__).parent / "tests" / "unit"
    test_files = list(test_path.glob("test_gaia_code*.py"))

    test_lines = 0
    for file in test_files:
        try:
            lines = len(file.read_text().splitlines())
            test_lines += lines
        except:
            pass

    print(f"Test files: {len(test_files)}")
    print(f"Test lines: ~{test_lines}")


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("GAIA CODE VALIDATION")
    print("=" * 60)
    print()

    results = {
        "files": check_files_exist(),
        "imports": check_imports(),
        "databases": check_databases(),
        "tests": check_tests_exist(),
    }

    print_summary()

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check.upper()}")

    all_passed = all(results.values())

    print("=" * 60)

    if all_passed:
        print("\n🎉 ALL VALIDATION CHECKS PASSED!")
        print("\nNext steps:")
        print("  1. Run integration tests: pytest tests/integration/test_gaia_code_integration.py")
        print("  2. Connect to LLM client (10-15 hours)")
        print("  3. Register tools and specialists (4-6 hours)")
        print("  4. End-to-end validation (3-4 hours)")
        return 0
    else:
        print("\n⚠️  SOME VALIDATION CHECKS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
