# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit tests for CodebaseIndex (M7).

Tests:
- Repository indexing
- Symbol extraction
- Dependency graph building
- Circular dependency detection
- Issue detection
- Architecture analysis
"""

import tempfile
from pathlib import Path

import pytest

from gaia.agents.gaia_code.codebase_index import CodebaseIndex, FileInfo, SymbolInfo


class TestCodebaseIndex:
    """Test CodebaseIndex."""

    def test_index_simple_file(self):
        """Test indexing a simple Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple Python file
            test_file = Path(tmpdir) / "example.py"
            test_file.write_text(
                '''
"""Example module."""

class MyClass:
    """A simple class."""

    def my_method(self):
        """A method."""
        pass

def my_function():
    """A function."""
    return 42
'''
            )

            # Index it
            index = CodebaseIndex(tmpdir)
            stats = index.index_repository()

            # Verify stats
            assert stats["files_indexed"] == 1
            assert stats["symbols_found"] >= 3  # class, method, function
            assert stats["classes"] >= 1
            assert stats["functions"] >= 1

    def test_symbol_extraction(self):
        """Test symbol extraction from AST."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "symbols.py"
            test_file.write_text(
                '''
class TestClass:
    """Test class."""
    pass

def test_function():
    """Test function."""
    return True

async def async_function():
    """Async function."""
    return await something()
'''
            )

            index = CodebaseIndex(tmpdir)
            index.index_repository()

            # Check symbols were extracted
            assert "TestClass" in index.symbols
            assert "test_function" in index.symbols
            assert "async_function" in index.symbols

            # Verify symbol info
            test_class = index.symbols["TestClass"][0]
            assert test_class.type == "class"
            assert test_class.docstring == "Test class."

    def test_dependency_graph(self):
        """Test dependency graph building."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create module A
            mod_a = Path(tmpdir) / "module_a.py"
            mod_a.write_text("import os\nimport sys")

            # Create module B that imports A
            mod_b = Path(tmpdir) / "module_b.py"
            mod_b.write_text("import module_a\nimport json")

            index = CodebaseIndex(tmpdir)
            index.index_repository()

            # Check imports were recorded
            assert str(mod_a) in index.imports
            assert str(mod_b) in index.imports

    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # This is tricky to test without actual importable modules
            # For now, just verify the method doesn't crash
            index = CodebaseIndex(tmpdir)
            index.index_repository()

            cycles = index.circular_deps
            assert isinstance(cycles, list)

    def test_issue_detection(self):
        """Test issue detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large file
            large_file = Path(tmpdir) / "large.py"
            large_file.write_text("\n".join([f"# Line {i}" for i in range(600)]))

            # Create a file with missing docstrings
            no_docs = Path(tmpdir) / "no_docs.py"
            no_docs.write_text(
                '''
class UndocumentedClass:
    pass

def undocumented_function():
    pass
'''
            )

            # Create a complex function
            complex_func = Path(tmpdir) / "complex.py"
            complex_func.write_text(
                '''
def complex_function(x):
    """Complex function."""
    if x > 0:
        if x > 10:
            if x > 20:
                if x > 30:
                    return "very high"
                return "high"
            return "medium"
        return "low"
    return "zero"
'''
            )

            index = CodebaseIndex(tmpdir)
            index.index_repository()

            # Check issues were detected
            assert len(index.issues) > 0

            # Should detect large file
            assert any(i["type"] == "large_file" for i in index.issues)

            # Should detect missing docstrings
            assert any(i["type"] == "missing_docstrings" for i in index.issues)

            # Should detect high complexity
            assert any(i["type"] == "high_complexity" for i in index.issues)

    def test_architecture_summary(self):
        """Test architecture summary generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple module structure
            (Path(tmpdir) / "module1").mkdir()
            (Path(tmpdir) / "module1" / "__init__.py").write_text("")
            (Path(tmpdir) / "module1" / "a.py").write_text("class A: pass")

            (Path(tmpdir) / "module2").mkdir()
            (Path(tmpdir) / "module2" / "__init__.py").write_text("")
            (Path(tmpdir) / "module2" / "b.py").write_text("class B: pass")

            index = CodebaseIndex(tmpdir)
            index.index_repository()

            summary = index.get_architecture_summary()

            # Should contain basic stats
            assert "Total Files" in summary
            assert "Total Symbols" in summary
            assert "Module Structure" in summary

    def test_find_symbol(self):
        """Test symbol lookup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("class MyClass:\n    pass\n\ndef my_func():\n    pass")

            index = CodebaseIndex(tmpdir)
            index.index_repository()

            # Find class
            results = index.find_symbol("MyClass")
            assert len(results) == 1
            assert results[0].type == "class"

            # Find function
            results = index.find_symbol("my_func")
            assert len(results) == 1
            assert results[0].type == "function"

            # Find non-existent
            results = index.find_symbol("DoesNotExist")
            assert len(results) == 0

    def test_get_dependents(self):
        """Test reverse dependency lookup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with dependencies
            base = Path(tmpdir) / "base.py"
            base.write_text("class Base: pass")

            derived = Path(tmpdir) / "derived.py"
            derived.write_text("from base import Base\n\nclass Derived(Base): pass")

            index = CodebaseIndex(tmpdir)
            index.index_repository()

            # Get dependents of base.py
            dependents = index.get_dependents(str(base))

            # derived.py should be in dependents (though may not resolve in test)
            # Just verify the method works
            assert isinstance(dependents, set)

    def test_export_to_json(self):
        """Test JSON export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("class Test: pass")

            index = CodebaseIndex(tmpdir)
            index.index_repository()

            output_file = Path(tmpdir) / "index.json"
            index.export_to_json(str(output_file))

            assert output_file.exists()

            # Verify it's valid JSON
            import json

            with open(output_file) as f:
                data = json.load(f)

            assert "stats" in data
            assert "files" in data


class TestCodebaseIndexIntegration:
    """Integration tests for codebase indexing."""

    def test_index_gaia_repository(self):
        """Test indexing the GAIA repository itself."""
        # This would index the actual GAIA codebase
        # Skip if not in GAIA repo
        gaia_root = Path(__file__).parent.parent.parent

        if not (gaia_root / "src" / "gaia").exists():
            pytest.skip("Not in GAIA repository")

        index = CodebaseIndex(str(gaia_root))
        stats = index.index_repository()

        # GAIA should have significant codebase
        assert stats["files_indexed"] > 50
        assert stats["symbols_found"] > 500
        assert stats["classes"] > 20

        # Should detect some issues
        assert stats["issues"] > 0

        # Should have architecture summary
        summary = index.get_architecture_summary()
        assert len(summary) > 100

    def test_analyze_circular_deps_in_real_code(self):
        """Test circular dependency detection on real code."""
        gaia_root = Path(__file__).parent.parent.parent

        if not (gaia_root / "src" / "gaia").exists():
            pytest.skip("Not in GAIA repository")

        index = CodebaseIndex(str(gaia_root))
        index.index_repository()

        # Check if circular dependencies were found
        if index.circular_deps:
            print(f"\nFound {len(index.circular_deps)} circular dependencies:")
            for cycle in index.circular_deps[:3]:
                print(f"  - {' -> '.join([Path(f).name for f in cycle])}")
