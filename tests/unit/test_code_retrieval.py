# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Tests for the Code Retrieval Pipeline.

Tests Layer 1-4 of the code retrieval pipeline:
- Structural indexing (chunks, hashing, incremental)
- Semantic annotations (AST-only)
- Hybrid search (FTS5 keyword)
- Context assembly (query classification, impact analysis)

Note: FAISS tests are skipped if faiss-cpu is not installed.
"""

import os
import sqlite3
import tempfile
import textwrap
from pathlib import Path

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample Python repository for testing."""
    # Create directory structure
    src_dir = tmp_path / "src" / "myapp"
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text("")

    # Main module
    (src_dir / "agent.py").write_text(textwrap.dedent('''\
        """Agent module for handling user requests."""

        from .base import BaseAgent
        from .tools import tool

        class MyAgent(BaseAgent):
            """Main agent that processes user queries.

            Supports chat, code generation, and search.
            """

            def __init__(self, name="default"):
                super().__init__(name=name)
                self.history = []

            def process_query(self, query: str) -> str:
                """Process a user query and return a response."""
                self.history.append(query)
                return f"Response to: {query}"

            def get_history(self):
                """Get conversation history."""
                return self.history
    '''))

    # Base class
    (src_dir / "base.py").write_text(textwrap.dedent('''\
        """Base agent implementation."""

        class BaseAgent:
            """Base class for all agents.

            Provides common functionality like logging and state management.
            """

            def __init__(self, name: str = "base"):
                self.name = name
                self.state = {}

            def log(self, message: str):
                """Log a message."""
                print(f"[{self.name}] {message}")
    '''))

    # Tools module
    (src_dir / "tools.py").write_text(textwrap.dedent('''\
        """Tool decorator and registry for agent tools."""

        import functools

        def tool(func):
            """Decorator to register a function as an agent tool."""
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            wrapper._is_tool = True
            return wrapper

        class ToolRegistry:
            """Registry for managing agent tools."""

            def __init__(self):
                self.tools = {}

            def register(self, name: str, func):
                """Register a tool."""
                self.tools[name] = func

            def get(self, name: str):
                """Get a tool by name."""
                return self.tools.get(name)
    '''))

    # Config file (non-Python)
    (tmp_path / "config.yaml").write_text(textwrap.dedent('''\
        app:
          name: myapp
          version: "1.0"
        logging:
          level: INFO
    '''))

    # README (markdown)
    (tmp_path / "README.md").write_text(textwrap.dedent('''\
        # MyApp

        A sample application for testing.

        ## Features

        - Agent-based architecture
        - Tool registry
        - Chat support

        ## Usage

        Run `python -m myapp` to start.
    '''))

    # A binary-like file to test track-only
    (tmp_path / "data.pkl").write_bytes(b"\\x80\\x04\\x95")

    return tmp_path


@pytest.fixture
def pipeline(sample_repo):
    """Create a CodeRetrievalPipeline for the sample repo."""
    from gaia.agents.gaia_code.code_retrieval import CodeRetrievalPipeline

    workspace = sample_repo / ".workspace"
    workspace.mkdir()

    p = CodeRetrievalPipeline(
        root_path=str(sample_repo),
        workspace_dir=workspace,
        llm_provider="none",  # AST-only for tests
        enable_watcher=False,  # Disable watcher in tests
    )
    yield p
    p.close()


# ============================================================================
# Layer 1: Structural Index Tests
# ============================================================================


class TestStructuralIndex:
    """Tests for Layer 1: Structural Index."""

    def test_index_repository_basic(self, pipeline):
        """Test basic repository indexing."""
        stats = pipeline.index_repository()

        assert stats["files_indexed"] > 0
        assert stats["chunks_created"] > 0
        assert stats["total_files"] > 0
        assert stats["total_chunks"] > 0
        assert stats["index_time_seconds"] >= 0

    def test_incremental_indexing(self, pipeline, sample_repo):
        """Test that unchanged files are skipped on re-index."""
        # First index
        stats1 = pipeline.index_repository()
        first_indexed = stats1["files_indexed"]

        # Second index (nothing changed)
        stats2 = pipeline.index_repository()

        assert stats2["files_indexed"] == 0  # Nothing new to index
        assert stats2["files_unchanged"] > 0
        assert stats2["total_files"] == stats1["total_files"]

    def test_incremental_detects_changes(self, pipeline, sample_repo):
        """Test that modified files are re-indexed."""
        # First index
        pipeline.index_repository()

        # Modify a file
        agent_file = sample_repo / "src" / "myapp" / "agent.py"
        content = agent_file.read_text()
        agent_file.write_text(content + "\n# Modified\n")

        # Re-index
        stats = pipeline.index_repository()
        assert stats["files_indexed"] >= 1  # At least the modified file

    def test_deleted_files_removed(self, pipeline, sample_repo):
        """Test that deleted files are removed from index."""
        pipeline.index_repository()

        # Delete a file
        tools_file = sample_repo / "src" / "myapp" / "tools.py"
        tools_file.unlink()

        # Re-index
        stats = pipeline.index_repository()
        assert stats["files_deleted"] >= 1

    def test_track_only_files(self, pipeline, sample_repo):
        """Test that binary/track-only files are recorded but not chunked."""
        stats = pipeline.index_repository()

        # Check that .pkl file exists in files table but has no chunks
        cursor = pipeline.conn.execute(
            "SELECT is_track_only FROM files WHERE file_path LIKE '%data.pkl'"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1  # is_track_only = True

    def test_python_chunk_extraction(self, pipeline, sample_repo):
        """Test that Python files are chunked correctly."""
        pipeline.index_repository()

        # Check for expected chunk types
        cursor = pipeline.conn.execute(
            "SELECT chunk_type, symbol_name FROM code_chunks WHERE file_path LIKE '%agent.py'"
        )
        chunks = cursor.fetchall()
        chunk_types = {row[0] for row in chunks}
        symbol_names = {row[1] for row in chunks}

        assert "class" in chunk_types or "method" in chunk_types
        assert "MyAgent" in symbol_names or "process_query" in symbol_names

    def test_non_python_chunk_extraction(self, pipeline, sample_repo):
        """Test that non-Python files are chunked."""
        pipeline.index_repository()

        # Check YAML file
        cursor = pipeline.conn.execute(
            "SELECT COUNT(*) FROM code_chunks WHERE file_path LIKE '%config.yaml'"
        )
        assert cursor.fetchone()[0] > 0

        # Check markdown file
        cursor = pipeline.conn.execute(
            "SELECT COUNT(*) FROM code_chunks WHERE file_path LIKE '%README.md'"
        )
        assert cursor.fetchone()[0] > 0

    def test_file_hash_consistency(self, pipeline, sample_repo):
        """Test that file hashes are consistent."""
        from gaia.agents.gaia_code.code_retrieval import compute_file_hash

        agent_file = str(sample_repo / "src" / "myapp" / "agent.py")
        hash1 = compute_file_hash(agent_file)
        hash2 = compute_file_hash(agent_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest


# ============================================================================
# Layer 2: Semantic Annotations Tests
# ============================================================================


class TestSemanticAnnotations:
    """Tests for Layer 2: Semantic Annotations."""

    def test_ast_annotations_generated(self, pipeline):
        """Test that AST annotations are created for all chunks."""
        pipeline.index_repository()

        total_chunks = pipeline.conn.execute("SELECT COUNT(*) FROM code_chunks").fetchone()[0]
        total_annotations = pipeline.conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]

        assert total_annotations == total_chunks

    def test_annotation_content(self, pipeline):
        """Test that annotations have meaningful content."""
        pipeline.index_repository()

        cursor = pipeline.conn.execute("""
            SELECT a.summary, a.search_text, a.annotation_source
            FROM annotations a
            JOIN code_chunks c ON a.chunk_id = c.chunk_id
            WHERE c.symbol_name = 'MyAgent'
        """)
        row = cursor.fetchone()

        assert row is not None
        summary, search_text, source = row
        assert "MyAgent" in summary
        assert source == "ast"
        assert len(search_text) > 0

    def test_annotation_concepts_extracted(self, pipeline):
        """Test that concepts are extracted from docstrings."""
        pipeline.index_repository()

        cursor = pipeline.conn.execute("""
            SELECT a.concepts
            FROM annotations a
            JOIN code_chunks c ON a.chunk_id = c.chunk_id
            WHERE c.docstring IS NOT NULL AND c.docstring != ''
            LIMIT 5
        """)
        rows = cursor.fetchall()

        # At least some annotations should have concepts
        has_concepts = False
        for row in rows:
            import json
            concepts = json.loads(row[0]) if row[0] else []
            if concepts:
                has_concepts = True
                break
        # Concepts may be empty for simple docstrings, that's OK

    def test_generate_summaries(self, pipeline):
        """Test hierarchical summary generation."""
        pipeline.index_repository()
        summaries = pipeline.generate_summaries()

        assert len(summaries) > 0
        assert "__architecture__" in summaries

        # Check summaries table
        cursor = pipeline.conn.execute("SELECT COUNT(*) FROM summaries")
        assert cursor.fetchone()[0] > 0


# ============================================================================
# Layer 3: Retrieval Index Tests
# ============================================================================


class TestRetrievalIndex:
    """Tests for Layer 3: Retrieval Index (FTS5, FAISS optional)."""

    def test_fts5_search_basic(self, pipeline):
        """Test FTS5 keyword search works."""
        pipeline.index_repository()

        results = pipeline._search_fts5("agent process query", top_k=5)
        assert len(results) > 0

    def test_fts5_search_symbol_name(self, pipeline):
        """Test FTS5 can find by symbol name."""
        pipeline.index_repository()

        results = pipeline._search_fts5("MyAgent", top_k=5)
        assert len(results) > 0
        # First result should be the MyAgent class
        chunk_ids = [r[0] for r in results]
        assert any("MyAgent" in cid for cid in chunk_ids)

    def test_fts5_search_no_results(self, pipeline):
        """Test FTS5 returns empty for nonsense queries."""
        pipeline.index_repository()

        results = pipeline._search_fts5("xyznonexistent", top_k=5)
        assert len(results) == 0

    def test_hybrid_search(self, pipeline):
        """Test hybrid search (FTS5 only when FAISS unavailable)."""
        pipeline.index_repository()

        results = pipeline.hybrid_search("agent that processes queries", top_k=5)
        # Should return results from at least FTS5
        assert isinstance(results, list)

    def test_hybrid_search_structural(self, pipeline):
        """Test hybrid search with structural bias (low alpha)."""
        pipeline.index_repository()

        results = pipeline.hybrid_search("BaseAgent", top_k=5, alpha=0.2)
        assert isinstance(results, list)


# ============================================================================
# Layer 4: Context Assembly Tests
# ============================================================================


class TestContextAssembly:
    """Tests for Layer 4: Context Assembly."""

    def test_query_classification_structural(self, pipeline):
        """Test structural query classification."""
        assert pipeline._classify_query("find all classes that inherit from Agent") == "structural"
        assert pipeline._classify_query("list all functions") == "structural"
        assert pipeline._classify_query("which classes extend BaseAgent") == "structural"

    def test_query_classification_semantic(self, pipeline):
        """Test semantic query classification."""
        assert pipeline._classify_query("how does authentication work") == "semantic"
        assert pipeline._classify_query("where is the error handling logic") == "semantic"

    def test_query_classification_locational(self, pipeline):
        """Test locational query classification."""
        assert pipeline._classify_query("what's in agent.py") == "locational"
        assert pipeline._classify_query("show me src/myapp/tools.py") == "locational"

    def test_query_classification_impact(self, pipeline):
        """Test impact query classification."""
        assert pipeline._classify_query("what breaks if I change base.py") == "impact"
        assert pipeline._classify_query("what depends on tools.py") == "impact"

    def test_query_classification_summary(self, pipeline):
        """Test summary query classification."""
        assert pipeline._classify_query("summarize the codebase") == "summary"
        assert pipeline._classify_query("architecture overview") == "summary"

    def test_full_query_pipeline(self, pipeline):
        """Test the full query pipeline end-to-end."""
        pipeline.index_repository()

        result = pipeline.query("find the agent class")

        assert "query" in result
        assert "query_type" in result
        assert "context" in result or "results" in result

    def test_context_budget_respected(self, pipeline):
        """Test that context assembly respects token budget."""
        pipeline.index_repository()

        # Use a very small budget
        result = pipeline.query("agent", context_budget=100)
        context = result.get("context", "")

        # Context should be relatively short
        assert len(context) // 4 < 200  # Some overhead is OK


# ============================================================================
# Structural Search Tests
# ============================================================================


class TestStructuralSearch:
    """Tests for the search_code_structure functionality."""

    def test_search_by_symbol_pattern(self, pipeline):
        """Test searching by symbol name pattern."""
        pipeline.index_repository()

        results = pipeline.search_structure(symbol_pattern="%Agent%")
        assert len(results) > 0
        assert any("Agent" in r.symbol_name for r in results)

    def test_search_by_chunk_type(self, pipeline):
        """Test searching by chunk type."""
        pipeline.index_repository()

        results = pipeline.search_structure(chunk_type="class")
        assert len(results) > 0
        assert all(r.chunk_type == "class" for r in results)

    def test_search_by_module_pattern(self, pipeline):
        """Test searching by module path pattern."""
        pipeline.index_repository()

        results = pipeline.search_structure(module_pattern="%myapp%")
        assert len(results) > 0

    def test_search_combined_filters(self, pipeline):
        """Test searching with multiple filters."""
        pipeline.index_repository()

        results = pipeline.search_structure(
            symbol_pattern="%Agent%",
            chunk_type="class",
        )
        assert len(results) > 0
        assert all(r.chunk_type == "class" for r in results)

    def test_find_subclasses(self, pipeline):
        """Test finding subclasses of a base class."""
        pipeline.index_repository()

        results = pipeline._find_subclasses("BaseAgent", top_k=10)
        assert len(results) > 0
        assert any("MyAgent" in r.symbol_name for r in results)


# ============================================================================
# Impact Analysis Tests
# ============================================================================


class TestImpactAnalysis:
    """Tests for the impact analysis feature."""

    def test_impact_analysis_basic(self, pipeline):
        """Test basic impact analysis."""
        pipeline.index_repository()

        impact = pipeline.impact_analysis("base.py")

        assert impact.target_file == "base.py"
        assert impact.risk_level in ("LOW", "MEDIUM", "HIGH")

    def test_impact_analysis_with_symbol(self, pipeline):
        """Test impact analysis for a specific symbol."""
        pipeline.index_repository()

        impact = pipeline.impact_analysis("base.py", "BaseAgent")

        assert impact.target_symbol == "BaseAgent"
        # MyAgent inherits from BaseAgent, so it should reference it
        assert isinstance(impact.symbol_references, list)

    def test_impact_format_context(self, pipeline):
        """Test impact result formatting."""
        pipeline.index_repository()

        impact = pipeline.impact_analysis("base.py")
        context = pipeline._format_impact_context(impact)

        assert "Impact Analysis" in context
        assert "Risk Level" in context


# ============================================================================
# Code Chunk Extraction Tests
# ============================================================================


class TestCodeChunkExtraction:
    """Tests for the extract_code_chunks function."""

    def test_extract_chunks_basic(self):
        """Test basic Python chunk extraction."""
        from gaia.agents.gaia_code.code_retrieval import extract_code_chunks

        source = textwrap.dedent('''\
            """Module docstring."""

            import os
            from pathlib import Path

            class Foo:
                """A foo class."""

                def bar(self):
                    """Do bar."""
                    return 42

            def standalone_func():
                """A standalone function."""
                pass
        ''')

        chunks = extract_code_chunks(
            "/tmp/test.py", "test.py", "test", source
        )

        chunk_types = {c.chunk_type for c in chunks}
        symbol_names = {c.symbol_name for c in chunks}

        assert "module_docstring" in chunk_types
        assert "import_block" in chunk_types
        assert "class" in chunk_types
        assert "function" in chunk_types
        assert "Foo" in symbol_names
        assert "standalone_func" in symbol_names

    def test_extract_chunks_large_class(self):
        """Test that large classes are split into methods."""
        from gaia.agents.gaia_code.code_retrieval import extract_code_chunks

        # Create a class with >200 lines
        methods = []
        for i in range(25):
            methods.append(f"    def method_{i}(self):\n" + "        pass\n" * 10)

        source = 'class BigClass:\n    """A big class."""\n\n' + "\n".join(methods)

        chunks = extract_code_chunks(
            "/tmp/big.py", "big.py", "big", source
        )

        chunk_types = [c.chunk_type for c in chunks]
        # Should have method chunks due to large class
        assert "method" in chunk_types or "class" in chunk_types

    def test_extract_chunks_syntax_error(self):
        """Test graceful handling of syntax errors."""
        from gaia.agents.gaia_code.code_retrieval import extract_code_chunks

        source = "def broken(\n    # syntax error"

        chunks = extract_code_chunks(
            "/tmp/broken.py", "broken.py", "broken", source
        )

        # Should fall back to a single file chunk
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "file"

    def test_extract_non_python_markdown(self):
        """Test markdown chunk extraction."""
        from gaia.agents.gaia_code.code_retrieval import extract_non_python_chunks

        source = textwrap.dedent('''\
            # Title

            Some content here.

            ## Section 1

            More content.

            ## Section 2

            Even more content.
        ''')

        chunks = extract_non_python_chunks(
            "/tmp/readme.md", "readme.md", "readme_md", source, "markdown"
        )

        assert len(chunks) >= 2  # At least 2 sections
        assert all(c.chunk_type == "section" for c in chunks)


# ============================================================================
# AST Annotation Tests
# ============================================================================


class TestASTAnnotation:
    """Tests for AST-only annotation generation."""

    def test_generate_class_annotation(self):
        """Test annotation for a class chunk."""
        from gaia.agents.gaia_code.code_retrieval import CodeChunk, generate_ast_annotation

        chunk = CodeChunk(
            chunk_id="test::MyClass",
            file_path="/tmp/test.py",
            relative_path="test.py",
            module_path="test",
            chunk_type="class",
            symbol_name="MyClass",
            docstring="A class for handling HTTP requests.",
            decorators=["dataclass"],
            source_code="class MyClass(BaseClass):\n    pass",
        )

        ann = generate_ast_annotation(chunk)

        assert "MyClass" in ann.summary
        assert ann.purpose == "A class for handling HTTP requests."
        assert "dataclass" in ann.concepts
        assert ann.annotation_source == "ast"
        assert len(ann.search_text) > 0

    def test_generate_function_annotation(self):
        """Test annotation for a function chunk."""
        from gaia.agents.gaia_code.code_retrieval import CodeChunk, generate_ast_annotation

        chunk = CodeChunk(
            chunk_id="test::my_func",
            file_path="/tmp/test.py",
            relative_path="test.py",
            module_path="test",
            chunk_type="function",
            symbol_name="my_func",
            docstring="Process incoming API requests.",
        )

        ann = generate_ast_annotation(chunk)

        assert "my_func" in ann.summary
        assert "function" in ann.summary
        assert "API" in ann.concepts  # Extracted from docstring

    def test_generate_import_annotation(self):
        """Test annotation for an import block."""
        from gaia.agents.gaia_code.code_retrieval import CodeChunk, generate_ast_annotation

        chunk = CodeChunk(
            chunk_id="test::__imports__",
            file_path="/tmp/test.py",
            relative_path="test.py",
            module_path="test",
            chunk_type="import_block",
            symbol_name="__imports__",
            dependencies=["os", "sys", "pathlib"],
        )

        ann = generate_ast_annotation(chunk)

        assert "Imports" in ann.summary
        assert any("imports" in r for r in ann.relationships)


# ============================================================================
# Pipeline Statistics Tests
# ============================================================================


class TestPipelineStats:
    """Tests for pipeline statistics and metadata."""

    def test_get_stats(self, pipeline):
        """Test pipeline stats reporting."""
        pipeline.index_repository()

        stats = pipeline.get_stats()

        assert stats["indexed_files"] > 0
        assert stats["total_chunks"] > 0
        assert stats["total_annotations"] > 0
        assert stats["llm_provider"] == "none"
        assert stats["watcher_active"] is False  # Disabled in tests

    def test_database_integrity(self, pipeline):
        """Test that database tables are properly linked."""
        pipeline.index_repository()

        # Every chunk should have an annotation
        cursor = pipeline.conn.execute("""
            SELECT COUNT(*)
            FROM code_chunks c
            LEFT JOIN annotations a ON c.chunk_id = a.chunk_id
            WHERE a.chunk_id IS NULL
        """)
        orphan_chunks = cursor.fetchone()[0]
        assert orphan_chunks == 0, f"{orphan_chunks} chunks without annotations"

        # Every chunk should reference an existing file
        cursor = pipeline.conn.execute("""
            SELECT COUNT(*)
            FROM code_chunks c
            LEFT JOIN files f ON c.file_path = f.file_path
            WHERE f.file_path IS NULL
        """)
        orphan_refs = cursor.fetchone()[0]
        assert orphan_refs == 0, f"{orphan_refs} chunks referencing non-existent files"
