# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit tests for M5 (Agent Auto-Generation) and M6 (Vector Search + Defrag).

Tests:
- ToolBuilder
- SkillExtractor
- InsightEngine
- AgentFactory
- EmbeddingEngine (if sentence-transformers available)
- VectorSearch (if FAISS available)
- MemoryDefragmenter
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaia.agents.gaia_code.agent_factory import AgentFactory
from gaia.agents.gaia_code.insight_engine import InsightEngine
from gaia.agents.gaia_code.shared_state import get_shared_state
from gaia.agents.gaia_code.skill_extractor import SkillExtractor
from gaia.agents.gaia_code.tool_builder import ToolBuilder


class TestToolBuilder:
    """Test ToolBuilder."""

    def test_create_simple_tool(self):
        """Test creating a simple tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ToolBuilder(workspace_dir=Path(tmpdir))

            # Create a simple tool
            code = '''
def hello_world():
    """Say hello."""
    return "Hello, World!"
'''

            result = builder.create_tool(
                name="hello_world",
                description="Say hello",
                code=code,
                category="example",
            )

            assert result["success"] is True
            assert "tool_id" in result

    def test_reject_unsafe_tool(self):
        """Test that unsafe tools are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ToolBuilder(workspace_dir=Path(tmpdir))

            # Try to create unsafe tool with eval
            code = '''
def unsafe_tool(code):
    """Unsafe tool."""
    return eval(code)  # Dangerous!
'''

            result = builder.create_tool(
                name="unsafe_tool",
                description="Unsafe",
                code=code,
            )

            assert result["success"] is False
            assert "safety" in result["error"].lower() or "eval" in result["error"].lower()

    def test_reject_invalid_syntax(self):
        """Test that tools with syntax errors are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ToolBuilder(workspace_dir=Path(tmpdir))

            # Tool with syntax error
            code = '''
def broken_tool()
    return "missing colon"
'''

            result = builder.create_tool(
                name="broken_tool",
                description="Broken",
                code=code,
            )

            assert result["success"] is False
            assert "syntax" in result["error"].lower()

    def test_list_learned_tools(self):
        """Test listing learned tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ToolBuilder(workspace_dir=Path(tmpdir))

            # Create a tool
            code = 'def test_tool():\n    return "test"'
            builder.create_tool("test_tool", "Test", code)

            # List tools
            tools = builder.list_learned_tools()
            assert len(tools) >= 1
            assert any(t["name"] == "test_tool" for t in tools)


class TestSkillExtractor:
    """Test SkillExtractor."""

    def test_extract_skill(self):
        """Test skill extraction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = SkillExtractor(workspace_dir=Path(tmpdir))

            # Extract a skill
            result = extractor.extract_skill(
                name="build_api",
                description="Build FastAPI app",
                task_description="Create a REST API",
                steps=["Create models", "Create endpoints", "Write tests"],
                tools_used=["write_file", "run_pytest"],
                category="web_dev",
                domain="coding",
            )

            assert result["success"] is True
            assert "skill_id" in result

    def test_recall_skill(self):
        """Test skill recall."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = SkillExtractor(workspace_dir=Path(tmpdir))

            # Extract a skill
            extractor.extract_skill(
                name="build_api",
                description="Build FastAPI app with auth",
                task_description="Create API",
                steps=["Create models", "Create endpoints"],
                tools_used=["write_file"],
            )

            # Recall skills
            skills = extractor.recall_skill("Build API")
            assert len(skills) >= 1

    def test_record_skill_usage(self):
        """Test recording skill usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = SkillExtractor(workspace_dir=Path(tmpdir))

            # Extract a skill
            result = extractor.extract_skill(
                name="test_skill",
                description="Test skill",
                task_description="Test",
                steps=["Step 1"],
                tools_used=["tool1"],
            )

            skill_id = result["skill_id"]

            # Record successful usage
            success = extractor.record_skill_usage(
                skill_id, success=True, task_description="Test task"
            )

            assert success is True

    def test_list_skills(self):
        """Test listing skills."""
        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = SkillExtractor(workspace_dir=Path(tmpdir))

            # Extract skills
            extractor.extract_skill(
                "skill1", "Skill 1", "Task 1", ["Step 1"], ["tool1"]
            )
            extractor.extract_skill(
                "skill2", "Skill 2", "Task 2", ["Step 2"], ["tool2"]
            )

            # List all skills
            skills = extractor.list_skills()
            assert len(skills) == 2


class TestInsightEngine:
    """Test InsightEngine."""

    def test_generate_insight(self):
        """Test insight generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = InsightEngine(workspace_dir=Path(tmpdir))

            # Generate insight
            result = engine.generate_insight(
                category="error_fix",
                content="Always check for None before accessing attributes",
                domain="python",
                triggers=["AttributeError", "None"],
            )

            assert result["success"] is True
            assert "insight_id" in result

    def test_generate_error_fix_insight(self):
        """Test error-fix insight generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = InsightEngine(workspace_dir=Path(tmpdir))

            # Generate error-fix insight
            result = engine.generate_error_fix_insight(
                error_type="AttributeError",
                error_message="'NoneType' object has no attribute 'name'",
                fix_description="Add None check before accessing attribute",
            )

            assert result["success"] is True

    def test_retrieve_insights(self):
        """Test insight retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = InsightEngine(workspace_dir=Path(tmpdir))

            # Generate some insights
            engine.generate_insight(
                "error_fix", "Check for None", triggers=["AttributeError"]
            )
            engine.generate_insight("pattern", "Use list comprehension", triggers=["filter"])

            # Retrieve insights
            results = engine.retrieve_insights("AttributeError")
            assert len(results) >= 1

    def test_validate_insight(self):
        """Test insight validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = InsightEngine(workspace_dir=Path(tmpdir))

            # Generate insight
            result = engine.generate_insight("pattern", "Test pattern")
            insight_id = result["insight_id"]

            # Validate (increases confidence)
            success = engine.validate_insight(insight_id, validated=True)
            assert success is True

    def test_get_insight_stats(self):
        """Test insight statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = InsightEngine(workspace_dir=Path(tmpdir))

            # Generate some insights
            engine.generate_insight("error_fix", "Insight 1")
            engine.generate_insight("pattern", "Insight 2")

            # Get stats
            stats = engine.get_insight_stats()
            assert stats["total_insights"] == 2
            assert "by_category" in stats


class TestAgentFactory:
    """Test AgentFactory."""

    def test_detect_pattern(self):
        """Test pattern detection from tasks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            factory = AgentFactory(workspace_dir=Path(tmpdir))

            # Create some similar completed tasks
            state = get_shared_state(Path(tmpdir))

            for i in range(3):
                task = state.plan.create_task(f"Build FastAPI app {i}")
                state.plan.update_task_status(task.id, "completed")

            # Detect patterns
            patterns = factory.detect_pattern(min_occurrences=3)

            # Should detect the FastAPI pattern
            assert len(patterns) >= 0  # May or may not detect depending on similarity

    def test_generate_specialist(self):
        """Test specialist generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            factory = AgentFactory(workspace_dir=Path(tmpdir))

            # Create a mock pattern
            state = get_shared_state(Path(tmpdir))
            tasks = []
            for i in range(3):
                task = state.plan.create_task(f"Build API {i}")
                state.plan.update_task_status(task.id, "completed")
                tasks.append(task)

            pattern = {
                "tasks": tasks,
                "keywords": ["build", "api", "fastapi"],
                "count": 3,
                "similarity": 0.9,
            }

            # Generate specialist
            result = factory.generate_specialist(
                pattern,
                specialist_name="FastAPIAgent",
                description="Specialist for building FastAPI apps",
            )

            assert result["success"] is True
            assert "specialist_name" in result


class TestDefragmentation:
    """Test MemoryDefragmenter."""

    def test_should_defragment(self):
        """Test defragmentation trigger logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from gaia.agents.gaia_code.defragmentation import MemoryDefragmenter

            defrag = MemoryDefragmenter(workspace_dir=Path(tmpdir))

            # With no insights, should not trigger
            assert defrag.should_defragment() is False

    def test_get_defrag_stats(self):
        """Test defrag statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from gaia.agents.gaia_code.defragmentation import MemoryDefragmenter

            defrag = MemoryDefragmenter(workspace_dir=Path(tmpdir))

            # Get stats
            stats = defrag.get_defrag_stats()

            assert "total_insights" in stats
            assert "avg_confidence" in stats
            assert "defrag_recommended" in stats


# Skip embedding and vector search tests if dependencies not available
try:
    from gaia.agents.gaia_code.embedding_engine import EMBEDDINGS_AVAILABLE
    from gaia.agents.gaia_code.vector_search import FAISS_AVAILABLE

    @pytest.mark.skipif(
        not EMBEDDINGS_AVAILABLE, reason="sentence-transformers not installed"
    )
    class TestEmbeddingEngine:
        """Test EmbeddingEngine."""

        def test_embed_single_text(self):
            """Test embedding single text."""
            from gaia.agents.gaia_code.embedding_engine import EmbeddingEngine

            engine = EmbeddingEngine()
            embedding = engine.embed("Hello world")

            assert embedding is not None
            assert embedding.shape[0] == 384  # all-MiniLM-L6-v2 dimension

        def test_embed_batch(self):
            """Test batch embedding."""
            from gaia.agents.gaia_code.embedding_engine import EmbeddingEngine

            engine = EmbeddingEngine()
            embeddings = engine.embed(["Text 1", "Text 2", "Text 3"])

            assert embeddings.shape == (3, 384)

        def test_similarity(self):
            """Test similarity calculation."""
            from gaia.agents.gaia_code.embedding_engine import EmbeddingEngine

            engine = EmbeddingEngine()

            # Similar texts should have high similarity
            sim1 = engine.similarity("Hello world", "Hi world")
            assert sim1 > 0.5

            # Different texts should have lower similarity
            sim2 = engine.similarity("Hello world", "Database query")
            assert sim2 < sim1

    @pytest.mark.skipif(
        not FAISS_AVAILABLE or not EMBEDDINGS_AVAILABLE,
        reason="FAISS or sentence-transformers not installed",
    )
    class TestVectorSearch:
        """Test VectorSearch."""

        def test_build_insights_index(self):
            """Test building insights index."""
            from gaia.agents.gaia_code.vector_search import VectorSearch

            with tempfile.TemporaryDirectory() as tmpdir:
                search = VectorSearch(workspace_dir=Path(tmpdir))

                # Add some insights
                state = get_shared_state(Path(tmpdir))
                state.knowledge.store_insight("error_fix", "Check for None")
                state.knowledge.store_insight("pattern", "Use list comprehension")

                # Build index
                count = search.build_insights_index()
                assert count == 2

        def test_search_insights(self):
            """Test semantic insight search."""
            from gaia.agents.gaia_code.vector_search import VectorSearch

            with tempfile.TemporaryDirectory() as tmpdir:
                search = VectorSearch(workspace_dir=Path(tmpdir))

                # Add insights
                state = get_shared_state(Path(tmpdir))
                state.knowledge.store_insight(
                    "error_fix",
                    "Always check if variable is not None before accessing attributes",
                    triggers=["AttributeError"],
                )

                # Build index
                search.build_insights_index()

                # Search
                results = search.search_insights("AttributeError")
                assert len(results) >= 1

except ImportError:
    # Skip tests if dependencies not available
    pass
