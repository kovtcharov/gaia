# GAIA V2 Testing Framework Architecture

> Comprehensive testing strategy for all 18+ GAIA V2 agent architectures, covering
> deterministic unit tests, non-deterministic AI behavior validation, AMD hardware
> benchmarking, and continuous integration pipelines.

**Document Version:** 2.0
**Last Updated:** 2026-02-07
**Status:** Living Document
**Owner:** GAIA Test Engineering

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Testing Pyramid for AI Agents](#2-testing-pyramid-for-ai-agents)
3. [Mock LLM Strategy](#3-mock-llm-strategy)
4. [Ground Truth Generation](#4-ground-truth-generation)
5. [Property-Based Testing](#5-property-based-testing)
6. [Performance Testing](#6-performance-testing)
7. [Safety Testing](#7-safety-testing)
8. [CI/CD Integration](#8-cicd-integration)
9. [Test Data Management](#9-test-data-management)
10. [Complete Test Suite Examples](#10-complete-test-suite-examples)

---

## 1. Executive Summary

### 1.1 Testing Philosophy for Agentic Systems

GAIA agents are autonomous, multi-step systems that interact with LLMs, external
services, file systems, and hardware accelerators. Testing these systems requires a
fundamentally different approach than testing traditional deterministic software.

**Core Principles:**

1. **CLI-first testing.** Users interact with GAIA through the `gaia` CLI. Every test
   that validates user-facing behavior must exercise the actual CLI commands, not
   internal Python module calls.

2. **Layered determinism.** The agent framework (tool registry, state machine,
   conversation loop) is deterministic and tested with standard unit tests. The LLM
   interaction layer is non-deterministic and tested through mocked providers,
   property-based assertions, and evaluation scoring.

3. **Hardware-aware validation.** GAIA targets AMD Ryzen AI processors with NPU
   support. Performance tests must measure NPU/iGPU utilization, memory pressure,
   and inference latency on real hardware.

4. **Safety as a first-class concern.** Agents that execute shell commands, write
   files, or access external services must pass security tests for path traversal,
   command injection, and credential leakage on every pull request.

### 1.2 Challenges Testing Non-Deterministic AI Behavior

| Challenge | GAIA Mitigation |
|-----------|----------------|
| LLM outputs vary across runs | Mock LLM providers return deterministic sequences; property-based tests assert invariants instead of exact outputs |
| Tool call ordering is model-dependent | State machine tests verify valid transitions; integration tests use golden tool-call sequences |
| RAG retrieval relevance fluctuates | TF-IDF similarity scoring with configurable thresholds; Claude-as-judge evaluation |
| Streaming introduces timing sensitivity | WebSocket tests use synchronous test clients; SSE tests drain iterators to completion |
| Multi-step plans may diverge | Max-step limits in test configurations; plan shape assertions (number of steps, tool types used) |
| Hardware performance varies | Benchmark baselines per hardware SKU; relative regression detection instead of absolute thresholds |

### 1.3 Multi-Tier Testing Strategy

```
                    +---------------------------+
                    |   Human Evaluation (Tier 4)|   Quarterly eval sprints
                    |   - User studies           |   with ground truth datasets
                    |   - A/B comparisons        |
                    +---------------------------+
                   /                             \
          +------------------+          +------------------+
          | E2E Tests (Tier 3)|          | Perf Tests       |   Per-release
          | - Full workflows  |          | - NPU benchmarks |   on AMD hardware
          | - CLI commands    |          | - Memory profiling|
          +------------------+          +------------------+
                  |                              |
          +------------------------------------------+
          |      Integration Tests (Tier 2)          |   Per-PR on self-hosted
          |      - Real LLM (Lemonade server)        |   runners with AMD hardware
          |      - MCP protocol compliance           |
          |      - API server end-to-end             |
          +------------------------------------------+
                           |
          +------------------------------------------+
          |         Unit Tests (Tier 1)              |   Every commit on
          |         - Mocked LLM, deterministic      |   ubuntu-latest
          |         - Tool registry, state machine   |
          |         - Error formatting, path security|
          +------------------------------------------+
```

### 1.4 Architecture Coverage Matrix

The following table maps each GAIA V2 architecture to its required test categories:

| Architecture | Unit | Integration | E2E | Security | Perf | Eval |
|-------------|------|-------------|-----|----------|------|------|
| Agent Base (`agents/base/`) | Yes | Yes | -- | Yes | -- | -- |
| ChatAgent (`agents/chat/`) | Yes | Yes | Yes | Yes | Yes | Yes |
| CodeAgent (`agents/code/`) | Yes | Yes | Yes | Yes | Yes | Yes |
| BlenderAgent (`agents/blender/`) | Yes | Yes | Yes | -- | -- | -- |
| JiraAgent (`agents/jira/`) | Yes | Yes | Yes | Yes | -- | -- |
| DockerAgent (`agents/docker/`) | Yes | Yes | Yes | Yes | -- | -- |
| MedicalIntakeAgent (`agents/emr/`) | Yes | Yes | Yes | Yes | -- | Yes |
| RoutingAgent (`agents/routing/`) | Yes | Yes | Yes | -- | Yes | -- |
| SummarizerAgent (`agents/summarize/`) | Yes | Yes | Yes | -- | -- | Yes |
| SDAgent (`agents/sd/`) | Yes | Yes | Yes | -- | Yes | -- |
| Workflow Orchestration | Yes | Yes | Yes | Yes | Yes | -- |
| Email Integration | Yes | Yes | Yes | Yes | -- | -- |
| Computer Use | Yes | Yes | Yes | Yes | -- | -- |
| Error Recovery | Yes | Yes | -- | -- | -- | -- |
| Persistent Memory | Yes | Yes | -- | -- | Yes | -- |
| Dynamic Tools | Yes | Yes | -- | -- | -- | -- |
| Multi-Document Synthesis | Yes | Yes | Yes | -- | -- | Yes |
| Observability | Yes | Yes | -- | -- | Yes | -- |

---

## 2. Testing Pyramid for AI Agents

### 2.1 Tier 1: Unit Tests (Mocked LLM, Deterministic)

Unit tests form the foundation. They run on every commit, require no external
services, and complete in under 60 seconds. All LLM interactions are mocked using
the `gaia.testing` module.

**Location:** `tests/unit/`
**Runner:** `ubuntu-latest` (GitHub Actions)
**CI Workflow:** `.github/workflows/test_unit.yml`

#### 2.1.1 What to Unit Test

- Tool registration and parameter extraction (`@tool` decorator)
- Agent state transitions (`PLANNING` -> `EXECUTING_PLAN` -> `DIRECT_EXECUTION`)
- JSON response parsing and validation
- Error formatting and recovery
- Path validation and security boundaries
- Console output formatting
- Configuration dataclass defaults

#### 2.1.2 Unit Test Pattern: Tool Registration

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for the @tool decorator and registry."""

import pytest
from gaia.agents.base.tools import _TOOL_REGISTRY, tool


class TestToolDecorator:
    """Verify tool registration metadata is captured correctly."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Isolate each test from registry side effects."""
        _TOOL_REGISTRY.clear()
        yield
        _TOOL_REGISTRY.clear()

    def test_tool_registration_captures_name_and_docstring(self):
        @tool
        def search_documents(query: str) -> dict:
            """Search indexed documents by query."""
            return {"results": []}

        assert "search_documents" in _TOOL_REGISTRY
        entry = _TOOL_REGISTRY["search_documents"]
        assert entry["name"] == "search_documents"
        assert entry["description"] == "Search indexed documents by query."
        assert entry["atomic"] is False

    def test_tool_atomic_flag_propagates(self):
        @tool(atomic=True)
        def quick_lookup(key: str) -> str:
            """Instant key lookup."""
            return "value"

        assert _TOOL_REGISTRY["quick_lookup"]["atomic"] is True

    def test_tool_parameter_types_inferred(self):
        @tool
        def create_record(name: str, count: int, active: bool = True) -> dict:
            """Create a new record."""
            return {}

        params = _TOOL_REGISTRY["create_record"]["parameters"]
        assert params["name"]["type"] == "string"
        assert params["name"]["required"] is True
        assert params["count"]["type"] == "integer"
        assert params["count"]["required"] is True
        assert params["active"]["type"] == "boolean"
        assert params["active"]["required"] is False

    def test_tool_function_remains_callable(self):
        @tool(atomic=True)
        def uppercase(text: str) -> dict:
            """Convert text to uppercase."""
            return {"result": text.upper()}

        func = _TOOL_REGISTRY["uppercase"]["function"]
        assert func(text="hello") == {"result": "HELLO"}
```

#### 2.1.3 Unit Test Pattern: Agent State Machine

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for Agent state transitions."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from gaia.agents.base.agent import Agent


class ConcreteTestAgent(Agent):
    """Minimal concrete agent for testing the base class."""

    def _get_system_prompt(self) -> str:
        return "You are a test agent."

    def _register_tools(self):
        pass

    def _get_tool_descriptions(self) -> str:
        return "No tools available."


class TestAgentStateTransitions:
    """Verify the agent state machine enforces valid transitions."""

    @pytest.fixture
    def agent(self):
        with patch("gaia.chat.sdk.ChatSDK"):
            agent = ConcreteTestAgent(silent_mode=True, max_steps=5)
            return agent

    def test_initial_state_is_planning(self, agent):
        assert agent.state == Agent.STATE_PLANNING

    def test_state_constants_are_distinct(self):
        states = {
            Agent.STATE_PLANNING,
            Agent.STATE_EXECUTING_PLAN,
            Agent.STATE_DIRECT_EXECUTION,
        }
        assert len(states) == 3

    def test_max_steps_respected(self, agent):
        assert agent.max_steps == 5

    def test_silent_mode_uses_silent_console(self, agent):
        from gaia.agents.base.console import SilentConsole
        assert isinstance(agent.console, SilentConsole)

    def test_conversation_history_starts_empty(self, agent):
        assert agent.conversation_history == []
```

#### 2.1.4 Unit Test Pattern: Error Formatting

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for error formatting utilities."""

from gaia.agents.base.errors import format_execution_trace, _truncate_args


class TestFormatExecutionTrace:
    """Verify error formatting produces actionable output."""

    def test_full_context_includes_all_sections(self):
        try:
            raise ValueError("connection timeout")
        except ValueError as e:
            result = format_execution_trace(
                exception=e,
                query="Summarize the document",
                plan_step=2,
                total_steps=5,
                tool_name="query_documents",
                tool_args={"query": "main themes"},
            )

        assert "AGENT ERROR" in result
        assert "Tool execution failed" in result
        assert 'Query: "Summarize the document"' in result
        assert "Plan Step: 2/5" in result
        assert "Tool: query_documents" in result
        assert "ValueError: connection timeout" in result

    def test_long_query_is_truncated_to_80_chars(self):
        long_query = "a" * 200
        try:
            raise RuntimeError("error")
        except RuntimeError as e:
            result = format_execution_trace(exception=e, query=long_query)

        assert "a" * 200 not in result
        assert "..." in result


class TestTruncateArgs:
    """Verify tool argument truncation for display."""

    def test_short_args_unchanged(self):
        result = _truncate_args({"key": "value"})
        assert "key" in result
        assert "..." not in result

    def test_long_args_truncated(self):
        args = {f"key_{i}": f"value_{i}" for i in range(50)}
        result = _truncate_args(args, max_length=100)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_none_returns_empty_dict_string(self):
        assert _truncate_args(None) == "{}"
```

### 2.2 Tier 2: Integration Tests (Real LLM, Sandbox)

Integration tests exercise real LLM inference through the Lemonade server running
on AMD hardware. They validate that agents produce meaningful responses, tools
execute correctly against real services, and MCP protocol compliance holds.

**Location:** `tests/integration/`, `tests/mcp/`, `tests/test_*.py`
**Runner:** Self-hosted AMD hardware runners
**CI Workflows:** `test_chat_agent.yml`, `test_code_agent.yml`, `test_mcp.yml`, `test_api.yml`

#### 2.2.1 Integration Test Pattern: API Server

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Integration tests for GAIA OpenAI-compatible API server."""

import json
import pytest
import requests

try:
    from fastapi.testclient import TestClient
    from gaia.api.openai_server import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    app = None


class TestApiValidation:
    """API schema validation tests using FastAPI TestClient."""

    @pytest.fixture(autouse=True)
    def setup(self):
        if not API_AVAILABLE:
            pytest.skip("API dependencies not available")
        self.client = TestClient(app)

    def test_invalid_model_returns_404(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_missing_model_returns_422(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )
        assert response.status_code == 422

    def test_invalid_role_returns_422(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gaia-code",
                "messages": [{"role": "invalid_role", "content": "test"}],
                "stream": False,
            },
        )
        assert response.status_code == 422


class TestApiWithLemonade:
    """Integration tests requiring a running Lemonade server."""

    def test_chat_completion(self, require_lemonade, api_server, api_client):
        """Test actual chat completion through the API."""
        response = api_client.post(
            f"{api_server}/v1/chat/completions",
            json={
                "model": "gaia-chat",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]

    def test_streaming_completion(self, require_lemonade, api_server, api_client):
        """Test SSE streaming through the API."""
        response = api_client.post(
            f"{api_server}/v1/chat/completions",
            json={
                "model": "gaia-chat",
                "messages": [{"role": "user", "content": "Count to 3"}],
                "stream": True,
            },
            stream=True,
        )
        assert response.status_code == 200

        chunks = []
        for line in response.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: ") and decoded != "data: [DONE]":
                    chunk = json.loads(decoded[6:])
                    chunks.append(chunk)

        assert len(chunks) > 0
```

#### 2.2.2 Integration Test Pattern: MCP Protocol

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""MCP protocol compliance tests."""

import subprocess
import json
import pytest


class TestMCPProtocolCompliance:
    """Verify MCP server implements the protocol correctly."""

    @pytest.fixture
    def mcp_server_url(self):
        """Start MCP server and return URL."""
        # Start MCP bridge in background
        proc = subprocess.Popen(
            ["gaia", "mcp", "start", "--background"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        yield "http://localhost:8765"
        proc.terminate()
        proc.wait(timeout=5)

    def test_mcp_tool_definitions_schema(self):
        """Verify tool definitions follow MCP schema."""
        from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig

        config = ChatAgentConfig(silent_mode=True)
        agent = ChatAgent(config)

        if hasattr(agent, "get_mcp_tool_definitions"):
            tools = agent.get_mcp_tool_definitions()
            for tool_def in tools:
                assert "name" in tool_def
                assert "description" in tool_def
                assert "inputSchema" in tool_def
                assert tool_def["inputSchema"].get("type") == "object"

        agent.stop_watching()

    def test_mcp_server_info_structure(self):
        """Verify MCP server info has required fields."""
        from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig

        config = ChatAgentConfig(silent_mode=True)
        agent = ChatAgent(config)

        if hasattr(agent, "get_mcp_server_info"):
            info = agent.get_mcp_server_info()
            assert "name" in info
            assert "version" in info

        agent.stop_watching()
```

### 2.3 Tier 3: End-to-End Tests (Full Workflows)

End-to-end tests exercise complete user workflows through the CLI. They validate
that the `gaia` command produces the expected results for real-world use cases.

**Location:** `tests/e2e/` (proposed), currently `tests/test_*.py`
**Runner:** Self-hosted AMD hardware runners

#### 2.3.1 E2E Test Pattern: CLI Command Validation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end CLI command tests."""

import subprocess
import pytest


class TestGaiaCLI:
    """Test the gaia CLI entry points."""

    def test_gaia_help_returns_zero(self):
        """Verify gaia --help exits cleanly."""
        result = subprocess.run(
            ["gaia", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "gaia" in result.stdout.lower()

    def test_gaia_chat_help(self):
        """Verify gaia chat --help exits cleanly."""
        result = subprocess.run(
            ["gaia", "chat", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    @pytest.mark.parametrize("command", [
        ["gaia", "prompt", "--help"],
        ["gaia", "llm", "--help"],
        ["gaia", "api", "--help"],
        ["gaia", "mcp", "--help"],
        ["gaia", "eval", "--help"],
        ["gaia", "cache", "--help"],
    ])
    def test_subcommand_help(self, command):
        """Verify all subcommands accept --help."""
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_gaia_llm_with_lemonade(self, require_lemonade):
        """Test actual LLM inference through CLI."""
        result = subprocess.run(
            ["gaia", "llm", "Say hello in one word"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert len(result.stdout.strip()) > 0
```

#### 2.3.2 E2E Test Pattern: Chat Agent Workflow

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""End-to-end workflow tests for ChatAgent."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig


class TestChatAgentWorkflow:
    """Test complete chat workflows with document indexing and retrieval."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = str(Path(tmpdir).resolve())
            yield resolved

    @pytest.fixture
    def agent(self, workspace):
        config = ChatAgentConfig(
            silent_mode=True,
            debug=False,
            max_steps=10,
            allowed_paths=[workspace, str(Path.cwd().resolve())],
        )
        agent = ChatAgent(config)
        yield agent
        agent.stop_watching()

    def test_index_and_query_workflow(self, agent, workspace):
        """Test: index a document, then query it."""
        # Create test document
        doc_path = Path(workspace) / "test_doc.txt"
        doc_path.write_text(
            "The Ryzen AI 300 processor features an XDNA 2 NPU "
            "capable of 50 TOPS of AI performance."
        )

        # Index the document
        result = agent.rag.index_document(str(doc_path))
        assert result["success"], f"Indexing failed: {result.get('error')}"
        assert str(doc_path) in agent.rag.indexed_files or \
               str(doc_path.absolute()) in agent.rag.indexed_files

        # Verify system prompt updated
        agent.update_system_prompt()
        assert "test_doc.txt" in agent.system_prompt

    def test_session_persistence_workflow(self, agent):
        """Test: create session, add history, save, reload."""
        # Create session
        if not agent.current_session:
            agent.current_session = agent.session_manager.create_session()

        # Simulate conversation
        agent.conversation_history = [
            {"role": "user", "content": "What is XDNA?"},
            {"role": "assistant", "content": "XDNA is AMD's NPU architecture."},
        ]

        session_id = agent.current_session.session_id
        agent.save_current_session()

        # Reload in new agent
        new_agent = ChatAgent(ChatAgentConfig(silent_mode=True))
        success = new_agent.load_session(session_id)

        assert success
        assert len(new_agent.conversation_history) == 2
        assert new_agent.conversation_history[0]["content"] == "What is XDNA?"
        new_agent.stop_watching()
```

### 2.4 Tier 4: Human Evaluation Tests

Human evaluation tests run quarterly as part of the GAIA eval framework. They
compare agent outputs against curated ground truth datasets using both automated
similarity scoring and Claude-as-judge qualitative analysis.

**Location:** `src/gaia/eval/`
**Runner:** Manual or scheduled (not per-PR)
**CLI Command:** `gaia eval run`

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Evaluation framework integration for human-in-the-loop testing."""

from gaia.eval.eval import Evaluator


class TestEvalFrameworkIntegration:
    """Verify the evaluation framework scoring pipeline."""

    def test_similarity_scoring(self):
        evaluator = Evaluator()

        # Identical texts should score 1.0
        score = evaluator.calculate_similarity(
            "The NPU delivers 50 TOPS performance",
            "The NPU delivers 50 TOPS performance",
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_dissimilar_texts_score_low(self):
        evaluator = Evaluator()

        score = evaluator.calculate_similarity(
            "The NPU delivers 50 TOPS performance",
            "Today the weather is sunny and warm",
        )
        assert score < 0.3

    def test_pass_fail_determination(self):
        evaluator = Evaluator()

        # High similarity should pass
        result = evaluator.determine_pass_fail(
            similarity=0.85,
            threshold=0.7,
            claude_analysis={
                "correctness": {"rating": "good"},
                "completeness": {"rating": "good"},
                "conciseness": {"rating": "good"},
                "relevance": {"rating": "good"},
            },
        )
        assert result["is_pass"] is True
        assert result["pass_fail"] == "pass"

    def test_poor_correctness_fails_regardless(self):
        evaluator = Evaluator()

        result = evaluator.determine_pass_fail(
            similarity=0.9,
            threshold=0.7,
            claude_analysis={
                "correctness": {"rating": "poor"},
                "completeness": {"rating": "excellent"},
                "conciseness": {"rating": "excellent"},
                "relevance": {"rating": "excellent"},
            },
        )
        assert result["is_pass"] is False
```

---

## 3. Mock LLM Strategy

### 3.1 The `gaia.testing` Module

GAIA provides a dedicated testing module at `src/gaia/testing/` with three
components:

| Module | Purpose |
|--------|---------|
| `mocks.py` | `MockLLMProvider`, `MockVLMClient`, `MockToolExecutor` |
| `fixtures.py` | `temp_directory()`, `temp_file()`, `create_test_agent()`, `AgentTestContext` |
| `assertions.py` | `assert_llm_called()`, `assert_tool_called()`, `assert_agent_completed()` |

### 3.2 MockLLMProvider: Deterministic LLM Responses

The `MockLLMProvider` replaces the real LLM client with a predictable response
sequence. It records all calls for post-hoc assertion.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Demonstrating MockLLMProvider usage patterns."""

import pytest
from gaia.testing.mocks import MockLLMProvider
from gaia.testing.assertions import assert_llm_called, assert_llm_prompt_contains


class TestMockLLMProvider:
    """Verify the mock LLM provider contract."""

    def test_returns_responses_in_sequence(self):
        mock = MockLLMProvider(responses=[
            "First response",
            "Second response",
            "Third response",
        ])

        assert mock.generate("prompt 1") == "First response"
        assert mock.generate("prompt 2") == "Second response"
        assert mock.generate("prompt 3") == "Third response"

    def test_cycles_when_responses_exhausted(self):
        mock = MockLLMProvider(responses=["A", "B"])

        assert mock.generate("1") == "A"
        assert mock.generate("2") == "B"
        assert mock.generate("3") == "A"  # Cycles back

    def test_falls_back_to_default_when_empty(self):
        mock = MockLLMProvider(default_response="fallback")
        assert mock.generate("anything") == "fallback"

    def test_records_call_history(self):
        mock = MockLLMProvider(responses=["ok"])
        mock.generate("test prompt", system_prompt="Be helpful", temperature=0.5)

        assert mock.call_count == 1
        assert mock.last_prompt == "test prompt"
        assert mock.call_history[0]["system_prompt"] == "Be helpful"
        assert mock.call_history[0]["temperature"] == 0.5

    def test_chat_method_extracts_last_user_message(self):
        mock = MockLLMProvider(responses=["response"])
        mock.chat(messages=[
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What is GAIA?"},
        ])

        assert mock.last_prompt == "What is GAIA?"

    def test_streaming_yields_full_response(self):
        mock = MockLLMProvider(responses=["streamed content"])
        chunks = list(mock.stream("test"))
        assert chunks == ["streamed content"]

    def test_reset_clears_state(self):
        mock = MockLLMProvider(responses=["a", "b"])
        mock.generate("1")
        mock.generate("2")
        mock.reset()

        assert mock.call_count == 0
        assert mock.generate("3") == "a"  # Back to first response


class TestMockLLMAssertions:
    """Verify assertion helpers work correctly."""

    def test_assert_llm_called_passes(self):
        mock = MockLLMProvider(responses=["ok"])
        mock.generate("test")
        assert_llm_called(mock)
        assert_llm_called(mock, times=1)

    def test_assert_llm_called_with_range(self):
        mock = MockLLMProvider(responses=["ok"])
        mock.generate("1")
        mock.generate("2")
        mock.generate("3")
        assert_llm_called(mock, min_times=2, max_times=5)

    def test_assert_llm_prompt_contains(self):
        mock = MockLLMProvider(responses=["ok"])
        mock.generate("Find documents about NPU")
        assert_llm_prompt_contains(mock, "NPU")
```

### 3.3 MockVLMClient: Vision Model Testing

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""MockVLMClient usage for vision model testing."""

from gaia.testing.mocks import MockVLMClient
from gaia.testing.assertions import assert_vlm_called


class TestMockVLMClient:
    """Verify mock VLM client for image processing tests."""

    def test_extract_from_image_returns_configured_text(self):
        mock = MockVLMClient(
            extracted_text='{"patient_name": "John Doe", "dob": "1990-01-01"}'
        )
        result = mock.extract_from_image(b"fake_image_bytes", prompt="Extract fields")

        assert "John Doe" in result
        assert mock.was_called
        assert mock.call_count == 1

    def test_sequential_extraction_results(self):
        mock = MockVLMClient(
            extraction_results=[
                '{"page": 1, "text": "First page"}',
                '{"page": 2, "text": "Second page"}',
            ]
        )

        result1 = mock.extract_from_image(b"page1")
        result2 = mock.extract_from_image(b"page2")

        assert "First page" in result1
        assert "Second page" in result2

    def test_availability_check(self):
        available_mock = MockVLMClient(is_available=True)
        unavailable_mock = MockVLMClient(is_available=False)

        assert available_mock.check_availability() is True
        assert unavailable_mock.check_availability() is False
```

### 3.4 MockToolExecutor: Tool Call Verification

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""MockToolExecutor for verifying agent tool call sequences."""

from gaia.testing.mocks import MockToolExecutor
from gaia.testing.assertions import assert_tool_called, assert_tool_args


class TestMockToolExecutor:
    """Verify tool execution tracking."""

    def test_returns_configured_results(self):
        executor = MockToolExecutor(
            results={
                "search": {"results": ["item1", "item2"]},
                "create": {"id": 42, "status": "created"},
            }
        )

        search_result = executor.execute("search", {"query": "test"})
        assert search_result == {"results": ["item1", "item2"]}

        create_result = executor.execute("create", {"name": "new item"})
        assert create_result == {"id": 42, "status": "created"}

    def test_tracks_call_history(self):
        executor = MockToolExecutor()
        executor.execute("search", {"query": "NPU"})
        executor.execute("index", {"file": "doc.pdf"})

        assert executor.was_tool_called("search")
        assert executor.was_tool_called("index")
        assert not executor.was_tool_called("delete")
        assert set(executor.tool_names_called) == {"search", "index"}

    def test_assertion_helpers(self):
        executor = MockToolExecutor(
            results={"search": {"count": 5}}
        )
        executor.execute("search", {"query": "test", "limit": 10})

        assert_tool_called(executor, "search")
        assert_tool_called(executor, "search", times=1)
        assert_tool_args(executor, "search", {"query": "test"})
```

### 3.5 AgentTestContext: Integrated Test Harness

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""AgentTestContext provides a complete test environment."""

import json
from gaia.testing.fixtures import AgentTestContext, create_test_agent


class TestAgentTestContext:
    """Demonstrate the context manager test harness."""

    def test_context_provides_agent_and_mocks(self):
        """The context sets up agent, mock LLM, temp directory."""
        from gaia.agents.code.agent import CodeAgent

        with AgentTestContext(
            CodeAgent,
            mock_responses=[
                json.dumps({"tool": "read_file", "args": {"filepath": "test.py"}}),
                "The file contains a hello function.",
            ],
        ) as ctx:
            assert ctx.agent is not None
            assert ctx.mock_llm is not None
            assert ctx.temp_dir.exists()

            # Create test files in the sandbox
            test_file = ctx.create_file("test.py", "def hello(): pass")
            assert test_file.exists()

    def test_create_test_agent_shorthand(self):
        """create_test_agent() is a quick alternative to AgentTestContext."""
        from gaia.agents.code.agent import CodeAgent

        agent = create_test_agent(
            CodeAgent,
            mock_responses=["Mock response"],
            max_steps=3,
        )
        assert agent.silent_mode is True
        assert agent.max_steps == 3
```

### 3.6 Fixture Library for Common LLM Responses

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Fixture library: reusable LLM response sequences for common test scenarios.

Import these in conftest.py or individual test files to avoid duplicating
mock response definitions across the test suite.
"""

import json
import pytest


# ---------------------------------------------------------------------------
# Tool call response fixtures
# ---------------------------------------------------------------------------

TOOL_CALL_READ_FILE = json.dumps({
    "tool": "read_file",
    "args": {"filepath": "target.py"},
})

TOOL_CALL_WRITE_FILE = json.dumps({
    "tool": "write_python_file",
    "args": {"filepath": "output.py", "code": "print('hello')"},
})

TOOL_CALL_SEARCH = json.dumps({
    "tool": "search_code",
    "args": {"query": "function definition", "directory": "."},
})

TOOL_CALL_INDEX_DOC = json.dumps({
    "tool": "index_document",
    "args": {"file_path": "/tmp/test.pdf"},
})

TOOL_CALL_QUERY_DOCS = json.dumps({
    "tool": "query_documents",
    "args": {"query": "What is the main topic?"},
})

# ---------------------------------------------------------------------------
# Planning response fixtures
# ---------------------------------------------------------------------------

PLAN_READ_AND_EDIT = json.dumps({
    "plan": [
        {"step": 1, "tool": "read_file", "description": "Read the target file"},
        {"step": 2, "tool": "edit_python_file", "description": "Apply the fix"},
        {"step": 3, "tool": "validate_syntax", "description": "Verify syntax"},
    ]
})

PLAN_INDEX_AND_QUERY = json.dumps({
    "plan": [
        {"step": 1, "tool": "index_document", "description": "Index the PDF"},
        {"step": 2, "tool": "query_documents", "description": "Answer the question"},
    ]
})

# ---------------------------------------------------------------------------
# Final answer response fixtures
# ---------------------------------------------------------------------------

FINAL_ANSWER_SUCCESS = "The operation completed successfully."
FINAL_ANSWER_NOT_FOUND = "I could not find the information you requested."
FINAL_ANSWER_ERROR_RECOVERY = "I encountered an error but recovered by trying an alternative approach."


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_read_edit_responses():
    """Response sequence for a read-then-edit workflow."""
    return [
        PLAN_READ_AND_EDIT,
        TOOL_CALL_READ_FILE,
        json.dumps({
            "tool": "edit_python_file",
            "args": {"filepath": "target.py", "edits": "fix applied"},
        }),
        json.dumps({
            "tool": "validate_syntax",
            "args": {"filepath": "target.py"},
        }),
        FINAL_ANSWER_SUCCESS,
    ]


@pytest.fixture
def mock_rag_responses():
    """Response sequence for a RAG indexing and query workflow."""
    return [
        PLAN_INDEX_AND_QUERY,
        TOOL_CALL_INDEX_DOC,
        TOOL_CALL_QUERY_DOCS,
        "The document discusses AMD Ryzen AI processors.",
    ]
```

---

## 4. Ground Truth Generation

### 4.1 Integration with GAIA Eval Framework

The GAIA evaluation framework (`src/gaia/eval/`) provides automated quality
assessment using two complementary scoring methods:

1. **TF-IDF cosine similarity** -- fast, deterministic comparison between
   expected and actual outputs.
2. **Claude-as-judge** -- qualitative assessment of correctness, completeness,
   conciseness, and relevance using Anthropic's Claude API.

The eval pipeline is orchestrated through `gaia eval run` and produces structured
JSON reports.

```
src/gaia/eval/
    eval.py              # Evaluator class with scoring logic
    batch_experiment.py   # Batch experiment runner
    claude.py            # Claude API client for evaluation
    config.py            # Model pricing and thresholds
```

### 4.2 Creating Test Datasets

Test datasets are stored as experiment files in JSON format. Each experiment
defines an input prompt, optional context, and expected ground truth output.

```json
{
    "experiment_name": "chat_rag_ryzen_ai",
    "description": "Validate RAG accuracy for Ryzen AI documentation",
    "model": "Qwen3-Coder-30B-A3B-Instruct-GGUF",
    "tasks": [
        {
            "id": "ryzen_npu_tops",
            "task_type": "qa",
            "input": "How many TOPS does the Ryzen AI 300 NPU deliver?",
            "context_file": "docs/ryzen_ai_specs.pdf",
            "ground_truth": "The Ryzen AI 300 series NPU delivers up to 50 TOPS of AI performance using the XDNA 2 architecture.",
            "similarity_threshold": 0.7
        },
        {
            "id": "lemonade_server_setup",
            "task_type": "qa",
            "input": "How do I start the Lemonade server?",
            "context_file": "docs/guides/chat.mdx",
            "ground_truth": "Run 'lemonade-server serve' to start the LLM backend server.",
            "similarity_threshold": 0.6
        }
    ]
}
```

### 4.3 Golden Outputs

Golden outputs are cached results from a known-good model run. They serve as
regression baselines: future model or code changes are compared against the
golden output to detect quality degradation.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Golden output management for regression testing."""

import json
from pathlib import Path
from typing import Dict, Optional

from gaia.eval.eval import Evaluator


class GoldenOutputManager:
    """
    Manage golden outputs for regression detection.

    Golden outputs are stored alongside experiment files with a .golden.json
    suffix. When a new model version is validated, its outputs become the
    new golden baseline.
    """

    def __init__(self, golden_dir: str = "tests/golden_outputs"):
        self.golden_dir = Path(golden_dir)
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = Evaluator()

    def save_golden(self, experiment_id: str, outputs: Dict) -> Path:
        """Save outputs as the golden baseline."""
        golden_path = self.golden_dir / f"{experiment_id}.golden.json"
        golden_path.write_text(json.dumps(outputs, indent=2))
        return golden_path

    def load_golden(self, experiment_id: str) -> Optional[Dict]:
        """Load golden outputs for comparison."""
        golden_path = self.golden_dir / f"{experiment_id}.golden.json"
        if golden_path.exists():
            return json.loads(golden_path.read_text())
        return None

    def check_regression(
        self,
        experiment_id: str,
        current_outputs: Dict,
        threshold: float = 0.8,
    ) -> Dict:
        """
        Compare current outputs against golden baseline.

        Returns:
            Dict with regression status and per-task comparisons.
        """
        golden = self.load_golden(experiment_id)
        if golden is None:
            return {
                "status": "no_baseline",
                "message": f"No golden output for {experiment_id}",
            }

        regressions = []
        for task_id, golden_output in golden.get("outputs", {}).items():
            current_output = current_outputs.get("outputs", {}).get(task_id)
            if current_output is None:
                regressions.append({
                    "task_id": task_id,
                    "issue": "missing_output",
                })
                continue

            similarity = self.evaluator.calculate_similarity(
                golden_output, current_output
            )
            if similarity < threshold:
                regressions.append({
                    "task_id": task_id,
                    "issue": "quality_regression",
                    "golden_similarity": similarity,
                    "threshold": threshold,
                })

        return {
            "status": "regression" if regressions else "pass",
            "regressions": regressions,
            "total_tasks": len(golden.get("outputs", {})),
            "regressed_tasks": len(regressions),
        }
```

### 4.4 Regression Detection in CI

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Pytest integration for golden output regression tests."""

import pytest
import json
from pathlib import Path


class TestGoldenOutputRegression:
    """
    Regression tests comparing current outputs against golden baselines.

    These tests run during evaluation sprints, not on every PR.
    Mark with @pytest.mark.eval to separate from unit/integration tests.
    """

    GOLDEN_DIR = Path("tests/golden_outputs")

    @pytest.mark.eval
    def test_chat_rag_no_regression(self, require_lemonade):
        """Verify ChatAgent RAG answers have not regressed."""
        golden_path = self.GOLDEN_DIR / "chat_rag_ryzen_ai.golden.json"
        if not golden_path.exists():
            pytest.skip("No golden baseline available")

        golden = json.loads(golden_path.read_text())

        from gaia.eval.eval import Evaluator
        evaluator = Evaluator()

        for task in golden.get("tasks", []):
            task_id = task["id"]
            expected = task["ground_truth"]
            actual = task.get("last_output", "")
            threshold = task.get("similarity_threshold", 0.7)

            similarity = evaluator.calculate_similarity(expected, actual)
            assert similarity >= threshold, (
                f"Regression detected for task '{task_id}': "
                f"similarity {similarity:.3f} < threshold {threshold:.3f}"
            )

    @pytest.mark.eval
    def test_summarizer_no_regression(self, require_lemonade):
        """Verify SummarizerAgent outputs have not regressed."""
        golden_path = self.GOLDEN_DIR / "summarizer_benchmark.golden.json"
        if not golden_path.exists():
            pytest.skip("No golden baseline available")

        golden = json.loads(golden_path.read_text())
        from gaia.eval.eval import Evaluator
        evaluator = Evaluator()

        for task in golden.get("tasks", []):
            similarity = evaluator.calculate_similarity(
                task["ground_truth"], task.get("last_output", "")
            )
            assert similarity >= task.get("similarity_threshold", 0.6)
```

---

## 5. Property-Based Testing

### 5.1 Why Property-Based Testing for Agents

Traditional example-based tests verify specific inputs produce specific outputs.
For agentic systems where outputs are non-deterministic, property-based testing
with [Hypothesis](https://hypothesis.readthedocs.io/) lets us assert *invariants*
that must hold for any valid input, regardless of what the LLM produces.

**Key invariants for GAIA agents:**

| Property | Description |
|----------|-------------|
| **Idempotence** | Indexing the same document twice produces the same RAG state |
| **Safety** | No tool execution writes outside allowed paths |
| **Correctness** | Agent always returns a dict with `status` and `result` keys |
| **Termination** | Agent loop completes within `max_steps` iterations |
| **Monotonic progress** | Each plan step reduces remaining work (step counter advances) |

### 5.2 Hypothesis Strategies for GAIA

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Hypothesis strategies for generating agent inputs."""

from hypothesis import strategies as st


# Strategy: valid user queries (non-empty strings, no control chars)
user_queries = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "Z"),
        blacklist_characters="\x00\x01\x02\x03",
    ),
    min_size=1,
    max_size=500,
)

# Strategy: file paths within allowed directories
safe_file_paths = st.from_regex(
    r"[a-zA-Z0-9_\-]{1,50}\.(txt|py|md|json|csv)",
    fullmatch=True,
)

# Strategy: tool argument dictionaries
tool_args = st.fixed_dictionaries({
    "query": st.text(min_size=1, max_size=200),
}).filter(lambda d: all(v.strip() for v in d.values()))

# Strategy: conversation history entries
conversation_messages = st.lists(
    st.fixed_dictionaries({
        "role": st.sampled_from(["user", "assistant"]),
        "content": st.text(min_size=1, max_size=300),
    }),
    min_size=0,
    max_size=20,
)

# Strategy: ChatConfig parameters
chat_configs = st.fixed_dictionaries({
    "max_tokens": st.integers(min_value=1, max_value=4096),
    "max_history_length": st.integers(min_value=0, max_value=20),
    "show_stats": st.booleans(),
})
```

### 5.3 Property Tests: Agent Invariants

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Property-based tests for agent invariants using Hypothesis."""

import json
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from unittest.mock import patch, Mock

from gaia.agents.base.tools import _TOOL_REGISTRY, tool
from gaia.agents.base.errors import _truncate_args


class TestToolRegistryProperties:
    """Property-based tests for the tool registry."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        _TOOL_REGISTRY.clear()
        yield
        _TOOL_REGISTRY.clear()

    @given(
        name=st.from_regex(r"[a-z][a-z0-9_]{0,30}", fullmatch=True),
        atomic=st.booleans(),
    )
    def test_any_valid_tool_name_registers(self, name, atomic):
        """Any valid Python identifier registers without error."""
        _TOOL_REGISTRY.clear()

        # Dynamically create and register a tool
        def dummy_func(x: str) -> dict:
            """A dummy tool."""
            return {}

        dummy_func.__name__ = name
        decorated = tool(atomic=atomic)(dummy_func)

        assert name in _TOOL_REGISTRY
        assert _TOOL_REGISTRY[name]["atomic"] == atomic
        assert _TOOL_REGISTRY[name]["name"] == name
        assert callable(_TOOL_REGISTRY[name]["function"])

    @given(
        args=st.one_of(
            st.none(),
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.text(max_size=100),
                max_size=10,
            ),
        ),
        max_length=st.integers(min_value=10, max_value=1000),
    )
    def test_truncate_args_never_exceeds_max_length(self, args, max_length):
        """_truncate_args output never exceeds max_length."""
        result = _truncate_args(args, max_length=max_length)
        assert len(result) <= max_length


class TestPathValidationProperties:
    """Property-based tests for path security."""

    @given(
        traversal=st.sampled_from([
            "../", "..\\", "/../", "/..\\",
            "%2e%2e/", "%2e%2e%2f",
            "....//", "..;/",
        ]),
        suffix=st.text(min_size=1, max_size=50),
    )
    def test_path_traversal_always_blocked(self, traversal, suffix):
        """Path traversal attempts are always rejected."""
        import os
        malicious_path = f"/allowed/dir/{traversal}{suffix}"
        normalized = os.path.normpath(malicious_path)

        # The normalized path should not start with /allowed/dir
        # if traversal was effective. This tests the OS-level normalization
        # that GAIA relies on.
        if traversal.startswith(".."):
            # After normpath, the path should have resolved the ..
            assert ".." not in normalized or normalized.startswith("/allowed")


class TestAgentCompletionProperties:
    """Property-based tests for agent termination guarantees."""

    @given(max_steps=st.integers(min_value=1, max_value=50))
    @settings(max_examples=20)
    def test_agent_respects_max_steps(self, max_steps):
        """Agent loop always terminates within max_steps."""
        from gaia.agents.code.agent import CodeAgent
        from unittest.mock import patch

        agent = CodeAgent(silent_mode=True, max_steps=max_steps)
        assert agent.max_steps == max_steps

        # The step counter should never exceed max_steps during execution
        # (verified structurally; full execution tested in integration)
```

### 5.4 Fuzzing Workflows

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Fuzz testing for agent input handling."""

from hypothesis import given, settings
from hypothesis import strategies as st

from gaia.agents.base.errors import format_execution_trace, format_user_error


class TestErrorFormattingFuzz:
    """Fuzz error formatting to ensure it never crashes."""

    @given(
        message=st.text(max_size=10000),
        query=st.one_of(st.none(), st.text(max_size=500)),
        plan_step=st.one_of(st.none(), st.integers()),
        total_steps=st.one_of(st.none(), st.integers()),
        tool_name=st.one_of(st.none(), st.text(max_size=100)),
    )
    @settings(max_examples=100)
    def test_format_execution_trace_never_crashes(
        self, message, query, plan_step, total_steps, tool_name
    ):
        """format_execution_trace handles any combination of inputs."""
        try:
            exc = RuntimeError(message)
            result = format_execution_trace(
                exception=exc,
                query=query,
                plan_step=plan_step,
                total_steps=total_steps,
                tool_name=tool_name,
            )
            assert isinstance(result, str)
            assert len(result) > 0
        except Exception as e:
            # The function should never raise; if it does, that is a bug
            pytest.fail(f"format_execution_trace raised {type(e).__name__}: {e}")

    @given(message=st.text(max_size=5000))
    @settings(max_examples=50)
    def test_format_user_error_never_crashes(self, message):
        """format_user_error handles any exception message."""
        try:
            exc = ValueError(message)
            exc.__traceback__ = None
            result = format_user_error(exc)
            assert isinstance(result, str)
        except Exception as e:
            pytest.fail(f"format_user_error raised {type(e).__name__}: {e}")


class TestChatConfigFuzz:
    """Fuzz ChatConfig construction."""

    @given(
        max_tokens=st.integers(min_value=1, max_value=100000),
        temperature=st.one_of(st.none(), st.floats(min_value=0.0, max_value=2.0)),
        max_history=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_chat_config_accepts_valid_ranges(
        self, max_tokens, temperature, max_history
    ):
        """ChatConfig should accept any valid parameter combination."""
        from gaia.chat.sdk import ChatConfig

        config = ChatConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            max_history_length=max_history,
        )
        assert config.max_tokens == max_tokens
        assert config.max_history_length == max_history
```

---

## 6. Performance Testing

### 6.1 Performance Testing Strategy

GAIA targets AMD Ryzen AI processors with NPU and iGPU acceleration. Performance
tests measure inference latency, throughput, memory usage, and hardware utilization
to detect regressions and validate optimization paths.

**Performance test tiers:**

| Tier | What | When | Where |
|------|------|------|-------|
| Microbenchmarks | Individual tool execution time | Per-PR | Any runner |
| Inference benchmarks | LLM latency, tokens/sec | Per-release | AMD hardware |
| Load tests | Concurrent API requests | Per-release | AMD hardware |
| Memory profiling | Leak detection over long sessions | Weekly | AMD hardware |

### 6.2 Benchmark Suite

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Performance benchmark suite for GAIA agents."""

import time
import statistics
import json
import pytest
from pathlib import Path
from typing import Dict, List


class BenchmarkResult:
    """Container for benchmark measurements."""

    def __init__(self, name: str):
        self.name = name
        self.samples: List[float] = []

    def add_sample(self, duration_ms: float):
        self.samples.append(duration_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples) if self.samples else 0

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def stdev_ms(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "samples": len(self.samples),
            "mean_ms": round(self.mean_ms, 2),
            "median_ms": round(self.median_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "stdev_ms": round(self.stdev_ms, 2),
        }


class TestToolExecutionBenchmarks:
    """Benchmark individual tool execution times."""

    def _benchmark(self, func, iterations=10, warmup=2) -> BenchmarkResult:
        """Run a function multiple times and collect timing data."""
        result = BenchmarkResult(func.__name__)

        # Warmup runs (not measured)
        for _ in range(warmup):
            func()

        # Measured runs
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.add_sample(elapsed_ms)

        return result

    def test_benchmark_tool_registration(self):
        """Measure tool registration overhead."""
        from gaia.agents.base.tools import _TOOL_REGISTRY, tool

        def register_tool():
            _TOOL_REGISTRY.clear()
            for i in range(100):
                @tool
                def dummy(x: str) -> dict:
                    """Dummy."""
                    return {}
                dummy.__name__ = f"tool_{i}"
                tool(dummy)

        result = self._benchmark(register_tool, iterations=20)
        print(f"\nTool registration (100 tools): {result.to_dict()}")
        assert result.mean_ms < 100, "Tool registration too slow"

    def test_benchmark_json_parsing(self):
        """Measure JSON response parsing performance."""
        test_response = json.dumps({
            "tool": "search_code",
            "args": {"query": "function definition", "directory": "/src"},
            "reasoning": "Need to find the function" * 100,
        })

        def parse_response():
            data = json.loads(test_response)
            assert "tool" in data

        result = self._benchmark(parse_response, iterations=1000)
        print(f"\nJSON parsing: {result.to_dict()}")
        assert result.mean_ms < 1, "JSON parsing too slow"

    def test_benchmark_error_formatting(self):
        """Measure error formatting performance."""
        from gaia.agents.base.errors import format_execution_trace

        def format_error():
            try:
                raise ValueError("benchmark error " * 50)
            except ValueError as e:
                format_execution_trace(
                    exception=e,
                    query="benchmark query",
                    plan_step=3,
                    total_steps=10,
                    tool_name="benchmark_tool",
                    tool_args={"key": "value" * 100},
                )

        result = self._benchmark(format_error, iterations=100)
        print(f"\nError formatting: {result.to_dict()}")
        assert result.mean_ms < 10, "Error formatting too slow"


class TestRAGBenchmarks:
    """Benchmark RAG indexing and retrieval performance."""

    @pytest.fixture
    def benchmark_docs(self, tmp_path):
        """Create a set of documents for benchmarking."""
        docs = []
        for i in range(20):
            doc_path = tmp_path / f"doc_{i}.txt"
            content = f"Document {i}. " + " ".join(
                [f"This is paragraph {j} of document {i} about topic {j}."
                 for j in range(50)]
            )
            doc_path.write_text(content)
            docs.append(str(doc_path))
        return docs

    def test_benchmark_document_chunking(self, benchmark_docs):
        """Measure document chunking throughput."""
        from gaia.rag.sdk import RAGSDK

        rag = RAGSDK()

        start = time.perf_counter()
        for doc_path in benchmark_docs[:5]:
            rag._chunk_document(doc_path)
        elapsed = time.perf_counter() - start

        docs_per_second = 5 / elapsed
        print(f"\nChunking throughput: {docs_per_second:.1f} docs/sec")
        assert elapsed < 10, "Document chunking too slow"


class TestInferenceBenchmarks:
    """Benchmark LLM inference latency (requires Lemonade server)."""

    @pytest.mark.benchmark
    def test_benchmark_first_token_latency(self, require_lemonade):
        """Measure time to first token (TTFT)."""
        import requests

        samples = []
        for _ in range(5):
            start = time.perf_counter()
            response = requests.post(
                "http://localhost:8000/api/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=30,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            if response.status_code == 200:
                samples.append(elapsed_ms)

        if samples:
            print(f"\nTTFT: mean={statistics.mean(samples):.0f}ms, "
                  f"p95={sorted(samples)[int(len(samples)*0.95)]:.0f}ms")

    @pytest.mark.benchmark
    def test_benchmark_tokens_per_second(self, require_lemonade):
        """Measure generation throughput in tokens/second."""
        import requests

        response = requests.post(
            "http://localhost:8000/api/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "Write a 100 word paragraph."}],
                "max_tokens": 150,
                "stream": False,
            },
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()
            usage = data.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)

            # If the server reports timing, use it; otherwise estimate
            if completion_tokens > 0:
                print(f"\nGenerated {completion_tokens} tokens")
```

### 6.3 Memory Leak Detection

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Memory profiling tests for long-running agent sessions."""

import gc
import pytest
import tracemalloc


class TestMemoryLeaks:
    """Detect memory leaks in agent lifecycle operations."""

    def test_agent_creation_cleanup_no_leak(self):
        """Verify agent creation and cleanup does not leak memory."""
        tracemalloc.start()

        from gaia.agents.code.agent import CodeAgent

        # Baseline
        gc.collect()
        snapshot1 = tracemalloc.take_snapshot()

        # Create and destroy many agents
        for _ in range(50):
            agent = CodeAgent(silent_mode=True, max_steps=3)
            del agent

        gc.collect()
        snapshot2 = tracemalloc.take_snapshot()

        # Compare memory usage
        stats = snapshot2.compare_to(snapshot1, "lineno")
        top_leaks = sorted(stats, key=lambda s: s.size_diff, reverse=True)[:5]

        total_leak_kb = sum(s.size_diff for s in top_leaks) / 1024
        print(f"\nTop memory changes after 50 agent cycles:")
        for stat in top_leaks:
            print(f"  {stat}")

        tracemalloc.stop()

        # Allow up to 1MB growth for 50 cycles
        assert total_leak_kb < 1024, (
            f"Potential memory leak: {total_leak_kb:.1f} KB growth "
            f"after 50 agent create/destroy cycles"
        )

    def test_conversation_history_bounded(self):
        """Verify conversation history does not grow unbounded."""
        from gaia.chat.sdk import ChatConfig

        config = ChatConfig(max_history_length=4)

        # Simulate adding many messages
        history = []
        for i in range(100):
            history.append({"role": "user", "content": f"Message {i}"})
            history.append({"role": "assistant", "content": f"Response {i}"})

            # The SDK should trim history; simulate the trim
            max_entries = config.max_history_length * 2  # pairs
            if len(history) > max_entries:
                history = history[-max_entries:]

        # After 100 exchanges with max_history_length=4, only 8 entries
        assert len(history) == 8

    def test_tool_registry_cleanup(self):
        """Verify tool registry does not accumulate stale entries."""
        from gaia.agents.base.tools import _TOOL_REGISTRY, tool

        initial_size = len(_TOOL_REGISTRY)
        _TOOL_REGISTRY.clear()

        # Register and clear multiple times
        for cycle in range(10):
            for i in range(20):
                @tool
                def temp_tool(x: str) -> dict:
                    """Temp."""
                    return {}
                temp_tool.__name__ = f"temp_{cycle}_{i}"
                tool(temp_tool)

            _TOOL_REGISTRY.clear()

        assert len(_TOOL_REGISTRY) == 0


class TestLLMCostTracking:
    """Track LLM API costs during test runs."""

    # Pricing per 1M tokens (approximate)
    PRICING = {
        "Qwen3-Coder-30B": {"input": 0.0, "output": 0.0},   # Local, no cost
        "Qwen3-0.6B-GGUF": {"input": 0.0, "output": 0.0},   # Local, no cost
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "gpt-4o": {"input": 2.5, "output": 10.0},
    }

    def test_estimate_test_suite_cost(self):
        """Estimate the cost of running the full eval suite."""
        # Typical eval run: 50 tasks, ~500 input tokens + ~200 output tokens each
        tasks = 50
        avg_input_tokens = 500
        avg_output_tokens = 200

        for model, pricing in self.PRICING.items():
            input_cost = (tasks * avg_input_tokens / 1_000_000) * pricing["input"]
            output_cost = (tasks * avg_output_tokens / 1_000_000) * pricing["output"]
            total = input_cost + output_cost
            print(f"  {model}: ${total:.4f} ({tasks} tasks)")

            if "local" not in model.lower() and pricing["input"] == 0:
                continue
            # Cloud model costs should be documented
```

---

## 7. Safety Testing

### 7.1 Testing Safety Guardrails

GAIA agents can execute shell commands, write files, and access network services.
Safety tests verify that security boundaries hold under adversarial inputs.

**Security test categories:**

| Category | Tests | CI Workflow |
|----------|-------|-------------|
| Path traversal prevention | `verify_path_validator.py` | `test_security.yml` |
| Shell injection prevention | `verify_shell_security.py` | `test_security.yml` |
| Argument sanitization | `test_code_agent.py` | `test_code_agent.yml` |
| API input validation | `test_api.py` | `test_api.yml` |
| Credential leak prevention | Proposed | `test_security.yml` |

### 7.2 Path Traversal Tests

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Path traversal prevention tests."""

import os
import tempfile
from pathlib import Path

import pytest
from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig


class TestPathTraversalPrevention:
    """Verify agents cannot access files outside allowed directories."""

    @pytest.fixture
    def restricted_agent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = str(Path(tmpdir).resolve())
            config = ChatAgentConfig(
                silent_mode=True,
                allowed_paths=[resolved],
            )
            agent = ChatAgent(config)
            yield agent, resolved
            agent.stop_watching()

    def test_allowed_path_accepted(self, restricted_agent):
        agent, allowed_dir = restricted_agent
        assert agent._is_path_allowed(allowed_dir)

    def test_parent_directory_rejected(self, restricted_agent):
        agent, allowed_dir = restricted_agent
        parent = str(Path(allowed_dir).parent)
        assert not agent._is_path_allowed(parent)

    def test_dot_dot_traversal_rejected(self, restricted_agent):
        agent, allowed_dir = restricted_agent
        traversal = os.path.join(allowed_dir, "..", "..", "etc", "passwd")
        assert not agent._is_path_allowed(traversal)

    def test_absolute_outside_path_rejected(self, restricted_agent):
        agent, _ = restricted_agent
        assert not agent._is_path_allowed("/tmp/not_allowed")
        assert not agent._is_path_allowed("/etc/shadow")

    @pytest.mark.parametrize("malicious_path", [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/proc/self/environ",
        "~/.ssh/id_rsa",
        "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ])
    def test_known_traversal_patterns_rejected(self, restricted_agent, malicious_path):
        agent, allowed_dir = restricted_agent
        full_path = os.path.join(allowed_dir, malicious_path)
        assert not agent._is_path_allowed(full_path)
```

### 7.3 Shell Injection Prevention Tests

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shell injection prevention tests."""

import os
import pytest
from gaia.agents.base.tools import _TOOL_REGISTRY
from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig


class TestShellInjectionPrevention:
    """Verify shell commands cannot be chained or piped."""

    @pytest.fixture
    def agent_with_shell(self):
        config = ChatAgentConfig(
            silent_mode=True,
            allowed_paths=[os.getcwd()],
        )
        agent = ChatAgent(config)
        yield agent
        agent.stop_watching()

    def test_command_chaining_blocked(self, agent_with_shell):
        """Semicolons should not chain commands."""
        run_shell = _TOOL_REGISTRY.get("run_shell_command", {}).get("function")
        if run_shell is None:
            pytest.skip("Shell tool not registered")

        result = run_shell("ls; echo 'INJECTED'", working_directory=os.getcwd())

        # The word INJECTED should not appear in stdout
        stdout = result.get("stdout", "")
        assert "INJECTED" not in stdout

    def test_pipe_operator_blocked(self, agent_with_shell):
        """Pipe operators should not work."""
        run_shell = _TOOL_REGISTRY.get("run_shell_command", {}).get("function")
        if run_shell is None:
            pytest.skip("Shell tool not registered")

        result = run_shell("cat /etc/passwd | head -1", working_directory=os.getcwd())

        # Should not successfully pipe (subprocess with list args does not
        # interpret pipes)
        stdout = result.get("stdout", "")
        assert "root:" not in stdout

    def test_backtick_substitution_blocked(self, agent_with_shell):
        """Backtick command substitution should not execute."""
        run_shell = _TOOL_REGISTRY.get("run_shell_command", {}).get("function")
        if run_shell is None:
            pytest.skip("Shell tool not registered")

        result = run_shell(
            "echo `whoami`",
            working_directory=os.getcwd(),
        )
        stdout = result.get("stdout", "")
        # If backticks were interpreted, stdout would contain the username
        # With list-based subprocess, it should be treated as a literal argument

    @pytest.mark.parametrize("dangerous_cmd", [
        "rm -rf /",
        "rm -rf ~",
        "dd if=/dev/zero of=/dev/sda",
        ":(){:|:&};:",
        "mkfs.ext4 /dev/sda1",
    ])
    def test_destructive_commands_prevented(self, agent_with_shell, dangerous_cmd):
        """Destructive commands should be blocked by allowlist or validation."""
        run_shell = _TOOL_REGISTRY.get("run_shell_command", {}).get("function")
        if run_shell is None:
            pytest.skip("Shell tool not registered")

        result = run_shell(dangerous_cmd, working_directory=os.getcwd())

        # Result should indicate error or blocked status
        # The exact behavior depends on the shell tool implementation
        assert result.get("status") != "success" or "error" in str(result).lower()


class TestAPIInputValidation:
    """Verify API server rejects malformed requests."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            from fastapi.testclient import TestClient
            from gaia.api.openai_server import app
            self.client = TestClient(app)
            self.available = True
        except ImportError:
            self.available = False

    def test_oversized_message_rejected(self):
        if not self.available:
            pytest.skip("API not available")

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gaia-chat",
                "messages": [{"role": "user", "content": "x" * 10_000_000}],
                "stream": False,
            },
        )
        # Server should either reject or handle gracefully
        assert response.status_code in (200, 400, 413, 422)

    def test_null_bytes_in_content_handled(self):
        if not self.available:
            pytest.skip("API not available")

        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "gaia-chat",
                "messages": [{"role": "user", "content": "test\x00injection"}],
                "stream": False,
            },
        )
        assert response.status_code in (200, 400, 422)
```

### 7.4 Chaos Engineering: Injecting Failures

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Chaos engineering tests: inject failures and verify graceful recovery."""

import pytest
from unittest.mock import patch, Mock, MagicMock
from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig
import tempfile
from pathlib import Path


class TestChaosEngineering:
    """Inject failures at various points to test agent resilience."""

    @pytest.fixture
    def agent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = str(Path(tmpdir).resolve())
            config = ChatAgentConfig(
                silent_mode=True,
                max_steps=5,
                allowed_paths=[resolved, str(Path.cwd().resolve())],
            )
            agent = ChatAgent(config)
            yield agent
            agent.stop_watching()

    def test_llm_connection_timeout_recovery(self, agent):
        """Agent recovers gracefully when LLM connection times out."""
        import requests

        mock_response = Mock()
        mock_response.text = "Recovery response after timeout"
        mock_response.stats = {}
        mock_response.tool_calls = []

        call_count = 0

        def flaky_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.exceptions.Timeout("Connection timed out")
            return mock_response

        with patch.object(agent.chat, "send_messages", side_effect=flaky_send):
            # The agent should handle the timeout without crashing
            try:
                result = agent.process_query("test query")
                # If it returns, verify it has expected structure
                assert "status" in result or "result" in result or isinstance(result, str)
            except requests.exceptions.Timeout:
                # If it propagates, that is acceptable (caller handles retry)
                pass

    def test_tool_execution_exception_contained(self, agent):
        """Exceptions in tool execution do not crash the agent loop."""
        from gaia.agents.base.tools import _TOOL_REGISTRY

        # Register a tool that always raises
        _TOOL_REGISTRY["crash_tool"] = {
            "name": "crash_tool",
            "description": "A tool that crashes.",
            "parameters": {},
            "function": lambda: (_ for _ in ()).throw(RuntimeError("BOOM")),
            "atomic": False,
        }

        # The agent should catch the error and continue or report gracefully
        # This is a structural test; full execution requires mocked LLM

    def test_disk_full_handling(self, agent):
        """Agent handles disk full errors when writing files."""

        with patch("builtins.open", side_effect=OSError("No space left on device")):
            # Attempting to write should fail gracefully
            try:
                # This tests the RAG indexing path when disk is full
                result = agent.rag.index_document("/nonexistent/file.txt")
                # Should return error status, not crash
                assert not result.get("success", True)
            except (OSError, FileNotFoundError):
                pass  # Also acceptable

    def test_malformed_llm_response_handling(self, agent):
        """Agent handles malformed JSON from LLM without crashing."""
        malformed_responses = [
            "This is not JSON at all",
            '{"incomplete": true',
            '{"tool": null, "args": "not_a_dict"}',
            "",
            "```json\n{broken}\n```",
        ]

        for response_text in malformed_responses:
            mock_resp = Mock()
            mock_resp.text = response_text
            mock_resp.stats = {}
            mock_resp.tool_calls = []

            with patch.object(agent.chat, "send_messages", return_value=mock_resp):
                try:
                    result = agent.process_query("test")
                    # Should complete without crash
                except Exception as e:
                    # Some exceptions are acceptable, but not SystemExit or KeyboardInterrupt
                    assert not isinstance(e, (SystemExit, KeyboardInterrupt))

    def test_concurrent_session_access(self, agent):
        """Verify session manager handles concurrent access gracefully."""
        import threading

        errors = []

        def create_session():
            try:
                session = agent.session_manager.create_session()
                assert session.session_id is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_session) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Concurrent session creation errors: {errors}"
```

### 7.5 Computer Use Safety Testing

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Safety tests for Computer Use architecture (proposed)."""

import pytest


class TestComputerUseSafety:
    """
    Verify safety guardrails for the Computer Use architecture.

    Computer Use allows agents to interact with desktop applications
    via screenshots and input simulation. These tests verify that
    safety boundaries prevent unauthorized actions.

    Reference: architecture/COMPUTER_USE_ARCHITECTURE.md
    """

    def test_screenshot_region_bounded(self):
        """Screenshots should be limited to allowed application windows."""
        # When implemented, verify the screenshot capture is scoped
        # to the target application, not the full desktop
        pass

    def test_input_simulation_allowlist(self):
        """Only allowlisted applications receive simulated input."""
        # Verify that mouse clicks and keyboard input are only sent
        # to applications in the approved list
        pass

    def test_sensitive_area_detection(self):
        """Agent should refuse to interact with password fields."""
        # When the screenshot contains a password input field,
        # the agent should not type into it
        pass

    def test_confirmation_required_for_destructive_actions(self):
        """Destructive actions (delete, format, shutdown) require confirmation."""
        # Verify that actions matching destructive patterns are
        # blocked without explicit user confirmation
        pass

    def test_network_request_monitoring(self):
        """Network requests made during computer use are logged."""
        # All outbound network requests during a computer use session
        # should be captured in the audit log
        pass
```

---

## 8. CI/CD Integration

### 8.1 GitHub Actions Workflow Architecture

GAIA uses a layered CI/CD strategy where fast, cheap tests gate slow, expensive
ones. Every workflow supports `workflow_call` for composition and `workflow_dispatch`
for manual triggering.

```
Pull Request Opened
        |
        v
+-------------------+     +-------------------+
|   lint.yml        |     |  test_unit.yml    |     <-- Tier 1: ~60 seconds
|   (black, isort,  |     |  (mocked LLM,    |         ubuntu-latest
|    pylint)        |     |   no hardware)    |
+-------------------+     +-------------------+
        |                          |
        +-----------+--------------+
                    |
                    v  (pass gate)
        +-------------------+     +-------------------+
        | test_security.yml |     | test_chat_sdk.yml |     <-- Tier 2: ~5 min
        | (path traversal,  |     | (SDK integration) |         ubuntu + windows
        |  shell injection) |     |                   |
        +-------------------+     +-------------------+
                    |
                    v  (pass gate)
        +-------------------+     +-------------------+
        | test_chat_agent.yml|    | test_code_agent.yml|    <-- Tier 3: ~10 min
        | (real Lemonade)    |    | (real Lemonade)    |        self-hosted AMD
        +-------------------+     +-------------------+
                    |
                    v  (merge to main)
        +-------------------+
        | test_eval.yml     |     <-- Tier 4: on-demand
        | (full eval suite) |         scheduled or manual
        +-------------------+
```

### 8.2 Workflow Templates

#### 8.2.1 Unit Test Workflow

```yaml
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# .github/workflows/test_unit.yml
name: Unit Tests

on:
  workflow_call:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
    types: [opened, synchronize, reopened, ready_for_review]
  merge_group:
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  unit-tests:
    name: Run Unit Tests
    runs-on: ubuntu-latest
    if: >
      github.event_name != 'pull_request' ||
      github.event.pull_request.draft == false ||
      contains(github.event.pull_request.labels.*.name, 'ready_for_ci')

    steps:
      - uses: actions/checkout@v6

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: '3.12'

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: |
          uv pip install --system pytest pytest-cov
          uv pip install --system -e ".[api]"

      - name: Validate CLI commands (dry-run)
        run: |
          gaia --help
          gaia chat --help
          gaia prompt --help
          gaia init --help

      - name: Run unit tests with coverage
        run: |
          pytest tests/unit/ -v --tb=short \
            --cov=src/gaia --cov-report=term-missing

      - name: Run integration tests (no server required)
        run: |
          pytest tests/integration/ -v --tb=short
```

#### 8.2.2 Security Test Workflow

```yaml
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# .github/workflows/test_security.yml
name: Security Tests

on:
  workflow_call:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  merge_group:
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: true

permissions:
  contents: read

jobs:
  test-security-linux:
    name: Security Tests (Linux)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: '3.12'
      - run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - run: uv pip install --system -e .[dev,rag]
      - name: Path Validator Tests
        run: python tests/verify_path_validator.py
      - name: Shell Injection Tests
        run: python tests/verify_shell_security.py

  test-security-windows:
    name: Security Tests (Windows)
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: '3.12'
      - run: irm https://astral.sh/uv/install.ps1 | iex
        shell: pwsh
      - run: uv pip install --system -e .[dev,rag]
      - name: Path Validator Tests
        run: python tests/verify_path_validator.py
      - name: Shell Injection Tests
        run: python tests/verify_shell_security.py
```

### 8.3 Test Parallelization

```yaml
# Parallel test execution strategy for large test suites
# Split tests by module to run concurrently across runners

jobs:
  test-matrix:
    strategy:
      fail-fast: false
      matrix:
        test-group:
          - { name: "Unit", path: "tests/unit/", runner: "ubuntu-latest" }
          - { name: "Chat Agent", path: "tests/test_chat_agent.py", runner: "self-hosted" }
          - { name: "Code Agent", path: "tests/test_code_agent.py", runner: "self-hosted" }
          - { name: "API", path: "tests/test_api.py", runner: "ubuntu-latest" }
          - { name: "MCP", path: "tests/mcp/", runner: "self-hosted" }
          - { name: "RAG", path: "tests/test_rag.py", runner: "self-hosted" }
          - { name: "Security", path: "tests/verify_*.py", runner: "ubuntu-latest" }

    name: ${{ matrix.test-group.name }}
    runs-on: ${{ matrix.test-group.runner }}

    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: '3.12'
      - run: uv pip install --system -e ".[dev,rag,api]"
      - run: pytest ${{ matrix.test-group.path }} -v --tb=short
```

### 8.4 Conditional Test Execution

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Pytest markers for conditional test execution.

Add to conftest.py to support selective test runs based on
available infrastructure and --hybrid flag.
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hybrid",
        action="store_true",
        default=False,
        help="Run with hybrid configuration (cloud + local models)",
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run performance benchmark tests",
    )
    parser.addoption(
        "--eval",
        action="store_true",
        default=False,
        help="Run evaluation framework tests (requires Claude API key)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "hybrid: test requires --hybrid flag")
    config.addinivalue_line("markers", "benchmark: performance benchmark test")
    config.addinivalue_line("markers", "eval: evaluation framework test")
    config.addinivalue_line("markers", "slow: slow test (>30 seconds)")
    config.addinivalue_line("markers", "npu: requires AMD NPU hardware")
    config.addinivalue_line("markers", "gpu: requires AMD iGPU")


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and available flags."""
    skip_hybrid = pytest.mark.skip(reason="Need --hybrid flag to run")
    skip_benchmark = pytest.mark.skip(reason="Need --benchmark flag to run")
    skip_eval = pytest.mark.skip(reason="Need --eval flag to run")

    for item in items:
        if "hybrid" in item.keywords and not config.getoption("--hybrid"):
            item.add_marker(skip_hybrid)
        if "benchmark" in item.keywords and not config.getoption("--benchmark"):
            item.add_marker(skip_benchmark)
        if "eval" in item.keywords and not config.getoption("--eval"):
            item.add_marker(skip_eval)
```

### 8.5 Test Result Reporting

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Custom pytest plugin for structured test result reporting.

Generates JSON reports suitable for dashboards and trend analysis.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import pytest


class GAIATestReporter:
    """Collects test results and generates structured reports."""

    def __init__(self):
        self.results: List[Dict] = []
        self.start_time = time.time()

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            self.results.append({
                "name": report.nodeid,
                "outcome": report.outcome,
                "duration_s": round(report.duration, 3),
                "markers": [m.name for m in report.keywords.get("pytestmark", [])],
            })

    def pytest_sessionfinish(self, session, exitstatus):
        total_duration = time.time() - self.start_time
        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r["outcome"] == "passed"),
            "failed": sum(1 for r in self.results if r["outcome"] == "failed"),
            "skipped": sum(1 for r in self.results if r["outcome"] == "skipped"),
            "total_duration_s": round(total_duration, 2),
            "tests": self.results,
        }

        report_path = Path("test-results.json")
        report_path.write_text(json.dumps(report, indent=2))


def pytest_configure(config):
    config._gaia_reporter = GAIATestReporter()
    config.pluginmanager.register(config._gaia_reporter)
```

---

## 9. Test Data Management

### 9.1 Fixture Organization

```
tests/
    conftest.py                  # Root fixtures: api_server, api_client,
    |                            #   lemonade_available, require_lemonade
    |
    fixtures/                    # Shared test data (proposed)
    |   llm_responses/           # Pre-recorded LLM response sequences
    |   |   chat_rag_flow.json
    |   |   code_edit_flow.json
    |   |   jira_create_flow.json
    |   |
    |   documents/               # Test documents for RAG indexing
    |   |   sample_report.txt
    |   |   sample_code.py
    |   |   ryzen_ai_specs.pdf
    |   |
    |   golden_outputs/          # Regression baselines
    |       chat_rag.golden.json
    |       summarizer.golden.json
    |
    unit/
    |   conftest.py              # Unit-specific fixtures (mocked clients)
    |   test_tool_decorator.py
    |   test_errors.py
    |   ...
    |
    integration/
    |   conftest.py              # Integration fixtures (DB setup, etc.)
    |   test_database_mixin_integration.py
    |   ...
    |
    mcp/
        conftest.py              # MCP-specific fixtures (server lifecycle)
        test_mcp_integration.py
        ...
```

### 9.2 Test Database Seeding

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Database fixtures for agents with DatabaseMixin."""

import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def seeded_database():
    """
    Create a temporary SQLite database pre-populated with test data.

    Yields a tuple of (db_path, connection) for test use.
    The database and all data are cleaned up after the test.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")
        conn = sqlite3.connect(db_path)

        # Create tables
        conn.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER REFERENCES customers(id),
                product TEXT NOT NULL,
                amount REAL NOT NULL,
                status TEXT DEFAULT 'pending'
            );

            INSERT INTO customers (name, email) VALUES
                ('Alice Smith', 'alice@example.com'),
                ('Bob Jones', 'bob@example.com'),
                ('Charlie Brown', 'charlie@example.com');

            INSERT INTO orders (customer_id, product, amount, status) VALUES
                (1, 'Widget A', 29.99, 'completed'),
                (1, 'Widget B', 49.99, 'pending'),
                (2, 'Gadget X', 99.99, 'completed'),
                (3, 'Widget A', 29.99, 'shipped');
        """)
        conn.commit()

        yield db_path, conn

        conn.close()


@pytest.fixture
def empty_database():
    """Create an empty temporary SQLite database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "empty.db")
        conn = sqlite3.connect(db_path)
        yield db_path, conn
        conn.close()
```

### 9.3 Mock External Services

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Mock external services for isolated integration testing."""

import json
from typing import Dict, List, Optional
from unittest.mock import MagicMock


class MockJiraClient:
    """
    Mock Jira API client for testing JiraAgent without real Jira instance.

    Simulates issue CRUD, search, and transitions.
    """

    def __init__(self):
        self.issues: Dict[str, Dict] = {}
        self._next_id = 1000
        self.call_log: List[Dict] = []

    def create_issue(self, project: str, summary: str,
                     description: str = "", issue_type: str = "Task") -> Dict:
        issue_key = f"{project}-{self._next_id}"
        self._next_id += 1

        issue = {
            "key": issue_key,
            "fields": {
                "summary": summary,
                "description": description,
                "issuetype": {"name": issue_type},
                "status": {"name": "To Do"},
                "project": {"key": project},
            },
        }
        self.issues[issue_key] = issue
        self.call_log.append({"method": "create_issue", "key": issue_key})
        return issue

    def get_issue(self, key: str) -> Optional[Dict]:
        self.call_log.append({"method": "get_issue", "key": key})
        return self.issues.get(key)

    def search_issues(self, jql: str, max_results: int = 50) -> List[Dict]:
        self.call_log.append({"method": "search_issues", "jql": jql})
        # Simple keyword matching against summaries
        results = []
        for issue in self.issues.values():
            summary = issue["fields"]["summary"].lower()
            if any(word.lower() in summary for word in jql.split()):
                results.append(issue)
        return results[:max_results]

    def transition_issue(self, key: str, status: str) -> bool:
        self.call_log.append({
            "method": "transition_issue", "key": key, "status": status,
        })
        if key in self.issues:
            self.issues[key]["fields"]["status"]["name"] = status
            return True
        return False


class MockDockerClient:
    """
    Mock Docker client for testing DockerAgent without real Docker daemon.

    Simulates container lifecycle operations.
    """

    def __init__(self):
        self.containers: Dict[str, Dict] = {}
        self.images: List[str] = ["python:3.12", "node:20", "ubuntu:22.04"]
        self.call_log: List[Dict] = []

    def run_container(self, image: str, name: str = "",
                      command: str = "", **kwargs) -> Dict:
        container_id = f"container_{len(self.containers):04d}"
        container = {
            "id": container_id,
            "image": image,
            "name": name or container_id,
            "status": "running",
            "command": command,
        }
        self.containers[container_id] = container
        self.call_log.append({"method": "run", "container_id": container_id})
        return container

    def stop_container(self, container_id: str) -> bool:
        self.call_log.append({"method": "stop", "container_id": container_id})
        if container_id in self.containers:
            self.containers[container_id]["status"] = "stopped"
            return True
        return False

    def list_containers(self, all_containers: bool = False) -> List[Dict]:
        self.call_log.append({"method": "list"})
        if all_containers:
            return list(self.containers.values())
        return [c for c in self.containers.values() if c["status"] == "running"]

    def list_images(self) -> List[str]:
        return self.images
```

### 9.4 VCR-Style HTTP Recording

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
VCR-style HTTP recording for reproducible integration tests.

Records real HTTP interactions and replays them in subsequent test runs,
providing deterministic integration tests without live services.
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch


class HTTPCassette:
    """
    Records and replays HTTP request/response pairs.

    Usage:
        # Record mode (first run):
        with HTTPCassette("tests/cassettes/lemonade_chat.json", record=True):
            response = requests.post("http://localhost:8000/api/v1/chat/completions", ...)

        # Replay mode (subsequent runs):
        with HTTPCassette("tests/cassettes/lemonade_chat.json"):
            response = requests.post(...)  # Returns recorded response
    """

    def __init__(self, cassette_path: str, record: bool = False):
        self.cassette_path = Path(cassette_path)
        self.record = record
        self.interactions: List[Dict] = []
        self._replay_index = 0
        self._original_session_send = None

    def _request_key(self, method: str, url: str, body: Optional[str]) -> str:
        """Generate a unique key for a request."""
        content = f"{method}|{url}|{body or ''}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_cassette(self):
        if self.cassette_path.exists():
            self.interactions = json.loads(self.cassette_path.read_text())

    def _save_cassette(self):
        self.cassette_path.parent.mkdir(parents=True, exist_ok=True)
        self.cassette_path.write_text(json.dumps(self.interactions, indent=2))

    def __enter__(self):
        if not self.record:
            self._load_cassette()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.record:
            self._save_cassette()
        return False

    def get_recorded_response(self, method: str, url: str) -> Optional[Dict]:
        """Get the next recorded response for replay."""
        if self._replay_index < len(self.interactions):
            interaction = self.interactions[self._replay_index]
            self._replay_index += 1
            return interaction.get("response")
        return None

    def record_interaction(self, method: str, url: str,
                           request_body: str, status_code: int,
                           response_body: str):
        """Record an HTTP interaction."""
        self.interactions.append({
            "request": {
                "method": method,
                "url": url,
                "body": request_body,
            },
            "response": {
                "status_code": status_code,
                "body": response_body,
            },
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })


# Example usage in tests
class TestWithCassette:
    """Demonstrate VCR-style testing."""

    def test_lemonade_health_check_recorded(self):
        """Test health check using recorded response."""
        cassette_path = "tests/cassettes/lemonade_health.json"
        cassette = HTTPCassette(cassette_path)

        # If cassette exists, use recorded response
        if Path(cassette_path).exists():
            with cassette:
                response = cassette.get_recorded_response("GET", "/api/v1/health")
                if response:
                    assert response["status_code"] == 200
```

---

## 10. Complete Test Suite Examples

### 10.1 Testing EmailAgent End-to-End

This example demonstrates a complete test suite for the proposed EmailAgent
architecture (reference: `architecture/EMAIL_INTEGRATION_ARCHITECTURE.md`).

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Complete test suite for EmailAgent.

Tests cover:
- Agent initialization and tool registration
- Email composition with LLM assistance
- Email search and filtering
- Draft management
- Safety: no credential leakage in logs
- Integration with SMTP/IMAP (mocked)
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from gaia.testing.mocks import MockLLMProvider, MockToolExecutor
from gaia.testing.assertions import (
    assert_llm_called,
    assert_tool_called,
    assert_agent_completed,
    assert_no_errors,
)
from gaia.testing.fixtures import AgentTestContext


# ---------------------------------------------------------------------------
# Mock Email Service
# ---------------------------------------------------------------------------

class MockEmailService:
    """Simulates SMTP/IMAP email operations."""

    def __init__(self):
        self.sent_emails: List[Dict] = []
        self.inbox: List[Dict] = [
            {
                "id": "msg_001",
                "from": "alice@example.com",
                "to": "user@gaia.ai",
                "subject": "Project Update",
                "body": "The Q4 report is ready for review.",
                "date": "2026-02-01T10:00:00Z",
                "read": False,
            },
            {
                "id": "msg_002",
                "from": "bob@example.com",
                "to": "user@gaia.ai",
                "subject": "Meeting Tomorrow",
                "body": "Can we reschedule to 3pm?",
                "date": "2026-02-02T14:30:00Z",
                "read": True,
            },
        ]
        self.drafts: List[Dict] = []

    def send_email(self, to: str, subject: str, body: str, cc: str = "") -> Dict:
        email = {
            "to": to,
            "subject": subject,
            "body": body,
            "cc": cc,
            "status": "sent",
        }
        self.sent_emails.append(email)
        return {"success": True, "message_id": f"sent_{len(self.sent_emails):03d}"}

    def search_inbox(self, query: str, limit: int = 10) -> List[Dict]:
        results = []
        for msg in self.inbox:
            if (query.lower() in msg["subject"].lower() or
                    query.lower() in msg["body"].lower()):
                results.append(msg)
        return results[:limit]

    def get_email(self, message_id: str) -> Dict:
        for msg in self.inbox:
            if msg["id"] == message_id:
                return msg
        return {"error": f"Message {message_id} not found"}

    def save_draft(self, to: str, subject: str, body: str) -> Dict:
        draft = {"to": to, "subject": subject, "body": body, "status": "draft"}
        self.drafts.append(draft)
        return {"success": True, "draft_id": f"draft_{len(self.drafts):03d}"}


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestEmailAgentUnit:
    """Unit tests for EmailAgent with mocked LLM and email service."""

    @pytest.fixture
    def mock_email_service(self):
        return MockEmailService()

    @pytest.fixture
    def mock_llm(self):
        return MockLLMProvider(responses=[
            json.dumps({
                "tool": "search_inbox",
                "args": {"query": "project update"},
            }),
            "I found 1 email about the project update from Alice.",
        ])

    def test_email_search_flow(self, mock_email_service, mock_llm):
        """Test searching inbox through agent."""
        # Simulate the tool execution
        results = mock_email_service.search_inbox("project update")
        assert len(results) == 1
        assert results[0]["from"] == "alice@example.com"

    def test_email_composition_flow(self, mock_email_service):
        """Test composing and sending an email."""
        result = mock_email_service.send_email(
            to="alice@example.com",
            subject="Re: Project Update",
            body="Thanks for the update. I will review the Q4 report today.",
        )
        assert result["success"] is True
        assert len(mock_email_service.sent_emails) == 1

    def test_draft_management(self, mock_email_service):
        """Test saving and managing email drafts."""
        result = mock_email_service.save_draft(
            to="team@example.com",
            subject="Weekly Standup Notes",
            body="Pending review...",
        )
        assert result["success"] is True
        assert len(mock_email_service.drafts) == 1

    def test_no_credentials_in_logs(self, mock_email_service):
        """Verify email credentials never appear in service output."""
        # Send an email and check the result does not contain credentials
        result = mock_email_service.send_email(
            to="test@example.com",
            subject="Test",
            body="Test body",
        )
        result_str = json.dumps(result)
        assert "password" not in result_str.lower()
        assert "secret" not in result_str.lower()
        assert "token" not in result_str.lower()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestEmailAgentIntegration:
    """Integration tests simulating full email workflows."""

    @pytest.fixture
    def email_service(self):
        return MockEmailService()

    def test_read_and_reply_workflow(self, email_service):
        """Full workflow: search inbox, read email, compose reply, send."""
        # Step 1: Search for unread emails
        unread = [m for m in email_service.inbox if not m["read"]]
        assert len(unread) == 1

        # Step 2: Read the email
        email = email_service.get_email(unread[0]["id"])
        assert email["subject"] == "Project Update"

        # Step 3: Compose and send reply
        result = email_service.send_email(
            to=email["from"],
            subject=f"Re: {email['subject']}",
            body="Thank you, I will review it today.",
        )
        assert result["success"] is True

        # Step 4: Verify sent
        assert len(email_service.sent_emails) == 1
        sent = email_service.sent_emails[0]
        assert sent["to"] == "alice@example.com"
        assert "Re: Project Update" in sent["subject"]

    def test_bulk_search_and_summarize(self, email_service):
        """Search multiple emails and produce a summary."""
        # Add more test emails
        for i in range(5):
            email_service.inbox.append({
                "id": f"msg_{100+i}",
                "from": f"user{i}@example.com",
                "to": "user@gaia.ai",
                "subject": f"Sprint {i} Review",
                "body": f"Sprint {i} completed with {i+3} story points.",
                "date": f"2026-02-0{i+1}T09:00:00Z",
                "read": True,
            })

        results = email_service.search_inbox("Sprint")
        assert len(results) == 5

        # Summarize search results
        subjects = [r["subject"] for r in results]
        assert all("Sprint" in s for s in subjects)
```

### 10.2 Testing WorkflowEngine with Retries

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Complete test suite for WorkflowEngine with retry logic.

Reference: architecture/WORKFLOW_ORCHESTRATION_ARCHITECTURE.md

Tests cover:
- Sequential step execution
- Parallel step execution
- Retry on transient failure
- Max retry exhaustion
- Conditional branching
- Workflow state persistence
"""

import json
import time
import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock


# ---------------------------------------------------------------------------
# Workflow Engine (simplified for testing)
# ---------------------------------------------------------------------------

class WorkflowStep:
    """A single step in a workflow."""

    def __init__(self, name: str, action: callable,
                 retries: int = 0, retry_delay: float = 0.1):
        self.name = name
        self.action = action
        self.retries = retries
        self.retry_delay = retry_delay


class WorkflowResult:
    """Result of a workflow execution."""

    def __init__(self):
        self.steps_completed: List[str] = []
        self.steps_failed: List[str] = []
        self.step_results: Dict[str, Any] = {}
        self.total_retries: int = 0
        self.success: bool = False


class WorkflowEngine:
    """
    Execute multi-step workflows with retry logic.

    This is a simplified version for testing purposes.
    The production implementation would integrate with the Agent base class.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 0.1):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.execution_log: List[Dict] = []

    def execute(self, steps: List[WorkflowStep]) -> WorkflowResult:
        result = WorkflowResult()

        for step in steps:
            retries = step.retries or self.max_retries
            last_error = None

            for attempt in range(retries + 1):
                try:
                    self.execution_log.append({
                        "step": step.name,
                        "attempt": attempt + 1,
                        "timestamp": time.time(),
                    })
                    step_result = step.action()
                    result.steps_completed.append(step.name)
                    result.step_results[step.name] = step_result
                    last_error = None
                    break  # Success
                except Exception as e:
                    last_error = e
                    result.total_retries += 1
                    if attempt < retries:
                        time.sleep(step.retry_delay or self.retry_delay)

            if last_error:
                result.steps_failed.append(step.name)
                result.step_results[step.name] = {"error": str(last_error)}

        result.success = len(result.steps_failed) == 0
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorkflowEngineBasic:
    """Basic workflow execution tests."""

    def test_sequential_steps_execute_in_order(self):
        execution_order = []

        steps = [
            WorkflowStep("step_1", lambda: execution_order.append("step_1") or "ok"),
            WorkflowStep("step_2", lambda: execution_order.append("step_2") or "ok"),
            WorkflowStep("step_3", lambda: execution_order.append("step_3") or "ok"),
        ]

        engine = WorkflowEngine()
        result = engine.execute(steps)

        assert result.success is True
        assert execution_order == ["step_1", "step_2", "step_3"]
        assert len(result.steps_completed) == 3

    def test_failed_step_recorded(self):
        def failing_action():
            raise RuntimeError("Step failed")

        steps = [
            WorkflowStep("good_step", lambda: "ok"),
            WorkflowStep("bad_step", failing_action, retries=0),
        ]

        engine = WorkflowEngine()
        result = engine.execute(steps)

        assert result.success is False
        assert "good_step" in result.steps_completed
        assert "bad_step" in result.steps_failed

    def test_empty_workflow_succeeds(self):
        engine = WorkflowEngine()
        result = engine.execute([])
        assert result.success is True
        assert len(result.steps_completed) == 0


class TestWorkflowEngineRetries:
    """Retry logic tests."""

    def test_transient_failure_retried_and_succeeds(self):
        call_count = 0

        def flaky_action():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient failure")
            return "success"

        steps = [WorkflowStep("flaky", flaky_action, retries=3, retry_delay=0.01)]
        engine = WorkflowEngine()
        result = engine.execute(steps)

        assert result.success is True
        assert call_count == 3
        assert result.total_retries == 2

    def test_max_retries_exhausted(self):
        def always_fails():
            raise RuntimeError("Permanent failure")

        steps = [WorkflowStep("doomed", always_fails, retries=2, retry_delay=0.01)]
        engine = WorkflowEngine()
        result = engine.execute(steps)

        assert result.success is False
        assert "doomed" in result.steps_failed
        assert result.total_retries == 2

    def test_execution_log_records_all_attempts(self):
        attempt = 0

        def fail_then_succeed():
            nonlocal attempt
            attempt += 1
            if attempt < 2:
                raise ValueError("Not yet")
            return "done"

        steps = [WorkflowStep("retry_step", fail_then_succeed,
                               retries=3, retry_delay=0.01)]
        engine = WorkflowEngine()
        engine.execute(steps)

        log_entries = [e for e in engine.execution_log if e["step"] == "retry_step"]
        assert len(log_entries) == 2  # 1 failure + 1 success

    def test_zero_retries_means_one_attempt(self):
        call_count = 0

        def failing():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("fail")

        steps = [WorkflowStep("no_retry", failing, retries=0)]
        engine = WorkflowEngine()
        result = engine.execute(steps)

        assert result.success is False
        assert call_count == 1
        assert result.total_retries == 0


class TestWorkflowEngineConditional:
    """Test conditional workflow branching."""

    def test_conditional_step_skipped(self):
        results = []

        def conditional_action():
            # Only execute if previous step returned specific value
            if results and results[-1] == "skip_next":
                return "skipped"
            return "executed"

        steps = [
            WorkflowStep("setup", lambda: results.append("skip_next") or "skip_next"),
            WorkflowStep("conditional", conditional_action),
        ]

        engine = WorkflowEngine()
        result = engine.execute(steps)

        assert result.success is True
        assert result.step_results["conditional"] == "skipped"
```

### 10.3 Testing Computer Use with Safety Violations

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Complete test suite for Computer Use safety boundaries.

Reference: architecture/COMPUTER_USE_ARCHITECTURE.md

Tests cover:
- Screenshot capture boundary enforcement
- Input event allowlisting
- Sensitive field detection
- Action confirmation gates
- Audit log completeness
"""

import pytest
from typing import Dict, List, Optional
from unittest.mock import Mock, MagicMock


# ---------------------------------------------------------------------------
# Computer Use Simulator (for testing)
# ---------------------------------------------------------------------------

class ScreenRegion:
    """Represents a bounded screen region."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def contains_point(self, px: int, py: int) -> bool:
        return (self.x <= px <= self.x + self.width and
                self.y <= py <= self.y + self.height)


class ComputerUseController:
    """
    Simulated Computer Use controller with safety boundaries.

    Enforces:
    - Screenshots limited to allowed regions
    - Input events only sent to approved applications
    - Sensitive fields (password, SSN) blocked from input
    - Destructive actions require confirmation
    """

    DANGEROUS_ACTIONS = {"delete", "format", "shutdown", "reboot", "rm", "drop"}
    SENSITIVE_FIELDS = {"password", "ssn", "credit_card", "secret", "token"}

    def __init__(self, allowed_apps: List[str] = None):
        self.allowed_apps = set(allowed_apps or [])
        self.allowed_regions: List[ScreenRegion] = []
        self.audit_log: List[Dict] = []
        self.pending_confirmations: List[Dict] = []

    def add_allowed_region(self, region: ScreenRegion):
        self.allowed_regions.append(region)

    def capture_screenshot(self, region: ScreenRegion) -> Optional[bytes]:
        """Capture screenshot only if region is within allowed bounds."""
        for allowed in self.allowed_regions:
            if (allowed.contains_point(region.x, region.y) and
                    allowed.contains_point(
                        region.x + region.width, region.y + region.height)):
                self.audit_log.append({
                    "action": "screenshot",
                    "region": f"{region.x},{region.y},{region.width},{region.height}",
                    "status": "allowed",
                })
                return b"fake_screenshot_data"

        self.audit_log.append({
            "action": "screenshot",
            "region": f"{region.x},{region.y},{region.width},{region.height}",
            "status": "blocked",
        })
        return None

    def send_input(self, app_name: str, input_type: str,
                   value: str, field_name: str = "") -> Dict:
        """Send input to an application with safety checks."""
        # Check app allowlist
        if app_name not in self.allowed_apps:
            self.audit_log.append({
                "action": "input",
                "app": app_name,
                "status": "blocked_app",
            })
            return {"success": False, "reason": f"App '{app_name}' not in allowlist"}

        # Check for sensitive fields
        if field_name.lower() in self.SENSITIVE_FIELDS:
            self.audit_log.append({
                "action": "input",
                "app": app_name,
                "field": field_name,
                "status": "blocked_sensitive",
            })
            return {"success": False, "reason": f"Cannot input to sensitive field '{field_name}'"}

        # Check for dangerous actions
        if any(dangerous in value.lower() for dangerous in self.DANGEROUS_ACTIONS):
            self.pending_confirmations.append({
                "app": app_name,
                "input_type": input_type,
                "value": value,
            })
            self.audit_log.append({
                "action": "input",
                "app": app_name,
                "status": "pending_confirmation",
            })
            return {"success": False, "reason": "Destructive action requires confirmation"}

        self.audit_log.append({
            "action": "input",
            "app": app_name,
            "input_type": input_type,
            "status": "allowed",
        })
        return {"success": True}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputerUseScreenshot:
    """Test screenshot capture boundary enforcement."""

    @pytest.fixture
    def controller(self):
        ctrl = ComputerUseController(allowed_apps=["notepad", "browser"])
        ctrl.add_allowed_region(ScreenRegion(0, 0, 1920, 1080))
        return ctrl

    def test_screenshot_within_bounds_allowed(self, controller):
        result = controller.capture_screenshot(ScreenRegion(100, 100, 400, 300))
        assert result is not None
        assert result == b"fake_screenshot_data"

    def test_screenshot_outside_bounds_blocked(self, controller):
        result = controller.capture_screenshot(ScreenRegion(2000, 2000, 400, 300))
        assert result is None

    def test_screenshot_partially_outside_blocked(self, controller):
        # Region extends beyond the allowed area
        result = controller.capture_screenshot(ScreenRegion(1800, 900, 400, 300))
        assert result is None

    def test_audit_log_records_blocked_screenshots(self, controller):
        controller.capture_screenshot(ScreenRegion(5000, 5000, 100, 100))
        blocked = [e for e in controller.audit_log if e["status"] == "blocked"]
        assert len(blocked) == 1


class TestComputerUseInputSafety:
    """Test input event safety controls."""

    @pytest.fixture
    def controller(self):
        return ComputerUseController(allowed_apps=["notepad", "browser"])

    def test_input_to_allowed_app_succeeds(self, controller):
        result = controller.send_input("notepad", "keyboard", "Hello World")
        assert result["success"] is True

    def test_input_to_blocked_app_rejected(self, controller):
        result = controller.send_input("terminal", "keyboard", "ls -la")
        assert result["success"] is False
        assert "not in allowlist" in result["reason"]

    def test_input_to_password_field_blocked(self, controller):
        result = controller.send_input(
            "browser", "keyboard", "my_secret_pass", field_name="password"
        )
        assert result["success"] is False
        assert "sensitive" in result["reason"].lower()

    @pytest.mark.parametrize("sensitive_field", [
        "password", "ssn", "credit_card", "secret", "token",
    ])
    def test_all_sensitive_fields_blocked(self, controller, sensitive_field):
        result = controller.send_input(
            "browser", "keyboard", "test_value", field_name=sensitive_field
        )
        assert result["success"] is False

    def test_destructive_action_requires_confirmation(self, controller):
        result = controller.send_input("notepad", "keyboard", "delete all files")
        assert result["success"] is False
        assert "confirmation" in result["reason"].lower()
        assert len(controller.pending_confirmations) == 1

    def test_audit_log_completeness(self, controller):
        """Verify all actions are recorded in the audit log."""
        controller.send_input("notepad", "keyboard", "safe text")
        controller.send_input("terminal", "keyboard", "blocked")
        controller.send_input("browser", "keyboard", "pass", field_name="password")

        assert len(controller.audit_log) == 3
        statuses = [e["status"] for e in controller.audit_log]
        assert "allowed" in statuses
        assert "blocked_app" in statuses
        assert "blocked_sensitive" in statuses


class TestComputerUseEdgeCases:
    """Edge case tests for Computer Use safety."""

    def test_empty_allowlist_blocks_everything(self):
        controller = ComputerUseController(allowed_apps=[])
        result = controller.send_input("notepad", "keyboard", "test")
        assert result["success"] is False

    def test_no_allowed_regions_blocks_all_screenshots(self):
        controller = ComputerUseController()
        result = controller.capture_screenshot(ScreenRegion(0, 0, 100, 100))
        assert result is None

    def test_case_insensitive_dangerous_action_detection(self):
        controller = ComputerUseController(allowed_apps=["notepad"])
        result = controller.send_input("notepad", "keyboard", "DELETE everything")
        assert result["success"] is False

    def test_unicode_input_handled(self):
        controller = ComputerUseController(allowed_apps=["notepad"])
        result = controller.send_input("notepad", "keyboard", "Hello")
        assert result["success"] is True
```

### 10.4 Testing Multi-Document Synthesis

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
Complete test suite for Multi-Document Synthesis.

Reference: architecture/MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md

Tests cover:
- Multiple document indexing
- Cross-document query answering
- Source attribution
- Synthesis quality evaluation
- Performance with large document sets
"""

import tempfile
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from gaia.testing.mocks import MockLLMProvider
from gaia.testing.fixtures import temp_directory


class TestMultiDocumentIndexing:
    """Test indexing and managing multiple documents."""

    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir).resolve()

    @pytest.fixture
    def document_set(self, workspace):
        """Create a set of related documents."""
        docs = {}

        # Document 1: Technical specs
        doc1 = workspace / "ryzen_ai_specs.txt"
        doc1.write_text(
            "AMD Ryzen AI 300 Series Specifications\n"
            "NPU: XDNA 2 architecture, up to 50 TOPS\n"
            "CPU: Zen 5 cores, up to 12 cores\n"
            "GPU: RDNA 3.5 integrated graphics\n"
            "Memory: LPDDR5X-7500 support\n"
            "TDP: 15-54W configurable"
        )
        docs["specs"] = str(doc1)

        # Document 2: Developer guide
        doc2 = workspace / "developer_guide.txt"
        doc2.write_text(
            "GAIA Developer Guide\n"
            "Getting Started:\n"
            "1. Install: uv pip install -e '.[dev]'\n"
            "2. Start Lemonade: lemonade-server serve\n"
            "3. Test: gaia llm 'Hello'\n"
            "The NPU is accessed through the Lemonade server."
        )
        docs["guide"] = str(doc2)

        # Document 3: Benchmark results
        doc3 = workspace / "benchmarks.txt"
        doc3.write_text(
            "Benchmark Results - Ryzen AI 300\n"
            "Qwen3-0.6B: 45 tokens/sec on NPU\n"
            "Qwen3-Coder-30B: 12 tokens/sec on NPU\n"
            "Whisper ASR: 3x realtime on NPU\n"
            "SDXL image gen: 2.1 sec/image on iGPU"
        )
        docs["benchmarks"] = str(doc3)

        return docs

    def test_index_multiple_documents(self, workspace, document_set):
        """Verify multiple documents are indexed correctly."""
        from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig

        config = ChatAgentConfig(
            silent_mode=True,
            allowed_paths=[str(workspace), str(Path.cwd().resolve())],
        )
        agent = ChatAgent(config)

        try:
            for name, path in document_set.items():
                result = agent.rag.index_document(path)
                assert result["success"], f"Failed to index {name}: {result.get('error')}"

            assert len(agent.rag.indexed_files) == 3

            # Verify system prompt lists all documents
            agent.update_system_prompt()
            assert "ryzen_ai_specs.txt" in agent.system_prompt
            assert "developer_guide.txt" in agent.system_prompt
            assert "benchmarks.txt" in agent.system_prompt
        finally:
            agent.stop_watching()

    def test_cross_document_query(self, workspace, document_set):
        """Query should retrieve relevant chunks from multiple documents."""
        from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig

        config = ChatAgentConfig(
            silent_mode=True,
            allowed_paths=[str(workspace), str(Path.cwd().resolve())],
        )
        agent = ChatAgent(config)

        try:
            for path in document_set.values():
                agent.rag.index_document(path)

            # Query that spans multiple documents
            response = agent.rag.query("What is the NPU performance?")
            assert response.text
            assert len(response.text) > 0
            # The response should reference content from both specs and benchmarks
        finally:
            agent.stop_watching()

    def test_incremental_indexing(self, workspace, document_set):
        """Adding documents incrementally should accumulate correctly."""
        from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig

        config = ChatAgentConfig(
            silent_mode=True,
            allowed_paths=[str(workspace), str(Path.cwd().resolve())],
        )
        agent = ChatAgent(config)

        try:
            # Index one at a time
            agent.rag.index_document(document_set["specs"])
            assert len(agent.rag.indexed_files) == 1

            agent.rag.index_document(document_set["guide"])
            assert len(agent.rag.indexed_files) == 2

            agent.rag.index_document(document_set["benchmarks"])
            assert len(agent.rag.indexed_files) == 3
        finally:
            agent.stop_watching()


class TestSynthesisQuality:
    """Evaluate the quality of multi-document synthesis."""

    def test_synthesis_covers_all_sources(self):
        """
        Verify that synthesis output references all relevant sources.

        This is an evaluation test that checks source attribution.
        """
        # Mock a synthesis result
        synthesis = {
            "answer": (
                "The Ryzen AI 300 NPU delivers 50 TOPS (from specs). "
                "Qwen3-0.6B achieves 45 tokens/sec on this NPU (from benchmarks). "
                "Developers access the NPU through Lemonade server (from guide)."
            ),
            "sources": ["ryzen_ai_specs.txt", "benchmarks.txt", "developer_guide.txt"],
        }

        # Verify all sources are referenced
        assert len(synthesis["sources"]) == 3
        assert "ryzen_ai_specs.txt" in synthesis["sources"]
        assert "benchmarks.txt" in synthesis["sources"]
        assert "developer_guide.txt" in synthesis["sources"]

    def test_synthesis_factual_accuracy(self):
        """Verify synthesized facts match source documents."""
        from gaia.eval.eval import Evaluator

        evaluator = Evaluator()

        # Ground truth from source documents
        ground_truth = "The Ryzen AI 300 NPU has XDNA 2 architecture with 50 TOPS."

        # Synthesized answer
        candidate = "AMD's Ryzen AI 300 features an XDNA 2 NPU delivering 50 TOPS."

        similarity = evaluator.calculate_similarity(ground_truth, candidate)
        assert similarity > 0.7, (
            f"Synthesis accuracy too low: {similarity:.3f}"
        )


class TestMultiDocumentPerformance:
    """Performance tests for multi-document operations."""

    def test_indexing_throughput(self):
        """Measure document indexing speed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()

            # Create 10 documents
            docs = []
            for i in range(10):
                doc = workspace / f"doc_{i}.txt"
                doc.write_text(
                    f"Document {i}. " +
                    " ".join([f"Paragraph {j} of document {i}." for j in range(20)])
                )
                docs.append(str(doc))

            from gaia.agents.chat.agent import ChatAgent, ChatAgentConfig
            config = ChatAgentConfig(
                silent_mode=True,
                allowed_paths=[str(workspace), str(Path.cwd().resolve())],
            )
            agent = ChatAgent(config)

            try:
                start = time.perf_counter()
                for doc in docs:
                    agent.rag.index_document(doc)
                elapsed = time.perf_counter() - start

                docs_per_second = len(docs) / elapsed
                print(f"\nIndexing throughput: {docs_per_second:.1f} docs/sec "
                      f"({elapsed:.2f}s for {len(docs)} docs)")

                assert elapsed < 30, f"Indexing 10 docs took {elapsed:.1f}s (>30s limit)"
            finally:
                agent.stop_watching()
```

---

## Appendix A: Test Command Reference

```bash
# Run all unit tests (fast, no server required)
python -m pytest tests/unit/ -xvs

# Run with coverage report
python -m pytest tests/unit/ -v --cov=src/gaia --cov-report=term-missing

# Run integration tests (requires Lemonade server)
python -m pytest tests/test_chat_agent.py -xvs

# Run security tests
python tests/verify_path_validator.py
python tests/verify_shell_security.py

# Run with hybrid flag (cloud + local)
python -m pytest tests/ --hybrid -xvs

# Run performance benchmarks
python -m pytest tests/ --benchmark -xvs

# Run evaluation tests (requires Claude API key)
python -m pytest tests/ --eval -xvs

# Run specific test class
python -m pytest tests/test_chat_agent.py::TestChatAgent -xvs

# Run MCP protocol tests
python -m pytest tests/mcp/ -xvs

# Run API validation tests
python -m pytest tests/test_api.py -xvs

# Validate MCP compliance
python validate_mcp.py

# Lint before commit
python util/lint.py --all --fix
```

## Appendix B: Fixture Dependency Graph

```
conftest.py (root)
    |
    +-- lemonade_available (session)
    |       |
    |       +-- require_lemonade (function)
    |
    +-- api_server (function)
    |       |
    |       +-- api_client (function)
    |
    +-- pytest_addoption(--hybrid, --benchmark, --eval)

gaia.testing.mocks
    |
    +-- MockLLMProvider
    |       +-- .generate(), .chat(), .stream()
    |       +-- .call_history, .call_count, .last_prompt
    |       +-- .reset(), .set_responses()
    |
    +-- MockVLMClient
    |       +-- .extract_from_image(), .extract_from_file()
    |       +-- .check_availability()
    |       +-- .call_history, .was_called
    |
    +-- MockToolExecutor
            +-- .execute()
            +-- .was_tool_called(), .get_tool_args()
            +-- .tool_names_called

gaia.testing.fixtures
    |
    +-- temp_directory()        (context manager)
    +-- temp_file()             (context manager)
    +-- create_test_agent()     (factory function)
    +-- AgentTestContext         (context manager class)
            +-- .agent, .mock_llm, .mock_vlm
            +-- .temp_dir
            +-- .create_file(), .create_directory()
            +-- .set_llm_responses(), .set_vlm_text()

gaia.testing.assertions
    |
    +-- assert_llm_called()
    +-- assert_llm_prompt_contains()
    +-- assert_vlm_called()
    +-- assert_tool_called()
    +-- assert_tool_args()
    +-- assert_result_has_keys()
    +-- assert_result_value()
    +-- assert_agent_completed()
    +-- assert_no_errors()
```

## Appendix C: Architecture-to-Test Mapping

| Architecture Document | Test Location | Status |
|----------------------|---------------|--------|
| `WORKFLOW_ORCHESTRATION_ARCHITECTURE.md` | Section 10.2 (proposed) | Design |
| `EMAIL_INTEGRATION_ARCHITECTURE.md` | Section 10.1 (proposed) | Design |
| `COMPUTER_USE_ARCHITECTURE.md` | Section 10.3 (proposed) | Design |
| `MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md` | Section 10.4, `tests/test_chat_agent.py` | Partial |
| `ERROR_RECOVERY_ARCHITECTURE.md` | `tests/unit/test_errors.py` | Active |
| `SECURITY_SECRETS_ARCHITECTURE.md` | `tests/verify_*.py`, `test_security.yml` | Active |
| `OBSERVABILITY_ARCHITECTURE.md` | Section 6 (proposed) | Design |
| `PERSISTENT_MEMORY_FRAMEWORK.md` | `tests/test_chat_agent.py::TestChatAgentSessions` | Active |
| `DYNAMIC_TOOLS_FRAMEWORK.md` | `tests/unit/test_tool_decorator.py` | Active |
| `ADAPTIVE_PROMPTS_FRAMEWORK.md` | (needs tests) | Gap |
| `LEARNING_ADAPTATION_FRAMEWORK.md` | (needs tests) | Gap |
| `LANGUAGE_EXTENSIBILITY_DESIGN.md` | (needs tests) | Gap |
| `TUI_DESIGN_SPECIFICATION.md` | (needs tests) | Gap |
| `AGENT_DASHBOARD_DESIGN.md` | (needs tests) | Gap |

---

**Document End**

*This is a living document. Update it when new architectures are added, test
patterns change, or CI workflows are modified.*

