# Multi-Agent Orchestration Architecture for GAIA

**Date**: February 7, 2026
**Version**: 1.0
**Status**: Specification
**Priority**: HIGH
**Estimated Effort**: 12-16 weeks (2-3 engineers)
**Target**: Enable complex tasks requiring multiple specialized agents working together with coordination, state sharing, and conflict resolution

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Patterns](#3-architecture-patterns)
4. [Orchestrator Agent](#4-orchestrator-agent)
5. [Message Passing Protocol](#5-message-passing-protocol)
6. [Shared State Management](#6-shared-state-management)
7. [Agent Lifecycle](#7-agent-lifecycle)
8. [Coordination Strategies](#8-coordination-strategies)
9. [Example Multi-Agent Workflows](#9-example-multi-agent-workflows)
10. [Integration with Workflow Orchestration](#10-integration-with-workflow-orchestration)

---

## 1. Executive Summary

### The Gap

GAIA currently has seven specialized agents (`ChatAgent`, `CodeAgent`, `JiraAgent`, `BlenderAgent`, `DockerAgent`, `MedicalIntakeAgent`, `RoutingAgent`) that operate independently. The `RoutingAgent` provides basic request-level routing -- it analyzes a user query, selects a single agent, and delegates execution. However, there is no mechanism for:

- Multiple agents working on the same task concurrently
- Agents sharing intermediate results with each other
- An orchestrator decomposing a complex task into sub-tasks for different specialists
- Conflict resolution when agents produce contradictory outputs
- Agent-to-agent communication during execution
- Coordinated state management across agent boundaries

### The Solution

A **Multi-Agent Orchestration Architecture** built on three foundational components:

1. **OrchestratorAgent** -- A new agent type that decomposes complex tasks, delegates sub-tasks to specialist agents, aggregates results, and resolves conflicts.
2. **Message Passing Protocol** -- A typed, async message system enabling request/reply, broadcast, and pub/sub communication patterns between agents.
3. **Shared State Manager** -- A blackboard-style shared workspace where agents read and write artifacts, with versioning and conflict detection.

### Design Principles

- **Backward compatible**: Existing agents work without modification; orchestration is opt-in.
- **Local-first**: All orchestration runs locally on the user's machine; no cloud services required.
- **AMD-optimized**: Designed for efficient resource sharing when multiple agents use the same Lemonade Server instance.
- **Incremental adoption**: Teams can start with simple sequential pipelines and progress to parallel orchestration.

### Impact

| Metric | Before | After |
|--------|--------|-------|
| Multi-agent task coverage | 0% (single-agent only) | 85% (hierarchical + pipeline + parallel) |
| Max concurrent agents | 1 | Configurable (default: 4) |
| Inter-agent communication | None | Full message passing + shared state |
| Task decomposition | Manual (user must pick agent) | Automatic via OrchestratorAgent |

---

## 2. Problem Statement

### Current Architecture Limitations

The current GAIA agent system operates on a **one-query, one-agent** model. The `Agent` base class in `src/gaia/agents/base/agent.py` defines the lifecycle:

```
User Query --> RoutingAgent --> Single Agent --> process_query() --> Result
```

The `RoutingAgent` in `src/gaia/agents/routing/agent.py` selects one agent based on keyword detection and LLM analysis, then hands off entirely. There is no mechanism for the selected agent to recruit other agents, share partial results, or coordinate parallel work.

**Concrete limitations in the current code:**

1. **`_TOOL_REGISTRY` is global and flat** (`src/gaia/agents/base/tools.py`): All tools share a single dictionary. Two agents running simultaneously would overwrite each other's tool registrations.

2. **`Agent.__init__` initializes a single `ChatSDK`** (line 203-213 of `agent.py`): Each agent manages its own conversation history with no mechanism to share context across agents.

3. **`RoutingAgent.process_query` returns a single agent** (line 162-168 of `routing/agent.py`): The routing logic assumes exactly one agent handles the entire query.

4. **State constants are per-agent** (`STATE_PLANNING`, `STATE_EXECUTING_PLAN`, etc.): No shared state machine exists for coordinating multiple agents through a joint execution plan.

### User Stories Requiring Multi-Agent Coordination

**Story 1: "Build a full-stack application with tests"**

This requires:
- `CodeAgent` (backend): Generate API endpoints, database models, server configuration
- `CodeAgent` (frontend): Generate UI components, routing, API client
- A testing specialist: Generate unit tests, integration tests, E2E tests
- `DockerAgent`: Create Dockerfile, docker-compose.yml, deployment config

Today: User must issue 4+ separate prompts, manually pass context between them, and resolve conflicts (e.g., mismatched API contracts between frontend and backend).

**Story 2: "Research a topic and produce a report with visualizations"**

This requires:
- `ChatAgent` with RAG: Search documents, extract relevant passages
- An analysis agent: Synthesize findings, identify patterns, draw conclusions
- A writing agent: Draft the report with proper structure and citations
- `BlenderAgent` or a chart agent: Generate visualizations from data

Today: User must manually shepherd data through each step.

**Story 3: "Triage incoming Jira issues and route to appropriate teams"**

This requires:
- `JiraAgent`: Fetch unresolved issues, read descriptions
- A classification agent: Categorize by severity, component, team
- An approval agent: Apply rules, request human approval for edge cases
- `JiraAgent` again: Update labels, assign teams, post comments

Today: Requires custom scripting outside GAIA.

### Challenges

| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Resource contention** | Multiple agents sharing one Lemonade Server instance creates serialized LLM calls | Performance bottleneck |
| **State coherence** | Agent A's output must be consistent with Agent B's assumptions | Correctness |
| **Error propagation** | If one agent fails mid-pipeline, downstream agents receive no input | Reliability |
| **Context window limits** | Aggregating results from multiple agents may exceed LLM context | Feasibility |
| **Tool registry conflicts** | Global `_TOOL_REGISTRY` cannot support concurrent agents with different tools | Architecture |
| **Determinism** | Parallel execution introduces non-deterministic ordering | Reproducibility |

---

## 3. Architecture Patterns

This section defines four orchestration patterns, each suited to different multi-agent scenarios. The implementation supports all four, allowing users to select the appropriate pattern for their use case.

### 3.1 Hierarchical Pattern (Orchestrator Delegates to Specialists)

```
                    +---------------------+
                    |  OrchestratorAgent  |
                    |  (task decomposer)  |
                    +---------------------+
                   /          |            \
                  v           v             v
        +-----------+  +-----------+  +-----------+
        | CodeAgent |  | JiraAgent |  |DockerAgent|
        | (backend) |  | (issues)  |  | (deploy)  |
        +-----------+  +-----------+  +-----------+
```

**When to use**: Complex tasks that decompose into independent sub-problems handled by different agent types.

**How it works**:
1. User submits query to `OrchestratorAgent`
2. Orchestrator uses LLM to decompose into sub-tasks
3. Each sub-task is assigned to a specialist agent
4. Specialists execute independently (possibly in parallel)
5. Orchestrator aggregates results and resolves conflicts

**Implementation approach**:

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from gaia.agents.base.agent import Agent
from gaia.agents.orchestration.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent(
    pattern="hierarchical",
    max_concurrent=3,
)

result = orchestrator.process_query(
    "Build a REST API with Docker deployment and create Jira tickets for the work"
)
# Orchestrator internally:
#   1. Decomposes into: [code_task, docker_task, jira_task]
#   2. Creates: CodeAgent, DockerAgent, JiraAgent
#   3. Executes sub-tasks (parallel where possible)
#   4. Aggregates results into unified response
```

**Advantages**: Clear authority hierarchy, predictable execution, easy to debug.
**Disadvantages**: Orchestrator is a bottleneck, specialists cannot communicate directly.

### 3.2 Peer-to-Peer Pattern (Agents Communicate Directly)

```
        +-----------+  <------>  +-----------+
        | CodeAgent |            | TestAgent |
        | (backend) |            | (tests)   |
        +-----------+  <------>  +-----------+
              ^                        ^
              |                        |
              v                        v
        +-----------+  <------>  +-----------+
        |DockerAgent|            | JiraAgent |
        | (deploy)  |            | (track)   |
        +-----------+            +-----------+
```

**When to use**: Agents need to collaborate iteratively (e.g., code agent generates code, test agent validates it, code agent fixes failures).

**How it works**:
1. Agents register with a message broker
2. Any agent can send messages to any other agent
3. Agents subscribe to topics they care about
4. No central coordinator; agents self-organize

**Implementation approach**:

```python
from gaia.agents.orchestration.message_bus import MessageBus
from gaia.agents.orchestration.peer import PeerAgent

bus = MessageBus()

code_agent = PeerAgent(agent=CodeAgent(), bus=bus, topics=["code", "review"])
test_agent = PeerAgent(agent=TestAgent(), bus=bus, topics=["test", "code"])

# code_agent generates code, publishes to "code" topic
# test_agent receives code, runs tests, publishes results to "review" topic
# code_agent receives review, fixes issues, republishes
```

**Advantages**: Flexible, supports iterative refinement, no single point of failure.
**Disadvantages**: Harder to debug, potential for infinite loops, non-deterministic ordering.

### 3.3 Blackboard Pattern (Shared State, Agents Contribute)

```
        +-----------+     +-----------+     +-----------+
        | CodeAgent |     | TestAgent |     |DockerAgent|
        +-----------+     +-----------+     +-----------+
              |                 |                 |
              v                 v                 v
        +---------------------------------------------+
        |              BLACKBOARD                      |
        |  +---------+ +---------+ +---------+        |
        |  | api.py  | |tests.py | |Dockerfile|       |
        |  +---------+ +---------+ +---------+        |
        |  status: { code: done, tests: running }     |
        +---------------------------------------------+
```

**When to use**: Multiple agents contribute to a shared artifact (e.g., building a project where files are the shared state).

**How it works**:
1. A shared blackboard holds all artifacts and metadata
2. Agents read from the blackboard, perform work, write results back
3. A controller monitors the blackboard and activates agents when their inputs are ready
4. Agents are stateless relative to each other; all shared state lives on the blackboard

**Implementation approach**:

```python
from gaia.agents.orchestration.blackboard import Blackboard, BlackboardController

blackboard = Blackboard()
controller = BlackboardController(blackboard)

controller.register_agent(
    agent=CodeAgent(),
    reads=["requirements"],
    writes=["source_code"],
    activation_condition=lambda bb: "requirements" in bb,
)
controller.register_agent(
    agent=TestAgent(),
    reads=["source_code"],
    writes=["test_results"],
    activation_condition=lambda bb: "source_code" in bb,
)

blackboard.write("requirements", "Build a REST API with user authentication")
controller.run()  # Activates agents as their inputs become available
```

**Advantages**: Clean separation of concerns, easy to add new agents, natural for artifact-based tasks.
**Disadvantages**: Requires careful design of blackboard schema, potential for stale reads.

### 3.4 Pipeline Pattern (Output of One Becomes Input of Next)

```
  +----------+     +----------+     +----------+     +----------+
  | Research |---->| Analyze  |---->|  Write   |---->| Review   |
  |  Agent   |     |  Agent   |     |  Agent   |     |  Agent   |
  +----------+     +----------+     +----------+     +----------+
    search &         synthesize       draft the        quality
    extract          findings         report           check
```

**When to use**: Tasks with clear sequential dependencies where each stage transforms the previous stage's output.

**How it works**:
1. Define an ordered list of (agent, transform) pairs
2. First agent receives the original query
3. Each subsequent agent receives the previous agent's output (possibly transformed)
4. Final agent's output is the pipeline result

**Implementation approach**:

```python
from gaia.agents.orchestration.pipeline import Pipeline, PipelineStage

pipeline = Pipeline(stages=[
    PipelineStage(
        agent=ChatAgent(),   # RAG-enabled search
        name="research",
        transform_input=lambda query: f"Search for: {query}",
        transform_output=lambda result: result["findings"],
    ),
    PipelineStage(
        agent=AnalysisAgent(),
        name="analyze",
        transform_input=lambda findings: f"Analyze these findings:\n{findings}",
    ),
    PipelineStage(
        agent=ChatAgent(),   # Writing mode
        name="write",
        transform_input=lambda analysis: f"Write a report based on:\n{analysis}",
    ),
])

report = pipeline.run("Impact of AMD NPU acceleration on local LLM inference")
```

**Advantages**: Simple to understand, predictable execution order, easy to test each stage.
**Disadvantages**: No parallelism, late stages blocked by early stages, no feedback loops.

### 3.5 Pattern Selection Guide

| Scenario | Recommended Pattern | Reason |
|----------|-------------------|--------|
| Complex task with independent sub-problems | Hierarchical | Clear decomposition, parallel execution |
| Iterative refinement (code + test cycles) | Peer-to-Peer | Agents need back-and-forth communication |
| Building a shared artifact (project files) | Blackboard | Multiple contributors to shared state |
| Sequential data transformation | Pipeline | Clear input/output chain |
| Unknown or mixed requirements | Hierarchical | Most flexible, orchestrator can adapt |

---

## 4. Orchestrator Agent

The `OrchestratorAgent` is the central component of the multi-agent system. It extends the existing `Agent` base class and adds task decomposition, agent selection, work distribution, result aggregation, and conflict resolution.

### 4.1 Class Design

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
OrchestratorAgent - Coordinates multiple specialist agents for complex tasks.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from gaia.agents.base.agent import Agent
from gaia.agents.base.tools import tool
from gaia.agents.orchestration.message_bus import MessageBus
from gaia.agents.orchestration.registry import AgentRegistry
from gaia.agents.orchestration.state import SharedStateManager


class OrchestrationPattern(Enum):
    """Supported orchestration patterns."""
    HIERARCHICAL = "hierarchical"
    PEER_TO_PEER = "peer_to_peer"
    BLACKBOARD = "blackboard"
    PIPELINE = "pipeline"


class SubTaskStatus(Enum):
    """Status of a sub-task in the orchestration plan."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class SubTask:
    """
    A decomposed unit of work assigned to a specialist agent.

    Attributes:
        id: Unique identifier for this sub-task.
        name: Human-readable name describing the sub-task.
        description: Detailed description for the assigned agent.
        agent_type: The type of agent to handle this sub-task.
        dependencies: IDs of sub-tasks that must complete first.
        priority: Execution priority (lower = higher priority).
        status: Current execution status.
        result: Output from the agent after completion.
        error: Error message if the sub-task failed.
        retries: Number of retry attempts remaining.
        metadata: Additional context passed to the agent.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    agent_type: str = "chat"
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationPlan:
    """
    The complete execution plan produced by task decomposition.

    Attributes:
        id: Unique plan identifier.
        query: Original user query.
        sub_tasks: Ordered list of sub-tasks.
        pattern: Orchestration pattern to use.
        created_at: Timestamp of plan creation.
        metadata: Additional plan-level metadata.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    sub_tasks: List[SubTask] = field(default_factory=list)
    pattern: OrchestrationPattern = OrchestrationPattern.HIERARCHICAL
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_ready_tasks(self) -> List[SubTask]:
        """Return sub-tasks whose dependencies are all completed."""
        completed_ids = {
            t.id for t in self.sub_tasks if t.status == SubTaskStatus.COMPLETED
        }
        return [
            t for t in self.sub_tasks
            if t.status == SubTaskStatus.PENDING
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def is_complete(self) -> bool:
        """Check if all sub-tasks are completed or failed."""
        return all(
            t.status in (SubTaskStatus.COMPLETED, SubTaskStatus.FAILED, SubTaskStatus.CANCELLED)
            for t in self.sub_tasks
        )

    def get_results(self) -> Dict[str, Any]:
        """Collect results from all completed sub-tasks."""
        return {
            t.name: t.result
            for t in self.sub_tasks
            if t.status == SubTaskStatus.COMPLETED and t.result is not None
        }


class OrchestratorAgent(Agent):
    """
    Coordinates multiple specialist agents to solve complex tasks.

    The OrchestratorAgent extends the base Agent with multi-agent
    capabilities. It uses the LLM to decompose tasks, then manages
    the execution lifecycle of specialist agents.

    Inherits from Agent to reuse:
    - LLM client and conversation management
    - Tool registration framework
    - State management (extended for orchestration states)
    - Console output and error handling

    Attributes:
        pattern: Default orchestration pattern.
        max_concurrent: Maximum number of agents running in parallel.
        registry: Agent registry for discovering available specialists.
        message_bus: Message passing infrastructure.
        state_manager: Shared state manager for inter-agent data.
        plans: History of orchestration plans.
    """

    # Extended state constants for orchestration
    STATE_DECOMPOSING = "DECOMPOSING"
    STATE_DISPATCHING = "DISPATCHING"
    STATE_MONITORING = "MONITORING"
    STATE_AGGREGATING = "AGGREGATING"
    STATE_RESOLVING_CONFLICTS = "RESOLVING_CONFLICTS"

    def __init__(
        self,
        pattern: str = "hierarchical",
        max_concurrent: int = 4,
        agent_timeout: int = 300,
        enable_message_bus: bool = True,
        enable_shared_state: bool = True,
        **kwargs,
    ):
        """
        Initialize OrchestratorAgent.

        Args:
            pattern: Default orchestration pattern
                ("hierarchical", "peer_to_peer", "blackboard", "pipeline").
            max_concurrent: Maximum number of concurrent agent executions.
            agent_timeout: Timeout in seconds for individual agent execution.
            enable_message_bus: If True, initialize message bus for inter-agent communication.
            enable_shared_state: If True, initialize shared state manager.
            **kwargs: Arguments passed to Agent.__init__().
        """
        super().__init__(**kwargs)

        self.pattern = OrchestrationPattern(pattern)
        self.max_concurrent = max_concurrent
        self.agent_timeout = agent_timeout

        # Initialize orchestration infrastructure
        self.registry = AgentRegistry()
        self.message_bus = MessageBus() if enable_message_bus else None
        self.state_manager = SharedStateManager() if enable_shared_state else None

        # Plan tracking
        self.plans: List[OrchestrationPlan] = []
        self.active_agents: Dict[str, Agent] = {}

        # Register built-in specialist agents
        self._register_default_agents()

    def _get_system_prompt(self) -> str:
        """Return the orchestrator-specific system prompt."""
        available_agents = self.registry.list_agents()
        agent_descriptions = "\n".join(
            f"  - {name}: {info['description']} (capabilities: {', '.join(info['capabilities'])})"
            for name, info in available_agents.items()
        )

        return f"""You are the GAIA Orchestrator Agent. Your role is to decompose complex
user requests into sub-tasks and assign them to specialist agents.

AVAILABLE SPECIALIST AGENTS:
{agent_descriptions}

TASK DECOMPOSITION RULES:
1. Break complex tasks into the smallest independent sub-tasks possible.
2. Identify dependencies between sub-tasks (which must complete before others start).
3. Assign each sub-task to the most appropriate specialist agent.
4. Maximize parallelism by minimizing unnecessary dependencies.
5. Each sub-task description must be self-contained (include all context the agent needs).

OUTPUT FORMAT for decomposition:
{{
    "thought": "Analysis of the task and decomposition strategy",
    "plan": {{
        "pattern": "hierarchical|pipeline|blackboard|peer_to_peer",
        "sub_tasks": [
            {{
                "name": "descriptive_name",
                "description": "Detailed instructions for the agent",
                "agent_type": "code|chat|jira|docker|blender",
                "dependencies": ["name_of_dependency"],
                "priority": 0
            }}
        ]
    }}
}}

OUTPUT FORMAT for aggregation:
{{
    "thought": "How I combined the results",
    "answer": "Unified response to the user incorporating all sub-task results"
}}

CONFLICT RESOLUTION:
When sub-tasks produce contradictory results, prefer:
1. Results with higher confidence scores
2. Results from more specialized agents
3. Results that are consistent with the majority of other sub-task outputs
4. When in doubt, present both options to the user
"""

    def _register_tools(self):
        """Register orchestrator-specific tools."""

        @tool
        def decompose_task(query: str) -> dict:
            """Decompose a complex query into sub-tasks for specialist agents."""
            return self._decompose_task(query)

        @tool
        def check_agent_status(agent_id: str) -> dict:
            """Check the current status of a running specialist agent."""
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                return {
                    "agent_id": agent_id,
                    "state": agent.execution_state,
                    "step": agent.current_step,
                }
            return {"error": f"No active agent with ID {agent_id}"}

        @tool
        def aggregate_results(plan_id: str) -> dict:
            """Aggregate results from all completed sub-tasks in a plan."""
            plan = next((p for p in self.plans if p.id == plan_id), None)
            if plan is None:
                return {"error": f"No plan with ID {plan_id}"}
            return plan.get_results()

    def _register_default_agents(self):
        """Register the built-in GAIA agents as available specialists."""
        self.registry.register(
            name="code",
            agent_class_path="gaia.agents.code.agent.CodeAgent",
            description="Generates, reviews, and refactors code. Supports TypeScript/Next.js.",
            capabilities=["code_generation", "code_review", "refactoring", "testing"],
        )
        self.registry.register(
            name="chat",
            agent_class_path="gaia.agents.chat.agent.ChatAgent",
            description="Conversational agent with RAG-based document search and Q&A.",
            capabilities=["document_search", "question_answering", "summarization"],
        )
        self.registry.register(
            name="jira",
            agent_class_path="gaia.agents.jira.agent.JiraAgent",
            description="Manages Jira issues: create, update, search, and triage.",
            capabilities=["issue_creation", "issue_search", "issue_update", "triage"],
        )
        self.registry.register(
            name="docker",
            agent_class_path="gaia.agents.docker.agent.DockerAgent",
            description="Manages Docker containers, images, and compose configurations.",
            capabilities=["container_management", "image_building", "compose"],
        )
        self.registry.register(
            name="blender",
            agent_class_path="gaia.agents.blender.agent.BlenderAgent",
            description="Automates Blender 3D scene creation and procedural modeling.",
            capabilities=["3d_modeling", "scene_creation", "procedural_generation"],
        )

    # -----------------------------------------------------------------
    # Task Decomposition
    # -----------------------------------------------------------------

    def _decompose_task(self, query: str) -> OrchestrationPlan:
        """
        Use the LLM to decompose a complex query into sub-tasks.

        Args:
            query: The user's original query.

        Returns:
            OrchestrationPlan with sub-tasks, dependencies, and agent assignments.
        """
        import datetime

        self.execution_state = self.STATE_DECOMPOSING
        self.console.print_state_info("Decomposing task into sub-tasks...")

        # Ask the LLM to decompose the task
        decomposition_prompt = f"""Decompose this task into sub-tasks for specialist agents.

User request: "{query}"

Remember:
- Each sub-task must specify an agent_type from the available agents.
- Identify dependencies (which sub-tasks must complete before others).
- Maximize parallelism where possible.
- Each sub-task description must be self-contained.
"""

        response = self.chat.send_message(
            decomposition_prompt,
            system_prompt=self.system_prompt,
        )

        # Parse the LLM response into an OrchestrationPlan
        plan_data = self._parse_decomposition(response)

        plan = OrchestrationPlan(
            query=query,
            pattern=self.pattern,
            created_at=datetime.datetime.now().isoformat(),
        )

        # Build sub-tasks from parsed data
        name_to_id = {}
        for task_data in plan_data.get("sub_tasks", []):
            sub_task = SubTask(
                name=task_data.get("name", "unnamed"),
                description=task_data.get("description", ""),
                agent_type=task_data.get("agent_type", "chat"),
                priority=task_data.get("priority", 0),
                metadata=task_data.get("metadata", {}),
            )
            name_to_id[sub_task.name] = sub_task.id
            plan.sub_tasks.append(sub_task)

        # Resolve dependency names to IDs
        for task_data, sub_task in zip(plan_data.get("sub_tasks", []), plan.sub_tasks):
            dep_names = task_data.get("dependencies", [])
            sub_task.dependencies = [
                name_to_id[name] for name in dep_names if name in name_to_id
            ]

        # Override pattern if LLM suggested one
        suggested_pattern = plan_data.get("pattern")
        if suggested_pattern:
            try:
                plan.pattern = OrchestrationPattern(suggested_pattern)
            except ValueError:
                pass  # Keep default pattern

        self.plans.append(plan)
        return plan

    def _parse_decomposition(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM decomposition response into structured data.

        Args:
            response: Raw LLM response text.

        Returns:
            Dictionary with 'sub_tasks' list and optional 'pattern' string.
        """
        import json
        import re

        # Try to extract JSON from the response
        # Handle markdown code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Try to find raw JSON
            brace_start = response.find("{")
            brace_end = response.rfind("}")
            if brace_start >= 0 and brace_end > brace_start:
                text = response[brace_start : brace_end + 1]
            else:
                # Fallback: create a single sub-task for the whole query
                return {
                    "sub_tasks": [
                        {
                            "name": "full_task",
                            "description": response,
                            "agent_type": "chat",
                            "dependencies": [],
                            "priority": 0,
                        }
                    ]
                }

        try:
            parsed = json.loads(text)
            # Handle nested "plan" key
            if "plan" in parsed:
                return parsed["plan"]
            return parsed
        except json.JSONDecodeError:
            return {
                "sub_tasks": [
                    {
                        "name": "full_task",
                        "description": response,
                        "agent_type": "chat",
                        "dependencies": [],
                        "priority": 0,
                    }
                ]
            }

    # -----------------------------------------------------------------
    # Agent Selection and Routing
    # -----------------------------------------------------------------

    def _select_agent(self, sub_task: SubTask) -> Agent:
        """
        Select and instantiate the appropriate agent for a sub-task.

        Uses the AgentRegistry to find the agent class, then instantiates
        it with appropriate configuration.

        Args:
            sub_task: The sub-task requiring an agent.

        Returns:
            Configured Agent instance ready for execution.

        Raises:
            ValueError: If the requested agent type is not registered.
        """
        agent_info = self.registry.get(sub_task.agent_type)
        if agent_info is None:
            raise ValueError(
                f"No agent registered for type '{sub_task.agent_type}'. "
                f"Available: {list(self.registry.list_agents().keys())}"
            )

        # Dynamically import and instantiate the agent
        agent_class = self.registry.import_agent_class(sub_task.agent_type)

        # Configure the agent with orchestration-aware settings
        agent = agent_class(
            silent_mode=True,          # Suppress individual agent console output
            max_steps=self.max_steps,   # Inherit step limits
            streaming=False,            # Collect full results, do not stream
        )

        return agent

    # -----------------------------------------------------------------
    # Work Distribution
    # -----------------------------------------------------------------

    async def _execute_plan(self, plan: OrchestrationPlan) -> Dict[str, Any]:
        """
        Execute an orchestration plan by dispatching sub-tasks to agents.

        Manages the execution lifecycle: dispatching ready tasks,
        monitoring progress, handling failures, and collecting results.

        Args:
            plan: The orchestration plan to execute.

        Returns:
            Aggregated results from all sub-tasks.
        """
        self.execution_state = self.STATE_DISPATCHING
        self.console.print_state_info(
            f"Executing plan with {len(plan.sub_tasks)} sub-tasks "
            f"(pattern: {plan.pattern.value})"
        )

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_sub_task(sub_task: SubTask):
            """Execute a single sub-task with concurrency control."""
            async with semaphore:
                sub_task.status = SubTaskStatus.RUNNING
                self.console.print_step_header(
                    plan.sub_tasks.index(sub_task) + 1, len(plan.sub_tasks)
                )
                self.console.print_thought(
                    f"Executing sub-task '{sub_task.name}' with {sub_task.agent_type} agent"
                )

                try:
                    agent = self._select_agent(sub_task)
                    agent_id = f"{sub_task.agent_type}_{sub_task.id}"
                    self.active_agents[agent_id] = agent

                    # Inject dependency results into the sub-task description
                    enriched_description = self._enrich_with_dependencies(
                        sub_task, plan
                    )

                    # Execute the agent in a thread to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, agent.process_query, enriched_description
                        ),
                        timeout=self.agent_timeout,
                    )

                    sub_task.result = result
                    sub_task.status = SubTaskStatus.COMPLETED

                    # Publish result to shared state if enabled
                    if self.state_manager:
                        self.state_manager.write(
                            key=sub_task.name,
                            value=result,
                            author=agent_id,
                        )

                    # Broadcast completion via message bus
                    if self.message_bus:
                        self.message_bus.publish(
                            topic=f"task.{sub_task.name}.completed",
                            payload={"task_id": sub_task.id, "result": result},
                            sender=agent_id,
                        )

                except asyncio.TimeoutError:
                    sub_task.status = SubTaskStatus.FAILED
                    sub_task.error = f"Agent timed out after {self.agent_timeout}s"
                except Exception as e:
                    if sub_task.retries > 0:
                        sub_task.retries -= 1
                        sub_task.status = SubTaskStatus.RETRYING
                        self.console.print_warning(
                            f"Sub-task '{sub_task.name}' failed: {e}. "
                            f"Retrying ({sub_task.retries} attempts left)..."
                        )
                        await run_sub_task(sub_task)
                    else:
                        sub_task.status = SubTaskStatus.FAILED
                        sub_task.error = str(e)
                        self.console.print_error(
                            f"Sub-task '{sub_task.name}' failed permanently: {e}"
                        )
                finally:
                    agent_id = f"{sub_task.agent_type}_{sub_task.id}"
                    self.active_agents.pop(agent_id, None)

        # Dispatch tasks in waves based on dependencies
        self.execution_state = self.STATE_MONITORING
        while not plan.is_complete():
            ready_tasks = plan.get_ready_tasks()
            if not ready_tasks:
                # Check for deadlock: no ready tasks but plan not complete
                pending = [
                    t for t in plan.sub_tasks
                    if t.status in (SubTaskStatus.PENDING, SubTaskStatus.ASSIGNED)
                ]
                if pending:
                    # Deadlock detected: cancel remaining tasks
                    self.console.print_error(
                        f"Deadlock detected: {len(pending)} tasks have unresolvable dependencies"
                    )
                    for t in pending:
                        t.status = SubTaskStatus.CANCELLED
                        t.error = "Cancelled due to dependency deadlock"
                break

            # Execute ready tasks concurrently
            await asyncio.gather(
                *[run_sub_task(task) for task in ready_tasks]
            )

        return plan.get_results()

    def _enrich_with_dependencies(
        self, sub_task: SubTask, plan: OrchestrationPlan
    ) -> str:
        """
        Inject dependency results into a sub-task's description.

        When a sub-task depends on other sub-tasks, their results are
        appended to the description so the agent has full context.

        Args:
            sub_task: The sub-task to enrich.
            plan: The full plan (to look up dependency results).

        Returns:
            Enriched description string with dependency context.
        """
        description = sub_task.description

        if not sub_task.dependencies:
            return description

        dependency_context = []
        for dep_id in sub_task.dependencies:
            dep_task = next(
                (t for t in plan.sub_tasks if t.id == dep_id), None
            )
            if dep_task and dep_task.result:
                result_str = str(dep_task.result)
                # Truncate very long results to avoid context window overflow
                if len(result_str) > 5000:
                    result_str = result_str[:5000] + "\n... (truncated)"
                dependency_context.append(
                    f"--- Result from '{dep_task.name}' ---\n{result_str}"
                )

        if dependency_context:
            context_block = "\n\n".join(dependency_context)
            description = (
                f"{description}\n\n"
                f"CONTEXT FROM PREVIOUS STEPS:\n{context_block}"
            )

        return description

    # -----------------------------------------------------------------
    # Result Aggregation
    # -----------------------------------------------------------------

    def _aggregate_results(
        self, plan: OrchestrationPlan
    ) -> str:
        """
        Use the LLM to synthesize results from all sub-tasks into a
        unified response.

        Args:
            plan: The completed orchestration plan.

        Returns:
            Aggregated response string.
        """
        self.execution_state = self.STATE_AGGREGATING
        self.console.print_state_info("Aggregating results from all sub-tasks...")

        results = plan.get_results()
        failed_tasks = [
            t for t in plan.sub_tasks if t.status == SubTaskStatus.FAILED
        ]

        # Build the aggregation prompt
        results_text = ""
        for name, result in results.items():
            result_str = str(result)
            if len(result_str) > 3000:
                result_str = result_str[:3000] + "\n... (truncated)"
            results_text += f"\n### {name}\n{result_str}\n"

        failures_text = ""
        if failed_tasks:
            failures_text = "\n\nFAILED SUB-TASKS:\n"
            for t in failed_tasks:
                failures_text += f"  - {t.name}: {t.error}\n"

        aggregation_prompt = f"""Synthesize the following sub-task results into a unified response.

Original request: "{plan.query}"

SUB-TASK RESULTS:
{results_text}
{failures_text}

Instructions:
1. Combine all successful results into a coherent response.
2. If any sub-tasks failed, acknowledge the failures and explain what was completed.
3. If results conflict with each other, resolve the conflict and explain your reasoning.
4. Structure the response clearly with sections for each major component.
"""

        response = self.chat.send_message(
            aggregation_prompt,
            system_prompt=self.system_prompt,
        )

        return response

    # -----------------------------------------------------------------
    # Conflict Resolution
    # -----------------------------------------------------------------

    def _resolve_conflicts(
        self, plan: OrchestrationPlan
    ) -> Dict[str, Any]:
        """
        Detect and resolve conflicts between sub-task results.

        Conflicts arise when:
        - Two agents produce contradictory outputs for overlapping domains
        - An agent's output violates assumptions made by another agent
        - File content conflicts (e.g., two agents wrote different versions)

        Args:
            plan: The completed orchestration plan.

        Returns:
            Dictionary mapping conflict descriptions to resolutions.
        """
        self.execution_state = self.STATE_RESOLVING_CONFLICTS
        results = plan.get_results()

        if len(results) < 2:
            return {}

        # Ask the LLM to identify conflicts
        results_summary = "\n".join(
            f"- {name}: {str(result)[:500]}" for name, result in results.items()
        )

        conflict_prompt = f"""Analyze these sub-task results for conflicts or inconsistencies.

Results:
{results_summary}

For each conflict found, provide:
1. What the conflict is
2. Which sub-tasks are involved
3. Your recommended resolution
4. Confidence level (high/medium/low)

If no conflicts exist, respond with: {{"conflicts": []}}
"""

        response = self.chat.send_message(
            conflict_prompt,
            system_prompt=self.system_prompt,
        )

        return self._parse_conflicts(response)

    def _parse_conflicts(self, response: str) -> Dict[str, Any]:
        """Parse conflict detection response from LLM."""
        import json
        import re

        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {"conflicts": []}

    # -----------------------------------------------------------------
    # Main Entry Point
    # -----------------------------------------------------------------

    def process_query(self, query: str, **kwargs) -> str:
        """
        Process a complex query using multi-agent orchestration.

        This is the main entry point. It:
        1. Decomposes the query into sub-tasks
        2. Executes sub-tasks via specialist agents
        3. Resolves any conflicts
        4. Aggregates results into a unified response

        Args:
            query: The user's complex query.
            **kwargs: Additional arguments (workspace_root, etc.).

        Returns:
            Unified response string.
        """
        self._current_query = query
        self.console.print_processing_start(
            query, self.max_steps, "gaia-orchestrator"
        )

        # Step 1: Decompose
        plan = self._decompose_task(query)
        self.console.print_plan(
            [f"[{t.agent_type}] {t.name}: {t.description[:80]}..." for t in plan.sub_tasks]
        )

        # Step 2: Execute
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(self._execute_plan(plan))
        finally:
            loop.close()

        # Step 3: Resolve conflicts
        conflicts = self._resolve_conflicts(plan)
        if conflicts.get("conflicts"):
            self.console.print_warning(
                f"Detected {len(conflicts['conflicts'])} conflicts. Resolving..."
            )

        # Step 4: Aggregate
        final_response = self._aggregate_results(plan)

        self.console.print_final_answer(final_response)
        self.console.print_completion(
            len([t for t in plan.sub_tasks if t.status == SubTaskStatus.COMPLETED]),
            len(plan.sub_tasks),
        )

        return final_response
```

### 4.2 Task Decomposition Strategy

The orchestrator uses a structured prompting approach to decompose tasks. The LLM is guided to produce sub-tasks that are:

1. **Atomic**: Each sub-task can be completed by a single agent in one `process_query` call.
2. **Independent where possible**: Sub-tasks without data dependencies can run in parallel.
3. **Typed**: Each sub-task specifies which agent type should handle it.
4. **Ordered**: Dependencies form a directed acyclic graph (DAG).

**Decomposition example for "Build a full-stack app with tests and deployment":**

```json
{
    "thought": "This requires code generation, testing, and Docker deployment. The backend and frontend can be built in parallel, tests depend on code, Docker depends on both code and tests.",
    "plan": {
        "pattern": "hierarchical",
        "sub_tasks": [
            {
                "name": "backend_api",
                "description": "Create a REST API with Express.js including user authentication endpoints (register, login, profile) and a PostgreSQL database schema.",
                "agent_type": "code",
                "dependencies": [],
                "priority": 0
            },
            {
                "name": "frontend_ui",
                "description": "Create a Next.js frontend with login form, registration form, and user dashboard. Use the API endpoints: POST /api/auth/register, POST /api/auth/login, GET /api/auth/profile.",
                "agent_type": "code",
                "dependencies": [],
                "priority": 0
            },
            {
                "name": "test_suite",
                "description": "Write unit tests and integration tests for the REST API and React components.",
                "agent_type": "code",
                "dependencies": ["backend_api", "frontend_ui"],
                "priority": 1
            },
            {
                "name": "docker_config",
                "description": "Create Dockerfile for the Node.js application and docker-compose.yml with PostgreSQL service, app service, and nginx reverse proxy.",
                "agent_type": "docker",
                "dependencies": ["backend_api", "frontend_ui"],
                "priority": 1
            },
            {
                "name": "project_tracking",
                "description": "Create Jira issues for: backend API implementation, frontend UI implementation, test coverage, Docker deployment, and documentation.",
                "agent_type": "jira",
                "dependencies": [],
                "priority": 2
            }
        ]
    }
}
```

### 4.3 Agent Selection and Routing

The orchestrator uses a capability-based routing strategy. Each registered agent declares its capabilities, and the orchestrator matches sub-tasks to agents based on:

1. **Agent type match**: Direct match from sub-task `agent_type` field
2. **Capability match**: For ambiguous tasks, match required capabilities to agent declarations
3. **Load balancing**: If multiple instances of an agent type exist, select the least loaded
4. **Fallback**: If no specialist matches, use `ChatAgent` as a general-purpose fallback

### 4.4 Execution DAG Visualization

The plan forms a DAG that the orchestrator traverses in topological order:

```
    backend_api ----+----> test_suite ----> [aggregation]
                    |
    frontend_ui ----+----> docker_config -> [aggregation]
                    |
    project_tracking ----------------------> [aggregation]

    Wave 0: backend_api, frontend_ui, project_tracking (parallel)
    Wave 1: test_suite, docker_config (parallel, after wave 0)
    Wave 2: aggregation (after all complete)
```

---

## 5. Message Passing Protocol

The message passing protocol enables inter-agent communication. It supports four message types and two communication patterns, all operating in-process using async queues (no external dependencies like Redis or RabbitMQ required for single-machine deployments).

### 5.1 Message Types

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Message types for inter-agent communication.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(Enum):
    """Types of messages agents can exchange."""
    REQUEST = "request"           # Agent asks another agent to do something
    RESPONSE = "response"         # Reply to a request
    BROADCAST = "broadcast"       # One-to-many notification
    NOTIFICATION = "notification" # Fire-and-forget status update


class MessagePriority(Enum):
    """Priority levels for message processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class DeliveryGuarantee(Enum):
    """Message delivery semantics."""
    AT_MOST_ONCE = "at_most_once"   # Fire and forget
    AT_LEAST_ONCE = "at_least_once" # Retry until acknowledged
    EXACTLY_ONCE = "exactly_once"   # Deduplicate on receiver


@dataclass
class Message:
    """
    A typed message exchanged between agents.

    Attributes:
        id: Unique message identifier.
        type: Message type (request, response, broadcast, notification).
        sender: ID of the sending agent.
        recipient: ID of the receiving agent (None for broadcasts).
        topic: Topic string for pub/sub routing.
        payload: The message content (must be JSON-serializable).
        correlation_id: ID linking a response to its original request.
        priority: Processing priority.
        timestamp: Creation time (Unix epoch seconds).
        ttl: Time-to-live in seconds (0 = no expiry).
        delivery: Delivery guarantee level.
        headers: Optional metadata headers.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.NOTIFICATION
    sender: str = ""
    recipient: Optional[str] = None
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    ttl: int = 0
    delivery: DeliveryGuarantee = DeliveryGuarantee.AT_MOST_ONCE
    headers: Dict[str, str] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the message has exceeded its TTL."""
        if self.ttl == 0:
            return False
        return (time.time() - self.timestamp) > self.ttl

    def create_reply(self, payload: Dict[str, Any], sender: str) -> "Message":
        """
        Create a response message linked to this request.

        Args:
            payload: Response content.
            sender: ID of the agent sending the reply.

        Returns:
            New Message with type RESPONSE and correlation_id set.
        """
        return Message(
            type=MessageType.RESPONSE,
            sender=sender,
            recipient=self.sender,
            topic=self.topic,
            payload=payload,
            correlation_id=self.id,
            priority=self.priority,
        )
```

### 5.2 Message Bus Implementation

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
In-process message bus for inter-agent communication.
"""

import asyncio
import fnmatch
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from gaia.agents.orchestration.messages import (
    DeliveryGuarantee,
    Message,
    MessagePriority,
    MessageType,
)

logger = logging.getLogger(__name__)

# Type alias for message handlers
MessageHandler = Callable[[Message], Awaitable[None]]


class Subscription:
    """
    A topic subscription with pattern matching.

    Attributes:
        subscriber_id: ID of the subscribing agent.
        topic_pattern: Glob pattern for topic matching (e.g., "task.*", "error.**").
        handler: Async callback invoked when a matching message arrives.
        filter_fn: Optional predicate to further filter messages.
    """

    def __init__(
        self,
        subscriber_id: str,
        topic_pattern: str,
        handler: MessageHandler,
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ):
        self.subscriber_id = subscriber_id
        self.topic_pattern = topic_pattern
        self.handler = handler
        self.filter_fn = filter_fn

    def matches(self, topic: str) -> bool:
        """Check if a topic matches this subscription's pattern."""
        return fnmatch.fnmatch(topic, self.topic_pattern)


class MessageBus:
    """
    In-process async message bus supporting pub/sub and request/reply.

    The MessageBus provides:
    - Topic-based publish/subscribe with glob pattern matching
    - Point-to-point request/reply with correlation tracking
    - Priority-based message ordering
    - TTL-based message expiration
    - Dead letter queue for undeliverable messages
    - Message history for debugging

    Usage:
        bus = MessageBus()

        # Subscribe to a topic
        async def handler(msg: Message):
            print(f"Received: {msg.payload}")

        bus.subscribe("agent_1", "task.completed.*", handler)

        # Publish a message
        bus.publish(
            topic="task.completed.backend",
            payload={"status": "done", "files": ["api.py"]},
            sender="code_agent_1",
        )

        # Request/reply
        response = await bus.request(
            topic="agent.code.generate",
            payload={"prompt": "Create a REST endpoint"},
            sender="orchestrator",
            recipient="code_agent_1",
            timeout=30.0,
        )
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the message bus.

        Args:
            max_history: Maximum number of messages to retain in history.
        """
        self._subscriptions: List[Subscription] = []
        self._queues: Dict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=1000)
        )
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._dead_letters: List[Message] = []
        self._history: List[Message] = []
        self._max_history = max_history
        self._seen_ids: Set[str] = set()  # For exactly-once deduplication
        self._running = False

    # -----------------------------------------------------------------
    # Pub/Sub Pattern
    # -----------------------------------------------------------------

    def subscribe(
        self,
        subscriber_id: str,
        topic_pattern: str,
        handler: MessageHandler,
        filter_fn: Optional[Callable[[Message], bool]] = None,
    ) -> str:
        """
        Subscribe to messages matching a topic pattern.

        Args:
            subscriber_id: Unique ID of the subscribing agent.
            topic_pattern: Glob pattern (e.g., "task.*", "error.**").
            handler: Async callback for matching messages.
            filter_fn: Optional additional filter predicate.

        Returns:
            Subscription ID for later unsubscription.
        """
        sub = Subscription(subscriber_id, topic_pattern, handler, filter_fn)
        self._subscriptions.append(sub)
        logger.debug(
            f"Agent '{subscriber_id}' subscribed to '{topic_pattern}'"
        )
        return f"{subscriber_id}:{topic_pattern}"

    def unsubscribe(self, subscriber_id: str, topic_pattern: str = None):
        """
        Remove subscriptions for an agent.

        Args:
            subscriber_id: ID of the agent to unsubscribe.
            topic_pattern: Specific pattern to unsubscribe (None = all).
        """
        if topic_pattern:
            self._subscriptions = [
                s for s in self._subscriptions
                if not (
                    s.subscriber_id == subscriber_id
                    and s.topic_pattern == topic_pattern
                )
            ]
        else:
            self._subscriptions = [
                s for s in self._subscriptions
                if s.subscriber_id != subscriber_id
            ]

    async def publish(
        self,
        topic: str,
        payload: Dict[str, Any],
        sender: str,
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: int = 0,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Publish a message to all matching subscribers.

        Args:
            topic: Message topic string.
            payload: Message content.
            sender: ID of the sending agent.
            priority: Message priority.
            ttl: Time-to-live in seconds (0 = no expiry).
            headers: Optional metadata headers.

        Returns:
            Message ID.
        """
        message = Message(
            type=MessageType.BROADCAST,
            sender=sender,
            topic=topic,
            payload=payload,
            priority=priority,
            ttl=ttl,
            headers=headers or {},
        )

        self._record_history(message)

        # Find matching subscriptions
        matching = [s for s in self._subscriptions if s.matches(topic)]

        if not matching:
            logger.debug(f"No subscribers for topic '{topic}'")
            self._dead_letters.append(message)
            return message.id

        # Deliver to all matching subscribers
        delivery_tasks = []
        for sub in matching:
            if sub.filter_fn and not sub.filter_fn(message):
                continue
            delivery_tasks.append(self._deliver(message, sub))

        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)

        return message.id

    async def _deliver(self, message: Message, subscription: Subscription):
        """Deliver a message to a single subscriber."""
        # Check exactly-once semantics
        if message.delivery == DeliveryGuarantee.EXACTLY_ONCE:
            dedup_key = f"{subscription.subscriber_id}:{message.id}"
            if dedup_key in self._seen_ids:
                return
            self._seen_ids.add(dedup_key)

        # Check TTL
        if message.is_expired():
            logger.debug(f"Message {message.id} expired, not delivering")
            return

        try:
            await subscription.handler(message)
        except Exception as e:
            logger.error(
                f"Error delivering message {message.id} to "
                f"{subscription.subscriber_id}: {e}"
            )
            if message.delivery == DeliveryGuarantee.AT_LEAST_ONCE:
                # Re-queue for retry
                self._dead_letters.append(message)

    # -----------------------------------------------------------------
    # Request/Reply Pattern
    # -----------------------------------------------------------------

    async def request(
        self,
        topic: str,
        payload: Dict[str, Any],
        sender: str,
        recipient: str,
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
    ) -> Message:
        """
        Send a request and wait for a reply.

        Args:
            topic: Request topic.
            payload: Request content.
            sender: ID of the requesting agent.
            recipient: ID of the target agent.
            timeout: Maximum seconds to wait for reply.
            priority: Request priority.

        Returns:
            Response Message from the recipient.

        Raises:
            asyncio.TimeoutError: If no reply within timeout.
        """
        request_msg = Message(
            type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            topic=topic,
            payload=payload,
            priority=priority,
        )

        self._record_history(request_msg)

        # Create a future for the response
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self._pending_requests[request_msg.id] = future

        # Deliver the request to the recipient's queue
        await self._queues[recipient].put(request_msg)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self._pending_requests.pop(request_msg.id, None)
            raise
        finally:
            self._pending_requests.pop(request_msg.id, None)

    async def reply(self, original_message: Message, payload: Dict[str, Any], sender: str):
        """
        Send a reply to a request message.

        Args:
            original_message: The request message being replied to.
            payload: Response content.
            sender: ID of the replying agent.
        """
        response = original_message.create_reply(payload, sender)
        self._record_history(response)

        # Resolve the pending future if it exists
        future = self._pending_requests.get(original_message.id)
        if future and not future.done():
            future.set_result(response)
        else:
            # No pending request; deliver via queue
            if original_message.sender:
                await self._queues[original_message.sender].put(response)

    # -----------------------------------------------------------------
    # Queue Management
    # -----------------------------------------------------------------

    async def get_messages(
        self, agent_id: str, timeout: float = 1.0
    ) -> List[Message]:
        """
        Get all pending messages for an agent.

        Args:
            agent_id: ID of the agent.
            timeout: Maximum seconds to wait for at least one message.

        Returns:
            List of pending messages (may be empty on timeout).
        """
        messages = []
        queue = self._queues[agent_id]

        try:
            # Wait for at least one message
            msg = await asyncio.wait_for(queue.get(), timeout=timeout)
            messages.append(msg)

            # Drain any additional messages without waiting
            while not queue.empty():
                messages.append(queue.get_nowait())
        except asyncio.TimeoutError:
            pass

        # Sort by priority (highest first)
        messages.sort(key=lambda m: m.priority.value, reverse=True)
        return messages

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def _record_history(self, message: Message):
        """Record a message in the history buffer."""
        self._history.append(message)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get message bus statistics.

        Returns:
            Dictionary with subscriber counts, queue depths, and history size.
        """
        return {
            "total_subscriptions": len(self._subscriptions),
            "active_queues": len(self._queues),
            "queue_depths": {
                agent_id: q.qsize()
                for agent_id, q in self._queues.items()
            },
            "pending_requests": len(self._pending_requests),
            "dead_letters": len(self._dead_letters),
            "history_size": len(self._history),
        }

    def get_dead_letters(self) -> List[Message]:
        """Return messages that could not be delivered."""
        return list(self._dead_letters)

    def clear_dead_letters(self):
        """Clear the dead letter queue."""
        self._dead_letters.clear()
```

### 5.3 Communication Patterns Reference

| Pattern | Message Type | Sender | Recipient | Use Case |
|---------|-------------|--------|-----------|----------|
| **Publish/Subscribe** | BROADCAST | Any agent | All subscribers matching topic | Status updates, event notifications |
| **Request/Reply** | REQUEST + RESPONSE | Requesting agent | Specific agent | Asking an agent to perform work |
| **Notification** | NOTIFICATION | Any agent | Specific agent or broadcast | Non-blocking status updates |
| **Fan-out** | BROADCAST | Orchestrator | All agents on a topic | Distributing work to multiple agents |

### 5.4 Topic Naming Convention

Topics follow a hierarchical dot-separated naming scheme with glob pattern matching:

```
<domain>.<entity>.<action>

Examples:
  task.backend_api.completed     # Specific task completed
  task.*.completed               # Any task completed
  task.**                        # All task-related messages
  agent.code.status              # Code agent status update
  error.code_agent_1.timeout     # Specific error from specific agent
  orchestrator.plan.created      # Plan lifecycle event
```

**Reserved topic prefixes:**

| Prefix | Purpose | Publisher |
|--------|---------|-----------|
| `orchestrator.*` | Orchestration lifecycle events | OrchestratorAgent |
| `task.*` | Sub-task status changes | Any agent executing a sub-task |
| `agent.*` | Agent lifecycle events | Any agent |
| `error.*` | Error notifications | Any agent |
| `state.*` | Shared state changes | SharedStateManager |

### 5.5 Message Flow Examples

**Example 1: Hierarchical task execution**

```
OrchestratorAgent                     CodeAgent                  DockerAgent
      |                                   |                          |
      |--- publish("task.backend.assigned", {...}) -->                |
      |                                   |                          |
      |                [executes task]     |                          |
      |                                   |                          |
      |<-- publish("task.backend.completed", {result}) ---|          |
      |                                   |                          |
      |--- publish("task.docker.assigned", {result_context}) ------->|
      |                                   |                          |
      |                                   |          [executes task] |
      |                                   |                          |
      |<-- publish("task.docker.completed", {result}) --------------|
      |                                   |                          |
      | [aggregates results]              |                          |
```

**Example 2: Peer-to-peer code review loop**

```
CodeAgent                              TestAgent
    |                                      |
    |--- request("agent.test.validate",    |
    |        {code: "..."})                |
    |                                      |
    |                          [runs tests]|
    |                                      |
    |<-- reply({passed: false,             |
    |          failures: [...]})           |
    |                                      |
    | [fixes code]                         |
    |                                      |
    |--- request("agent.test.validate",    |
    |        {code: "...(fixed)"})         |
    |                                      |
    |<-- reply({passed: true})             |
    |                                      |
```

---

## 6. Shared State Management

The shared state system provides a centralized, versioned data store that multiple agents can read from and write to. It implements a blackboard architecture with conflict detection and optional conflict-free replicated data types (CRDTs) for concurrent writes.

### 6.1 State Entry Model

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Shared state management for multi-agent orchestration.
"""

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class ConflictResolution(Enum):
    """Strategies for resolving write conflicts."""
    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MERGE = "merge"
    REJECT = "reject"
    MANUAL = "manual"


@dataclass
class StateEntry:
    """
    A versioned entry in the shared state.

    Attributes:
        key: Unique identifier for this state entry.
        value: The current value.
        version: Monotonically increasing version number.
        author: ID of the agent that last wrote this entry.
        timestamp: Time of last write (Unix epoch seconds).
        history: List of (version, value, author, timestamp) tuples.
        locked_by: Agent ID if this entry is currently locked (None = unlocked).
        lock_expiry: Timestamp when the lock expires.
        tags: Set of tags for categorization and querying.
    """
    key: str = ""
    value: Any = None
    version: int = 0
    author: str = ""
    timestamp: float = 0.0
    history: List[Tuple[int, Any, str, float]] = field(default_factory=list)
    locked_by: Optional[str] = None
    lock_expiry: float = 0.0
    tags: Set[str] = field(default_factory=set)

    def is_locked(self) -> bool:
        """Check if this entry is currently locked."""
        if self.locked_by is None:
            return False
        if time.time() > self.lock_expiry:
            # Lock expired, auto-release
            self.locked_by = None
            return False
        return True
```

### 6.2 SharedStateManager Implementation

```python
class SharedStateManager:
    """
    Thread-safe shared state manager with versioning and conflict detection.

    Provides a blackboard-style shared workspace where agents can:
    - Read and write key-value pairs with automatic versioning
    - Lock entries for exclusive access during multi-step operations
    - Subscribe to state changes via callbacks
    - Query entries by tags
    - Resolve write conflicts using configurable strategies

    Thread safety is ensured via a reentrant lock, making this safe
    for use with concurrent agent execution.

    Usage:
        state = SharedStateManager()

        # Write a value
        state.write("api_schema", {"endpoints": ["/users"]}, author="code_agent")

        # Read a value
        schema = state.read("api_schema")

        # Subscribe to changes
        def on_schema_change(key, value, author):
            print(f"Schema updated by {author}")
        state.subscribe("api_schema", on_schema_change)

        # Lock for exclusive access
        with state.lock("api_schema", agent_id="code_agent"):
            schema = state.read("api_schema")
            schema["endpoints"].append("/products")
            state.write("api_schema", schema, author="code_agent")
    """

    def __init__(
        self,
        conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITER_WINS,
        max_history: int = 50,
    ):
        """
        Initialize the shared state manager.

        Args:
            conflict_resolution: Default strategy for resolving write conflicts.
            max_history: Maximum number of historical versions to retain per entry.
        """
        self._entries: Dict[str, StateEntry] = {}
        self._lock = threading.RLock()
        self._subscribers: Dict[str, List[Callable]] = {}
        self._conflict_resolution = conflict_resolution
        self._max_history = max_history
        self._change_log: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Core Read/Write Operations
    # -----------------------------------------------------------------

    def read(self, key: str, default: Any = None) -> Any:
        """
        Read a value from shared state.

        Args:
            key: The state key to read.
            default: Default value if key does not exist.

        Returns:
            The current value (deep-copied to prevent external mutation).
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return default
            # Return a deep copy to prevent external mutation
            return copy.deepcopy(entry.value)

    def read_with_version(self, key: str) -> Tuple[Any, int]:
        """
        Read a value along with its version number.

        Useful for optimistic concurrency control (compare-and-swap).

        Args:
            key: The state key to read.

        Returns:
            Tuple of (value, version). Returns (None, 0) if key does not exist.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return (None, 0)
            return (copy.deepcopy(entry.value), entry.version)

    def write(
        self,
        key: str,
        value: Any,
        author: str,
        expected_version: Optional[int] = None,
        tags: Optional[Set[str]] = None,
    ) -> int:
        """
        Write a value to shared state.

        Args:
            key: The state key to write.
            value: The new value.
            author: ID of the writing agent.
            expected_version: If set, only write if current version matches
                (optimistic concurrency control). Set to None to skip check.
            tags: Optional tags for this entry.

        Returns:
            The new version number.

        Raises:
            ConflictError: If expected_version does not match current version.
            LockError: If the entry is locked by another agent.
        """
        with self._lock:
            entry = self._entries.get(key)

            if entry is None:
                # New entry
                entry = StateEntry(
                    key=key,
                    value=copy.deepcopy(value),
                    version=1,
                    author=author,
                    timestamp=time.time(),
                    tags=tags or set(),
                )
                self._entries[key] = entry
                self._notify_subscribers(key, value, author)
                self._log_change("create", key, author, 1)
                return 1

            # Check lock
            if entry.is_locked() and entry.locked_by != author:
                raise LockError(
                    f"Entry '{key}' is locked by '{entry.locked_by}' "
                    f"(expires at {entry.lock_expiry})"
                )

            # Optimistic concurrency check
            if expected_version is not None and entry.version != expected_version:
                resolution = self._handle_conflict(
                    entry, value, author, expected_version
                )
                if resolution == "rejected":
                    raise ConflictError(
                        f"Version conflict on '{key}': "
                        f"expected {expected_version}, actual {entry.version}"
                    )

            # Save history
            entry.history.append(
                (entry.version, copy.deepcopy(entry.value), entry.author, entry.timestamp)
            )
            if len(entry.history) > self._max_history:
                entry.history = entry.history[-self._max_history:]

            # Update entry
            entry.value = copy.deepcopy(value)
            entry.version += 1
            entry.author = author
            entry.timestamp = time.time()
            if tags:
                entry.tags.update(tags)

            self._notify_subscribers(key, value, author)
            self._log_change("update", key, author, entry.version)
            return entry.version

    def delete(self, key: str, author: str) -> bool:
        """
        Delete an entry from shared state.

        Args:
            key: The state key to delete.
            author: ID of the deleting agent.

        Returns:
            True if the entry was deleted, False if it did not exist.

        Raises:
            LockError: If the entry is locked by another agent.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False

            if entry.is_locked() and entry.locked_by != author:
                raise LockError(
                    f"Entry '{key}' is locked by '{entry.locked_by}'"
                )

            del self._entries[key]
            self._notify_subscribers(key, None, author)
            self._log_change("delete", key, author, 0)
            return True

    # -----------------------------------------------------------------
    # Locking
    # -----------------------------------------------------------------

    def acquire_lock(
        self, key: str, agent_id: str, timeout: float = 30.0
    ) -> bool:
        """
        Acquire an exclusive lock on a state entry.

        Args:
            key: The state key to lock.
            agent_id: ID of the locking agent.
            timeout: Lock duration in seconds.

        Returns:
            True if lock acquired, False if already locked by another agent.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                # Create a placeholder entry for locking
                entry = StateEntry(key=key)
                self._entries[key] = entry

            if entry.is_locked() and entry.locked_by != agent_id:
                return False

            entry.locked_by = agent_id
            entry.lock_expiry = time.time() + timeout
            return True

    def release_lock(self, key: str, agent_id: str) -> bool:
        """
        Release a lock on a state entry.

        Args:
            key: The state key to unlock.
            agent_id: ID of the agent releasing the lock.

        Returns:
            True if lock released, False if not locked or locked by another agent.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return False

            if entry.locked_by != agent_id:
                return False

            entry.locked_by = None
            entry.lock_expiry = 0.0
            return True

    class _LockContext:
        """Context manager for state entry locks."""

        def __init__(self, manager: "SharedStateManager", key: str, agent_id: str, timeout: float):
            self._manager = manager
            self._key = key
            self._agent_id = agent_id
            self._timeout = timeout

        def __enter__(self):
            if not self._manager.acquire_lock(self._key, self._agent_id, self._timeout):
                raise LockError(
                    f"Could not acquire lock on '{self._key}' for '{self._agent_id}'"
                )
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._manager.release_lock(self._key, self._agent_id)
            return False

    def lock(self, key: str, agent_id: str, timeout: float = 30.0) -> _LockContext:
        """
        Create a context manager for locking a state entry.

        Args:
            key: The state key to lock.
            agent_id: ID of the locking agent.
            timeout: Lock duration in seconds.

        Returns:
            Context manager that acquires lock on enter and releases on exit.
        """
        return self._LockContext(self, key, agent_id, timeout)

    # -----------------------------------------------------------------
    # Subscriptions
    # -----------------------------------------------------------------

    def subscribe(
        self, key_pattern: str, callback: Callable[[str, Any, str], None]
    ):
        """
        Subscribe to state changes matching a key pattern.

        Args:
            key_pattern: Glob pattern for key matching (e.g., "api_*", "*").
            callback: Function called with (key, new_value, author) on change.
        """
        if key_pattern not in self._subscribers:
            self._subscribers[key_pattern] = []
        self._subscribers[key_pattern].append(callback)

    def _notify_subscribers(self, key: str, value: Any, author: str):
        """Notify subscribers of a state change."""
        import fnmatch

        for pattern, callbacks in self._subscribers.items():
            if fnmatch.fnmatch(key, pattern):
                for callback in callbacks:
                    try:
                        callback(key, value, author)
                    except Exception as e:
                        logger.error(f"Subscriber callback error: {e}")

    # -----------------------------------------------------------------
    # Querying
    # -----------------------------------------------------------------

    def keys(self) -> List[str]:
        """Return all keys in shared state."""
        with self._lock:
            return list(self._entries.keys())

    def query_by_tags(self, tags: Set[str]) -> Dict[str, Any]:
        """
        Query entries that have all specified tags.

        Args:
            tags: Set of required tags.

        Returns:
            Dictionary of matching key-value pairs.
        """
        with self._lock:
            return {
                key: copy.deepcopy(entry.value)
                for key, entry in self._entries.items()
                if tags.issubset(entry.tags)
            }

    def query_by_author(self, author: str) -> Dict[str, Any]:
        """
        Query entries written by a specific agent.

        Args:
            author: Agent ID to filter by.

        Returns:
            Dictionary of matching key-value pairs.
        """
        with self._lock:
            return {
                key: copy.deepcopy(entry.value)
                for key, entry in self._entries.items()
                if entry.author == author
            }

    def get_history(self, key: str) -> List[Tuple[int, Any, str, float]]:
        """
        Get version history for a state entry.

        Args:
            key: The state key.

        Returns:
            List of (version, value, author, timestamp) tuples.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return []
            return list(entry.history)

    # -----------------------------------------------------------------
    # Conflict Handling
    # -----------------------------------------------------------------

    def _handle_conflict(
        self,
        entry: StateEntry,
        new_value: Any,
        author: str,
        expected_version: int,
    ) -> str:
        """
        Handle a write conflict based on the configured resolution strategy.

        Args:
            entry: The existing state entry.
            new_value: The value the writer wants to set.
            author: ID of the writing agent.
            expected_version: The version the writer expected.

        Returns:
            "accepted" if the write should proceed, "rejected" if not.
        """
        if self._conflict_resolution == ConflictResolution.LAST_WRITER_WINS:
            return "accepted"
        elif self._conflict_resolution == ConflictResolution.FIRST_WRITER_WINS:
            return "rejected"
        elif self._conflict_resolution == ConflictResolution.REJECT:
            return "rejected"
        elif self._conflict_resolution == ConflictResolution.MERGE:
            # Attempt automatic merge for dict types
            if isinstance(entry.value, dict) and isinstance(new_value, dict):
                merged = {**entry.value, **new_value}
                entry.value = merged
                return "accepted"
            return "rejected"
        else:
            return "rejected"

    # -----------------------------------------------------------------
    # Change Log
    # -----------------------------------------------------------------

    def _log_change(self, operation: str, key: str, author: str, version: int):
        """Record a state change in the change log."""
        self._change_log.append({
            "operation": operation,
            "key": key,
            "author": author,
            "version": version,
            "timestamp": time.time(),
        })

    def get_change_log(
        self, since: Optional[float] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent state changes.

        Args:
            since: Only return changes after this timestamp (Unix epoch).
            limit: Maximum number of changes to return.

        Returns:
            List of change log entries, newest first.
        """
        log = self._change_log
        if since is not None:
            log = [entry for entry in log if entry["timestamp"] > since]
        return log[-limit:]

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get a complete snapshot of the current state.

        Returns:
            Dictionary with all keys, values, versions, and metadata.
        """
        with self._lock:
            return {
                key: {
                    "value": copy.deepcopy(entry.value),
                    "version": entry.version,
                    "author": entry.author,
                    "timestamp": entry.timestamp,
                    "tags": list(entry.tags),
                    "locked_by": entry.locked_by,
                }
                for key, entry in self._entries.items()
            }


class ConflictError(Exception):
    """Raised when a write conflict is detected and rejected."""
    pass


class LockError(Exception):
    """Raised when a lock operation fails."""
    pass
```

### 6.3 Conflict-Free Replicated Data Types (CRDTs)

For scenarios where multiple agents must concurrently update the same state entry without coordination, CRDTs provide eventual consistency guarantees.

**Supported CRDT types:**

| CRDT | Use Case | Merge Strategy |
|------|----------|----------------|
| **GCounter** | Counting events across agents | Sum of per-agent counts |
| **GSet** | Accumulating items (files, issues) | Set union |
| **LWWRegister** | Latest value wins (simple overwrites) | Higher timestamp wins |
| **ORSet** | Add/remove items across agents | Observed-remove semantics |

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
CRDT implementations for conflict-free concurrent state updates.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple


@dataclass
class GCounter:
    """
    Grow-only counter. Each agent maintains its own counter.
    The total is the sum of all agent counters.
    """
    _counts: Dict[str, int] = field(default_factory=dict)

    def increment(self, agent_id: str, amount: int = 1):
        """Increment the counter for an agent."""
        self._counts[agent_id] = self._counts.get(agent_id, 0) + amount

    def value(self) -> int:
        """Get the total count across all agents."""
        return sum(self._counts.values())

    def merge(self, other: "GCounter") -> "GCounter":
        """Merge with another GCounter (takes max of each agent's count)."""
        merged = GCounter()
        all_agents = set(self._counts.keys()) | set(other._counts.keys())
        for agent in all_agents:
            merged._counts[agent] = max(
                self._counts.get(agent, 0),
                other._counts.get(agent, 0),
            )
        return merged


@dataclass
class GSet:
    """
    Grow-only set. Items can be added but never removed.
    Merge is set union.
    """
    _items: Set[Any] = field(default_factory=set)

    def add(self, item: Any):
        """Add an item to the set."""
        self._items.add(item)

    def contains(self, item: Any) -> bool:
        """Check if an item is in the set."""
        return item in self._items

    def items(self) -> Set[Any]:
        """Get all items."""
        return set(self._items)

    def merge(self, other: "GSet") -> "GSet":
        """Merge with another GSet (set union)."""
        merged = GSet()
        merged._items = self._items | other._items
        return merged


@dataclass
class LWWRegister:
    """
    Last-Writer-Wins register. Highest timestamp wins on merge.
    """
    value: Any = None
    timestamp: float = 0.0
    author: str = ""

    def set(self, value: Any, timestamp: float, author: str):
        """Set the value if the timestamp is newer."""
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
            self.author = author

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        """Merge with another LWWRegister (highest timestamp wins)."""
        if other.timestamp > self.timestamp:
            return LWWRegister(
                value=other.value,
                timestamp=other.timestamp,
                author=other.author,
            )
        return LWWRegister(
            value=self.value,
            timestamp=self.timestamp,
            author=self.author,
        )
```

---

## 7. Agent Lifecycle

This section defines the complete lifecycle of agents within the orchestration system, from discovery and registration through health monitoring to graceful shutdown.

### 7.1 Agent Registry

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Agent registry for discovering and managing available specialist agents.
"""

import importlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type

from gaia.agents.base.agent import Agent

logger = logging.getLogger(__name__)


class AgentHealth(Enum):
    """Health status of a registered agent."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AgentRegistration:
    """
    Registration record for an available specialist agent.

    Attributes:
        name: Unique name for this agent type.
        agent_class_path: Fully qualified Python class path.
        description: Human-readable description of agent capabilities.
        capabilities: Set of capability tags.
        max_instances: Maximum concurrent instances allowed.
        active_instances: Number of currently active instances.
        health: Current health status.
        last_heartbeat: Timestamp of last health check.
        metadata: Additional registration metadata.
    """
    name: str = ""
    agent_class_path: str = ""
    description: str = ""
    capabilities: Set[str] = field(default_factory=set)
    max_instances: int = 4
    active_instances: int = 0
    health: AgentHealth = AgentHealth.UNKNOWN
    last_heartbeat: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """Check if this agent type can accept new work."""
        return (
            self.active_instances < self.max_instances
            and self.health in (AgentHealth.HEALTHY, AgentHealth.UNKNOWN)
        )


class AgentRegistry:
    """
    Registry for discovering and managing specialist agents.

    Provides:
    - Agent type registration with capability declarations
    - Dynamic agent class loading
    - Instance counting and concurrency limits
    - Health tracking
    - Capability-based agent discovery

    Usage:
        registry = AgentRegistry()

        # Register an agent type
        registry.register(
            name="code",
            agent_class_path="gaia.agents.code.agent.CodeAgent",
            description="Code generation agent",
            capabilities=["code_generation", "refactoring"],
        )

        # Find agents by capability
        agents = registry.find_by_capability("code_generation")

        # Import and instantiate
        agent_class = registry.import_agent_class("code")
        agent = agent_class()
    """

    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        self._class_cache: Dict[str, Type[Agent]] = {}

    def register(
        self,
        name: str,
        agent_class_path: str,
        description: str = "",
        capabilities: Optional[List[str]] = None,
        max_instances: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Register a specialist agent type.

        Args:
            name: Unique name for this agent type.
            agent_class_path: Fully qualified Python class path
                (e.g., "gaia.agents.code.agent.CodeAgent").
            description: Human-readable description.
            capabilities: List of capability tags.
            max_instances: Maximum concurrent instances.
            metadata: Additional metadata.
        """
        registration = AgentRegistration(
            name=name,
            agent_class_path=agent_class_path,
            description=description,
            capabilities=set(capabilities or []),
            max_instances=max_instances,
            metadata=metadata or {},
        )
        self._agents[name] = registration
        logger.info(f"Registered agent '{name}' with capabilities: {capabilities}")

    def unregister(self, name: str):
        """Remove an agent type from the registry."""
        self._agents.pop(name, None)
        self._class_cache.pop(name, None)

    def get(self, name: str) -> Optional[AgentRegistration]:
        """Get registration info for an agent type."""
        return self._agents.get(name)

    def list_agents(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered agent types with their info.

        Returns:
            Dictionary mapping agent names to their info dictionaries.
        """
        return {
            name: {
                "description": reg.description,
                "capabilities": list(reg.capabilities),
                "max_instances": reg.max_instances,
                "active_instances": reg.active_instances,
                "health": reg.health.value,
                "available": reg.is_available(),
            }
            for name, reg in self._agents.items()
        }

    def find_by_capability(self, capability: str) -> List[str]:
        """
        Find agent types that have a specific capability.

        Args:
            capability: Capability tag to search for.

        Returns:
            List of agent names that have the capability.
        """
        return [
            name for name, reg in self._agents.items()
            if capability in reg.capabilities and reg.is_available()
        ]

    def import_agent_class(self, name: str) -> Type[Agent]:
        """
        Dynamically import and return the agent class.

        Args:
            name: Registered agent name.

        Returns:
            The agent class (not an instance).

        Raises:
            ValueError: If the agent name is not registered.
            ImportError: If the class cannot be imported.
        """
        if name in self._class_cache:
            return self._class_cache[name]

        reg = self._agents.get(name)
        if reg is None:
            raise ValueError(f"Agent '{name}' is not registered")

        # Split "gaia.agents.code.agent.CodeAgent" into module + class
        parts = reg.agent_class_path.rsplit(".", 1)
        if len(parts) != 2:
            raise ImportError(
                f"Invalid class path: {reg.agent_class_path}. "
                f"Expected 'module.path.ClassName'."
            )

        module_path, class_name = parts
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        if not issubclass(agent_class, Agent):
            raise TypeError(
                f"{agent_class} is not a subclass of Agent"
            )

        self._class_cache[name] = agent_class
        return agent_class

    # -----------------------------------------------------------------
    # Instance Tracking
    # -----------------------------------------------------------------

    def checkout(self, name: str) -> bool:
        """
        Mark an instance of an agent type as active.

        Args:
            name: Agent type name.

        Returns:
            True if instance was checked out, False if at capacity.
        """
        reg = self._agents.get(name)
        if reg is None or not reg.is_available():
            return False
        reg.active_instances += 1
        return True

    def checkin(self, name: str):
        """
        Mark an instance of an agent type as released.

        Args:
            name: Agent type name.
        """
        reg = self._agents.get(name)
        if reg and reg.active_instances > 0:
            reg.active_instances -= 1

    # -----------------------------------------------------------------
    # Health Monitoring
    # -----------------------------------------------------------------

    def update_health(self, name: str, health: AgentHealth):
        """
        Update the health status of an agent type.

        Args:
            name: Agent type name.
            health: New health status.
        """
        reg = self._agents.get(name)
        if reg:
            reg.health = health
            reg.last_heartbeat = time.time()

    def get_unhealthy_agents(self) -> List[str]:
        """Return names of agents that are unhealthy or have stale heartbeats."""
        stale_threshold = time.time() - 60  # 60 seconds
        return [
            name for name, reg in self._agents.items()
            if reg.health == AgentHealth.UNHEALTHY
            or (
                reg.last_heartbeat > 0
                and reg.last_heartbeat < stale_threshold
            )
        ]
```

### 7.2 Agent Lifecycle States

```
    +-------------+
    |  REGISTERED |  (agent type known to registry)
    +------+------+
           |
           v  instantiate()
    +------+------+
    | INITIALIZING|  (loading model, connecting to LLM)
    +------+------+
           |
           v  ready()
    +------+------+
    |    IDLE     |  (ready to accept work)
    +------+------+
           |
           v  assign(sub_task)
    +------+------+
    |   ACTIVE    |  (executing sub-task via process_query)
    +------+------+
           |
      +----+----+
      |         |
      v         v  error
  +---+---+  +--+------+
  |COMPLETE|  |  ERROR  |  (retryable or fatal)
  +---+---+  +--+------+
      |         |
      v         v  retry or give up
    +------+------+
    |  RELEASING  |  (cleanup, checkin to registry)
    +------+------+
           |
           v
    +------+------+
    |  TERMINATED |  (resources freed)
    +-------------+
```

### 7.3 Health Monitoring

The orchestrator periodically checks the health of active agents:

```python
# Health check configuration
HEALTH_CHECK_INTERVAL = 10  # seconds
HEALTH_CHECK_TIMEOUT = 5    # seconds
STALE_THRESHOLD = 60        # seconds without heartbeat = unhealthy

class HealthMonitor:
    """
    Monitors the health of active agents in the orchestration system.

    Runs as a background task, periodically checking:
    - Agent responsiveness (heartbeat)
    - LLM client connectivity
    - Resource usage (memory, context window)
    - Task progress (stuck detection)
    """

    def __init__(
        self,
        registry: AgentRegistry,
        check_interval: float = HEALTH_CHECK_INTERVAL,
    ):
        self._registry = registry
        self._check_interval = check_interval
        self._active_agents: Dict[str, Agent] = {}
        self._task = None

    async def start(self):
        """Start the health monitoring loop."""
        self._task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the health monitoring loop."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def register_agent(self, agent_id: str, agent: Agent):
        """Register an active agent instance for monitoring."""
        self._active_agents[agent_id] = agent

    def unregister_agent(self, agent_id: str):
        """Unregister an agent instance from monitoring."""
        self._active_agents.pop(agent_id, None)

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            await asyncio.sleep(self._check_interval)

            for agent_id, agent in list(self._active_agents.items()):
                health = self._check_agent_health(agent_id, agent)

                # Extract agent type from ID (format: "type_uuid")
                agent_type = agent_id.split("_")[0]
                self._registry.update_health(agent_type, health)

    def _check_agent_health(self, agent_id: str, agent: Agent) -> AgentHealth:
        """
        Check the health of a single agent.

        Returns:
            AgentHealth status based on checks.
        """
        # Check 1: Is the agent stuck? (same state for too long)
        if hasattr(agent, "_state_timestamp"):
            state_duration = time.time() - agent._state_timestamp
            if state_duration > 120:  # 2 minutes in same state
                logger.warning(
                    f"Agent {agent_id} stuck in state "
                    f"'{agent.execution_state}' for {state_duration:.0f}s"
                )
                return AgentHealth.DEGRADED

        # Check 2: Has the agent exceeded its step limit?
        if agent.current_step >= agent.max_steps:
            return AgentHealth.DEGRADED

        # Check 3: Is the agent in an error state?
        if agent.execution_state == Agent.STATE_ERROR_RECOVERY:
            return AgentHealth.DEGRADED

        return AgentHealth.HEALTHY
```

### 7.4 Graceful Shutdown

```python
class GracefulShutdown:
    """
    Manages graceful shutdown of the multi-agent orchestration system.

    Shutdown sequence:
    1. Stop accepting new tasks
    2. Wait for active tasks to complete (up to timeout)
    3. Cancel remaining tasks
    4. Release all locks
    5. Flush message bus
    6. Save state snapshot
    7. Clean up resources
    """

    def __init__(
        self,
        orchestrator: "OrchestratorAgent",
        shutdown_timeout: float = 60.0,
    ):
        self._orchestrator = orchestrator
        self._shutdown_timeout = shutdown_timeout
        self._shutting_down = False

    async def shutdown(self):
        """Execute graceful shutdown sequence."""
        if self._shutting_down:
            return
        self._shutting_down = True

        logger.info("Initiating graceful shutdown...")

        # Step 1: Stop accepting new tasks
        self._orchestrator.execution_state = "SHUTTING_DOWN"

        # Step 2: Wait for active tasks with timeout
        start_time = time.time()
        while self._orchestrator.active_agents:
            elapsed = time.time() - start_time
            if elapsed > self._shutdown_timeout:
                logger.warning(
                    f"Shutdown timeout after {elapsed:.0f}s. "
                    f"Cancelling {len(self._orchestrator.active_agents)} active agents."
                )
                break
            await asyncio.sleep(1.0)

        # Step 3: Cancel remaining tasks
        for plan in self._orchestrator.plans:
            for task in plan.sub_tasks:
                if task.status in (SubTaskStatus.PENDING, SubTaskStatus.RUNNING):
                    task.status = SubTaskStatus.CANCELLED
                    task.error = "Cancelled due to system shutdown"

        # Step 4: Release all locks
        if self._orchestrator.state_manager:
            state = self._orchestrator.state_manager
            for key in state.keys():
                entry = state._entries.get(key)
                if entry and entry.is_locked():
                    entry.locked_by = None
                    entry.lock_expiry = 0.0

        # Step 5: Flush message bus
        if self._orchestrator.message_bus:
            stats = self._orchestrator.message_bus.get_stats()
            if stats["dead_letters"] > 0:
                logger.warning(
                    f"Discarding {stats['dead_letters']} dead letter messages"
                )

        # Step 6: Save state snapshot
        if self._orchestrator.state_manager:
            snapshot = self._orchestrator.state_manager.get_snapshot()
            logger.info(
                f"Final state snapshot: {len(snapshot)} entries"
            )

        # Step 7: Clean up
        self._orchestrator.active_agents.clear()
        logger.info("Graceful shutdown complete")
```

---

## 8. Coordination Strategies

This section defines the four execution strategies the orchestrator uses to coordinate agents. Each strategy maps to specific use cases and can be mixed within a single orchestration plan.

### 8.1 Sequential Execution

The simplest strategy: sub-tasks execute one after another, each receiving the previous task's result as context.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Sequential execution strategy for multi-agent orchestration.
"""

from typing import Any, Dict, List, Optional

from gaia.agents.base.agent import Agent
from gaia.agents.orchestration.orchestrator import (
    OrchestrationPlan,
    SubTask,
    SubTaskStatus,
)


class SequentialStrategy:
    """
    Execute sub-tasks one after another in dependency order.

    Each sub-task receives the accumulated results from all
    preceding tasks. This is the safest strategy: no concurrency
    issues, deterministic ordering, easy to debug.

    Use when:
    - Tasks have strict sequential dependencies
    - Debugging a multi-agent workflow
    - LLM context must build incrementally
    - Resource-constrained environments (single LLM instance)
    """

    async def execute(
        self,
        plan: OrchestrationPlan,
        select_agent_fn,
        enrich_fn,
        console,
    ) -> Dict[str, Any]:
        """
        Execute all sub-tasks sequentially.

        Args:
            plan: The orchestration plan.
            select_agent_fn: Callable to instantiate an agent for a sub-task.
            enrich_fn: Callable to inject dependency results into descriptions.
            console: Output handler for progress reporting.

        Returns:
            Dictionary mapping sub-task names to their results.
        """
        # Topological sort: respect dependencies
        execution_order = self._topological_sort(plan.sub_tasks)

        for i, sub_task in enumerate(execution_order):
            console.print_step_header(i + 1, len(execution_order))
            console.print_thought(
                f"Sequential step {i + 1}/{len(execution_order)}: "
                f"'{sub_task.name}' via {sub_task.agent_type} agent"
            )

            sub_task.status = SubTaskStatus.RUNNING
            enriched_desc = enrich_fn(sub_task, plan)

            try:
                agent = select_agent_fn(sub_task)
                result = agent.process_query(enriched_desc)
                sub_task.result = result
                sub_task.status = SubTaskStatus.COMPLETED
            except Exception as e:
                sub_task.status = SubTaskStatus.FAILED
                sub_task.error = str(e)
                console.print_error(f"Sub-task '{sub_task.name}' failed: {e}")

                # In sequential mode, failure can optionally halt the pipeline
                # or continue with remaining tasks
                if sub_task.metadata.get("halt_on_failure", False):
                    console.print_warning("Halting pipeline due to critical failure")
                    break

        return plan.get_results()

    def _topological_sort(self, sub_tasks: List[SubTask]) -> List[SubTask]:
        """Sort sub-tasks respecting dependency order."""
        id_to_task = {t.id: t for t in sub_tasks}
        visited = set()
        order = []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = id_to_task.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    visit(dep_id)
                order.append(task)

        for task in sub_tasks:
            visit(task.id)

        return order
```

### 8.2 Parallel Execution with Merge

Sub-tasks without dependencies execute concurrently. Results are merged once all parallel tasks complete, before downstream tasks begin.

```python
class ParallelMergeStrategy:
    """
    Execute independent sub-tasks in parallel, merge results at
    synchronization points (dependency boundaries).

    Execution proceeds in waves:
    - Wave 0: All tasks with no dependencies (parallel)
    - Wave 1: Tasks whose dependencies are all in Wave 0 (parallel)
    - Wave N: Tasks whose dependencies are all in Waves 0..N-1

    Use when:
    - Multiple independent sub-tasks exist
    - Performance is critical (minimize wall-clock time)
    - Sufficient resources for concurrent LLM calls
    """

    def __init__(self, max_concurrent: int = 4):
        self._max_concurrent = max_concurrent

    async def execute(
        self,
        plan: OrchestrationPlan,
        select_agent_fn,
        enrich_fn,
        console,
    ) -> Dict[str, Any]:
        """
        Execute sub-tasks in parallel waves.

        Args:
            plan: The orchestration plan.
            select_agent_fn: Callable to instantiate agents.
            enrich_fn: Callable to inject dependency results.
            console: Output handler.

        Returns:
            Dictionary mapping sub-task names to results.
        """
        import asyncio

        semaphore = asyncio.Semaphore(self._max_concurrent)
        wave_num = 0

        while not plan.is_complete():
            ready_tasks = plan.get_ready_tasks()
            if not ready_tasks:
                # Detect deadlock
                pending = [
                    t for t in plan.sub_tasks
                    if t.status == SubTaskStatus.PENDING
                ]
                if pending:
                    console.print_error(
                        f"Deadlock: {len(pending)} tasks with unresolvable deps"
                    )
                    for t in pending:
                        t.status = SubTaskStatus.CANCELLED
                break

            console.print_state_info(
                f"Wave {wave_num}: executing {len(ready_tasks)} tasks in parallel"
            )

            async def run_task(sub_task: SubTask):
                async with semaphore:
                    sub_task.status = SubTaskStatus.RUNNING
                    enriched = enrich_fn(sub_task, plan)
                    try:
                        agent = select_agent_fn(sub_task)
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None, agent.process_query, enriched
                        )
                        sub_task.result = result
                        sub_task.status = SubTaskStatus.COMPLETED
                    except Exception as e:
                        sub_task.status = SubTaskStatus.FAILED
                        sub_task.error = str(e)

            await asyncio.gather(*[run_task(t) for t in ready_tasks])
            wave_num += 1

        return plan.get_results()
```

### 8.3 Conditional Branching

Sub-tasks include conditions that determine whether they execute, based on the results of preceding tasks.

```python
from typing import Callable


class ConditionalStrategy:
    """
    Execute sub-tasks conditionally based on results of prior tasks.

    Conditions are Python callables that receive the current plan state
    and return True/False. Sub-tasks with unmet conditions are skipped.

    Use when:
    - Workflow branches based on intermediate results
    - Optional sub-tasks depend on classification outcomes
    - Error handling requires alternative paths

    Example conditions:
        - "Run Docker deploy only if all tests pass"
        - "Create Jira issue only if code review found issues"
        - "Use GPU agent if model size > 7B, otherwise CPU agent"
    """

    def __init__(
        self,
        conditions: Optional[Dict[str, Callable[[OrchestrationPlan], bool]]] = None,
    ):
        """
        Args:
            conditions: Mapping of sub-task names to condition functions.
                Each function receives the plan and returns True if the
                sub-task should execute.
        """
        self._conditions = conditions or {}

    async def execute(
        self,
        plan: OrchestrationPlan,
        select_agent_fn,
        enrich_fn,
        console,
    ) -> Dict[str, Any]:
        """Execute plan with conditional branching."""
        import asyncio

        execution_order = self._topological_sort(plan.sub_tasks)

        for sub_task in execution_order:
            # Check condition
            condition_fn = self._conditions.get(sub_task.name)
            if condition_fn and not condition_fn(plan):
                console.print_info(
                    f"Skipping '{sub_task.name}': condition not met"
                )
                sub_task.status = SubTaskStatus.CANCELLED
                sub_task.error = "Condition not met, skipped"
                continue

            # Check if all dependencies completed successfully
            dep_failed = any(
                t.status == SubTaskStatus.FAILED
                for t in plan.sub_tasks
                if t.id in sub_task.dependencies
            )
            if dep_failed and not sub_task.metadata.get("run_on_dep_failure", False):
                console.print_warning(
                    f"Skipping '{sub_task.name}': dependency failed"
                )
                sub_task.status = SubTaskStatus.CANCELLED
                sub_task.error = "Dependency failed"
                continue

            # Execute
            sub_task.status = SubTaskStatus.RUNNING
            enriched = enrich_fn(sub_task, plan)
            try:
                agent = select_agent_fn(sub_task)
                result = agent.process_query(enriched)
                sub_task.result = result
                sub_task.status = SubTaskStatus.COMPLETED
            except Exception as e:
                sub_task.status = SubTaskStatus.FAILED
                sub_task.error = str(e)

        return plan.get_results()

    def _topological_sort(self, sub_tasks: List[SubTask]) -> List[SubTask]:
        """Sort sub-tasks respecting dependency order (same as sequential)."""
        id_to_task = {t.id: t for t in sub_tasks}
        visited = set()
        order = []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            task = id_to_task.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    visit(dep_id)
                order.append(task)

        for task in sub_tasks:
            visit(task.id)
        return order
```

### 8.4 Sub-Agent Spawning

An agent dynamically spawns new sub-agents during execution when it discovers additional work. This enables recursive decomposition.

```python
class SpawningStrategy:
    """
    Allow agents to dynamically spawn sub-agents during execution.

    An active agent can request the orchestrator to spawn additional
    agents by writing spawn requests to the shared state. The orchestrator
    monitors for spawn requests and creates new sub-tasks on the fly.

    Use when:
    - Task complexity is unknown upfront
    - Agents discover additional work during execution
    - Recursive task decomposition is needed

    Example: A CodeAgent generating a backend discovers it needs a
    database migration agent, so it spawns a DatabaseAgent sub-task.
    """

    SPAWN_REQUEST_KEY = "__spawn_requests__"

    def __init__(
        self,
        state_manager,
        registry,
        max_spawned: int = 10,
    ):
        """
        Args:
            state_manager: SharedStateManager for spawn communication.
            registry: AgentRegistry for instantiating spawned agents.
            max_spawned: Maximum number of dynamically spawned sub-tasks.
        """
        self._state = state_manager
        self._registry = registry
        self._max_spawned = max_spawned
        self._spawn_count = 0

    def create_spawn_tool(self):
        """
        Create a tool that agents can use to request sub-agent spawning.

        Returns:
            A tool function that can be registered with @tool.
        """
        state = self._state

        @tool
        def spawn_agent(agent_type: str, description: str, priority: int = 1) -> dict:
            """Request the orchestrator to spawn a new sub-agent for additional work.

            Args:
                agent_type: Type of agent to spawn (code, chat, jira, docker, blender).
                description: Task description for the spawned agent.
                priority: Execution priority (0=highest).

            Returns:
                Acknowledgment with spawn request ID.
            """
            import uuid

            request_id = str(uuid.uuid4())[:8]
            requests = state.read(SpawningStrategy.SPAWN_REQUEST_KEY, [])
            requests.append({
                "id": request_id,
                "agent_type": agent_type,
                "description": description,
                "priority": priority,
            })
            state.write(
                SpawningStrategy.SPAWN_REQUEST_KEY,
                requests,
                author="spawn_system",
            )
            return {"status": "spawn_requested", "request_id": request_id}

        return spawn_agent

    async def process_spawn_requests(
        self, plan: OrchestrationPlan, select_agent_fn, console
    ):
        """
        Check for and process pending spawn requests.

        Called periodically by the orchestrator during plan execution.
        """
        requests = self._state.read(self.SPAWN_REQUEST_KEY, [])
        if not requests:
            return

        for req in requests:
            if self._spawn_count >= self._max_spawned:
                console.print_warning(
                    f"Max spawn limit ({self._max_spawned}) reached. "
                    f"Ignoring spawn request for '{req['agent_type']}'."
                )
                break

            # Create a new sub-task
            sub_task = SubTask(
                name=f"spawned_{req['id']}",
                description=req["description"],
                agent_type=req["agent_type"],
                priority=req.get("priority", 1),
                metadata={"spawned": True, "spawn_request_id": req["id"]},
            )
            plan.sub_tasks.append(sub_task)
            self._spawn_count += 1

            console.print_info(
                f"Spawned new sub-task '{sub_task.name}' "
                f"({sub_task.agent_type} agent)"
            )

        # Clear processed requests
        self._state.write(
            self.SPAWN_REQUEST_KEY, [], author="spawn_system"
        )
```

### 8.5 Strategy Selection Matrix

| Criteria | Sequential | Parallel Merge | Conditional | Spawning |
|----------|-----------|---------------|-------------|----------|
| **Task independence** | Low | High | Medium | Unknown |
| **Determinism** | Full | Partial | Full | Low |
| **Performance** | Slowest | Fastest | Medium | Variable |
| **Debuggability** | Easy | Moderate | Easy | Hard |
| **Resource usage** | Minimal | High | Medium | Variable |
| **Failure handling** | Simple | Complex | Configurable | Complex |
| **Use case** | Data pipelines | Parallel builds | Decision trees | Discovery |

---

## 9. Example Multi-Agent Workflows

### 9.1 Example 1: "Build a Full-Stack Application"

**User query**: "Build a task management app with user authentication, a dashboard, and Docker deployment"

**Agents involved**: CodeAgent (x2), DockerAgent, JiraAgent

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Example: Full-stack application build with multi-agent orchestration.
"""

from gaia.agents.orchestration.orchestrator import (
    OrchestratorAgent,
    OrchestrationPlan,
    OrchestrationPattern,
    SubTask,
)


def build_fullstack_app():
    """
    Demonstrates a hierarchical orchestration for building a
    full-stack application.
    """
    orchestrator = OrchestratorAgent(
        pattern="hierarchical",
        max_concurrent=3,
    )

    # The orchestrator decomposes this automatically, but here
    # is what the plan looks like:
    #
    # Wave 0 (parallel):
    #   - backend_api: CodeAgent creates Express.js REST API
    #   - frontend_ui: CodeAgent creates Next.js dashboard
    #   - project_setup: JiraAgent creates tracking issues
    #
    # Wave 1 (parallel, depends on Wave 0):
    #   - integration_tests: CodeAgent writes tests using both APIs
    #   - docker_config: DockerAgent creates deployment config
    #
    # Wave 2 (sequential, depends on Wave 1):
    #   - final_review: Orchestrator aggregates and checks consistency

    result = orchestrator.process_query(
        "Build a task management app with user authentication, "
        "a React dashboard showing task statistics, and Docker deployment. "
        "Also create Jira tickets to track the implementation."
    )

    return result


# Expected execution flow:
#
# [Orchestrator] Decomposing task...
# [Orchestrator] Plan: 5 sub-tasks, pattern: hierarchical
#
# Wave 0:
#   [CodeAgent-1] Generating Express.js API with auth endpoints...
#   [CodeAgent-2] Generating Next.js dashboard with charts...
#   [JiraAgent]   Creating 5 tracking issues...
#
# Wave 1:
#   [CodeAgent-3] Writing integration tests (using API schema from Wave 0)...
#   [DockerAgent] Creating Dockerfile and docker-compose.yml...
#
# [Orchestrator] Aggregating results...
# [Orchestrator] Checking for conflicts...
#   - Conflict: Frontend expects /api/tasks, Backend exposes /api/v1/tasks
#   - Resolution: Updating frontend API client to use /api/v1/tasks
# [Orchestrator] Final answer: <unified response with all artifacts>
```

**Shared state during execution:**

```
Blackboard State:
+-------------------+----------+-----------+-------------------+
| Key               | Version  | Author    | Value (preview)   |
+-------------------+----------+-----------+-------------------+
| api_schema        | 1        | code_1    | {endpoints: [...]}|
| frontend_routes   | 1        | code_2    | ["/", "/login"...]|
| jira_issues       | 1        | jira_1    | [TASK-101, ...]   |
| test_results      | 1        | code_3    | {passed: 12, ...} |
| docker_compose    | 1        | docker_1  | {services: {...}} |
+-------------------+----------+-----------+-------------------+
```

**Message flow:**

```
Orchestrator --[task.backend.assigned]--> CodeAgent-1
Orchestrator --[task.frontend.assigned]--> CodeAgent-2
Orchestrator --[task.jira.assigned]--> JiraAgent

CodeAgent-1 --[task.backend.completed {api_schema}]--> Orchestrator
CodeAgent-2 --[task.frontend.completed {routes}]--> Orchestrator
JiraAgent   --[task.jira.completed {issues}]--> Orchestrator

Orchestrator --[task.tests.assigned {api_schema, routes}]--> CodeAgent-3
Orchestrator --[task.docker.assigned {api_schema}]--> DockerAgent

CodeAgent-3 --[task.tests.completed {results}]--> Orchestrator
DockerAgent --[task.docker.completed {compose}]--> Orchestrator

Orchestrator --[orchestrator.plan.completed]--> (broadcast)
```

### 9.2 Example 2: "Research and Write a Technical Report"

**User query**: "Research AMD NPU acceleration performance and write a technical report with data analysis"

**Agents involved**: ChatAgent (RAG search), ChatAgent (analysis), ChatAgent (writing)

```python
def research_and_write_report():
    """
    Demonstrates a pipeline orchestration for research and report writing.
    """
    from gaia.agents.orchestration.pipeline import Pipeline, PipelineStage
    from gaia.agents.chat.agent import ChatAgent

    pipeline = Pipeline(stages=[
        PipelineStage(
            agent=ChatAgent(),
            name="research",
            description="Search documents and web for AMD NPU performance data",
            transform_input=lambda q: (
                f"Search for performance benchmarks, technical specs, and "
                f"comparisons related to: {q}. "
                f"Include specific numbers, dates, and sources."
            ),
            transform_output=lambda result: {
                "findings": result,
                "type": "raw_research",
            },
        ),
        PipelineStage(
            agent=ChatAgent(),
            name="analysis",
            description="Synthesize research findings into structured analysis",
            transform_input=lambda data: (
                f"Analyze these research findings and identify key patterns, "
                f"trends, and insights:\n\n{data['findings']}\n\n"
                f"Structure your analysis with: "
                f"1) Key metrics 2) Trends 3) Comparisons 4) Conclusions"
            ),
            transform_output=lambda result: {
                "analysis": result,
                "type": "structured_analysis",
            },
        ),
        PipelineStage(
            agent=ChatAgent(),
            name="writing",
            description="Draft the final technical report",
            transform_input=lambda data: (
                f"Write a professional technical report based on this analysis:\n\n"
                f"{data['analysis']}\n\n"
                f"Format: Executive Summary, Methodology, Findings, "
                f"Analysis, Recommendations, Conclusion."
            ),
        ),
    ])

    report = pipeline.run(
        "AMD NPU acceleration for local LLM inference on Ryzen AI processors"
    )
    return report


# Expected execution flow:
#
# [Pipeline] Stage 1/3: research
#   [ChatAgent] Searching indexed documents for NPU benchmarks...
#   [ChatAgent] Found 12 relevant passages across 4 documents
#   Output: Raw research findings (3200 tokens)
#
# [Pipeline] Stage 2/3: analysis
#   [ChatAgent] Analyzing findings...
#   [ChatAgent] Identified 5 key metrics, 3 trends
#   Output: Structured analysis (2100 tokens)
#
# [Pipeline] Stage 3/3: writing
#   [ChatAgent] Drafting technical report...
#   Output: Final report (4500 tokens, 6 sections)
```

### 9.3 Example 3: "Triage Incoming Issues with Approval Workflow"

**User query**: "Triage all unresolved Jira issues in the GAIA project, classify by urgency, and assign to appropriate teams"

**Agents involved**: JiraAgent (fetch), ChatAgent (classify), ChatAgent (approval), JiraAgent (update)

```python
def triage_issues_with_approval():
    """
    Demonstrates a conditional branching orchestration for issue triage.
    """
    orchestrator = OrchestratorAgent(
        pattern="hierarchical",
        max_concurrent=2,
    )

    # Define conditions for the conditional strategy
    conditions = {
        # Only run approval step if high-urgency issues found
        "human_approval": lambda plan: any(
            t.name == "classify_issues"
            and t.status == SubTaskStatus.COMPLETED
            and t.result
            and "high_urgency" in str(t.result)
            for t in plan.sub_tasks
        ),
        # Only update issues if approval was given or all are low/medium
        "update_issues": lambda plan: any(
            (t.name == "human_approval" and t.status == SubTaskStatus.COMPLETED)
            or (
                t.name == "classify_issues"
                and t.status == SubTaskStatus.COMPLETED
                and "high_urgency" not in str(t.result)
            )
            for t in plan.sub_tasks
        ),
    }

    # The plan:
    #
    # Step 1: Fetch all unresolved issues
    #   [JiraAgent] GET /rest/api/2/search?jql=status!=Done
    #
    # Step 2: Classify each issue by urgency and team
    #   [ChatAgent] Analyze issue descriptions, assign urgency + team
    #
    # Step 3 (conditional): Request human approval for high-urgency assignments
    #   [ChatAgent] Present high-urgency items for confirmation
    #   (Only runs if high-urgency issues exist)
    #
    # Step 4 (conditional): Update issues with labels and assignments
    #   [JiraAgent] PUT labels, assignees, and priorities
    #   (Only runs after approval or if no high-urgency items)

    result = orchestrator.process_query(
        "Triage all unresolved Jira issues in GAIA: "
        "fetch them, classify by urgency (high/medium/low), "
        "determine the responsible team, and update the issues "
        "with labels and assignments. "
        "Require approval before assigning high-urgency issues."
    )

    return result


# Expected execution flow:
#
# [Orchestrator] Decomposing task into 4 sub-tasks...
#
# Step 1: fetch_issues
#   [JiraAgent] Fetching unresolved issues from GAIA project...
#   Result: 23 issues found
#
# Step 2: classify_issues
#   [ChatAgent] Classifying 23 issues...
#   Result: {high_urgency: 3, medium: 12, low: 8, teams: {...}}
#
# Step 3: human_approval (CONDITIONAL - high_urgency found)
#   [ChatAgent] Presenting 3 high-urgency issues for approval:
#     - GAIA-456: "NPU driver crash on Ryzen 9" -> Team: Platform, P0
#     - GAIA-489: "Security: token exposure in logs" -> Team: Security, P0
#     - GAIA-501: "Data loss during RAG indexing" -> Team: Core, P1
#   [Approval required - present to user]
#
# Step 4: update_issues (CONDITIONAL - after approval)
#   [JiraAgent] Updating 23 issues with labels and assignments...
#   Result: 23 issues updated successfully
```

---

## 10. Integration with Workflow Orchestration

This section clarifies the relationship between multi-agent orchestration (this document) and the workflow orchestration system defined in `architecture/WORKFLOW_ORCHESTRATION_ARCHITECTURE.md`.

### 10.1 How Multi-Agent Differs from Workflow Orchestration

| Aspect | Multi-Agent Orchestration | Workflow Orchestration |
|--------|--------------------------|----------------------|
| **Primary concern** | Coordinating multiple agents on a single complex task | Automating multi-step processes with triggers and scheduling |
| **Trigger** | User query (synchronous) | Cron, webhook, file watcher, manual (async) |
| **Duration** | Seconds to minutes (interactive) | Minutes to hours (background) |
| **State persistence** | In-memory (session-scoped) | Persistent (survives restarts) |
| **Agent model** | Multiple agents running concurrently | Single agent per workflow step |
| **Communication** | Message bus + shared state | Data passing between steps |
| **Error handling** | Retry + conflict resolution | Retry + backoff + dead letter |
| **Scheduling** | None (on-demand) | Cron, interval, event-driven |
| **Defined in** | This document | `WORKFLOW_ORCHESTRATION_ARCHITECTURE.md` |

### 10.2 When to Use Which Pattern

**Use Multi-Agent Orchestration when:**
- A single user query requires expertise from multiple agent types
- Agents need to communicate during execution
- Results from multiple agents must be merged or reconciled
- The task is interactive and the user expects a real-time response

**Use Workflow Orchestration when:**
- Tasks should run on a schedule (daily reports, hourly checks)
- External events trigger processing (webhook, file drop)
- Multi-step processes need persistence and recovery
- Tasks run in the background without user interaction

**Use Both when:**
- A scheduled workflow triggers a multi-agent task
  (e.g., "Every morning, run a multi-agent analysis pipeline")

### 10.3 Hybrid Architecture

The two systems compose naturally. A workflow step can invoke the `OrchestratorAgent` as its agent, enabling scheduled multi-agent tasks.

```python
# Copyright(C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Hybrid example: Workflow step that triggers multi-agent orchestration.
"""

# Workflow definition (from WORKFLOW_ORCHESTRATION_ARCHITECTURE.md)
workflow_definition = {
    "name": "daily_code_review",
    "trigger": {"type": "cron", "schedule": "0 8 * * 1-5"},  # 8 AM weekdays
    "steps": [
        {
            "name": "fetch_prs",
            "agent": "jira",
            "action": "Fetch all open PRs from the last 24 hours",
        },
        {
            "name": "multi_agent_review",
            "agent": "orchestrator",  # <-- Multi-agent step
            "action": (
                "For each PR, run a code review (CodeAgent), "
                "check test coverage (TestAgent), and verify "
                "Docker compatibility (DockerAgent). "
                "Produce a consolidated review report."
            ),
            "config": {
                "pattern": "hierarchical",
                "max_concurrent": 3,
            },
        },
        {
            "name": "post_results",
            "agent": "jira",
            "action": "Post review summary as PR comment",
        },
    ],
}
```

**Integration points:**

```
Workflow Engine                        Multi-Agent Orchestration
+-----------------+                    +---------------------+
| Trigger (cron)  |                    |                     |
|    |            |                    |                     |
|    v            |                    |                     |
| Step 1: Jira   |                    |                     |
|    |            |                    |                     |
|    v            |    process_query() |                     |
| Step 2: ------->|------------------->| OrchestratorAgent   |
|  Orchestrator   |                    |   |                 |
|                 |                    |   +-> CodeAgent      |
|                 |                    |   +-> TestAgent      |
|                 |                    |   +-> DockerAgent    |
|                 |    result          |   |                 |
|    <------------|<-------------------| Aggregate           |
|    v            |                    |                     |
| Step 3: Jira   |                    |                     |
+-----------------+                    +---------------------+
```

### 10.4 Shared Infrastructure

Both systems share core GAIA infrastructure:

| Component | Multi-Agent Usage | Workflow Usage |
|-----------|-------------------|----------------|
| `Agent` base class | Specialist agents | Step executors |
| `@tool` decorator | Agent tools | Step actions |
| `AgentConsole` | Progress output | Step logging |
| `ChatSDK` | LLM interaction | LLM-powered steps |
| `LemonadeClient` | Local LLM backend | Local LLM backend |
| `AgentRegistry` | Discovering specialists | Discovering step agents |

### 10.5 Future Integration Roadmap

| Phase | Milestone | Target |
|-------|-----------|--------|
| Phase 1 | Multi-agent orchestration (this document) | Q2 2026 |
| Phase 2 | Workflow orchestration (`WORKFLOW_ORCHESTRATION_ARCHITECTURE.md`) | Q3 2026 |
| Phase 3 | Hybrid integration (workflow steps can be multi-agent) | Q3 2026 |
| Phase 4 | Visual workflow builder with drag-and-drop agent assignment | Q4 2026 |
| Phase 5 | Distributed multi-agent across multiple machines | 2027 |

---

## Appendix A: Module Layout

The proposed module structure within the GAIA codebase:

```
src/gaia/agents/orchestration/
    __init__.py
    orchestrator.py          # OrchestratorAgent (Section 4)
    messages.py              # Message, MessageType, etc. (Section 5.1)
    message_bus.py           # MessageBus (Section 5.2)
    state.py                 # SharedStateManager, CRDTs (Section 6)
    registry.py              # AgentRegistry (Section 7.1)
    health.py                # HealthMonitor (Section 7.3)
    shutdown.py              # GracefulShutdown (Section 7.4)
    strategies/
        __init__.py
        sequential.py        # SequentialStrategy (Section 8.1)
        parallel.py          # ParallelMergeStrategy (Section 8.2)
        conditional.py       # ConditionalStrategy (Section 8.3)
        spawning.py          # SpawningStrategy (Section 8.4)
    pipeline.py              # Pipeline, PipelineStage (Section 3.4)
    blackboard.py            # Blackboard, BlackboardController (Section 3.3)
```

## Appendix B: Configuration Schema

```json
{
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GAIA Multi-Agent Orchestration Configuration",
    "type": "object",
    "properties": {
        "orchestration": {
            "type": "object",
            "properties": {
                "default_pattern": {
                    "type": "string",
                    "enum": ["hierarchical", "peer_to_peer", "blackboard", "pipeline"],
                    "default": "hierarchical",
                    "description": "Default orchestration pattern"
                },
                "max_concurrent_agents": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 16,
                    "default": 4,
                    "description": "Maximum number of agents running in parallel"
                },
                "agent_timeout_seconds": {
                    "type": "integer",
                    "minimum": 30,
                    "maximum": 3600,
                    "default": 300,
                    "description": "Timeout for individual agent execution"
                },
                "max_retries": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 5,
                    "default": 2,
                    "description": "Maximum retry attempts per sub-task"
                },
                "enable_message_bus": {
                    "type": "boolean",
                    "default": true,
                    "description": "Enable inter-agent message passing"
                },
                "enable_shared_state": {
                    "type": "boolean",
                    "default": true,
                    "description": "Enable shared state manager"
                },
                "conflict_resolution": {
                    "type": "string",
                    "enum": ["last_writer_wins", "first_writer_wins", "merge", "reject", "manual"],
                    "default": "last_writer_wins",
                    "description": "Default conflict resolution strategy"
                },
                "max_spawn_depth": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10,
                    "default": 3,
                    "description": "Maximum depth for recursive sub-agent spawning"
                },
                "health_check_interval_seconds": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 120,
                    "default": 10,
                    "description": "Interval between agent health checks"
                },
                "shutdown_timeout_seconds": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 300,
                    "default": 60,
                    "description": "Maximum time to wait for graceful shutdown"
                }
            }
        }
    }
}
```

## Appendix C: Testing Strategy

### Unit Tests

```python
# tests/unit/test_orchestration.py

import pytest
from gaia.agents.orchestration.orchestrator import (
    OrchestratorAgent,
    OrchestrationPlan,
    SubTask,
    SubTaskStatus,
)
from gaia.agents.orchestration.state import SharedStateManager, ConflictError
from gaia.agents.orchestration.messages import Message, MessageType
from gaia.agents.orchestration.message_bus import MessageBus
from gaia.agents.orchestration.registry import AgentRegistry


class TestSubTask:
    """Tests for SubTask dataclass."""

    def test_default_status_is_pending(self):
        task = SubTask(name="test")
        assert task.status == SubTaskStatus.PENDING

    def test_default_retries(self):
        task = SubTask(name="test")
        assert task.retries == 2


class TestOrchestrationPlan:
    """Tests for OrchestrationPlan."""

    def test_get_ready_tasks_no_dependencies(self):
        plan = OrchestrationPlan(sub_tasks=[
            SubTask(id="a", name="task_a"),
            SubTask(id="b", name="task_b"),
        ])
        ready = plan.get_ready_tasks()
        assert len(ready) == 2

    def test_get_ready_tasks_with_dependencies(self):
        plan = OrchestrationPlan(sub_tasks=[
            SubTask(id="a", name="task_a", status=SubTaskStatus.COMPLETED),
            SubTask(id="b", name="task_b", dependencies=["a"]),
            SubTask(id="c", name="task_c", dependencies=["b"]),
        ])
        ready = plan.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == "b"

    def test_is_complete_all_done(self):
        plan = OrchestrationPlan(sub_tasks=[
            SubTask(status=SubTaskStatus.COMPLETED),
            SubTask(status=SubTaskStatus.FAILED),
        ])
        assert plan.is_complete()

    def test_is_complete_pending(self):
        plan = OrchestrationPlan(sub_tasks=[
            SubTask(status=SubTaskStatus.COMPLETED),
            SubTask(status=SubTaskStatus.PENDING),
        ])
        assert not plan.is_complete()


class TestSharedStateManager:
    """Tests for SharedStateManager."""

    def test_write_and_read(self):
        state = SharedStateManager()
        state.write("key1", {"data": 42}, author="agent_1")
        value = state.read("key1")
        assert value == {"data": 42}

    def test_version_increments(self):
        state = SharedStateManager()
        v1 = state.write("key1", "first", author="a")
        v2 = state.write("key1", "second", author="b")
        assert v1 == 1
        assert v2 == 2

    def test_optimistic_concurrency_conflict(self):
        state = SharedStateManager(
            conflict_resolution=ConflictResolution.REJECT
        )
        state.write("key1", "v1", author="a")
        with pytest.raises(ConflictError):
            state.write("key1", "v2", author="b", expected_version=0)

    def test_lock_prevents_other_writes(self):
        state = SharedStateManager()
        state.write("key1", "v1", author="a")
        state.acquire_lock("key1", "agent_a")
        with pytest.raises(LockError):
            state.write("key1", "v2", author="agent_b")

    def test_deep_copy_prevents_mutation(self):
        state = SharedStateManager()
        original = {"items": [1, 2, 3]}
        state.write("key1", original, author="a")
        read_value = state.read("key1")
        read_value["items"].append(4)
        assert state.read("key1") == {"items": [1, 2, 3]}


class TestMessageBus:
    """Tests for MessageBus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = MessageBus()
        received = []

        async def handler(msg: Message):
            received.append(msg)

        bus.subscribe("agent_1", "task.*", handler)
        await bus.publish(
            topic="task.completed",
            payload={"result": "done"},
            sender="agent_2",
        )

        assert len(received) == 1
        assert received[0].payload == {"result": "done"}

    @pytest.mark.asyncio
    async def test_request_reply(self):
        bus = MessageBus()

        # Set up responder
        async def responder():
            messages = await bus.get_messages("code_agent", timeout=5.0)
            for msg in messages:
                if msg.type == MessageType.REQUEST:
                    await bus.reply(msg, {"answer": 42}, sender="code_agent")

        import asyncio
        asyncio.create_task(responder())

        response = await bus.request(
            topic="compute",
            payload={"question": "what"},
            sender="orchestrator",
            recipient="code_agent",
            timeout=5.0,
        )

        assert response.payload == {"answer": 42}


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_and_list(self):
        registry = AgentRegistry()
        registry.register(
            name="test",
            agent_class_path="gaia.agents.chat.agent.ChatAgent",
            capabilities=["search"],
        )
        agents = registry.list_agents()
        assert "test" in agents

    def test_find_by_capability(self):
        registry = AgentRegistry()
        registry.register(
            name="code", agent_class_path="x.CodeAgent",
            capabilities=["code_gen"],
        )
        registry.register(
            name="chat", agent_class_path="x.ChatAgent",
            capabilities=["search"],
        )
        found = registry.find_by_capability("code_gen")
        assert found == ["code"]

    def test_checkout_checkin(self):
        registry = AgentRegistry()
        registry.register(
            name="test", agent_class_path="x.Agent",
            max_instances=2,
        )
        assert registry.checkout("test") is True
        assert registry.checkout("test") is True
        assert registry.checkout("test") is False  # At capacity
        registry.checkin("test")
        assert registry.checkout("test") is True
```

### Integration Tests

```python
# tests/integration/test_orchestration_e2e.py

import pytest


@pytest.mark.integration
class TestOrchestratorEndToEnd:
    """End-to-end tests for the multi-agent orchestrator."""

    def test_sequential_pipeline(self, require_lemonade):
        """Test a simple sequential pipeline with real LLM."""
        from gaia.agents.orchestration.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent(
            pattern="pipeline",
            max_concurrent=1,
            silent_mode=True,
        )

        result = orchestrator.process_query(
            "Summarize what GAIA is in one sentence"
        )
        assert result is not None
        assert len(result) > 0

    def test_parallel_decomposition(self, require_lemonade):
        """Test that the orchestrator decomposes tasks into parallel sub-tasks."""
        from gaia.agents.orchestration.orchestrator import OrchestratorAgent

        orchestrator = OrchestratorAgent(
            pattern="hierarchical",
            max_concurrent=2,
            silent_mode=True,
        )

        plan = orchestrator._decompose_task(
            "Create a Python CLI tool and write tests for it"
        )
        assert len(plan.sub_tasks) >= 2
        # At least one task should have no dependencies (can run first)
        root_tasks = [t for t in plan.sub_tasks if not t.dependencies]
        assert len(root_tasks) >= 1
```

## Appendix D: Migration Guide

### For Existing Agent Users

Multi-agent orchestration is entirely opt-in. Existing agents continue to work without any changes:

```python
# Before (still works exactly the same)
from gaia.agents.chat.agent import ChatAgent

agent = ChatAgent()
result = agent.process_query("What is GAIA?")

# After (new capability, opt-in)
from gaia.agents.orchestration.orchestrator import OrchestratorAgent

orchestrator = OrchestratorAgent()
result = orchestrator.process_query(
    "Research GAIA features, write a summary, and create Jira tickets"
)
```

### For Agent Developers

To make a custom agent available for orchestration, register it with the `AgentRegistry`:

```python
from gaia.agents.orchestration.registry import AgentRegistry

registry = AgentRegistry()
registry.register(
    name="my_agent",
    agent_class_path="my_package.agents.MyCustomAgent",
    description="My custom agent that does specialized work",
    capabilities=["custom_analysis", "report_generation"],
)
```

No changes to the agent class itself are required. The orchestrator calls `process_query()` on each specialist agent, which is already the standard interface defined by the `Agent` base class.

### For Tool Developers

Tools registered with `@tool` work within orchestrated agents without modification. The tool registry scoping issue (global `_TOOL_REGISTRY`) is resolved by the orchestrator instantiating each agent in isolation:

```python
# Tools are scoped to their agent instance during orchestration.
# No changes needed to existing @tool decorated functions.

@tool
def my_custom_tool(param: str) -> dict:
    """My tool works in both single-agent and multi-agent modes."""
    return {"result": param}
```

---

## Appendix E: Glossary

| Term | Definition |
|------|-----------|
| **Orchestration** | Coordinating multiple agents to accomplish a complex task |
| **Sub-task** | An atomic unit of work assigned to a single specialist agent |
| **Specialist agent** | An existing GAIA agent (CodeAgent, JiraAgent, etc.) used as a worker |
| **Orchestrator** | The OrchestratorAgent that decomposes, delegates, and aggregates |
| **Message bus** | In-process async communication channel between agents |
| **Blackboard** | Shared state store where agents read/write artifacts |
| **DAG** | Directed Acyclic Graph of sub-task dependencies |
| **Wave** | A set of sub-tasks that can execute in parallel (same dependency depth) |
| **Conflict** | When two agents produce contradictory outputs for overlapping domains |
| **CRDT** | Conflict-Free Replicated Data Type for concurrent state updates |
| **Spawn** | Dynamic creation of new sub-tasks by an active agent during execution |
| **Heartbeat** | Periodic health signal from an active agent to the monitor |
