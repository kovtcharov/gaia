# GAIA Code: The World's Most Autonomous Coding Agent

**Status**: M0-M3 Implemented (Days 1-10)
**Architecture**: Recursive Agent Composition (RAC)
**Version**: 1.0.0

## Overview

GAIA Code is an autonomous coding agent that exceeds Claude Code in capability, autonomy, and reliability through the Recursive Agent Composition (RAC) architecture.

### Key Differentiators

1. **Context-Lean**: Never fills context window. Stores knowledge externally in databases.
2. **Continuous Execution**: No step limits. Runs until quality gates pass.
3. **Quality-First**: Verifies output works before declaring "done".
4. **Recursive**: Decomposes complex tasks via `agent_query()`.
5. **Persistent**: Learns across sessions via knowledge DB.

## Architecture

GAIA Code is built on 5 pillars:

1. **Context-Lean Architecture** (M0-M1)
   - Knowledge DB for cross-session persistence
   - RLMs for within-session recursive decomposition
   - Never fills context window

2. **Continuous Execution** (M2)
   - No step limits
   - Quality-driven completion (not step-driven)
   - Escalation ladder: retry → decompose → cloud → ask user

3. **Quality Gates** (M2)
   - Syntax check
   - Import check
   - Test check
   - Agent cannot complete until all gates pass

4. **Checkpoint/Resume** (M3)
   - State-based resume (not summary-based)
   - Survives crashes and interruptions
   - Zero information loss

5. **Persistent Memory** (M1)
   - 7 databases (memory, knowledge, tools, skills, agents, plan, manifest)
   - FTS5 full-text search
   - Cross-session learning

## Implementation Status

### ✅ Completed (M0-M3)

**M0: Prompting Foundation** (Days 1-2)
- ✅ System prompt with RLM patterns
- ✅ Error recovery prompts
- ✅ Tool usage guidelines
- ✅ Smart escalation (local → cloud)

**M1: SharedAgentState + RAC Foundation** (Days 3-5)
- ✅ memory.db (working cache)
- ✅ knowledge.db (cross-session learning)
- ✅ tools.db (tool registry)
- ✅ skills.db (learned workflows)
- ✅ agents.db (specialist registry)
- ✅ MasterPlan (hierarchical task tree)
- ✅ ProjectManifest (live project state)
- ✅ AgentCallStack (recursion tracking)
- ✅ MessageQueue (async communication)
- ✅ agent_query() tool (RAC mechanism)
- ✅ recall() tool (knowledge search)
- ✅ Automatic persistence

**M2: Quality Gates + Continuous Execution** (Days 5-8)
- ✅ SyntaxGate (AST parsing)
- ✅ ImportGate (module validation)
- ✅ TestGate (pytest/jest)
- ✅ QualityGateRunner (orchestration)
- ✅ EscalationLadder (retry → decompose → cloud → ask)
- ✅ Gate-driven completion
- ✅ Continuous execution (no max_steps)

**M3: Checkpoint/Resume + Audit Log** (Days 8-10)
- ✅ Checkpoint serialization (state-based)
- ✅ Resume from checkpoint
- ✅ Audit log (all actions timestamped)
- ✅ Time awareness
- ✅ Progress tracking

### 🔄 Pending (M4-M6)

**M4: Agent Registry + Core Specialists** (Days 10-13)
- ⏳ 7 core specialized agents
- ⏳ Specialist auto-selection
- ⏳ Dynamic tool loading
- ⏳ FAISS semantic search

**M5: Agent Auto-Generation** (Days 13-16)
- ⏳ ToolBuilder
- ⏳ Skill extraction
- ⏳ AgentFactory
- ⏳ Insight generation engine

**M6: Memory Defragmentation** (Days 16-19)
- ⏳ Embedding engine
- ⏳ FAISS integration
- ⏳ Defragmentation
- ⏳ Auto-defrag trigger

## Usage

### Basic Usage

```bash
# Execute a coding task
gaia code "Build a REST API with JWT authentication"

# Check progress
gaia code --status

# View audit log
gaia code --audit

# Resume from checkpoint
gaia code --resume

# Create checkpoint
gaia code --checkpoint
```

### Advanced Options

```bash
# Disable quality gates (not recommended)
gaia code "task" --no-quality-gates

# Disable continuous execution (use step limits)
gaia code "task" --no-continuous

# Use Claude API instead of local LLM
gaia code "task" --claude

# Enable debug output
gaia code "task" --debug

# Custom workspace
gaia code "task" --workspace /path/to/workspace
```

### Python API

```python
from pathlib import Path
from gaia.agents.gaia_code import GaiaCodeAgent

# Create agent
agent = GaiaCodeAgent(
    workspace_dir=Path.home() / ".gaia" / "workspace",
    enable_quality_gates=True,
    enable_continuous_execution=True,
)

# Execute task
result = agent.process_query("Build a REST API with auth")

if result["success"]:
    print("✅ Task completed!")
    print(f"Result: {result['result']}")

    # Check quality gates
    for gate in result["quality_gates"]:
        print(f"{gate.gate_name}: {gate.passed}")
else:
    print("❌ Task failed!")
    print(f"Error: {result['error']}")

# Get progress
progress = agent.get_progress()
print(f"Progress: {progress['progress_percent']}%")

# Get audit log
audit_log = agent.get_audit_log()
for entry in audit_log:
    print(f"{entry['timestamp']}: {entry['action_type']}")
```

## Architecture Details

### Recursive Agent Composition (RAC)

RAC is the core innovation. Instead of a single monolithic agent, GAIA Code is a system of agents that can:
- Recursively spawn sub-agents via `agent_query()`
- Each sub-agent has fresh context window
- All agents share the same SharedAgentState
- Coordination happens through shared plan and manifest

Example:
```python
# Main agent receives: "Build a full-stack app"

# Decomposes into subtasks:
backend_result = agent_query("Build FastAPI backend with auth")
frontend_result = agent_query("Build React frontend")

# Each subtask:
# - Gets fresh context
# - Has full tool access
# - Runs quality gates
# - Can recursively decompose further
```

### Context-Lean Design

Traditional agents fill the context window:
```
Claude Code:
Context Window (200K tokens)
├── System prompt
├── File contents (all)
├── Tool results (all)
├── Conversation history (all)
└── ... fills up, needs compaction
```

GAIA Code keeps context lean:
```
GAIA Code:
Context Window (<50% usage)
├── System prompt
├── Current task
├── Active file snippet
└── Last few tool results

Knowledge DBs (external)
├── memory.db (session cache)
├── knowledge.db (learnings)
├── tools.db (tool registry)
├── skills.db (workflows)
└── agents.db (specialists)
```

### Quality Gates

Quality gates are automated checks that run BEFORE the agent can declare "done":

1. **Syntax Gate**: All code parses without errors (AST)
2. **Import Gate**: All imports resolve successfully
3. **Test Gate**: All tests pass (pytest/jest auto-detected)

If gates fail, escalation ladder:
1. **Retry** (2 attempts)
2. **Decompose** into smaller subtasks via `agent_query()`
3. **Escalate** to cloud LLM (if available)
4. **Ask user** for help (last resort)

### Persistent Memory

GAIA Code learns across sessions:

**knowledge.db** stores:
- Insights (error-fix patterns, best practices)
- Preferences (user preferences, code style)
- Learnings (what works, what doesn't)
- Conventions (project-specific patterns)

**Example:**
```python
# Session 1: Agent fixes an error
agent encounters: AttributeError accessing None.attribute
agent learns: "Check if variable is not None before accessing attributes"
agent stores: insight("error_fix", "None check before attribute access", triggers=["AttributeError", "None"])

# Session 2: Similar error
agent recalls: "AttributeError" → finds stored insight
agent applies: Check for None first
agent succeeds: Faster fix using past learning
```

## File Structure

```
src/gaia/agents/gaia_code/
├── __init__.py              # Package exports
├── agent.py                 # Main GaiaCodeAgent class
├── shared_state.py          # RAC foundation (7 databases)
├── quality_gates.py         # Quality gates + escalation
├── system_prompt.py         # Prompting foundation
├── tools.py                 # GAIA-specific tools
├── cli.py                   # CLI commands
└── README.md                # This file

tests/unit/
├── test_gaia_code_agent.py           # Agent tests
├── test_gaia_code_shared_state.py    # SharedAgentState tests
└── test_gaia_code_quality_gates.py   # Quality gate tests
```

## Tests

Comprehensive test suite covers:

**SharedAgentState Tests**
- ✅ Singleton pattern
- ✅ Database initialization
- ✅ Memory operations
- ✅ Knowledge operations
- ✅ Master plan operations
- ✅ Call stack operations
- ✅ Message queue operations

**Quality Gate Tests**
- ✅ Syntax gate (valid/invalid syntax)
- ✅ Import gate (valid/invalid imports)
- ✅ Test gate (pytest/jest detection)
- ✅ Quality gate runner
- ✅ Escalation ladder progression

**Agent Tests**
- ✅ Agent initialization
- ✅ Tool registration
- ✅ Checkpoint/resume
- ✅ Audit logging
- ✅ Progress tracking
- ✅ System prompt generation

Run tests:
```bash
pytest tests/unit/test_gaia_code*.py -v
```

## Performance

Expected performance (based on architecture):

| Metric | Target | Status |
|--------|--------|--------|
| Context usage | <50% always | ✅ Implemented |
| Session continuity | No compaction ever | ✅ Implemented |
| Quality assurance | 100% syntax valid | ✅ Implemented |
| Crash recovery | Zero data loss | ✅ Implemented |
| Cross-session learning | Persistent knowledge | ✅ Implemented |

## Next Steps (M4-M6)

### M4: Specialists (Days 10-13)
Create 7 core specialized agents:
- DebuggerAgent
- SecurityAgent
- RefactoringAgent
- TestingAgent
- DocumentationAgent
- PerformanceAgent
- ArchitectureAgent

### M5: Auto-Generation (Days 13-16)
Enable the agent to create its own tools and specialists:
- ToolBuilder (agent creates new tools)
- SkillExtractor (learns workflows from experience)
- AgentFactory (generates specialists from patterns)
- InsightEngine (structured learning)

### M6: Defragmentation (Days 16-19)
Keep the knowledge DB clean and efficient:
- Embedding engine (semantic search)
- FAISS integration (vector search)
- Memory defragmentation (dedupe, reconcile, prune)
- Auto-defrag triggers

## References

- **Main Spec**: `architecture/gaia-code/GAIA_CODE_AUTONOMOUS_AGENT.md`
- **Milestones**: `architecture/gaia-code/GAIA_CODE_MILESTONES.md`
- **Eval Plan**: `architecture/gaia-code/GAIA_CODE_EVALUATION_PLAN.md`
- **RLM Paper**: [Recursive Language Models](https://arxiv.org/abs/2512.24601)

## License

Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
