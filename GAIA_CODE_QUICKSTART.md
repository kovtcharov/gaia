# GAIA Code: Developer Quickstart

**Get started with GAIA Code in 5 minutes**

---

## What You Need to Know

GAIA Code is an autonomous coding agent with:
- **Interactive planning**: Asks questions, doesn't assume (NEW!)
- **Recursive decomposition**: Breaks complex tasks into smaller subtasks
- **Quality gates**: Verifies output works before declaring "done"
- **Execution & observation**: Runs code and verifies it works (NEW!)
- **8 computer scientist personas**: Choose Torvalds (brutal), Knuth (teacher), Pike (simple), etc. (NEW!)
- **Persistent memory**: Learns across sessions
- **90+ tools**: Including web search, Playwright testing, codebase analysis (NEW!)

---

## Quick Start (5 minutes)

### 1. Setup with uv (Recommended - Fast!)

**Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create virtual environment and install**:
```bash
cd C:\Users\14255\Work\gaia

# Create venv and install (uv is much faster than pip)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install GAIA with all dependencies
uv pip install -e ".[dev]"

# Install Playwright for web app testing (optional but recommended)
uv pip install playwright
playwright install chromium
```

### Alternative: Traditional pip Setup

```bash
cd C:\Users\14255\Work\gaia

# Activate existing virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with pip
pip install -e ".[dev]"
pip install playwright
playwright install chromium
```

### 2. Configure LLM

**Option A: Claude Opus 4.6** (Default, Recommended):
```bash
# Set API key
export ANTHROPIC_API_KEY=your_api_key_here

# Agent automatically uses Claude Opus 4.6
```

**Option B: Local LLM** (Lemonade Server):
```bash
# Start Lemonade server with large context
lemonade-server serve --ctx-size 32768

# Tell agent to use local LLM
gaia code "task" --local
```

### 3. Run Interactive GAIA Code (NEW!)

**With Interactive Planning** (asks questions, shows options):
```bash
# Interactive mode with Simple TUI (default)
gaia code "Build a REST API with authentication"

# Shows:
# ┌────────────────────────────────────┐
# │ ❗ CRITICAL: Authentication?        │
# └────────────────────────────────────┘
#   1. Yes - JWT tokens ← RECOMMENDED
#   2. Yes - Session-based
#   3. No authentication
#   4. Custom answer
#
# Your choice [1]:
```

**With Different Personas**:
```bash
# Brutally honest (Linus Torvalds)
gaia code "Build API" --persona torvalds

# Thorough teacher (Donald Knuth)
gaia code "Build API" --persona knuth

# Simplicity advocate (Rob Pike)
gaia code "Build API" --persona pike

# Performance-focused (John Carmack)
gaia code "Build API" --persona carmack
```

**With Different TUI Modes**:
```bash
# Full mode (detailed, multi-panel)
gaia code "Build API" --persona pike --tui full

# Simple mode (default, clean)
gaia code "Build API" --persona pike --tui simple

# Minimal mode (one line)
gaia code "Build API" --persona pike --tui minimal

# No TUI (verbose logs)
gaia code "Build API" --persona pike --tui off
```

### 4. Launch Electron Observability App

```bash
# Open full observability dashboard
cd src/gaia/agents/gaia_code/electron
npm install
npm start

# Shows:
# - Real-time agent activity
# - All 7 databases (browsable)
# - Execution plan
# - Quality gate status
# - Specialist activity
# - Performance metrics
# - Screenshots of apps being tested
```

### 5. Check Status & Progress

```bash
# View current progress
gaia code --status

# View complete audit log
gaia code --audit

# View workspace and databases
ls ~/.gaia/workspace/
# memory.db, knowledge.db, tools.db, skills.db, agents.db, plan.db
```

---

## Python API (Quick Reference)

```python
from pathlib import Path
from gaia.agents.gaia_code import GaiaCodeAgent

# Create agent with persona and TUI
agent = GaiaCodeAgent(
    workspace_dir=Path.home() / ".gaia" / "workspace",
    persona="pike",  # Simplicity advocate (or torvalds, knuth, carmack, etc.)
    tui_mode="simple",  # Beautiful interactive TUI
    enable_quality_gates=True,
    enable_continuous_execution=True,
)

# Execute task - shows interactive planning
result = agent.process_query("Build a calculator app with tests")

# Interactive planning will:
# 1. Ask clarifying questions with selectable options
# 2. Show detailed plan preview
# 3. Ask for approval
# 4. Execute with beautiful TUI
# 5. Run and observe code
# 6. Loop until functional

# Check result
if result["success"]:
    print("✅ Task completed!")

    # View quality gates
    for gate in result["quality_gates"]:
        print(f"{gate.gate_name}: {'✅' if gate.passed else '❌'}")

    # View progress
    progress = agent.get_progress()
    print(f"Completed {progress['completed']}/{progress['total_tasks']} tasks")
else:
    print(f"❌ Failed: {result['error']}")
```

## Interactive Planning Example

```python
# Agent will show interactive questions:

┌──────────────────────────────────────────┐
│ ❗ CRITICAL: What language?               │
└──────────────────────────────────────────┘
  1. Python ← RECOMMENDED
  2. TypeScript
  3. Go
  4. Custom answer

Your choice [1]: 1
✓ Answered: Python

┌──────────────────────────────────────────┐
│ ⚠️ IMPORTANT: Test coverage?             │
└──────────────────────────────────────────┘
  1. Comprehensive (80%+) ← RECOMMENDED
  2. Basic (happy path only)
  3. None (skip tests)
  4. Custom answer

Your choice [1]: 1
✓ Answered: Comprehensive

# ... more questions ...

═══ PLAN PREVIEW ═══

Goal: Build a calculator app

Tasks (5):
  1. Create calculator.py [low]
     • Basic operations (+, -, *, /)
     • Input validation
     • Error handling

  2. Create test_calculator.py [medium]
     → TestingAgent
     • Unit tests for all operations
     • Edge case tests
     • Error handling tests

  [...continues...]

Does this plan look good? [yes/no/revise]: yes

✓ Plan approved! Starting execution...

▶ GAIA Code
  Build a calculator app

⠋ Executing • Writing calculator.py • 30% • 45s
```

---

## Understanding the Architecture (2 minutes)

### The RAC Pattern

When you give GAIA Code a complex task, it:

1. **Creates a plan** (hierarchical task tree)
2. **Decomposes** into subtasks
3. **Delegates** via `agent_query()` - each subtask gets a fresh context window
4. **Coordinates** through shared state (all agents see the same plan/manifest)
5. **Verifies** with quality gates (syntax, imports, tests)
6. **Returns** verified result

Example:
```
You: "Build a full-stack app"
  │
  ├─> Sub-agent 1: "Build backend"
  │   ├─> Sub-agent 1.1: "Create models"
  │   ├─> Sub-agent 1.2: "Create endpoints"
  │   └─> Sub-agent 1.3: "Write tests"
  │
  └─> Sub-agent 2: "Build frontend"
      ├─> Sub-agent 2.1: "Setup React"
      └─> Sub-agent 2.2: "Create components"
```

Each sub-agent has fresh context. Context never fills up.

### The 7 Databases

```
memory.db       → Session cache (files, tool results)
knowledge.db    → Cross-session learning (insights, preferences)
tools.db        → Tool registry (70+ tools, usage stats)
skills.db       → Learned workflows (multi-step patterns)
agents.db       → Specialist registry (7+ specialists)
plan.db         → Task hierarchy (parent-child tasks)
manifest        → Project state (files, APIs, decisions)
```

All agents in the recursion tree share these databases → perfect coordination.

### The Quality Gates

Before the agent can say "done", these MUST pass:

1. **SyntaxGate**: All code parses without errors (AST)
2. **ImportGate**: All imports resolve successfully
3. **TestGate**: All tests pass (pytest/jest)

If gates fail → Escalation ladder:
1. Retry (2 attempts)
2. Decompose (break into smaller subtasks)
3. Escalate to cloud (if available)
4. Ask user (last resort)

---

## Common Tasks

### Debug Failing Tests

```bash
gaia code "Debug the failing test in tests/test_user.py"
```

Agent will:
1. Auto-select DebuggerAgent
2. Read test output
3. Diagnose root cause
4. Fix the bug
5. Re-run test to verify

### Refactor Code

```bash
gaia code "Refactor user_service.py to reduce complexity"
```

Agent will:
1. Auto-select RefactoringAgent
2. Analyze code smells
3. Apply refactorings (extract function, reduce nesting)
4. Run tests to ensure behavior unchanged
5. Return cleaner code

### Generate Tests

```bash
gaia code "Write comprehensive tests for the auth module"
```

Agent will:
1. Auto-select TestingAgent
2. Analyze code to understand behavior
3. Generate tests (happy path + edge cases + errors)
4. Run tests to verify they pass
5. Check coverage

### Security Audit

```bash
gaia code "Scan this codebase for security vulnerabilities"
```

Agent will:
1. Auto-select SecurityAgent
2. Scan for OWASP Top 10 issues
3. Identify vulnerabilities (SQL injection, XSS, etc.)
4. Apply security fixes
5. Re-scan to verify

---

## Advanced Features

### Custom Workspace

```bash
gaia code "task" --workspace /path/to/project
```

### Use Cloud LLM

```bash
gaia code "task" --claude   # Use Claude API
gaia code "task" --chatgpt  # Use ChatGPT API
```

### Disable Quality Gates (Not Recommended)

```bash
gaia code "task" --no-quality-gates
```

### Direct Execution (No Planning)

```bash
gaia code "task" --no-plan
```

---

## Development Workflow

### 1. Write Code

```bash
# Let GAIA Code write the initial implementation
gaia code "Create a user authentication system with JWT"
```

### 2. Review Output

```bash
# Check what was created
gaia code --status

# View full audit log
gaia code --audit
```

### 3. Iterate

```bash
# Add features
gaia code "Add password reset functionality to auth system"

# Fix issues
gaia code "Fix the failing test in test_auth.py"

# Optimize
gaia code "Optimize the database queries in user_service.py"
```

### 4. Validate

```bash
# GAIA Code runs tests automatically
# Check quality gates passed
gaia code --audit | grep "QUALITY_GATES"
```

---

## Troubleshooting

### Agent Won't Start

```bash
# Check Lemonade server
lemonade-server serve --ctx-size 32768

# Or use cloud LLM
gaia code "task" --claude
```

### Quality Gates Keep Failing

```bash
# View detailed errors
gaia code --audit

# Try with decomposition (smaller subtasks)
# This happens automatically after 2 retries

# Or disable gates temporarily
gaia code "task" --no-quality-gates
```

### Out of Memory

```bash
# Defragment knowledge DB
gaia memory defrag

# Clear session cache
gaia cache clear

# Use smaller workspace
gaia code "task" --workspace /tmp/gaia-temp
```

---

## Architecture Deep Dive (10 minutes)

### How Recursion Works

```python
# Main agent receives complex task
def process_query(task):
    # Too complex for one pass?
    if is_complex(task):
        # Decompose into subtasks
        subtasks = decompose(task)

        # Execute each with fresh context
        results = []
        for subtask in subtasks:
            result = agent_query(subtask)  # ← Recursion!
            results.append(result)

        # Synthesize results
        return combine(results)
    else:
        # Simple enough - execute directly
        return execute(task)
```

Each `agent_query()` call:
- Creates a new agent instance
- Pushes onto call stack (tracking recursion)
- Gets fresh context window (context never fills)
- Has access to all tools
- Shares knowledge DB (coordination)
- Runs quality gates on output
- Returns verified result

### How Quality Gates Work

```python
def execute_with_quality_gates(task):
    attempts = 0

    while attempts < max_attempts:
        # Execute task
        result = execute_task(task)

        # Run quality gates
        all_passed, gate_results = run_quality_gates(result)

        if all_passed:
            return result  # Done!

        # Gates failed - escalate
        action = escalation_ladder.get_action()

        if action == "retry":
            # Try again
            attempts += 1
            continue

        elif action == "decompose":
            # Break into smaller pieces
            return decompose_and_retry(task, gate_results)

        elif action == "cloud":
            # Use more capable model
            return escalate_to_cloud(task)

        else:
            # Ask user
            return ask_user_for_help(task, gate_results)
```

### How Memory Works

```python
# Session 1
agent.process_query("Build API with pathlib")
  ├─> Uses pathlib for file operations
  ├─> Stores insight: "User prefers pathlib"
  └─> knowledge.db.store_insight("preference", "Use pathlib")

# Session 2 (new agent instance)
agent.process_query("Add file upload to API")
  ├─> Recalls: agent.recall("file operations")
  ├─> Finds: "Use pathlib" (from Session 1)
  └─> Uses pathlib (consistent with preference)
```

---

## Next Steps

### Learn More
- [Complete User Guide](docs/guides/gaia-code.mdx)
- [Architecture Diagram](GAIA_CODE_ARCHITECTURE_DIAGRAM.md)
- [Implementation Report](FINAL_IMPLEMENTATION_REPORT.md)
- [Architecture Spec](architecture/gaia-code/GAIA_CODE_AUTONOMOUS_AGENT.md)

### Contribute
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Development Guide](docs/reference/dev.mdx)

### Get Help
- GitHub Issues: https://github.com/amd/gaia/issues
- Documentation: https://amd-gaia.ai

---

**Happy coding with GAIA Code!** 🚀
