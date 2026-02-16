# Gaia Code: Evaluation & Battle-Hardening Plan

**Date**: February 10, 2026
**Version**: 1.0
**Status**: Evaluation Strategy
**Goal**: Prove Gaia Code is state-of-the-art through rigorous benchmarks + custom tests

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [External Benchmarks](#external-benchmarks)
3. [Custom Benchmark: GaiaCodeBench](#custom-benchmark-gaiacodebench)
4. [Real-World Validation Scenarios](#real-world-validation-scenarios)
5. [Battle-Hardening Tests](#battle-hardening-tests)
6. [Regression Suite](#regression-suite)
7. [Evaluation Infrastructure](#evaluation-infrastructure)
8. [Success Criteria](#success-criteria)

---

## Philosophy

> **An agent that hasn't been rigorously tested is not state-of-the-art -- it's a prototype.**

Four levels of confidence:
1. **External benchmarks** -- Proves we compete with the best (Terminal-Bench, SWE-bench)
2. **Custom benchmark** -- Proves we excel at what matters to *our* users (real coding workflows)
3. **Real-world validation** -- Proves the agent works on actual projects, not just test cases
4. **Battle-hardening** -- Proves we don't break under stress (edge cases, long tasks, bad inputs)

---

## External Benchmarks

### 1. Terminal-Bench 2.0

**What it is:** 229 tasks across system administration, security, data science, and coding in real terminal environments. The gold standard for CLI agent evaluation.

**Current leaderboard (Feb 2026):**

| Agent | Model | Score |
|-------|-------|-------|
| Simple Codex | GPT-5.3-Codex | 75.0% |
| CodeBrain-1 | GPT-5.3-Codex | 70.3% |
| Droid | Claude Opus 4.6 | 69.9% |
| Qwen3-Coder-Next | 3B active params | 36.2% |

**Our target:** ≥65% on Terminal-Bench 2.0 with local LLM (Qwen3-Coder-30B). ≥75% with cloud fallback.

**How to run:**
```bash
# Terminal-Bench uses the Harbor harness
# We need to implement a Gaia Code adapter for Harbor
harbor run --agent gaia-code --model qwen3-coder-30b --benchmark terminal-bench-2.0

# OR run individual task categories
harbor run --agent gaia-code --category coding
harbor run --agent gaia-code --category system-admin
harbor run --agent gaia-code --category security
```

**Integration work needed:**
- Build Harbor adapter for Gaia Code (maps Harbor's agent interface to our CLI)
- Ensure Gaia Code can run in Docker containers (Terminal-Bench uses isolated environments)
- Add telemetry to capture per-task timing, token usage, and step count

**Reference:** [Terminal-Bench](https://www.tbench.ai/) | [GitHub](https://github.com/laude-institute/terminal-bench)

### 2. SWE-bench Verified

**What it is:** Real GitHub issues from popular Python repositories. Agent must read the issue, understand the codebase, write a patch that passes the repo's test suite.

**Current leaderboard (Feb 2026):**

| Agent | Model | Score |
|-------|-------|-------|
| Claude Opus 4.5 | Opus 4.5 | 80.9% |
| Claude Opus 4.6 | Opus 4.6 | 80.8% |
| GPT-5.2 | GPT-5.2 | 80.0% |
| Claude Sonnet 4.5 | Sonnet 4.5 | 77.2% |
| Qwen3-Coder-Next | 3B active | 70.6% |

**Our target:** ≥70% on SWE-bench Verified with local LLM (Qwen3-Coder-30B). ≥80% with cloud fallback.

**How to run:**
```bash
# SWE-bench uses its own harness
pip install swebench
swebench run --agent gaia-code --model qwen3-coder-30b --dataset verified
```

**Reference:** [SWE-bench](https://www.swebench.com/) | [SWE-bench Verified](https://epoch.ai/benchmarks/swe-bench-verified)

### 3. SWE-bench Pro

**What it is:** Harder variant of SWE-bench. More realistic, less gameable. Significant performance drop for all models.

**Current leaderboard (Feb 2026):**

| Agent | Model | Score |
|-------|-------|-------|
| GPT-5.3-Codex | Codex | 56.8% |
| GPT-5.2-Codex | Codex | 56.4% |
| GPT-5.2 | GPT-5.2 | 55.6% |

**Our target:** ≥45% on SWE-bench Pro with local LLM. ≥55% with cloud fallback.

**Reference:** [SWE-bench Pro](https://scale.com/leaderboard/swe_bench_pro_public)

---

## Custom Benchmark: GaiaCodeBench

> **External benchmarks test general capability. GaiaCodeBench tests what actually matters for daily developer workflows.**

### Why a Custom Benchmark?

External benchmarks have limitations:
- **Terminal-Bench** focuses on system administration, not full-stack development
- **SWE-bench** focuses on bug fixing in existing repos, not greenfield development
- Neither tests memory, learning, continuous execution, or plan quality

GaiaCodeBench tests the **end-to-end developer experience** that Gaia Code is built for.

### GaiaCodeBench Categories

#### Category 1: Greenfield Project Generation (20 tasks)

Build complete projects from scratch. Tests planning, task decomposition, multi-file coherence, and quality.

```
TASK GB-001: "Create a Python CLI tool that converts CSV to JSON with filtering"
  Expected:
  - cli.py with argparse
  - converter.py with business logic
  - tests/test_converter.py with ≥5 test cases
  - requirements.txt
  - README.md
  Validation:
  - All tests pass (pytest)
  - CLI runs correctly (csv_to_json input.csv --filter column=value)
  - Code passes black + isort
  - No security issues (bandit)

TASK GB-002: "Create a FastAPI REST API with user authentication"
  Expected:
  - Project structure (src/, tests/, etc.)
  - User model with SQLAlchemy
  - Register, login, refresh endpoints
  - JWT authentication middleware
  - 10+ tests (unit + integration)
  - Docker-compatible (Dockerfile or docker-compose.yml)
  Validation:
  - All tests pass
  - API responds to curl requests
  - JWT flow works end-to-end (register → login → access protected route)
  - No hardcoded secrets

TASK GB-003: "Create a React + TypeScript todo app with local storage"
  Expected:
  - Component structure (App, TodoList, TodoItem, AddTodo)
  - TypeScript types for all props and state
  - Local storage persistence
  - Add, complete, delete, filter functionality
  - Tests for core components
  Validation:
  - npm run build succeeds
  - npm test passes
  - TypeScript compilation clean
  - ESLint passes

TASK GB-004: "Create a Python package with proper packaging (pyproject.toml)"
  Expected:
  - pyproject.toml (not setup.py)
  - src/ layout
  - Tests
  - Entry point (CLI command)
  - Type hints throughout
  Validation:
  - pip install -e . works
  - Entry point command runs
  - Tests pass
  - mypy passes

... (16 more greenfield tasks covering Go, Rust, Next.js, Django, etc.)
```

#### Category 2: Bug Fixing & Debugging (20 tasks)

Given a project with known bugs, fix them. Tests diagnostic ability, error reading, and targeted fixes.

```
TASK GB-021: "Fix the authentication bypass in auth.py"
  Given: FastAPI project with a flawed JWT verification
  Bug: Token expiry check uses wrong comparison operator
  Validation:
  - Existing tests that were failing now pass
  - No regression (other tests still pass)
  - Fix is minimal (only changes the bug, not surrounding code)
  - Agent identifies root cause in ≤3 steps

TASK GB-022: "Fix the circular import between models.py and utils.py"
  Given: Python project that crashes on import
  Bug: models.py imports from utils.py which imports from models.py
  Validation:
  - Project imports without error
  - Both modules functional
  - Fix doesn't break any tests
  - Agent explains the circular dependency

TASK GB-023: "Debug why tests pass locally but fail in CI"
  Given: Python project with flaky test
  Bug: Test depends on file system ordering (os.listdir is non-deterministic)
  Validation:
  - Test passes deterministically
  - Agent identifies the non-determinism source
  - Fix uses sorted() or equivalent

... (17 more debugging tasks)
```

#### Category 3: Refactoring & Enhancement (15 tasks)

Modify existing code without breaking it. Tests understanding of existing code, safe modification, and regression avoidance.

```
TASK GB-041: "Add pagination to all API endpoints"
  Given: FastAPI project with 5 CRUD endpoints, no pagination
  Validation:
  - All endpoints support ?page=N&per_page=M parameters
  - Default page size is 20
  - Returns total count in response
  - Existing tests still pass
  - New tests for pagination

TASK GB-042: "Refactor synchronous code to async/await"
  Given: Flask-like sync project
  Validation:
  - All I/O operations are async
  - No blocking calls in async context
  - Performance improvement measurable
  - Tests adapted and passing

TASK GB-043: "Add type hints to an untyped Python project"
  Given: 500-line Python project with zero type hints
  Validation:
  - All function signatures have type hints
  - mypy passes with strict mode
  - No runtime behavior changes
  - Tests still pass

... (12 more refactoring tasks)
```

#### Category 4: Multi-File Coherence (15 tasks)

Tasks that require understanding and modifying multiple files in coordination. Tests project-level understanding.

```
TASK GB-056: "Add a new database model and wire it through the entire stack"
  Given: Full-stack project (FastAPI + React)
  Task: Add a "Comment" model with API endpoints and React components
  Validation:
  - Database model created
  - Migration generated
  - API endpoints (CRUD) created
  - React components created
  - Tests for backend and frontend
  - All existing tests still pass
  - API schema consistent between backend and frontend

TASK GB-057: "Rename a core module and update all references"
  Given: Project where 'utils.py' is imported by 15 files
  Task: Rename to 'helpers.py' and update everything
  Validation:
  - All imports updated
  - No broken references
  - All tests pass
  - Git diff shows clean rename

... (13 more multi-file tasks)
```

#### Category 5: Long-Running Tasks (10 tasks)

Tasks that require >100 steps to complete. Tests continuous execution, checkpoint/resume, and sustained quality.

```
TASK GB-071: "Build a complete blog platform (backend + tests + docs)"
  Expected: 30+ files, 1000+ lines of code
  Validation:
  - All files created
  - All tests pass (≥80% coverage)
  - API documentation generated
  - Quality consistent from first file to last file
  - No context degradation (last files are as good as first files)
  Time limit: 30 minutes

TASK GB-072: "Set up a complete CI/CD pipeline with Docker, GitHub Actions, and deployment"
  Expected: Dockerfile, docker-compose.yml, .github/workflows/*, deployment scripts
  Validation:
  - Docker builds successfully
  - GitHub Actions workflow is syntactically valid
  - Environment variables properly handled (no secrets in code)
  Time limit: 20 minutes

... (8 more long-running tasks)
```

#### Category 6: Memory & Learning (10 tasks)

Tests that require cross-session knowledge. Only Gaia Code (and similar agents with persistent memory) can pass these.

```
TASK GB-081: "Session 1: Build a Python project with specific conventions"
  Setup: User corrects agent 3 times (use pathlib, use dataclasses, use pytest)
  No validation yet -- just establish preferences

TASK GB-082: "Session 2: Build another Python project"
  Validation:
  - Agent uses pathlib (not os.path) without being reminded
  - Agent uses dataclasses (not dicts) without being reminded
  - Agent uses pytest (not unittest) without being reminded
  - Agent references "your preference" when making these choices
  Note: Claude Code would FAIL all three checks

TASK GB-083: "Session 1: Fix a tricky async bug"
  Setup: Agent debugs and fixes an asyncio.gather() exception handling bug
  No validation yet

TASK GB-084: "Session 2: Write async code that has a similar pattern"
  Validation:
  - Agent proactively handles exceptions in asyncio.gather()
  - Agent mentions the past bug or the learned pattern
  Note: Claude Code would not remember the past fix

... (6 more memory/learning tasks)
```

#### Category 7: Plan Quality (10 tasks)

Specifically tests the quality of plans the agent creates before coding.

```
TASK GB-091: "Create a plan for building a REST API with auth, don't implement yet"
  Validation:
  - Plan has clear steps (≥5 steps)
  - Steps are ordered logically (models before endpoints, endpoints before tests)
  - Dependencies between steps are identified
  - Estimated complexity per step
  - No missing critical steps (auth middleware, error handling, etc.)
  - Plan is readable by a human developer

TASK GB-092: "Create a plan, start implementing, then replan when you discover a problem"
  Given: Task that reveals a hidden complexity mid-implementation
  Validation:
  - Agent creates initial plan
  - Agent starts implementing
  - Agent discovers the hidden complexity
  - Agent updates the plan (not starts over)
  - Agent communicates the replan to the user
  - Agent completes the updated plan

... (8 more plan quality tasks)
```

### GaiaCodeBench Scoring

```
Total: 100 tasks across 7 categories

Scoring per task:
  - PASS (1.0): All validation criteria met
  - PARTIAL (0.5): ≥70% of validation criteria met
  - FAIL (0.0): <70% of validation criteria met

Category weights:
  - Greenfield (20 tasks):     20%
  - Bug Fixing (20 tasks):     20%
  - Refactoring (15 tasks):    15%
  - Multi-File (15 tasks):     15%
  - Long-Running (10 tasks):   10%
  - Memory (10 tasks):         10%
  - Plan Quality (10 tasks):   10%

Overall score = weighted sum of category scores
```

### GaiaCodeBench Targets

| Agent | Expected Score | Notes |
|-------|:---:|-------|
| Claude Code (Opus 4.6) | ~75% | Strong on categories 1-4, fails category 6 entirely |
| Gaia Code (Qwen3-30B) | ≥80% | Should match Claude on 1-4, win on 5-7 |
| Gaia Code (cloud fallback) | ≥85% | Best of both worlds |
| Cursor (Sonnet 4.5) | ~65% | IDE-focused, weaker on CLI tasks |
| Cline (various models) | ~55% | Good reasoning but limited autonomy |

**The bar:** If Gaia Code scores ≥80% on GaiaCodeBench with a local LLM, we are state-of-the-art for autonomous coding.

---

## Real-World Validation Scenarios

> **Benchmarks only tell part of the story.** A perfect benchmark score doesn't mean the agent can actually build real software. These scenarios prove the agent works on actual projects that matter.

### Why Real-World Tests?

| Benchmarks | Real-World |
|-----------|------------|
| Fixed inputs, expected outputs | Ambiguous requirements, evolving scope |
| Short tasks (minutes) | Long tasks (hours/days) |
| Clean environments | Messy codebases with legacy code |
| One correct answer | Many valid approaches |
| No user interaction | Mid-task clarifications, requirement changes |
| Fresh start | Builds on existing code |

### RW-1: Build Gaia Code V2 Using Gaia Code V1

**The ultimate dogfooding test.** If Gaia Code can build the next version of itself, it's proven.

```
Task: "Implement the Checkpoint/Resume pillar for Gaia Code V2"

Given:
  - The GAIA codebase (1M+ LoC)
  - The architecture spec (GAIA_CODE_AUTONOMOUS_AGENT.md)
  - The existing CodeAgent (70+ tools, 13 mixins)

Expected:
  - Agent indexes the GAIA codebase
  - Agent reads the architecture spec
  - Agent creates a plan (tasks, dependencies, estimates)
  - Agent implements CheckpointStore (SQLite)
  - Agent implements Serializable protocol
  - Agent writes tests
  - Agent integrates with existing CodeAgent
  - Agent runs full test suite (no regressions)
  - Total: ~8 hours of autonomous work

Validation:
  ✅ Agent completes without human intervention
  ✅ All new code passes quality gates
  ✅ Existing tests still pass (no regressions)
  ✅ Checkpoint/resume actually works (verified by test)
  ✅ Code follows existing GAIA patterns (not alien code)
  ✅ Agent produces audit log of all actions
  ✅ Agent checkpoints its own work (meta-test: uses checkpoint to build checkpoint)
```

### RW-2: Analyze and Fix Issues in a Large Open-Source Repo

**Proves the agent can work with code it has never seen before.**

```
Task: "Clone the FastAPI repository. Find and fix 3 open issues."

Given:
  - FastAPI repo (~200K LoC, well-structured but large)
  - List of open GitHub issues

Expected:
  - Agent clones and indexes the repo (~30 seconds)
  - Agent reads 3 selected issues
  - Agent navigates to relevant code
  - Agent understands the bug/feature
  - Agent implements fixes
  - Agent runs existing test suite
  - Agent creates patches

Validation:
  ✅ Agent successfully indexes a 200K LoC repo
  ✅ Agent finds the relevant code for each issue (not brute force)
  ✅ Fixes are correct (tests pass)
  ✅ Fixes are minimal (doesn't over-modify)
  ✅ Agent explains its reasoning for each fix
  ✅ Total time: <2 hours for 3 issues
```

### RW-3: Multi-Day Feature Build on Existing Codebase

**Proves the agent can work on a real project over multiple sessions.**

```
Day 1 (Session 1): "Add a complete notification system to this Django app"
  Given: Existing Django project with 50K LoC, users, posts, comments
  Expected:
    - Agent indexes codebase, understands existing models and patterns
    - Agent creates plan (estimated 15 tasks)
    - Agent implements: Notification model, signals, API endpoints
    - Agent checkpoints at end of session
    - Session time: 3 hours

Day 1 (Session 2): "Continue building the notification system"
  Expected:
    - Agent resumes from checkpoint (no re-reading, no re-indexing)
    - Agent recalls: models created, endpoints implemented, remaining tasks
    - Agent implements: WebSocket real-time delivery, email digest
    - Agent checkpoints at end of session
    - Session time: 2 hours

Day 2 (Session 3): "Finish the notifications and add push notifications too"
  Expected:
    - Agent resumes, recalls all previous work
    - Agent replans: adds push notification tasks
    - Agent implements: Push notification service, tests
    - Agent runs full quality gates
    - Total: notification system complete across 3 sessions, 7 hours

Validation:
  ✅ Agent resumes correctly across 3 sessions (no lost context)
  ✅ Code is consistent across all sessions (same patterns, same style)
  ✅ Replan in session 3 is clean (adds tasks, doesn't break existing)
  ✅ All tests pass (unit + integration)
  ✅ Django migrations work
  ✅ WebSocket notifications actually work
  ✅ Agent remembers project conventions from session 1 in session 3
```

### RW-4: Architectural Analysis of a Legacy Codebase

**Proves the agent can provide value without writing any code.**

```
Task: "Analyze this 500K LoC Java codebase and produce an architecture report"

Given: Legacy enterprise Java app (Spring Boot, 500K LoC, 8 years old)

Expected:
  - Agent indexes entire codebase
  - Agent identifies:
    - Module structure and boundaries
    - Dependency graph (internal and external)
    - Circular dependencies
    - Dead code (unused classes, methods)
    - Security vulnerabilities (SQL injection, XSS, auth bypass)
    - Test coverage gaps
    - API inconsistencies
    - Database schema issues
  - Agent produces structured report with severity ratings

Validation:
  ✅ Agent completes indexing of 500K LoC in <10 minutes
  ✅ Report identifies ≥5 real architectural issues
  ✅ Circular dependencies are correctly identified
  ✅ Dead code detection has <10% false positive rate
  ✅ Security findings are actionable (file + line number)
  ✅ Report is useful to a senior developer (not generic/obvious)
```

### RW-5: Requirement Change Mid-Build

**Proves the agent can pivot gracefully.**

```
Task progression:
  1. User: "Build a REST API for a todo app"
  2. Agent builds for 30 minutes, ~60% complete
  3. User: "Actually, make it a GraphQL API instead of REST"
  4. Agent must:
     - Assess what can be reused (models, business logic, tests)
     - Identify what must change (endpoints → resolvers, routing → schema)
     - Update the plan
     - Refactor without starting over
     - Complete the GraphQL version

Validation:
  ✅ Agent reuses models and business logic (doesn't rewrite from scratch)
  ✅ Agent updates plan (shows diff: "removed REST tasks, added GraphQL tasks")
  ✅ GraphQL API works correctly
  ✅ Tests updated and passing
  ✅ Time to complete: <50% of building from scratch (proves reuse)
  ✅ Agent communicates the pivot clearly in audit log
```

### RW-6: The "Use It Every Day" Test

**The most important test. Can a developer use Gaia Code as their primary coding tool for a week?**

```
Protocol:
  - Developer uses Gaia Code for ALL coding tasks for 5 work days
  - Tasks include: bug fixes, new features, refactoring, code review, testing
  - Developer records:
    - Tasks attempted vs. completed
    - Times agent needed help vs. worked autonomously
    - Times agent made mistakes vs. got it right
    - Times agent was faster vs. slower than manual coding
    - Overall satisfaction (1-10)

Success criteria:
  ✅ ≥80% of tasks completed without intervention
  ✅ ≥90% of completed tasks have correct output
  ✅ Agent is faster than manual coding for ≥60% of tasks
  ✅ Developer satisfaction ≥7/10
  ✅ Developer wants to continue using it after the week
  ✅ Agent improves over the 5 days (learns preferences, conventions)
```

---

## Battle-Hardening Tests

> **Benchmarks test capability. Battle-hardening tests resilience.**

These tests specifically target failure modes, edge cases, and stress conditions.

### BH-1: Context Pressure Tests

```
BH-1.1: "Build a project, then ask about a file you created 200 steps ago"
  Pass: Agent retrieves the file from knowledge DB without re-reading
  Fail: Agent re-reads the file (wasted step) or says it doesn't remember

BH-1.2: "Build a 50-file project without any conversation compaction"
  Pass: Agent completes without compacting, context stays <50% full
  Fail: Agent triggers compaction or runs out of context

BH-1.3: "Give a task that produces 100KB of terminal output"
  Pass: Agent stores output in knowledge DB, keeps summary in context
  Fail: Agent fills context with raw output
```

### BH-2: Error Recovery Tests

```
BH-2.1: "Task where the first 3 approaches fail"
  Pass: Agent tries different approaches, doesn't repeat failed ones
  Fail: Agent retries the same failed approach

BH-2.2: "Task where a dependency is missing (pip install needed)"
  Pass: Agent detects ImportError, installs dependency, continues
  Fail: Agent gets stuck on the error

BH-2.3: "Task where the test suite has a flaky test"
  Pass: Agent identifies the flaky test, fixes or skips it
  Fail: Agent enters infinite retry loop

BH-2.4: "Kill the LLM server mid-task"
  Pass: Agent checkpoints state, waits for server, resumes
  Fail: Agent crashes or loses progress
```

### BH-3: Quality Under Pressure Tests

```
BH-3.1: "Build 10 files. File 10 must reference patterns from file 1"
  Pass: Consistent patterns throughout (naming, structure, style)
  Fail: Style drift between early and late files

BH-3.2: "Build code that intentionally has a subtle bug, then ask agent to find it"
  Pass: Agent finds the bug through systematic analysis
  Fail: Agent claims code is correct

BH-3.3: "Build code, then change requirements mid-task"
  Pass: Agent updates plan, modifies code, re-runs tests
  Fail: Agent continues with original requirements or starts over from scratch
```

### BH-4: Adversarial Input Tests

```
BH-4.1: "User gives contradictory requirements"
  Pass: Agent asks for clarification
  Fail: Agent guesses or builds both versions

BH-4.2: "User asks agent to write code with a security vulnerability"
  Pass: Agent warns about the vulnerability and suggests safe alternative
  Fail: Agent writes vulnerable code

BH-4.3: "User asks to modify a file that doesn't exist"
  Pass: Agent reports the file doesn't exist, asks for correct path
  Fail: Agent creates the file silently or crashes

BH-4.4: "Extremely vague task: 'make it better'"
  Pass: Agent asks what 'better' means or analyzes code for specific improvements
  Fail: Agent makes random changes
```

### BH-5: Endurance Tests

```
BH-5.1: "100-step task with checkpoints every 10 steps"
  Pass: Agent maintains quality throughout, all checkpoints are resumable
  Fail: Quality degrades in later steps

BH-5.2: "Task that spans 3 context windows (checkpoint/resume twice)"
  Pass: Agent resumes correctly both times, final result is coherent
  Fail: Agent loses context on resume

BH-5.3: "Task where the agent must recall information from 50+ steps ago"
  Pass: Agent retrieves from knowledge DB accurately
  Fail: Agent fails to recall or recalls incorrect information
```

---

## Regression Suite

Every feature gets a regression test. Every bug that's fixed gets a test to prevent recurrence.

```
tests/
├── eval/
│   ├── external/
│   │   ├── test_terminal_bench_adapter.py    # Harbor adapter works
│   │   ├── test_swe_bench_adapter.py         # SWE-bench adapter works
│   │   └── conftest.py                       # Shared eval fixtures
│   ├── gaiacodebench/
│   │   ├── tasks/                            # 100 task definitions (YAML)
│   │   │   ├── greenfield/                   # GB-001 to GB-020
│   │   │   ├── debugging/                    # GB-021 to GB-040
│   │   │   ├── refactoring/                  # GB-041 to GB-055
│   │   │   ├── multifile/                    # GB-056 to GB-070
│   │   │   ├── longrunning/                  # GB-071 to GB-080
│   │   │   ├── memory/                       # GB-081 to GB-090
│   │   │   └── planning/                     # GB-091 to GB-100
│   │   ├── validators/                       # Automated validation scripts
│   │   │   ├── test_validator.py             # Tests pass?
│   │   │   ├── lint_validator.py             # Lint clean?
│   │   │   ├── security_validator.py         # No security issues?
│   │   │   └── coherence_validator.py        # Multi-file consistency?
│   │   ├── runner.py                         # Run GaiaCodeBench suite
│   │   └── scorer.py                         # Calculate scores per category
│   ├── battle_hardening/
│   │   ├── test_context_pressure.py          # BH-1.x tests
│   │   ├── test_error_recovery.py            # BH-2.x tests
│   │   ├── test_quality_under_pressure.py    # BH-3.x tests
│   │   ├── test_adversarial_input.py         # BH-4.x tests
│   │   └── test_endurance.py                 # BH-5.x tests
│   └── regression/
│       ├── test_pillar_context_lean.py        # Context never exceeds 50%
│       ├── test_pillar_continuous_execution.py # No step limit hit
│       ├── test_pillar_quality_gates.py       # All gates fire correctly
│       ├── test_pillar_checkpoint_resume.py   # State survives restart
│       └── test_pillar_persistent_memory.py   # Knowledge persists across sessions
```

### Task Definition Format (YAML)

```yaml
# tasks/greenfield/gb-001.yaml
id: GB-001
category: greenfield
difficulty: easy
title: "CSV to JSON CLI tool"
description: |
  Create a Python CLI tool that converts CSV files to JSON format.
  Support filtering rows by column value.
prompt: "Create a Python CLI tool that converts CSV to JSON with filtering"
time_limit_minutes: 10
expected_files:
  - "cli.py"
  - "converter.py"
  - "tests/test_converter.py"
  - "requirements.txt"
validation:
  - type: tests_pass
    command: "pytest tests/ -xvs"
  - type: cli_runs
    command: "python cli.py sample.csv --filter name=Alice"
    expected_exit_code: 0
  - type: lint_clean
    command: "black --check . && isort --check ."
  - type: no_security_issues
    command: "bandit -r . -q"
  - type: file_exists
    paths: ["cli.py", "converter.py", "tests/test_converter.py", "requirements.txt"]
setup:
  # Create sample CSV for testing
  files:
    "sample.csv": |
      name,age,city
      Alice,30,NYC
      Bob,25,LA
      Charlie,35,Chicago
```

---

## Evaluation Infrastructure

### Running Evaluations

```bash
# Run full GaiaCodeBench suite
gaia eval run --benchmark gaiacodebench --model qwen3-coder-30b

# Run specific category
gaia eval run --benchmark gaiacodebench --category greenfield

# Run single task
gaia eval run --benchmark gaiacodebench --task GB-001

# Run Terminal-Bench via Harbor
gaia eval run --benchmark terminal-bench --model qwen3-coder-30b

# Run battle-hardening tests
gaia eval run --benchmark battle-hardening

# Run regression suite
pytest tests/eval/regression/ -xvs

# Generate report
gaia eval report --format markdown --output eval_results.md
```

### Evaluation Report Format

```
═══════════════════════════════════════════════════════════
              GAIA CODE EVALUATION REPORT
              Date: 2026-02-10
              Model: Qwen3-Coder-30B (local)
═══════════════════════════════════════════════════════════

EXTERNAL BENCHMARKS
───────────────────
Terminal-Bench 2.0:    67.2%  (target: ≥65%)  ✅
SWE-bench Verified:    72.1%  (target: ≥70%)  ✅
SWE-bench Pro:         46.3%  (target: ≥45%)  ✅

GAIACODEBENCH (100 tasks)
─────────────────────────
Category              Score    Pass  Partial  Fail
────────────────────  ─────    ────  ───────  ────
Greenfield (20)       85.0%    15      4       1
Bug Fixing (20)       80.0%    14      4       2
Refactoring (15)      73.3%    10      3       2
Multi-File (15)       80.0%    11      2       2
Long-Running (10)     85.0%     8      1       1
Memory (10)           90.0%     9      0       1
Plan Quality (10)     85.0%     8      1       1
────────────────────  ─────    ────  ───────  ────
Overall               82.0%    75     15      10

BATTLE-HARDENING
────────────────
Context Pressure:      3/3 passed  ✅
Error Recovery:        4/4 passed  ✅
Quality Under Stress:  3/3 passed  ✅
Adversarial Input:     3/4 passed  ⚠️ (BH-4.4 partial)
Endurance:             3/3 passed  ✅

PILLAR REGRESSION
─────────────────
Context-Lean:          ✅ Max context usage: 42%
Continuous Execution:  ✅ No step limit hits
Quality Gates:         ✅ All gates functional
Checkpoint/Resume:     ✅ Full fidelity verified
Persistent Memory:     ✅ Cross-session recall working

═══════════════════════════════════════════════════════════
VERDICT: STATE-OF-THE-ART (82.0% GaiaCodeBench, all
         external benchmarks above target)
═══════════════════════════════════════════════════════════
```

---

## Success Criteria

### "State-of-the-Art" Threshold

We declare Gaia Code **state-of-the-art** when ALL of the following are met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Terminal-Bench 2.0 | ≥65% (local), ≥75% (cloud) | Competitive with frontier agents |
| SWE-bench Verified | ≥70% (local), ≥80% (cloud) | Matches top-tier agent frameworks |
| GaiaCodeBench Overall | ≥80% | Proves practical coding capability |
| GaiaCodeBench Memory | ≥85% | Proves our unique advantage |
| GaiaCodeBench Long-Running | ≥80% | Proves continuous execution works |
| Battle-Hardening | ≥90% pass rate | Proves resilience |
| Context Pressure | Zero compactions in 100-task run | Proves context-lean architecture |
| Checkpoint/Resume | 100% fidelity on 3-window task | Proves crash recovery |

### Progressive Milestones

```
Phase 0 (Infrastructure):
  ✅ Eval infrastructure exists
  ✅ GaiaCodeBench tasks defined (YAML)
  ✅ Automated validators work
  ✅ Terminal-Bench adapter works

Phase 1 (MVP - Pillars):
  ✅ GaiaCodeBench Greenfield ≥60%
  ✅ GaiaCodeBench Bug Fixing ≥60%
  ✅ Context never exceeds 50%
  ✅ No step limit hits on any task

Phase 2 (Quality):
  ✅ GaiaCodeBench Overall ≥70%
  ✅ Terminal-Bench ≥50%
  ✅ SWE-bench Verified ≥60%
  ✅ All battle-hardening tests pass

Phase 3 (State-of-the-Art):
  ✅ All success criteria met
  ✅ GaiaCodeBench ≥80%
  ✅ Memory category ≥85%
  ✅ Published results on website
```

---

## How Evaluation Feeds Back Into Development

```
Run eval → Identify weak category → Analyze failures → Fix agent → Re-run eval
                                          ↓
                                   Add new test to prevent regression
```

Every eval run generates:
1. **Score report** (what passed/failed)
2. **Failure analysis** (why it failed, with full transcript)
3. **Regression test** (automated test added for each failure pattern)
4. **Knowledge entry** (failure pattern stored in knowledge DB so agent can learn from it)

---

*Battle-hardened agents ship. Untested agents break.*
*External benchmarks prove competitiveness. Custom benchmarks prove utility.*
*The bar is high. That's the point.*
