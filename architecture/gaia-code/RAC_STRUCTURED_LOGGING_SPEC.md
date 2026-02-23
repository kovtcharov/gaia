# RAC Structured Logging Specification (v3)

## Purpose

Add comprehensive structured logging to the RAC architecture's two core components that currently have **zero observability**:
- `src/gaia/agents/base/shared_state.py`
- `src/gaia/agents/base/quality_gates.py`

## Design Principles

1. **No upward dependencies** - Base modules use `logging.getLogger(__name__)`, NOT `gaia.logger.get_logger`. Base layer must remain pure stdlib. The `GaiaLogger`/`log_manager` captures these loggers automatically via Python's logging hierarchy. This prevents circular import risk.
2. **Tagged messages** - `[Component]` prefix tags (matches existing `[JSON]`, `[PARSE]` convention)
3. **Comprehensive DEBUG** - Log EVERY operation at DEBUG level including read-only lookups. DEBUG is free when disabled and invaluable when enabled. We want full visibility into what each agent does and when.
4. **Log outside locks** - Capture data inside lock, log after releasing. Avoids holding locks during I/O.
5. **Lazy formatting for DEBUG** - Use `logger.debug("[Tag] msg %s", val)` not f-strings, to avoid evaluation cost when DEBUG is disabled
6. **No behavior changes** - Logging is additive only

## Log Level Policy

| Level | When to Use | Example |
|-------|-------------|---------|
| **DEBUG** | Every operation: reads, writes, queries, lookups, cache hits/misses | `[MemoryDB] cache miss path=/src/main.py` |
| **INFO** | State mutations, lifecycle events, gate pass | `[MasterPlan] task created id=abc` |
| **WARNING** | Expected failures: gate failures, max depth, escalation | `[SyntaxGate] failed errors=2` |
| **ERROR** | Unexpected exceptions inside gate/DB logic | `[TestGate] pytest crashed: OSError(...)` |

**Key decision**: Quality gate failures are WARNING, not ERROR. A gate failing is expected behavior - the escalation ladder handles it. ERROR is for unexpected exceptions only.

## Component: shared_state.py

### Logger Setup

```python
import logging

logger = logging.getLogger(__name__)
```

### _sanitize_fts5_query (module-level function)

| Operation | Level | Format |
|-----------|-------|--------|
| empty/invalid input | DEBUG | `[FTS5] query empty/invalid, returning None` |
| sanitized | DEBUG | `[FTS5] sanitized %r -> %r` |

### SharedAgentState

| Operation | Level | Format |
|-----------|-------|--------|
| `__new__` (creating) | INFO | `[SharedState] creating singleton instance` |
| `__new__` (existing) | DEBUG | `[SharedState] returning existing singleton` |
| `__init__` | INFO | `[SharedState] initialized workspace=%s` |
| `__init__` (skip) | DEBUG | `[SharedState] already initialized, skipping` |
| `reset_session` | INFO | `[SharedState] session reset` |

### MemoryDB

| Operation | Level | Format |
|-----------|-------|--------|
| `cache_file` | DEBUG | `[MemoryDB] cached path=%s size=%d` |
| `get_file` (hit) | DEBUG | `[MemoryDB] cache hit path=%s` |
| `get_file` (miss) | DEBUG | `[MemoryDB] cache miss path=%s` |
| `store_tool_result` | DEBUG | `[MemoryDB] tool result stored tool=%s` |

### KnowledgeDB

| Operation | Level | Format |
|-----------|-------|--------|
| `_create_tables` (migration) | INFO | `[KnowledgeDB] FTS5 schema migrated (added domain/category)` |
| `store_insight` | INFO | `[KnowledgeDB] insight stored id=%s category=%s domain=%s` |
| `recall` (query) | DEBUG | `[KnowledgeDB] recall query=%r sanitized=%r results=%d` |
| `recall` (skipped) | DEBUG | `[KnowledgeDB] recall skipped, empty/invalid query` |
| `store_preference` | INFO | `[KnowledgeDB] preference stored key=%s` |
| `get_preference` (hit) | DEBUG | `[KnowledgeDB] preference hit key=%s` |
| `get_preference` (miss) | DEBUG | `[KnowledgeDB] preference miss key=%s` |

### ToolsDB

| Operation | Level | Format |
|-----------|-------|--------|
| `register_tool` | INFO | `[ToolsDB] registered name=%s category=%s` |
| `find_tools` (query) | DEBUG | `[ToolsDB] find query=%r results=%d` |
| `find_tools` (skipped) | DEBUG | `[ToolsDB] find skipped, empty/invalid query` |
| `get_tool` (hit) | DEBUG | `[ToolsDB] get_tool name=%s found=True` |
| `get_tool` (miss) | DEBUG | `[ToolsDB] get_tool name=%s found=False` |
| `record_usage` | DEBUG | `[ToolsDB] usage tool=%s success=%s duration=%dms` |
| `get_tool_stats` | DEBUG | `[ToolsDB] stats tool=%s total=%d successes=%d` |

### SkillsDB

| Operation | Level | Format |
|-----------|-------|--------|
| `register_skill` | INFO | `[SkillsDB] registered name=%s category=%s steps=%d` |
| `find_skills` | DEBUG | `[SkillsDB] find category=%s domain=%s results=%d` |
| `record_usage` | DEBUG | `[SkillsDB] usage skill=%s success=%s confidence=%.2f` |

### AgentsDB

| Operation | Level | Format |
|-----------|-------|--------|
| `register_agent` | INFO | `[AgentsDB] registered name=%s` |
| `find_agent` (hit) | DEBUG | `[AgentsDB] find_agent name=%s found=True` |
| `find_agent` (miss) | DEBUG | `[AgentsDB] find_agent name=%s found=False` |
| `list_agents` | DEBUG | `[AgentsDB] list_agents count=%d` |
| `record_usage` | DEBUG | `[AgentsDB] usage agent=%s success=%s confidence=%.2f` |

### MasterPlan

| Operation | Level | Format |
|-----------|-------|--------|
| `create_task` | INFO | `[MasterPlan] task created id=%s parent=%s desc=%.80s` |
| `get_task` (hit) | DEBUG | `[MasterPlan] get_task id=%s found=True` |
| `get_task` (miss) | DEBUG | `[MasterPlan] get_task id=%s found=False` |
| `update_task_status` | INFO | `[MasterPlan] task %s: %s -> %s` |
| `get_all_tasks` | DEBUG | `[MasterPlan] get_all_tasks count=%d` |
| `clear_all_tasks` | INFO | `[MasterPlan] all tasks cleared` |

### AgentCallStack

| Operation | Level | Format |
|-----------|-------|--------|
| `push` (success) | INFO | `[CallStack] push depth=%d task=%.80s specialist=%s` |
| `push` (rejected) | WARNING | `[CallStack] max depth %d reached, push rejected` |
| `pop` | INFO | `[CallStack] pop depth=%d status=%s` |
| `current` | DEBUG | `[CallStack] current depth=%d task=%.80s` |
| `current` (empty) | DEBUG | `[CallStack] current: stack empty` |

### MessageQueue

| Operation | Level | Format |
|-----------|-------|--------|
| `send` | INFO | `[MessageQueue] sent id=%s priority=%s %s -> %s` |
| `receive` | DEBUG | `[MessageQueue] receive recipient=%s messages=%d` |
| `respond` | INFO | `[MessageQueue] response to id=%s` |

### ProjectManifest

| Operation | Level | Format |
|-----------|-------|--------|
| `add_file` (new) | INFO | `[Manifest] file added path=%s` |
| `add_file` (update) | DEBUG | `[Manifest] file updated path=%s` |
| `add_api` | INFO | `[Manifest] api added %s %s` |
| `add_decision` | INFO | `[Manifest] decision: %.80s` |
| `get_file` (hit) | DEBUG | `[Manifest] get_file path=%s found=True` |
| `get_file` (miss) | DEBUG | `[Manifest] get_file path=%s found=False` |
| `list_files` | DEBUG | `[Manifest] list_files count=%d` |

## Component: quality_gates.py

### Logger Setup

```python
import logging

logger = logging.getLogger(__name__)
```

### SyntaxGate

| Operation | Level | Format |
|-----------|-------|--------|
| `check` (no files) | DEBUG | `[SyntaxGate] no files to check` |
| per-file OK | DEBUG | `[SyntaxGate] %s syntax valid` |
| per-file error | DEBUG | `[SyntaxGate] %s:%d: %s` |
| `check` (pass) | INFO | `[SyntaxGate] passed files=%d` |
| `check` (fail) | WARNING | `[SyntaxGate] failed errors=%d` |

### ImportGate

| Operation | Level | Format |
|-----------|-------|--------|
| `check` (no files) | DEBUG | `[ImportGate] no files to check` |
| per-import OK | DEBUG | `[ImportGate] %s: import %s OK` |
| per-import error | DEBUG | `[ImportGate] %s: cannot import %s` |
| `check` (pass) | INFO | `[ImportGate] passed files=%d` |
| `check` (fail) | WARNING | `[ImportGate] failed errors=%d` |

### TestGate

| Operation | Level | Format |
|-----------|-------|--------|
| `check` (inferred dir) | DEBUG | `[TestGate] project_dir inferred: %s` |
| `check` (no tests) | DEBUG | `[TestGate] no test files in %s` |
| `check` (no framework) | DEBUG | `[TestGate] no test framework detected in %s` |
| `_find_test_files` | DEBUG | `[TestGate] found %d test files in %s` |
| `_run_pytest` (pass) | INFO | `[TestGate] passed test_files=%d` |
| `_run_pytest` (fail) | WARNING | `[TestGate] failed errors=%d` |
| `_run_pytest` (timeout) | WARNING | `[TestGate] timed out after 60s` |
| `_run_pytest` (exception) | ERROR | `[TestGate] exception: %s` |
| `_run_jest` (pass) | INFO | `[TestGate] jest passed` |
| `_run_jest` (fail) | WARNING | `[TestGate] jest failed` |
| `_run_jest` (timeout) | WARNING | `[TestGate] jest timed out after 60s` |
| `_run_jest` (exception) | ERROR | `[TestGate] jest exception: %s` |

### QualityGateRunner

| Operation | Level | Format |
|-----------|-------|--------|
| `run_all` (start) | INFO | `[QualityGates] running %d gates on %d files` |
| `run_all` (gate skipped) | DEBUG | `[QualityGates] %s disabled, skipping` |
| `run_all` (gate exception) | ERROR | `[QualityGates] %s threw exception: %s` |
| `run_all` (complete) | INFO | `[QualityGates] complete: %d/%d passed` |
| `enable_gate` | INFO | `[QualityGates] enabled: %s` |
| `disable_gate` | INFO | `[QualityGates] disabled: %s` |

### EscalationLadder

| Operation | Level | Format |
|-----------|-------|--------|
| `increment` | INFO | `[Escalation] retry %d/%d` |
| `escalate` | INFO | `[Escalation] escalated count=%d action=%s` |
| `reset` | INFO | `[Escalation] reset (was count=%d)` |
| `get_action` | DEBUG | `[Escalation] action=%s retry_count=%d` |

## Implementation Rules

1. **Import**: `import logging` + `logger = logging.getLogger(__name__)` (stdlib only in base modules)
2. **Lazy formatting**: `logger.debug("[Tag] msg %s %d", val1, val2)` for ALL levels. Consistent and avoids cost.
3. **Log outside locks**: Capture return values/state inside lock, log after lock release
4. **Truncate descriptions**: `%.80s` in format strings for task descriptions, decisions
5. **No sensitive data**: Never log file contents, API keys, or user data
6. **No exception swallowing**: Logging must not catch/suppress any exceptions
7. **Consistent gate detail**: All gates log file count on pass, error count on fail

## Log Point Count

| Component | Points |
|-----------|--------|
| shared_state.py | 38 |
| quality_gates.py | 25 |
| **Total** | **63** |

Comprehensive coverage: every public method in both files has at least one log point.

## Verification

After implementation with DEBUG enabled, a pipeline run should produce:

```
[SharedState] creating singleton instance
[SharedState] initialized workspace=/tmp/rac_test
[KnowledgeDB] insight stored id=abc category=error_pattern domain=python
[FTS5] sanitized 'auth patterns' -> 'auth OR patterns'
[KnowledgeDB] recall query='auth patterns' sanitized='auth OR patterns' results=1
[ToolsDB] registered name=read_file category=file_io
[ToolsDB] find query='file operations' results=1
[ToolsDB] get_tool name=read_file found=True
[SkillsDB] registered name=debug_workflow category=debugging steps=3
[AgentsDB] registered name=debugger
[AgentsDB] find_agent name=debugger found=True
[AgentsDB] list_agents count=1
[MasterPlan] task created id=def parent=None desc=Build calculator
[MasterPlan] get_task id=def found=True
[MasterPlan] task def: pending -> in_progress
[CallStack] push depth=0 task=Build calculator specialist=None
[CallStack] current depth=0 task=Build calculator
[MemoryDB] cached path=/tmp/calc/main.py size=245
[MemoryDB] cache hit path=/tmp/calc/main.py
[Manifest] file added path=/tmp/calc/main.py
[Manifest] file added path=/tmp/calc/test_main.py
[Manifest] list_files count=2
[QualityGates] running 3 gates on 2 files
[SyntaxGate] /tmp/calc/main.py syntax valid
[SyntaxGate] /tmp/calc/test_main.py syntax valid
[SyntaxGate] passed files=2
[ImportGate] /tmp/calc/main.py: import os OK
[ImportGate] passed files=2
[TestGate] found 1 test files in /tmp/calc
[TestGate] passed test_files=1
[QualityGates] complete: 3/3 passed
[MessageQueue] sent id=msg1 priority=FYI agent -> user
[MessageQueue] receive recipient=user messages=1
[MessageQueue] response to id=msg1
[CallStack] pop depth=0 status=completed
[MasterPlan] task def: in_progress -> completed
[MasterPlan] get_all_tasks count=1
```
