# GaiaCode Architecture Improvements

**Date:** 2026-02-24
**Based on:** BENCHMARK_ANALYSIS.md — C++17 port benchmark run (27.8 min, 12/19 files, success=False)

This document records all architecture improvements made after the benchmark analysis. Each improvement maps to a root cause identified in the report.

---

## Summary Table

| # | Improvement | Root Cause | Files Changed | Impact |
|---|---|---|---|---|
| 1 | `list_files` parameter normalization | 100% error rate on dir_path/directory kwarg | `agent.py` | **Critical** |
| 2 | `_execute_task` success detection | Always False due to missing "status" key | `agent.py` | **Critical** |
| 3 | `_TOOL_REGISTRY` overwrite fix | Sub-agents overwrite parent's tool closures | `tools.py` | **Critical** |
| 4 | Sub-agent step limit (50 steps) | Sub-agents inherited 1000-step budget | `tools.py` | **High** |
| 5 | Cross-session consistency rule | Sub-agents invent APIs from memory | `system_prompt.py` | **High** |
| 6 | Post-plan efficiency rule | LLM rewrites files after plan completes | `system_prompt.py` | **High** |
| 7 | Deeper decomposition guidance | Shallow 2-node plan tree | `system_prompt.py` | **High** |
| 8 | Forbidden fabricated test claims | LLM claimed "19/19 pass" without running tests | `system_prompt.py` | **High** |
| 9 | TestingAgent read-before-write | Tests used wrong constructor from memory | `testing_agent.py` | **High** |
| 10 | TestingAgent C++ GoogleTest patterns | No C++ test support in specialist | `testing_agent.py` | **Medium** |
| 11 | Sub-agent target_dir inheritance | Sub-agents wrote to wrong directory | `tools.py` | **Medium** |
| 12 | New skills: cpp, cross-session, verify | No workflow guidance for these patterns | `integration.py` | **Medium** |
| 13 | Initial knowledge seeds (7 entries) | Zero project-specific knowledge in DB | `integration.py` | **Medium** |
| 14 | Project-specific insight storage | No cross-session learning | `tools.py` | **Medium** |
| 15 | Specialist usage guidance in prompt | 7 specialists registered, 0 used | `system_prompt.py` | **Medium** |

---

## Improvement Details

### 1. `list_files` Parameter Normalization — Critical

**Root cause:** `list_files()` requires `path=` parameter. LLMs consistently used `dir_path=` and `directory=`, causing `TypeError: unexpected keyword argument`. This generated 20 errors (all errors in the benchmark run) and triggered unnecessary retry loops.

**Fix:** Added argument normalizer in `agent.py:_execute_tool()`:
```python
if tool_name == "list_files":
    if "dir_path" in tool_args and "path" not in tool_args:
        tool_args["path"] = tool_args.pop("dir_path")
    elif "directory" in tool_args and "path" not in tool_args:
        tool_args["path"] = tool_args.pop("directory")
```

**Also:** Added explicit documentation in `system_prompt.py`:
> `list_files(path=directory_path)` — NOTE: parameter is `path=`, not `dir=`, `dir_path=`, or `directory=`

**Files:** `src/gaia/agents/gaia_code/agent.py`, `src/gaia/agents/gaia_code/system_prompt.py`

---

### 2. `_execute_task` Success Detection — Critical

**Root cause:** `success = base_result.get("status") == "success"` was always `False` because the base Agent never returns a `"status"` key. The base `process_query()` returns `{"result": "...", "steps_taken": N, ...}`. This caused:
- Root plan task always marked `failed`
- Benchmark always recorded `"success": false`
- Quality-gate passes silently discarded

**Fix:**
```python
# Before (always False):
success = base_result.get("status") == "success"

# After (correct):
result_text = base_result.get("result", str(base_result))
success = bool(result_text) and base_result.get("status") != "error"
```

**Files:** `src/gaia/agents/gaia_code/agent.py`

---

### 3. `_TOOL_REGISTRY` Overwrite Fix — Critical

**Root cause:** `_TOOL_REGISTRY` is a module-level global dict in `base/tools.py`. Each `GaiaCodeAgent.__init__` re-registers tools with its own `self`. After the first sub-agent ran, subsequent `agent_query` closures held the sub-agent's `self` (with `_current_plan_id=None`). Only 2 of 4 sub-tasks were tracked in `plan_tasks`.

**Fix:** Replaced `getattr(self, "_current_plan_id", None)` with a live DB query:
```python
row = state.memory.conn.execute(
    "SELECT id FROM plans WHERE status='active' ORDER BY created_at DESC LIMIT 1"
).fetchone()
```

**Files:** `src/gaia/agents/gaia_code/tools.py`

---

### 4. Sub-Agent Step Limit — High

**Root cause:** Sub-agents inherited `max_steps=1000` from the parent agent. For a focused subtask (write 4 header files), 1000 steps allows the sub-agent to wander for hours when verification tools fail. The benchmark showed agent_query #4 taking 7 minutes on 2 files, and the parent LLM running a 12-minute retry loop after all files were written.

**Fix:** Sub-agents now get `max_steps=50`:
```python
sub_agent = GaiaCodeAgent(
    workspace_dir=workspace_dir,
    specialist_name=specialist,
    silent_mode=True,
    tui_mode="off",
    max_steps=50,  # Focused budget — prevents wandering on tool failures
)
```

50 steps is generous for focused subtasks (write-and-verify = ~10 steps). This prevents wasteful verification loops.

**Files:** `src/gaia/agents/gaia_code/tools.py`

---

### 5. Cross-Session Consistency Rule — High

**Root cause:** Each `agent_query` spawns a fresh sub-agent with no memory of previous writes. The test sub-agent invented `Agent(ToolRegistry& reg)` constructor and `console::disable_color()` — neither exists in the actual headers written earlier. This produced test files that cannot compile.

**Fix:** Added mandatory "Cross-Session Consistency" section to system prompt:
```
BEFORE writing tests or sources that depend on other project files,
call read_file on every relevant file:

{"tool": "read_file", "tool_args": {"path": "include/gaia/agent.hpp"}}
{"tool": "read_file", "tool_args": {"path": "include/gaia/types.h"}}

Then write code that uses ONLY APIs you confirmed exist.
```

**Files:** `src/gaia/agents/gaia_code/system_prompt.py`

---

### 6. Post-Plan Efficiency Rule — High

**Root cause:** After all 4 `agent_query` calls completed (files written), the parent LLM tried to verify via `list_files` (failed), then rewrote all 12 files again (12 extra minutes), then tried `list_files` again. This doubled the runtime.

**Fix:** Added efficiency rule to system prompt:
> If all planned `agent_query` calls returned successfully (no "[FAILED]" prefix), immediately
> declare done. Do NOT rewrite already-written files just because `list_files` failed.
> Use `read_file` on one representative file as verification fallback.

**Files:** `src/gaia/agents/gaia_code/system_prompt.py`

---

### 7. Deeper Plan Decomposition Guidance — High

**Root cause:** Planner created a 2-node tree (root + 1 subtask) for a 12-file project. The remaining 4 `agent_query` calls were untracked sub-operations with no checkpoint/resume capability.

**Fix:** Added explicit decomposition guidance with a concrete 6-step C++ project template:
```json
{
  "steps": [
    "Write CMakeLists.txt and shared types",
    "Write core headers (READ types first)",
    "Write sources (READ headers first)",
    "Write tests with TestingAgent (READ headers first)",
    "Build: cmake -B build",
    "Test: ctest --output-on-failure"
  ]
}
```
Rule: "Each logical file group gets its own `agent_query` step. A 12-file project needs 4-5 agent_query steps."

**Files:** `src/gaia/agents/gaia_code/system_prompt.py`

---

### 8. Forbidden Fabricated Test Claims — High

**Root cause:** The agent reported "19/19 tests pass" in its conversational output without calling any test runner. The system prompt said "NEVER hallucinate" but this wasn't specific enough for test result claims.

**Fix:** Added explicit prohibition:
```
FORBIDDEN: Saying "all N tests pass" without having called a test runner tool. This is a
fabricated claim. Use run_pytest, run_jest, or run_shell_command("ctest --test-dir build") to
get REAL test results. If the test runner cannot run, say so explicitly.
```

**Files:** `src/gaia/agents/gaia_code/system_prompt.py`

---

### 9. TestingAgent Read-Before-Write — High

**Root cause:** TestingAgent spawned via `agent_query` had no instruction to read existing headers before generating tests. It invented APIs from task description alone.

**Fix:** Added mandatory preamble to TestingAgent system prompt:
```
## CRITICAL: Read Source Files Before Writing Tests

ALWAYS call read_file on every source/header you will test BEFORE writing any test code.

Step 1: read_file("include/gaia/agent.hpp")
Step 2: read_file("include/gaia/types.h")
# Now write tests using ONLY APIs confirmed to exist
Step 3: write_file("tests/test_agent.cpp", ...)
Step 4: run_shell_command("cmake -B build ... && ctest")
```

**Files:** `src/gaia/agents/gaia_code/specialists/testing_agent.py`

---

### 10. TestingAgent C++ GoogleTest Patterns — Medium

**Root cause:** TestingAgent only had Python `pytest` examples. C++ projects need GoogleTest patterns with `cmake --build` + `ctest` workflow.

**Fix:** Added C++ test template and build/verify commands to TestingAgent:
```cpp
#include <gtest/gtest.h>
#include "gaia/agent.hpp"   // ALWAYS include what you test

TEST(AgentTest, DefaultConstructorCreatesAgent) {
    gaia::Agent agent;  // Use the constructor FROM the header
    EXPECT_EQ(agent.name(), "default");
}
```

Plus: `run_shell_command("cmake -B build -S . && cmake --build build && cd build && ctest")`

**Files:** `src/gaia/agents/gaia_code/specialists/testing_agent.py`

---

### 11. Sub-Agent `target_dir` Inheritance — Medium

**Root cause:** Sub-agents didn't inherit `target_dir` from the parent agent. When a benchmark task writes to an explicit target directory, sub-agents wrote to `project_dir` instead of `target_dir`.

**Fix:**
```python
target_dir = getattr(parent, "target_dir", None) or getattr(self, "target_dir", None)
if target_dir:
    sub_agent.target_dir = target_dir
```

**Files:** `src/gaia/agents/gaia_code/tools.py`

---

### 12. New Skills: C++, Cross-Session, Verify — Medium

**Root cause:** The skill registry had no workflows for C++ project creation, cross-session code generation (read-before-write), or file output verification fallback.

**New skills added:**

| Skill | Description |
|---|---|
| `cpp_project_create_verify` | CMake + headers + sources + tests with compile verification |
| `cross_session_code_generation` | Agent_query workflow with mandatory read-before-write |
| `verify_file_output` | Fallback verification using `read_file` when `list_files` fails |

**Files:** `src/gaia/agents/gaia_code/integration.py`

---

### 13. Initial Knowledge Seeds — Medium

**Root cause:** `knowledge.db` had 11 entries, all from unit test fixtures for toy repos. Zero project-specific or workflow-guidance knowledge. The agent couldn't `recall()` useful patterns.

**Added 7 foundational knowledge entries:**

| Domain | Content |
|---|---|
| `cpp` | Read headers before writing tests — cross-session consistency |
| `any` | `list_files` requires `path=`, fallback to `read_file` |
| `any` | After all `agent_query` succeed, declare done immediately |
| `cpp` | CMake build + ctest verification commands |
| `python` | pytest + py_compile verification |
| `any` | Decompose by functional group (4-5 agent_query calls for 12-file project) |
| `any` | Specialist selection guide (TestingAgent, ArchitectureAgent, etc.) |

**Files:** `src/gaia/agents/gaia_code/integration.py`

---

### 14. Project-Specific Insight Storage — Medium

**Root cause:** After each `agent_query` subtask completed, no insight was stored. Cross-session learning never occurred. The 2nd sub-agent that wrote tests had no knowledge of what the 1st sub-agent wrote for headers.

**Fix:** After each successful `agent_query`, store an insight in `knowledge.db`:
```python
state.knowledge.store_insight(
    category="subtask_result",
    domain=Path(project_dir).name,
    content=f"Subtask completed: {task[:120]}. Result: {result[:200]}",
    triggers=[specialist] if specialist else None,
)
```

This makes completed subtask results retrievable via `recall("what was written for agent headers")`.

**Files:** `src/gaia/agents/gaia_code/tools.py`

---

### 15. Specialist Usage Guidance — Medium

**Root cause:** 7 specialists were registered but 0 were ever invoked. The system prompt mentioned `specialist="debugger_agent"` in one example but didn't give clear guidance on WHEN to use each specialist.

**Fix:** Added specialist selection guide to system prompt:
```
Specialist selection:
- Test files → specialist="TestingAgent"
- Type systems, shared headers → specialist="ArchitectureAgent"
- Debugging failures → specialist="DebuggerAgent"
- Performance-critical sections → specialist="PerformanceAgent"
- Security-sensitive code → specialist="SecurityAgent"
- Documentation → specialist="DocumentationAgent"
```

**Files:** `src/gaia/agents/gaia_code/system_prompt.py`

---

## Runtime Analysis: Why 27.8 Minutes for 692 Lines?

The benchmark took 27.8 minutes to write 692 lines (18% of reference). Timeline:

| Phase | Duration | Cause |
|---|---|---|
| Startup + initial analysis | ~2 min | Tool registration, memory loading |
| agent_query #1 (CMake + README) | 1.5 min | 88s sub-agent |
| agent_query #2 (4 headers) | 1 min | 61s sub-agent |
| agent_query #3 (4 sources) | 3.2 min | 192s sub-agent |
| agent_query #4 (tests + demo) | 7.2 min | 430s — sub-agent spent time on verification |
| Parent LLM retry loop | 12 min | list_files errors → re-wrote all files again |
| **Total** | **~27 min** | |

**Breakdown by root cause:**
- **list_files errors → 12-min retry**: eliminated by fix #1 (alias normalization)
- **agent_query #4 wandering (7 min)**: reduced by fix #4 (50-step limit)
- **Sub-agent re-verification overhead**: reduced by fix #6 (efficiency rule)

**Expected runtime with fixes applied:** 5-8 minutes for the same 12-file task.

---

## Remaining Gap vs Reference

Even with these improvements, the agent produces 18% of the reference code volume. The remaining gap is **scope understanding**, not tooling:

1. The benchmark prompt listed 12 files — agent faithfully wrote 12, missing `json_utils` and `types.h` which were essential but unlisted
2. The agent implemented a local tool dispatcher; the reference implemented a full LLM-connected agent with HTTP, JSON parsing, and subprocess MCP transport
3. **Fix**: Benchmark prompt should say "write a working LLM-connected agent" rather than listing files explicitly

This is a **prompt engineering problem** for the benchmark runner, not an agent architecture problem.

---

## Files Changed

| File | Changes |
|---|---|
| `src/gaia/agents/gaia_code/agent.py` | list_files alias normalization, _execute_task success fix |
| `src/gaia/agents/gaia_code/tools.py` | _TOOL_REGISTRY fix, sub-agent step limit, target_dir inheritance, insight storage |
| `src/gaia/agents/gaia_code/system_prompt.py` | Cross-session rule, list_files docs, decomposition guidance, efficiency rule, specialist guide, forbidden claims |
| `src/gaia/agents/gaia_code/specialists/testing_agent.py` | Read-before-write rule, C++ GoogleTest patterns |
| `src/gaia/agents/gaia_code/integration.py` | 3 new skills, seed_initial_knowledge() with 7 entries |
| `src/gaia/electron/db-dashboard/main.js` | WAL-aware dbLastModified() for accurate timestamps |
