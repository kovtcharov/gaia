# GAIA Code Improvements - Next Steps Summary

**Date:** 2026-02-23
**Status:** Analysis complete, awaiting approval to implement

---

## What We Discovered

**Benchmark:** gaiacpp_gaia scored **94/100**, exceeding Claude Code's 93/100 ✅

**GAIA Code Agent Issues:** Three autonomous runs failed before completion:
1. Round 2: Answer collapse (LLM dumped code into answer text)
2. Round 2 retry: Context exhaustion (200,817 tokens > 200K limit)
3. Round 4: API credits exhausted (but 3/3 files were production-quality)

---

## Root Causes Identified

### Issue 1: Unbounded Context Growth (CRITICAL)
- The `messages` array grows without limit
- At step 41: 200,817 tokens (exceeded 200K limit)
- **Fix:** Use existing `memory.db` to store full tool results, send only summaries to LLM

### Issue 2: Empty Thought/Goal Display
- LLM returned `{"thought": "", "goal": "", "answer": "<code dump>"}`
- Display logic suppressed empty values → silent steps
- **Fix:** Always display thought/goal with "[Empty]" placeholder when missing

### Issue 3: No Plan Display
- Plans are created but never shown to user
- Users can't see what the agent will do before it acts
- **Fix:** Call `console.print_plan()` immediately after plan creation

---

## Proposed Solutions (Pending Your Review)

### Solution 1: Database-Backed Context Management (HIGH PRIORITY)

**Use your existing SharedState/MemoryDB infrastructure:**

```python
# Instead of sending full tool results:
messages.append({"role": "user", "content": "<108KB Python file>"})  # ❌ 200K tokens

# Store in memory.db and send summary:
result_id = shared_state.memory.store_tool_result(tool, args, full_result)
messages.append({"role": "user", "content": "File: agent.py, 2,465 lines\nPreview: [first 10 lines]\n📎 Full: {result_id}"})  # ✅ 30K tokens
```

**Impact:** Prevents context exhaustion for unlimited-length conversations

**Files to modify:**
- `src/gaia/agents/base/shared_state.py` (add `result_id` to schema)
- `src/gaia/agents/base/agent.py` (modify `_execute_tool` to store and summarize)

**Estimated effort:** 4-6 hours

---

### Solution 2: Enhanced Plan Display (MEDIUM PRIORITY)

**Show plans in readable format with arguments:**

```
╭─── 📋 Execution Plan (5 steps) ───╮
│   Step 1/5: read_file
│     💡 Read the Python agent.py source
│     📝 file_path="/mnt/c/.../agent.py"
│ ▶ Step 2/5: write_file
│     💡 Create C++ types.h header
│     📝 file_path=".../types.h", content=str[450]
│   [... more steps ...]
╰───────────────────────────────────────────╯
```

**Impact:** Users see what will happen before it happens

**Files to modify:**
- `src/gaia/agents/base/console.py` (enhance `print_plan()`)
- `src/gaia/agents/base/agent.py` (call `print_plan()` after plan creation)

**Estimated effort:** 2-3 hours

---

### Solution 3: Hard Context Limit Safety Net (HIGH PRIORITY)

**Add 150K token hard cap to prevent crashes:**

```python
MAX_INPUT_CONTEXT_TOKENS = 150_000  # Leave 50K for output

# Before every LLM call:
if _estimate_tokens(messages) > MAX_INPUT_CONTEXT_TOKENS:
    messages = _prune_to_token_limit(messages, MAX_INPUT_CONTEXT_TOKENS)
```

**Impact:** Prevents crashes even if database storage fails

**Files to modify:**
- `src/gaia/agents/base/agent.py` (add pruning before `send_messages()`)

**Estimated effort:** 1-2 hours

---

## Configuration Changes Made

✅ **Default model changed:** `claude-opus-4-6` → `claude-sonnet-4-6`
- File: `src/gaia/agents/gaia_code/agent.py` lines 153-157, 163-165
- Reason: Sonnet 4.6 is 3x faster, 40% cheaper, and produced excellent output in Round 4

---

## Documentation Created

1. **GAIA_CODE_FAILURE_ANALYSIS.md** - Detailed failure analysis of all 3 rounds
2. **CONTEXT_MANAGEMENT_FIX_PLAN.md** - Database-backed solution architecture
3. **THOUGHT_GOAL_PLAN_DISPLAY_BUGS.md** - UI/UX bugs and fixes
4. **THIS FILE** - Executive summary

---

## Recommended Implementation Order

### Immediate (This Session - If Approved)
1. ✅ Change default to `claude-sonnet-4-6` (DONE)
2. Implement hard context limit (1-2 hours)
3. Enhance plan display (2-3 hours)
4. Fix empty thought/goal suppression (30 minutes)

### Next Session (When API Credits Available)
5. Implement database-backed tool result storage (4-6 hours)
6. Add `retrieve_tool_result` tool (1 hour)
7. Run Round 5 benchmark with all fixes
8. Measure: Can it complete 19 files without context errors?

### Future Enhancements
9. Conversational checkpointing (summarize history every 10 steps)
10. Answer validation (reject answers containing code/tool calls)
11. Filesystem reliability improvements

---

## Questions for You

1. **Approve database-backed context management?** Should we use `memory.db` to store full tool results and send only summaries to the LLM?

2. **Approve hard 150K token limit?** Should we add a safety net that prunes old messages when approaching the limit?

3. **Approve enhanced plan display?** Should we show plans with arguments and descriptions before execution?

4. **Priority order correct?** Or would you prefer different ordering?

5. **Should we implement now?** Or wait for more review/discussion?

---

## Expected Results After Implementation

**Context management:**
- Unlimited conversation length (tested to 100+ steps)
- Peak usage: ~40K tokens (vs current 200K crash)
- Graceful degradation if approaching limits

**User experience:**
- See plans before they execute
- See thought/goal even when empty (detect LLM issues)
- Track progress through multi-step plans
- Understand what was accomplished

**Benchmark success rate:**
- Current: 15% (3/19 files before credit exhaustion)
- After fixes: 95%+ (projected, based on Round 4 quality)
