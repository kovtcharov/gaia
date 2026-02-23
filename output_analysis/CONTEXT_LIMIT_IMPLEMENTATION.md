# Hard Input Context Limit Implementation

**Date:** 2026-02-23
**Implemented by:** Claude Code
**Status:** ✅ Complete, ready for testing

---

## What Was Implemented

A **hard 32K token input context limit** with monitoring, warnings, and emergency compaction.

**Philosophy:** Agents should DECOMPOSE tasks to stay well under this limit. Hitting the limit indicates improper task design.

---

## Configuration

### Hard Limits (Class Constants in `Agent`)

```python
DEFAULT_MAX_INPUT_TOKENS = 32768   # Hard limit (32K tokens = ~130KB text)
WARNING_INPUT_TOKENS = 24576       # Warn at 75% (24K tokens)
EMERGENCY_INPUT_TOKENS = 30720     # Emergency alert at 93.75% (30K tokens)
```

### Configurable Per-Agent

```python
agent = GaiaCodeAgent(
    max_input_tokens=32768,  # Override default if needed
    # ...
)
```

**Default:** 32K tokens (configurable via `max_input_tokens` parameter)

---

## How It Works

### 1. Before Every LLM Call

```python
# Check context size
needs_compaction = self._check_context_size(messages, self.system_prompt, steps_taken)
if needs_compaction:
    messages = self._compact_messages_to_limit(messages, self.system_prompt)

# Then call LLM
chat_response = self.chat.send_messages(messages=messages, system_prompt=self.system_prompt)
```

### 2. Monitoring Thresholds

#### At 75% (24K tokens):
```
⚠️  Context approaching limit: 24,576/32,768 tokens (75.0%)
   Consider using agent_query() to decompose remaining work into subtasks.
   Each sub-agent gets fresh 32,768 token context.
```

**Logged:**
```
[WARNING] [CONTEXT] At 75.0% of limit (24,576 tokens at step 12)
```

#### At 93.75% (30K tokens):
```
🚨 CRITICAL: Context near limit: 30,720/32,768 tokens (93.8%)
   Next LLM call may fail. STRONGLY RECOMMENDED: Use agent_query() to delegate remaining work.
   Automatic compaction will activate if limit is exceeded.
```

**Logged:**
```
[ERROR] [CONTEXT] At CRITICAL level: 93.8% (30,720 tokens at step 18)
```

#### At 100% (32K+ tokens):
```
❌ Context limit exceeded: 33,500/32,768 tokens
   Automatic compaction activated. Will summarize older messages.
   RECOMMENDATION: Redesign task to use agent_query() decomposition.
```

**Logged (with detailed diagnostics):**
```
[ERROR] [CONTEXT] LIMIT EXCEEDED at step 20: 33,500 tokens
[ERROR] [CONTEXT] This should NEVER happen with proper task decomposition.
[ERROR] [CONTEXT] Agent should have used agent_query() to delegate subtasks.
[ERROR] [CONTEXT] Emergency compaction activating...
```

### 3. Emergency Compaction (Fallback)

When the hard limit is exceeded, the system automatically:

1. **Summarizes** messages 1 through -6 using an LLM call:
   ```
   "Summarize this conversation history concisely (max 500 tokens):
   [JSON of middle messages]

   Include:
   - Key decisions made
   - Files created/modified
   - Tools used
   - Any blockers encountered"
   ```

2. **Keeps:**
   - First message (original user query)
   - Summary message (replaces middle)
   - Last 5 messages (recent context)

3. **Logs diagnostics:**
   ```
   [ERROR] [CONTEXT] EMERGENCY COMPACTION TRIGGERED
   [ERROR] [CONTEXT] This indicates improper task decomposition.
   [ERROR] [CONTEXT] Messages before compaction: 41
   [ERROR] [CONTEXT] Total estimated tokens: 33,500
   [ERROR] [CONTEXT] Compaction complete:
   [ERROR] [CONTEXT] - Removed 34 messages
   [ERROR] [CONTEXT] - Kept 7 messages
   [ERROR] [CONTEXT] - Tokens after compaction: 12,450
   [ERROR] [CONTEXT] - Reduction: 21,050 tokens
   ```

4. **Warns user:**
   ```
   ⚠️  Emergency compaction: 34 messages summarized
      Tokens reduced: 33,500 → 12,450
      This should NEVER happen. Use agent_query() for large tasks.
   ```

---

## Expected Behavior

### Scenario: C++ Port Benchmark (19 files, should use RAC)

#### With Proper RAC Decomposition (Expected):

```
Step 1: Create plan with agent_query decomposition
  Messages: 1, Tokens: 2,500
  Plan:
    1. agent_query("Generate headers: types.h, json_utils.h, tool_registry.h")
    2. agent_query("Generate MCP client: mcp_client.h/cpp")
    3. agent_query("Generate Agent: agent.h/cpp")
    4. agent_query("Generate tests: test_*.cpp")

Step 2-5: Execute agent_query calls (each sub-agent has 32K fresh context)
  Messages: 3-5, Tokens: 3,500-4,500 (stays low)
  [No warnings triggered]

Step 6: Build and test
  Messages: 6, Tokens: 5,000
  Result: Success, no context issues
```

**Peak usage: ~5K tokens (15% of limit)**

#### Without RAC (What Happened in Round 2):

```
Step 1-5: Read files, create structure
  Messages: 10, Tokens: 12,000

Step 10: Wrote 5 headers
  Messages: 20, Tokens: 22,000

Step 15: Hit write_file truncation
  Messages: 30, Tokens: 28,000
  ⚠️  WARNING displayed: "75% of limit"

Step 20: Error recovery loop
  Messages: 40, Tokens: 31,500
  🚨 CRITICAL displayed: "93.75% of limit"

Step 21: Exceeded limit
  Messages: 42, Tokens: 33,200
  ❌ LIMIT EXCEEDED → Emergency compaction
  Compacted to 7 messages, 11,000 tokens

Step 22-41: Continue with compacted context
  [May hit limit again if errors persist]
```

**Peak usage: 33,200 tokens → compacted to 11,000 → likely to hit again**

---

## Testing Plan

### Test 1: Warning Triggers

```python
def test_context_warning_at_75_percent():
    agent = Agent(max_input_tokens=32768)

    # Create 24,576+ tokens worth of messages
    large_msg = {"role": "user", "content": "x" * 98304}  # ~24,576 tokens
    messages = [large_msg]

    # Should trigger warning
    agent._check_context_size(messages, "", step_num=1)

    # Verify warning was shown
    assert "warning" in agent._context_warnings_shown
```

### Test 2: Emergency Compaction

```python
def test_emergency_compaction_when_limit_exceeded():
    agent = Agent(max_input_tokens=32768)

    # Create 35K tokens worth of messages (exceeds limit)
    messages = [{"role": "user", "content": "x" * 1000}] * 35  # ~35K tokens

    # Should trigger compaction
    needs_compaction = agent._check_context_size(messages, "", step_num=1)
    assert needs_compaction == True

    # Compact
    compacted = agent._compact_messages_to_limit(messages, "")

    # Verify under limit
    assert agent._estimate_message_tokens(compacted) < 32768
    # Verify keeps first and last
    assert compacted[0] == messages[0]
    assert compacted[-1] == messages[-1]
```

### Test 3: Never Triggers with Proper RAC

```python
def test_rac_decomposition_stays_under_limit():
    agent = GaiaCodeAgent()

    # Task that requires >5 files (should trigger agent_query)
    result = agent.process_query(
        "Port GAIA Python framework to C++ (19 files)",
        max_steps=20
    )

    # Should never trigger warnings if properly decomposed
    assert "warning" not in agent._context_warnings_shown
    assert "emergency" not in agent._context_warnings_shown
```

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `src/gaia/agents/base/agent.py` | Added context limits, monitoring, compaction | +165 |
| `src/gaia/agents/gaia_code/agent.py` | Default model: opus → sonnet-4-6 | 2 changed |
| `src/gaia/agents/gaia_code/tools.py` | Enhanced agent_query description | +40 |

---

## Key Design Decisions

### 1. Conservative Token Estimation

Uses `1 token ≈ 4 chars` which **overestimates** token count. This ensures we trigger warnings BEFORE hitting actual API limits, giving the agent time to course-correct.

### 2. Logged as ERROR Level

Emergency compaction logs at `ERROR` level, not `WARNING`, because:
- It should NEVER happen with proper design
- When it does happen, it indicates a fundamental flaw
- We want it to be highly visible in logs for debugging

### 3. Compaction, Not Pruning

Uses **summarization** (compacts many messages into summary) rather than **pruning** (drops messages entirely). This preserves information, just in condensed form.

### 4. Warnings Only Show Once

`self._context_warnings_shown` prevents spam. Each threshold warning shows exactly once per session.

### 5. Configurable Limit

The `max_input_tokens` parameter allows overriding for:
- Testing (set to 1000 to force compaction)
- Large context models (set to 100K if using 1M context Claude)
- Conservative agents (set to 16K for safety)

But **default stays at 32K** to enforce good decomposition habits.

---

## Integration with RAC

The monitoring system **encourages proper RAC usage**:

1. **At 75%:** Suggests using agent_query
2. **At 93.75%:** STRONGLY RECOMMENDS agent_query
3. **At 100%:** States "This should NEVER happen" and logs why

Combined with the enhanced planning guidance (prioritize agent_query for >5 files), this creates a feedback loop:
- Agent creates linear plan → context grows → warning appears → (ideally) agent should re-plan with agent_query
- But if agent ignores warnings → compaction activates → logs full diagnostics for debugging

---

## Expected Log Output (Proper RAC)

```
[INFO] Processing query: "Port GAIA to C++ (19 files)"
[DEBUG] Step 1: Creating plan
[DEBUG] Plan created with 4 steps (all agent_query calls)
[DEBUG] Step 1 token usage: 2,450 tokens (7.5% of limit)
[DEBUG] Step 2: Executing agent_query for headers
[DEBUG] Step 2 token usage: 3,100 tokens (9.5% of limit)
[DEBUG] Step 3: Executing agent_query for MCP client
[DEBUG] Step 3 token usage: 3,800 tokens (11.6% of limit)
[DEBUG] Step 4: Executing agent_query for Agent class
[DEBUG] Step 4 token usage: 4,200 tokens (12.8% of limit)
[DEBUG] Step 5: Executing agent_query for tests
[DEBUG] Step 5 token usage: 4,900 tokens (15.0% of limit)
[DEBUG] Step 6: Build and test
[DEBUG] Step 6 token usage: 5,200 tokens (15.9% of limit)
[INFO] Task complete. Peak context usage: 5,200/32,768 tokens (15.9%)
```

**No warnings = proper design ✅**

---

## Expected Log Output (Improper Design - Sequential)

```
[INFO] Processing query: "Port GAIA to C++ (19 files)"
[DEBUG] Step 1: Creating plan
[DEBUG] Plan created with 19 steps (all write_file calls)
[WARNING] Large plan without agent_query detected (19 steps)
[DEBUG] Step 1-10 token usage: 18,500 tokens (56.5% of limit)
[WARNING] [CONTEXT] At 75.0% of limit (24,650 tokens at step 12)
⚠️  Context approaching limit: 24,650/32,768 tokens (75.2%)
   Consider using agent_query() to decompose remaining work...

[DEBUG] Step 15: write_file truncation error
[ERROR] [CONTEXT] At CRITICAL level: 93.8% (30,720 tokens at step 17)
🚨 CRITICAL: Context near limit: 30,720/32,768 tokens (93.8%)
   Next LLM call may fail...

[ERROR] [CONTEXT] LIMIT EXCEEDED at step 18: 33,100 tokens
[ERROR] [CONTEXT] This should NEVER happen with proper task decomposition.
[ERROR] [CONTEXT] Emergency compaction activating...
[ERROR] [CONTEXT] EMERGENCY COMPACTION TRIGGERED
[ERROR] [CONTEXT] Messages before compaction: 36
[ERROR] [CONTEXT] Compaction complete: Removed 29 messages, kept 7
⚠️  Emergency compaction: 29 messages summarized
   This should NEVER happen. Use agent_query() for large tasks.
```

**Warnings + ERROR logs = improper design detected ⚠️**

---

## Comparison with Previous Behavior

| Aspect | Before | After |
|--------|--------|-------|
| **Hard limit** | None (crashed at 200K) | 32K tokens (configurable) |
| **Monitoring** | None | Warns at 75%, 93.75% |
| **Warnings** | None | User-facing + logger warnings |
| **Fallback** | Crash | Emergency compaction with logging |
| **Diagnostics** | None | Detailed ERROR logs when limit hit |
| **Guidance** | Generic | Specific: "Use agent_query()" |

---

## Integration with Other Fixes

Works together with:

1. **RAC-first planning guidance** (already implemented)
   - Planning prompt now prioritizes agent_query for complex tasks
   - Prevents hitting the limit in the first place

2. **Enhanced agent_query description** (already implemented)
   - Clear examples of when to use it
   - Explains fresh context benefit

3. **Plan display** (already implemented)
   - Shows plan before execution
   - User can see if plan uses agent_query or not

Together, these changes create a **defense in depth**:
- Layer 1: Prompt guides proper RAC usage (prevents problem)
- Layer 2: Monitoring warns when approaching limit (early detection)
- Layer 3: Emergency compaction handles edge cases (safety net)

---

## Testing Checklist

- [ ] Test warning triggers at 75% threshold
- [ ] Test emergency alert at 93.75% threshold
- [ ] Test emergency compaction at 100% threshold
- [ ] Test that proper RAC usage never triggers warnings
- [ ] Test compaction preserves first and last messages
- [ ] Test compaction summary is coherent
- [ ] Test configurable max_input_tokens parameter
- [ ] Run C++ benchmark with all fixes enabled

---

## Next Steps

1. **Commit these changes** (recommended message below)
2. **Test with synthetic long conversation** (50+ steps without agent_query)
3. **Verify warnings appear at expected thresholds**
4. **Run Round 5 benchmark** when API credits available
5. **Verify no warnings triggered** (indicating proper RAC usage)

---

## Recommended Commit Message

```
GAIA Code: Add hard 32K input context limit with monitoring

- Add DEFAULT_MAX_INPUT_TOKENS = 32768 (configurable)
- Implement _check_context_size() with 75% and 93.75% warnings
- Add emergency _compact_messages_to_limit() as fallback
- Log detailed diagnostics when limit is approached/exceeded
- Encourage agent_query() decomposition at all warning levels

Philosophy: Agents should decompose tasks (via agent_query) to stay
under 32K. Hitting warnings indicates improper task design.

Emergency compaction is a safety net that should NEVER trigger with
proper RAC usage.
```

---

## Configuration Examples

### Strict (Force Early Decomposition)
```python
agent = GaiaCodeAgent(
    max_input_tokens=16384,  # 16K limit forces aggressive decomposition
)
# Will warn at 12K, critical at 15K
```

### Lenient (For Testing)
```python
agent = GaiaCodeAgent(
    max_input_tokens=65536,  # 64K limit allows more sequential work
)
# Will warn at 49K, critical at 61K
```

### Default (Recommended)
```python
agent = GaiaCodeAgent()  # Uses 32K default
# Will warn at 24K, critical at 30K
```

---

## Comparison with Claude Code

| Feature | Claude Code | GAIA Code (After Fix) |
|---------|-------------|----------------------|
| **Hard limit** | Unknown (auto-compression) | 32K tokens (configurable) |
| **Monitoring** | Silent (automatic) | Explicit warnings at 75%, 93.75% |
| **Fallback** | Automatic summarization | Emergency compaction with diagnostics |
| **User feedback** | None (transparent) | Visible warnings encouraging decomposition |
| **Logging** | Not observable | Detailed ERROR logs with token counts |

**GAIA Code is more transparent** — users and developers can see when limits are approached and why.

---

## Success Metrics

After this implementation, successful agent runs should show:

```
grep "\[CONTEXT\]" agent.log
[Should be empty or only show single-digit percentage logs]
```

If you see:
- `[WARNING] [CONTEXT]` → Task could be better decomposed
- `[ERROR] [CONTEXT]` → Task design is flawed, must use agent_query
- `EMERGENCY COMPACTION` → Fundamental failure, requires task redesign

The goal: **Zero context warnings** for well-designed tasks.
