# GAIA Code Context Management Fix Plan

**Date:** 2026-02-23
**Problem:** GAIA Code agent hit 200,817 tokens at step 41, exceeding Claude Opus's 200K limit
**Goal:** Implement database-backed context management to handle unlimited conversation length

---

## Current State Analysis

### What Exists Already

✅ **SharedAgentState with 7 databases** (`shared_state.py`):
1. `memory.db` - Session-scoped cache (file contents, tool results)
2. `knowledge.db` - Cross-session insights
3. `tools.db` - Tool registry with FTS5 search
4. `skills.db` - Learned workflows
5. `agents.db` - Specialist registry
6. `plan` - Hierarchical task tree
7. `audit_log` - Action audit trail

✅ **GaiaCodeAgent enables SharedState** (line 197):
```python
kwargs["enable_shared_state"] = True
kwargs["workspace_dir"] = str(workspace_dir) if workspace_dir else None
```

✅ **MemoryDB has storage methods**:
- `store_file_content(path, content)` - Caches read files
- `get_file_content(path)` - Retrieves with cache check
- `store_tool_result(tool_name, args, result)` - Stores tool outputs
- `retrieve_tool_result(tool_name, args)` - Retrieves cached results

❌ **What's MISSING:** The `messages` array in `process_query()` doesn't use these databases — it accumulates unbounded in-memory.

---

## Problem: Dual Message Arrays

GAIA Code agent has **TWO separate message tracking systems** that don't coordinate:

### System 1: In-Memory (Current, Problematic)

```python
# agent.py line 1607-1609
def process_query(self, user_input: str, max_steps: int = None):
    conversation = []  # LEGACY: not used anymore
    messages = []      # ACTIVE: grows unbounded to 200K+ tokens

    # Every step appends:
    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "user", "content": tool_result})  # Full result!
    messages.append({"role": "user", "content": error_prompt})
    # ... 41 steps → 200K tokens
```

### System 2: Database (Available, Unused)

```python
# shared_state.py line 240-250
class MemoryDB:
    def store_tool_result(self, tool_name: str, args: Dict, result: str):
        """Store a tool call result."""
        # Stores in SQLite: tool_results table
        # Can be retrieved later by tool_name + args

    # But process_query() doesn't call this!
```

**The disconnect:** `process_query()` builds `messages` from scratch each step and sends the entire array to the LLM. It never stores/retrieves from MemoryDB.

---

## Solution: Database-Backed Context Compression

### Architecture

Instead of sending full tool results to the LLM, send:
1. **Tool result summary** (first 500 chars) in the messages array
2. **Reference ID** to the full result stored in MemoryDB
3. **On-demand retrieval** when the LLM needs details (via a `retrieve_tool_result` tool)

### Implementation

#### Step 1: Store Tool Results in MemoryDB

```python
# src/gaia/agents/base/agent.py

def _execute_tool(self, tool_name: str, tool_args: Dict) -> str:
    """Execute a tool and store the result in MemoryDB."""

    # Execute tool
    result = self._TOOL_REGISTRY[tool_name]["func"](**tool_args)
    result_json = json.dumps(result, indent=2)

    # Store full result in database if shared state enabled
    result_id = None
    if self.shared_state:
        result_id = str(uuid4())
        self.shared_state.memory.store_tool_result(
            tool_name=tool_name,
            args=tool_args,
            result=result_json,
            result_id=result_id  # Add this field to schema
        )

    # Return summary + reference
    if len(result_json) > 2000:  # Large result
        summary = result_json[:500] + "\n... (truncated, " + \
                  f"{len(result_json) - 500} more chars)\n" + \
                  f"[Full result stored as {result_id}]"
        return summary
    else:
        return result_json  # Small results sent in full
```

#### Step 2: Add retrieve_tool_result Tool

```python
# src/gaia/agents/base/agent.py

@tool(
    name="retrieve_tool_result",
    description="Retrieve the full result of a previous tool call by its result ID",
    parameters=[
        ToolParam(
            name="result_id",
            type="string",
            description="Result ID from a truncated tool result message",
            required=True,
        )
    ],
)
def retrieve_tool_result(self, result_id: str) -> Dict[str, Any]:
    """Retrieve a full tool result from the database."""

    if not self.shared_state:
        return {"status": "error", "error": "Shared state not enabled"}

    result = self.shared_state.memory.retrieve_tool_result_by_id(result_id)

    if not result:
        return {"status": "error", "error": f"Result {result_id} not found"}

    return {
        "status": "success",
        "tool_name": result["tool_name"],
        "result": result["result"],
        "timestamp": result["timestamp"]
    }
```

#### Step 3: Update MemoryDB Schema

```python
# src/gaia/agents/base/shared_state.py

class MemoryDB:
    def _init_db(self):
        # Add result_id and timestamp to tool_results table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_results (
                result_id TEXT PRIMARY KEY,
                tool_name TEXT NOT NULL,
                args TEXT NOT NULL,
                result TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT
            )
        """)

    def store_tool_result(self, tool_name: str, args: Dict, result: str,
                          result_id: str = None, session_id: str = None):
        """Store a tool result with unique ID."""
        if not result_id:
            result_id = str(uuid4())

        self.conn.execute("""
            INSERT INTO tool_results (result_id, tool_name, args, result, timestamp, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (result_id, tool_name, json.dumps(args), result,
              datetime.now().isoformat(), session_id))
        self.conn.commit()
        return result_id

    def retrieve_tool_result_by_id(self, result_id: str) -> Optional[Dict]:
        """Retrieve a tool result by its ID."""
        cursor = self.conn.execute("""
            SELECT tool_name, args, result, timestamp
            FROM tool_results WHERE result_id = ?
        """, (result_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "tool_name": row[0],
            "args": json.loads(row[1]),
            "result": row[2],
            "timestamp": row[3]
        }
```

#### Step 4: Hard Input Limit as Safety Net

Even with compression, add a hard cap matching Claude Code's approach:

```python
# src/gaia/agents/base/agent.py

MAX_INPUT_CONTEXT_TOKENS = 150_000  # Safe limit for 200K models

def _estimate_message_tokens(messages: List[Dict]) -> int:
    """Estimate tokens (rough: 1 token ≈ 4 chars)."""
    total_chars = sum(len(json.dumps(m)) for m in messages)
    return total_chars // 4

def _prune_to_token_limit(messages: List[Dict], max_tokens: int) -> List[Dict]:
    """Keep most recent messages that fit within token limit."""

    if len(messages) <= 3:  # Always keep minimal context
        return messages

    # Always keep: [0]=first user query, [-2]=last assistant, [-1]=last user
    essential = [messages[0], messages[-2], messages[-1]]
    essential_tokens = _estimate_message_tokens(essential)

    budget = max_tokens - essential_tokens
    keep = essential.copy()

    # Add middle messages from newest to oldest until budget exhausted
    for msg in reversed(messages[1:-2]):
        msg_tokens = _estimate_message_tokens([msg])
        if essential_tokens + msg_tokens > budget:
            break
        keep.insert(-2, msg)  # Insert before last 2
        essential_tokens += msg_tokens

    pruned_count = len(messages) - len(keep)
    if pruned_count > 0:
        logger.info(f"Pruned {pruned_count} messages (estimated {pruned_count * 5000} tokens)")

    return keep

# In process_query, before LLM calls:
if _estimate_message_tokens(messages) > MAX_INPUT_CONTEXT_TOKENS:
    messages = _prune_to_token_limit(messages, MAX_INPUT_CONTEXT_TOKENS)
```

---

## Expected Impact

### Before (Current Behavior)

```
Step 1:  2K tokens
Step 10: 85K tokens
Step 20: 170K tokens
Step 30: 255K tokens ❌ CRASH
```

### After (With Compression)

```
Step 1:  2K tokens (full result)
Step 10: 10K tokens (9 summaries + 1 full recent result)
Step 20: 15K tokens (19 summaries + 1 full recent result)
Step 30: 15K tokens (29 summaries + 1 full recent result)
Step 100: 20K tokens (99 summaries + 1 full recent result)
```

**Why this works:**

- **Summary overhead:** 500 chars per tool result = ~125 tokens
- **100 steps:** 100 × 125 = 12,500 tokens for summaries
- **Recent full results:** Last 3 steps × 5K = 15K tokens
- **Total:** ~30K tokens (well under 150K limit)

---

## Alternative: Conversational Checkpointing

Instead of keeping all messages, periodically summarize and restart:

```python
def _checkpoint_conversation(self, messages: List[Dict]) -> List[Dict]:
    """Every 10 steps, summarize history and restart with fresh context."""

    if len(messages) < 20:  # Don't checkpoint early
        return messages

    # Extract key facts from history
    summary_prompt = f"""
    Summarize the key accomplishments and current state from this conversation:
    {json.dumps(messages[:-5], indent=2)}

    Format: bullet list of files created, decisions made, blockers encountered.
    """

    summary = self.chat.send_messages(
        messages=[{"role": "user", "content": summary_prompt}],
        system_prompt="You summarize conversation history concisely."
    ).text

    # New context: summary + recent messages
    return [
        {"role": "system", "content": f"Previous work summary:\n{summary}"},
        *messages[-5:]  # Keep last 5 messages
    ]

# In process_query loop:
if steps_taken % 10 == 0 and steps_taken > 0:
    messages = _checkpoint_conversation(messages)
```

---

## Recommended Approach

**Hybrid strategy:**

1. **Primary:** Database-backed summaries (elegant, scalable)
2. **Backup:** Hard token limit with pruning (safety net)
3. **Future:** Conversational checkpointing (for ultra-long runs)

**Implementation priority:**

1. **HIGH:** Add `_prune_to_token_limit()` with 150K hard cap (2 hours to implement)
2. **HIGH:** Store tool results in MemoryDB with summaries (4 hours to implement)
3. **MEDIUM:** Add `retrieve_tool_result` tool (1 hour to implement)
4. **LOW:** Conversational checkpointing (research needed)

---

## Testing Plan

### Test 1: Synthetic Long Run

Create a test that forces 50+ steps to validate context management:

```python
def test_long_conversation_context_limit():
    agent = GaiaCodeAgent(workspace_dir="/tmp/test")

    # Query that requires 50+ tool calls
    query = "Read all 50 Python files in src/gaia/agents/ and list their classes"

    result = agent.process_query(query)

    # Should complete without context error
    assert "200817 tokens" not in result
    assert "too long" not in result
```

### Test 2: Database Storage Verification

```python
def test_tool_result_database_storage():
    agent = GaiaCodeAgent(workspace_dir="/tmp/test")

    # Generate large tool result
    result = agent._execute_tool("read_file", {"file_path": "large_file.py"})

    # Verify it was stored
    assert agent.shared_state.memory.retrieve_tool_result_by_id(result_id)

    # Verify summary was sent to LLM
    assert len(result) < 2000  # Should be truncated
    assert "[Full result stored as" in result
```

### Test 3: Context Pruning

```python
def test_message_pruning_at_limit():
    messages = [{"role": "user", "content": "x" * 50000}] * 20  # 250K tokens

    pruned = _prune_to_token_limit(messages, MAX_INPUT_CONTEXT_TOKENS)

    # Should fit under limit
    assert _estimate_message_tokens(pruned) < MAX_INPUT_CONTEXT_TOKENS

    # Should keep first and last
    assert pruned[0] == messages[0]
    assert pruned[-1] == messages[-1]
```

---

## Code Changes Required

### File 1: `src/gaia/agents/base/shared_state.py`

**Changes:**
1. Add `result_id` column to `tool_results` table
2. Add `retrieve_tool_result_by_id(result_id)` method
3. Update `store_tool_result()` to accept and return `result_id`

**Lines affected:** ~30 lines modified, ~20 lines added

---

### File 2: `src/gaia/agents/base/agent.py`

**Changes:**
1. Add `MAX_INPUT_CONTEXT_TOKENS = 150_000` constant
2. Add `_estimate_message_tokens()` helper
3. Add `_prune_to_token_limit()` method
4. Add `_summarize_tool_result()` method (creates 500-char summary + result ID)
5. Modify `_execute_tool()` to store results in MemoryDB and return summary
6. Add pruning call before `send_messages()` (lines 2023, 2104)
7. Add `retrieve_tool_result` tool registration

**Lines affected:** ~100 lines modified, ~80 lines added

---

### File 3: `src/gaia/chat/sdk.py`

**Changes:**
None needed — the ChatSDK's `max_history_length` works correctly for its internal state. The issue is the externally-provided `messages` array.

---

### File 4: `tests/unit/test_shared_state.py`

**Changes:**
1. Add test for `store_tool_result()` with `result_id`
2. Add test for `retrieve_tool_result_by_id()`

**Lines affected:** ~40 lines added

---

### File 5: `tests/unit/test_agent.py`

**Changes:**
1. Add test for `_estimate_message_tokens()`
2. Add test for `_prune_to_token_limit()`
3. Add test for tool result summarization
4. Add integration test for long conversation (50+ steps)

**Lines affected:** ~100 lines added

---

## Implementation Example

### Elegant Tool Result Handling

```python
# src/gaia/agents/base/agent.py

def _execute_tool(self, tool_name: str, tool_args: Dict) -> str:
    """Execute a tool and return a context-efficient result."""

    try:
        # Execute the tool
        result = self._TOOL_REGISTRY[tool_name]["func"](**tool_args)
        result_json = json.dumps(result, indent=2)

        # For small results, return as-is
        if len(result_json) <= 2000:
            return result_json

        # For large results (e.g., read_file), use database + summary
        if self.shared_state:
            result_id = self.shared_state.memory.store_tool_result(
                tool_name=tool_name,
                args=tool_args,
                result=result_json
            )

            # Create intelligent summary based on tool type
            if tool_name == "read_file":
                summary = self._summarize_file_content(result_json, tool_args.get("file_path"))
            elif tool_name == "run_cli_command":
                summary = self._summarize_command_output(result_json)
            else:
                summary = result_json[:500] + "\n...(truncated)"

            return (
                f"{summary}\n\n"
                f"📎 Full result: {len(result_json):,} chars stored as `{result_id}`\n"
                f"💡 Use retrieve_tool_result(result_id=\"{result_id}\") to access full content"
            )

        # Fallback if no shared state: simple truncation
        return result_json[:2000] + f"\n...(truncated {len(result_json) - 2000} chars)"

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return json.dumps({"status": "error", "error": str(e)})

def _summarize_file_content(self, content: str, file_path: str) -> str:
    """Create intelligent summary of file content."""

    data = json.loads(content)
    if "content" in data:
        lines = data["content"].split("\n")
        return (
            f"File: {file_path}\n"
            f"Lines: {len(lines):,}\n"
            f"Size: {len(data['content']):,} chars\n"
            f"Preview (first 10 lines):\n" +
            "\n".join(f"  {i+1:4d} | {line[:80]}" for i, line in enumerate(lines[:10]))
        )
    return content[:500]

def _summarize_command_output(self, content: str) -> str:
    """Create intelligent summary of command output."""

    data = json.loads(content)
    stdout = data.get("stdout", "")
    stderr = data.get("stderr", "")

    lines = stdout.split("\n") if stdout else []
    if len(lines) > 20:
        return (
            f"Command output: {len(lines)} lines\n"
            f"First 5 lines:\n" + "\n".join(f"  {line[:80]}" for line in lines[:5]) +
            f"\n\nLast 5 lines:\n" + "\n".join(f"  {line[:80]}" for line in lines[-5:])
        )
    return content[:1000]
```

---

## Message Structure Comparison

### Before (Unbounded)

```json
[
  {"role": "user", "content": "Read agent.py and port to C++"},
  {"role": "assistant", "content": "{\"thought\": \"...\", \"tool\": \"read_file\", ...}"},
  {"role": "user", "content": "Tool 'read_file' returned: <108KB of Python code>"},
  {"role": "assistant", "content": "{\"thought\": \"...\", \"tool\": \"write_file\", ...}"},
  {"role": "user", "content": "Tool 'write_file' returned: <20KB C++ code>"},
  ... (40 more of these → 200K tokens)
]
```

### After (Database-Backed)

```json
[
  {"role": "user", "content": "Read agent.py and port to C++"},
  {"role": "assistant", "content": "{\"thought\": \"...\", \"tool\": \"read_file\", ...}"},
  {"role": "user", "content": "File: agent.py\nLines: 2,465\nPreview:\n  1 | # Copyright...\n  2 | import json\n...\n\n📎 Full result: 108,077 chars stored as `abc123`"},
  {"role": "assistant", "content": "{\"thought\": \"...\", \"tool\": \"write_file\", ...}"},
  {"role": "user", "content": "File written: agent.cpp (20KB)\n📎 Full result stored as `def456`"},
  ... (100 more of these → 30K tokens)
]
```

**Token savings:** 200K → 30K (85% reduction)

---

## Rollout Plan

### Phase 1: Hard Limit (1 day)
- Implement `_prune_to_token_limit()` with 150K cap
- Add to both streaming and non-streaming LLM calls
- Test with synthetic long conversation
- **Impact:** Prevents crashes, but loses context

### Phase 2: Database Storage (2 days)
- Update MemoryDB schema with `result_id`
- Modify `_execute_tool()` to store and summarize
- Add intelligent summarizers for read_file, run_cli_command
- Test with C++ benchmark
- **Impact:** Maintains full context via database

### Phase 3: Retrieval Tool (1 day)
- Implement `retrieve_tool_result` tool
- Update system prompt to explain retrieval mechanism
- Test retrieval workflow
- **Impact:** LLM can access full results on-demand

### Phase 4: Validation (1 day)
- Run full C++ benchmark (Round 5) with all fixes
- Compare against Claude Code reference
- Measure token usage across 19-file generation
- **Success criterion:** Complete without context errors

---

## Expected Benchmark Results (Round 5)

With all fixes:

| Metric | Round 4 (Failed) | Round 5 (Projected) |
|--------|------------------|---------------------|
| **Steps** | 4 (blocked by credits) | 25-30 (full completion) |
| **Files** | 3/19 | 19/19 |
| **Peak tokens** | N/A | ~40K (under 150K limit) |
| **Duration** | 8 minutes | 30-40 minutes |
| **Quality** | 95/100 (extrapolated) | 95+/100 (actual) |
| **Status** | Blocked | Success ✓ |

---

## Comparison with Claude Code

| Feature | GAIA Code (Current) | GAIA Code (After Fixes) | Claude Code |
|---------|---------------------|-------------------------|-------------|
| **Context limit** | Unbounded (crashes at 200K) | 150K hard cap | Automatic compression |
| **Message pruning** | None | Prune to last 20-30 | Intelligent summarization |
| **Tool result storage** | In-memory only | SQLite database | Not observable |
| **On-demand retrieval** | N/A | `retrieve_tool_result` tool | N/A (not needed) |
| **Max conversation** | ~40 steps | Unlimited | Unlimited |

After fixes, GAIA Code will match Claude Code's reliability for long conversations while adding database-backed persistence as a bonus.
