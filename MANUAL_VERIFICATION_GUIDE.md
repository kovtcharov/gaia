# GAIA Code: Manual Verification Guide

**Complete guide to test every feature manually**

---

## Setup

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY=your_key  # Or already in .env
```

---

## 1. Basic Functionality

### Test 1.1: Agent Starts

```bash
gaia code --help
```

**Expected**:
- Shows help with all options
- No errors

**Validates**: CLI integration

---

### Test 1.2: Simple Query

```bash
gaia code "What is 5+5?" --claude --tui off
```

**Expected**:
- Credential check passes
- Claude responds with "10"
- Clean exit

**Validates**: Basic LLM connection

---

## 2. File Operations

### Test 2.1: List Files

```bash
gaia code "List all Python files in current directory" --claude --tui off
```

**Expected**:
- Uses `list_files` tool
- Returns actual file list
- Shows Python files

**Validates**: File I/O tools

---

### Test 2.2: Read File

```bash
gaia code "Read setup.py and tell me the project name" --claude --tui off
```

**Expected**:
- Uses `read_file` or `cat` command
- Reads actual file content
- Reports "amd-gaia"

**Validates**: File reading

---

### Test 2.3: Create File

```bash
gaia code "Create hello.py with print('Hello GAIA')" --claude --tui off
```

**Expected**:
- Uses `write_file` or `write_python_file`
- Creates hello.py
- File exists on disk

**Verify**:
```bash
cat hello.py
# Should show: print('Hello GAIA')
```

**Validates**: File creation

---

### Test 2.4: Execute Code

```bash
gaia code "Create and run a script that calculates 7*8" --claude --tui off
```

**Expected**:
- Creates file
- Executes it
- Returns "56"

**Validates**: Code execution

---

## 3. Testing Features

### Test 3.1: Create with Tests

```bash
gaia code "Create calculator.py with add/subtract/multiply/divide, and test_calculator.py with pytest tests" --claude --tui off
```

**Expected**:
- Creates calculator.py
- Creates test_calculator.py
- Files exist on disk

**Verify**:
```bash
ls -l calculator.py test_calculator.py
```

**Validates**: Multi-file creation

---

### Test 3.2: Run Tests

```bash
gaia code "Run the tests in test_calculator.py" --claude --tui off
```

**Expected**:
- Uses `run_tests` tool
- Executes pytest
- Shows test results

**Validates**: Test execution

---

## 4. Knowledge Database

### Test 4.1: Store Insight

```bash
gaia code "Store this insight: 'Use type hints for better code clarity' category=pattern" --claude --tui off --workspace /tmp/test_gaia
```

**Expected**:
- Uses `store_insight` tool
- Returns insight_id
- Success message

**Validates**: Knowledge storage

---

### Test 4.2: Recall Insight

```bash
gaia code "Recall insights about type hints" --claude --tui off --workspace /tmp/test_gaia
```

**Expected**:
- Uses `recall` tool
- Finds 1 result
- Shows stored insight

**Validates**: Knowledge retrieval, FTS5 search

---

### Test 4.3: Cross-Session Memory

```bash
# Session 1
gaia code "Store: Always use parameterized queries for SQL" --workspace /tmp/test_gaia

# Session 2 (new command)
gaia code "Recall insights about SQL" --workspace /tmp/test_gaia
```

**Expected**:
- Second session finds insight from first
- Proves persistence

**Validates**: Cross-session persistence

---

## 5. Codebase Analysis (M7)

### Test 5.1: Index Repository

```bash
gaia code "Index this codebase and tell me how many Python files there are" --claude --tui off
```

**Expected**:
- Uses `index_codebase` tool
- Scans all Python files
- Returns count

**Validates**: M7 codebase indexing

---

### Test 5.2: Find Symbol

```bash
gaia code "Find where the GaiaCodeAgent class is defined" --claude --tui off
```

**Expected**:
- Uses `find_symbol` tool
- Returns file path and line number

**Validates**: Symbol extraction

---

### Test 5.3: Analyze Architecture

```bash
gaia code "Analyze the architecture of this codebase" --claude --tui off
```

**Expected**:
- Uses `analyze_architecture` tool
- Shows module structure
- Lists dependencies

**Validates**: Architecture analysis

---

### Test 5.4: Detect Issues

```bash
gaia code "Detect any code issues in this repository" --claude --tui off
```

**Expected**:
- Uses `detect_issues` tool
- Finds circular deps, missing docs, etc.
- Returns issue list

**Validates**: Issue detection

---

## 6. Personas

### Test 6.1: Torvalds (Brutal Honesty)

```bash
gaia code "Should I use global variables?" --persona torvalds --claude --tui off
```

**Expected**:
- Harsh, direct response
- "No." or similar
- Explains why it's bad

**Validates**: Torvalds persona

---

### Test 6.2: Knuth (Thorough Teacher)

```bash
gaia code "Explain what Big O notation is" --persona knuth --claude --tui off
```

**Expected**:
- Detailed, pedagogical explanation
- Examples and theory
- Thorough coverage

**Validates**: Knuth persona

---

### Test 6.3: Pike (Simplicity)

```bash
gaia code "Create a user authentication system" --persona pike --claude --tui off
```

**Expected**:
- Simple, minimal solution
- Pushback on over-engineering
- "Keep it simple" philosophy

**Validates**: Pike persona

---

## 7. Interactive Mode

### Test 7.1: Start Interactive Session

```bash
gaia code -i --persona pike
```

**Expected**:
- Welcome message
- Prompt appears
- Can type and chat

**Type**: `Create a simple calculator`

**Expected**:
- Agent responds
- Creates code
- Continues conversation

**Type**: `/exit`

**Validates**: Interactive chat mode

---

## 8. Planning Features

### Test 8.1: Multi-Step Task

```bash
gaia code "Create a web scraper with requests and BeautifulSoup, include tests" --claude --tui off
```

**Expected**:
- Creates multi-step plan
- Shows plan before executing
- Executes each step
- Creates scraper.py and tests

**Validates**: Planning system

---

## 9. Quality Gates

### Test 9.1: Syntax Errors

```bash
# Create file with syntax error
echo "def broken(\n    print('missing colon')" > broken.py

gaia code "Fix the syntax errors in broken.py" --claude --tui off
```

**Expected**:
- Detects syntax error
- Fixes it
- Validates with syntax gate

**Validates**: SyntaxGate

---

## 10. Web Search

### Test 10.1: Search Web

```bash
gaia code "Search the web for: how to use FastAPI websockets" --claude --tui off
```

**Expected**:
- Uses `search_web` tool (Perplexity)
- Returns current information
- Provides answer

**Validates**: Perplexity integration

---

## 11. Advanced Tools

### Test 11.1: Agent Query (RAC)

```bash
gaia code "Use agent_query to delegate a subtask: calculate fibonacci(10)" --claude --tui off
```

**Expected**:
- Uses `agent_query` tool
- Tracks in call stack
- Returns result

**Validates**: RAC mechanism

---

### Test 11.2: Get Plan

```bash
gaia code "Show me the current execution plan" --claude --tui off
```

**Expected**:
- Uses `get_plan` tool
- Returns task list
- Shows progress

**Validates**: Plan tracking

---

## 12. Execution & Observation

### Test 12.1: Run and Observe

```bash
gaia code "Create a script that prints numbers 1-5, then run it and verify the output" --claude --tui off
```

**Expected**:
- Creates script
- Executes it
- Observes output
- Verifies correctness

**Validates**: Execution observer

---

## 13. Database Inspection

### Test 13.1: Check Databases Exist

```bash
ls -lh ~/.gaia/workspace/
```

**Expected**:
```
memory.db
knowledge.db
tools.db
skills.db
agents.db
plan.db
```

**Validates**: All 7 databases created

---

### Test 13.2: Query Database Directly

```bash
sqlite3 ~/.gaia/workspace/knowledge.db "SELECT count(*) FROM insights;"
```

**Expected**:
- Shows count of stored insights
- Should match number of store_insight calls

**Validates**: Database actually storing data

---

## 14. TUI Modes

### Test 14.1: Simple TUI

```bash
gaia code "Create hello world" --persona pike --tui simple
```

**Expected**:
- Clean progress bar
- Shows stages
- Minimal output
- No log spam

**Validates**: Simple TUI mode

---

### Test 14.2: Full TUI

```bash
gaia code "Create calculator" --persona pike --tui full
```

**Expected**:
- Multi-panel layout
- Header, main, sidebar
- Quality gates visible
- Plan shown

**Validates**: Full TUI mode

---

### Test 14.3: Minimal TUI

```bash
gaia code "What is 10*10" --persona pike --tui minimal
```

**Expected**:
- Single line output
- Updates in place
- Very minimal

**Validates**: Minimal TUI mode

---

## 15. Error Recovery

### Test 15.1: Handles Missing Dependencies

```bash
gaia code "Use a library that doesn't exist: import nonexistent_lib" --claude --tui off
```

**Expected**:
- Detects import error
- Suggests installing package or alternative
- Recovers gracefully

**Validates**: Error recovery

---

## 16. Credential Management

### Test 16.1: Credential Check on Startup

```bash
# Rename .env temporarily
mv .env .env.bak
gaia code "test"
# Should prompt for API key
mv .env.bak .env
```

**Expected**:
- Detects missing API key
- Prompts user to enter it
- Offers to save to credentials.json

**Validates**: Credential management

---

## Validation Checklist

After running all tests, verify:

### Core Functionality
- [ ] Agent starts and responds
- [ ] CLI commands work
- [ ] Tools execute

### File Operations
- [ ] List files works
- [ ] Read files works
- [ ] Create files works
- [ ] Execute code works

### Testing
- [ ] Can create tests
- [ ] Can run pytest
- [ ] Tests actually pass

### Knowledge/Memory
- [ ] store_insight works
- [ ] recall works
- [ ] Persists across sessions

### Codebase Analysis (M7)
- [ ] Index repository works
- [ ] Find symbols works
- [ ] Analyze architecture works
- [ ] Detect issues works

### Personas
- [ ] All 8 personas available
- [ ] Different communication styles
- [ ] Authentic voices

### Interactive Mode
- [ ] Chat mode starts
- [ ] Context maintained
- [ ] Commands work (/help, /exit, etc.)

### Planning
- [ ] Multi-step plans created
- [ ] Plans shown to user
- [ ] Steps executed in order

### Quality Gates
- [ ] Syntax checking works
- [ ] Import checking works
- [ ] Test running works

### TUI
- [ ] All 3 modes work
- [ ] Clean output
- [ ] No log spam

### Error Recovery
- [ ] Recovers from failures
- [ ] Tries alternatives
- [ ] Doesn't crash

---

## Expected Results Summary

After all tests:
- ✅ 95+ tools functional
- ✅ All 7 databases created and working
- ✅ Knowledge persists across sessions
- ✅ Multi-step tasks complete successfully
- ✅ Error recovery works
- ✅ All personas respond appropriately
- ✅ Interactive mode functional
- ✅ TUI displays correctly

---

## If Issues Found

For each failing test:
1. Note the exact error
2. Check logs (if --debug used)
3. Verify prerequisites (API keys, etc.)
4. Report issue with:
   - Command run
   - Expected result
   - Actual result
   - Error message

---

**Status**: Use this guide to comprehensively validate GAIA Code

**All tests passing = Fully functional autonomous agent!**
