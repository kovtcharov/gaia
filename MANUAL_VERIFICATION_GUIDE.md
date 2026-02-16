# GAIA Code: Manual Verification Guide

**Complete guide to test every feature manually. Updated 2026-02-15 with real test results.**

---

## Setup

```bash
source .venv/bin/activate   # Or .venv-linux/bin/activate on WSL
export ANTHROPIC_API_KEY=your_key        # Required
export PERPLEXITY_API_KEY=your_key       # Optional: for web search
```

---

## 1. Basic Functionality

### Test 1.1: CLI Help

```bash
gaia code --help
```

**Expected**: Shows help with all options (--persona, --tui, --claude, etc.)
**Validates**: CLI integration
**Status**: PASSED (2026-02-15)

---

### Test 1.2: Simple Query

```bash
gaia code "What is 2+2?" --claude --tui off
```

**Expected**: Returns "4" with clean output
**Validates**: Basic LLM connection, response parsing
**Status**: PASSED (2026-02-15)

---

### Test 1.3: Simple TUI Mode

```bash
gaia code "What is 2+2?" --persona pike --tui simple
```

**Expected**:
- Credential status box
- Execution plan display
- Clean progress output
- Final answer with timing

**Validates**: Simple TUI mode
**Status**: PASSED (2026-02-15)

---

## 2. File Operations

### Test 2.1: Create and Execute Code

```bash
gaia code "Create and run a script that calculates 7*8" --claude --tui off
```

**Expected**:
- Uses `write_python_file` tool to create file
- Uses `execute_python_file` or `run_cli_command` to run it
- Returns "56"

**Validates**: File creation + code execution
**Status**: PASSED (2026-02-15)

---

### Test 2.2: Multi-File Python Project (Level 1)

```bash
rm -rf /tmp/calc_project
gaia code "Create a complete Python calculator project at /tmp/calc_project/ with: 1) calc/__init__.py exporting Calculator and History, 2) calc/calculator.py with Calculator class (add, subtract, multiply, divide with zero-check, power), 3) calc/history.py with History class (records with timestamps, list_records, clear), 4) main.py that demos all 5 operations and prints history, 5) tests/test_calculator.py with pytest tests. Create ALL 5 files, run the tests, and run main.py." --claude --tui off
```

**Expected**:
- Creates all 5+ files (may also create tests/__init__.py)
- Runs tests: 9/9 passing
- Runs main.py: shows all operations + history

**Verify**:
```bash
find /tmp/calc_project -name "*.py" | sort
cd /tmp/calc_project && python main.py
cd /tmp/calc_project && python -m pytest tests/ -v
```

**Validates**: Multi-file project creation, cross-file imports, test execution
**Status**: PASSED (2026-02-15) - 6 files created, 9/9 tests, main.py runs clean

---

### Test 2.3: REST API with Database (Level 2)

```bash
rm -rf /tmp/rest_api
gaia code "Create a complete REST API project at /tmp/rest_api/ with Flask and SQLite. Requirements: 1) app.py with GET/POST/PUT/DELETE /tasks endpoints, 2) models.py with task validation, 3) database.py with SQLite CRUD, 4) tests/test_api.py with tests for all endpoints. Create ALL files and run the tests." --claude --tui off
```

**Expected**:
- Creates 5 files (app.py, models.py, database.py, tests/__init__.py, tests/test_api.py)
- Installs Flask if needed (agent handles missing dependencies)
- Runs tests: 12/12 passing

**Verify**:
```bash
find /tmp/rest_api -name "*.py" | sort
cd /tmp/rest_api && python -m pytest tests/ -v
```

**Validates**: Complex multi-file project, dependency management, API code generation
**Status**: PASSED (2026-02-15) - 5 files, 12/12 tests, agent bootstrapped pip+Flask

---

### Test 2.4: Webpage Mockup (Level 2.5)

```bash
gaia code "Create a mockup webpage for a product called 'AeroFit Pro' fitness tracker at /tmp/aerofit/index.html. Include: hero section, features grid with 3 cards, pricing section with 2 tiers ($9 Basic, $99 Pro), footer. Dark theme, modern design, single HTML file with embedded CSS." --claude --tui off
```

**Expected**:
- Creates a single professional HTML file
- Dark theme with modern design
- All sections present (hero, features, pricing, footer)

**Verify**:
```bash
wc -l /tmp/aerofit/index.html  # Should be 200+ lines
# Open in browser to visually verify
```

**Validates**: Web development, HTML/CSS generation, design quality
**Status**: PASSED (2026-02-15) - 246 lines, professional design

---

## 3. Knowledge Database

### Test 3.1: Store Insight

```bash
gaia code "Store this insight: category=convention, content='Always use pathlib.Path instead of os.path for cross-platform path handling', domain='python', triggers=['path', 'pathlib', 'os.path']" --claude --tui off --workspace /tmp/test_gaia_kb
```

**Expected**:
- Uses `store_insight` tool
- Returns insight_id (UUID)
- Success status

**Validates**: Knowledge storage
**Status**: PASSED (2026-02-15)

---

### Test 3.2: Recall Insight

```bash
gaia code "Recall insights about path handling" --claude --tui off --workspace /tmp/test_gaia_kb
```

**Expected**:
- Uses `recall` tool
- Finds the stored insight
- Shows content about pathlib

**Validates**: Knowledge retrieval, FTS5 search
**Status**: PASSED (2026-02-15)

---

### Test 3.3: Cross-Session Memory

```bash
# Session 1: Store
gaia code "Store insight: category=pattern, content='Use context managers for resource cleanup'" --claude --tui off --workspace /tmp/test_gaia_kb

# Session 2: Recall (completely new process)
gaia code "Recall insights about context managers" --claude --tui off --workspace /tmp/test_gaia_kb
```

**Expected**: Second session finds the insight from the first session
**Validates**: Cross-session persistence via SQLite

---

### Test 3.4: Direct Database Inspection

After running any knowledge tests, manually verify the database contents:

```bash
# Check all databases exist (6 databases)
ls -lh /tmp/test_gaia_kb/
# Expected: agents.db, knowledge.db, memory.db, plan.db, skills.db, tools.db

# Inspect knowledge database - list all insights
python -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/knowledge.db')
print('=== INSIGHTS TABLE ===')
for row in conn.execute('SELECT id, category, domain, content, triggers, created_at FROM insights'):
    print(f'  ID: {row[0][:8]}...')
    print(f'  Category: {row[1]}')
    print(f'  Domain: {row[2]}')
    print(f'  Content: {row[3][:80]}')
    print(f'  Triggers: {row[4]}')
    print(f'  Created: {row[5]}')
    print()
count = conn.execute('SELECT count(*) FROM insights').fetchone()[0]
print(f'Total insights: {count}')

print()
print('=== FTS INDEX ===')
fts_count = conn.execute('SELECT count(*) FROM insights_fts').fetchone()[0]
print(f'FTS entries: {fts_count} (should match insights count)')
"

# Inspect tools database
python -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/tools.db')
count = conn.execute('SELECT count(*) FROM tools').fetchone()[0]
print(f'Registered tools: {count}')
for row in conn.execute('SELECT name, category, description FROM tools LIMIT 10'):
    print(f'  {row[0]} [{row[1]}]: {row[2][:60]}')
"

# Inspect plan database
python -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/plan.db')
count = conn.execute('SELECT count(*) FROM tasks').fetchone()[0]
print(f'Plan tasks: {count}')
for row in conn.execute('SELECT id, description, status FROM tasks LIMIT 10'):
    print(f'  [{row[2]}] {row[1][:60]}')
"
```

**Expected**:
- 6 databases present
- Insights count matches number of store_insight calls
- FTS index count matches insights count
- Plan shows tasks from most recent execution

**Validates**: Database persistence, data integrity, FTS5 indexing
**Status**: PASSED (2026-02-15) - All databases verified with correct data

---

## 4. Web Search (Perplexity)

### Test 4.1: Search Web

```bash
gaia code "Search the web for: what is the latest Python version" --claude --tui off
```

**Expected**:
- Uses `search_web` tool
- Calls Perplexity API directly (not MCP subprocess)
- Returns current, accurate information
- Includes citations/sources

**Validates**: Perplexity direct HTTP API integration
**Status**: PASSED (2026-02-15) - Returns accurate info about Python 3.14

---

## 5. Personas

### Test 5.1: Torvalds (Brutal Honesty)

```bash
gaia code "Should I use global variables?" --persona torvalds --claude --tui off
```

**Expected**:
- Harsh, direct response ("No. Just no.")
- Technical explanation of why globals are bad
- Code examples showing better alternatives
- Authentic Torvalds voice

**Validates**: Torvalds persona
**Status**: PASSED (2026-02-15) - Authentic harsh response with code examples

---

### Test 5.2: Knuth (Thorough Teacher)

```bash
gaia code "Explain what Big O notation is" --persona knuth --claude --tui off
```

**Expected**:
- Detailed, pedagogical explanation
- Mathematical rigor
- Examples and theory

**Validates**: Knuth persona

---

### Test 5.3: Pike (Simplicity)

```bash
gaia code "Create a user authentication system" --persona pike --claude --tui off
```

**Expected**:
- Simple, minimal solution
- Emphasis on simplicity
- Pushback on over-engineering

**Validates**: Pike persona

---

## 6. TUI Modes

### Test 6.1: Simple TUI (Default)

```bash
gaia code "What is 10*10?" --tui simple
```

**Expected**:
- Credential status box
- Execution plan display
- Clean progress
- Final result with timing
- No log spam

**Validates**: Simple TUI mode
**Status**: PASSED (2026-02-15)

---

### Test 6.2: Full TUI

```bash
gaia code "Create a hello world script" --tui full
```

**Expected**:
- Multi-panel Rich layout
- Quality gates visible
- Plan displayed

**Validates**: Full TUI mode

---

### Test 6.3: Minimal TUI

```bash
gaia code "What is 10*10?" --tui minimal
```

**Expected**:
- Single line output
- Updates in place
- Very minimal footprint

**Validates**: Minimal TUI mode

---

### Test 6.4: TUI Off (Verbose)

```bash
gaia code "What is 10*10?" --tui off
```

**Expected**:
- Step-by-step output with boxes
- Shows thought/goal for each step
- Tool arguments and results visible

**Validates**: Verbose/off mode

---

## 7. Codebase Analysis (M7)

### Test 7.1: Index Repository

```bash
gaia code "Index this codebase and tell me how many Python files there are" --claude --tui off
```

**Expected**:
- Uses `index_codebase` tool
- Reports file count and symbol count

**Validates**: M7 codebase indexing

---

### Test 7.2: Find Symbol

```bash
gaia code "Find where the GaiaCodeAgent class is defined" --claude --tui off
```

**Expected**:
- Uses `find_symbol` tool
- Returns: src/gaia/agents/gaia_code/agent.py with line number

**Validates**: Symbol extraction

---

## 8. Interactive Mode

### Test 8.1: Start and Chat

```bash
gaia code -i --persona pike
```

**Expected**:
- Welcome message with persona name
- Prompt appears
- Can chat naturally
- Multi-turn context maintained

**Test sequence**:
1. Type: `Create a simple function to check if a number is prime`
2. Type: `Now add tests for it`
3. Type: `/help` (shows commands)
4. Type: `/status` (shows progress)
5. Type: `/exit` (clean exit)

**Validates**: Interactive chat mode, conversation memory

---

## 9. Error Recovery

### Test 9.1: Handles Missing Dependencies

```bash
gaia code "Create a REST API with Flask and run tests" --claude --tui off
```

**Expected**: If Flask not installed, agent should:
- Detect the missing dependency
- Attempt to install it (pip/uv)
- Re-run after installation

**Validates**: Error recovery, dependency management
**Status**: PASSED (2026-02-15) - Agent bootstrapped pip and installed Flask

---

## 10. Checkpoint/Resume

### Test 10.1: Create Checkpoint

```bash
gaia code --checkpoint --workspace /tmp/test_checkpoint
```

**Expected**: Creates checkpoint.json in workspace
**Validates**: Checkpoint creation

### Test 10.2: Check Status

```bash
gaia code --status --workspace /tmp/test_checkpoint
```

**Expected**: Shows task counts and progress
**Validates**: Status reporting

---

## Validation Checklist

After running all tests, verify:

### Core Functionality
- [x] Agent starts and responds (Test 1.1, 1.2)
- [x] CLI help works with all flags (Test 1.1)
- [x] JSON response parsing handles code fences and embedded JSON

### File Operations
- [x] Create files works (Test 2.1)
- [x] Execute code works (Test 2.1)
- [x] Multi-file projects work with cross-file imports (Test 2.2)
- [x] Complex projects with tests work (Test 2.3)

### Testing
- [x] Can create pytest tests (Test 2.2, 2.3)
- [x] Can run pytest and get results (Test 2.2: 9/9, Test 2.3: 12/12)
- [x] Tests actually pass independently

### Knowledge/Memory
- [x] store_insight works (Test 3.1)
- [x] recall works with FTS5 search (Test 3.2)
- [x] FTS5 handles special characters (dots, colons) safely
- [x] Databases persist to disk (Test 3.4)
- [x] All 6 databases created correctly

### Web Search
- [x] Perplexity direct HTTP API works (Test 4.1)
- [x] Returns accurate, current information
- [x] MCP subprocess fallback available

### Personas
- [x] Torvalds: harsh, direct, code examples (Test 5.1)
- [ ] Knuth: pedagogical, thorough (Test 5.2 - untested this session)
- [ ] Pike: simple, minimal (Test 5.3 - untested this session)
- [x] All 8 persona names accepted by CLI

### TUI Modes
- [x] Simple TUI: clean progress (Test 6.1)
- [ ] Full TUI: multi-panel (Test 6.2)
- [ ] Minimal TUI: single line (Test 6.3)
- [x] TUI Off: verbose output (Test 6.4)

### Error Recovery
- [x] Recovers from path security denials (adapts to workspace dir)
- [x] Recovers from missing dependencies (bootstraps pip)
- [x] Recovers from FTS5 query errors
- [x] Recovers from plan placeholder args

### Plan Execution
- [x] Plans created for multi-step tasks
- [x] Placeholder args ("...") filtered from plans
- [x] Agent doesn't declare completion prematurely
- [x] Completion verification prompts check all files exist

---

## Bugs Fixed During Testing (2026-02-15)

| Bug | Fix | File |
|-----|-----|------|
| Code-fence JSON not parsed | Added code-fence detection before plain-text fast path | base/agent.py |
| Plain text + JSON not parsed | Added embedded JSON extraction after text | base/agent.py |
| Plan placeholder "..." executed literally | Added placeholder detection + filtering | base/agent.py |
| Conversation memory duplicated | Removed pre-add in interactive session | interactive_session.py |
| TUI interface mismatch | Unified all 3 TUI classes to same interface | tui.py |
| TUI double-complete | Added `_completed` guard to all TUI classes | tui.py |
| Perplexity MCP fragile | Added direct HTTP API as primary method | external_services.py |
| FTS5 dot syntax error | Added `_sanitize_fts5_query()` method | shared_state.py |
| /tmp path blocked | Added /tmp and ~/.gaia to default allowed paths | security.py |
| `gaia-code-rac` confusion | Removed standalone entry point, single `gaia code` CLI | setup.py |
| System prompt: premature completion | Added CRITICAL completion verification section | system_prompt.py |
| CLI missing main() | Added main() for standalone entry point | gaia_code/cli.py |

---

## Expected Results Summary

After all tests:
- All 6 databases created and populated
- Knowledge persists across sessions
- Multi-step complex projects complete successfully (Level 1: 9/9 tests, Level 2: 12/12 tests)
- Error recovery handles missing deps, bad paths, FTS5 errors
- All 8 personas respond with authentic voice
- Web search returns current information via Perplexity API
- TUI modes display correctly without log spam

---

**All core tests passing = Fully functional autonomous coding agent!**
