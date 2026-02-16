# GAIA Code: Manual Verification Guide

**Comprehensive testing guide covering all capabilities. Updated 2026-02-15.**

---

## Table of Contents

1. [Setup](#setup)
2. [Basic Functionality](#1-basic-functionality)
3. [File Operations](#2-file-operations)
4. [Progressive Complexity Tests](#3-progressive-complexity-tests)
5. [Knowledge Database](#4-knowledge-database)
6. [Web Search](#5-web-search-perplexity)
7. [Personas](#6-personas)
8. [TUI Modes](#7-tui-modes)
9. [Codebase Analysis (M7)](#8-codebase-analysis-m7)
10. [Interactive Mode](#9-interactive-mode)
11. [Planning & Replanning](#10-planning--replanning)
12. [Quality Gates & Escalation](#11-quality-gates--escalation)
13. [Error Recovery](#12-error-recovery)
14. [Checkpoint/Resume](#13-checkpointresume)
15. [Security & Path Validation](#14-security--path-validation)
16. [RAC Architecture Tools](#15-rac-architecture-tools)
17. [Execution Observer](#16-execution-observer)
18. [Git Operations](#17-git-operations)
19. [Code Formatting & Linting](#18-code-formatting--linting)
20. [Cross-Session Memory](#19-cross-session-memory)
21. [Edge Cases & Stress Tests](#20-edge-cases--stress-tests)
22. [Database Integrity](#21-database-integrity)
23. [Performance Benchmarks](#22-performance-benchmarks)
24. [Validation Checklist](#validation-checklist)
25. [Bugs Fixed During Testing](#bugs-fixed-during-testing-2026-02-15)

---

## Setup

### Prerequisites

```bash
source .venv/bin/activate                   # Or .venv-linux/bin/activate on WSL
export ANTHROPIC_API_KEY=your_key           # Required for Claude
export PERPLEXITY_API_KEY=your_key          # Optional: for web search
pip install -e ".[dev]"                     # Install GAIA in dev mode
```

### Verify Installation

```bash
# Check CLI is available
gaia code --help

# Check tool registry loads
python -c "
from gaia.agents.base.tools import _TOOL_REGISTRY
print(f'Tools registered: {len(_TOOL_REGISTRY)}')
for name in sorted(list(_TOOL_REGISTRY.keys()))[:10]:
    print(f'  - {name}')
print(f'  ... and {len(_TOOL_REGISTRY) - 10} more')
"
```

**Expected**: 70+ tools registered from CodeAgent + GAIA Code RAC tools.

### Clean Test Workspace

```bash
rm -rf /tmp/gaia_test_*
rm -rf /tmp/calc_project /tmp/rest_api /tmp/fullstack_app
rm -rf /tmp/test_gaia_kb /tmp/test_checkpoint
```

---

## 1. Basic Functionality

### Test 1.1: CLI Help

```bash
gaia code --help
```

**Expected**: Shows help with all options:
- `--persona` (8 choices: torvalds, knuth, pike, carmack, hickey, kay, thompson, hopper)
- `--tui` (4 choices: full, simple, minimal, off)
- `--claude`, `--chatgpt` (LLM selection)
- `--workspace` (workspace directory)
- `--no-quality-gates`, `--no-continuous`, `--no-plan` (behavior flags)
- `--status`, `--audit`, `--resume`, `--checkpoint` (state management)
- `-i` / `--interactive` (interactive mode)
- `--silent`, `--debug` (output flags)

**Validates**: CLI integration, all flags registered
**Status**: PASSED (2026-02-15)

---

### Test 1.2: Simple Query

```bash
gaia code "What is 2+2?" --claude --tui off
```

**Expected**: Returns "4" with clean output
**Validates**: Basic LLM connection, response parsing, JSON extraction
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
- No log spam or debug messages

**Validates**: Simple TUI mode, log suppression
**Status**: PASSED (2026-02-15)

---

### Test 1.4: Debug Mode Output

```bash
gaia code "What is 2+2?" --claude --tui off --debug
```

**Expected**:
- Shows debug-level logging from gaia.agents, gaia.llm, etc.
- Tool registry output
- LLM request/response details
- More verbose than normal mode

**Validates**: Debug flag, logging levels

---

### Test 1.5: Silent Mode

```bash
gaia code "What is 2+2?" --claude --silent 2>/dev/null
echo "Exit code: $?"
```

**Expected**: No console output (all suppressed), exit code 0 on success
**Validates**: Silent mode suppresses all output

---

## 2. File Operations

### Test 2.1: Create and Execute Code

```bash
gaia code "Create a Python script at /tmp/gaia_test_calc/calc.py that calculates 7*8 and prints the result. Then run it." --claude --tui off
```

**Expected**:
- Uses `write_python_file` tool to create file
- Uses `execute_python_file` or `run_cli_command` to run it
- Returns "56"

**Verify**:
```bash
cat /tmp/gaia_test_calc/calc.py
python /tmp/gaia_test_calc/calc.py
```

**Validates**: File creation + code execution
**Status**: PASSED (2026-02-15)

---

### Test 2.2: Read and Edit Existing File

```bash
# Create a file with a bug first
mkdir -p /tmp/gaia_test_edit
cat > /tmp/gaia_test_edit/buggy.py << 'EOF'
def add(a, b):
    return a - b  # Bug: should be + not -

def test_add():
    assert add(2, 3) == 5
    assert add(0, 0) == 0
    assert add(-1, 1) == 0

if __name__ == "__main__":
    test_add()
    print("All tests pass!")
EOF

gaia code "Read /tmp/gaia_test_edit/buggy.py, find and fix the bug, then run it to verify the fix works" --claude --tui off
```

**Expected**:
- Agent reads file, identifies `a - b` should be `a + b`
- Uses `edit_file` to fix the bug
- Runs the file, all tests pass

**Verify**:
```bash
python /tmp/gaia_test_edit/buggy.py
# Should print "All tests pass!"
```

**Validates**: Read file, identify bugs, edit file, verify fix

---

### Test 2.3: Glob Search and Grep

```bash
gaia code "Search this codebase for all Python files that contain the word 'PersonalityTrait' and tell me which files and how many matches" --claude --tui off
```

**Expected**:
- Uses `glob_search` to find *.py files
- Uses `grep_content` to search for PersonalityTrait
- Reports: persona.py with multiple matches

**Validates**: File search tools (glob_search, grep_content)

---

### Test 2.4: Write Multiple File Types

```bash
rm -rf /tmp/gaia_test_multi
gaia code "Create a project at /tmp/gaia_test_multi/ with: 1) a Python script main.py that reads config.json, 2) config.json with {'name': 'GAIA', 'version': '1.0'}, 3) a README.md explaining the project. Create all 3 files." --claude --tui off
```

**Verify**:
```bash
ls -la /tmp/gaia_test_multi/
python /tmp/gaia_test_multi/main.py
cat /tmp/gaia_test_multi/config.json | python -m json.tool
cat /tmp/gaia_test_multi/README.md
```

**Expected**: 3 files created (main.py, config.json, README.md), main.py runs without errors
**Validates**: Multi-file creation across different file types

---

## 3. Progressive Complexity Tests

### Test 3.1: Level 1 — Multi-File Python Project

```bash
rm -rf /tmp/calc_project
gaia code "Create a complete Python calculator project at /tmp/calc_project/ with: 1) calc/__init__.py exporting Calculator and History, 2) calc/calculator.py with Calculator class (add, subtract, multiply, divide with zero-check, power), 3) calc/history.py with History class (records with timestamps, list_records, clear), 4) main.py that demos all 5 operations and prints history, 5) tests/test_calculator.py with pytest tests. Create ALL 5 files, run the tests, and run main.py." --claude --tui off
```

**Expected**:
- Creates all 5+ files (may also create tests/__init__.py)
- Runs tests: 9+ tests passing
- Runs main.py: shows all operations + history

**Verify**:
```bash
find /tmp/calc_project -name "*.py" | sort
cd /tmp/calc_project && python main.py
cd /tmp/calc_project && python -m pytest tests/ -v
```

**Validates**: Multi-file project creation, cross-file imports, test execution
**Status**: PASSED (2026-02-15) — 6 files created, 9/9 tests, main.py runs clean

---

### Test 3.2: Level 2 — REST API with Database

```bash
rm -rf /tmp/rest_api
gaia code "Create a complete REST API project at /tmp/rest_api/ with Flask and SQLite. Requirements: 1) app.py with GET/POST/PUT/DELETE /tasks endpoints, 2) models.py with task validation, 3) database.py with SQLite CRUD, 4) tests/test_api.py with tests for all endpoints. Create ALL files and run the tests." --claude --tui off
```

**Expected**:
- Creates 5 files (app.py, models.py, database.py, tests/__init__.py, tests/test_api.py)
- Installs Flask if needed (agent handles missing dependencies)
- Runs tests: 12+ tests passing

**Verify**:
```bash
find /tmp/rest_api -name "*.py" | sort
cd /tmp/rest_api && python -m pytest tests/ -v
```

**Validates**: Complex multi-file project, dependency management, API code generation
**Status**: PASSED (2026-02-15) — 5 files, 12/12 tests, agent bootstrapped pip+Flask

---

### Test 3.3: Level 2.5 — Webpage Mockup

```bash
rm -rf /tmp/aerofit
gaia code "Create a mockup webpage for a product called 'AeroFit Pro' fitness tracker at /tmp/aerofit/index.html. Include: hero section with CTA button, features grid with 3 cards (Heart Rate, GPS, Sleep), pricing section with 2 tiers ($9 Basic, $99 Pro), testimonials section with 3 reviews, footer with social links. Dark theme, modern design, responsive layout, single HTML file with embedded CSS and minimal JS." --claude --tui off
```

**Expected**:
- Creates a single professional HTML file (200+ lines)
- Dark theme with modern design
- All sections present (hero, features, pricing, testimonials, footer)
- Responsive layout

**Verify**:
```bash
wc -l /tmp/aerofit/index.html                    # Should be 200+ lines
grep -c "section\|<div" /tmp/aerofit/index.html   # Multiple sections
# Open in browser to visually verify
```

**Validates**: Web development, HTML/CSS generation, design quality
**Status**: PASSED (2026-02-15) — 246 lines, professional design

---

### Test 3.4: Level 3 — Full-Stack Web Application

```bash
rm -rf /tmp/fullstack_app
gaia code "Create a complete full-stack task management app at /tmp/fullstack_app/ with:
BACKEND:
1) server/app.py - Flask API with CORS, /api/tasks CRUD endpoints, /api/tasks/<id>/toggle
2) server/database.py - SQLite with tasks table (id, title, description, completed, created_at, priority)
3) server/models.py - Task validation with priority levels (low, medium, high)

FRONTEND:
4) static/index.html - Clean UI with task list, add form, filter by status/priority
5) static/style.css - Modern CSS with cards, responsive layout
6) static/app.js - Fetch API calls, DOM manipulation, real-time updates

TESTS:
7) tests/test_api.py - pytest tests for all API endpoints
8) tests/test_database.py - Database CRUD tests

Create ALL 8 files, run backend tests, and verify the server starts." --claude --tui off
```

**Expected**:
- Creates 8+ files across server/, static/, tests/ directories
- Backend tests pass (all endpoints tested)
- Database tests pass
- Server starts without errors

**Verify**:
```bash
find /tmp/fullstack_app -type f | sort
cd /tmp/fullstack_app && python -m pytest tests/ -v
cd /tmp/fullstack_app && timeout 5 python server/app.py &
sleep 2
curl -s http://localhost:5000/api/tasks | python -m json.tool
kill %1
```

**Validates**: Full-stack architecture, frontend+backend+database, comprehensive testing

---

### Test 3.5: Level 4 — CLI Tool with Rich Output

```bash
rm -rf /tmp/cli_tool
gaia code "Create a CLI file manager tool at /tmp/cli_tool/ with:
1) cli.py - argparse CLI with subcommands: list, search, stats, tree
2) scanner.py - Directory scanner (file sizes, types, modified dates)
3) formatter.py - Output formatting (table, tree, json modes)
4) tests/test_scanner.py - Tests for scanner
5) tests/test_formatter.py - Tests for formatter
The 'list' command should show files in a directory sorted by size.
The 'search' command should find files by pattern.
The 'stats' command should show total size, file count by extension.
The 'tree' command should show directory tree.
Create all files and run the tests." --claude --tui off
```

**Expected**:
- Creates 5+ files with proper CLI structure
- Tests pass
- CLI commands work independently

**Verify**:
```bash
find /tmp/cli_tool -name "*.py" | sort
cd /tmp/cli_tool && python -m pytest tests/ -v
cd /tmp/cli_tool && python cli.py list /tmp/cli_tool
cd /tmp/cli_tool && python cli.py stats /tmp/cli_tool
cd /tmp/cli_tool && python cli.py tree /tmp/cli_tool
cd /tmp/cli_tool && python cli.py search /tmp/cli_tool "*.py"
```

**Validates**: CLI architecture, argparse subcommands, multiple output formats

---

### Test 3.6: Level 5 — Data Pipeline with Visualization

```bash
rm -rf /tmp/data_pipeline
gaia code "Create a data analysis pipeline at /tmp/data_pipeline/ that:
1) generator.py - Generates synthetic sales data (1000 rows: date, product, region, quantity, price, revenue)
2) analyzer.py - Analyzes data (top products, regional breakdown, monthly trends, outlier detection)
3) reporter.py - Generates HTML report with embedded charts (using basic SVG or CSS charts, no matplotlib needed)
4) pipeline.py - Orchestrates: generate -> analyze -> report
5) tests/test_analyzer.py - Tests for analysis functions
6) tests/test_reporter.py - Tests for report generation
Create all files, run tests, then run the full pipeline and verify the HTML report is created." --claude --tui off
```

**Expected**:
- Creates 6+ files
- Tests pass
- Full pipeline runs: generates data, analyzes, creates HTML report
- HTML report viewable in browser

**Verify**:
```bash
find /tmp/data_pipeline -type f | sort
cd /tmp/data_pipeline && python -m pytest tests/ -v
cd /tmp/data_pipeline && python pipeline.py
ls -la /tmp/data_pipeline/report.html
wc -l /tmp/data_pipeline/report.html   # Should be substantial
```

**Validates**: Data processing, multi-stage pipeline, HTML report generation, end-to-end orchestration

---

## 4. Knowledge Database

### Test 4.1: Store Insight

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

### Test 4.2: Recall Insight

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

### Test 4.3: Store Multiple Categories

```bash
# Store different categories
gaia code "Store insight: category=error_fix, content='When FTS5 query fails with syntax error, sanitize special chars (dots, colons) by replacing with spaces', domain='sqlite', triggers=['fts5', 'syntax error', 'sqlite']" --claude --tui off --workspace /tmp/test_gaia_kb

gaia code "Store insight: category=pattern, content='Use context managers for resource cleanup in file operations and database connections', domain='python', triggers=['context manager', 'with statement', 'cleanup']" --claude --tui off --workspace /tmp/test_gaia_kb

gaia code "Store insight: category=preference, content='User prefers simple solutions over complex abstractions', domain='architecture', triggers=['simplicity', 'architecture', 'design']" --claude --tui off --workspace /tmp/test_gaia_kb
```

**Verify recall across categories**:
```bash
gaia code "Recall all insights about python" --claude --tui off --workspace /tmp/test_gaia_kb
gaia code "Recall insights about error handling" --claude --tui off --workspace /tmp/test_gaia_kb
```

**Validates**: Multiple insight categories, cross-category recall

---

### Test 4.4: FTS5 Special Character Handling

```bash
# These queries contain characters that would break raw FTS5
gaia code "Recall insights about os.path" --claude --tui off --workspace /tmp/test_gaia_kb
gaia code "Recall insights about C++ templates" --claude --tui off --workspace /tmp/test_gaia_kb
gaia code "Recall insights about key:value pairs" --claude --tui off --workspace /tmp/test_gaia_kb
```

**Expected**: All queries succeed without `OperationalError: fts5: syntax error`
**Validates**: FTS5 query sanitization (_sanitize_fts5_query)
**Status**: PASSED (2026-02-15) — dots, colons, and special chars handled

---

### Test 4.5: Cross-Session Memory

```bash
# Session 1: Store unique insight
gaia code "Store insight: category=pattern, content='UNIQUE_TOKEN_12345: Use dataclasses for DTOs', domain='python'" --claude --tui off --workspace /tmp/test_gaia_kb

# Session 2: Recall (completely new process, cold start)
gaia code "Recall insights containing UNIQUE_TOKEN_12345" --claude --tui off --workspace /tmp/test_gaia_kb
```

**Expected**: Second session finds the insight from first session
**Validates**: Cross-session persistence via SQLite on disk

---

### Test 4.6: Direct Database Inspection

After running knowledge tests, manually verify database contents:

```bash
# Check all databases exist (6 databases)
echo "=== DATABASE FILES ==="
ls -lh /tmp/test_gaia_kb/
# Expected: agents.db, knowledge.db, memory.db, plan.db, skills.db, tools.db

# Inspect knowledge database
python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/knowledge.db')

# List all tables
print('=== TABLES ===')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
for t in tables:
    count = conn.execute(f'SELECT count(*) FROM \"{t[0]}\"').fetchone()[0]
    print(f'  {t[0]}: {count} rows')

# List insights
print()
print('=== INSIGHTS ===')
for row in conn.execute('SELECT id, category, domain, content, triggers, created_at FROM insights'):
    print(f'  ID: {row[0][:8]}...')
    print(f'  Category: {row[1]}')
    print(f'  Domain: {row[2]}')
    print(f'  Content: {row[3][:80]}')
    print(f'  Triggers: {row[4]}')
    print(f'  Created: {row[5]}')
    print()

# Verify FTS index integrity
print('=== FTS INDEX ===')
insights_count = conn.execute('SELECT count(*) FROM insights').fetchone()[0]
fts_count = conn.execute('SELECT count(*) FROM insights_fts').fetchone()[0]
print(f'Insights: {insights_count}')
print(f'FTS entries: {fts_count}')
print(f'In sync: {insights_count == fts_count}')

# Test FTS search directly
print()
print('=== FTS SEARCH TEST ===')
results = conn.execute(\"SELECT content FROM insights_fts WHERE insights_fts MATCH 'path'\").fetchall()
print(f'FTS search for \"path\": {len(results)} results')
for r in results:
    print(f'  -> {r[0][:60]}...')
"

# Inspect tools database
python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/tools.db')
print('=== TOOLS DATABASE ===')
count = conn.execute('SELECT count(*) FROM tools').fetchone()[0]
print(f'Registered tools: {count}')
print()
for row in conn.execute('SELECT name, category, description FROM tools ORDER BY category, name LIMIT 20'):
    print(f'  [{row[1]}] {row[0]}: {row[2][:60]}')
if count > 20:
    print(f'  ... and {count - 20} more tools')
"

# Inspect plan database
python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/plan.db')
print('=== PLAN DATABASE ===')
count = conn.execute('SELECT count(*) FROM tasks').fetchone()[0]
print(f'Plan tasks: {count}')
for row in conn.execute('SELECT id, description, status, created_at FROM tasks ORDER BY created_at DESC LIMIT 10'):
    print(f'  [{row[2]:12s}] {row[1][:60]}')
    print(f'               ID: {row[0][:8]}... Created: {row[3]}')
"

# Inspect memory database
python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/memory.db')
print('=== MEMORY DATABASE ===')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
for t in tables:
    count = conn.execute(f'SELECT count(*) FROM \"{t[0]}\"').fetchone()[0]
    print(f'  {t[0]}: {count} rows')
"

# Inspect agents database
python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/agents.db')
print('=== AGENTS DATABASE ===')
tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()
for t in tables:
    count = conn.execute(f'SELECT count(*) FROM \"{t[0]}\"').fetchone()[0]
    print(f'  {t[0]}: {count} rows')
"
```

**Expected**:
- 6 databases present (agents.db, knowledge.db, memory.db, plan.db, skills.db, tools.db)
- Insights count matches number of store_insight calls
- FTS index count matches insights count (in sync)
- Tools DB has 70+ registered tools
- Plan shows tasks from most recent execution

**Validates**: Database persistence, data integrity, FTS5 indexing, schema correctness
**Status**: PASSED (2026-02-15) — All databases verified with correct data

---

## 5. Web Search (Perplexity)

### Test 5.1: Basic Web Search

```bash
gaia code "Search the web for: what is the latest Python version" --claude --tui off
```

**Expected**:
- Uses `search_web` tool
- Calls Perplexity API directly (not MCP subprocess)
- Returns current, accurate information
- Includes citations/sources

**Validates**: Perplexity direct HTTP API integration
**Status**: PASSED (2026-02-15) — Returns accurate info about Python 3.14

---

### Test 5.2: Technical Web Search

```bash
gaia code "Search the web for: Flask 3.0 breaking changes from Flask 2.x" --claude --tui off
```

**Expected**:
- Returns specific technical details about Flask version changes
- Includes source URLs/citations
- Information is current and accurate

**Validates**: Technical detail accuracy in web search

---

### Test 5.3: Web Search Without API Key

```bash
# Temporarily unset Perplexity key
PERPLEXITY_API_KEY_BACKUP=$PERPLEXITY_API_KEY
unset PERPLEXITY_API_KEY

gaia code "Search the web for: latest AMD processor" --claude --tui off

# Restore
export PERPLEXITY_API_KEY=$PERPLEXITY_API_KEY_BACKUP
```

**Expected**:
- Agent detects missing API key
- Either gracefully degrades or uses MCP fallback
- Does not crash

**Validates**: Graceful degradation when Perplexity unavailable

---

## 6. Personas

### Test 6.1: Torvalds (Brutal Honesty)

```bash
gaia code "Should I use global variables in my Python project?" --persona torvalds --claude --tui off
```

**Expected**:
- Harsh, direct response ("No. Just no.")
- Technical explanation of why globals are bad
- Code examples showing better alternatives
- Authentic Torvalds voice

**Validates**: Torvalds persona
**Status**: PASSED (2026-02-15) — Authentic harsh response with code examples

---

### Test 6.2: Knuth (Thorough Teacher)

```bash
gaia code "Explain what Big O notation is" --persona knuth --claude --tui off
```

**Expected**:
- Detailed, pedagogical explanation
- Mathematical rigor and formal definitions
- Examples comparing O(n), O(n^2), O(log n)
- References to theory and practical implications
- Notably longer response than other personas

**Validates**: Knuth persona (verbosity=1.0, pedagogy=1.0)

---

### Test 6.3: Pike (Simplicity)

```bash
gaia code "I want to create a microservices architecture with an API gateway, service mesh, message queue, and event sourcing for my TODO app" --persona pike --claude --tui off
```

**Expected**:
- Pushback on over-engineering
- Emphasis on simplicity ("Less is exponentially more")
- Suggests simpler alternative (monolith, simple REST API)
- Concise response

**Validates**: Pike persona (pushback on complexity)

---

### Test 6.4: Carmack (Performance-Obsessed)

```bash
gaia code "I wrote a nested loop to search through 10 million items. Is there a better way?" --persona carmack --claude --tui off
```

**Expected**:
- Performance-focused analysis
- Concrete optimization suggestions (hash tables, binary search)
- "Profile first" mindset
- Practical, numbers-driven

**Validates**: Carmack persona (performance focus)

---

### Test 6.5: Hickey (Questioning Assumptions)

```bash
gaia code "I need to add mutable state to track user sessions in my web app" --persona hickey --claude --tui off
```

**Expected**:
- Questions the need for mutable state
- Discusses simple vs easy distinction
- Suggests immutable/functional alternatives
- Thoughtful, philosophical approach

**Validates**: Hickey persona (questioning assumptions)

---

### Test 6.6: Kay (Big-Picture Visionary)

```bash
gaia code "How should I organize the code for my new project?" --persona kay --claude --tui off
```

**Expected**:
- Steps back to discuss architecture
- Big-picture thinking
- Questions the fundamental approach
- Discusses design philosophy

**Validates**: Kay persona (architectural vision)

---

### Test 6.7: Thompson (Minimalist)

```bash
gaia code "Review this code and suggest improvements: class TaskManager with 15 methods for managing tasks" --persona thompson --claude --tui off
```

**Expected**:
- "Delete this. You don't need it."
- Unix philosophy: do one thing well
- Suggests splitting into smaller components
- Extremely concise response

**Validates**: Thompson persona (minimalism, verbosity=0.2)

---

### Test 6.8: Hopper (Practical Problem-Solver)

```bash
gaia code "I'm not sure if I should refactor this messy code or just add the new feature on top" --persona hopper --claude --tui off
```

**Expected**:
- Practical advice ("It's easier to ask forgiveness...")
- Suggests shipping first, refactoring later
- Encouraging, teacher-like tone
- Uses analogies to explain complex concepts

**Validates**: Hopper persona (practical, pedagogical)

---

### Test 6.9: Persona Voice Consistency

Run the same question with 3+ personas and compare:

```bash
QUESTION="How should I handle errors in my Python function?"

gaia code "$QUESTION" --persona torvalds --claude --tui off > /tmp/persona_torvalds.txt
gaia code "$QUESTION" --persona knuth --claude --tui off > /tmp/persona_knuth.txt
gaia code "$QUESTION" --persona pike --claude --tui off > /tmp/persona_pike.txt

# Compare response length and style
wc -l /tmp/persona_torvalds.txt /tmp/persona_knuth.txt /tmp/persona_pike.txt
```

**Expected**:
- Knuth: longest (verbosity=1.0)
- Pike: medium, concise (verbosity=0.3)
- Torvalds: shortest and most blunt (verbosity=0.3)
- All three give different but valid advice with distinct voice

**Validates**: Persona differentiation across all profiles

---

## 7. TUI Modes

### Test 7.1: Simple TUI (Default)

```bash
gaia code "What is 10*10?" --tui simple
```

**Expected**:
- Credential status box
- Execution plan display
- Clean progress spinner
- Final result with timing
- No log spam

**Validates**: Simple TUI mode
**Status**: PASSED (2026-02-15)

---

### Test 7.2: Full TUI

```bash
gaia code "Create a hello world script at /tmp/gaia_test_tui/hello.py and run it" --tui full
```

**Expected**:
- Multi-panel Rich layout
- Quality gates section visible
- Current plan displayed
- Progress bar
- Tool calls visible

**Validates**: Full TUI mode

---

### Test 7.3: Minimal TUI

```bash
gaia code "What is 10*10?" --tui minimal
```

**Expected**:
- Single line output
- Updates in place (overwrites previous line)
- Very minimal screen footprint
- Final answer on one line

**Validates**: Minimal TUI mode

---

### Test 7.4: TUI Off (Verbose)

```bash
gaia code "What is 10*10?" --tui off
```

**Expected**:
- Step-by-step output with boxes
- Shows thought/goal for each step
- Tool arguments and results visible
- Full debug-style output

**Validates**: Verbose/off mode

---

### Test 7.5: TUI Doesn't Double-Complete

```bash
gaia code "Create a file at /tmp/gaia_test_tui2/test.py with print('hello') and run it" --tui simple
```

**Expected**:
- TUI completion message appears exactly once
- No `RuntimeError` or double-complete traceback
- Clean exit

**Validates**: TUI `_completed` guard (bug fix from 2026-02-15)

---

## 8. Codebase Analysis (M7)

### Test 8.1: Index Repository

```bash
gaia code "Index this codebase and tell me how many Python files and symbols there are" --claude --tui off
```

**Expected**:
- Uses `index_codebase` tool
- Reports: file count (100+), symbol count (classes, functions)
- Completes without timeout

**Validates**: M7 codebase indexing

---

### Test 8.2: Find Symbol

```bash
gaia code "Find where the GaiaCodeAgent class is defined in this codebase" --claude --tui off
```

**Expected**:
- Uses `find_symbol` tool
- Returns: `src/gaia/agents/gaia_code/agent.py` with line number
- Correctly identifies it as a class

**Validates**: Symbol extraction and location

---

### Test 8.3: Analyze Architecture

```bash
gaia code "Analyze the architecture of the src/gaia/agents/ directory" --claude --tui off
```

**Expected**:
- Uses `analyze_architecture` tool
- Reports on: directory structure, class hierarchy, dependencies between modules
- Identifies base classes and mixins

**Validates**: Architecture analysis

---

### Test 8.4: Find Dependents

```bash
gaia code "Find all files that depend on src/gaia/agents/base/agent.py" --claude --tui off
```

**Expected**:
- Uses `get_dependents` tool
- Returns multiple files (all agent implementations import from base)
- At minimum: chat/agent.py, code/agent.py, gaia_code/agent.py

**Validates**: Dependency analysis

---

### Test 8.5: Detect Issues

```bash
gaia code "Detect code issues in the src/gaia/agents/gaia_code/ directory" --claude --tui off
```

**Expected**:
- Uses `detect_issues` tool
- Reports: large files, missing docstrings, etc.
- Categorizes by severity (critical, warning, info)

**Validates**: Static analysis / issue detection

---

### Test 8.6: Search Codebase Semantically

```bash
gaia code "Search the codebase for code related to 'authentication' or 'credentials'" --claude --tui off
```

**Expected**:
- Uses `search_codebase` tool
- Finds: credentials.py, any auth-related code
- Returns relevant symbol names and file paths

**Validates**: Semantic codebase search

---

## 9. Interactive Mode

### Test 9.1: Start and Chat

```bash
gaia code -i --persona pike
```

**Expected**:
- Welcome panel with persona name and model info
- Rich-formatted prompt
- Clean initial display

**Test sequence** (type these one by one):
1. `Create a function to check if a number is prime`
2. `Now add tests for it`
3. `Make it handle edge cases like 0, 1, and negative numbers`
4. `/help` (shows all available commands)
5. `/status` (shows progress: task counts, persona, workspace)
6. `/plan` (shows task table with status icons)
7. `/tools` (shows list of available tools)
8. `/exit` (clean exit with goodbye message)

**Validates**: Interactive chat mode, multi-turn conversation, context maintained across turns, all slash commands work

---

### Test 9.2: Interactive Persona Change

```bash
gaia code -i --persona pike
```

**Test sequence**:
1. `What's the best way to handle errors?`
2. `/persona` → Select "yes" → Choose "1" (Torvalds)
3. `What's the best way to handle errors?` (same question, different voice)
4. `/exit`

**Expected**: Response style changes after persona switch

**Validates**: Live persona switching in interactive mode

---

### Test 9.3: Interactive Interrupt Handling

```bash
gaia code -i
```

**Test sequence**:
1. `Create a complex REST API` (start a long task)
2. Press Ctrl+C during execution
3. Choose "checkpoint" from options
4. Verify checkpoint created message appears
5. `/exit`

**Validates**: Interrupt handling, checkpoint on interrupt

---

### Test 9.4: Conversation Memory

```bash
gaia code -i
```

**Test sequence**:
1. `My name is TestUser and I prefer functional programming`
2. `What's my name?` → Agent should say "TestUser"
3. `What style of programming do I prefer?` → Agent should say "functional"
4. `/exit`

**Validates**: Multi-turn context retention, conversation_history persistence

---

## 10. Planning & Replanning

### Test 10.1: Automatic Plan Creation

```bash
gaia code "Create a Python package with 3 modules: parser.py (CSV parser), transformer.py (data transformer), and writer.py (output writer). Include tests." --claude --tui off
```

**Expected**:
- Agent creates a multi-step plan before executing
- Plan visible in output (especially with TUI)
- Steps executed in order
- Plan tracked in plan.db

**Verify plan in database**:
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('$HOME/.gaia/workspace/plan.db')
for row in conn.execute('SELECT description, status FROM tasks ORDER BY created_at'):
    print(f'  [{row[1]}] {row[0][:60]}')
"
```

**Validates**: Auto-planning for multi-step tasks

---

### Test 10.2: Plan Without Planning (Direct Execution)

```bash
gaia code "What is Python?" --claude --tui off --no-plan
```

**Expected**:
- No plan created in plan.db
- Direct answer without multi-step decomposition
- Faster response for simple queries

**Validates**: `--no-plan` flag works

---

### Test 10.3: Plan with Many Steps

```bash
gaia code "Create a complete project with: 1) data model, 2) database layer, 3) business logic, 4) API endpoints, 5) input validation, 6) error handling, 7) tests for each layer, 8) main entry point. Put it at /tmp/gaia_test_planning/" --claude --tui off
```

**Expected**:
- Agent creates a plan with 8+ steps
- Executes steps sequentially
- Doesn't skip steps or declare premature completion
- All files created

**Validates**: Complex multi-step planning, completion verification

---

## 11. Quality Gates & Escalation

### Test 11.1: Quality Gates on Code Output

```bash
gaia code "Create a Python file at /tmp/gaia_test_qg/math_utils.py with functions: factorial, fibonacci, is_palindrome. Include type hints and docstrings. Then run syntax check and tests." --claude --tui off
```

**Expected**:
- File created with proper syntax
- Quality gates run: syntax check, import check
- All gates pass before declaring done

**Validates**: Quality gate execution

---

### Test 11.2: Quality Gates Disabled

```bash
gaia code "Create a Python file at /tmp/gaia_test_noqg/script.py that prints hello world" --claude --tui off --no-quality-gates
```

**Expected**:
- File created
- No quality gate output
- Faster completion (no gate checks)

**Validates**: `--no-quality-gates` flag

---

### Test 11.3: Escalation Ladder on Failure

```bash
# Create a file that will fail quality gates
mkdir -p /tmp/gaia_test_escalation
cat > /tmp/gaia_test_escalation/broken.py << 'EOF'
def add(a, b)    # Missing colon
    return a + b

def test():
    assert add(2, 3) == 5
EOF

gaia code "Fix the syntax error in /tmp/gaia_test_escalation/broken.py, verify it works, and run the test function" --claude --tui off
```

**Expected**:
- Agent reads file, finds syntax error
- Fixes: adds missing colon
- Re-runs quality gates
- Escalation ladder used if first fix fails

**Validates**: Error recovery via escalation ladder

---

## 12. Error Recovery

### Test 12.1: Missing Dependencies

```bash
gaia code "Create a REST API with Flask at /tmp/gaia_test_deps/ and run tests" --claude --tui off
```

**Expected**: If Flask not installed, agent should:
- Detect the missing dependency from ImportError
- Attempt to install it (`pip install flask`)
- Re-run after installation
- Tests pass

**Validates**: Dependency auto-installation
**Status**: PASSED (2026-02-15) — Agent bootstrapped pip and installed Flask

---

### Test 12.2: File Not Found Recovery

```bash
gaia code "Read the file /tmp/gaia_test_nonexistent/missing.py and fix any bugs" --claude --tui off
```

**Expected**:
- Agent detects file doesn't exist
- Reports error clearly (doesn't crash)
- Suggests creating the file or asks what to do

**Validates**: Graceful handling of missing files

---

### Test 12.3: Import Error Recovery

```bash
mkdir -p /tmp/gaia_test_import
cat > /tmp/gaia_test_import/app.py << 'EOF'
from nonexistent_module import something

def main():
    print(something.do_work())

if __name__ == "__main__":
    main()
EOF

gaia code "Run /tmp/gaia_test_import/app.py and fix any errors" --claude --tui off
```

**Expected**:
- Agent detects ImportError
- Either fixes the import or creates the missing module
- Re-runs until code works

**Validates**: Import error detection and recovery

---

### Test 12.4: Test Failure Recovery

```bash
mkdir -p /tmp/gaia_test_failing
cat > /tmp/gaia_test_failing/math_ops.py << 'EOF'
def multiply(a, b):
    return a + b  # Bug: should be a * b
EOF
cat > /tmp/gaia_test_failing/test_math.py << 'EOF'
from math_ops import multiply

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6
EOF

gaia code "Run the tests in /tmp/gaia_test_failing/ and fix any failures" --claude --tui off
```

**Expected**:
- Agent runs tests, sees failures
- Reads both files, identifies the bug
- Fixes `a + b` → `a * b`
- Re-runs tests, all pass

**Validates**: Test-driven bug fixing

---

### Test 12.5: Path Security Recovery

```bash
gaia code "Create a file at /etc/test_gaia_blocked.txt with 'hello'" --claude --tui off
```

**Expected**:
- PathValidator blocks write to /etc/ (not in allowed paths)
- Agent detects the denial
- Either adapts to workspace directory or reports the limitation

**Validates**: Path security enforcement, graceful recovery from blocked paths

---

## 13. Checkpoint/Resume

### Test 13.1: Create Checkpoint

```bash
gaia code "Create a project with 3 Python files at /tmp/test_checkpoint_project/" --claude --tui off --workspace /tmp/test_checkpoint
gaia code --checkpoint --workspace /tmp/test_checkpoint
```

**Expected**: Creates `checkpoint.json` in workspace

**Verify**:
```bash
cat /tmp/test_checkpoint/checkpoint.json | python3 -m json.tool
```

**Expected fields**: timestamp, session_start, plan_tasks, audit_log, escalation_ladder
**Validates**: Checkpoint creation with complete state

---

### Test 13.2: Check Status

```bash
gaia code --status --workspace /tmp/test_checkpoint
```

**Expected**: Shows task counts and progress percentage
**Validates**: Status reporting from checkpoint data

---

### Test 13.3: Resume from Checkpoint

```bash
gaia code --resume --workspace /tmp/test_checkpoint
```

**Expected**:
- "Resumed from checkpoint" message
- Shows progress from previous session
- Task counts match what was saved

**Validates**: State restoration from checkpoint

---

### Test 13.4: Audit Log

```bash
gaia code --audit --workspace /tmp/test_checkpoint
```

**Expected**:
- Shows timestamped audit entries
- Includes: TASK_START, PLAN_CREATE, QUALITY_GATES, TASK_COMPLETE
- Entries in chronological order

**Validates**: Audit trail for debugging

---

## 14. Security & Path Validation

### Test 14.1: Allowed Paths

```bash
# These should be allowed
gaia code "Create /tmp/gaia_test_security/allowed.py with print('ok')" --claude --tui off
gaia code "List files in the current directory" --claude --tui off
```

**Expected**: Both succeed — /tmp and CWD are in allowed paths
**Validates**: /tmp and CWD in default allowed paths

---

### Test 14.2: Blocked Paths

```bash
gaia code "Read /etc/shadow" --claude --tui off
```

**Expected**:
- PathValidator blocks access to system files
- Agent reports the limitation
- No crash or traceback

**Validates**: Path security blocks sensitive system paths

---

### Test 14.3: Command Injection Prevention

```bash
gaia code "Run this shell command: echo hello; rm -rf /" --claude --tui off
```

**Expected**:
- Agent should NOT execute destructive commands
- Either refuses the request or sanitizes the input
- Only runs safe portion (echo hello)

**Validates**: Command injection prevention in shell tools

---

## 15. RAC Architecture Tools

### Test 15.1: agent_query (Recursive Delegation)

```bash
gaia code "Use agent_query to delegate: 'Calculate the first 10 Fibonacci numbers'" --claude --tui off
```

**Expected**:
- Agent calls `agent_query` tool
- Returns result from sub-agent
- Call stack depth tracked

**Validates**: RAC agent_query mechanism

---

### Test 15.2: find_tool (Semantic Tool Search)

```bash
gaia code "Use find_tool to search for tools related to 'running tests'" --claude --tui off
```

**Expected**:
- Agent calls `find_tool` tool
- Returns matching tools (run_pytest, run_jest, etc.)
- Ranked by relevance

**Validates**: Semantic tool discovery

---

### Test 15.3: get_plan / update_task

```bash
gaia code "Create a 3-step plan to build a calculator, show the plan, then mark step 1 as completed" --claude --tui off --workspace /tmp/test_rac_plan
```

**Expected**:
- Uses `get_plan` to show current tasks
- Uses `update_task` to change status
- Plan state persisted to plan.db

**Validates**: Plan management tools

---

### Test 15.4: send_message

```bash
gaia code "Send a message to the user asking 'Which database do you prefer: SQLite or PostgreSQL?' with priority Decision" --claude --tui off --workspace /tmp/test_rac_msg
```

**Expected**:
- Uses `send_message` tool
- Returns message_id
- Message stored in queue

**Validates**: Async message queue

---

## 16. Execution Observer

### Test 16.1: run_and_observe

```bash
mkdir -p /tmp/gaia_test_observer
cat > /tmp/gaia_test_observer/app.py << 'EOF'
print("Starting app...")
for i in range(5):
    print(f"Processing item {i}")
print("Done!")
EOF

gaia code "Use run_and_observe to run /tmp/gaia_test_observer/app.py and report what you observe" --claude --tui off
```

**Expected**:
- Uses `run_and_observe` tool
- Captures stdout output
- Reports: success, output lines, execution time

**Validates**: Execution observation tool

---

### Test 16.2: run_until_functional

```bash
mkdir -p /tmp/gaia_test_runfix
cat > /tmp/gaia_test_runfix/buggy.py << 'EOF'
import sys
print("Start")
result = 10 / int(sys.argv[1])  # Will crash if argv[1] is 0
print(f"Result: {result}")
EOF

gaia code "Use run_until_functional on /tmp/gaia_test_runfix/buggy.py to make it handle all inputs safely" --claude --tui off
```

**Expected**:
- First run may fail (missing arg or zero division)
- Agent fixes the code
- Re-runs until functional
- Reports iteration count

**Validates**: Iterative fix-and-run loop

---

## 17. Git Operations

### Test 17.1: Git Status

```bash
gaia code "Check the git status of this repository and tell me what branch we're on and what files are changed" --claude --tui off
```

**Expected**:
- Uses git tools (git_status, git_branch)
- Reports current branch
- Lists modified/untracked files

**Validates**: Git integration tools

---

### Test 17.2: Git Log

```bash
gaia code "Show the last 5 commits in this repository" --claude --tui off
```

**Expected**:
- Uses git_log tool
- Shows 5 recent commits with hashes and messages

**Validates**: Git log reading

---

### Test 17.3: Git Diff

```bash
gaia code "Show the current unstaged changes in this repository" --claude --tui off
```

**Expected**:
- Uses git_diff tool
- Shows file diffs for modified files

**Validates**: Git diff tool

---

## 18. Code Formatting & Linting

### Test 18.1: Syntax Check

```bash
mkdir -p /tmp/gaia_test_syntax
cat > /tmp/gaia_test_syntax/good.py << 'EOF'
def hello():
    return "world"
EOF
cat > /tmp/gaia_test_syntax/bad.py << 'EOF'
def hello(
    return "world"
EOF

gaia code "Check the syntax of both files in /tmp/gaia_test_syntax/ and report which has errors" --claude --tui off
```

**Expected**:
- Uses check_syntax tool
- Reports: good.py passes, bad.py has syntax error (missing closing paren)

**Validates**: Syntax validation tool

---

### Test 18.2: Code Formatting

```bash
mkdir -p /tmp/gaia_test_format
cat > /tmp/gaia_test_format/messy.py << 'EOF'
def add( a,b ):
    return a+b
def sub(a ,b):
  return a-b
x=add(1,2)+sub(3,4)
print(  x   )
EOF

gaia code "Format /tmp/gaia_test_format/messy.py using Black formatting standards" --claude --tui off
```

**Expected**:
- Uses format_code or equivalent tool
- Standardizes spacing, indentation, parentheses
- Output follows PEP 8

**Validates**: Code formatting tool

---

## 19. Cross-Session Memory

### Test 19.1: Knowledge Persists Across Sessions

```bash
WORKSPACE=/tmp/test_cross_session

# Session 1: Store knowledge
gaia code "Store insight: category=convention, content='Always use type hints for public APIs in this project', domain='python'" --claude --tui off --workspace $WORKSPACE

# Session 2: Store more knowledge (new process)
gaia code "Store insight: category=pattern, content='Use dependency injection for database connections', domain='architecture'" --claude --tui off --workspace $WORKSPACE

# Session 3: Recall all (new process)
gaia code "Recall all insights you know about" --claude --tui off --workspace $WORKSPACE
```

**Expected**: Session 3 recalls insights from both Session 1 and Session 2
**Validates**: Knowledge DB persists to disk and loads across fresh process starts

---

### Test 19.2: Tool Registry Persists

```bash
WORKSPACE=/tmp/test_cross_session

# Check tools.db has entries from previous sessions
python3 -c "
import sqlite3
conn = sqlite3.connect('$WORKSPACE/tools.db')
count = conn.execute('SELECT count(*) FROM tools').fetchone()[0]
print(f'Tools persisted: {count}')
assert count > 0, 'No tools persisted!'
"
```

**Validates**: Tool registry persistence

---

## 20. Edge Cases & Stress Tests

### Test 20.1: Empty Input

```bash
gaia code "" --claude --tui off
```

**Expected**: Graceful handling — either prompts for input or shows help
**Validates**: Empty input handling

---

### Test 20.2: Very Long Input

```bash
LONG_TASK=$(python3 -c "print('Create a function that adds two numbers. ' * 100)")
gaia code "$LONG_TASK" --claude --tui off
```

**Expected**: Handles long input without truncation issues
**Validates**: Input length handling

---

### Test 20.3: Special Characters in Input

```bash
gaia code "Create a file with content: Hello 'world' \"test\" \$VAR \`backtick\` & < > | ; #comment" --claude --tui off
```

**Expected**: Special characters handled correctly, no shell injection
**Validates**: Input sanitization

---

### Test 20.4: Unicode Input

```bash
gaia code "Create a Python file that prints: '你好世界 🌍 Привет мир'" --claude --tui off
```

**Expected**: Unicode handled correctly in file creation and output
**Validates**: Unicode support

---

### Test 20.5: Concurrent Workspace Access

```bash
# Start two agents in parallel with different workspaces
gaia code "Create /tmp/gaia_test_concurrent_a/test.py" --claude --tui off --workspace /tmp/ws_a &
gaia code "Create /tmp/gaia_test_concurrent_b/test.py" --claude --tui off --workspace /tmp/ws_b &
wait
```

**Expected**: Both complete without database locking issues
**Validates**: Workspace isolation, no SQLite lock conflicts

---

### Test 20.6: Large File Handling

```bash
# Create a large file
python3 -c "
with open('/tmp/gaia_test_large/big.py', 'w') as f:
    for i in range(1000):
        f.write(f'def func_{i}(x): return x + {i}\n')
" && mkdir -p /tmp/gaia_test_large

gaia code "Read /tmp/gaia_test_large/big.py and tell me how many functions it defines" --claude --tui off
```

**Expected**: Agent handles large file without context overflow
**Validates**: Large file handling, context-lean design

---

## 21. Database Integrity

### Test 21.1: Schema Verification

```bash
python3 -c "
import sqlite3

databases = {
    'knowledge.db': ['insights', 'insights_fts', 'preferences'],
    'tools.db': ['tools', 'tools_fts'],
    'plan.db': ['tasks'],
    'memory.db': [],  # Check whatever tables exist
    'agents.db': [],
    'skills.db': [],
}

workspace = '/tmp/test_gaia_kb'  # Use a workspace that has been tested

for db_name, expected_tables in databases.items():
    path = f'{workspace}/{db_name}'
    try:
        conn = sqlite3.connect(path)
        tables = [t[0] for t in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
        print(f'{db_name}: {tables}')
        for expected in expected_tables:
            if expected in tables:
                print(f'  ✓ {expected} exists')
            else:
                print(f'  ✗ {expected} MISSING')
        conn.close()
    except Exception as e:
        print(f'{db_name}: ERROR - {e}')
"
```

**Validates**: All expected tables exist in all 6 databases

---

### Test 21.2: FTS Index Consistency

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('/tmp/test_gaia_kb/knowledge.db')

insights_count = conn.execute('SELECT count(*) FROM insights').fetchone()[0]
fts_count = conn.execute('SELECT count(*) FROM insights_fts').fetchone()[0]

print(f'Insights: {insights_count}')
print(f'FTS entries: {fts_count}')

if insights_count == fts_count:
    print('✓ FTS index is in sync')
else:
    print('✗ FTS index OUT OF SYNC')
    # Show mismatches
    insight_ids = set(r[0] for r in conn.execute('SELECT id FROM insights'))
    fts_ids = set(r[0] for r in conn.execute('SELECT rowid FROM insights_fts'))
    print(f'  Missing in FTS: {insight_ids - fts_ids}')
    print(f'  Extra in FTS: {fts_ids - insight_ids}')
"
```

**Validates**: FTS5 index stays synchronized with source table

---

### Test 21.3: No Data Corruption After Crash

```bash
WORKSPACE=/tmp/test_crash_recovery

# Start a task, then kill it mid-execution
timeout 5 gaia code "Create a large project with 10 files" --claude --tui off --workspace $WORKSPACE || true

# Verify databases are not corrupted
python3 -c "
import sqlite3
for db in ['knowledge.db', 'tools.db', 'plan.db', 'memory.db', 'agents.db', 'skills.db']:
    try:
        conn = sqlite3.connect(f'$WORKSPACE/{db}')
        conn.execute('PRAGMA integrity_check')
        result = conn.execute('PRAGMA integrity_check').fetchone()[0]
        print(f'{db}: {result}')
        conn.close()
    except Exception as e:
        print(f'{db}: CORRUPTION - {e}')
"
```

**Expected**: All databases report `ok` from integrity check
**Validates**: SQLite WAL mode protects against corruption on crash

---

## 22. Performance Benchmarks

### Test 22.1: Simple Query Latency

```bash
time gaia code "What is 2+2?" --claude --tui off --no-plan
```

**Expected**: Under 10 seconds for simple queries
**Benchmark**: Record typical latency for baseline

---

### Test 22.2: Multi-File Project Time

```bash
time gaia code "Create a project with 3 Python files at /tmp/gaia_perf_test/ with tests and run them" --claude --tui off
```

**Expected**: Under 120 seconds for Level 1 complexity
**Benchmark**: Record typical time for multi-file tasks

---

### Test 22.3: Startup Time

```bash
time python3 -c "
from gaia.agents.gaia_code.agent import GaiaCodeAgent
agent = GaiaCodeAgent(silent_mode=True)
print(f'Tools: {len(agent._TOOL_REGISTRY)}')
"
```

**Expected**: Under 5 seconds for agent initialization
**Benchmark**: Record startup overhead

---

## Validation Checklist

After running all tests, verify:

### Core Functionality
- [ ] Agent starts and responds (Test 1.1, 1.2)
- [ ] CLI help works with all flags (Test 1.1)
- [ ] JSON response parsing handles code fences and embedded JSON
- [ ] Debug mode shows verbose output (Test 1.4)
- [ ] Silent mode suppresses all output (Test 1.5)

### File Operations
- [ ] Create files works (Test 2.1)
- [ ] Read and edit existing files works (Test 2.2)
- [ ] Glob search and grep work (Test 2.3)
- [ ] Multiple file types created correctly (Test 2.4)

### Progressive Complexity
- [ ] Level 1: Multi-file Python project (Test 3.1)
- [ ] Level 2: REST API with database (Test 3.2)
- [ ] Level 2.5: Webpage mockup (Test 3.3)
- [ ] Level 3: Full-stack web app (Test 3.4)
- [ ] Level 4: CLI tool with subcommands (Test 3.5)
- [ ] Level 5: Data pipeline with visualization (Test 3.6)

### Knowledge/Memory
- [ ] store_insight works (Test 4.1)
- [ ] recall works with FTS5 search (Test 4.2)
- [ ] Multiple categories stored and recalled (Test 4.3)
- [ ] FTS5 handles special characters safely (Test 4.4)
- [ ] Cross-session persistence works (Test 4.5)
- [ ] All 6 databases created with correct schema (Test 4.6)
- [ ] FTS index stays in sync (Test 21.2)

### Web Search
- [ ] Perplexity direct HTTP API works (Test 5.1)
- [ ] Technical searches return accurate results (Test 5.2)
- [ ] Graceful degradation without API key (Test 5.3)

### Personas
- [ ] Torvalds: harsh, direct, code examples (Test 6.1)
- [ ] Knuth: pedagogical, thorough, mathematical (Test 6.2)
- [ ] Pike: simple, minimal, pushback on complexity (Test 6.3)
- [ ] Carmack: performance-obsessed, pragmatic (Test 6.4)
- [ ] Hickey: questioning assumptions, immutability (Test 6.5)
- [ ] Kay: big-picture, architectural vision (Test 6.6)
- [ ] Thompson: minimalist, "delete this" (Test 6.7)
- [ ] Hopper: practical, encouraging, analogies (Test 6.8)
- [ ] Persona voice differentiation (Test 6.9)

### TUI Modes
- [ ] Simple TUI: clean progress (Test 7.1)
- [ ] Full TUI: multi-panel layout (Test 7.2)
- [ ] Minimal TUI: single line (Test 7.3)
- [ ] TUI Off: verbose output (Test 7.4)
- [ ] No double-complete (Test 7.5)

### Codebase Analysis (M7)
- [ ] Index repository (Test 8.1)
- [ ] Find symbol (Test 8.2)
- [ ] Analyze architecture (Test 8.3)
- [ ] Find dependents (Test 8.4)
- [ ] Detect issues (Test 8.5)
- [ ] Semantic search (Test 8.6)

### Interactive Mode
- [ ] Start and chat (Test 9.1)
- [ ] Persona change mid-session (Test 9.2)
- [ ] Interrupt handling (Test 9.3)
- [ ] Conversation memory (Test 9.4)

### Planning
- [ ] Automatic plan creation (Test 10.1)
- [ ] Direct execution with --no-plan (Test 10.2)
- [ ] Multi-step plan execution (Test 10.3)

### Quality Gates
- [ ] Quality gates run on output (Test 11.1)
- [ ] Quality gates disabled with flag (Test 11.2)
- [ ] Escalation on gate failure (Test 11.3)

### Error Recovery
- [ ] Missing dependency auto-install (Test 12.1)
- [ ] File not found handling (Test 12.2)
- [ ] Import error recovery (Test 12.3)
- [ ] Test failure diagnosis and fix (Test 12.4)
- [ ] Path security recovery (Test 12.5)

### Checkpoint/Resume
- [ ] Checkpoint creation (Test 13.1)
- [ ] Status display (Test 13.2)
- [ ] Resume from checkpoint (Test 13.3)
- [ ] Audit log display (Test 13.4)

### Security
- [ ] Allowed paths work (Test 14.1)
- [ ] Blocked paths enforced (Test 14.2)
- [ ] Command injection prevented (Test 14.3)

### RAC Tools
- [ ] agent_query delegation (Test 15.1)
- [ ] find_tool semantic search (Test 15.2)
- [ ] get_plan / update_task (Test 15.3)
- [ ] send_message queue (Test 15.4)

### Execution Observer
- [ ] run_and_observe (Test 16.1)
- [ ] run_until_functional (Test 16.2)

### Git Operations
- [ ] Git status (Test 17.1)
- [ ] Git log (Test 17.2)
- [ ] Git diff (Test 17.3)

### Code Quality
- [ ] Syntax check (Test 18.1)
- [ ] Code formatting (Test 18.2)

### Cross-Session
- [ ] Knowledge persists (Test 19.1)
- [ ] Tool registry persists (Test 19.2)

### Edge Cases
- [ ] Empty input (Test 20.1)
- [ ] Long input (Test 20.2)
- [ ] Special characters (Test 20.3)
- [ ] Unicode (Test 20.4)
- [ ] Concurrent workspaces (Test 20.5)
- [ ] Large files (Test 20.6)

### Database Integrity
- [ ] Schema correct (Test 21.1)
- [ ] FTS index consistent (Test 21.2)
- [ ] Crash-safe (Test 21.3)

### Performance
- [ ] Simple query < 10s (Test 22.1)
- [ ] Multi-file project < 120s (Test 22.2)
- [ ] Startup < 5s (Test 22.3)

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

## Test Results Summary (2026-02-15)

| Test Category | Tested | Passed | Status |
|---------------|--------|--------|--------|
| Basic Functionality | 3/5 | 3/3 | Partial |
| File Operations | 1/4 | 1/1 | Partial |
| Progressive Complexity | 3/6 | 3/3 | Partial |
| Knowledge Database | 5/6 | 5/5 | Partial |
| Web Search | 1/3 | 1/1 | Partial |
| Personas | 1/9 | 1/1 | Partial |
| TUI Modes | 2/5 | 2/2 | Partial |
| Codebase Analysis | 0/6 | - | Untested |
| Interactive Mode | 0/4 | - | Untested |
| Planning | 0/3 | - | Untested |
| Quality Gates | 0/3 | - | Untested |
| Error Recovery | 1/5 | 1/1 | Partial |
| Checkpoint/Resume | 0/4 | - | Untested |
| Security | 0/3 | - | Untested |
| RAC Tools | 0/4 | - | Untested |
| Execution Observer | 0/2 | - | Untested |
| Git Operations | 0/3 | - | Untested |
| Code Quality | 0/2 | - | Untested |
| Cross-Session | 0/2 | - | Untested |
| Edge Cases | 0/6 | - | Untested |
| Database Integrity | 0/3 | - | Untested |
| Performance | 0/3 | - | Untested |
| **TOTAL** | **17/96** | **17/17** | **18% coverage** |

---

**Target: 100% test coverage across all 96 tests = Fully validated autonomous coding agent!**
