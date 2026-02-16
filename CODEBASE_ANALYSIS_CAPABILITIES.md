# GAIA Code: Codebase Analysis Capabilities

**Question**: Does this architecture support analyzing existing code and building a comprehensive knowledge base?

**Answer**: ✅ **YES** - The architecture is designed for this, and the foundation is complete. However, the specific codebase indexing tools (M7) need to be added.

---

## What's Already There (Foundation Complete)

### ✅ 1. ProjectManifest (Live State Tracking)

**Location**: `shared_state.py` - `ProjectManifest` class

**Already implemented**:
```python
class ProjectManifest:
    """Live project state shared across all agents."""

    def __init__(self):
        self.files: Dict[str, Dict] = {}           # path -> {content, created, modified}
        self.apis: Dict[str, Dict] = {}            # endpoint -> {method, params, response}
        self.schemas: Dict[str, Dict] = {}         # table -> {columns, types}
        self.decisions: List[Dict] = []            # Architecture decisions
        self.dependencies: List[str] = []          # Dependencies
```

**What it does NOW**:
- Tracks files created/modified during session
- Records API endpoints defined
- Stores database schemas
- Logs architecture decisions

**What it NEEDS** (M7):
- Symbol index (classes, functions, variables)
- Import graph (who imports whom)
- Call graph (who calls whom)
- Dependency analysis
- Circular dependency detection

### ✅ 2. ArchitectureAgent Specialist

**Location**: `specialists/architecture_agent.py`

**Already has**:
- SOLID principles knowledge
- Design pattern expertise
- Architecture pattern understanding
- Anti-pattern detection knowledge

**Tools mentioned in prompt**:
- `analyze_dependencies()` - NEEDS IMPLEMENTATION
- `detect_circular_deps()` - NEEDS IMPLEMENTATION
- `check_solid_violations()` - NEEDS IMPLEMENTATION
- `generate_architecture_diagram()` - NEEDS IMPLEMENTATION

### ✅ 3. Knowledge Storage Infrastructure

**Already complete**:
- `knowledge.db` - Stores insights about code
- `memory.db` - Caches file contents
- FTS5 search - Full-text search over knowledge
- Vector search - Semantic search (M6)
- Insight system - Structured learning

**Perfect for storing**:
- Architecture analysis results
- Dependency graphs
- Code patterns discovered
- Bug patterns found
- Documentation gaps

### ✅ 4. Recursive Analysis via RLM

**Already built into prompts**:
```
Pattern 1: FILTER (when codebase is large)
  1. Filter to relevant files
  2. Chunk and recurse
  3. Synthesize results

Pattern 2: CHUNK & RECURSE
  - Analyze module by module
  - Each module gets fresh context
  - Synthesize into full architecture
```

**This enables**: Analyzing 1M+ LoC by recursively breaking down the analysis.

---

## What Needs to Be Added (M7: Codebase Indexing)

### Missing Component: CodebaseIndex Class

**From Architecture Spec (C1)**:

The spec clearly defines this capability:

```python
class CodebaseIndex:
    """Indexes a codebase for fast navigation and understanding."""

    def __init__(self, root: str):
        self.root = root
        self.files: dict[str, FileInfo] = {}
        self.symbols: dict[str, SymbolInfo] = {}      # name → definition location
        self.imports: dict[str, list[str]] = {}       # file → imported modules
        self.dependents: dict[str, list[str]] = {}    # file → files that import it
        self.call_graph: dict[str, list[str]] = {}    # function → functions it calls

    def index(self):
        """Build the full index. Runs once, updates incrementally."""
        for path in self._walk_source_files():
            self._index_file(path)
        self._build_dependency_graph()
        self._detect_circular_deps()
        self._persist()  # Save to knowledge DB

    def find_symbol(self, name: str) -> list[SymbolInfo]:
        """Find where a symbol is defined and used."""
        ...

    def get_dependents(self, path: str) -> list[str]:
        """What files would break if I change this file?"""
        ...

    def get_architecture_summary(self) -> str:
        """High-level summary of the project architecture."""
        ...

    def find_related_code(self, query: str) -> list[CodeSnippet]:
        """Semantic search across the codebase."""
        ...

    def detect_issues(self) -> list[Issue]:
        """Find architectural issues: circular deps, unused code, etc."""
        ...
```

**Scale targets from spec**:
- 10K LoC: <5 sec index, <100ms query
- 100K LoC: <30 sec index, <100ms query
- 1M LoC: <5 min index, <200ms query

---

## How to Add Codebase Analysis (Implementation Guide)

### Step 1: Create CodebaseIndex Class (4-6 hours)

**File**: `src/gaia/agents/gaia_code/codebase_index.py`

```python
import ast
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class SymbolInfo:
    """Information about a symbol (class, function, variable)."""
    name: str
    type: str  # "class", "function", "variable"
    file_path: str
    line_number: int
    docstring: Optional[str] = None

@dataclass
class FileInfo:
    """Information about a source file."""
    path: str
    symbols: List[SymbolInfo]
    imports: List[str]
    size_bytes: int
    last_modified: datetime

class CodebaseIndex:
    """Indexes codebase for fast navigation and analysis."""

    def __init__(self, root_path: str, state: SharedAgentState):
        self.root = Path(root_path)
        self.state = state
        self.files: Dict[str, FileInfo] = {}
        self.symbols: Dict[str, List[SymbolInfo]] = {}
        self.imports: Dict[str, List[str]] = {}
        self.dependents: Dict[str, Set[str]] = {}

    def index_repository(self) -> Dict:
        """Index the entire repository."""
        # Walk all Python files
        py_files = list(self.root.rglob("*.py"))

        for file_path in py_files:
            self._index_file(file_path)

        # Build dependency graph
        self._build_dependency_graph()

        # Detect issues
        issues = self._detect_issues()

        # Store in knowledge DB
        self._persist_to_knowledge_db()

        return {
            "files_indexed": len(self.files),
            "symbols_found": sum(len(syms) for syms in self.symbols.values()),
            "issues_found": len(issues),
        }

    def _index_file(self, file_path: Path):
        """Index a single Python file using AST."""
        try:
            with open(file_path, 'r') as f:
                source = f.read()

            tree = ast.parse(source)

            symbols = []
            imports = []

            for node in ast.walk(tree):
                # Extract class definitions
                if isinstance(node, ast.ClassDef):
                    symbols.append(SymbolInfo(
                        name=node.name,
                        type="class",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        docstring=ast.get_docstring(node)
                    ))

                # Extract function definitions
                elif isinstance(node, ast.FunctionDef):
                    symbols.append(SymbolInfo(
                        name=node.name,
                        type="function",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        docstring=ast.get_docstring(node)
                    ))

                # Extract imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Store file info
            self.files[str(file_path)] = FileInfo(
                path=str(file_path),
                symbols=symbols,
                imports=imports,
                size_bytes=file_path.stat().st_size,
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
            )

            # Index symbols
            for symbol in symbols:
                if symbol.name not in self.symbols:
                    self.symbols[symbol.name] = []
                self.symbols[symbol.name].append(symbol)

            # Store imports
            self.imports[str(file_path)] = imports

        except Exception as e:
            logger.warning(f"Failed to index {file_path}: {e}")

    def _build_dependency_graph(self):
        """Build reverse dependency graph."""
        for file_path, imported_modules in self.imports.items():
            for module in imported_modules:
                # Find file that defines this module
                module_file = self._resolve_import(module, file_path)
                if module_file:
                    if module_file not in self.dependents:
                        self.dependents[module_file] = set()
                    self.dependents[module_file].add(file_path)

    def _detect_issues(self) -> List[Dict]:
        """Detect architectural issues."""
        issues = []

        # Detect circular dependencies
        circular = self._find_circular_dependencies()
        for cycle in circular:
            issues.append({
                "type": "circular_dependency",
                "severity": "warning",
                "files": cycle,
                "message": f"Circular dependency: {' -> '.join(cycle)}"
            })

        # Detect unused imports
        # Detect missing docstrings
        # Detect large files (>500 lines)
        # etc.

        return issues

    def get_architecture_summary(self) -> str:
        """Generate architecture summary."""
        total_files = len(self.files)
        total_symbols = sum(len(syms) for syms in self.symbols.values())
        total_classes = sum(1 for syms in self.symbols.values()
                           for s in syms if s.type == "class")
        total_functions = sum(1 for syms in self.symbols.values()
                             for s in syms if s.type == "function")

        # Find main modules (directories with most files)
        modules = self._group_by_module()

        return f"""Repository Analysis:

Files: {total_files}
Symbols: {total_symbols} ({total_classes} classes, {total_functions} functions)
Main modules: {', '.join(modules.keys())}

Dependencies: {len(self.imports)} files with imports
Issues found: {len(self._detect_issues())}
"""
```

### Step 2: Add Codebase Analysis Tools (2-3 hours)

**Add to tools.py**:

```python
@tool
def index_codebase(root_path: str = ".") -> Dict:
    """
    Index a codebase for analysis.

    Extracts:
    - All symbols (classes, functions, variables)
    - Import graph and dependencies
    - Architecture structure
    - Potential issues

    Args:
        root_path: Root directory of codebase

    Returns:
        Index statistics
    """
    from .codebase_index import CodebaseIndex

    index = CodebaseIndex(root_path, self.shared_state)
    stats = index.index_repository()

    # Store in knowledge DB for future sessions
    self.shared_state.knowledge.store_insight(
        category="codebase_analysis",
        content=index.get_architecture_summary(),
        domain="architecture"
    )

    return stats

@tool
def analyze_architecture(scope: str = "full") -> str:
    """
    Analyze codebase architecture.

    Args:
        scope: "full" | "dependencies" | "issues" | "summary"

    Returns:
        Architecture analysis report
    """
    # Delegate to ArchitectureAgent specialist
    result = self.tool_agent_query(
        task=f"Analyze the codebase architecture (scope: {scope})",
        specialist="ArchitectureAgent"
    )
    return result["result"]

@tool
def find_symbol(name: str) -> List[Dict]:
    """
    Find where a symbol is defined and used.

    Args:
        name: Symbol name (class, function, variable)

    Returns:
        List of locations where symbol appears
    """
    # Query the codebase index from knowledge DB
    # Or use grep as fallback
    pass

@tool
def get_dependents(file_path: str) -> List[str]:
    """
    Find what files depend on this file.

    Args:
        file_path: Path to file

    Returns:
        List of files that import this file
    """
    # Query dependency graph from codebase index
    pass

@tool
def detect_circular_deps() -> List[List[str]]:
    """
    Find circular dependencies in codebase.

    Returns:
        List of dependency cycles
    """
    # Use codebase index
    pass
```

### Step 3: Enhance ArchitectureAgent (1-2 hours)

**Update** `specialists/architecture_agent.py`:

Add actual workflow implementation:

```python
def analyze_codebase_workflow(self, root_path: str) -> Dict:
    """
    Complete workflow for analyzing an existing codebase.

    Phases:
    1. INDEX: Build symbol and dependency index
    2. ANALYZE: Understand architecture patterns
    3. IDENTIFY: Find issues and anti-patterns
    4. DOCUMENT: Generate architecture documentation
    5. REPORT: Comprehensive analysis report
    """

    # Phase 1: INDEX
    self.transition_state("indexing")
    index_stats = self.call_tool("index_codebase", root_path=root_path)

    # Phase 2: ANALYZE
    self.transition_state("analyzing")
    summary = self.call_tool("get_architecture_summary")
    dependencies = self.call_tool("analyze_dependencies")

    # Phase 3: IDENTIFY
    self.transition_state("identifying_issues")
    circular_deps = self.call_tool("detect_circular_deps")
    antipatterns = self.call_tool("detect_antipatterns")

    # Phase 4: DOCUMENT
    self.transition_state("documenting")
    adr = self.generate_adr(summary, dependencies)

    # Phase 5: REPORT
    self.transition_state("reporting")
    return {
        "index_stats": index_stats,
        "architecture_summary": summary,
        "dependencies": dependencies,
        "issues": circular_deps + antipatterns,
        "documentation": adr
    }
```

---

## What You Get: Complete Codebase Analysis System

### Capability 1: Repository Indexing

**Use case**: Understand a large existing codebase

```bash
# Index the codebase
gaia code "Index this repository and analyze its architecture"
```

**What happens**:
1. CodebaseIndex walks all Python files
2. Extracts symbols using AST (classes, functions, variables)
3. Builds import graph and dependency graph
4. Stores in knowledge.db
5. Returns architecture summary

**Result**:
```
Repository Analysis:
  Files: 1,247
  Symbols: 8,945 (523 classes, 8,422 functions)
  Main modules: agents/, llm/, rag/, mcp/, api/
  Dependencies: 1,247 files with imports
  Issues found: 3 circular dependencies, 45 large files

Index stored in knowledge DB for future queries.
```

### Capability 2: Deep Dependency Analysis

**Use case**: Understand how components connect

```bash
# Analyze dependencies
gaia code "Analyze the dependency structure and find circular dependencies"
```

**What happens**:
1. ArchitectureAgent specialist auto-selected
2. Queries codebase index from knowledge DB
3. Analyzes import graph
4. Detects circular dependencies
5. Generates dependency diagram

**Result**:
```
Dependency Analysis:

Circular Dependencies (3):
  1. agents/base/agent.py ↔ agents/base/tools.py
  2. llm/client.py ↔ llm/factory.py
  3. api/server.py ↔ api/middleware.py

Module Structure:
  agents/ (45 files)
    ├─ base/ (8 files) - Core agent framework
    ├─ chat/ (12 files) - Chat agent
    └─ code/ (25 files) - Code agent

  llm/ (23 files)
    ├─ providers/ (15 files) - LLM providers
    └─ clients/ (8 files) - Client implementations

Recommendations:
  - Break circular dependency in agents/base via interface
  - Consider splitting api/middleware.py
```

### Capability 3: Bug Pattern Detection

**Use case**: Find common bug patterns

```bash
# Find potential bugs
gaia code "Analyze this codebase for common bug patterns and security issues"
```

**What happens**:
1. SecurityAgent + DebuggerAgent auto-selected
2. Scan for common patterns:
   - Unchecked None access
   - Missing error handling
   - SQL injection vulnerabilities
   - XSS vulnerabilities
   - Resource leaks
3. Generate report with locations

**Result**:
```
Bug Pattern Analysis:

High Priority (3):
  1. auth/login.py:45 - SQL injection risk (string concatenation)
  2. api/users.py:123 - Unchecked None access
  3. utils/file.py:67 - Path traversal vulnerability

Medium Priority (12):
  - Missing error handling in 8 functions
  - 4 functions with high cyclomatic complexity

Recommendations:
  1. Use parameterized queries in auth/login.py
  2. Add None check before .attribute access
  3. Validate and sanitize file paths
```

### Capability 4: Documentation Gap Analysis

**Use case**: Find undocumented code

```bash
# Find documentation gaps
gaia code "Analyze what code lacks documentation and generate comprehensive docs"
```

**What happens**:
1. DocumentationAgent auto-selected
2. Scans codebase index for:
   - Functions without docstrings
   - Classes without docstrings
   - Modules without README
   - APIs without documentation
3. Generates missing documentation

**Result**:
```
Documentation Analysis:

Missing Docstrings:
  - 234 functions (52% of total)
  - 45 classes (31% of total)

Missing Module Docs:
  - agents/routing/ - No README
  - llm/providers/ - No README

Generating Documentation:
  ✅ Added docstrings to 234 functions
  ✅ Added docstrings to 45 classes
  ✅ Created agents/routing/README.md
  ✅ Created llm/providers/README.md
```

### Capability 5: Architecture Documentation

**Use case**: Generate comprehensive architecture docs

```bash
# Generate architecture documentation
gaia code "Generate complete architecture documentation for this repository"
```

**What happens**:
1. ArchitectureAgent analyzes structure
2. DocumentationAgent generates docs
3. Creates:
   - High-level architecture diagram
   - Module-by-module documentation
   - Dependency graphs
   - API documentation
   - ADRs (Architecture Decision Records)

---

## Implementation Effort

### To Add Full Codebase Analysis (M7):

**Estimated Time**: 15-20 hours total

1. **CodebaseIndex class** (4-6 hours)
   - AST-based symbol extraction
   - Import graph building
   - Dependency analysis
   - Circular dependency detection

2. **Analysis tools** (3-4 hours)
   - `index_codebase()`
   - `analyze_architecture()`
   - `find_symbol()`
   - `get_dependents()`
   - `detect_circular_deps()`
   - `find_issues()`

3. **Tree-sitter integration** (4-6 hours) - OPTIONAL
   - Faster parsing for large codebases
   - Multi-language support (not just Python)
   - Better accuracy

4. **ArchitectureAgent enhancement** (2-3 hours)
   - Wire up the analysis tools
   - Implement complete workflow
   - Add visualization generation

5. **Testing** (2-3 hours)
   - Test on GAIA repo itself (100K+ LoC)
   - Validate analysis accuracy
   - Benchmark performance

---

## Current Status vs Full Capability

### ✅ Already Implemented (M0-M6)

| Component | Status | Purpose |
|-----------|--------|---------|
| **ProjectManifest** | ✅ | Tracks files, APIs, schemas, decisions |
| **ArchitectureAgent** | ✅ | SOLID, patterns, anti-pattern knowledge |
| **knowledge.db** | ✅ | Stores analysis results |
| **Vector search** | ✅ | Semantic code search |
| **Recursive analysis** | ✅ | RLM patterns for large codebases |
| **InsightEngine** | ✅ | Learn patterns from analysis |

### ⏳ Needs to Be Added (M7)

| Component | Status | Purpose |
|-----------|--------|---------|
| **CodebaseIndex** | ⏳ | Symbol + dependency indexing |
| **Analysis tools** | ⏳ | `index_codebase()`, `analyze_architecture()`, etc. |
| **Tree-sitter** | ⏳ Optional | Multi-language parsing |
| **Issue detection** | ⏳ | Automated bug pattern finding |

---

## The Answer: YES, with M7 Addition

### Current Capability (M0-M6 implemented)

**What works NOW**:
- ✅ Agent can analyze code file-by-file
- ✅ Agent can use ArchitectureAgent for design review
- ✅ Agent can store insights about code in knowledge.db
- ✅ Agent can recall past insights about similar code
- ✅ Agent can use recursive decomposition for large analysis tasks

**Example (works now)**:
```bash
gaia code "Review the architecture of agents/base/agent.py and suggest improvements"
```

Agent will:
1. Read the file
2. Use ArchitectureAgent to analyze
3. Check for SOLID violations
4. Suggest improvements
5. Store insights for future

### Full Capability (After M7 - 15-20 hours)

**What will work AFTER M7**:
- ✅ Agent can index 1M+ LoC repositories in minutes
- ✅ Agent can query "find all classes that implement X"
- ✅ Agent can build complete dependency graphs
- ✅ Agent can detect circular dependencies automatically
- ✅ Agent can generate architecture diagrams
- ✅ Agent can find bug patterns across entire codebase
- ✅ Agent can identify documentation gaps
- ✅ Agent can understand system without reading every file

**Example (after M7)**:
```bash
gaia code "Analyze this 100K LoC repository: architecture, dependencies, bugs, and docs gaps"
```

Agent will:
1. Index entire codebase (30 seconds)
2. Build dependency graph
3. Detect issues (circular deps, antipatterns, bugs)
4. Find documentation gaps
5. Generate comprehensive report
6. Store in knowledge.db for future queries

---

## Recommendation: Add M7 for Production Use

### Why M7 is Important

If you plan to use GAIA Code on **existing codebases** (not just greenfield projects), you should add M7.

**Benefits**:
- **Fast**: Index 100K LoC in <30 seconds
- **Comprehensive**: Full symbol + dependency analysis
- **Persistent**: Index stored in knowledge DB
- **Smart**: Semantic search over code
- **Accurate**: AST-based (not regex)

### How to Add M7

I can implement M7 for you right now if you'd like. It would add:

1. **CodebaseIndex class** - Complete indexing system
2. **6 analysis tools** - index_codebase, analyze_architecture, find_symbol, etc.
3. **Integration with ArchitectureAgent** - Wire up the tools
4. **Tests** - Validate on GAIA repo itself
5. **Documentation** - How to use

**Time**: 15-20 hours of implementation

**Result**: Complete codebase analysis capability for existing repositories

---

## Bottom Line

**Current State**: ✅ Architecture SUPPORTS codebase analysis
- Foundation is complete (M0-M6)
- ArchitectureAgent exists
- Knowledge storage ready
- Recursive analysis patterns ready

**To Enable Full Analysis**: Add M7 (15-20 hours)
- CodebaseIndex class
- Analysis tools
- Integration

**Should You Add It?**:
- ✅ YES if working with existing codebases
- ⏳ LATER if only building new projects

Would you like me to implement M7 now to add complete codebase analysis?
