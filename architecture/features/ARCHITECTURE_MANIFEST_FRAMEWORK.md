# Architecture Manifest Framework for Gaia Agent SDK

**Date**: February 6, 2026
**Version**: 1.0
**Scope**: Real-time project tracking and dependency management for long-running multi-file tasks
**Target**: Gaia agents performing complex, multi-phase work (code generation, system design, documentation)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Manifest Data Model](#3-manifest-data-model)
4. [Real-Time Tracking System](#4-real-time-tracking-system)
5. [Dependency Management](#5-dependency-management)
6. [Integration with Continuous Execution](#6-integration-with-continuous-execution)
7. [Manifest Persistence & Database](#7-manifest-persistence--database)
8. [ManifestToolsMixin](#8-manifesttoolsmixin)
9. [Automatic Manifest Updates](#9-automatic-manifest-updates)
10. [Multi-Project Support](#10-multi-project-support)
11. [Configuration](#11-configuration)
12. [Risks & Mitigations](#12-risks--mitigations)
13. [Examples](#13-examples)

---

## 1. Problem Statement

### The Challenge

When a Gaia agent is tasked with long-running, complex projects like:
- "Build a full-stack e-commerce web application with user auth, product catalog, cart, payment integration, admin dashboard, and deployment"
- "Analyze 50 GPU trace files and generate a comprehensive performance comparison report"
- "Refactor a 20,000-line codebase to use dependency injection"

...the agent must track:
- **What exists**: Which files have been created, what's their status
- **What's pending**: What still needs to be built
- **How things relate**: File dependencies, architecture patterns
- **What's been decided**: Architectural decisions and their rationale
- **Progress**: How close to completion, what's blocking
- **Quality status**: Which components are tested, which have errors

### Without a Manifest

After creating 30+ files across a 4-hour continuous execution:
- Agent **forgets what it already built** → creates duplicate components
- Agent **violates its own architecture decisions** → inconsistent patterns
- Agent **can't estimate progress** → "I'm 50% done" with no basis
- Agent **doesn't know what breaks when changing a file** → cascade failures
- Agent **loses context after restart** → can't resume effectively

### The Solution

A **living architecture manifest** that:
1. **Tracks all project entities** in real-time (files, goals, dependencies, decisions)
2. **Updates automatically** as the agent works (no manual bookkeeping)
3. **Persists to database** with full timestamp history
4. **Provides structured queries** ("what depends on file X?", "which goals are blocked?")
5. **Enables resume** after interruption or restart
6. **Feeds into agent context** (manifest summary injected into system prompt)

---

## 2. Architecture Overview

### High-Level Design

```
┌──────────────────────────────────────────────────────────────┐
│                     Agent (continuous execution)              │
│                                                               │
│  Every tool call → Update manifest → Persist to DB          │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 │ Writes
                 ▼
┌──────────────────────────────────────────────────────────────┐
│          Manifest Database (manifest.db - SQLite)             │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Files      │  │ Dependencies │  │   Goals      │      │
│  │  - path      │  │ - from_file  │  │ - id         │      │
│  │  - type      │  │ - to_file    │  │ - status     │      │
│  │  - status    │  │ - type       │  │ - progress   │      │
│  │  - timestamp │  │              │  │ - blocked_by │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Decisions   │  │  Components  │  │  Events      │      │
│  │ - id         │  │ - name       │  │ - timestamp  │      │
│  │ - decision   │  │ - type       │  │ - event_type │      │
│  │ - rationale  │  │ - files      │  │ - details    │      │
│  │ - timestamp  │  │              │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────────────────────────────────────────────┘
                 │
                 │ Reads (context injection)
                 ▼
┌──────────────────────────────────────────────────────────────┐
│              Agent System Prompt (dynamic)                    │
│                                                               │
│  Current project: E-commerce app (React + FastAPI)          │
│  Progress: 45% (18/32 files complete)                       │
│  Active goals: [Implement cart API, Write cart tests]       │
│  Recent decision: Use Stripe for payments (not manual)      │
│  Architecture: Monorepo, JWT auth, PostgreSQL database      │
│  Blocked: Checkout (waiting for Stripe API key)             │
└──────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Single Source of Truth**: Manifest is the definitive record of project state
2. **Automatic Updates**: Agent tools trigger manifest changes (no manual updates)
3. **Timestamped History**: Every change recorded with ISO timestamp
4. **Queryable**: Structured queries + semantic search over manifest
5. **Context-Aware**: Manifest summary fed to agent on every query
6. **Resumable**: Full state restoration from manifest after restart

---

## 3. Manifest Data Model

### 3.1 Core Entities

#### Project Metadata

```python
@dataclass
class ProjectManifest:
    """Root manifest object."""

    project_id: str                    # UUID
    name: str                          # "E-Commerce Web App"
    description: str                   # Full task description
    created_at: str                    # ISO timestamp
    updated_at: str                    # ISO timestamp (updated on every change)
    root_directory: str                # Absolute path

    architecture: ArchitectureSpec     # High-level architecture
    goals: GoalTracker                 # Goal hierarchy
    files: Dict[str, FileEntry]        # {path: FileEntry}
    dependencies: DependencyGraph      # File/component dependencies
    decisions: List[ArchitectureDecision]  # Design decisions
    components: Dict[str, Component]   # Logical components
    progress: ProgressMetrics          # Completion tracking
    metadata: Dict[str, Any]           # Extensible
```

#### File Entry

```python
@dataclass
class FileEntry:
    """Represents a single file in the project."""

    path: str                          # Relative to project root
    absolute_path: str                 # Full path
    type: str                          # "component", "model", "api_endpoint", "test", "config", etc.
    purpose: str                       # Human-readable description
    status: str                        # "planned", "in_progress", "complete", "failing", "deprecated"
    created_at: str                    # When file was created
    last_modified_at: str              # Most recent modification
    created_by_tool: str               # "write_file", "generate_component", etc.
    modified_by_tools: List[str]       # All tools that touched this file

    # Code-specific
    language: Optional[str]            # "python", "typescript", etc.
    entry_point: bool                  # Is this a main entry point?
    lines_of_code: int
    dependencies: List[str]            # Files this file imports/requires
    dependents: List[str]              # Files that depend on this file

    # Component membership
    component: Optional[str]           # Which logical component (e.g., "auth-system")

    # Quality tracking
    quality_checks: Dict[str, str]     # {"syntax": "pass", "tests": "pass", "lint": "3 warnings"}
    test_coverage: Optional[float]     # Percentage (if applicable)

    # Metadata
    size_bytes: int
    checksum: str                      # SHA-256 of content (detect external changes)
    metadata: Dict[str, Any]
```

#### Goal

```python
@dataclass
class Goal:
    """A project goal or milestone."""

    id: str                            # "implement-user-auth"
    title: str                         # "User Authentication System"
    description: str                   # Detailed requirement
    status: str                        # "pending", "in_progress", "complete", "blocked"
    priority: int                      # 0 (highest) to N
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    # Hierarchy
    parent_goal: Optional[str]         # ID of parent goal (if sub-goal)
    sub_goals: List[str]               # IDs of child goals

    # Dependencies
    depends_on: List[str]              # Goal IDs that must complete first
    blocks: List[str]                  # Goal IDs waiting on this

    # Implementation
    related_files: List[str]           # Files implementing this goal
    acceptance_criteria: List[str]     # How to verify completion
    verification_status: Dict[str, bool]  # {"tests_pass": True, "manual_test": False}

    # Progress
    progress_percent: float            # 0.0 to 100.0
    progress_notes: List[str]          # Status updates
```

#### Dependency

```python
@dataclass
class Dependency:
    """Dependency between files or components."""

    id: str
    from_entity: str                   # File path or component ID
    to_entity: str                     # File path or component ID
    dependency_type: str               # "import", "api_call", "config", "data_flow"
    strength: str                      # "strong" (import), "weak" (optional), "runtime" (dynamic)
    detected_at: str                   # ISO timestamp
    detected_by: str                   # Tool that discovered this ("analyze_imports", "manual")

    # For imports
    import_statement: Optional[str]    # Actual code line

    # For API calls
    endpoint: Optional[str]            # If calling an API endpoint
```

#### Architecture Decision

```python
@dataclass
class ArchitectureDecision:
    """Record of an architectural decision (ADR pattern)."""

    id: str                            # "adr-001-use-jwt"
    timestamp: str                     # ISO timestamp
    decision: str                      # "Use JWT tokens for authentication"
    rationale: str                     # "Stateless, enables horizontal scaling, widely supported"
    alternatives_considered: List[str] # ["Session cookies", "OAuth2 only"]
    consequences: List[str]            # ["Must store refresh tokens", "Client handles expiry"]
    status: str                        # "proposed", "accepted", "deprecated", "superseded"
    superseded_by: Optional[str]       # ID of decision that replaced this
    affected_files: List[str]          # Files implementing this decision
    affected_components: List[str]     # Components affected
```

#### Component

```python
@dataclass
class Component:
    """Logical grouping of related files (e.g., 'auth-system', 'payment-integration')."""

    id: str
    name: str                          # "User Authentication System"
    type: str                          # "feature", "infrastructure", "service", "library"
    purpose: str
    created_at: str
    status: str                        # "planned", "in_progress", "complete"

    files: List[str]                   # File paths in this component
    dependencies: List[str]            # Component IDs this depends on
    api_surface: List[str]             # Public API (functions/endpoints exposed)

    # Implementation
    tech_stack: List[str]              # ["FastAPI", "SQLAlchemy", "bcrypt"]
    entry_points: List[str]            # File paths that are entry points

    # Quality
    test_coverage: float
    quality_score: float               # Composite score
```

#### Progress Metrics

```python
@dataclass
class ProgressMetrics:
    """Computed progress statistics."""

    overall_percent: float             # 0.0 to 100.0
    last_updated_at: str

    # Files
    files_total: int
    files_complete: int
    files_in_progress: int
    files_failing: int
    files_pending: int

    # Goals
    goals_total: int
    goals_complete: int
    goals_blocked: int

    # Quality
    tests_total: int
    tests_passing: int
    tests_failing: int
    test_coverage_percent: float

    # Effort
    tokens_used: int
    time_elapsed_minutes: int
    estimated_time_remaining_minutes: Optional[int]
```

### 3.2 Database Schema

SQLite database at `.gaia/manifests/{project_id}/manifest.db`:

```sql
-- Project root
CREATE TABLE project (
    project_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    root_directory TEXT NOT NULL,
    architecture_json TEXT,    -- JSON blob
    metadata_json TEXT
);

-- Files
CREATE TABLE files (
    path TEXT PRIMARY KEY,
    absolute_path TEXT NOT NULL,
    type TEXT,
    purpose TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    last_modified_at TEXT NOT NULL,
    created_by_tool TEXT,
    language TEXT,
    entry_point BOOLEAN DEFAULT 0,
    lines_of_code INTEGER,
    component_id TEXT,
    size_bytes INTEGER,
    checksum TEXT,
    quality_checks_json TEXT,
    test_coverage REAL,
    metadata_json TEXT,

    FOREIGN KEY (component_id) REFERENCES components(id)
);

CREATE INDEX idx_files_status ON files(status);
CREATE INDEX idx_files_type ON files(type);
CREATE INDEX idx_files_component ON files(component_id);
CREATE INDEX idx_files_modified ON files(last_modified_at DESC);

-- File modification history (for time-based queries)
CREATE TABLE file_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event TEXT NOT NULL,       -- 'created', 'modified', 'deleted', 'status_changed'
    old_value TEXT,            -- Previous content/status
    new_value TEXT,            -- New content/status
    tool TEXT,                 -- Tool that made the change

    FOREIGN KEY (path) REFERENCES files(path)
);

CREATE INDEX idx_file_history_timestamp ON file_history(timestamp DESC);

-- Dependencies
CREATE TABLE dependencies (
    id TEXT PRIMARY KEY,
    from_entity TEXT NOT NULL,
    to_entity TEXT NOT NULL,
    dependency_type TEXT,
    strength TEXT,
    detected_at TEXT NOT NULL,
    detected_by TEXT,
    import_statement TEXT,
    endpoint TEXT,
    metadata_json TEXT
);

CREATE INDEX idx_dependencies_from ON dependencies(from_entity);
CREATE INDEX idx_dependencies_to ON dependencies(to_entity);
CREATE INDEX idx_dependencies_type ON dependencies(dependency_type);

-- Goals
CREATE TABLE goals (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT,
    priority INTEGER,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    parent_goal TEXT,
    progress_percent REAL DEFAULT 0.0,
    metadata_json TEXT,

    FOREIGN KEY (parent_goal) REFERENCES goals(id)
);

CREATE INDEX idx_goals_status ON goals(status);
CREATE INDEX idx_goals_priority ON goals(priority);
CREATE INDEX idx_goals_parent ON goals(parent_goal);

-- Goal dependencies
CREATE TABLE goal_dependencies (
    goal_id TEXT NOT NULL,
    depends_on_goal_id TEXT NOT NULL,
    created_at TEXT NOT NULL,

    PRIMARY KEY (goal_id, depends_on_goal_id),
    FOREIGN KEY (goal_id) REFERENCES goals(id),
    FOREIGN KEY (depends_on_goal_id) REFERENCES goals(id)
);

-- Goal → File relationships
CREATE TABLE goal_files (
    goal_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    relevance REAL DEFAULT 1.0,    -- How relevant this file is to the goal

    PRIMARY KEY (goal_id, file_path),
    FOREIGN KEY (goal_id) REFERENCES goals(id),
    FOREIGN KEY (file_path) REFERENCES files(path)
);

-- Architecture Decisions
CREATE TABLE decisions (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT,
    alternatives_considered TEXT,  -- JSON array
    consequences TEXT,             -- JSON array
    status TEXT,
    superseded_by TEXT,
    metadata_json TEXT,

    FOREIGN KEY (superseded_by) REFERENCES decisions(id)
);

CREATE INDEX idx_decisions_timestamp ON decisions(timestamp DESC);
CREATE INDEX idx_decisions_status ON decisions(status);

-- Decision → File relationships
CREATE TABLE decision_files (
    decision_id TEXT NOT NULL,
    file_path TEXT NOT NULL,

    PRIMARY KEY (decision_id, file_path),
    FOREIGN KEY (decision_id) REFERENCES decisions(id),
    FOREIGN KEY (file_path) REFERENCES files(path)
);

-- Components
CREATE TABLE components (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT,
    purpose TEXT,
    created_at TEXT NOT NULL,
    status TEXT,
    tech_stack TEXT,               -- JSON array
    entry_points TEXT,             -- JSON array
    api_surface TEXT,              -- JSON array
    test_coverage REAL,
    quality_score REAL,
    metadata_json TEXT
);

CREATE INDEX idx_components_status ON components(status);

-- Component dependencies
CREATE TABLE component_dependencies (
    from_component TEXT NOT NULL,
    to_component TEXT NOT NULL,

    PRIMARY KEY (from_component, to_component),
    FOREIGN KEY (from_component) REFERENCES components(id),
    FOREIGN KEY (to_component) REFERENCES components(id)
);

-- Events log (all manifest changes)
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,      -- 'file_created', 'goal_completed', 'decision_made', etc.
    entity_type TEXT,               -- 'file', 'goal', 'component', 'decision'
    entity_id TEXT,
    details_json TEXT,
    triggered_by_tool TEXT
);

CREATE INDEX idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX idx_events_type ON events(event_type);
```

---

## 4. Real-Time Tracking System

### 4.1 Automatic Manifest Updates

The manifest updates in real-time as the agent works. This is implemented by intercepting tool executions:

```python
class ManifestTracker:
    """Real-time manifest updates during agent execution."""

    def __init__(self, project_id: str, db_path: str):
        self.project_id = project_id
        self.db = sqlite3.connect(db_path)
        self._init_schema()

    def on_tool_call(self, tool_name: str, tool_args: dict, tool_result: dict, timestamp: str) -> None:
        """Called after every tool execution (via _post_process_tool_result hook)."""

        # File creation
        if tool_name in ["write_file", "create_file"] and tool_result.get("status") == "success":
            file_path = tool_args.get("file_path") or tool_args.get("path")
            self.add_file(
                path=file_path,
                type=self._infer_file_type(file_path),
                status="complete",
                created_at=timestamp,
                created_by_tool=tool_name,
            )
            self.log_event("file_created", "file", file_path, timestamp, tool_name)

        # File modification
        elif tool_name in ["edit_file", "modify_file"] and tool_result.get("status") == "success":
            file_path = tool_args.get("file_path") or tool_args.get("path")
            self.update_file(path=file_path, last_modified_at=timestamp)
            self.log_file_modification(file_path, timestamp, tool_name)
            self.log_event("file_modified", "file", file_path, timestamp, tool_name)

        # Test execution (update quality checks)
        elif tool_name in ["run_tests", "pytest"] and tool_result.get("status") == "success":
            test_results = tool_result.get("test_results", {})
            for file_path, result in test_results.items():
                status = "pass" if result.get("passed") else "fail"
                self.update_file_quality(file_path, "tests", status, timestamp)

        # Dependency detection
        elif tool_name in ["analyze_imports", "find_dependencies"]:
            deps = tool_result.get("dependencies", [])
            for dep in deps:
                self.add_dependency(
                    from_file=dep["from"],
                    to_file=dep["to"],
                    type="import",
                    detected_at=timestamp,
                    detected_by=tool_name,
                )

        # Goal completion
        elif tool_name == "mark_goal_complete":
            goal_id = tool_args.get("goal_id")
            self.complete_goal(goal_id, timestamp)
            self.log_event("goal_completed", "goal", goal_id, timestamp, tool_name)

    def _infer_file_type(self, path: str) -> str:
        """Infer file type from path and extension."""
        if "/test" in path or path.startswith("test_") or path.endswith("_test.py"):
            return "test"
        elif path.endswith(".tsx") or path.endswith(".jsx"):
            return "component"
        elif "/api/" in path or "/routes/" in path:
            return "api_endpoint"
        elif "/models/" in path:
            return "model"
        elif path.endswith(".sql"):
            return "migration"
        elif path in ["Dockerfile", "docker-compose.yml", ".github/workflows"]:
            return "config"
        else:
            return "source"
```

### 4.2 Timestamp Management

**Critical requirement**: Every entity has multiple timestamps:

- `created_at`: When entity was first added to manifest
- `updated_at`: Last modification time
- `last_accessed_at`: When entity was last read/queried (for LRU)
- Event timestamps: Every change logged with precise timestamp

**Time-aware queries**:

```python
def get_recent_files(self, hours: int = 24) -> List[FileEntry]:
    """Get files modified in the last N hours."""
    cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
    cursor = self.db.execute("""
        SELECT * FROM files
        WHERE last_modified_at > ?
        ORDER BY last_modified_at DESC
    """, (cutoff,))
    return [self._row_to_file_entry(row) for row in cursor.fetchall()]

def get_file_history(self, path: str, since: Optional[str] = None) -> List[dict]:
    """Get modification history for a file."""
    query = "SELECT * FROM file_history WHERE path = ?"
    params = [path]
    if since:
        query += " AND timestamp > ?"
        params.append(since)
    query += " ORDER BY timestamp DESC"
    cursor = self.db.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]
```

### 4.3 Recency Prioritization

Recent information prioritized over old:

```python
def retrieve_context_for_agent(self, max_tokens: int = 2000) -> str:
    """Generate manifest summary for agent system prompt.

    Prioritizes:
    1. Recent activity (last 1 hour) - highest priority
    2. Current goals (in_progress status)
    3. Recent decisions (last 7 days)
    4. Files with failing tests
    5. High-priority pending goals
    """
    sections = []

    # Recent activity (last hour)
    recent = self.get_recent_files(hours=1)
    if recent:
        sections.append(f"RECENT ACTIVITY ({len(recent)} files modified in last hour):")
        for f in recent[:5]:
            sections.append(f"  - {f.path} ({f.type}) - {f.status}")

    # Active goals
    active_goals = self.get_goals_by_status("in_progress")
    if active_goals:
        sections.append(f"\nACTIVE GOALS:")
        for g in active_goals[:5]:
            sections.append(f"  - [{g.progress_percent:.0f}%] {g.title}")

    # Recent decisions
    decisions = self.get_recent_decisions(days=7)
    if decisions:
        sections.append(f"\nRECENT DECISIONS:")
        for d in decisions[:3]:
            sections.append(f"  - {d.decision} (rationale: {d.rationale[:50]}...)")

    # Quality issues
    failing = [f for f in self.get_all_files() if f.quality_checks.get("tests") == "fail"]
    if failing:
        sections.append(f"\nFAILING TESTS ({len(failing)} files):")
        for f in failing[:5]:
            sections.append(f"  - {f.path}")

    return "\n".join(sections)
```

---

## 5. Dependency Management

### 5.1 Dependency Graph

The manifest maintains a directed graph of dependencies:

```python
class DependencyGraph:
    """Manage file and component dependencies."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def add_dependency(self, from_entity: str, to_entity: str, dep_type: str, timestamp: str) -> None:
        """Add a dependency edge."""
        dep_id = f"{from_entity}→{to_entity}".replace("/", "_")
        self.db.execute("""
            INSERT OR REPLACE INTO dependencies
            VALUES (?, ?, ?, ?, 'strong', ?, 'auto', NULL, NULL, NULL)
        """, (dep_id, from_entity, to_entity, dep_type, timestamp))
        self.db.commit()

    def get_dependencies(self, entity: str) -> List[str]:
        """Get all entities this entity depends on."""
        cursor = self.db.execute(
            "SELECT to_entity FROM dependencies WHERE from_entity = ?",
            (entity,)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_dependents(self, entity: str) -> List[str]:
        """Get all entities that depend on this entity."""
        cursor = self.db.execute(
            "SELECT from_entity FROM dependencies WHERE to_entity = ?",
            (entity,)
        )
        return [row[0] for row in cursor.fetchall()]

    def find_impact(self, entity: str, max_depth: int = 5) -> Dict[str, List[str]]:
        """Find all entities affected if this entity changes (transitive closure)."""
        affected = {"direct": [], "indirect": []}

        # Direct dependents
        direct = self.get_dependents(entity)
        affected["direct"] = direct

        # Transitive dependents (BFS)
        visited = set([entity])
        queue = direct.copy()
        depth = 0

        while queue and depth < max_depth:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            affected["indirect"].append(current)

            # Add dependents of current
            deps = self.get_dependents(current)
            queue.extend(deps)
            depth += 1

        return affected

    def detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependency cycles."""
        # Tarjan's algorithm for finding strongly connected components
        # (Implementation omitted for brevity - standard graph algorithm)
        pass

    def visualize_graph(self, output_path: str = "dependency_graph.dot") -> None:
        """Export graph as Graphviz DOT file for visualization."""
        cursor = self.db.execute("SELECT from_entity, to_entity, dependency_type FROM dependencies")
        with open(output_path, "w") as f:
            f.write("digraph Dependencies {\n")
            for from_e, to_e, dtype in cursor:
                label = dtype or ""
                f.write(f'  "{from_e}" -> "{to_e}" [label="{label}"];\n')
            f.write("}\n")
```

### 5.2 Automatic Dependency Detection

Dependencies discovered automatically from code analysis:

```python
class DependencyDetector:
    """Detect dependencies from code."""

    def analyze_file(self, file_path: str, language: str) -> List[Dependency]:
        """Extract dependencies from source file."""
        if language == "python":
            return self._analyze_python(file_path)
        elif language in ["typescript", "javascript"]:
            return self._analyze_typescript(file_path)
        else:
            return []

    def _analyze_python(self, file_path: str) -> List[Dependency]:
        """Extract Python imports."""
        import ast
        with open(file_path) as f:
            tree = ast.parse(f.read())

        deps = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.append(Dependency(
                        id=f"{file_path}→{alias.name}",
                        from_entity=file_path,
                        to_entity=alias.name,
                        dependency_type="import",
                        strength="strong",
                        detected_at=datetime.now().isoformat(),
                        detected_by="auto_import_analysis",
                        import_statement=f"import {alias.name}",
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    deps.append(Dependency(
                        id=f"{file_path}→{module}.{alias.name}",
                        from_entity=file_path,
                        to_entity=f"{module}.{alias.name}",
                        dependency_type="import",
                        strength="strong",
                        detected_at=datetime.now().isoformat(),
                        detected_by="auto_import_analysis",
                        import_statement=f"from {module} import {alias.name}",
                    ))
        return deps

    def _analyze_typescript(self, file_path: str) -> List[Dependency]:
        """Extract TypeScript imports using regex (or TSC API)."""
        import re
        with open(file_path) as f:
            content = f.read()

        deps = []
        # Match: import {...} from "..."  and  import ... from "..."
        import_pattern = r'import\s+.*?\s+from\s+["\'](.+?)["\']'
        for match in re.finditer(import_pattern, content):
            imported_module = match.group(1)
            deps.append(Dependency(
                id=f"{file_path}→{imported_module}",
                from_entity=file_path,
                to_entity=imported_module,
                dependency_type="import",
                strength="strong",
                detected_at=datetime.now().isoformat(),
                detected_by="auto_import_analysis",
                import_statement=match.group(0),
            ))
        return deps
```

---

## 6. Integration with Continuous Execution

The manifest is critical for continuous execution because it enables:

### 6.1 Task Completion Verification

```python
def verify_task_complete(self, task_description: str) -> Dict[str, bool]:
    """Check if all task requirements are met based on manifest state."""
    criteria = {
        "all_goals_complete": self._all_goals_complete(),
        "no_failing_tests": self._no_failing_tests(),
        "no_incomplete_files": self._no_incomplete_files(),
        "all_dependencies_satisfied": self._all_dependencies_satisfied(),
    }
    return {
        "complete": all(criteria.values()),
        "criteria": criteria,
        "blocking_issues": self._get_blocking_issues(),
    }

def _all_goals_complete(self) -> bool:
    cursor = self.db.execute("SELECT COUNT(*) FROM goals WHERE status != 'complete'")
    return cursor.fetchone()[0] == 0

def _no_failing_tests(self) -> bool:
    cursor = self.db.execute(
        "SELECT COUNT(*) FROM files WHERE quality_checks_json LIKE '%\"tests\": \"fail\"%'"
    )
    return cursor.fetchone()[0] == 0

def _get_blocking_issues(self) -> List[str]:
    """Get list of issues preventing task completion."""
    issues = []

    # Failing tests
    failing = self.db.execute(
        "SELECT path FROM files WHERE quality_checks_json LIKE '%\"tests\": \"fail\"%'"
    ).fetchall()
    if failing:
        issues.append(f"{len(failing)} files with failing tests: {[f[0] for f in failing[:3]]}")

    # Blocked goals
    blocked = self.db.execute(
        "SELECT id, title FROM goals WHERE status = 'blocked'"
    ).fetchall()
    if blocked:
        issues.append(f"{len(blocked)} blocked goals: {[g[1] for g in blocked[:3]]}")

    # Incomplete files
    incomplete = self.db.execute(
        "SELECT path FROM files WHERE status IN ('in_progress', 'planned')"
    ).fetchall()
    if incomplete:
        issues.append(f"{len(incomplete)} incomplete files: {[f[0] for f in incomplete[:3]]}")

    return issues
```

### 6.2 Progress Estimation

```python
def compute_progress(self) -> ProgressMetrics:
    """Compute overall project progress."""
    # File progress
    file_stats = self.db.execute("""
        SELECT status, COUNT(*) FROM files GROUP BY status
    """).fetchall()
    file_counts = dict(file_stats)

    files_total = sum(file_counts.values())
    files_complete = file_counts.get("complete", 0)

    # Goal progress
    goal_stats = self.db.execute("""
        SELECT status, COUNT(*) FROM goals GROUP BY status
    """).fetchall()
    goal_counts = dict(goal_stats)

    goals_total = sum(goal_counts.values())
    goals_complete = goal_counts.get("complete", 0)

    # Weighted progress (60% files, 40% goals)
    file_progress = files_complete / max(files_total, 1)
    goal_progress = goals_complete / max(goals_total, 1)
    overall = (file_progress * 0.6) + (goal_progress * 0.4)

    # Test metrics
    test_results = self.db.execute("""
        SELECT json_extract(quality_checks_json, '$.tests') as test_status, COUNT(*)
        FROM files WHERE type = 'test'
        GROUP BY test_status
    """).fetchall()
    test_counts = dict(test_results)

    return ProgressMetrics(
        overall_percent=overall * 100,
        last_updated_at=datetime.now().isoformat(),
        files_total=files_total,
        files_complete=files_complete,
        files_in_progress=file_counts.get("in_progress", 0),
        files_failing=file_counts.get("failing", 0),
        files_pending=file_counts.get("planned", 0),
        goals_total=goals_total,
        goals_complete=goals_complete,
        goals_blocked=goal_counts.get("blocked", 0),
        tests_total=sum(test_counts.values()),
        tests_passing=test_counts.get("pass", 0),
        tests_failing=test_counts.get("fail", 0),
        test_coverage_percent=self._compute_overall_coverage(),
        tokens_used=self._get_tokens_used(),
        time_elapsed_minutes=self._get_time_elapsed(),
        estimated_time_remaining_minutes=self._estimate_remaining_time(),
    )

def _estimate_remaining_time(self) -> Optional[int]:
    """Estimate time remaining based on current pace."""
    progress = self.compute_progress()
    if progress.overall_percent < 5:
        return None  # Too early to estimate

    time_elapsed = progress.time_elapsed_minutes
    completion_rate = progress.overall_percent / time_elapsed  # percent per minute
    remaining_percent = 100 - progress.overall_percent

    return int(remaining_percent / completion_rate)
```

### 6.3 Resume from Checkpoint

When agent restarts mid-task:

```python
def resume_from_manifest(self, project_id: str) -> Dict[str, Any]:
    """Load full project state from manifest to resume execution."""
    manifest = self.load_manifest(project_id)

    # What's been done
    completed_goals = [g for g in manifest.goals if g.status == "complete"]
    completed_files = [f for f in manifest.files.values() if f.status == "complete"]

    # What's in progress
    current_goals = [g for g in manifest.goals if g.status == "in_progress"]
    current_files = [f for f in manifest.files.values() if f.status == "in_progress"]

    # What's next
    next_goals = [g for g in manifest.goals
                  if g.status == "pending"
                  and not g.depends_on]  # No blockers

    return {
        "project_id": project_id,
        "project_name": manifest.name,
        "progress": manifest.progress.overall_percent,
        "completed_goals": [g.title for g in completed_goals],
        "current_goals": [g.title for g in current_goals],
        "next_goals": [g.title for g in next_goals[:3]],
        "files_complete": len(completed_files),
        "files_in_progress": [f.path for f in current_files],
        "blocking_issues": manifest.get_blocking_issues(),
        "resume_prompt": self._generate_resume_prompt(manifest),
    }

def _generate_resume_prompt(self, manifest: ProjectManifest) -> str:
    """Generate a prompt for the agent to resume from manifest state."""
    return f"""You are resuming a task that was interrupted.

PROJECT: {manifest.name}
ORIGINAL TASK: {manifest.description}

PROGRESS SO FAR: {manifest.progress.overall_percent:.0f}% complete

WHAT'S BEEN DONE:
{self._format_completed_goals(manifest)}

WHAT'S IN PROGRESS:
{self._format_current_work(manifest)}

WHAT'S NEXT:
{self._format_next_steps(manifest)}

IMPORTANT: Review the manifest to understand project structure before continuing.
Use the `get_manifest_summary` tool to refresh your understanding."""
```

---

## 7. Manifest Persistence & Database

### 7.1 Atomic Updates

All manifest changes are atomic (transaction-based):

```python
def update_file_status(self, path: str, new_status: str, timestamp: str) -> None:
    """Update file status atomically."""
    try:
        self.db.execute("BEGIN TRANSACTION")

        # Update file
        self.db.execute("""
            UPDATE files
            SET status = ?, last_modified_at = ?
            WHERE path = ?
        """, (new_status, timestamp, path))

        # Log to history
        self.db.execute("""
            INSERT INTO file_history (path, timestamp, event, new_value, tool)
            VALUES (?, ?, 'status_changed', ?, 'system')
        """, (path, timestamp, new_status))

        # Log event
        self.db.execute("""
            INSERT INTO events (timestamp, event_type, entity_type, entity_id, details_json)
            VALUES (?, 'file_status_changed', 'file', ?, ?)
        """, (timestamp, path, json.dumps({"old": "...", "new": new_status})))

        self.db.execute("COMMIT")
    except Exception as e:
        self.db.execute("ROLLBACK")
        raise
```

### 7.2 Memory Consolidation with Contradictions

When new information contradicts existing knowledge:

```python
def add_or_update_knowledge(self, new_entry: dict, timestamp: str) -> str:
    """Add knowledge entry, detecting and resolving conflicts."""
    # Check for contradictions
    conflicts = self._find_conflicts(new_entry)

    if conflicts:
        # Log conflict for human review
        self.db.execute("""
            INSERT INTO knowledge_conflicts (timestamp, new_entry_json, conflicting_ids, status)
            VALUES (?, ?, ?, 'pending_review')
        """, (timestamp, json.dumps(new_entry), json.dumps([c["id"] for c in conflicts])))

        # Apply resolution strategy
        if new_entry.get("confidence", 0) > 0.9:
            # High confidence → deprecate old
            for conflict in conflicts:
                self.db.execute("""
                    UPDATE knowledge
                    SET status = 'superseded', superseded_by = ?, superseded_at = ?
                    WHERE id = ?
                """, (new_entry["id"], timestamp, conflict["id"]))
        else:
            # Low confidence → mark for review, don't auto-supersede
            return "conflict_detected"

    # Insert or update
    existing = self.db.execute("SELECT * FROM knowledge WHERE id = ?", (new_entry["id"],)).fetchone()
    if existing:
        self._update_knowledge(new_entry, timestamp)
        return "updated"
    else:
        self._insert_knowledge(new_entry, timestamp)
        return "inserted"
```

---

## 8. ManifestToolsMixin

Agent-facing tools for manifest interaction:

```python
class ManifestToolsMixin:
    """Mixin providing project manifest management tools."""

    def register_manifest_tools(self) -> None:
        """Register all manifest tools."""

        @tool(atomic=True, name="get_manifest_summary")
        def get_manifest_summary() -> Dict[str, Any]:
            """Get high-level summary of current project state.

            Use at the start of a session or when you need to understand
            project structure. Returns architecture, progress, active goals.
            """
            manifest = self.manifest_tracker.get_current_manifest()
            return {
                "project_name": manifest.name,
                "progress": manifest.progress.overall_percent,
                "architecture": manifest.architecture,
                "active_goals": [g.title for g in manifest.get_active_goals()],
                "recent_files": [f.path for f in manifest.get_recent_files(hours=1)],
                "blocking_issues": manifest.get_blocking_issues(),
                "total_files": manifest.progress.files_total,
                "files_complete": manifest.progress.files_complete,
            }

        @tool(atomic=True, name="get_file_dependencies")
        def get_file_dependencies(file_path: str) -> Dict[str, Any]:
            """Get dependencies for a specific file.

            Use before modifying a file to understand impact.
            """
            deps = self.manifest_tracker.graph.get_dependencies(file_path)
            dependents = self.manifest_tracker.graph.get_dependents(file_path)
            impact = self.manifest_tracker.graph.find_impact(file_path)

            return {
                "file": file_path,
                "depends_on": deps,
                "depended_on_by": dependents,
                "impact_if_changed": impact,
                "warning": "Changing this file affects {len(impact['direct']) + len(impact['indirect'])} other files" if impact else None,
            }

        @tool(name="update_goal_status")
        def update_goal_status(goal_id: str, new_status: str, notes: str = "") -> Dict[str, Any]:
            """Update the status of a goal.

            Use when you complete a goal or hit a blocker.

            Args:
                goal_id: Goal identifier
                new_status: 'pending', 'in_progress', 'complete', 'blocked'
                notes: Optional status update notes
            """
            timestamp = datetime.now().isoformat()
            self.manifest_tracker.update_goal(goal_id, new_status, timestamp, notes)

            # Check if completing this goal unblocks others
            unblocked = self.manifest_tracker.get_unblocked_goals(goal_id)

            return {
                "status": "success",
                "goal_id": goal_id,
                "new_status": new_status,
                "unblocked_goals": [g.title for g in unblocked],
            }

        @tool(name="record_architecture_decision")
        def record_architecture_decision(
            decision: str,
            rationale: str,
            alternatives: str = "",
        ) -> Dict[str, Any]:
            """Record an architectural decision for future reference.

            Use when making significant design choices (auth method,
            database choice, API pattern, etc.).
            """
            timestamp = datetime.now().isoformat()
            decision_id = self.manifest_tracker.add_decision(
                decision=decision,
                rationale=rationale,
                alternatives=alternatives.split(",") if alternatives else [],
                timestamp=timestamp,
            )

            return {
                "status": "success",
                "decision_id": decision_id,
                "message": "Decision recorded and will guide future development",
            }

        @tool(atomic=True, name="check_architecture_consistency")
        def check_architecture_consistency() -> Dict[str, Any]:
            """Check if current code follows established architecture decisions.

            Use before major changes or periodically during long tasks.
            """
            violations = self.manifest_tracker.find_architecture_violations()

            return {
                "consistent": len(violations) == 0,
                "violations": violations,
                "recent_decisions": self.manifest_tracker.get_recent_decisions(days=7),
            }

        @tool(atomic=True, name="get_next_task")
        def get_next_task() -> Dict[str, Any]:
            """Get the next recommended task based on goal priorities and dependencies.

            Use when you complete a task and need to know what to work on next.
            """
            next_goal = self.manifest_tracker.get_highest_priority_unblocked_goal()

            if not next_goal:
                return {"status": "all_complete", "message": "No pending goals"}

            return {
                "status": "success",
                "goal_id": next_goal.id,
                "goal_title": next_goal.title,
                "goal_description": next_goal.description,
                "required_files": next_goal.get_required_files(),
                "dependencies": next_goal.depends_on,
            }
```

---

## 9. Automatic Manifest Updates

### 9.1 Integration with Agent Loop

Manifest updates happen transparently via `_post_process_tool_result()`:

```python
class ManifestAwareAgent(Agent, ...):
    def __init__(self, config):
        super().__init__(config)
        self.manifest_tracker = ManifestTracker(
            project_id=config.project_id,
            db_path=os.path.join(config.manifest_dir, "manifest.db"),
        )

    def _post_process_tool_result(
        self,
        tool_name: str,
        tool_args: dict,
        tool_result: dict
    ) -> None:
        """Hook called after every tool execution."""
        super()._post_process_tool_result(tool_name, tool_args, tool_result)

        # Update manifest based on tool call
        timestamp = datetime.now().isoformat()
        self.manifest_tracker.on_tool_call(tool_name, tool_args, tool_result, timestamp)

        # Re-compute progress
        self.current_progress = self.manifest_tracker.compute_progress()

    def _compose_system_prompt(self) -> str:
        """Inject manifest summary into prompt."""
        parts = [
            super()._compose_system_prompt(),
            "\n--- PROJECT CONTEXT ---",
            self.manifest_tracker.retrieve_context_for_agent(max_tokens=1000),
        ]
        return "\n".join(parts)
```

### 9.2 External Change Detection

If files are modified outside the agent (by user, by other tool):

```python
def scan_for_external_changes(self) -> List[str]:
    """Detect files modified externally (checksum mismatch)."""
    changed = []

    for file_entry in self.get_all_files():
        if os.path.exists(file_entry.absolute_path):
            current_checksum = self._compute_checksum(file_entry.absolute_path)
            if current_checksum != file_entry.checksum:
                changed.append(file_entry.path)
                # Update manifest
                self.update_file(
                    path=file_entry.path,
                    checksum=current_checksum,
                    last_modified_at=datetime.now().isoformat(),
                    modified_by_tools=[...file_entry.modified_by_tools, "external"],
                )

    return changed
```

---

## 10. Multi-Project Support

Agents may work on multiple projects concurrently or sequentially:

```python
class MultiProjectManager:
    """Manage multiple project manifests."""

    def __init__(self, base_dir: str = ".gaia/manifests"):
        self.base_dir = base_dir
        self.active_project_id = None

    def create_project(self, name: str, description: str, root_dir: str) -> str:
        """Initialize a new project manifest."""
        project_id = str(uuid.uuid4())[:8]
        project_dir = os.path.join(self.base_dir, project_id)
        os.makedirs(project_dir, exist_ok=True)

        db_path = os.path.join(project_dir, "manifest.db")
        tracker = ManifestTracker(project_id, db_path)
        tracker.initialize_project(name, description, root_dir)

        self.active_project_id = project_id
        return project_id

    def switch_project(self, project_id: str) -> None:
        """Switch active project context."""
        if not self._project_exists(project_id):
            raise ValueError(f"Project {project_id} not found")
        self.active_project_id = project_id

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects with summaries."""
        projects = []
        for project_id in os.listdir(self.base_dir):
            manifest_path = os.path.join(self.base_dir, project_id, "manifest.db")
            if os.path.exists(manifest_path):
                tracker = ManifestTracker(project_id, manifest_path)
                manifest = tracker.get_current_manifest()
                projects.append({
                    "project_id": project_id,
                    "name": manifest.name,
                    "created_at": manifest.created_at,
                    "updated_at": manifest.updated_at,
                    "progress": manifest.progress.overall_percent,
                })
        return sorted(projects, key=lambda p: p["updated_at"], reverse=True)
```

---

## 11. Configuration

```python
@dataclass
class ManifestConfig:
    """Configuration for manifest system."""

    # Storage
    manifest_dir: str = ".gaia/manifests"
    enable_manifest: bool = True
    auto_detect_dependencies: bool = True
    auto_update: bool = True

    # Context injection
    inject_into_prompt: bool = True
    max_context_tokens: int = 1000

    # Checkpointing
    checkpoint_interval_steps: int = 50
    checkpoint_on_goal_complete: bool = True

    # External change detection
    scan_interval_minutes: int = 5
    warn_on_external_changes: bool = True

    # Progress estimation
    enable_time_estimation: bool = True
    estimation_min_progress: float = 5.0  # Don't estimate until 5% done

    # Quality
    require_tests: bool = True
    min_test_coverage: float = 0.8
```

---

## 12. Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Manifest drift** — manifest out of sync with actual files | HIGH | Automatic checksum verification, periodic scans, atomic updates |
| **Database corruption** — crash during write | MEDIUM | SQLite transactions, write-ahead logging (WAL mode) |
| **Performance** — large projects slow down queries | MEDIUM | Indexes on all query paths, lazy loading, manifest summary caching |
| **Incorrect dependency detection** — AST parser misses dynamic imports | MEDIUM | Conservative: if unsure, add dependency (false positive OK) |
| **Storage bloat** — history tables grow unbounded | LOW | Retention policy: archive events > 90 days |

---

## 13. Examples

### Example 1: Code Agent Building Web App

```python
# User task: "Build e-commerce web app"
agent = CodeAgent(ManifestConfig(enable_manifest=True))

# Agent starts, manifest initialized
project_id = agent.manifest.create_project(
    name="E-Commerce App",
    description="Full-stack app with React + FastAPI",
    root_dir="./ecommerce-app"
)

# Agent creates files, manifest auto-updates
# After 2 hours:
progress = agent.manifest.compute_progress()
# → overall_percent=35, files_complete=12, files_in_progress=2

# Agent can query manifest
deps = agent.manifest.get_dependencies("backend/models/user.py")
# → ["database.py", "auth_utils.py"]

# Agent checks if done
completion = agent.manifest.verify_task_complete("Build e-commerce app")
# → complete=False, blocking_issues=["3 files with failing tests", "Checkout goal blocked (needs Stripe key)"]

# Agent resumes after restart
resume_info = agent.manifest.resume_from_manifest(project_id)
# → Full project state loaded, agent continues from where it left off
```

---

*Architecture Manifest Framework for tracking long-running multi-file projects in Gaia agents.*
*Designed for continuous execution with real-time progress tracking, dependency management, and resumability.*
