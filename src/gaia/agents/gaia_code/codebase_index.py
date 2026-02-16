# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
CodebaseIndex: Comprehensive repository indexing and analysis.

M7: Codebase Indexing + Architecture Analysis

Enables the agent to:
- Index 1M+ lines of code in minutes
- Extract all symbols (classes, functions, variables)
- Build dependency and call graphs
- Detect circular dependencies
- Find architectural issues
- Generate architecture summaries

This is essential for working with existing codebases.
"""

import ast
import json
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .shared_state import get_shared_state

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a code symbol (class, function, variable)."""

    name: str
    type: str  # "class", "function", "method", "variable", "import"
    file_path: str
    line_number: int
    module_path: str  # Python module path (e.g., "gaia.agents.base.agent")
    docstring: Optional[str] = None
    parent_class: Optional[str] = None  # For methods
    decorators: List[str] = field(default_factory=list)
    is_public: bool = True  # Not starting with _
    complexity: int = 0  # Cyclomatic complexity for functions


@dataclass
class FileInfo:
    """Information about a source file."""

    path: str
    relative_path: str
    module_path: str
    symbols: List[SymbolInfo] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    from_imports: Dict[str, List[str]] = field(default_factory=dict)
    size_bytes: int = 0
    line_count: int = 0
    last_modified: Optional[datetime] = None
    has_tests: bool = False
    docstring: Optional[str] = None


@dataclass
class DependencyEdge:
    """A dependency between two files."""

    source: str  # File that imports
    target: str  # File that is imported
    import_type: str  # "import" or "from"
    symbols_imported: List[str] = field(default_factory=list)


class CodebaseIndex:
    """
    Comprehensive codebase indexing for fast navigation and analysis.

    Indexes:
    - All symbols (classes, functions, methods, variables)
    - All imports and dependencies
    - Module structure
    - Test coverage
    - Documentation coverage

    Enables:
    - Fast symbol lookup ("where is class X defined?")
    - Dependency analysis ("what depends on file Y?")
    - Circular dependency detection
    - Architecture visualization
    - Issue detection (antipatterns, missing docs, etc.)
    """

    def __init__(self, root_path: str, workspace_dir: Optional[Path] = None):
        """
        Initialize codebase index.

        Args:
            root_path: Root directory of codebase to index
            workspace_dir: Workspace directory for SharedAgentState
        """
        self.root = Path(root_path).resolve()
        self.state = get_shared_state(workspace_dir)

        # Index data structures
        self.files: Dict[str, FileInfo] = {}
        self.symbols: Dict[str, List[SymbolInfo]] = defaultdict(list)
        self.imports: Dict[str, List[str]] = {}  # file -> list of imported modules
        self.dependents: Dict[str, Set[str]] = defaultdict(set)  # file -> files that import it
        self.dependencies: List[DependencyEdge] = []

        # Analysis results
        self.circular_deps: List[List[str]] = []
        self.issues: List[Dict] = []

        # Statistics
        self.stats = {
            "files_indexed": 0,
            "symbols_found": 0,
            "classes": 0,
            "functions": 0,
            "imports": 0,
            "issues": 0,
        }

    def index_repository(
        self,
        include_tests: bool = True,
        include_venv: bool = False,
        extensions: List[str] = None,
    ) -> Dict:
        """
        Index the entire repository.

        Args:
            include_tests: Include test files (default: True)
            include_venv: Include virtual environment (default: False)
            extensions: File extensions to index (default: [".py"])

        Returns:
            Index statistics
        """
        logger.info(f"Indexing repository: {self.root}")
        start_time = datetime.now()

        if extensions is None:
            extensions = [".py"]

        # Walk all source files
        for file_path in self._walk_source_files(include_tests, include_venv, extensions):
            self._index_file(file_path)

        # Build dependency graph
        self._build_dependency_graph()

        # Detect circular dependencies
        self.circular_deps = self._detect_circular_dependencies()

        # Detect issues
        self.issues = self._detect_issues()

        # Update statistics
        self._update_stats()

        # Persist to knowledge DB
        self._persist_to_knowledge_db()

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Indexed {self.stats['files_indexed']} files "
            f"with {self.stats['symbols_found']} symbols in {elapsed:.2f}s"
        )

        return {
            **self.stats,
            "index_time_seconds": elapsed,
            "circular_dependencies": len(self.circular_deps),
        }

    def _walk_source_files(
        self, include_tests: bool, include_venv: bool, extensions: List[str]
    ) -> List[Path]:
        """Walk and collect all source files."""
        files = []

        exclude_patterns = [
            "__pycache__",
            ".git",
            ".tox",
            "node_modules",
            ".pytest_cache",
            "build",
            "dist",
            "*.egg-info",
        ]

        if not include_venv:
            exclude_patterns.extend(["venv", ".venv", "env", "virtualenv"])

        for ext in extensions:
            for file_path in self.root.rglob(f"*{ext}"):
                # Skip excluded patterns
                if any(pattern in str(file_path) for pattern in exclude_patterns):
                    continue

                # Skip test files if requested
                if not include_tests and ("test_" in file_path.name or file_path.name.endswith("_test.py")):
                    continue

                files.append(file_path)

        return files

    def _index_file(self, file_path: Path):
        """Index a single Python file using AST."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            # Parse AST
            tree = ast.parse(source, filename=str(file_path))

            # Extract file-level info
            relative_path = file_path.relative_to(self.root)
            module_path = self._path_to_module(relative_path)

            file_info = FileInfo(
                path=str(file_path),
                relative_path=str(relative_path),
                module_path=module_path,
                size_bytes=file_path.stat().st_size,
                line_count=len(source.splitlines()),
                last_modified=datetime.fromtimestamp(file_path.stat().st_mtime),
                has_tests="test" in file_path.name.lower(),
                docstring=ast.get_docstring(tree),
            )

            # Extract symbols
            symbols = []
            imports = []
            from_imports = {}

            for node in ast.walk(tree):
                # Classes
                if isinstance(node, ast.ClassDef):
                    symbols.append(
                        SymbolInfo(
                            name=node.name,
                            type="class",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            module_path=module_path,
                            docstring=ast.get_docstring(node),
                            decorators=[
                                self._get_decorator_name(d) for d in node.decorator_list
                            ],
                            is_public=not node.name.startswith("_"),
                        )
                    )

                # Functions
                elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    # Determine if it's a method or function
                    parent_class = None
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            if node in ast.walk(parent):
                                parent_class = parent.name
                                break

                    symbol_type = "method" if parent_class else "function"

                    symbols.append(
                        SymbolInfo(
                            name=node.name,
                            type=symbol_type,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            module_path=module_path,
                            docstring=ast.get_docstring(node),
                            parent_class=parent_class,
                            decorators=[
                                self._get_decorator_name(d) for d in node.decorator_list
                            ],
                            is_public=not node.name.startswith("_"),
                            complexity=self._calculate_complexity(node),
                        )
                    )

                # Imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.module not in from_imports:
                            from_imports[node.module] = []
                        from_imports[node.module].extend([alias.name for alias in node.names])

            # Store file info
            file_info.symbols = symbols
            file_info.imports = imports
            file_info.from_imports = from_imports

            self.files[str(file_path)] = file_info

            # Index symbols by name
            for symbol in symbols:
                self.symbols[symbol.name].append(symbol)

            # Store imports
            all_imports = imports + list(from_imports.keys())
            self.imports[str(file_path)] = all_imports

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}:{e.lineno}: {e.msg}")
        except Exception as e:
            logger.warning(f"Failed to index {file_path}: {e}")

    def _path_to_module(self, relative_path: Path) -> str:
        """Convert file path to Python module path."""
        parts = list(relative_path.parts)

        # Remove .py extension
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]

        # Remove __init__
        if parts[-1] == "__init__":
            parts = parts[:-1]

        # Skip src/ if present
        if parts and parts[0] == "src":
            parts = parts[1:]

        return ".".join(parts)

    def _get_decorator_name(self, decorator) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        return "unknown"

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            # Each branching statement adds 1
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _build_dependency_graph(self):
        """Build dependency graph with reverse dependencies."""
        for file_path, imported_modules in self.imports.items():
            for module in imported_modules:
                # Try to resolve module to file
                target_file = self._resolve_import_to_file(module, file_path)

                if target_file:
                    # Add to dependents (reverse dependency)
                    self.dependents[target_file].add(file_path)

                    # Create dependency edge
                    self.dependencies.append(
                        DependencyEdge(
                            source=file_path,
                            target=target_file,
                            import_type="import",
                        )
                    )

    def _resolve_import_to_file(self, module: str, importing_file: str) -> Optional[str]:
        """
        Resolve an import statement to a file in the codebase.

        Args:
            module: Module name (e.g., "gaia.agents.base.agent")
            importing_file: File doing the importing

        Returns:
            Path to the file defining the module, or None
        """
        # Convert module to potential file paths
        module_parts = module.split(".")

        # Try different variations
        potential_paths = [
            # module.py
            self.root / "/".join(module_parts[:-1]) / f"{module_parts[-1]}.py",
            # module/__init__.py
            self.root / "/".join(module_parts) / "__init__.py",
            # src/module.py
            self.root / "src" / "/".join(module_parts[:-1]) / f"{module_parts[-1]}.py",
            # src/module/__init__.py
            self.root / "src" / "/".join(module_parts) / "__init__.py",
        ]

        for path in potential_paths:
            if path.exists() and str(path) in self.files:
                return str(path)

        return None

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """
        Detect circular dependencies using Tarjan's algorithm.

        Returns:
            List of cycles, where each cycle is a list of file paths
        """
        # Build adjacency list
        graph = defaultdict(list)
        for edge in self.dependencies:
            graph[edge.source].append(edge.target)

        # Find strongly connected components (SCCs)
        sccs = self._tarjan_scc(graph)

        # Return only SCCs with more than 1 node (circular dependencies)
        cycles = [scc for scc in sccs if len(scc) > 1]

        logger.info(f"Found {len(cycles)} circular dependency cycle(s)")

        return cycles

    def _tarjan_scc(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """Tarjan's algorithm for finding strongly connected components."""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = defaultdict(bool)
        sccs = []

        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True

            for successor in graph.get(node, []):
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif on_stack[successor]:
                    lowlinks[node] = min(lowlinks[node], index[successor])

            if lowlinks[node] == index[node]:
                scc = []
                while True:
                    successor = stack.pop()
                    on_stack[successor] = False
                    scc.append(successor)
                    if successor == node:
                        break
                sccs.append(scc)

        for node in graph:
            if node not in index:
                strongconnect(node)

        return sccs

    def _detect_issues(self) -> List[Dict]:
        """Detect various code issues and antipatterns."""
        issues = []

        # 1. Circular dependencies
        for cycle in self.circular_deps:
            issues.append(
                {
                    "type": "circular_dependency",
                    "severity": "warning",
                    "files": cycle,
                    "message": f"Circular dependency: {' -> '.join([Path(f).name for f in cycle])}",
                }
            )

        # 2. Large files (>500 lines)
        for file_info in self.files.values():
            if file_info.line_count > 500:
                issues.append(
                    {
                        "type": "large_file",
                        "severity": "info",
                        "file": file_info.path,
                        "lines": file_info.line_count,
                        "message": f"Large file ({file_info.line_count} lines) - consider splitting",
                    }
                )

        # 3. Missing docstrings
        for file_path, file_info in self.files.items():
            missing_docs = []

            for symbol in file_info.symbols:
                if symbol.is_public and not symbol.docstring:
                    if symbol.type == "class" or (symbol.type == "function" and not symbol.name.startswith("_")):
                        missing_docs.append(f"{symbol.type} {symbol.name}")

            if len(missing_docs) > 5:  # More than 5 missing
                issues.append(
                    {
                        "type": "missing_docstrings",
                        "severity": "info",
                        "file": file_path,
                        "count": len(missing_docs),
                        "message": f"{len(missing_docs)} public symbols without docstrings",
                    }
                )

        # 4. High complexity functions (>10)
        for file_info in self.files.values():
            for symbol in file_info.symbols:
                if symbol.type in ("function", "method") and symbol.complexity > 10:
                    issues.append(
                        {
                            "type": "high_complexity",
                            "severity": "warning",
                            "file": file_info.path,
                            "symbol": symbol.name,
                            "complexity": symbol.complexity,
                            "line": symbol.line_number,
                            "message": f"High complexity ({symbol.complexity}) in {symbol.name}",
                        }
                    )

        # 5. Missing tests
        source_files = [f for f in self.files.values() if not f.has_tests]
        test_files = [f for f in self.files.values() if f.has_tests]

        if len(source_files) > 0:
            # Check if source files have corresponding test files
            for source_file in source_files:
                if "test" not in source_file.relative_path:
                    # Look for corresponding test file
                    test_name = f"test_{Path(source_file.path).name}"
                    has_test = any(test_name in f.relative_path for f in test_files)

                    if not has_test and len(source_file.symbols) > 0:
                        issues.append(
                            {
                                "type": "missing_tests",
                                "severity": "info",
                                "file": source_file.path,
                                "message": f"No test file found for {Path(source_file.path).name}",
                            }
                        )

        return issues

    def _update_stats(self):
        """Update index statistics."""
        self.stats["files_indexed"] = len(self.files)
        self.stats["symbols_found"] = sum(
            len(file_info.symbols) for file_info in self.files.values()
        )
        self.stats["classes"] = sum(
            1
            for file_info in self.files.values()
            for symbol in file_info.symbols
            if symbol.type == "class"
        )
        self.stats["functions"] = sum(
            1
            for file_info in self.files.values()
            for symbol in file_info.symbols
            if symbol.type in ("function", "method")
        )
        self.stats["imports"] = sum(len(imports) for imports in self.imports.values())
        self.stats["issues"] = len(self.issues)

    def _persist_to_knowledge_db(self):
        """Persist index to knowledge DB for cross-session access."""
        # Store architecture summary
        summary = self.get_architecture_summary()
        self.state.knowledge.store_insight(
            category="architecture",
            content=summary,
            domain="codebase_analysis",
            triggers=["architecture", "codebase", "structure"],
        )

        # Store each issue as a learning
        for issue in self.issues:
            if issue["severity"] in ("warning", "critical"):
                self.state.knowledge.store_insight(
                    category="code_issue",
                    content=f"{issue['type']}: {issue['message']}",
                    domain="code_quality",
                    triggers=[issue["type"], "issue", "codebase"],
                )

        logger.info(f"Persisted index to knowledge DB: {summary[:100]}...")

    # ========================================================================
    # Query Methods
    # ========================================================================

    def find_symbol(self, name: str) -> List[SymbolInfo]:
        """Find all occurrences of a symbol."""
        return self.symbols.get(name, [])

    def find_symbols_by_type(self, symbol_type: str) -> List[SymbolInfo]:
        """Find all symbols of a given type."""
        results = []
        for symbols in self.symbols.values():
            results.extend([s for s in symbols if s.type == symbol_type])
        return results

    def get_file_info(self, file_path: str) -> Optional[FileInfo]:
        """Get information about a file."""
        return self.files.get(file_path)

    def get_dependents(self, file_path: str) -> Set[str]:
        """Get files that depend on this file."""
        return self.dependents.get(file_path, set())

    def get_dependencies(self, file_path: str) -> List[str]:
        """Get files this file depends on."""
        return self.imports.get(file_path, [])

    def get_architecture_summary(self) -> str:
        """Generate high-level architecture summary."""
        # Group files by top-level module
        modules = defaultdict(list)
        for file_info in self.files.values():
            parts = file_info.module_path.split(".")
            if parts:
                top_module = parts[0]
                modules[top_module].append(file_info)

        lines = [
            "# Repository Architecture Summary",
            "",
            f"**Total Files**: {self.stats['files_indexed']}",
            f"**Total Symbols**: {self.stats['symbols_found']} ({self.stats['classes']} classes, {self.stats['functions']} functions)",
            f"**Total Dependencies**: {self.stats['imports']}",
            "",
            "## Module Structure",
            "",
        ]

        for module_name, files in sorted(modules.items(), key=lambda x: -len(x[1]))[:10]:
            symbol_count = sum(len(f.symbols) for f in files)
            lines.append(f"- **{module_name}/** - {len(files)} files, {symbol_count} symbols")

        if self.circular_deps:
            lines.extend(
                [
                    "",
                    "## Issues Detected",
                    "",
                    f"**Circular Dependencies**: {len(self.circular_deps)}",
                ]
            )
            for cycle in self.circular_deps[:5]:
                cycle_names = [Path(f).name for f in cycle]
                lines.append(f"  - {' → '.join(cycle_names)}")

        if len(self.circular_deps) > 5:
            lines.append(f"  - ... and {len(self.circular_deps) - 5} more")

        return "\n".join(lines)

    def get_dependency_report(self) -> str:
        """Generate dependency analysis report."""
        lines = ["# Dependency Analysis", ""]

        # Most depended-upon files
        lines.append("## Most Depended-Upon Files")
        lines.append("")

        dep_counts = [(f, len(deps)) for f, deps in self.dependents.items()]
        dep_counts.sort(key=lambda x: -x[1])

        for file_path, count in dep_counts[:10]:
            rel_path = Path(file_path).relative_to(self.root)
            lines.append(f"- {rel_path} - {count} dependents")

        # Files with most dependencies
        lines.append("")
        lines.append("## Files With Most Dependencies")
        lines.append("")

        imp_counts = [(f, len(imps)) for f, imps in self.imports.items()]
        imp_counts.sort(key=lambda x: -x[1])

        for file_path, count in imp_counts[:10]:
            rel_path = Path(file_path).relative_to(self.root)
            lines.append(f"- {rel_path} - imports {count} modules")

        return "\n".join(lines)

    def get_issues_report(self) -> str:
        """Generate issues report."""
        lines = ["# Code Issues Report", ""]

        # Group by severity
        by_severity = defaultdict(list)
        for issue in self.issues:
            by_severity[issue["severity"]].append(issue)

        for severity in ["critical", "warning", "info"]:
            if severity in by_severity:
                lines.append(f"## {severity.title()} ({len(by_severity[severity])})")
                lines.append("")

                for issue in by_severity[severity][:20]:
                    file_name = Path(issue.get("file", "")).name if "file" in issue else "multiple files"
                    lines.append(f"- [{issue['type']}] {file_name}: {issue['message']}")

                if len(by_severity[severity]) > 20:
                    lines.append(f"- ... and {len(by_severity[severity]) - 20} more")

                lines.append("")

        return "\n".join(lines)

    def export_to_json(self, output_file: str):
        """Export index to JSON file."""
        data = {
            "root": str(self.root),
            "indexed_at": datetime.now().isoformat(),
            "stats": self.stats,
            "files": {
                path: {
                    "module": info.module_path,
                    "symbols": [asdict(s) for s in info.symbols],
                    "imports": info.imports,
                    "line_count": info.line_count,
                }
                for path, info in self.files.items()
            },
            "circular_dependencies": self.circular_deps,
            "issues": self.issues,
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Exported index to {output_file}")
