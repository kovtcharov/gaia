# Language Extensibility Design for Gaia V2

**Date**: February 6, 2026
**Version**: 1.0
**Scope**: Design principles for language-agnostic, extensible architecture
**Goal**: Gaia V2 core should support any language via plugins, not hardcoded implementations

---

## 1. Design Principle: Core vs. Extensions

```
┌────────────────────────────────────────────────────────────┐
│              GAIA V2 CORE (Language-Agnostic)              │
│                                                            │
│  - Continuous Execution Engine                            │
│  - Task Queue & Management                                │
│  - 4-Tier Persistent Memory                               │
│  - State Machine Framework                                │
│  - Architecture Manifest (generic file tracking)          │
│  - Learning Loop                                          │
│  - Dynamic Tools & SKILLS Framework                       │
│  - TUI & Dashboard                                        │
│  - Voice Interface                                        │
│                                                            │
│  NO language-specific code in core!                       │
└────────────────────────────────────────────────────────────┘
                           │
                           │ Plugin API
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐ ┌──────▼───────┐ ┌───────▼────────┐
│  Python        │ │  TypeScript  │ │  Swift         │
│  Language      │ │  Language    │ │  Language      │
│  Plugin        │ │  Plugin      │ │  Plugin        │
│                │ │              │ │                │
│  - Validator   │ │  - Validator │ │  - Validator   │
│  - Builder     │ │  - Builder   │ │  - Builder     │
│  - Tester      │ │  - Tester    │ │  - Tester      │
│  - Analyzer    │ │  - Analyzer  │ │  - Analyzer    │
│  - Patterns    │ │  - Patterns  │ │  - Patterns    │
└────────────────┘ └──────────────┘ └────────────────┘

        [Future: Go, Rust, Java, Kotlin, C++, etc.]
```

---

## 2. Language Plugin Interface

### 2.1 Abstract Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ValidationResult:
    """Language-agnostic validation result."""
    success: bool
    errors: List[Dict[str, Any]]      # Structured errors
    warnings: List[Dict[str, Any]]
    suggestions: List[str]

@dataclass
class BuildResult:
    """Language-agnostic build result."""
    success: bool
    output_path: Optional[str]        # Compiled output (if applicable)
    errors: List[Dict[str, Any]]
    duration_sec: float

@dataclass
class TestResult:
    """Language-agnostic test result."""
    total: int
    passed: int
    failed: int
    skipped: int
    failures: List[Dict[str, Any]]    # Test name, error, line
    coverage_percent: Optional[float]
    duration_sec: float

@dataclass
class FileStructure:
    """Language-agnostic file structure analysis."""
    imports: List[str]
    exports: Dict[str, str]           # {name: type}
    classes: List[str]
    functions: List[str]
    public_api: List[str]             # Exported symbols
    dependencies: List[str]           # Other files this imports

class LanguagePlugin(ABC):
    """Abstract base for language-specific implementations."""

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Return language name (e.g., 'python', 'swift', 'typescript')."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Return file extensions (e.g., ['.py', '.pyi'])."""
        pass

    @abstractmethod
    def validate_syntax(self, file_path: str, code: str) -> ValidationResult:
        """Check syntax without executing."""
        pass

    @abstractmethod
    def build(self, project_dir: str, build_config: dict) -> BuildResult:
        """Build/compile the project."""
        pass

    @abstractmethod
    def run_tests(self, project_dir: str, test_pattern: str = "") -> TestResult:
        """Execute test suite."""
        pass

    @abstractmethod
    def analyze_file_structure(self, file_path: str, code: str) -> FileStructure:
        """Extract imports, exports, classes, functions."""
        pass

    @abstractmethod
    def get_error_recovery_strategy(self, error: Dict[str, Any]) -> Optional[str]:
        """Suggest how to fix a specific error type."""
        pass

    @abstractmethod
    def get_proven_patterns(self) -> List[Dict[str, str]]:
        """Return language-specific proven code patterns."""
        pass

    @abstractmethod
    def get_language_states(self) -> List[Dict[str, Any]]:
        """Return language/framework-specific agent states."""
        pass

    # Optional methods (have defaults)
    def format_code(self, code: str) -> str:
        """Format code according to language conventions."""
        return code  # Default: no formatting

    def lint(self, file_path: str) -> ValidationResult:
        """Run linter."""
        return ValidationResult(success=True, errors=[], warnings=[], suggestions=[])

    def get_dependency_manager_commands(self) -> Dict[str, str]:
        """Return commands for package management.

        Returns:
            {
                "install": "pip install {package}",
                "add": "pip install {package}",
                "list": "pip list",
            }
        """
        return {}
```

---

## 3. Language Plugin Registry

```python
class LanguageRegistry:
    """Central registry of language plugins."""

    def __init__(self):
        self._plugins: Dict[str, LanguagePlugin] = {}
        self._extension_map: Dict[str, str] = {}  # {'.py': 'python'}

    def register(self, plugin: LanguagePlugin):
        """Register a language plugin."""
        self._plugins[plugin.language_name] = plugin

        # Map file extensions
        for ext in plugin.file_extensions:
            self._extension_map[ext] = plugin.language_name

        logging.info(f"Registered language plugin: {plugin.language_name}")

    def get_plugin(self, language: str) -> Optional[LanguagePlugin]:
        """Get plugin by language name."""
        return self._plugins.get(language)

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect language from file extension."""
        _, ext = os.path.splitext(file_path)
        return self._extension_map.get(ext.lower())

    def get_plugin_for_file(self, file_path: str) -> Optional[LanguagePlugin]:
        """Get appropriate plugin for a file."""
        language = self.detect_language(file_path)
        return self.get_plugin(language) if language else None

# Global registry
LANGUAGE_REGISTRY = LanguageRegistry()
```

---

## 4. Example Plugin: Python

```python
class PythonLanguagePlugin(LanguagePlugin):
    """Python language support."""

    @property
    def language_name(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> List[str]:
        return [".py", ".pyi"]

    def validate_syntax(self, file_path: str, code: str) -> ValidationResult:
        """Compile Python to check syntax."""
        try:
            compile(code, file_path, "exec")
            return ValidationResult(success=True, errors=[], warnings=[], suggestions=[])
        except SyntaxError as e:
            return ValidationResult(
                success=False,
                errors=[{
                    "line": e.lineno,
                    "column": e.offset,
                    "message": e.msg,
                    "severity": "error",
                }],
                warnings=[],
                suggestions=[],
            )

    def build(self, project_dir: str, build_config: dict) -> BuildResult:
        """Python doesn't need compilation, return success."""
        return BuildResult(
            success=True,
            output_path=None,
            errors=[],
            duration_sec=0.0,
        )

    def run_tests(self, project_dir: str, test_pattern: str = "") -> TestResult:
        """Run pytest."""
        cmd = ["pytest", "-v", "--tb=short"]
        if test_pattern:
            cmd.append(test_pattern)

        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Parse pytest output
        test_results = self._parse_pytest_output(result.stdout)

        return TestResult(
            total=test_results["total"],
            passed=test_results["passed"],
            failed=test_results["failed"],
            skipped=test_results["skipped"],
            failures=test_results["failures"],
            coverage_percent=test_results.get("coverage"),
            duration_sec=test_results.get("duration", 0.0),
        )

    def analyze_file_structure(self, file_path: str, code: str) -> FileStructure:
        """Parse Python file with AST."""
        import ast

        tree = ast.parse(code)

        imports = []
        exports = {}
        classes = []
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
                # Public if doesn't start with _
                if not node.name.startswith("_"):
                    exports[node.name] = "function"

        return FileStructure(
            imports=imports,
            exports=exports,
            classes=classes,
            functions=functions,
            public_api=list(exports.keys()),
            dependencies=[],  # Would need import resolution
        )

    def get_proven_patterns(self) -> List[Dict[str, str]]:
        """Return Python-specific patterns."""
        return [
            {
                "name": "FastAPI Endpoint",
                "code": FASTAPI_ENDPOINT_PATTERN,
                "critical_requirements": ["Use async/await", "Pydantic validation", "Error handling"],
            },
            {
                "name": "pytest Fixture",
                "code": PYTEST_FIXTURE_PATTERN,
                "critical_requirements": ["Use @pytest.fixture decorator", "Return reusable object"],
            },
        ]

    def get_language_states(self) -> List[Dict[str, Any]]:
        """Return Python-specific states."""
        return [
            {
                "state_id": "fastapi_development",
                "name": "FastAPI Development",
                "prompt_module": "Focus on async endpoints, Pydantic models, dependency injection...",
                "enabled_tools": ["write_file", "run_tests", "validate_python"],
            },
            {
                "state_id": "data_science",
                "name": "Data Science Mode",
                "prompt_module": "Focus on pandas, numpy, visualization...",
                "enabled_tools": ["write_file", "run_notebook", "plot"],
            },
        ]

# Register plugin
LANGUAGE_REGISTRY.register(PythonLanguagePlugin())
```

---

## 5. Language-Agnostic Tool Implementation

All core tools use the plugin API:

```python
@tool(name="validate_code")
def validate_code(file_path: str) -> Dict[str, Any]:
    """Validate code syntax (language-agnostic).

    Automatically detects language and uses appropriate plugin.
    """
    # Detect language
    plugin = LANGUAGE_REGISTRY.get_plugin_for_file(file_path)

    if not plugin:
        return {
            "status": "error",
            "error": f"No language plugin for {file_path}",
        }

    # Read file
    with open(file_path) as f:
        code = f.read()

    # Validate using plugin
    result = plugin.validate_syntax(file_path, code)

    return {
        "status": "success" if result.success else "error",
        "errors": result.errors,
        "warnings": result.warnings,
        "language": plugin.language_name,
    }

@tool(name="build_project")
def build_project(project_dir: str, language: str = "") -> Dict[str, Any]:
    """Build project (language-agnostic)."""
    # Auto-detect if not specified
    if not language:
        language = self._detect_project_language(project_dir)

    plugin = LANGUAGE_REGISTRY.get_plugin(language)

    if not plugin:
        return {"status": "error", "error": f"No plugin for {language}"}

    result = plugin.build(project_dir, {})

    return {
        "status": "success" if result.success else "error",
        "build_succeeded": result.success,
        "errors": result.errors,
        "duration": result.duration_sec,
    }

@tool(name="run_tests")
def run_tests(project_dir: str, language: str = "", test_pattern: str = "") -> Dict[str, Any]:
    """Run tests (language-agnostic)."""
    if not language:
        language = self._detect_project_language(project_dir)

    plugin = LANGUAGE_REGISTRY.get_plugin(language)

    if not plugin:
        return {"status": "error", "error": f"No plugin for {language}"}

    result = plugin.run_tests(project_dir, test_pattern)

    return {
        "status": "success",
        "total": result.total,
        "passed": result.passed,
        "failed": result.failed,
        "failures": result.failures,
        "coverage": result.coverage_percent,
    }
```

---

## 6. Manifest Extensibility

The manifest is **language-agnostic** with **language-specific analyzers**:

```python
class ManifestTracker:
    """Language-agnostic manifest tracker."""

    def __init__(self, language_registry: LanguageRegistry):
        self.language_registry = language_registry
        self.files: Dict[str, FileEntry] = {}

    def add_file(self, file_path: str, code: str, timestamp: str) -> FileEntry:
        """Add file to manifest (language-agnostic).

        Automatically detects language and uses appropriate analyzer.
        """
        # Detect language
        plugin = self.language_registry.get_plugin_for_file(file_path)

        # Basic file entry
        entry = FileEntry(
            path=file_path,
            type=self._infer_type(file_path),  # Generic: "source", "test", "config"
            status="complete",
            created_at=timestamp,
            last_modified_at=timestamp,
        )

        # Language-specific analysis (if plugin available)
        if plugin:
            structure = plugin.analyze_file_structure(file_path, code)

            entry.imports = structure.imports
            entry.exports = structure.exports
            entry.classes = structure.classes
            entry.functions = structure.functions
            entry.public_api = structure.public_api
            entry.dependencies = structure.dependencies
            entry.language = plugin.language_name

        self.files[file_path] = entry
        return entry
```

---

## 7. Quality Gates Extensibility

```python
class QualityGateSystem:
    """Language-agnostic quality gate orchestrator."""

    def __init__(self, language_registry: LanguageRegistry):
        self.language_registry = language_registry

    def verify_all_gates(
        self,
        project_dir: str,
        language: str,
        gates_config: Dict[str, bool],
    ) -> Dict[str, GateResult]:
        """Run all configured gates for a language."""
        plugin = self.language_registry.get_plugin(language)

        if not plugin:
            raise ValueError(f"No plugin for language: {language}")

        results = {}

        # Level 0: Syntax (always run)
        if gates_config.get("syntax", True):
            results["syntax"] = self._verify_syntax_all_files(project_dir, plugin)

        # Level 1: Build (if applicable)
        if gates_config.get("build", True):
            results["build"] = self._verify_build(project_dir, plugin)

        # Level 2: Tests
        if gates_config.get("tests", True):
            results["tests"] = self._verify_tests(project_dir, plugin)

        # Level 3: Lint (if plugin supports)
        if gates_config.get("lint", False):
            results["lint"] = self._verify_lint(project_dir, plugin)

        # Level 4: Type check (if applicable)
        if gates_config.get("type_check", False):
            results["type_check"] = self._verify_types(project_dir, plugin)

        # Level 5: Runtime (if plugin supports)
        if gates_config.get("runtime", False):
            results["runtime"] = self._verify_runtime(project_dir, plugin)

        return results

    def _verify_syntax_all_files(self, project_dir, plugin) -> GateResult:
        """Validate syntax for all files of this language."""
        all_files = self._find_files_by_extension(project_dir, plugin.file_extensions)

        errors = []
        for file_path in all_files:
            with open(file_path) as f:
                code = f.read()

            result = plugin.validate_syntax(file_path, code)
            if not result.success:
                errors.extend(result.errors)

        return GateResult(
            passed=len(errors) == 0,
            errors=errors,
            message=f"Syntax check: {len(all_files)} files, {len(errors)} errors",
        )
```

---

## 8. Proven Patterns Extensibility

```python
class ProvenPatternsLibrary:
    """Language-agnostic pattern library."""

    def __init__(self, language_registry: LanguageRegistry):
        self.language_registry = language_registry
        self.db = sqlite3.connect(".gaia/patterns/patterns.db")

    def add_pattern(
        self,
        name: str,
        language: str,
        code: str,
        description: str,
        critical_requirements: List[str],
        common_mistakes: List[str],
        timestamp: str,
    ) -> str:
        """Add pattern (language field determines which plugin handles it)."""
        # Validate language is supported
        plugin = self.language_registry.get_plugin(language)
        if not plugin:
            raise ValueError(f"Unsupported language: {language}")

        # Store pattern
        pattern_id = str(uuid.uuid4())[:12]
        self.db.execute("""
            INSERT INTO proven_patterns VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
        """, (pattern_id, name, language, code, description,
              json.dumps(critical_requirements), json.dumps(common_mistakes),
              timestamp, json.dumps({})))

        self.db.commit()
        return pattern_id

    def search_patterns(
        self,
        query: str,
        language: Optional[str] = None,
        min_success_count: int = 3,
    ) -> List[dict]:
        """Search patterns (optionally filtered by language)."""
        sql = "SELECT * FROM proven_patterns WHERE success_count >= ?"
        params = [min_success_count]

        if language:
            sql += " AND language = ?"
            params.append(language)

        cursor = self.db.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def load_language_defaults(self, language: str):
        """Load default patterns from language plugin."""
        plugin = self.language_registry.get_plugin(language)

        if plugin:
            default_patterns = plugin.get_proven_patterns()

            for pattern in default_patterns:
                self.add_pattern(
                    name=pattern["name"],
                    language=language,
                    code=pattern["code"],
                    description=pattern.get("description", ""),
                    critical_requirements=pattern.get("critical_requirements", []),
                    common_mistakes=pattern.get("common_mistakes", []),
                    timestamp=datetime.now().isoformat(),
                )
```

---

## 9. State Machine Extensibility

```python
class StateMachine:
    """Language-agnostic state machine."""

    def __init__(self, language_registry: LanguageRegistry):
        self.language_registry = language_registry
        self.states: Dict[str, AgentState] = {}

        # Load generic states (work for all languages)
        self._load_generic_states()

    def _load_generic_states(self):
        """Load universal states."""
        self.states.update({
            "planning": AgentState(
                state_id="planning",
                name="Planning",
                system_prompt_module="You are in planning mode. Design architecture, no code writing.",
                enabled_tools=["read_file", "analyze_codebase", "search_files"],
            ),
            "implementation": AgentState(
                state_id="implementation",
                name="Implementation",
                system_prompt_module="You are writing code. Follow project conventions.",
                enabled_tools=["write_file", "edit_file", "validate_code", "run_tests"],
            ),
            "testing": AgentState(
                state_id="testing",
                name="Testing",
                system_prompt_module="Run tests and verify quality.",
                enabled_tools=["run_tests", "validate_code", "lint"],
            ),
            "debug": AgentState(
                state_id="debug",
                name="Debug",
                system_prompt_module="Fix errors iteratively with minimal changes.",
                enabled_tools=["read_file", "edit_file", "validate_code", "run_tests"],
            ),
        })

    def load_language_specific_states(self, language: str):
        """Load states specific to a language."""
        plugin = self.language_registry.get_plugin(language)

        if plugin:
            language_states = plugin.get_language_states()

            for state_def in language_states:
                state = AgentState(
                    state_id=f"{language}_{state_def['state_id']}",
                    name=state_def["name"],
                    system_prompt_module=state_def["prompt_module"],
                    enabled_tools=state_def.get("enabled_tools"),
                    disabled_tools=state_def.get("disabled_tools", []),
                    created_at=datetime.now().isoformat(),
                    created_by="language_plugin",
                )

                self.states[state.state_id] = state
```

---

## 10. Adding a New Language (Example: Swift)

**Step 1**: Implement `SwiftLanguagePlugin`

```python
# gaia/plugins/swift.py
from gaia.plugins.base import LanguagePlugin, ValidationResult, BuildResult, TestResult, FileStructure

class SwiftLanguagePlugin(LanguagePlugin):
    @property
    def language_name(self) -> str:
        return "swift"

    @property
    def file_extensions(self) -> List[str]:
        return [".swift"]

    def validate_syntax(self, file_path: str, code: str) -> ValidationResult:
        """Use swiftc -typecheck."""
        result = subprocess.run(
            ["swiftc", "-typecheck", file_path],
            capture_output=True,
            text=True,
        )
        # Parse Swift compiler errors
        # Return ValidationResult
        pass

    def build(self, project_dir: str, build_config: dict) -> BuildResult:
        """Use xcodebuild."""
        pass

    def run_tests(self, project_dir: str, test_pattern: str = "") -> TestResult:
        """Use xcodebuild test."""
        pass

    def analyze_file_structure(self, file_path: str, code: str) -> FileStructure:
        """Parse Swift imports, classes, protocols, structs."""
        # Could use SourceKit or regex parsing
        pass

    def get_proven_patterns(self) -> List[Dict[str, str]]:
        """Return SwiftUI, UIKit, Core Data patterns."""
        return [
            {
                "name": "SwiftUI View + ViewModel",
                "code": SWIFTUI_VIEWMODEL_PATTERN,
                "critical_requirements": [
                    "Use @StateObject for owned viewModel",
                    "@Published for reactive properties",
                    "$ syntax for bindings",
                ],
                "common_mistakes": [
                    "@ObservedObject for owned viewModel (causes recreation)",
                    "Forgetting @Published (UI won't update)",
                ],
            },
        ]

    def get_language_states(self) -> List[Dict[str, Any]]:
        """Return iOS-specific states."""
        return [
            {
                "state_id": "swiftui_development",
                "name": "SwiftUI Development",
                "prompt_module": "Focus on SwiftUI views, @State, @Binding, previews...",
            },
            {
                "state_id": "core_data_setup",
                "name": "Core Data Setup",
                "prompt_module": "Focus on NSManagedObject, fetch requests, migrations...",
            },
        ]
```

**Step 2**: Register plugin

```python
# gaia/plugins/__init__.py
from gaia.plugins.python import PythonLanguagePlugin
from gaia.plugins.typescript import TypeScriptLanguagePlugin
from gaia.plugins.swift import SwiftLanguagePlugin
from gaia.plugins.registry import LANGUAGE_REGISTRY

# Auto-register all built-in plugins
LANGUAGE_REGISTRY.register(PythonLanguagePlugin())
LANGUAGE_REGISTRY.register(TypeScriptLanguagePlugin())
LANGUAGE_REGISTRY.register(SwiftLanguagePlugin())
```

**Step 3**: Use with any agent

```python
# Agent automatically uses Swift plugin when working with .swift files
agent = AgentV2(AgentV2Config(enable_all=True))

task = agent.task_manager.create_task(
    title="Build iOS Weather App",
    description="SwiftUI app...",
)

# Agent detects .swift files, loads Swift plugin, uses Swift-specific validation/build/test
```

---

## 11. Plugin Discovery (Future: Third-Party Plugins)

```python
class LanguageRegistry:
    """Extended with plugin discovery."""

    def discover_plugins(self, plugin_dir: str = ".gaia/plugins"):
        """Auto-discover and load plugins from directory."""
        import importlib.util

        for plugin_file in os.listdir(plugin_dir):
            if plugin_file.endswith("_plugin.py"):
                spec = importlib.util.spec_from_file_location(
                    plugin_file[:-3],
                    os.path.join(plugin_dir, plugin_file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for LanguagePlugin subclasses
                for name, obj in vars(module).items():
                    if isinstance(obj, type) and issubclass(obj, LanguagePlugin):
                        plugin_instance = obj()
                        self.register(plugin_instance)
                        logging.info(f"Discovered plugin: {plugin_instance.language_name}")

# Third-party developers can create plugins:
# .gaia/plugins/rust_plugin.py
# .gaia/plugins/go_plugin.py
# .gaia/plugins/kotlin_plugin.py
```

---

## 12. Configuration Extensibility

```python
@dataclass
class AgentV2Config:
    """Language-agnostic configuration."""

    # Core (no language assumptions)
    enable_continuous_execution: bool = True
    enable_tasks: bool = True
    enable_memory: bool = True
    enable_manifest: bool = True
    enable_states: bool = True
    enable_learning: bool = True
    enable_voice: bool = False

    # Language support (auto-detected or specified)
    primary_language: Optional[str] = None    # Auto-detect if None
    supported_languages: List[str] = None     # Auto: all registered plugins

    # Language-specific overrides (optional)
    language_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

# Example: Multi-language project
config = AgentV2Config(
    enable_all=True,
    language_configs={
        "python": {
            "validator": "mypy",
            "formatter": "black",
            "test_runner": "pytest",
        },
        "typescript": {
            "validator": "tsc",
            "formatter": "prettier",
            "test_runner": "jest",
        },
        "swift": {
            "validator": "swiftc",
            "formatter": "swiftformat",
            "test_runner": "xcodebuild",
        },
    }
)
```

---

## 13. Extensibility Checklist

For each Gaia V2 framework, ensure:

- [ ] **Persistent Memory**: Language field in all entities, but no language-specific logic
- [ ] **Adaptive Prompts**: States can be language-specific, loaded via plugins
- [ ] **Manifest**: Uses plugin API for file analysis, stores language field
- [ ] **Quality Gates**: Delegates to plugins for validation/build/test
- [ ] **Learning Loop**: Learns language-agnostic patterns + language-specific via plugins
- [ ] **Dynamic Tools**: ToolBuilderAgent can generate tools for any language
- [ ] **SKILLS**: Skills tagged with language, search filtered by language
- [ ] **TUI**: Shows language name in file listings, syntax highlights any language
- [ ] **Task Interface**: Tasks have optional `language` field

---

## 14. Multi-Language Project Support

```python
# Agent working on full-stack project (Python backend + TypeScript frontend)

manifest = {
    "project_languages": ["python", "typescript"],
    "files": {
        "backend/main.py": {"language": "python", "type": "api_endpoint"},
        "frontend/App.tsx": {"language": "typescript", "type": "component"},
    },
}

# Quality gates run for each language:
python_plugin.run_tests("backend/")
typescript_plugin.build("frontend/")

# Agent switches states based on current file:
if working_on.endswith(".py"):
    enter_state("python_fastapi_development")
elif working_on.endswith(".tsx"):
    enter_state("typescript_react_development")
```

---

## 15. Plugin Development Guide

For future: **anyone can add language support**

```markdown
# Adding a New Language to Gaia V2

1. Create plugin file: `gaia/plugins/<language>_plugin.py`

2. Implement `LanguagePlugin` interface:
   - `validate_syntax()` - Use language compiler/parser
   - `build()` - Invoke build tool
   - `run_tests()` - Invoke test runner
   - `analyze_file_structure()` - Parse imports/exports
   - `get_proven_patterns()` - Provide initial patterns
   - `get_language_states()` - Provide specialized states

3. Register plugin: `LANGUAGE_REGISTRY.register(YourPlugin())`

4. Test with simple project: Create task, verify quality gates work

5. Contribute: Submit PR to Gaia SDK with plugin

**Example**: Add Go support in ~200 lines of code + patterns
```

---

## 16. Summary: Extensibility Design Principles

**1. Core is Language-Agnostic**
- No `if language == "python"` in core code
- All language logic in plugins

**2. Plugin API is Simple**
- ~10 methods to implement
- Defaults provided for optional methods
- Can start with minimal implementation, add features later

**3. Automatic Detection**
- File extension → language plugin
- No manual configuration needed

**4. Composable**
- Multiple languages in one project
- Each file uses appropriate plugin

**5. Extensible by Third Parties**
- Clear plugin interface
- Auto-discovery mechanism
- Anyone can add language support

**6. Backward Compatible**
- Existing agents work without plugins
- Plugins are enhancement, not requirement

---

## 17. Future Languages (Easy to Add)

With this plugin architecture, adding support for:

| Language | Effort (with Claude Code) | Key Tools |
|----------|--------------------------|-----------|
| **Go** | 2-3 days | go build, go test, gofmt |
| **Rust** | 2-3 days | cargo build, cargo test, rustfmt |
| **Java** | 3-4 days | javac, gradle, junit |
| **Kotlin** | 3-4 days | kotlinc, gradle, JUnit |
| **C++** | 4-5 days | gcc/clang, cmake, gtest |
| **Ruby** | 2 days | ruby -c, rspec, rubocop |
| **C#** | 3-4 days | dotnet build, dotnet test |
| **Dart/Flutter** | 3-4 days | dart analyze, flutter test |

**All languages benefit from**:
- Continuous execution
- Task queue
- Memory and learning
- State machine
- Manifest tracking
- TUI/Dashboard

Only need language-specific validation/build/test implementations.

---

*Language Extensibility Design for Gaia V2.*
*Plugin architecture enables support for any programming language without modifying core framework.*
*Swift, Go, Rust, Java, Kotlin, C++, and others can be added via ~200-line plugins.*
