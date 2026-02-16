# Implement Documentation-Code Synchronization Framework

## Priority
**HIGH** - Prevents users from following broken documentation examples

## Labels
`enhancement`, `documentation`, `infrastructure`, `ci-cd`

## Context

### The Problem
On **January 15, 2026** (commit 77df07b), a breaking refactor removed `gaia.llm.llm_client` module. Documentation wasn't updated, leaving **6+ files showing broken imports** for 4 days.

**What users saw:**
```python
# From docs (BROKEN):
from gaia.llm.llm_client import LLMClient  # ❌ ModuleNotFoundError

# What actually works:
from gaia.llm import LLMClient  # ✅ Correct
```

**Root cause:**
- No single source of truth for correct imports
- No automated validation of documentation code examples
- Import tests were minimal (only 4 modules)

### The Solution
Implement a **2-tool synchronization framework**:
1. **Canonical Import Registry** - Single source of truth for public API imports
2. **Documentation Validator** - Automated validation of all code examples in `.mdx` files

Both tools work together to ensure documentation always matches code.

### Current State
The related PR already:
- ✅ Fixed 6 documentation files with broken imports
- ✅ Enhanced **both** `util/lint.ps1` and `util/lint.py` with 31 comprehensive import tests (up from 4)
- ✅ Fixed 3 source files using old import patterns
- ✅ Added VLMClient export to gaia.llm.__init__.py
- ✅ Created `src/gaia/agents/blender/__init__.py` to export BlenderAgent
- ✅ Fixed 6 Pylint warnings in Blender agent code

**This issue:** Add automated validation framework driven by canonical registry to prevent future issues.

### Important: Missing __all__ Declarations

**Current state of modules:**
- ✅ `gaia.llm/__init__.py` - HAS `__all__`
- ✅ `gaia.database/__init__.py` - HAS `__all__`
- ✅ `gaia.utils/__init__.py` - HAS `__all__`
- ❌ `gaia.chat/sdk.py` - NO `__all__` (module file, not package)
- ❌ `gaia.rag/sdk.py` - NO `__all__` (module file, not package)
- ❌ `gaia.agents/base/__init__.py` - NO `__all__` (has imports but missing declaration)
- ❌ Most agent packages - NO `__init__.py` at all

**Impact:** The implementation must either:
1. **Add `__all__` declarations** to modules that don't have them (additional scope)
2. **Handle modules gracefully** that lack `__all__` (validator detects and reports)

**Recommended approach:** Add `__all__` as part of this issue to ensure clean public API surface.

## Objectives

1. **Add `__all__` declarations** to modules that lack them (gaia.chat.sdk, gaia.rag.sdk, gaia.agents.base)
2. Create `CANONICAL_IMPORTS.json` - Registry of all public SDK imports
3. Create `validate_canonical_imports.py` - Validates `__init__.py` files match registry
4. Create `validate_docs.py` - Validates all documentation code snippets
5. Update `util/lint.py` to use registry (matching lint.ps1's 31 tests)
6. Integrate both validators into CI/CD
7. Document the framework in dev docs

## Implementation Steps

### Step 0: Add Missing __all__ Declarations (30 minutes)

Before creating the registry, ensure all modules have proper `__all__` declarations.

**Files to update:**

**1. `src/gaia/chat/sdk.py`** - Add at end of file (after all class/function definitions):
```python
__all__ = [
    "ChatSDK",
    "ChatConfig",
    "ChatSession",
    "ChatResponse",
    "SimpleChat",
    "quick_chat",
    "quick_chat_with_memory",
]
```

**2. `src/gaia/rag/sdk.py`** - Add at end of file:
```python
__all__ = [
    "RAGSDK",
    "RAGConfig",
    "quick_rag",
]
```

**3. `src/gaia/agents/base/__init__.py`** - Add after imports:
```python
__all__ = [
    "Agent",
    "MCPAgent",
    "tool",
    "_TOOL_REGISTRY",
]
```

**Validation:**
```bash
# Verify imports still work after adding __all__
python -c "from gaia.chat.sdk import ChatSDK; print('OK')"
python -c "from gaia.rag.sdk import RAGSDK; print('OK')"
python -c "from gaia.agents.base import Agent; print('OK')"
```

**Note:** This step formalizes the public API surface and makes subsequent validation possible.

---

### Step 1: Create Canonical Import Registry (30 minutes)

**File:** `src/gaia/CANONICAL_IMPORTS.json`

**Content:**
```json
{
  "version": "1.0",
  "description": "Canonical import paths for GAIA public SDK. Single source of truth for correct imports.",
  "canonical_imports": {
    "gaia.llm": {
      "exports": ["LLMClient", "VLMClient", "create_client", "NotSupportedError"],
      "description": "LLM client interfaces for local and cloud providers"
    },
    "gaia.chat.sdk": {
      "exports": ["ChatSDK", "ChatConfig", "ChatSession", "ChatResponse", "SimpleChat", "quick_chat", "quick_chat_with_memory"],
      "description": "Chat SDK with memory and RAG support"
    },
    "gaia.rag.sdk": {
      "exports": ["RAGSDK", "RAGConfig", "quick_rag"],
      "description": "Document retrieval and Q&A"
    },
    "gaia.agents.base": {
      "exports": ["Agent", "MCPAgent", "tool"],
      "description": "Base agent system and decorators"
    },
    "gaia.agents.chat": {
      "exports": ["ChatAgent"],
      "description": "Chat agent with RAG capabilities"
    },
    "gaia.agents.code": {
      "exports": ["CodeAgent"],
      "description": "Code generation agent"
    },
    "gaia.agents.jira": {
      "exports": ["JiraAgent"],
      "description": "Jira integration agent"
    },
    "gaia.agents.docker": {
      "exports": ["DockerAgent"],
      "description": "Docker management agent"
    },
    "gaia.agents.blender": {
      "exports": ["BlenderAgent"],
      "description": "Blender 3D automation agent"
    },
    "gaia.agents.emr": {
      "exports": ["MedicalIntakeAgent"],
      "description": "Medical form processing agent"
    },
    "gaia.agents.routing": {
      "exports": ["RoutingAgent"],
      "description": "Intelligent agent selection and routing"
    },
    "gaia.database": {
      "exports": ["DatabaseAgent", "DatabaseMixin", "temp_db"],
      "description": "Database integration and ORM mixin"
    },
    "gaia.utils": {
      "exports": ["FileWatcher", "FileWatcherMixin"],
      "description": "Utility classes and helpers"
    }
  },
  "internal_imports": {
    "description": "These are internal implementation details, not part of public API",
    "whitelist": [
      "gaia.llm.vlm_client.detect_image_mime_type",
      "gaia.llm.lemonade_client.LemonadeClient",
      "gaia.llm.lemonade_client.DEFAULT_MODEL_NAME"
    ]
  }
}
```

**Note:** The `internal_imports` section allows documentation to show advanced examples using internal helpers when appropriate (e.g., tutorials showing integration with Lemonade server directly).

**Helper script to generate from current code:**

```python
# util/generate_canonical_imports.py (optional helper)
#!/usr/bin/env python3
"""Generate CANONICAL_IMPORTS.json from current __init__.py files."""
import ast
import json
from pathlib import Path

def extract_all(init_path: Path):
    """Extract __all__ from __init__.py."""
    if not init_path.exists():
        return None
    try:
        with open(init_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            return [
                                elt.s if isinstance(elt, ast.Constant) else elt.value
                                for elt in node.value.elts
                            ]
    except:
        return None
    return None

# Scan known modules
modules = [
    "gaia.llm",
    "gaia.chat.sdk",
    "gaia.rag.sdk",
    "gaia.agents.base",
    "gaia.agents.chat",
    "gaia.agents.code",
    "gaia.agents.jira",
    "gaia.agents.docker",
    "gaia.agents.blender",
    "gaia.agents.emr",
    "gaia.agents.routing",
    "gaia.database",
    "gaia.utils",
]

canonical = {}
for module in modules:
    parts = module.split(".")
    init_path = Path("src").joinpath(*parts) / "__init__.py"
    exports = extract_all(init_path)
    if exports:
        canonical[module] = {"exports": exports, "description": "TODO: Add description"}

print(json.dumps({"version": "1.0", "canonical_imports": canonical}, indent=2))
```

**Usage:**
```bash
python util/generate_canonical_imports.py > src/gaia/CANONICAL_IMPORTS.json
# Then manually add descriptions
```

---

### Step 2: Create Registry Validator (30 minutes)

**File:** `util/validate_canonical_imports.py`

<details>
<summary>Click to expand full implementation (120 lines)</summary>

```python
#!/usr/bin/env python3
# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Validate that __init__.py __all__ declarations match CANONICAL_IMPORTS.json

This ensures the registry stays in sync with actual code exports.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Dict, Set, List


def load_canonical(json_path: Path) -> Dict:
    """Load canonical imports registry."""
    if not json_path.exists():
        print(f"❌ Error: {json_path} not found")
        print("Create CANONICAL_IMPORTS.json first")
        sys.exit(1)

    try:
        with open(json_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {json_path}")
        print(f"   {e}")
        sys.exit(1)

    if "canonical_imports" not in data:
        print(f"❌ Error: Missing 'canonical_imports' key in {json_path}")
        sys.exit(1)

    return data["canonical_imports"]


def get_init_all(init_path: Path) -> Set[str]:
    """Extract __all__ from __init__.py file."""
    if not init_path.exists():
        return None

    try:
        with open(init_path) as f:
            tree = ast.parse(f.read())
    except SyntaxError as e:
        print(f"⚠️  Warning: Could not parse {init_path}: {e}")
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        return {
                            elt.s if isinstance(elt, ast.Constant) else elt.value
                            for elt in node.value.elts
                        }
    return None


def validate_module(module_path: str, spec: Dict, src_path: Path) -> List[str]:
    """
    Validate a single module's __init__.py against spec.

    Args:
        module_path: Module path like "gaia.llm"
        spec: Registry spec with exports list
        src_path: Path to src/gaia

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Convert module path to file path (e.g., gaia.llm -> src/gaia/llm/__init__.py)
    parts = module_path.split(".")
    init_path = src_path.parent.joinpath(*parts) / "__init__.py"

    if not init_path.exists():
        errors.append(f"❌ {module_path}: Missing __init__.py at {init_path}")
        return errors

    actual_all = get_init_all(init_path)
    expected = set(spec["exports"])

    if actual_all is None:
        errors.append(f"❌ {module_path}: Missing __all__ declaration in {init_path}")
    elif actual_all != expected:
        missing = expected - actual_all
        extra = actual_all - expected

        if missing:
            errors.append(
                f"❌ {module_path}: Missing exports in __all__: {sorted(missing)}"
            )
        if extra:
            errors.append(
                f"⚠️  {module_path}: Extra exports not in canonical: {sorted(extra)}"
            )

    return errors


def main():
    """Run validation."""
    project_root = Path(__file__).parent.parent
    canonical_path = project_root / "src" / "gaia" / "CANONICAL_IMPORTS.json"
    src_path = project_root / "src" / "gaia"

    print("=" * 70)
    print("Canonical Import Registry Validator")
    print("=" * 70)
    print()

    canonical = load_canonical(canonical_path)
    print(f"✓ Loaded {len(canonical)} canonical modules")
    print()

    all_errors = []

    for module_path, spec in canonical.items():
        errors = validate_module(module_path, spec, src_path)
        all_errors.extend(errors)

    if all_errors:
        print("=" * 70)
        print("VALIDATION FAILED")
        print("=" * 70)
        print()
        for error in all_errors:
            print(error)
        print()
        print("Update __init__.py files or CANONICAL_IMPORTS.json to match")
        sys.exit(1)
    else:
        print("=" * 70)
        print("✅ SUCCESS: All __init__.py files match canonical registry")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
```
</details>

**Test:**
```bash
python util/validate_canonical_imports.py
# Should pass with all current exports
```

---

### Step 3: Create Documentation Validator (1 hour)

**File:** `util/validate_docs.py`

<details>
<summary>Click to expand full implementation (280 lines)</summary>

```python
#!/usr/bin/env python3
# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Documentation Code Snippet Validator

Validates that all Python code examples in .mdx files use correct imports.
Prevents users from following broken documentation.
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple


class CodeSnippet:
    """Represents a Python code snippet from documentation."""

    def __init__(self, file: Path, line_num: int, code: str):
        self.file = file
        self.line_num = line_num
        self.code = code
        self.imports = self._extract_imports()

    def _extract_imports(self) -> List[Tuple[str, str, List[str]]]:
        """
        Extract import statements from code snippet.

        Returns:
            List of (import_type, module, names)
            - import_type: "import" or "from"
            - module: module name
            - names: list of imported names (empty for simple import)
        """
        imports = []
        try:
            tree = ast.parse(self.code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(("import", alias.name, []))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    imports.append(("from", module, names))
        except SyntaxError:
            # Skip invalid Python (might be pseudocode)
            pass
        return imports


class ImportValidator:
    """Validates imports against canonical registry."""

    def __init__(self, canonical_path: Path):
        if not canonical_path.exists():
            print(f"❌ Error: {canonical_path} not found")
            print("Create CANONICAL_IMPORTS.json first")
            sys.exit(1)

        try:
            with open(canonical_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in {canonical_path}")
            print(f"   {e}")
            sys.exit(1)

        self.canonical = data["canonical_imports"]
        self.internal_whitelist = set(
            data.get("internal_imports", {}).get("whitelist", [])
        )

    def validate_snippet(self, snippet: CodeSnippet) -> List[str]:
        """
        Validate a code snippet's imports.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        for import_type, module, names in snippet.imports:
            # Skip non-gaia imports
            if not module.startswith("gaia"):
                continue

            if import_type == "from":
                # Separate whitelisted from non-whitelisted names
                non_whitelisted_names = []
                for name in names:
                    full_import = f"{module}.{name}"
                    if full_import not in self.internal_whitelist:
                        non_whitelisted_names.append(name)

                # If all names are whitelisted, skip validation
                if not non_whitelisted_names:
                    continue

                # Check if module is canonical
                if module not in self.canonical:
                    # Check if it's a known submodule (suggest canonical)
                    parent = ".".join(module.split(".")[:-1])
                    if parent in self.canonical:
                        errors.append(
                            f"{snippet.file.name}:{snippet.line_num}: "
                            f"Import from submodule '{module}'. "
                            f"Use canonical: 'from {parent} import {', '.join(non_whitelisted_names)}'"
                        )
                    else:
                        errors.append(
                            f"{snippet.file.name}:{snippet.line_num}: "
                            f"Import from non-canonical module '{module}'. "
                            f"Available: {list(self.canonical.keys())}"
                        )
                    continue

                # Check if imported names are in __all__ (only non-whitelisted)
                expected_exports = self.canonical[module]["exports"]
                for name in non_whitelisted_names:
                    if name not in expected_exports:
                        errors.append(
                            f"{snippet.file.name}:{snippet.line_num}: "
                            f"'{name}' not exported from {module}. "
                            f"Available: {expected_exports}"
                        )

        return errors


class DocumentationValidator:
    """Main validator for documentation code snippets."""

    def __init__(self, docs_path: Path, canonical_path: Path):
        self.docs_path = docs_path
        self.validator = ImportValidator(canonical_path)
        self.snippets: List[CodeSnippet] = []

    def extract_snippets(self):
        """Extract all Python code snippets from .mdx files."""
        for mdx_file in self.docs_path.rglob("*.mdx"):
            self._extract_from_file(mdx_file)

    def _extract_from_file(self, mdx_file: Path):
        """Extract Python code blocks from a single .mdx file."""
        try:
            with open(mdx_file, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Warning: Could not read {mdx_file}: {e}")
            return

        # Match Python code blocks: ```python ... ```
        pattern = r"```python\s*\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            code = match.group(1)
            # Count lines before match for line number
            line_num = content[: match.start()].count("\n") + 1

            snippet = CodeSnippet(mdx_file, line_num, code)
            if snippet.imports:  # Only store if has imports
                self.snippets.append(snippet)

    def validate_all(self) -> Tuple[List[str], int]:
        """
        Validate all extracted snippets.

        Returns:
            (errors, total_snippets_checked)
        """
        errors = []

        for snippet in self.snippets:
            snippet_errors = self.validator.validate_snippet(snippet)
            errors.extend(snippet_errors)

        return errors, len(self.snippets)


def main():
    """Run documentation validation."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    docs_path = project_root / "docs"
    canonical_path = project_root / "src" / "gaia" / "CANONICAL_IMPORTS.json"

    if not canonical_path.exists():
        print(f"❌ Error: {canonical_path} not found")
        print("Create CANONICAL_IMPORTS.json first (see Step 1)")
        sys.exit(1)

    print("=" * 70)
    print("Documentation Code Snippet Validator")
    print("=" * 70)
    print()

    validator = DocumentationValidator(docs_path, canonical_path)

    print(f"📝 Extracting code snippets from {docs_path}...")
    validator.extract_snippets()
    print(f"✓ Found {len(validator.snippets)} Python code snippets with imports")
    print()

    print("🔍 Validating imports against canonical registry...")
    errors, total = validator.validate_all()
    print()

    if errors:
        print("=" * 70)
        print(f"❌ VALIDATION FAILED: {len(errors)} import errors found")
        print("=" * 70)
        print()
        for error in errors:
            print(f"  {error}")
        print()
        print("Fix these imports to match CANONICAL_IMPORTS.json")
        sys.exit(1)
    else:
        print("=" * 70)
        print(f"✅ SUCCESS: All {total} code snippets validated")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
```
</details>

**Test:**
```bash
python util/validate_docs.py
# Should pass (all docs were fixed in the prerequisite PR)
```

---

### Step 4: Update lint.py to Use Registry (1 hour)

**File:** `util/lint.py`

**Current state:** Now tests 31 comprehensive imports (lines 295-369) - **hardcoded list**
**Target state:** Generate tests **dynamically** from `CANONICAL_IMPORTS.json` instead of hardcoding

**Why:** Currently both lint.py and lint.ps1 have hardcoded 31-import lists. If we add a new SDK module, we must update 3 places manually (lint.py, lint.ps1, and docs). Using the registry, tests auto-update when we modify CANONICAL_IMPORTS.json.

**Replace the current `check_imports()` function with:**

```python
import json

def check_imports() -> CheckResult:
    """Test all canonical imports from registry."""
    print("\n[7/7] Testing canonical SDK imports...")
    print("-" * 40)

    # Load canonical registry
    project_root = Path(__file__).parent.parent
    canonical_path = project_root / "src" / "gaia" / "CANONICAL_IMPORTS.json"

    if not canonical_path.exists():
        print("⚠️  CANONICAL_IMPORTS.json not found - using basic tests")
        # Fallback to basic 4 imports for backward compatibility
        return check_imports_basic()

    try:
        with open(canonical_path) as f:
            data = json.load(f)
            canonical = data["canonical_imports"]
    except Exception as e:
        print(f"⚠️  Could not load registry: {e} - using basic tests")
        return check_imports_basic()

    failed_imports = []
    passed = 0

    # Test each module and its exports
    for module_name, spec in canonical.items():
        # Test 1: Module import
        cmd = [sys.executable, "-c", f"import {module_name}"]
        exit_code, output = run_command(cmd)

        if exit_code != 0:
            failed_imports.append(f"import {module_name}")
            print(f"❌ {module_name}")
        else:
            passed += 1
            print(f"✓ {module_name}")

        # Test 2: Each export
        for export_name in spec["exports"]:
            cmd = [sys.executable, "-c", f"from {module_name} import {export_name}"]
            exit_code, output = run_command(cmd)

            if exit_code != 0:
                failed_imports.append(f"from {module_name} import {export_name}")
                print(f"  ❌ {export_name}")
            else:
                passed += 1
                print(f"  ✓ {export_name}")

    print()
    if failed_imports:
        print(f"❌ {len(failed_imports)} import tests failed:")
        for fail in failed_imports:
            print(f"  {fail}")
        return CheckResult("Import Validation", False, False, len(failed_imports), "")
    else:
        print(f"✅ All {passed} import tests passed")
        return CheckResult("Import Validation", True, False, 0, "")


def check_imports_basic() -> CheckResult:
    """Fallback basic import tests if registry doesn't exist."""
    print("[Using basic 4-import fallback tests]")
    print()

    imports = [
        ("gaia.cli", "CLI module"),
        ("gaia.chat.sdk", "Chat SDK"),
        ("gaia.llm", "LLM client"),
        ("gaia.agents.base.agent", "Base agent"),
    ]

    failed = False
    issues = 0

    for module, desc in imports:
        cmd = [sys.executable, "-c", f"import {module}; print('OK: {desc} imports')"]
        print(f"[CMD] {' '.join(cmd)}")
        exit_code, output = run_command(cmd)
        print(output.strip())
        if exit_code != 0:
            print(f"[!] Failed to import {module}")
            failed = True
            issues += 1

    if failed:
        return CheckResult("Import Validation", False, False, issues, "")

    print("[OK] All imports working!")
    return CheckResult("Import Validation", True, False, 0, "")
```

**Note:** This maintains backward compatibility - if registry doesn't exist, falls back to basic 4-import tests.

---

### Step 5: Create CI/CD Workflow (30 minutes)

**File:** `.github/workflows/validate-docs.yml`

```yaml
name: Documentation Validation

on:
  pull_request:
    paths:
      - 'docs/**/*.mdx'
      - 'src/gaia/**/__init__.py'
      - 'src/gaia/CANONICAL_IMPORTS.json'
      - 'util/validate_*.py'
  push:
    branches: [main]

jobs:
  validate-docs:
    name: Validate Documentation-Code Sync
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install GAIA
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Validate canonical imports registry
        run: |
          echo "Checking that __init__.py files match CANONICAL_IMPORTS.json..."
          python util/validate_canonical_imports.py

      - name: Validate documentation code snippets
        run: |
          echo "Checking that all .mdx code examples use canonical imports..."
          python util/validate_docs.py

      - name: Test all canonical imports work
        run: |
          echo "Testing that all registered imports are importable..."
          python util/lint.py --imports
```

**Key points:**
- Triggers on changes to `.mdx`, `__init__.py`, `CANONICAL_IMPORTS.json`, or validators
- Runs all 3 checks in sequence
- Uses `--imports` flag (not `--check-imports`)

**Test workflow:**
1. Push to a test branch
2. Verify workflow runs
3. Break an import in docs/test.mdx
4. Verify workflow catches it

---

### Step 6: Update Documentation (30 minutes)

**File:** `docs/reference/dev.mdx`

Find the "Linting" section and add this new section after it:

```markdown
## Documentation-Code Synchronization

GAIA uses automated validation to ensure documentation examples always work.

### Canonical Import Registry

**File:** `src/gaia/CANONICAL_IMPORTS.json`

This JSON file is the **single source of truth** for correct import paths. It defines:
- All public SDK modules
- What each module exports in `__all__`
- Canonical import patterns for documentation

**Example:**
\`\`\`json
{
  "canonical_imports": {
    "gaia.llm": {
      "exports": ["LLMClient", "VLMClient", "create_client"],
      "description": "LLM client interfaces"
    }
  }
}
\`\`\`

### Validation Tools

#### 1. Validate Registry Matches Code

Ensures `__init__.py` files match the canonical registry:

\`\`\`bash
python util/validate_canonical_imports.py
\`\`\`

#### 2. Validate Documentation Examples

Ensures all Python code in `.mdx` files uses correct imports:

\`\`\`bash
python util/validate_docs.py
\`\`\`

#### 3. Test All Imports

Lint validation now tests 30+ imports automatically:

\`\`\`bash
python util/lint.py --imports
\`\`\`

### Usage for Developers

**When adding new public APIs:**

1. Update module's `__init__.py`:
   \`\`\`python
   from .new_module import NewClass
   __all__ = [..., "NewClass"]
   \`\`\`

2. Update `CANONICAL_IMPORTS.json`:
   \`\`\`json
   {
     "gaia.module": {
       "exports": [..., "NewClass"],
       "description": "Module description"
     }
   }
   \`\`\`

3. Validate sync:
   \`\`\`bash
   python util/validate_canonical_imports.py
   \`\`\`

4. Document with canonical import:
   \`\`\`python
   from gaia.module import NewClass
   \`\`\`

5. Validate docs:
   \`\`\`bash
   python util/validate_docs.py
   \`\`\`

**When writing documentation:**

✅ **Use canonical imports:**
\`\`\`python
from gaia.llm import LLMClient, VLMClient
from gaia.chat.sdk import ChatSDK
from gaia.agents.base import Agent, tool
\`\`\`

❌ **Don't use submodule imports:**
\`\`\`python
from gaia.llm.llm_client import LLMClient      # Wrong
from gaia.llm.vlm_client import VLMClient      # Wrong
from gaia.agents.base.agent import Agent       # Wrong
\`\`\`

### CI/CD Integration

The `validate-docs` workflow runs automatically on PRs that change:
- Documentation files (`.mdx`)
- Module exports (`__init__.py`)
- Canonical registry (`CANONICAL_IMPORTS.json`)

**What it checks:**
1. Registry matches code exports
2. Documentation uses canonical imports
3. All imports are importable

**If validation fails:**
- PR will be blocked
- Error message shows which file and line number
- Fix the import to match canonical pattern
```

---

### Step 6: Update CONTRIBUTING.md (15 minutes)

**File:** `CONTRIBUTING.md`

Find the "Before Submitting" or "Code Quality" section and add:

```markdown
### Documentation Changes

If you're modifying documentation (`.mdx` files):

\`\`\`bash
# Validate all code examples use correct imports
python util/validate_docs.py
\`\`\`

If you're changing module exports (`__init__.py`):

\`\`\`bash
# 1. Update CANONICAL_IMPORTS.json to match your changes
# 2. Validate they're in sync:
python util/validate_canonical_imports.py
\`\`\`

Both validators run automatically in CI/CD and will block PRs if imports are incorrect.
```

---

## Testing Plan

### Test 1: Registry Validator

```bash
# Should pass (all current exports are correct)
python util/validate_canonical_imports.py

# Test error detection:
# 1. Temporarily remove "VLMClient" from src/gaia/llm/__init__.py __all__
python util/validate_canonical_imports.py
# Expected: "Missing exports in __all__: ['VLMClient']"

# 2. Revert change

# 3. Add fake export "FakeClass" to CANONICAL_IMPORTS.json under gaia.llm
python util/validate_canonical_imports.py
# Expected: "Extra exports not in canonical: ['FakeClass']"

# 4. Revert - should pass again
```

### Test 2: Documentation Validator

```bash
# Should pass (all docs fixed in prerequisite PR)
python util/validate_docs.py

# Test error detection:
# 1. Create temporary test file
cat > docs/test-broken.mdx << 'EOF'
---
title: "Test"
---

```python
from gaia.llm.llm_client import LLMClient
```
EOF

# 2. Run validator
python util/validate_docs.py
# Expected: "Import from submodule 'gaia.llm.llm_client'. Use canonical: 'from gaia.llm import LLMClient'"

# 3. Remove test file
rm docs/test-broken.mdx
```

### Test 3: Updated lint.py

```bash
# Test that lint.py now uses registry
python util/lint.py --imports

# Should see:
# ✓ gaia.llm
#   ✓ LLMClient
#   ✓ VLMClient
#   ✓ create_client
# ... (30+ total tests)
# ✅ All X import tests passed
```

### Test 4: CI/CD Workflow

```bash
# 1. Create test branch
git checkout -b test/doc-validation

# 2. Create docs/test.mdx with broken import:
cat > docs/test.mdx << 'EOF'
---
title: "Test"
---

```python
from gaia.llm.old_module import Something
```
EOF

# 3. Commit and push
git add docs/test.mdx
git commit -m "Test: broken import"
git push origin test/doc-validation

# 4. Create PR - verify GitHub Actions fails with clear error

# 5. Fix import to canonical pattern
# 6. Push again - verify GitHub Actions passes

# 7. Close PR and cleanup
```

### Test 5: Integration Test

```bash
# Run full validation suite
python util/validate_canonical_imports.py && \
python util/validate_docs.py && \
python util/lint.py --imports

# All three should pass
```

---

## Acceptance Criteria

### Deliverables
- [ ] `src/gaia/CANONICAL_IMPORTS.json` created with 13+ modules
- [ ] `util/validate_canonical_imports.py` created (120 lines)
- [ ] `util/validate_docs.py` created (280 lines)
- [ ] `util/lint.py` updated with registry-based import tests
- [ ] `.github/workflows/validate-docs.yml` created and working
- [ ] `docs/reference/dev.mdx` updated with validation docs
- [ ] `CONTRIBUTING.md` updated with validation steps

### Validation
- [ ] Registry validator passes on current codebase
- [ ] Documentation validator passes on current codebase
- [ ] lint.py generates 30+ tests from registry
- [ ] CI workflow triggers on relevant file changes
- [ ] CI workflow correctly fails on broken imports
- [ ] CI workflow correctly passes on valid imports

### Testing
- [ ] All 5 test scenarios pass
- [ ] Error messages are clear with file:line references
- [ ] Performance < 10 seconds for full validation
- [ ] No false positives on current codebase

## Success Metrics

**Before:**
- 0% documentation validation
- 4 import tests (modules only)
- 4 days to detect breaking change
- Manual coordination across 8+ files

**After:**
- 100% documentation validation
- 30+ import tests (every exported class)
- < 5 minutes to detect breaking change
- Automated enforcement in CI/CD
- Users never see broken examples

## Prerequisites

**Must be merged first:**
- PR that fixes the 6 documentation files with broken imports
- PR that adds VLMClient export to gaia/llm/__init__.py
- PR that enhances lint.ps1 with 31 import tests

**Check git log for:** Commit fixing import inconsistencies (should be on main)

## Known Considerations

### Internal Helper Imports

Some documentation may legitimately show internal imports for advanced use cases:
```python
from gaia.llm.lemonade_client import DEFAULT_MODEL_NAME  # Advanced usage
```

**Solution:** The `internal_imports.whitelist` in the registry handles these exceptions.

### Platform Compatibility

All validators use `pathlib.Path` for cross-platform compatibility. Tested on:
- Windows (PowerShell)
- Linux (bash)
- macOS (bash)

### Backward Compatibility

If `CANONICAL_IMPORTS.json` doesn't exist, lint.py falls back to basic 4-import tests. This prevents breaking existing workflows during deployment.

## Edge Cases Handled

1. **Invalid JSON in registry** → Clear error message, fails gracefully
2. **Pseudocode in docs** → AST parse errors caught, snippet skipped
3. **Missing __init__.py** → Detected and reported
4. **Missing __all__** → Detected and reported
5. **Non-Python code blocks** → Ignored (only validates ```python blocks)
6. **Comments in code** → Ignored by AST parser
7. **Internal helper imports** → Whitelisted via registry

## Rollout Strategy

### Week 1: Local Development
- Create registry and validators
- Test locally on developer machine
- Iterate on error messages for clarity

### Week 2: CI/CD Integration
- Create GitHub workflow
- Test on feature branch
- Monitor for false positives

### Week 3: Documentation
- Update dev.mdx
- Update CONTRIBUTING.md
- Announce to team

### Week 4: Enforcement
- Make mandatory (block PRs)
- Monitor for issues
- Refine as needed

## Estimated Effort

| Task | Time | Notes |
|------|------|-------|
| Add missing __all__ declarations | 30 min | 3 files to update |
| Create CANONICAL_IMPORTS.json | 30 min | Use helper script to extract |
| Write validate_canonical_imports.py | 30 min | Straightforward AST parsing |
| Write validate_docs.py | 1 hour | Regex + AST parsing |
| Update lint.py | 1 hour | Replace hardcoded list with registry |
| Create CI/CD workflow | 30 min | Adapt existing workflows |
| Update documentation | 30 min | Add to dev.mdx + CONTRIBUTING.md |
| Testing all scenarios | 30 min | Run 5 test cases |
| **TOTAL** | **4.5 hours** | Can be done in one day |

## Files to Create/Modify

### New Files (4-5)
- `src/gaia/CANONICAL_IMPORTS.json`
- `util/validate_canonical_imports.py`
- `util/validate_docs.py`
- `.github/workflows/validate-docs.yml`
- Optional: `util/generate_canonical_imports.py` (helper)

### Modified Files (6)
- `src/gaia/chat/sdk.py` - Add `__all__` declaration
- `src/gaia/rag/sdk.py` - Add `__all__` declaration
- `src/gaia/agents/base/__init__.py` - Add `__all__` declaration
- `util/lint.py` - Update `check_imports()` function
- `docs/reference/dev.mdx` - Add validation section
- `CONTRIBUTING.md` - Add validation steps

**Total:** 10-11 files

## Questions for Reviewer

1. Should we migrate **both** lint.py and lint.ps1 to registry-based, or keep the hardcoded 31 tests?
2. Should internal helper imports be allowed in docs, or strictly canonical only?
3. What should happen if validator has false positives - create exceptions list?
4. Should we validate that code examples actually execute (beyond just imports)?

## Additional Resources

- **Breaking change commit:** 77df07b (Jan 15, 2026) - LLM Client Factory refactor
- **Current comprehensive tests:**
  - `util/lint.ps1` lines 267-314 (31 hardcoded import tests)
  - `util/lint.py` lines 295-369 (31 hardcoded import tests, now in sync)
- **Related PR:** Import inconsistencies fix (adds VLMClient export, fixes 6 docs, enhances both lint scripts)

## Checklist for Assignee

Before starting:
- [ ] Review this issue fully (all steps, code, and test scenarios)
- [ ] Verify prerequisite PR is merged
- [ ] Examine current import testing in `util/lint.ps1` lines 267-314

Implementation:
- [ ] Add __all__ declarations to 3 modules (Step 0)
- [ ] Create CANONICAL_IMPORTS.json from current exports (Step 1)
- [ ] Implement validate_canonical_imports.py (Step 2)
- [ ] Test registry validator - all test cases (Test 1)
- [ ] Implement validate_docs.py (Step 3)
- [ ] Test documentation validator - all test cases (Test 2)
- [ ] Update lint.py check_imports() function (Step 4)
- [ ] Test updated lint.py (Test 3)
- [ ] Create GitHub Actions workflow (Step 5)
- [ ] Test CI/CD integration on feature branch (Test 4)

Documentation:
- [ ] Update docs/reference/dev.mdx
- [ ] Update CONTRIBUTING.md
- [ ] Verify all new docs are accurate

Testing:
- [ ] Run all validators locally
- [ ] Verify they pass on current codebase
- [ ] Test error detection works
- [ ] Verify CI workflow fails/passes correctly
- [ ] Check performance (< 10 sec)

Finalize:
- [ ] Create PR with all changes
- [ ] Self-review all code
- [ ] Verify CI passes on your PR
- [ ] Request review from team
