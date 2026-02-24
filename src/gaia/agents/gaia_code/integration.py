# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Integration utilities for GAIA Code.

This module handles:
- Tool registration from existing CodeAgent tools
- Specialist registration in agents.db
- LLM client integration
- Initial workspace setup
"""

import logging
from pathlib import Path
from typing import List, Optional

from .shared_state import get_shared_state
from .specialists import (
    ArchitectureAgent,
    DebuggerAgent,
    DocumentationAgent,
    PerformanceAgent,
    RefactoringAgent,
    SecurityAgent,
    TestingAgent,
)

logger = logging.getLogger(__name__)


def register_core_tools(state) -> int:
    """
    Register core tools in tools.db.

    Returns:
        Number of tools registered
    """
    core_tools = [
        # File I/O tools
        {
            "name": "read_file",
            "category": "file_io",
            "description": "Read contents of a file",
            "source": "core",
        },
        {
            "name": "write_file",
            "category": "file_io",
            "description": "Write contents to a file",
            "source": "core",
        },
        {
            "name": "edit_file",
            "category": "file_io",
            "description": "Edit a file by replacing old text with new text",
            "source": "core",
        },
        {
            "name": "glob_search",
            "category": "file_io",
            "description": "Search for files matching a pattern",
            "source": "core",
        },
        {
            "name": "grep_content",
            "category": "file_io",
            "description": "Search for content within files",
            "source": "core",
        },
        # Code execution tools
        {
            "name": "run_python",
            "category": "execution",
            "description": "Run Python code",
            "source": "core",
        },
        {
            "name": "run_shell_command",
            "category": "execution",
            "description": "Run a shell command",
            "source": "core",
        },
        # Testing tools
        {
            "name": "run_pytest",
            "category": "testing",
            "description": "Run pytest tests",
            "source": "core",
        },
        {
            "name": "run_jest",
            "category": "testing",
            "description": "Run Jest tests",
            "source": "core",
        },
        {
            "name": "check_coverage",
            "category": "testing",
            "description": "Check test coverage",
            "source": "core",
        },
        # Git tools
        {
            "name": "git_status",
            "category": "git",
            "description": "Show git status",
            "source": "core",
        },
        {
            "name": "git_diff",
            "category": "git",
            "description": "Show git diff",
            "source": "core",
        },
        {
            "name": "git_commit",
            "category": "git",
            "description": "Create git commit",
            "source": "core",
        },
        {
            "name": "git_branch",
            "category": "git",
            "description": "Git branch operations",
            "source": "core",
        },
        {
            "name": "git_log",
            "category": "git",
            "description": "Show git log",
            "source": "core",
        },
        # Quality tools
        {
            "name": "check_syntax",
            "category": "quality",
            "description": "Check Python syntax",
            "source": "core",
        },
        {
            "name": "run_linter",
            "category": "quality",
            "description": "Run code linter",
            "source": "core",
        },
        {
            "name": "check_imports",
            "category": "quality",
            "description": "Check if all imports resolve",
            "source": "core",
        },
        {
            "name": "format_code",
            "category": "quality",
            "description": "Format code with Black",
            "source": "core",
        },
        # GitHub tools
        {
            "name": "gh_clone",
            "category": "github",
            "description": "Clone GitHub repository",
            "source": "core",
        },
        {
            "name": "gh_pr_create",
            "category": "github",
            "description": "Create GitHub pull request",
            "source": "core",
        },
        {
            "name": "gh_issue_list",
            "category": "github",
            "description": "List GitHub issues",
            "source": "core",
        },
    ]

    count = 0
    for tool in core_tools:
        try:
            state.tools.register_tool(
                name=tool["name"],
                category=tool["category"],
                description=tool["description"],
                source=tool["source"],
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to register tool {tool['name']}: {e}")

    logger.info(f"Registered {count} core tools")
    return count


def register_specialists(state) -> int:
    """
    Register the 7 core specialist agents in agents.db.

    Returns:
        Number of specialists registered
    """
    specialists = [
        DebuggerAgent(),
        SecurityAgent(),
        RefactoringAgent(),
        TestingAgent(),
        DocumentationAgent(),
        PerformanceAgent(),
        ArchitectureAgent(),
    ]

    count = 0
    for specialist in specialists:
        try:
            metadata = specialist.get_metadata()

            # Skip if already registered (idempotent on repeated startups)
            existing = state.agents.conn.execute(
                "SELECT id FROM agents WHERE name = ?", (metadata["name"],)
            ).fetchone()
            if existing:
                continue

            # Use register_agent() so capabilities/tool_packs are stored as JSON
            state.agents.register_agent(
                name=metadata["name"],
                description=metadata["description"],
                capabilities=metadata["capabilities"],
                system_prompt=specialist.get_system_prompt(),
                tool_packs=metadata["tool_packs"],
            )
            count += 1

            logger.info(f"Registered specialist: {metadata['name']}")

        except Exception as e:
            logger.warning(f"Failed to register specialist {specialist.__class__.__name__}: {e}")

    logger.info(f"Registered {count} specialists")
    return count


def register_initial_skills(state) -> int:
    """
    Register initial coding workflow patterns in skills.db.

    These are common multi-step patterns the agent uses repeatedly.
    Registered once on startup; actual usage is tracked via record_usage().

    Returns:
        Number of skills registered
    """
    initial_skills = [
        {
            "name": "debug_error_fix",
            "description": "Read failing file, analyse error, apply targeted fix, verify",
            "category": "debugging",
            "domain": "python",
            "steps": [
                {"step": 1, "action": "read_file", "description": "Read the file with the error"},
                {"step": 2, "action": "analyze", "description": "Identify root cause of the error"},
                {"step": 3, "action": "edit_file", "description": "Apply minimal targeted fix"},
                {"step": 4, "action": "run_syntax_check", "description": "Verify fix is syntactically valid"},
            ],
            "tools_used": ["read_file", "edit_file", "check_syntax", "run_pytest"],
        },
        {
            "name": "create_feature",
            "description": "Plan feature, create implementation file, write tests, run tests",
            "category": "coding",
            "domain": "python",
            "steps": [
                {"step": 1, "action": "plan", "description": "Outline the feature structure"},
                {"step": 2, "action": "write_file", "description": "Implement the feature"},
                {"step": 3, "action": "write_file", "description": "Write tests for the feature"},
                {"step": 4, "action": "run_pytest", "description": "Run tests and verify"},
            ],
            "tools_used": ["write_file", "run_pytest", "check_syntax"],
        },
        {
            "name": "refactor_code",
            "description": "Read code, identify improvement areas, refactor, run tests",
            "category": "refactoring",
            "domain": "python",
            "steps": [
                {"step": 1, "action": "read_file", "description": "Read the code to refactor"},
                {"step": 2, "action": "analyze", "description": "Identify code smells and improvement areas"},
                {"step": 3, "action": "edit_file", "description": "Apply refactoring changes"},
                {"step": 4, "action": "run_pytest", "description": "Run existing tests to verify no regressions"},
            ],
            "tools_used": ["read_file", "edit_file", "run_pytest", "format_code"],
        },
        {
            "name": "git_commit_workflow",
            "description": "Stage changed files, write descriptive commit message, commit",
            "category": "git",
            "domain": "version_control",
            "steps": [
                {"step": 1, "action": "git_status", "description": "Check what files changed"},
                {"step": 2, "action": "git_diff", "description": "Review the changes"},
                {"step": 3, "action": "git_commit", "description": "Create commit with descriptive message"},
            ],
            "tools_used": ["git_status", "git_diff", "git_commit"],
        },
        {
            "name": "add_tests",
            "description": "Read existing code, write comprehensive unit tests, run and verify",
            "category": "testing",
            "domain": "python",
            "steps": [
                {"step": 1, "action": "read_file", "description": "Read the file to test"},
                {"step": 2, "action": "analyze", "description": "Identify testable units and edge cases"},
                {"step": 3, "action": "write_file", "description": "Write unit tests with good coverage"},
                {"step": 4, "action": "run_pytest", "description": "Run tests to verify they pass"},
                {"step": 5, "action": "check_coverage", "description": "Check coverage is adequate"},
            ],
            "tools_used": ["read_file", "write_file", "run_pytest", "check_coverage"],
        },
        {
            "name": "write_documentation",
            "description": "Read code, write docstrings and README, verify accuracy",
            "category": "documentation",
            "domain": "python",
            "steps": [
                {"step": 1, "action": "read_file", "description": "Read the code to document"},
                {"step": 2, "action": "edit_file", "description": "Add/update docstrings"},
                {"step": 3, "action": "write_file", "description": "Update or create README"},
            ],
            "tools_used": ["read_file", "edit_file", "write_file"],
        },
        {
            "name": "security_audit",
            "description": "Scan code for vulnerabilities, report findings, apply fixes",
            "category": "security",
            "domain": "python",
            "steps": [
                {"step": 1, "action": "glob_search", "description": "Find all source files"},
                {"step": 2, "action": "analyze", "description": "Scan for OWASP top 10 vulnerabilities"},
                {"step": 3, "action": "edit_file", "description": "Apply security fixes"},
                {"step": 4, "action": "run_pytest", "description": "Verify fixes don't break functionality"},
            ],
            "tools_used": ["glob_search", "grep_content", "edit_file", "run_pytest"],
        },
        {
            "name": "codebase_exploration",
            "description": "Index codebase, search for relevant symbols, understand architecture",
            "category": "analysis",
            "domain": "any",
            "steps": [
                {"step": 1, "action": "index_codebase", "description": "Build semantic index of repository"},
                {"step": 2, "action": "analyze_architecture", "description": "Get architecture overview"},
                {"step": 3, "action": "search_codebase", "description": "Search for relevant components"},
            ],
            "tools_used": ["index_codebase", "analyze_architecture", "search_codebase", "find_symbol"],
        },
    ]

    count = 0
    for skill in initial_skills:
        try:
            # Skip if already registered (idempotent on repeated startups)
            existing = state.skills.conn.execute(
                "SELECT id FROM skills WHERE name = ?", (skill["name"],)
            ).fetchone()
            if existing:
                continue

            state.skills.register_skill(
                name=skill["name"],
                description=skill["description"],
                category=skill["category"],
                steps=skill["steps"],
                domain=skill.get("domain"),
                tools_used=skill.get("tools_used"),
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to register skill {skill['name']}: {e}")

    logger.info(f"Registered {count} initial skills")
    return count


def seed_common_tool_recipes(state) -> int:
    """
    Pre-seed tools.db with shell command recipes for common development tools.

    These are NOT callable Python functions — they are discovery hints so the
    agent can call find_tool("cmake build") and get back the exact shell command
    to pass to run_shell_command(). Registered with source="recipe".

    Returns:
        Number of recipes seeded (0 if already seeded)
    """
    # Check if already seeded (idempotent)
    existing = state.tools.conn.execute(
        "SELECT COUNT(*) FROM tools WHERE source = 'recipe'"
    ).fetchone()[0]
    if existing > 0:
        logger.debug(f"[ToolsDB] recipes already seeded ({existing} entries), skipping")
        return 0

    recipes = [
        # ── Python ───────────────────────────────────────────────────────────
        {
            "name": "pytest_run",
            "category": "python_testing",
            "description": "Run pytest tests. Command: pytest {path} (e.g. pytest tests/ or pytest tests/test_foo.py). Use run_shell_command.",
        },
        {
            "name": "pytest_coverage",
            "category": "python_testing",
            "description": "Run pytest with coverage report. Command: pytest --cov={src_dir} --cov-report=term-missing {test_dir}. Use run_shell_command.",
        },
        {
            "name": "pytest_verbose",
            "category": "python_testing",
            "description": "Run pytest with verbose output and stop on first failure. Command: pytest -xvs {path}. Use run_shell_command.",
        },
        {
            "name": "pytest_single_test",
            "category": "python_testing",
            "description": "Run a single pytest test function by name. Command: pytest -xvs tests/test_foo.py::test_function_name. Use run_shell_command.",
        },
        {
            "name": "black_format",
            "category": "python_quality",
            "description": "Format Python code with black. Command: black {path} (e.g. black src/ or black file.py). Use run_shell_command.",
        },
        {
            "name": "ruff_check",
            "category": "python_quality",
            "description": "Lint Python code with ruff (fast linter). Command: ruff check {path}. Use run_shell_command.",
        },
        {
            "name": "ruff_fix",
            "category": "python_quality",
            "description": "Auto-fix Python lint issues with ruff. Command: ruff check --fix {path}. Use run_shell_command.",
        },
        {
            "name": "mypy_typecheck",
            "category": "python_quality",
            "description": "Run mypy static type checker on Python code. Command: mypy {path} --ignore-missing-imports. Use run_shell_command.",
        },
        {
            "name": "isort_imports",
            "category": "python_quality",
            "description": "Sort Python imports with isort. Command: isort {path}. Use run_shell_command.",
        },
        {
            "name": "pip_install",
            "category": "python_packages",
            "description": "Install Python packages with pip. Command: pip install {package} or pip install -r requirements.txt. Use run_shell_command.",
        },
        {
            "name": "uv_install",
            "category": "python_packages",
            "description": "Install Python packages with uv (fast pip replacement). Command: uv pip install {package} or uv pip install -e '.[dev]'. Use run_shell_command.",
        },
        {
            "name": "uv_sync",
            "category": "python_packages",
            "description": "Sync Python environment with uv. Command: uv sync or uv sync --all-extras. Use run_shell_command.",
        },
        {
            "name": "bandit_security_scan",
            "category": "python_security",
            "description": "Scan Python code for security issues with bandit. Command: bandit -r {src_dir} -ll. Use run_shell_command.",
        },
        {
            "name": "python_syntax_check",
            "category": "python_quality",
            "description": "Check Python file syntax without running it. Command: python -m py_compile {file.py} && echo OK. Use run_shell_command.",
        },
        {
            "name": "python_profile",
            "category": "python_performance",
            "description": "Profile Python script execution. Command: python -m cProfile -s cumulative {script.py}. Use run_shell_command.",
        },
        # ── C++ ──────────────────────────────────────────────────────────────
        {
            "name": "cmake_configure",
            "category": "cpp_build",
            "description": "Configure a CMake C++ project. Command: cmake -B build -S . (or cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug). Use run_shell_command.",
        },
        {
            "name": "cmake_configure_release",
            "category": "cpp_build",
            "description": "Configure CMake project for release (optimized). Command: cmake -B build -S . -DCMAKE_BUILD_TYPE=Release. Use run_shell_command.",
        },
        {
            "name": "cmake_configure_debug",
            "category": "cpp_build",
            "description": "Configure CMake project for debug build with symbols. Command: cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON. Use run_shell_command.",
        },
        {
            "name": "cmake_build",
            "category": "cpp_build",
            "description": "Build a CMake C++ project. Command: cmake --build build or cmake --build build --parallel $(nproc). Use run_shell_command.",
        },
        {
            "name": "cmake_build_target",
            "category": "cpp_build",
            "description": "Build a specific CMake target. Command: cmake --build build --target {target_name}. Use run_shell_command.",
        },
        {
            "name": "cmake_install",
            "category": "cpp_build",
            "description": "Install CMake build artifacts. Command: cmake --install build or cmake --install build --prefix /usr/local. Use run_shell_command.",
        },
        {
            "name": "cmake_clean",
            "category": "cpp_build",
            "description": "Clean CMake build directory. Command: rm -rf build && mkdir build. Use run_shell_command.",
        },
        {
            "name": "make_build",
            "category": "cpp_build",
            "description": "Build with make. Command: make -j$(nproc) or make -j4. Use run_shell_command with cwd set to build directory.",
        },
        {
            "name": "make_clean",
            "category": "cpp_build",
            "description": "Clean make build artifacts. Command: make clean. Use run_shell_command with cwd set to build directory.",
        },
        {
            "name": "ctest_run",
            "category": "cpp_testing",
            "description": "Run C++ tests with CTest. Command: ctest --test-dir build or cd build && ctest. Use run_shell_command.",
        },
        {
            "name": "ctest_verbose",
            "category": "cpp_testing",
            "description": "Run CTest with verbose output and show test output on failure. Command: ctest --test-dir build --output-on-failure -V. Use run_shell_command.",
        },
        {
            "name": "gtest_run",
            "category": "cpp_testing",
            "description": "Run GoogleTest binary directly. Command: ./build/{test_binary} --gtest_output=xml:test_results.xml. Use run_shell_command.",
        },
        {
            "name": "gpp_compile",
            "category": "cpp_compile",
            "description": "Compile C++ file with g++. Command: g++ -std=c++17 -Wall -Wextra -o {output} {source.cpp}. Use run_shell_command.",
        },
        {
            "name": "clang_compile",
            "category": "cpp_compile",
            "description": "Compile C++ file with clang++. Command: clang++ -std=c++17 -Wall -Wextra -o {output} {source.cpp}. Use run_shell_command.",
        },
        {
            "name": "clang_format",
            "category": "cpp_quality",
            "description": "Format C++ code with clang-format. Command: clang-format -i {file.cpp} or find src -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i. Use run_shell_command.",
        },
        {
            "name": "clang_tidy",
            "category": "cpp_quality",
            "description": "Run clang-tidy static analyzer on C++ code. Command: clang-tidy {file.cpp} -- -std=c++17 -I include. Use run_shell_command.",
        },
        {
            "name": "cppcheck_analyze",
            "category": "cpp_quality",
            "description": "Run cppcheck static analyzer. Command: cppcheck --enable=all --suppress=missingIncludeSystem src/. Use run_shell_command.",
        },
        {
            "name": "valgrind_memcheck",
            "category": "cpp_debug",
            "description": "Check C++ program for memory leaks with valgrind. Command: valgrind --leak-check=full --show-leak-kinds=all ./{binary}. Use run_shell_command.",
        },
        {
            "name": "gdb_debug",
            "category": "cpp_debug",
            "description": "Start GDB debugger on a C++ binary. Command: gdb ./{binary} (then run, bt, break main, etc.). Use run_shell_command.",
        },
        {
            "name": "address_sanitizer",
            "category": "cpp_debug",
            "description": "Build C++ with AddressSanitizer to detect memory errors. Command: cmake -B build -DCMAKE_CXX_FLAGS='-fsanitize=address -g' && cmake --build build. Use run_shell_command.",
        },
        {
            "name": "conan_install",
            "category": "cpp_packages",
            "description": "Install C++ dependencies with Conan package manager. Command: conan install . --output-folder=build --build=missing. Use run_shell_command.",
        },
        {
            "name": "vcpkg_install",
            "category": "cpp_packages",
            "description": "Install C++ package with vcpkg. Command: vcpkg install {package} or vcpkg install --triplet x64-linux. Use run_shell_command.",
        },
        {
            "name": "doxygen_generate",
            "category": "cpp_docs",
            "description": "Generate C++ documentation with Doxygen. Command: doxygen Doxyfile (create with doxygen -g if missing). Use run_shell_command.",
        },
        # ── Frontend / JavaScript / TypeScript ───────────────────────────────
        {
            "name": "npm_install",
            "category": "frontend_packages",
            "description": "Install npm packages. Command: npm install or npm install {package}. Use run_shell_command with cwd set to project root.",
        },
        {
            "name": "npm_ci",
            "category": "frontend_packages",
            "description": "Clean install npm packages from lockfile (CI-safe). Command: npm ci. Use run_shell_command.",
        },
        {
            "name": "npm_build",
            "category": "frontend_build",
            "description": "Build frontend project with npm. Command: npm run build. Use run_shell_command with cwd set to project root.",
        },
        {
            "name": "npm_dev",
            "category": "frontend_build",
            "description": "Start npm development server. Command: npm run dev or npm start. Use run_shell_command.",
        },
        {
            "name": "npm_test",
            "category": "frontend_testing",
            "description": "Run npm tests. Command: npm test or npm run test. Use run_shell_command.",
        },
        {
            "name": "yarn_install",
            "category": "frontend_packages",
            "description": "Install packages with yarn. Command: yarn install or yarn add {package}. Use run_shell_command.",
        },
        {
            "name": "yarn_build",
            "category": "frontend_build",
            "description": "Build with yarn. Command: yarn build. Use run_shell_command.",
        },
        {
            "name": "pnpm_install",
            "category": "frontend_packages",
            "description": "Install packages with pnpm. Command: pnpm install or pnpm add {package}. Use run_shell_command.",
        },
        {
            "name": "tsc_typecheck",
            "category": "frontend_quality",
            "description": "Run TypeScript compiler type check without emitting files. Command: tsc --noEmit or npx tsc --noEmit. Use run_shell_command.",
        },
        {
            "name": "eslint_check",
            "category": "frontend_quality",
            "description": "Lint JavaScript/TypeScript with ESLint. Command: npx eslint {path} or eslint src/. Use run_shell_command.",
        },
        {
            "name": "eslint_fix",
            "category": "frontend_quality",
            "description": "Auto-fix ESLint issues. Command: npx eslint --fix {path} or eslint --fix src/. Use run_shell_command.",
        },
        {
            "name": "prettier_format",
            "category": "frontend_quality",
            "description": "Format code with Prettier. Command: npx prettier --write {path} or prettier --write 'src/**/*.{ts,tsx,js,jsx,json,css}'. Use run_shell_command.",
        },
        {
            "name": "jest_run",
            "category": "frontend_testing",
            "description": "Run Jest tests. Command: npx jest or npx jest --testPathPattern={pattern}. Use run_shell_command.",
        },
        {
            "name": "jest_coverage",
            "category": "frontend_testing",
            "description": "Run Jest with coverage report. Command: npx jest --coverage. Use run_shell_command.",
        },
        {
            "name": "vitest_run",
            "category": "frontend_testing",
            "description": "Run Vitest tests. Command: npx vitest run or npx vitest run --reporter=verbose. Use run_shell_command.",
        },
        {
            "name": "playwright_test",
            "category": "frontend_testing",
            "description": "Run Playwright end-to-end tests. Command: npx playwright test or npx playwright test --reporter=list. Use run_shell_command.",
        },
        {
            "name": "vite_build",
            "category": "frontend_build",
            "description": "Build project with Vite. Command: npx vite build or vite build. Use run_shell_command.",
        },
        {
            "name": "next_build",
            "category": "frontend_build",
            "description": "Build Next.js project for production. Command: npm run build or next build. Use run_shell_command.",
        },
        {
            "name": "next_dev",
            "category": "frontend_build",
            "description": "Start Next.js development server. Command: npm run dev or next dev. Use run_shell_command.",
        },
        # ── Shell / General utilities ─────────────────────────────────────────
        {
            "name": "find_files",
            "category": "shell_utility",
            "description": "Find files by name or extension recursively. Command: find {dir} -name '*.py' or find . -name '*.cpp' -not -path '*/build/*'. Use run_shell_command.",
        },
        {
            "name": "grep_recursive",
            "category": "shell_utility",
            "description": "Search file contents recursively with grep. Command: grep -rn '{pattern}' {dir} or grep -rn 'TODO' src/. Use run_shell_command.",
        },
        {
            "name": "sed_replace",
            "category": "shell_utility",
            "description": "Replace text in files with sed. Command: sed -i 's/old/new/g' {file} or find . -name '*.py' | xargs sed -i 's/old/new/g'. Use run_shell_command.",
        },
        {
            "name": "curl_get",
            "category": "shell_network",
            "description": "Make HTTP GET request with curl. Command: curl -s {url} or curl -s -H 'Authorization: Bearer {token}' {url}. Use run_shell_command.",
        },
        {
            "name": "curl_post_json",
            "category": "shell_network",
            "description": "POST JSON with curl. Command: curl -s -X POST -H 'Content-Type: application/json' -d '{\"key\":\"val\"}' {url}. Use run_shell_command.",
        },
        {
            "name": "jq_parse",
            "category": "shell_utility",
            "description": "Parse and query JSON with jq. Command: echo '{json}' | jq '.field' or cat file.json | jq '.results[0]'. Use run_shell_command.",
        },
        {
            "name": "tar_extract",
            "category": "shell_archive",
            "description": "Extract tar archive. Command: tar -xzf {file.tar.gz} -C {dest_dir} or tar -xjf {file.tar.bz2}. Use run_shell_command.",
        },
        {
            "name": "tar_create",
            "category": "shell_archive",
            "description": "Create tar archive. Command: tar -czf {archive.tar.gz} {dir} or tar -czf backup.tar.gz src/. Use run_shell_command.",
        },
        {
            "name": "zip_create",
            "category": "shell_archive",
            "description": "Create zip archive. Command: zip -r {archive.zip} {dir} or zip -r dist.zip dist/. Use run_shell_command.",
        },
        {
            "name": "zip_extract",
            "category": "shell_archive",
            "description": "Extract zip archive. Command: unzip {archive.zip} -d {dest_dir}. Use run_shell_command.",
        },
        {
            "name": "docker_build",
            "category": "docker",
            "description": "Build Docker image. Command: docker build -t {image_name}:{tag} . or docker build -f Dockerfile.prod -t myapp:latest .. Use run_shell_command.",
        },
        {
            "name": "docker_run",
            "category": "docker",
            "description": "Run Docker container. Command: docker run -it --rm {image_name} or docker run -p 8080:80 {image_name}. Use run_shell_command.",
        },
        {
            "name": "docker_compose_up",
            "category": "docker",
            "description": "Start services with docker-compose. Command: docker-compose up -d or docker compose up --build. Use run_shell_command.",
        },
        {
            "name": "docker_compose_down",
            "category": "docker",
            "description": "Stop docker-compose services. Command: docker-compose down or docker compose down -v (to also remove volumes). Use run_shell_command.",
        },
        {
            "name": "env_check",
            "category": "shell_utility",
            "description": "Check environment variables. Command: env | grep {PATTERN} or printenv {VAR_NAME}. Use run_shell_command.",
        },
        {
            "name": "process_kill",
            "category": "shell_utility",
            "description": "Find and kill processes. Command: pkill -f {process_name} or kill $(lsof -t -i:{port}). Use run_shell_command.",
        },
        {
            "name": "port_check",
            "category": "shell_network",
            "description": "Check what is listening on a port. Command: lsof -i :{port} or ss -tlnp | grep {port}. Use run_shell_command.",
        },
        {
            "name": "disk_usage",
            "category": "shell_utility",
            "description": "Check disk usage of directories. Command: du -sh {dir} or du -sh * | sort -h. Use run_shell_command.",
        },
        {
            "name": "chmod_executable",
            "category": "shell_utility",
            "description": "Make file executable. Command: chmod +x {file} or chmod 755 {file}. Use run_shell_command.",
        },
        {
            "name": "symlink_create",
            "category": "shell_utility",
            "description": "Create symbolic link. Command: ln -s {target} {link_name} or ln -sf {target} {link_name} to force. Use run_shell_command.",
        },
        {
            "name": "watch_command",
            "category": "shell_utility",
            "description": "Run a command repeatedly and watch output. Command: watch -n 2 {command} (e.g. watch -n 2 'ls -lh build/'). Use run_shell_command.",
        },
        {
            "name": "git_log_pretty",
            "category": "git",
            "description": "Show git log with graph and color. Command: git log --oneline --graph --decorate --all. Use run_shell_command.",
        },
        {
            "name": "git_stash",
            "category": "git",
            "description": "Stash or restore working tree changes. Command: git stash or git stash pop or git stash list. Use run_shell_command.",
        },
        {
            "name": "git_reset_soft",
            "category": "git",
            "description": "Undo last commit keeping changes staged. Command: git reset --soft HEAD~1. Use run_shell_command.",
        },
        {
            "name": "rsync_copy",
            "category": "shell_utility",
            "description": "Copy files efficiently with rsync, preserving permissions. Command: rsync -avz {src}/ {dest}/ or rsync -avz --exclude='*.pyc' src/ dest/. Use run_shell_command.",
        },
    ]

    count = 0
    for recipe in recipes:
        try:
            state.tools.register_tool(
                name=recipe["name"],
                category=recipe["category"],
                description=recipe["description"],
                source="recipe",
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to seed recipe {recipe['name']}: {e}")

    logger.info(f"Seeded {count} common tool recipes")
    return count


def initialize_workspace(workspace_dir: Optional[Path] = None) -> Path:
    """
    Initialize GAIA Code workspace.

    Creates directory structure and initializes databases.

    Args:
        workspace_dir: Optional workspace directory (default: ~/.gaia/workspace)

    Returns:
        Path to workspace directory
    """
    if workspace_dir is None:
        workspace_dir = Path.home() / ".gaia" / "workspace"

    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Get shared state (creates databases)
    state = get_shared_state(workspace_dir)

    # Register core tools
    tools_count = register_core_tools(state)

    # Seed common tool recipes (Python, C++, Frontend, Shell)
    recipes_count = seed_common_tool_recipes(state)

    # Register specialists
    specialists_count = register_specialists(state)

    # Register initial skills
    skills_count = register_initial_skills(state)

    logger.info(f"Workspace initialized at {workspace_dir}")
    logger.info(f"  - {tools_count} tools registered")
    logger.info(f"  - {recipes_count} tool recipes seeded")
    logger.info(f"  - {specialists_count} specialists registered")
    logger.info(f"  - {skills_count} skills registered")

    return workspace_dir


def find_specialist_for_task(task: str) -> Optional[str]:
    """
    Find the best specialist for a task by scoring agents.db capabilities.

    The capabilities column is stored as a JSON list (e.g. ["debugging", "error_fix"]).
    Each agent is scored by how many capabilities appear in the task string; the highest
    scorer with score >= 1 is returned.

    Args:
        task: Task description

    Returns:
        Specialist name or None if no good match
    """
    import json

    state = get_shared_state()
    task_lower = task.lower()

    try:
        rows = state.agents.conn.execute(
            "SELECT name, capabilities FROM agents ORDER BY confidence DESC"
        ).fetchall()
    except Exception:
        return None

    best_name, best_score = None, 0
    for name, caps_json in rows:
        if not caps_json:
            continue
        caps = json.loads(caps_json) if isinstance(caps_json, str) else (caps_json or [])
        score = sum(1 for cap in caps if cap.lower() in task_lower)
        if score > best_score:
            best_score, best_name = score, name

    if best_name and best_score >= 1:
        logger.info("[RAC] auto-selected specialist=%s score=%d", best_name, best_score)
        return best_name

    return None


def get_tool_packs_for_specialist(specialist_name: str) -> List[str]:
    """
    Get tool packs for a specialist.

    Args:
        specialist_name: Name of specialist

    Returns:
        List of tool pack names
    """
    state = get_shared_state()

    try:
        cursor = state.agents.conn.execute(
            "SELECT tool_packs FROM agents WHERE name = ?", (specialist_name,)
        )
        row = cursor.fetchone()

        if row and row[0]:
            return row[0].split(",")

    except Exception as e:
        logger.warning(f"Failed to get tool packs for {specialist_name}: {e}")

    # Default tool packs if query fails
    return ["core", "coding"]


def get_specialist_system_prompt(specialist_name: str) -> Optional[str]:
    """
    Get system prompt for a specialist.

    Args:
        specialist_name: Name of specialist

    Returns:
        System prompt or None
    """
    state = get_shared_state()

    try:
        cursor = state.agents.conn.execute(
            "SELECT system_prompt FROM agents WHERE name = ?", (specialist_name,)
        )
        row = cursor.fetchone()

        if row:
            return row[0]

    except Exception as e:
        logger.warning(f"Failed to get system prompt for {specialist_name}: {e}")

    return None


def list_available_specialists() -> List[dict]:
    """
    List all registered specialists.

    Returns:
        List of specialist info dicts
    """
    state = get_shared_state()

    try:
        cursor = state.agents.conn.execute(
            "SELECT name, description, capabilities, confidence FROM agents"
        )

        specialists = []
        for row in cursor.fetchall():
            specialists.append(
                {
                    "name": row[0],
                    "description": row[1],
                    "capabilities": row[2].split(",") if row[2] else [],
                    "confidence": row[3],
                }
            )

        return specialists

    except Exception as e:
        logger.warning(f"Failed to list specialists: {e}")
        return []


def get_workspace_stats() -> dict:
    """
    Get statistics about the workspace.

    Returns:
        Dict with statistics
    """
    state = get_shared_state()

    stats = {
        "tools": {
            "total": 0,
            "core": 0,
            "learned": 0,
        },
        "skills": {
            "total": 0,
        },
        "specialists": {
            "total": 0,
        },
        "knowledge": {
            "insights": 0,
            "preferences": 0,
            "learnings": 0,
        },
        "plan": {
            "total_tasks": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
        },
    }

    try:
        # Tool stats
        cursor = state.tools.conn.execute("SELECT COUNT(*) FROM tools")
        stats["tools"]["total"] = cursor.fetchone()[0]

        cursor = state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools WHERE source = 'core'"
        )
        stats["tools"]["core"] = cursor.fetchone()[0]

        cursor = state.tools.conn.execute(
            "SELECT COUNT(*) FROM tools WHERE source = 'learned'"
        )
        stats["tools"]["learned"] = cursor.fetchone()[0]

        # Skills stats
        cursor = state.skills.conn.execute("SELECT COUNT(*) FROM skills")
        stats["skills"]["total"] = cursor.fetchone()[0]

        # Specialists stats
        cursor = state.agents.conn.execute("SELECT COUNT(*) FROM agents")
        stats["specialists"]["total"] = cursor.fetchone()[0]

        # Knowledge stats
        cursor = state.knowledge.conn.execute("SELECT COUNT(*) FROM insights")
        stats["knowledge"]["insights"] = cursor.fetchone()[0]

        cursor = state.knowledge.conn.execute("SELECT COUNT(*) FROM preferences")
        stats["knowledge"]["preferences"] = cursor.fetchone()[0]

        cursor = state.knowledge.conn.execute("SELECT COUNT(*) FROM learnings")
        stats["knowledge"]["learnings"] = cursor.fetchone()[0]

        # Plan stats
        tasks = state.plan.get_all_tasks()
        stats["plan"]["total_tasks"] = len(tasks)
        stats["plan"]["completed"] = len([t for t in tasks if t.status == "completed"])
        stats["plan"]["in_progress"] = len(
            [t for t in tasks if t.status == "in_progress"]
        )
        stats["plan"]["pending"] = len([t for t in tasks if t.status == "pending"])

    except Exception as e:
        logger.warning(f"Failed to get workspace stats: {e}")

    return stats
