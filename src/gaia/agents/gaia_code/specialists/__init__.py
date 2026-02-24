# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code Specialist Agents

M4: Agent Registry + Core Specialists

The 7 core specialists:
1. DebuggerAgent - Debugging and error diagnosis
2. SecurityAgent - Security scanning and vulnerability detection
3. RefactoringAgent - Code refactoring and cleanup
4. TestingAgent - Test generation and validation
5. DocumentationAgent - Documentation generation
6. PerformanceAgent - Performance analysis and optimization
7. ArchitectureAgent - Architecture analysis and design

Domain-specific analysis + bug-bashing pairs:
8.  CppCodeAnalysisAgent  - Full C++ codebase static analysis (produces JSON report)
9.  CppBugBasherAgent     - Apply C++ fixes from analysis report + recompile
10. PythonCodeAnalysisAgent - Full Python codebase static analysis (produces JSON report)
11. PythonBugBasherAgent  - Apply Python fixes from analysis report + retest
12. CodeAnalysisAgent     - General-purpose analysis for TS/JS/Rust/Go/Java/other
13. BugBasherAgent        - General-purpose bug fixing from analysis report
"""

from .base_specialist import BaseSpecialist
from .debugger_agent import DebuggerAgent
from .security_agent import SecurityAgent
from .refactoring_agent import RefactoringAgent
from .testing_agent import TestingAgent
from .documentation_agent import DocumentationAgent
from .performance_agent import PerformanceAgent
from .architecture_agent import ArchitectureAgent
from .cpp_code_analysis_agent import CppCodeAnalysisAgent
from .cpp_bug_basher_agent import CppBugBasherAgent
from .python_code_analysis_agent import PythonCodeAnalysisAgent
from .python_bug_basher_agent import PythonBugBasherAgent
from .code_analysis_agent import CodeAnalysisAgent
from .bug_basher_agent import BugBasherAgent

__all__ = [
    "BaseSpecialist",
    "DebuggerAgent",
    "SecurityAgent",
    "RefactoringAgent",
    "TestingAgent",
    "DocumentationAgent",
    "PerformanceAgent",
    "ArchitectureAgent",
    "CppCodeAnalysisAgent",
    "CppBugBasherAgent",
    "PythonCodeAnalysisAgent",
    "PythonBugBasherAgent",
    "CodeAnalysisAgent",
    "BugBasherAgent",
]
