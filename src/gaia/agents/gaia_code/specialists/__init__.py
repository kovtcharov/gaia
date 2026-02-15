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
"""

from .base_specialist import BaseSpecialist
from .debugger_agent import DebuggerAgent
from .security_agent import SecurityAgent
from .refactoring_agent import RefactoringAgent
from .testing_agent import TestingAgent
from .documentation_agent import DocumentationAgent
from .performance_agent import PerformanceAgent
from .architecture_agent import ArchitectureAgent

__all__ = [
    "BaseSpecialist",
    "DebuggerAgent",
    "SecurityAgent",
    "RefactoringAgent",
    "TestingAgent",
    "DocumentationAgent",
    "PerformanceAgent",
    "ArchitectureAgent",
]
