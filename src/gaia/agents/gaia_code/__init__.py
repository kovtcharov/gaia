# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code: The World's Most Autonomous Coding Agent

This module implements the Recursive Agent Composition (RAC) architecture
for autonomous coding with:
- Context-lean design (never fills context window)
- Continuous execution (no step limits)
- Quality gates (verifies output works)
- Checkpoint/resume (survives interruptions)
- Persistent memory (learns across sessions)
- Recursive decomposition via agent_query()
"""

from .agent import GaiaCodeAgent

__all__ = ["GaiaCodeAgent"]
