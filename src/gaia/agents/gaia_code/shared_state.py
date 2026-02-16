# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
SharedAgentState: Re-exported from base agent module.

The canonical implementation lives in gaia.agents.base.shared_state.
This module re-exports everything for backward compatibility.
"""

# Re-export everything from the base module
from gaia.agents.base.shared_state import (
    AgentCallFrame,
    AgentCallStack,
    AgentsDB,
    KnowledgeDB,
    MasterPlan,
    MemoryDB,
    Message,
    MessageQueue,
    ProjectManifest,
    SharedAgentState,
    SkillsDB,
    TaskNode,
    ToolsDB,
    get_shared_state,
)

__all__ = [
    "TaskNode",
    "AgentCallFrame",
    "Message",
    "MemoryDB",
    "KnowledgeDB",
    "ToolsDB",
    "SkillsDB",
    "AgentsDB",
    "MasterPlan",
    "AgentCallStack",
    "MessageQueue",
    "ProjectManifest",
    "SharedAgentState",
    "get_shared_state",
]
