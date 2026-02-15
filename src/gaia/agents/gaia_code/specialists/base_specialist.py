# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Base Specialist Agent

All specialist agents inherit from this base class.
Specialists have:
- Custom state machines
- Domain-specific tools
- Specialized system prompts
- Workflow templates
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional


class SpecialistState(Enum):
    """State machine states for specialists."""

    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseSpecialist(ABC):
    """
    Base class for all specialist agents.

    Each specialist:
    - Has a custom state machine
    - Uses domain-specific tools
    - Has a specialized system prompt
    - Follows a specific workflow

    Specialists are invoked via agent_query(specialist="specialist_name")
    """

    def __init__(self):
        self.state = SpecialistState.ANALYZING
        self.workflow_steps = self.define_workflow()
        self.current_step = 0
        self.results = {}

    @abstractmethod
    def define_workflow(self) -> List[str]:
        """
        Define the specialist's workflow.

        Returns:
            List of workflow step names

        Example:
            return ["analyze", "diagnose", "fix", "validate"]
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the specialist's system prompt.

        Returns:
            Specialized system prompt
        """
        pass

    @abstractmethod
    def get_tool_packs(self) -> List[str]:
        """
        Get the tool packs this specialist uses.

        Returns:
            List of tool pack names

        Example:
            return ["core", "coding", "debugging"]
        """
        pass

    def get_capabilities(self) -> List[str]:
        """
        Get the specialist's capabilities.

        Returns:
            List of capability descriptions
        """
        return []

    def transition_state(self, new_state: SpecialistState):
        """Transition to a new state."""
        self.state = new_state

    def next_step(self) -> Optional[str]:
        """
        Get the next workflow step.

        Returns:
            Next step name or None if workflow complete
        """
        if self.current_step < len(self.workflow_steps):
            step = self.workflow_steps[self.current_step]
            self.current_step += 1
            return step
        return None

    def reset(self):
        """Reset the specialist to initial state."""
        self.state = SpecialistState.ANALYZING
        self.current_step = 0
        self.results = {}

    def get_metadata(self) -> Dict:
        """
        Get specialist metadata for registration.

        Returns:
            Dict with name, description, capabilities, tool_packs
        """
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__ or "",
            "capabilities": self.get_capabilities(),
            "tool_packs": self.get_tool_packs(),
            "workflow": self.workflow_steps,
        }
