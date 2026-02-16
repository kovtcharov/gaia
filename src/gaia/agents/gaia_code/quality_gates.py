# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Quality Gates: Re-exported from base agent module.

The canonical implementation lives in gaia.agents.base.quality_gates.
This module re-exports everything for backward compatibility.
"""

# Re-export everything from the base module
from gaia.agents.base.quality_gates import (
    EscalationLadder,
    GateResult,
    ImportGate,
    QualityGate,
    QualityGateRunner,
    SyntaxGate,
    TestGate,
)

__all__ = [
    "GateResult",
    "QualityGate",
    "SyntaxGate",
    "ImportGate",
    "TestGate",
    "QualityGateRunner",
    "EscalationLadder",
]
