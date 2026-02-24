# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Unit test configuration.

Provides fixtures shared across all unit tests, including singleton isolation
to prevent SharedAgentState from leaking between tests that use temporary
directories.
"""

import pytest


@pytest.fixture(autouse=True)
def reset_shared_agent_state():
    """
    Reset the SharedAgentState singleton before and after every unit test.

    Without this, a test that initialises SharedAgentState with a
    TemporaryDirectory will leave the singleton pointing at that (now deleted)
    path.  The next test then tries to write to that deleted path and gets:
        sqlite3.OperationalError: attempt to write a readonly database
    """
    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None
    yield
    SharedAgentState._instance = None
