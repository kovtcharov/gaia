#!/usr/bin/env python3
# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
GAIA Code TUI Demo

Demonstrates the three TUI modes without requiring full agent integration.
"""

import time

from src.gaia.agents.gaia_code.tui import (
    GaiaCodeMinimalTUI,
    GaiaCodeSimpleTUI,
    GaiaCodeTUI,
    TaskProgress,
)


def demo_simple_mode():
    """Demonstrate simple mode (default)."""
    print("\n" + "=" * 60)
    print("DEMO 1: Simple Mode (Default) - Clean and Balanced")
    print("=" * 60)
    print()

    tui = GaiaCodeSimpleTUI()

    # Start task
    tui.start("Build a REST API with JWT authentication")
    time.sleep(1)

    # Phase 1: Planning
    tui.update(stage="Planning", current="Analyzing requirements", percent=10)
    time.sleep(1)

    tui.update(stage="Planning", current="Creating task tree", percent=15)
    time.sleep(1)

    # Show plan
    tui.progress.stop()
    tui.show_plan([
        {"description": "Create project structure", "status": "completed"},
        {"description": "Define data models", "status": "in_progress"},
        {"description": "Implement auth endpoints", "status": "pending"},
        {"description": "Write tests", "status": "pending"},
        {"description": "Quality gates", "status": "pending"},
    ])
    time.sleep(2)
    tui.progress.start()

    # Phase 2: Executing
    tui.update(stage="Executing", current="Writing models.py", percent=30)
    time.sleep(1)

    tui.update(stage="Executing", current="Writing auth.py", percent=50)
    time.sleep(1)

    tui.update(stage="Executing", current="Writing endpoints.py", percent=70)
    time.sleep(1)

    # Phase 3: Testing
    tui.update(stage="Testing", current="Running pytest", percent=85)
    time.sleep(1)

    # Phase 4: Quality Gates
    tui.progress.stop()
    tui.update_quality_gates({
        "syntax": True,
        "imports": True,
        "tests": True,
    })
    time.sleep(1)

    tui.show_quality_gates_detail([
        {"name": "Syntax", "passed": True, "message": "All 8 files valid"},
        {"name": "Imports", "passed": True, "message": "All imports resolve"},
        {"name": "Tests", "passed": True, "message": "24/24 tests passing"},
    ])
    time.sleep(2)

    # Complete
    tui.complete(
        success=True,
        message="Created 8 files, 24 tests passing, all quality gates passed",
    )


def demo_minimal_mode():
    """Demonstrate minimal mode."""
    print("\n" + "=" * 60)
    print("DEMO 2: Minimal Mode - Absolute Minimum")
    print("=" * 60)
    print()

    tui = GaiaCodeMinimalTUI()

    tui.start("Create a calculator app")
    time.sleep(1)

    tui.update(step=1, total=5, current="Planning", percent=20)
    time.sleep(1)

    tui.update(step=3, total=5, current="Writing code", percent=60)
    time.sleep(1)

    tui.update(step=5, total=5, current="Testing", percent=100)
    time.sleep(1)

    tui.complete(success=True, message="All tests passing")


def demo_full_mode():
    """Demonstrate full mode."""
    print("\n" + "=" * 60)
    print("DEMO 3: Full Mode - Maximum Visibility")
    print("=" * 60)
    print()

    tui = GaiaCodeTUI()

    # Start task
    tui.start("Build a full-stack application", estimated_steps=12)
    time.sleep(1)

    # Add some task progress
    tasks = [
        TaskProgress("Create backend", "completed"),
        TaskProgress("Create frontend", "running"),
        TaskProgress("Write tests", "pending"),
        TaskProgress("Deploy", "pending"),
    ]
    tui.update_task_progress(tasks)
    time.sleep(1)

    # Update step
    tui.update_step(5, "Implementing React components")
    time.sleep(1)

    # Update quality gates
    tui.update_quality_gates({
        "syntax": "pass",
        "imports": "pass",
        "tests": "running",
    })
    time.sleep(2)

    # Update gates to all pass
    tui.update_quality_gates({
        "syntax": "pass",
        "imports": "pass",
        "tests": "pass",
    })
    time.sleep(1)

    # Add some activity
    tui.add_activity("Backend tests passing")
    time.sleep(0.5)
    tui.add_activity("Frontend build complete")
    time.sleep(0.5)
    tui.add_activity("Integration tests passing")
    time.sleep(1)

    # Complete
    tui.complete(success=True, message="Full-stack app created with 47 files, 156 tests")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("GAIA CODE TUI DEMONSTRATION")
    print("=" * 60)

    demo_simple_mode()
    time.sleep(2)

    demo_minimal_mode()
    time.sleep(2)

    demo_full_mode()

    print("\n" + "=" * 60)
    print("DEMOS COMPLETE")
    print("=" * 60)
    print("\nTo use in your code:")
    print("  agent = GaiaCodeAgent(tui_mode='simple')  # Default")
    print("\nTo use in CLI:")
    print("  gaia code 'task'  # Uses simple mode by default")
    print("  gaia code 'task' --tui full  # Full mode")
    print("  gaia code 'task' --tui minimal  # Minimal mode")
    print()


if __name__ == "__main__":
    main()
