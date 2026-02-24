#!/usr/bin/env python3
# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Plan Walkthrough: Shows the MasterPlan system in action.

Simulates a multi-agent coding project with milestones, tasks, subtasks,
and real-time status transitions — the same data the dashboard PlanView reads.

Run:
    python tests/system/run_plan_walkthrough.py [--workspace /path/to/.gaia/workspace]
    python tests/system/run_plan_walkthrough.py --keep  # don't delete workspace after
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on the path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gaia.agents.base.shared_state import SharedAgentState


def sep(label=""):
    width = 60
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * pad}")
    else:
        print(f"\n{'─' * width}")


def show_summary(plan):
    summary = plan.get_summary()
    if summary:
        print(summary)
    else:
        print("  (no active plan)")


def main():
    parser = argparse.ArgumentParser(description="Plan walkthrough demo")
    parser.add_argument("--workspace", help="Workspace directory (default: temp in /tmp)")
    parser.add_argument("--keep", action="store_true", help="Keep workspace after run")
    args = parser.parse_args()

    import tempfile
    if args.workspace:
        ws = Path(args.workspace)
        ws.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmpdir = tempfile.mkdtemp(prefix="gaia_plan_walkthrough_")
        ws = Path(tmpdir)
        cleanup = not args.keep

    print(f"\n{'═' * 60}")
    print("  GAIA MasterPlan Walkthrough")
    print(f"  Workspace: {ws}")
    print(f"{'═' * 60}")

    # Reset singleton so we get a fresh state
    SharedAgentState._instance = None
    state = SharedAgentState(workspace_dir=ws)
    plan = state.plan

    # ----------------------------------------------------------------
    # Step 1: Create the plan
    # ----------------------------------------------------------------
    sep("STEP 1: Create plan")
    plan_id = plan.create_plan(
        title="Build FastAPI REST service with auth and tests",
        project_dir="/code/fastapi-service",
        target_dir="/code/fastapi-service",
    )
    print(f"Plan ID: {plan_id[:8]}...")

    # ----------------------------------------------------------------
    # Step 2: Create milestone structure
    # ----------------------------------------------------------------
    sep("STEP 2: Add milestones")
    m_setup   = plan.create_task(plan_id, "Project setup",           depth=0, priority=1)
    m_models  = plan.create_task(plan_id, "Data models & schema",    depth=0, priority=2)
    m_api     = plan.create_task(plan_id, "API endpoints",           depth=0, priority=3)
    m_auth    = plan.create_task(plan_id, "Authentication",          depth=0, priority=4)
    m_tests   = plan.create_task(plan_id, "Test suite",              depth=0, priority=5)
    m_docs    = plan.create_task(plan_id, "Documentation",           depth=0, priority=6)
    print("Created 6 milestones")

    # ----------------------------------------------------------------
    # Step 3: Decompose each milestone into tasks
    # ----------------------------------------------------------------
    sep("STEP 3: Decompose into tasks")

    # Setup tasks
    t_dirs     = plan.create_task(plan_id, "Create directory structure",   parent_id=m_setup, created_by="gaia-code")
    t_deps     = plan.create_task(plan_id, "Write requirements.txt",        parent_id=m_setup, created_by="gaia-code")
    t_config   = plan.create_task(plan_id, "Configure pyproject.toml",      parent_id=m_setup, created_by="gaia-code")

    # Model tasks
    t_schema   = plan.create_task(plan_id, "Design database schema",        parent_id=m_models, created_by="gaia-code")
    t_models   = plan.create_task(plan_id, "Create SQLAlchemy models",      parent_id=m_models, created_by="gaia-code")
    t_migrate  = plan.create_task(plan_id, "Write Alembic migrations",      parent_id=m_models, created_by="gaia-code")

    # API tasks
    t_users    = plan.create_task(plan_id, "Implement /users CRUD",         parent_id=m_api, created_by="api-specialist")
    t_items    = plan.create_task(plan_id, "Implement /items CRUD",         parent_id=m_api, created_by="api-specialist")
    t_health   = plan.create_task(plan_id, "Implement /health endpoint",    parent_id=m_api, created_by="api-specialist")

    # Auth tasks
    t_jwt      = plan.create_task(plan_id, "JWT token generation",          parent_id=m_auth, created_by="security-specialist")
    t_login    = plan.create_task(plan_id, "Login/logout endpoints",        parent_id=m_auth, created_by="security-specialist")
    t_guard    = plan.create_task(plan_id, "Protected route middleware",    parent_id=m_auth, created_by="security-specialist")

    # Test tasks
    t_unit     = plan.create_task(plan_id, "Unit tests for models",         parent_id=m_tests, created_by="test-specialist")
    t_integ    = plan.create_task(plan_id, "Integration tests for API",     parent_id=m_tests, created_by="test-specialist")

    # Doc tasks
    t_readme   = plan.create_task(plan_id, "Write README.md",               parent_id=m_docs, created_by="docs-specialist")
    t_openapi  = plan.create_task(plan_id, "Verify OpenAPI schema",         parent_id=m_docs, created_by="docs-specialist")

    print("Created 17 tasks across 6 milestones")

    sep("Initial plan state")
    show_summary(plan)
    time.sleep(0.3)

    # ----------------------------------------------------------------
    # Step 4: Simulate agent execution
    # ----------------------------------------------------------------
    sep("STEP 4: Simulate execution")

    # Milestone 1: Setup (gaia-code agent)
    print("\n[gaia-code] Starting setup milestone...")
    plan.start_task(m_setup, owner="gaia-code")
    plan.start_task(t_dirs,  owner="gaia-code"); time.sleep(0.1)
    plan.complete_task(t_dirs, result="Created src/, tests/, docs/")
    plan.start_task(t_deps,  owner="gaia-code"); time.sleep(0.1)
    plan.complete_task(t_deps, result="requirements.txt written")
    plan.start_task(t_config, owner="gaia-code"); time.sleep(0.1)
    plan.complete_task(t_config, result="pyproject.toml configured")
    plan.complete_task(m_setup, result="Project structure ready")
    print("  Setup milestone complete ✓")

    # Milestone 2: Models (gaia-code agent)
    print("[gaia-code] Starting models milestone...")
    plan.start_task(m_models, owner="gaia-code")
    plan.start_task(t_schema, owner="gaia-code"); time.sleep(0.1)
    plan.complete_task(t_schema, result="Schema designed: users, items, sessions")
    plan.start_task(t_models, owner="gaia-code"); time.sleep(0.1)
    plan.complete_task(t_models, result="SQLAlchemy models written")
    plan.start_task(t_migrate, owner="gaia-code"); time.sleep(0.1)
    plan.complete_task(t_migrate, result="Initial migration created")
    plan.complete_task(m_models, result="All models and migrations ready")
    print("  Models milestone complete ✓")

    # Milestone 3: API — parallel agents
    print("[api-specialist] Starting API milestone...")
    plan.start_task(m_api, owner="api-specialist")
    plan.start_task(t_health, owner="api-specialist"); time.sleep(0.1)
    plan.complete_task(t_health, result="/health returns 200 with uptime")
    plan.start_task(t_users, owner="api-specialist"); time.sleep(0.1)
    plan.start_task(t_items, owner="api-specialist")  # parallel!
    plan.complete_task(t_users, result="GET/POST/PUT/DELETE /users implemented")
    plan.complete_task(t_items, result="GET/POST/PUT/DELETE /items implemented")
    plan.complete_task(m_api, result="All API endpoints implemented")
    print("  API milestone complete ✓")

    # Milestone 4: Auth — security specialist, one task blocked
    print("[security-specialist] Starting auth milestone...")
    plan.start_task(m_auth, owner="security-specialist")
    plan.start_task(t_jwt, owner="security-specialist"); time.sleep(0.1)
    plan.complete_task(t_jwt, result="JWT with RS256, 1h expiry")
    plan.start_task(t_login, owner="security-specialist"); time.sleep(0.1)
    plan.complete_task(t_login, result="POST /auth/login and /auth/logout")
    plan.start_task(t_guard, owner="security-specialist"); time.sleep(0.1)
    # Simulate a temporary block
    plan.block_task(t_guard, reason="Waiting for user model finalization")
    print("  [!] Protected route middleware blocked — dependency issue")

    sep("Mid-execution plan state")
    show_summary(plan)
    time.sleep(0.3)

    # Unblock and complete auth
    print("\n[security-specialist] Issue resolved, resuming...")
    plan.update_task_status(t_guard, "in_progress", owner="security-specialist")
    time.sleep(0.1)
    plan.complete_task(t_guard, result="Bearer token middleware added to all /users and /items routes")
    plan.complete_task(m_auth, result="Auth complete with JWT + protected routes")
    print("  Auth milestone complete ✓")

    # Milestone 5: Tests — test-specialist, one failure then recovery
    print("[test-specialist] Starting test milestone...")
    plan.start_task(m_tests, owner="test-specialist")
    plan.start_task(t_unit, owner="test-specialist"); time.sleep(0.1)
    plan.complete_task(t_unit, result="47 unit tests passing")
    plan.start_task(t_integ, owner="test-specialist"); time.sleep(0.1)
    # Simulate failure then fix
    plan.fail_task(t_integ, error="TestClient fails on /auth/login — missing test DB fixture")
    print("  [!] Integration tests failed — fixing fixture...")
    # Re-create the task as a retry
    t_integ2 = plan.create_task(plan_id, "Integration tests (retry)",
                                 parent_id=m_tests, created_by="test-specialist",
                                 description="Fixed: added SQLite test DB fixture")
    plan.start_task(t_integ2, owner="test-specialist"); time.sleep(0.1)
    plan.complete_task(t_integ2, result="23 integration tests passing after fixture fix")
    plan.complete_task(m_tests, result="70 tests total, all passing")
    print("  Test milestone complete ✓")

    # Milestone 6: Docs
    print("[docs-specialist] Starting documentation...")
    plan.start_task(m_docs, owner="docs-specialist")
    plan.start_task(t_readme, owner="docs-specialist"); time.sleep(0.1)
    plan.complete_task(t_readme, result="README with quickstart, API reference, deployment")
    plan.start_task(t_openapi, owner="docs-specialist"); time.sleep(0.1)
    plan.complete_task(t_openapi, result="OpenAPI schema validated, 12 endpoints documented")
    plan.complete_task(m_docs, result="Documentation complete")
    print("  Docs milestone complete ✓")

    # Complete the overall plan
    plan.complete_plan(plan_id)

    # ----------------------------------------------------------------
    # Step 5: Final state
    # ----------------------------------------------------------------
    sep("STEP 5: Final plan state")
    show_summary(plan)

    # Show progress stats
    sep("Progress stats")
    active = plan.get_active_plan()
    tasks = active["tasks"]
    total      = len(tasks)
    completed  = sum(1 for t in tasks if t["status"] == "completed")
    failed     = sum(1 for t in tasks if t["status"] == "failed")
    blocked    = sum(1 for t in tasks if t["status"] == "blocked")
    milestones = sum(1 for t in tasks if t["depth"] == 0)
    subtasks   = sum(1 for t in tasks if t["depth"] == 1)
    agents     = set(t["owner"] for t in tasks if t["owner"])

    print(f"  Plan status:   {active['status']}")
    print(f"  Total tasks:   {total}")
    print(f"  Milestones:    {milestones}")
    print(f"  Subtasks:      {subtasks}")
    print(f"  Completed:     {completed}")
    print(f"  Failed:        {failed}")
    print(f"  Agents:        {', '.join(sorted(agents))}")

    # Verify in memory.db
    sep("Verification")
    import sqlite3
    conn = sqlite3.connect(str(ws / "memory.db"))
    plan_count = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
    task_count = conn.execute("SELECT COUNT(*) FROM plan_tasks").fetchone()[0]
    event_count = conn.execute("SELECT COUNT(*) FROM plan_task_events").fetchone()[0]
    conn.close()

    print(f"  memory.db → plans:            {plan_count}")
    print(f"  memory.db → plan_tasks:       {task_count}")
    print(f"  memory.db → plan_task_events: {event_count}")
    print(f"\n  Workspace: {ws}")
    print("  Point the dashboard at this workspace to see the PlanView!")

    if cleanup:
        import shutil
        shutil.rmtree(ws)
        print(f"\n  (Workspace cleaned up. Use --keep to preserve it for the dashboard.)")

    sep()
    print("  Walkthrough complete.\n")


if __name__ == "__main__":
    main()
