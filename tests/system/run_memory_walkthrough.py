#!/usr/bin/env python3
# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Memory DB Walkthrough — Real LLM, No Mocking
=============================================

Exercises memory.db end-to-end using a real GaiaCodeAgent with real Claude API.
Nothing is mocked — real ChatSDK, real tool execution, real SQLite.

Requirements:
    ANTHROPIC_API_KEY must be set in the environment.

Run from the repo root:
    ANTHROPIC_API_KEY=sk-ant-... python tests/system/run_memory_walkthrough.py

What it does (step by step):
  1. Clears ~/.gaia/workspace/ and creates a fresh GaiaCodeAgent workspace
  2. Manually stores one fact in active_state via store_memory()
  3. Shows memory.db BEFORE the query
  4. Runs process_query() — Claude decides what to do (no scripting)
  5. Shows memory.db AFTER the query
  6. Validates key invariants and prints a summary
"""

import json
import os
import sqlite3
import sys
from pathlib import Path

# ── make sure we can import from src/ ──────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

# ── load .env if present ────────────────────────────────────────────────────
_env_file = REPO_ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    print(f"Loaded .env from {_env_file}")

WORKSPACE = Path.home() / ".gaia" / "workspace"
SEP = "─" * 72


# ============================================================================
# Helpers
# ============================================================================


def hr(title=""):
    if title:
        print(f"\n{SEP}\n  {title}\n{SEP}")
    else:
        print(SEP)


def dump_table(conn: sqlite3.Connection, table: str):
    """Print every row in a table with column headers."""
    cursor = conn.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in cursor.description]
    rows = cursor.fetchall()

    print(f"\n  table: {table}  ({len(rows)} row{'s' if len(rows) != 1 else ''})")
    if not rows:
        print("    (empty)")
        return

    widths = [len(c) for c in cols]
    for row in rows:
        for i, v in enumerate(row):
            widths[i] = max(widths[i], len(str(v) if v is not None else "NULL"))

    fmt = "    " + "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    print("    " + "  ".join("-" * w for w in widths))
    for row in rows:
        vals = [str(v) if v is not None else "NULL" for v in row]
        vals = [v[:70] + "…" if len(v) > 70 else v for v in vals]
        print(fmt.format(*vals))


# ============================================================================
# Setup
# ============================================================================


def check_api_key():
    hr("Pre-flight check")
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("  ERROR: ANTHROPIC_API_KEY is not set.")
        print()
        print("  Set it before running:")
        print("    export ANTHROPIC_API_KEY=sk-ant-...")
        print("    python tests/system/run_memory_walkthrough.py")
        sys.exit(1)
    print(f"  ANTHROPIC_API_KEY: {'*' * 20}{key[-6:]}")
    print("  LLM backend: Claude (real API, no mock)")


def setup_workspace() -> Path:
    hr("STEP 1 — Clear and prepare workspace")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    deleted = []
    for db in WORKSPACE.glob("*.db"):
        db.unlink()
        deleted.append(db.name)
    for extra in ["checkpoint.json"]:
        p = WORKSPACE / extra
        if p.exists():
            p.unlink()
    if deleted:
        print(f"  Deleted: {', '.join(sorted(deleted))}")
    else:
        print("  Workspace was already empty")
    print(f"  Workspace: {WORKSPACE}  (fresh start)")
    return WORKSPACE


def build_agent(ws: Path):
    hr("STEP 2 — Initialize GaiaCodeAgent (real ChatSDK, real SQLite)")
    from gaia.agents.base.shared_state import SharedAgentState

    SharedAgentState._instance = None

    from gaia.agents.gaia_code.agent import GaiaCodeAgent

    agent = GaiaCodeAgent(
        workspace_dir=ws,
        silent_mode=True,
        tui_mode="off",
    )

    print(f"  Agent:       {agent.__class__.__name__}")
    print(f"  LLM model:   {agent.model_id}")
    print(f"  Chat SDK:    {agent.chat.__class__.__name__}")
    print(f"  Workspace:   {ws}")
    print(f"  memory.db:   {ws / 'memory.db'}")
    return agent


# ============================================================================
# Main walkthrough
# ============================================================================


def main():
    print()
    print("=" * 72)
    print("  GAIA memory.db Walkthrough  (real LLM, no mocking)")
    print("=" * 72)

    check_api_key()

    ws = setup_workspace()
    agent = build_agent(ws)
    state = agent.shared_state
    file_path = str(ws / "hello.py")

    # ── STEP 3: seed one memory fact ─────────────────────────────────────
    hr("STEP 3 — Seed one fact into active_state")
    state.memory.store_memory(
        "project_lang",
        "Python 3.12",
        tags=["project", "config"],
    )
    print("  store_memory('project_lang', 'Python 3.12', tags=['project','config'])")
    print("  → 1 row written to active_state in memory.db")

    # ── STEP 4: snapshot BEFORE ──────────────────────────────────────────
    hr("STEP 4 — memory.db BEFORE process_query()")
    conn = sqlite3.connect(str(ws / "memory.db"))
    for t in ("active_state", "tool_results", "file_cache"):
        dump_table(conn, t)
    conn.close()

    # ── STEP 5: run the real LLM query ───────────────────────────────────
    hr("STEP 5 — Run process_query() with real Claude")
    task = (
        f"Create the file {file_path} with exactly this content:\n\n"
        "x = 1\n\n"
        "Do not add anything else. Just write that one line."
    )
    print(f"  Task: {task[:100]}")
    print()

    result = agent.process_query(task)

    print()
    print(f"  Result: {str(result)[:200]}")
    print(f"  hello.py on disk: {Path(file_path).exists()}")
    if Path(file_path).exists():
        print(f"  Content: {Path(file_path).read_text().strip()!r}")

    # ── STEP 6: snapshot AFTER ───────────────────────────────────────────
    hr("STEP 6 — memory.db AFTER process_query()")
    conn = sqlite3.connect(str(ws / "memory.db"))
    for t in ("active_state", "tool_results", "file_cache"):
        dump_table(conn, t)

    # ── STEP 7: assertions ───────────────────────────────────────────────
    hr("STEP 7 — Assertions")

    # active_state: seeded fact must still be there
    row = conn.execute(
        "SELECT key, value, tags, stored_at FROM active_state WHERE key='project_lang'"
    ).fetchone()
    assert row is not None, "FAIL: project_lang missing from active_state"
    assert row[1] == "Python 3.12"
    print(f"  ✓ active_state has project_lang = 'Python 3.12'")
    print(f"    stored_at={row[3]}  tags={row[2]}")

    # active_state: count all rows (agent may have stored additional facts)
    all_facts = conn.execute("SELECT COUNT(*) FROM active_state").fetchone()[0]
    print(f"  ✓ active_state total rows: {all_facts}  (seeded=1, agent may add more)")

    # tool_results: must have at least one entry (agent called at least one tool)
    tr_total = conn.execute("SELECT COUNT(*) FROM tool_results").fetchone()[0]
    assert tr_total >= 1, f"FAIL: tool_results is empty — agent called no tools"
    print(f"  ✓ tool_results: {tr_total} row(s)")
    tr_rows = conn.execute(
        "SELECT tool_name, timestamp FROM tool_results ORDER BY id"
    ).fetchall()
    for name, ts in tr_rows:
        print(f"      {ts}  {name}")

    # file_cache: show count (write_file won't populate, read_file would)
    fc_total = conn.execute("SELECT COUNT(*) FROM file_cache").fetchone()[0]
    print(f"  ✓ file_cache: {fc_total} row(s)")

    conn.close()

    # ── STEP 8: workspace summary ─────────────────────────────────────────
    hr("STEP 8 — All DB files in workspace")
    for db in sorted(ws.glob("*.db")):
        size = db.stat().st_size
        print(f"  {db.name:<20} {size:>8,} bytes")

    hr("DONE")
    print()
    print("  Inspect memory.db live:")
    print(f"    sqlite3 {ws / 'memory.db'}")
    print()

    from gaia.agents.base.shared_state import SharedAgentState
    SharedAgentState._instance = None


if __name__ == "__main__":
    main()
