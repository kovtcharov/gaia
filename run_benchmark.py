#!/usr/bin/env python
# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Benchmark script: Run GaiaCodeAgent on C++ port task.

Uses GaiaCodeAgent (the autonomous coding agent) which:
- Reads existing codebases directly via file tools
- Writes output to absolute paths specified in the query
- Uses Claude without Lemonade/Qwen dependencies
- Works with any language (not web-stack locked)
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

# Load .env (handles CRLF line endings correctly, unlike bash `source`)
_env = Path(__file__).parent / ".env"
if _env.exists():
    import os
    for line in _env.read_bytes().decode("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# Add gaia to path
sys.path.insert(0, str(Path("/mnt/c/Users/14255/Work/gaia/src")))

OUTPUT_DIR = "/mnt/c/Users/14255/Work/Projects/GaiaCodeExperiments/gaiacpp_gaia"

# Enumerate the exact files the agent MUST create so it can't skip writing them.
REQUIRED_FILES = [
    f"{OUTPUT_DIR}/CMakeLists.txt",
    f"{OUTPUT_DIR}/include/gaia/agent.hpp",
    f"{OUTPUT_DIR}/include/gaia/tool_registry.hpp",
    f"{OUTPUT_DIR}/include/gaia/console.hpp",
    f"{OUTPUT_DIR}/include/gaia/mcp_client.hpp",
    f"{OUTPUT_DIR}/src/agent.cpp",
    f"{OUTPUT_DIR}/src/tool_registry.cpp",
    f"{OUTPUT_DIR}/src/console.cpp",
    f"{OUTPUT_DIR}/src/mcp_client.cpp",
    f"{OUTPUT_DIR}/tests/test_agent.cpp",
    f"{OUTPUT_DIR}/demo/main.cpp",
    f"{OUTPUT_DIR}/README.md",
]

BENCHMARK_QUERY = (
    f"TASK: Write a complete, working C++17 implementation of the GAIA agent framework to disk.\n"
    f"\n"
    f"OUTPUT DIRECTORY: {OUTPUT_DIR}/\n"
    f"\n"
    f"SCOPE: Write ALL files needed for a WORKING, COMPILABLE system — not just the 12 listed below.\n"
    f"If a working program needs additional utility files (e.g. json_utils.hpp, types.h), write them.\n"
    f"Ask: 'What does a WORKING, compilable C++ agent framework need?' — not 'What was explicitly listed?'\n"
    f"\n"
    f"MINIMUM REQUIRED FILES (12 files — may write more):\n"
    + "\n".join(f"  {f}" for f in REQUIRED_FILES)
    + f"\n"
    f"\n"
    f"IMPLEMENTATION PLAN — execute these 11 agent_query calls in sequence:\n"
    f"\n"
    f"  STEP 1 — Build system (write files only, do NOT run cmake):\n"
    f"    agent_query(task='Write {OUTPUT_DIR}/CMakeLists.txt and {OUTPUT_DIR}/README.md. "
    f"CMake 3.17+, C++17, FetchContent for nlohmann_json + GTest. "
    f"include_directories(include). README includes cmake -B build && cmake --build build && ctest. "
    f"IMPORTANT: Just write the 2 files. Do NOT run cmake, do NOT run any shell commands.')\n"
    f"\n"
    f"  STEP 2 — Shared types header (write file only, do NOT compile):\n"
    f"    agent_query(task='Write {OUTPUT_DIR}/include/gaia/types.h with shared types: "
    f"ToolFunction (std::function<std::string(const std::string&)>), AgentState enum (IDLE, RUNNING, DONE, ERROR), "
    f"ToolCall struct (name, args, result). C++17, include guards. "
    f"IMPORTANT: Just write the 1 file. Do NOT compile, do NOT run cmake, do NOT run any shell commands.')\n"
    f"\n"
    f"  STEP 3 — Core headers (READ types.h first, then write headers, do NOT compile):\n"
    f"    agent_query(task='READ {OUTPUT_DIR}/include/gaia/types.h FIRST. Then write 4 headers in {OUTPUT_DIR}/include/gaia/: "
    f"agent.hpp (Agent class, uses AgentState+ToolFunction from types.h, run(query)->string, add_tool), "
    f"tool_registry.hpp (ToolRegistry, register_tool, dispatch, list_tools, uses ToolFunction), "
    f"console.hpp (print_thought, print_tool_usage, print_final_answer with ANSI color), "
    f"mcp_client.hpp (MCPClient, call_tool via JSON-RPC 2.0 over stdin/stdout). C++17, include guards. "
    f"IMPORTANT: Just write the 4 header files. Do NOT compile, do NOT run cmake, do NOT run any shell commands.')\n"
    f"\n"
    f"  STEP 4a — Write agent.cpp (read only types.h + agent.hpp, do NOT compile):\n"
    f"    agent_query(task='READ ONLY these 2 files using read_file: "
    f"{OUTPUT_DIR}/include/gaia/types.h and {OUTPUT_DIR}/include/gaia/agent.hpp. "
    f"Do NOT read any other files. The file {OUTPUT_DIR}/src/agent.cpp does NOT exist yet. "
    f"After reading, write {OUTPUT_DIR}/src/agent.cpp from scratch: Agent class implementation with "
    f"processQuery loop that dispatches tools from ToolRegistry and uses Console for output. "
    f"#include relative path to agent.hpp. Write the complete file with write_file NOW. "
    f"IMPORTANT: Do NOT compile, do NOT run cmake, do NOT run any shell commands.')\n"
    f"\n"
    f"  STEP 4b — Write tool_registry.cpp (read only types.h + tool_registry.hpp, do NOT compile):\n"
    f"    agent_query(task='READ ONLY these 2 files using read_file: "
    f"{OUTPUT_DIR}/include/gaia/types.h and {OUTPUT_DIR}/include/gaia/tool_registry.hpp. "
    f"Do NOT read any other files. The file {OUTPUT_DIR}/src/tool_registry.cpp does NOT exist yet. "
    f"After reading, write {OUTPUT_DIR}/src/tool_registry.cpp from scratch: ToolRegistry implementation "
    f"using unordered_map<string,ToolFunction> for dispatch and list_tools. "
    f"#include relative path to tool_registry.hpp. Write the complete file with write_file NOW. "
    f"IMPORTANT: Do NOT compile, do NOT run cmake, do NOT run any shell commands.')\n"
    f"\n"
    f"  STEP 4c — Write console.cpp (read only console.hpp, do NOT compile):\n"
    f"    agent_query(task='READ ONLY this 1 file using read_file: "
    f"{OUTPUT_DIR}/include/gaia/console.hpp. "
    f"Do NOT read any other files. The file {OUTPUT_DIR}/src/console.cpp does NOT exist yet. "
    f"After reading, write {OUTPUT_DIR}/src/console.cpp from scratch: Console implementation "
    f"with ANSI color codes using std::cout (print_thought in cyan, print_tool_usage in yellow, "
    f"print_final_answer in green). "
    f"#include relative path to console.hpp. Write the complete file with write_file NOW. "
    f"IMPORTANT: Do NOT compile, do NOT run cmake, do NOT run any shell commands.')\n"
    f"\n"
    f"  STEP 4d — Write mcp_client.cpp (read only types.h + mcp_client.hpp, do NOT compile):\n"
    f"    agent_query(task='READ ONLY these 2 files using read_file: "
    f"{OUTPUT_DIR}/include/gaia/types.h and {OUTPUT_DIR}/include/gaia/mcp_client.hpp. "
    f"Do NOT read any other files. The file {OUTPUT_DIR}/src/mcp_client.cpp does NOT exist yet. "
    f"After reading, write {OUTPUT_DIR}/src/mcp_client.cpp from scratch: MCPClient implementation "
    f"that calls tools via JSON-RPC 2.0 over stdin/stdout using nlohmann_json. "
    f"#include relative path to mcp_client.hpp. Write the complete file with write_file NOW. "
    f"IMPORTANT: Do NOT compile, do NOT run cmake, do NOT run any shell commands.')\n"
    f"\n"
    f"  STEP 5 — Analyze codebase for bugs (specialist: CppCodeAnalysisAgent):\n"
    f"    agent_query(task='Analyze ALL C++ files in {OUTPUT_DIR} for bugs. "
    f"Check: header vs implementation API mismatches, missing #includes, constructor signature mismatches, "
    f"CMakeLists.txt linkage gaps, ODR violations. "
    f"Read every .hpp and .cpp file. Run cmake -B {OUTPUT_DIR}/build -S {OUTPUT_DIR} 2>&1. "
    f"Output a structured JSON bug report.', specialist='CppCodeAnalysisAgent')\n"
    f"\n"
    f"  STEP 6 — Fix all bugs from analysis (specialist: CppBugBasherAgent):\n"
    f"    agent_query(task='Fix ALL bugs from the CppCodeAnalysisAgent report. "
    f"Read each file before editing. Apply targeted edits (edit_file). "
    f"After fixes: cmake -B {OUTPUT_DIR}/build -S {OUTPUT_DIR} && cmake --build {OUTPUT_DIR}/build. "
    f"Output must compile cleanly.', specialist='CppBugBasherAgent')\n"
    f"\n"
    f"  STEP 7a — Write test_agent.cpp (specialist: TestingAgent, read agent.hpp + tool_registry.hpp + console.hpp, do NOT compile):\n"
    f"    agent_query(task='READ ONLY these 3 files: "
    f"{OUTPUT_DIR}/include/gaia/agent.hpp, {OUTPUT_DIR}/include/gaia/tool_registry.hpp, "
    f"{OUTPUT_DIR}/include/gaia/console.hpp. "
    f"Do NOT read any other files. Then write {OUTPUT_DIR}/tests/test_agent.cpp: "
    f"GoogleTest suite with ≥10 tests for Agent, ToolRegistry, Console — "
    f"use ONLY constructors/methods confirmed in the headers you read. "
    f"Write the complete file with write_file NOW. "
    f"IMPORTANT: Do NOT compile, do NOT run cmake, do NOT run any shell commands.', specialist='TestingAgent')\n"
    f"\n"
    f"  STEP 7b — Write demo/main.cpp (specialist: TestingAgent, read agent.hpp + tool_registry.hpp, do NOT compile):\n"
    f"    agent_query(task='READ ONLY these 2 files: "
    f"{OUTPUT_DIR}/include/gaia/agent.hpp and {OUTPUT_DIR}/include/gaia/tool_registry.hpp. "
    f"Do NOT read any other files. Then write {OUTPUT_DIR}/demo/main.cpp: "
    f"demo program that creates an Agent, registers echo and reverse tools, runs one query, prints result. "
    f"Write the complete file with write_file NOW. "
    f"IMPORTANT: Do NOT compile, do NOT run cmake, do NOT run any shell commands.', specialist='TestingAgent')\n"
    f"\n"
    f"  STEP 8 — Compile and run tests:\n"
    f"    run_shell_command('cmake -B {OUTPUT_DIR}/build -S {OUTPUT_DIR} -DCMAKE_BUILD_TYPE=Debug && "
    f"cmake --build {OUTPUT_DIR}/build --parallel && cd {OUTPUT_DIR}/build && ctest --output-on-failure')\n"
    f"\n"
    f"Execute all 11 steps. After all steps complete, verify all 12+ files exist with "
    f"list_files(path='{OUTPUT_DIR}'), then answer with the compilation and test results."
)


def clear_workspace(output_dir: Path):
    """
    Full workspace wipe for a reproducible benchmark run.

    Deletes all DB files in ~/.gaia/workspace/ (they are recreated fresh on
    next agent startup, with agents.db reseeded with the 7 default specialists).
    Also wipes the benchmark output directory.
    """
    workspace = Path.home() / ".gaia" / "workspace"
    if workspace.exists():
        for f in workspace.iterdir():
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
        print(f"  Cleared workspace: {workspace}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"  Cleared output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="GAIA Code Benchmark: C++ Port")
    parser.add_argument(
        "--no-fresh",
        action="store_true",
        help="Skip workspace reset and reuse state from the previous run",
    )
    args = parser.parse_args()
    fresh = not args.no_fresh

    print("=" * 70)
    print("GAIA Code Benchmark: C++ Port of Core Agent Framework")
    print("=" * 70)
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Query:   {BENCHMARK_QUERY[:100]}...")
    print(f"Fresh:   {fresh} (use --no-fresh to reuse previous state)")
    print("=" * 70)

    out = Path(OUTPUT_DIR)

    if fresh:
        print("\nClearing workspace for fresh run...")
        clear_workspace(out)
        print()

    # Ensure output directory exists (agent will create C++ files inside it)
    out.mkdir(parents=True, exist_ok=True)

    from gaia.agents.gaia_code.agent import GaiaCodeAgent

    start_time = time.time()
    agent = GaiaCodeAgent(
        # Use default ~/.gaia/workspace for DBs; gaiacpp_gaia is only the code output target
        silent_mode=False,
        tui_mode="off",
        # Tell the agent where output files will live so path validation passes
        target_dir=OUTPUT_DIR,
    )

    print(f"\nAgent initialized in {time.time() - start_time:.1f}s")
    print("\nRunning benchmark query...\n")

    result = agent.process_query(BENCHMARK_QUERY)

    # --- Retry guard: if any required files are missing, prompt the agent ---
    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        print(f"\n[BENCHMARK] {len(missing)} required files missing — sending corrective follow-up...\n")
        for mf in missing:
            print(f"  MISSING: {mf}")
        followup = (
            f"The following {len(missing)} required files are still missing from {OUTPUT_DIR}/:\n"
            + "\n".join(f"  {f}" for f in missing)
            + f"\n\nPlease write ALL missing files now using write_file. "
            f"Write each file with complete, working C++17 content. "
            f"Do NOT return an answer until all files exist — verify with "
            f"list_files(\"{OUTPUT_DIR}\")."
        )
        result = agent.process_query(followup, create_plan=False)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"Benchmark completed in {elapsed:.1f}s")
    success = result.get("success", False)
    print(f"Success: {success}")
    answer = result.get("result") or result.get("answer") or "No result"
    print(f"Result:  {str(answer)[:300]}")
    print("=" * 70)

    # List generated source files (exclude any .db files)
    files = sorted(out.rglob("*") if out.exists() else [])
    cpp_files = [f for f in files if f.is_file() and f.suffix != ".db"]
    print(f"\nFiles created: {len(cpp_files)}")
    for f in cpp_files[:30]:
        print(f"  {f.relative_to(out)}")
    if len(cpp_files) > 30:
        print(f"  ... and {len(cpp_files) - 30} more")

    # Save result summary
    result_path = out / "benchmark_result.json"
    with open(result_path, "w") as f:
        json.dump(
            {
                "success": success,
                "result": str(answer)[:1000],
                "elapsed_seconds": elapsed,
                "files_created": [str(f.relative_to(out)) for f in cpp_files],
            },
            f,
            indent=2,
        )
    print(f"\nResult saved to {result_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
