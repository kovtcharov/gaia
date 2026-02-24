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

import json
import sys
import time
from pathlib import Path

# Add gaia to path
sys.path.insert(0, str(Path("/mnt/c/Users/14255/Work/gaia/src")))

OUTPUT_DIR = "/mnt/c/Users/14255/Work/Projects/GaiaCodeExperiments/gaiacpp_gaia"

BENCHMARK_QUERY = (
    f"Analyze the GAIA Python agent framework source code in "
    f"/mnt/c/Users/14255/Work/gaia/src/gaia/agents/base/ and "
    f"/mnt/c/Users/14255/Work/gaia/src/gaia/mcp/, "
    f"then implement a C++17 port of the core framework. "
    f"Write all output files to {OUTPUT_DIR}/. "
    f"Include: "
    f"(1) base Agent class with state machine (idle/running/done/error), "
    f"(2) tool registry with registration and dispatch, "
    f"(3) console output interface (print_thought, print_tool_usage, print_final_answer), "
    f"(4) MCP client (JSON-RPC over stdio). "
    f"Use nlohmann/json for JSON, CMake as the build system, and Google Test for unit tests. "
    f"Include a simple demo agent that registers two tools and runs one query. "
    f"Create: CMakeLists.txt, include/gaia/*.hpp, src/*.cpp, tests/test_*.cpp, README.md."
)


def main():
    print("=" * 70)
    print("GAIA Code Benchmark: C++ Port of Core Agent Framework")
    print("=" * 70)
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Query:   {BENCHMARK_QUERY[:100]}...")
    print("=" * 70)

    # Ensure output directory exists (agent will create C++ files inside it)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    from gaia.agents.gaia_code.agent import GaiaCodeAgent

    start_time = time.time()
    agent = GaiaCodeAgent(
        # Use default ~/.gaia/workspace for DBs; gaiacpp_gaia is only the code output target
        silent_mode=False,
        tui_mode="off",
    )

    print(f"\nAgent initialized in {time.time() - start_time:.1f}s")
    print("\nRunning benchmark query...\n")

    result = agent.process_query(BENCHMARK_QUERY)

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"Benchmark completed in {elapsed:.1f}s")
    success = result.get("success", False)
    print(f"Success: {success}")
    answer = result.get("result") or result.get("answer") or "No result"
    print(f"Result:  {str(answer)[:300]}")
    print("=" * 70)

    # List generated source files (exclude any .db files)
    out = Path(OUTPUT_DIR)
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
