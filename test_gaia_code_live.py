#!/usr/bin/env python3
"""
GAIA Code Live Test - With Real Dependencies

Now that dependencies are installed, test the agent properly.
"""

import sys
from pathlib import Path

print("=" * 70)
print("GAIA CODE - LIVE TEST WITH DEPENDENCIES")
print("=" * 70)
print()

# Test 1: Import GAIA Code
print("[1] Importing GAIA Code...")
try:
    from gaia.agents.gaia_code import GaiaCodeAgent
    print("✓ GaiaCodeAgent imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import persona system
print("\n[2] Checking persona system...")
try:
    from gaia.agents.gaia_code.persona import PERSONALITY_PROFILES, list_personas

    personas = list_personas()
    print(f"✓ {len(personas)} personas available:")
    for p in personas:
        print(f"  - {p['name']:15} {p['description'][:50]}...")
except Exception as e:
    print(f"✗ Persona check failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check tool registration
print("\n[3] Checking tool registry...")
try:
    from gaia.agents.base.tools import _TOOL_REGISTRY

    print(f"✓ Tool registry available ({len(_TOOL_REGISTRY)} tools currently)")

    if _TOOL_REGISTRY:
        print("  Sample tools:")
        for tool_name in list(_TOOL_REGISTRY.keys())[:5]:
            print(f"    - {tool_name}")
except Exception as e:
    print(f"✗ Tool registry check failed: {e}")

# Test 4: Try to instantiate agent
print("\n[4] Instantiating GAIA Code agent...")
print("   (This will check for API key)")
try:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        agent = GaiaCodeAgent(
            workspace_dir=Path(tmpdir),
            persona="pike",
            tui_mode="off",  # No TUI for testing
            silent_mode=True,  # Quiet mode
        )

        print(f"✓ Agent instantiated successfully!")
        print(f"  Workspace: {agent.shared_state.workspace_dir}")
        print(f"  Persona: {agent.persona.profile.name}")
        print(f"  Quality gates: {len(agent.quality_gates.gates)}")

        # Check tools registered
        from gaia.agents.base.tools import _TOOL_REGISTRY
        print(f"  Registered tools: {len(_TOOL_REGISTRY)}")

        # Show sample of registered tools
        print(f"\n  Sample tools registered:")
        for tool_name in list(_TOOL_REGISTRY.keys())[:10]:
            print(f"    ✓ {tool_name}")

        if len(_TOOL_REGISTRY) > 10:
            print(f"    ... and {len(_TOOL_REGISTRY) - 10} more tools")

except Exception as e:
    print(f"✗ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n  This is expected if ANTHROPIC_API_KEY is not set.")
    print("  GAIA Code will prompt for it on first interactive run.")

print("\n" + "=" * 70)
print("LIVE TEST RESULTS")
print("=" * 70)
print("\nIf all tests passed, GAIA Code is ready!")
print("\nTry: gaia code 'Create a calculator' --persona pike")
print("=" * 70)
