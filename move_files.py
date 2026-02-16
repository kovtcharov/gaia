#!/usr/bin/env python3
import os
import shutil

# Map path components to destinations
# Filenames have Unicode: \uf03a for : and \uf05c for \
mappings = [
    ("gaia_code\uf05cagent.py", "src/gaia/agents/gaia_code/agent.py"),
    ("gaia_code\uf05ccli.py", "src/gaia/agents/gaia_code/cli.py"),
    ("gaia_code\uf05cintegration.py", "src/gaia/agents/gaia_code/integration.py"),
    ("gaia_code\uf05cREADME.md", "src/gaia/agents/gaia_code/README.md"),
    ("specialists\uf05c__init__.py", "src/gaia/agents/gaia_code/specialists/__init__.py"),
    ("gaia_code\uf05ctools.py", "src/gaia/agents/gaia_code/tools.py"),
    ("integration\uf05ctest_gaia_code_integration.py", "tests/integration/test_gaia_code_integration.py"),
    ("unit\uf05ctest_gaia_code_agent.py", "tests/unit/test_gaia_code_agent.py"),
    ("unit\uf05ctest_gaia_code_m5_m6.py", "tests/unit/test_gaia_code_m5_m6.py"),
    ("unit\uf05ctest_gaia_code_quality_gates.py", "tests/unit/test_gaia_code_quality_gates.py"),
    ("unit\uf05ctest_gaia_code_shared_state.py", "tests/unit/test_gaia_code_shared_state.py"),
]

moved = 0
errors = []

for pattern, dest in mappings:
    # Find file matching pattern
    found = False
    for fname in os.listdir('.'):
        if pattern in fname and '\uf03a' in fname:  # Check for Unicode :
            try:
                shutil.move(fname, dest)
                print(f"✓ {dest}")
                moved += 1
                found = True
                break
            except Exception as e:
                errors.append(f"Error moving {fname}: {e}")

    if not found:
        errors.append(f"Not found: {pattern}")

print(f"\nMoved {moved} files")
print(f"Errors: {len(errors)}")
