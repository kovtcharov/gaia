#!/usr/bin/env python3
"""Fix corrupted file paths from WSL instance."""

import os
import shutil
from pathlib import Path

# Mapping of corrupted filenames to correct paths
FILE_MAPPINGS = {
    # Documentation
    "C:Users14255Workgaiadocsguidesgaia-code.mdx": "docs/guides/gaia-code.mdx",

    # Root-level markdown files
    "C:Users14255WorkgaiaFINAL_IMPLEMENTATION_REPORT.md": "FINAL_IMPLEMENTATION_REPORT.md",
    "C:Users14255WorkgaiaGAIA_CODE_IMPLEMENTATION_REPORT.md": "GAIA_CODE_IMPLEMENTATION_REPORT.md",
    "C:Users14255WorkgaiaIMPLEMENTATION_SUMMARY.md": "IMPLEMENTATION_SUMMARY.md",

    # Agent source files
    "C:Users14255Workgaiasrcgaiaagentsgaia_code__init__.py": "src/gaia/agents/gaia_code/__init__.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeagent.py": "src/gaia/agents/gaia_code/agent.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeagent_factory.py": "src/gaia/agents/gaia_code/agent_factory.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codecli.py": "src/gaia/agents/gaia_code/cli.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codedefragmentation.py": "src/gaia/agents/gaia_code/defragmentation.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeembedding_engine.py": "src/gaia/agents/gaia_code/embedding_engine.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeinsight_engine.py": "src/gaia/agents/gaia_code/insight_engine.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeintegration.py": "src/gaia/agents/gaia_code/integration.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codequality_gates.py": "src/gaia/agents/gaia_code/quality_gates.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeREADME.md": "src/gaia/agents/gaia_code/README.md",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeshared_state.py": "src/gaia/agents/gaia_code/shared_state.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codeskill_extractor.py": "src/gaia/agents/gaia_code/skill_extractor.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codesystem_prompt.py": "src/gaia/agents/gaia_code/system_prompt.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codetool_builder.py": "src/gaia/agents/gaia_code/tool_builder.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codetools.py": "src/gaia/agents/gaia_code/tools.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codevector_search.py": "src/gaia/agents/gaia_code/vector_search.py",

    # Specialist agents
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialists__init__.py": "src/gaia/agents/gaia_code/specialists/__init__.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialistsarchitecture_agent.py": "src/gaia/agents/gaia_code/specialists/architecture_agent.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialistsbase_specialist.py": "src/gaia/agents/gaia_code/specialists/base_specialist.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialistsdebugger_agent.py": "src/gaia/agents/gaia_code/specialists/debugger_agent.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialistsdocumentation_agent.py": "src/gaia/agents/gaia_code/specialists/documentation_agent.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialistsperformance_agent.py": "src/gaia/agents/gaia_code/specialists/performance_agent.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialistsrefactoring_agent.py": "src/gaia/agents/gaia_code/specialists/refactoring_agent.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialistssecurity_agent.py": "src/gaia/agents/gaia_code/specialists/security_agent.py",
    "C:Users14255Workgaiasrcgaiaagentsgaia_codespecialiststesting_agent.py": "src/gaia/agents/gaia_code/specialists/testing_agent.py",

    # Tests
    "C:Users14255Workgaiatestsintegrationtest_gaia_code_integration.py": "tests/integration/test_gaia_code_integration.py",
    "C:Users14255Workgaiatestsunittest_gaia_code_agent.py": "tests/unit/test_gaia_code_agent.py",
    "C:Users14255Workgaiatestsunittest_gaia_code_m5_m6.py": "tests/unit/test_gaia_code_m5_m6.py",
    "C:Users14255Workgaiatestsunittest_gaia_code_quality_gates.py": "tests/unit/test_gaia_code_quality_gates.py",
    "C:Users14255Workgaiatestsunittest_gaia_code_shared_state.py": "tests/unit/test_gaia_code_shared_state.py",
}

def main():
    """Move corrupted files to correct locations."""
    moved = []
    errors = []

    for corrupted_name, correct_path in FILE_MAPPINGS.items():
        # Check if source file exists
        if not os.path.exists(corrupted_name):
            errors.append(f"Source not found: {corrupted_name}")
            continue

        # Create destination directory if needed
        dest_path = Path(correct_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if destination already exists
        if dest_path.exists():
            # Backup existing file
            backup_path = str(dest_path) + ".backup"
            shutil.move(str(dest_path), backup_path)
            print(f"Backed up existing file: {correct_path} -> {backup_path}")

        # Move the file
        try:
            shutil.move(corrupted_name, str(dest_path))
            moved.append(f"{corrupted_name} -> {correct_path}")
            print(f"✓ Moved: {correct_path}")
        except Exception as e:
            errors.append(f"Error moving {corrupted_name}: {e}")
            print(f"✗ Error: {corrupted_name}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successfully moved: {len(moved)} files")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors:")
        for msg in errors:
            print(f"  - {msg}")

if __name__ == "__main__":
    print("GAIA Code Path Fix Utility")
    print("=" * 70)
    main()
