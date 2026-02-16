#!/bin/bash
# Fix corrupted file paths

echo "GAIA Code Path Fix Utility"
echo "======================================================================"

# Create necessary directories
mkdir -p docs/guides
mkdir -p src/gaia/agents/gaia_code/specialists
mkdir -p tests/integration tests/unit

moved=0
errors=0

# Function to move a file
move_file() {
    local pattern="$1"
    local dest="$2"

    file=$(ls 2>/dev/null | grep "$pattern" | head -1)
    if [ -n "$file" ]; then
        if mv "$file" "$dest" 2>/dev/null; then
            echo "✓ Moved: $dest"
            ((moved++))
        else
            echo "✗ Error moving: $pattern"
            ((errors++))
        fi
    else
        echo "⚠ Not found: $pattern"
        ((errors++))
    fi
}

# Root-level markdown files
move_file "FINAL_IMPLEMENTATION_REPORT.md" "FINAL_IMPLEMENTATION_REPORT.md"
move_file "GAIA_CODE_IMPLEMENTATION_REPORT.md" "GAIA_CODE_IMPLEMENTATION_REPORT.md"
move_file "IMPLEMENTATION_SUMMARY.md" "IMPLEMENTATION_SUMMARY.md"

# Agent source files
move_file "agentsgaia_code__init__.py" "src/gaia/agents/gaia_code/__init__.py"
move_file "agentsgaia_codeagent.py" "src/gaia/agents/gaia_code/agent.py"
move_file "agentsgaia_codeagent_factory.py" "src/gaia/agents/gaia_code/agent_factory.py"
move_file "agentsgaia_codecli.py" "src/gaia/agents/gaia_code/cli.py"
move_file "agentsgaia_codedefragmentation.py" "src/gaia/agents/gaia_code/defragmentation.py"
move_file "agentsgaia_codeembedding_engine.py" "src/gaia/agents/gaia_code/embedding_engine.py"
move_file "agentsgaia_codeinsight_engine.py" "src/gaia/agents/gaia_code/insight_engine.py"
move_file "agentsgaia_codeintegration.py" "src/gaia/agents/gaia_code/integration.py"
move_file "agentsgaia_codequality_gates.py" "src/gaia/agents/gaia_code/quality_gates.py"
move_file "agentsgaia_codeREADME.md" "src/gaia/agents/gaia_code/README.md"
move_file "agentsgaia_codeshared_state.py" "src/gaia/agents/gaia_code/shared_state.py"
move_file "agentsgaia_codeskill_extractor.py" "src/gaia/agents/gaia_code/skill_extractor.py"
move_file "agentsgaia_codesystem_prompt.py" "src/gaia/agents/gaia_code/system_prompt.py"
move_file "agentsgaia_codetool_builder.py" "src/gaia/agents/gaia_code/tool_builder.py"
move_file "agentsgaia_codetools.py" "src/gaia/agents/gaia_code/tools.py"
move_file "agentsgaia_codevector_search.py" "src/gaia/agents/gaia_code/vector_search.py"

# Specialist agents
move_file "codespecialists__init__.py" "src/gaia/agents/gaia_code/specialists/__init__.py"
move_file "codespecialistsarchitecture_agent.py" "src/gaia/agents/gaia_code/specialists/architecture_agent.py"
move_file "codespecialistsbase_specialist.py" "src/gaia/agents/gaia_code/specialists/base_specialist.py"
move_file "codespecialistsdebugger_agent.py" "src/gaia/agents/gaia_code/specialists/debugger_agent.py"
move_file "codespecialistsdocumentation_agent.py" "src/gaia/agents/gaia_code/specialists/documentation_agent.py"
move_file "codespecialistsperformance_agent.py" "src/gaia/agents/gaia_code/specialists/performance_agent.py"
move_file "codespecialistsrefactoring_agent.py" "src/gaia/agents/gaia_code/specialists/refactoring_agent.py"
move_file "codespecialistssecurity_agent.py" "src/gaia/agents/gaia_code/specialists/security_agent.py"
move_file "codespecialiststesting_agent.py" "src/gaia/agents/gaia_code/specialists/testing_agent.py"

# Tests
move_file "testsintegrationtest_gaia_code_integration.py" "tests/integration/test_gaia_code_integration.py"
move_file "testsunittest_gaia_code_agent.py" "tests/unit/test_gaia_code_agent.py"
move_file "testsunittest_gaia_code_m5_m6.py" "tests/unit/test_gaia_code_m5_m6.py"
move_file "testsunittest_gaia_code_quality_gates.py" "tests/unit/test_gaia_code_quality_gates.py"
move_file "testsunittest_gaia_code_shared_state.py" "tests/unit/test_gaia_code_shared_state.py"

echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "Successfully moved: $moved files"
echo "Errors/Not found: $errors"
