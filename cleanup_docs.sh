#!/bin/bash
# Cleanup redundant documentation

echo "Removing redundant/outdated documentation..."

# Delete redundant "FINAL" and "COMPLETE" status documents (keep only one)
rm -f ABSOLUTELY_EVERYTHING_COMPLETE.md
rm -f ABSOLUTELY_FINAL_COMPLETE.md
rm -f ABSOLUTELY_FINAL_STATUS.md
rm -f ACTUAL_STATUS.md
rm -f ALL_COMPLETE.txt
rm -f COMPLETE_SUCCESS.md
rm -f COMPLETE_IMPLEMENTATION.md
rm -f COMPLETE_M0_M7_TUI.md
rm -f COMPLETE_TOOL_INTEGRATION_PLAN.md
rm -f CONGRATULATIONS.md
rm -f EXECUTIVE_SUMMARY.md
rm -f FINAL_COMPLETE_STATUS.md
rm -f FINAL_DELIVERY_COMPLETE.md
rm -f FINAL_DELIVERY_SUMMARY.txt
rm -f FINAL_IMPLEMENTATION_REPORT.md
rm -f FINAL_STATUS_M0_M7.md
rm -f FINAL_STATUS.md
rm -f FINAL_SUMMARY.txt
rm -f FINAL_SYSTEM_COMPLETE.md
rm -f FINAL_WORKING_STATUS.md
rm -f GAIA_CODE_IMPLEMENTATION_REPORT.md
rm -f IMPLEMENTATION_COMPLETE.md
rm -f IMPLEMENTATION_FINAL_SUMMARY.md
rm -f IMPLEMENTATION_STATUS_REAL.txt
rm -f IMPLEMENTATION_SUMMARY.md
rm -f IMPLEMENTATION_VALIDATED.md
rm -f INTEGRATION_COMPLETE.md

# Delete incremental/temporary status documents
rm -f BUGS_FIXED_SUMMARY.md
rm -f CODE_REVIEW_FINDINGS.md
rm -f CURRENT_STATUS_HONEST.md
rm -f HONEST_ASSESSMENT.md
rm -f HONEST_STATUS.md
rm -f PLACEHOLDER_AUDIT.md
rm -f READY_FOR_EXECUTION.md
rm -f READY_TO_TEST.md
rm -f REVIEW_COMPLETE.md
rm -f SUCCESS.md
rm -f WORKING_STATUS.md
rm -f QUICKSTART_UPDATED.txt
rm -f TOOLS_STATUS.md
rm -f IMPLEMENTATION_STRATEGY.md
rm -f DATABASE_ACCESS_STATUS.md

# Delete redundant feature-specific docs (info is in main docs)
rm -f INTERACTIVE_MODE.md
rm -f INTERACTIVE_SESSION_COMPLETE.md
rm -f MISSING_TOOLS_ANALYSIS.md
rm -f LEGACY_CODE_AGENT_ANALYSIS.md

# Keep these essential documents:
# - START_HERE.md (entry point)
# - GAIA_CODE_QUICKSTART.md (how to use)  
# - GAIA_CODE_ARCHITECTURE_DIAGRAM.md (architecture reference)
# - README_GAIA_CODE.md (developer guide)
# - README_INSTALLATION.md (setup guide)
# - M7_IMPLEMENTATION_COMPLETE.md (codebase analysis features)
# - PERSONA_SYSTEM_COMPLETE.md (persona guide)
# - TUI_IMPLEMENTATION.md (TUI guide)
# - CLI_INTERACTION_COMPLETE.md (CLI tool interaction)
# - CODEBASE_ANALYSIS_CAPABILITIES.md (M7 features)
# - AUTONOMOUS_SELF_REVIEW.md (explains autonomous features)
# - TOOL_COMPARISON_CLAUDE_CODE.md (vs Claude Code)
# - EXCELLENT_CLI_UX_PACKAGES.md (package recommendations)
# - WORKING_NOW.md (current status)
# - FINAL_REVIEW_AND_FIXES.md (bug fixes applied)

echo "Cleanup complete!"
echo ""
echo "Kept essential documentation:"
ls -1 *.md | wc -l
echo "markdown files remaining"
