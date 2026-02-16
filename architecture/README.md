# GAIA Architecture

**Status**: Active development
**Priority**: `gaia code` (autonomous coding agent) is the #1 priority. Everything else follows.

---

## Directory Structure

```
architecture/
├── README.md                    # This file
│
├── gaia-code/                   # PRIMARY FOCUS: Autonomous Coding Agent
│   ├── README.md                        # Start here - overview of the 4 docs
│   ├── RECURSIVE_AGENT_COMPOSITION.md   # NEW: The RAC paradigm (read this first!)
│   ├── GAIA_CODE_AUTONOMOUS_AGENT.md    # Complete spec - all capabilities
│   ├── GAIA_CODE_MILESTONES.md          # Implementation roadmap (M0-M10)
│   └── GAIA_CODE_EVALUATION_PLAN.md     # Benchmarks + validation
│
├── planning/                    # Roadmaps, timelines, branch plans
│   ├── V2_FEATURE_BRANCH_PLAN.md        # 26 feature branches across 2 tracks
│   ├── AI_ACCELERATED_TIMELINE.md       # 12-week timeline
│   ├── IMPLEMENTATION_ROADMAP.md        # Implementation phases
│   ├── ARCHITECTURE_MASTER_INDEX.md     # Index of all architecture docs
│   ├── ARCHITECTURE_UPDATES_SUMMARY.md  # Changelog of architecture changes
│   ├── CATEGORY_ENABLEMENT_GAPS.md      # Gaps for 4 agent categories
│   └── FINAL_ARCHITECTURE_SUITE_REPORT.md
│
├── features/                    # Agent framework feature specs
│   ├── PERSISTENT_MEMORY_FRAMEWORK.md   # 4-tier memory system
│   ├── ADAPTIVE_PROMPTS_FRAMEWORK.md    # State machine + dynamic prompts
│   ├── DYNAMIC_TOOLS_FRAMEWORK.md       # Tool creation + SKILLS
│   ├── LEARNING_ADAPTATION_FRAMEWORK.md # 6-step learning loop
│   ├── ARCHITECTURE_MANIFEST_FRAMEWORK.md # Project tracking
│   └── REQUIREMENTS_GATHERING_STATE.md  # Clarifying questions before planning
│
├── infrastructure/              # Platform: security, testing, deployment
│   ├── ERROR_RECOVERY_ARCHITECTURE.md
│   ├── OBSERVABILITY_ARCHITECTURE.md
│   ├── SECURITY_REVIEW_RECOMMENDATIONS.md
│   ├── SECURITY_SECRETS_ARCHITECTURE.md
│   ├── TESTING_FRAMEWORK_ARCHITECTURE.md
│   └── DEPLOYMENT_ARCHITECTURE.md
│
├── agents/                      # Domain-specific agent architectures
│   ├── COMPUTER_USE_ARCHITECTURE.md     # CUA (Computer Use Agent)
│   ├── EMAIL_INTEGRATION_ARCHITECTURE.md
│   ├── INTEGRATION_ARCHITECTURE.md
│   ├── MULTI_AGENT_ORCHESTRATION_ARCHITECTURE.md
│   └── WORKFLOW_ORCHESTRATION_ARCHITECTURE.md
│
├── ui/                          # User interface specs
│   ├── TUI_DESIGN_SPECIFICATION.md      # Terminal UI (Textual)
│   ├── TUI_UX_ENHANCED_SPECIFICATION.md # Enhanced TUI with animations
│   ├── AGENT_DASHBOARD_DESIGN.md        # Web dashboard (React)
│   └── TASK_CENTRIC_INTERFACE_PARADIGM.md # Task > Chat interaction model
│
└── research/                    # Analysis, references, research
    ├── ADVANCED_RAG_ARCHITECTURE.md
    ├── ANTHROPIC_SKILLS_STANDARD.md
    ├── CROSS_CUTTING_CONCERNS_ARCHITECTURE.md
    ├── GAIA4_INSIGHTS_INTEGRATION.md
    ├── LANGUAGE_EXTENSIBILITY_DESIGN.md
    ├── MISSING_ARCHITECTURES_ASSESSMENT.md
    ├── MULTI_DOCUMENT_SYNTHESIS_ARCHITECTURE.md
    └── SWIFT_IOS_APPLICABILITY.md
```

---

## Where to Start

### If you're building `gaia code` (the coding agent):

1. **Read**: `gaia-code/RECURSIVE_AGENT_COMPOSITION.md` -- understand the RAC paradigm (the innovation)
2. **Plan**: `gaia-code/GAIA_CODE_MILESTONES.md` -- what to build and in what order
3. **Reference**: `gaia-code/GAIA_CODE_AUTONOMOUS_AGENT.md` -- complete spec (all capabilities, architecture)
4. **Validate**: `gaia-code/GAIA_CODE_EVALUATION_PLAN.md` -- how to prove it works

### If you're working on other agents:

- **CUA**: `agents/COMPUTER_USE_ARCHITECTURE.md`
- **Workflows**: `agents/WORKFLOW_ORCHESTRATION_ARCHITECTURE.md`
- **Email**: `agents/EMAIL_INTEGRATION_ARCHITECTURE.md`

### If you're working on shared infrastructure:

- Feature specs in `features/` -- memory, tools, learning, prompts
- Platform specs in `infrastructure/` -- security, testing, deployment

---

## Key Principles

1. **Recursive** -- agent_query() enables unlimited decomposition via specialized sub-agents
2. **Shared State** -- All agents share: memory.db, knowledge.db, manifest, plan (7 databases)
3. **Self-Extending** -- Creates tools → skills → agents from patterns
4. **Context-Lean** -- RLMs + knowledge DB, never fills context window
5. **Composable** -- Works with 0.6B to 200B LLMs, every feature opt-in
6. **Quality-Focused** -- Quality gates enforced at every recursion level
7. **Simple** -- Single-threaded + async/await, not multi-threaded
8. **Time-Aware** -- Everything timestamped, full audit trail
9. **Adaptive** -- Smart escalation: local first, cloud per-subtask with cost ceiling
10. **Learning** -- Exponential improvement (specialists create specialists)

---

*40 architecture documents organized by domain. Start with `gaia-code/`.*
