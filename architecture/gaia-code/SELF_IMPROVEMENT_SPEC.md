# Self-Improvement Specification (v1)

## Context

GAIA Code currently treats itself as a black box — it can index and modify any user codebase, but has no awareness of its own implementation. When it hits a limitation (e.g., a tool doesn't handle an edge case, a prompt produces poor results, the retrieval pipeline misses relevant code), it has no mechanism to:

1. Identify that the problem is in *its own code* rather than the user's
2. Locate the specific file/function in the gaia SDK responsible
3. Propose a fix
4. Apply the fix and verify it works

This spec adds **self-introspection** — a separate, always-available index of the gaia SDK codebase that GAIA Code can query while working on any user project. When the agent struggles, it can examine its own internals, diagnose the root cause, and propose (or apply) improvements.

**Key principle:** The gaia SDK index is *read-only by default*. Modifications require explicit user approval at the configured autonomy level.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                GAIA Code Agent                   │
│                                                  │
│  ┌──────────────┐    ┌──────────────────────┐   │
│  │ User Project  │    │ Self-Introspection   │   │
│  │ Pipeline      │    │ Pipeline             │   │
│  │               │    │                      │   │
│  │ code_index.db │    │ gaia_self_index.db   │   │
│  │ (user code)   │    │ (gaia SDK code)      │   │
│  └──────────────┘    └──────────────────────┘   │
│         │                      │                 │
│         └──────┐   ┌──────────┘                 │
│                ▼   ▼                             │
│         ┌──────────────┐                        │
│         │ Diagnosis    │                        │
│         │ Engine       │                        │
│         │              │                        │
│         │ "Is this my  │                        │
│         │  limitation  │                        │
│         │  or theirs?" │                        │
│         └──────┬───────┘                        │
│                │                                 │
│                ▼                                 │
│         ┌──────────────┐                        │
│         │ Improvement  │                        │
│         │ Workflow     │                        │
│         │              │                        │
│         │ propose →    │                        │
│         │ fix →        │                        │
│         │ test →       │                        │
│         │ PR           │                        │
│         └──────────────┘                        │
└─────────────────────────────────────────────────┘
```

### Two Pipelines, One Agent

The agent maintains **two separate `CodeRetrievalPipeline` instances**:

| Pipeline | Database | Root Path | Purpose |
|----------|----------|-----------|---------|
| **User Pipeline** | `code_index.db` | User's project root | Index the codebase being worked on |
| **Self Pipeline** | `gaia_self_index.db` | gaia SDK root (`site-packages/gaia/` or repo root) | Index gaia's own code for introspection |

The self pipeline is initialized lazily on first self-introspection query and reuses the same `CodeRetrievalPipeline` class from the code retrieval pipeline.

---

## Autonomy Levels

Configurable via `KnowledgeDB.store_preference("self_improvement_autonomy", level)` or `CredentialManager.set_setting("self_improvement_autonomy", level)`.

| Level | Name | Behavior | Default |
|-------|------|----------|---------|
| 0 | `propose_only` | Identify issue, suggest fix in chat. No code changes. | **Yes** |
| 1 | `fix_and_ask` | Write fix to staging area, show diff, ask permission to apply. | |
| 2 | `auto_pr` | Create branch, apply fix, run tests, open PR on `amd/gaia`. User reviews PR. | |
| 3 | `self_modify` | Apply fix to local installation, hot-reload, continue task. PR as follow-up. | |

**Safety rules regardless of level:**
- Never modify files outside the gaia SDK scope
- Never force-push or modify `main` branch
- Always run existing tests before proposing any change
- Always show the user what changed and why
- Rate limit: Max 3 self-improvement proposals per session (prevent infinite loops)
- Escalation: If self-fix doesn't resolve the issue after 1 attempt, stop and ask the user

---

## Component 1: Self-Introspection Database

### Initialization

```python
class SelfIntrospection:
    """Manages the gaia SDK self-index for agent introspection."""

    def __init__(self, workspace_dir: Path):
        self.gaia_root = self._find_gaia_root()
        self.pipeline = None  # Lazy init

    def _find_gaia_root(self) -> Path:
        """Find the gaia SDK root directory.

        Priority:
        1. GAIA_SOURCE_ROOT env var (for development)
        2. Git repo root (if running from source checkout)
        3. site-packages/gaia/ (if installed as package)
        """
        ...

    def ensure_indexed(self):
        """Ensure the self-index is built and up to date."""
        if self.pipeline is None:
            self.pipeline = CodeRetrievalPipeline(
                root_path=str(self.gaia_root),
                workspace_dir=self.workspace_dir,
                db_name="gaia_self_index.db",
                llm_provider="none",  # AST-only for self-index (fast, no API cost)
                enable_watcher=False,  # No watcher for self-index
            )
            self.pipeline.index_repository()
```

### Self-Query API

```python
def query_self(self, query: str, top_k: int = 10) -> Dict:
    """Query the gaia SDK codebase.

    Examples:
        query_self("how does search_codebase work")
        query_self("where is the tool registry")
        query_self("what handles FTS5 search")
    """
    self.ensure_indexed()
    return self.pipeline.query(query, top_k=top_k)

def get_self_source(self, module_path: str) -> str:
    """Get the source code of a gaia module.

    Examples:
        get_self_source("gaia.agents.gaia_code.tools")
        get_self_source("gaia.agents.base.agent")
    """
    ...

def get_self_architecture(self) -> str:
    """Get architecture summary of the gaia SDK."""
    self.ensure_indexed()
    return self.pipeline.query("architecture overview")

def impact_on_self(self, file_path: str, symbol: str = None) -> Dict:
    """What would break if we change this gaia module?"""
    self.ensure_indexed()
    return self.pipeline.impact_analysis(file_path, symbol)
```

---

## Component 2: Diagnosis Engine

### When to Diagnose

The agent should consider self-diagnosis when:

1. **Repeated tool failures** — Same tool fails 3+ times on different inputs
2. **Quality gate failures** — Output consistently fails quality checks
3. **Search misses** — `search_codebase` returns no relevant results when they should exist
4. **Escalation ladder exhaustion** — Retry → decompose → cloud all failed
5. **User frustration signals** — User says "that's wrong", "try again", "you keep missing X"

### Diagnosis Flow

```
1. Detect struggle signal (repeated failures, user feedback, etc.)
2. Classify: Is this likely a user-code issue or a gaia-code issue?
   - User-code signals: compile errors, test failures, unfamiliar framework
   - Self-code signals: tool returns wrong format, search misses obvious results,
     same error pattern across different user projects
3. If self-code suspected:
   a. Query self-index for the relevant gaia module
   b. Read the source code of the struggling component
   c. Identify the root cause (missing edge case, bad regex, etc.)
   d. Generate a diagnosis report
4. Pass to Improvement Workflow
```

### Diagnosis Report Format

```python
@dataclass
class SelfDiagnosis:
    trigger: str              # What triggered the diagnosis
    component: str            # e.g., "code_retrieval.py:_search_fts5"
    root_cause: str           # Natural language explanation
    severity: str             # "bug", "limitation", "enhancement"
    suggested_fix: str        # Natural language description of the fix
    affected_files: List[str] # Files that need modification
    confidence: float         # 0.0-1.0 confidence in the diagnosis
    evidence: List[str]       # Specific observations that led to this diagnosis
```

### Diagnosis Classification Heuristics

| Signal | Threshold | Classification |
|--------|-----------|---------------|
| Tool returns empty/error 3+ times | 3 consecutive | Possible tool bug |
| FTS5/FAISS search returns 0 results for obvious query | 1 occurrence | Possible indexing gap |
| Quality gate fails with same pattern 2+ times | 2 consecutive | Possible gate misconfiguration |
| LLM response parsing fails | 2 consecutive | Possible parser limitation |
| User says "wrong" or "try again" 3+ times | 3 in session | Check recent tool output |

---

## Component 3: Improvement Workflow

### Level 0: Propose Only (Default)

```
Agent → User:
  "I identified a limitation in my own code that's affecting this task.

  **Component:** code_retrieval.py:_search_fts5 (line 1580)
  **Issue:** FTS5 query sanitization strips hyphens, so searching for
  'well-known' becomes 'well known' and misses exact matches.
  **Suggested fix:** Preserve hyphens in FTS5 queries by quoting terms.

  Would you like me to fix this? (Current autonomy: propose_only)
  You can change this with: set_preference('self_improvement_autonomy', 'fix_and_ask')"
```

### Level 1: Fix and Ask

```
1. Write the fix to a staging directory (~/.gaia/self_improvements/)
2. Run gaia's own test suite against the fix
3. Show the user:
   - The diff
   - Test results (pass/fail)
   - Impact analysis (what else might be affected)
4. Ask: "Apply this fix? [Yes / No / Open PR instead]"
5. If approved: Apply to local installation
```

### Level 2: Auto PR

```
1. Clone amd/gaia to ~/.gaia/self_improvements/gaia/
2. Create branch: self-fix/{component}-{timestamp}
3. Apply the fix
4. Run test suite
5. If tests pass:
   a. Commit with message describing the fix
   b. Push branch
   c. Open PR via `gh pr create`:
      Title: "fix: {short description}"
      Body: Full diagnosis report + test results
   d. Report PR URL to user
6. If tests fail: Report to user, do not open PR
```

### Level 3: Self-Modify (Hot Reload)

```
1. Apply fix to the running gaia installation (site-packages or source)
2. Hot-reload the modified module:
   importlib.reload(module)
3. Re-run the failed operation with the fix applied
4. If it works:
   - Report success to user
   - Optionally open PR (Level 2 flow)
5. If it doesn't work:
   - Revert the change
   - Report to user
   - Stop self-improvement for this issue
```

**Hot-reload safety:**
- Only reload leaf modules (not base agent or shared state)
- Snapshot module state before reload for revert
- Never hot-reload during active tool execution
- Log all hot-reloads to audit log

---

## Component 4: GitHub Integration

### Using `gh` CLI

Since gaia has no existing GitHub API integration, we use the `gh` CLI (GitHub CLI) which is widely available and handles authentication.

```python
def create_self_improvement_pr(
    diagnosis: SelfDiagnosis,
    diff: str,
    test_results: Dict,
) -> Optional[str]:
    """Create a PR on amd/gaia with the self-improvement fix.

    Returns PR URL or None if failed.
    """
    repo_url = "https://github.com/amd/gaia"
    branch_name = f"self-fix/{diagnosis.component.replace('.', '-')}-{timestamp}"

    # 1. Fork/clone if needed
    run_command(f"gh repo fork {repo_url} --clone=true", cwd=staging_dir)

    # 2. Create branch
    run_command(f"git checkout -b {branch_name}", cwd=repo_dir)

    # 3. Apply changes (already written to staging)
    apply_staged_changes(repo_dir)

    # 4. Commit
    run_command(
        f'git commit -am "fix({diagnosis.component}): {diagnosis.root_cause[:72]}"',
        cwd=repo_dir,
    )

    # 5. Push
    run_command(f"git push origin {branch_name}", cwd=repo_dir)

    # 6. Create PR
    pr_body = format_pr_body(diagnosis, test_results)
    result = run_command(
        f'gh pr create --repo {repo_url} --title "fix: {title}" --body "{pr_body}"',
        cwd=repo_dir,
    )

    return extract_pr_url(result)
```

### PR Template

```markdown
## Self-Improvement Fix

**Triggered by:** {diagnosis.trigger}
**Component:** `{diagnosis.component}`
**Severity:** {diagnosis.severity}

### Root Cause
{diagnosis.root_cause}

### Fix Description
{diagnosis.suggested_fix}

### Evidence
{bulleted list of diagnosis.evidence}

### Test Results
- Existing tests: {pass_count}/{total_count} passing
- New tests added: {new_test_count}
- No regressions detected

### Impact Analysis
{impact analysis from self-pipeline}

---
*This PR was automatically generated by GAIA Code's self-improvement system.*
*Autonomy level: {level_name}*
*Review by a human maintainer is required before merging.*
```

---

## Tool Interface

### New Tools

```python
@tool
def introspect_self(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Search GAIA's own codebase for understanding its internals.

    Use when you need to understand how your own tools, prompts,
    or pipeline work. Searches the gaia SDK source code.
    """

@tool
def diagnose_self(issue_description: str) -> Dict[str, Any]:
    """Diagnose whether a problem is in GAIA's own code.

    Use when you've struggled with a task and suspect the issue
    might be in your own implementation rather than the user's code.
    Returns a diagnosis with root cause and suggested fix.
    """

@tool
def propose_self_improvement(
    diagnosis_id: str,
    fix_description: str,
) -> Dict[str, Any]:
    """Propose an improvement to GAIA's own code.

    Behavior depends on self_improvement_autonomy setting:
    - propose_only: Show suggestion to user
    - fix_and_ask: Write fix, show diff, ask permission
    - auto_pr: Apply fix, run tests, open PR
    - self_modify: Apply fix, hot-reload, continue
    """
```

### Configuration Tool

```python
@tool
def set_self_improvement_autonomy(level: str) -> Dict[str, Any]:
    """Set the self-improvement autonomy level.

    Levels: propose_only (default), fix_and_ask, auto_pr, self_modify
    """
```

---

## Storage

### Self-Improvement History

Stored in `knowledge.db` as insights with category `self_improvement`:

```python
state.knowledge.store_insight(
    category="self_improvement",
    content=json.dumps({
        "diagnosis": asdict(diagnosis),
        "action_taken": "proposed",  # or "fixed", "pr_created", "hot_reloaded"
        "outcome": "success",  # or "failed", "reverted", "pending_review"
        "pr_url": "https://github.com/amd/gaia/pull/123",
    }),
    domain="self_improvement",
    triggers=["self-fix", diagnosis.component],
)
```

This enables:
- Tracking which self-improvements have been proposed/applied
- Avoiding duplicate proposals for the same issue
- Learning which types of issues are most common
- Reporting self-improvement statistics

---

## Safety & Guardrails

1. **Rate limiting**: Max 3 self-improvement proposals per session
2. **Scope enforcement**: Only modify files within the gaia SDK (validated by path prefix)
3. **Test gate**: Any fix must pass existing tests before being proposed
4. **Revert capability**: All changes are reversible (git branch or backup)
5. **Audit logging**: All self-improvement actions logged to audit log
6. **User override**: User can always say "no" or disable self-improvement entirely
7. **No infinite loops**: If a self-fix doesn't resolve the issue, stop after 1 attempt
8. **PR review required**: Even at auto_pr level, the PR requires human review to merge

---

## Configuration Defaults

```python
DEFAULT_SELF_IMPROVEMENT_CONFIG = {
    "autonomy_level": "propose_only",       # Default: safest level
    "max_proposals_per_session": 3,          # Rate limit
    "max_fix_attempts": 1,                   # Stop after 1 failed fix
    "enable_hot_reload": False,              # Disabled by default
    "auto_run_tests": True,                  # Always test fixes
    "staging_dir": "~/.gaia/self_improvements/",
    "gaia_repo_url": "https://github.com/amd/gaia",
    "allowed_scope": "full_sdk",             # full_sdk | gaia_code_only | tools_only
}
```

---

## Implementation Phases

### Phase 1: Self-Introspection (Foundation)
- `SelfIntrospection` class with lazy pipeline initialization
- `introspect_self` tool
- Gaia root detection (source vs installed)
- **Modify:** `code_retrieval.py` (add `db_name` parameter), `tools.py`

### Phase 2: Diagnosis Engine
- `DiagnosisEngine` class with struggle detection
- `diagnose_self` tool
- Diagnosis report generation
- Integration with quality gates and escalation ladder
- **Create:** `self_improvement.py`

### Phase 3: Improvement Workflow (Level 0-1)
- `propose_self_improvement` tool
- Staging directory management
- Diff generation and presentation
- Test runner integration
- **Modify:** `self_improvement.py`

### Phase 4: GitHub Integration (Level 2)
- `gh` CLI wrapper for fork/branch/commit/push/PR
- PR template generation
- PR status tracking
- **Modify:** `self_improvement.py`

### Phase 5: Hot-Reload (Level 3)
- Module snapshot and restore
- `importlib.reload` integration
- Safety checks for reload-safe modules
- **Modify:** `self_improvement.py`

---

## Key Files

| File | Action |
|------|--------|
| `src/gaia/agents/gaia_code/self_improvement.py` | **CREATE** — SelfIntrospection, DiagnosisEngine, ImprovementWorkflow |
| `src/gaia/agents/gaia_code/code_retrieval.py` | **MODIFY** — Add db_name parameter for separate self-index DB |
| `src/gaia/agents/gaia_code/tools.py` | **MODIFY** — Add introspect_self, diagnose_self, propose_self_improvement tools |
| `src/gaia/agents/gaia_code/agent.py` | **MODIFY** — Initialize SelfIntrospection, wire diagnosis triggers |
| `tests/unit/test_self_improvement.py` | **CREATE** — Tests for introspection, diagnosis, workflow |

---

## Verification

1. Index gaia's own codebase via self-pipeline
2. Query: "how does search_codebase work?" → returns relevant tool code
3. Query: "where is the FTS5 search?" → returns _search_fts5 method
4. Simulate a diagnosis trigger (repeated tool failure)
5. Verify diagnosis report is accurate
6. Test propose_only flow (show suggestion, no changes)
7. Test fix_and_ask flow (write fix, show diff, user approves)
8. Test auto_pr flow (create branch, PR) — requires gh CLI
9. Verify rate limiting (max 3 per session)
10. Verify scope enforcement (can't modify files outside gaia)
