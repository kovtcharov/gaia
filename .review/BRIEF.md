# Review brief — shared rules for every reviewer task

You are one reviewer in a parallel, comprehensive code review of the GAIA repo at the
HEAD of `main` (amd/gaia, commit 211f08c5, tagged release v0.23.1 era).

Repo checkout (already at that commit, detached HEAD):
  C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f

## Hard rules
1. READ-ONLY. Do NOT modify, create, or delete any tracked file. Do NOT run `git checkout`,
   `git stash`, `git reset`, `git commit`, `git pull`, or anything that changes git state.
   Other reviewers share this checkout concurrently.
2. Your ONLY output file is the one named in your task prompt, under
   `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f\.review\`.
   Write it incrementally (append as you go) so partial progress survives if you are interrupted.
3. Python: use `.venv\Scripts\python.exe` in the checkout (an editable install of THIS commit;
   it is being installed in the background — if `import gaia` fails, wait a minute and retry,
   or check `.venv-install.log`). Never use the system python or `C:\Users\14255\Work\gaia\.venv`
   (that one points at a different branch).
   Running unit tests is allowed and encouraged: `.venv\Scripts\python.exe -m pytest tests/unit/<x> -q`.
   Do NOT start servers on port 4001. Do NOT run `gaia eval agent` (needs a Lemonade backend).
4. `gh` CLI is authenticated. Before reporting a finding, check whether it is already tracked:
   `gh issue list -R amd/gaia --state open --search "<keywords>" --limit 5` and note the issue #
   if one exists (a finding that duplicates an open issue is still worth listing, but mark it).
5. Read `CLAUDE.md` and `REVIEW.md` at the repo root first — they define project conventions
   (no silent fallbacks, docs must be updated with code, tests required, etc.) that you review against.
   Skip what REVIEW.md tells you to skip (black/isort formatting, copyright headers, type-hint nags).

## What counts as a finding
A finding must be VERIFIED: you read the actual code (or ran it) and can quote the lines.
Do not report anything you have not confirmed. Unconfirmed suspicions go in a separate
"Hypotheses (unverified)" section at the end, briefly.

Each finding uses this exact template:

### [SEV] Short title
- **Where:** `path/to/file.py:LINE` (+ symbol name)
- **What:** one or two sentences a non-author understands.
- **Failure scenario:** concrete inputs/state -> wrong outcome (crash, data loss, wrong answer, silent degradation, security impact).
- **Evidence:** quoted code lines (short) and/or command output.
- **Fix:** concrete suggestion.
- **Confidence:** High | Medium
- **Tracked:** #NNNN | none found

SEV is one of: 🔴 Critical (security, data loss, breaking API, bug that fires in normal use),
🟡 Important (real bug on an edge path, missing tests for real logic, convention violation a
maintainer would fix before merge, doc that contradicts code), 🟢 Minor.

## Output structure (markdown)
1. `## Scope covered` — the files/dirs you actually read (be honest about what you skipped).
2. `## Findings` — sorted 🔴 → 🟡 → 🟢, template above.
3. `## Test gaps` — modules/paths with no or weak tests; mocks that only prove a call happened, not that the call was valid.
4. `## Documentation gaps` — docs contradicting code, missing docs for shipped features, stale references.
5. `## Improvement opportunities` — refactors / robustness / DX wins, each with a one-line why.
6. `## High-impact feature opportunities` — grounded in what the code/docs/roadmap/open issues show is missing or half-built; say why it matters to users and roughly what it would take.
7. `## Checked and fine` — brief list of things you verified are OK (so the integrator does not re-check).
8. `## Hypotheses (unverified)`.

## Depth expectations
Be thorough, not fast. Read the large files fully (don't skim the first 200 lines). Trace
code paths end-to-end (caller -> callee -> external contract). Prefer fewer, confirmed,
high-value findings over many shallow ones. Cross-check docs vs. code and tests vs. code.
When you run tests, report exact failures with output.
