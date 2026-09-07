# Critique brief — adversarial verification of REPORT.md

You are a skeptical verifier. The draft report is `.review/REPORT.md` (checkout root: `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f`, detached HEAD at amd/gaia main 211f08c5). Raw per-dimension evidence is in `.review/0*.md`. Your job is to find what is WRONG, OVERSTATED, UNDER-EVIDENCED, MIS-RANKED, or MISSING — not to agree.

## Hard rules
- READ-ONLY on the repo: never modify/create/delete tracked files; never run git state-changing commands. Other verifiers share this checkout.
- Write ONLY to the output file named in your prompt, incrementally.
- Python: `.venv\Scripts\python.exe` (editable install of this commit, `[dev,ui]` extras). Probes go in a temp dir OUTSIDE the repo (`%TEMP%`), never in the tree. Do not start servers on port 4001. Do not run `gaia eval agent`.
- `gh` is authenticated for `gh issue list -R amd/gaia --search "..."`.

## Verdicts (use exactly these)
- **CONFIRMED** — you independently re-derived it from source (cite the lines you read) and, where the report says *probe*, you re-ran an equivalent probe or explain why you could not.
- **CONFIRMED-ADJUST** — real, but severity, wording, scope, line numbers, or the proposed fix needs a specific change (state the exact edit).
- **OVERSTATED** — the mechanism exists but the impact/failure scenario is exaggerated or gated by something the report omits (say what).
- **REFUTED** — the claim is false at this commit (show the code that disproves it).
- **UNVERIFIABLE** — could not be checked here (say what is needed).

## Per-finding output template
### <ID> <verdict>
- Evidence: <what you read / ran, with file:line and probe output>
- Required edit to REPORT.md: <exact text change, or "none">
- Severity check: <keep 🔴/🟡/🟢 or change to … because …>
- Fix check: <does the proposed fix work / could it break something / is there a simpler one>
- Tracked check: <issue # found by gh, or "none found">

## Standards
- Prefer running code over reading it when a claim is executable.
- Distrust line numbers: re-open the file and confirm the cited lines contain what the report says.
- For security findings, state the attacker position required (remote page / local process / needs a click / needs auto-approve) and whether the report states it correctly.
- If you find a finding the report MISSED that is at least 🟡 while verifying, add it under a final "## Missed" section with full evidence.
- End with a "## Summary" listing counts per verdict and the 5 most important corrections.
