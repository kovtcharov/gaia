# Final check — `.review/REPORT.md` (post-revision pass)

Scope: correction audit against `_crit_C1-15.md`, `_crit_C16-30.md`, `_crit_I.md`,
`_crit_report.md`; internal consistency; 14 spot re-verifications against source at
`211f08c5`; readability defects. Nothing else.

## 1. Correction audit

**Applied correctly** (checked against the crit text, sampled against source): all C1 edits;
C4 (`<&` refuted, `>&` downgraded to a validator gap, `:192-194`, `-enc` already-blocked made
explicit); C6 precondition; C7 (always-on sandbox, `screenshot_tools.py:29-79`, absolute-path
example, `index_directory:1954-2016`, `hasattr` sequencing, `PermissionError` re-raise);
C10's refuted CORS clause struck + `HostOriginMiddleware`/`token_ok`/`is_exempt_path`;
C11's headline (41 of 92, guard on 24, `agent-server/start` moved to the JSON-gated side, live
tunnel proof, four guard copies, the six previously-unnamed routes); C13 loopback clause;
C14 confidence reword + `\\host\share` form; C15 `:791-802`/`:824-833`; C22 (`:93-95`,
`:154-164`, `:166-178`, `:445-473`, "attacker position: none"); C23 (substring example,
decode wording, Windows/Unix asymmetry); C24 (decorators re-applied, `ast.walk:1038-1042`);
C25 (`.env` example dropped, missing-validator caveat, C31 sequencing); C26 source-only
clarification; C27 (`:1673-1680`, `:1698-1727`, 100-turn scale, corrected doc cites);
C29 `release_agent_gaia.yml:412`; C30 (`bypass.go:89-126`/`:143-154`, C29 timing gate,
`stdio.py:1132`); C31 (`client.py:125-139`, hub-install bound, 11 hosts); C32 5 s join;
C33 (`--base-url` honoured for the pre-flight, `default=None`); C38 (Lemonade-version
rationale, `negotiate.go`); I96 (#3147/#2969, primary CTA correct); the whole I17–I33 block;
I34–I45 (including the I44 split that produced I41 and every `schedule/store.py` re-cite);
I46–I64; I65–I73; C40's tallies and CI wording; I79–I96; every §3.12 correction including the
four refuted sub-claims; all ten "edits that matter most" in `_crit_report.md` except those
listed below; §10 ordering, the Week-1 header, and SNI + `_encode_texts` moved to Week 1.

**Not applied / applied wrongly — ranked:**

1. **Draft C3 (skill-name traversal: `remove_skill(".")` deletes `~/.gaia`) was silently
   demoted from 🔴 to one §4 line.** `_crit_C1-15.md` verdict: *CONFIRMED-ADJUST, "keep 🔴 on
   the data-loss axis"*, plus add "Attacker position: none — a mistyped or copy-pasted name".
   The revision kept only "`remove_skill` docstring says hub-only; deletes any dir" in §4,
   dropped the probe (`remove_skill("..")` → `gaia_home exists: False`), and never landed the
   required citation fix (`cli.py:587-604` → **`src/gaia/skills/cli.py:580-598`**;
   `:186-187`→`:187`; `install.py` rmtree at `:271`). **Appendix E has no row for draft C3** —
   the map jumps C2→C4 — and Appendix A's "0 findings refuted outright … 7 demoted to 🟢" does
   not account for a 🔴→🟢 move. It also contradicts C22, which keeps 🔴 for
   user-misconfiguration data loss on the same axis. *Fix:* restore it as a critical in §2.2
   with the corrected cites and the "attacker position: none; the model-facing wrapper
   rejects" bounds — or, if the demotion is deliberate, record it in Appendix E and in
   Appendix A's counts.

2. **I78 — four required edits unapplied, one of which leaves a refuted claim standing.**
   - "(git-ignored, one file for all categories)" is **false**: `eval/results/.gitignore` ends
     `!baseline.json` and `git ls-files` lists `eval/results/baseline.json` (verified).
     Replace with "a **tracked** file, so `--save-baseline` silently dirties committed state".
   - "nothing reads/writes `tests/fixtures/eval_baselines/`" is false as written: no code in
     `src/gaia` does, but `test_eval_agent_gemma_consolidation.yml:168,192` and
     `test_eval_rag.yml:74` name it (verified). Reword to "no code in `src/gaia` reads or
     writes …".
   - "**71** of 91 scenarios" → **74** (recounted at HEAD: 91 total; baselined categories are
     rag_quality 7 + tool_selection 5 + context_retention 4 = 16).
   - Missing clause: the judge-mismatch banner's own remedy (`--save-baseline`) writes to a
     path the documented workflow never reads.
   *(Declining the crit's "split `test_eval_rag.yml` out as a 🔴" is the right call and should
   stay — that workflow's header says it is FULLY DISABLED pending #1315, which the critic
   missed. §6.3's "disabled" wording is accurate.)*

3. **I98 — three citation corrections unapplied, all wrong at HEAD.**
   `ui/server.py:112-115` → the `tunnel.active` gate is **`:126-128`** (`:112` is
   `async def dispatch`); `tunnel.py:301-308` → the `active` property is **`:303-310`**;
   `tunnel.py:407-417`, cited for "0.5 s polls, up to 15 s", is the `Popen` block — the poll
   loop is **`_poll_ngrok_api` `:589-608`, called at `:420`**. (`_kill_stale_ngrok`
   `:520-535` / `:391` and `ui/server.py:576` are correct.)

4. **I1's size-cap correction was applied inverted.** Verified: `filesystem_tools.py:27
   MAX_READ_BYTES = 50 * 1024 * 1024`, and `file_io_tools.py` has **no** read cap at all. The
   report says the losing filesystem `read_file` is "10 MB-capped" and the winning
   `file_io.read_file` has a "50 MB cap" — both halves wrong. Rewrite: "the sandboxed,
   **50 MB-capped** filesystem `read_file` is replaced by `file_io.read_file`, which has **no
   size cap** and dereferences `path_validator` unguarded (#3316)".

5. **I77 — the required tightening is missing and the declined promotion is unexplained.**
   Crit: replace "a fixer timeout is swallowed into the fix log" with "the timeout is returned
   as `{"error": …}` and the caller (`runner.py:1960`) never inspects the result, so the loop
   re-evaluates a tree Claude Code was killed halfway through editing"; add that
   `eval/prompts/fixer.md`'s "do NOT commit" is the only restraint and names no file
   allowlist. The 🟡→🔴 proposal was also declined silently — defensible, but Appendix A
   advertises "13 re-ranked upward" without saying which proposals were rejected.

6. **I76 — "CLAUDE.md's serial rule relies on it" was marked OVERSTATED and survives
   verbatim.** The rule's stated enforcement is a human `ps aux | grep` (itself POSIX-only),
   so the honest line is "the rule is unenforced on Windows both ways". Also add the contrast
   the crit supplied: the sibling read-only-`/tmp` branch (`runner.py:114-122`) *does* warn.

7. **Smaller citation fixes requested and not made (each re-checked at HEAD):**
   - I6: `EXTRACTION_TIMEOUT_S = 8` is `memory.py:`**`189`**, not `:187` (also in the I89 table row).
   - I12: `DEFAULT_MAX_STEPS = 50` is `agent.py:`**`105`**, not `:113`.
   - I4: `file_tools.py:2733-2762` → `:2598` (def) + `:2732-2762`, and quote the "Return all
     files …" comment — it makes this a design decision, not an oversight.
   - I45: the bare-"…" truncation is `sse_translation.py:`**`643-644`** (`INVOCATION_TOTAL_CHARS`
     at `:618`), not `:640-641`.
   - §4 `scan_all` bullet: `discovery.py:3899` → **`:3872-3877`** (the silent filter itself).
   - C10: `POST /v1/<id>/init` is **`:595-625`** (`agent_provision`), not `:565-620`; the CORS
     block starts at `app.add_middleware` `:441`, so `:441-447` rather than `:442-446`.
   - C30: the wrong cite (`client.go:965-970`) was deleted rather than replaced, so the merged
     history-loss clause now carries no citation. Restore `client.go:32-37`, `sse.go:793,856`,
     `chat/model.go:1298-1300`.
   - I97: crit point 2 — the file's own `:82` "never execute code from the PR" invariant vs
     `auto-fix`'s `pip install` at `:1057-1062` — was dropped; point 1 landed.

## 2. Internal consistency

Checks that pass: **40 C-IDs and 98 I-IDs**, gap-free; raw totals 34/153/82 and every
Appendix B per-file row match the recount; **exactly 36 findings carry a `*probe*` marker**,
matching §0 and Appendix A; ≈120 minor; ~150 in Appendix C; 698 open issues in all four
places. Appendix E's draft→final map is otherwise complete, and every cross-reference in
§5–§10 resolves to the correct renumbered finding.

Defects:

8. **§1 contradicts Appendix A on the verification outcome.** §1: "one sub-claim was refuted
   …, two were overstated and re-ranked, seven line citations were corrected". Appendix A:
   "2 sub-claims refuted … 3 overstated and re-ranked … ~40 citation, count and wording
   corrections". Appendix A is right; align §1.

9. **"Five new criticals" is six.** §1 and Appendix A both say five; Appendix B's row says
   6 🔴 / 9 🟡, Appendix E lists six new criticals (M1→C3, M2→C19, M3→C8, M4→C11, M5→C17,
   M-A→C12), and Appendix A's own file list names six files. Change both to six.

10. **The tracked-criticals list is wrong in one slot.** §1: "Ten criticals carry or name an
    open issue (#124, #630, #1128, #1344, #1460, #2474, #2768, #2951, #3101, #3316)".
    **#3101 belongs to I96**, a demoted finding — no critical cites it. C19 (#2803) and C18
    (#3239 / #2062) are criticals that do. Swap #3101 → #2803; the count "ten" then holds
    (C4, C7, C10, C13, C18, C19, C25, C29, C33, C38).

11. **"≈270 Windows-red tests" contradicts C40's own tallies.** C40 lists 33 F/217 E +
    213 F/42 E + 124 E + 53 F + ~157 F; the verifier's own figure is **~640** after overlap.
    "≈270" appears three times (§1 structural themes, §1 theme 10, §5 item 1). Use ~640, or
    write "≈640 across the sampled trees (overlapping)".

12. **Stale range `C30–C30` twice** — §8 item 7 and §9 A8; both mean **C29–C30**.

13. **Six `I-series` placeholders left where the ID now exists:** C1 → **I14**; C8 → **I13**;
    C11 → **I34**; C18 → **I39**; C38 → **I80**; C30's trailing "— I-series" has no target
    (the history loss is now inside C30) → delete it.

14. **§4 carries 9 ↓ marks against Appendix A's "7 demoted to 🟢".** The nine are the seven
    full demotions plus two partials (I59's CHANGELOG/regex half, the `@amd-gaia/gaia`
    CHANGELOG line). Say "7 demoted plus 2 partial", or drop ↓ from the partials.

15. **I97 and I98 sit out of numeric sequence** (I97 between I81 and I82; I98 after I48) with
    no explanation. One clause in Appendix E — "demoted criticals keep tail numbers and are
    filed by topic" — prevents a "typo?" reaction.

## 3. Spot re-verification (14 findings re-opened at the cited lines)

CONFIRMED as written: **C7** (`screenshot_tools.py` is 96 lines, so `:29-79` is valid);
**C8** (`_chat_helpers.py:926-940` is exactly `_compute_allowed_paths` granting parent dirs;
`_prompt_overwrite` `:623-640` auto-approves when non-interactive — only nit:
`is_write_blocked` is reached at `:588`, not `:584`); **C11** (`email_sidecar_router` included
unconditionally at `server.py:638-640`); **C12** (`ui/server.py:549-563`; regex `:559`,
`allow_credentials=True` `:560`, `allow_headers=["*"]` `:562` — exact); **C16** (all five
cites exact: `memory_store.py:105-118` invariant text, `store()` `:751`, `memory.py:1442`
unchecked `update` op, `VALID_CATEGORIES` gate `:2550`, `routers/memory.py:1557`/`:1584`);
**C19** (`mcp_client.cpp` `connect()` `:547`, `quoteArg` `:551-556`, `execl("/bin/sh"…)`
`:448`); **C36** (`_chat_helpers.py:494` `silent_mode`, `security.py:462` `input(...)`);
**C37** (`logger.py:50` `Path.home()`, module-scope `log_manager` `:281`); **C39**
(`publish.yml:290` `… || echo "Backend tests skipped …"`, verbatim); **I13**
(`security.py:271` `_load_persisted_paths()` called unconditionally in `__init__`); **I36**
(`index.py:709` raw `files_fts MATCH :name`); **I50** (`rag/sdk.py:606-612`, unchecked
`response.get("data", [])` extend).

Needs edit: **C10** and **I45** (item 7), **I1**, **I78**, **I98** (items 2–4).

## 4. Readability

No paragraph in §1–§3 is internally contradictory beyond the numeric defects above. Two real
reading defects: the six `I-series` placeholders (13) and `C30–C30` (12) — both stop a reader
mid-sentence. One imprecise claim: §8 item 1 lists **I4** among what a `FileAccessPolicy`
closes, but I4 is an output-cap/context-overflow finding, not a path-check one; drop it or
write "the listing-tool half of I4". The crit's request to table **I43** and **I88** (as was
done for I89) was not carried out — cosmetic, not blocking.

## Verdict

Not ship-as-is. Apply, in order:

1. Restore or explicitly account for draft C3 (skill-name traversal) — Appendix E row plus a stated severity decision.
2. I78: strike "git-ignored" (the baseline is tracked), fix "nothing reads/writes `tests/fixtures/eval_baselines/`", 71 → 74.
3. I98: three citation fixes (`ui/server.py:126-128`, `tunnel.py:303-310`, `_poll_ngrok_api:589-608` / `:420`).
4. I1: un-invert the read-file caps (50 MB loser, uncapped winner).
5. §1 ↔ Appendix A: refuted / overstated / correction counts; "five new criticals" → six; #3101 → #2803.
6. "≈270" Windows-red tests → ~640 (three places).
7. `C30–C30` → `C29–C30` (×2); the six `I-series` → I14 / I13 / I34 / I39 / I80 / delete.
8. I77 timeout clause; I76 "relies on it" reword.
9. The seven small citation fixes in item 7 (I6, I12, I4, I45, §4 `scan_all`, C10, C30) plus the I97 `:82` clause.
10. Optional: the ↓ / demotion-count wording, an I97–I98 ordering note, and §8.1's I4 reference.

Everything else in the revision is faithful to the four critique files and to source.
