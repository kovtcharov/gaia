# Adversarial verification — findings C16–C30 (§2 of REPORT.md)

Verifier scope: C16–C30. Checkout `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f`
(detached HEAD, amd/gaia main 211f08c5). Repo untouched; every probe written to
`%TEMP%\critC1630`. Python = the worktree `.venv`. `gh` used for dedup.

**Headline:** all 15 findings are real. 8 CONFIRMED outright, 7 CONFIRMED-ADJUST
(citations, gating, or an interaction the report omits). Nothing refuted. The two
biggest corrections are (a) three of the report's file:line cites do not exist at
those lines (C21, C22, C23), and (b) C22/C23 are *gated on* C21 — on the released
one-file binary the respawn they describe is deferred by one message, which the
report never says. My own C24 probe is **worse** than the report's: 9 of 11 hosts
fail, including `example.com`.

---

### C16 CONFIRMED-ADJUST

- **Evidence:**
  - Re-opened `src/gaia/cli.py`. `kill_process_by_port` is at **4432–4543**. The
    Windows branch is **4441** (`subprocess.check_output(["netstat","-ano"]).decode()`),
    **4446** (`if f":{port}" in line` — bare substring, no `LISTENING` filter, no
    column parse), **4450** (`pid = int(parts[-1])`), **4452–4456** (`taskkill /F`).
    The Unix branch is **4471–4483** (`lsof -ti:{port}` → `kill -9` over *every*
    returned pid). Cited lines are accurate.
  - Callers confirmed: `gaia kill --lemonade` fallback (**3510**, **3518**, port
    13305), `gaia kill --port` (**3526**), `gaia api stop` (**4925**).
  - **Live probe** (`p16.py`, real `netstat -ano` on this box):

    ```
    --- port 80: 2 matching lines
        TCP  192.168.1.178:51799  192.168.1.48:8009  ESTABLISHED  18040
        TCP  192.168.1.178:52576  192.168.1.53:8009  ESTABLISHED  18040
    --- port 443: 150 matching lines (all ESTABLISHED client sockets)
    ```

    `gaia kill --port 80` on this machine would `taskkill /F` PID 18040 — an
    unrelated app with an outbound connection to a **:8009** foreign address.
    The report predicted `:80` matching `:8080`; the live failure is `:8009`,
    same class. The foreign-address / client-kill claim is reproduced.
- **Required edit to REPORT.md:** three wording fixes.
  1. `:80` matches `:8080` → `":80" substring-matches ":8009" and 149 other rows on
     this machine`, and note the match is against the **whole netstat line**, so
     foreign addresses and TIME_WAIT rows count.
  2. `UnicodeDecodeError swallowed` is wrong — it is caught by the function-level
     `except Exception` at **4539–4543** and returned as
     `"Error killing process on port N: …"`. Change to *"surfaces as a generic
     `Error killing process on port N: 'utf-8' codec can't decode…` instead of a
     port-not-found result, so `gaia api stop` exits 1 for an unrelated reason."*
  3. Add the asymmetry the report misses: **Windows returns after the first kill**
     (4457–4460), so it kills exactly one wrong process; **Unix kills every pid
     `lsof` returns**, i.e. both ends of every connection. The "kills the Agent UI
     backend, daemon, and any `gaia chat`" consequence is Unix-only.
- **Severity check:** keep 🔴. Data-loss-adjacent (kills a user's unrelated process
  without confirmation) and fires in normal use — `gaia api stop` with the default
  port on a busy machine is enough.
- **Fix check:** the proposed fix is right and minimal; `lsof -nP -iTCP:{port}
  -sTCP:LISTEN -t` is exactly the correct incantation. One addition the report's fix
  text omits: the Unix `netstat -tulpn` **fallback** at **4501–4525** has the
  identical unanchored `f":{port}" in line` bug and must be fixed in the same change.
  Also worth requiring the process **name** to match a GAIA/Lemonade allowlist before
  `taskkill`, as `lemonade_embedded._daemon_alive` does.
- **Tracked check:** **#789** (CLOSED) — *"Security: kill_process_on_port uses
  shell=True with unvalidated port interpolation"*. Same function, different bug; the
  `shell=True` half was fixed, the matching half was not. Change `Tracked: none` →
  `Tracked: none (#789 fixed the shell=True half of this same function and left the
  matching logic)`. That is useful signal: this code was reviewed and the port
  matching was never questioned.

---

### C17 CONFIRMED

- **Evidence:** `src/gaia/agents/tools/file_io_tools.py` re-read.
  **1052** `start_line = function_node.lineno - 1` (the `def` line — **not** the
  decorator), **1054–1067** the "next `def`/`class` at same indent" scan,
  **1080–1082** `lines[:start_line] + [new] + lines[end_line:]`, **1086–1099** the
  syntax check (which passes, so the tool reports success), **1102–1103** the write.
  Backup at **1069–1077** only writes `.bak` when `backup=True`.
  - **Probe** (`p17.py`, replaying the exact algorithm on
    `def foo / CONSTANT = 42 / @decorator / def bar`):

    ```
    start_line 0 end_line 6
    def foo():
        return 99

    def bar():
        return 2

    CONSTANT present: False | @decorator present: False
    syntax valid -> tool reports success
    ```

    Independently reproduced.
- **Required edit to REPORT.md:** none required, but two free upgrades worth adding in
  one clause: (a) because `start_line` is the `def` line, the **target's own
  decorators are left behind** and silently re-applied to the replacement — a
  semantic change, not just a deletion; (b) `ast.walk` at **1038–1042** takes the
  *first* function of that name anywhere in the module, so `replace_function("run")`
  on a file with a nested or method-level `run` rewrites the wrong one.
- **Severity check:** keep 🔴. Silent code deletion reported as `success` is textbook
  data loss.
- **Fix check:** AST spans are correct and simpler than the scan. Precise form:
  `start = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1`
  and `end = node.end_lineno` (Python ≥3.8, satisfied). Note this *changes* behaviour
  for a decorated target — the decorators now get replaced too. That is correct, but
  call it out so it is not read as a pure bugfix.
- **Tracked check:** none found. Nearest is **#955** (CLOSED, *"CodeAgent write tools
  missing blocklist/guardrails"*) — different concern (path guardrails, not span
  computation). `Tracked: none` stands.

---

### C18 CONFIRMED-ADJUST

- **Evidence:**
  - `src/gaia/web/client.py` **770–785** filename from `Content-Disposition`
    (`re.search(r'filename[*]?=["\']?([^"\';]+)', cd)`), **788** `_sanitize_filename`,
    **793** `save_path = save_dir / filename`, **813** `with open(save_path, "wb")` —
    **no `exists()` check anywhere between 788 and 813**. Confirmed.
  - `src/gaia/agents/tools/browser_tools.py` **294–305**: `is_write_blocked(saved_path)`
    runs *after* `mixin._web_client.download(...)` at **277–281**, and reacts with
    `Path(saved_path).unlink(missing_ok=True)` at **299**. Confirmed.
  - **Probe** (`p18.py`, real `WebClient._sanitize_filename`):

    ```
    'report.pdf'       -> 'report.pdf'
    'credentials.json' -> 'credentials.json'
    '../../evil.txt'   -> 'evil.txt'
    '.env'             -> '_.env'
    ```

    The report's two examples behave as claimed.
- **Required edit to REPORT.md:** three.
  1. Never use `.env` as the example anywhere in this finding —
     `_sanitize_filename` prefixes leading dots (`.env` → `_.env`).
     `credentials.json` is correct and the report already uses it.
  2. Add the reachability caveat: the post-write `unlink` only fires when a
     `_path_validator` is attached (`hasattr(mixin, "_path_validator") and
     mixin._path_validator`). Without one — any custom agent composing
     `BrowserToolsMixin` — the file is silently **overwritten and kept**, which is the
     worse half. The report attributes both halves to the same code path.
  3. Add the C24 interaction: today most HTTPS downloads never reach the
     `open(...,"wb")` because `PinnedIPAdapter` fails the handshake (C24). Fixing C24
     **un-gates** C18, so C18 must land in or before the C24 change, not after.
- **Severity check:** keep 🔴, but the justification is the *overwrite*, not the
  delete. State the attacker position, which the report omits: a **remote page the
  model was asked to download from** — no click needed once the model calls
  `download_file`.
- **Fix check:** proposed fix is right. One trap: `.part` + atomic rename must run the
  blocklist check **before** creating `.part`, or a `credentials.json.part` is still
  created in a sensitive directory. Safest concrete form: resolve → blocklist →
  `exists()` → refuse; then download to `save_dir/f"{uuid4().hex}.part"` and
  `os.replace`.
- **Tracked check:** none. **#1460** (OPEN, *"EPIC: L2 Web Search & Extraction"*) is
  the umbrella the report cites and is correctly labelled as such.

---

### C19 CONFIRMED

- **Evidence:**
  - `src/gaia/api/openai_server.py` **364** `agent.process_query(user_message,
    workspace_root=workspace_root)` (non-streaming) and **496**
    `lambda: agent.process_query(query, workspace_root=workspace_root)` (streaming).
    Both cites exact. The kwarg is passed **unconditionally**, even when
    `extract_workspace_root` returned `None` (**311**) — so it is not gated on the
    caller sending a workspace hint.
  - The only `try/except` in the handler (**336–339**) wraps `registry.get_agent` and
    catches `ValueError` only. Nothing wraps line 364 → the `TypeError` becomes a 500.
  - `AGENT_MODELS` (`api/agent_registry.py` **35–47**) has exactly one entry:
    `"gaia" → gaia_agent.agent.GaiaAgent`. That is the only model the endpoint serves.
  - **Override audit** (`grep -rn "def process_query" src/ hub/`): outside tests there
    are exactly **three** definitions — `agents/base/agent.py:4442`,
    `agents/base/memory.py:2183`, and the email agent
    (`hub/agents/email/.../agent.py:1221`, not reachable from this registry).
    **No `ChatAgent` or `GaiaAgent` override exists.** Confirmed as asked.
  - **MRO probe** (`p19b.py`):

    ```
    MRO: GaiaAgent, ChatAgent, MemoryMixin, ProceduralMemoryMixin, Agent, ABC, ...
    resolved process_query from: MemoryMixin.process_query  (src/gaia/agents/base/memory.py)
    sig: (self, user_input, **kwargs)
    BIND FAIL base: got an unexpected keyword argument 'workspace_root'
    ```

    `MemoryMixin.process_query` forwards verbatim:
    `return super().process_query(augmented, **kwargs)` (**memory.py:2208**).
    Confirmed as asked.
  - **Runtime probe** (`p19c.py` — a real `class Probe(MemoryMixin, Agent)`, not a
    signature simulation):

    ```
    TypeError: Agent.process_query() got an unexpected keyword argument 'workspace_root'
    ```

  - Test-blindness confirmed: `tests/test_api.py:71` and `:124` both do
    `fake_agent = mocker.MagicMock()` + `mocker.patch.object(server_registry,
    "get_agent", ...)`, and `:100–102` assert only *that it was called* with the right
    positional arg. Exactly the "mock proves we called it" trap CLAUDE.md names.
- **Required edit to REPORT.md:** one clarification. In a **source-only checkout** the
  failure surfaces earlier and differently: `gaia_agent` is not importable, so
  `get_agent` raises `ImportError` and the caller gets a 404 with an install hint. The
  `TypeError` is what a user with the flagship installed (the supported path) hits.
  Either way `/v1/chat/completions` never answers, but the report's flat "crashes"
  should name which install state produces which failure.
- **Severity check:** keep 🔴. A public REST surface returning 500 on 100% of requests
  is as blocking as it gets.
- **Fix check:** deleting the kwarg at 364 and 496 is correct and complete —
  `workspace_root` has no other consumer (only 115, 311–313, 349, 364, 434, 446, 496).
  Also delete the now-dead `extract_workspace_root` and the `workspace_root` parameter
  on `create_sse_stream`, or the next reader re-adds the call. The proposed regression
  test (a real `Agent` subclass + stubbed LLM) is the right shape.
- **Tracked check:** none found. Adjacent: **#3254** (OPEN) *"`--step-through` is a
  dead flag — it sets an env var nothing reads"* — same `gaia api` surface, same class
  of untested plumbing. Worth citing.

---

### C20 CONFIRMED-ADJUST (understated, not wrong)

- **Evidence:** every claim re-derived, plus a **fresh probe I wrote against a real
  `MemoryStore` with a 25-turn session**, as required.
  - `memory.py:1628` `turns = store.get_history(session_id, limit=20)`; **1734–1736**
    `store.mark_turns_consolidated(turn_ids)` over only those ids.
  - `memory_store.get_history` is `SELECT … ORDER BY id DESC LIMIT ?` re-sorted ASC —
    the **last** 20, confirming the report (and contradicting the spec doc, below).
  - `memory_store.get_unconsolidated_sessions` eligibility SQL:
    `HAVING COUNT(*)>=? AND MAX(timestamp)<? AND SUM(CASE WHEN consolidated_at IS NULL
    THEN 1 ELSE 0 END) > 0` — one unconsolidated turn keeps the session eligible
    forever. `ORDER BY MAX(timestamp) ASC LIMIT ?` → the same oldest sessions occupy
    all 5 slots on every run.
  - `memory.py:590` sets `_memory_post_init_pending = True` (steps 6–7 **deferred** to
    first query); `memory.py:593` runs `self._memory_store.prune()` (step 8) **at
    init**. `MemoryStore.prune(days=90)` does
    `DELETE FROM conversations WHERE timestamp < ?` with **no `consolidated_at`
    condition**, so unconsolidated turns are hard-deleted.
  - **My probe** (`p20.py`, fresh `MemoryStore`, 25 turns aged 30 days):

    ```
    BEFORE: eligible=['sess-25'] total_turns=25 unconsolidated=25
    get_history(limit=20) returned 20 turns, ids 6..25 ('turn 6 content' .. 'turn 25 content')
    mark_turns_consolidated -> 20 rows
    AFTER : eligible=['sess-25'] total_turns=25 unconsolidated=5
    2nd pass get_history ids: 6 .. 25 | already-consolidated in that window: 20
    mark 2nd pass -> 0 rows newly marked
    AFTER2: eligible=['sess-25'] total_turns=25 unconsolidated=5
    ```

    Turns **1–5 are never reachable**. The session is still eligible after two full
    passes and makes **zero** further progress. Exactly as reported.
- **Required edit to REPORT.md:** the finding is *understated* on two points; add both,
  and fix one doc cite.
  1. **Every re-run stores a fresh duplicate.** The re-summarised window still reaches
     `store.store(category="note", source="consolidation", …)` (**1673–1680**) and the
     extracted-knowledge loop (**1698–1727**) on every startup. The cost is not just a
     wasted LLM call — the knowledge table accrues a near-duplicate summary note plus
     duplicate items per boot, forever, for each stuck session.
  2. **Scale the loss correctly.** For a 100-turn session only the last 20 are ever
     distilled; turns 1–80 are hard-deleted at day 90. "Pruned undistilled" without
     the ratio undersells it.
  3. Tighten the doc-contradiction cite. `docs/spec/agent-memory-architecture.md` at
     **~700** actually *documents* the 20-turn fetch, so "contradicts :700" reads
     wrong. The real contradictions are: that doc says the fetch is **"(oldest
     first)"** (the code takes the newest 20); that **"`consolidated_at` prevents
     re-processing"** (it does not, for >20-turn sessions); and at **~401** that
     *"Conversations are consolidated before the 90-day prune"* (prune runs at init,
     consolidation is deferred to first query).
- **Severity check:** keep 🔴 — silent hard-delete of user conversation data plus a
  per-boot LLM cost that never terminates. It is gated on memory being enabled and a
  >20-turn session older than 14 days, which is ordinary for a real user; state that
  gate in the finding so a reader can judge it.
- **Fix check:** the proposed `mark_session_consolidated(session_id)` is the obvious
  one-liner but is **not sufficient and is slightly dangerous**: marking all 25 turns
  while summarising only 20 silently discards turns 1–5 from the summary — trading a
  loud bug for a silent one. The honest fix is windowed consolidation keyed on the
  *oldest unconsolidated* turn, then mark. Add: `prune()` (**memory.py:593**) must move
  after `_run_memory_post_init`, or a session that never gets a query still loses its
  turns regardless.
- **Tracked check:** none found. Adjacent memory bugs exist (**#2865**, **#2676**, both
  OPEN) but neither covers consolidation coverage. `Tracked: none` stands.

---

### C21 CONFIRMED

- **Evidence:** both halves independently reproduced.
  - **Empirical half — reproduced on this machine** against the *installed*
    `C:\Users\14255\AppData\Local\Programs\GAIA\gaia-agent.exe` (49 MB, one-file).
    Probe `p21.py` (spawned with `CREATE_NO_WINDOW`, cleaned up afterwards):

    ```
    parent pid: 41224
    children of parent before kill: [{"ProcessId":24348,"Name":"conhost.exe"},
                                     {"ProcessId":49100,"Name":"gaia-agent.exe"}]
    parent poll after kill: 1
      child 49100 (gaia-agent.exe) 2s after parent kill: ALIVE
    stdout read returned within 8s: False | bytes: None
    after closing stdin, stdout read returned: True | bytes: 245
    cleanup done
    ```

    Post-run `Get-Process gaia-agent` → empty; nothing left behind. So
    `Process.Kill()` kills the bootloader only, the real Python child survives, and
    **stdout never EOFs** — it only unblocks once stdin is closed. Exactly the
    report's claim, verified independently.
  - One-file confirmed at the build, not just the header comment:
    `.github/workflows/release_agent_gaia.yml:412` →
    `"$PY" "${PACKAGING}/freeze.py" --onefile`.
  - **Code half**, all re-derived:
    - `subprocess.go:222-228` — the cancel goroutine calls only `st.proc.kill()`.
    - `procHandle.kill()` (**69–73**) is `cmd.Process.Kill()` — no process group, no
      Job Object.
    - The reader goroutine's reset is a `defer` (**238–246**) that only runs when the
      goroutine **exits**; it is parked in `st.scanner.Scan()` (**263**), which the
      probe shows does not return. So `resetDeadChild` → `discard` →
      `s.started=false` (**513–524**) is never reached: **`s.started` stays true**,
      confirming the report against the obvious counter-reading of `discard`.
    - Next `Send` → `startLocked` (**135–140**) sees `s.started` and hands back the
      *same* `stdin/stdout/proc`, writing into the orphan.
    - `stdio.py:661-668` queues the new line behind the running turn
      (`queries.put(parse_query(line))`); `apply_control` (re-read in full) supports
      only `bypass` and `tool_decision` — **there is no `cancel` verb**.
      `state.cancel_active()` exists but is called only at **682**, on stdin close.
    - When the orphan finally emits, the old reader wakes, `emit` returns false
      (**251–261**, **297–299**) → `return` → deferred `resetDeadChild` →
      `proc.reap()` → Go's `cmd.Wait()` closes the pipes **under turn 2's reader**
      (two goroutines on one `*bufio.Scanner`) → `scanner.Err()` at **328–332** emits
      `"agent stdout read error: …"`. Chain confirmed end to end.
    - `tui/test/mockagent/main.go` is 169 lines with **no** `exec.Command` /
      `os.StartProcess` — single-process, so no existing test can observe this.
      Confirmed.
- **Required edit to REPORT.md:** change the freeze citation from
  `release_agent_gaia.yml:10-11` (prose header) to **`:412`** (the actual `--onefile`
  flag). Nothing else — this is the best-evidenced finding in the section.
- **Severity check:** keep 🔴. The "Esc on `delete all *.tmp under ~/proj` still
  deletes" scenario is exactly right: cancel is a safety control, and it does not
  cancel.
- **Fix check:** the proposed fix is correct and correctly ordered (cooperative cancel
  first, tree-kill as backstop). Two additions: (a) a `gaia_control: cancel` verb must
  be handled on the **pump thread** (as `tool_decision` already is) or it queues
  behind the very turn it is cancelling — the same trap; (b) tree-kill alone is not
  enough on Windows without a Job Object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, since `taskkill /T` races the orphan.
  Shipping `--onedir` also fixes it and is the cheapest interim mitigation.
- **Tracked check:** none. **#2917** (OPEN) — *"a cancelled free-text turn holds its
  session lock until generation finishes, so a resend conflicts"* — is the same
  failure shape on the daemon/email transport and is worth citing as corroboration
  that cancel is systematically non-cooperative. **#2474** is correctly identified as
  different.

---

### C22 CONFIRMED-ADJUST

- **Evidence:**
  - Respawn-from-original-argv confirmed: `subprocess.go:145`
    `cmd := exec.Command(s.path, s.args...)`, and `s.args` is never mutated
    (`grep s.args` → only 145, 463, 475).
  - `BypassAtLaunch()` (**458–468**) scans argv for `--bypass-permissions`; the flag
    is forwarded to the child, so a respawn re-arms bypass **in the agent**.
  - `bypass.go` re-read in full (169 lines): `setBypass` (**89–126**) sends
    `SetBypassPermissions` and sets `m.bypassPermissions`; `applyLaunchBypass`
    (**143–154**) runs **once**, at startup. **Nothing re-sends permission state after
    a respawn** — `grep -rn "SetBypassPermissions" tui/internal` shows call sites only
    at `bypass.go:102` and the test double.
  - Net effect confirmed: after respawn the **agent** is bypassing again while the
    **UI** still holds `bypassPermissions == false` and shows no banner — the agent is
    more permissive than the UI claims, the dangerous direction.
  - The model-revert contrast is real and correctly cited: `chat/model.go:283-289`
    (the comment) and `chat/canonical.go:44-57` (the actual
    `"[!] the agent process restarted and reverted to …"` message). Note it only fires
    `if m.modelID != "" && e.ModelID != m.modelID && !m.awaitingModelSwitch` — a user
    who never ran `/model` gets **no signal at all** that a respawn happened.
- **Required edit to REPORT.md:** three, one of which is a factual error.
  1. **Wrong citation.** `bypass.go:2138-2175` does not exist —
     `tui/internal/ui/chat/bypass.go` is **169 lines**. Replace with
     `bypass.go:89-126` (`setBypass`) and `bypass.go:143-154` (`applyLaunchBypass`,
     the once-only launch reflection). Line 2138 appears to be a stray from
     `chat/model.go` (2730 lines), which has unrelated rendering code there.
  2. **Add the C21 gate.** The scenario `--bypass-permissions` → `/bypass off` → Esc →
     next message runs unprompted does **not** happen on the *released one-file
     binary*: C21 proves the respawn is deferred, so the next message goes to the
     surviving orphan, which still has bypass **off**. The revert lands one message
     later, after the orphan's stale output trips `resetDeadChild`. On a source /
     `--onedir` install (where `kill()` really kills the child) it is immediate. State
     both, or a maintainer will run the repro on the shipped binary, not see it, and
     close the finding.
  3. Minor: "loaded skills / 'always' grants / `/model` switch are lost" is correct —
     all three live in the child's `PermissionState` / in-process skill registry — but
     is unevidenced in the text. One cite (`stdio.py`'s `PermissionState`) carries it.
- **Severity check:** keep 🔴 🔒. Silently re-arming a permission bypass the user
  explicitly turned off is a security-control failure, not a UX bug. Attacker position:
  none needed — the user's own agent plus a prompt-injected or merely wrong tool call.
- **Fix check:** "carry `bypass` in the model-state ping and warn/resync like the model
  revert" is the right shape and reuses existing plumbing, but is **not sufficient
  alone**: between argv re-exec and the ping arriving, the child is already in bypass.
  Safer primary fix — stop passing `--bypass-permissions` to *respawned* children
  entirely (bypass becomes purely a runtime control message), so a respawn always fails
  closed. Keep the resync as the mechanism that restores the user's real intent.
- **Tracked check:** none found (`gh` searches for `bypass respawn`, `TUI Esc cancel
  agent` return nothing on point). `Tracked: none` stands.

---

### C23 CONFIRMED-ADJUST

- **Evidence:**
  - `conversation_history` is child-owned:
    `hub/agents/gaia/python/gaia_agent/stdio.py` **743–771** (`_record_turn` appends to
    `agent.conversation_history`, trimmed to `MAX_HISTORY_TURNS = 12` pairs at
    **740**). The docstring at **746–750** states the base `Agent` never appends, so
    this transport is the *only* holder. A respawned child starts with an empty list.
  - Host-owned transcript exists only for SSE: `TranscriptResetter` is declared at
    `tui/internal/client/client.go:32-37`; the only implementation is `sse.go:793`
    with the compile-time assertion
    `var _ TranscriptResetter = (*SSEClient)(nil)` at `sse.go:856`. `SubprocessClient`
    does not implement it. The only call site is `chat/model.go:1298-1300` (`/clear`).
  - So on a subprocess respawn the on-screen transcript is unchanged while the agent's
    prompt context is empty — the report's claim, confirmed.
- **Required edit to REPORT.md:** three.
  1. **Wrong citation.** `client.go:965-970` does not exist —
     `tui/internal/client/client.go` is **114 lines**. Replace with `client.go:32-37`
     (the interface), `sse.go:793,856` (the only implementation), and
     `chat/model.go:1298-1300` (the only call site).
  2. Add the C21 gate, same wording as C22: on the shipped one-file binary the history
     loss lands one message after the Esc, not immediately.
  3. Soften "the whole conversation history". For the flagship, `MemoryMixin` is also
     recording turns into SQLite and injecting recalled context, so the agent is not
     fully amnesiac. What is lost is the **verbatim 12-pair prompt window** that
     carries pronoun and back-reference resolution. Still a real and confusing loss —
     but say it precisely, so the finding survives a "but we have memory" rebuttal.
- **Severity check:** **downgrade 🔴 → 🟡.** REVIEW.md reserves 🔴 for "security,
  breaking changes, data-loss risk, or a bug that will fire in normal use". Nothing is
  destroyed: the SQLite turn log survives and the visible transcript survives. The user
  gets a confused answer and can restate. Next to C21 (a cancel that does not cancel a
  destructive tool call) and C22 (a permission bypass silently re-armed), this is a
  different tier. Keep it in §2 prose as a consequence of C21 if preferred, but as a
  standalone finding 🟡 is the honest call.
- **Fix check:** "cooperative cancel (C21)" subsumes this entirely — if the turn is
  cancelled in-process there is no respawn and no loss, so C23 needs no independent fix
  once C21 lands. The "agent restarted — earlier context lost" line is the right
  interim and costs nothing. A host-owned transcript for the subprocess transport is a
  much larger change (it would have to push context every turn like `SSEClient` does)
  and should not be listed as if it were cheap.
- **Tracked check:** none found.

---

### C24 CONFIRMED

- **Evidence:** re-derived and **re-probed live**; my numbers are worse than the
  report's.
  - `src/gaia/web/client.py` **202–245** (`PinnedIPAdapter.send`): **228**
    `new_netloc = f"{host}@{url_ip}:{port}"` — the hostname goes into **userinfo**,
    which urllib3 never uses for SNI; **242** `request.url = new_url`. **247–264**
    (`get_connection` / `get_connection_with_tls_context`) set only
    `pool.assert_hostname`, never `server_hostname`. Cited range accurate.
  - **Live probe** (`p24.py`, `WebClient` vs plain `requests`, same process, same box):

    ```
    URL                              WebClient      plain requests
    https://github.com/amd/gaia      SSLError       OK 200
    https://amd-gaia.ai/             SSLError       OK 200
    https://pypi.org/                SSLError       OK 200
    https://stackoverflow.com/       SSLError       OK 403
    https://huggingface.co/          SSLError       OK 200
    https://arxiv.org/               SSLError       OK 200
    https://lemonade-server.ai/      SSLError       OK 200
    https://www.amd.com/             ReadTimeout    OK 200
    https://example.com/             SSLError       OK 200
    https://www.python.org/          OK 200         OK 200
    https://en.wikipedia.org/…       OK 200         OK 200
    ```

    **9 of 11 fail**, including `example.com` — the canonical "always works" URL — and
    both of GAIA's own sites (`amd-gaia.ai`, `lemonade-server.ai`). Independently
    reproduced and then some.
- **Answer to "is there any code path or config that already fixes SNI that the report
  missed?" — No.** Checked specifically:
  - The adapter is mounted **unconditionally** for both schemes at **301–305**
    (`_adapter = PinnedIPAdapter(); mount("https://"); mount("http://")`). No env var,
    constructor arg, or config key disables pinning (`grep` for `GAIA_WEB`,
    `allow_private`, `pinned` → nothing).
  - There is no second, unpinned session for HTTPS.
  - Other download paths do **not** route through it — `src/gaia/hub/catalog.py:109`
    uses plain `requests.get`, so `gaia hub install` is unaffected. That bounds the
    blast radius but does not fix the web tools.
- **What the report misses and should say:** the limitation is **already documented
  in-code as an accepted trade-off** — the class docstring at **125–139** ("Residual
  HTTPS limitation (documented, not silently ignored) … that is intentionally out of
  scope here in favour of a correct, narrower guarantee") plus a runtime `log.warning`
  at **216–227**. That matters two ways: (a) a reviewer will otherwise think the report
  caught an oversight, when the author knew and scoped it out; (b) the report's real
  contribution is the **measurement** — the author's "most CDNs / shared hosts *may*
  fail" is in fact ~80% of the web, including GAIA's own docs site. Reframed as *"the
  documented residual limitation is not residual — here is the number"*, the finding
  gets stronger, not weaker, and survives the obvious rebuttal.
- **Required edit to REPORT.md:** add one clause naming `client.py:125-139` as the
  in-code acknowledgement, and one clause bounding scope (`gaia hub install` uses plain
  `requests` and is unaffected). Refresh the probe table with these 11 hosts — adding
  `example.com` is the single most persuasive row.
- **Severity check:** keep 🔴, and consider making it the **top** functional finding of
  §2.3. "Summarise https://github.com/amd/gaia" — the exact task GAIA's own README
  demos — returns an SSL error.
- **Fix check:** the proposed fix is the right one and matches what the docstring
  itself says is needed. Concretely: keep `send()` rewriting the *connect address*, but
  pass `server_hostname=host` through `pool_kwargs` on `connection_from_host`, or
  subclass `HTTPSConnection` and set `server_hostname` independently of `.host`. Risk
  to watch: `assert_hostname` and `server_hostname` must both be the real hostname or
  the SSRF pin silently weakens — assert it, don't comment it. The proposed
  skip-if-offline live test against 3–4 CDN hosts is essential; a mocked test cannot
  catch this (same lesson as C19).
- **Tracked check:** none open. **#1207** (CLOSED) *"[Bug]: SEARCH_WEB tool has SSLCert
  error"* is very likely this same bug reported by a user and closed without the root
  cause — cite it as evidence it already reached users. **#2687** and **#956** (both
  CLOSED) are prior fixes to this exact adapter, so it has been edited twice since the
  limitation was written down and never measured.

---

### C25 CONFIRMED

- **Evidence:**
  - `src/gaia/talk/sdk.py` **258–262** (inside `start_voice_session`'s
    `voice_processor`): `chat_response = self.chat_sdk.send(text)` then
    `await self.audio_client.speak_text(chat_response.text)`. Cite exact. Same pattern
    at **211–215** in `TalkSDK.process_voice_input`.
  - `src/gaia/audio/audio_client.py` **345–365** `speak_text`: builds a queue, starts
    `tts.generate_speech_streaming` in a thread with **no `status_callback`**, then
    `tts_thread.join(timeout=5.0)` and returns. **No `pause_recording()` anywhere in
    the function.** Cite exact.
  - The correct pipeline is real and dead: `AudioClient.process_voice_input`
    (**170–237**) calls `whisper_asr.pause_recording()` at **188**, installs a
    `tts_status_callback` at **217–226** that pauses/resumes around speech, and passes
    it as `"status_callback"` at **232**. `grep -rn "process_voice_input" src/ hub/` →
    only its own definition and `TalkSDK.process_voice_input` (`sdk.py:197`), which is
    a *different* method that also calls `speak_text`. **Nothing calls
    `AudioClient.process_voice_input`.** Confirmed as reported.
  - `join(timeout=5.0)` then return means for any reply longer than ~5 s of audio the
    function returns while TTS is still playing and the loop resumes consuming
    transcription.
- **Required edit to REPORT.md:** none. Optionally add that the 5-second join is a
  second, independent defect — even with pausing wired, a reply longer than 5 s would
  resume the mic mid-sentence — so the fix must be "pause → full `join()` → drain →
  resume", not "add a pause".
- **Severity check:** keep 🔴, and keep the hedge already in the text. REVIEW.md's 🔴
  bar is "a bug that will fire in normal use"; laptop speakers are normal use for
  `gaia talk`. A reviewer could argue 🟡 because headphones avoid it — the existing
  parenthetical "(mechanism; audible loop depends on room level)" pre-empts that
  correctly. Leave it.
- **Fix check:** the fix is right, and "collapse the two pipelines" is the important
  half — `AudioClient.process_voice_input` is already-written, already-correct dead
  code, so this is a delete-and-rewire, not new logic. One caveat: the dead path also
  installs a keyboard listener (**195**) for interrupting playback; wiring it in
  changes CLI behaviour (Enter now stops TTS) and should be called out rather than
  shipped as a side effect.
- **Tracked check:** none for the bug. **#2985** (OPEN) *"voice stack's degraded and
  streaming-lifecycle paths are untested"* is correctly cited as adjacent — and is in
  fact the reason this survived. **#702** is the voice-first roadmap item.

---

### C26 CONFIRMED-ADJUST

- **Evidence:** every listed flag checked by grep **and** by a real `parse_args` probe,
  as required.
  - The talk branch is `src/gaia/cli.py` **751–785**. `TalkConfig(...)` at **759–774**
    reads exactly these keys: `index`, `whisper_model_size`, `audio_device_index`,
    `silence_threshold`, `mic_threshold`, `no_tts`, `stats`, `logging_level`.
    **`model`, `max_tokens`, `use_claude`, `use_chatgpt`, `claude_model` and
    `base_url` are never read in this branch.** Each one is genuinely unread. ✔
  - **`--stats` is doubly dead.** `cli.py:1241-1247` declares
    `"--stats", "--show-stats", action="store_true", dest="show_stats"`, but **768**
    reads `kwargs.get("stats", False)`. **parse_args probe** (`p26.py`):

    ```
    namespace['show_stats'] = True
    namespace['stats']      = <ABSENT>
    ```

    The key `stats` does not exist in the namespace at all → always `False`.
    Confirmed. (Small correction: the report writes `kwargs.get("stats")`; the code has
    `kwargs.get("stats", False)`. Same outcome — quote it right.)
  - Same probe confirms the others **are** parsed and available and simply not used:
    `model='MYMODEL'`, `max_tokens=9`, `use_claude=True`,
    `claude_model='claude-opus-5'`, `base_url='http://x:1'`.
  - The flags **would** work if wired: `TalkConfig` has `model`, `max_tokens`,
    `use_claude`, `use_chatgpt`, `show_stats`, and `TalkSDK.__init__` plumbs them into
    `AgentConfig` (`model=self.config.model`, `use_claude=self.config.use_claude`). Pure
    missing wiring.
  - The pre-flight / actual-model split is confirmed: `cli.py:520-532` passes
    `base_url`, `use_claude`, `use_chatgpt` to `initialize_lemonade_for_agent` for the
    `"talk"` profile, so those flags *do* change what is checked and loaded — and then
    the conversation runs on `TalkConfig`'s defaults. Worse than "ignored"; the
    report's "the wrong model is loaded and then a different one used" is exactly right.
  - Secondary examples verified:
    - `gaia mcp start --ctx-size` (declared `cli.py:2410-2415`, default 32768): the
      **only** `args.ctx_size` reads in the file are **4094 / 4097 / 4118**, all in the
      `gaia eval` branch. Never read by `mcp start`. ✔
    - `gaia eval agent --device` is dead as claimed. Probe:
      `parse_args(['eval','agent','--device','npu'])` → `model='claude-opus-5'`, so
      `if eval_device and not eval_model` (**cli.py:3910**) can never be true —
      `--model` carries a non-None default. ✔
- **Required edit to REPORT.md:** three precision fixes.
  1. Quote the code as `show_stats=kwargs.get("stats", False)`.
  2. **`--base-url` is not "silently ignored"** — it is honoured for the Lemonade
     pre-flight (**520**, **531**) and then dropped, exactly like `--model`. Move it out
     of the "ignored" list into the "honoured for the pre-flight, dropped for the
     conversation" clause. Add that `TalkConfig` has **no `base_url` or `claude_model`
     field at all**, so those two cannot be wired without a dataclass change — a
     slightly bigger fix than the others.
  3. Name the `--device` mechanism explicitly: it is dead **because `--model` defaults
     to `claude-opus-5`**, not because the branch is unreachable in general. That is the
     actual one-line fix (`default=None`), and a reader needs it.
- **Severity check:** keep 🔴. Accepting a flag, changing the pre-flight with it, then
  running a different model is a wrong-answer bug, not a UX nit — and #124 shows a user
  already lost time to it.
- **Fix check:** the proposed CLI test (monkeypatch `TalkSDK`, assert the built
  `TalkConfig`) is the right test and catches all of these. Add one assertion shape the
  report omits: a generic test that every flag on a subcommand's parser is read
  somewhere in that subcommand's branch would catch the whole class (`--ctx-size`,
  `--device`, `--stats`) rather than these four instances.
- **Tracked check:** **#124** (OPEN since v0.13) confirmed and reproduces — the body
  shows `gaia talk --model Qwen3-Coder-30B-A3B-Instruct-GGUF` running against
  `localhost:8000` with the wrong model. Correctly cited; the other flags are not in
  it, as the report says.

---

### C27 CONFIRMED

- **Evidence:**
  - `grep -rn setNotificationCount src/ tests/` → **exactly two producers of the
    name**: the call at
    `src/gaia/apps/webui/services/notification-service.cjs:321` and the Jest mock at
    `tests/electron/test_notification_service.js:78`
    (`return { setNotificationCount: jest.fn() };`), plus nine assertions against that
    mock. Confirmed exactly as the report says.
  - `_updateTrayBadge` is at **319–323**; its guard is `if (this.trayManager)` — a
    truthiness check, not a method check.
  - `TrayManager` (`services/tray-manager.cjs`) methods: `constructor`, `create`,
    `destroy`, `refresh`, `_rebuildContextMenu`, `_showWindow`, `_loadTrayIcon`,
    `_loadIcon`, `_loadConfig`, `_saveConfig`, `_registerIpcHandlers`,
    `_applyLoginItemSetting`. **No `setNotificationCount`.**
  - The guard is always true in production: `main.cjs:604` `trayManager = new
    TrayManager(...)`, **608–611** `notificationService = new NotificationService(…,
    trayManager)`. So every call throws `TypeError`.
  - Callers confirmed at **149** (`_addNotification`), **170** (`markAllRead`), **179**
    (`clearAll`). Cites exact.
  - Normal path confirmed: notifications arrive through `_handleJsonRpcMessage`, called
    inside the `try` at `agent-process-manager.cjs:706-713`, whose `catch` logs
    `"[agent-mgr] Non-JSON stdout from …"` (**711**) — so a `TypeError` from the badge
    update is mislabelled as a parse error and swallowed.
  - Crash path confirmed: `agent-process-manager.cjs:826`
    `this.emit("agent-crash-limit", …)` sits inside `child.on("exit", …)` (registered
    at **189**), so a throwing listener is synchronous and unguarded;
    `notification-service.cjs:352` is the listener; `main-safety-net.cjs:119`
    `process.on("uncaughtException", (err) => fatal(err))` → **115**
    `try { process.exit(1); } catch {}`. Chain complete.
- **Required edit to REPORT.md:** none. Optionally sharpen one line: the Jest suite
  doesn't merely fail to catch this, it **actively asserts the broken contract**
  (`expect(trayManager.setNotificationCount).toHaveBeenCalledWith(2)` at
  `test_notification_service.js:716`) — a green test proving a method that has never
  existed. That is the strongest version of the point.
- **Severity check:** keep 🔴. The crash-limit path terminating the app and orphaning
  the backend and sidecars is data-loss-adjacent and fires without user action.
- **Fix check:** implementing `setNotificationCount` on `TrayManager` is correct and
  small (`app.setBadgeCount` on macOS, tray tooltip / overlay icon on Windows). The
  report's "wrap listener invocations" is the more important half — a throwing
  EventEmitter listener taking down the app is a general hazard here, not specific to
  this method. "Test against the real `TrayManager`" is exactly right and is the only
  change that would have prevented it.
- **Tracked check:** none found.

---

### C28 CONFIRMED

- **Evidence:** `src/gaia/apps/webui/main.cjs`, startup sequence re-read verbatim:
  - **1007** `backendProcess = await startBackend();` — `startBackend` spawns the child
    and returns it; its body sets `healthCheckUrl` and calls `spawn(...)`, with **no**
    `waitForBackend`.
  - **1010** `createWindow()`, **1014** `initializeServices()` (this is where
    `agentProcessManager` becomes non-null), **1034** `processStartupDeepLinks()`,
    **1037** `await loadApp()`, **1044–1046** `waitForBackend(STARTUP_TIMEOUT)`. All
    four cites exact and in the order the report states.
  - `handleDeepLink` re-read: its only readiness guard is
    `if (!agentProcessManager) { pendingDeepLinkUrl = rawUrl; return; }` — and
    `initializeServices()` at 1014 has already set it, so at 1034 the link is dispatched
    immediately via `void runDeepLink(command)`.
  - Fail-closed path confirmed: `confirmDeepLinkInstall` (**829–844**) →
    `agentProcessManager.fetchCatalogEntry` (`agent-process-manager.cjs:319-350`), a
    bare `fetch(\`${base}/api/agents/catalog\`)` with **no retry**, which on a connect
    error throws `"Could not reach the GAIA backend to look up …"` →
    `dialog.showErrorBox("Could not verify this agent", message)` (**839**).
  - The `second-instance` (already-running) path is unaffected because the backend is
    long since up — matching "only the already-running path works".
- **Required edit to REPORT.md:** none. The report is precise here.
- **Severity check:** keep 🔴. It breaks the website's primary onboarding CTA ("Open in
  GAIA") for every user whose app is closed — the common case, and the case the feature
  exists for.
- **Fix check:** "dispatch after `waitForBackend()` resolves" is correct and minimal.
  One caveat to state: `waitForBackend` can time out (**1049–1051** logs a warning and
  continues), so the deep link must still be dispatched on the timeout branch —
  otherwise a slow backend converts a wrong error into a *silently dropped* link, which
  is worse. A bounded retry inside `fetchCatalogEntry` would be a cheaper, more robust
  fix and also helps the macOS `open-url` race.
- **Tracked check:** none found. **#1725** is referenced in the code comment at
  **1032–1033** as the issue that *added* `processStartupDeepLinks` — so this ordering
  was introduced by a fix and never verified cold. Worth citing.

---

### C29 CONFIRMED

- **Evidence:** I ran the validators myself rather than trusting the report.
  - `docs/docs.json` **505–510**: navbar label is literally
    `"v0.23.0 \u00b7 Lemonade 11.5.0"`. `src/gaia/version.py`: `__version__ = "0.23.1"`,
    `LEMONADE_VERSION = "11.8.1"`. The label is wrong on **both** halves.
  - `.github/workflows/publish.yml` **152–159** is the navbar gate:
    `for link in docs.get("navbar",{}).get("links",[]): if tag in link.get("label","")`
    → `"v0.23.1" in "v0.23.0 · Lemonade 11.5.0"` is **False** → appended to `errors` →
    **161–166** `sys.exit(1)`. Hard fail confirmed.
  - **Ran it:** `.venv/Scripts/python.exe util/validate_release_notes.py
    docs/releases/v0.23.1.mdx --tag v0.23.1`

    ```
    ❌ docs/releases/v0.23.1.mdx:
       - Missing required section: '## What's New' or '## Key Changes'
    EXIT=1
    ```

    `publish.yml:169-172` runs exactly that command. Second hard fail confirmed.
    (`grep '^## '` on the notes → `## Breaking Changes`, `## Bug Fixes`,
    `## Full Changelog`. No `## What's New`.)
  - `util/check_doc_versions.py` contains **zero** references to `docs.json` —
    confirming "reports OK because it never scans docs.json".
  - `.github/workflows/update-release-branch.yml`: **8–10** `on: push: tags: ['v*']`
    (independent of publish.yml's gates), **23–29** pre-release tags `exit 1` (red, not
    skipped), **41–42** `git branch -f release "$TAG_NAME"; git push -f origin release`.
    All three cites exact.
  - The bad commands are real. Against the actual parser: `parse_args(['list'])` →
    `SystemExit 2`, `"invalid choice: 'list'"`; `gaia install` exists but is the
    Lemonade installer and takes no agent id (`gaia` lands in unrecognised args).
    Meanwhile `tui/internal/client/negotiate.go`'s `updateCommand()` returns
    `"gaia hub %s %s"` and its own comment says *"`gaia install` / `gaia uninstall`
    also exist and look right, which is the"* trap — the codebase already knows this is
    the confusable pair. `parse_args(['agent','install','gaia'])` also succeeds
    (`agent_action='install'`), so there are two right spellings; the v0.23.1 notes use
    the one wrong spelling, four times (lines 8, 11, 19, 25).
- **Required edit to REPORT.md:** none factually. Two strengtheners: (a) the navbar
  label is wrong on the Lemonade version too (11.5.0 vs 11.8.1), so the fix text
  `v0.23.1 · Lemonade 11.8.1` is right — say *why*, citing `version.py:LEMONADE_VERSION`;
  (b) cite `negotiate.go`'s `updateCommand` comment as in-repo proof that
  `gaia install <agent>` is a known-confusable and the notes fell into exactly the trap
  the code warns about.
- **Severity check:** keep 🔴. This is a release blocker by definition — the tag would
  fail two gates and still force-move `release`.
- **Fix check:** the fix list is correct and complete, and the ordering is right:
  chaining `update-release-branch` after `github-release` is the part that prevents a
  *half-published* release and matters more than either content fix. One addition:
  `update-release-branch.yml:26-28` should `exit 0` (or use an `if:`) for pre-release
  tags — `exit 1` marks legitimate rc tags red, which trains maintainers to ignore red
  on release tags, which is how the other two failures get missed.
- **Tracked check:** **#1128** (OPEN) *"feat(github): release-prep — changelog
  generation, CI check, draft release notes"* confirmed open and is the right systemic
  ticket. The concrete v0.23.1 breakage is untracked, as the report says.

---

### C30 CONFIRMED-ADJUST

- **Evidence:** verified live and against source.
  - **Live probe:** `curl -s https://amd-gaia.ai/hub/gaia` → **HTTP 200**, page
    contains:

    ```
    data-im-copy="gaia agent install gaia"
    data-im-copy="pip install gaia-agent-gaia"
    data-im-copy="git clone https://github.com/amd/gaia.git"
    ```

    `curl https://pypi.org/pypi/gaia-agent-gaia/json` → **HTTP 404**.
    `curl https://registry.npmjs.org/@amd-gaia%2Fgaia` → **HTTP 200**.
    All three of the report's live claims reproduced exactly.
  - **Source:** `website/src/data/catalog.ts` — **503–521** the non-agent branch,
    **523–538** the `agent.npm_package` branch (returns GAIA + npm, no pip),
    **540–546** the GAIA method, and **548–555**:

    ```ts
    if (agent.language === "python") {
      methods.push({ key: "pip", label: "pip",
        command: `pip install gaia-agent-${agent.id}`, … });
    }
    ```

    The gate is exactly "is an agent, has no `npm_package`, `language === 'python'`" —
    the report's description is correct and **548–555** is the right cite.
  - `hub/agents/gaia/python/gaia-agent.yaml` declares `language: python` (line 25) and
    **no `npm_package`** — so the flagship takes the pip branch. Confirmed.
  - I also checked the page's other command: `gaia agent install gaia` is **valid**
    (`parse_args(['agent','install','gaia'])` → `agent_action='install'`), so the page's
    primary CTA is fine and only the pip tab is wrong. Worth stating so the finding is
    not read as "the hub page is broken".
- **Required edit to REPORT.md:** two.
  1. Add the missing dedup. `Tracked: #3101` is defensible but incomplete — the same
     `installMethods()` function was edited twice recently for the *mirror image* of
     this bug: **#3147** (CLOSED) *"fix(hub): agent pages offer npm as the only install
     path for npm-distributed agents"* and **#2969** (CLOSED) *"docs(email): hub page
     wrongly says npm is the only install path"*. Change to
     `Tracked: #3101 (same root cause, different surface); #3147/#2969 fixed the mirror
     case in this same function without adding the artifact-kind gate that would have
     covered both.` That reframes it from a one-off typo to a missing invariant — which
     is the report's actual point.
  2. Add that the page's primary CTA is correct (verified), so the impact is one tab of
     three.
- **Severity check:** **downgrade 🔴 → 🟡.** REVIEW.md reserves 🔴 for security /
  breaking / data-loss / fires-in-normal-use. This is a wrong command on one tab of a
  public page; the page's **primary, recommended** CTA is correct, and a user who
  copies the pip line gets an immediate, unambiguous
  `No matching distribution found` — no data loss, no silent wrong behaviour, and a
  working path one tab away. Real and embarrassing on the flagship's own page, but
  that is 🟡. Note the report files it under §2.4 "Release is blocked", and it does not
  block a release.
- **Fix check:** both proposed fixes work but are not equivalent, and the report should
  say which it prefers. Adding `npm_package: "@amd-gaia/gaia"` to the manifest is a
  one-line change that fixes *this* page today (routing it into the **523–538**
  branch) — but it is a data patch, and the next binary-only Python agent with no npm
  package regresses. The `artifact_kinds` / `has_wheel` field emitted by `toIndexEntry`
  is the real fix and the one that makes the proposed `catalog.test.ts` case
  ("binary-only python agent gets no pip method") meaningful. Recommend both: manifest
  now, gate next.
- **Tracked check:** **#3101** (OPEN) confirmed; **#3147**, **#2969** (both CLOSED) as
  above.

---

## Missed

Nothing at 🟡 or above surfaced during this pass that §2 does not already cover. Two
things I chased and cleared, recorded so nobody re-chases them:

- **Does the SNI bug (C24) also break `gaia hub install` / `gaia skill install`?**
  **No.** `src/gaia/hub/catalog.py:109` uses plain `requests.get`, not `WebClient`, and
  `src/gaia/skills/hub.py` does not import `WebClient`. The blast radius is the
  web-fetch / search / download tool surface and the `rss-digest` skill only, as the
  report states. (Had it routed through `WebClient`, agent installation would be broken
  against `amd-gaia.ai`, which my probe shows fails the handshake — worth re-checking
  if that transport is ever unified.)
- **Does the hub page's primary CTA `gaia agent install gaia` also not exist?**
  **It exists.** `parse_args(['agent','install','gaia'])` resolves to
  `agent_action='install', agent_id='gaia'`. Only C30's pip tab is wrong.
  (Side note, 🟢 and out of scope: `CLAUDE.md` documents `gaia agent {export|import}`
  and omits `install` / `uninstall` — stale docs, below the reporting bar.)

---

## Summary

**Counts** — CONFIRMED **8** · CONFIRMED-ADJUST **7** · OVERSTATED **0** ·
REFUTED **0** · UNVERIFIABLE **0**.

- **CONFIRMED (8):** C17, C19, C21, C24, C25, C27, C28, C29.
- **CONFIRMED-ADJUST (7):** C16, C18, C20, C22, C23, C26, C30.
  (C20 is "adjust" only because the finding is *understated*, not wrong.)

No finding in C16–C30 is false. Every claimed probe I could re-run, I re-ran, and every
one reproduced — including the two hardest: the PyInstaller orphan against the real
installed `gaia-agent.exe`, and the live SNI failure.

**The five most important corrections:**

1. **C22 and C23 are gated on C21, and the report never says so.** On the *shipped
   one-file binary* the respawn they describe does not happen on the next message —
   C21 proves the reader stays parked and `s.started` stays true, so the next message
   goes to the surviving orphan, which still has `/bypass off` and its history. The
   revert lands one message later. A maintainer running C22's repro on the release
   build will not see it and will close the finding. Add the gate to both, and note it
   *is* immediate on a source / `--onedir` install.

2. **Three cited line ranges do not exist.** `bypass.go:2138-2175` (C22) — the file is
   **169 lines**; use `bypass.go:89-126` and `:143-154`. `client.go:965-970` (C23) —
   the file is **114 lines**; use `client.go:32-37`, `sse.go:793,856`,
   `chat/model.go:1298-1300`. `release_agent_gaia.yml:10-11` (C21) is prose; the
   `--onefile` flag is at **:412**. Distrusting line numbers paid off.

3. **C24 should lead §2.3 — and must acknowledge the in-code disclaimer.** My probe is
   worse than the report's: **9 of 11** hosts fail, including `example.com` and both of
   GAIA's own sites. But `client.py:125-139` already documents the limitation and calls
   the fix "intentionally out of scope", and the report currently reads as if it caught
   an oversight. Reframe as *"the documented 'residual' limitation is ~80% of the web,
   measured"* — stronger, and it survives the author's obvious rebuttal. Also bound it:
   `gaia hub install` uses plain `requests` and is unaffected.

4. **Two severities are one tier too high.** **C23 → 🟡**: the SQLite turn log and the
   visible transcript both survive; what is lost is the 12-pair prompt window, so the
   user gets a confused answer, not destroyed data. **C30 → 🟡**: the hub page's
   *primary* CTA is correct (I verified it parses), only the third tab is wrong, and it
   fails loudly with `No matching distribution found`. Neither belongs under a "Release
   is blocked" heading. Leaving them at 🔴 is the inflation that makes a reviewer
   discount C21 and C24.

5. **C20 is understated and its proposed fix is incomplete.** My 25-turn probe shows
   the session is *permanently* stuck and makes **zero** progress on pass two — and
   because the re-summarised window still reaches `store()` at `memory.py:1673` and
   `:1704`, every reboot writes a **duplicate** summary note and duplicate knowledge
   items, forever. Also, `mark_session_consolidated()` alone would mark turns 1–5
   consolidated without ever summarising them — trading a loud bug for a silent one.
   The fix must be windowed consolidation keyed on the oldest unconsolidated turn,
   *plus* moving `prune()` (`memory.py:593`) after `_run_memory_post_init`.

Two smaller corrections that will otherwise be argued in review: **C16**'s
`UnicodeDecodeError` is *reported*, not swallowed (caught at `cli.py:4539`), and Windows
kills exactly one wrong process while Unix kills every pid `lsof` returns; **C26**'s
`--base-url` is honoured for the Lemonade pre-flight and then dropped — the same shape
as `--model`, not "ignored".
