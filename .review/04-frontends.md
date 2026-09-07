# Review 04 — Frontends (Go TUI, React Agent UI, Electron, website, Cloudflare workers)

Reviewer scope: `tui/`, `src/gaia/apps/webui/`, `src/gaia/electron/`, `src/gaia/apps/{llm,example,_shared}`,
`src/gaia/ui/build.py`, `website/`, `workers/`, `tests/electron/`. Commit 211f08c5.

Status: COMPLETE for TUI + Electron + React renderer. Workers/website are in `04b-workers-website.md`.

## Scope covered

**Go TUI (`tui/`) — read fully:** `cmd/gaia/main.go`; `internal/cli/{root,chat,agents,control,version}.go`;
`internal/control/{server,state,keys,paths,debuglog,doc}.go`; `internal/client/{subprocess,factory,cmdline,sse,sse_frame,client,negotiate,memory(part),prescan(part)}.go`;
`internal/daemon/*.go`; `internal/catalog/{agent,catalog,hub}.go`; `internal/gaiainit/gaiainit.go`; `internal/event/*.go`;
`internal/ui/{app,oneshot}.go`; `internal/ui/root/*.go`; `internal/ui/chat/{model,canonical,message,memoryview,narrate,palette,hints,mousecapture,selectmode,introspect,sanitize,clipboard,clipboardimage_windows,setup,claude,bypass,modelcmd,devlog}.go`;
`internal/ui/components/{confirmation,question,markdown,helpoverlay,helpstate,statusbar,band,confirm}.go`; `internal/ui/preflight/*.go`;
`internal/ui/cards/{cards,box,fallback,primitives}.go`; `internal/ui/status/status.go`; `test/` (listing + `mockagent/main.go`).
Skipped: `ui/theme`, `ui/brand`, `cards/{emailprescan,attention}.go`, `chat/{turnmetrics,tokenrate,lemonadechip,questionmouse,widthlayout}`, most `_test.go` bodies.

**Toolchain runs:** `go vet ./...` clean; `go test ./...` all 16 packages pass (39 s for `tui/test`).
`go test -race` could not run: `-race requires cgo` and no gcc on this box (CI's `build_tui.yml` does run it per the Makefile comment).
No `node_modules` anywhere in the checkout, so `tsc`/`vitest`/`wrangler` could not be run (not installed to keep the shared checkout untouched).

**Empirical checks on this machine (installed GAIA 0.23.x, `C:\Users\<u>\AppData\Local\Programs\GAIA\`):** spawned the released
`gaia-agent.exe` from PowerShell with redirected pipes, killed the parent pid, and observed process tree + stdout EOF behaviour (see 🔴 #1).

**Electron main process + services (`src/gaia/apps/webui/`) — delegated sub-review, every finding below re-verified by me against the source (line numbers quoted are mine):** `main.cjs` (window creation, `setWindowOpenHandler`, `loadApp`, startup ordering, `will-quit`), `main-safety-net.cjs`, `preload.cjs` (full), `bin/gaia-ui.cjs` (listen), `services/{notification-service,agent-process-manager,backend-installer,auto-updater,tray-manager}.cjs` (targeted), `electron-builder.yml`, `src/gaia/electron/src/preload/preload.js` + `services/base-ipc-handlers.js`, `src/gaia/ui/build.py` (full), `tests/electron/` inventory + `test_notification_service.js`.

**React renderer (`src/gaia/apps/webui/src/`) — read by me:** `utils/markdown.ts` (full), `services/api.ts` (full), `components/MessageBubble.tsx` (full), `utils/apiBase.ts` (full), `hooks/useConnectorsSSE.ts` (full), `stores/{chatStore(part),agentChatStore,terminalStore(part),notificationStore(part)}.ts`, `components/{ChatView(targeted regions),AgentTerminal(part),PermissionPrompt(grep)}.tsx`, `App.tsx` (SSE + nav regions), plus repo-wide greps for `dangerouslySetInnerHTML`/`innerHTML`/`window.open`/`target=_blank`/`rehype-raw`/`localStorage`/empty `catch`. Cross-checked `src/gaia/ui/routers/files.py` `/api/files/open`. Skipped: `MemoryDashboard.tsx` (2.6K lines, no XSS sinks per grep), `ConnectorsSection.tsx`, `FileBrowser.tsx`, `DocumentLibrary.tsx` bodies, onboarding, most `__tests__`.

**Not covered here:** `workers/` + `website/` (separate report `.review/04b-workers-website.md`).

## Findings

### 🔴 Esc/Ctrl+C on the flagship agent does NOT stop the released `gaia-agent` binary — the real process keeps running the cancelled turn (and the user's next message), and the TUI then reports "file already closed"
- **Where:** `tui/internal/client/subprocess.go:222-228` (`case <-ctx.Done(): st.proc.kill()`), `:263` (`for st.scanner.Scan()` — only unblocks on EOF or a line), `:135-140` (`startLocked` reuses `s.stdin`/`s.stdout` while `s.started` is true), `:505-509` (`resetDeadChild` = kill → `reap()`/`Wait()` → `discard`). Released binary: `.github/workflows/release_agent_gaia.yml:10-11` ("frozen with PyInstaller into a one-file native binary"); installed here as `gaia-agent.exe` (49 MB, no `_internal/` → one-file).
- **What:** A PyInstaller one-file binary is a bootloader *parent* plus the real Python *child*. `Process.Kill()` kills only the parent; the child keeps both pipe ends. Verified on this machine: after `Kill()` on the parent, a second `gaia-agent.exe` whose `ParentProcessId` is the killed pid stays alive; a read on its stdout **blocks** (no EOF after 8 s) and only completes once the stdin write-end is closed and the child exits on its own (~16 s when idle). In the TUI that means: (1) the reader goroutine stays parked in `Scan()`, so the deferred `resetDeadChild` never runs and `s.started` stays `true`; (2) the next Enter takes `startLocked`'s "already started" branch and writes the new query into the **orphan's** stdin — the orphan's stdin pump (`hub/agents/gaia/python/gaia_agent/stdio.py:661-672`) queues it behind the "cancelled" turn, which runs to completion, tool calls included, because `state.cancel_active()` only fires on EOF (`stdio.py:683`); (3) the first event the orphan writes wakes the old reader, whose `emit` returns false → deferred `resetDeadChild` → `Wait()` closes the pipes underneath the *second* turn's reader (two goroutines now `Scan()` one `bufio.Scanner` — a data race) → the second turn ends with `agent stdout read error: read |0: file already closed`; (4) the orphan then executes the user's second query with nobody listening, then exits on EOF.
- **Failure scenario:** `gaia-tui` → "delete all *.tmp under ~/proj" → the agent starts a shell tool → user presses Esc to stop it → the deletion still happens; user types "never mind, just list them" → red "agent stdout read error … file already closed"; a third message finally starts a fresh agent while two Python agents are alive hitting Lemonade at once.
- **Evidence:** PowerShell against the installed binary — `Get-CimInstance Win32_Process -Filter "Name='gaia-agent.exe'"` after `$p.Kill()` → `ProcessId 33000 ParentProcessId 43488` still listed; `ReadAsync` on stdout: "second read within 8s after PARENT kill: done=False … BLOCKED", then "done=True bytes=55" only after `StandardInput.Close()`. Design comment `subprocess.go:219-221` ("A cancelled turn must actually stop the child … Kill only — the reader reaps") assumes the killed pid owns the pipe. The Go test double `tui/test/mockagent/main.go` is a single-process Go binary, so `cancel_test.go`/`subprocess_test.go` can never observe this. Same structure on macOS/Linux (one-file per platform, SIGKILL cannot be forwarded by the bootloader).
- **Fix:** Stop relying on process death for EOF. On cancel (a) send a `gaia_control: cancel` line (the pump already handles control lines mid-turn and `state.cancel_active()` exists) and (b) close the stdin write end / kill the whole tree (Job Object on Windows, `Setpgid`+`kill(-pgid)` on POSIX) — or ship `--onedir` so pid == process. Make `startLocked` refuse to reuse a client whose previous turn has not reaped (`turnDone` still open) instead of writing into it. Add an integration test whose mock agent spawns a grandchild that inherits the pipes.
- **Confidence:** High (frozen-binary behaviour verified empirically; TUI code path traced)
- **Tracked:** none found (`gh issue list --search "cancel kills agent" / "gaia-agent orphan"` → only #2474, which is the daemon's sidecar reaping — a different path)

### 🔴 Cancelling a turn respawns the flagship agent from its ORIGINAL launch flags — `/bypass off` is silently undone, with the warning banner gone
- **Where:** `tui/internal/client/subprocess.go:238-246` (cancel → `resetDeadChild`) and `:145` (`exec.Command(s.path, s.args...)` on respawn); `tui/internal/ui/chat/bypass.go:2138-2175` (`setBypass` only sends a control line and flips the local flag); `tui/internal/ui/chat/canonical.go:39-74` (the model-state ping re-syncs the model, not bypass); `stdio.py:333-340` (ping fields) and `:1132` (`PermissionState(bypass=args.bypass_permissions)`).
- **What:** Whether the respawn happens promptly (Go mock agent) or after the orphan finally exits (🔴 above), the new child starts with whatever argv the catalog carried. Launched with `--bypass-permissions` then `/bypass off` → after a cancel the child is back in bypass mode while `m.bypassPermissions == false`, so no banner, no `/bypass off` hint in the status bar. The reverse (turned on mid-session, then cancel) silently re-enables prompts while the banner still says BYPASS ON. Loaded skills, `always`-granted tools (`PermissionState._grants`) and the `/model` switch are lost the same way — the `/model` case at least prints "[!] the agent process restarted and reverted…" (`canonical.go:51-57`); bypass does not.
- **Failure scenario:** `gaia-tui --bypass-permissions` → `/bypass off` → ask something → Esc → "clean up the build dir" → `run_shell_command rm -rf …` runs with no prompt and no banner on screen.
- **Evidence:** `model.go:283-289` comment: "a cancelled turn respawns the child from its ORIGINAL launch flags, silently reverting any live switch"; `subprocess.go:461-468` `BypassAtLaunch` scans the same `s.args` the respawn reuses; no `bypass`/`grants` field in the ping.
- **Fix:** After any respawn re-send the session's permission state before the first query (`SetBypassPermissions(m.bypassPermissions)` + grants), or carry `bypass` in the model-state ping and warn/resync like the model revert does. Test: `bypass_test.go` "launch on → /bypass off → cancel → next turn: child spawned without the flag / control line re-sent".
- **Confidence:** High
- **Tracked:** none found

### 🔴 Cancelling a turn silently drops the whole conversation history for the flagship agent
- **Where:** `tui/internal/client/subprocess.go:222-246`; `stdio.py:737-771` (`conversation_history` lives only in the child process); `tui/internal/client/client.go:965-970` — the host-owned transcript (`TranscriptResetter`) exists only for `SSEClient`, never for the subprocess transport.
- **What:** Esc during a flagship turn ends (eventually — see 🔴 #1) the `gaia-agent` process; the next turn is a brand-new process with empty history. The TUI transcript still shows every earlier exchange, so nothing tells the user the agent forgot them.
- **Failure scenario:** "read report.pdf and summarise" → follow-up is slow → Esc → "what was the second point?" → the agent answers from nothing (or invents one).
- **Evidence:** `catalog.go:664-668` "The child is started once and kept … what makes a skill loaded in one turn still loaded in the next"; `docs/guides/terminal-hub.mdx:234` documents Esc as "cancel the turn" with no mention of a restart.
- **Fix:** Cooperative cancel that keeps the process (see 🔴 #1), or keep a host-owned transcript for the subprocess transport and replay it on respawn (the SSE client already does exactly this via `context`). At minimum a visible "agent restarted — earlier context was lost" status line.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Claude-credential preflight walks UP the directory tree for `.env`, but the agent's `load_dotenv()` searches from a different root — the gate can pass on a key the child never sees
- **Where:** `tui/internal/ui/preflight/local.go:133-152` (`claudeCredentialConfigured` loops `parent := filepath.Dir(dir)` up to the root); `src/gaia/llm/providers/claude.py:156-158` (`load_dotenv()` with no `usecwd=True` — `find_dotenv` walks from the *calling module's* directory; inside the frozen `gaia-agent.exe` that is the extraction dir, not the TUI's cwd).
- **What:** The comment claims "accepting a .env here cannot turn a launch into a false pass"; that only holds if both sides search the same tree. A `.env` two directories above cwd satisfies the gate and the first message fails inside the agent with an auth error.
- **Failure scenario:** `cd ~/work/foo` (with `~/work/.env` holding `ANTHROPIC_API_KEY`) → `gaia-tui --use-claude` → row "Claude credential: set" → first message → 401 / "ANTHROPIC_API_KEY not set" from the child.
- **Evidence:** `local.go:142-151` walk-up loop; `claude.py:158` `load_dotenv()`.
- **Fix:** Check only `os.Getenv` + `./.env` in cwd **and** pass the value into the child's environment explicitly (the TUI already builds `cmd.Env` for `LEMONADE_BASE_URL`, `subprocess.go:150-152`); or make the agent use `find_dotenv(usecwd=True)` and mirror the identical walk, pinned by a test.
- **Confidence:** Medium (search-root mismatch verified; exact python-dotenv behaviour under PyInstaller not run)
- **Tracked:** none found

### 🟡 Subprocess transport buffers the agent's stderr without bound for the life of the process
- **Where:** `tui/internal/client/subprocess.go:146-147` (`stderr := &bytes.Buffer{}; cmd.Stderr = stderr`), only read at `:345` after a non-zero exit.
- **What:** `stdio.py:811` redirects `sys.stdout = sys.stderr`, so every stray `print` from a tool, skill or third-party library, plus Python warnings, accumulates in the TUI's memory for the whole session and is surfaced only if the child dies non-zero.
- **Failure scenario:** A long session with a chatty tool (progress bars, an MCP server logging to stdout) grows the TUI's RSS by the full volume of that output; nothing drains it.
- **Fix:** Ring buffer (last 64 KB) or stream it to `~/.gaia/logs/gaia-tui.log` through `control.logf`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Confirmation timeout is hardcoded as the literal "30s" in three user-visible strings while the constant lives elsewhere
- **Where:** `tui/internal/ui/components/confirmation.go:175` (`ConfirmationTimeout = 30 * time.Second`), `:434` ("timed out after 30s"), `:455` ("auto-denies in 30s"), `tui/internal/ui/chat/canonical.go:437` ("denied (30s timeout — no response)").
- **What:** Changing the constant leaves the UI lying about the timeout.
- **Fix:** Format from the constant; the existing confirmation tests pin the text.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `Esc` means "deny" on the confirmation panel but "cancel the whole turn" on the look-alike question panel
- **Where:** `tui/internal/ui/chat/model.go:905-923`; hints at `components/question.go:757` ("esc cancel the turn") vs `confirmation.go:449` ("n/esc deny"); `docs/guides/terminal-hub.mdx:289`.
- **What:** Documented, but two inline panels with opposite Esc semantics is how a user kills a run they meant to keep. Ctrl+C already cancels the turn everywhere.
- **Fix:** Make Esc on a question "decline to answer" and leave turn-cancel to Ctrl+C / a second key, matching the confirmation panel.
- **Confidence:** High (UX)
- **Tracked:** none found

### 🟢 `tui/README.md` download table (and a `download-target.ts` comment) still name `0.23.0` artefacts while the package is `0.23.1`
- **Where:** `tui/README.md:44-49` (`…/terminal-hub/0.23.0/gaia-win-x64.exe` …); `website/src/scripts/download-target.ts:199`; `src/gaia/version.py:9` (`__version__ = "0.23.1"`).
- **What:** A hand-written version in the doc table drifts every release; the README points at the previous build.
- **Fix:** Link `…/terminal-hub/latest/…` or generate the table from `manifest.json` in the release skill.
- **Confidence:** High
- **Tracked:** none found

---
*Electron main process, preload, services and the React renderer. Findings marked (sub-review) originated from the delegated Electron pass; every one was re-verified against the code.*

### 🔴 Unvalidated `shell.openExternal` in the main process + a renderer link allow-list that passes root-relative/protocol-relative URLs — one click on an LLM-rendered link can launch a local binary or a UNC path
- **Where:** `src/gaia/apps/webui/main.cjs:515-518` (`setWindowOpenHandler(({url}) => { shell.openExternal(url); return {action:"deny"} })`, no scheme check, no `will-navigate` guard anywhere in `main.cjs`); `main.cjs:569-571` (window is loaded with `mainWindow.loadURL(pathToFileURL(indexPath).href)` → `file://` origin); `src/gaia/apps/webui/src/utils/markdown.ts:52` (`if (url.startsWith('#') || url.startsWith('/')) return url;` and line 56 passes any scheme-less relative URL); `src/gaia/apps/webui/src/components/MessageBubble.tsx:676-687` (every markdown link is `<a href target="_blank" rel="noopener noreferrer">`).
- **What:** `safeUrlTransform` blocks `javascript:`/`data:` but deliberately lets `/…` (and therefore `//host/…` and `foo/bar`) through. In a `file://` window a `target=_blank` click on `/Windows/System32/calc.exe` resolves to `file:///C:/Windows/System32/calc.exe`, `//attacker/share/x.exe` to `file://attacker/share/x.exe`; Chromium routes the new-window request to `setWindowOpenHandler`, which hands the resolved URL to `shell.openExternal` = ShellExecute on Windows.
- **Failure scenario:** A fetched web page / PDF contains `[open the report](/Users/me/Downloads/report.exe)` or `[details](//attacker/share/payload.exe)`; the model repeats the link in its answer; user clicks → the binary runs (or an SMB fetch to an attacker host occurs). This is exactly the prompt-injection model the markdown hardening (#976) was written for.
- **Evidence:** quoted lines above; `grep -n will-navigate main.cjs` → nothing.
- **Fix:** In `main.cjs` parse the URL and allow only `http:`/`https:`/`mailto:` (log + deny otherwise); add `webContents.on('will-navigate', e => e.preventDefault())`; in `markdown.ts` keep only `#` fragments for chat content (a chat answer has no legitimate root-relative link). 🔒 security — tag @kovtcharov-amd.
- **Confidence:** High for the main-process gap and the allow-list; Medium on the exact WHATWG resolution of `/…` against a `file:///C:/…` base (drive letter is retained per the spec's file-slash state).
- **Tracked:** none found (#2950 is about shell-string interpolation)

### 🔴 "Always allow this tool" is persisted for ever in `localStorage`, keyed by tool *name*, auto-approved client-side with `remember=false`, and there is no UI to revoke it
- **Where:** `src/gaia/apps/webui/src/stores/notificationStore.ts:22` (`ALWAYS_ALLOW_TOOLS_KEY = 'gaia_always_allow_tools'`), `:109-117` (push tool name into localStorage when `remember`); `src/gaia/apps/webui/src/components/ChatView.tsx:791-800` and `:825-834` (`alwaysAllowed.includes(toolName)` → `api.confirmToolExecution(sessionId, confirm_id, 'allow', false)` / `confirmTool(sessionId, true)` without showing the prompt); `components/PermissionPrompt.tsx:195-202` (the checkbox); `grep gaia_always_allow_tools src/` → only those two files (no settings/permission-manager entry, no `removeItem`).
- **What:** One tick of "Always allow this tool" on e.g. `run_shell_command` silently approves every future shell command, in every session, for the lifetime of the browser profile; the backend sees each approval as a fresh explicit "allow" (`remember=false`), so its own grant store and audit log never record a standing grant. The TUI's confirmation model was deliberately built the other way (invocation-scoped `always_scope`, session-only — `components/confirmation.go:338-348`), so the two surfaces now enforce different permission semantics for the same agent.
- **Failure scenario:** User allows `run_shell_command` "always" once for `ls`; a week later a prompt-injected page makes the agent run `rm -rf ~/projects` — no prompt, nothing in Settings shows the grant, the only way out is clearing site data.
- **Evidence:** lines above; `PermissionManager.tsx` and `SettingsPage.tsx` contain no reference to the key.
- **Fix:** Persist grants server-side via the existing `remember` flag (`/chat/confirm` already accepts it) with the backend's invocation scope, drop the localStorage bypass, and list/revoke grants in Settings → Permissions. At minimum scope the client list per session and show it in `PermissionManager`. 🔒 security-adjacent — tag @kovtcharov-amd.
- **Confidence:** High
- **Tracked:** none found

### 🔴 `NotificationService` calls `TrayManager.setNotificationCount`, which does not exist — badge never updates, and the agent crash-limit path throws inside an EventEmitter listener (sub-review)
- **Where:** `src/gaia/apps/webui/services/notification-service.cjs:319-323` (`_updateTrayBadge` → `this.trayManager.setNotificationCount(...)`, called from `:149,170,179`); `services/tray-manager.cjs` has no such method (`grep -rn setNotificationCount src/` → only the caller and the Jest mock `tests/electron/test_notification_service.js:78`); listeners registered at `notification-service.cjs:332` (`agent-notification`) and `:352` (`agent-crash-limit`), emitted from `agent-process-manager.cjs:745` and `:826`.
- **What:** Every notification path ends in a `TypeError`. On the normal path it is caught by the stdout handler and logged as "Non-JSON stdout", so the badge silently never updates and the notification is not persisted. On the crash-limit path the `emit` is inside the child `exit` handler with no try/catch, so the error escapes to `uncaughtException` → `main-safety-net.cjs:115` `process.exit(1)`.
- **Failure scenario:** A sidecar agent with `restartOnCrash` crash-loops → "GAIA crashed" dialog, whole app exits, backend + sidecars orphaned (no `will-quit` cleanup on `process.exit`).
- **Evidence:** the unit test only passes because the tray manager is mocked with the missing method (`test_notification_service.js:78,259,317,…`).
- **Fix:** Implement `TrayManager.setNotificationCount(n)` (tooltip/overlay badge), wrap listener invocations, and add a test that constructs the real `TrayManager` against the Electron mock.
- **Confidence:** High (method absence + mock verified; crash propagation traced from the emit site)
- **Tracked:** none found

### 🔴 Cold-start `gaia://hub/install/<id>` deep links are dispatched before the backend is listening, so "Open in GAIA" from the website fails whenever GAIA was closed (sub-review)
- **Where:** `src/gaia/apps/webui/main.cjs:1007` (`backendProcess = await startBackend()`), `:1034` (`processStartupDeepLinks()` → `handleDeepLink` → `confirmDeepLinkInstall` `:829-832` → `agentProcessManager.fetchCatalogEntry` — an HTTP fetch to the backend), `:1046` (`waitForBackend(STARTUP_TIMEOUT)` only after `loadApp`).
- **What:** The catalog fetch runs milliseconds after the Python backend is spawned; it gets `ECONNREFUSED`, and the fail-closed confirmation shows "Could not verify this agent". Only the already-running (`second-instance`) path works.
- **Failure scenario:** GAIA closed → click "Open in GAIA" on the hub → app launches → error dialog → nothing installed.
- **Fix:** Dispatch startup deep links after `waitForBackend()` resolves (and show a loud dialog if it does not), or make `fetchCatalogEntry` wait for `/api/health` first. Pin the ordering with a test in the `test_loadapp_query.mjs` style.
- **Confidence:** High (ordering unambiguous)
- **Tracked:** none found

### 🟡 Packaged Windows build falls back to `irm https://astral.sh/uv/install.ps1 | iex` and then to an unverified system `uv`, contradicting the file's own "dev-only, never fires for end users" contract (sub-review)
- **Where:** `src/gaia/apps/webui/services/backend-installer.cjs:1282-1332` (`if (IS_WINDOWS && !isDev) { … "irm https://astral.sh/uv/install.ps1 | iex" …}`, then `report(STAGES.ENSURE_UV, 100, "uv ready (system, unverified)")`); header comments `:72-78`, `:1205-1206`.
- **What:** A packaged build missing `vendor/uv/win-x64/uv.exe` (or a user who deleted `resources/vendor`) executes a remote PowerShell script with `-ExecutionPolicy Bypass`, no checksum, and then trusts whatever `uv` is on PATH — the exact hole the SHA-256 pinning in `installBundledUv` closes.
- **Fix:** In packaged builds a missing bundled `uv` is an `InstallError` ("installer is corrupt — re-download"); if a rescue must remain, gate it behind an explicit dialog and verify the download's SHA-256.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `gaia init` failure during first-run install is reported as "Lemonade Server setup complete" and the install is marked READY (sub-review)
- **Where:** `src/gaia/apps/webui/services/backend-installer.cjs:1579-1586` (`if (initResult.code !== 0) { log("Warning: gaia init exited with code … Continuing anyway."); } report(STAGES.GAIA_INIT, 100, "Lemonade Server setup complete");`).
- **What:** Model download / Lemonade install failures are hidden; the first chat then fails with a backend error the user has no context for. Violates the fail-loudly rule.
- **Fix:** Surface the failure in the progress dialog with the tail of the output and a Retry / "Skip model download" choice; persist `initFailed` so the renderer's onboarding can show it.
- **Confidence:** High
- **Tracked:** #3307 (same pattern elsewhere), no Electron issue

### 🟡 Docs promise auto-update, but shipped builds have no update channel (`NO_CHANNEL`), the feed accepts plain `http://`, and no Authenticode check is configured (sub-review)
- **Where:** `docs/guides/install.mdx:101-109` ("GAIA checks for updates automatically: 10 seconds after launch, every 4 hours"); `src/gaia/apps/webui/services/auto-updater.cjs:190-227` (feed resolved ONLY from `GAIA_UPDATE_FEED_URL` or `~/.gaia/update-config.json`, otherwise `STATES.NO_CHANNEL`); `:231` (`setFeedURL({provider:"generic", url, …})` with no `https:` requirement); `electron-builder.yml` (no `win.publisherName`; `grep publisherName` → nothing).
- **What:** Nothing in the repo sets the feed for end users, so every install pauses updates with "No update channel configured". If a user sets an `http://` feed (or a captive portal MITMs one), electron-updater's only integrity check is the sha512 in that same feed — arbitrary installer execution via `quitAndInstall`.
- **Fix:** Bake the R2 channel (`publish.url` in `electron-builder.yml` is already `https://hub.amd-gaia.ai/updates`) as the default; reject non-`https:` feeds loudly; set `win.publisherName` once Windows signing lands. Until then, fix the docs. 🔒 (feed half) — tag @kovtcharov-amd.
- **Confidence:** High for NO_CHANNEL + docs; Medium for the electron-updater signature behaviour (library not inspected — no `node_modules`)
- **Tracked:** #1729 (signing/provenance, partial)

### 🟡 Agent child stdin has no `error` listener — an EPIPE from the 30 s health ping to a just-died sidecar is an uncaught exception that exits the app (sub-review)
- **Where:** `src/gaia/apps/webui/services/agent-process-manager.cjs:858` (only `stdin.destroyed` is checked), `:870` (`entry.process.stdin.write(payload)`); `grep -n "stdin.on" agent-process-manager.cjs` → nothing.
- **What:** A write to a pipe whose reader just exited emits `'error'` on `child.stdin`; with no listener Node raises it as `uncaughtException` → `main-safety-net.cjs:115` `process.exit(1)`.
- **Fix:** `child.stdin.on('error', …)` after spawn; check `child.exitCode !== null` before writing.
- **Confidence:** Medium (standard Node semantics, not reproduced)
- **Tracked:** none found

### 🟡 Legacy Electron framework exposes a generic `invoke(channel, …args)` and an unvalidated `open-external-link` handler, and the docs still call it the shell that ships the Agent UI (sub-review)
- **Where:** `src/gaia/electron/src/preload/preload.js:17-20` (`invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args)`); `src/gaia/electron/src/services/base-ipc-handlers.js:92-95` (`ipcMain.handle('open-external-link', (e, url) => shell.openExternal(url))`); `docs/spec/electron-integration.mdx:26` ("currently shipping the flagship Agent UI (`webui`)") and `:16` ("Electron 31.0.0+" vs `^44` in `package.json`).
- **What:** No app under `src/gaia/apps/` uses this framework (the Agent UI uses `main.cjs`/`preload.cjs`; `example` has its own `main.js`), but it is documented as canonical and CI still tests it — the next app scaffolded from it inherits an IPC surface where any renderer script can open any URL or file dialog.
- **Fix:** Delete `src/gaia/electron/` and its tests, or allow-list the channels and validate the URL scheme, and rewrite the spec page around `main.cjs`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Fatal-path `process.exit()` skips `will-quit`, orphaning the backend and every sidecar (sub-review)
- **Where:** `src/gaia/apps/webui/main-safety-net.cjs:83,115` (`process.exit(2/1)` in `fatal()`), `main.cjs:1118-1122` (cleanup lives only in `app.on("will-quit")`, which does not fire on `process.exit`).
- **What:** Any renderer crash / uncaught exception (including 🔴 above) leaves `gaia chat --ui` and the sidecars running with no owner; only the backend pidfile sweep on the next launch reaps them, and it does not cover agents.
- **Fix:** Give `fatal()` an injected synchronous best-effort cleanup (`backendProcess.kill()`, kill each agent), and persist sidecar PIDs like the backend pidfile.
- **Confidence:** Medium
- **Tracked:** #2146 / #2474 cover the daemon side only

### 🟡 `cleanLLMJsonBlocks` deletes legitimate answer content: any `{…}` mentioning `"tool"`/`"thought"` within 50 chars, and *everything after* an unbalanced `{`
- **Where:** `src/gaia/apps/webui/src/components/MessageBubble.tsx:130-194` (`MARKERS = ['"thought"', '"answer"', '"tool"']`, `if (depth !== 0) { break; } // suppress partial/unclosed JSON block`, "thought-only and tool/tool_args blocks are dropped silently"), applied to every assistant message via `cleanToolCallContent` (`:242-274`).
- **What:** A developer asking "show me the tool-call JSON format" gets an answer whose example block vanishes from the rendered bubble; a message containing an unbalanced `{` (a code snippet, a shell one-liner) loses the rest of the message on screen. Copy still copies the raw content, so the transcript and the rendering disagree.
- **Failure scenario:** "Give me a Python dict with a `tool` key" → the model writes `{"tool": "search", "tool_args": {...}}` in prose → the rendered answer is empty/half.
- **Fix:** Only strip blocks that parse as a complete tool-call envelope (`tool` + `tool_args` keys, nothing else) and never truncate on an unbalanced brace once streaming has finished; rely on the server-side filter (`sse_handler.py`) as the primary path.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `stripBogusCodeFences` unwraps any fence whose tag is not in a hand-maintained list — `shell`, `console`, `cmd`, `bat`, `jsonc`, `mermaid`, `env`, `http`, `hcl`/`terraform`, `groovy`, `latex` blocks are rendered as Markdown prose
- **Where:** `src/gaia/apps/webui/src/components/MessageBubble.tsx:197-240` (`KNOWN_CODE_LANGS` and the replace that returns `inner.trim()` for anything else).
- **What:** A ```` ```shell ```` block with `# install deps` becomes an H1, `* glob` becomes a bullet, `<tag>` disappears, indentation is lost. `shell`/`console` are two of the most common fence tags models emit.
- **Fix:** Invert the rule — only strip fences whose tag is 1–2 characters and not a known language (the Qwen-Coder ```` ```i ```` case the comment describes), keep everything else.
- **Confidence:** High
- **Tracked:** none found

### 🟡 No Content-Security-Policy anywhere in the Agent UI (no `<meta>` in `index.html`, no `session.webRequest` header in `main.cjs`)
- **Where:** `grep -in content-security-policy src/gaia/apps/webui/index.html main.cjs vite.config.ts` → nothing.
- **What:** With `contextIsolation`/`nodeIntegration:false` the blast radius of a renderer XSS is limited, but a CSP is the standard second layer Electron's security checklist asks for, and the renderer only ever needs `connect-src 127.0.0.1/localhost`, `img-src` self/blob/data, no remote scripts.
- **Fix:** Add a strict meta CSP to `index.html` (Vite can inline hashes) and verify in `test_loadapp_query.mjs`.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `bin/gaia-ui.cjs --serve` binds on all interfaces (sub-review)
- **Where:** `src/gaia/apps/webui/bin/gaia-ui.cjs:255` (`server.listen(port, () => …)` — no host, unlike `dev-server.js`'s `127.0.0.1`).
- **Fix:** `server.listen(port, '127.0.0.1', …)`.
- **Confidence:** High
- **Tracked:** none found

### 🟢 ~1,900 lines of unreferenced renderer code: `AgentChat.tsx`, `AgentManager.tsx` (+`AgentCard.tsx`), `SettingsModal.tsx`, `stores/agentChatStore.ts`
- **Where:** `grep -rln "from './AgentChat'|SettingsModal|AgentManager|agentChatStore" src/` → only the files themselves (`SettingsPage.tsx:17` imports just `SettingsModal.css`).
- **What:** Dead IPC-chat and modal-settings code paths that still get maintained (and that the sub-review had to read).
- **Fix:** Delete, or note in `README.md` why they are kept.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `isErrorContent` heuristics style ordinary answers as errors
- **Where:** `MessageBubble.tsx:93-105` (`lower.includes('connection refused') || lower.includes('failed to fetch') || …`) → red "Something went wrong" banner (`:450-455`).
- **What:** An assistant answer that *explains* a connection-refused error to the user gets the error banner.
- **Fix:** Drive the banner from the SSE `error`/`agent_error` event (already available) rather than prose matching.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `saveTitle` has no error handling — a failed rename is an unhandled promise rejection and the title editor stays open with no message
- **Where:** `src/gaia/apps/webui/src/components/ChatView.tsx:1339-1346` (`await api.updateSession(...)` outside any try; `setEditingTitle(false)` never reached on throw).
- **Fix:** try/catch → toast, like `handleTogglePrivate` right below it.
- **Confidence:** High
- **Tracked:** none found

## Test gaps

- **No test exercises a multi-process agent.** `tui/test/mockagent/main.go` is a single Go process; every cancel/close/respawn test (`subprocess_test.go`, `chat/cancel_test.go`, `cli_run_test.go`) therefore proves the *opposite* of what the shipped PyInstaller one-file binary does (🔴 #1). A mock that `exec`s a grandchild holding the pipes would have caught it.
- `go test -race` is only run in CI (`build_tui.yml`); the concurrent-`Scan()` race in 🔴 #1 is not reachable by the current tests anyway.
- `preflight/local.go` `claudeCredentialConfigured` has no test for the walk-up vs. cwd distinction (🟡 above), and nothing pins that the Go search matches the Python `load_dotenv` search.
- `internal/ui/root` has no tests of its own (coverage comes only via `tui/test`), and `internal/gaiainit` has none at all — `Start`'s cancel path (kill the launcher → does `gaia init`'s Python child die?) is untested on Windows.
- `daemon/client.go:517` builds `/agents/<id>/ensure` without `url.PathEscape` (every other path escapes); unreachable today because `catalog.Get` gates ids, but nothing tests that guard.

- **Electron:** `test_notification_service.js` mocks a `TrayManager` method the real class lacks (🔴 above) — the canonical "mock proves the call, not its validity" case. `main.cjs` has no behavioural tests: `test_loadapp_query.mjs` / `test_electron_chat_app.js` regex-scan its source text, so `setWindowOpenHandler`, the deep-link/backend ordering and `cleanup()` are untested. `backend-installer-lock.test.cjs` tests three regex classifiers, not a lock (there is none between `gaia-ui` and Electron sharing `~/.gaia/venv`). No test rejects an `http://` update feed. `tests/electron/README.md:50-51` still describes "test_example_app.js (11 tests) / test_functional.js (11 tests)"; the directory has 24 test files. (sub-review, spot-checked)
- **Renderer:** no test covers `safeUrlTransform` with `/…` or `//host/…` inputs, `cleanLLMJsonBlocks` with a legitimate `{"tool": …}` example or an unbalanced brace, or `stripBogusCodeFences` with `shell`/`console`; no test for the localStorage always-allow bypass in `ChatView`; `consumeSSEResponse` has no test for a stream that closes mid-frame or without `done` (the synthesized `done` is untested). Issue **#3327** already tracks the missing backend↔TypeScript conformance check.

## Documentation gaps

- `tui/README.md` download table pinned to `0.23.0` (🟢 above).
- `docs/guides/terminal-hub.mdx:234` documents Esc as "cancel the turn"; it does not say the flagship agent is restarted (history, skills, grants and any `/model` switch lost) — and after 🔴 #1 that description is not even true for the released binary. Issue **#3140** already tracks that `--use-claude`, `--claude-model`, `--bypass-permissions`, `--dev` are undocumented on the docs site.
- `local.go:129-132` comment ("accepting a .env here cannot turn a launch into a false pass") contradicts the code (🟡 above).

- `docs/guides/install.mdx:101-109` describes working auto-update (10 s after launch, every 4 h); shipped builds sit in `NO_CHANNEL` (🟡 above). `GAIA_UPDATE_FEED_URL`, `GAIA_UPDATE_PRERELEASE`, `~/.gaia/update-config.json` and the `gaia://hub/install/<id>` deep link are documented nowhere under `docs/`.
- `docs/spec/electron-integration.mdx:26` says `src/gaia/electron/` is "currently shipping the flagship Agent UI (`webui`)" and `:16` "Electron 31.0.0+" — the Agent UI uses `main.cjs` and pins Electron `^44`. `docs/deployment/testing-electron.mdx:234,370` still reference `src/gaia/apps/jira/webui`, which no longer exists.
- `backend-installer.cjs:72-78` header ("DEV-ONLY … never fires for end users") contradicts the packaged-Windows rescue at `:1282`.
- `markdown.ts:26-29` says `mailto:` is acceptable because only developer-authored catalog content goes through `SafeMarkdown` — but `grep ReactMarkdown src/` shows the only consumer is `MessageBubble` (LLM output); the catalog-markdown path the comment describes does not exist.

## Improvement opportunities

- **Cooperative cancel over the control channel** (`gaia_control: cancel`) instead of kill-and-respawn — fixes 🔴 #1–#3 in one move and keeps the ~2.5 s warm-agent turn the catalog comment brags about.
- **Host-owned transcript for the subprocess transport**, mirroring `SSEClient.transcript` — makes `/clear` and restarts behave identically on both transports.
- `subprocess.go` `detectLemonadeURL` and `preflight/local.go` `probeLemonade` duplicate the port list and probe logic; one helper.
- `control/server.go` `send()` sleeps while holding `injectMu` (up to the 10 s delay budget) — a second `/keys` caller blocks silently; return 409 instead.
- `status.Sanitize`, `chat.sanitizeErrorText`, `cards.clean`, `chat.clean`, `devlog.scrubDisplayControls` are five near-identical scrubbers with slightly different rule sets (C1 range handled by two of them, bidi overrides by one). One package-level sanitizer with documented modes would close the gaps (e.g. `sanitizeErrorText` lets C1 controls and bidi overrides through into `RoleError` panels).

- **Electron:** one `openExternalSafe(url)` helper (scheme allow-list) used by `setWindowOpenHandler`, tray "Open in browser", the progress dialog and the legacy handler; make `cleanup()` idempotent and callable from `fatal()`; persist sidecar PIDs like the backend pidfile; narrow `_handleStdout`'s catch to `JSON.parse` so listener bugs surface with a stack; delete `src/gaia/electron/` and the four structure-scan Jest files.
- **Renderer:** move the always-allow list server-side (the `remember` flag already exists on `/chat/confirm`); replace the prose-based `isErrorContent` with event-driven state; the JSON/fence "cleaners" in `MessageBubble.tsx` and `ChatView.tsx` (`stripStreamingEnvelope`, `TOOL_CALL_JSON_SAFETY_RE`) are three overlapping heuristics for one problem — keep the server-side filter as the single source and reduce the client to fence handling; `DocumentLibrary` swallows all poll/refresh errors (`catch { /* ignore */ }` at `:144,190`) — surface at least once as a banner.

## High-impact feature opportunities

- **Real cancel for the flagship agent** (see above) — today "Esc" is the one thing a user reaches for when the agent is about to do something wrong, and it does not stop it.
- **Session persistence/resume for the TUI**: the SSE transport already has `session_id` (#2829); the subprocess transport has nothing, so a crash or Esc loses the conversation. A `~/.gaia/tui/sessions/*.jsonl` transcript replay would make both restarts and `gaia-tui --resume` cheap.
- **Control API is loopback-token-gated but has no client library in-repo besides the Python MCP** — a `gaia-tui control` subcommand (screen/keys/wait) would make the driving-the-tui skill and CI screenshots one-liners.

- **Ship a default update channel** — the whole updater (checks, pin/rollback UI, `VersionPicker`) is built and tested but unreachable; one baked URL turns it on.
- **Cold-start deep-link install** — the hub website's primary acquisition path; the ordering fix plus a "GAIA is starting… installing *X* in a moment" toast makes "Open in GAIA" reliable.
- **A permissions panel that shows and revokes standing grants** (both surfaces) — today the TUI has session-scoped `always` and the Agent UI has a hidden permanent list; a single backend grant store with a Settings view would make the permission model explainable.

## Checked and fine

- Control API (`control/server.go`, `paths.go`): binds 127.0.0.1 only, 256-bit random bearer compared with `subtle.ConstantTimeCompare`, discovery file written 0600 via O_EXCL temp + rename, port 4001 refused at both CLI and listener, `MaxBytesReader` 1 MiB + `DisallowUnknownFields` on every POST, wait loop takes the change channel before checking (no lost wakeups), `Stop` closes `done` so parked waits return, resize refuses sizes larger than the real terminal.
- Daemon client: two-check trust (pid alive + token-authed status probe answering our service id/pid), token rotation retried exactly once, MAJOR/MINOR contract gates, advisory lock for start (flock / LockFileEx), instance token never logged (`Instance.String` redacts).
- SSE client: exactly-one-terminal-event contract enforced (synthesised error on EOF/timeout/close), read-idle watchdog reset by heartbeats, cancel POST on its own background context, contract negotiation before sending optional fields, `url.PathEscape` on agent/run ids, superseded-turn handles cancelled.
- Event parsing: unknown/malformed canonical events surface as visible events, never dropped; legacy parser errors become status warnings; tool-error extraction handles string-encoded and truncated envelopes.
- Chat model: late `eventMsg`/`doneMsg`/`errMsg` deliveries scoped by channel identity / `turnSeq`; queued follow-ups restored to the composer on cancel; bypass needs a two-step confirm and is never persisted; confirmation `always` only offered with a deliverable channel and an agent-supplied scope; question `RequestID` checked before answering; clipboard image path written under `~/.gaia/paste` (inside the agent's sandbox) and swept after 7 days; `sanitizeErrorText`/`cards.clean` strip ANSI + C0 before agent text hits the terminal; image cards reject `data:image/svg+xml`.
- Preflight gate: every non-OK row carries a Disposition (pinned by test), unknown ≠ ok, optional mailbox row never blocks, `cancelBox` pointer so abandoning the gate cancels the probe/`gaia init` child, stale `ProceedMsg` for a different agent ignored, remedies resolved against the machine (no `lemonade-server serve` ghost command).
- `gaiainit.Check` treats only exit 1 as "not ready" (exit 2 from an old CLI is "unanswered", not a fresh install).
- Windows clipboard image reader: `LockOSThread`, `GlobalLock/Unlock` balanced, DIB decoder bounds-checks `need` before indexing, all-zero-alpha quirk handled.
- `go vet` clean; all Go tests green.

- **Electron:** `nodeIntegration:false`, `contextIsolation:true`, default sandbox, no `webSecurity` override (`main.cjs:504-508`); `preload.cjs` exposes only fixed channels with fixed arities (no generic `invoke`); `utils/apiBase.ts` validates the `?api=` query against a loopback allow-list and `index-query.cjs` round-trips it; `/api/files/open` (`files.py:286-350`) is home-restricted, rejects symlinks and only *reveals* files (`explorer /select,`), so `FilePathLink` clicks cannot execute a file; `deep-link.cjs` id regex + fail-closed confirmation; `installBundledUv` SHA-256/codesign + atomic rename; `dev-server.js` binds loopback; `build.py` uses array-form `npm` invocations, no `shell=True`.
- **Renderer:** `react-markdown` without `rehype-raw`, `disallowedElements` for script/iframe/object/embed/style, `javascript:`/`data:` hrefs neutralised; no `dangerouslySetInnerHTML`/`innerHTML` anywhere in `src/`; `AgentTerminal` renders stdout/stderr as text nodes; `consumeSSEResponse` synthesizes `done` on EOF, tolerates malformed frames, aborts cleanly; `useConnectorsSSE`/App session-events use exponential backoff and cancel on unmount; `ChatView` scopes stale stream callbacks by `currentSessionId` (#1580) and batches chunks via rAF; `agentChatStore` guards TOCTOU on session creation; clipboard copy falls back to `execCommand` for non-secure contexts.

## Hypotheses (unverified)

- `gaiainit.Start` cancel (Esc during first-run setup) kills only the `gaia.exe` launcher on Windows; whether the Python `gaia init` (and the Lemonade MSI it spawns) survives depends on the launcher (distlib/uv trampolines use a Job Object; a PyInstaller-frozen `gaia` would not). Not tested — see next section for a quick repro.
- Two goroutines calling `bufio.Scanner.Scan` concurrently (🔴 #1 step 3) would be flagged by `-race`; could not run it here (no cgo).
- `control.WriteInfo` relies on `os.Chmod(path, 0o600)` which is a no-op on Windows; `control.json` (with the bearer token) inherits the `%USERPROFILE%\.gaia\tui` ACL — fine for a per-user profile, worth a note in `doc.go`.
- On a pinned launch the updater may check the baked `app-update.yml` feed (`https://hub.amd-gaia.ai/updates`) that `electron-builder.yml:91-96` calls "inert" — `init()` with a saved pin skips `_applyForwardFeed` but still schedules `checkForUpdates()` (sub-review, not re-traced).
- electron-updater's generic provider may itself refuse non-`https` feeds in the pinned `^6.8` version; if so the `http://` half of the updater finding reduces to the missing `publisherName`.
- `bootstrapBackend`'s fast path trusts `findGaiaBin()` from PATH, which may be a different venv than the packaged one (sub-review).
