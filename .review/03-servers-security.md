# Review 03 — Servers, Daemon, Connectors, Security

Repo: amd/gaia @ 211f08c5 (v0.23.1 era). Reviewer dimension: SERVERS / DAEMON / CONNECTORS / SECURITY.
Status: COMPLETE. 48 findings (5 🔴, 26 🟡, 17 🟢) across the Agent UI backend, MCP/Telegram/schedulers, daemon/custody/sidecars, and connectors/API/shell/filesystem sub-areas.

## Scope covered
(see the full "## Scope covered" section at the end of this file)

## Findings
Lead-reviewer findings first; delegated sub-area reports follow, each with its own inventory, test-gap, doc-gap and checked-fine lists. Cross-cutting closing sections are at the end.

### [🔴] Agent UI backend: state-changing "simple request" POSTs are CSRF-able from any website (X-Gaia-UI guard applied to only 17 of ~45 mutating routes)
- **Where:** `src/gaia/ui/server.py:560-580` (CORS config), `src/gaia/ui/routers/connectors.py:106` `_require_ui_header` (the guard); unguarded examples: `src/gaia/ui/routers/tunnel.py:21` `start_tunnel`, `src/gaia/ui/routers/documents.py:331` `upload_document_blob`, `src/gaia/ui/routers/files.py:62` `upload_file`, `src/gaia/ui/routers/memory.py:963` `prune_memory`, `src/gaia/ui/routers/mcp.py:223` `start_agent_mcp_server`, `src/gaia/ui/routers/documents.py:557` `reindex_document`.
- **What:** The backend listens on `localhost:4200` with no auth when the tunnel is off and relies on browser CORS to keep other websites out. CORS only blocks *reading* responses; a POST with no body, a query-string-only body, or a `multipart/form-data` body is a "simple request" the browser sends without preflight. The project knows this — `_require_ui_header` exists precisely to force a preflight ("drive-by form POSTs from malicious pages cannot forge this header") — but it is wired on only the connectors / hub / agents / memory-reinitialize routes; the rest of the mutating surface is unguarded.
- **Failure scenario:** User has GAIA Agent UI running and visits any web page. The page auto-submits `<form method=POST enctype=multipart/form-data action=http://localhost:4200/api/documents/upload>` → an attacker-chosen `.md`/`.txt` is written to `~/.gaia/documents/` and **indexed into the user's RAG knowledge base** (persistent prompt injection — the content is later fed to the local LLM). Same trick with `POST /api/tunnel/start` (no body) makes the machine open a public ngrok tunnel; `POST /api/mcp/agent-server/start` (body optional) spawns an MCP server subprocess on port 8765; `POST /api/memory/prune?days=7` deletes memory older than 7 days; `POST /api/files/upload` plants files in `~/.gaia/chat/uploads` (served back from the app origin as `.html`/`.svg`). Exploitable **remotely** (any site the user visits); no tunnel needed.
- **Evidence:**
  - `connectors.py:106` `def _require_ui_header(request)` — docstring: "Custom request headers trigger a CORS preflight in browsers, so drive-by form POSTs from malicious pages cannot forge this header."
  - `grep -rn "dependencies=\[Depends(_require_ui_header)\]" src/gaia/ui | wc -l` → **17**
  - `tunnel.py:21` `@router.post("/api/tunnel/start") async def start_tunnel(tunnel=Depends(get_tunnel))` — no body
  - `documents.py:331` `async def upload_document_blob(file: UploadFile = File(...))` — multipart = simple request
  - `memory.py:963` `@router.post("/api/memory/prune") def prune_memory(days: int = Query(90, ge=7, le=365))` — query only
  - `mcp.py:223` `async def start_agent_mcp_server(body: Optional[StartAgentServerRequest] = None)`
  - No Host/Origin validation anywhere: `grep -rn -iE "TrustedHost|allowed_hosts" src/gaia/ui src/gaia/daemon src/gaia/api` → no matches. That also leaves the *read* side open to DNS rebinding (attacker hostname resolving to 127.0.0.1 → same-origin reads of `/api/sessions`, `/api/files/preview?path=~/.ssh/id_rsa`; see next finding).
- **Fix:** Enforce the `X-Gaia-UI` check in a middleware for every non-GET `/api/*` and `/v1/*` request (exempt only the OAuth loopback callback GET). Add `TrustedHostMiddleware(allowed_hosts=["localhost","127.0.0.1","[::1]", <tunnel host>])` and reject mutating requests whose `Origin` is not in the CORS allowlist — that closes DNS rebinding too. Add a route-table introspection test asserting every mutating route carries the guard so the next router can't forget it.
- **Confidence:** High
- **Tracked:** none found (`gh issue list` searches for "CSRF", "X-Gaia-UI", "DNS rebinding" → no open issues)

### [🔴] Tunnel auth is bypassable for the first seconds of every tunnel start (auth gate keys off `tunnel.active`, which stays False until the ngrok URL has been polled)
- **Where:** `src/gaia/ui/server.py:112-115` `TunnelAuthMiddleware.dispatch`; `src/gaia/ui/tunnel.py:301-308` `TunnelManager.active`; `src/gaia/ui/tunnel.py:407-417` `_start_unlocked`.
- **What:** The middleware enforces the bearer/cookie token only when `tunnel.active` is True, and `active` additionally requires `self._url is not None`. `_url` is set only after `_poll_ngrok_api()` returns (polls every 0.5 s, up to 15 s). ngrok forwards public traffic as soon as its control connection is up — before the local 4040 API lists the tunnel — so there is a window where remote requests reach the backend and the middleware passes them through with no token check.
- **Failure scenario:** User has a paid ngrok `--domain` (fixed, known hostname — `TunnelManager(domain=...)`) or a reused free-tier static domain and clicks "Start tunnel". An attacker polling that hostname gets requests through to `/api/sessions`, `/api/files/preview?path=…` etc. while `_url is None`. Also fires when a start *times out*: the ngrok process stays alive for the full 15 s poll before it is killed. Exploitable **remotely**; bounded by the window (~1–15 s per start) and hostname knowledge.
- **Evidence:**
  - `server.py:114-115` `if tunnel is None or not tunnel.active: return await call_next(request)`
  - `tunnel.py:303` `return (self._process is not None and self._process.poll() is None and self._url is not None)`
  - `tunnel.py:408` `self._process = subprocess.Popen(cmd, ...)` then `tunnel.py:416` `self._url = await self._poll_ngrok_api()` (up to 15 s later)
- **Fix:** Gate on "a tunnel process has been spawned and not stopped" (e.g. a `tunnel.auth_required` property = `_process is not None`), not on `_url`. `_token` is already generated before `Popen`, so enforcement can begin immediately. Add a test: `_process` set, `_url=None`, non-local request → 401.
- **Confidence:** High
- **Tracked:** none found

### [🟡] The tunnel token is a full read-token for the user's entire home directory (including `~/.gaia` OAuth/token stores, `~/.ssh`, `~/.aws`)
- **Where:** `src/gaia/ui/routers/files.py:509` `preview_file`, `files.py:150` `browse_files`, `files.py:606` `serve_local_image`; containment helper `src/gaia/ui/utils.py:395` `ensure_within_home`.
- **What:** Every file endpoint's only containment is "inside `Path.home()`", and `preview_file` reads *any* extension (`ext in TEXT_EXTENSIONS or size < 1 MB`) up to 200 lines. Whoever holds the tunnel token (scanned QR code, or the `gaia_tunnel_token` cookie on a shared/lost phone) can list and read dotfiles, private keys, and the connector credential store under `~/.gaia`. The docs pitch the tunnel as mobile chat access, not remote read of `$HOME`.
- **Failure scenario:** `GET https://<tunnel>/api/files/preview?path=/home/u/.ssh/id_ed25519` with the bearer token → returns the key. `GET /api/files/browse?path=/home/u/.gaia` lists the credential files. Combined with the CSRF/DNS-rebinding finding above this is also reachable without a tunnel.
- **Evidence:**
  - `files.py:527` `ensure_within_home(resolved)` is the only check before reading
  - `files.py:562` `if ext in TEXT_EXTENSIONS or stat.st_size < 1_000_000:  # Try text for < 1MB`
  - `utils.py:400` `resolved.relative_to(home)` — the whole containment rule
- **Fix:** Add a sensitive-roots denylist to `ensure_within_home` (`~/.gaia`, `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gcloud`, `~/.docker`, `~/.kube`, `~/.netrc`, `~/AppData/Local/ngrok`) — or better, a positive allowlist of Documents/Desktop/Downloads plus explicitly indexed folders — and restrict `preview_file` to `TEXT_EXTENSIONS`. Document in `docs/guides/agent-ui.mdx` exactly what a tunnel token grants.
- **Confidence:** High
- **Tracked:** none found

### [🟡] Windows dev box: 21 failed + 29 errored unit tests on `main` because the unit-test network guard blocks asyncio's socketpair fallback
- **Where:** `tests/unit/conftest.py:65` `_blocked_connect`; failing modules `tests/unit/connectors/test_account_type.py`, `test_activation_api.py`, `test_activations.py`, `test_agent_bridge.py`, `test_agent_mcps_router.py`, `test_context.py`, `test_e2e_smoke.py`, `test_activation_watcher.py`, `test_api.py`, `test_disconnect_clears_*.py`, `test_enable_disable.py`.
- **What:** On Windows, `asyncio.new_event_loop()` builds its self-pipe with `socket._fallback_socketpair`, which does a real `connect()` to 127.0.0.1. The conftest guard raises for that, so every test that calls `asyncio.run()` or gets a fresh pytest-asyncio loop errors in setup. Linux CI has a native `socketpair` and never sees it — the suite is green there and red for every Windows contributor, the platform GAIA primarily targets.
- **Failure scenario:** `.venv\Scripts\python.exe -m pytest tests/unit/connectors -q` on Windows → `21 failed, 218 passed, 9 skipped, 29 errors` (run on this checkout, 2026-09-02).
- **Evidence:**
  - `..\Lib\asyncio\proactor_events.py:786: in _make_self_pipe` → `..\Lib\socket.py:623: in _fallback_socketpair   csock.connect((addr, port))` → `tests\unit\conftest.py:65: in _blocked_connect`
  - `E   ConnectionError: Unit tests must not make real network connections (mark the test @pytest.mark.allow_network if it uses a local socket)`
- **Fix:** In `_blocked_connect`, permit loopback connects originating from `socket._fallback_socketpair` (inspect the calling frame) or pre-patch `socket.socketpair` under the guard. Add a Windows runner to the unit-test workflow so this can't regress silently.
- **Confidence:** High
- **Tracked:** none found

### [🟡] Tool bodies run in a bare `threading.Thread`, dropping the agent-identity contextvar — any tool that calls `get_access_token()` without an explicit `agent_id` silently skips the per-agent grant check
- **Where:** `src/gaia/agents/base/agent.py:3179` `_call_tool_bounded`; `src/gaia/agents/base/agent.py:4463` `process_query` (enters `_agent_identity_context`); `src/gaia/connectors/api.py:187-195` `_check_grant_and_scopes`; `src/gaia/connectors/context.py:16-21` (design assumption).
- **What:** The connectors grant model binds the agent id in a `ContextVar` in `process_query` and documents that "the sync→async bridge relies on ThreadPoolExecutor inheriting the worker thread's context". But #1591 later moved every tool invocation into a fresh `threading.Thread`, and Python threads start with an **empty** context (no `copy_context()`), so inside a tool `current_agent_id()` is `None`. `_check_grant_and_scopes` treats `None` as "BYPASSES the per-agent grant check" (its own docstring, step 3). The grant UI in Settings → Connectors therefore does not gate tools that use the implicit-identity path.
- **Failure scenario:** A hub skill / custom tool does `asyncio.run(get_access_token(provider="google", scopes=[...]))` (the documented usage) while the user has *revoked* that agent's Google grant. Expected: `AuthRequiredError(AGENT_NOT_GRANTED)`. Actual: the token is returned; only the coarse OAuth scope check runs. Exploitable **locally** by any tool body the agent loads (third-party skills); no remote vector.
- **Evidence:**
  - `agent.py:3179` `worker = threading.Thread(target=_target, name=f"tool:{tool_name}", daemon=True)` — no `contextvars.copy_context().run(...)`
  - `api.py:187` `resolved_agent = agent_id if agent_id is not None else current_agent_id()` / `api.py:191` `if resolved_agent is not None: if not check_agent_grant(...)` — check skipped when `None`
  - `context.py:16-21` "``Agent.process_query`` runs in a ``ThreadPoolExecutor`` worker … inherits the worker thread's context — see the bridge test in ``test_agent_bridge.py``" — the bridge test covers the executor, not the bare thread from #1591
  - Blast radius today is limited: every in-tree caller passes `agent_id` explicitly (`hub/agents/email/.../gmail_backend.py:1237`, `calendar_backend.py:268`, `mailbox_state.py:339`, `src/gaia/daemon/forward.py:122`). The bug bites the documented public path for third-party skills and any future in-core tool that follows the docs.
- **Fix:** In `_call_tool_bounded` run the target under `contextvars.copy_context().run(...)`; and make the connectors gate **fail closed**: when `current_agent_id()` is `None` and no explicit `agent_id` was passed, raise an actionable error instead of skipping the grant check (the "explicit opt-out" should be a named kwarg like `agent_id=NO_AGENT`, not the absence of context). Add a unit test that calls a tool through `_call_tool_bounded` and asserts `current_agent_id()` is preserved.
- **Confidence:** High
- **Tracked:** none found (`gh issue list --search "contextvar grant"` → none)


---
## Sub-area: MCP bridge / MCP clients / Telegram / schedulers
(Delegated deep-read; the 🔴 and the Telegram-allowlist / port-collision findings were independently re-verified against the source by the lead reviewer.)

### [🔴] MCP bridge answers every origin with `Access-Control-Allow-Origin: *` and runs unauthenticated by default — any website can drive the local model/agent and read the replies
- **Where:** `src/gaia/mcp/mcp_bridge.py:436-443` `do_OPTIONS`, `:453` `send_json`, `:505` (auth optional)
- **What:** Every CORS preflight and every data response carries `ACAO: *`, and `gaia mcp start` defaults to no `--auth-token` ("Auth: none"). Localhost binding does not help because the request originates in the user's own browser.
- **Failure scenario:** User runs `gaia mcp start`, then visits a malicious page. The page does `fetch("http://localhost:8765/chat", {method:"POST", headers:{"Content-Type":"application/json"}, body:…})` and **reads the answer** — including the shared conversation history (next finding) and any RAG context the AgentSDK loads. Exploitable **remotely** (drive-by) with no tunnel.
- **Evidence:** live `OPTIONS /chat` with `Origin: http://evil.example` → `200`, `Access-Control-Allow-Origin: *`, `Access-Control-Allow-Headers: Authorization, Content-Type`; code `self.send_header("Access-Control-Allow-Origin", "*")` at :440 and :453; `tests/unit/test_mcp_bridge_auth.py:227 test_cors_preflight_stays_open_and_allows_authorization` pins this as intended.
- **Fix:** Reflect only an allow-listed origin (or none) when no token is configured; drop `ACAO: *` from data responses; or require a token whenever CORS is on. Flip the test that enshrines the open preflight. Related: #2951 narrows wildcard CORS on the EMR dashboard / base agent server — the bridge should ride the same fix.
- **Confidence:** High
- **Tracked:** #2951 (adjacent — EMR/base agent server CORS), none for the bridge itself

### [🟡] Bridge `/chat` is one shared conversation for every caller
- **Where:** `src/gaia/mcp/mcp_bridge.py:173-182` `_execute_chat`
- **What:** One `AgentSDK` is lazily created and reused for every request, so all clients (n8n, LAN callers with `--host 0.0.0.0`, the drive-by page above) share one history.
- **Failure scenario:** Two integrations talk to the bridge; each sees the other's prior turns. With the CORS hole, a web page reads whatever the user's other tooling said.
- **Evidence:** `if self.chat_sdk is None: … self.chat_sdk = AgentSDK(config=config)` then `self.chat_sdk.send(query)` with no session key.
- **Fix:** Key sessions on a client-supplied `session_id` (or the token/peer), or make `/chat` stateless like `/llm`.
- **Confidence:** High — **Tracked:** none found

### [🟡] Telegram allowlist is opt-in: the default bot answers anyone on Telegram, indexes their uploads into the user's global RAG library, and the guide contradicts the code
- **Where:** `src/gaia/messaging/telegram.py:76-79` `_allowed`, `src/gaia/cli.py:1625-1626`; `docs/guides/telegram-adapter.mdx:61,71`
- **What:** With no `--allowed-users`, `_allowed()` returns True for every Telegram user; each message is an `AgentSDK.send_stream` against the local model; documents go through `ingest_document_to_rag` → `RAGSDK(RAGConfig()).index_document` (global index); photos hit the VLM. No rate limit. The guide says off-list users are "ignored" while the code replies `UNAUTHORIZED_REPLY`.
- **Failure scenario:** User follows Quick Start (`gaia telegram start --token …`); a stranger finds the bot, uploads PDFs → they land in the owner's RAG library and surface in the owner's future answers; or floods the bot to pin the local GPU. **Remote**, limited to chat/RAG privileges (AgentSDK carries no shell/file tools — verified in `src/gaia/chat/sdk.py`).
- **Evidence:** `telegram.py:77 if not self.allowed_users: return True`; `test_empty_allowlist_intentionally_allows_all` pins it; `ingest_document_to_rag` uses default `RAGConfig()`.
- **Fix:** Refuse to start without `--allowed-users` (or require explicit `--allow-all`), scope Telegram ingestion to a per-user index, fix the guide wording.
- **Confidence:** High — **Tracked:** #3239 (docs wording), #2062 (no e2e coverage)

### [🟡] `gaia telegram start --background` never polls (daemon threads die with the CLI)
- **Where:** `src/gaia/messaging/telegram.py:313-337`, `src/gaia/cli.py:3258-3266`
- **What:** Background mode spawns the health server and `app.run_polling()` on `daemon=True` threads and returns; the CLI exits and kills both. PTB's `run_polling` also needs the main thread.
- **Failure scenario:** `gaia telegram start --token X --background` writes `~/.gaia/telegram.pid` with a dead PID; `gaia telegram status` reports "PID file exists, but health check failed".
- **Evidence:** `poll_thread = threading.Thread(target=_run_polling, daemon=True); poll_thread.start()` … `return` at :337.
- **Fix:** Spawn a detached child process (as `handle_mcp_start` does) whose main thread runs `run_polling()`.
- **Confidence:** High — **Tracked:** #3133

### [🟡] Telegram health server hard-codes `127.0.0.1:8765` — same default port as the MCP bridge and the Agent-UI MCP server
- **Where:** `src/gaia/messaging/telegram.py:308`; `src/gaia/cli.py:2383` (bridge `--port 8765`); `src/gaia/mcp/servers/agent_ui_mcp.py:42` `MCP_DEFAULT_PORT = 8765`; `src/gaia/cli.py:1656` (`--health-port 8765`)
- **What:** Three services default to 8765; `tui_mcp.py:95` even documents "8766 is the MCP bridge", which is wrong (bridge = 8765; `gaia mcp agent` = 8766).
- **Failure scenario:** Bridge running → Telegram health thread raises `OSError: address in use` inside a daemon thread (swallowed); `gaia telegram status` then hits the bridge's 404 on `/healthz` and reports the adapter unhealthy.
- **Evidence:** `HTTPServer(("127.0.0.1", 8765), HealthHandler)`; no `start --health-port` flag.
- **Fix:** Own default port + `start --health-port`; fix the `tui_mcp.py` comment.
- **Confidence:** High — **Tracked:** none found

### [🟡] Bridge handler drops the connection (no HTTP response) on any unsupported method or malformed `Content-Length`
- **Where:** `src/gaia/mcp/mcp_bridge.py:463` `log_message`, `:344` `do_POST`
- **What:** `log_message` does `"/health" not in args[0]`, but `send_error` passes an `HTTPStatus` as `args[0]` → `TypeError` before the error response is written. `int(self.headers.get("Content-Length", 0))` is unguarded.
- **Failure scenario:** `curl -X PUT http://localhost:8765/tools` → "Remote end closed connection without response" + server traceback. Same for `Content-Length: abc`.
- **Evidence:** live repro: `TypeError: argument of type 'HTTPStatus' is not iterable` at :463; `ValueError: invalid literal for int()` at :344.
- **Fix:** `if args and isinstance(args[0], str) and "/health" in args[0]: return`; wrap the `Content-Length` parse and reply 400.
- **Confidence:** High — **Tracked:** none found

### [🟡] `gaia schedule add` accepts an invalid cron; every later command then crashes
- **Where:** `src/gaia/cli.py:3071-3080`, `src/gaia/schedule/store.py:314-321`, `src/gaia/schedule/daemon.py:67,98`
- **What:** Nothing validates `--cron` at add time; `list`/`show`/`run`/`daemon` all call `CronTrigger.from_crontab`, and `list`/`daemon` iterate every schedule, so one bad row breaks all of them.
- **Failure scenario:** `gaia schedule add --name x --cron "every 5 minutes" --prompt hi` succeeds; `gaia schedule list` → `ValueError: Wrong number of fields; got 3, expected 5`; `gaia schedule daemon` refuses to start.
- **Evidence:** scratch run reproduced both errors.
- **Fix:** Validate in `Schedule.__post_init__` (or `add`); have `build_scheduler` skip-and-log a bad row.
- **Confidence:** High — **Tracked:** none found

### [🟡] `schedules.toml` writes are not atomic and CLI/daemon race on it
- **Where:** `src/gaia/schedule/store.py:404-408` `save`, `:450-462` `mark_run`; `daemon.py:41-53`
- **What:** `save` truncates in place (`open(path,"wb")`); the daemon rewrites the whole file after every fire (`mark_run` = load→mutate→save, no lock) and only reads the store at start.
- **Failure scenario:** Daemon fires job A and rewrites while the user runs `gaia schedule add B` → B silently lost, or a half-written file makes `tomllib` fail on next load.
- **Fix:** temp file + `os.replace`; file lock around read-modify-write; daemon re-reads on mtime change.
- **Confidence:** High — **Tracked:** none found

### [🟡] Three schedulers coexist with different semantics and no shared ledger
- **Where:** `src/gaia/schedule/` (APScheduler cron, TOML), `src/gaia/ui/scheduler.py` (asyncio timers, `scheduled_tasks` SQLite), `src/gaia/daemon/scheduler/` (`DaemonClock`, `daemon_jobs` SQLite)
- **What:** `daemon/scheduler/models.py:291-294` names the UI Scheduler and the `gaia schedule` CLI as sources to reconcile, but `reconcile_jobs` is only called for the email adapter. Different storage, interval grammar, sinks and privilege (`AgentSDK` plain chat vs `ChatAgent` with confirmation tools denied).
- **Failure scenario:** "daily at 9am" in the UI + `0 9 * * *` in the CLI → two agents fire unaware of each other; the UI one dies with the backend, the CLI one only runs while `gaia schedule daemon` is open; nobody can list "all my schedules".
- **Fix:** Route UI `create_task` and `gaia schedule add` into `daemon_jobs` (a `"prompt"` kind + executor); make the legacy drivers read-only views.
- **Confidence:** High — **Tracked:** #2156 / #2379 (daemon clock); none for wiring UI/CLI into it

### [🟡] UI scheduler computes fixed-time schedules in UTC and double-computes windowed `next_run_at`
- **Where:** `src/gaia/ui/scheduler.py:554-623` `compute_next_run`, `:1031`, `:1141`
- **What:** `datetime.now(timezone.utc)` + `now.replace(hour=hour…)` with no tz conversion → "daily at 9am" fires at 09:00 UTC. Windowed schedules compute `next_dt` from *now* before and after each run, so the persisted `next_run_at` and the actual sleep disagree.
- **Failure scenario:** PST user sets "daily at 9am" → fires at 1–2 am local.
- **Fix:** Store an IANA tz with the config; compute `next_run_at` once and sleep until it.
- **Confidence:** High (tz) / Medium (drift) — **Tracked:** none found

### [🟡] Telegram sink token persisted in clear text in `schedules.toml`
- **Where:** `src/gaia/schedule/sinks.py:247`, `store.py:334-335`, `docs/reference/cli.mdx:2413`
- **What:** `sink_args.token` (documented) is written verbatim to `~/.gaia/schedules.toml` with default perms.
- **Evidence:** scratch run wrote `[schedules.t.sink_args] token = "123:SECRET"`.
- **Fix:** Drop `sink_args.token` (env or keyring only), or `chmod 600` the store and redact on `show`.
- **Confidence:** High — **Tracked:** none found

### [🟡] `gaia telegram stop --force` is documented but a no-op
- **Where:** `src/gaia/cli.py:1638-1642` (arg), `:3268-3296` (handler never reads `args.force`); `docs/reference/cli.mdx:1013`, `docs/guides/telegram-adapter.mdx:84`
- **Fix:** Implement (SIGKILL after a wait) or delete the flag and both doc rows.
- **Confidence:** High — **Tracked:** none found

### [🟢] Telegram: edited messages / channel posts crash the handlers (fails closed, but noisy)
- **Where:** `src/gaia/messaging/telegram.py:59-62`, `:90`, `:270` — `filters.ALL & ~filters.COMMAND` also receives `edited_message`/`channel_post`, where `update.message`/`effective_user` are `None` → `AttributeError` before any model call.
- **Fix:** `filters.UpdateType.MESSAGE & ~filters.COMMAND`, or `effective_message` + guard. Add an allowlist test with an `edited_message` update.
- **Confidence:** Medium (PTB not installed in the review venv) — **Tracked:** none found

### [🟢] Telegram downloads use predictable, unbounded temp paths and never clean up
- **Where:** `src/gaia/messaging/telegram.py:97-99, 115-117` — `tempfile.gettempdir()/gaia_telegram_<file_id>`, no size cap, no extension check, never deleted.
- **Fix:** `mkstemp`, enforce `file.file_size` cap, delete after ingest. — **Confidence:** High — **Tracked:** none found

### [🟢] Bridge default `--base-url` and `mcp.json` ignore `LEMONADE_BASE_URL`
- **Where:** `src/gaia/mcp/mcp_bridge.py:60, 575`; `src/gaia/mcp/mcp.json:342` — hard-coded `http://localhost:13305/api/v1` while every other entry point resolves the env var.
- **Fix:** Default `None` and resolve via the shared Lemonade URL helper. — **Confidence:** High — **Tracked:** none found

### [🟢] Bridge `/status` and `/llm` error bodies leak internals; unbounded POST body
- **Where:** `mcp_bridge.py:150-152, 193-195` (`{"error": str(e)}`), `:348` (`self.rfile.read(content_length)` with no cap)
- **Fix:** Generic message + correlation id; cap body size (~1 MiB). — **Confidence:** High — **Tracked:** none found

### [🟢] `agent_ui_mcp.take_screenshot` swallows `SetForegroundWindow` failure
- **Where:** `src/gaia/mcp/servers/agent_ui_mcp.py:494-497` `except Exception: pass` — captures the wrong window silently.
- **Fix:** Narrow to `pywintypes.error`, log, report `foreground: false`. — **Confidence:** High — **Tracked:** none found

**Silent-fallback inventory (MCP / messaging / schedule):** 37 broad `except Exception` in sub-scope; discarding ones: `mcp/servers/agent_ui_mcp.py:496-497, 765-766`; `mcp/client/transports/stdio.py:275-276`; `mcp/mixin.py:422-423` (`__del__`, acceptable); `mcp/external_services.py:594-597` (`list_tools` → `[]`), `:565-567` (boundary, OK); `mcp/context7_cache.py:152-154, 165-167, 236-238, 290-292, 306-307, 317-318` (cache reset, acceptable); `mcp/mcp_bridge.py:96-97, 118-119` (server continues with fewer tools); `messaging/telegram.py:201-203` (`finally: pass` dead code), `:333-335`; `cli.py:3281-3283, 3287-3289, 3312-3313` (telegram pid/health); `ui/scheduler.py:1076-1084, 1096-1102, 1126-1132` (logged, documented non-fatal). `schedule/*` and `daemon/scheduler/*`: **zero** swallowed exceptions.

**Sub-area test gaps:** no bridge test for non-GET/POST, malformed `Content-Length`, oversized bodies, or shared-`chat_sdk` session; `test_cors_preflight_stays_open_and_allows_authorization` pins the insecure default. `test_telegram_allowlist.py` (4 tests) only uses plain `message` updates; `test_telegram_background.py` passes only because `GAIA_TEST_MODE` short-circuits before the daemon threads (the path #3133 says is broken); `python-telegram-bot` absent in the unit env so handlers never run against real `Update` objects (#2062). No test that `gaia schedule add` rejects a bad cron; no atomicity test for `TomlScheduleStore`; `runner.fire` only tested with a mocked `AgentSDK`. On Windows `tests/unit/test_scheduler.py` + `test_scheduler_api.py` error at setup (41 + 55) from the socketpair guard (see the conftest finding above). Nothing asserts UI/CLI schedules reconcile into the daemon clock — because they don't.

**Sub-area doc gaps:** `docs/guides/telegram-adapter.mdx:71` "ignored" vs code's refusal reply (#3239); `:46` "no environment variable for the token" vs `sinks.py` reading `GAIA_TELEGRAM_TOKEN`; `:84` + `cli.mdx:1013` document the no-op `stop --force`; `:53` health endpoint on 8765 collides with `cli.mdx:1233` bridge default; `cli.mdx:2346` "hand-editable" `schedules.toml` (hand edits lost to the race); `tui_mcp.py:95` wrong bridge port; `src/gaia/mcp/n8n.json` posts to `/jira` / tool `gaia.jira` which no longer exist (Jira agent deleted); `docs/sdk/infrastructure/mcp.mdx` never says `AgentMCPServer` is unauthenticated or that `--host 0.0.0.0` exposes every agent tool to the LAN; `cli.mdx:1249` shows `gaia mcp start --host 0.0.0.0` without `--auth-token`.

**Sub-area checked and fine:** `StdioTransport` always `shell=False`, legacy single-string commands tokenised with `shlex` and shell operators rejected, `.cmd` shims via `shutil.which` — no injection from `mcp_servers.json`. `_resolve_keyring_refs` refuses non-`gaia.connections` keyring services and fails closed. `mcp_tool_requires_confirmation` fails closed. Bridge auth: constant-time compare, token passed to the child via env not argv, `/health` the only public path, `resolve_bind_host` warns on wildcard without a token. `tui_mcp.py` refuses non-loopback hosts from `control.json`, disables proxies, redacts the token, validates pid+service id. `agent_ui_mcp._normalize_error` strips the backend URL. UI scheduled executor denies confirmation-gated tools and refuses to run while a tunnel is active; `Scheduler` bounds concurrency. `schedule/sinks.py` sinks raise actionable errors; AppleScript escaped; `notify-send` argv. `daemon/scheduler`: atomic `pending→firing` claim, ledger UNIQUE key, loud failure on unknown kinds. `context7_cache` file names md5-hashed, no traversal. `MCPConfig._read_servers` raises on corrupt JSON.


---
## Sub-area: daemon / custody / sidecars / caller_auth / email sidecar
(Delegated deep-read; the custody-secret-in-env, stale-secret-file, and 401-path-disclosure claims were re-verified against the source by the lead reviewer.)

### [🟡] Re-ensuring a crashed sidecar leaves the previous launch-secret file on disk, still holding a valid token
- **Where:** `src/gaia/daemon/sidecars/manager.py:674-691` `_start_locked`, `:589-634` `_write_secret_file`, `:254` (`auth_token` minted once per manager)
- **What:** When a sidecar dies and the daemon respawns it, the manager writes a *new* 0600 secret file but never deletes the old one; because `auth_token` is generated once per manager instance, the leftover file holds the exact token the *new* sidecar authenticates with (relay bearer + broker credential).
- **Failure scenario:** sidecar crashes → `is_running` False → `start()` → `_spawn_process` → `_write_secret_file` overwrites `self._secret_path`; the previous `gaia-<id>-secret-*/launch-secret` under TMPDIR persists until the next clean shutdown (which cleans only the latest path) — potentially forever, one live-credential file per crash. **Local only** (0600/DACL) but the "removed on sidecar exit" contract in `caller_auth.py:139-141` is false.
- **Evidence:** scratch repro (fake Popen, exit code 1, `start()` again): `FIRST STILL ON DISK AFTER RE-ENSURE: True  same token: True`. `_start_locked` only guards `if self.is_running: return`; `_cleanup_secret_file` runs only from `_shutdown_locked`/spawn-failure.
- **Fix:** call `_cleanup_secret_file()` (+ `_fire_reaped()`) at the top of `_start_locked` when `self._proc is not None and not self.is_running`; rotate `auth_token` per spawn. Add the crash-then-restart case to `test_agent_sidecar_manager.py` (only clean shutdown is tested, lines 478-486).
- **Confidence:** High — **Tracked:** none found

### [🟡] The custody secret is always delivered through the bare process environment, defeating the #2149 file-delivery posture
- **Where:** `src/gaia/daemon/sidecars/manager.py:463-470` `_spawn_process`, vs `:540-587` `_apply_secret_delivery`
- **What:** The *launch* secret was moved to a 0600/DACL file (#2149, #2250) because bare env is "visible to local process inspection". The `/host/v1` custody secret — read/write access to the agent's memory, sessions, RAG corpus and audit log — is injected as `GAIA_HOST_CUSTODY_SECRET=<secret>` unconditionally, even on the "file" leg, and is inherited by the sidecar's children.
- **Failure scenario:** `ps eww`, `/proc/<pid>/environ`, Windows process inspectors, crash reporters, or a sidecar that logs `os.environ` expose a live custody credential while the launch token in the same process is protected. **Local only**, same user.
- **Evidence:** `spawn_env[CUSTODY_URL_ENV_VAR] = self.custody_url; spawn_env[CUSTODY_SECRET_ENV_VAR] = self.custody_secret` with no leg negotiation; the deprecation warning at `:578-587` fires only for the launch token.
- **Fix:** put the custody secret in the same owner-only secret file (or sibling) and pass a `..._FILE` path; teach `select_custody_provider` to read a file env var first, mirroring `caller_auth.config_from_env`.
- **Confidence:** High — **Tracked:** none found (#2149 / #2153 closed; launch token only)

### [🟡] On Windows the daemon client token in `instance.json` (and `daemon.log`, `sidecars.json`, `custody.db`) gets no ACL hardening — only the launch-secret file does
- **Where:** `src/gaia/daemon/paths.py:99-132` `atomic_write_json(mode=0o600)`, `instance.py:58-65`, `client.py:114-120`, vs `sidecars/manager.py:113-193` `_lock_down_windows_acl` (only called from `_write_secret_file`, `manager.py:605,622`)
- **What:** #2250 established `chmod 0600` is inert on NTFS and added an owner-only DACL — but only for the sidecar launch secret. The *daemon client token* (can install/uninstall third-party agents, relay to any sidecar, forward OAuth tokens, stop the daemon) is written with inert mode bits and inherits whatever ACL `%USERPROFILE%\.gaia\host` has.
- **Failure scenario:** default profile ACLs protect it; on shared machines / lab boxes / a `GAIA_DAEMON_HOME` on a shared volume, any local user reads the token and gets full daemon authority incl. mailbox access via the relay. **Local only.**
- **Evidence:** `instance.py:6-7` "It is written mode 0600 via temp-file-then-rename"; `paths.py:85-86` "Mode bits are meaningless on Windows and skipped there"; `grep _lock_down_windows_acl src/gaia` → only `manager.py`.
- **Fix:** move `_lock_down_windows_acl` into `paths.py` and apply in `ensure_host_dir()` and `atomic_write_json()`; add a Windows test next to `test_daemon_secret_acl.py::test_real_lockdown_leaves_only_the_current_user` for `instance.json`.
- **Confidence:** High — **Tracked:** none found

### [🟡] The v0→v1 custody migration copies legacy DBs to paths nothing reads
- **Where:** `src/gaia/daemon/migrate.py:115-122` (`custody_sessions_db`, `custody_memory_db`), `:40-42` docstring, vs `src/gaia/daemon/paths.py:48-56` `custody_db_path` = `host/custody.db`, `custody/store.py`
- **What:** `run_migrations()` snapshots `~/.gaia/chat/gaia_chat.db` → `host/custody/agents/chat/sessions.db` and `~/.gaia/memory.db` → `host/custody/memory/user/memory.db`, then stamps schema v1. `CustodyStore` opens `host/custody.db` with a different schema and never reads those copies.
- **Failure scenario:** every legacy install now carries two never-read SQLite copies; the stamp is already at v1, so the real import needs a new step, and the docstring ("#2153 reads from the same layout") misleads whoever writes it. Non-destructive today.
- **Evidence:** `grep custody_sessions_db\|custody_memory_db src hub` → only `migrate.py`; `CustodyStore.__init__` connects to `custody_db_path()`.
- **Fix:** wire `CustodyStore` to ingest the snapshots in a v2 step, or drop the copy until there is a consumer and fix the docstring.
- **Confidence:** High — **Tracked:** none found (#2153 closed)

### [🟡] The daemon unit suite does not run on Windows: 157 of 746 tests fail, every daemon HTTP-route test included
- **Where:** `tests/unit/conftest.py:64-74` (same socketpair guard as the conftest finding above), `tests/unit/test_daemon_sidecar_dev_gate.py` (POSIX-only `\fake\checkout-a` paths → `DevSrcDirResolutionError`), `tests/unit/test_agent_sidecar_manager.py` (`_FakeProc` patch of `subprocess.Popen` but `_shutdown_locked` calls `subprocess.run(["taskkill", …])` on Windows → `TypeError: '_FakeProc' object does not support the context manager protocol`)
- **Evidence:** `157 failed, 571 passed, 18 skipped`; `test_daemon_hub_routes.py` 47, `test_daemon_agents_routes.py` 26, `test_daemon_custody.py` 19, `test_email_sidecar_router.py` 15, `test_daemon_broker_routes.py` 14, `test_daemon_sidecar_dev_gate.py` 10, `test_agent_sidecar_manager.py` 9, `test_daemon_forward.py` 8 … Traceback: `anyio/_backends/_asyncio.py:2480 … socket.socketpair() → socket.py:623 _fallback_socketpair → tests/unit/conftest.py:65 _blocked_connect` (every `TestClient` request → 500).
- **Fix:** fix the guard (see above); build fake paths with `tmp_path`; patch `subprocess.run` alongside `Popen` in the manager tests.
- **Confidence:** High — **Tracked:** none found

### [🟢] Unauthenticated 401 responses disclose the user's home directory path
- **Where:** `src/gaia/daemon/app.py:37` `build_require_token`, `broker_routes.py:82-85` — `where = f"the client token in {instance_path()}"` computed before the token is checked, so any local process or drive-by browser `fetch` to `127.0.0.1:<port>` (no Host/Origin check on the daemon, unlike `caller_auth.HostOriginMiddleware`) learns the OS username / profile path.
- **Fix:** say `~/.gaia/host/instance.json` in the 401; mount `HostOriginMiddleware` on the daemon app too. — **Confidence:** High — **Tracked:** none found

### [🟢] Broker lease `timeout` is taken from the body unvalidated
- **Where:** `src/gaia/daemon/broker_routes.py:149`, `broker.py:170` — a string → `TypeError` → 500; `null` → wait forever, parking a threadpool thread per request (anyio default 40) → all `run_in_threadpool` daemon routes stall. Same-trust caller, but violates the route's 422 contract.
- **Fix:** validate as positive finite number with an upper bound. — **Confidence:** High — **Tracked:** none found

### [🟢] Broker `release` does not verify the caller owns the lease
- **Where:** `src/gaia/daemon/broker_routes.py:180-188`, `broker.py:226-249` — `holder` recorded but never compared; a leaked `lease_id` lets another process release it and race-evict the model.
- **Fix:** reject `release` when `caller != lease.holder` (409). — **Confidence:** High — **Tracked:** none found

### [🟢] Sidecar stdout/stderr logs are world-readable while the daemon log is 0600
- **Where:** `src/gaia/daemon/sidecars/manager.py:401-410` `_open_log` (`open(path, "wb")`, umask perms) vs `client.py:117-120` — sidecar logs carry uvicorn access lines and tracebacks that can include mailbox subjects/addresses.
- **Fix:** `os.open(..., 0o600)` + DACL helper; `logs/` dir at 0700. — **Confidence:** High — **Tracked:** none found

### [🟢] `docs/security/connections.mdx:9` claims GAIA never writes tokens to plaintext files; the daemon does by design
- **Where:** vs `src/gaia/daemon/instance.py:58-65`, `sidecars/manager.py:589-634` — daemon client token and every sidecar launch secret are plaintext owner-only files. The daemon/custody trust model is documented nowhere under `docs/security/`.
- **Fix:** scope the sentence to connector credentials; add a "Daemon & sidecar credentials" section. — **Confidence:** High — **Tracked:** none found

### [🟢] `caller_auth` docstring promises a loud warning when no token is configured; none is logged
- **Where:** `src/gaia/sidecar/caller_auth.py:27-29, 116-119, 143-144` — `config_from_env` returns `CallerAuthConfig(token=None)` silently; `token_ok` returns True for every request with no log line.
- **Fix:** log once at `configure()`. — **Confidence:** High — **Tracked:** none found

### [🟢] Custody `user` memory scope is documented as shared/cross-agent but stored and read per-agent, ungated
- **Where:** `src/gaia/daemon/custody/constants.py:40-46`, `store.py:165-186` — `get_memory` always filters `WHERE agent_id = ?` (no leak), any agent may write `scope=user` with no grant check (comment defers to V2-3).
- **Fix:** implement the shared read + grant gate or rename the scope. — **Confidence:** High — **Tracked:** none found

**Silent-fallback inventory (daemon / sidecar / email_sidecar):** 40 narrow swallows + 24 broad `except Exception` (all re-raise, translate at a boundary, or log). Per file — `daemon/sidecars/manager.py` 7 (426, 434, **644** `_remove_secret_dir` swallows `OSError` silently, 815, 850, 863, 867); `daemon/instance.py` 7 (78 logged, 115, 154, 160, 196, 202, 208); `daemon/sidecars/ledger.py` 6 (44, 107, 113, 130, 147, 157); `daemon/paths.py` 3 (93, 124, **132** `chmod` failure after rename silent); `daemon/lock.py` 3 (46, 51, 80); `daemon/migrate.py` 2 (217, 225); `daemon/client.py` 1 (**120** `chmod(daemon.log)` silent); `daemon/custody/store.py` 1 (**131** `close()` swallows `sqlite3.Error`); `daemon/sidecars/install.py` 2 (404, 408); `daemon/sidecars/routes.py` 2 (75, 87); `daemon/broker_client.py` 1 (298); `daemon/relay.py` 1 (175); `daemon/scheduler/store.py` 1 (275, documented exactly-once); `daemon/sidecars/fetch.py` 1 (55); `ui/email_sidecar/proxy.py` 1 (53); `ui/email_sidecar/relay.py` 1 (375, debug-logged). Bolded ones deserve a `logger.warning`.

**Sub-area test gaps:** crashed-sidecar restart never tested (the secret-file leak lives there); no test of `auth_token`/custody-secret rotation; `test_daemon_secret_acl.py` covers only the launch-secret file — nothing for `instance.json`/`daemon.log` on NTFS; no broker test for non-numeric/null/negative `timeout` or foreign-caller `release`; `test_daemon_migrate.py` verifies copies+stamp but not that the custody API can read them (it can't); no custody test of `user`-scope visibility across agents; `forward.py`/`relay.py` upstreams fully stubbed — the integration suite (`tests/integration/test_daemon*.py`, not run here) should assert the sidecar accepts the forwarded JSON shape.

**Sub-area doc gaps:** `docs/plans/security-model.mdx` has zero content on the daemon, custody API, per-spawn secrets, broker or `instance.json` (only a banner to an agent-UI plan §0.11/§0.24); no user-facing daemon trust-model page under `docs/security/`; `connections.mdx:9` contradiction; `migrate.py:40-42` docstring false; `caller_auth.py:27-29` and `:139-141` claims false; `ui/email_sidecar/router.py:15-17` says the UI "still talks straight to the sidecar port until V2-7" but V2-7 (#2150) shipped the relay while the email router still bypasses it and `acquire_handle` still hands the UI backend the sidecar bearer — contradicting `client.ensure_agent`'s "a thin client never holds sidecar credentials"; `custody/constants.py:40-43` documents an unimplemented grant-gated shared scope.

**Sub-area improvement opportunities:** rotate `auth_token` + custody secret per spawn; one ACL routine in `paths.py` for all five secret-bearing files; pass sidecars a minimal environment instead of `{**os.environ}` (`manager.py:448` — the user's shell `ANTHROPIC_API_KEY`/`GITHUB_TOKEN`, listed as a threat in `security-model.mdx:361-368`, is inherited by third-party agent binaries today); mount `HostOriginMiddleware` on the daemon app; gate the "test-only" `GAIA_DAEMON_EXTRA_SPECS` seam (`server.py:54-88`) behind an explicit opt-in so a stray env var can't add arbitrary `dev_src_dir` specs; Pydantic-validate broker bodies; escape `%`/`_` in `CustodyStore.get_memory`/`query_rag` `LIKE` patterns.

**Sub-area checked and fine:** daemon binds `127.0.0.1` only (`constants.HOST`, `server.run`, `_find_free_port`), port 4001 never used, OpenAPI disabled; every `/daemon/v1/*`, `/v1/<agent>/*` relay and connections route shares one `require_token` with `secrets.compare_digest`; `/host/v1/*` resolves identity from a per-spawn secret via constant-time scan; only `/host/v1/version` is unauthenticated and carries no data; token minted with `secrets.token_urlsafe(32)` per daemon start; `instance.json` written atomically (`O_EXCL` temp + fsync + `os.replace`), 0600/0700 on POSIX; probe requires service id + pid; `terminate_instance` checks cmdline (PID-reuse safe); no `shell=True` anywhere; agent ids validated with `_ID_RE` before becoming path segments; hub `executable` validated as a bare filename (`gaia/hub/installer.py:241-255`); binary downloads and hub installs SHA-256-verified, placeholder SHAs refused; secrets never logged (only the secret-file *path*); relay strips `Authorization`/hop-by-hop headers both ways, bounds the SSE buffer (16 MiB), propagates cancel; forward-out never forwards an ungranted provider, narrows scopes to grant ∩ connection, withdraws on revoke, never ships the refresh token; custody store parameterized throughout, every query scoped by `agent_id`, session ownership checked, `k` bounded 1..50, WAL + busy_timeout; migration WAL-aware, non-destructive, atomic, refuses newer-than-known schema; start lock uses OS advisory locks; registry closes the capacity TOCTOU with `_starting`; ledger reap never kills on port evidence alone.

**Sub-area hypotheses (unverified):** relay passes `{path:path}` unnormalised (`/v1/email/../../health`) — same trust level, so no boundary crossed; `install.py:_run_install` abandoned mid-download on daemon shutdown with the in-process install slot held; `_read_log_tail` in `SidecarSpawnError` could echo a sidecar's env dump if the sidecar prints it; `EmailSidecarProxy` reads `GAIA_EMAIL_SIDECAR_TIMEOUT` with a bare `float()` — non-numeric value probably becomes a 500 on every email route rather than a startup error.


---
## Sub-area: connectors / governance / OpenAI-compatible API / shell tool / filesystem index
(Delegated deep-read; the API-crash, Windows shell-bypass and contextvar claims were independently re-run by the lead reviewer on this checkout — outputs quoted below are from those re-runs.)

### [🔴] Every `/v1/chat/completions` request crashes: the API server passes a `workspace_root` kwarg no agent accepts
- **Where:** `src/gaia/api/openai_server.py:364` (non-streaming) and `:496` (streaming, inside `create_sse_stream`) → `src/gaia/agents/base/memory.py:2183` `MemoryMixin.process_query(self, user_input, **kwargs)` → `src/gaia/agents/base/agent.py` `Agent.process_query(self, user_input, max_steps=None, trace=False, filename=None)`
- **What:** The OpenAI-compatible server always calls `agent.process_query(msg, workspace_root=…)`. The served agent is `GaiaAgent(ChatAgent, …)`; neither it nor `ChatAgent` defines `process_query`, so the call lands on `MemoryMixin.process_query`, which forwards `**kwargs` to `Agent.process_query`, which has no such parameter → `TypeError` on every request, streaming or not. The only consumer of `workspace_root` was the deleted code agent.
- **Failure scenario:** `gaia api start` → `POST /v1/chat/completions {"model":"gaia","messages":[{"role":"user","content":"hi"}]}` → HTTP 500 (non-streaming) or a stream that emits the role chunk and aborts. No chat-completions request can succeed against the shipped agent. `tests/test_api.py:71,124` replace the agent with `MagicMock`, which accepts any kwarg, so the suite is green — the exact "mock proves we called it, not that the call is valid" trap CLAUDE.md warns about.
- **Evidence:**
  - `openai_server.py:364` `result = agent.process_query(user_message, workspace_root=workspace_root)`; `:496` `lambda: agent.process_query(query, workspace_root=workspace_root)`
  - `grep -rn workspace_root src hub --include=*.py` → only `openai_server.py`
  - re-run on this checkout: `Agent.process_query (self, user_input, max_steps=None, trace=False, filename=None)` / `sig.bind(None,"x",workspace_root=None)` → `bind FAILS -> got an unexpected keyword argument 'workspace_root'`; `MemoryMixin.process_query (self, user_input, **kwargs)`; `grep "def process_query" hub/agents/gaia hub/agents/chat src/gaia/agents/base` → only `memory.py:2183`.
- **Fix:** Drop the kwarg (or set `agent.workspace_root` before the call). Add one `tests/test_api.py` case that drives a real `Agent` subclass with a stubbed LLM so the call shape is validated.
- **Confidence:** High
- **Tracked:** none found

### [🔴] Windows shell tool: the read-only whitelist is bypassable — bare `&` and PowerShell .NET/WMI calls run arbitrary commands via `cmd.exe`
- **Where:** `src/gaia/agents/tools/shell_tools.py:200-203` `DANGEROUS_SHELL_OPERATORS`, `:570-616` (PowerShell branch of `_validate_command`), `:949` `use_shell = os.name == "nt" and not lone_granted_segment`, `:1029` `shell=use_shell`
- **What:** On Windows the validated *string* is handed to `cmd.exe` (`shell=True`). (a) The operator regex only blocks `&` followed by whitespace/end, so `echo x&calc.exe` passes and cmd.exe runs both commands. (b) The PowerShell filter only looks for hyphenated cmdlet names, so `.NET`/WMI method calls — `[System.Diagnostics.Process]::Start('calc')`, `[IO.File]::WriteAllText(...)`, `(Get-WmiObject -List Win32_Process).Create('…')`, `$x='calc'; &$x` — are accepted as "read-only". POSIX is unaffected (argv, `shell=False`).
- **Failure scenario:** Prompt-injected content (web page, email, issue body) steers the model to `run_shell_command` with a WMI `Create(...)`; `policy_refusal_for_call` says "not refused", so the user sees a confirmation modal for what looks like a WMI query; with `GAIA_AUTO_APPROVE_TOOLS=1` (documented opt-in for the API server) nobody is asked. Exploitable locally and via remote injected content; needs the confirmation click unless auto-approve is on. `ChatAgent`/`GaiaAgent` compose `ShellToolsMixin` (`hub/agents/chat/python/gaia_agent_chat/agent.py:181`).
- **Evidence:** re-run of `ShellToolsMixin._validate_shell_command` on this checkout: `'echo x&calc.exe' -> (None, [['echo', 'x&calc.exe']])` (allowed) and `"powershell -Command \"[System.Diagnostics.Process]::Start('calc.exe')\"" -> (None, [...])` (allowed), while `'echo hi > out.txt'`, `'ls; rm -rf /'`, `'git status && rm x'` are refused. `subprocess.run('echo x&echo INJECTED', shell=True)` → `x\nINJECTED`. `:198-199` comment says the word-boundary `&` is deliberate ("avoid false positives inside quoted PowerShell strings"); `test_shell_guardrails.py:143` only asserts `sleep 10 &`.
- **Fix:** Never hand the raw string to `cmd.exe`: run non-pipeline commands as argv and resolve Windows built-ins explicitly (`cmd /c dir …` as argv); implement `|` pipelines by spawning each segment as argv. Treat any `&` as an operator. For PowerShell, allowlist the whole command against a grammar (no `[Type]::`, `.Method(`, `&`/`.` invocation, `$`) or drop `powershell` from the whitelist. Add regression tests for the allowed cases above.
- **Confidence:** High
- **Tracked:** none for the bypass; #2785 (doc classifies `run_shell_command` as auto-approve READ tier) is adjacent

### [🟡] Tavily API key is handed to any caller with no grant check
- **Where:** `src/gaia/web/tavily.py:125` / `:145`, `src/gaia/connectors/handler.py:399-410`
- **What:** `get_credential_sync(_CONNECTOR_ID)` passes neither `agent_id` nor `required_scopes`; the dispatcher checks grants only `if resolved_agent and required_scopes`, so the `mcp-tavily` secret is returned unconditionally. Activations gate only MCP tool *visibility* (`mcp_client_manager.py:150-152`), never this Python-native path. (This is the concrete in-tree instance of the contextvar finding above.)
- **Failure scenario:** User configures `mcp-tavily` and grants it to no agent; any agent's web-search tool spends the user's quota. Contradicts `docs/security/connections.mdx:63`. **Local only.**
- **Evidence:** `tavily.py:125 cred = get_credential_sync(_CONNECTOR_ID)`; `handler.py:401 if resolved_agent and required_scopes:`; `tests/unit/connectors/test_handler.py:161 test_no_agent_id_skips_grant_check` locks it in.
- **Fix:** Pass `agent_id=<namespaced id>` + `required_scopes=["use"]`; make `handler.get_credential` default `required_scopes` to `["use"]` for `mcp_server` specs.
- **Confidence:** High — **Tracked:** none found

### [🟡] MCP-server secrets skip the insecure-keyring refusal the OAuth path enforces
- **Where:** `src/gaia/connectors/mcp_server.py:168`, `:215`, `:334`
- **What:** `store.verify_keyring_backend()` refuses `PlaintextKeyring`/`EncryptedKeyring`/`Win32CryptoKeyring` before every OAuth save/load; `McpServerHandler` calls `keyring.set_password/get_password` directly with no check (grep: no `verify_keyring_backend` in the module).
- **Failure scenario:** headless Linux with `keyrings.alt` resolved → `gaia connectors configure mcp-github --set GITHUB_TOKEN=ghp_…` writes the PAT to `~/.local/share/python_keyring/keyring_pass.cfg` in cleartext, no error, while `docs/security/connectors.mdx:21-25` says every save is refused. **Local only.**
- **Fix:** Route through `store._kr_set/_kr_get/_kr_delete` (also gains >1280-char chunking on Windows). Add a refused-backend test for `McpServerHandler.configure`.
- **Confidence:** High — **Tracked:** none found

### [🟡] Index scan deletes a sibling directory's entries when names share a prefix
- **Where:** `src/gaia/filesystem/index.py:349-354`, `:371-375` `scan_directory`, `:566-569` `_update_directory_stats`
- **What:** Stale detection loads `SELECT path FROM files WHERE path LIKE '<root>%'` — no path separator, `%`/`_` unescaped — so scanning `…/doc` also loads `…/docs/**` as "existing", never sees them, and deletes them.
- **Failure scenario:** verified: scan `docs` (1 added), scan sibling `doc` → `files_removed=1`, `docs/b.txt` gone from the index.
- **Fix:** `root_str + os.sep + "%"` (+ `OR path = :root`) with `ESCAPE '\'`.
- **Confidence:** High — **Tracked:** none found

### [🟡] `query_files(name=…)` raises on ordinary file names (FTS5 syntax injection)
- **Where:** `src/gaia/filesystem/index.py:709-712`, caller `src/gaia/agents/tools/filesystem_tools.py:663`
- **What:** LLM-supplied `name` is passed verbatim as an FTS5 `MATCH` expression.
- **Failure scenario:** verified: `"report (final)"` → `sqlite3.OperationalError: fts5: syntax error`; `'c++'` → syntax error; `'my-file'` → `no such column: file`; `'name:report OR path:secret'` executes as a column-filtered boolean query.
- **Fix:** quote the term (`'"' + name.replace('"','""') + '"'`, optional trailing `*`).
- **Confidence:** High — **Tracked:** none found

### [🟡] Non-streaming `/v1/chat/completions` blocks the whole event loop
- **Where:** `src/gaia/api/openai_server.py:364` — sync minutes-long `agent.process_query` inside `async def create_chat_completion` (the streaming branch uses `run_in_executor`, `:495`). `/health`, `/v1/models`, other completions and the daemon relay stall meanwhile.
- **Fix:** `await loop.run_in_executor(None, …)`. — **Confidence:** High — **Tracked:** none found

### [🟡] `/v1/chat/completions` has no authentication, and the docs' first example binds it to `0.0.0.0`
- **Where:** `src/gaia/api/openai_server.py:276` (no dependency), `src/gaia/api/agent_proxy.py:21-24` ("`/v1/chat/completions` and `/v1/models` are untouched"), `docs/sdk/infrastructure/api-server.mdx:25` `uvicorn.run(app, host="0.0.0.0", port=8080)`
- **What:** Only the `/v1/<agent>/*` relay is gated by `GAIA_API_KEY`; chat completions drive the full flagship agent (web fetch, file reads, memory, and with `GAIA_AUTO_APPROVE_TOOLS=1` shell + file writes) with no auth.
- **Failure scenario:** user copies the doc example on a LAN → any LAN host runs agent queries and reads whatever the agent's tools can read. **Remote (LAN)** when non-loopback bound.
- **Fix:** apply `require_api_key` to `/v1/chat/completions` when `GAIA_API_KEY` is set; refuse non-loopback binds without a key; change the doc example to `127.0.0.1`.
- **Confidence:** High — **Tracked:** #630

### [🟢] `gaia connectors connect --device` silently ignores `--grant-agent`
- **Where:** `src/gaia/connectors/cli.py:373-377` `_handle_connect_device` → `poll_device_flow` without `grant_agents` (supported at `flow.py:742`). Headless Microsoft users end up connected-but-ungranted, the dead end `flow.py:236-239` says the flag exists to prevent.
- **Fix:** pass `grant_agents=` as `_handle_connect` does. — **Confidence:** High — **Tracked:** none found

### [🟢] `McpServerHandler.configure` docstring promises writing non-secret env keys; the code drops them silently
- **Where:** `src/gaia/connectors/mcp_server.py:195-197` vs `:210-226` (loops only over `spec.mcp_env_keys`). `--set GITHUB_API_URL=…` vanishes with no error.
- **Fix:** implement or reject unknown keys loudly; fix the docstring. — **Confidence:** High — **Tracked:** none found

### [🟢] `JsonlReceiptService` default audit path is relative to CWD
- **Where:** `src/gaia/governance/adapter.py:186` `default(audit_log="receipts.jsonl")` — the audit trail lands wherever the process started.
- **Fix:** default to `~/.gaia/governance/receipts.jsonl` (0600). — **Confidence:** High — **Tracked:** none found

**Silent-fallback inventory (connectors / governance / api / shell / filesystem):** `connectors/store.py` 3 (216-217, 265-266, **558-559** corrupt client-credential blob → "not configured", no log), `connectors/flow.py` 4 (121-131, 170-177, 206-212, 321-329, 387-391 — all warn), `connectors/tokens.py` 2 (241-242, 269-270), `connectors/api.py` 3 (425-427, 733-735 `continue` per provider, 743-744), `connectors/grants.py` 3 (76-80, 147-148, 153-154), `connectors/activations.py` 3 (197-198, 202-203 + chmod warn), `connectors/activation_watcher.py` 3 (**43-47** `_safe_load` any exception → `{}` → spurious "deactivated" events for every pair, 90-91, 117-118), `connectors/mcp_server.py` 5 (65-66, 74-75, 79-80, 84-85, 262-263), `connectors/events.py` 1 (213-214 logged), `connectors/store.py:595-598` (`list_connections` skips a provider on keyring error), `governance/mixin.py` 5 (192-193, 194-200, 232-233, 264-270, 353-359 fail-closed ✔, 386-391), `governance/receipt_service.py` 2 (**113-124** malformed JSONL skipped at DEBUG — audit-log corruption invisible at default level), `api/app.py` 1 (269-270), `api/agent_registry.py` 1 (211-217 broken agent import still advertises the model), `filesystem/index.py` 6 (170-176, 191-196, 250-251 integrity failure → DB deleted, 429-431, 438-439, 446-447), `agents/tools/shell_tools.py` 3 (417-418 fail-closed ✔, **929-936** path-resolve failure → argument executed unchecked, flagged in-code, 1099-1101), `shell/prompt.py` 1 (175-176). Worth fixing: bolded.

**Sub-area test gaps:** `/v1/chat/completions` only exercised with a `MagicMock` agent (`tests/test_api.py:62-145`) — how the fatal kwarg shipped; shell guardrail tests exercise regex/whitelist in isolation, nothing asserts what `cmd.exe` receives, no test for bare `&`, `.NET`/WMI calls, `&$var`; no test executes a tool through `_execute_tool` and checks `current_agent_id()`; `test_handler.py::test_no_agent_id_skips_grant_check` enshrines the ungated path; no refused-backend test for `McpServerHandler`; `test_filesystem_index.py` never scans prefix-sharing siblings nor FTS5-operator names; no test that a `PolicyEngine.evaluate_action` exception fails closed; on Windows `tests/unit/connectors` → `33 failed, 508 passed, 12 skipped, 217 errors` (socketpair guard).

**Sub-area doc gaps:** `docs/sdk/infrastructure/connectors.mdx:257-262` shows the ambient (`agent_id`-less) `get_access_token_sync` pattern that is now ungated; `docs/security/connections.mdx:37,63` and `connectors.mdx:21-25` promise grant enforcement / plaintext-keyring refusal the Tavily path and MCP handler don't deliver; `connectors.mdx:389` lists `~/.gaia/connectors/state.json` (0600) that no code writes (`oauth_pkce.py:139-141`: keyring blob is the source of truth); `api-server.mdx:25` binds `0.0.0.0` with no note that chat completions are unauthenticated; `McpServerHandler.configure` docstring. `src/gaia/governance/README.md` matches the code.

**Sub-area checked and fine:** OAuth loopback server binds `127.0.0.1` on an ephemeral port (`flow.py:280-286`); `state` = 32 random bytes compared with `hmac.compare_digest` after a `None` guard (`:401-409`); success/error pages are static literals; PKCE S256 with 64-byte verifier; `redirect_uri` generated, never caller-supplied; token exchange uses the provider's constant `token_url` — no SSRF; refresh tokens only in the OS keyring with plaintext-backend refusal on every OAuth save/load; grants/activations/`mcp_servers.json` written atomically (`mkstemp` + `os.replace`, 0600/0700 POSIX); no token/secret/auth-code in any `logger` call under `connectors/`; `list_connections` strips `refresh_token`; `import_forwarded_connection` never returns secrets; refresh path double-checked-locked per `(provider, account)`, 60 s buffer, rotation persisted before exposure, `invalid_grant` → `ConnectionRevokedError`; `_agent_context` not re-exported from `gaia.connectors`; MCP catalog `command`/`args` only from frozen `ConnectorSpec`s (no injection via configure), `mcp_servers.json` holds `$keyring` refs only; `activate()` refuses non-`mcp_server`; `disconnect` wipes grants and activations; `agent_proxy.py` key compared with `secrets.compare_digest`, unset → 503, hop-by-hop + `Authorization` stripped, reserved ids refused, upstream closed on disconnect; API CORS never wildcard+credentials (`openai_server.py:161-190`); debug logging redacts headers/bodies; `SSEOutputHandler` denies confirmation-gated tools by default; governance opt-in, REVIEW fails closed, receipts hash the full envelope; shell tool on POSIX is argv, `find -exec/-delete`, `sort -o`, git write subcommands, `-EncodedCommand/-File` refused, stdin `DEVNULL`; filesystem index parameterised except the two LIKE/MATCH cases, symlinks not followed. Passing suites: `tests/unit/api`, `test_api_agent_proxy.py`, `test_api_extras.py`, `test_governance_*`, `test_agent_required_connectors.py`, `test_skill_binary_grants.py`, `test_filesystem_index.py`, `test_shell_guardrails.py`, `test_shell_output_encoding.py` (460 passed).

**Sub-area hypotheses (unverified):** cmd.exe newline / `^` escape sequences may offer further separators the regex misses (only `&` and the PowerShell cases were verified); `_resolve_account_email` userinfo GET (`flow.py:195-198`) is a constant URL today but becomes a token-exfil SSRF if a future spec makes it configurable; `activation_watcher` on a transiently unreadable ledger flips the Settings UI; whether the Agent-UI shell confirmation modal renders the full command string (so a user could spot `.Create(`) was not checked. (The lead reviewer tested the "cross-site `text/plain` JSON body" hypothesis against FastAPI 0.141.1: a JSON body with no/`text/plain` Content-Type gets **422**, so Pydantic-body routes are *not* CSRF-able that way — only body-less / query-only / multipart / manual `request.json()` routes are; see the UI CSRF finding.)


---
## Additional lead-reviewer findings (Agent UI backend / security.py)

### [🟡] Agent file-write guardrails do not block shell-profile / autostart files, so an in-allowlist write is a persistence-to-RCE vector
- **Where:** `src/gaia/security.py:28-55` `SENSITIVE_FILE_NAMES`, `:112-167` `_get_blocked_directories`, `:503-560` `is_write_blocked`; allowlist source `src/gaia/ui/_chat_helpers.py:926-940` `_compute_allowed_paths`
- **What:** The write denylist covers `.env`, key material, `~/.ssh`, `~/.gnupg`, macOS `LaunchAgents` and the Windows Startup folder, but not the files that execute on next login/shell: `~/.bashrc`, `~/.bash_profile`, `~/.profile`, `~/.zshrc`, `~/.zprofile`, `~/.config/autostart/*.desktop`, `~/.config/fish/config.fish`, `~/.gitconfig` (`core.hooksPath`/aliases), `.git/hooks/*`, `Documents\WindowsPowerShell\profile.ps1` / `PowerShell\Microsoft.PowerShell_profile.ps1`, `~/.tmux.conf`, `~/.vimrc`. In the Agent UI the allowlist is the parent directory of every attached document — attach a file that lives in `$HOME` (or `~/Documents`) and the agent may write anywhere under it.
- **Failure scenario:** prompt-injected document steers the model to `write_file(path="~/.bashrc", content="curl … | sh")` while a `$HOME` document is attached; `validate_write` passes (allowlist ✔, not a blocked dir, not a sensitive name), the confirmation modal shows a benign-looking "write ~/.bashrc", and the payload runs at the next shell. **Local** (needs the user's approval click — which the modal shape makes easy to give — or `GAIA_AUTO_APPROVE_TOOLS=1`).
- **Evidence:**
  - `security.py:28-55` — set contains `.env*`, `credentials.json`, `id_rsa`, `authorized_keys`, `.netrc`, `.npmrc`, `.pypirc` … no shell rc / profile / autostart entries
  - `security.py:117-131` Windows blocklist = `WINDIR`, `Program Files*`, `ProgramData\Microsoft`, `.ssh`, Start-Menu `Startup`; POSIX (`:133-157`) = system dirs, `.ssh`, `.gnupg`, `Library/LaunchAgents` — no `~/.config/autostart`, no shell rc
  - `_chat_helpers.py:934-936` `for fp in rag_file_paths: dirs.add(str(Path(fp).parent))` — a document in `$HOME` makes `$HOME` writable
- **Fix:** add shell rc / profile / autostart / git-hook / PowerShell-profile names and dirs to the blocklists; consider blocking writes to any dotfile directly under `$HOME` unless explicitly allowlisted; surface the "this file runs on login" fact in the confirmation modal.
- **Confidence:** High
- **Tracked:** none found (`gh issue list --search "bashrc profile persistence"` / "sensitive file write guard" → none; #2768 audits `ALLOWED_COMMANDS`, not the write guard)

### [🟢] `gaia chat --ui` launches uvicorn without the `proxy_headers=False, forwarded_allow_ips=""` hardening that `python -m gaia.ui.server` uses (and that `TunnelAuthMiddleware`'s comments rely on)
- **Where:** `src/gaia/cli.py:856-862` vs `src/gaia/ui/server.py:900-913` `main()`
- **What:** The standalone runner disables uvicorn's proxy-header rewrite so `request.client.host` is always the raw TCP peer; the CLI path (the documented default) does not, so uvicorn's default (`proxy_headers=True`, trust `127.0.0.1`) applies and ngrok's local agent can rewrite `client.host`. The middleware is still safe because its localhost bypass *also* requires no `X-Forwarded-*` header, so this is an inconsistency with a misleading comment rather than a bypass — but a future change that relaxes the header check would open the spoof the comment claims is closed.
- **Evidence:** `cli.py:856` `uvicorn.run(app, host="127.0.0.1", port=port, log_level=…, access_log=debug)`; `server.py:900-913` passes `proxy_headers=False, forwarded_allow_ips=""` with a comment saying the middleware relies on it.
- **Fix:** factor a `run_server(app, host, port, …)` helper in `server.py` used by both entry points.
- **Confidence:** High — **Tracked:** none found

### [🟢] Agent-UI MCP server subprocess is spawned with `stderr=PIPE` that is never drained
- **Where:** `src/gaia/ui/routers/mcp.py:270-275` `start_agent_mcp_server`
- **What:** `stderr=subprocess.PIPE` is only read if the process exits within the 1 s grace period; a long-lived server that logs to stderr eventually fills the pipe (64 KiB on Linux) and blocks on its next write — the MCP server hangs with no diagnostic.
- **Fix:** `stderr=subprocess.DEVNULL` (or a log file), read the last lines from the file on failure.
- **Confidence:** High — **Tracked:** none found

**Silent-fallback inventory — `src/gaia/ui` (bare `except …: pass / return None / continue`, per file, line of the `except`):** `routers/system.py` **6** (473, 589, 622, 678, 689, 744 — all "keep the status endpoint alive" swallows; 622/689 hide Lemonade `/stats` and catalog failures with no log), `_chat_helpers.py` **4** (90, 1620, 2737, 2765), `routers/memory.py` **3** (268, 809, 919), `tunnel.py` **2** (492 `_kill_stale_ngrok`, 547 poll loop — documented), `routers/chat.py` **2** (49 `_notify_loop`, 203 auto-titling), `agent_loop.py` **2** (213, 462), `server.py` **1** (350 model preload re-check), `routers/mcp.py` **1** (298 stderr read), `routers/goals.py` **1** (45), `routers/files.py` **1** (602 CSV header parse). Total **23** in `src/gaia/ui` — CLAUDE.md calls these pre-existing tech debt; `system.py` is the densest and also the least tested (see Test gaps).


---
## Scope covered

**Read fully by the lead reviewer:** `src/gaia/ui/server.py` (921 lines), `ui/tunnel.py` (680), `ui/routers/tunnel.py`, `ui/routers/files.py` (665), `ui/routers/documents.py` (749), `ui/routers/system.py` (1060), `ui/routers/chat.py` (307), `ui/routers/mcp.py` (spawn section 200-349; the rest sampled), `ui/utils.py` (598), `ui/dispatch.py`, `ui/run_manager.py`, `src/gaia/security.py` (735), `src/gaia/ui/_chat_helpers.py` (`_compute_allowed_paths` + session kwargs only — the remaining ~2,800 lines were **not** read), `ui/agent_loop.py` (header, tunnel gate, tick entry — sampled), `ui/sse_handler.py` (confirmation gate only — sampled), `ui/email_sidecar/router.py` (POST signatures only), `ui/database.py` (SQL-construction grep + `get_stats`; not read line-by-line), `src/gaia/connectors/context.py`, `connectors/api.py:170-300`, `src/gaia/agents/base/agent.py` (`_call_tool_bounded`, `process_query`, `_agent_identity_context`), `src/gaia/cli.py:840-875`, `src/gaia/api/openai_server.py` (the `workspace_root` call sites), `src/gaia/mcp/mcp_bridge.py:434-512`, `src/gaia/messaging/telegram.py:74-80, 306-310`, `src/gaia/daemon/sidecars/manager.py:460-472, 672-692`, `src/gaia/daemon/app.py:35-40`. Docs: `SECURITY.md`, `docs/plans/security-model.mdx` (header + grep), `docs/spec/agent-ui-server.mdx` (security section), `docs/sdk/sdks/agent-ui.mdx` (tunnel section), `docs/guides/agent-ui.mdx` (grep), `ls docs/security`. Tests: `tests/unit/conftest.py` guard, `tests/unit/chat/ui/test_tunnel_auth.py` (structure), `tests/test_api.py` (mock usage).

**Read fully by delegated sub-reviewers (reports integrated above; their headline claims spot-checked by the lead):** all of `src/gaia/daemon/**`, `src/gaia/sidecar/caller_auth.py`, `src/gaia/ui/email_sidecar/**`, all of `src/gaia/connectors/**` (except `errors.py`/`setup_routes.py` skimmed), `src/gaia/governance/**`, `src/gaia/api/**`, `src/gaia/agents/tools/shell_tools.py`, `src/gaia/filesystem/**`, `src/gaia/shell/prompt.py` (swallow sites only), all of `src/gaia/mcp/**`, `src/gaia/messaging/**`, `src/gaia/schedule/**`, `src/gaia/ui/scheduler.py`, `ui/routers/schedules.py`, `src/gaia/daemon/scheduler/**`, plus the connectors/api-server/mcp/telegram/schedule docs and `docs/security/*.mdx`.

**Not read / only sampled:** `ui/routers/{sessions,agents,hub,goals,memory,onboarding,connectors}.py` (route signatures were enumerated for the CSRF finding; bodies not audited — `memory.py` 1,717 lines and `connectors.py` 1,181 lines are the largest unaudited surfaces in this dimension), `ui/sse_translation.py`, `ui/event_narration.py`, `ui/document_monitor.py`, `ui/models.py`, `ui/build.py`, `src/gaia/mcp/servers/*` beyond what the sub-reviewer covered, `tests/mcp/**` (needs a live server / `mcp` package not in the venv), `tests/integration/**` (not run), `docs/plans/agent-ui-agent-capabilities-plan.md` §0 (the "v2" security spec the model doc defers to).

**Test runs on this checkout (Windows, `.venv`, fastapi installed by the reviewer via `uv pip install -e ".[ui,dev]"`):** `tests/unit/connectors` → 21 failed / 218 passed / 29 errors (socketpair guard); daemon + sidecar + email-sidecar suites → 157 failed / 571 passed / 18 skipped (same guard + 2 Windows-only test bugs); `tests/unit/api`, `test_api_agent_proxy.py`, `test_api_extras.py`, `test_governance_*`, `test_agent_required_connectors.py`, `test_skill_binary_grants.py`, `test_filesystem_index.py`, `test_shell_guardrails.py`, `test_shell_output_encoding.py` → 460 passed; `tests/unit/mcp` → 229 passed / 2 skipped; `tests/unit/chat/ui` → **213 failed / 582 passed / 13 skipped / 42 errors** — 235 of the failures are the same `ConnectionError` from the socketpair guard (`test_server.py` 98, `test_tunnel_auth.py` 30, `test_agents_router.py` 28, `test_chat_helpers_model_resolution.py` 13, plus setup errors in `test_run_manager/test_dispatch/test_document_monitor/test_chat_concurrency`); every `TestClient` request in those files returns a false 500, so on Windows the Agent-UI backend has effectively zero runnable unit coverage on `main`. `tests/verify_shell_security.py` and `tests/verify_path_validator.py` are not pytest files — they `SystemExit` at import when `gaia-agent-chat` is absent (it is absent in this venv), so they were not executed.

## Test gaps
- **CSRF / origin enforcement has no test at all.** `tests/unit/chat/ui/test_tunnel_auth.py` covers bearer/cookie handling thoroughly but nothing asserts that a mutating route without `X-Gaia-UI` is rejected, and no route-table introspection test guards new routers. Needed: a parametrised test over `app.routes` asserting every non-GET `/api|/v1` route depends on the guard (or a middleware test).
- **Tunnel pre-URL window untested:** every `test_tunnel_auth.py` fixture sets `tunnel._url` directly (`:34`); no test puts the manager in "process spawned, `_url is None`" and asserts a remote request still gets 401.
- **`ui/routers/system.py` (1,060 lines, 6 swallowed exceptions) has no dedicated unit test module** — `grep -rl "routers.system\|/api/system"` finds only `test_server.py`/`test_startup_init.py` (startup jobs) and integration/stress suites. `update_settings` validation, `_stream_lemonade_pull` SSE parsing/byte accounting, `download-model`/`load-model` fire-and-forget tasks are unverified.
- **File endpoints test containment, not sensitivity:** `test_toctou.py` + `test_utils_helpers.py` prove home-directory containment; nothing asserts `~/.gaia`/`~/.ssh` are unreadable (they are readable — finding above).
- **`security.py` write guard:** `tests/verify_path_validator.py` is a script, not a pytest module, and needs `gaia-agent-chat`; no pytest coverage that `.bashrc`/autostart writes are blocked (they aren't).
- **Mock-only boundaries (CLAUDE.md "mocks prove we called it"):** `/v1/chat/completions` (`MagicMock` agent → the fatal kwarg shipped); shell guardrails (regex only, never what `cmd.exe` receives); `forward.py`/`relay.py` upstreams fully stubbed; `runner.fire` with mocked `AgentSDK`; Telegram handlers never run against real `Update` objects (PTB absent).
- **Windows: the unit suite is effectively unrunnable** (socketpair guard) — ~400 errors across connectors/daemon/scheduler/UI; no Windows job in CI for `tests/unit`.
- **Contextvar propagation through `_call_tool_bounded`** — no test executes a tool via `_execute_tool` and reads `current_agent_id()`; `test_agent_bridge.py` covers only the executor hop; `test_handler.py::test_no_agent_id_skips_grant_check` enshrines the ungated path.
- **Crashed-sidecar restart, `instance.json` ACL on NTFS, broker `timeout`/foreign `release`, custody `user`-scope visibility, migration consumer** — none tested (daemon sub-area).
- **Bridge:** no test for non-GET/POST methods, malformed `Content-Length`, oversized bodies, shared `chat_sdk`; the CORS `*` default is pinned *as intended* by `test_cors_preflight_stays_open_and_allows_authorization`.
- **Schedulers:** no bad-cron rejection test, no `TomlScheduleStore` atomicity/concurrency test, no timezone test for `compute_next_run`.
- **Filesystem index:** no prefix-sibling scan test, no FTS5-operator name test.

## Documentation gaps
- `docs/spec/agent-ui-server.mdx:251` says "`gaia chat --ui` binds to `0.0.0.0` for Electron/browser access" — stale: `cli.py:858` binds `127.0.0.1`. `:252` "No authentication: Designed for single-user local use. Do not expose to untrusted networks" is the only security statement for the UI backend and says nothing about the tunnel-token grant, CSRF posture, or the `X-Gaia-UI` convention new routers must follow.
- `docs/spec/agent-ui-server.mdx:71` and `docs/sdk/sdks/agent-ui.mdx:878-893` describe "Ngrok/Cloudflared tunnel lifecycle" and a status with "provider, uptime" — the code is ngrok-only (`tunnel.py`) and `get_status()` returns `active/url/token/startedAt/error/publicIp`; no `provider`.
- **No doc anywhere states what a tunnel token grants** (full API incl. home-directory read, chat with tools, memory, connectors settings). `docs/guides/agent-ui.mdx` mentions the tunnel only to say background runs are suspended while it is active.
- `docs/plans/security-model.mdx:89-97` "All GAIA services bind exclusively to `127.0.0.1`. No public ports are opened" — the tunnel feature, `gaia api start --host 0.0.0.0` (`api/app.py:279` docstring example), `gaia mcp start --host 0.0.0.0` (`cli.mdx:1249`), and `AgentMCPServer(host=…)` all open non-loopback surfaces; the doc's own banner admits the localhost-trust framing is superseded but the superseding text (§0.11/§0.24) lives in a plan file, not under `docs/security/`. The daemon/custody/broker trust model is documented nowhere user-facing.
- `docs/security/connections.mdx:9` "never writes tokens … to plaintext files" vs `instance.json` + launch-secret files; `:37,63` grant-enforcement promises vs the Tavily path and the contextvar drop; `docs/security/connectors.mdx:21-25` plaintext-keyring refusal vs `McpServerHandler`; `docs/sdk/infrastructure/connectors.mdx:257-262` shows the now-ungated ambient pattern; `:389` lists a `state.json` no code writes.
- `docs/sdk/infrastructure/api-server.mdx:25` binds `0.0.0.0` without saying chat completions are unauthenticated.
- Telegram / schedule / MCP doc contradictions are itemised in the sub-area sections (`stop --force` no-op, "ignored" vs refusal reply, port 8765 collision, `n8n.json` stale `/jira`, `tui_mcp.py:95` wrong port, `mcp.mdx` silent on auth).
- `SECURITY.md` is fine as a reporting policy; it does not link to any threat-model / trust-boundary page — there is none to link.
- Code-comment claims that are false: `migrate.py:40-42`, `caller_auth.py:27-29, 139-141`, `email_sidecar/router.py:15-17`, `custody/constants.py:40-43`, `McpServerHandler.configure` docstring, `server.py:900-913` (hardening comment true only for one of two entry points).

## Improvement opportunities
- **One request-guard middleware for the UI backend** (`X-Gaia-UI` on every mutating `/api|/v1` request + `TrustedHost` + `Origin` allowlist) — replaces 17 per-route `Depends` and closes CSRF + DNS rebinding in one place.
- **`tunnel.auth_required` property** independent of `_url`; also stop `_kill_stale_ngrok` from killing *every* ngrok on the machine (`taskkill /im ngrok.exe`, `pkill -x ngrok`) — it takes down the user's unrelated tunnels.
- **Sensitive-path denylist shared by `ensure_within_home` and `security.PathValidator`** (one list, two consumers) so read and write guards agree on what is sensitive.
- **`contextvars.copy_context().run(...)` in `_call_tool_bounded`** + fail-closed `_authorize_access` when identity is absent — fixes the grant bypass and the Tavily gap together.
- **Argv-based Windows shell execution with an explicit built-in table** — the only way the read-only whitelist becomes a security boundary; unblocks #2785's auto-approve tier honestly.
- **Rotate sidecar `auth_token` + custody secret per spawn; one `paths.py` ACL routine for all secret-bearing files; custody secret via file leg; minimal child environment** (daemon sub-area).
- **Fix the unit-test network guard for Windows** (allow the `_fallback_socketpair` loopback connect) and add a `windows-latest` unit job — the keyring/DACL/cmd.exe paths are Windows-specific and currently validated only by hand.
- **Collapse the three schedulers onto `DaemonClock`**; atomic `schedules.toml` writes; cron validation at `add`.
- **Bridge:** auth-token-by-default (print once, store 0600 like `tui/control.json`), origin-scoped CORS, proper 405/400 responses, body cap, per-client sessions.
- `system.py`: route the six swallowed exceptions through one `_probe(name, coro)` helper that logs at WARNING and records the failure in `SystemStatus` (e.g. `probe_errors: list[str]`) so degraded status is visible instead of silent.
- Non-streaming `/v1/chat/completions` in the executor; drop `workspace_root`.
- `filesystem/index.py`: escape `LIKE`, quote FTS5 terms.

## High-impact feature opportunities
- **Real remote-access security for the tunnel (#894 PWA, #898 tunnel UX, #2527 headless mode all push toward remote use):** scoped tokens (chat-only vs. full), per-token expiry/revocation, a "what this QR grants" screen, and a sensitive-path denylist. Today the tunnel is a full home-directory read token — the consumer-app track can't ship remote/mobile on it. Roughly: token model + middleware + 3 endpoints + settings UI; 1–2 weeks.
- **Security posture page in `docs/security/`** (trust boundaries: browser ↔ UI backend ↔ daemon ↔ sidecars ↔ providers; what each secret file is; what each env flag disables). Half a day of writing that would have caught three of the doc/code contradictions above and is the prerequisite for the "grant model as a boundary" story in the connectors docs.
- **Grant enforcement that actually gates third-party agents/skills** (contextvar propagation + fail-closed + `required_scopes` default) — turns the Settings → Connectors grant UI from advisory into a real boundary; needed before the hub opens to third-party agents (#977, #1020).
- **Windows-green unit CI** — GAIA's primary platform has no automated unit coverage today; a single conftest fix + one runner unblocks every Windows contributor and would have caught the two Windows-only test bugs the daemon sub-review found.
- **OpenAI-API auth + loopback-by-default (#630)** so `gaia api` can be safely used from VS Code/Continue on a LAN.
- **Unified "Automations" ledger** (daemon clock as the single scheduler with a UI/CLI view) — the autonomy-engine roadmap item depends on one clock with exactly-once semantics; the daemon already has it, the other two schedulers don't.

## Checked and fine
- `TunnelAuthMiddleware` token compare is `hmac.compare_digest`; UUID4 token; localhost bypass requires *both* raw-peer loopback and no `X-Forwarded-*` header (spoof-resistant in both entry points); cookie is `HttpOnly; SameSite=Strict; Secure` (when https); `?token=` bootstrap only on the SPA index path, stripped by 303, `Referrer-Policy: no-referrer`; `/api/health` is the only exempt path and returns only row counts.
- Starlette 1.6 `allow_origin_regex` is full-matched, and ngrok domains are on the Public Suffix List, so the `*.ngrok-free.app` CORS allowance plus `SameSite=Strict` does not let another ngrok tenant ride the cookie.
- Static SPA serving: `sanitize_static_path` rejects NUL/`..` and re-checks containment after `resolve()`; `/assets/*` misses are real 404s.
- File endpoints: NUL rejected first; `resolve(strict=False)` then containment before any FS call (avoids `PermissionError` oracles); leaf symlinks rejected; `safe_open_document` is lexical + physical containment + `lstat` + `O_NOFOLLOW` + `fstat` S_ISREG (TOCTOU-safe); uploads get UUID names, size caps (20 MB), extension allowlists, and blob uploads stream with abort-on-overflow and atomic rename; `_is_server_owned` gates on-disk deletion.
- `database.py`: every query parameterised (the one f-string builds `?` placeholders); `role` CHECK constraint; `get_stats` is counts only.
- `open_file_or_folder`: argv `Popen`, home-contained, no shell.
- Chat concurrency: global semaphore + per-session lock + `run_manager` guard, 409 on overlap, resources released via `BackgroundTask` on stream teardown; tool confirmation blocks the agent thread with a timeout and denies on expiry; bypass mode logs and emits a warning event per call.
- Global exception handler returns a generic 500 and logs the traceback server-side; router 500s use fixed strings.
- `DispatchQueue`: dependency timeout, terminal states, pruning; `RunManager` buffers freed on finish.
- `PathValidator`: `realpath` + macOS `/private` normalisation, prefix-with-separator containment, fail-closed on exceptions in both allow and write-block checks, non-interactive auto-deny for allowlist prompts, audit log rotated.
- Agent loop / scheduler / scheduled executor: suspended while a tunnel is active unless `GAIA_AUTONOMOUS_ALLOW_TUNNEL=1`; confirmation-gated tools denied in unattended runs.
- Sub-area "checked and fine" lists above cover daemon auth/tokens/relay/custody/migration, OAuth/PKCE/keyring/grants, API-key proxy, governance, POSIX shell tool, MCP stdio transport, bridge auth, TUI MCP, schedule sinks, daemon clock.

## Hypotheses (unverified)
- **DNS rebinding read access** is inferred from the absence of Host/Origin validation and confirmed CORS-only defence; not exercised end-to-end against a live server here.
- `_fetch_public_ip` posts the machine's public IP to logs and status (`tunnel.py:520-530`) — harmless but a privacy nit if logs are shared.
- `index_folder` (`documents.py:642`) indexes every matching file synchronously with no cap — pointing it at `$HOME` recursively could hold the event loop's executor for a long time; not measured.
- `upload_locks` / `session_locks` dicts grow without bound (documented as intentional) — a tunnel client cycling random paths could grow memory; not measured.
- The `X-Forwarded-*`-absent localhost bypass plus `proxy_headers=True` in the CLI path: a *local* attacker process cannot spoof either way, but a reverse proxy other than ngrok that strips `X-Forwarded-*` before hitting the backend would make every remote request look local — configuration-dependent, not a code bug.
- Sub-area hypotheses are listed inline above (relay path normalisation, `GAIA_EMAIL_SIDECAR_TIMEOUT` parsing, cmd.exe `^`/newline separators, PTB `run_polling` off-main-thread, `activation_watcher` corrupt-ledger flapping, `_USER_SESSIONS` growth).
