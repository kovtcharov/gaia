
---

## 4. Minor findings (🟢) — condensed

One line each; locations in the raw reports and verification logs. Items the verification pass demoted from 🟡 are marked ↓.

**Agent loop / registry (01).**
- Skill-discovery note is emitted in the *static* prompt head, so a discovery turn invalidates the whole KV-cache prefix (`VOLATILE_PROMPT_FRAGMENTS` misses `get_skill_discovery_system_prompt`).
- Parse-error recovery double-increments `steps_taken`; `error_count` is shared between tool and parse errors.
- `_LEGACY_ID_ALIASES` comment says `gaia-lite` → flagship but maps to `doc`.
- `get_lemonade_models` caches forever after one success and swallows failures.
- `system_context.py` has 20+ `except Exception: pass` and stores the hostname despite "no personal data".
- Blender-era regex fallback returns a token as the *answer* for unparseable `{`-prefixed responses.
- ↓ `SystemDiscovery.scan_all(sources=[…])` silently drops unknown source names (`discovery.py:3872-3877`) — latent: no CLI flag or in-tree caller passes `sources`.
- ↓ `tool_grants` "unbounded binary" list misses `python3.12`/`pythonw`/`ipython`/`pip`/`wsl`/`ssh`/`docker` — hardening only: none is in `ALLOWED_COMMANDS`, validation runs before the prompt, so no "always" is ever offered for them today.

**Skills (01).**
- `remove_skill` docstring says hub-only; deletes any dir.
- `gaia skill install` accepts `http://` despite help text; no size/ratio cap on download/unpack (zip bomb before traversal check).
- Private key written with default umask then `chmod` (no-op on Windows).
- `skill_synthesis._num` falls back to defaults on invalid config (prohibited pattern).
- `loader.py:91` module-name mangling collides `foo-bar`/`foo_bar`.
- `install_skill` writes the provenance lock *after* `copytree`; a lock failure leaves an installed skill with no provenance.

**Tool mixins / SDK (01).**
- `search_file_content` opens each file twice and swallows every read error.
- Dead validators in `gaia_agent_chat/session.py` with blocking `input()`.
- `create_session(**kwargs)` drops `base_url`, `temperature`, `claude_model`, `use_local_llm` (4 of 13 `AgentConfig` fields).
- `_validate_and_open_file` tests `errno == 40` (Linux `ELOOP`; macOS is 62) and has no callers.

**Memory (01).**
- Procedures FAISS uses module-level `EMBEDDING_DIM` (768) instead of the live embedder dim.
- Dedup replaces DB content but the in-memory vector stays stale until restart.
- Several `except Exception: pass` in `memory.py`/`memory_store.py`; superseded rows and embeddings never pruned; `_faiss_remove` rebuilds the whole index per delete; `GoalStore` has no `busy_timeout` while sharing `goals.db` with the UI router; `recall` pagination silently ends at 4 pages.

**Readiness (01).**
- `InitResponse` docstring vs behaviour (`compatible is None` passes — deliberate); `AgentRequirements.from_manifest` never gets `min_lemonade_version` because `hub.manifest.Requirements` doesn't parse it, so the documented version gate never runs.

**LLM client (02).**
- `launch_server()` kills whatever owns the port when the health probe fails for a non-"down" reason — reachable only via dead entry points.
- `gaia.util.kill_process_on_port` is dead code whose Windows predicate would kill Lemonade's *clients*.
- Every fresh model download in `gaia init` waits an extra 30 s (sleep-before-check).
- Disk-space guard measures CWD, not the model cache.
- `device.py` VRAM floor (24 GB) still sized for Qwen3.5-35B; Gemma-4 needs ~3 GB.
- `LemonadeStatus.url` drops scheme/path; `/completions` forwards llama.cpp-only kwargs to the OpenAI SDK and skips `_ensure_model_loaded`; `_check_version_compatibility` reachable only via dead `initialize()` — `LEMONADE_MIN_VERSION` is enforced only by `gaia init`; `_try_reload_with_ctx` "assumes reload succeeded" silently.

**Installer / CLI (02).**
- `scripts/install-ui.{ps1,sh}` print `amd-gaia.ai/guides/…` (404 — `/docs/` prefix rule; `scripts/` is outside the URL test's scan) and name different products.
- `refresh_path_from_registry` replaces the process PATH, dropping the activated venv.
- `tools/lemonade_stub.py` is unreferenced dead code binding the real port with a contract-unfaithful shape.
- Linux gate docstring "Ubuntu 24.04+ only" while a `debian13` URL is built that no path uses.
- Top-level `--base-url`/`--ui-port`/`--ui-dist` are silently discarded when a subcommand follows (argparse subparser defaults win).
- `gaia prompt` prints the answer then the raw dict; `gaia youtube` with no flag falls to "Unknown action"; `gaia agent` help omits `pack`, `publish`, `login`; `gaia install --lemonade` `input()` without EOF guard; `--hub-token`/`--pypi-token`/`telegram start --token` accept secrets on argv.

**Servers / daemon / connectors (03).**
- Unauthenticated daemon 401s disclose the user's home path; broker lease `timeout` unvalidated (`null` parks a threadpool thread forever); broker `release` never checks the holder; sidecar stdout/stderr logs world-readable while `daemon.log` is 0600; `caller_auth` docstring promises a loud no-token warning that is never logged; custody `user` memory scope documented as shared but stored per-agent and ungated.
- `gaia connectors connect --device` ignores `--grant-agent`; `McpServerHandler.configure` drops non-secret env keys silently; `JsonlReceiptService` default audit path is CWD-relative (`receipts.jsonl`).
- `gaia chat --ui` launches uvicorn without the `proxy_headers=False, forwarded_allow_ips=""` hardening `python -m gaia.ui.server` uses; Agent-UI MCP subprocess spawned with `stderr=PIPE` never drained.
- Telegram: edited messages/channel posts crash handlers; downloads use predictable unbounded temp paths never cleaned.
- `agent_ui_mcp.take_screenshot` swallows `SetForegroundWindow` failure; `system.py` has 6 swallowed exceptions hiding Lemonade `/stats` and catalog failures.
- Silent-swallow inventory: 23 bare `pass/return None/continue` handlers in `src/gaia/ui` (`system.py` 6, `_chat_helpers.py` 4, `memory.py` 3, …) plus the 18 in `ui/routers/memory.py` (I46); ≈76 across `src+hub` outside `ui/`, densest in `agents/base/system_context.py` (20).

**RAG / web / data (08).**
- `PinnedIPAdapter` pin cache never expires (CDN failover → timeouts for the process lifetime).
- `Content-Length` parsed with bare `int()`; `except OSError: pass` in `download_file` cleanup; `remove_document` swallows all exceptions.
- `_llm_based_chunking` applies split positions from a 2,000-char preview to a 6,000-char segment and can loop forever when `chunk_overlap*4 >= segment_size`.
- Encrypted PDFs with an empty user password are refused although pypdf can open them.
- Two `test_hub_installer.py` tests shell out to real `pip` (absent in uv venvs); `_hot_register` puts the agent's site-packages at `sys.path[0]`; single-install guard is process-local (CLI vs UI race); `_write_agent_yaml` swallows fetch errors.
- Scratchpad `_sanitize_name` rewrites names so distinct names collide; `get_size_bytes` swallows exceptions and returns 0, disabling the 100 MB cap; `db_query` has no type check on `sql`.
- Malformed code-index `metadata.json` → bare `KeyError`; `_read_gitignore_patterns` swallows read errors.
- `AsyncTavilyClient` lacks `crawl` (the sync `crawl` at `tavily.py:477` has no `gaia knowledge` subcommand); `_cache_key` raises `TypeError` on non-serialisable kwargs; `extract_json_from_text` gives up at the first unparsable `{`.
- `FileChangeHandler` debounce keyed by path only (the `modified` after every `created` is dropped); `FileWatcher.stop()` forgets a still-running observer.
- `rag/app.py:76` and `rag/demo.py` advertise a `gaia rag` command that does not exist (no `add_parser("rag")`); the docs-vs-argparse diff could not see it because the strings live in `src/`.

**Voice / SD / VLM (08).**
- `AudioRecorder._get_default_input_device` silently falls back to device 0; `TalkSDK` constructs a second unused LLM client.
- SD tool schema tells the LLM the default is `SD-Turbo` (code: `SDXL-Turbo`); two generations within one second overwrite each other; default SD output dir is CWD-relative and the chat agent hides `init_sd` failure at debug.
- `init_vlm()` default `base_url` bypasses `LEMONADE_BASE_URL`; `_parse_page_range("0")` returns the *last* page.

**Email (08).**
- ↓ Email CHANGELOG has no entry for #3234/#3269 and `outlook_query.py:47-52` duplicates `gmail_query.DURATION_OP_RE`.
- `move_to_label` archives unconditionally (behaviour ask #2626); `trust.REVERSIBLE_AUTO_ACTIONS` lists actions the candidate map never emits; `_HTMLStripper` flattens lists/tables into one line.

**C++ (08).**
- cpp-httplib pinned to 0.15.3 (Feb 2024) with no floor on the system-package path; `vcpkg.json` has no baseline; `::tolower` on signed `char`; `HttpClient` builds a fresh client per request; `getStatus`/`checkModelLoaded` swallow `listModels()` failures.

**Frontends (04).**
- Esc means "deny" on the confirmation panel but "cancel the whole turn" on the look-alike question panel.
- `tui/README.md` download table and a `download-target.ts` comment still name `0.23.0` artefacts.
- ↓ The confirmation timeout is hard-coded as "30s" in three strings (`confirmation.go:434,455`, `canonical.go:437`) while `ConfirmationTimeout` is defined at `confirmation.go:175` — drift hazard, all agree today.
- `daemon/client.go:517` builds `/agents/<id>/ensure` without `url.PathEscape`; `control/server.go send()` sleeps while holding `injectMu`; five near-identical text scrubbers with different rule sets (`sanitizeErrorText` lets C1 controls and bidi overrides through).
- `bin/gaia-ui.cjs --serve` binds all interfaces; ~1,900 lines of unreferenced renderer code (`AgentChat.tsx`, `AgentManager.tsx`, `SettingsModal.tsx`, `agentChatStore.ts`); `isErrorContent` prose heuristics style ordinary answers as errors; `saveTitle` has no error handling; `DocumentLibrary` swallows all poll/refresh errors (`:144,190`); `tests/electron/README.md` describes 2 test files; the directory has 24.
- ↓ `src/gaia/electron/` is dead framework code with an unsafe shape (generic `invoke(channel,…)` preload, unvalidated `open-external-link`), loaded by nothing — the root `package.json:8` scripts point at the deleted `src/gaia/apps/jira/webui` — while `docs/spec/electron-integration.mdx:25` still calls it the shell "currently shipping the flagship Agent UI" and pins "Electron 31.0.0+" against `^44`. Delete or archive; fix the dead `install:jira`/`app:jira:*` scripts together with it.

**CI (05).**
- `claude.yml`/`self-assign.yml`/`merge-queue-notify.yml` create a skipped run record on every issue/comment event (~70 % of run records on `main` are no-ops; the auth canary's 5 failures hide in the noise).
- 98 of ~180 jobs have no `timeout-minutes` (incl. every `publish.yml` job); `gh workflow list --all` shows 95 registered vs 72 files (ghosts from deleted branches); `test_unit.yml` macOS job is advisory; `publish.yml:238-243`'s "#1315" comment is stale; `hub-publishing.mdx:326` "redeploy is triggered automatically" is true only for the `workflow_dispatch` path.
- ↓ `runner_heartbeat.yml`'s 10 cancellations pre-date #2306's fix; the live gap is the unmonitored `lemonade-eval` box (I81).
- No unit test exists for `util/validate_release_notes.py` itself.

**Docs (07).**
- ↓ Roadmap and plan status are stale: `docs/roadmap.mdx` "Updated April 13, 2026", `## Shipped` ends at v0.17.2 and v0.18.0–v0.25.0 are all listed as future with April–June due dates while `version.py` is 0.23.1; #768/#746 sit in the v0.17.3 "in progress" table; the v0.20.0 memory work and the v0.23.0 autonomy section are "future"; deleted `code`/`sd` agents are scheduled (#695/#771). Six `docs/plans/*` say "Planning/0%" for shipped features (`email-triage-agent`, `messaging-integrations-plan`, `desktop-installer`, `connectors`, `agent-hub`, `autonomy-engine`); four plans are orphaned from `docs.json` (`bash-agent`, `email-full-autonomy`, `package-publishing`, `typescript-sdk`); `docs/eval.md` is an unreferenced "moved" stub beside `docs/eval.mdx`. Contributor-time cost, not user breakage.
- ↓ `@amd-gaia/gaia` CHANGELOG says 0.1.1 is "unreleased" though it is on npm and tagged.
- `docs/reference/cli.mdx` (3,163 lines) duplicates `terminal-hub.mdx` and `hub-publishing.mdx` and the copies drift (#3087); `cli.mdx:2725` shows `gaia hub install email` reporting 0.5.0; `docs/quickstart.mdx:13-15` calls the Electron app "the primary install path" while terminal-hub.mdx, the v0.23.1 notes and the gaia npm CHANGELOG lead with the terminal hub; `CODEOWNERS:30` and `labeler.yml` (13 targets) cite `.md` doc paths that are all `.mdx` (plus a nonexistent `docs/guides/cpp.mdx`), so a docs-only PR to those pages gets only the generic `documentation` label; `hub/agents/gaia/npm/SKILL.md:25` says skills load from `gaia_agent/skills/<name>/` — on disk only `gaia-voice` is there, the 13 starter skills live in `hub/skills/` (§9 A3).

**Hub Worker / website (04b).**
- Bearer tokens looked up on a plain object without `Object.hasOwn` — `Authorization: Bearer constructor` yields `500 server_misconfigured` (verified live; no bypass, wrong status + operator-blaming message); `/reindex` token compared with `!==`.
- Malformed percent-encoding on a download path → 500 `internal_error` (live); `robots.txt` advertises a sitemap the site never generates (404); `parseScorecardScore` swallows YAML errors so a typo silently removes the eval badge.
- README drift: bucket name `gaia-agent-hub` vs `wrangler.toml` `gaia-hub`; "uncomment the routes line" (it is active); `POST /reindex` + `REINDEX_TOKEN` undocumented (permanent 500 on the Railway demo); website README says the router source "is being brought into the repo" (it landed); `catalog.ts:215-219` hides `agent-ui` as "no longer maintained" while `release_components.yml` still publishes its installers every release and CLAUDE.md/docs present it as supported — `/hub/agent-ui` is a 404 today.

**Tests (06).**
- ↓ Optional-dependency tests error at collection instead of skipping: 30 modules import `fastapi` at module level (176 collection errors on a core-only venv); four of six hub package test dirs (`chat`, `connectors-demo`, `hello-world`, `word-count`) have neither a `conftest.py` nor an `importorskip`; `allow_network` is registered only in `tests/unit/conftest.py` so it is unusable elsewhere under `--strict-markers` (the marker the C41 fix would reach for); `pytest-timeout` is used by five workflows but in no extra (`--timeout` errors locally).
- Order-dependent tests (`test_hotreload_*`, `test_sse_confirmation_gate`, `test_longthread_corpus_integrity`) — `_TOOL_REGISTRY`/`sys.modules` leak; no `pytest-randomly`. `tests/test_agent_sdk.py` spends 31 s probing Lemonade before skipping 8 tests (per-test, not session-scoped).

---

## 5. Test suite assessment

**What is good.** 534 files / 215K lines; `--strict-markers`; 316 `importorskip` guards vs only 2 unconditional skips; the memory/goal stores are tested against real SQLite (~750 tests, all green on Windows); 396 payload-asserting mock calls vs 124 existence-only; several suites (`tests/unit/mcp`, `factory`, `rag`, `test_eval.py` 140) are clean; `test_email_agent.yml` fails when corpus-wired tests are *skipped* — a pattern worth copying. The verification pass re-ran every tally in this section and all matched.

**What is structurally wrong.**

1. **No Windows lane for `tests/unit`** (C41, I74): ≈640 tests (overlapping tallies across the sampled trees) error or fail on a clean Windows checkout for five environmental reasons, none visible to the ubuntu/macos lanes; the narrow Windows lanes that exist miss the guard. Several Windows-specific bugs in this report would have been caught on the day they landed.
2. **Orphaned tests** (I75): ≈40–43 files / ≈780 tests run nowhere; the two largest need only `TestClient`; `tests/test_sdk.py` and `tests/test_lemonade_client.py`'s mock classes have rotted into red because nobody runs them.
3. **Mocks that prove invocation, not validity — at least nine tests pin a bug as the contract**: `tests/test_api.py` (`MagicMock` agent — how C27 shipped); `test_lemonade_manager_preload.py` asserts the buggy call (`load_model(DEFAULT_MODEL_NAME)`); `test_init_ctx_size.py` pins the wrong literal (I16); `test_scope_specific_path` pins arbitrary-path scope (C7); `test_cors_preflight_stays_open` pins C13; `test_no_agent_id_skips_grant_check` pins I14; `test_empty_allowlist_intentionally_allows_all` pins C18; `test_notification_service.js` mocks a method the real class lacks (C35); `test_structured_vlm_extraction.py` pins zero-fill (I20); shell-guardrail tests never exercise what `cmd.exe` receives (C4); `tui/test/mockagent` is single-process (C30); `test_pull_model` never asserts the request body.
4. **Coverage holes by module** (grep-verified table in `06`): `hub/agents/chat` (the flagship's base class) has 3 test files / 221 lines against 4.9K lines of source; `src/gaia/skills/` (12.3K lines) has no integration tier; zero-reference modules ≈5.6K lines incl. `ui/routers/system.py` (1,060), `chat/prompts.py`, `chat/app.py`, `utils/parsing.py`, `shell/prompt.py`, `util.py` (the last five also have no *importer* — dead code). Zero tests reference `_shrink_messages_for_overflow`, `_resolve_plan_parameters`, or `STATE_EXECUTING_PLAN` plan execution (`$PREV`/`$STEP_N`, error → recovery).
5. **Worker/R2 tests prove the wrong thing**: `FakeR2.put` ignores `onlyIf` (a future conditional-write fix for I90 would pass without proving anything), `FakeR2.list` always returns `truncated: false` (the cursor loops are untested), no test injects a mid-publish R2 failure (I91) or a reserved filename (I92), and nothing runs a real by-reference PUT against R2 (`util/check_r2_credentials.py` is run by no workflow).
6. **Order-dependent tests** pass in isolation and fail in chunked runs (§4).

**Per-area gaps worth tests first** (each maps to a finding): contextvar propagation through `_call_tool_bounded`; shell bypass strings; `find_files`/`index_directory`/binary-extractor sandbox; `replace_function` with a decorated neighbour; skill name traversal; hub filename traversal; memory >20-turn consolidation; category minting on `update`; `AgentSDK.send` request shape; CSRF route-table introspection; tunnel pre-URL window; sensitive-path reads via tunnel and via a `$HOME` allowlist; `.bashrc` write; `_encode_texts` row-count invariant; HMAC tamper path; code-index desync; `TalkConfig` from CLI flags; pause/resume around `speak_text`; Outlook-shaped timestamps; Electron `setWindowOpenHandler`; deep-link ordering; a multi-process mock agent for the TUI; docs.json navbar vs `version.py`; release-notes command validation; `validate_release_notes.py` itself; baseline-drift and never-run-file guards; `compare_scorecards` precedence; the judge request shape; PPTX filename quoting; grants scope ceiling; `put_grant`/`prune` CSRF.

## 6. CI/CD and release assessment

**What is good.** `publish.yml`'s core flow is sound: tag must be on `main`, `version.py` == webui `package.json` == tag, PyPI/npm via OIDC trusted publishing, a single `publish` environment approval, post-publish `pip install amd-gaia==<tag>` smoke, Sigstore on dists, reproducible TUI builds with `SHA256SUMS`; the v0.23.0 run went green end-to-end. `pypi.yml` builds on every PR with a negative test for the wheel verifier. Script-injection sweep is clean (no `github.event.*` text in `run:`); `claude-code-action` is SHA-pinned; eval workflows gate the key-bearing path on same-repo PRs; `lint.yml` chains the `util/check_*` scripts.

**What is structurally wrong.**

1. **Release gates run only at tag time** (C39): the release PR merged in an untaggable state because the navbar/notes validators live in `publish.yml`, not in any PR check; `update-release-branch.yml` and `release_components.yml` fire on the tag independently of validation/approval; the gate that *is* required (`email-eval`) is red on billing.
2. **Required checks are unknowable**: `rules/branches/main` is empty, the only ruleset is disabled, classic protection is not visible to the token. Every "gate" in this report is "gate if required".
3. **The eval gate has never produced a scorecard** (`test_eval_agent_gemma_consolidation.yml`: 10 skipped / 4 failed / 0 green in 15 runs; #3016/#2960), `test_eval_rag.yml` is disabled, `email_scorecard_refresh` last ran 2026-07-16. The CLAUDE.md eval rule is enforced by honour only, and `find_scenarios` overrides make a clean scorecard fakeable (I78).
4. **Supply chain** (C21, I23, I67, I94, I81): mutable action tags in the publish/sign path; unpinned Lemonade MSI baked into signed installers; `irm | iex` bootstraps in two shipped paths; install one-liners served from `main` by an out-of-repo rule; Dependabot blind to the Go TUI, npm sidecars, website and Worker.
5. **Untrusted code on persistent self-hosted runners** (C20) and **community-triggered Claude with `Bash` + `contents: write`**, an accepted risk whose bound is unverifiable from outside (I97).
6. **Silent fallbacks in the one workflow that ships** (C40, I79, I80).
7. **Noise that hides signal**: ~70 % of run records on `main` are fire-and-skip; dead/racing workflows; 95 registered vs 72 files; five workflows without a `permissions:` block (I82).

## 7. Documentation assessment

**What is good.** `docs.json` is complete (195 nav pages exist, no broken relative links/images); the `amd-gaia.ai/docs/` URL rule holds everywhere; CLAUDE.md's `KNOWN_TOOLS` table, default model, ctx sizes and `DEFAULT_MAX_STEPS` match code; every `ChatAgentConfig`/`AgentConfig`/`GaiaAgentConfig` kwarg in docs is real; the 13 starter skills conform to the skill format and their `tools_required` all resolve; the gaia and email npm doc bundles (README/SPEC/SKILL) are internally consistent and match `server.py` — the #1841 miss has not recurred; `README.md:131-151` carries a real release-process table.

**What is structurally wrong.**

1. **Nothing ties docs to the CLI or the code at PR time**: 21 `gaia tui …` invocations, `gaia install gaia` in the release notes, 25 dead `gaia eval -d` examples, four impossible imports, seven inert documented flags — all would be caught by a ~60-line "docs-as-tests" pytest that resolves every fenced `gaia …` invocation against `build_parser()` and every `from gaia… import` in the venv (extend it to strings in `src/` too — `gaia rag`).
2. **Status drift**: roadmap 4.7 months / 6 releases stale, six plans claiming "Planning/0%" for shipped features, CLAUDE.md advertising a cancelled rename and omitting five packages and ~20 subcommands. Agents following CLAUDE.md → `docs/plans/` before building are invited to re-plan shipped work.
3. **Security posture is documented nowhere as a whole**, and the pieces that exist contradict the code (I88); the most prominent "Security Model" page is a self-superseded plan.
4. **Missing guides for shipped surfaces**: `gaia daemon` (the process model every sidecar depends on), `gaia schedule`, `gaia eval code`, governance, `GAIA_UPDATE_FEED_URL`/deep links.
5. **Hub package doc bundles are not gated on tags**: the email npm README links to a tag that does not exist; CHANGELOG says a shipped version is unreleased.

---

## 8. Architecture and improvement opportunities

Ordered by leverage (how many findings each closes).

1. **One `FileAccessPolicy` enforced at tool registration, consumed by every file/shell/RAG/screenshot/VLM/SD tool** (closes C7, C8, I13, I34, I58 and the "forgot to call the validator" class). Start with C8's narrow fix (grant files, not parents; sensitive-file denylist on reads; rc/autostart names in the write blocklist) so the refactor does not revert it; include a sensitive-roots denylist shared by `ensure_within_home` and `security.PathValidator`, and a "this file runs on login" tier for the confirmation modal.
2. **argv-based Windows shell execution with an explicit built-in table**; `shell=True` only for that table (closes C4 for good; would let read-only commands skip per-call confirmation honestly, which #2785 wants). Two interim regex fixes first.
3. **`contextvars.copy_context().run(...)` in `_call_tool_bounded` + fail-closed `_authorize_access` + `required_scopes` default for `mcp_server` specs + a scope ceiling in `grants.grant_agent`** (closes C1, C17, I14; turns the Settings → Connectors grant UI into a real boundary before the hub opens to third-party agents). Update `connectors.mdx:257-262` in the same PR.
4. **One request-guard middleware for the UI backend** (`X-Gaia-UI` on every mutating request + `TrustedHost` + `Origin` allowlist; `APIRouter(dependencies=[...])` defaults) replacing four copies of the guard and 24 per-route `Depends` (closes C11 and DNS rebinding) — together with deleting the `*.ngrok-free.app`/devtunnels origin allowance (C12), without which the header is forgeable; `tunnel.auth_required` independent of `_url` (I98); copy `openai_server._cors_config()` into `AgentServer` and mount `caller_auth` (C10, C13).
5. **Collapse duplicate implementations**: `_require_ui_header` ×4; three `read_file`s, two `browse_directory`s, two `search_web`s, `file_tools` vs `file_io_tools` write/edit, ChatAgent inline `list_files`/`fetch_webpage`/`open_url` vs the mixins; `cli._unpack` vs `install._unpack_bundle`; hub `/init` probe vs `readiness.compute_init_status`; `session_registry.py` vs `gaia_agent_email.agent_routes._SessionRegistry`; three email-fact builders in `discovery.py`; `subprocess.go detectLemonadeURL` vs `preflight/local.go probeLemonade`; three schedulers vs `DaemonClock`; three renderer JSON/fence cleaners vs the server-side filter; five TUI text scrubbers; two voice pipelines; the C++ error decoder ×2; `release_agent_{email,chat,gaia}.yml` (2,370 lines over one skeleton). Make `tool()` raise on a name collision (I1).
6. **Single source of truth for the context window** (`ctx_for(device)` used by `INIT_PROFILES`, `MODELS`, `AGENT_PROFILES`, `LemonadeManager`, `_maybe_load_expected_model`, the overflow classifier) — closes I15, I16, I18, I32; device-aware idle preload.
7. **Cooperative cancel + host-owned transcript for the TUI subprocess transport** (closes C30–C31 and the history loss in one move and keeps the warm-agent turn).
8. **Fail-loud sweep of the ≈95 silent-swallow handlers** — start with `ui/routers/memory.py` (18, incl. the silent standalone fallback), `system_context.py` (20), `ui/routers/system.py` (6, hides Lemonade/catalog failures), `memory.py`/`memory_store.py`, `daemon/sidecars/manager.py:644`, `paths.py:132`, `client.py:120`, `receipt_service.py:113-124` (audit-log corruption at DEBUG), `connectors/store.py:558`, `activation_watcher.py:43-47`, `filesystem_tools.py:686-689` (index fallback), `pdf_utils.py:183`.
9. **Tests-as-guards**: Windows unit job (after its four prerequisites); loopback-allowing network guard; never-run-test-file guard; baseline-drift guard; docs-as-tests for CLI invocations and imports; navbar-vs-`version.py` unit test; route-table CSRF introspection; registry-shadowing test; a multi-process TUI mock agent; `pytest-randomly`; `pytest-timeout` in `[dev]`.
10. **Release DAG, not a race**: reusable `validate` job called from `docs.yml`/`pypi.yml` on `version.py` changes; `update-release-branch`/`release_components`/`publish_agents` chained after `github-release`; SHA-pin every third-party action; pin the Lemonade MSI in both places; `timeout-minutes` everywhere; `permissions:` blocks; delete dead workflows; separate "eval infra failed" from "eval measured a regression" in the release gate.
11. **Memory hygiene**: category validation in `MemoryStore.store/update` (C16); windowed consolidation + prune after (C28); background cancellable extraction (I6); FAISS-id candidate fetch (I7); end-of-day normalisation for date-only bounds (I8); prune superseded rows/embeddings; `IndexIDMap.remove_ids`; `busy_timeout` on `GoalStore`; fence scanner output in `_INFER_PROMPT` (I47).
12. **Voice**: collapse to the pausing pipeline; one stdin listener per session; `is_recording` as a property over thread liveness; surface TTS failure; non-zero exit on loop crash (C33, C34, I60).
13. **RAG**: length-check every embedding batch (I50); embedder-aware chunking (I51); persist embeddings in the signed cache + `IndexIDMap2` removal (I52 and minutes of re-embedding at every process start); distinguish integrity failure from staleness and validate the key length (I53); chunker fingerprint in the cache key; `~/.gaia/cache/rag/` default; re-raise `PermissionError` in `_get_cache_path` (C7).
14. **Data layer**: one `readonly_query(conn, sql, *, timeout_s, max_rows)` in `sql_safety.py` used by `DatabaseMixin` and scratchpad (I54); sign `code_index/metadata.json` like the RAG cache *and* replace the count check with an identity check (I55).
15. **Docs**: generate `cli.mdx` option tables from `build_parser()`; front-matter `status:` on every plan; fold the TUI section of `cli.mdx` into `terminal-hub.mdx`; a `docs/security/` trust-boundary page (and archive the superseded plan); CLAUDE.md: replace the hand-maintained CLI list and tree with "run `gaia -h` / `ls src/gaia`" plus non-obvious pointers; extend `check_doc_links.py` to `.claude/**`, `AGENTS.md`, `CONTRIBUTING.md`, `hub/**/*.md`.

---

## 9. High-impact feature opportunities (ranked)

Scoring: user impact × how much already exists ÷ size. "Finish what is half-built" ranks first because each item closes a gap between the website's promise and the install. Evidence: the roadmap/plans-vs-code table and the 698-issue clustering in `08 §B`; the landing page's ten claims vs code in `08 §B.2` (three ✅, five ⚠️ half, two ❌). Every item below names both an issue and a code path; the verification pass traced each.

### A. Finish what is half-built

1. **Make the eval gate real** (M, low risk). The automated gate has never produced a scorecard (#3016: 0 successes in 186 runs since 2026-07-18 — 62 failures / 124 cancelled), the judge key is empty on PRs (#2960), the RAG gate never ran (#1315), the tool-cost baseline never runs (#3294), the 80-min email eval blocks releases (#2958) and is currently red on billing (C39), and — from this review — the serial guard is a no-op on Windows (I76), baselines exist for 3 of 12 categories, `--save-baseline` writes somewhere the docs don't mention, and local scenario overrides are invisible (I78). Everything else in this list is un-regressable until this works; the fix is mostly ops plus a small runner change plus a nightly job on the existing Strix Halo lane.
2. **Cloud-inference option for the agent-under-test** (S/M, low). Judge already rides the subscription; `AgentConfig(use_claude=…)` exists; `gaia eval agent --provider claude|openai` + a per-provider baseline set + a GitHub-hosted fast lane on every PR (hardware lane nightly). Opt-in only — "no silent cloud fallback". Unblocks (1) without waiting on hardware.
3. **Ship the skills story end-to-end** (M, medium). Website: "Skills, not more installs"; the per-task agents were deleted in favour of skills — yet no workflow publishes `hub/skills/*` (#3090), the frozen flagship ships 1 of 12 skills (#3057), the daemon has no skill route, the TUI has no skills screen, `default_skill_set` is commented out pending an eval gate. Everything client-side exists (`src/gaia/skills/` 8.8K lines, Worker `POST /publish/skill`, `skill_audit.yml`, `TrustStore`, the TUI trust gate). Missing: `release_skills.yml`, `freeze.py` staging, `GET/POST /daemon/v1/skills`, a TUI list/install screen, the `index.schema.json` fix (I93). Prerequisite: fix C3, C5, C6 and add load-time verification (`gaia skill verify`) first — the ecosystem should not open on today's substrate.
4. **Long-conversation policy on top of memory (#686) + memory reliability** (M, medium; eval-gated). Today: a hard ring buffer (`chat/sdk.py:113`), one shrink-and-retry on overflow (#2780 recovers 81 of 6,902 tokens), memory silently disables without the embedder (#2831), neither documented way to turn memory on works (#3173), recall discards the KV prefix (#2686), procedures truncate mid-step (#2676), outcome counters never update (#2865), consolidation broken for >20-turn sessions (C28), category minting (C16), the top-200 pool (I7), the 8 s timeout that isn't (I6). Needed: extract durable facts before a turn falls off the deque, a pinned "safety + preferences" block, a token-budgeted history using the real tokenizer (#2772), fail loud when memory can't start.
5. **NPU path reliability** (M/L, high). The most-discussed external bug is the FLM↔Vulkan thrash (#1676/#1746); `--profile npu` → zero agents (#2972); the email sidecar loads the NPU model anyway (#3152); overflow misrouted to a reload (#2884 = I18); `flm:npu` install rejected (#3175); this review adds I15 (init pulls the GPU model on NPU boxes), I16 (32K vs 64K), I32 (UI reload halves the window). One device profile resolved once and honoured by every process, a hardware-lane baseline, and a cold `gaia init --profile npu --force-models` integration test.
6. **Cold-start / installer correctness suite** (M, low). `gaia init → gaia chat` fails from plain PyPI (#2260); quickstart omits extras (#1074) and the terminal hub (#3293); uninstall leaves binaries (#3290/#3236) — and this review adds C23, C39, I23 (unpinned MSI), I24 (NSIS dialog), I26, I67/I68 (Electron rescue paths), C35/C36, I94 (install script served from `main`). A CI matrix that installs the built artifact on a clean VM per OS and runs `gaia init && gaia chat "hi"` + uninstall; SignPath/notarization wiring; a Lemonade-version check at init (#3175).
7. **Real remote-access security for the tunnel + a grant/permission model that holds** (S/M each, low). Scoped tokens (chat-only vs full), expiry/revocation, a "what this QR grants" screen, sensitive-path denylist (I34, I98, C12); server-side standing grants with a Settings view (C15, I13, C31); a scope ceiling on the grants ledger (C17); `GovernedAgentMixin`/receipts wired behind a flag with `/api/audit` + a timeline tab (#3096, #2214) so "asks before it acts" becomes inspectable, with per-argument rendering in the consent prompt (I45). Needed before #894/#898/#2527 (PWA, tunnel UX, headless) can ship on this backend.
8. **TUI ↔ Agent UI convergence, starting with a dynamic daemon catalog** (L, or S for the first step). Two full frontends (59K Go, 31K TS) against one contract; the daemon catalog is a hard-coded 2-entry table (`daemon/sidecars/spec.py:322-344`) so neither can show a third agent. First step: hub index + installed `gaia.agent` entry points → both frontends inherit new agents; then converge rich rendering on `tool_result.render` cards (#2351). Fix C30–C31 as part of it — "Esc" is the one thing a user reaches for when the agent is about to do something wrong.

### B. Net-new

9. **Second messaging adapter (Slack or Signal) on a shared `MessagingAdapter` ABC** (M, medium). Only Telegram exists, open by default (C18), broken `--background` (I39), no e2e (#2062); the plan doc still says "no implementation". Slack Socket Mode is lowest friction; Signal needs a `signal-cli` sidecar. Rate limiting (#689) and a restricted tool set per adapter (#690) first.
10. **Opt-in hybrid routing — local-first, cloud on request** (M, medium). #632/#1236/#2875; `use_claude`/`use_chatgpt`, both providers and the factory exist; missing a per-session switch in TUI + Agent UI, RAG synthesis on the cloud model, cost display (#649), and an explicit logged never-default policy. Doubles as (2)'s runtime.
11. **Voice-first parity (#702)** (M, medium): wire `--model` (C34), pause the mic (C33), barge-in + echo suppression (I60), server-side ASR/TTS (#373/#386) to drop local model downloads, push-to-talk in the TUI. Most pieces exist unused in `AudioClient.process_voice_input`.
12. **Dogfood the GitHub agent on the backlog** (S/M, low): 698 open issues, 54 unlabeled, 56 nightly-audit filings with no owner, 5 good-first-issues, external bugs (#1394, #1382, #999, #1068) idle for months, a roadmap five months stale. A weekly stale-sweep that closes/merges duplicates, re-labels, and regenerates the roadmap "Shipped" section from release notes.

### C. Small, concrete module-level wins (each unblocks a promise above)

Web fetch that reaches the web (C32 — one adapter method); length-checked embedding batches (I50); embedder-aware chunking (I51 — the biggest retrieval-quality lever); persist embeddings in the signed RAG cache (minutes of GPU time per process start today); a query budget for LLM SQL (I54); scratchpad `load_table_from_file` (CSV/XLSX → table; today rows go through a 10 MB JSON call); multi-repo code index (I55); Outlook parity for the email agent (I61/I62 — one timestamp normaliser + ~6 operator mappings); a generic IMAP/SMTP provider (#2619 — every backend already speaks the Gmail payload shape through one decoder); hub artifact signing beyond same-origin SHA-256 (skills already have ed25519 to reuse); a structured-extraction confidence surface (I20); C++ P1.2 embeddings / P2.2 RAG on the existing `VectorIndex` and an HTTP MCP transport (after C19); a default Electron update channel (I69 — the whole updater is built and unreachable); docs-as-tests; a hub package doc-bundle gate (README/SPEC/SKILL/CHANGELOG present, top dated entry == package version, every `blob/<tag>/` link resolves); a `gaia daemon` guide.

### Explicitly deprioritised
CUA/desktop control (#224/#460), Home Assistant (#705), finance/CRM/photo/meeting agents (#1490–#1502), C++ `gaia-agent` (#2804 — readiness doc says no-go), Docker containers, OS-agents MCP servers, Power-Automate Outlook bypass: plan-only, ≤3 comments, zero external reactions.
