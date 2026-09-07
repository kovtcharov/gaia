# 02 — LLM layer, Lemonade client, installer / init, CLI entrypoint

Reviewer dimension: `src/gaia/llm/`, `src/gaia/installer/`, `src/gaia/cli.py`, `src/gaia/cli_agent.py`,
`src/gaia/{version,device,config,util,logger,security}.py`, `installer/`, `scripts/`, `tools/lemonade_stub.py`,
and the matching tests. Commit 211f08c5 (v0.23.1).

## Test run on main (`.venv\Scripts\python.exe -m pytest <scope> -q`)

Command: `tests/test_lemonade_client.py tests/test_lemonade_embeddings.py tests/test_lemonade_health.py tests/unit/installer tests/unit/cli tests/unit/test_cli_*.py tests/unit/test_device_check.py tests/unit/test_init_command.py tests/unit/test_lemonade_*.py tests/unit/test_security_edge_cases.py tests/unit/test_claude_provider.py tests/unit/test_base_agent_lemonade_api_key.py tests/installer`

Result: **34 failed, 886 passed, 33 skipped, 15 errors** (log: `.review/_02_pytest.log`). `tests/unit/test_lemonade_error_classification.py` cannot even be collected in a core-only venv (`ModuleNotFoundError: fastapi` via `gaia.ui._chat_helpers`) — it lives under `tests/unit/` but needs the `[ui]` extra.

Classification of the failures (details in Findings / Test gaps):

| Bucket | Tests | Real bug on main? |
|---|---|---|
| Mock tests drifted from the client | `TestLemonadeClientMock::{test_chat_completions, test_chat_completions_nonstream_includes_auth_header, test_get_required_models_for_code, test_validate_context_size_insufficient}` | Yes — tests are stale; CI only runs `-k Integration` so nobody sees it |
| Posix-only test assumptions run on Windows | `test_uninstall_command.py` (7), `test_lemonade_launcher.py` darwin cases (7), `test_init_command.py::TestInstallViaPpa` (7) + `test_check_installation_finds_macos_install`, `test_lemonade_macos_install.py` errors (15) | Test portability gap, not product bug |
| Env-dependent harness | `test_cli_refusal_exit_codes.py` (4: subprocess env has `HOME` but not `USERPROFILE` → `Path.home()` raises at `gaia.logger` import), `test_cli_smoke.py::test_gaia_binary_on_path` (3: venv Scripts not on PATH), `test_install_scripts_terminal_hub.py::test_sh_parses_under_dash` (CRLF checkout; index is LF — no `.gitattributes` pin) | Harness / checkout artifacts; one small product edge (logger import crashes when home dir unknown) |
| Unexplained | `tests/installer/test_custom_agent_mcp_harness.py::test_custom_agent_dummy_mcp_path_uses_installed_bundle` | see below |


## Scope covered

Read fully: `src/gaia/llm/lemonade_client.py` (4818 lines), `lemonade_manager.py`, `lemonade_launcher.py`, `lemonade_embedded.py`, `vlm_client.py`, `factory.py`, `base_client.py`, `embedding_cache.py`, `exceptions.py`, `providers/{lemonade,claude,openai_provider,litellm}.py`; `src/gaia/installer/init_command.py` (2438 lines); `src/gaia/{version,device,config,util,logger,security}.py`; `.claude/skills/lemonade-client-patterns/SKILL.md`; the relevant CI workflow steps; the failing tests' sources. Cross-checked callers in `rag/sdk.py`, `code_index/sdk.py`, `ui/_chat_helpers.py`, `cli.py:79-210`.

Delegated to two sub-reviewers (their verified findings are merged below, marked *(sub-review)*): `src/gaia/cli.py` + `cli_agent.py` handlers and the docs/parser cross-check; `installer/lemonade_installer.py`, `uninstall_command.py`, `export_import.py`, `mcp_init.py`, the `installer/` tree, `scripts/`, `tools/lemonade_stub.py`.

Not read line-by-line: `tests/test_lemonade_client.py` beyond the failing tests and the pull/recipe contract tests (lines 860-1070), `hub/` agents, `src/gaia/ui/`.

**Version premise check:** `src/gaia/version.py` is `__version__ = "0.23.1"` on this commit and `.venv` metadata reports `0.23.1`; the release commit 72241b2f bumped it. Not a bug.

## Findings

### 🔴 `GAIA_HOME` pointed at the user's home turns `gaia uninstall --purge` into deletion of `~/Documents` and `~/venv` *(sub-review, quotes re-verified)*
- **Where:** `src/gaia/installer/uninstall_command.py:94-96` (`_gaia_home`), `:154-164` (`_purge_paths`), `:166-175` (`_safe_roots`), `:449-471` (`_remove_path` containment guard).
- **What:** `GAIA_HOME` is honoured as the state root by `security.py:89` and `lemonade_embedded.py:144`, so setting it to `~` is a realistic misconfiguration. The purge list is then `$HOME/venv`, `$HOME/chat`, `$HOME/documents`, … and the containment guard is vacuous because the allowed root *is* `$HOME`. On Windows (and default-APFS macOS) `documents` resolves case-insensitively to the real `Documents` folder.
- **Failure scenario:** `GAIA_HOME=%USERPROFILE%` + `gaia uninstall --purge --yes` → `shutil.rmtree` on the user's real `Documents` and any `~/venv`. Reproduced by the sub-reviewer with a fake home: the plan lists `…\fakehome\documents exists=True`, resolves to `…\fakehome\Documents`, `is_relative_to(root)` is `True`.
- **Evidence:** `env_override = os.environ.get("GAIA_HOME"); if env_override: return Path(env_override).expanduser().resolve()` (94-96); `return [gaia / "venv", gaia / "chat", gaia / "documents", …]` (155-163); `if resolved.is_relative_to(root): inside_any = True` (462-465).
- **Fix:** Refuse to build a plan when the resolved GAIA home equals or contains `Path.home()`, a drive root, or `/` (`RuntimeError` naming `GAIA_HOME`); require a GAIA marker (`config.json`) before purging; print the resolved root in the confirmation prompt.
- **Confidence:** High on the code path; Medium on how often users set `GAIA_HOME=~`.
- **Tracked:** none found

### 🔴 `gaia kill --port N` / `gaia api stop` can kill an unrelated process — substring match on the whole netstat line, no LISTENING filter *(sub-review, quotes re-verified)*
- **Where:** `src/gaia/cli.py:4441-4456` (`kill_process_by_port`, Windows), `:4470-4483` (Unix `lsof -ti:PORT` + `kill -9`); callers `:3517/3524` (`gaia kill`), `:4923` (`gaia api stop`).
- **What:** The predicate is `if f":{port}" in line` over every `netstat -ano` line — no state filter, no anchoring. `:80` matches `:8080`/`:8000`, `:420` matches `:4200`, and a line whose **foreign** address is the port (a client connected to the target) matches too. On Unix `lsof -ti:PORT` lists both ends of every connection and every PID gets `kill -9`.
- **Failure scenario:** (1) Nothing on 80, `gaia kill --port 80` → first line containing `:80` is `0.0.0.0:8080 … LISTENING <pid>` → kills whatever owns 8080. (2) Linux `gaia kill --lemonade` when `lemonade-server stop` fails → `lsof -ti:13305` returns Lemonade **and** the Agent UI backend, the daemon, and any `gaia chat` connected to it → all SIGKILLed. (3) Windows `gaia api stop` with a browser tab open: the ESTABLISHED client line can precede the listener and the browser is killed instead. `gaia api status` (`:4914-4917`) already notes "a bare socket probe can't tell it from any process on the port" — `stop` does exactly that.
- **Evidence:** `output = subprocess.check_output(["netstat", "-ano"]).decode()` … `if f":{port}" in line: … pid = int(parts[-1]) … subprocess.run(["taskkill", "/PID", str(pid), "/F"], …)` (4441-4456); `subprocess.check_output(["lsof", f"-ti:{port}"])` … `subprocess.run(["kill", "-9", str(pid)], …)` (4472-4483).
- **Fix:** Parse netstat columns; require `state == "LISTENING"` and `local.rsplit(":",1)[1] == str(port)`; on Unix `lsof -nP -iTCP:{port} -sTCP:LISTEN -t`. For `api stop` / `kill --lemonade` confirm the owner (`psutil.Process(pid).name()` contains `lemonade`/`gaia`, or a `/health` probe) before killing — the same identity check `lemonade_embedded._daemon_alive` already does. Also `.decode()` is strict UTF-8 on the OEM-code-page `netstat` output, so non-English Windows hits `UnicodeDecodeError` → swallowed at `:4537` as "Error killing process on port" (use `errors="replace"`).
- **Confidence:** High
- **Tracked:** none found

### 🟡 NPU users get the GPU model downloaded and loaded on every cold start
- **Where:** `src/gaia/llm/lemonade_manager.py:729-733` + `:797-847` (`LemonadeManager.ensure_ready` → `_try_preload_with_ctx`); triggered from `src/gaia/cli.py:186-197` and `src/gaia/installer/init_command.py:1933-1935` (`_verify_setup`).
- **What:** When the server is idle (no LLM loaded), `ensure_ready` unconditionally preloads `DEFAULT_MODEL_NAME` (`Gemma-4-E4B-it-GGUF`, the Vulkan/llama.cpp build) with `auto_download=True`. The NPU profile deliberately never downloads that model (`init_command.py:1667-1670`: "NPU profile uses FLM models exclusively — don't append GGUF model") and its agent uses `gemma4-it-e2b-FLM` (`agents/registry.py:336`).
- **Failure scenario:** Fresh Ryzen AI box, `gaia init --profile npu` → models pulled (nothing loaded) → step "Verifying setup" calls `ensure_ready(min_context_size=32768)` → server idle → downloads ~3 GB Gemma-4 GGUF and loads it on Vulkan. Same on every `gaia chat --device npu` after a Lemonade restart. The FLM chat model then loads on top (the two-backend thrash class of #1676). A dev box that already has a model resident never sees this (hidden-state masking).
- **Evidence:** `if context_size_value == 0 and not llm_models_loaded: cls._try_preload_with_ctx(client, min_context_size, quiet, cls._lock)` (729-732); `client.load_model(DEFAULT_MODEL_NAME, ctx_size=min_context_size, prompt=False, auto_download=True)` (842-847); `_download_models`: `if self.profile not in ("sd", "npu") and not self.skip_chat_model: ... + [DEFAULT_MODEL_NAME]` (1667-1671) vs `_verify_setup`: `if self.profile != "sd" and not self.skip_chat_model:` (1955) — verify does *not* exclude `npu`.
- **Fix:** Resolve the preload model from the configured device/profile (`GaiaConfig.default_device == "npu"` → `MODELS["gemma-4-e2b"].model_id`), or skip the idle-server preload when the caller names a model; make `_verify_setup` use the same `("sd","npu")` exclusion as `_download_models`.
- **Confidence:** High on the code path; Medium on hardware behaviour (not run on NPU here).
- **Tracked:** related #1676 (FLM↔GGUF loading loop); the init/preload trigger is not named there.

### 🟡 `gaia init` pins Gemma-4 to a 32K window that the runtime immediately reloads to 64K
- **Where:** `src/gaia/installer/init_command.py:92,102,154` (`INIT_PROFILES[...]["min_context_size"] = 32768`), `:1791-1800` (`_test_model_inference` → `load_model(..., ctx_size=min_ctx, save_options=True)`), `:1933-1937` (`ensure_ready(min_context_size=min_ctx)` → "Context size verified: 32768 tokens"); vs `src/gaia/llm/lemonade_client.py:173,334` (`GPU_CTX_SIZE = 65536`, `MODELS["gemma-4-e4b"].min_ctx_size=GPU_CTX_SIZE`) and `:3245-3260` (reload when `loaded_ctx < expected_ctx`).
- **What:** Two sources of truth for the GPU context window disagree. `gaia init --profile chat` loads Gemma-4 at 32768 **and persists it** (`save_options=True`) into Lemonade's per-model config, prints "Context size verified: 32768", then the first `gaia chat` sees 32768 < 65536 and does a full unload/reload (the ~100 s cold reload the comment at `rag/sdk.py:506` describes).
- **Failure scenario:** User runs `gaia init --profile chat`, sees success, runs `gaia chat`; first turn stalls for a model reload. Any Lemonade-side auto-load (tray, `lemonade load`) uses the persisted 32K and re-triggers the reload.
- **Evidence:** `"min_context_size": 32768,` (init_command.py:92); `client.load_model(model_id, auto_download=False, prompt=False, ctx_size=min_ctx, save_options=True)` (1795-1800); `min_ctx_size=GPU_CTX_SIZE,` (lemonade_client.py:334); `"loaded at ctx={loaded_ctx} but GAIA expects ctx={expected_ctx}; reloading."` (3257-3260). `tests/unit/installer/test_init_ctx_size.py:116-122` pins the literal `32768` for profile `chat`.
- **Fix:** Derive `min_context_size` for GPU profiles from `GPU_CTX_SIZE` (keep 32768 only for `npu`) and update `test_init_ctx_size.py`; or drop `save_options=True` from the verify step so init never persists a window smaller than the runtime's.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `check_model_loaded()` checks "downloaded", and init's verify step evicts every resident model because of it
- **Where:** `src/gaia/llm/lemonade_client.py:4155-4175` (`check_model_loaded`), `src/gaia/installer/init_command.py:1786-1789` (`_test_model_inference`).
- **What:** `check_model_loaded` reads `/models` (the downloaded catalog, not `/health.all_models_loaded`) and also matches on substring, so it returns True for any downloaded model. `_test_model_inference` then calls `client.unload_model()` with **no model name** — a global unload — before loading the LLM under test.
- **Failure scenario:** `gaia init --profile chat` verify loop: embedder loaded and verified → next model is the LLM → `check_model_loaded("Gemma-4-E4B-it-GGUF")` is True because it is on disk → `/unload` (global) evicts the embedder and everything else → each LLM verification costs a full cold reload of all slots; with the `all` profile this repeats per LLM.
- **Evidence:** `models_response = self.list_models() ... if model_id.lower() in model.get("id", "").lower(): return True` (4166-4172); `if client.check_model_loaded(model_id): client.unload_model()` (1788-1789).
- **Fix:** Implement `check_model_loaded` on `get_status().loaded_models` (`_find_loaded_entry`), drop the substring match, and pass `model_name=model_id, ignore_if_not_loaded=True` to the unload.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Four `TestLemonadeClientMock` tests fail on `main`; CI never runs the mock class
- **Where:** `tests/test_lemonade_client.py::TestLemonadeClientMock::{test_chat_completions, test_chat_completions_nonstream_includes_auth_header, test_get_required_models_for_code, test_validate_context_size_insufficient}`; `.github/workflows/test_gaia_cli_windows.yml:312` and `test_gaia_cli_linux.yml:398` run `-k "Integration"` only.
- **What:** The non-streaming `chat_completions` path now calls `_ensure_model_loaded` (a `/load` POST) before the request; the two chat tests only mock `/chat/completions`, so they die with "Connection refused by Responses … POST /api/v1/load". `test_get_required_models_for_code` asserts a `code` agent profile that no longer exists in `AGENT_PROFILES`. `test_validate_context_size_insufficient` asserts a `--ctx-size` hint that the modern-Windows launcher intentionally no longer emits.
- **Failure scenario:** Any client change is "validated" against a suite that is already red, so a real regression in the non-streaming path (the exact class of bug #2513/#1030 fixed) is indistinguishable from the pre-existing failures. The skill file itself warns about this ("Never assume a failure is pre-existing").
- **Evidence:** `.review/_02_pytest.log`; `AssertionError: 'Gemma-4-E4B-it-GGUF' not found in []` (test line 1300); `REM Run only integration tests (skip mock tests which don't need server)` / `python -m pytest tests\test_lemonade_client.py -vs --tb=short -k "Integration"` (workflow :311-312).
- **Fix:** Mock `/health`, `/models`, `/load` (or patch `_ensure_model_loaded`, as the skill file recommends) in the two chat tests; replace `code` with `chat`; make the ctx-hint assertion tooling-aware; add `tests/test_lemonade_client.py -k Mock` to the unit-test job so it runs without a server.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Context-overflow on the NPU profile is mis-classified as "retryable" and re-sent once
- **Where:** `src/gaia/llm/providers/lemonade.py:237-245` (`_classify_lemonade_response`), `src/gaia/ui/_chat_helpers.py:236-244` and `:1610-1625`.
- **What:** Overflow errors are marked `retryable=True` whenever the reported `n_ctx < 65536` — a GPU-profile literal. The NPU profile's ceiling is `NPU_CTX_SIZE = 32768` (`lemonade_client.py:174`), so a genuine "conversation too long" on NPU is always flagged as a wrong-ctx load.
- **Failure scenario:** NPU user overflows 32K → provider raises `LemonadeContextOverflowError(retryable=True)` → chat layer logs "reloading model and retrying once", `_maybe_load_expected_model` is a no-op (already at the expected 32K), the identical request is re-sent and fails again; the user waits twice as long and reads "Reloading the model — give it a few seconds" instead of "start a fresh task".
- **Evidence:** `if 0 < n_ctx_reported < 65536: err_instance.retryable = True` (lemonade.py:243-244); `# Threshold tracks the chat / rag profile default (65536)` … `if 0 < n_ctx < 65536: err.retryable = True` (_chat_helpers.py:240-242); `if classified is None or not classified.retryable: raise` then `result = agent.process_query(request.message)` again (1610-1625).
- **Fix:** Compare `n_ctx` against the expected ctx for the loaded model/device (`profile_ctx_size(device)` or `MODELS[...].min_ctx_size`) instead of a literal; the provider comment already admits the pinned case (#1892) is wrong.
- **Confidence:** High
- **Tracked:** none found (adjacent: #2513)

### 🟡 `VLMClient` rewrites `https://…` and any path prefix to `http://host:port/api/v1`
- **Where:** `src/gaia/llm/vlm_client.py:100-110` (`VLMClient.__init__`); reached from `providers/lemonade.py:500-505` (`LemonadeProvider.vision`).
- **What:** The constructor parses `base_url` into host/port and hands `host=`/`port=` to `LemonadeClient`, whose explicit-host branch always builds `http://{host}:{port}/api/v1` (`lemonade_client.py:995-999`). Scheme and any reverse-proxy path prefix are lost; `server_url` shown to the user is also `http://`.
- **Failure scenario:** `LEMONADE_BASE_URL=https://xyz.ngrok.app/api/v1` (the remote setup `cli.py:142-146` explicitly preserves for chat) → every vision/OCR call goes to `http://xyz.ngrok.app:443/api/v1/chat/completions` → connection error → `"[VLM extraction failed: …]"` text (next finding) is stored as the page content.
- **Evidence:** `parsed = urlparse(base_url); host = parsed.hostname or "localhost"; port = parsed.port or 13305; self.server_url = f"http://{host}:{port}"; self.client = LemonadeClient(model=vlm_model, host=host, port=port, api_key=api_key)`.
- **Fix:** Pass `base_url=base_url` through to `LemonadeClient` (its `base_url` branch preserves scheme) and derive `server_url` from `client.base_url`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 VLM extraction failures are returned as document text instead of raised
- **Where:** `src/gaia/llm/vlm_client.py:204-207, 278-291` (`extract_from_image`), `:169-182` (`_ensure_vlm_loaded` swallows the load error).
- **What:** Every failure path returns the string `"[VLM extraction failed: …]"`. No caller checks that prefix (grep over `src/`, `hub/`: zero matches outside `vlm_client.py`), so a failed OCR silently becomes the extracted content and gets chunked/embedded by RAG or returned from `LemonadeProvider.vision()` as the answer.
- **Failure scenario:** Lemonade evicts the VLM mid-batch (or the ctx is too small) → 200 pages of `[VLM extraction failed: Backend error: …]` are indexed; later Q&A "finds" them and the user sees a confident answer quoting an error string. Violates CLAUDE.md "No Silent Fallbacks".
- **Evidence:** `return f"[VLM extraction failed: {error_msg}]"` (207, 282, 291); `logger.error(f"Failed to load VLM model: {e}") … return False` (178-182).
- **Fix:** Raise a typed `VLMExtractionError` (carrying page/image numbers) and let `extract_from_page_images` / the RAG pipeline decide whether to skip or abort; keep the placeholder string only behind an explicit opt-in.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `.env.example` points `LEMONADE_BASE_URL` at the pre-10.1 port 8000
- **Where:** `.env.example:14`; vs `src/gaia/llm/lemonade_client.py:50-55` (`DEFAULT_PORT = 13305`, with the migration note).
- **What:** The one documented env-file example uses `http://localhost:8000/api/v1`. Nothing has listened on 8000 since Lemonade 10.1; the minimum supported version is 10.2.0.
- **Failure scenario:** New user copies `.env.example` → `.env` (the recommended way to set `ANTHROPIC_API_KEY`), keeps the sample value → `load_dotenv()` in `lemonade_client.py:43` makes every GAIA command target port 8000 → "Lemonade server is not running" while the tray says it is.
- **Evidence:** `LEMONADE_BASE_URL=http://localhost:8000/api/v1` (.env.example:14).
- **Fix:** `LEMONADE_BASE_URL=http://localhost:13305/api/v1` (or comment it out so the code default applies).
- **Confidence:** High
- **Tracked:** none found

### 🟡 `create_lemonade_client()` / `initialize_lemonade()` ignore `LEMONADE_BASE_URL` and force `http://`
- **Where:** `src/gaia/llm/lemonade_client.py:4573-4592` (`create_lemonade_client`), `:4692-4723` (`initialize_lemonade`); `.claude/skills/lemonade-client-patterns/SKILL.md` ("Factory must mirror `__init__` … callers that use the factory (like CLI entry points)").
- **What:** The public factory reads undocumented `LEMONADE_HOST`/`LEMONADE_PORT`/`LEMONADE_MODEL` env vars and always passes `host=`/`port=`, which takes the branch that discards `LEMONADE_BASE_URL` (scheme, path). `initialize_lemonade` hard-defaults `host=DEFAULT_HOST, port=DEFAULT_PORT`. Neither has an in-tree caller (grep: only the module's own `__main__`), and CI still exports `LEMONADE_PORT=13305` (`test_gaia_cli_linux.yml:398`) which nothing else reads.
- **Failure scenario:** An SDK user follows the docstring example (`create_lemonade_client(model=…, auto_start=True)`) with `LEMONADE_BASE_URL=https://remote/api/v1` set → the client silently targets `http://localhost:13305`; with `auto_start=True` it then runs `kill_process_on_port(13305)` + launches a local server (next finding).
- **Evidence:** `server_host = host or env_host or DEFAULT_HOST; server_port = port or (int(env_port) if env_port else DEFAULT_PORT); client = LemonadeClient(model=model_name, host=server_host, port=server_port, …)`.
- **Fix:** Route both through `_get_lemonade_config()` (pass `base_url=`), or delete the two dead entry points and the skill-file sentence that claims the CLI uses them.
- **Confidence:** High (Medium on user impact — dead in-tree, public in the SDK)
- **Tracked:** none found

### 🟡 Lemonade installers run elevated with no integrity check beyond TLS *(sub-review, re-verified)*
- **Where:** `src/gaia/installer/lemonade_installer.py:445-460` (`download_installer`), `:603-636` (`_install_windows` → `msiexec /i`), `:742-752` (`_install_macos` → `sudo installer -pkg`); `.github/workflows/build-installers.yml:343-356` (bundled MSI, size-only guard); grep for `hashlib|sha256|Authenticode|check-signature` in the installer: none.
- **What:** The MSI/.pkg is downloaded from GitHub and handed straight to `msiexec` (UAC) / `sudo installer`. No SHA-256 pin, no Authenticode / `pkgutil --check-signature`; the workflow that bundles the MSI only checks `size > 1MB`. The repo already states and enforces the rule for the sidecar (`installer/tui/fetch_sidecar.py:9-13`) and for the embedded server (`lemonade_embedded.EMBEDDABLE_SHA256`) — the system-wide install path contradicts both.
- **Failure scenario:** TLS-intercepting proxy / compromised CDN edge / typo'd `GITHUB_RELEASE_BASE` delivers a different MSI → GAIA installs it as admin, and the shipped GAIA installer bundles it.
- **Evidence:** `with open(dest_path, "wb") as f: … f.write(chunk)` (448-455, no digest); `cmd = ["msiexec", "/i", str(installer_path)]` (613).
- **Fix:** Add `LEMONADE_SHA256 = {asset: hex}` next to `LEMONADE_VERSION` (mirroring `EMBEDDABLE_SHA256`), verify in `download_installer` and the two workflow steps; minimum: `Get-AuthenticodeSignature` / `pkgutil --check-signature` before executing.
- **Confidence:** High
- **Tracked:** none found (#936 is smoke tests, not integrity)

### 🟡 Electron uninstaller wipes all of `~/.gaia`, contradicting the CLI's "keep `~/.gaia` itself" contract and its own dialog *(sub-review, re-verified)*
- **Where:** `installer/nsis/installer.nsh:142-145` (`customUnInstall`); same construct in `installer/tui/nsis/gaia-setup.nsi:464-467`.
- **What:** `RmDir /r "$PROFILE\.gaia"` removes `agents/` (custom agents), `skills/`, `memory/`, `mcp_servers.json`, connector tokens, `bin/` (Terminal Hub binaries), `lemonade/` (embedded runtime). `uninstall_command.py:151-152` promises "We keep `~/.gaia/` itself so other tooling that lives alongside us (e.g. MCP config) is preserved"; the dialog names only "chats, documents, Python environment". Two products share the directory, so uninstalling one and answering Yes destroys the other's data.
- **Evidence:** `MessageBox … "Also remove your GAIA data (chats, documents, Python environment)?…" /SD IDNO IDNO +2` / `RmDir /r "$PROFILE\.gaia"`.
- **Fix:** Have NSIS invoke `gaia uninstall --purge --yes` (the tiered path) or enumerate the same subpaths; at minimum make the dialog truthful.
- **Confidence:** High
- **Tracked:** none found (#3290 is the *opposite* gap — `--purge` leaving `bin/` behind)

### 🟡 A symlinked entry under `~/.gaia` crashes `--purge` mid-run with a traceback *(sub-review, re-verified)*
- **Where:** `src/gaia/installer/uninstall_command.py:449-471` (`_remove_path`), `execute_plan` (~`:650-660`), `src/gaia/cli.py:4424`.
- **What:** The guard resolves the *target* of a symlink before deciding. `~/.gaia/documents -> /mnt/data/docs` resolves outside the roots → `RuntimeError`, which nothing catches. `venv` and `chat` have already been deleted; exit code is Python's 1 (not the documented 2/64); the link itself is never unlinked even though unlinking without following is the safe action the code already implements a few lines later.
- **Evidence:** `resolved = path.resolve(strict=False)` (449) … `raise RuntimeError(f"Refusing to delete {resolved}: outside allowed roots …")` (470-473).
- **Fix:** Check `path.is_symlink()` first; if the link itself lives inside a root, `unlink()` it and skip the resolved-target check (resolve only `path.parent`). Catch `RuntimeError` in `execute_plan` → `EXIT_FS_ERROR`.
- **Confidence:** High (code path; symlink repro blocked by Windows privilege)
- **Tracked:** none found

### 🟡 `apt purge` silently deletes the invoking user's chats and documents *(sub-review, re-verified)*
- **Where:** `installer/debian/postrm:34-37`.
- **What:** On `purge`, the maintainer script runs `gaia uninstall --purge --yes` as `$SUDO_USER` with output discarded. Debian Policy forbids maintainer scripts touching files in user home directories; `sudo apt purge gaia-agent-ui` removes the admin's personal `~/.gaia/chat` and `~/.gaia/documents` with no prompt, and `2>/dev/null || true` hides both the deletion and any failure.
- **Evidence:** `sudo -u "$SUDO_USER" -H sh -c 'command -v gaia >/dev/null 2>&1 && gaia uninstall --purge --yes' 2>/dev/null || true`.
- **Fix:** Drop the home-dir purge from `postrm` (leave it to the documented `gaia uninstall --purge`), or at least print what will be removed / ask via `debconf`.
- **Confidence:** Medium (the design is plan-sanctioned in `docs/plans/desktop-installer.mdx §5`; the risk is the silent default)
- **Tracked:** none found

### 🟡 PPA install reports success even when the post-install probe finds nothing *(sub-review, re-verified)*
- **Where:** `src/gaia/installer/lemonade_installer.py:983-998` (`_install_via_ppa`).
- **What:** After `apt-get install`, `check_installation()` is only used for the version string; `success=True` is returned even when `verify.installed` is False. The macOS branch (`:757-768`) does the opposite (trusts the probe).
- **Failure scenario:** apt exits 0 but the package doesn't put `lemonade-server` on PATH (shadowed repo, arch mismatch) → `gaia init` prints "Installed Lemonade vunknown via PPA" and later fails with an unrelated "server not reachable".
- **Evidence:** `verify = self.check_installation(); installed_version = verify.version or "unknown"; … return InstallResult(success=True, version=installed_version, …)`.
- **Fix:** `if not verify.installed: return InstallResult(success=False, error=…)`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Windows uninstall fallback blames the Installer registry for network failures *(sub-review, re-verified)*
- **Where:** `src/gaia/installer/lemonade_installer.py:1095-1170` (`_uninstall_windows`).
- **What:** When the registry lookup fails, both MSI variants are *downloaded* at uninstall time; every exception in that loop is `log.debug`'d and swallowed, and the final error says "product not found in Windows Installer registry". `gaia uninstall --purge-lemonade` uses this path, so an offline machine gets a wrong diagnosis.
- **Evidence:** `except Exception as e: log.debug(f"Failed to uninstall with {variant} MSI: {e}"); continue` (1160-1162) → `error="Could not uninstall: product not found in Windows Installer registry. …"` (1165-1168).
- **Fix:** Collect per-variant errors into `InstallResult.error`; skip the download loop when the failure is a non-definitive network error.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Documented CLI flags that are accepted and silently ignored *(sub-review, re-verified)*
- **Where:** `src/gaia/cli.py:759-774` (`talk` branch), `:1241-1246` (`--stats` registered with `dest="show_stats"`), `:354-357` (`GaiaCliClient.__init__` → `create_client("lemonade", …)`), `:543-560` (`prompt` branch), `:3906-3920` (`eval agent --device`), `:2410-2415` (`mcp start --ctx-size`, never read in `handle_mcp_start` 7288-7475).
- **What:** `gaia talk --model/--max-tokens/--use-claude` are never passed into `TalkConfig`, and `--stats` is read under the wrong key (`kwargs.get("stats")` vs `dest="show_stats"` — confirmed via `parse_args(["talk","--stats"])`). `gaia prompt --device/--use-claude/--claude-model` never reach `GaiaCliClient`, which hard-codes the Lemonade provider; worse, `--use-claude` sets `skip_if_external=True` so the Lemonade preflight is skipped and the command then talks to Lemonade anyway. `gaia eval agent --device` is dead: `eval_model = args.model` always holds the judge default (`claude-opus-5`), so `if eval_device and not eval_model:` never fires (and would overwrite the judge model if it did). `gaia mcp start --ctx-size` is registered, documented (`cli.mdx:1239`), and unused (the only `args.ctx_size` consumer is `eval benchmark`, `:4094`).
- **Failure scenario:** `gaia talk --model Qwen3-… --max-tokens 2000 --stats` runs the default model at 512 tokens with no stats and no warning; `gaia prompt "hi" --use-claude` with Lemonade down fails with a raw connection error instead of the "start Lemonade" hint; `gaia eval agent --device npu` runs untagged on whatever is configured. `docs/reference/cli.mdx:869, 964-971, 1239, 1702` document all of these as working.
- **Evidence:** `show_stats=kwargs.get("stats", False),` (768); `self.llm_client = create_client("lemonade", model=model)` (356); `eval_model = args.model … if eval_device and not eval_model:` (3908-3910).
- **Fix:** Wire `model/max_tokens/use_claude/show_stats` into `TalkConfig`; route `prompt` through the same device/provider resolution `chat` uses (or drop the flags and reject them explicitly); pass `device=` into `AgentEvalRunner`; either forward `--ctx-size` to `initialize_lemonade_for_agent` or delete the flag + doc row.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `gaia init --profile mcp --check` performs the install instead of a side-effect-free check *(sub-review, re-verified)*
- **Where:** `src/gaia/cli.py:4310-4318` precedes `:4319-4335` (`main`, init branch); `src/gaia/installer/mcp_init.py:59-81` writes `~/.gaia/mcp_servers.json`.
- **What:** The MCP-profile short-circuit runs before the `--check` branch, so `--check` is ignored for that profile — contradicting the flag's help ("no install, no download, no side effects") and `cli.mdx:200`.
- **Evidence:** `if profile == "mcp": … exit_code = run_mcp_init(…); sys.exit(exit_code)` (4310-4317) then `if args.check:` (4319).
- **Fix:** Move the `args.check` block above the `profile == "mcp"` block and give `check_setup_status` an mcp probe (`mcp_servers.json` exists).
- **Confidence:** High
- **Tracked:** none found

### 🟡 Failure paths that exit 0 — `gaia chat -q`, `llm`, `download`, `mcp start/stop`, `api start`, bare `cache`/`memory`/`knowledge`/`daemon`/`mcp` *(sub-review, re-verified)*
- **Where:** `src/gaia/cli.py:729` (returns `0/1` from `chat -q`), `:3343-3346` (`main` prints any truthy return and exits 0), `:740-743` (chat `except Exception: … return`); `llm` `:3787-3788`; `download` `:3560-3561, 3666-3667`; `api start` `:4849-4850`; `mcp start` `:7302, 7315, 7335, 7473-7475`; `mcp stop` `:7492-7493, 7525`; bare-subcommand refusals returning `None`: `cache` `:5075-5078`, `memory` `:5160-5163`, `knowledge` `:4966-4969`, `daemon` `:6516-6521`, `mcp` `:7138-7141`.
- **What:** Each prints ❌ then `return`s from `main` → exit 0. `chat -q` returns an int that `main` `print`s as output — stdout ends with a stray `1` and the exit code is 0. Meanwhile `hub`, `lemonade`, `config`, `agent`, `kill`, `cache clear` refuse with exit 1, and `tests/unit/test_cli_refusal_exit_codes.py` codifies the exit-1 rule for only `kill`/`cache clear`.
- **Failure scenario:** `gaia chat -q "…" && deploy.sh` runs `deploy.sh` after the agent failed; `gaia mcp start --background && gaia mcp test` runs the test against nothing when deps/port/Lemonade fail; `gaia download` with Lemonade down exits 0 having downloaded nothing.
- **Evidence:** `return 0 if result["status"] == "success" else 1` (729); `result = run_cli(args.action, **kwargs); if result: print(result)` (3344-3346); `if not success: return` (3787-3788).
- **Fix:** `sys.exit(result)` in `main` when `isinstance(result, int)`; replace each ❌-then-`return` with `sys.exit(1)`; extend the refusal test parametrization to `mcp`, `cache`, `memory`, `daemon`, `knowledge`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `gaia mcp stop` deletes the PID file even when the kill failed, and the PID/log files depend on CWD *(sub-review, re-verified)*
- **Where:** `src/gaia/cli.py:7424` (`os.path.abspath("gaia.mcp.pid")`), `:7485` (read), `:7523-7532`.
- **What:** PID and log live in whatever directory the user was in; `stop` from elsewhere says "No MCP bridge PID file found" while the bridge keeps running. On `CalledProcessError` the code prints ❌ and still falls through to `os.remove(pid_file_path)`, so the next `stop` has nothing to act on.
- **Evidence:** `except subprocess.CalledProcessError: print(f"❌ Failed to stop process {pid}")` … `os.remove(pid_file_path)` (7523-7531).
- **Fix:** Store the PID under `~/.gaia/mcp.pid` (as `telegram` does at `:3269`); remove it only after a confirmed stop.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `gaia agent test --live` tracebacks on a bad manifest *(sub-review, re-verified)*
- **Where:** `src/gaia/cli_agent.py:755` (`_run_live_gates` → bare `hub_manifest.parse(pkg_dir)`), dispatcher `:300-305` catches only `AgentWorkflowError`.
- **What:** `--lint` wraps `parse` in `try/except ManifestError` (`:558-566`); `--live` does not, and `ManifestError` subclasses `ValueError`, so the user gets a raw traceback instead of the "Error: …" line every other gate prints.
- **Fix:** `except hub_manifest.ManifestError as exc: raise AgentWorkflowError(str(exc)) from exc`.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `launch_server()` kills whatever owns the port when the health probe fails for a non-"down" reason
- **Where:** `src/gaia/llm/lemonade_client.py:1065-1080` (`launch_server`), `:688-702` (`kill_process_on_port`).
- **What:** Any exception from `health_check()` — including `LemonadeAuthError` from an authenticated server with a wrong/missing key, or a 5xx while a model loads — is treated as "no healthy server", and `kill_process_on_port(self.port)` then force-kills the listener (Lemonade or an unrelated service on 13305) before launching a new one. Only reachable today through the dead `initialize()`/`create_lemonade_client(auto_start=True)` paths, hence 🟢.
- **Evidence:** `except Exception as e: … health = None` / `if isinstance(health, dict) and health.get("status") == "ok": … return` / `kill_process_on_port(self.port)`.
- **Fix:** Re-raise `LemonadeAuthError`; only kill when the socket is open **and** the owner is a Lemonade binary (as `lemonade_embedded._daemon_alive` already does).
- **Confidence:** High
- **Tracked:** none found

### 🟢 `gaia.util.kill_process_on_port` is dead code whose Windows predicate would kill Lemonade's *clients*
- **Where:** `src/gaia/util.py:9-77`.
- **What:** No in-tree caller (grep `from gaia.util import` / `gaia.util.kill_process_on_port`: none). Its Windows branch matches `netstat` lines by substring `":13305"` with `ESTABLISHED`, which includes connections whose **foreign** address is `:13305` — i.e. the GAIA UI server, daemon, browser, or the current process — and taskkills them.
- **Evidence:** `if f":{port}" in line and ("LISTENING" in line or "ESTABLISHED" in line): … pids_to_kill.add(pid)`.
- **Fix:** Delete the module (the psutil-based `lemonade_client.kill_process_on_port` checks `laddr` only), or restrict to `LISTENING`.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `import gaia` crashes when the home directory cannot be resolved
- **Where:** `src/gaia/logger.py:50` (`GaiaLogger.__init__`), instantiated at import (`:281`).
- **What:** `Path.home()` raises `RuntimeError("Could not determine home directory.")` when `USERPROFILE`/`HOMEDRIVE+HOMEPATH` (Windows) or `HOME` (POSIX) are unset; the call sits outside the `try` that handles the unwritable-home case, so the whole package import fails.
- **Failure scenario:** Observed: `tests/unit/test_cli_refusal_exit_codes.py` spawns `python -m gaia.cli` with only `HOME` set → 4 failures on Windows. Real-world: a Windows Scheduled Task / service with a minimal environment running `gaia daemon` dies at import with a stack trace.
- **Evidence:** `log_file = Path.home() / ".gaia" / "gaia.log"` … `RuntimeError: Could not determine home directory.` (pytest output).
- **Fix:** Wrap `Path.home()` in `try/except RuntimeError` → fall through to the tempdir fallback that already exists at lines 118-140.
- **Confidence:** High
- **Tracked:** none found

### 🟢 Shell installers are not pinned to LF; a Windows checkout ships CRLF scripts that `sh`/`dash` reject
- **Where:** `.gitattributes` (no `*.sh` rule), `installer/scripts/install.sh`, `scripts/*.sh`, `installer/linux`, `installer/macos`.
- **What:** `git ls-files --eol installer/scripts/install.sh` → `i/lf w/crlf` on this checkout. `dash -n` fails at line 84 (`case "$OS_NAME" in` with a trailing CR), which is exactly what `test_install_scripts_terminal_hub.py::test_sh_parses_under_dash` caught.
- **Failure scenario:** A release/installer bundle or docs walkthrough built from a Windows clone ships a `curl | sh` installer that dies with "Syntax error: word unexpected".
- **Fix:** Add `*.sh text eol=lf` and `installer/scripts/** text eol=lf` to `.gitattributes`.
- **Confidence:** High
- **Tracked:** none found

### 🟢 Every fresh model download in `gaia init` waits an extra 30 s after the pull already finished
- **Where:** `src/gaia/llm/lemonade_client.py:2833-2846` (`ensure_model_downloaded`), `:2990-3003` (`_wait_for_model_download`).
- **What:** `pull_model` is the **synchronous** `/pull` (blocks until the download completes, timeout 7200 s). It is then followed by `_wait_for_model_download`, whose loop sleeps `poll_interval = 30` **before** its first `/models` check.
- **Evidence:** `self.pull_model(model_name, …, timeout=timeout)` then `return self._wait_for_model_download(...)`; in the waiter: `time.sleep(poll_interval); elapsed += poll_interval; try: models_response = self.list_models()`.
- **Fix:** Check first, sleep after; or skip the waiter when the synchronous pull returned success.
- **Confidence:** High
- **Tracked:** none found

### 🟢 Disk-space guard checks the current working directory, not the model cache
- **Where:** `src/gaia/llm/lemonade_client.py:913-952` (`_check_disk_space`), called at `:3638`.
- **What:** Free space is measured on `os.getcwd()`; models land in `%LOCALAPPDATA%\lemonade` / `~/.cache/huggingface` (paths `_prompt_user_for_delete` already knows at 843-849). On Windows with a small C: and a project on D:, the guard passes and the download fails at 100 %. The docstring admits the gap.
- **Fix:** Measure the Lemonade cache path (`LEMONADE_CACHE_DIR` if set, else the platform default already listed in `_prompt_user_for_delete`).
- **Confidence:** High
- **Tracked:** none found

### 🟢 Agent UI device gate still sized for Qwen3.5-35B
- **Where:** `src/gaia/device.py:15-18` (`_MIN_GPU_VRAM_GB = 24.0`, "minimum to load Qwen3.5-35B-A3B-GGUF"), consumed by `src/gaia/ui/routers/system.py:741`.
- **What:** The default model everywhere is `Gemma-4-E4B-it-GGUF` (~3 GB); a discrete Radeon with 12–16 GB runs it fine but is reported unsupported by the Agent UI's device check.
- **Fix:** Re-derive the floor from `MODELS[DEFAULT_MODEL_NAME]` (or demote the VRAM gate to a warning).
- **Confidence:** Medium (UI reviewer owns the gate's UX; the stale constant is in scope here)
- **Tracked:** none found

### 🟢 Smaller client-contract nits (one pattern, listed once)
- `LemonadeStatus(url=f"http://{self.host}:{self.port}")` at `lemonade_client.py:3942` and `:4363` drops the scheme/path of a `base_url` client — status output shows the wrong URL for https/proxied servers.
- `_stream_completions_with_openai` (`:2198-2207`) forwards raw `**kwargs` into `client.completions.create`; llama.cpp-only kwargs (`repeat_penalty`, `repeat_last_n` — the ones `LemonadeProvider.chat` sets by default) raise `TypeError` in the OpenAI SDK, unlike the chat path which routes them via `extra_body`. It also holds no model-slot lease, and the non-streaming `completions()` never calls `_ensure_model_loaded`, so the 64K ctx guard is skipped for `/completions`.
- `_check_version_compatibility` (`:4227-4296`) is only reachable via the dead `initialize()`; the `LEMONADE_MIN_VERSION` floor in `version.py` is therefore enforced only by `gaia init` (`init_command.py:1000-1131`), never at runtime against a server that was downgraded after init.
- `lemonade_manager._try_reload_with_ctx` (`:935-955`) sets `cls._context_size = min_context_size` even when the server still reports a smaller window ("Assuming reload succeeded to prevent reload loop") — a deliberate but silent degrade; only a WARNING log distinguishes it.

### 🟢 Installer-side nits *(sub-review; the URL and `iex` lines re-verified)*
- `scripts/install-ui.ps1:78` and `scripts/install-ui.sh:75` print `https://amd-gaia.ai/guides/…` (404; the `/docs/` prefix is required — the same rule `tests/unit/test_amd_gaia_urls.py` enforces for `src/gaia/`, but `scripts/` is outside its scan). The `.ps1` also names `chat-ui`, the `.sh` `agent-ui`.
- `installer/scripts/install.ps1:71` runs `irm https://astral.sh/uv/install.ps1 | iex` — unpinned, unverified third-party bootstrap, while the same script SHA-256-verifies GAIA's own hub artifacts; `install.sh:172-180` at least stages to a file. Pin `UV_VERSION` (as `build-installers.yml:228` already does).
- `lemonade_installer.refresh_path_from_registry` (`:165-171`, duplicated in `init_command.py:468-516`) replaces/prepends the process `PATH` from the registry, dropping session-only entries (activated venv `Scripts`, uv's `~/.local/bin`) and reversing Windows' system-before-user order for later `shutil.which` calls.
- `tools/lemonade_stub.py` is unreferenced dead code that binds the real default port 13305, answers only three POST routes with a fixed body, and has no `/health`, `/models`, `/pull`, `/load` — anything pointed at it "passes" while the real server would 4xx. Delete or move under `tests/fixtures/` with a contract-faithful shape.
- `lemonade_installer.py:863-897` Linux gate: docstring says "Ubuntu 24.04+ only" but Debian ≥ 13 and non-apt distros pass through to a misleading `add-apt-repository` error, while `get_download_url` builds a `debian13` `.deb` URL no install path uses.

### 🟢 CLI nits *(sub-review)*
- Top-level `--base-url` / `--ui-port` / `--ui-dist` (`cli.py:1150-1178`) are silently discarded when a subcommand follows — argparse copies the subparser defaults over the parent's (confirmed: `gaia --base-url http://remote chat` → `args.base_url is None`; `gaia --ui-port 9999 chat --ui` → `4200`; same for `gaia mcp --base-url X start`). Use `argparse.SUPPRESS` on the subparser copies.
- `gaia prompt` prints the streamed answer and then the raw return dict again (`:372` streams chunks, `:560` returns `{"response": …}`, `main` prints it at `:3345-3346`) → `<answer>{'response': '<answer>'}`.
- `gaia youtube` with no flag falls through every later `if args.action == …` to "Unknown action specified: youtube" + full help (`:3465-3489`, `:4426-4429`).
- `gaia agent` help lists `init, version, test, configure, health, status, export, import, install, list` (`:6268-6272`) but `cli_agent.register_subparsers` also registers `pack`, `publish`, `login`.
- `gaia install --lemonade` calls `input()` without an EOF guard (`:4384`) → traceback when stdin is closed; every other prompt in the file catches `EOFError`.
- `--hub-token` / `--pypi-token` (`cli_agent.py:261-270`) and the required `telegram start --token` (`cli.py:1613`) accept secrets on argv (shell history, `ps`); `mcp start` deliberately moved its token to the environment (`:7362-7366`). Nothing echoes the values.

## Test gaps

- **Mock suite for the primary HTTP client is red on `main` and excluded from CI** — see the 🟡 finding above. The Integration class only runs on the two hardware CLI workflows, so a PR that only touches `lemonade_client.py` gets no client test at all in the fast unit lane.
- **Posix-only tests are not skipped on Windows** (30 of the 49 failures/errors on this box): `tests/unit/test_init_command.py::TestInstallViaPpa` and `test_lemonade_macos_install.py` patch `os.geteuid` (absent on Windows → `AttributeError` at setup); `tests/unit/installer/test_uninstall_command.py::TestLemonadePythonResolution` uses pyfakefs which calls `os.getuid` at teardown; `test_lemonade_launcher.py` darwin cases compare `"/usr/local/bin/lemond"` to a `str(Path)` that Windows renders as `\usr\local\bin\lemond`; `test_uninstall_command.py::{TestBuildPlan,TestDryRun,TestPurgeModels,TestGaiaHomeEnvVar}` expect `~/.cache/lemonade` but the code (correctly) uses `%LOCALAPPDATA%` on Windows. Add `pytest.mark.skipif(sys.platform == "win32")` (or `platform.system()` patches that also patch the path builders) so the primary target platform can produce a green run.
- **`tests/unit/test_cli_refusal_exit_codes.py`** builds a subprocess env with `HOME` only; on Windows Python needs `USERPROFILE` — the harness never exercises the refusal paths there. (It also happens to expose the `logger.py` import crash above.)
- **`tests/unit/test_lemonade_error_classification.py`** lives under `tests/unit/` but imports `gaia.ui._chat_helpers` → `fastapi`; it cannot even be collected in a core-only venv. Move it under a `ui`-gated path or guard with `pytest.importorskip("fastapi")`.
- **`tests/unit/cli/test_cli_smoke.py::test_gaia_binary_on_path`** asserts the console-script shim is on `PATH` — an environment property, not code under test; fails in any venv that is not activated.
- **No test for the idle-server preload path choosing the right model per device** (`LemonadeManager._try_preload_with_ctx` — `tests/unit/test_lemonade_manager_preload.py` asserts *that* `load_model(DEFAULT_MODEL_NAME, …)` is called, which is the bug, not the contract).
- **No test that `INIT_PROFILES[*].min_context_size` agrees with `MODELS[*].min_ctx_size` / `profile_ctx_size`** — `test_init_ctx_size.py` pins the literal 32768, so the disagreement is enshrined rather than caught.
- **Cold-start pull contract is tested only at the request-shape level** (`tests/test_lemonade_client.py:977-1070` asserts `user.` + checkpoint + recipe + embedding in the `/pull` body — good), but there is no integration test that registers `user.embeddinggemma-300m-GGUF` on a server that has never seen it, which is the #1655 failure class. `tests/test_lemonade_embeddings.py` runs against a warm server in `test_embeddings.yml`.
- **`VLMClient`** has no unit test covering the `base_url` → host/port rewrite or the error-string return; `tests/unit/test_vlm*` (if any) mock `LemonadeClient` at the source.
- **`tests/installer/test_custom_agent_mcp_harness.py::test_custom_agent_dummy_mcp_path_uses_installed_bundle`** fails on Windows with `Unknown tool name` from `Agent._execute_tool` — the MCP mixin connected (the sibling "diagnosable connection failure" test passes) but `mcp_dummy_add_two_numbers` was not registered. Not root-caused here (MCP is another reviewer's dimension); flagged in Hypotheses.
- *(sub-review)* Uninstall containment is only tested for paths outside `~/.gaia` (`test_uninstall_command.py:536-560`); nothing covers `GAIA_HOME` resolving to `$HOME`/`/`, case-insensitive `documents` vs `Documents`, or symlinked entries — the 🔴 and the symlink 🟡 above.
- *(sub-review)* `_install_windows`, `_uninstall_windows`, `wait_for_msi_mutex` have no behavioural tests (only the missing-path guard at `test_init_command.py:1629`); the msiexec argv, the 1602/1603/1618 mapping and the MSI-download fallback are unexercised. `_install_via_ppa`'s "success without probe" path is untested and `TestInstallViaPpa` is POSIX-only.
- *(sub-review)* Download integrity has no test because there is no integrity check; `test_lemonade_download_urls.py:140-172` mocks `urlopen` and asserts the bytes landed — "we wrote what we were sent". `tests/integration/test_lemonade_release_assets.py` is the only upstream-URL check and is `@pytest.mark.network`-skipped offline, so an asset rename can pass CI silently.
- *(sub-review)* NSIS and Debian maintainer scripts (`RmDir /r "$PROFILE\.gaia"`, `postrm` purge) have no smoke test (#936). `test_install_scripts_terminal_hub.py` executes `install.sh` against a fake hub but only string-asserts `install.ps1` (#3295).
- *(sub-review)* `export_import.py` is well covered (zip-slip, symlink, absolute path, size/count, reserved names, overwrite); missing: an archive containing a `.backup-<id>` directory colliding with the rollback dir (`:403-404` → uncaught `OSError`), and backslash-separated entries on POSIX.

## Documentation gaps

- `.env.example:14` — `LEMONADE_BASE_URL=http://localhost:8000/api/v1` contradicts `DEFAULT_PORT = 13305` (🟡 above).
- `.claude/skills/lemonade-client-patterns/SKILL.md` — "Factory must mirror `__init__` … callers that use the factory (like CLI entry points)": no in-tree caller uses `create_lemonade_client`; the CLI goes through `LemonadeManager.ensure_ready` + `LemonadeClient(...)` directly.
- `lemonade_client.py:329` comment — "Low-memory users can dial down via the `GAIA_CTX_SIZE` env var": true only for the CLI path (`cli.py:164`); the Agent UI server and `LemonadeManager` never read `GAIA_CTX_SIZE`, and it is not in `.env.example`.
- `lemonade_client.py:4298-4336` (`initialize` docstring) and `AGENT_PROFILES` — still describe "code, blender, jira, docker" agents and a 32768 default; the profiles were deleted and the GPU default is 65536.
- `LemonadeClient.check_model_loaded` docstring says "Check if a specific model is loaded"; the implementation checks the downloaded catalog (🟡 above).
- `lemonade_manager._RECIPE_BY_DEVICE` (`:26-37`) carries a `TODO: Confirm full recipe vocabulary` and maps every AMD device to `oga-hybrid`, while the real recipes in use are `llamacpp` and `flm`; the value is only logged, but the table misleads readers.
- CI `test_gaia_cli_linux.yml:398` sets `LEMONADE_PORT=13305`, an env var no runtime path reads (only the dead factory) — stale knob.
- **`docs/reference/cli.mdx` vs the argparse tree** *(sub-review; 135 parsers enumerated programmatically)*:
  - Documented but not registered (argparse rejects): `gaia mcp test-client --config PATH` (`cli.mdx:1173,1180`; `mcp_test_client_parser` has only the positional `name`, `cli.py:2503-2505`).
  - Documented flags that are inert (🟡 above): `talk --model/--max-tokens/--stats`, `prompt --device`, `eval agent --device`, `mcp start --ctx-size`. `mcp start --log-file` is documented as default `stdout` (`cli.mdx:1237`) but defaults to `gaia.mcp.log` and only applies with `--background`.
  - Registered but never mentioned in the docs: `gaia eval code`, `gaia eval sessions`, `gaia mcp serve`, `gaia schedule show|remove`, `gaia connectors {list,status,test,disconnect}`, `gaia connectors activations {list,activate,deactivate}`, `gaia connectors grants {list,revoke}`.
  - Registered flags undocumented: `memory bootstrap --infer/--system/--reset-system`; `install --silent`; `eval code --workspace`; `eval sessions --dataset-only/--project`; `report --eval-dir/--output-file/--summary-only`; `connectors configure --client-id/--client-secret`; `connectors connect --grant-agent`; the shared parent flags `--use-chatgpt`, `--claude-model`, `--trace`, `--no-lemonade-check` are attached to ~20 subcommands (including `kill`, `download`, `install`, `uninstall`, `perf-vis`, `youtube`, `test` where they are meaningless) and documented nowhere.
  - `gaia tui` / `gaia status` are documented as the Go TUI binary (`cli.mdx:2683-2687`), not `cli.py` — fine.

## Improvement opportunities

- Single source of truth for the context window: one `ctx_for(device)` used by `INIT_PROFILES`, `MODELS`, `AGENT_PROFILES`, `LemonadeManager`, and the overflow classifier — three of those currently disagree (32768 vs 65536 literal) and two bugs above stem from it.
- Make the idle-server preload device-aware (or drop it in favour of the per-request `_ensure_model_loaded`, which already knows the model) — removes the NPU/GGUF thrash trigger.
- Delete dead API: `create_lemonade_client`, `initialize_lemonade`, `LemonadeClient.initialize`, `_check_version_compatibility`, `gaia.util.kill_process_on_port`, `InitCommand._find_lemonade_server` (self-described "compatibility surface only — no in-tree callers"). Less surface to keep contract-correct.
- Run `LEMONADE_MIN_VERSION` enforcement at runtime (`LemonadeManager.ensure_ready` reads `/health.version` already) so a server downgraded after `gaia init` fails loudly instead of via an obscure 400.
- `_ensure_model_loaded_locked` prints via `rich.Console()` to stdout unconditionally (`:3281-3303`); inside `gaia mcp serve --stdio` this corrupts the JSON-RPC stream unless every caller remembered `route_console_logging_to_stderr` (which only reroutes *logging* handlers, not `print`/`Console`). Route through the logger.
- `_wait_for_model_download` / `ensure_model_downloaded`: poll-then-sleep, and reuse the SSE `pull_model_stream` so `gaia init` shows progress instead of a silent 30 s gap per model.
- `providers/lemonade.py` typed errors duplicate `is_context_overflow_error`'s phrase table (`lemonade_client.py:566-571`, which already includes FastFlowLM's "max length reached") — `_classify_lemonade_response` only matches the llama.cpp phrasing, so the NPU backend's overflow falls through to the generic `LemonadeError` in the provider path. Reuse the shared classifier.
- `.gitattributes`: pin `*.sh`, `*.ps1` (CRLF is fine), and `installer/**` scripts to LF.

## High-impact feature opportunities

- **Device-aware setup verification.** `gaia init --check` (`check_setup_status`) and `_verify_setup` know the profile but not whether the *runtime* will pick the same model/ctx (they don't for NPU, and they disagree with the 64K GPU window). A single "what will `gaia chat` actually load?" preflight — model id, ctx, backend, resident-vs-download — shown at the end of init and by `gaia cache status` would remove the whole class of "init said OK, first chat downloads/reloads something else" reports (#1676, #3152 are both this shape). Roughly: one resolver function + wiring into init's completion panel and the UI system router.
- **Embedded Lemonade as the default path.** `lemonade_embedded.py` is a well-built, checksum-verified, private-port, API-key-protected server (the best-engineered module in this area), but `gaia init` still drives the system-wide MSI/PPA/pkg installer with all its platform branches, PATH refreshes and MSI-mutex waits. Making the embedded artifact the default for `gaia init` (system install as opt-in) would delete most of `lemonade_installer.py`'s risk surface and give every user the same reproducible server. Needs: init profile → `EmbeddedLemonade.start()` + `install_backend()` per device, `LemonadeManager` reading `state.json` for base URL/key.
- **Structured VLM/OCR failures surfaced in the RAG index** (per-page status instead of error strings), so the Agent UI can show "3 pages failed OCR — retry" rather than answering from error text.
- **Runtime version gate + upgrade hint.** Read `/health.version` once per process, compare to `LEMONADE_MIN_VERSION` and the profile floor (EmbeddingGemma needs ≥10.9.0), and print the platform-correct upgrade instruction. Today only `gaia init` checks; users who upgrade GAIA but not Lemonade hit "cannot load embedder" at first index.

## Checked and fine

- `src/gaia/version.py`: `__version__ = "0.23.1"`, `LEMONADE_VERSION = "11.8.1"`, `LEMONADE_MIN_VERSION = "10.2.0"`; `get_package_version()` reads installed metadata (0.23.1 in the venv). Consistent with the release commit.
- `/pull` contract (#1655): built-ins are pulled by name only; `user.` models carry checkpoint + recipe + `embedding=True` — enforced identically in `init_command._download_models`, `rag/sdk.py:479-503`, `code_index/sdk.py:750-767`, and asserted at the request-body level in `tests/test_lemonade_client.py:977-1036`. `_model_ids_match` handles the stripped-id listing.
- 401 handling: fixed-string `LemonadeAuthError` everywhere (`_send_request`, both streaming paths, `/pull`, `/responses`, `/chat`, `/completions`); `openai.AuthenticationError` is caught before the generic `APIError` branch; `get_status()` re-raises auth errors instead of reporting "not running". API key is never logged (presence only at DEBUG).
- Auth header omitted when no key; `api_key or "lemonade"` placeholder for the OpenAI SDK; env value `.strip() or None`.
- `_execute_with_auto_download` no longer retries non-missing-model errors (#2513); `_is_corrupt_download_error` excludes "llama-server failed to start"; transient-load retry is bounded (3 × escalating backoff) and only for the named phrase; corrupt repair is bounded to resume + one delete/re-download and honours `prompt=False` (no `input()` in server contexts, #1293).
- Exact-pin path (`ctx_size_override`, #1892) settles unload/load against `/health` with deadlines and fails loudly on a clamped ctx; probe failures are treated as UNKNOWN, not "absent".
- Model-slot lease is held across load **and** inference for both chat paths (#2380); re-entrant per thread; no-op without a broker.
- Streaming tool-call fragments are accumulated by index and emitted as the same sentinel envelope as the non-streaming path; `reasoning_content` is line-buffered into `<think>` tags.
- `ClaudeProvider`: `ANTHROPIC_API_KEY` missing → actionable error; OAuth tokens routed via `auth_token` + beta header; non-Anthropic kwargs dropped with a debug log; `max_tokens` floored at 8192; prompt-cache breakpoints on system block and last tool; usage accounting sums cached + uncached; refusals raise. `OpenAIProvider` lets the SDK raise on a missing key. `LiteLLMProvider` raises `ImportError` with the install hint.
- `lemonade_embedded.py`: SHA-256 pins for all four assets, per-member archive extraction with containment + link checks, `O_EXCL|0600` state/env files, pid identity checked by image name before kill, API key doubles as identity for health probes, refuses double-start / cross-version start.
- `lemonade_launcher.resolve_lemonade` precedence (env override → canonical modern path → legacy PATH) and `build_start_command` (`LEMONADE_CTX_SIZE` env for modern; argv `--ctx-size` for legacy; never `shell=True`; parent env merged not replaced). `describe_start_hint` never invents a command on tray/app platforms.
- `GaiaConfig.load` fails loudly on a corrupt file; missing file → defaults; `resolve_model` precedence flag > config > builtin.
- `security.PathValidator`: symlink-resolved allowlist with `os.sep`-suffixed prefix check, macOS `/private` normalisation, blocklist includes `~/.gaia` (except user-content subdirs), sensitive names/extensions, 10 MB cap, non-interactive auto-deny for new paths, fail-closed on validation errors, rotating audit log.
- `gaia init` refuses to run non-interactively without `--yes`; `--yes` documented as authorising an unattended Lemonade uninstall/upgrade; profile below `min_lemonade_version` forces upgrade; newer-than-pinned is accepted with a note.
- `is_context_overflow_error` parses the nested Lemonade envelope first and includes FastFlowLM's "max length reached" (#2513).
- *(sub-review)* `export_import.py`: traversal defended twice (validate + re-check in staging, `:254-258`, `:339-343`), symlink entries rejected, real-bytes zip-bomb accounting, atomic `os.replace` with rollback, agent IDs validated before any FS write; the UI upload endpoint caps at 100 MB and deletes the temp file. `mcp_init.py` never overwrites an existing `mcp_servers.json`. `_stdin.py` is the single TTY predicate; `--purge` on a non-TTY requires explicit `--yes`; dry-run returns before any FS mutation.
- *(sub-review)* `--purge-models` / `--purge-hf-cache` do delete models pulled outside GAIA, but both are opt-in, listed in the plan, and documented in `--help`. `_remove_path` unlinks (does not follow) symlinks that resolve inside a root; `rmtree` does not follow junctions.
- *(sub-review)* macOS install: Intel refused up front (`lemonade_installer.py:303-310`), sudo announced before prompting, `KeyboardInterrupt` at the sudo prompt handled, post-install probe trusted over exit code. `LemonadeAssetError.definitive` split: 404/410 fails before download; HEAD-rejecting proxies fall through to GET (logged tolerance, not a silent fallback).
- *(sub-review, CLI)* Secrets: `mcp start --background` passes the auth token via environment, never argv or the banner (`cli.py:7362-7366, 7385-7394`); `daemon start-agent` prints only whitelisted fields (`:6707-6753`); `diagnostics` redacts env keys matching `key|token|secret|pass|auth|credential` (`:6178-6189`); `lemonade embedded start` prints a `source <file>` command, not the key. No `shell=True` in `cli.py`/`cli_agent.py`; no `os.killpg`; background spawns use `CREATE_NEW_PROCESS_GROUP` / `start_new_session`. `gaia kill` with no target exits 1; `gaia init` (non-mcp) paths all `sys.exit(code)`; `config`, `hub`, `lemonade`, `daemon *`, `agent {install,import,export}` exit 1 on failure with actionable messages. `_gaia_cli_client_params` keeps stray flags from reaching `GaiaCliClient`.
- *(sub-review)* `installer/tui/*`: sidecar digest from the committed lock, placeholder digests are a hard stop, NSIS PATH editing refuses PATHs it cannot read whole, uninstaller uses non-recursive `RMDir` on `$INSTDIR`, every build define guarded by `!ifndef … !error`; `LEMONADE_VERSION` is parsed from `version.py` by both build scripts and both workflows (the `11.5.0` in `gaia-setup.nsi:24` is a comment). `install.sh`/`install.ps1`: `set -eu` / `$ErrorActionPreference = "Stop"`, TLS 1.2 forced on PS 5.1, timeouts everywhere, hub artifacts SHA-256-verified, Windows-off-Windows guarded, PATH edits idempotent. `cloud_bootstrap.sh` arms `set -e` after its environment guards; `installer/tui/macos/scripts/postinstall` is `set -euo pipefail`.

## Hypotheses (unverified)

- `tests/installer/test_custom_agent_mcp_harness.py::test_custom_agent_dummy_mcp_path_uses_installed_bundle` fails on Windows with `Unknown tool name` after a successful MCP connect — possibly the tool-name prefixing (`mcp_<server>_<tool>`) or the stdio subprocess on Windows returning an empty tool list. Needs the MCP reviewer's eye; it may be a real Windows regression rather than a harness artefact.
- `lemonade_launcher.build_start_command` starts modern Linux via `systemctl --user start lemond`; if the Debian package installs `lemond` as a **system** unit (the module docstring says "managed by the `lemond` systemd unit" without saying which), the user-unit call fails and `_auto_start_server` reports "Server failed to start after 30s" with no hint to try `sudo systemctl start lemond`.
- `LemonadeProvider.chat` sets `repeat_penalty`/`repeat_last_n` for every backend; on the FLM/NPU server these llama.cpp-specific fields may be rejected or ignored — the same server already 500s on the `tools` field per the `MODELS` comment.
- `_ensure_model_loaded_locked` matches `MODELS` by exact `model_id` string (`_req.model_id == model`) whereas every other lookup uses `_model_ids_match`; a caller passing `user.`-prefixed or differently-cased ids falls back to `DEFAULT_CONTEXT_SIZE` (32768) instead of the registry ctx.
- *(sub-review)* NSIS `RmDir /r` on a `%USERPROFILE%\.gaia` that is a junction to another drive: NSIS ≥ 3.0 is believed not to descend into reparse points; unverified — if it does, the Electron uninstaller follows the link and deletes the target.
- *(sub-review)* The hub manifest supplies both the download URL and the SHA-256 for `install.sh`/`install.ps1` — transfer integrity, not origin (the same self-verification `fetch_sidecar.py` refuses for the sidecar). Whether the hub Worker signs manifests was not checked.
- *(sub-review)* `_uninstall_windows` Strategy 2 downloads `lemonade-server-minimal.msi` for the *installed* version; older upstream releases may not publish the minimal variant → a spurious 404 before the full-MSI attempt. `_install_windows` reads the MSI log as UTF-16 (`:674`) while `msiexec /l*v` writes ANSI unless a Unicode flag is used → timeout diagnostics may be mojibake. Neither executed.
- *(sub-review, CLI)* `download`'s outer `except` blocks reference `console` (`cli.py:3768, 3775`), which is only bound at `:3672`; no reachable `UnboundLocalError` was constructed (earlier calls swallow or wrap), but any new pre-`console` raise would surface as one. `telegram stop` / `mcp stop` kill whatever PID the file names without checking its command line (PID reuse after reboot). `kill_process_by_port` could kill the CLI's own parent if that parent holds a connection to the target port (e.g. an Electron/terminal wrapper on 4200) — plausible given the foreign-address matching, not reproduced.
