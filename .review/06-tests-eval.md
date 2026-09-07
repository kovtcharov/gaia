# 06 — Test suite quality + eval framework (main @ 211f08c5)

_Status: COMPLETE._

## Scope covered
Ran (worktree `.venv`, Windows): every `tests/unit/**` chunk twice (before/after `[ui]`), `tests/test_eval.py`, `tests/test_sdk.py`, `tests/test_agent_sdk.py`, all six `hub/agents/*/python/tests` trees, plus the memory/goal/turn-metrics suites (see `sub-145-memory.md`). Read: `tests/unit/conftest.py`, `pyproject.toml [tool.pytest.ini_options]`, `.gitattributes`, `setup.py` extras, the four workflows that install `pytest-timeout`, `test_unit.yml`; `src/gaia/eval/{runner.py (lock, find_scenarios, aggregate_scorecard, run_fix_iteration, compare_scorecards, AgentEvalRunner.run/_run_locked), scorecard.py, claude.py, config.py}`; `src/gaia/cli.py` eval-agent flag definitions and `--compare`/`--save-baseline` handlers; every `tests/fixtures/eval_baselines/*/meta.json` + scorecard; all 91 `eval/scenarios/**/*.yaml` (ids/categories only). Skimmed: `scorecard_gate.py`, `release_scorecard.py`, `.review/_05tmp/part3.md`. Not read: the other 22 `src/gaia/eval/*.py` modules (quality-metric/harness files), `tests/integration/**` bodies, Go/TS test suites.

## Task 1 — Unit-suite run on a clean checkout (Windows dev box, worktree venv, run #1: before `[ui]` extra installed)

Command (chunked by subdirectory; no `pytest-timeout` in venv so `--timeout` omitted):
`.venv\Scripts\python.exe -m pytest tests/unit/<chunk> -q -p no:cacheprovider -rfEsx`

| chunk | result | wall |
|---|---|---|
| tests/unit/agents | 3 failed, 590 passed, 68 skipped | 71s |
| tests/unit/api | 6 failed, 37 passed | 3s |
| tests/unit/chat | 15 collection ERRORS (all `tests/unit/chat/ui/*` — `No module named 'fastapi'`) | 2s |
| tests/unit/cli | 3 failed, 191 passed | 7s |
| tests/unit/connectors | 22 failed, 506 passed, 60 skipped, 182 errors | 31s |
| tests/unit/email | 2 failed, 141 passed, 8 skipped | 3s |
| tests/unit/eval | 38 failed, 443 passed, 1 skipped | 23s |
| tests/unit/factory | 61 passed | 2s |
| tests/unit/installer | 7 failed, 92 passed, 15 skipped, 3 errors | 28s |
| tests/unit/mcp | 229 passed, 2 skipped | 5s |
| tests/unit/rag | 5 passed, 3 skipped | 1s |
| tests/unit/*.py (top-level files) | 14 skipped, 15 collection errors (fastapi) | 6s |

Root-cause tally of every `E` line in run #1 (`grep "^E  " | sort | uniq -c`):
- 176 × `No module named 'fastapi'` / `gaia.ui.routers.connectors` import error — environmental (no `[ui]` extra); **but these are collection ERRORS, not skips**: the tests import FastAPI at module level rather than `pytest.importorskip`, so a contributor without `[ui]` gets red, not "skipped".
- 122 × `ConnectionError: Unit tests must not make real network connections` + 8 × `'ProactorEventLoop' object has no attribute '_ssock'` — the `_block_network` autouse guard in `tests/unit/conftest.py` breaks `asyncio` on Windows (see finding below).
- 38 × `UnicodeEncodeError: 'charmap' codec can't encode character '\u03a3'` — `tests/unit/eval/test_scorecard_gate.py`, `test_release_scorecard.py` (see finding below).
- 3 × `AttributeError: module 'os' has no attribute 'getuid'` — `tests/unit/installer/test_uninstall_command.py::TestLemonadePythonResolution` (POSIX-only test, no skipif).
- 3 × `missing '.gaia/venv' in dry-run output` + 1 × `WindowsPath('C:/srv/gaia-alt') == WindowsPath('/srv/gaia-alt')` — `test_uninstall_command.py` assumes `/` separators (Windows-only test bugs).
- 3 × `test_cli_smoke.py::test_gaia_binary_on_path[...]` — asserts `shutil.which("gaia")` is not None (venv not activated on PATH → machine-state dependent).
- 3 × `test_builder_agent.py::TestCreateAgentImpl::test_hotreload_*`, 6 × `test_sse_confirmation_gate.py`, 2 × `test_longthread_corpus_integrity.py`, 1 × `test_claude_judge.py::test_raises_on_missing_api_key`, 1 × `test_install_scripts_terminal_hub.py::test_sh_parses_under_dash` — investigated individually below.

## Task 2 — Coverage map (src/gaia + hub) — grep-verified

Method (reproducible): per-package `grep -rlE "gaia\.<pkg>([.\" ]|import)|from +gaia +import +[^#]*\b<pkg>\b"` over `tests/unit`, `tests/integration`, `tests/mcp`, `tests/*.py`, `hub/agents/*/python/tests`; zero claims double-checked with a bare class/def symbol grep; "integration" = a referencing file uses `require_lemonade` / `@pytest.mark.integration` / `real_model`.

| package | files | lines | unit files | integ files | hub/other | class | notes |
|---|---|---|---|---|---|---|---|
| `src/gaia/shell/` | 1 | 231 | 0 | 0 | 0 | **ZERO** | `prompt.py`; no `__init__.py`; **no importer anywhere in src/ or hub/** — dead code |
| `src/gaia/util.py` | 1 | 77 | 0 | 0 | 0 | **ZERO** | not `gaia/utils/`; no importer in src/ or hub/ — dead code |
| `src/gaia/vlm/` | 3 | 929 | 1 | 0 | 0 | mock-only | `vlm/mixin.py` (270 L) zero refs |
| `src/gaia/testing/` | 4 | 1243 | 1 | 0 | 0 | unit-only | one file `test_testing_utilities.py` |
| `src/gaia/sidecar/` | 2 | 277 | 1 | 0 | 0 | mock-only | |
| `src/gaia/utils/` | 3 | 988 | 2 | 0 | 1 top | mock-only | `parsing.py` (253 L) zero refs |
| `src/gaia/perf_analysis.py` | 1 | 361 | 1 | 0 | 0 | mock-only | |
| `src/gaia/device.py` | 1 | 164 | 2 | 0 | 0 | mock-only | |
| `src/gaia/cli_agent.py` | 1 | 1576 | 3 | 0 | 0 | unit-only | |
| `src/gaia/sd/` | 3 | 745 | 2 | 1 | 0 | integration | `sd/prompts.py` zero refs |
| `src/gaia/factory/` | 8 | 3341 | 3 | 0 | 0 | unit-only | |
| `src/gaia/scratchpad/` | 2 | 576 | 4 | 0 | 0 | unit-only | |
| `src/gaia/schedule/` | 5 | 463 | 5 | 0 | 0 | unit-only | |
| `src/gaia/messaging/` | 2 | 401 | 5 | 0 | 0 | unit-only | |
| `src/gaia/filesystem/` | 3 | 1226 | 4 | 0 | 0 | unit-only | |
| `src/gaia/security.py` | 1 | 735 | 3 | 0 | 2 top | unit-only | |
| `src/gaia/web/` | 3 | 1562 | 6 | 0 | 0 | unit-only | no live-network test |
| `src/gaia/audio/` | 5 | 1839 | 5 | 0 | 1 top | unit-only | + in-source `src/gaia/audio/tests/` (533 L) outside `testpaths` — never collected by default |
| `src/gaia/talk/` | 3 | 842 | 4 | 0 | 1 top | mock-only | `talk/app.py` (287 L) zero refs |
| `src/gaia/chat/` | 4 | 2397 | 7 | 0 | 4 top, 1 hub | unit-only | `chat/app.py` (428 L), `chat/prompts.py` (522 L) zero refs |
| `src/gaia/api/` | 7 | 2492 | 5 | 0 | 2 top | integration-marked | |
| `src/gaia/code_index/` | 3 | 1534 | 3 | 1 | 0 | integration | |
| `src/gaia/skills/` | 28 | 12271 | 24 | 0 | 4 hub | unit-only | **no integration tier for 12K lines** |
| `src/gaia/installer/` | 7 | 5184 | 14 | 0 | 1 | mock-only | + `tests/installer/` harness; `_stdin.py` zero refs |
| `src/gaia/governance/` | 13 | 1447 | 7 | 5 | 0 | integration | |
| `src/gaia/rag/` | 6 | 4654 | 10 | 1 | 7 top | integration | `rag/demo.py` (304 L) zero refs |
| `src/gaia/database/` | 5 | 1138 | 7 | 4 | 16 hub | integration | |
| `src/gaia/mcp/` | 17 | 5789 | 20 | 1 | 4 mcp | integration | `client/transports/{base,http}.py` never exercised (only `StdioTransport` imported) |
| `src/gaia/hub/` | 9 | 5157 | 29 | 1 | 3 | integration | |
| `src/gaia/cli.py` | 1 | 8087 | 30 | 4 | 3 | integration | |
| `src/gaia/eval/` | 28 | 17025 | 33 | 4 | 4 | integration | |
| `src/gaia/daemon/` | 43 | 9904 | 42 | 6 | 2 | integration | `daemon/lock.py::StartLock` zero refs |
| `src/gaia/connectors/` | 31 | 7984 | 69 | 1 | 28 | integration | |
| `src/gaia/llm/` | 15 | 9314 | 50 | 7 | 21 | integration | best covered |
| `src/gaia/ui/` | 38 | 21446 | 72 | 11 | 7 | integration | `routers/system.py` (1060 L) and `routers/tunnel.py` (54 L) never named by a test (only import-mounted via `create_app`) |
| `src/gaia/agents/` | 41 | 40217 | 125 | 10 | 74 | integration | `agents/tools/skill_library_tools.py` (646 L) **zero refs** — the hub test with a similar name tests `gaia_agent.skill_tools`; `screenshot_tools.py` effectively zero |

Individual modules with zero test references (verified twice): `ui/routers/system.py` 1060, `agents/tools/skill_library_tools.py` 646, `chat/prompts.py` 522, `chat/app.py` 428, `mcp/context7_cache.py` 332, `rag/demo.py` 304, `talk/app.py` 287, `vlm/mixin.py` 270, `skills/audit/permission_truth.py` 264, `utils/parsing.py` 253, `shell/prompt.py` 231, `eval/code_bench_fixtures.py` 216, `sd/prompts.py` 146, `mcp/client/transports/http.py` 135, `connectors/prior_state.py` 119, `daemon/lock.py` 85, `util.py` 77, `mcp/client/transports/base.py` 56, `ui/routers/tunnel.py` 54, `installer/_stdin.py` 23 — **≈5.6K lines**.

Hub agents: `email` 91 src files / 42.5K L ↔ 128 test files / 43.9K L (+ vitest in `npm/`); `gaia` 14 / 4.8K ↔ 14 / 5.5K (+ vitest); **`chat` 8 / 4.9K ↔ 3 files / 221 L** (two are dependency-floor checks) — the flagship's base class is essentially untested in its own package; `connectors-demo`, `word-count`, `hello-world` one file each.

Markers (pyproject `[tool.pytest.ini_options]`, `--strict-markers` on): `integration` 30 uses, `real_model` 8, `slow` 4, `real_slm_build` 4 (hub only), `network` 1, `gmail_live` 1, `distributed_seams` 1; `allow_network` 38 uses (registered dynamically in `tests/unit/conftest.py::pytest_configure`, not in pyproject — so it is *only* valid when tests/unit/conftest.py loads, i.e. unusable from tests/integration). No `hardware` marker exists. `require_lemonade` used by 11 files.

## Task 1 (cont.) — Run #2 with `[ui]` extra installed, plus hub / top-level / mcp / integration files

| target | result |
|---|---|
| tests/unit/connectors | 33 failed, 508 passed, 12 skipped, **217 errors** (more than run #1: fastapi now imports so the router tests reach the asyncio/socketpair guard — see F1) |
| tests/unit/email | 143 passed, 8 skipped (`gaia_agent_email` not installed → importorskip) |
| tests/unit/eval | 37 failed (all cp1252 `Σ` in test helper), 457 passed |
| tests/unit/installer | 7 failed, 3 errors (Windows path/`os.getuid` assumptions) |
| tests/unit/factory, mcp, rag | all pass (61 / 229+2 skipped / 55) |
| tests/test_eval.py | 140 passed |
| tests/test_sdk.py | **17 failed**, 66 passed, 1 skipped, 2 xfailed, 78 s — stale API references + one test that really calls Lemonade (see F5) |
| tests/test_agent_sdk.py | 8 skipped (31 s spent probing Lemonade before skipping) |
| hub/agents/{word-count,connectors-demo}/python/tests | collection ERROR (package not installed; no `importorskip`) |
| hub/agents/email/python/tests | 92 skipped, 18 collection ERRORS (partially guarded) |
| hub/agents/gaia, chat, hello-world | see log; chat/hello-world same "package not installed" pattern |

Individual failure root causes (all reproduced, `--tb=short`):
- `tests/unit/eval/test_scorecard_gate.py` ×36 + `test_release_scorecard.py` ×8 (+ `test_release_scorecard` ones overlap): the **test helper** `_write_card` does `path.write_text(render_scorecard(payload))` with no `encoding=` (test_scorecard_gate.py:46) while the template contains `Σ` (release_scorecard.py:255) → `UnicodeEncodeError: 'charmap'` on any cp1252 console. Product code uses `encoding="utf-8"` (release_scorecard.py:365, scorecard_gate.py:366) — test-only bug, but it makes the eval unit suite red on every Windows dev box.
- `tests/unit/eval/test_claude_judge.py::test_raises_on_missing_api_key` — `DID NOT RAISE`: `gaia/eval/claude.py:24` calls `load_dotenv()` at import time, which walks *up* from `src/gaia/eval/` and found `C:/Users/14255/Work/gaia/.env` (contains `ANTHROPIC_API_KEY`) two directories above the worktree. The test `monkeypatch.delenv`s then imports the module → the key is re-injected from the ancestor `.env`. Machine-state dependency (any `.env` in an ancestor dir).
- `tests/unit/installer/test_install_scripts_terminal_hub.py::test_sh_parses_under_dash` — worktree has `installer/scripts/install.sh` checked out CRLF (`git ls-files --eol` → `i/lf w/crlf`; `core.autocrlf=true`; `.gitattributes` has no `*.sh text eol=lf`), `dash -n` → `Syntax error: word unexpected`. Any Windows contributor with autocrlf sees this.
- `tests/unit/installer/test_uninstall_command.py` ×7+3: asserts `'.gaia/venv' in captured.text` against `\fake\home\user\.gaia\venv`; `Path("/srv/gaia-alt") == WindowsPath('C:/srv/gaia-alt')`; `TestLemonadePythonResolution::*_posix` call `os.getuid` with no `skipif(sys.platform == "win32")`.
- `tests/unit/cli/test_cli_smoke.py::test_gaia_binary_on_path[gaia|gaia-cli|gaia-mcp]` — `shutil.which("gaia")` must be non-None: depends on the venv being on PATH, not on the package.
- `tests/unit/agents/test_builder_agent.py::TestCreateAgentImpl::test_hotreload_*` ×3 and `tests/unit/api/test_sse_confirmation_gate.py` ×6 and `tests/unit/email/test_longthread_corpus_integrity.py` ×2 failed in the chunked run #1 but **pass in isolation and in run #2** — they needed `fastapi`/order; not Windows bugs.

**Correction to the Task 2 table (re-verified before writing Findings):** "zero refs" there meant *zero test references*. Re-grepped for *importers* in `src/`+`hub/`: `daemon/lock.py::StartLock` IS used (`daemon/client.py:34,94`), `mcp/context7_cache.py` IS used (`cli.py:5080`, `mcp/external_services.py:200`), `connectors/prior_state.py` IS used (`connectors/flow.py:55`, `oauth_pkce.py:208`), and `agents/tools/skill_library_tools.py` + `vlm/mixin.py` are loaded **by name** through `registry.py:51-52` `KNOWN_TOOLS` (a `from … import` grep misses them). The confirmed no-importer set is therefore only: `shell/prompt.py` (231), `util.py` (77), `utils/parsing.py` (253), `chat/prompts.py` (522), `rag/demo.py` (304) — F10 below is scoped to those five.

`test_unit.yml` runs on `ubuntu-latest` (line 51), so every Windows-only breakage below is invisible to CI and lands only on Windows contributors — which is the platform the product targets.

## Findings

### 🟡 F1 — `_block_network` guard breaks asyncio on Windows: 217 `TestClient` tests error before running
- **Where:** `tests/unit/conftest.py:53-73` (`_block_network`, autouse) — surfaces in every FastAPI-router test module (`tests/unit/connectors/*router*`, `tests/unit/test_memory_router.py:77`, `tests/unit/chat/ui/*`)
- **What:** The guard monkeypatches `socket.socket.connect`. On Windows, `socket.socketpair()` is emulated with a loopback `connect()`, and asyncio's event loop creates one for its self-pipe, so `TestClient.__enter__` → anyio portal → asyncio `Runner` dies with the guard's `ConnectionError` (or `'ProactorEventLoop' object has no attribute '_ssock'`) before any test body runs.
- **Failure scenario:** A Windows contributor runs `pytest tests/unit/connectors` → 217 errors; `tests/unit/test_memory_router.py` → 124 errors. Every HTTP-router test in the unit tier is red on the platform GAIA targets, so router regressions are only caught by ubuntu CI.
- **Evidence:** run #2: `33 failed, 508 passed, 12 skipped, 217 errors`; traceback `test_memory_router.py:77 in client → TestClient.__enter__ → anyio.from_thread.start_blocking_portal → asyncio Runner → BaseEventLoop._close_self_pipe`; guard body: `monkeypatch.setattr(socket.socket, "connect", _blocked_connect)`.
- **Fix:** In `_blocked_connect`, inspect the address and allow loopback (`127.0.0.1`, `::1`, `localhost`); alternatively patch `socket.create_connection`/`getaddrinfo` for non-loopback hosts only. Add one Windows job (or `windows-latest` matrix entry) to `test_unit.yml` so this class of breakage is seen.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F2 — cp1252 test helper turns the whole eval unit suite red on Windows (37-44 failures)
- **Where:** `tests/unit/eval/test_scorecard_gate.py:46` (`_write_card`) and the sibling helper in `tests/unit/eval/test_release_scorecard.py`; template `src/gaia/eval/release_scorecard.py:255`
- **What:** The helper writes the rendered scorecard with `path.write_text(render_scorecard(payload))` — no `encoding=` — while the template contains `Σ`/`×`/subscript characters. Product code writes UTF-8 (`release_scorecard.py:365`, `scorecard_gate.py:366`); only the test fixture is wrong.
- **Failure scenario:** Any cp1252-default Python (every Windows console without `PYTHONUTF8=1`) → `UnicodeEncodeError: 'charmap' codec can't encode character '\u03a3'` × 37 (run #2) / 38 (run #1) in `tests/unit/eval`.
- **Evidence:** `path.write_text(render_scorecard(payload))` (test_scorecard_gate.py:46); `Formula: \`round(100 × Σ(weightᵢ × valueᵢ) / Σ(weightᵢ), 2)\`` (release_scorecard.py:255); run #2 `37 failed (all cp1252 Σ in test helper), 457 passed`.
- **Fix:** `path.write_text(…, encoding="utf-8")` in both helpers; consider a repo-wide lint (`ruff` `PLW1514` / `encoding` rule) since product code already follows it.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F3 — `test_raises_on_missing_api_key` depends on there being no `.env` in any ancestor directory
- **Where:** `tests/unit/eval/test_claude_judge.py:51-59`; root cause `src/gaia/eval/claude.py:24` (`load_dotenv()` at import time)
- **What:** The test deletes `ANTHROPIC_API_KEY` and imports `gaia.eval.claude`, but the module runs `load_dotenv()` at import, which searches *upward* from the module and re-injects the key from any ancestor `.env`. On this box it found `C:/Users/14255/Work/gaia/.env` two directories above the worktree.
- **Failure scenario:** Any developer whose repo (or a parent of a worktree) holds a `.env` with the key → `Failed: DID NOT RAISE ValueError`; conversely the product silently picks up credentials from outside the repo, which is also a surprise for `gaia eval` users.
- **Evidence:** `monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)` … `from gaia.eval.claude import ClaudeClient` … `with pytest.raises(ValueError, match="ANTHROPIC_API_KEY")`; module top: `load_dotenv()`.
- **Fix:** Move `load_dotenv()` into `ClaudeClient.__init__` (or the CLI entry) and pass an explicit path (`find_dotenv(usecwd=True)` bounded to the repo root); in the test, `monkeypatch.setattr("gaia.eval.claude.load_dotenv", lambda *a, **k: None)` or set `override=False` semantics and assert on the env after import.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F4 — `install.sh` has no `eol=lf` attribute, so autocrlf checkouts ship a CRLF installer and fail `test_sh_parses_under_dash`
- **Where:** `.gitattributes` (no `*.sh` rule); `installer/scripts/install.sh`; `tests/unit/installer/test_install_scripts_terminal_hub.py::test_sh_parses_under_dash`
- **What:** With `core.autocrlf=true` (the Git-for-Windows default) the shell installer is checked out CRLF; `dash -n` rejects it. The same CRLF file is what a Windows developer would package or `scp` to a Linux box.
- **Failure scenario:** Windows contributor: `git ls-files --eol installer/scripts/install.sh` → `i/lf w/crlf`; test fails with `Syntax error: word unexpected`. Real-world: a Windows-built artifact containing `install.sh` breaks on POSIX with the same error.
- **Evidence:** `.gitattributes` contents (binary/fixture rules only, no `*.sh text eol=lf`); `git config core.autocrlf` → `true`; `git ls-files --eol` → `i/lf w/crlf`.
- **Fix:** Add `*.sh text eol=lf` (and `install.ps1 text eol=crlf` if desired) to `.gitattributes`; add a unit test asserting no `\r` in `installer/scripts/*.sh` bytes so the property is enforced on every platform.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F5 — `tests/test_sdk.py`: 17 tests assert an SDK surface that no longer exists, and one really boots an agent against Lemonade
- **Where:** `tests/test_sdk.py` (88 tests; never run by any workflow per the CI reviewer, `.review/_05tmp/part3.md` §"43 test files")
- **What:** The file pins a stale contract: `gaia.api.openai_server.create_app` (3 tests), `gaia.agents.base.agent.STATE_COMPLETION`, `Agent.execute_tool` (public name; today it is `_execute_tool`), `RAGSDK.index_documents`, `KokoroTTS.synthesize`, `AudioClient.play_audio`, `ApiAgent.format_for_api`, `gaia.mcp.agent_mcp_server`, `docs/SDK.md`, and `LLMClient()` as a concrete class — none exist at HEAD. `TestAgentIntegration::test_agent_with_mocked_llm` patches `AgentSDK().complete` (no longer the call path) so the real agent runs, tries to load `Gemma-4-E4B-it-GGUF` on `localhost:13305`, and fails after ~16 s with a connection error — a "unit" test that requires Lemonade.
- **Failure scenario:** `pytest tests/test_sdk.py` → `17 failed, 66 passed, 1 skipped, 2 xfailed` in 78 s on a machine without Lemonade; with Lemonade running, `test_agent_with_mocked_llm` would hit the real model. Because no workflow runs the file, nobody notices the SDK contract drifted.
- **Evidence:** `ImportError: cannot import name 'create_app' from 'gaia.api.openai_server'` ×3; `ImportError: cannot import name 'STATE_COMPLETION'`; `AttributeError: 'TestAgent' object has no attribute 'execute_tool'`; `assert False where False = hasattr(<class 'gaia.rag.sdk.RAGSDK'>, 'index_documents')`; `AssertionError: SDK.md should exist in docs/`; `--durations`: `16.39s call tests/test_sdk.py::TestAgentIntegration::test_agent_with_mocked_llm` with `lemonade_client.py:3602 Failed to load Gemma-4-E4B-it-GGUF … localhost:13305 … actively refused`.
- **Fix:** Either delete `tests/test_sdk.py` (its live assertions are duplicated by `tests/unit/agents/*`) or rewrite it against the current public surface (`docs/sdk/*.mdx` is the contract) and wire it into `test_unit.yml`; give `test_agent_with_mocked_llm` a `require_lemonade` fixture or patch `LemonadeClient`/`send_messages`, not `AgentSDK.complete`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F6 — The eval "one run at a time" guard is a silent no-op on Windows
- **Where:** `src/gaia/eval/runner.py:40-43, 94-110` (`_acquire_eval_lock`)
- **What:** The serial-execution lock is `fcntl.flock` on `<tmp>/gaia-eval-agent.lock`; on Windows `fcntl` is `None` and the context manager just `yield`s. CLAUDE.md's "Run agent evals SERIALLY, never in parallel" rule relies on this guard, and the runner's own comment lists the concurrency failure modes it exists to prevent (`n_ctx=4096` overflows, `model_load_error`, spurious `BLOCKED_BY_ARCHITECTURE`). The Ryzen AI target platform is Windows.
- **Failure scenario:** Two `gaia eval agent` invocations on a Windows box (e.g. `--fix` mode plus a manual run) race-evict each other's Lemonade model; the scorecard records bogus failures with no warning that the guard was absent.
- **Evidence:** `if sys.platform == "win32": fcntl = None` … `if fcntl is None: yield; return` — no log line, unlike the read-only-`/tmp` branch which prints `[WARN] … Skipping concurrency guard`.
- **Fix:** Use `msvcrt.locking` on Windows (or an `O_CREAT|O_EXCL` PID file with the same stale-PID reclaim), and at minimum print the same `[WARN]` the OSError branch prints so the degradation is visible.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F7 — `--fix` runs Claude Code with `--dangerously-skip-permissions` on the developer's working tree with no safety rails
- **Where:** `src/gaia/eval/runner.py:1329-1415` (`run_fix_iteration`), `src/gaia/cli.py:2077-2091` (`--fix`, `--max-fix-iterations`)
- **What:** Each fix iteration executes `claude -p <prompt> --dangerously-skip-permissions` with `cwd=REPO_ROOT`, up to 3 times by default, with a 600 s timeout. Nothing checks that the tree is clean, that the user is on a branch, or that the fixer stayed inside the files the prompt names; the only "do NOT commit" rule is prose inside the prompt. A fixer timeout is swallowed into the fix log (`"error": "Fixer timed out after 600s"`) and the loop proceeds to re-eval whatever half-applied edits exist.
- **Failure scenario:** Developer with uncommitted work runs `gaia eval agent --category tool_selection --fix`; the fixer edits `agent.py`/`_chat_helpers.py` over their in-progress changes, times out mid-edit, and the next iteration evaluates a mixed tree. Recovering requires manual `git diff` archaeology.
- **Evidence:** `cmd = [claude_cmd, "-p", prompt, "--dangerously-skip-permissions"]`; `subprocess.run(cmd, …, timeout=600, cwd=str(REPO_ROOT), check=False)`; no `git status --porcelain` / branch check anywhere in `runner.py` (grep `porcelain|git ` → none).
- **Fix:** Refuse to start `--fix` on a dirty tree or on `main` unless `--fix-allow-dirty`; run the fixer in a `git worktree` (or at least `git stash`-guard) and print the resulting `git diff --stat` after each iteration; abort the loop on fixer timeout instead of re-evaluating.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F8 — Baseline plumbing contradicts the documented workflow: `--save-baseline` writes a file the docs never mention, and the default judge differs from every committed baseline
- **Where:** `src/gaia/cli.py:3843-3859` (`--compare` single-path → `eval/results/baseline.json`), `:3954-3963` (`--save-baseline` → `eval/results/baseline.json`); `src/gaia/eval/config.py:16` (`DEFAULT_CLAUDE_MODEL = "claude-opus-5"`); `tests/fixtures/eval_baselines/*/scorecard_*.json` (`config.model == "claude-sonnet-4-6"`); `CLAUDE.md:405-414`, `docs/reference/troubleshooting.mdx:402`
- **What:** CLAUDE.md and the troubleshooting guide tell developers to compare against `tests/fixtures/eval_baselines/<model>-<commit>/scorecard_<cat>.json` and to "regenerate the baseline with `--save-baseline`". But `--save-baseline` writes only `eval/results/baseline.json` (git-ignored results dir, one file for all categories, overwritten by the next `--save-baseline`), and nothing in `src/gaia` reads or writes `tests/fixtures/eval_baselines/` — those directories are hand-assembled. Separately, all seven committed scorecards were judged by `claude-sonnet-4-6` while the CLI default judge is `claude-opus-5`, so every documented `--compare` prints the `_warn_on_judge_mismatch` banner ("not directly comparable") unless the developer also passes `--model claude-sonnet-4-6` — which no doc says.
- **Failure scenario:** Developer follows CLAUDE.md verbatim: runs the eval (opus judge), compares against the gemma-4-e4b-d71cd914 sonnet baseline → judge-mismatch banner + deltas that mix judge drift with real change; then runs `--save-baseline` expecting to refresh the committed fixture and gets a file in `eval/results/` that the next `--save-baseline` for a *different* category silently overwrites.
- **Evidence:** `baseline_path = RESULTS_DIR / "baseline.json"` (both handlers); `grep -rn eval_baselines src/gaia` → only `sidecar_harness.py` (a different feature); `DEFAULT_CLAUDE_MODEL = "claude-opus-5"` vs baseline `config: {'model': 'claude-sonnet-4-6'}` (all 7 files); `_warn_on_judge_mismatch` compares `config.model`.
- **Fix:** Make `--save-baseline` write `tests/fixtures/eval_baselines/<model-slug>-<short-sha>/scorecard_<category>.json` + `meta.json` (or take a `--baseline-dir`), make `--compare CURRENT` resolve the committed baseline for the run's model/category, and either pin the judge for regression compares (`--judge` default read from the baseline's `config.model`) or document `--model claude-sonnet-4-6`. Fix CLAUDE.md/troubleshooting to match whichever is chosen.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F9 — Committed baselines lag the scenario set; 9 of 12 eval categories have no baseline at all
- **Where:** `tests/fixtures/eval_baselines/**`, `eval/scenarios/**` (91 YAML scenarios)
- **What:** Cross-checking scenario ids per category: every `rag_quality` baseline lacks `safety_handbook_water` (8 current vs 7 in baseline) and every `tool_selection` baseline lacks `data_vs_recall_disambiguation` (5 vs 4); no orphans. Categories `memory` (25 scenarios), `real_world` (19), `mcp_reliability` (10), `web_system` (6), `adversarial`, `error_recovery`, `personality`, `vision` (3 each) and `captured` (2) have **no committed baseline**, so the CLAUDE.md rule "compare to the committed baseline" cannot be followed for 71 of 91 scenarios — including the entire memory category that this review's memory sub-report found regressions in.
- **Failure scenario:** A prompt change to the memory extractor passes unit tests; the developer runs `gaia eval agent --category memory` as required, gets a scorecard, and has nothing to `--compare` against — the eval gate is advisory for the biggest category.
- **Evidence:** script output above: `missing_from_baseline=['safety_handbook_water']` ×3, `['data_vs_recall_disambiguation']` ×2; `categories with NO baseline at all: ['adversarial','captured','error_recovery','mcp_reliability','memory','personality','real_world','vision','web_system']`; the newest Gemma baseline (`95e4b372`, 2026-05-11) lists `"categories_not_captured": ["tool_selection","context_retention"]`.
- **Fix:** Regenerate the three existing categories under the current judge and add `memory`, `mcp_reliability`, `web_system` baselines (the ones with the most scenarios); add a unit test that fails when a scenario id in a baselined category has no baseline entry (a "baseline drift" guard) so new scenarios come with baseline updates.
- **Confidence:** High
- **Tracked:** none found

### 🟡 F10 — Five modules (~1.4K lines) have no importer anywhere in `src/` or `hub/` and no tests
- **Where:** `src/gaia/shell/prompt.py` (231, no `__init__.py`), `src/gaia/util.py` (77), `src/gaia/utils/parsing.py` (253), `src/gaia/chat/prompts.py` (522), `src/gaia/rag/demo.py` (304)
- **What:** Nothing in the package imports them (`grep -rlE` over `src`+`hub`, re-verified — see the correction note above for the modules that *do* have importers), and no test references them. They still ship in the wheel and are part of the `[ui]`/`[eval]` review surface.
- **Failure scenario:** A refactor changes a signature they call (e.g. `chat/prompts.py` templates vs. `hub/agents/chat` prompt assembly); nothing fails, and the dead copy drifts into a misleading reference for the next reader.
- **Evidence:** `gaia.shell → 0`, `gaia.util\b → 0`, `gaia.utils.parsing → 0`, `chat.prompts → 0`, `rag.demo → 0` importers (src+hub); Task 2 table for test refs.
- **Fix:** Delete them (or move `rag/demo.py` under `examples/`), and add `tests/unit/test_dead_modules.py`-style guard that every `src/gaia/**.py` is imported by at least one non-test module or listed in an explicit allowlist.
- **Confidence:** High
- **Tracked:** none found

### 🟢 F11 — POSIX-only and PATH-dependent unit tests lack `skipif`, so a clean Windows checkout is red for reasons unrelated to code
- **Where:** `tests/unit/installer/test_uninstall_command.py` (`TestLemonadePythonResolution::*_posix` → `os.getuid`; `'.gaia/venv'` substring against a `WindowsPath`; `Path("/srv/gaia-alt")` equality at `:650`); `tests/unit/cli/test_cli_smoke.py:204-209` (`test_gaia_binary_on_path` asserts `shutil.which("gaia")`)
- **What:** Ten `test_uninstall_command.py` cases assume `/` separators or `os.getuid`; `test_gaia_binary_on_path` asserts on the *shell PATH* (venv activated) rather than on the installed package — it already has a `skipif` for "package not installed" but not for "shim not on PATH".
- **Failure scenario:** Fresh Windows venv, `python -m pytest tests/unit/installer tests/unit/cli` → 7 failed + 3 errors + 3 failed with `AttributeError: module 'os' has no attribute 'getuid'`, `WindowsPath('C:/srv/gaia-alt') == WindowsPath('/srv/gaia-alt')`, `Binary 'gaia' not found on PATH`.
- **Evidence:** run #1/#2 tallies above; test bodies cited.
- **Fix:** `pytest.mark.skipif(sys.platform == "win32", …)` on the `_posix` cases; build expected paths with `Path(...)`/`os.path.join` and compare `Path` objects; for the PATH test, resolve `sysconfig.get_path("scripts")` and assert the shim file exists there instead of `shutil.which`.
- **Confidence:** High
- **Tracked:** none found

### 🟢 F12 — Optional-dependency tests error at collection instead of skipping (176 `fastapi` errors; every hub agent package)
- **Where:** 31 test modules import `fastapi`/`TestClient` at module level (e.g. `tests/unit/chat/ui/test_agents_router.py:9`) vs 13 that use `pytest.importorskip("fastapi")`; `hub/agents/{word-count,connectors-demo,hello-world,chat,gaia}/python/tests/` have no `conftest.py` and no `importorskip`; `hub/agents/email/python/tests` is only partially guarded (18 collection errors)
- **What:** Without the `[ui]` extra, `tests/unit/chat/ui/*` and 14 top-level `tests/unit/*.py` files are 176 collection ERRORS (run #1); without a hub package installed, its test dir is a collection ERROR (`ModuleNotFoundError: No module named 'gaia_agent_word_count'`). The `allow_network` marker is registered only in `tests/unit/conftest.py::pytest_configure`, so with `--strict-markers` it is unusable from `tests/integration` or hub tests.
- **Failure scenario:** `pytest tests/unit` on a core-only install stops with "Interrupted: N errors during collection" — the contributor cannot tell environmental from real failures; `pytest hub/agents/word-count/python/tests` cannot even be used to *check* whether the package is installed.
- **Evidence:** counts above; `pyproject.toml [tool.pytest.ini_options].markers` lacks `allow_network`; `tests/unit/conftest.py:50` registers it dynamically.
- **Fix:** `fastapi = pytest.importorskip("fastapi")` at the top of each router test module (or a `tests/unit/chat/ui/conftest.py` doing it once); a `conftest.py` per hub package with `pytest.importorskip("gaia_agent_<id>")`; move `allow_network` into `pyproject.toml` markers.
- **Confidence:** High
- **Tracked:** none found

## Task 3 — Mock validity (do the tests prove the call is *valid*, or only that it happened?)

Numbers (tests/unit, grep): 396 payload-asserting calls (`assert_called_once_with(` / `assert_called_with(` / `call_args`) vs 124 existence-only assertions (`assert_called_once()` / `.called`). The ratio is healthy overall; the concentration is not — `tests/unit/test_init_command.py` alone has 19 existence-only assertions, i.e. the install/`gaia init` path (the CLAUDE.md #1655 canonical case) is still the file that most often proves "we called it" rather than "the call would be accepted".

Concrete examples found:
- **Contract stubbed at the HTTP layer, shape never asserted** — `tests/test_lemonade_client.py::test_pull_model` (lines 961-990) registers `responses.POST /pull → {"status":"success"}` and asserts `result == pull_response`; it never asserts the outgoing JSON body (that a built-in model is sent *without* `recipe`, the exact #1655 bug). Per the CI reviewer this file is also never run in CI and 6 of its tests fail today.
- **Real machine state leaking through a partial stub** — `tests/unit/test_memory_discovery.py::TestCredentialManagerContractShape::test_windows_email_failures_warn_like_the_other_platforms[_scan_outlook_registry-…]` stubs only the Outlook scanner and expects `[]`, but the other Windows scanners run for real and returned the developer's actual Gmail account (`Email account: kalin.ovtcharov@gmail.com`). Fails on any box with a configured mail client; passes on a sterile CI runner — the inverse of the "hidden-state masking" rule, but the same root cause (the test does not control its inputs).
- **Mock aimed at a call path that no longer exists** — `tests/test_sdk.py::TestAgentIntegration::test_agent_with_mocked_llm` patches `AgentSDK().complete`; the agent no longer calls it, so the "mock" is bypassed and the test really contacts Lemonade (F5). A mock that is never hit is indistinguishable from a passing mock unless the test also asserts it was called — this one does (`.called`), which is the only reason the drift is visible.
- **Good pattern to copy** — `tests/unit/test_memory_store.py` and `tests/unit/test_goal_store.py` run against a real SQLite file in `tmp_path` (≈750 tests, all pass on Windows); `tests/unit/eval/test_scorecard_gate.py` renders real scorecards to disk. Where a boundary has a local, deterministic implementation, the suite already prefers the real thing.
- **Judge boundary** — `tests/unit/eval/test_claude_judge.py` mocks `anthropic` wholesale (`_make_mock_anthropic()`); nothing asserts the request shape (`messages` structure, `max_tokens`, absence of `temperature` — the `_sampling_kwargs` contract that the docstring says newer models 400 on). One test asserting `client.messages.create.call_args.kwargs` would pin it.

## Task 4 — Test hygiene

- **Layout:** `testpaths = ["tests"]`; `src/gaia/audio/tests/{test_audio_pipeline,test_mic_simple,test_talk_basic}.py` (533 L) live in-source, outside `testpaths`, and are collected by nothing (they also need a microphone). No tracked `__pycache__`/`.pyc` (0 in `git ls-files`; `.gitignore:5,9` cover them) — the `.pyc` files seen in `find` output are local build artifacts.
- **Markers:** `--strict-markers` is on; 7 markers in `pyproject.toml`; `allow_network` registered only via `tests/unit/conftest.py` (F12). No `hardware`/`windows_only` marker despite Windows-only and POSIX-only tests existing (F11).
- **Skips:** 316 `pytest.importorskip(` (the dominant guard — good), 165 runtime `pytest.skip(` (mostly `require_lemonade`-style), only 2 `pytest.mark.skip(` (one reason: "Parameter setting API is still in development") and 1 `xfail` — the suite is not hiding failures behind blanket skips.
- **Plugin parity:** `pytest-timeout` is installed ad hoc by `test_unit.yml:90`, `test_api.yml:71`, `test_distributed_seams.yml:79`, `test_examples.yml:69` and `--timeout=300` is used by `test_unit.yml:122` and `test_examples.yml`, but `pytest-timeout` is in **no** extra in `setup.py`/`pyproject.toml` — a developer copying a CI command locally gets `unrecognized arguments: --timeout`. Add it to `[dev]`.
- **Runtime:** `tests/unit` ≈ 5 min on this box excluding the F1/F2 error storm; `tests/test_agent_sdk.py` spends 31 s probing Lemonade before skipping 8 tests (probe timeout is per-test, not per-session) — cache the `require_lemonade` result at session scope.
- **Chunk-order sensitivity:** `test_builder_agent.py::test_hotreload_*`, `test_sse_confirmation_gate.py`, `test_longthread_corpus_integrity.py` fail in a chunked run but pass in isolation — global state (`_TOOL_REGISTRY`, `sys.modules` for fastapi) leaks between tests; `pytest-randomly` is not used, so the order that CI happens to run is the only order known to pass.

## Task 5 — Eval framework (`src/gaia/eval`, 28 files / 17,025 lines; read: `runner.py` lock/scenarios/aggregate/fix/compare/run paths, `scorecard.py`, `claude.py`, `config.py`; skimmed: `scorecard_gate.py`, `release_scorecard.py`)

**Judge handling** — `ClaudeClient` (`claude.py:30-100`) fails loudly on missing `anthropic`/`bs4`/`ANTHROPIC_API_KEY` with actionable messages (subscription-token path documented); retries are explicit (`max_retries` ctor arg → SDK backoff; CLAUDE.md-compliant). `temperature` is omitted unless pinned. `load_dotenv()` at import time is the one landmine (F3). `_warn_on_judge_mismatch` (`runner.py:1416`) correctly refuses to let two judges' scores be diffed silently — but the default judge (`claude-opus-5`) mismatches every committed baseline (F8).

**Scorecard semantics** (`scorecard.py:16-207`) — sound and worth keeping: `avg_score` counts only judged statuses and caps FAIL scores at 5.99 so a "high-scoring failure" cannot lift the average; `judged_pass_rate` excludes infra/setup/skipped; unknown statuses are bucketed as `errored` **and** surfaced in `scorecard["warnings"]` + stderr rather than dropped. `SKIPPED_NO_DOCUMENT` is tracked separately.

**`--compare` semantics** (`runner.py:1439-1693`, `cli.py:3843-3878`) — per-scenario, keyed by `scenario_id`; results missing an id are skipped with a `[WARN]` (not silently). Classes: PASS→FAIL `regressed`, FAIL→PASS `improved`, same-status drop ≥ `_SCORE_REGRESSION_THRESHOLD = 2.0` → `score_regressed`, elapsed > 2× → `time_regressed`, one side `SKIPPED_NO_DOCUMENT` → `corpus_changed` (neutral), `only_in_baseline`/`only_in_current` reported but **not** counted as issues. Exit code: 2 on any regressed/score_regressed/time_regressed, 0 otherwise, 1 on missing file. Masking risks worth knowing: (a) a scenario that *disappears* from the current run (crash before judging → no entry, or renamed) is reported under `ONLY IN BASELINE` and does **not** fail the compare; (b) a non-numeric `overall_score` is coerced to 0 for the delta, so PASS→PASS with a null current score reads as a −9 "score changed" line but only trips the ≥2.0 rule if the baseline had a numeric score; (c) `time_regressed` takes precedence over `regressed` in the `elif` chain (`runner.py:1540-1553`), so a scenario that both got slower and went PASS→FAIL is listed as a *time* regression only — still exit 2, but the report under-states it; (d) rounding: summary deltas print at `.0f`% / `.1f` while the threshold is applied to unrounded deltas — consistent, no off-by-rounding.

**Baseline selection** — no code selects a baseline by model/commit; `--compare CURRENT` uses `eval/results/baseline.json`, the committed fixtures are manual (F8, F9). The newest Gemma baseline is 4 months old (2026-05-11) and predates the memory/procedural-memory work.

**`--fix` safety** — F7. Additionally the fixer prompt is loaded from `eval/prompts/fixer.md` if present, with `str.replace` templating (safe against braces), and `fix_history.json` is written even on error — good; the loop's stop condition uses `judged_pass_rate` (correct denominator).

**Serial guard** — POSIX-only (F6); the bypass env `GAIA_EVAL_NO_LOCK=1` is documented in the module comment; stale-PID reclaim uses `os.kill(pid, 0)` which is fine on POSIX.

**Scenarios vs baselines** — 91 YAML scenarios in 12 categories; `find_scenarios` validates schema and lets `~/.gaia/eval/scenarios` and `--scenario-dir` override built-ins by id (logged at INFO — a user-local override silently changing a "baseline" category run is a foot-gun worth a WARN). Orphans/missing: F9.

## Task 6 — Tests never run by any workflow (cross-check of `.review/_05tmp/part3.md` §"43 test files / ≈780 test functions")

The CI reviewer's method (every `pytest`/`python tests/…` invocation in `.github/workflows/*.yml` vs `tests/**/test_*.py`) and list are consistent with what this review saw from the inside:
- Confirmed by running them: `tests/test_sdk.py` (88 tests; 17 stale failures + one real-Lemonade test — F5) and `tests/test_agent_sdk.py` (8 tests, all skip without Lemonade after 31 s of probing) are exactly the shape of a file nobody runs.
- Confirmed by reading: `tests/integration/test_memory_integration.py`, `test_memory_api_integration.py`, `test_memory_eval.py` (155 tests) cover the memory subsystem end-to-end and are the tier that would have caught the consolidation/extraction findings in the memory sub-report (`sub-145-memory.md` 🔴/🟡) — none run in CI.
- Additional observation not in part3: `src/gaia/audio/tests/*.py` (3 files, 533 L) are outside `testpaths` and therefore outside *any* invocation, including local `pytest` — they belong on the same list (or should be deleted).
- The `hub/agents/*/python/tests` directories are run only by their own `release_agent_*.yml`/hub workflows (per part3); locally they are unrunnable without the package installed (F12).
No numbers were recomputed here; the 43 / ≈780 figures are the CI reviewer's and were not re-derived.

## Test gaps
- **Windows tier does not exist:** `test_unit.yml` is ubuntu-only, so F1/F2/F4/F11 (≈270 red tests on a Windows checkout) are invisible to CI while Windows/Ryzen AI is the product's primary platform.
- **No integration tier for `src/gaia/skills/` (12.3K lines)** and **`hub/agents/chat`** (the flagship's base class: 3 test files / 221 L against 4.9K L of source).
- **Eval baselines missing for 9 of 12 categories** (F9) — the "run the eval and compare" gate in CLAUDE.md is unenforceable for `memory`, `mcp_reliability`, `web_system`, `real_world`, etc.
- **No test pins the judge request shape** (`ClaudeClient` → `messages.create` kwargs) nor the `_sampling_kwargs` "no temperature" contract.
- **No test pins `compare_scorecards` precedence/exit semantics** for the masking cases in Task 5 (disappeared scenario, null current score, time-and-status regression on the same scenario).
- **`tests/test_lemonade_client.py` HTTP-body assertions** — `test_pull_model` and siblings stub responses but never assert the request payload (the #1655 class of bug), and the file is not in CI (CI reviewer).
- **Order-dependent tests** (`test_hotreload_*`, `test_sse_confirmation_gate`, `test_longthread_corpus_integrity`) — no `pytest-randomly` / isolation fixture for `_TOOL_REGISTRY`.
- **Real-state leakage**: `test_memory_discovery` credential-manager cases stub one scanner and run the rest against the host (Task 3).

## Documentation gaps
- `CLAUDE.md:405-414` and `docs/reference/troubleshooting.mdx:402` describe a baseline workflow (`tests/fixtures/eval_baselines/…` + `--save-baseline`) that the CLI does not implement; neither mentions that the default judge differs from every committed baseline (F8).
- `CLAUDE.md` "Run agent evals SERIALLY" implies a guard that is a no-op on Windows (F6); the runner's module comment only describes the POSIX behaviour.
- `docs/reference/dev.mdx` testing section does not say that `[ui]` is required for `tests/unit` to *collect* (F12) nor that `pytest-timeout` must be installed to copy CI commands (Task 4).
- `.claude/skills/gaia-testing/SKILL.md:116` repeats the `tests/fixtures/eval_baselines/<model>-<hash>/scorecard_<cat>.json` path without saying how such a directory is produced.
- `tests/test_sdk.py::TestSDKDocumentation` asserts `docs/SDK.md` exists — the SDK docs moved to `docs/sdk/*.mdx`; the test is the only "doc" still pointing at the old path (F5).

## Improvement opportunities
- Add a `windows-latest` job to `test_unit.yml` running `tests/unit` with `PYTHONUTF8=1` **unset** — it would have caught F1, F2, F4, F11 on the day they landed.
- Add `pytest-randomly` (or at least `-p no:randomly` opt-out) to surface the order dependencies found in Task 4.
- Session-scope the Lemonade reachability probe in `tests/conftest.py::require_lemonade` (31 s wasted per skipped file today).
- `tests/unit/conftest.py::_block_network`: allow loopback and log the blocked host in the error so the offending call is identifiable without a traceback.
- Add a "baseline drift" unit test (scenario ids in a baselined category ⊆ baseline entries) and a "never-run test file" guard (every `tests/**/test_*.py` referenced by some workflow or tagged `manual`) — both are cheap and close the two biggest process holes (F9, Task 6).
- `find_scenarios`: log user-local overrides of built-in scenario ids at WARNING, and include the override path in the scorecard `config` so a baseline run can be told from a customised one.
- `compare_scorecards`: treat `only_in_baseline` as an issue when the baseline entry was `PASS` (a passing scenario that vanished is the most common way a regression hides), and evaluate status change before time regression.
- Move `pytest-timeout`, `pytest-randomly` and `responses` into `[dev]` so local commands match CI.

## High-impact feature opportunities
- **Eval-as-CI on the self-hosted Strix Halo lane** — the runner already produces JUnit (`write_junit_xml`) and exit codes; wiring `gaia eval agent --category {rag_quality,tool_selection,context_retention} --compare <committed>` into a nightly job on the existing self-hosted Windows runners would make the CLAUDE.md eval rule enforced instead of aspirational, and F6/F8/F9 are the only blockers. Roughly: fix F6, implement `--save-baseline` to the fixture layout, one workflow (~1 day).
- **Baseline registry command** (`gaia eval baseline {list,save,diff}`) — resolves model/commit/judge, writes `meta.json`, and refuses to compare across judges; removes the hand-assembled fixture directories and the doc/CLI contradiction (F8).
- **Fixer sandbox** — run `--fix` in a `git worktree` with a diff summary and an explicit "apply" step; turns a dangerous flag into a reviewable one (F7) and makes the eval-driven fix loop usable by contributors, not just the maintainer.

## Checked and fine
- `pyproject` pytest config: `--strict-markers`, `asyncio_mode = "auto"`, `testpaths`, marker docs — coherent; no tracked `.pyc`/`__pycache__`; only 2 unconditional `pytest.mark.skip` in the whole tree.
- `build_scorecard`: FAIL score cap, judged-only denominators, unknown-status surfacing, `SKIPPED_NO_DOCUMENT` separation — all correct and tested (`tests/test_eval.py` 140 pass).
- `compare_scorecards`: judge-mismatch banner, per-scenario keying with `[WARN]` on missing ids, exit 2 on regressions — behaves as documented for the common path.
- `ClaudeClient`: no silent fallbacks; every failure path re-raises with context; retries are explicit and opt-in; dependency errors name the install command.
- `_acquire_eval_lock` on POSIX: NB flock, PID stamping, stale-PID reclaim, cleanup in `finally` — correct.
- `find_scenarios`: schema-validates every YAML and raises `RuntimeError` on parse failure (fail-loud); persona list enforced.
- `tests/unit/test_memory_store.py`, `test_memory_mixin.py`, `test_goal_store.py`, `test_turn_metrics*.py`, `tests/unit/mcp`, `tests/unit/factory`, `tests/unit/rag`: pass on Windows with no environment dependencies.
- `hub/agents/gaia/python/tests/test_session_registry.py`: 32/33 pass; the one failure needs `gaia_agent_chat` installed (environment).

## Hypotheses (unverified)
- The 8 `'ProactorEventLoop' object has no attribute '_ssock'` errors in run #1 are the same root cause as F1 (self-pipe creation aborted mid-`__init__`), not a second bug — not traced.
- `tests/unit/connectors` 33 failures in run #2 (vs 22 in run #1) were not individually root-caused beyond the F1 storm; some may be genuine Windows path/URL-encoding bugs in the connectors layer.
- `eval/prompts/fixer.md` (if present in the repo) may contain guidance that partially mitigates F7 (e.g. "only touch these files"); the prompt template was not read.
- `_common_prefix`-style cache-hit metrics in `turn_metrics.py` are recorded by `runner.py` performance aggregation only if the sidecar reports them; whether any eval scenario exercises `GAIA_TURN_LOG` was not checked.
- The `tests/unit/connectors` router tests may pass on Windows under `asyncio_mode=strict` with a `SelectorEventLoop` policy; not tried.

## Summary (5 lines)
1. On a clean Windows checkout the unit suite is red for **environmental** reasons — ≈270 tests error/fail from a loopback-blocking network guard (F1), a cp1252 test helper (F2), a CRLF installer (F4) and POSIX-only assertions (F11) — none visible to ubuntu-only CI.
2. `tests/test_sdk.py` asserts an SDK that no longer exists (17 failures, one real-Lemonade "unit" test) and is one of ≈43 test files no workflow runs (F5, Task 6, cross-checked with the CI reviewer).
3. The eval framework's *scoring* is sound, but its *process* is not: the serial guard is a no-op on Windows (F6), `--fix` runs Claude with permissions off on the live tree (F7), and the documented baseline workflow is not what the CLI implements — default judge ≠ every committed baseline, `--save-baseline` writes elsewhere, 9/12 categories have no baseline (F8, F9).
4. Test quality is otherwise good — real-SQLite store tests, 316 `importorskip` guards, 396 payload-asserting mocks vs 124 existence-only — with hot spots in `test_init_command.py` and the never-run `test_lemonade_client.py` HTTP stubs.
5. Cheapest high-leverage fixes: a Windows unit job, loopback-allowing guard, `encoding="utf-8"` in two helpers, `*.sh text eol=lf`, and a baseline-drift + never-run-file guard test.

