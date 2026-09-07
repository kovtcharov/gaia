# Reviewer 08 — Hub client, Email agent, teaching templates, skills

Commit reviewed: 211f08c5 (HEAD of main). Python: `.venv\Scripts\python.exe` (CPython 3.13.11, uv venv).

## Scope covered

Read fully, line by line:
- `src/gaia/hub/installer.py` (1645 lines) — download/verify/extract/backup/rollback/uninstall/setup executor
- `src/gaia/hub/catalog.py` (598 lines) — index fetch, TTL + disk cache, semver, registry merge

Read via preview only / skimmed headers (interrupted before full read — NOT reviewed end to end):
- `src/gaia/hub/manifest.py`, `native_launcher.py`, `lifecycle.py`, `compatibility.py`, `packager.py`, `publisher.py`

Grep-level checks only (targeted, not a full read):
- `hub/agents/email/python/gaia_agent_email/outlook_query.py` (#3269 follow-ups), `CHANGELOG.md`, `except`/token-logging sweep
- `hub/agents/{hello-world,word-count,connectors-demo}/python/` — real import + instantiation with the venv, README vs `agent.py`
- `hub/skills/*/SKILL.md` front matter (`tools_required`, `permissions`) vs `@tool` defs; `hub/skills/README.md`

Ran (full output in `.review/_08_pytest_hub.log`, `.review/_08_pytest_email.log`):
- hub unit suite (11 files), email unit suite (8 files + `tests/unit/email/`), `tests/unit/test_starter_skills.py`

Checked against GitHub: PRs #3232, #3299 (both merged, `website/`-only — `website/src/lib/catalog.ts` / `InstallCard.astro`;
nothing in the Python hub client is affected, so "does HEAD still have the problem" is N/A for `src/gaia/hub/`) and #3269
(merged; the Outlook grouped-duration fix IS at HEAD — `outlook_query.py:47-52` stops at `)}]` — but both follow-ups its
review asked for are not, see findings below).

NOT covered (interrupted): the email agent's `agent.py`, Gmail/Outlook MIME parsing, sidecar client/relay/proxy/router
source, and the README/SPEC/SKILL cross-check for the email agent beyond the CHANGELOG grep.

## Findings

### [🟡] Wheel install writes the `.installed` sentinel before the `.pth` step, so a `.pth` failure leaves an agent that is simultaneously "installed" and "failed"
- **Where:** `src/gaia/hub/installer.py:1142-1160` (`install`), `:1219-1241` (`_write_sentinel`), `:797-828` (`_add_wheel_agent_to_active_env_path`)
- **What:** For a wheel artifact, `install()` writes the sentinel and *then* writes the `gaia-hub-agents.pth` entry into the active interpreter's site-packages. `_add_wheel_agent_to_active_env_path` deliberately raises `InstallError` on `OSError` (fail-loudly), but by then the sentinel already exists. On a fresh install there is no backup, so `_restore_backup_if_present` is a no-op and the install dir + sentinel survive the raise.
- **Failure scenario:** A user whose active Python `purelib` is not writable (system-site Python on Linux/macOS, a locked-down venv, read-only conda env) installs a wheel agent. `install()` raises, progress is recorded `status="failed"`, the UI shows an error — yet `list_installed()` / `installed_versions()` / the catalog now report the agent as `installed` at that version. Every retry takes the `updated=True` path (snapshot → install → `.pth` raise → restore backup), so it never self-heals; the agent is listed but is not importable from any new process (the very thing #2358's `.pth` mechanism exists to guarantee).
- **Evidence:**
  ```
  1142:                _write_sentinel(
  ...
  1156:                if artifact_kind == ARTIFACT_KIND_WHEEL:
  1157:                    _add_wheel_agent_to_active_env_path(
  ...
  1161:            except Exception:
  1164:                _restore_backup_if_present(agent_id, root)
  1165:                raise
  ```
  `_restore_backup_if_present` (`:1374-1382`) returns early when `.backup/<id>` is absent — nothing removes the freshly written install dir. `tests/unit/test_hub_installer.py` has no test where `_add_wheel_agent_to_active_env_path` raises after the sentinel is written (grep for `PermissionError` only hits the binary-write and uninstall cases at lines 1074/1121).
- **Fix:** Move the `.pth` write before `_write_sentinel` (the sentinel is the commit marker and should be written last), or on any exception in the fresh-install path `rmtree(install_dir)` when no backup existed. Add a unit test that injects a non-writable `active_env_site_packages` and asserts `read_sentinel()` is `None` after the raise.
- **Confidence:** High
- **Tracked:** none found

### [🟡] Hub/email unit tests hard-fail on Windows: the unit-test network guard blocks `socket.socketpair()`, which `ProactorEventLoop` needs — 35 hub + 18 email tests fail at HEAD on this platform
- **Where:** `tests/unit/conftest.py:56-67` (`_block_network`); every `TestClient`-based test in `tests/unit/test_hub_router.py`, `tests/unit/test_email_sidecar_router.py`, `tests/unit/test_email_sidecar_server_wiring.py`; one real-socket test in `tests/unit/test_email_sidecar_proxy.py`
- **What:** The autouse guard raises `ConnectionError("Unit tests must not make real network connections …")` for any socket connect. On Windows `socket.socketpair()` is emulated via a real loopback connect (`socket.py:623 _fallback_socketpair`), and asyncio's `ProactorEventLoop.__init__` calls it — so every Starlette `TestClient` request dies in the guard before the route runs. The tests are not marked `@pytest.mark.allow_network`, and the guard doesn't whitelist loopback.
- **Failure scenario:** `.venv\Scripts\python.exe -m pytest tests/unit/test_hub_router.py` on Windows → 35/35 router tests fail with `ConnectionError`; a developer cannot use the router suites as a local regression gate, and a Windows CI lane would be red. PR #3269's test plan already notes needing "the repository's `allow_network` test override because Windows blocks socketpair" — known to contributors, not fixed in the fixture.
- **Evidence:** `.review/_08_pytest_hub.log` lines 943-949:
  ```
  self._ssock, self._csock = socket.socketpair()
  ..\Lib\socket.py:623: in _fallback_socketpair
  E   ConnectionError: Unit tests must not make real network connections (mark the test @pytest.mark.allow_network if it uses a local socket)
  ```
  Failing set (hub, 35 of 37): every test in `test_hub_router.py` — `test_catalog_returns_merged_list`, `test_catalog_not_swallowed_by_agents_route`, `test_catalog_offline_503_when_no_cache`, `test_install_returns_202_and_schedules`, `test_install_non_verified_python_refused_403`, `test_install_native_non_verified_allowed_with_trust`, `test_install_duplicate_returns_409`, `test_install_requires_ui_header`, `test_install_status_polling`, `test_install_status_unknown_404`, `test_uninstall_success`, `test_uninstall_builtin_refused`, `test_uninstall_not_installed_404`, `test_uninstall_requires_ui_header`, `test_rollback_success`, `test_rollback_no_backup_400`, `test_set_config_success`, `test_set_config_replace_flag`, `test_set_config_requires_ui_header`, `test_get_config`, `test_health_endpoint`, `test_status_endpoint`, `test_setup_returns_202_and_schedules`, `test_setup_empty_ids_400`, `test_setup_requires_ui_header`, `test_setup_status_polling`, `test_setup_status_unknown_404`, `test_setup_status_not_swallowed_by_agents_route`, `test_install_error_surfaces_via_install_status`, `test_email_install_shuts_down_running_sidecar_first`, `test_email_uninstall_shuts_down_running_sidecar_first`, `test_email_rollback_shuts_down_running_sidecar_first`, `test_email_install_proceeds_when_stop_sidecar_noops`, `test_email_install_aborts_when_stop_fails`, `test_non_email_install_does_not_shutdown_sidecar`.
  Failing set (email, 18 of 18): `test_email_sidecar_proxy.py::test_cross_thread_close_unblocks_parked_read_promptly_real_socket`; all 15 tests in `test_email_sidecar_router.py` (`test_each_request_acquires_a_fresh_handle`, `test_health_and_version_proxied`, `test_init_ready_200_body_passthrough`, `test_init_not_ready_503_body_passthrough`, `test_init_post_streams_provisioning_output_200`, `test_init_post_503_unreachable_streamed_body_passthrough`, `test_init_post_sidecar_http_error_passthrough`, `test_prescan_route_forwards_and_preserves_card_envelope`, `test_search_and_archive_routes_mounted`, `test_calendar_events_get_forwards_query_params`, `test_sidecar_http_error_status_and_detail_passthrough`, `test_start_failure_returns_503_with_remedy`, `test_acquire_failure_returns_503`, `test_connection_error_surfaces_as_503_with_actionable_detail`, `test_connector_routes_not_exposed`); `test_email_sidecar_server_wiring.py::test_sidecar_router_mounted_dev_mode`, `::test_sidecar_router_mounted_flag_unset_defaults_to_user_mode`.
  All of these are **environment/test-infra failures, not product regressions** — the route code is never reached.
- **Fix:** In `_block_network`, allow loopback connects (`127.0.0.1` / `::1`) or let `socket.socketpair` through (wrap it to bypass the guard), or set `WindowsSelectorEventLoopPolicy` for the unit session. Marking the `TestClient` suites `allow_network` also works but the fixture fix covers every future `TestClient` test.
- **Confidence:** High
- **Tracked:** none found (searched "socketpair")

### [🟡] Email agent CHANGELOG does not record the Outlook grouped-duration fix (#3234 / PR #3269) — the review on that PR flagged it as 🟡 and it was merged without it
- **Where:** `hub/agents/email/python/CHANGELOG.md` (no entry); fix lives at `hub/agents/email/python/gaia_agent_email/outlook_query.py:47-52`
- **What:** CLAUDE.md requires a hub agent's user-visible behaviour change to be named in `CHANGELOG.md`, and the `[Unreleased] → Fixed` section already records neighbouring query-translation fixes. The PR review supplied the exact entry text; it was never added.
- **Failure scenario:** An Outlook user whose `(newer_than:7d)` search was failing has no release note telling them it landed; the next package version ships with a silent behaviour change.
- **Evidence:** `grep -n "3234\|grouping paren\|grouped" hub/agents/email/python/CHANGELOG.md` → no matches. The code change is present:
  ```
  47:_DURATION_RE = re.compile(
  48:    # Stop at grouping punctuation so `(newer_than:7d)` validates `7d`, not
  49:    # `7d)`, just like the Gmail query normalizer.
  50:    r'\b(?P<op>newer_than|older_than):(?P<val>"[^"]*"|[^\s)}\]]+)',
  ```
- **Fix:** Add the entry the PR review drafted under `## [Unreleased] → ### Fixed`.
- **Confidence:** High
- **Tracked:** none found (PR #3269 review comment only)

### [🟢] Two `test_hub_installer.py` tests fail in a uv-created venv because they shell out to real `pip`, which uv venvs don't ship
- **Where:** `tests/unit/test_hub_installer.py::test_install_real_wheel_lands_in_site_packages_and_is_importable`, `::test_default_run_pip_falls_back_to_python_pip_when_uv_missing` (line 1590); `src/gaia/hub/installer.py:401-442` (`_default_run_pip`)
- **What:** Both run the real `python -m pip install --target …` frontend. A uv-managed venv has no `pip` module, so the last frontend fails and `_default_run_pip` raises. PR #3232's body reports the same failures ("`No module named pip` inside the uv venv, unrelated to this diff") — known-flaky in the documented dev setup (`uv venv && uv pip install -e ".[dev]"`).
- **Failure scenario:** Any contributor following the documented setup sees 2 red tests on a clean `main`; the tests cannot distinguish "the fallback chain is broken" from "this box lacks pip".
- **Evidence:** log lines 25-28:
  ```
  E   gaia.hub.installer.InstallError: Could not install the agent's Python package -- every pip frontend failed:
  E     - uv (not found on PATH)
  E     - …\.venv\Scripts\python.exe (not found on PATH)
  E     - …\.venv\Scripts\python.exe -m pip install (exit 1): …python.exe: No module named pip
  ```
  Second bullet is also a misleading message: `attempts.append(f"{frontend[0]} (not found on PATH)")` (`:429`) names only argv[0], so the `python -m uv` frontend is reported as "`python.exe` (not found on PATH)" even though `sys.executable` obviously exists.
- **Fix:** `pytest.importorskip("pip")` / skip when neither `uv` nor `pip` is available, or drive the test through an injected `run_pip`. Print `' '.join(frontend)` in the `FileNotFoundError` branch label.
- **Confidence:** High
- **Tracked:** none found as an issue (mentioned in PR #3232's body only)

### [🟢] Outlook duration grammar is still a copy of `gmail_query.DURATION_OP_RE` — the drift that caused #3234 can recur
- **Where:** `hub/agents/email/python/gaia_agent_email/outlook_query.py:40,47-52`; `gmail_query.py:38`
- **What:** PR #3269's review asked to import the shared pattern instead of copying it. At HEAD the module imports only `parse_gmail_duration_value` and re-declares `_DURATION_RE` with a comment saying it must match the Gmail one.
- **Failure scenario:** The next terminator/grammar change lands on one path and not the other — exactly how #3234 happened.
- **Evidence:**
  ```
  40:from gaia_agent_email.gmail_query import parse_gmail_duration_value
  47:_DURATION_RE = re.compile(
  ```
  vs `gmail_query.py:38: DURATION_OP_RE = re.compile(`.
- **Fix:** `from gaia_agent_email.gmail_query import DURATION_OP_RE, parse_gmail_duration_value` and `_DURATION_RE = DURATION_OP_RE`.
- **Confidence:** High
- **Tracked:** none found

### [🟢] `_hot_register` puts the per-agent `site-packages` at `sys.path[0]`, ahead of the active venv
- **Where:** `src/gaia/hub/installer.py:881-885` (`_hot_register`); `:462` (`run_pip` without `--no-deps`)
- **What:** `run_pip(["--target", site_packages, wheel])` installs the agent's full dependency closure under `~/.gaia/agents/<id>/site-packages`. `_hot_register` then does `sys.path.insert(0, sp)`, so any dependency not yet imported by the UI server resolves from the agent's private copy in preference to the venv's pinned version. The `.pth` path used by later processes appends to the *end* of `sys.path` — the two mechanisms give different precedence for the same package.
- **Failure scenario:** `gaia chat --ui` installs a hub wheel whose closure pins an older `requests`/`pydantic`; the server's next lazy import of that package gets the agent's copy, while a fresh `gaia` process gets the venv's — behaviour differs between "just installed" and "after restart", the exact class of bug hot-register exists to prevent.
- **Evidence:**
  ```
  883:        sp = str(site_packages)
  884:        if sp not in sys.path:
  885:            sys.path.insert(0, sp)
  ```
- **Fix:** `sys.path.append(sp)` (matching `.pth` semantics), or `--no-deps` on the `--target` install with a loud error when a required dep is absent from the venv.
- **Confidence:** Medium (precedence verified from code; no concrete package clash reproduced)
- **Tracked:** none found

### [🟢] The single-install guard is process-local; the CLI and the UI server can race on the same install dir
- **Where:** `src/gaia/hub/installer.py:308-330` (`_install_slot`, `_IN_PROGRESS`); docstring `:25-26` promises "One install per id"
- **What:** `_IN_PROGRESS` is an in-memory set. A `gaia hub install <id>` in a shell while the Agent UI server installs the same id runs both bodies concurrently: both call `_snapshot_backup` (`shutil.rmtree(backup)` + `shutil.move` of the same dir, `:731-734`), so the second raises a raw `FileNotFoundError` instead of the documented `InstallInProgressError`, and both write into `install_dir`.
- **Failure scenario:** Second installer surfaces a traceback rather than a 409; interleaved `rmtree(backup)`/`move` can leave no usable backup for `rollback`.
- **Evidence:** `:314-320` (`if agent_id in _IN_PROGRESS: raise …` under a `threading.Lock`) — no cross-process lock.
- **Fix:** `O_EXCL` lock file under `install_root/.locks/<id>` around the body, mapped to `InstallInProgressError`.
- **Confidence:** High for the code shape; Medium for user impact (needs two concurrent installers)
- **Tracked:** none found

### [🟢] `_write_agent_yaml` swallows every fetch error and installs without the package's `gaia-agent.yaml`
- **Where:** `src/gaia/hub/installer.py:1257-1262`
- **What:** Documented "best-effort", but the yaml is what `lifecycle`/registry read for identity/permissions; a hub that 404s it yields an install with no manifest and only a `logger.warning`. It's the same origin the artifact just came from, so a failure is a hub inconsistency the caller should see, not only the log.
- **Evidence:**
  ```
  1259:    except Exception as exc:  # noqa: BLE001 - optional asset
  1260:        logger.warning("installer: could not fetch %s: %s", url, exc)
  1261:        return
  ```
- **Fix:** Narrow to transport errors and either fail the install or add a `warnings` field to `InstallResult`/progress so the UI shows it.
- **Confidence:** Medium (convention call; behaviour verified)
- **Tracked:** none found

## Test results (exact)

- Hub suite (11 files): **37 failed, 328 passed, 4 skipped** (`.review/_08_pytest_hub.log`). 35 failures = Windows `socketpair` guard (finding 2); 2 = missing `pip` in the uv venv (finding 4). **No failure traced to a product regression at HEAD.**
- Email suite: **18 failed, 270 passed, 19 skipped** (`.review/_08_pytest_email.log`). All 18 = the same `socketpair` guard. **No product regression identified.** `tests/unit/email/*` (corpus integrity, gmail client, triage heuristics, phishing precision, text signals, attachments) all passed.
- `tests/unit/test_starter_skills.py`: **143 passed, 27 skipped** (skips are the Lemonade-backed "load into an agent" cases).

## Skill→tool cross-check table

Source of truth: `tools_required` in each SKILL.md front matter, checked against `def <name>(` under `@tool` in
`src/gaia/agents/{tools,base}/`, `hub/agents/gaia/python`, `hub/agents/chat/python`, `hub/skills/*/tools.py`.
Every declared tool resolves; `test_starter_skill_tools_required_are_real_tools` enforces this against a live
registry and passes.

| Skill | `tools_required` | Exists? (file) | Permissions / connectors |
|---|---|---|---|
| check-in | recall, search_past_conversations, remember, update_memory | Y — all `src/gaia/agents/base/memory.py` | none |
| coding | read_file, edit_file, search_file_content, search_code_index, execute_python_file | Y — `file_io_tools.py`, `file_io_tools.py`, `file_tools.py`, `code_index_tools.py`, `hub/agents/chat/…/agent.py` | `shell:execute:pytest` |
| daily-brief | search_web, fetch_page, recall | Y — `browser_tools.py` (also a `search_web` in chat `agent.py`), `browser_tools.py`, `memory.py` | `network:read` |
| data-explore | create_table, insert_data, query_data, list_tables | Y — all `scratchpad_tools.py` | none |
| document-brief | index_document, index_directory, list_indexed_documents, query_documents, summarize_document, rag_status | Y — all `rag_tools.py` | none |
| file-ops | read_file, write_file, edit_file, find_files, search_file_content, get_file_info, request_user_input | Y — `file_io_tools.py` ×3, `filesystem_tools.py`, `file_tools.py` ×2, chat `agent.py` | none |
| github-triage | run_shell_command | Y — `shell_tools.py` | `shell:execute:gh`; uses the `gh` CLI's own auth, explicitly **no** GAIA connector (SKILL.md:22) |
| price-watch | fetch_page, recall, remember | Y | `network:read` |
| recommendations | recall, search_web, fetch_page, remember | Y | `network:read` |
| research-report | search_web, fetch_page, write_file, index_document, query_documents | Y | `network:read` |
| rss-digest | (declares its own `tools:` — `fetch_rss`) | Y — `hub/skills/rss-digest/tools.py`; `test_rss_digest_registers_its_declared_tool` + 10 `fetch_rss` parser tests | `network:read`; `requirements.python >=3.10` |
| source-watch | fetch_page, recall, remember | Y | `network:read` |
| summarize | index_document, summarize_document, list_indexed_documents, dump_document, read_file | Y — `rag_tools.py` ×4, `file_io_tools.py`/`filesystem_tools.py` | none |

Notes:
- `hub/skills/README.md` says "thirteen worked examples" — there are 13 skill directories (matches).
- No skill references a connector or MCP server; `github-triage` deliberately routes through the `gh` binary, so the "connector that doesn't exist" check is vacuous. `src/gaia/connectors/` ships `google`/`github` providers (used by `connectors-demo`, not by skills).
- Name collisions worth knowing: `read_file` is defined in both `file_io_tools.py` and `filesystem_tools.py`; `search_web` in both `browser_tools.py` and chat `agent.py`. Which one a skill gets depends on the composing agent's MRO — not a bug, but the SKILL.md bodies assume the `file_io` semantics (edit + write present).
- Skill "tests": per skill there is schema validation, byte-identical round-trip, provenance, publishability, permission-catalog resolution, `tools_required` reality check, body-mentions-declared-tools, memory-category and scratchpad-prefix lint (`test_starter_skills.py:64-300`). Only `rss-digest` has behaviour tests of its tool. **No skill has a test that runs its procedure against an agent** (the `pack_manager` load-into-agent cases are Lemonade-gated and skipped here).

## Teaching templates (hello-world, word-count, connectors-demo)

Real import + instantiation with the venv (`PYTHONPATH=hub/agents/<id>/python`): all three `agent.py` modules import and
`HelloWorldAgent()`, `WordCountAgent()`, `ConnectorsDemoAgent()` construct against the current base `Agent`
(`model_id` resolves to `Gemma-4-E4B-it-GGUF`; the only output was the expected "Lemonade not running" notice).
- hello-world: README's "four things every agent needs" match the numbered docstring in `agent.py:12-16`. OK.
- word-count: README names `count_text` (the `@tool`) and `count_text_stats` (the pure helper) — both exist (`agent.py:47`, `:96`). OK.
- connectors-demo: code registers four tools (`calendar_today`, `drive_recent_files`, `github_my_repos`, `gmail_recent_subjects`) and `REQUIRED_CONNECTORS` → `required_connections` in `__init__.py:72`. The README (only `## Install` and `## Develop / test` sections) never names the tools or the connector ids the agent needs granted — a user can't tell from the README what `gaia connectors` grant to run. Doc gap, listed below.
- Each template ships a `tests/` dir (`test_hello_world.py`, `test_word_count.py`, `test_connectors_demo.py`); not executed here.

## Test gaps

- `installer.install()` failure *after* the sentinel is written (`.pth` write raising) — no test; the current suite only exercises `PermissionError` on the binary write and on `uninstall` (finding 1).
- `test_hub_router.py` (all 35 tests) and the three email sidecar suites are unrunnable on Windows without touching the fixture — effectively zero router coverage on that platform (finding 2).
- `_default_run_pip` real-subprocess tests depend on host `pip`/`uv` presence instead of injecting a runner (finding 4).
- Cross-process install race — no test (finding 7); `_install_slot` is only tested in-process (`test_install_duplicate_returns_409`).
- `_install_cpp_artifact`: `test_hub_security.py` exists (not read in full); no test observed for a zip member whose `external_attr` marks a symlink, nor for a tar member with an absolute Windows path (`C:\…`) — the `_assert_member_within` logic handles both by inspection (see Checked and fine) but the tests weren't confirmed.
- Skills: no per-skill behavioural test beyond `rss-digest`; the procedure bodies are only linted.

## Documentation gaps

- `hub/agents/email/python/CHANGELOG.md` — missing the #3234 Outlook grouped-duration fix entry (finding 3). Per CLAUDE.md the same claim usually needs checking in README/SPEC/SKILL; not verified here (interrupted).
- `hub/agents/connectors-demo/python/README.md` — does not list the four tools or the `google`/`github` connector grants the agent requires; the `gaia-agent.yaml` only carries them as `tags`.
- `src/gaia/hub/installer.py` module docstring (`:25-26`) says "One install per id — a re-entrant install … raises `InstallInProgressError`"; true only within one process (finding 7). Worth qualifying.
- PRs #3232/#3299 fixed the *website's* rendering of app/component entries with `npm_package`; the Python `catalog.merge_with_registry` (`catalog.py:439-488`) still emits every non-skill entry — including `type: "app"`/`"component"` — with `status: available` and no install-method hint, so the Agent UI's "Install" affordance for those lanes depends on the frontend re-deriving the lane (same drift the website had). Flagged as a hypothesis below since the UI side wasn't read.

## Improvement opportunities

- Write the sentinel last in `install()` (commit-marker semantics) — closes finding 1 structurally instead of per-step cleanup.
- Replace `_default_run_pip`'s argv[0]-only "not found" label with the full frontend string — the current error blames `python.exe`.
- Add a `--no-deps` / dependency-policy decision for `--target` installs; today every hub wheel duplicates its closure (and possibly `amd-gaia` itself) under `~/.gaia/agents/<id>/site-packages`.
- `catalog._parse_version` treats `1.0.0-rc.1` vs `1.0.0-beta` by string compare of the prerelease tag; fine for the "is newer" use, but `compare_versions` is public — note the limitation in its docstring.
- `_block_network` guard: allow loopback so `TestClient`/`socketpair` work on Windows; this unblocks ~50 tests for every Windows contributor.
- Import `DURATION_OP_RE` in `outlook_query.py` (finding 5).

## High-impact feature opportunities

- **Artifact signing / provenance beyond SHA-256.** `_download_and_verify` compares against a checksum served by the *same* origin as the artifact (`installer.py:382-393`, manifest from `catalog.fetch_manifest`). A compromised or MITM'd `GAIA_HUB_URL` can serve a matching pair. The trust-tier gate (`ensure_trust_ack`) is the only other defence and it's a manifest field from that same origin. A detached signature over `manifest.json` (public key pinned in the client) would make `verified` mean something cryptographic. Cost: a signing step in the hub Worker's publish path + one verify call here.
- **Cross-process install lock + crash-recovery sweep.** `~/.gaia/agents/<id>/` has no lock file and no "in-progress" marker; a partial dir with no sentinel is invisible to `list_installed` and silently reused by the next install. A tiny lock/marker protocol would make `gaia hub install`, the UI, and `run_setup` safe together.

## Checked and fine

- **Zip-slip / tar traversal** (`_install_cpp_artifact`, `_assert_member_within`, `installer.py:466-537`): every member is resolved against `install_dir.resolve()` and refused unless `is_relative_to(base)`; absolute POSIX (`/etc/x`) and Windows (`C:\x`) member names both resolve outside base and are refused; tar sym/hard links and non-regular members refused; zip symlinks detected via `external_attr >> 16`; extraction is per-member (no `extractall`). Validation happens before any write.
- **Checksum enforced, not optional**: `_resolve_version` raises if `sha256` is missing (`:711-715`); `_download_and_verify` raises `ChecksumError` on mismatch before anything touches the install dir.
- **Agent id → path safety**: `_require_safe_agent_id` (`:152-173`) restricts to one path component; used by `agent_install_dir`, `_backup_dir`, so `rmtree`/`move` in `uninstall`/`rollback` cannot escape `install_root`. `list_installed` skips non-matching dirs. `read_sentinel` rejects an `executable` containing separators.
- **Artifact filename safety**: `_sanitize_artifact_filename` refuses `/`, `\`, `..`, empty (`:603-610`) before the temp-file write.
- **Uninstall of builtins refused** (`is_builtin`, `:1288-1291`); `uninstall` raises `NotInstalledError` when no sentinel; `.pth` line removed for wheel kinds; backup discarded.
- **Binary write is atomic** (`mkstemp` + `os.replace`, `:553-558`), exec bit set on POSIX; rollback explicitly unsupported for binary kinds with an actionable message.
- **Trust gate** applied both at the router and inside `install()` (`ensure_trust_ack`); default tier `experimental` when absent (least privilege).
- **Catalog offline behaviour**: `load_index` fails loudly (`CatalogError`) when neither network nor disk cache; `build_catalog` degrades to registry-only with `offline=True` flagged, never hidden. Skills lane filtered out of the agent list (`agent_entries`) so no broken install button.
- **No `except Exception: pass`** in `hub/agents/email/python/gaia_agent_email/` (grep count 0); the bare `pass` handlers found are `asyncio.CancelledError` and narrow `ValueError/TypeError` cases. No `access_token`/`refresh_token` reaches a logger/print in that package (grep).
- **Prompt-injection delimiting exists** in the email agent: `body_normalize.py`, `tools/llm_triage.py`, `read_tools.py`, `summarize_tools.py`, `thread_fold.py`, `calendar_tools.py` all contain untrusted-content framing strings (not read in depth).
- **#3232 / #3299**: website-only; `src/gaia/hub/` unaffected at HEAD.
- **#3269 code fix is present** at `outlook_query.py:47-52`.
- Teaching templates import and instantiate against the current base `Agent` API.

## Hypotheses (unverified)

- The Agent UI's Hub panel may still offer "Install" for `type: app`/`component` catalog entries because `merge_with_registry` emits them with `status: available` (the Python analogue of #3231/#3298). Needs a read of `src/gaia/apps/webui` hub components.
- `manifest.py`, `native_launcher.py`, `lifecycle.py`, `compatibility.py`, `packager.py`, `publisher.py` were not read; the `native_launcher` command-injection check (agent id / binary path interpolated into a shell) is therefore **unverified** — the installer side only ever launches `subprocess.run([...])` argv lists, never a shell string.
- On Python ≥3.12, `tf.extract(member, install_dir)` without `filter=` emits `DeprecationWarning`, and 3.14 switches the default to the `data` filter, which additionally strips setuid/world-writable bits — behaviour of `_install_cpp_artifact` will change slightly across interpreter versions. Passing `filter="data"` explicitly would pin it.
- Email README/SPEC/SKILL may all be silent on the Outlook grouped-duration behaviour; only CHANGELOG was grepped.
- A partial, sentinel-less install dir left by a failed *fresh* install (e.g. pip failure) is reused by the next attempt without cleanup (`install_dir.mkdir(exist_ok=True)`, `:1125`); harmless for pip `--target` (overwrites), but stale files from a previous version's cpp archive could survive. Not reproduced.
