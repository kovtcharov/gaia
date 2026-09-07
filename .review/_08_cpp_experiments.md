# Sub-review 08 — `cpp/`, `experiments/whatsapp-webjs/`, `skills/community/`

Reviewer scope: C++ agent framework, the WhatsApp experiment, and the community-skills README.
Commit 211f08c5. READ-ONLY; nothing built in-tree.

## Scope covered

Read in full:
- `cpp/README.md`, `cpp/vcpkg.json`, `cpp/third_party/sqlite/README.md`, `SQLITE_VERSION.txt`
- `docs/plans/cpp-framework-parity.md` (current-state table, dead-code note, all phases, testing, risks), `docs/plans/cpp-webui-integration.md` (skimmed for the capability matrix only)
- `cpp/CMakeLists.txt` (header + packaging section; dependency pins grepped)
- `cpp/src/http_client.cpp` (547 lines, full), `cpp/src/lemonade_client.cpp` (434, full), `cpp/src/sse_parser.cpp` (149, full)
- `.github/workflows/{build_cpp,build_agents,benchmark_cpp}.yml` (grepped for cmake/ctest steps)
- `setup.py` / `MANIFEST.in` / `pyproject.toml` (grepped for `cpp`), `src/gaia/**` (grepped for `cpp/`, ctypes, cffi, pybind)
- `experiments/whatsapp-webjs/index.js`, `skills/community/README.md`, `hub/skills/README.md`, the `skill` subparser in `src/gaia/cli.py` (grep-level, see below)

Skipped / grep-only (honest gaps): `cpp/src/agent.cpp` (1459 lines), `mcp_client.cpp` (804), `process.cpp` (1102), `file_tools.cpp`, `skill.cpp`, `database.cpp`, `vector_index.cpp`, `chunking.cpp`, all of `cpp/tests/`, `cpp/agents/*`, `cpp/examples/*`, `cpp/benchmarks/`. Memory-safety review of those files is grep-sampled only; anything not quoted below is unverified and lives in Hypotheses.

Line counts (`wc -l`, excluding `third_party/`): `cpp/` is **47,477 lines** total (src+include = 19,943; the rest is tests, agents, examples, docs). The "~60K" figure in the task brief only holds if the vendored SQLite amalgamation (283,998 lines) is counted, which it should not be.

## Architecture + parity table

Modules under `cpp/include/gaia/` → `cpp/src/`: `agent` (loop state machine), `tool_registry` (registration, fuzzy resolution, schema validation, ALLOW/CONFIRM/DENY policies), `http_client` (pimpl over cpp-httplib), `lemonade_client` (OpenAI `/chat/completions` blocking + SSE streaming, `/health`, `/load`, `/models`), `sse_parser`, `mcp_client` (stdio JSON-RPC, Win32+POSIX), `mcp_registry` (`~/.gaia/mcp.json`), `json_utils`, `console`/`clean_console`/`tui_*` (FTXUI), `json_event_handler`, `database` (vendored SQLite + FTS5), `vector_index` (flat index), `chunking`, `file_tools`, `git_tools`, `process`, `ignore` (.gitignore), `image`, `security`, `session`, `skill`/`skill_yaml`/`skill_sets`, `model_registry`, `repl`. Consumers: `cpp/agents/{bash,health,process,security-demo,vlm,wifi}` + `cpp/examples/*.cpp`, packaged by `cpp/packaging/package_agents.py`.

Parity, judged from code present in the tree (not the README):

| Python feature (`src/gaia/`) | C++ equivalent | Status |
|---|---|---|
| Agent loop (`agents/base/agent.py`) | `src/agent.cpp` | complete (multi-step, error recovery, cancellation) |
| Tool registry / `@tool` | `tool_registry.cpp` | complete |
| LLM client — Lemonade `/api/v1` | `lemonade_client.cpp` over `http_client.cpp` | complete |
| OpenAI-compatible `/v1` (Ollama, #773) | `LemonadeClient::normalizeUrl` + `ensureModelLoaded` skip | complete for chat; model mgmt endpoints intentionally skipped |
| Native `tools`/`tool_calls` (streaming + parallel) | `sse_parser.cpp::accumulateToolCalls`, `test_native_tool_calls.cpp` | complete |
| Claude / OpenAI cloud providers (`llm/providers/`) | none | missing (README says "planned") |
| MCP client — stdio | `mcp_client.cpp` | complete |
| MCP client — HTTP/streamable-HTTP transport | none found (`grep -l HttpTransport cpp/` empty) | missing (plan P4.1) |
| MCP server registry (`~/.gaia/mcp.json`) | `mcp_registry.cpp` | complete |
| RAG SDK (`rag/sdk.py`) | `chunking.cpp` + `vector_index.cpp` only; no `rag.h`, no embeddings call | partial (plan P2.2 not landed) |
| Code index (`code_index/`) | none | missing (plan P2.3) |
| Embeddings API | none in `lemonade_client.cpp` (no `/embeddings` path) | missing (plan P1.2) |
| `SKILL.md` parse/validate/sets | `skill.cpp`, `skill_yaml.cpp`, `skill_sets.cpp` | complete for parse + sets |
| SkillManager discovery + `Agent::loadSkill` | plan P3.3 — no `skill_manager.h` in tree | missing; `loadSkillSet()` resolves but registers nothing (per plan) |
| Memory / SQLite persistence (`memory/`) | `database.cpp` (RAII, migrations, FTS5) | partial — library exists, no consumer |
| Sessions (`ui/` sessions) | `session.cpp` (JSON under `~/.gaia/sessions/`) | complete |
| File IO / git / shell tools | `file_tools.cpp`, `git_tools.cpp`, `process.cpp` | complete |
| Security (path validation, confirm callbacks, allowed-tools store) | `security.cpp`, registry policies | complete |
| Audio / SD / VLM | `image.cpp` + `agents/vlm` only | VLM partial; audio/SD missing by design |
| REST API server / Agent UI backend | none | missing by design |
| TUI | `tui_app.cpp` (FTXUI) | partial (plan P7.x) |

Build/CI reality: `cpp/` **is** gated. `.github/workflows/build_cpp.yml` configures (`cmake -B cpp/build -S cpp -DGAIA_BUILD_INTEGRATION_TESTS=OFF`), builds, and runs `ctest --test-dir cpp/build -C Release --output-on-failure` on an OS matrix, plus install/`find_package` round-trip, shared-lib build, and a Windows integration-build leg; `build_agents.yml` produces static binaries via vcpkg; `benchmark_cpp.yml` runs `gaia_benchmarks` after the build job. The task brief's presumption ("60K lines with no CI gate") is **false** — recorded so the integrator does not re-check.

Wheel / Python coupling: `setup.py`, `MANIFEST.in`, `pyproject.toml` contain no `cpp` reference → `cpp/` is not shipped in the wheel. `src/gaia/` has no `ctypes`/`cffi`/`pybind`/`cpp/` dependency (only `skills/audit/code.py` naming `ctypes.CDLL` as an *audit sink string*). The two runtimes are fully decoupled; the only shared contracts are `~/.gaia/mcp.json`, `~/.gaia/skills`, `~/.gaia/sessions`, and the chunking parity fixture `tests/fixtures/chunking/parity_expected.json`.

Third-party: vendored `third_party/sqlite` = **3.53.4** (2026-07-24, current; README documents flags + upgrade procedure, checksums recorded). FetchContent pins per `cpp/README.md` dependency table: nlohmann/json 3.11.3, cpp-httplib **0.15.3**, yaml-cpp 0.8.0, FTXUI 6.1.9, GoogleTest 1.14.0 (verified against the `URL …/refs/tags/…` lines at `cpp/CMakeLists.txt:87,106,134,163,238`). vcpkg manifest lists only `openssl` (unpinned, no `builtin-baseline`).

## Findings

### [🟡] Streaming LLM errors are thrown away — the user sees "contained no tokens" instead of the server's message
- **Where:** `cpp/src/sse_parser.cpp:72-74` + `:92-94` (`SseParser::processData`), `cpp/src/lemonade_client.cpp:386-417` (`chatCompletionsStreaming`), `cpp/src/agent.cpp:661`
- **What:** When an OpenAI-compatible server reports an error *inside* the SSE stream (HTTP 200, `data: {"error":{...}}`), the parser drops the event because it has no `choices`; the client's embedded-error check then `json::parse`s the raw SSE bytes (which begin with `data: `, so parsing fails and is swallowed); the agent finally throws a generic error. Non-streaming mode (`chatCompletions`, `:303-340`) extracts the same error and even produces the "context window too small" remedy; streaming mode loses it. `GAIA_STREAMING=1` is the documented opt-in, so the two modes give different diagnostics for the same fault.
- **Failure scenario:** Lemonade/llama.cpp with `stream: true` and a prompt exceeding `n_ctx` → stream body is one `data: {"error": {...n_ctx...}}` event → user gets `Streaming response contained no tokens` and no hint to raise `--ctx-size`; the non-streaming path prints the `lemonade-server serve --ctx-size 32768` remedy.
- **Evidence:**
  ```cpp
  // sse_parser.cpp:72
  if (!j.contains("choices") || !j["choices"].is_array() || j["choices"].empty()) { return; }
  // lemonade_client.cpp:388
  const json responseJson = json::parse(rawBytes);   // rawBytes == "data: {...}\n\n" -> parse_error
  } catch (...) { // Not valid JSON — return raw bytes for caller's fallback handling
  // agent.cpp:661
  throw std::runtime_error("Streaming response contained no tokens");
  ```
- **Fix:** In `SseParser::processData`, when a payload parses and carries a top-level `error`, record it and stop; have `chatCompletionsStreaming` raise the same formatted message the non-streaming path builds (factor the duplicated `errMsg` builder at `:311-331` / `:389-411` into one helper). Add a `test_sse_parser` case for `data: {"error":...}` and a `test_lemonade_client` case asserting the context-size remedy text on the streaming path.
- **Confidence:** High (path traced end-to-end; in-stream error emission is server-dependent, but Lemonade's 200-with-`error` body is exactly why the non-streaming check exists)
- **Tracked:** none found (#773 is merged; no open issue)

### [🟡] `SseParser` silently discards malformed JSON events — a no-silent-fallback violation that hides lost deltas
- **Where:** `cpp/src/sse_parser.cpp:92-94` (`SseParser::processData`)
- **What:** Any `data:` payload that fails to parse is dropped with `catch (...) {}`. `feed()` already buffers until `\n`, so a legitimately partial event never reaches `processData`; the only thing this catches is a genuinely corrupt event, and it vanishes without a log line or counter.
- **Failure scenario:** One corrupt `data:` line mid-answer → that token is lost, the answer is missing a word, `hasTokens_` stays true so nothing notices. For a tool-call delta, `arguments` silently loses a fragment; if the brace balance survives, a wrong argument set is executed — the very risk `lemonade_client.cpp:419-429` guards against for stream *truncation* but not event *corruption*.
- **Evidence:**
  ```cpp
  } catch (...) {
      // Silently skip malformed JSON — servers occasionally send partial events
  }
  ```
  The comment's premise does not hold for this parser: `feed()` at `:18-19` dispatches complete lines only (`if (nl == std::string::npos) break; // Incomplete line — wait for more data`).
- **Fix:** Count and surface: increment `malformedEvents_`, expose it, and have `chatCompletionsStreaming` throw (or at minimum log under debug and refuse tool_calls) when it is non-zero. Unit test: feed `data: {"choices":[{"delta":{"content":"a"\n` and assert the error.
- **Confidence:** High
- **Tracked:** none found

### [🟡] `experiments/whatsapp-webjs/` is an orphaned spike with no ignore rules for the secrets it writes
- **Where:** `experiments/whatsapp-webjs/index.js:14,26`, `experiments/whatsapp-webjs/package.json`
- **What:** A 72-line whatsapp-web.js echo bot. `LocalAuth()` writes a `.wwebjs_auth/` session-credential directory next to the script and the bot appends to `run.log` in the same directory. Neither path is ignored anywhere (`grep -n "wwebjs\|run.log\|experiments" .gitignore` → no match; no `experiments/.gitignore` exists). Nothing references the directory: `docs/plans/messaging-integrations-plan.mdx` marks WhatsApp **Deferred** (`:176`, `:343`, `:356 "no WhatsApp in v1"`) and never mentions `experiments/`; no other doc, workflow, or packaging file does.
- **Failure scenario:** A developer runs `npm start`, scans the QR, later runs `git add -A` — `.wwebjs_auth/` (a live WhatsApp Web session) and `run.log` (sender ids; bodies if `LOG_MESSAGE_BODIES=1`) land in a commit. The file's own privacy note (`:8-13`) acknowledges this; the repo does nothing to enforce it.
- **Evidence:**
  ```js
  const LOG_PATH = path.resolve(__dirname, 'run.log');
  const client = new Client({ authStrategy: new LocalAuth() });
  ```
  `package.json`: `"test": "node index.js"` — the test script launches the bot.
- **Fix:** Delete the directory (the plan defers WhatsApp and the spike's conclusion is captured in the plan's platform matrix). If it must stay: add `experiments/whatsapp-webjs/.gitignore` (`.wwebjs_auth/`, `.wwebjs_cache/`, `run.log`, `node_modules/`), link it from the messaging plan as the deferred spike, and drop the `"test"` script.
- **Confidence:** High
- **Tracked:** none found (#635 covers Telegram/Discord/Slack adapters, not this spike)

### [🟢] cpp-httplib is pinned to 0.15.3 (Feb 2024) with no version floor on the system-package path
- **Where:** `cpp/CMakeLists.txt:101,106`
- **What:** `find_package(httplib QUIET)` accepts any system version, else FetchContent pulls `v0.15.3.tar.gz` — more than two years old at review date; upstream has shipped many hardening and API releases since. This library parses every byte the LLM server (and, after P4.1, third-party HTTP MCP servers) sends. nlohmann/json 3.11.3 and GoogleTest 1.14.0 are likewise behind; SQLite 3.53.4, yaml-cpp 0.8.0, FTXUI 6.1.9 are current.
- **Failure scenario:** No demonstrated bug; risk is inheriting a fixed parser flaw once the client is pointed at untrusted servers, and FetchContent vs. system builds diverging in HTTP behaviour.
- **Evidence:** `URL      https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.15.3.tar.gz`
- **Fix:** Bump to a current release, add a minimum to `find_package(httplib …)`, and record pins in the SBOM work.
- **Confidence:** Medium (staleness verified; no specific CVE asserted without advisory lookup)
- **Tracked:** #368 (C++ SBOM) — partial

### [🟢] `checkModelLoaded` lowercases with `::tolower` on signed `char`
- **Where:** `cpp/src/lemonade_client.cpp:200,204` (`LemonadeClient::checkModelLoaded`)
- **What:** `std::transform(..., ::tolower)` passes `char` directly to `tolower`, which is undefined for negative values (non-ASCII bytes where `char` is signed: MSVC, x86 GCC). `http_client.cpp:19-23` already has the correct `unsigned char`-casting `toLower`.
- **Failure scenario:** A model id with UTF-8 bytes (Lemonade permits arbitrary `user.` names) → UB; MSVC debug CRT asserts.
- **Evidence:** `std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);`
- **Fix:** Share the `toLower` helper (move to a small header) or cast through `unsigned char`.
- **Confidence:** High
- **Tracked:** none found

### [🟢] Lemonade embedded-error decoding is duplicated verbatim between the blocking and streaming paths
- **Where:** `cpp/src/lemonade_client.cpp:309-337` and `:387-416`
- **What:** ~30 identical lines, including the hard-coded `.\\installer\\scripts\\start-lemonade.ps1 -CtxSize 32768` remedy, appear twice; the blocks differ only in indentation.
- **Failure scenario:** Drift — one path learns a new Lemonade error shape, the other keeps the old one.
- **Evidence:** compare the two `if (responseJson.contains("error"))` blocks.
- **Fix:** One `static std::string describeLemonadeError(const json&)` — also the natural home for finding 1's fix.
- **Confidence:** High
- **Tracked:** none found

## Test gaps
- `cpp/tests/test_sse_parser.cpp` covers split lines (§3), both `[DONE]` spellings (§4, §5, §15) and post-`[DONE]` suppression, but has **no malformed-`data:` case and no in-stream `{"error":…}` case** (`grep -n "error\|malformed"` hits only `[DONE]` comments). One test each would catch findings 1 and 2.
- `cpp/tests/test_lemonade_client.cpp`: `normalizeUrl` is thoroughly covered (`:16-111`) and there is a mock-server leg (`:182-211`); nothing asserts that the streaming path surfaces a Lemonade `error` body with the context-size remedy.
- `HttpClient::postStreaming` non-2xx drain (`http_client.cpp:478-483`, 512-byte cap → `HttpError` with status/body): not confirmed covered — `test_http_client.cpp` was not read.
- `cpp/tests/integration/` is compiled only on the Windows CI leg (`-DGAIA_BUILD_INTEGRATION_TESTS`) and needs a live LLM server, so it never *executes* in CI. The live-Lemonade embeddings/RAG tests the parity plan mandates do not exist because P1.2/P2.2 have not landed.
- `experiments/whatsapp-webjs`: no tests; `npm test` launches the bot.

## Documentation gaps
- `cpp/README.md` feature matrix still lists RAG as "Python-only" while the same README documents `gaia/chunking.h` and `vector_index.cpp` ships; the parity plan (P5.2) already notes this and that `docs/plans/cpp-webui-integration.md` asserts `rag: ❌`. Both uncorrected at HEAD.
- `cpp/README.md` env-var table documents `GAIA_CPP_BASE_URL` as the base-URL override, but `LemonadeClient` reads `LEMONADE_BASE_URL` (`lemonade_client.cpp:52`; `lemonade_client.h:39`), and `http_client.cpp:214-215`'s error text names `LEMONADE_BASE_URL`. The plan calls the `GAIA_CPP_*` names deprecated; the README should document both and which wins.
- `cpp/README.md` "Project Structure" lists 10 `src/` files and 12 tests; the tree has 33 sources and 34 tests (`chunking`, `database`, `file_tools`, `git_tools`, `ignore`, `image`, `mcp_registry`, `process`, `repl`, `security`, `session`, `skill*`, `tui_*`, `vector_index` absent). Stale, not wrong.
- The README's Ollama example (`GAIA_CPP_BASE_URL=http://localhost:11434/v1 LEMONADE_MODEL=gemma4:e2b`) matches `normalizeUrl` behaviour — fine, modulo the env-var name above.
- `skills/community/README.md` vs `hub/skills/README.md` vs CLI: **consistent.** Both route contributions to `skills/community/<name>/` by PR (`hub/skills/README.md:63`), both name `gaia skill audit ./<path>/` as the pre-PR gate and `gaia skill publish` as maintainer-only; `src/gaia/skills/cli.py` defines `audit` (`:216`), `publish` (`:326`), `import` (`:135`, into `~/.gaia/skills/`) and `install` (`:298`, Hub → `~/.gaia/skills/`); `.github/workflows/skill_audit.yml:361-381` implements the `skill-audit-reviewed` label gate the README describes. Nit: `skills/community/` holds only the README, so there is no in-place worked example of the "one directory per skill" layout (the README does point at `hub/skills/`).
- `experiments/whatsapp-webjs/` is referenced by no doc (see finding 3).

## Improvement opportunities
- Fix findings 1, 2 and 5 together: one Lemonade error decoder, an error-aware `SseParser`, one set of tests — same two files.
- `LemonadeClient::getStatus` (`:157-164`) and `checkModelLoaded` (`:208`) swallow `listModels()` failures with `catch (...) {}`; `getStatus` already has an `error` field — populate it instead of dropping the reason.
- `HttpClient` builds a fresh `httplib::Client` per request (`http_client.cpp:409,440,517`): no keep-alive across the agent loop's many turns. Correct, but a per-target cached client removes a TCP(+TLS) handshake per LLM call.
- `vcpkg.json` lacks `builtin-baseline`, so OpenSSL in the static-agent builds floats with the runner's vcpkg checkout; pin it for reproducible `build_agents.yml` artifacts (ties to #368).
- `find_package(httplib QUIET)` with no version floor vs. a pinned FetchContent tarball: add a minimum so both build modes agree.

## High-impact feature opportunities
- **P1.2 embeddings + P2.2 RAG on the existing `VectorIndex`/`chunking`.** The index, the Python-parity splitter and the SQLite layer already ship; one `POST /embeddings` call and a `RAGSDK` unlock document Q&A in a native binary — the gap that keeps every OEM native agent tool-only.
- **P4.1 HTTP MCP transport** over the now-public `HttpClient`: stdio-only MCP excludes every hosted MCP server, and the client abstraction was built to unblock exactly this.
- **P3.3 SkillManager**: per the plan's own ordering note, `loadSkillSet()` resolves a set and registers nothing, so `SKILL.md` support in C++ is parse-only; "one signed binary retargeted by skills" is not deliverable until it lands.

## Checked and fine
- `cpp/` **is** built and unit-tested in CI (`build_cpp.yml`: cmake + `ctest --output-on-failure` on an OS matrix, install round-trip, shared build; plus `build_agents.yml`, `benchmark_cpp.yml`).
- Not shipped in the wheel; no Python→C++ dependency.
- `LemonadeClient::normalizeUrl` (#773): strips trailing slashes, preserves `/v1` and `/api/v1`, appends `/api/v1` otherwise — no `/v1/v1` duplication; `ensureModelLoaded` skips `/health`/`/load` for non-`/api/v1` bases so Ollama never sees Lemonade-only endpoints; unit-tested.
- `HttpClient::Impl::resolve`: single-separator join, IPv6 brackets, strict port parse (rejects `8080abc`), rejects control chars/spaces in the target; header CR/LF/NUL injection rejected; case-insensitive header merge; repeated headers collapsed per RFC 9110; timeouts validated `> 0` and applied per request; moved-from client re-armed so `impl_` is never null.
- `postStreaming`: non-2xx bodies drained (512 B cap, UTF-8-safe cut) and surfaced as `HttpError` with status; callback stop (`[DONE]`) distinguished from transport `Canceled`.
- `SseParser::feed`: buffers across arbitrary chunk boundaries, handles `\r\n`, `data:` with/without space, comments, `[DONE]`; tool-call deltas keyed by `index` with positional fallback; id-vs-continuation logic prevents `echoecho`; `chatCompletionsStreaming` refuses tool_calls from a stream that ended before `[DONE]`.
- `agent.cpp:596-601`: unsolicited `tool_calls` dropped when the request sent no `tools`; non-streaming parse failure throws with a 200-char body preview (`:670-675`).
- Memory-safety sample: one raw `new` (`process.cpp:973`) owned by `std::unique_ptr<Impl>` (`process.h:183`); the three `memcpy`s are bounded (`file_tools.cpp:85-86` SHA-256 block fill with `take = min(len, 64 - bufferLen_)`; `vector_index.cpp:48,71` `sizeof` type-punning); no `strcpy`/`sprintf`/`strcat`/`gets`; no `.detach()`; `process.cpp:344-372` reader threads joined after child exit/terminate; `agent.cpp:1339-1360` `optional::value()` calls sit under `has_value()` guards.
- Vendored SQLite 3.53.4 current, checksummed, hardened (`SQLITE_DQS=0`, `SQLITE_TRUSTED_SCHEMA=0`, defensive + no extension loading at runtime).
- `skills/community/README.md` agrees with `hub/skills/README.md`, the `gaia skill` CLI and `skill_audit.yml`.
- `experiments/whatsapp-webjs/index.js` handles no API keys; the only secret material is the LocalAuth session (finding 3).

## Hypotheses (unverified)
- `agent.cpp` was read only around the LLM-call site (`:580-700`); the loop body, plan handling and the dead `resolvePlanParameters()` the parity plan flags were not re-verified at HEAD.
- `mcp_client.cpp`: the parity plan (verified at `9bf0042a`) lists `protocolVersion: "1.0.0"`, no `notifications/initialized`, naive argv concatenation in `StdioTransport`, raw `result` returned without `content[]`/`isError` unwrapping (P4.2). Not re-checked at 211f08c5; if the argv concatenation persists it is a 🟡 shell-metacharacter issue on POSIX.
- `tui_app.cpp:445` (`worker_` thread) and `repl.cpp:420` (per-query worker): join/cancellation on agent destruction not read.
- `database.cpp`: whether a `gaia::Database` handle is safe to share across threads without an external mutex (compiled `SQLITE_THREADSAFE=1`) not read.
- cpp-httplib 0.15.3 may carry parser CVEs fixed later; not checked against an advisory database.

## Summary
1. 🟡 Streaming mode loses the LLM server's error text (in-stream `{"error":…}` → generic "no tokens") while non-streaming prints the context-size remedy — same fault, worse diagnosis under `GAIA_STREAMING=1`.
2. 🟡 `SseParser` swallows malformed events with `catch(...){}` although it already line-buffers, so corrupt deltas vanish silently from answers and tool arguments.
3. 🟡 `experiments/whatsapp-webjs/` is an orphaned, undocumented spike whose `LocalAuth` session dir and `run.log` have no `.gitignore`; the messaging plan defers WhatsApp — delete or fence it.
4. 🟢 cpp-httplib pinned at 0.15.3 (2024) with no `find_package` version floor; `vcpkg.json` unpinned — fold into SBOM work (#368).
5. Presumptions to drop: `cpp/` is ~47K lines (not 60K), **is** CI-built and ctest-gated on a matrix, is not in the wheel, Python has no dependency on it; `normalizeUrl`/Ollama `/v1` handling and the community-skills docs/CLI agreement check out.

