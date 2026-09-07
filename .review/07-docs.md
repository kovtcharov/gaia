# 07 — Documentation accuracy and completeness review

Reviewer dimension: docs/ (Mintlify), root docs, hub agent bundled docs, hub/skills, .claude/agents + .claude/skills, CLAUDE.md claims. Repo at 211f08c5 (`__version__ = "0.23.1"`, not yet tagged — `gh release list` latest is v0.23.0).

## Scope covered

Mechanical checks (helper scripts in `.review/_07_*.py`, all read-only):
- `docs/docs.json` walked against disk (195 nav pages, 199 `.mdx` on disk); relative links/images scanned.
- Full `gaia` argparse tree dumped via `gaia.cli.build_parser()` (`.review/_07_cli_tree.json`, 130 command nodes) and diffed against `docs/reference/cli.mdx` (3163 lines) and `README.md`.
- Every `from gaia… import …` in `docs/**`, `hub/**/*.md`, `.claude/**`, root docs (120 distinct imports) resolved in `.venv` (hub packages via `sys.path`).
- Every `ChatAgentConfig(...)`, `AgentConfig(...)`, `GaiaAgentConfig(...)` kwarg used in docs checked against dataclass fields.
- Every repo path cited in `.claude/agents/*.md` (20), `.claude/skills/*/SKILL.md` (17), CLAUDE.md, AGENTS.md, CONTRIBUTING.md, REVIEW.md, SECURITY.md, hub READMEs checked for existence.
- 13 `hub/skills/*/SKILL.md` front-matters parsed and compared with `docs/plans/skill-format.mdx`; every `tools_required` name grepped as a `def <name>(` tool.
- `tests/unit/test_amd_gaia_urls.py` run; `amd-gaia.ai` links grepped across docs/hub/root/src/.claude.
- Hub packages gaia / chat / email (+ hello-world, word-count, connectors-demo): pyproject / package.json / gaia-agent.yaml versions vs CHANGELOG headings vs GitHub tags (curl) vs npm registry (`npm view`); README/SPEC/SKILL lifecycle, port, and endpoint claims vs `server.py` / `agent.py`.
- Read in full: README.md, CLAUDE.md, hub/agents/README.md, hub/skills/README.md, docs/roadmap.mdx status sections, docs/releases/v0.23.1.mdx, heads of quickstart/index/guides/install/guides/terminal-hub, cli.mdx sections for API / MCP / TUI / Pull / Connectors, docs/reference/eval.mdx head, the status line of every docs/plans/*.mdx.
- TUI (`tui/`) subcommand set read from cobra `Use:` strings; binary name from `tui/Makefile`.

Skipped / grep-only: bodies of the ~150 non-reference `.mdx` pages, `docs/cpp/**`, `docs/connectors/**` prose, `docs/spikes`, `docs/superpowers`, `docs/issues`, non-nav `.md` files under docs, `hub/agents/email/node/README.md`.

## Findings

### 🔴 docs.json navbar label still says `v0.23.0 · Lemonade 11.5.0`; the v0.23.1 tag will fail the publish workflow's own gate
- **Where:** `docs/docs.json:508` (`navbar.links[0].label`) vs `src/gaia/version.py:9,12` (`__version__ = "0.23.1"`, `LEMONADE_VERSION = "11.8.1"`)
- **What:** The release commit (72241b2f, "Release v0.23.1") added `releases/v0.23.1` to the Releases tab but left the navbar version label at v0.23.0 / Lemonade 11.5.0. README.md's own release checklist (step 3b) says the label must be updated, and `.github/workflows/publish.yml:152-159` hard-fails when the navbar label does not contain the tag.
- **Failure scenario:** Pushing tag `v0.23.1` → publish.yml "Validate docs.json" step exits 1 (`docs/docs.json navbar version label does not reference v0.23.1`); no PyPI / GitHub release ships until a follow-up commit lands. Meanwhile the live docs header advertises Lemonade 11.5.0 while `gaia init` installs 11.8.1.
- **Evidence:** `git show 72241b2f -- docs/docs.json` diff is a single line `+ "releases/v0.23.1",`; `gh release list` latest = v0.23.0; `curl -o /dev/null -w "%{http_code}" https://github.com/amd/gaia/tree/v0.23.1` → 404.
- **Fix:** Set the label to `"v0.23.1 · Lemonade 11.8.1"` before tagging; add a PR-time unit test asserting the navbar label contains `__version__` and `LEMONADE_VERSION` (see Test gaps).
- **Confidence:** High
- **Tracked:** none found

### 🟡 v0.23.1 release notes tell users to run `gaia install gaia` / `gaia list` — neither exists in this build
- **Where:** `docs/releases/v0.23.1.mdx:8,11,19,25` vs `src/gaia/cli.py:3009` (`gaia install` = Lemonade installer; flags `--lemonade/--yes/--silent`, no agent positional) and `src/gaia/cli.py:2691` (`gaia hub install <agent_id>`); TUI subcommands are only `run`, `status`, `chat`, `version` (`tui/internal/cli/{agents,chat,version}.go` `Use:` strings).
- **What:** The headline "Why upgrade" bullet and the Breaking Changes migration advice both say `gaia install gaia`; the intro says `gaia list` hid the agent. `docs/guides/terminal-hub.mdx` says the TUI's `list/install/uninstall/hub` subcommands "are gone" and installing is "the Python CLI's job". The working commands are `gaia hub install gaia --trust` and `gaia hub list` — which `docs/releases/v0.23.0.mdx:41` documents correctly.
- **Failure scenario:** A user following the v0.23.1 notes runs `gaia install gaia` → `gaia: error: unrecognized arguments: gaia`; `gaia install` alone starts the Lemonade installer. The one thing the patch release exists to fix is described with a command that does not run.
- **Evidence:** `gaia install --help` → `usage: gaia install [-h] … [--lemonade] [--yes] [--silent]`; `gaia hub install --help` → `positional arguments: agent_id  Agent to install (e.g. email)`.
- **Fix:** Replace with `gaia hub install gaia --trust` / `gaia hub list`. `util/validate_release_notes.py` checks structure only — add a check that every documented `gaia …` invocation resolves in `build_parser()`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 cli.mdx documents a `gaia tui …` command family that the `gaia` CLI it describes rejects
- **Where:** `docs/reference/cli.mdx:2683-2900` ("Terminal UI (`gaia tui`)"; Note "`gaia tui status` and `gaia status` are the same command"; table rows `gaia tui run/chat/status/version`) vs `src/gaia/cli.py:1260` subparser choices and `tui/Makefile:9-13` ("Built as gaia-tui, never as gaia").
- **What:** 11 `gaia tui` examples on the CLI reference; the binary is `gaia-tui`, and `docs/guides/terminal-hub.mdx` says so in bold ("never `gaia`"). Two pages in the same nav contradict each other.
- **Failure scenario:** `gaia tui status` on a pip install → `gaia: error: argument action: invalid choice: 'tui'` (reproduced with `.venv/Scripts/gaia.exe tui --help`).
- **Evidence:** `grep -c 'gaia tui' docs/reference/cli.mdx` → 11; argparse choices contain no `tui` or `status`.
- **Fix:** `gaia tui` → `gaia-tui` throughout cli.mdx; drop the "same as `gaia status`" note; or replace the section with a pointer to terminal-hub.mdx (#3087 already flags it as a contradicting duplicate).
- **Confidence:** High
- **Tracked:** #3219, #2709, #3087

### 🟡 npm `@amd-gaia/agent-email` 0.6.0 is published with every doc link pinned to a tag that does not exist
- **Where:** `hub/agents/email/npm/README.md:14,153,166,171,176-179`, `hub/agents/email/npm/CHANGELOG.md:5` (all `https://github.com/amd/gaia/blob/agent-pkg-email-v0.6.0/…`) vs GitHub tags.
- **What:** `npm view @amd-gaia/agent-email version` → `0.6.0`; both CHANGELOGs carry `## [0.6.0] - 2026-08-12`; `pyproject.toml`, `package.json`, `gaia-agent.yaml` all say 0.6.0 — but the tag the README links through was never created.
- **Failure scenario:** An integrator on npmjs.com clicks SPEC / SKILL / SCORECARD / EVALUATION / CHANGELOG → GitHub 404 on all seven links. The README is the only rendered doc the package has.
- **Evidence:** `curl -o /dev/null -w "%{http_code}" https://github.com/amd/gaia/tree/agent-pkg-email-v0.6.0` → `404` (the same check on `agent-pkg-gaia-v0.1.1` → `200`, so the method is sound); `gh release list` newest email pre-release is `agent-pkg-email-v0.5.0`.
- **Fix:** Create `agent-pkg-email-v0.6.0` at the published commit (or re-publish under a tag) and make `release_agent_email.yml` refuse `npm publish` unless the tag named in the README exists.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `@amd-gaia/gaia` CHANGELOG says 0.1.1 is "unreleased" — it is on npm, tagged, and named in the release notes
- **Where:** `hub/agents/gaia/npm/CHANGELOG.md:7` (`## [0.1.1] — unreleased`) vs `npm view @amd-gaia/gaia version` → `0.1.1`, tag `agent-pkg-gaia-v0.1.1` (HTTP 200), `docs/releases/v0.23.1.mdx:8` ("published on the Agent Hub since 0.1.1").
- **What:** The package's own changelog contradicts the registry and the release notes.
- **Failure scenario:** A user checking whether the 503/409 refusals documented under 0.1.1 are in their build reads "unreleased" and concludes they are not.
- **Fix:** Date the entry.
- **Confidence:** High
- **Tracked:** none found

### 🟡 CLAUDE.md carries stale claims that steer agents at the wrong code
- **Where / What (each verified):**
  1. `CLAUDE.md:689` — "GaiaAgent rename planned (#696) — not yet landed; the chat agent class is still `ChatAgent`". `hub/agents/gaia/python/gaia_agent/agent.py:211` is `class GaiaAgent(ChatAgent, SkillLibraryToolsMixin, CodeIndexToolsMixin)`; the same file's agent table (:578-580) already calls GaiaAgent the flagship, so CLAUDE.md contradicts itself.
  2. `CLAUDE.md:256` — "Check existing mixins in agent packages (e.g., `hub/agents/chat/python/gaia_agent_chat/tools/`)". No such directory; mixins live in `src/gaia/agents/tools/` (the KNOWN_TOOLS table at :601-616 is correct).
  3. `CLAUDE.md:455-520` project tree omits shipped top-level packages `src/gaia/{daemon,hub,schedule,sidecar,skills}/` (plus `cli_agent.py`, `config.py`, `device.py`, `security.py`) — while the CLI list at :636-660 documents `gaia daemon/hub/schedule/skill`.
  4. `CLAUDE.md:640,650,660` CLI list: `gaia mcp {…}` omits `tui`; `gaia config {get|set}` omits `show`; `gaia agent {export|import}` omits `init|version|test|pack|publish|configure|health|status|login|install|list` (all documented in cli.mdx:433-712); `gaia eval {benchmark|sessions|code}` and `gaia lemonade embedded {start|stop|status|install|install-backend}` are absent.
  5. `CLAUDE.md:675` — Guides "one per feature: chat, agent-ui, email, talk, memory, install, custom-agent, hardware-advisor, npu"; `docs/guides/` has 21 pages, including the flagship `gaia.mdx`, `terminal-hub`, `starter-skills`, `composing-skills`, `hub-publishing`, `code-index`, `telegram-adapter`, `email-integration`, `eval*`.
  6. `CLAUDE.md:662` — "Evaluation & analysis (see `docs/reference/eval.mdx`)" points at a page whose banner says it was deprecated in v0.18.0 (next finding).
- **Failure scenario:** An agent following CLAUDE.md greps a non-existent mixin dir, assumes GaiaAgent has not landed, or documents `gaia agent` as a two-subcommand tool.
- **Fix:** One pass over CLAUDE.md against `gaia -h` and `ls src/gaia`; delete the #696 bullet.
- **Confidence:** High
- **Tracked:** none found (#2275 is the epic the tree drifted from)

### 🟡 `docs/reference/eval.mdx` stays in the nav with 25 `gaia eval -d …` examples that error
- **Where:** `docs/reference/eval.mdx:1-12` (Warning: "removed in v0.18.0"), `:231,281,321,357…` (`gaia eval -d ./output/experiments -o …`, 25 occurrences) vs `src/gaia/cli.py:1957` (`gaia eval` requires `{agent,benchmark,sessions,code}`); `docs/docs.json:340` still lists `reference/eval`.
- **What:** The page self-declares deprecated but is navigable, and CLAUDE.md cites it as the eval reference. Nothing on it runs.
- **Failure scenario:** `gaia eval -d ./x` → `gaia eval: error: argument eval_command: invalid choice: './x'`.
- **Fix:** Remove from `docs.json` with a redirect to `guides/eval`; point CLAUDE.md at `docs/guides/eval.mdx` + `docs/reference/eval-scorecard.mdx`. While there, document `gaia eval code` (zero mentions in docs/) and `gaia eval sessions --dataset-only/--project`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Roadmap is five months and six releases stale, and still schedules deleted agents
- **Where:** `docs/roadmap.mdx:387` ("*Updated: April 13, 2026*"), `:137` ("v0.17.3 — **Status:** In progress — Due: April 17, 2026"), `:23-66` (timeline: Shipped ends at v0.17.2; v0.18–v0.23 are Near/Mid-term), `:235` (CodeAgent → #695), `:285` ("Consolidate SD agent (gaia sd)" → #771) vs `src/gaia/version.py` (0.23.1) and PR #2995 which deleted the `code` and `sd` agents.
- **What:** The public roadmap lists as future work what shipped in v0.18–v0.23 (memory, email, hub, skills, messaging, autonomy per `docs/releases/`) and names two deleted agents as deliverables.
- **Failure scenario:** External readers conclude v0.17.3 is current and that a `gaia sd` agent is coming.
- **Fix:** Move v0.18–v0.23 into Shipped (release notes already have the summaries); drop or reword the #695 / #771 rows; bump the Updated stamp.
- **Confidence:** High
- **Tracked:** none found (#1081 covers related stale version strings)

### 🟡 `docs/plans/*` status headers contradict shipped code; four plans are unreachable from the nav
- **Where / What:**
  - `docs/plans/email-triage-agent.mdx` — "Planning (0% implemented)"; the agent ships at 0.6.0 with a SCORECARD.
  - `docs/plans/messaging-integrations-plan.mdx` — "Planning (no implementation)"; `gaia telegram {start,stop,status}` (`cli.py:1610`), `src/gaia/messaging/`, `docs/guides/telegram-adapter.mdx` exist.
  - `docs/plans/desktop-installer.mdx` — "Planning"; installers ship (`gaia-agent-ui-0.23.0-{x64-setup.exe,arm64.dmg,amd64.deb,x86_64.AppImage}` in the v0.23.0 release assets); the roadmap lists it as shipped in v0.17.2.
  - `docs/plans/connectors.mdx` — "Target v0.18.x | implementation underway"; `gaia connectors` has 14 subcommands and 22 doc pages.
  - `docs/plans/agent-hub.mdx` — "Target Q2 2026 | Planning"; `gaia hub {list,install,uninstall}` shipped in v0.23.0.
  - `docs/plans/image-agent.mdx` — "a Stable Diffusion agent already ships today (`gaia sd`)"; deleted in #2995, no `sd` subparser.
  - `docs/plans/autonomy-engine.mdx` — "Planning (0% implemented)" while `docs/plans/email-full-autonomy.mdx` says "Phases 1–5 shipped" and `gaia email autonomy {status,trust,set-level,pause,resume,kill,run}` exists.
  - Orphans (on disk, not in `docs.json`): `plans/bash-agent`, `plans/email-full-autonomy`, `plans/package-publishing`, `plans/typescript-sdk`.
- **Failure scenario:** CLAUDE.md sends agents to `docs/plans/` before building; "0% implemented" invites re-planning shipped work.
- **Fix:** Add a "Shipped in vX — see guide" banner or an archive group; register or delete the four orphans.
- **Confidence:** High
- **Tracked:** none found

### 🟡 Doc code examples import names that do not exist
- **Where / What:**
  - `docs/spec/test-utilities.mdx:12` — "**Import:** `from gaia.testing import MockLLMProvider, MockVLMClient, create_test_agent, temp_database`"; `grep -rn temp_database src/gaia/testing/` → no hits, and `src/gaia/testing/__init__.py` `__all__` has no such name. The page documents `temp_database()` in full at :272-300 and a test for it at :464.
  - `docs/spec/test-utilities.mdx:525` — `from gaia import SilentConsole`; defined only at `src/gaia/agents/base/console.py:2525`, not re-exported by `src/gaia/__init__.py`.
  - `docs/spec/llm-client.mdx:1191` — `from gaia.agents import Agent`; `src/gaia/agents/__init__.py` exports nothing but a logger. Real path: `gaia.agents.base.agent`.
  - `docs/sdk/patterns.mdx:211,239` — `from gaia.agents.hello.agent import HelloAgent, HelloAgentConfig` under "Add to `AgentRegistry._register_builtin_agents`" (method exists, `registry.py:596`; module is fictional) and again as a runnable `# tests/test_hello_agent.py`.
- **Failure scenario:** Copy-paste → `ImportError` on line one.
- **Evidence:** `.review/_07_sdk_imports.py` — 120 imports checked; these are the only failures apart from the optional `mcp` extra.
- **Fix:** Point the first three at their real modules (or ship the documented `temp_database` and the re-exports); base Pattern 5/6 on `hub/agents/hello-world` (`gaia_agent_hello_world.agent.HelloWorldAgent`), which exists.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `.claude/` agents and skills, AGENTS.md, CONTRIBUTING.md cite files and agents that were deleted
- **Where / What (all verified missing on disk):**
  - `.claude/agents/cli-developer.md:20,35,87` — `tests/test_cli.py` (CLI tests are `tests/unit/test_cli_agent.py` etc.).
  - `.claude/agents/mcp-developer.md:37` — `src/gaia/mcp/blender_mcp_server.py` + `blender_mcp_client.py` (Blender agent deleted in #2995).
  - `.claude/agents/test-engineer.md:40` — "code-agent tests live in the hub package: `hub/agents/code/python/tests/`".
  - `.claude/agents/gaia-agent-builder.md:80` — cites `CodeAgent`, `JiraAgent` as living examples.
  - `.claude/skills/github-issue-response/SKILL.md:112` — `hub/agents/jira/python/`.
  - `.claude/skills/gaia-executive-presentation/SKILL.md:24`, `gaia-technical-presentation/SKILL.md:22` — `hub/agents/email/python/README.md` (the README is at `hub/agents/email/npm/README.md`).
  - `AGENTS.md:157` — `docs/spec/orchestrator.mdx` (no such file).
  - `CONTRIBUTING.md:61` — example test plan `pytest tests/unit/test_chat.py -k startup` (no such file).
- **Failure scenario:** An agent launched with one of these definitions greps for a missing file and stalls, or recreates a deleted agent's test dir.
- **Fix:** Extend `util/check_doc_links.py` to walk `.claude/**`, `AGENTS.md`, `CONTRIBUTING.md` (~30 lines; see `.review/_07_claude_paths.py`).
- **Confidence:** High
- **Tracked:** none found

### 🟡 CLI reference gaps: implemented subcommands/flags documented nowhere
- **Where:** `.review/_07_cli_tree.json` vs `docs/reference/cli.mdx`.
- **What:** `gaia eval code` (0 mentions in docs/); `gaia eval sessions --dataset-only/--project`; `gaia report --eval-dir/--output-file/--summary-only` (cli.mdx:1667 shows only `gaia report -d`); `gaia memory bootstrap --infer/--system/--reset-system`; `gaia connectors connect --grant-agent`; `gaia connectors configure --client-id/--client-secret`; `gaia install --silent`.
- **Failure scenario:** `--grant-agent` and `--client-id/--client-secret` are how a connector gets wired for a sidecar agent non-interactively; the Connectors section (cli.mdx:1316-1382) never shows them.
- **Fix:** Add table rows; better, generate the option tables from `build_parser()` (the dump script here is ~25 lines).
- **Confidence:** High
- **Tracked:** #3140 (related: undocumented TUI flags)

### 🟢 hub/agents/README.md says every package ships `<id>/python/README.md`; the two products don't
- **Where:** `hub/agents/README.md:33-44` vs `hub/agents/gaia/python/` and `hub/agents/email/python/` (no README.md; docs are under `npm/`).
- **Fix:** "README.md (or `npm/README.md` for sidecar packages)". — **Confidence:** High — **Tracked:** none found

### 🟢 README.md names the wrong default VLM and omits macOS from requirements
- **Where:** `README.md:76` ("Extract text from images with Qwen3-VL-4B") vs `src/gaia/vlm/mixin.py:55` (`model: str = "Gemma-4-E4B-it-GGUF"`) and CLAUDE.md:620; `README.md:112-116` lists OS "Windows 11, Linux" while the Download section offers a macOS `.dmg` (present in v0.23.0 assets) and `docs/guides/install.mdx:12` says macOS 14+.
- **Fix:** "Gemma-4 (Qwen3-VL also supported)"; add macOS. — **Confidence:** High — **Tracked:** none found

## Test gaps
- **No PR-time check ties `docs/docs.json`'s navbar label to `version.py`.** `publish.yml:128-165` checks it only on tag push, which is why 72241b2f merged with the stale label. A ~10-line `tests/unit/test_docs_json_release.py` (load docs.json; assert `__version__` and `LEMONADE_VERSION` appear in the navbar label and `releases/v{__version__}` is in the Releases pages) would have failed the release PR.
- **`util/validate_release_notes.py` validates structure, not commands.** The v0.23.1 notes passed it while naming `gaia install gaia`. Parsing every `` `gaia <sub> …` `` in a release note and checking it against `build_parser()` choices would catch this class.
- **`tests/unit/test_amd_gaia_urls.py` scans only `src/gaia/`.** docs/, hub/, and `.claude/` were clean today, but hub READMEs render on npm; extend the glob to `hub/**/*.md`.
- **Nothing checks that hub README tag-pinned links resolve** (email 0.6.0). `release_agent_email.yml` could `git rev-parse` the tag named in the README before `npm publish`.
- **`.claude/` is outside `util/check_doc_links.py`'s walk**, so eight stale paths in agent/skill definitions went unnoticed.
- The docs-example import check done here (`.review/_07_sdk_imports.py`) has no CI equivalent; four broken imports in `docs/spec` and `docs/sdk` have been shipping.

## Documentation gaps
- Contradictions (detailed in Findings): navbar vs `version.py`; v0.23.1 notes vs CLI; cli.mdx `gaia tui` vs terminal-hub.mdx; CLAUDE.md #696 bullet vs the `GaiaAgent` class; plans/roadmap status vs shipped code; `reference/eval` deprecated-but-navigated.
- Missing docs for shipped surfaces: `gaia eval code` (nothing); `gaia daemon` has no guide (only cli.mdx:2540 — the daemon is the machine-wide custody process every sidecar depends on); `gaia schedule` has no guide (cli.mdx:2344 plus a mention in `starter-skills.mdx`); the `governance` package has only `sdk/sdks/governance` + `integrations/guard-proxy`, no user guide; `gaia lemonade embedded` is cli.mdx-only (acceptable).
- Stale references to deleted agents are confined to `docs/roadmap.mdx`, `docs/plans/image-agent.mdx`, `docs/spec/agent-hub-restructure.mdx` (historical), and `docs/playbooks/chat-agent/part-1-getting-started.mdx:513-594` (a tutorial-defined `DocQAAgent`, acceptable). Live guides / sdk / reference pages contain no `gaia code|docqa|analyst|sd|jira|blender|docker|emr|summarize` invocations.
- Onboarding narrative disagrees: `docs/quickstart.mdx:13` calls the Electron desktop app "the primary install path for end users", while `docs/guides/terminal-hub.mdx`, the v0.23.1 notes ("the website tells you to download the terminal hub") and `hub/agents/gaia/npm/CHANGELOG.md` ("`npx @amd-gaia/gaia` is now the single command that gets a user running GAIA") lead with the terminal hub. Both ship in the v0.23.0 assets; the docs just don't agree on which to lead with.
- `docs/reference/cli.mdx` at 3163 lines duplicates `terminal-hub.mdx` (TUI) and `hub-publishing.mdx` (`gaia agent …`); #3087 already notes the copies drift.

## Improvement opportunities
- Generate cli.mdx option tables from `build_parser()` — the dump in `.review/_07_cli_dump.py` already yields opts/defaults/choices/help; ends the flag-drift class.
- Give every `docs/plans/*.mdx` a front-matter `status:` and render it; a script can then flag "Planning" plans whose feature has a guide.
- Replace `docs/reference/eval.mdx` with a redirect to `guides/eval` and delete the 25 dead examples.
- Make `util/check_doc_links.py` also walk `.claude/**`, `AGENTS.md`, `CONTRIBUTING.md`, `hub/**/*.md`.
- Fold the TUI section of cli.mdx into terminal-hub.mdx and leave one link.
- CLAUDE.md: replace the hand-maintained CLI list and project tree with "run `gaia -h` / `ls src/gaia`" plus only the non-obvious pointers (the file already does this for skills at :735-745).

## High-impact feature opportunities
- **Docs-as-tests for the CLI** — a pytest that extracts every `gaia …` invocation from fenced bash blocks in `docs/**` and `docs/releases/**` and validates subcommand + flag existence against `build_parser()` (no execution). ~60 lines; would have caught the release-notes, `gaia tui`, and `eval.mdx` findings. Matters because the docs site is the onboarding path the release notes say users actually follow.
- **Hub package doc-bundle gate** (extend `util/check_agent_conventions.py`): README/SPEC/SKILL/CHANGELOG present; top dated CHANGELOG entry == `pyproject`/`package.json`/`gaia-agent.yaml` version; every `blob/<tag>/` link in README resolves to an existing tag. The email 0.6.0 case shows the npm README is the only doc most integrators see.
- **A `gaia daemon` guide** — it is the process model for the flagship, email, and any future sidecar, and is documented today only as a cli.mdx section.

## Checked and fine
- `docs/docs.json`: all 195 pages exist on disk, no duplicates; the relative-link/image scan found nothing broken (the six regex hits were code-sample noise). `.github/workflows/docs.yml` + `util/check_doc_links.py` already gate docs links.
- `tests/unit/test_amd_gaia_urls.py` passes; no `amd-gaia.ai/<non-docs path>` links anywhere in docs/, hub/, root docs, or `.claude/`.
- CLAUDE.md KNOWN_TOOLS table (12 entries) matches `registry.py:41-52` exactly; `DEFAULT_MODEL_NAME`, `GPU_CTX_SIZE=65536`, `NPU_CTX_SIZE=32768`, `DEFAULT_MAX_STEPS=50` and `GAIA_AGENT_MAX_STEPS` match code; `.claude/agents/` really has 20 files; console scripts (`gaia`, `gaia-cli`, `gaia-mcp`) match `setup.py:342-345`.
- README.md quickstart example: `_get_system_prompt`, `_register_tools`, `process_query` all exist on `Agent`; `cpp/README.md` exists; the release-process table names the right three files.
- cli.mdx defaults spot-checked against argparse: `--ui-port 4200`, api `--port 8080`, mcp start `--port 8765` / `--ctx-size 32768`, `--whisper-model-size base`, `--max-tokens 512`, `gaia init --profile` choices, `gaia report -d` alias — all correct. The `mcp add/remove` removal note (#977) and the Pull Command redirect to `lemonade-server pull` are accurate.
- Every `ChatAgentConfig(...)` / `AgentConfig(...)` / `GaiaAgentConfig(...)` kwarg used in docs is a real dataclass field.
- 13 `hub/skills/*/SKILL.md` front-matters conform to `docs/plans/skill-format.mdx` (name == dir, version, license MIT, `metadata.gaia.security_tier`, provenance); all 27 distinct `tools_required` names exist as `def <name>(` tools; hub/skills/README.md's "thirteen" count is right and its links (`skills/community/`, `docs/spec/agent-skills.mdx`) resolve.
- gaia npm README/SPEC/SKILL agree with each other and with code: `DEFAULT_PORT = 8141` (`server.py:749`), `/v1/gaia/{init,query,query/{run_id}/cancel,respond}` routes, 409/503 refusals, `GAIA_DYNAMIC_SKILLS[_TAU]` env vars, tree-kill/autoCleanup lifecycle.
- email npm README/SPEC/SKILL lifecycle claims are consistent (auto-reap by default, `shutdown` for graceful stop, port 8131 == `server.py:50`) — the #1841 miss has not recurred. Python and npm CHANGELOGs carry matching `[0.6.0] - 2026-08-12` entries; `gaia-agent.yaml`, `pyproject.toml`, `package.json` all say 0.6.0.
- `docs/releases/v0.23.0.mdx` and `v0.23.1.mdx` exist and are registered; `version.py` is 0.23.1; v0.23.0 release assets include the Electron installers install.mdx describes plus the TUI binaries.
- Hub teaching templates (hello-world, word-count, connectors-demo) each have README + tests as hub/agents/README.md promises; `gaia agent init … --layout hub` exists.

## Hypotheses (unverified)
- `docs/reference/cli.mdx:2725` shows `gaia hub install email` reporting "Version 0.5.0" — likely an old capture, but if the live hub catalog still serves 0.5.0 while npm has 0.6.0 it is a second symptom of the missing 0.6.0 tag; not checked against the live catalog.
- `hub/agents/gaia/npm/SKILL.md:25` says the agent loads skills from `gaia_agent/skills/<name>/SKILL.md`; on disk that directory contains only `gaia-voice`, while the 13 starter skills live in `hub/skills/`. Presumably the freeze/packaging step copies them in (not traced) — one for the packaging reviewer.
- `.claude/agents/*.md` bodies may cite symbols beyond the path check done here; only paths and `CodeAgent` / `JiraAgent` / `ChatSDK` / `DocQAAgent` were swept.
- `git ls-remote --tags origin` returned nothing for any `agent-pkg-*` pattern (origin is `kovtcharov/gaia`, a fork); the 404/200 conclusions above come from `curl` against `github.com/amd/gaia`, which is the repo the README links target.
