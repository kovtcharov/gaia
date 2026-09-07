
---
# PART B — High-impact feature opportunities (main deliverable)

## B.1 Inputs used
- `docs/roadmap.mdx` (last updated **April 13, 2026**, still says v0.17.3 "In progress" — the code is at v0.23.1, see finding D-1 below).
- All 56 files in `docs/plans/` (headers + status sections; the ones with code-grounded status tables — `unified-tui-composability.md`, `flagship-installer.mdx`, `gaia-agent-readiness.md`, `skill-format.mdx`, `adaptive-skills.mdx`, `tool-loader.mdx`, `email-full-autonomy.mdx` — read in more depth).
- Open issues: **697** open (pulled via `gh api repos/amd/gaia/issues --paginate`, PRs excluded, saved to `.review/_08_issues_all.json`). Label distribution: enhancement 406, agent 226, p1 211, track:consumer-app 142, domain:agent-core 103, **bug 89**, weekly-audit 56, security 31, cpp 37, agent-ui 39, tui 14, eval 19, installer 21, rag 21. 54 unlabeled. `good first issue`: **5** only.
- Issue velocity: 148 opened in Jul-2026, 149 in Aug-2026 — the backlog grows ~5/day, dominated by `weekly-audit` bot filings and email-agent follow-ups.
- `AGENTS.md`, `hub/agents/README.md`, `hub/skills/README.md`, `website/src/pages/index.astro` (the product promise).
- Code state checked directly (paths cited per item).

## B.2 What the product promises (website `index.astro`) vs. what the code delivers

| Landing-page claim | Code reality (HEAD) | Verdict |
|---|---|---|
| "Answers from your documents / indexed on the machine" | `src/gaia/rag/sdk.py` (3.4K lines), RAGToolsMixin | ✅ shipped |
| "Reads and edits your files, scoped to home folder" | `file_io_tools.py`, `filesystem_tools.py` | ✅ shipped (#3316: mixins need host attrs nothing sets — first tool call can fail) |
| "Searches code by meaning" | `src/gaia/code_index/` (FAISS) | ✅ shipped |
| "Works your spreadsheets (CSV/Excel → local SQL scratchpad)" | `src/gaia/scratchpad/service.py` | ✅ shipped |
| "Looks things up on the web" | `src/gaia/web/client.py`, `browser_tools.py`; search needs a Tavily key | ❌ **`fetch_page` fails on 8 of 14 CDN-fronted HTTPS sites** (incl. amd-gaia.ai, github.com, pypi.org, huggingface.co) — the IP-pinning adapter sends no SNI (Part A 🔴); search also depends on a cloud API next to "no silent cloud fallback" |
| "Asks before it acts — every tool call needs approval" | Permission prompt exists (`tui/internal/ui/components/confirmation.go`); durable audit via `GovernedAgentMixin` is **not wired** — zero references outside `src/gaia/governance/` (#3096 open) | ⚠️ half |
| "The AI chip in your PC does the work (NPU)" | NPU profile exists (`gemma4-it-e2b-FLM`, `NPU_CTX_SIZE=32768`); FLM↔Vulkan thrash epic #1676/#1746 open; #2972 npu profile → zero agents; #3175 `flm:npu` install rejected by Lemonade | ⚠️ least reliable path |
| "One model stays loaded — switching costs nothing" | every agent defaults to `DEFAULT_MODEL_NAME` | ✅ by design; #3152 email sidecar loads the NPU model anyway |
| "It remembers instead of forgetting — not a summary of your history" | `memory.py`/`memory_store.py` (6.2K lines), post-query hook `agent.py:6575`; but history is a hard ring buffer `deque(maxlen=max_history_length*2)` (`chat/sdk.py:113`, 20 pairs for agents) and #686 (memory-based long-conversation handling) is open with every acceptance box unchecked | ⚠️ half |
| "Skills, not more installs — small signed packages" | Signing (`skills/signing.py`, ed25519) + hub client (`skills/hub.py`) exist; **no workflow publishes the 13 starter skills** (#3090; verified: `grep -l "skill publish\|publish/skill" .github/workflows/*.yml` → nothing); flagship release ships 1 of 12 skills (#3057); daemon has no skill route (`grep skill src/gaia/daemon/sidecars/routes.py` → nothing) | ❌ not deliverable to an installer user today |
| "Embed the agent in your app: `npm install @amd-gaia/gaia`" | `hub/agents/gaia/npm/package.json` name `@amd-gaia/gaia` exists | ✅ (publish state is the CI reviewer's) |
| "No silent cloud fallback" | CLAUDE.md rule; 76 silent-swallow handlers remain in src+hub (coordinator notes); #3315 RAG `query_documents` hides tool errors behind a general-knowledge answer | ⚠️ |

## B.3 Roadmap / plan status judged by code

| Plan | Doc says | Code says |
|---|---|---|
| Memory + long conversations (#542/#543/#686) | v0.20 "planned" | MemoryStore/MemoryMixin **shipped** (memory v2, #606); long-conversation policy (#686) **not started**; 5 open memory bugs (#2831 silent-disable, #2686, #2676, #2865, #3173) |
| Messaging adapters (#635/#693) | v0.23 "Signal P0" | **Only Telegram** shipped (`src/gaia/messaging/telegram.py`, 354 lines). `messaging-integrations-plan.mdx:11` still says "Status: Planning (no implementation)". Signal/Discord/Slack/Teams **not started** (#693, #1002, #1003, #694). `experiments/whatsapp-webjs` is a 72-line spike. |
| Skills ecosystem (#691/#647/#692) | v0.24 "later" | Format, loader, CLI, signing, tiers, marketplace client, OpenClaw migrate **shipped** (`src/gaia/skills/`, 8.8K lines); synthesis (`agents/base/skill_synthesis.py`, 737 lines) shipped. Adaptive skills v2 (#2674–#2682, #2866, #2867) **not started**. Skill publish pipeline **missing** (#3090). Sandbox (#2863) not started. |
| Agent UI v2 thin host (#2014) | epic open | Daemon + sidecar contract shipped; daemon catalog hardcoded to 2 agents (`daemon/sidecars/spec.py:322-344` `builtin_specs()`, filtered at `install.py:151`); TUI (59K Go lines) does hub browse/install/trust/chat; webui (31K TS) is the Electron app. **Two full frontends** against one contract. |
| Installer / cold start (#530/#597) | v0.19 "planned" | One-line installers exist (`scripts/install-ui.*`, `installer/`); flagship installer "mostly built, not yet released" (`flagship-installer.mdx` §9: signing unaddressed, docs drift); #2260 `gaia init → gaia chat` fails from plain PyPI; #2972 npu profile → zero agents; #3290/#3236 uninstall leaves binaries. |
| Eval framework (#573) | v0.18 | Shipped (`src/gaia/eval/`, 28 modules, scorecards, baselines). **The CI gate has never produced a scorecard** (#3016: 0 successes / 62 failures / 124 cancelled since 2026-07-18; #2960 judge key empty on every PR; #2958 80-min email eval blocks releases in report mode; #3294 tool-prompt cost baseline never runs and is failing; #1315 RAG eval gate never ran). |
| Cloud inference for evals / hybrid routing (#632/#1236/#1344) | v0.20 | Judge already rides the Claude Code subscription (`eval/runner.py:936-950`); the agent-under-test has no cloud option in `gaia eval agent`; `use_claude`/`use_chatgpt` exist on `AgentConfig` (`agent.py:858-863`) but scenarios pin Lemonade models. |
| Observability (#697) / config dashboard (#701) | v0.20 | `MemoryDashboard.tsx` + `/api/memory/*` shipped; audit trail (`governance/receipt_service.py`) exists but unwired (#3096); no activity timeline; #2214 decision-trace log open. |
| Autonomy engine (#634/#555) | v0.23 | `gaia schedule` (`src/gaia/schedule/`), daemon (`src/gaia/daemon/`), email autonomy phases 1–5 **shipped for the email agent only**; #2615 TUI has no autonomy control surface; not generalized to the flagship. |
| C++ framework parity (#2791–#2815) | v0.22.5 "OEM" | `gaia-agent-readiness.md` verdict **no-go**: 4 of 6 deps unstarted; #2955/#3127 26 SkillCorpus tests failing on main and the suite dark behind a path filter. |
| CUA (#224/#460), Home Assistant (#705), Vision pipeline (#325), finance/CRM/photo agents | v0.21–v0.24 | **Not started** (plans only). |
| App consolidation (#759–#771) | v0.22 | **Done differently** — per-task agents deleted and collapsed into skills (`hub/agents/README.md`). Roadmap still lists the consolidation tickets as future work. |

## B.4 The 15 most-discussed / highest-signal open issues
Reactions are near-zero repo-wide (max 1), so ranking is by comments + cross-references, filtered to user-facing impact.

| # | Title (abridged) | Why it matters |
|---|---|---|
| #1676 / #1746 | Infinite FLM↔Vulkan model-load loop on NPU (11 comments, epic) | The headline hardware feature thrashes on every prompt for NPU users |
| #1394 / #1382 | GAIA hangs / custom agent not working (10 + 6) | The "Build on it" promise broken for external reporters |
| #2014 | Agent UI v2 thin host epic (8) | Architecture north star; drives daemon/sidecar work |
| #2762 | Email agent validation index (8) | Email is the only autonomous agent; its quality bar is still being defined |
| #124 | Model selection ignored by `gaia talk` (7) | Voice path ignores `--model` |
| #555 / #634 | Autonomous mode / always-on engine (the only issues with reactions) | The "proactive agent" story |
| #2996 | Gmail operator syntax never translated for Graph (4) | Outlook users get wrong search results |
| #2564 | HTTP 409 "chat already in progress" (4, unlabeled) | Users see a raw 409 from the UI |
| #2260 | `gaia init → gaia chat` fails from a plain PyPI install (3) | Cold-start failure for pip users |
| #1344 | Route eval judge through Claude Code subscription (4) | Unblocks evals for maintainers without API keys |
| #1236 / #632 | Cloud LLM backend for RAG synthesis / hybrid routing (4) | Users on weak hardware want an opt-in cloud path |
| #1220 | Native FLM/NPU profile (4) | Same NPU theme as #1676 |
| #2992 | Reports 65,536 ctx while llama.cpp effective is 40,960 (3) | Silent context truncation; docs and reality disagree |
| #3016 / #2960 | Agent eval gate has never produced a scorecard | The CLAUDE.md eval rule has no automated backstop |
| #3057 / #3090 | Flagship ships 1 of 12 skills; nothing publishes skills | The skills story is undeliverable to installer users |

Bug backlog shape (89 `bug`): email agent ~25, CI/eval ~10, TUI 8, memory 5, NPU 5, installer/init 5, daemon/sidecar 5, cpp 3, misc. Most are self-filed by maintainers or the nightly audit bot; the external-user signal is concentrated in #1676, #1394, #1382, #124, #999/#1068 (Linux AppImage), #72 (proxy), #844 (AppImage guidelines).

## B.5 Ranked opportunities

Scoring: user impact × how much already exists ÷ size. "Finish what's half-built" comes first because the marginal cost is lowest and each closes a gap between the website promise and the install.

### A. Finish what's half-built

**A1. Make the eval gate real (CI feedback loop)** — Size **M**, Risk low — **Rank #1**
- *Problem:* CLAUDE.md makes `gaia eval agent` mandatory for LLM-affecting changes, but the automated gate has **never produced a scorecard** (#3016: 0 of 235 runs), the judge key is empty on every PR (#2960), the RAG gate never ran (#1315), the tool-cost baseline test never runs and currently fails (#3294), and the 80-min email eval blocks releases in report-only mode (#2958). A regression like #1030 (Gemma-4 RAG timeout) is only caught if a maintainer runs evals by hand.
- *Exists:* `src/gaia/eval/` (runner, scorecard, scorecard_gate, release_scorecard; baselines in `tests/fixtures/eval_baselines/`), `test_eval_agent_gemma_consolidation.yml`, `weekly_eval.yml`, judge via subscription (`runner.py:936-950`).
- *Missing:* a runner where the embedder loads (`user.embeddinggemma-300m-GGUF` 500s on `sjlab-stx-halo-18`), a working judge secret, a **cloud-inference lane** (A2) so the gate doesn't hinge on one flaky self-hosted box, and a `--compare` step that actually fails the PR.
- *Why #1:* every other quality claim (memory, skills, NPU) is un-regressable until this works, and the fix is mostly ops plus a small runner change.

**A2. Cloud-inference option for evals (agent-under-test, not just the judge)** — Size **S/M**, Risk low — **Rank #2**
- *Problem:* Evals need a Lemonade box; runners without AMD hardware can't run them (#3016; #2122 six workflows fighting over one stx runner). #1344 asks to route the remaining Anthropic paths through Claude Code. A prior Claudia task ("Scope cloud inference for evals", #130) scoped this; nothing landed.
- *Exists:* `AgentConfig(use_claude=…, use_chatgpt=…)` (`agent.py:858-863`), `LLMClientFactory`, judge already on the subscription.
- *Missing:* `gaia eval agent --provider claude|openai` pinning the agent-under-test to a cloud model, a baseline set per provider, and a GitHub-hosted fast lane on every PR while the hardware lane runs nightly. Must stay opt-in per "no silent cloud fallback".
- *Why here:* unblocks A1 without waiting for hardware fixes, and separates "did the prompt change break on a strong model" from "does the 4B model cope".

**A3. Ship the skills story end-to-end (publish pipeline + release staging + daemon route)** — Size **M**, Risk medium — **Rank #3**
- *Problem:* Website: "Skills, not more installs". `hub/agents/README.md`: per-task agents were collapsed into skills. But no workflow publishes `hub/skills/*` (#3090), the frozen flagship ships 1 of 12 skills (#3057), the daemon has no skill route, the TUI has no skills screen (`unified-tui-composability.md` §1, re-verified at HEAD), and `default_skill_set` is commented out in both flagship manifests pending an eval gate (#2848/#2695 → depends on A1). Docs disagree on the count (README says thirteen, the guide says ten — #3088).
- *Exists:* everything client-side — `src/gaia/skills/{format,manager,loader,signing,tiers,hub,install,publish,audit}.py` (8.8K lines), Worker `POST /publish/skill`, `skill_audit.yml` gate, `TrustStore`, the TUI trust gate (`tui/internal/ui/hub/trust.go`) reusable as the skill-consent screen.
- *Missing:* `release_skills.yml` (audit → sign → publish), `freeze.py` staging of `hub/skills/` into the flagship artifact, `GET/POST /daemon/v1/skills`, a TUI list/install screen.
- *Why here:* the capability of ~8 deleted agents is "gone" for anyone who installs instead of clones — the largest promise/reality gap.

**A4. Long-conversation policy on top of memory (#686) + memory reliability** — Size **M**, Risk medium — **Rank #4**
- *Problem:* CLAUDE.md: "no compaction — memory + RAG handles long conversations". The only mechanism today is a hard ring buffer (`chat/sdk.py:113`) plus a one-shot shrink-and-retry on overflow (`agent.py:3950 _shrink_messages_for_overflow`; #2780: recovers 81 of 6,902 tokens). Memory silently disables when the embedding model is missing (#2831), recall rebuilds the prompt head and discards the KV prefix (#2686), recalled procedures are truncated mid-step (#2676), outcome counters never update (#2865), and neither documented way to turn memory on works (#3173). The email agent's overflow (#2763/#2781) is caused by cross-turn history.
- *Exists:* MemoryStore + hybrid FAISS/FTS5/RRF recall (`memory.py:1178`), post-query storage hook (`agent.py:6575`), session consolidation (`memory.py:1598`), memory dashboard.
- *Missing:* the #686 policy itself — before a turn falls off the deque, extract durable facts/decisions into memory and keep a pinned "safety + preferences" block; a token-budgeted history (not pair-count) using the real tokenizer (#2772: estimator is 3–4× off); fail loud when memory can't start.
- *Why here:* touches the flagship's core loop; must be eval-gated (A1), hence not #1 despite high impact.

**A5. NPU path reliability (the hardware differentiator)** — Size **M/L**, Risk high — **Rank #5**
- *Problem:* The most-discussed external bug is the FLM↔Vulkan thrash (#1676, epic #1746); `--profile npu` skips the chat agent so the UI shows zero agents (#2972, p0); the email sidecar loads the NPU model anyway (#3152); NPU context overflow is misrouted to a reload instead of a trim (#2884); the `flm:npu` install request is rejected by current Lemonade (#3175); no NPU eval baseline (#1335, #2090); no auto-detect/prefer-NPU (#1783).
- *Exists:* `NPU_CTX_SIZE`/`GPU_CTX_SIZE` split and `resolve_ctx_size()` (`lemonade_client.py:170-222`), FLM embedder co-residency (#1744, `:412-423`), `gaia init --profile npu`.
- *Missing:* one "device profile" resolved once and honored by every process (agent, sidecar, embedder), a hardware-lane eval baseline, and an integration test that runs from a cold `gaia init --profile npu --force-models` (the #1655 lesson).
- *Why here:* highest external-user pain, but needs hardware and Lemonade coordination, so risk is high.

**A6. Cold-start / installer correctness suite** — Size **M**, Risk low — **Rank #6**
- *Problem:* `gaia init → gaia chat` fails from a plain PyPI install (#2260); Quickstart omits `[rag]`/`[ui]` extras (#1074) and the terminal hub (#3293); the Windows one-line installer's checksum logic is only string-asserted (#3295); `gaia uninstall --purge` leaves binaries (#3290, #3236); flagship installer signing unaddressed (`flagship-installer.mdx` §9.2, #1710); no installer smoke tests on PRs (#936, #989–#992); proxy support requested since #72.
- *Exists:* `src/gaia/installer/` (init, lemonade installer, uninstall, export/import), `scripts/install-ui.{ps1,sh}`, `installer/` NSIS/DMG/AppImage, TUI self-updater, `claude-weekly-doc-walkthrough.yml` (the execution-based audit that found several of these).
- *Missing:* a CI matrix that installs the built artifact on a clean VM per OS and runs `gaia init && gaia chat "hi"` + uninstall; a Lemonade-version compatibility check at init (#3175); SignPath/Apple notarization wiring.
- *Why here:* cheap, deterministic, protects the first five minutes for every new user.

**A7. Governance/audit-trail wiring + observability panel (#3096, #697)** — Size **S/M**, Risk low — **Rank #7**
- *Problem:* Website: "Asks before it acts — every tool call needs your approval". The prompt exists, but nothing durable records approvals/denials — `GovernedAgentMixin` and `receipt_service.py` are referenced nowhere outside `src/gaia/governance/`. #2214 asks for a decision-trace log; #890 asks for a verifiable zero-telemetry property.
- *Exists:* the full governance package (mixin, policy binding, checkpoint bridge, receipt service, schemas), memory dashboard + `/api/memory/*`, TUI confirmation cards, the SSE event stream.
- *Missing:* mix `GovernedAgentMixin` into ChatAgent behind a flag, persist receipts under `~/.gaia/`, expose `/api/audit` + a timeline tab next to the memory dashboard, and a `gaia audit` CLI.
- *Why here:* mostly wiring; turns a security promise into an inspectable property.

**A8. TUI ↔ Agent UI convergence decision (start with a dynamic daemon catalog)** — Size **L** (or **S** for the first step), Risk medium — **Rank #8**
- *Problem:* Two full frontends (Go TUI 59K lines, React/Electron 31K lines) plus the website hub page are maintained against one daemon contract. `unified-tui-composability.md` verdict: "go, but the work is not in the TUI" — the daemon catalog is a hardcoded 2-entry table (`spec.py:322-344`), so neither frontend can show a third agent. TUI gaps: no skills/components/memory screens, connector onboarding leaves the TUI (#2352), 16 hints reference a non-existent `gaia tui` command (#2709). Agent UI gaps: send-confirmation dead-end (#2404), model-management dashboard (#1537), first-run wizard (#597).
- *Exists:* both frontends chat; daemon REST/SSE contract; TUI loopback control API for tests.
- *Missing:* a decision. Cheapest high-value step: make the daemon catalog dynamic (hub index + installed `gaia.agent` entry points) so both frontends inherit new agents for free, then converge rich rendering on `tool_result.render` cards (#2351).
- *Why here:* strategic, not a quick win; but the catalog unblock is S-sized and also unblocks A3.

### B. Net-new

**B1. Second messaging adapter (Slack or Signal) on a shared adapter ABC** — Size **M**, Risk medium — **Rank #9**
- *Problem:* Roadmap v0.23 promised Signal (P0), Telegram, Discord, Slack; only Telegram exists, with no e2e coverage (#2062), a `--background` that exits without polling (#3133), and docs contradicting the allow-list behaviour (#3239). The plan doc still says "no implementation". A prior Claudia task (#104) scoped Slack "as easy as enabling it via the TUI by chatting".
- *Exists:* `TelegramAdapter` with allow-list decorator, streaming edit-in-place replies, PID/log files; `messaging/ingest.py` (media → VLM/RAG); `gaia schedule` sinks (`schedule/sinks.py`) for outbound.
- *Missing:* the `MessagingAdapter` ABC + `MessageRouter` + SQLite session map from the plan; rate limiting (#689); restricted tool set per adapter (#690); one more adapter. Slack Socket Mode is the lowest-friction technically; Signal needs a `signal-cli` (Java) sidecar — heavier packaging.
- *Why here:* real user value (talk to your local agent from your phone) and the autonomy engine needs an outbound channel other than Telegram.

**B2. Opt-in hybrid routing (local-first, cloud on request)** — Size **M**, Risk medium — **Rank #10**
- *Problem:* #632, #1236, #2875 (Atlas Cloud), #1220 — users on weaker hardware want to say "use Claude for this run". The website already frames it correctly: "A remote model is something you switch on for a run, never a default."
- *Exists:* `use_claude`/`use_chatgpt` on `AgentConfig`, `providers/claude.py`, `providers/openai_provider.py`, `LLMClientFactory`; the Claude model-picker (#62) and prompt-caching (#75) Claudia tasks landed pieces.
- *Missing:* a per-session switch in TUI + Agent UI (`/model claude` style), RAG synthesis on the cloud model (#1236), cost display (#649), and an explicit, logged, never-default policy.
- *Why here:* unlocks GAIA on non-Ryzen machines for evaluation and doubles as A2's runtime.

**B3. Voice-first parity (#702): `gaia talk` honors `--model`, mic in TUI/Agent UI** — Size **M**, Risk medium — **Rank #11**
- *Problem:* Roadmap calls voice "P0 enabling technology"; today `gaia talk` ignores `--model`/`--max-tokens`/`--use-claude`/`--stats` (#124, 7 comments — re-verified at HEAD, Part A 🔴), never pauses the mic while it speaks so on laptop speakers it transcribes its own reply (Part A 🔴), has no working barge-in, degraded/streaming paths are untested (#2985), ASR file transcription needs an undocumented ffmpeg (#3180), and no UI has a mic.
- *Exists:* `WhisperASR`, `KokoroTTS`, `TalkSDK` (with the restart command, #3280); audio tests (which don't collect without `sounddevice` — see Part A).
- *Missing:* wiring `--model` through TalkSDK → AgentSDK, Lemonade server-side ASR/TTS (#373/#386) to drop local model downloads, push-to-talk in the TUI.

**B4. Dogfood the GitHub agent on the backlog itself** — Size **S/M**, Risk low — **Rank #12**
- *Problem:* 697 open issues, 54 unlabeled, 56 nightly-audit filings with no owner, only 5 good-first-issues; external users' bugs (#1394, #1382, #999, #1068) sit for months with no state change; `docs/roadmap.mdx` is five months stale.
- *Exists:* `claude-weekly-audit.yml`, `claude.yml`, the GitHub-agent track (#2552–#2563), the `github-triage` starter skill.
- *Missing:* a weekly stale-sweep that closes/merges duplicates, re-labels, and regenerates the roadmap "Shipped" section from release notes so the public roadmap stops contradicting the version number.

### C. Module-level opportunities surfaced by the Part A code review (small, concrete, each unblocks a promise above)

| # | Opportunity | Why it matters | Exists today | Size |
|---|---|---|---|---|
| C1 | **Web fetch that reaches the web** — pinned-IP + correct SNI in `PinnedIPAdapter`, plus a skip-if-offline test against 3–4 CDN hosts | Without it `search_web` results mostly can't be opened, undercutting every research skill (`research-report`, `daily-brief`, `rss-digest`, `source-watch`) and #1146/#1148 | the adapter, the validator, the redirect loop — one method to change | S |
| C2 | **Barge-in + echo suppression for `gaia talk`** — one stdin/VAD interrupt that stops playback, drains queues, pauses/resumes the mic; drop transcriptions that fuzzy-match the last spoken reply | the difference between a demo and a usable voice assistant (#702 P0); most pieces exist unused in `AudioClient.process_voice_input` | pause/resume + interrupt event code paths | S/M |
| C3 | **Persist embeddings in the signed RAG cache** (`.npy`/float16 keyed by `embedding_model`) + `IndexIDMap2` removal | today every process start re-embeds every cached document through Lemonade (`sdk.py:2777-2781`) — minutes of GPU time before the first query on a 100-doc corpus; every LRU eviction is a full re-index | HMAC machinery, `file_embeddings` in memory | S/M |
| C4 | **Embedder-aware chunking** — read the embedder's context length from Lemonade instead of the hard-coded 1,200 chars, size chunks to it | the single biggest retrieval-quality lever in `rag/sdk.py`; makes the docs' tuning guide honest | `_encode_texts`, `RAGConfig` | S |
| C5 | **Query budget for LLM SQL** (timeout + row cap + `truncated` flag) shared by `db_query` and scratchpad `query_data` | turns two hang/flood paths into model-correctable errors | `query_readonly` authorizer | S |
| C6 | **Scratchpad `load_table_from_file`** (CSV/XLSX → `scratch_` table, `allowed_paths`-checked, header sanitised) | today the model must transcribe rows through a 10 MB / 10,000-row JSON call — the bottleneck for the "financial analysis" the mixin advertises; ties to #1499 | `RAGSDK` already parses CSV/XLSX | S/M |
| C7 | **Multi-repo code index** — fix the root ratchet, key the SDK map by `repo_path`, add `stale: true` per hit | the flagship can index exactly one repository per lifetime today; #870 | `CodeIndexSDK`, per-file SHA-256 already in memory | S/M |
| C8 | **Outlook parity for the email agent** — one timestamp normaliser + ~6 operator → OData `$filter` mappings + provider-aware `search_messages` docstring | closes the gap between "Outlook is supported" (README) and search/needs-you/reply-target actually working for the M365 audience (#2996, #2628/#2629) | `outlook_query.py`, `graph_message_to_gmail` | S |
| C9 | **Generic IMAP/SMTP provider** (#2619) | every backend already speaks the Gmail `payload` shape through one decoder; an IMAP backend is `graph_message_to_gmail`'s twin over `email.message_from_bytes`; unlocks every non-Google/Microsoft mailbox | `gmail_backend.decode_message_body` | M |
| C10 | **Hub artifact signing beyond same-origin SHA-256** — detached signature over `manifest.json` with a pinned public key | today `verified` means "checksum served by the same origin as the artifact" (`installer.py:382-393`); a compromised `GAIA_HUB_URL` can serve a matching pair; skills already have ed25519 signing (`skills/signing.py`) to reuse | `TrustStore`, `verify_bundle` | S/M |
| C11 | **Structured-extraction confidence surface** — `parse_ok`/`errors` instead of zero-fill | makes VLM extraction eval-able (today every failure scores as a low result, not an error) | `structured_extraction.py` | S |
| C12 | **C++ P1.2 embeddings + P2.2 RAG on the existing `VectorIndex`/`chunking`; P4.1 HTTP MCP transport** | the gap that keeps every OEM native agent tool-only; stdio-only MCP excludes hosted servers | index, Python-parity splitter, SQLite layer, `HttpClient` all ship | M |

### Explicitly deprioritized (plans exist, no code, low pull)
CUA/desktop control (#224, #460), Home Assistant (#705), personal finance/CRM/photo/meeting agents (#1490–#1502), C++ `gaia-agent` (#2804 — readiness doc says no-go), Docker containers plan, OS-agents MCP servers, Power-Automate Outlook bypass. Each is plan-only with ≤3 comments and zero external reactions; none should displace A1–A6.
