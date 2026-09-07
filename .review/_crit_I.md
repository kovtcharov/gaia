# Adversarial verification — the 99 🟡 findings I1–I99 (§3 of REPORT.md)

**Scope.** Every one of I1–I99, verified against source at `211f08c5`. A completeness pass,
not a sample. Numbering audited first: §3 contains exactly 99 findings, IDs 1–99, no gaps and
no duplicates.

**Method.** I34–I45 (§3.3 servers/daemon/connectors/security — the chunk carrying five 🔒
findings and four of the brief's proposed escalations) I verified myself, re-opening every
cited line and re-running probes. I1–I33 and I46–I99 went to eight parallel sub-verifiers
under the same CRITIQUE_BRIEF rules, each re-deriving its claims from source; their blocks are
merged below in ID order with their section summaries. I spot-audited their sharpest calls
against source (`multipart.ts` really is 91 lines with the regex at `:20`; `find_dotenv` really
does use `os.getcwd()` under `sys.frozen`; the ten fork-PR workflows really carry zero
`head.repo` gates; `pypa/gh-action-pypi-publish@release/v1` really is a branch) and every one
held.

The repo was read-only throughout. Probes ran from `%TEMP%` with `.venv\Scripts\python.exe`;
nothing was written into the tree; no server was started; no eval was run.

**One claim about the report itself, checked and withdrawn.** Appendix A says "~40 findings
were additionally reproduced by executing a probe". Counting the literal `*probe` marker gives
only 16, which looked like an overclaim — but 38 findings mention a probe in some form. The
claim is fair; I record the check so nobody re-runs it.

---

## §3.1 Agent framework, tools, memory, skills (I1–I16)

# Adversarial verification — REPORT.md §3.1 findings I1–I16

Checkout: `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f` @ 211f08c5 (verified `git log --oneline -1`, worktree clean except untracked `.review/`).
Probes run with `.venv\Scripts\python.exe`, scripts in `%TEMP%`. No repo file touched.

---

### I1 CONFIRMED-ADJUST
- Evidence: `tools.py:79` `_TOOL_REGISTRY[tool_name] = {…}` unconditional (no collision check); `profiles.py:45-50` `file_fs` = file→filesystem→file_search→file_io; three `read_file` (`file_tools.py:572`, `filesystem_tools.py:773`, `file_io_tools.py:38`), two `browse_directory` (`file_tools.py:1488` `directory_path`, `filesystem_tools.py:124` `path`), two `search_web` (`browser_tools.py:188` DDG vs `gaia_agent_chat/agent.py:2127` Perplexity, registered *after* the groups at `agent.py:1307`). **Probe** (registrars called in profile order): `read_file -> agents\tools\file_io_tools.py:37 ['file_path']`, `browse_directory -> agents\tools\file_tools.py:1466 ['directory_path','show_hidden','sort_by']` — last-wins confirmed exactly as claimed. Prompt claim `search_web (DuckDuckGo, no key)` is real (`agent.py:970`).
- Edit: **(a)** The size cap on the losing `filesystem.read_file` is **50 MB**, not 10 MB (`filesystem_tools.py:27 MAX_READ_BYTES = 50 * 1024 * 1024`) — change "no 10 MB cap" → "no 50 MB cap". **(b)** Drop / soften *"the prompt describes `browse_directory(path=…)` … → 'Unexpected argument(s): path' on every browse"*: the FILE SYSTEM TOOLS block (`agent.py:945`) names the tool but **not** its parameters, and `_format_tools_for_prompt` (`agent.py:1194`) renders the *winner's* signature, so the model is shown `directory_path` and the "every browse fails" scenario is unsupported. The real, verified harm is the other three: sandboxed+capped `read_file` replaced by the uncapped `file_io` one that dereferences `self.path_validator` unguarded (#3316), and DDG→Perplexity silent paid/exfil swap contradicting the prompt.
- Severity: keep 🟡 (the Perplexity swap is a quiet egress change; the `read_file` swap is a real sandbox weakening — but both need the specific profile/env, and neither is a direct exploit).
- Dup: overlaps `01-core-agent.md:138-142` (listing tools bypass `path_validator`) — I1 should cross-reference it rather than restate the cap loss.

### I2 CONFIRMED-ADJUST
- Evidence: `gaia_agent_chat/agent.py:1494-1517` `open_url` and `:1519-1564` `fetch_webpage` — only `url.startswith(("http://","https://"))`, then `httpx.get(url, timeout=15, follow_redirects=True)`; no DNS resolution check, no redirect re-validation. `src/gaia/web/client.py:45-74, 327-365, 458` implements exactly that guard (`ip.is_private`, per-redirect re-validation, rebind note) for `WebClient.fetch_page`. Gated on `spec.web_tools` → profiles `web` and `full` (`profiles.py:135-166`) — the flagship is `prompt_profile="full"` (`hub/agents/gaia/python/gaia_agent/agent.py:140`), so this is on by default in the shipped agent.
- Edit: fix the line range to `agent.py:1519-1564` (fetch_webpage) and `:1494-1517` (open_url); add the attacker position sentence: *"reachable via indirect prompt injection — an indexed document or fetched page instructing the model to fetch a loopback/metadata URL — not by a remote request alone."*
- Severity: **change 🟡 → 🔴.** REVIEW.md puts security in 🔴, and the codebase itself treats private-IP fetching as a threat worth a dedicated guard in `web/client.py`; these two tools are registered in the flagship's default profile and re-open it. The daemon loopback API and `169.254.169.254` are both reachable.
- Dup: none found. `gh issue list -R amd/gaia --search "SSRF fetch_webpage"` → no results; untracked.

### I3 CONFIRMED
- Evidence: `agent.py:6455-6488` — `if steps_taken == steps_limit and final_answer is None:` … `has_stdin = sys.stdin and sys.stdin.isatty()`; `if has_stdin and not (hasattr(self,"silent_mode") and self.silent_mode): … input("\nContinue with 50 more steps? (y/n): ")` (`:6472`). `src/gaia/ui/_chat_helpers.py:494` `kwargs: dict = {"silent_mode": not streaming, "debug": False}` → streaming agents get `silent_mode=False`. `gaia chat --ui` runs uvicorn in the foreground, so `stdin.isatty()` is True. Cited lines are exact.
- Edit: none (the report already flags it as not reproduced live; the "(Medium confidence)" qualifier is warranted and should stay).
- Severity: keep 🟡 — it fires only when a streaming UI turn actually hits `max_steps` (50), an edge path, but when it does the SSE request hangs on the server operator's terminal indefinitely.
- Dup: none.

### I4 CONFIRMED — propose 🔴
- Evidence: `shell_tools.py:569-579` (git: subcommand-only check against `SAFE_GIT_COMMANDS`), `:580-592` (wmic: blocks only the words `call/create/delete/set`), vs `:669-695` (`sort -o/--output` explicitly refused, incl. GNU abbreviations) and `:697-725` (`uniq` output operand refused). **Probe** through the real validator (`ShellToolsMixin._validate_command`, reached from `_validate_shell_command:286-338` for every pipeline segment): `ALLOWED | git log --output=…/pwn.txt -1`, `ALLOWED | wmic /output:C:\Temp\pwn.txt os get caption`, `BLOCKED | sort -o out.txt in.txt`. **Second probe** in a throwaway `%TEMP%` repo: `git log --output=out.txt --format="ARBITRARY-CONTENT-%H" -1` → exit 0, `out.txt` contains `ARBITRARY-CONTENT-3388…`.
- Edit: strengthen the claim — this is not just "writes files", it is a **near-arbitrary-content file write**: `--format=` is fully model-controlled, so the written bytes are chosen by the caller, not merely leaked git output. Add the `--format` half to the finding and to the fix (refusing `--output`/`-o` is sufficient; no need to touch `--format`).
- Severity: **change 🟡 → 🔴.** An arbitrary-path, arbitrary-content write out of a policy the code calls "read-only" is a sandbox escape (drop a `.bat` in Startup, overwrite a config). It needs no user click once shell tools are registered.
- Dup: partially tracked — `gh issue list -R amd/gaia --search "shell allowlist output write"` → **#2768 OPEN "Audit ALLOWED_COMMANDS for remaining write/exec side-doors (sysctl -w, date -s)"**, same class. Add `Tracked: #2768 (same class, these two instances not listed).`

### I5 CONFIRMED
- Evidence: `chat/sdk.py:100-110` passes `system_prompt=self.config.system_prompt` into `create_client`; `providers/lemonade.py:311` stores it and `:355-357` prepends `{"role":"system", …}` on every `chat()`; `sdk.py:495-539` `send()` builds `full_prompt` via `Prompts.format_chat_history(..., system_prompt=self.config.system_prompt)` (`chat/prompts.py:89-107`), which embeds it in the model's `<start_of_turn>system` template, and hands that whole string to `generate()` (`providers/lemonade.py:322-334` → one `user` message). **Probe** (fake `LemonadeClient`, real `AgentSDK`+`LemonadeProvider`): `roles: ['system','user']`, `count of SYS-ONE across request: 2` — matches the report exactly. **Second half also reproduced:** after `sdk.update_config(system_prompt="SYS-TWO")` (`sdk.py:839`) the next request contains `SYS-ONE: 1  SYS-TWO: 1` — the provider keeps the stale prompt and the two now contradict each other.
- Edit: none required. Optionally sharpen: after `update_config` the model receives *both* the old and the new system prompt in the same request, which is worse than "leaves the old one in place".
- Severity: keep 🟡 — wasted context and a contradictory prompt on a public SDK surface, but no data loss or security impact.
- Dup: none.

### I6 CONFIRMED-ADJUST
- Evidence: `sdk.py:1168-1172` — `except Exception as e: self.log.warning(f"RAG enhancement failed: {e}, falling back to direct query"); return message, {"rag_used": False, "error": str(e)}`. The caller discards it: `sdk.py:514` `enhanced_message, _rag_metadata = self._enhance_with_rag(...)` and `_rag_metadata` is never read in `send()` or `send_stream()` (`grep _rag_metadata` → only the two assignments), so `AgentResponse` carries no signal. History desync: `sdk.py:526` (`send`) and `sdk.py:600` (`send_stream`) append `f"user: {message}"` *before* `llm_client.generate`; the handlers at `:577-579` / `:648-650` log and `raise` with no pop, leaving a dangling `user:` turn in `chat_history` for the rest of the session. Cited line numbers are exact.
- Edit: replace "silently falls back" with "logs a `warning` but returns a normal answer — the `{'rag_used': False, 'error': …}` metadata is discarded by `send`/`send_stream` (`_rag_metadata` is never read), so nothing reaches the caller or the user." That is the accurate, and stronger, statement.
- Severity: keep 🟡. Directly violates the repo's own "No Silent Fallbacks — Fail Loudly" rule in CLAUDE.md, which is worth naming in the finding.
- Dup: report already notes #3315 is the tool-layer twin — correct; keep.

### I7 CONFIRMED-ADJUST
- Evidence: `file_tools.py:2598` `def list_recent_files(..., max_results: int = 20, …)`; the return block at `:2751-2762` sends `"files": recent_files[:max_results]` **and** `"all_files": recent_files` (`:2756`, uncapped) **and** a `display_message` that enumerates every extra file inside a `<details>` block (`:2745-2750`). The comment at `:2732-2733` states the intent outright: *"Return all files — first batch shown directly, rest in a collapsible section so the LLM doesn't truncate them."*
- Edit: fix the line cite `file_tools.py:2733-2762` → `file_tools.py:2598` (signature) and `:2732-2762` (the deliberate uncapped return; quote the `all_files` key and the comment). The finding is stronger with the comment quoted — it is an intentional design, not an oversight, so the fix needs a decision, not a patch.
- Severity: keep 🟡 — needs a large recent-file set (OneDrive/Documents) plus the 32K NPU profile to actually overflow; not a normal-use break on a 64K GPU box.
- Dup: none.

### I8 CONFIRMED-ADJUST
- Evidence: `skills/install.py:418-422` compares exactly `bundled_skill.name`, `.security_tier`, `sorted(.gaia.permissions)` — nothing else; `version` is a parameter used only in the message text. The install-time text it can misdescribe is at `:365-373`: `code_warning` branches on `skill.gaia.tools` (the **R2-served, unsigned** copy — the docstring at `:403-408` says "The signature covers the bundle's copy; the R2 object is what we parse"), so a bundle whose own SKILL.md declares `metadata.gaia.tools` can be described to the user as *"instruction-only: it ships no code"*.
- Edit: add the gating the report omits — that message is only rendered on the **experimental-tier refusal path** (`if tier == LOWEST_TIER and not allow_experimental`, `:357`), i.e. it is part of the error that *blocks* the install and is only seen before the user re-runs with `--allow-experimental`. And add the sharper framing: the risk description shown to the user is computed from the **unsigned** manifest while the **signed** bundle is what actually loads.
- Severity: keep 🟡 (it is a consistency gap in a decision prompt, gated behind an explicit `--allow-experimental` opt-in — not a bypass of the signature itself, since name/tier/permissions *are* compared).
- Dup: none.

### I9 OVERSTATED
- Evidence: `tool_grants.py:45-79` `_UNBOUNDED_BINARIES` — the seven names the report lists (`python3.12`, `pythonw`, `ipython`, `pip`, `wsl`, `ssh`, `docker`) are indeed absent. **But the grant is unreachable for all of them.** `agent.py:3428-3444` — "Validate first, confirm second: a call the guardrails already refuse must never reach a prompt" — runs `_policy_refusal` (`shell_tools.py:361-373` → `_validate_shell_command`) **before** `confirm_tool_execution`, and `grant_scope` is only ever called from the confirmation UI (`console.py:14,425,721`, `sse_handler.py:25`). None of those seven binaries is in `ALLOWED_COMMANDS` (`shell_tools.py:30-90`: cat/head/grep/find/git/wmic/powershell/… only) or in `BINARY_POLICIES` (`skills/binaries.py:387,437` — exactly `gh` and `pytest`), so the call is refused before any prompt and no "always allow" is ever offered. The one Windows name that *is* allowlisted, `powershell.exe`, is handled: `_binary_name` (`tool_grants.py:186-193`) strips `.exe/.cmd/.bat/.com/.ps1` and `powershell`/`pwsh` are both in the set. `run_cli_command` (the other member of `_SHELL_TOOLS`) is a dead name — `grep run_cli_command src/ hub/` returns only `agent.py:200` and `tool_grants.py:87`; no tool is registered under it. `execute_python_file` — the one real arbitrary-code runner — is in neither `_SHELL_TOOLS` nor `_PATH_TOOLS`, so `grant_scope` returns `None` and "always" is correctly not offered.
- Edit: rewrite as a latent / defence-in-depth gap, not a live hole. Suggested: "`_UNBOUNDED_BINARIES` (`tool_grants.py:45-79`) omits `python3.12`, `pythonw`, `ipython`, `pip`, `wsl`, `ssh`, `docker`. None is reachable today — the shell allowlist refuses them before the confirmation prompt that would offer the grant — but the list is the only thing standing between a future `ALLOWED_COMMANDS`/`BINARY_POLICIES` addition and a session-wide arbitrary-code grant. Fix: match by regex/prefix so new runners are covered by default." Delete the claim "'always allow' can be offered for an arbitrary-code runner" — it is false at this commit.
- Severity: **change 🟡 → 🟢.** Under REVIEW.md's definitions this cannot today break or mislead a user; it is hardening for a change that has not happened.
- Dup: none.

### I10 CONFIRMED
- Evidence: `memory.py:1393-1402` — `with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:` … `future.result(timeout=EXTRACTION_TIMEOUT_S)`; the `with` exit calls `shutdown(wait=True)`, so the block does not return until `_call_llm` finishes. `EXTRACTION_TIMEOUT_S = 8` (`memory.py:189`) while the docstring says `Timeout: 3s.` (`:1346`) and the inline comment says `(spec: 3s)` (`:1384`). The caller is synchronous and on the turn's critical path: `_after_process_query` (`memory.py:2432`, extraction at `:2471`) is invoked from `agent.py:6576-6578`, before `process_query` returns. **Probe** (`%TEMP%`, py 3.13.11): 3 s task with `timeout=0.5` → `TimeoutError raised at 0.51s`, `elapsed after with-block exit: 3.00s` — reproduces the report's numbers exactly.
- Edit: none. Optionally add: the result is not merely discarded — the whole turn's return is delayed by the *full* LLM latency, so the timeout buys nothing at all.
- Severity: keep 🟡 — latency plus a doc/constant contradiction, no incorrect output.
- Dup: none.

### I11 CONFIRMED — propose 🔴
- Evidence: `memory.py:1431-1441` — an `add` op is dropped unless `op["category"] in EXTRACTABLE_CATEGORIES`, with the comment "Privileged category (system/profile/permission): only explicit tools may write these — never the extractor." `memory.py:1442-1443` — `elif op_type == "update" and "knowledge_id" in op and "content" in op: valid_ops.append(op)` — **no category check**. `memory.py:1507-1521` then calls `store.store(category=op.get("category", …), …, source="llm_extract")`; `existing_item` defaults to `{}` when the `knowledge_id` matches nothing (`:1509-1511`), so even a fabricated id still creates the row. `memory_store.py:751-782` `store()` validates content and `due_at` only — never `category`. The invariant it breaks is written down at `memory_store.py:110-118`: "A chat turn must not be able to mint a permission grant, a system fact, or a profile entry by emitting that category." Impact confirmed at `memory.py:2011-2031`: `_build_stable_memory_prompt` renders `system` (section 0a) and `profile` (0b) **first**, at `context="global"`, in every future system prompt. The `remember` tool has the wider gap the report claims: `memory.py:2550-2553` validates against `VALID_CATEGORIES`, not `EXTRACTABLE_CATEGORIES`, so the model can write `profile` / `system` / **`permission`** directly by tool call.
- Edit: add the `permission` category to the impact sentence — it is the autonomous-mode approval grant (`memory_store.py:101-107`), so this is not only persistent prompt injection but a self-granted approval. Name the `remember` gap precisely (`VALID_CATEGORIES` where `EXTRACTABLE_CATEGORIES` was intended) rather than "same gap".
- Severity: **change 🟡 → 🔴.** A single conversation turn — including one driven by indirect prompt injection from an indexed document — writes a row that is prepended to every subsequent system prompt and can mint a permission grant. That is a security boundary the code documents and then does not enforce.
- Dup: none.

### I12 CONFIRMED
- Evidence: `memory.py:1221-1236` — `candidate_pool = store.get_items_with_embeddings(..., top_k=max(oversample * 2, 200), …)`, then `pool_by_id = {…}` and `for kid, _score in faiss_hits: item = pool_by_id.get(kid); if item is not None: vector_results.append(item)` — a FAISS hit outside the pool is silently dropped. `memory_store.py:1589-1590` — `ORDER BY confidence DESC, updated_at DESC LIMIT ?`. Rich-get-richer confirmed: the pool is the 200 highest-confidence rows, not the 200 nearest.
- Edit: none; cited lines are exact.
- Severity: keep 🟡 — degrades recall quality on a large store; no crash, no visibly wrong output.
- Dup: none.

### I13 CONFIRMED (all four sub-claims)
- Evidence: **(a)** `memory.py:1986` `if not hasattr(self, "_memory_store"): return ""` — but the disabled paths set `self._memory_store = None` (`:437`, `:562`), so `hasattr` is True, `_build_stable_memory_prompt` dereferences `None`, and `:1990-1991` logs `warning("failed to build stable memory prompt: …")` on every prompt build. The sibling `get_memory_dynamic_context` (`:2003`) gets it right with `getattr(self, "_memory_store", None) is None` — the asymmetry sits four lines apart in one file. **(b)** `memory.py:3033-3034` `if hasattr(self, "_memory_store"): self._memory_store.apply_confidence_decay()` → `AttributeError` on a memory-disabled agent. **(c)** `memory.py:2661` docstring example `recall(time_from='2026-01-01', time_to='2026-03-31')  → entries from Q1`; the no-query branch (`:2761-2783`) filters `if time_to and created > time_to: continue`, and `created_at` is a full ISO timestamp. **Probe**: `created="2026-03-31T14:22:05"` → `time_to='2026-03-31': 0 kept`, `time_to='2026-04-01': 1 kept` — the tool's own docstring example returns nothing. Same bug in the SQL path (`memory_store.py:1579-1581` `created_at <= ?`) and in `_hybrid_search` (`memory.py:1248-1256`). **(d)** `goal_store.py:8, 14-27` draws two explicit state machines in the module docstring; `update_task_status` (`:497-517`) and `update_goal_status` (`:419-431`) issue a bare `UPDATE … SET status=?` with no transition check, and `GoalStatus`/`TaskStatus` are `typing.Literal` (`:48`, `:58`) — erased at runtime.
- Edit: split (c) out of the "misc" bucket into its own finding, and cite all three places, not one (`memory.py:1254`, `memory.py:2782`, `memory_store.py:1580`). It is the only one of the four that silently returns **wrong data** through a documented tool call.
- Severity: keep 🟡 for the remaining bucket; the extracted (c) is at least 🟡 and arguably 🔴 — the docstring example is the call shape the model copies, so date-only `time_to` is the *normal* case, and it silently drops the whole final day.
- Dup: none.

### I14 CONFIRMED-ADJUST
- Evidence: `hub/agents/gaia/python/gaia_agent/server.py:369-373` — `base = (os.environ.get("LEMONADE_BASE_URL") or getattr(GaiaAgentConfig(), "base_url", None) or "http://localhost:13305").rstrip("/")`; `:384` — `requests.get(f"{base}/api/v1/models", timeout=5)`. With the documented `/api/v1`-suffixed value the probe URL is `http://localhost:13305/api/v1/api/v1/models` (computed directly), the request 404s, the bare `except Exception` at `:398` returns `reachable=False`, and `/init` (`:439-472`) renders `ready=False` with "Local Lemonade Server is not reachable at …". The normaliser the report names exists and is correct: `src/gaia/agents/base/readiness.py:223-238 resolve_probe_base` (tested at `tests/unit/test_agent_readiness.py:139-147`), and the email agent's re-export `_resolve_probe_base` (`hub/agents/email/python/gaia_agent_email/model_select.py:53-70`) — the flagship simply does not call either. ChatAgent's own default is the `/api/v1` form (`gaia_agent_chat/agent.py:1468`), which is what makes the mismatch likely rather than exotic.
- Edit: **delete "and reads a non-existent config field".** `GaiaAgentConfig` really does have `base_url` — inherited from `ChatAgentConfig` (`gaia_agent_chat/agent.py:75 base_url: Optional[str] = None`); I constructed it and confirmed `hasattr(c,'base_url') == True`. The `getattr(..., None)` is defensive but not wrong, and it is not part of the bug. Also fix the line cite to `server.py:369-373` (base) + `:384` (request) and name the endpoint `GET /v1/gaia/init` → the handler is `@router.get("/init")` at `:439`.
- Severity: keep 🟡 — a real, user-facing false negative, but only for a non-default `LEMONADE_BASE_URL`.
- Dup: `gh issue view 3203` → **OPEN**, "LEMONADE_BASE_URL needs a bare origin, but the GAIA CLI documents the /api/v1-suffixed form - the suffixed value produces a false 503" — the report's `Tracked: #3203` is correct. Also the same root cause as **I23** (`.env.example` shipping the `/api/v1` form) — the report should cross-link them; fixing #3203 without fixing this probe leaves the flagship broken.

### I15 OVERSTATED (first half) / CONFIRMED (second half)
- Evidence: **First half** — the mechanism is real: `discovery.py:3872-3877` `if sources is None: sources = all_sources else: # Validate source names / sources = [s for s in sources if s in all_sources]` — a typo'd name is dropped with no error and no log (the loop at `:3898-3905` then just never runs it). **But the failure scenario is not reachable:** `grep -rn -- "--sources" .` over the whole repo (py/md/mdx) returns **nothing** — there is no `gaia memory bootstrap --sources` flag — and the only caller in-tree is `src/gaia/cli.py:5800 discovery.scan_all()` with **no** `sources` argument. So this is an SDK-surface silent-drop, not a CLI misuse anyone can hit today. Cited line `3888-3891` is also wrong by ~11 lines. **Second half CONFIRMED** — `gaia_agent/session_registry.py:276-283` `delete()` pops under `self._lock` then calls `close_agent(session.agent)` with no `run_lock` claim, while both eviction paths do claim it (`:228-247`, `:250-270`) and the module comment at `:129` states the invariant ("never while a session's `run_lock` is held"). `grep` for callers → only `hub/agents/gaia/python/tests/test_session_registry.py:64,72`; the `agent_routes.py:256` hit is the *email* agent's separate registry class. "No production caller yet" is accurate.
- Edit: rewrite the first half. Suggested: "`SystemDiscovery.scan_all(sources=[…])` (`discovery.py:3872-3877`) silently drops any name not in `all_sources` — no error, no log — so an SDK caller that typos a source gets a successful-looking empty result. No CLI path reaches it today (`cli.py:5800` calls `scan_all()` with no sources, and no `--sources` flag exists), so this is a latent SDK trap, not a live bug." Delete the invented `gaia memory bootstrap --discover --sources browser_histroy` example; and fix the line cite `3888-3891` → `3872-3877`.
- Severity: split — first half **🟡 → 🟢** (unreachable, latent); second half keep 🟡 (a real lock-invariant violation, correctly flagged as having no production caller). If the report keeps them as one bullet, 🟡 is defensible on the second half alone; the fabricated CLI example must go either way.
- Dup: none.

### I16 CONFIRMED-ADJUST
- Evidence: `src/gaia/agents/base/agent.py:105 DEFAULT_MAX_STEPS = 50` (with `GAIA_AGENT_MAX_STEPS` override resolved at `:113-129`). `docs/sdk/core/agent-system.mdx:785` — `def process_query(self, user_input, max_steps=None):  # defaults to self.max_steps (20)` — the only place that states a **default**, and it is wrong. The walkthrough divergence is real: the doc's `for step in range(max_steps): … return {"status": "incomplete", …}` (`:797-827`) has no counterpart to the live loop's max-steps continuation prompt (`agent.py:6455-6488`, which raises `steps_limit` by 50 on a `y`).
- Edit: **(a)** fix the source cite `agent.py:113` → **`agent.py:105`** (`:113` is inside the env-var resolver's docstring, not the constant). **(b)** Narrow the doc cites: `:785` is the false default claim; `:365` is an example config block and `:465` a "quick vs. complex" recommendation table — both reinforce the wrong number but neither asserts a default, so calling all three "docs say the default is 20" overstates by two. Reword to "`agent-system.mdx:785` states the default is 20; `:365` and `:465` repeat 20 as the example/recommended value, so nothing in the doc hints at 50."
- Severity: keep 🟡 — doc contradicting code, exactly REVIEW.md's 🟡 definition.
- Dup: none.

---

## Summary (I1–I16)

### Counts per verdict
| Verdict | Count | IDs |
|---|---|---|
| CONFIRMED | 7 | I3, I4, I5, I10, I11, I12, I13 |
| CONFIRMED-ADJUST | 7 | I1, I2, I6, I7, I8, I14, I16 |
| OVERSTATED | 2 | I9, I15 (first half; its second half is CONFIRMED) |
| REFUTED | 0 | — |
| UNVERIFIABLE | 0 | — |

(7 CONFIRMED · 7 CONFIRMED-ADJUST · 2 OVERSTATED · 0 REFUTED · 0 UNVERIFIABLE.)

Every claim that hinged on runtime behaviour was re-run: I1 (registry collision winners), I4 (shell validator + a real `git log --output` write), I5 (`count of SYS-ONE across request: 2`), I10 (`elapsed after with-block exit: 3.00s`), I13c (`time_to='2026-03-31': 0 kept`), I14 (double-suffixed probe URL). All reproduced the report's stated results.

### The corrections that matter most
1. **I9 is not a live hole — downgrade to 🟢.** Validation runs *before* the confirmation prompt (`agent.py:3428`), and none of the seven missing binaries is in `ALLOWED_COMMANDS` or `BINARY_POLICIES`, so no "always allow" is ever offered for them. `powershell.exe` — the one that *is* allowlisted — is already covered via `.exe` stripping. The sentence "'always allow' can be offered for an arbitrary-code runner" is false at this commit and must go.
2. **I15's failure scenario is fabricated — delete it.** There is no `gaia memory bootstrap --sources` flag anywhere in the repo, and the only in-tree caller passes no `sources` at all. The silent-drop mechanism is real but latent (and the line cite is off by 11). The second half (`session_registry.delete()` skipping `run_lock`) is fully confirmed and carries the bullet.
3. **I14 contains a false sub-claim — delete "reads a non-existent config field".** `GaiaAgentConfig.base_url` exists (inherited from `ChatAgentConfig`, default `None`); I constructed the object to check. The double-`/api/v1` bug is entirely real without it.
4. **I1's blast-radius claim is half wrong.** The size cap on the losing `read_file` is 50 MB, not 10 MB, and the "'Unexpected argument(s): path' on every browse" scenario is unsupported — the prompt never names `browse_directory`'s parameters and `_format_tools_for_prompt` renders the *winner's* signature. The `read_file` downgrade and the silent DuckDuckGo→Perplexity swap are the real, verified harms.
5. **I13(c) is buried.** Date-only `time_to` silently dropping the entire final day is the only sub-claim that returns *wrong data* through a documented tool call — and the docstring's own example (`memory.py:2661`) is the broken shape. It appears in three places (`memory.py:1254`, `memory.py:2782`, `memory_store.py:1580`) and deserves its own finding, not a slot in a four-item "misc" line.

### Proposed severity changes
| ID | Now | Proposed | Why |
|---|---|---|---|
| I2 | 🟡 | **🔴** | SSRF: `fetch_webpage`/`open_url` are registered in the flagship's default `full` profile and re-open exactly what `web/client.py:45-74` was written to close (loopback, `169.254.169.254`, redirect-to-private). Attacker position: indirect prompt injection from an indexed doc or fetched page. Untracked (`gh` search → nothing). |
| I4 | 🟡 | **🔴** | `git log --output=<path> --format=<anything>` is a **near-arbitrary-content, arbitrary-path file write** through a policy the code calls read-only (probe-verified). Needs no click once shell tools are registered. Class partially tracked by **#2768** (add that). |
| I11 | 🟡 | **🔴** | An `update` op — or a `remember` tool call — mints `system` / `profile` / **`permission`** rows that `_build_stable_memory_prompt` renders *first* in every future system prompt, breaking an invariant `memory_store.py:110-118` writes down verbatim. Persistent prompt injection + self-granted autonomous-mode approval. |
| I9 | 🟡 | **🟢** | Unreachable at this commit; defence-in-depth for a change that has not happened. |
| I15 (first half) | 🟡 | **🟢** | No CLI surface, no in-tree caller passing `sources`; latent SDK trap only. Second half stays 🟡. |
| I13(c) | (inside 🟡 misc) | **own 🟡, arguably 🔴** | Wrong data through the documented call shape; "fires in normal use" is the 🔴 test. |

---

## §3.2 LLM layer, installer, CLI (I17–I33)

# Adversarial verification — I17–I33 (section 3.2, LLM layer / installer / CLI)

Verifier B. Checkout 211f08c5, read-only. Python: `.venv\Scripts\python.exe`.

### I17 CONFIRMED
- Evidence: `lemonade_manager.py:729-732` (`if context_size_value == 0 and not llm_models_loaded: cls._try_preload_with_ctx(...)`) → `:797,826,843-847` (`client.load_model(DEFAULT_MODEL_NAME, ctx_size=…, auto_download=True)`); `init_command.py:1660` `if self.profile not in ("sd","npu")` vs `:1955` `if self.profile != "sd"`; `INIT_PROFILES["npu"].min_context_size = 32768` (`:140`) so `_verify_setup:1933` does call `ensure_ready` on npu.
- Edit: line numbers — `init_command.py:1667-1670` → **`:1660-1664`**; `lemonade_manager.py:729-733` → `:729-732`. Also note the verify step is behind a `Run model verification?` prompt (default **yes**), so it fires on a default `gaia init`.
- Severity: keep 🟡 — wasted ~3 GB download + a wrong-backend load, not corruption.
- Dup: none.

### I18 CONFIRMED
- Evidence: `init_command.py:92,102,140,154` all `"min_context_size": 32768`; `:1792-1798` `client.load_model(model_id, auto_download=False, prompt=False, ctx_size=min_ctx, save_options=True)`; `lemonade_client.py:173 GPU_CTX_SIZE = 65536`, `:334 min_ctx_size=GPU_CTX_SIZE` for `gemma-4-e4b`, `:3226-3232 + 3249-3259` reload when `loaded_ctx < expected_ctx`; `tests/unit/installer/test_init_ctx_size.py:116-122` pins the literal `"32768"`.
- Edit: cite range `:1791-1798` (not `1795-1800`) for the `save_options=True` load; add `:140` (npu, where 32768 is *correct*) to the profile list so the fix scope is obvious.
- Severity: keep 🟡 — one slow first turn, no wrong answers.
- Dup: none.

### I19 CONFIRMED-ADJUST
- Evidence: `lemonade_client.py:4155-4175` — `check_model_loaded` calls `list_models()` → `GET {base}/models` (docstring `:2409-2411`: "returns only **downloaded** models") and additionally substring-matches; `init_command.py:1788-1789` `if client.check_model_loaded(model_id): client.unload_model()` with `unload_model(model_name=None)` documented at `:3718-3730` as "unload all models (global)".
- Edit: the consequence clause is order-dependent — for `chat`/`rag` the LLM is verified **before** the embedder (`INIT_PROFILES[...]["models"]` lists the GGUF first), so nothing is evicted there; what is really burned is an unload+reload of the model `ensure_ready` just preloaded. The eviction case is the `all` profile, whose model order comes from `list(set(...))` (`lemonade_client.py:4041-4057`) and is therefore arbitrary. Reword to: "→ init's verify step globally unloads whatever is resident before each LLM test (a wasted cold reload of the model `ensure_ready` just preloaded; on the `all` profile, whose model order is a `list(set(...))`, it can also evict the embedder)."
- Severity: keep 🟡.
- Dup: none.

### I20 CONFIRMED
- Evidence: `providers/lemonade.py:243-244` `if 0 < n_ctx_reported < 65536: err_instance.retryable = True` (comment at `:230` admits the literal tracks the chat/rag profile); `ui/_chat_helpers.py:241-242` same literal with the comment "Threshold tracks the chat / rag profile default (65536)"; `:1608-1625` retry path (`_maybe_load_expected_model(effective)` then re-`process_query`). `lemonade_client.py:174 NPU_CTX_SIZE = 32768`, `:359 min_ctx_size=NPU_CTX_SIZE` for the FLM model → the reload is a genuine no-op on NPU.
- Edit: none (cited ranges 237-245 / 236-244 / 1610-1625 all contain the quoted code).
- Severity: keep 🟡 — misleading message + doubled latency on an edge path, no data loss.
- Dup: none.

### I21 CONFIRMED
- Evidence: `vlm_client.py:101-110` — `parsed = urlparse(base_url)`, `host = parsed.hostname`, `port = parsed.port or 13305`, `self.server_url = f"http://{host}:{port}"`, `LemonadeClient(model=…, host=host, port=port, …)`; `lemonade_client.py:995-999` — the `if host is not None or port is not None:` branch builds `f"http://{self.host}:{self.port}/api/{LEMONADE_API_VERSION}"`, discarding scheme and path; reached from `providers/lemonade.py:505`.
- Edit: none (line numbers hold).
- Severity: keep 🟡 — only bites the remote/reverse-proxy `LEMONADE_BASE_URL` setup, which the chat path explicitly supports.
- Dup: none. Note it compounds with I22: the resulting connection error is returned as page text.

### I22 CONFIRMED
- Evidence: `vlm_client.py:203-207` (VLM-not-loaded → `return f"[VLM extraction failed: {error_msg}]"`), `:281-282`, `:290-291`; `:169-182` `_ensure_vlm_loaded` logs and `return False`. Repo-wide grep for `"VLM extraction failed"` outside `vlm_client.py`: **zero** callers check the prefix. It flows into `extract_from_page_images:354-368` → `rag/sdk.py:825, 1186` (indexed as page content), `vlm/mixin.py:162,205`, `providers/lemonade.py:505` (returned as the vision answer), `messaging/ingest.py:30`. `structured_extraction.py:306 return []`, `:445 return {}`, `:469 return 0.0`, and `:200-231` sums the zeros into `result["aggregated_data"]["timeline_totals"]`; `tests/unit/test_structured_vlm_extraction.py:63,70,76,160,167,268,326` pin `[]`/`{}`/`{"Q1":0.0,"Q2":0.0}` as the contract.
- Edit: line refs `204-207, 278-291` → **`203-207, 281-282, 290-291`**; `313-608` → **`306, 445, 469` (+ the aggregation at `200-231`)**.
- Severity: keep 🟡, but it is the strongest 🟡 in this section — it is a clean "No Silent Fallbacks" violation that ends in a confident wrong answer. Not 🔴 only because it needs a VLM failure to fire; if the reviewer wants one 🔴 in 3.2 on user-visible-wrongness grounds, this is the candidate, not I25/I26.
- Dup: none.

### I23 CONFIRMED
- Evidence: `.env.example:14` is literally `LEMONADE_BASE_URL=http://localhost:8000/api/v1`; `lemonade_client.py:55 DEFAULT_PORT = 13305`, `:43 load_dotenv()` at import. Probe (`%TEMP%`, venv python): `DEFAULT_PORT = 13305 | with .env.example value -> ('localhost', 8000, 'http://localhost:8000/api/v1')`.
- Edit: soften "breaks every command" → "breaks every command run from a directory under the copied `.env`" — `load_dotenv()` is CWD-relative, so this bites developers running from the repo root, not users of the installed wheel from an arbitrary cwd.
- Severity: keep 🟡.
- Dup: none.

### I24 CONFIRMED-ADJUST
- Evidence: `lemonade_client.py:4573-4592` — `env_host = os.environ.get("LEMONADE_HOST")` … `client = LemonadeClient(model=…, host=server_host, port=server_port, …)`, i.e. always the host/port branch that discards `LEMONADE_BASE_URL`; `:4692-4693` `host: str = DEFAULT_HOST, port: int = DEFAULT_PORT` and `:4723` `LemonadeClient(host=host, port=port, keep_alive=True)`. `auto_start` → `:4606 client.launch_server(...)` → `:1080 kill_process_on_port(self.port)` (guarded by a health probe at `:1072-1076`, so it only kills when nothing healthy answers on 13305). `.claude/skills/lemonade-client-patterns/SKILL.md:27` does claim "callers that use the factory (like CLI entry points)" — cli.py in fact uses its own `initialize_lemonade_for_agent` (`cli.py:95`), never the factory.
- Edit: "no in-tree caller" is not quite right — `create_lemonade_client` is called by the module's own `__main__` (`:4762`) **and by `tests/test_lemonade_client.py:55,92`**. Say "no in-tree production caller (only `__main__` and the client's own tests)". Also add that the `kill_process_on_port` step is preceded by a health check, so it kills only a non-healthy occupant of 13305.
- Severity: keep 🟡 (public SDK surface, dead in-tree).
- Dup: none.

### I25 CONFIRMED-ADJUST (severity: keep 🟡, do NOT promote to 🔴)
- Evidence: `lemonade_installer.py:440-461` downloads the asset with `urllib.request.urlopen` straight to disk; `:602-638` `cmd = ["msiexec", "/i", str(installer_path)]` (+`/qn`) run elevated; `:742-752` `sudo installer -pkg <path> -target /`. Repo-wide grep in that file for `sha256|hashlib|Authenticode|pkgutil`: **zero hits**. Contrast `llm/lemonade_embedded.py:76 EMBEDDABLE_SHA256` + `:493-498` (refuses on mismatch) and `installer/tui/fetch_sidecar.py:107-124, 178, 271, 306` (`verified sha256 … against binaries.lock`). CI: `.github/workflows/build-installers.yml:343-356` — `curl -fsSL … lemonade-server-minimal.msi` then only `SIZE -lt 1048576` → fail. Only pre-flight is `verify_download_url` (`:346-392`), an HTTP **HEAD 200** check — availability, not integrity. `gh issue list` on "installer integrity/supply chain/checksum+signature": nothing tracking this (closest is #3295, about the *one-line* Windows script's checksum logic being untested).
- Edit: replace "no integrity check beyond TLS" with the sharper, defensible claim: "**the download is authenticated only by TLS to github.com — GAIA pins a SHA-256 for its own embedded server and sidecar but not for the third-party MSI/PKG it then runs elevated**". Also correct the CI cite `338-356` → **`343-356`**, and add that `verify_download_url` is a HEAD availability probe, not integrity.
- Severity: **keep 🟡.** Arguing 🔴 (promote): the artifact runs with Administrator/root, GitHub release assets are mutable under a fixed tag, and macOS `installer -pkg` from the CLI does not apply Gatekeeper, so a compromised upstream release is a direct root-code-execution path with no second gate. Arguing 🟡 (keep — my verdict): exploitation requires compromising `lemonade-sdk/lemonade`'s release assets or breaking TLS to github.com; nothing here fires in normal use, and pinning a hash to a mutable upstream tag is a hardening improvement, not a live defect. REVIEW.md reserves 🔴 for "a bug that will fire in normal use" or a *present* security hole; this is defense-in-depth against a third-party compromise. It is however the top 🟡 in the installer group — worth filing.
- Dup: overlaps source 05 (report already flags "(Also 05.)"); make sure the two are merged into one finding, not counted twice.

### I26 OVERSTATED
- Evidence: `installer/nsis/installer.nsh:142-145` and `installer/tui/nsis/gaia-setup.nsi:464-467` — the `RmDir /r "$PROFILE\.gaia"` is the *body* of a `MessageBox MB_YESNO|MB_ICONQUESTION … /SD IDNO IDNO +2`, i.e. **opt-in, default No, and silent/GPO/SCCM uninstalls skip it** (`/SD IDNO`). The report's line reads as unconditional deletion and omits the consent gate entirely. `uninstall_command.py:148-163` docstring "We keep ``~/.gaia/`` itself" is confirmed at `:151-152`. #3290 confirmed OPEN and is indeed the opposite gap (`--purge` leaves `~/.gaia/bin` behind).
- Edit: rewrite the finding around the **real** defect, which survives: *the confirmation dialog under-describes what it destroys.* Electron says "chats, documents, Python environment"; TUI says "chats, documents, memory, config" — but `RmDir /r "$PROFILE\.gaia"` also takes `agents/` (custom agents), `connectors/` + `grants.json` + `activations.json` (OAuth tokens/grants), `goals.db`, `mcp_servers.json`, `bin/` and the embedded runtime (paths enumerated by grep over `src/gaia/`). And because both products share `$PROFILE\.gaia`, uninstalling one offers to delete the other's data. Proposed line: "**I26. The NSIS 'also remove your GAIA data?' dialog under-describes what it deletes** — `RmDir /r "$PROFILE\.gaia"` (`installer/nsis/installer.nsh:145`, `installer/tui/nsis/gaia-setup.nsi:467`) is correctly opt-in (`/SD IDNO`, default No) but also removes custom agents, connector OAuth tokens/grants, memory and the shared `bin/` runtime the *other* GAIA product installed — none of which the dialog names. Fix: enumerate them in the prompt, or scope the delete to the uninstalling product."
- Severity: **keep 🟡, do NOT promote to 🔴.** No 🔴: the user must affirmatively answer Yes to a dialog that already warns "This cannot be undone", and every non-interactive path defaults to keeping the data. That is consented deletion, not data loss.
- Dup: none; related to #3290 (opposite direction).

### I27 CONFIRMED
- Evidence: `uninstall_command.py:448-473` — `resolved = path.resolve(strict=False)` follows the link, then `raise RuntimeError(f"Refusing to delete {resolved}: outside allowed roots …")` at `:471-474`, **uncaught**: `execute_plan:658-660` calls `_remove_path` bare, `run():806` returns `execute_plan(...)` with no try, and `cli.py:4421-4424` does `sys.exit(_run_uninstall(args))` inside `main()` (grep for `^    try:` in `cli.py:3142-4430`: none) → traceback, interpreter exit 1, not `EXIT_FS_ERROR = 2` (`:74`) or `EXIT_USAGE = 64` (`:78`). Probe (junction created and removed under `%TEMP%`, repo untouched): `_remove_path(gaiahome/documents → elsewhere, allowed_roots=[gaiahome])` → `RAISED RuntimeError: Refusing to delete …\elsewhere: outside allowed roots (…)`. `_purge_paths:155-163` order is `venv, chat, documents, …`, so a symlinked `documents` aborts after `venv` and `chat` are already gone.
- Edit: cite `:448-473` (the resolve is at 448-452, the raise at 471-474); note a Windows **directory junction** triggers it too (`Path.is_symlink()` is False for a junction, but `resolve()` still follows it) — that widens the trigger beyond POSIX symlinks.
- Severity: keep 🟡 — a half-done purge plus an undocumented exit code; needs a user who relocated a `~/.gaia` subdir.
- Dup: none. `gh` search: no tracking issue.

### I28 CONFIRMED
- Evidence: `installer/debian/postrm:34-38` — `if [ -n "$SUDO_USER" ] && command -v sudo …; then sudo -u "$SUDO_USER" -H sh -c 'command -v gaia … && gaia uninstall --purge --yes' 2>/dev/null || true`, i.e. runs as the invoking admin with **stderr discarded and every failure swallowed** (`|| true`). `_purge_paths` (`uninstall_command.py:155-163`) includes `~/.gaia/chat` and `~/.gaia/documents`; `--yes` bypasses the confirm.
- Edit: cite `:34-38` (the `elif` fallback at 36-37 is part of the same block). The report's own "(Medium — plan-sanctioned; the risk is the silent default.)" is accurate and should stay.
- Severity: keep 🟡.
- Dup: shares the "purge deletes user documents" mechanism with I27 and with source 01's `GAIA_HOME` 🔴; they are distinct triggers, keep separate.

### I29 CONFIRMED
- Evidence: (a) PPA — `lemonade_installer.py:983-996`: after `apt-get install` returns 0, `verify = self.check_installation()`, `installed_version = verify.version or "unknown"`, then an **unconditional** `return InstallResult(success=True, version=installed_version, message=f"Installed Lemonade v{installed_version} via PPA")`. `verify.installed is False` is never checked, so a probe that finds nothing still yields `success=True` and the literal message "Installed Lemonade vunknown via PPA". (b) Windows uninstall — `:1108-1168`: inside `for use_minimal in [True, False]` every failure is `except Exception as e: log.debug(...); continue`, including a `download_installer()` network/404 failure, and the loop exit returns `error="Could not uninstall: product not found in Windows Installer registry. Try uninstalling manually via Windows Settings > Apps."` — a registry diagnosis for what may have been a download failure.
- Edit: tighten ranges to `:983-996` and `:1108-1168`; add the concrete misdiagnosis ("a failed MSI *download* is reported as 'product not found in Windows Installer registry'") — that is the actionable half.
- Severity: keep 🟡 (two No-Silent-Fallbacks violations, both on recovery paths).
- Dup: same class as I31 (reporting success for work that did not happen); group under #3307 in the fix plan but keep separate — different files, different fixes.

### I30 CONFIRMED-ADJUST (impact overstated)
- Evidence: `cli.py:4303-4316` — `if args.action == "init":` -> `profile = "minimal" if args.minimal else args.profile` -> `if profile == "mcp": ... run_mcp_init(...); sys.exit(exit_code)` at `:4309-4316`, **before** `if args.check:` at `:4318`. The `--check` help text (`cli.py:3001-3005`) promises "no install, no download, no side effects. Exit code 0 means ready, 1 means `gaia init` still has work to do." `mcp_init.MCPInitCommand.run` (`:42-91`) creates `~/.gaia/` and writes `~/.gaia/mcp_servers.json`, then `return 0`. Verified statically only — running it would write to `~/.gaia`, which the brief forbids.
- Edit: "performs the install" overstates it — nothing is downloaded or installed. Reword to: "**`gaia init --profile mcp --check` takes the write path and always exits 0** — the mcp short-circuit (`cli.py:4309-4316`) precedes the `--check` branch (`:4318`), so `--check` creates `~/.gaia/` and `~/.gaia/mcp_servers.json` and can never report NOT READY, contradicting its own help text ('no side effects ... 1 means gaia init still has work to do'). A setup script gating on it always believes mcp is ready."
- Severity: keep 🟡.
- Dup: the always-exit-0 half belongs to the #3307 family (I31).

### I31 CONFIRMED
- Evidence (probe: `%TEMP%` cwd, venv python, exit code read per process with no pipe): `gaia cache -> 0`, `gaia memory -> 0`, `gaia knowledge -> 0`, `gaia mcp -> 0`, `gaia daemon -> 0`, each after printing a "No <x> action specified" error. Source: `cli.py:5075-5078, 5160-5163, 4966-4969, 7138-7141, 6516-6521` (bare `return`). Chat chain confirmed: `async_main` (`:507`) `return 0 if result["status"] == "success" else 1` at `:729` -> `run_cli` (`:799-800`) -> `cli.py:3334-3346` `result = run_cli(args.action, **kwargs); if result: print(result)` — the failure code is **printed, not exited on**. `llm`: `:3787-3788` `if not success: return` after the Lemonade preflight; `api start`: `:4849-4850`; `download --clear-cache`: `:3560-3561`. `tests/unit/test_cli_refusal_exit_codes.py:44-49` parametrizes exactly two cases, `["kill"]` and `["cache","clear"]` — the coverage claim is exact. #3307 verified OPEN: "Commands report success for operations that did not happen".
- Edit: none needed; optionally add that `gaia chat -q` prints the literal `1` to stdout before exiting 0, which also corrupts any script capturing the answer.
- Severity: **keep 🟡, do not promote to 🔴.** It is a genuine No-Silent-Fallbacks violation that fires in normal use (`gaia llm` with Lemonade down), but the interactive user does see the error text, there is no data loss or security exposure, and an open umbrella issue already tracks it. Promote only on an explicit "`cmd && next-step` in CI is a supported idiom" argument, and then say so.
- Dup: shares the "success for work not done" mechanism with I29 and with the `--check` half of I30.

### I32 CONFIRMED
- Evidence: `cli.py:7424` and `:7485` — `pid_file_path = os.path.abspath("gaia.mcp.pid")`, i.e. the **current working directory**, not `~/.gaia`; the log default is likewise `"gaia.mcp.log"` (`:2402-2403`). In `handle_mcp_stop` (`:7477`) the kill-failure paths only `print` (`except subprocess.CalledProcessError: print("Failed to stop process ...")` at `:7524-7527`), and the `# Clean up PID file` block at `:7529-7534` runs **unconditionally** afterwards — a failed `taskkill` still deletes the PID file and the orphan becomes unfindable. `cli_agent.py:755` `parsed = hub_manifest.parse(pkg_dir)` is bare; `manifest.ManifestError` is a `ValueError` (`hub/manifest.py:110`), not an `AgentWorkflowError`, and the dispatcher catches only `AgentWorkflowError` (`cli_agent.py:304-308`) -> `gaia agent test --live` on a malformed `gaia-agent.yaml` tracebacks instead of printing an actionable error.
- Edit: cite `:7529-7534` (not `7523-7532`) for the unconditional cleanup, and add the log-file default at `:2402-2403` as the second CWD artifact.
- Severity: keep 🟡. Note `handle_mcp_stop` also exits 0 on every one of those failure prints — same defect as I31, so the fixes should land together.
- Dup: partial overlap with I31 (exit 0 on failure).

### I33 CONFIRMED-ADJUST (two sub-claims are wrong)
- Evidence, all re-run against the real parser via `python -m gaia.cli ... --help`:
  - `gaia mcp test-client --config foo` -> `cli.py: error: unrecognized arguments: --config`, while `docs/reference/cli.mdx:1173` documents `gaia mcp test-client <server-name> [--config PATH]`. Confirmed.
  - Registered and absent from `cli.mdx` (`grep -c` = 0 for each): `eval {code,sessions}` (subparsers are `{agent,benchmark,sessions,code}`), `mcp serve`, `schedule {show,remove}` (subparsers `{add,list,show,remove,pause,resume,run,daemon}`), `connectors {list,status,test,disconnect}`, `memory bootstrap --infer/--system/--reset-system`, `install --silent`, `report --eval-dir/--output-file/--summary-only`, `connectors connect --grant-agent`, `connectors configure --client-id/--client-secret`. Confirmed.
  - **Wrong:** `connectors activations ...` and `connectors grants ...` **are** documented — `cli.mdx:1337-1338` list `grants {list|grant|revoke}` and `activations {list|activate|deactivate}`, with a worked example at `:1366`. Remove both from the undocumented list.
  - Shared parent flags: **19** of the 31 top-level subcommands accept `--use-chatgpt` (`prompt chat talk email api telegram schedule download stats test youtube kill llm eval report perf-vis mcp install uninstall`), so "~20" is accurate. Worst concrete example: `gaia install --help` advertises `--max-steps`, `--stream`, `--show-stats`, `--use-claude`, `--claude-model`, `--trace`.
- Edit: (1) drop `activations ..., grants ...` from the undocumented list; (2) fix the broken cross-reference — "inert documented flags (**I26**/C26)" points at the NSIS-uninstaller finding; it should read "(C26)" alone, or name the source-07 finding it means; (3) optionally cite `gaia install` as the concrete shared-flag example. The "135 parsers enumerated" figure is a methodology claim I did not re-count; it is not load-bearing.
- Severity: keep 🟡.
- Dup: the inert-flag half overlaps C26; keep I33 as the *docs-vs-parser drift* finding and let C26 own the ignored-flag behaviour.

## Summary (I17-I33)

**Counts (17 findings):** CONFIRMED 11 (I17, I18, I20, I21, I22, I23, I27, I28, I29, I31, I32) - CONFIRMED-ADJUST 5 (I19, I24, I25, I30, I33) - OVERSTATED 1 (I26) - REFUTED 0 - UNVERIFIABLE 0.

**Most important corrections**

1. **I26 is materially overstated and must be rewritten.** Both `RmDir /r "$PROFILE\.gaia"` calls are the body of a `MessageBox MB_YESNO ... /SD IDNO` — opt-in, default **No**, and skipped entirely by silent/GPO/SCCM uninstalls. The report's line reads as unconditional deletion. The defect that survives is narrower and still worth filing: the dialog names "chats, documents, Python environment / memory, config" but the delete also takes `agents/`, `connectors/` + `grants.json` + `activations.json` (OAuth tokens), `goals.db`, `mcp_servers.json`, `bin/` and the *other* GAIA product's data.
2. **I33 lists two things as undocumented that are documented** — `connectors grants` and `connectors activations` are both at `cli.mdx:1337-1338`. It also carries a broken cross-reference: "inert documented flags (I26/C26)" — I26 is the uninstaller finding.
3. **I30's "performs the install" is wrong.** `run_mcp_init` downloads and installs nothing; it `mkdir`s `~/.gaia` and writes an empty `mcp_servers.json`, then returns 0. The real defect is that `--check` takes a write path and can never report NOT READY, contradicting its own help text.
4. **I19's consequence is order-dependent.** On `chat`/`rag` the LLM is verified before the embedder, so nothing is evicted; what is wasted is an unload+reload of the model `ensure_ready` just preloaded. The eviction case needs the `all` profile, whose model order comes from `list(set(...))`.
5. **I24's "no in-tree caller" is inaccurate** (`create_lemonade_client` is used by the module's `__main__` and by `tests/test_lemonade_client.py`), and **I25's "no integrity check beyond TLS" should be sharpened** to the internally-inconsistent version: GAIA pins SHA-256 for its own embedded server and sidecar, but not for the third-party MSI/PKG it runs elevated.

**Proposed severity changes: none. All 17 stay 🟡.**

- **I25 (elevated install, no hash/signature): keep 🟡.** Exploitation requires compromising `lemonade-sdk/lemonade`'s GitHub release assets or breaking TLS to github.com — nothing fires in normal use, so this is defense-in-depth, not a present hole. (The 🔴 case, for the record: the artifact runs as Administrator/root, GitHub release assets are mutable under a fixed tag, and CLI `installer -pkg` on macOS applies no Gatekeeper check, so one upstream compromise is root RCE with no second gate. I do not think that clears REVIEW.md's 🔴 bar, but it is the closest call in the section.)
- **I26: keep 🟡** — consented, opt-in, default-No deletion is not data loss.
- **I31 (failure paths exit 0): keep 🟡** — it does fire in normal use, but the interactive user sees the error, there is no data loss or security exposure, and #3307 already tracks it.
- If the section needs one 🔴, the better candidate than I25 or I26 is **I22** — a silent fallback that puts `[VLM extraction failed: ...]` into the RAG index and then quotes it back as a confident answer. Still 🟡 by my reading, because it needs an actual VLM failure to fire.

**Tracked-issue check (`gh`):** I31 -> #3307 (OPEN, exact match). I26 -> #3290 (OPEN, the opposite gap; correctly cited). I17 -> #1676 (related, as stated). I25/I27/I28/I29/I30/I32/I33 -> no tracking issue found (nearest for I25 is #3295, about the one-line Windows installer's *untested* checksum logic — a different code path).

---

## §3.3 Servers, daemon, connectors, security (I34–I45) — verified directly

### I34 CONFIRMED-ADJUST
- Evidence: `ui/routers/files.py:568` gates text reads on `if ext in TEXT_EXTENSIONS or stat.st_size < 1_000_000` — **any** extension under 1 MB is read and up to 200 lines × 500 chars returned; the only containment is `ensure_within_home` (`ui/utils.py:391-404`, *not* :395 — :395 is inside the docstring), which is a bare `relative_to(Path.home())` with no sensitive-root denylist. `server.py:151-186` confirms a valid tunnel bearer/cookie satisfies the middleware for **every** `/api/*` path, so the token does grant `GET /api/files/preview?path=~/.ssh/id_ed25519`.
- Edit: fix the cite to `ui/utils.py:391`; add "preview line cap is 200 lines × 500 chars, which fully contains an ed25519 or RSA private key". **De-duplicate against C9**: the sentence "Also reachable without a tunnel via C9's DNS-rebinding read side" restates C9, which already names `/api/files/preview?path=~/.ssh/id_rsa` as its read-side proof. Cut that sentence from I34 and add a `See also I34` pointer inside C9 instead — as written the same exploit is counted twice, once at 🔴 and once at 🟡.
- Severity: **keep 🟡**, but only after the de-duplication. The remotely-exploitable half is C9's (🔴, drive-by, no credential needed). What is left in I34 is over-broad scope on a credential the user *deliberately* shares via QR code — real, security-relevant, but the attacker must already hold the token. Escalating I34 to 🔴 while C9 is also 🔴 would double-count one exploit.
- Dup: **partial dup of C9** (read side). Merge as described.
- Tracked: none found (`gh issue list --search "tunnel token files preview"` → only #875, unrelated).

### I35 CONFIRMED — recommend 🔴
- Evidence: `security.py:29-52` (`SENSITIVE_FILE_NAMES`) and `:112-190` (`_get_blocked_directories`) contain no rc/profile/autostart entry — `grep -n "bashrc\|zshrc\|profile\|autostart\|gitconfig\|git/hooks" src/gaia/security.py` → **no matches**. *Probe* (`PathValidator(allowed_paths=[home])`, `validate_write(..., prompt_user=False)`): `.bashrc → (True,'')`, `.profile → (True,'')`, `.zshrc → (True,'')`, `.gitconfig → (True,'')`, `.config/autostart/x.desktop → (True,'')`, `Documents/WindowsPowerShell/profile.ps1 → (True,'')`, while `.ssh/id_ed25519` and `.env` are correctly refused. The allowlist half also holds: `_chat_helpers.py:926-940 _compute_allowed_paths` adds `str(Path(fp).parent)` per attached document, so one document attached directly in `$HOME` makes all of `$HOME` writable.
- Edit: add the probe line as evidence; correct the cite to `security.py:29-52, 112-190, 495-560` and `_chat_helpers.py:926-940`. **Add the missing aggravating fact:** `security.py:622-640 _prompt_overwrite` auto-approves the overwrite whenever `_is_interactive()` is false — i.e. in the Agent UI server context the "are you sure you want to overwrite your existing `.bashrc`" step does not happen at all.
- Severity: **change 🟡 → 🔴.** REVIEW.md puts security in the 🔴 tier, and the outcome here is *persistent arbitrary code execution* — `write_file("~/.bashrc", "curl … | sh")` runs on the user's next shell. That is the same outcome class as C4 (shell allowlist bypass), which the report already ranks 🔴; the only difference is that it is reached through the write tool rather than the shell tool. The prompt-injection vector is not hypothetical — C9 already establishes "upload → indexed into RAG = persistent prompt injection". The single mitigation is a confirmation modal that (a) shows a benign-looking home-directory path and (b) is skipped entirely on the auto-approve / non-interactive paths.
- Dup: none. Complements C7 (read-side sandbox escapes) and C4 (shell RCE); this is the write-side.
- Tracked: none found.

### I36 CONFIRMED-ADJUST
- Evidence: `web/tavily.py:125` (`get_credential_sync(_CONNECTOR_ID)`) and `:146` (`await get_credential(_CONNECTOR_ID)`) — neither passes `agent_id` nor `required_scopes`. The gate is `connectors/handler.py:152-155`: `resolved_agent = agent_id or current_agent_id()` then `if resolved_agent and required_scopes:` — so the grant check is skipped when **either** conjunct is falsy. `tests/unit/connectors/test_handler.py:161 test_no_agent_id_skips_grant_check` confirmed at that exact line. Doc contradiction confirmed: `docs/security/connections.mdx:63` threat-model row says "`get_credential_sync` checks the grants ledger before returning; unapproved calls raise `AuthRequiredError`".
- Edit: two corrections. (1) The cite `connectors/handler.py:399-410` is **wrong** — the code is at `:137-168`, and `:399-410` is a different function. (2) The mechanism is stated as half-true: it is not only "skips grants when `required_scopes` is empty", it is "skips when the agent id is unresolved **or** `required_scopes` is empty". (3) Strengthen the framing — "(Concrete instance of C1)" *understates* it. Tavily passes neither argument, so **fixing C1's contextvar loss would not gate this call**; it is an independent bypass that needs its own fix at the call site. Reword to "(C1 makes every tool ungated; this call site is ungated even after C1 is fixed.)"
- Severity: keep 🟡. The asset is a paid-search API key, and the caller is in-process code the connectors threat model already classifies as trusted ("Token exfiltration via a rogue custom agent → custom agents are trusted code you install yourself"). Do not escalate to 🔴 — C1 already carries the 🔴 for the systemic bypass and escalating the instance double-counts.
- Dup: overlaps C1 by design; keep both but sharpen the relationship per the edit above.
- Tracked: none found.

### I37 CONFIRMED
- Evidence: `grep -n "keyring\.\(get\|set\|delete\)_password" src/gaia/connectors/mcp_server.py` → lines **168, 215, 261, 334**; the report lists 168/215/334 and **misses 261** (`delete_password` in the disconnect path). `verify_keyring_backend` (`store.py:99-140`, refusing `PlaintextKeyring`/`EncryptedKeyring`/`Win32CryptoKeyring`) is called only from `store.py:354,402,486,504,530,550,564` (the OAuth connection-blob paths) and `api.py:509` (forwarded-connection import) — never from `mcp_server.py`. `docs/security/connectors.mdx:21-25` claims refusal happens "at the entry of every save and load"; false for MCP-server secrets.
- Edit: add line 261 to the cite list.
- Severity: keep 🟡. Requires a headless Linux box with `keyrings.alt` installed and a local reader of `keyring_pass.cfg` — a hardening gap plus a doc that overstates the guarantee, not a remotely reachable one. Squarely "convention violation / doc contradicting code".
- Dup: none. Feeds I92's security-doc-contradiction list — keep the I92 bullet as a pointer, not a restatement.
- Tracked: none found.

### I38 OVERSTATED
- Evidence: both mechanisms are real and I reproduced both. *Probe:* scan `…/_p38/docs` → 1 row; then scan `…/_p38/doc` → `removed=1` and `docs/b.txt` is gone (`index.py:349-354`, `path LIKE :prefix` with `root_str + "%"`, no separator, no LIKE escape). *Probe:* `query_files(name="report (final)")` → `OperationalError: fts5: syntax error near "report"`; `name="c++"` → `syntax error near "+"`; `name="name:report OR path:secret"` → executes, 0 rows (`index.py:707-710`, raw `files_fts MATCH :name`).
- Edit: two reachability facts the report omits, both of which cut the impact hard, must be added. (1) **`scan_directory` has zero callers in the entire repo** — `grep -rn "scan_directory" --include=*.py .` matches only `index.py` itself. `FileSystemIndexService` is instantiated by ChatAgent (`gaia_agent_chat/agent.py:327`) but the scan is never invoked, so the sibling-deletion bug is **latent, not user-reachable today**. (2) The FTS5 half has exactly one caller, `filesystem_tools.py:663`, and it is wrapped in `except Exception: logger.debug("Index search failed, falling back to filesystem")` (`:686-689`) — so the syntax error never surfaces to the user; it silently degrades to a full filesystem walk. Worse, `:640-654` routes any query containing `(` to `effective_type = "content"`, which skips the index path entirely, so **the report's own headline example, `report (final)`, never reaches `query_files`**. Rewrite as: "latent, no in-tree caller" for the scan half, and "the only caller swallows the error and silently falls back to a filesystem walk (a No-Silent-Fallbacks violation in its own right)" for the query half. Fix the line cite `709-712` → `707-710`.
- Severity: keep 🟡, but only on the rewritten text. As written it reads like user-visible index corruption plus a user-visible crash; neither reaches a user at this commit.
- Dup: none.

### I39 CONFIRMED
- Evidence: `api/openai_server.py:249 async def create_chat_completion` and `:364 result = agent.process_query(...)` — a synchronous call directly in the coroutine body. The streaming branch does it correctly at `:495-496` (`loop.run_in_executor(None, lambda: agent.process_query(...))`). Cite is exact.
- Edit: none. Optionally add "the streaming branch 130 lines below already has the fix, so this is a one-line asymmetry, not a design problem."
- Severity: keep 🟡, though it is the closest call in this section. Non-streaming is the default for many OpenAI clients, so the loop stalls for a whole agent turn on a normal path — `/health`, `/v1/models` and every concurrent request block. It stops short of 🔴 because the consequence is degraded concurrency on a single-user local server, not a wrong answer or data loss.
- Dup: none.

### I40 CONFIRMED
- Evidence: all sub-claims hold. (a) `_execute_chat` is a method of the **`GAIAMCPBridge` singleton** (`mcp_bridge.py:45`; one instance created at `:511` and shared by every handler via `MCPHTTPHandler.bridge`), so `self.chat_sdk` (`:173-176`) is one `AgentSDK` — one conversation history — across all callers. (b) `:464 elif "/health" not in args[0]` — *probe* of the stdlib confirms `send_error` → `log_error("code %d, message %s", code, message)` → `log_message` with `args[0]` an int/`HTTPStatus`, and `"/health" not in 501` raises `TypeError: argument of type 'int' is not iterable`. Any unsupported method, an over-long URI, or any base-class 4xx/5xx kills the handler. (c) `:344 content_length = int(self.headers.get("Content-Length", 0))` — unguarded, and the following `self.rfile.read(content_length)` is uncapped. (d) `:60` and argparse `:575` both default to the literal `"http://localhost:13305/api/v1"`; neither reads `LEMONADE_BASE_URL`.
- Edit: fix the `log_message` cite `:463` → `:464`, and add the gate the report omits — the `TypeError` fires **only when `--verbose` is off** (`VERBOSE` at `:33`/`:69`; the `if VERBOSE: pass` branch short-circuits before the `elif`). That is the default, so the finding stands, but the condition belongs in the text.
- Severity: keep 🟡.
- Dup: none.

### I41 CONFIRMED — recommend 🔴
- Evidence: `messaging/telegram.py:76-79 _allowed()` → `if not self.allowed_users: return True`. `tests/unit/test_telegram_allowlist.py:27 test_empty_allowlist_intentionally_allows_all` pins it (and `:32` pins the `None` case). No rate limiting anywhere — `grep -n "rate_limit\|throttle" src/gaia/messaging/telegram.py` → no matches. The RAG half is exact: `:119 rag_result = ingest_document_to_rag(tmp_path)` on any document a stranger sends. Doc contradiction confirmed verbatim at `docs/guides/telegram-adapter.mdx:71`: "Messages from anyone not on the allow-list are ignored."
- Edit: add the doc cite (`docs/guides/telegram-adapter.mdx:71`) — the report names no file. State the attacker position explicitly: **unauthenticated, remote, no click** — Telegram bots are reachable by username by anyone who finds one; the attacker needs no credential and no prior relationship with the victim.
- Severity: **change 🟡 → 🔴.** An unauthenticated stranger gets a full conversation with the user's local agent (its tools, its files) and can write documents into the user's global RAG library — which C9 already characterises as *persistent prompt injection*. The usual "but the user opted in" defence fails here, because the guide tells the user in plain words that off-list senders are ignored: a user who read the docs and set no allowlist believes they are protected and is not. State the one honest mitigation alongside the escalation — the exposure requires the user to have created a bot, configured the token, and run `gaia telegram start` in the **foreground** (per I42, `--background` never polls at all).
- Dup: none.
- Tracked: **#3239** (open, weekly-audit: "The Telegram guide says off-allowlist messages are ignored — the bot now replies to them") and **#2062** (no e2e coverage), as the report states. Note #3239 is filed as a *docs* bug; escalating I41 to 🔴 means the fix is the default, not the sentence.

### I42 CONFIRMED-ADJUST
- Evidence: background mode confirmed dead — `telegram.py:326-337`: `poll_thread = threading.Thread(target=_run_polling, daemon=True)`, `.start()`, then `return`; the CLI process exits and both daemon threads die. Health server hard-coded at `:308 HTTPServer(("127.0.0.1", 8765), HealthHandler)`. `--force` confirmed inert: registered at `cli.py:1638-1642`, and grepping `force` over the stop handler (`cli.py:3266-3300`) → no matches.
- Edit: **the port paragraph is wrong and needs replacing.** The real map, re-derived: `gaia mcp start` (the MCP bridge) defaults to **8765** (`cli.py:2383`); `gaia mcp serve` (Agent UI MCP) to **8766** (`cli.py:2490`) — while `agent_ui_mcp.py:42` sets its own module default to **8765**; `gaia mcp tui` to 8767 (`cli.py:2511`, matching `tui_mcp.py:95`). So the comment at `tui_mcp.py:95` ("8765 is agent_ui_mcp, 8766 is the MCP bridge") has the two **swapped** relative to the CLI, and the genuine 8765 collision is three-way: MCP bridge, `agent_ui_mcp`'s module default, and the Telegram health server. **Add the stronger fact the report missed:** `gaia telegram start --health-port` exists (`cli.py:1656`, default 8765) and is read *only* by the `status` action (`cli.py:3304`); `run_telegram` takes no `health_port` parameter, so `telegram.py:308` hard-codes the port and the flag cannot move the server — another "accepted and ignored" flag that belongs in C26's list.
- Severity: keep 🟡.
- Dup: the `--health-port` and `--force` no-ops are instances of C26's "accepted and ignored" pattern — cross-reference rather than restate.
- Tracked: #3133 as stated.

### I43 CONFIRMED-ADJUST — every line number in this finding is wrong
- Evidence: (a)+(b) reproduced. *Probe:* `Schedule(name="bad", cron="not a cron", prompt="hi")` constructs fine (`store.py:41-49 __post_init__` validates only skill-xor-prompt, never the cron) and `store.add()` writes it to TOML; then `daemon.next_fire_time("not a cron")` → `ValueError: Wrong number of fields; got 3, expected 5`, and `build_scheduler` raises the same. That function (`daemon.py:85-88`, bare `CronTrigger.from_crontab`) is called from `list` (`cli.py:3090`), `show` (`:3105`), `run` (`:3127`) and `build_scheduler` (`daemon.py:56`) — so all four break exactly as claimed. (c) confirmed: `store.py:126-130 save()` is a bare `open(path,"wb")` + `tomli_w.dump` — truncate-then-write, no temp+rename, no lock — and every mutator (`add`/`remove`/`set_enabled`/`mark_run`) is load→mutate→save.
- Edit: **fix the citations — they point past the end of the files.** `src/gaia/schedule/store.py` is **191 lines** and `sinks.py` is **125**, so `store.py:314-321`, `store.py:404-408`, `store.py:450-462` and `sinks.py:247` cannot exist. Correct to: cron unvalidated → `store.py:41-49` + `cli.py:3071-3079`; crash sites → `daemon.py:85-88` (called from `cli.py:3090, 3105, 3127` and `daemon.py:56`); non-atomic write → `store.py:126-130`; Telegram token → `sinks.py:100-105`. Also **reword the token claim**: GAIA does not persist the token. `cli.py:3067-3070` sets only `sink_args["to"]`; there is no `--token` flag. What exists is `sinks.py:100` reading `sink_args.get("token")` and an error message at `:104-105` *inviting* the user to hand-edit it into `schedules.toml`, with `docs/reference/cli.mdx:2412` offering it as the alternative to `GAIA_TELEGRAM_TOKEN` (which the doc lists first). Correct text: "the Telegram sink accepts a bot token hand-written in cleartext in `schedules.toml`, and both the error message and `cli.mdx:2412` present that as a supported option."
- Severity: keep 🟡. (a)+(b) is a real crash on a normal path, but recovery is one hand-edit of a documented human-readable file.
- Dup: none.
- Tracked: none found.

### I44 CONFIRMED — recommend SPLITTING; the UTC bug is buried
- Evidence: three schedulers confirmed as separate implementations — `src/gaia/schedule/` (APScheduler + TOML), `src/gaia/ui/scheduler.py` (asyncio timers + SQLite), `src/gaia/daemon/scheduler/` (`clock.py`, `store.py`, `migration.py`). `reconcile_jobs` (`daemon/scheduler/migration.py:33`) has exactly **one** caller outside its own package: `hub/agents/email/python/gaia_agent_email/daemon_migration.py:131` — the email adapter, as claimed. UTC bug confirmed at `ui/scheduler.py:564-576`: `now = after or datetime.now(timezone.utc)`, then `candidate = now.replace(hour=hour, minute=minute, …)` — the user's "9am" is applied to a UTC clock. `compute_next_run` is the single next-run source for the UI scheduler (`:811, :935, :1031, :1141`) and the schedules router (`routers/schedules.py:132`).
- Edit: **split this finding in two.** "Daily at 9am fires at 09:00 UTC" is a concrete, user-visible, 100 %-reproducible defect for every user not on UTC; it is currently a subordinate clause inside an architecture-sprawl bullet, where a maintainer triaging by severity will not see it. Promote it to its own finding with the `ui/scheduler.py:564-576` cite, and leave I44 as the (correct) architecture observation.
- Severity: keep 🟡 for the architecture half; the extracted UTC finding is a **candidate for 🔴** — REVIEW.md's 🔴 tier includes "a bug that will fire in normal use", and this one fires on every fixed-time schedule for every non-UTC user. It falls short of 🔴 only because the consequence is a mistimed job rather than data loss; at minimum it should be the top-ranked 🟡 rather than invisible.
- Dup: none.
- Tracked: #2156/#2379 as stated (daemon clock only — neither covers the UTC bug).

### I45 CONFIRMED-ADJUST
- Evidence: all five sub-claims hold. (1) `manager.py:254 self.auth_token = secrets.token_urlsafe(32)` — minted once per manager; `_write_secret_file` (`:589-633`) creates a **fresh** `mkdtemp` dir on each call and overwrites `self._secret_path`, while cleanup (`:647-650`) only unlinks the current path — so a re-ensure orphans the previous file carrying the same still-valid token. (2) `:470 spawn_env[CUSTODY_SECRET_ENV_VAR] = self.custody_secret` is unconditional, so the custody secret rides bare process env even when the caller-auth token takes the file leg at `:558`. (3) Windows DACL confirmed: the `_restrict_to_current_user` helper exists only in `manager.py:114-193` and is applied only to the launch-secret dir/file, while `paths.py:99-132 atomic_write_json` — which writes `instance.json` and the sidecar ledger — relies on `os.open(..., 0o600)` + `os.chmod`, both inert on NTFS. (4) `manager.py:448 spawn_env = {**os.environ, …}`; `docs/plans/security-model.mdx:359-368` lists this exact leak (including `GITHUB_TOKEN`) as a known problem. (5) Orphan migration confirmed: `daemon/migrate.py:243-263` copies to `custody_sessions_db()` / `custody_memory_db()`, and `grep -rn` shows those two paths are read by **nothing but tests** — the daemon opens `CustodyStore(custody_db_path())` = `host_dir()/custody.db` (`daemon/server.py:235`).
- Edit: fix two paths. The migration file is `src/gaia/daemon/migrate.py:243-263`, **not** `custody/migrate.py:115-122` (that file does not exist). And narrow the DACL sentence: `instance.json` and the sidecar ledger are the two confirmed `atomic_write_json` callers; `daemon.log`, `sidecars.json` and `custody.db` are written elsewhere and were not verified — either re-check them or drop them from the list.
- Severity: keep 🟡. The report's own "(local-only)" qualifier is correct and honest — every item requires a local process running as another user, or a local reader of the temp dir.
- Dup: sub-claim (4) overlaps I92's security-doc list; keep one.
- Tracked: none found.

---

## §3.4–3.6 RAG, data, code index, voice, email (I46–I59)

# Adversarial verification — I46 … I59 (§3.4 RAG/data/code-index, §3.5 voice, §3.6 email)

Checkout: `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f` @ 211f08c5. Read-only.
Python: `.venv\Scripts\python.exe` (3.13.11). Probes run in `%TEMP%`.

---

### I46 CONFIRMED
- Evidence: read `web/client.py:44-52` (`_is_blocked_ip` = `is_private|is_loopback|is_link_local|is_reserved|is_multicast`), used by both `_assert_ip_allowed:72` (pinned-IP adapter, `:162`) and `validate_url:363`; probe → `100.100.100.100 blocked=False is_global=False`, `100.64.0.1 blocked=False`, while `10/127/192.168/169.254/198.18` all `blocked=True` — so `not ip.is_global` is the correct single-predicate fix and does not over-block `8.8.8.8`/`192.88.99.1`.
- Edit: append the attacker position and the gate, which the report omits: "…*probe:* `100.100.100.100` ALLOWED (`is_private=False`, `is_global=False`). Reachable when an injected URL (fetched page, indexed doc, email) steers `web_fetch`; **only exploitable on a host that actually routes 100.64/10** — a Tailscale/CGNAT LAN or a k8s pod CIDR. Cloud metadata (169.254.169.254) and RFC1918 are already blocked. Fix: `not ip.is_global`."
- Severity: **keep 🟡**, not 🔴. It is a real bypass of a stated security boundary, but the high-value SSRF targets (loopback, RFC1918, link-local metadata) are all still blocked, and reaching anything requires the host to be joined to a CGNAT/Tailscale mesh. It does not fire in normal use, and it is not data loss or a broken API. Related closed issue #956 hardened the same predicate's DNS-rebind window — worth citing as "same function, second gap".
- Dup: none.

### I47 CONFIRMED-ADJUST
- Evidence: read `rag/sdk.py:534-655`. Batch path `:606-611` builds `batch_embeddings` from `response.get("data", [])` with **no length check**; the one-by-one fallback fires only on `len(batch_embeddings)==0` (`:613`) and on a still-empty single response does `self.log.warning(... "skipping")` (`:631-635`) — chunk kept, vector dropped. Consumers are positional: `:3075` `file_chunks = [chunks_snapshot[i] …]`, `:3082` `_encode_texts(file_chunks)`, `:3086` `file_index.add(...)`, `:3108` `retrieved_chunks.append(file_chunks[idx])`. Contrast `code_index/sdk.py:840-871` (`_encode_texts` **raises** `RuntimeError` on short batch) and `:874-932` (`_encode_texts_with_sync` returns `(embeddings, valid_chunks)` in lockstep) — the correct contract exists one package over. All-dropped → `np.array([], float32)` shape `(0,)` → `file_embeddings.shape[1]` `IndexError` at `:3083`.
- Edit: the report understates it. Replace "drops a vector but keeps the chunk when Lemonade returns nothing" with: "**a partial batch (`data` shorter than the 25 texts sent) is extended silently with no length check, no fallback and no warning** (`:606-611`); only a *fully* empty batch triggers the one-by-one path, which then skips individual failures with a `warning` (`:631-635`). Either way chunk *i* no longer matches vector *i*…". Also cite `:3075-3108` as the positional consumer rather than a vague "every consumer".
- Severity: **keep 🟡**, argued: it produces silent wrong answers, but only once the embedding backend returns short — it does not fire on a healthy Lemonade, so it fails REVIEW.md's "fires in normal use" test for 🔴. Two things push it to the top of 🟡 rather than the bottom: the `code_index` docstring says the identical failure "previously surfaced as *no matches*", i.e. this is an observed-in-the-wild backend behaviour, not a theoretical one; and the RAG path degrades *silently* where `code_index` now fails loudly, which is a direct CLAUDE.md "No Silent Fallbacks" violation worth naming in the finding.
- Dup: none (I50/I52 are different integrity paths).

### I48 CONFIRMED-ADJUST
- Evidence: read `rag/sdk.py:546-563` (`MAX_EMBED_CHARS = 1200`, silent truncation logged at `info`), `:92` `chunk_size: int = 500` tokens, `:2103` `para_tokens = len(para) // 4` — so the chunker targets ~2,000 chars at defaults. `docs/sdk/sdks/rag.mdx:207-217` states "500 tokens ≈ 375 words ≈ 2-3 paragraphs" and recommends `1024` for technical papers; `:945,1040,1051` recommend `768`; `:1076,1105` recommend `1024` for speed.
- Edit: the arithmetic is wrong for the default. At `chunk_size=500` (≈2,000 chars) 1,200/2,000 = **40 % invisible**, not 60–70 %. The 60–70 % figure is right for the *documented recommendations*: `768` → ~3,072 chars → 61 % invisible; `1024` → ~4,096 chars → 71 %. Rewrite as: "…**40 % of a default chunk, and 61–71 % of the 768/1024-token chunks `rag.mdx` recommends, is never embedded** — so the tuning guide's advice makes retrieval strictly worse, which is the dishonest part."
- Severity: keep 🟡. Silent retrieval degradation, no crash, and the truncation is at least logged (`info`, `:558-561`).
- Dup: none. (Note `code_index` applies the same 1,200 cap but *chunks to fit it* — `parsers.py:357 _MAX_CHUNK_CHARS = MAX_EMBED_CHARS - 100` — which is the fix pattern to cite.)

### I49 CONFIRMED
- Evidence: read `rag/sdk.py:2316-2415`. `:2327` `with self._state_lock:` wraps the whole body; `:2371-2372` `if new_chunks: new_index, new_chunks = self._create_vector_index(new_chunks)` → `:2266 _encode_texts(chunks)` for the **entire remaining corpus**; `:2413-2415` `except Exception as e: self.log.error(...); return False`. Callers confirmed: `:2437` (`reindex_document`), `:2477` (`_evict_lru_document`), `:2482-2513` (`_check_memory_limits`, two `while` loops). Defaults `:101-102` `max_indexed_files=100`, `max_total_chunks=10000` — at `BATCH_SIZE=25` (`:566`) a 10,000-chunk corpus is exactly the report's ~400 Lemonade calls.
- Edit: two strengtheners the report misses, and one softener. (a) `_check_memory_limits` evicts in a `while` loop (`:2489-2513`), so crossing the cap by several files pays the full re-index **once per evicted file**, not once. (b) Say the lock explicitly: `_state_lock` is held across all of it (`:2327`), so every concurrent `query`/`index` blocks for the duration. (c) Soften "`except Exception: return False`": it does `log.error` first (`:2414`) and the `False` is honoured by `_evict_lru_document:2478` and `_check_memory_limits:2496`, which `log.warning` and break — the limit is genuinely left unmet, but it is **not** silent, so drop the implied CLAUDE.md fail-loudly framing.
- Severity: keep 🟡. A latency/throughput cliff on an edge path (corpus at the cap), not data loss and not a normal-use failure.
- Dup: none.

### I50 CONFIRMED-ADJUST (one sub-claim REFUTED)
- Evidence: read `rag/sdk.py:322-412` and `:2811-2845`. Tamper path: `_verify_and_load_cache:408-411` raises `ValueError("Cache integrity check failed — file may have been tampered with")`; the caller `:2832-2844` catches bare `Exception`, emits `log.warning("Cache load failed: {e}, reindexing")` + a `show_stats` print reading **"Cache outdated or corrupted"**, then `os.remove()`s both the cache and its `.sig`. Cache key `_get_cache_path:446-460` = `sha256(path)[:16] + sha256(content)[:32]` only — no `chunk_size`, `chunk_overlap` or `use_llm_chunking`, and the cached blob contains `chunks` (`_save_cache:353-374`). `cache_dir: str = ".gaia"` at `:95` (CWD-relative) vs the key at `Path.home()/".gaia"/"cache"/"hmac.key"` (`:336-337`).
- Edit: **delete the 0-byte-key sentence — it is false.** Probe: `hmac.new(b"", data, sha256)` signs and verifies fine, so a truncated `hmac.key` self-heals after one reindex (old caches fail once, get deleted, and are re-signed with the empty key); it does **not** "make every cache fail forever". The real defect there is the opposite one — worth substituting: `_get_hmac_key:339-341` accepts whatever `hmac.key` contains with **no length check**, so a zero-length or attacker-chosen key silently downgrades the signature to forgeable. Also correct "silently deleted": it is logged at `warning` and printed under `show_stats` — the accurate charge is that **a signature failure is reported with the same wording and the same recovery as ordinary staleness ("Cache outdated or corrupted") and then destroys the evidence**, so #768's integrity signal is unactionable. Line refs `2833-2843` → `2832-2844`; `445-448` → `446-460`.
- Severity: keep 🟡.
- Dup: none.

### I51 CONFIRMED
- Evidence: read `database/mixin.py:176-229` — `query_readonly` does `cursor = self._db.execute(sql, params or {})` then `rows = [dict(row) for row in cursor.fetchall()]` (`:214-215`); `grep` for `set_progress_handler|max_rows|LIMIT|timeout` in the file → **no hits**. Tool wrapper `database/agent.py:109-128` returns `{"rows": rows, "count": len(rows)}` uncapped. Probe through the real `gaia.database.sql_safety.readonly` authorizer: 3 M-row recursive CTE ran to completion, `denied: []`; a 300,000-row `SELECT` materialised 300,000 dicts, `denied: []`; `SELECT length(randomblob(50000000))` allowed. Scratchpad probe against the real `_strip_sql_string_literals`: `WITH t AS (SELECT 1 x) SELECT * FROM t` → REJECT (not SELECT); `SELECT * FROM scratch_a -- update the totals later` → REJECT (UPDATE); `SELECT * FROM pragma_database_list` → PASS; `SELECT name FROM sqlite_master` → PASS.
- Edit: two additions. `pragma_table_list` and `SELECT load_extension('x')` also pass the deny-list (`scratchpad/service.py:271-285`) — the `\b[A-Z]+\b` tokenizer can never see `PRAGMA` inside `pragma_database_list` because `_` is a word character. And name the missing primitive by name in the fix: SQLite exposes `conn.set_progress_handler(cb, n)` (plus an explicit row cap), which is exactly what neither path installs.
- Severity: keep 🟡. Local, self-inflicted DoS / context blow-up driven by the model's own SQL; no privilege crossing and the authorizer still blocks writes.
- Dup: none.

### I52 CONFIRMED-ADJUST (one parenthetical REFUTED)
- Evidence: read `code_index/sdk.py:1029-1039` — the `ntotal` check **does exist**: `expected = len(meta.get("chunks", []))`, `if index.ntotal != expected: log.warning("Index/metadata mismatch…"); return False`, in `_ensure_index_loaded`. The comment the report cites (`:965-968`, "`_load_metadata` will detect stale metadata via ntotal check") names the **wrong function** — `_load_metadata:985-1003` only checks file-existence and `version`. Consequences confirmed: it is a **count** check, so metadata rotated by one keeps `len(chunks)` equal and sails through; and `get_status:502-520` calls `_load_metadata` only, so a count *mismatch* yields `{"indexed": True, …}` while `search` (gated on `_ensure_index_loaded`) returns `[]`. Sandbox ratchet confirmed at `agents/tools/code_index_tools.py:125-152`: the guard compares `repo_path` against `self._repo_path` and then does `self._repo_path = resolved` (`:151`), so each successful call permanently narrows the ceiling — a sibling repo is unreachable for the rest of the agent's life.
- Edit: replace "comment at `:967-968` claims an `ntotal` check that does not exist" with "the `ntotal` check exists but lives in `_ensure_index_loaded` (`:1029-1037`), not in `_load_metadata` where the comment at `:965-968` places it — and it compares **counts only**, so rotated metadata passes". Also add the second half of the split-brain, which is the sharper symptom: `get_status():502-520` never runs the check, so it reports `indexed: True` for a cache that `search` refuses to load.
- Severity: keep 🟡. The rotation case needs a local writer of `~/.gaia`'s code-index cache; the split-brain `indexed: True` / empty-search case needs only a crash between the two `replace()`s (`:972-975`) and is the part that fires without an attacker.
- Dup: none; distinct from I50 (RAG cache, HMAC-signed) — worth one cross-reference since the *pattern* (unsigned/weakly-checked cache feeding the model context) is shared.

### I53 CONFIRMED
- Evidence: read `web/tavily.py:303-308` (`_credits_used` = `SELECT COALESCE(SUM(credits),0) FROM tavily_ledger` — **no session, run-id or time predicate**) and `:328-348` (`_check_budget` compares `used + est` against `self._budget.cap`); ledger is on disk at `_DEFAULT_DB_PATH = C:\Users\…\.gaia\tavily_cache.db` (`:162`). Docs say session: `cli.mdx:2967` "Credit cap for the **session**", `cli.py:1770` help "Credit cap for **this session**". Probe (temp db, fake key, no network): seeded 12 credits, then a **brand-new client with `cap=5`** → first `_check_budget("search","basic")` raised `TavilyBudgetExceeded: 12 credits used + ~1 for this search would reach 13, but the cap is 5.`
- Edit: none to the mechanism. Optional precision: name the doc line (`cli.mdx:2967`) and the CLI help string (`cli.py:1770`) — both say "session", which is what makes this a doc-contradicts-code finding rather than a design preference. The ledger has `created_at` (`:298-300`), so the fix is a predicate, not a schema change.
- Severity: keep 🟡. Feature is unusable after normal accumulated use, but it fails *closed* (over-blocking, never over-spending) and `--no-block` is an escape hatch that the error message itself names.
- Dup: none.

### I54 CONFIRMED-ADJUST
- Evidence: read `hub/installer.py:1139-1165`. Order is `_write_agent_yaml:1139` → `_write_sentinel:1142-1150` → `_add_wheel_agent_to_active_env_path:1156-1161`. That last one raises `InstallError` on `OSError` by design (`:797-827`, docstring cites CLAUDE.md fail-loudly). The handler `:1161-1165` calls `_restore_backup_if_present(agent_id, root)` then re-raises.
- Edit: qualify the "no self-heal". `_restore_backup_if_present` (read in full) **returns immediately when no backup exists**, and it only `rmtree`s the install dir when one does. So on an **upgrade** the bad sentinel is wiped with the restored tree and the state is consistent; the split state is specific to a **first install of that agent** (no backup ⇒ the sentinel written at `:1142` survives the raised `InstallError`). Rewrite as: "…on a **first** install (no backup to restore) a non-writable site-packages leaves the `.installed` sentinel on disk after the install has raised, so the agent reads as installed but is not importable outside the installing process."
- Severity: keep 🟡. Edge path (non-writable active-env site-packages), and re-running `gaia hub install` re-attempts the `.pth`.
- Dup: none.

### I55 CONFIRMED-ADJUST (line refs wrong on two sub-claims; one sub-claim OVERSTATED)
- Evidence, sub-claim by sub-claim:
  - **Mic failure → stuck on "Listening…"**: the cited `audio_recorder.py:154-156` is the **base class**, and `gaia talk` does not run it. `WhisperAsr(AudioRecorder)` (`whisper_asr.py:34`) overrides the loop: `_record_audio_streaming:97-172` ends `except Exception as e: log.error("Error reading from stream"); break` at **`:160-162`**, `is_recording` set at `:176` (`start_recording_streaming`) and never cleared here, so `_process_audio`'s `while self.is_recording:` (`:189`) spins on. Mechanism confirmed, file/line wrong.
  - **`_check_mic_levels` hides device errors at debug**: confirmed, `audio_client.py:402-403` `except Exception as e: self.log.debug("Mic level check skipped: %s")`.
  - **Enter-to-interrupt**: `talk/sdk.py` has **zero** `input()`/`stdin`/keyboard references (grep); `start_voice_session:240-272` → `audio_client.start_voice_chat` → `_process_audio_wrapper:410-518`, none of which installs a listener. Meanwhile `start_voice_chat:87` prints the banner **"Press Enter key to stop during audio playback."** at runtime.
  - **The `input()`-thread leak**: `keyboard_listener` lives in `AudioClient.process_voice_input:193-211` — and that method **has no caller in `src/`, `hub/`, or any live path** (`grep -rn process_voice_input`: only `docs/spec/audio-client.mdx:112,117,282` and `tests/unit/test_asr.py`, which replaces it with an `AsyncMock`). `TalkSDK.process_voice_input:197-224` is a different method with no listener.
  - **`"__HALT__"` unknown to `kokoro_tts.py`**: literally true (`grep __HALT__` → one hit, `audio_client.py:206`; kokoro only knows `"__END__"`, `:379-391`) — but **inert**: `keyboard_listener` sets `interrupt_event` at `:205` *before* the put at `:206`, and kokoro's branch is `chunk == "__END__" or (interrupt_event and interrupt_event.is_set())` (`:380-382`), so the sentinel is consumed on the interrupt branch and never spoken.
  - **TTS-thread death hangs the LLM stream**: confirmed — `text_queue = queue.Queue(maxsize=100)` (`audio_client.py:214`, `:353`) with blocking `put` (`:276-293`), and `kokoro_tts.py:333-341` creates and starts `sd.OutputStream` **above** the `try:` at `:375`, so a device failure kills the consumer thread before the loop and the producer blocks forever at chunk 100.
  - **Docs say "exit"/"quit"**: confirmed — `docs/guides/talk.mdx:79,87` vs `audio_client.py:434 if cleaned_text in ["stop"]`. `talk.mdx:95` "Natural pauses (>1 second)" corresponds to no constant on the live path (`WhisperAsr` has no `SILENCE_LIMIT`; only a per-chunk energy gate, `:147-151`).
- Edit: (1) change `audio_recorder.py:154-156` → `whisper_asr.py:160-162` (with `:176`/`:189` for the un-cleared flag). (2) **Delete the "leaks one `input()` thread per turn, oldest waiter wins — *probe*" clause** and replace with the stronger, simpler fact: "*the only Enter-to-interrupt implementation lives in `AudioClient.process_voice_input:193-211`, which nothing calls — `docs/spec/audio-client.mdx:112` documents it as the API and `tests/unit/test_asr.py` only ever mocks it, so no live path has ever run it, while `start_voice_chat:87` prints 'Press Enter key to stop during audio playback' on every launch.*" (3) Drop or footnote the `__HALT__` clause — `interrupt_event` covers it. (4) Add the mitigation the report omits: `_process_audio_wrapper:501-512` does print one actionable "No speech detected… try `gaia talk --audio-device-index <N>`" after 10 s, so the stuck state is not *completely* silent — it is silent *after* that one line.
- Severity: keep 🟡, but note this is really five findings sharing one bullet; the banner-promises-a-feature-that-does-not-exist part (`start_voice_chat:87`) is the one a user hits on **every single launch**, which is the closest thing here to 🔴-by-normal-use. Recommend splitting that clause out so it is not buried.
- Dup: overlaps C25/C26 (`gaia talk` transcribes its own voice / ignores `--model`, #124) — same command, different mechanisms; keep separate but cross-reference.

### I56 CONFIRMED
- Evidence: `outlook_backend.py:195` `"internalDate": msg.get("receivedDateTime") or ""` (ISO-8601). The three consumers all **guard** the `int()` — `read_tools.py:412-418` `_thread_message_sort_key` (`try: int(...) except (TypeError, ValueError): return 0`), `read_tools.py:1655-1665` `_parse_epoch_millis` (same), `reply_tools.py:164-168` `_internal_ms` (same) — so the report's "**collapse to 0**" is the correct mechanism and the "it would raise" hypothesis is wrong. Probe against the real functions: `_thread_message_sort_key ISO -> 0 | GMAIL -> 1767225600000`; `_parse_epoch_millis ISO -> 0`; `_internal_ms ISO -> 0`; `_timestamp_ms ISO -> 1767225600000` and `_compute_reply_latency_seconds ISO -> 21305828` (the two ISO-aware consumers); bare `int(iso)` raises `ValueError`. Downstream, all three sub-impacts re-derived: **ages vanish** — `read_tools.py:1859-1860` `age_seconds = ... if internal_date else None`, so `0` gives `age_seconds: None`; **ordering degrades** — `read_tools.py:349,509,631` sort by an all-zero key (stable sort, so a no-op); **"latest wins" never fires** — `reply_tools.py:322` `if current is None or _internal_ms(msg) > _internal_ms(current[1])`, and `0 > 0` is False, so `draft_reply` keeps the *first* message it saw. Probe: `sorted([old, new], key=_internal_ms)` gave keys `[0, 0]`, order unchanged.
- Edit: none to the mechanism. Two precision improvements: (a) name the internal inconsistency — `reply_tools.py` contains **both** the correct ISO parser (`_compute_reply_latency_seconds:87-119`, whose docstring explicitly documents the Outlook ISO case) and the broken one (`_internal_ms:164-168`) **in the same file**, which is the sharpest evidence this is an oversight, not a design choice; the other correct one is `followup_tools._timestamp_ms:57-86`. (b) Note one mitigation: `outlook_backend.py:514` pre-sorts a thread lexicographically by the ISO string, which happens to be chronologically correct, so `get_thread` ordering survives — it is `read_tools`' *defensive re-sort* that becomes a no-op, and the un-mitigated damage is `age_seconds: None` plus `draft_reply` targeting.
- Severity: keep 🟡. Outlook-only, degrades triage quality and reply targeting; no data loss, and it never fires for Gmail users.
- Dup: none. Closely related to I57/I59 (all Outlook-parity gaps) — worth grouping under one "Outlook parity" heading.

### I57 CONFIRMED
- Evidence: read `outlook_query.py:94-150` — only `_IS_RE` (`is:unread|read`, `:45`) and `_DURATION_RE` (`newer_than|older_than`, `:47-52`) become a `$filter`; everything else falls through to `_graph_search_param(query)` at `:150`. Probe against the real `translate_query`: `after:2026/01/01` gives `search='"after:2026/01/01"'`; `before:...`, `label:promotions`, `has:attachment`, `is:starred` all become quoted dead phrases with `filter=None`. Steering confirmed at `read_tools.py:2839` (`label:promotions`) and `:2842-2843` ("Date operators require `YYYY/MM/DD` — e.g. `after:2026/07/01 before:2026/07/08`"). #2996 confirmed **OPEN** with the matching title.
- Edit: two qualifiers. (a) `from:` and `subject:` **do** work — they are genuine Graph KQL scoping keywords inside the quotes (`:103-107`), and the `CHANGELOG.md` "Unreleased" entry for #2996 claims exactly that partial fix; so write "**every Gmail operator except `from:`/`subject:`/`is:unread`/`is:read`/`newer_than:`/`older_than:`**" rather than "most ... untranslated". (b) The failure is silent only for a *pure* search query — mixing a dead operator with `is:unread` raises a loud `ValueError` (`:135-143`; probe: `is:unread label:promotions` gave ValueError). Add: "the CHANGELOG already records #2996 as fixed while the issue is still open and `after:`/`label:`/`has:` remain dead — the partial fix closed the `from:`/`subject:` half only."
- Severity: keep 🟡.
- Dup: none.

### I58 CONFIRMED
- Evidence: `read_tools.py:145-177` — `body` is the only field that gets `normalize_email_body` + `wrap_untrusted_body` (`:160-161`, `:173`); `subject`, `from`, `to`, `date`, `snippet` are copied raw (`:167-172`). `_format_message_metadata_for_llm:180-208` is the same shape with no body at all. Prompt builders confirmed: `llm_triage.py:195-201`, `summarize_tools.py:108-114`, `calendar_tools.py:413-417` — each interpolates `Subject: {subject}` / `From: {sender}` **outside** the delimiter pair and wraps only the body. `normalize_email_body:158-192` performs `scrub_delimiter_tokens(body)` unconditionally, so the scrub exists — and `snippet` (Gmail's ~200-char body prefix; Outlook's `bodyPreview`, `outlook_backend.py:194`) is a verbatim body prefix that never passes through it. That is the bypass.
- Edit: none required; mechanism, sites and bounding are all accurate. Optional strengthener: the `snippet` bypass is not merely "a prefix" — `_format_message_metadata_for_llm` is the **listing/counting** formatter, so a listing hands the model up to 100 unscrubbed, undelimited snippets at once with no body path to scrub them.
- Severity: **keep 🟡** — I checked the bounding and it holds. `run_autonomy_cycle:2201-2217` only builds `Proposal` objects and persists them to the GoalStore; it does not send. The mutating REST actions are archive/quarantine and each requires a single-use `POST /v1/email/confirm` token bound to `(action, message_id)` (`docs/guides/email.mdx:393`), and `agent.py:2235-2242` adds a batch confirmation above 5 ops / 3 senders. So the realistic outcome really is mis-classification or a misleading summary, not an attacker-driven send. Do **not** raise to 🔴 — the finding's own impact paragraph is the reason.
- Dup: none.

### I59 CONFIRMED-ADJUST
- Evidence: `grep -n "3234|3269|grouped|paren" hub/agents/email/python/CHANGELOG.md` returned **no hits**, while #3234 is CLOSED and PR #3269 ("fix(email): parse grouped Outlook duration queries") is MERGED. `outlook_query.py:47-52` `_DURATION_RE` is character-for-character the body of `gmail_query.py:38-48` `DURATION_OP_RE` (`\b(?P<op>newer_than|older_than):(?P<val>"[^"]*"|[^\s)}\]]+)`) — and `outlook_query.py:40` **already imports** `parse_gmail_duration_value` from that module, so the duplication is gratuitous. `docs/guides/email.mdx:639-641`: heading "Limitations (as of v0.23)" then "Outlook / Exchange — tracked in [#963]"; #963 is **CLOSED**.
- Edit: strengthen the doc half — the contradiction is on the same page several times over: `email.mdx:204` ("Works across every connected mailbox (Gmail, personal Outlook, and work Microsoft 365)"), `:393` (Outlook-specific quarantine refusal plus `post_archive_id` for Outlook folder moves), `:433,442-444` (Outlook Calendar selection), `:629` (Outlook mailbox refusal message). Cite `:204` as the direct rebuttal to `:641`.
- Severity: **split.** The CHANGELOG-entry gap and the regex duplication are hygiene, not a bug on any path — those halves are 🟢. The `email.mdx:641` half stays 🟡: a user reading Limitations is told a shipped, documented, released feature does not exist, which is doc-contradicting-code under REVIEW.md's own 🟡 definition. Recommend splitting the bullet rather than leaving one mixed-severity item.
- Dup: the `email.mdx:641` half belongs with I56/I57 under an "Outlook parity" grouping.

## Summary (I46–I59)

**Counts (14 findings):**
- **CONFIRMED — 7:** I46, I49, I51, I53, I56, I57, I58
- **CONFIRMED-ADJUST — 7:** I47, I48, I50, I52, I54, I55, I59
- **OVERSTATED — 0** (as whole findings; one *sub-claim* in I55 is overstated)
- **REFUTED — 0** (as whole findings; two *sub-claims* are refuted, in I50 and I52)
- **UNVERIFIABLE — 0**

Every finding in this range is real. Nothing here should be dropped from the report; six need a correction to their stated mechanism or line refs, and three need a severity split.

**The five corrections that matter most**

1. **I50 — the 0-byte-`hmac.key` sentence is false; delete it.** Probe: `hmac.new(b"", ...)` signs and verifies consistently, so a truncated key self-heals after one reindex — it does not "make every cache fail forever". The real defect at that line is the opposite: `_get_hmac_key:339-341` accepts any key length, so a zero-length or attacker-planted key silently makes cache signatures forgeable. Substitute that.
2. **I52 — "an `ntotal` check that does not exist" is false.** It exists at `code_index/sdk.py:1029-1037`; the *comment* at `:965-968` merely attributes it to the wrong function. The defensible claim is that it is a **count-only** check (rotated metadata passes) and that `get_status:502-520` skips it entirely, so a cache `search` refuses to load still reports `indexed: True`.
3. **I55 — the `input()`-thread-leak sub-claim is about dead code.** `AudioClient.process_voice_input:193-211` has **no caller** in `src/` or `hub/`; it appears only in `docs/spec/audio-client.mdx` and as an `AsyncMock` in `tests/unit/test_asr.py`. Replace it with the stronger fact: no live path implements Enter-to-interrupt at all, yet `start_voice_chat:87` prints "Press Enter key to stop during audio playback." on every launch. Also: the stuck-mic mechanism is at `whisper_asr.py:160-162`, not the cited `audio_recorder.py:154-156` (the base class `gaia talk` never runs), and the `"__HALT__"` clause is inert — `interrupt_event` is set first and kokoro checks it on the same branch (`kokoro_tts.py:380-382`).
4. **I48 — the arithmetic is wrong for the default.** At `chunk_size=500` (~2,000 chars) truncation hides **40 %**, not 60–70 %. The 60–70 % figure describes the 768/1024-token settings `rag.mdx` itself recommends — which is the more damning framing, so state both.
5. **I47 — the report describes the milder half of the bug.** A *partially* short batch (`data` shorter than the 25 texts sent) is appended with **no length check, no fallback and no warning** (`rag/sdk.py:606-611`); only a fully-empty batch reaches the one-by-one path. Cite `:3075-3108` as the positional consumer, and `code_index/sdk.py:840-871` as the fail-loud contract that already exists one package over.

**Proposed severity changes**

- **I46 — argued for 🔴 and rejected; keep 🟡.** A real bypass of a stated boundary, but loopback, RFC1918 and link-local metadata (169.254.169.254) are all still blocked, and 100.64/10 is only routable on a CGNAT / Tailscale / k8s-pod network. Add the attacker position and that gate to the finding text; cite closed issue #956 as the previous hardening of the same predicate.
- **I47 — argued for 🔴 and rejected; keep 🟡**, but flag it as top-of-🟡: it produces silent wrong answers, and it is a direct "No Silent Fallbacks" violation whose correct implementation already exists in `code_index`.
- **I58 — argued for 🔴 and rejected; keep 🟡.** I independently verified the bounding: `run_autonomy_cycle:2201-2217` only persists `Proposal`s, and the mutating REST actions need a single-use confirm token (`email.mdx:393`).
- **I59 — split 🟡 / 🟢** (doc contradiction stays 🟡; CHANGELOG gap and duplicated regex drop to 🟢).
- **I55 — split.** The "Press Enter" banner promising a feature no live path implements fires on every launch and should not be buried inside a five-mechanism bullet.

**Structural note for the report.** I56, I57 and the `email.mdx:641` half of I59 are three faces of one thing: Outlook shipped and was documented as shipped, but the timestamp parser, the query translator and the Limitations section were never brought along. Grouping them under one "Outlook parity" heading tells that story in one line instead of three.

---

## §3.7–3.8 Frontends and C++ (I60–I71)

# Adversarial verification — I60–I71 (frontends + C++/experiments)

Checkout: detached HEAD @ 211f08c5. All citations re-opened at this commit.

### I60 CONFIRMED-ADJUST
- Evidence: `tui/internal/ui/preflight/local.go:128-152` walks up from `os.Getwd()` for `.env`; `src/gaia/llm/providers/claude.py:156-158` calls bare `load_dotenv()`, and the installed `dotenv/main.py find_dotenv` resolves the start dir as **`os.getcwd()` when `getattr(sys,"frozen",False)`** (I read the function source in `.venv/Lib/site-packages/dotenv/main.py`) — so for a FROZEN sidecar it matches preflight exactly, and `tui/internal/client/subprocess.go:145` never sets `cmd.Dir`, so cwd is inherited. The divergence is the **non-frozen** case: `find_dotenv` then walks up from `claude.py`'s own directory (site-packages), which for a non-editable pip install is nowhere near the TUI's cwd.
- Edit: replace "the agent's `load_dotenv()` searches from the frozen module's directory" with "the agent's bare `load_dotenv()` starts its walk at `claude.py`'s own directory (python-dotenv uses the calling frame's file, and `os.getcwd()` only when `sys.frozen`)". Add the gate: "fires for a **non-editable pip-installed** agent — a frozen sidecar resolves `.env` from cwd and matches preflight."
- Severity: keep 🟡 — the report's own "(Medium.)" is right, but it is narrower than written (does not fire for the frozen shipped sidecar, nor for an editable install inside the repo).
- Dup: none.

### I61 CONFIRMED
- Evidence: `tui/internal/client/subprocess.go:146-147` `stderr := &bytes.Buffer{}; cmd.Stderr = stderr`; stored at `:179` and only nil'd on cancel/close (`:522`, `:538`), read at `:345` — never truncated, so it grows for the process lifetime. `hub/agents/gaia/python/gaia_agent/stdio.py:811` is exactly `sys.stdout = sys.stderr`, confirming every stray `print` lands in that buffer.
- Edit: none required. Optional addition — the same `*bytes.Buffer` is written by `exec.Cmd`'s copy goroutine while `:345` calls `st.stderr.String()`, which is an unsynchronised concurrent read (a `-race` finding the report notes it could not run).
- Severity: keep 🟡.
- Dup: none.

### I62 CONFIRMED-ADJUST
- Evidence: only **three** hard-coded "30s" strings exist — `tui/internal/ui/components/confirmation.go:434` and `:455`, and `tui/internal/ui/chat/canonical.go:437`. `confirmation.go:175` is `const ConfirmationTimeout = 30 * time.Second` — i.e. the constant is in the *same* file, not "elsewhere"; `:171` is prose in its doc comment.
- Edit: rewrite as "…in three UI strings (`ui/components/confirmation.go:434,455`, `ui/chat/canonical.go:437`) while `ConfirmationTimeout` is defined at `ui/components/confirmation.go:175`." Drop "the constant lives elsewhere".
- Severity: change 🟡 → 🟢 — no user-visible bug today (both values are 30s); it is a drift hazard only. REVIEW.md 🟢 = "would not break or mislead a user".
- Dup: none.

### I63 CONFIRMED
- Evidence: `src/gaia/apps/webui/services/backend-installer.cjs:1282` — the guard is literally `if (IS_WINDOWS && !isDev) {`, with `isDev` set at `:1207` as `isPackaged === false || !process.resourcesPath`. So the block runs **only** in a packaged build. `:1284-1296` spawns `powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"`, and success reports `"uv ready (system, unverified)"` (`:1313`); `:1324-1330` then accepts any system `uv` on PATH unverified. Contrast the bundled path `:1178-1199`, which hard-fails on a SHA-256 mismatch. `installer/scripts/install.ps1:71` is the same `irm … | iex`. Attacker position: MITM / DNS-hijack of `astral.sh` against a machine that already trusts the signed GAIA installer — no click needed; the app runs it during first-run bootstrap.
- Edit: the "contradicting the file's own header" clause is unfair and should go. The `// DEV-ONLY … Never fires for end users.` comment at `:1205-1206` scopes the *`isDev`* block only; the packaged rescue at `:1276-1281` documents itself honestly as an end-user last resort. Replace with: "reached only when the build shipped **no bundled uv resource** for the platform and no system `uv` is present — the exact condition of closed issue #966 (`no bundled uv shipped for win-x64`), so it is a path that has really fired on shipping Windows builds."
- Severity: keep 🟡 🔒 — genuinely gated behind a missing bundled resource, so not "normal use"; but #966 shows the gate has opened in a real release. Do not promote to 🔴.
- Dup: I98 cites the same unpinned uv installer (report already cross-references); keep both, they are different delivery paths.

### I64 CONFIRMED
- Evidence: `src/gaia/apps/webui/services/backend-installer.cjs:1579-1586` — `if (initResult.code !== 0) { … log("Warning: gaia init exited with code … Continuing anyway.") }` followed unconditionally by `report(STAGES.GAIA_INIT, 100, "Lemonade Server setup complete")`; the run then reaches `setState(STATES.READY, …)` at `:1649`. Cited line range is exact.
- Edit: none.
- Severity: keep 🟡 (a first-run user is told setup completed when Lemonade/models may be absent — it degrades on the next launch, it does not destroy data).
- Dup: none — it is one instance of the #3307 pattern the report already links; the executive summary at line 62 also lists it, which is fine.

### I65 CONFIRMED
- Evidence: `services/auto-updater.cjs:188-205` `_resolveFeedUrl()` reads only `GAIA_UPDATE_FEED_URL` or `feedUrl` in `~/.gaia/update-config.json` — it never reads the built `app-update.yml`; `:216-229` sets `STATES.NO_CHANNEL` when neither is present, and nothing in the repo sets either for a shipped build (grep for `GAIA_UPDATE_FEED_URL` outside `services/` and `tests/electron/` returns only a comment in `electron-builder.yml:92`). `electron-builder.yml:95-104` states the `publish.url` "exists only because electron-builder requires a non-empty, expandable `url`" and notes the R2 feed is "not live yet (#1719)". No scheme check on the resolved URL → a `http://` feed is accepted. `win:` block (`electron-builder.yml:111-120`) has no `publisherName`, and `forceCodeSigning: false` at `:108`; electron-updater skips Windows signature verification entirely when `publisherName` is unset. `docs/guides/install.mdx:102-109` promises "GAIA checks for updates automatically … 10 seconds after launch / Every 4 hours". `VersionPicker.tsx` exists in `src/components/`.
- Edit: cite `install.mdx:102-109` (not 101-109) and `auto-updater.cjs:188-231`.
- Severity: keep 🟡 🔒. The `http://` + missing-Authenticode parts are only reachable once someone configures a feed, so they are a latent hardening gap, not a live exploit; the live user-facing defect is the doc promising a feature that is off in every shipped build. Recommend the report say so explicitly.
- Dup: none. Tracking: #1724 (CLOSED — "R2-primary auto-update: generic feed + mutable latest channel") and #1731 (OPEN epic) are the better references than the report's #1729; #2383 (CLOSED) fixed the resume-after-NO_CHANNEL case. Suggest changing "Tracked: #1729 (partial)" → "Related: #1724 (closed, feed design), #1731 (open epic)."

### I66 CONFIRMED
- Evidence: `services/agent-process-manager.cjs:857-871` `_sendJsonRpcRaw` ends in `entry.process.stdin.write(payload)` at `:870`; the only `"error"` listener in the whole file is `child.on("error", …)` at `:184` (spawn failure, not the stdin stream) — no `stdin.on("error")` anywhere. `HEALTH_CHECK_INTERVAL = 30000` at `:42`, driven by `setInterval` at `:197-199`. `main-safety-net.cjs:119` `process.on("uncaughtException", (err) => fatal(err))`; `fatal()` ends `try { process.exit(1); }` at `:115` (and `process.exit(2)` at `:83` on re-entry) — both cited lines exact. `main.cjs:1118-1156` puts `agentProcessManager.stopAll()` inside the `will-quit` handler, which `process.exit()` bypasses. The file's own header says `installLogTee` was the "root-cause fix for #934" (stream errors without listeners) — the same class of bug, unfixed for the agent's stdin.
- Edit: none. Optional strengthening: note that the repo already knows this failure mode (#934, `installLogTee`) and simply never applied it to `child.stdin`.
- Severity: keep 🟡 (needs a sidecar to die between the liveness check and the write — a narrow race, but the consequence is a full app exit that orphans the backend).
- Dup: the report's C27 is also an "agent crash loop crash-exits the app" finding; keep both but cross-reference.

### I67 OVERSTATED
- Evidence: the code is exactly as described — `src/gaia/electron/src/preload/preload.js:17-20` exposes `invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args)`, and `src/gaia/electron/src/services/base-ipc-handlers.js:92-95` does `shell.openExternal(url)` with no scheme check. **But nothing loads this framework.** The only entry point is the root `package.json:8` script `app:jira:run:electron` → `src/gaia/apps/jira/webui`, a directory that **does not exist** (`src/gaia/apps/` holds only `_shared`, `example`, `llm`, `webui`); the example app has its own `src/main.js`/`preload.js` and requires only `electron`, `path`, `dotenv`, `electron-squirrel-startup`; the Agent UI uses `src/gaia/apps/webui/main.cjs`. No packaged artifact, no CI build target. It reaches users through zero shipping surfaces — only `tests/electron/*.js` reference `FRAMEWORK_PATH`.
- Edit: drop the 🔒 and reframe as dead code + a stale spec. Suggested replacement: "**I67. `src/gaia/electron/` is dead framework code with an unsafe shape, still documented as shipping.** Its preload exposes a generic `invoke(channel, …args)` and `base-ipc-handlers.js:92` calls `shell.openExternal(url)` unvalidated — but no app loads it: the root `package.json:8` script points at `src/gaia/apps/jira/webui`, which was deleted, and the Agent UI uses `main.cjs`. `docs/spec/electron-integration.mdx:25` still calls it the shell 'currently shipping the flagship Agent UI (`webui`)' and pins 'Electron 31.0.0+' (`:16`, `:548`) against the root `^44.0.0`. Delete the package or mark it archived; the spec is describing a shell that does not exist."
- Severity: change 🟡 🔒 → 🟢 for the code (unreachable, ships to nobody) — the doc contradiction alone would be 🟡; net 🟢/🟡, and the 🔒 marker is not warranted.
- Dup: none. Also note the doc citation is line **25**, not 26.

### I68 CONFIRMED
- Evidence: probe — I extracted `MessageBubble.tsx:130-240` verbatim into `%TEMP%\critIE\fns.js` and ran the real functions under node 22. `stripBogusCodeFences`: ```` ```shell ```` → `"Run this:\n# install deps\nnpm ci\nDone."` (fence gone; `# install deps` becomes an H1), and `console`/`cmd`/`mermaid`/`jsonc`/`env` all unwrap identically, while ```` ```bash ```` survives. `cleanLLMJsonBlocks`: `'Here is a tool definition:\n\n```json\n{"tool": "search_web", "tool_args": {...}}\n```\n\nUse it.'` → the fence body is emptied (```` ```json\n\n``` ````); `'Example: {"tool": "x", "args": {"a":1} -- …EVERYTHING AFTER THIS IS LOST.'` → `"Example: "` — the whole tail is discarded at `:167` (`if (depth !== 0) { break; }`). `cleanToolCallContent(message.content)` is applied to **every** message at `:308-311`, user messages included; `handleCopy` at `:326-328` copies raw `message.content`.
- Edit: two corrections. (a) `stripBogusCodeFences` is at `:225-240`, not `:197-240` — `:197-222` is the `KNOWN_CODE_LANGS` allowlist. (b) the marker match is the literal token `"tool"` / `"thought"` / `"answer"` within the **first 50 characters** after the `{` (`:141-143`), so `{"tool_choice":…}` is *not* matched — the report's "any `{…}` mentioning `"tool"`" overstates the trigger; say "any `{…}` whose first 50 chars contain the exact key `"tool"`, `"thought"` or `"answer"`". Worth adding: the cleaner also runs on the **user's own** messages, so a pasted tool schema is mangled in the user's bubble.
- Severity: promote 🟡 → 🔴. This fires in normal use with no edge condition: asking the shipped agent anything that produces a ```` ```shell ```` block or a JSON tool schema silently deletes or re-renders content, and the unbalanced-brace case truncates the rest of the answer with no indication. REVIEW.md 🔴 covers "a bug that fires in NORMAL use".
- Dup: none.

### I69 CONFIRMED
- Evidence: `grep -rni "content-security-policy|session.webRequest|onHeadersReceived"` over all of `src/gaia/apps/webui` (excluding `node_modules`/lockfile) returns **zero** code hits; `src/gaia/apps/webui/index.html` (13 lines) has only charset/viewport/icon/title metas. The same grep over `src/gaia/ui` and `src/gaia/api` — the FastAPI server that serves this bundle for `gaia chat --ui` in a real browser — is also empty. Corroboration the report should quote: `src/components/render/ImageCard.tsx:10` says "no CSP in this app, so this allowlist is the actual security boundary."
- Edit: add the two things that bound it, so the finding is not read as unmitigated — `main.cjs:505-506` sets `nodeIntegration: false, contextIsolation: true`, so renderer XSS does not get Node; and add that the browser-served path (`gaia chat --ui`) has no CSP header from the Python server either, where the Electron mitigations do not apply. Cite `ImageCard.tsx:10` as in-repo acknowledgement.
- Severity: keep 🟡 🔒 — defence-in-depth gap, not itself an exploit; it becomes the amplifier for any of the report's other injection findings. Attacker position: needs an existing injection into rendered content (e.g. LLM-controlled markdown), not remote-unauthenticated on its own.
- Dup: none found by `gh issue list -R amd/gaia --search "Content-Security-Policy"` (0 results; a CSP/renderer-headers search returns only unrelated browser-use issues #458/#459). Report should say "no tracking issue".

### I70 CONFIRMED-ADJUST
- Evidence: `cpp/src/sse_parser.cpp:72-74` returns silently when the event has no `choices` array (an `{"error":…}` event lands here) and `:92-94` is `} catch (...) { // Silently skip malformed JSON }` — both cited lines exact. `cpp/src/agent.cpp:661` is the `throw std::runtime_error("Streaming response contained no tokens")`. **But** `cpp/src/lemonade_client.cpp:385-416` — the range the report cites as the defect — is actually the *recovery* code: it re-parses `rawBytes` and does print the `--ctx-size` remedy. It only fails when the error is SSE-**framed**, because `rawBytes` then contains `data: {…}\n\n` and `json::parse` at `:387` throws into the swallow at `:412-414`. The duplication claim holds: `:311-330` (blocking) and `:387-410` (streaming) are near-identical ~25-line blocks including the identical remedy string.
- Edit: rewrite the first clause so the citation matches the mechanism — "C++ streaming mode recovers a plain-JSON error body (`lemonade_client.cpp:385-416`) but throws away an **SSE-framed** one: `rawBytes` holds raw `data:` framing, so `json::parse` at `:387` throws into the `catch (...)` at `:412-414` and the user gets `agent.cpp:661` 'Streaming response contained no tokens' instead of the `--ctx-size` remedy."
- Severity: keep 🟡. Not probed — building `cpp/` was out of scope; the `json::parse` failure on `data:`-prefixed text is deterministic, so this is read-verified rather than executed. Flag it as such rather than leaving it in the report's "confirmed by probe" bucket.
- Dup: none.

### I71 CONFIRMED-ADJUST
- Evidence: `git ls-files experiments/whatsapp-webjs` → only `index.js` and `package.json`. `index.js:26` `new Client({ authStrategy: new LocalAuth() })` writes `.wwebjs_auth/`, and `git check-ignore -v experiments/whatsapp-webjs/.wwebjs_auth/session/x` matches **nothing** — that half is right. But `git check-ignore -v …/run.log` prints `.gitignore:57:*.log`, so **`run.log` IS already ignored**; the report (and `_08_cpp_experiments.md`, which grepped only for `wwebjs|run.log|experiments`) missed the generic `*.log` rule. `docs/plans/messaging-integrations-plan.mdx:176` "**WhatsApp.** Deferred." and `:343`, and no doc anywhere references the directory. `package.json` maps `"test": "node index.js"` onto an interactive QR-login bot.
- Edit: drop `run.log` from the claim. Replace with: "…whose `.wwebjs_auth/` WhatsApp Web session credentials have no ignore rule (`run.log` is covered by the generic `*.log` at `.gitignore:57`) and no doc references it…". Keep the `"test": "node index.js"` point from the raw sub-review — it is the sharper detail and is absent from REPORT.md.
- Severity: keep 🟢/🟡 as filed — nothing is committed today; it is a foot-gun for whoever runs the spike. Note `index.js:8-13` already carries a privacy note and redacts bodies by default, which the report's phrasing does not credit.
- Dup: none.

---

## Summary (I60–I71)

**Counts:** CONFIRMED 6 (I61, I63, I64, I65, I66, I68, I69 — 7) · CONFIRMED-ADJUST 4 (I60, I62, I70, I71) · OVERSTATED 1 (I67) · REFUTED 0 · UNVERIFIABLE 0.
Exact: **7 CONFIRMED, 4 CONFIRMED-ADJUST, 1 OVERSTATED**.

Every cited file was re-opened and every line number re-checked. One executable claim (I68) was re-run as a node probe against the functions extracted verbatim from `MessageBubble.tsx`; all others are read-verified.

### The five corrections that matter

1. **I68 is under-ranked — it should be 🔴, not 🟡.** The probe confirms ```` ```shell ````, `console`, `cmd`, `mermaid`, `jsonc` and `env` fences are all unwrapped into prose, and that a `{"tool":…}` with an unbalanced brace silently deletes **every remaining character of the message**. This needs no adversary and no edge condition, it runs on user messages too, and `MessageBubble.fence.test.tsx` covers none of it. It is the most user-visible defect in this block.
2. **I67 is OVERSTATED and should lose its 🔒.** The unsafe `invoke(channel, …args)` preload and unvalidated `open-external-link` are real, but nothing loads `src/gaia/electron/`: its only entry point (`package.json:8`) points at `src/gaia/apps/jira/webui`, a directory that no longer exists, and the Agent UI runs `main.cjs`. It ships to nobody. What survives is a stale spec (`electron-integration.mdx:25`, not `:26`) claiming it is the shell behind the Agent UI.
3. **I60's stated mechanism is wrong.** python-dotenv's `find_dotenv` uses `os.getcwd()` precisely when `sys.frozen` is set — so a frozen sidecar matches the Go preflight exactly (`subprocess.go:145` never sets `cmd.Dir`). The divergence exists only for a **non-editable pip-installed** agent, where the walk starts at `claude.py`'s own site-packages directory. Fix the sentence or the finding reads as false.
4. **I70's citation points at the wrong code.** `lemonade_client.cpp:385-416` is the block that *does* print the `--ctx-size` remedy; the real failure is that `rawBytes` carries SSE `data:` framing, so `json::parse` at `:387` throws into the `catch (...)` at `:412-414`. Also: this one was not probed — it should not sit in the report's "reproduced by probe" bucket.
5. **I71 half-refuted on the facts.** `run.log` **is** ignored by the generic `*.log` at `.gitignore:57`; only `.wwebjs_auth/` is unignored. The raw sub-review grepped for `wwebjs|run.log|experiments` and missed the wildcard rule. `index.js:8-13` also already redacts message bodies by default, which the report's wording does not credit.

### Proposed severity changes

| Finding | Report | Proposed | Why |
|---|---|---|---|
| I68 | 🟡 | **🔴** | Fires in normal rendering; silently truncates answers. |
| I67 | 🟡 🔒 | **🟢** (drop 🔒) | Dead code loaded by no shipping app; only the stale spec is a real 🟡. |
| I62 | 🟡 | **🟢** | Drift hazard only — all four sites currently agree on 30s. |
| I63 | 🟡 🔒 | keep 🟡 🔒 | Gated on a missing bundled uv resource — but closed issue **#966** shows that gate has opened in a shipped Windows build, so it is not theoretical. Do **not** promote to 🔴; do drop the unfair "contradicts its own header" clause (the `Never fires for end users` comment at `:1205` scopes the `isDev` block, and the packaged rescue at `:1276-1281` documents itself honestly). |
| I65 | 🟡 🔒 | keep 🟡 🔒 | Retarget tracking: **#1724** (closed, feed design) / **#1731** (open epic) fit better than #1729. |

### Other required edits

- **I62** — the constant is at `ui/components/confirmation.go:175`, i.e. the *same* file as two of the three strings; drop "the constant lives elsewhere". Paths are `ui/components/confirmation.go` and `ui/chat/canonical.go`.
- **I68** — `stripBogusCodeFences` is `:225-240`; `:197-222` is the allowlist. The trigger is the exact key `"tool"`/`"thought"`/`"answer"` inside the **first 50 chars**, so `{"tool_choice":…}` is untouched.
- **I69** — add that `main.cjs:505-506` sets `contextIsolation: true, nodeIntegration: false` (bounds the Electron impact) and that the browser-served `gaia chat --ui` path has no CSP header from FastAPI either (where those mitigations do not apply). Quote `render/ImageCard.tsx:10` — the code already admits "no CSP in this app". `gh` finds no tracking issue.
- **I61** — optional addition: the `*bytes.Buffer` is written by `exec.Cmd`'s copy goroutine and read unsynchronised at `subprocess.go:345`, i.e. a data race as well as a leak.
- **I66** — optional strengthening: `main-safety-net.cjs` header names `installLogTee` as the "root-cause fix for #934" (stream `error` with no listener) — the repo already fixed exactly this class of bug and never applied it to `child.stdin`.

### Missed (≥🟡, found while verifying)

- **Dead workspace entries in the root `package.json`.** `"install:apps"` → `"install:jira"` → `cd src/gaia/apps/jira/webui`, and `"app:jira:build"` → the same path; `src/gaia/apps/jira/` does not exist at this commit. Both scripts fail immediately for any contributor following them, and `workspaces: ["src/gaia/apps/*/webui"]` plus `src/gaia/electron` still pull `@amd-gaia/electron` into `package-lock.json` for a package nothing runs. 🟡 (convention/dead code, adjacent to I67 — fix them together.)

---

## §3.9–3.10 Tests and eval framework (I72–I78)

# Adversarial verification — I72–I78 (tests-as-a-system + eval framework)

Verifier notes: all pytest runs below were executed on this Windows 11 box with
`.venv\Scripts\python.exe -m pytest <target> -q -p no:cacheprovider`.
`--timeout` could **not** be passed — `pytest-timeout` is not installed in this venv
(`error: unrecognized arguments: --timeout=120`), which is itself an independent
confirmation of one of I75's sub-claims.

---

### I72 CONFIRMED-ADJUST

- **Evidence — mechanism (read + reproduced).** `tests/unit/conftest.py:53-73` is exactly the
  `_block_network` autouse fixture the report cites (`@pytest.fixture(autouse=True)` at :54,
  `def _block_network(request, monkeypatch)` at :55, `monkeypatch.setattr(socket.socket, "connect",
  _blocked_connect)` at :73). Traceback from a single-test run of
  `tests/unit/connectors/test_tokens.py::TestRefresh::test_refreshes_when_expired`:

  ```
  Lib/socket.py:623  (_fallback_socketpair)   csock.connect((addr, port))
  args = (<socket.socket [closed] ...>, ('127.0.0.1', 59253))
  tests\unit\conftest.py:65: ConnectionError: Unit tests must not make real network connections
  ... plus PytestUnraisableExceptionWarning:
  asyncio/proactor_events.py:778 _close_self_pipe -> AttributeError: 'ProactorEventLoop' object
  has no attribute '_ssock'
  ```
  The mechanism is **exactly** what the report states: Windows `socket.socketpair()` falls back to a
  loopback `connect()`, the guard raises, and the ProactorEventLoop self-pipe never gets built. Both
  the `ConnectionError` and the `'_ssock'` AttributeError come from the same root cause (the report's
  source doc listed that as an unverified hypothesis — it is now verified; they are one bug).

- **Sub-claims (tallies re-run today):**
  - a. `tests/unit/connectors` = **33 F / 217 E** — **CONFIRMED exactly.**
    Actual: `33 failed, 508 passed, 12 skipped, 217 errors in 42.12s`.
  - b. `tests/unit/chat/ui` = **213 F / 42 E** — **CONFIRMED exactly.**
    Actual: `213 failed, 582 passed, 13 skipped, 42 errors in 56.81s`.
  - c. `test_memory_router.py` = **124 E** — **CONFIRMED exactly.**
    Actual: `3 passed, 124 errors in 26.19s`.
  - d. "hub/email router tests 53 F" — **CONFIRMED as a number, MISLABELLED as a location.**
    Actual: `53 failed, 45 passed, 2 skipped` — but the 53 are
    `tests/unit/test_hub_router.py` (35), `test_email_sidecar_router.py` (15),
    `test_email_sidecar_server_wiring.py` (2), `test_email_sidecar_proxy.py` (1). They live in
    `tests/unit/`, not in `hub/agents/email/python/tests/` — and "hub" here is the **Agent Hub
    router**, not the email hub package. A reader chasing "hub/email" looks in the wrong tree.
    (The hub package test trees do *not* load `tests/unit/conftest.py` at all, so they cannot be
    hit by this guard — source 06 recorded them as `92 skipped, 18 collection ERRORS`.)
  - e. "daemon+sidecar suites 157 F of 746" — **CONFIRMED by source, NOT re-run** (there is no
    `tests/unit/daemon` directory; the figure comes from source `03-servers-security.md:239,412`
    over a hand-picked file set: `test_daemon_hub_routes.py` 47, `test_daemon_agents_routes.py` 26,
    `test_daemon_custody.py` 19, `test_email_sidecar_router.py` 15, …). **This double-counts with
    sub-claim (d)** — `test_email_sidecar_router.py`'s 15 failures appear in both the "157" and the
    "53". The tally list reads as five disjoint buckets and is not.
  - f. **"Ubuntu-only CI never sees it" — REFUTED as written.** `test_unit.yml` has a
    **macos-latest** job (`:215`) that also runs `pytest tests/unit/` (`:250`). Both lanes are POSIX,
    where `socketpair()` is native, so the conclusion survives — but the stated reason does not.
  - g. **"GAIA's primary platform has no automated unit coverage" — OVERSTATED.** Windows CI lanes
    exist: `test_eval.yml:47` and `test_mcp.yml`/`test_security.yml:144` run on `windows-latest`,
    and `test_gaia_cli_windows.yml` is a self-hosted Windows lane. They are narrow
    (`test_gaia_cli_windows.yml:103` runs exactly one unit file, `tests/unit/test_daemon_secret_acl.py`,
    plus `-k Integration` lemonade tests), and they happen to select suites that don't trip the guard
    (`tests/unit/mcp` = 229 passed, `tests/test_eval.py` = 140 passed on this box). Accurate wording:
    *no CI lane runs the `tests/unit/` tree on Windows.*

- **Edit to REPORT.md (line 247):** replace the last two sentences of I72 with:
  > Tallies on this box (re-run): `tests/unit/connectors` 33 F / 217 E; `tests/unit/chat/ui` 213 F /
  > 42 E; `test_memory_router.py` 124 E; the hub/email-sidecar router files under `tests/unit/`
  > (`test_hub_router.py`, `test_email_sidecar_*.py`) 53 F; plus ~157 F across the daemon/sidecar
  > files (overlapping the previous bucket). No CI lane runs `tests/unit/` on Windows —
  > `test_unit.yml` is ubuntu-latest + macos-latest, and the Windows lanes
  > (`test_gaia_cli_windows.yml`, `test_eval.yml`, `test_mcp.yml`) each select a narrow file set that
  > misses the guard. Fix: allow loopback in the guard (or `WindowsSelectorEventLoopPolicy`), add a
  > `windows-latest` job to `test_unit.yml`.

  Also change "`test_memory_router.py` 124 E" → keep, and drop the flat "Ubuntu-only CI" clause.

- **Severity: change 🟡 → 🔴.** REVIEW.md's 🔴 bar is "a bug that fires in normal use". *Every*
  Windows contributor running `pytest tests/unit/` on a clean `main` checkout gets ~640 red
  tests that have nothing to do with their change — that is normal use, 100% reproduction, and it
  disables the primary local gate on the product's primary platform. The counter-argument (it is
  "only" test infra, product code is unaffected) is real, which is why I would not argue past 🔴
  into anything higher; but 🟡 under-ranks a defect with a 100% hit rate on the target platform.
  Note this also makes I72 the single highest-leverage fix in §3.9 — it is a ~5-line change to
  `_blocked_connect`.

- **Dup:** I72 is the root cause of the counts quoted in I73 (which mixes it with unrelated
  environmental failures) and overlaps `03-servers-security.md`'s daemon finding and
  `01-core-agent.md`'s duplicate of the same guard. Within REPORT.md itself: no duplicate ID —
  keep as one finding, but I73 should stop implying its own tallies are independent of I72's.

- **Tracked check:** none found.

---

### I73 CONFIRMED

Every sub-claim independently re-derived; two are stronger than the report says.

- **a. cp1252 `Σ` in an eval test helper — CONFIRMED (numbers exact).**
  `tests/unit/eval/test_scorecard_gate.py:44` is `path.write_text(render_scorecard(payload))` with no
  `encoding=`; the rendered template contains `Σ` at `src/gaia/eval/release_scorecard.py:255`
  (`Formula: round(100 × Σ(weightᵢ × valueᵢ) / Σ(weightᵢ), 2)`). Re-run:
  `tests/unit/eval` → `37 failed, 457 passed` — **29 in `test_scorecard_gate.py` + 8 in
  `test_release_scorecard.py`**, i.e. exactly the "37–44 F" the report claims. Product code does use
  `encoding="utf-8"`, so this is a test-only bug. Note the helper is at **:44**, not :46 as the
  source doc says (the report itself cites no line — fine).

- **b. `gaia/eval/claude.py:24 load_dotenv()` at import walking up ancestors — CONFIRMED, and it is
  worse than "a flaky test".** Line 24 is a bare module-level `load_dotenv()`. Probe (stripped
  `ANTHROPIC_API_KEY`, CWD = repo root):

      before import: False
      found dotenv at: C:\Users\14255\Work\gaia\.env      <-- TWO directories ABOVE the checkout
      after import:  True
      prefix: sk-ant-a

  So merely importing `gaia.eval.claude` injects a live API key sourced from **outside the
  repository** into the process environment. `tests/unit/eval/test_claude_judge.py::TestClaudeClientInit::test_raises_on_missing_api_key`
  fails in isolation (`Failed: DID NOT RAISE ValueError`, log line
  `Initialized ClaudeClient with model: claude-opus-5`) and passes inside the full `tests/unit/eval`
  run (an earlier module already imported it) — the classic order-dependent flake.
  **Suggest promoting the product half of this to its own sentence:** a credential silently read
  from an ancestor directory the user never pointed at is a supply-chain-shaped surprise, not just
  test hygiene.

- **c. `install.sh` missing an `eol=lf` `.gitattributes` entry — CONFIRMED.** Read `.gitattributes`
  in full: only binary-fixture rules (`cpp/tests/fixtures/*.png binary`,
  `tests/fixtures/skill_audit_digest/** -text`, `cpp/tests/fixtures/chunking/* -text`),
  vendored-source rules, and one `merge=union`. **No `*.sh` rule of any kind.** Re-run:
  `tests/unit/installer/test_install_scripts_terminal_hub.py::test_sh_parses_under_dash` FAILS on
  this checkout.

- **d. POSIX-only tests without `skipif` — CONFIRMED, but the `os.getuid` attribution is wrong.**
  Re-runs:
  - `tests/unit/installer` → `7 failed, 92 passed, 15 skipped, 3 errors`. The 3 errors are
    `TestLemonadePythonResolution::{test_resolves_direct_shebang_posix, test_resolves_env_shebang_posix, test_script_without_shebang_returns_none}`.
  - `tests/unit/test_lemonade_macos_install.py` + `test_lemonade_launcher.py` →
    `7 failed, 29 passed, 12 errors`.
  - `tests/unit/cli/test_cli_smoke.py` → `3 failed` (`test_gaia_binary_on_path[gaia|gaia-cli|gaia-mcp]`,
    `assert shutil.which(binary) is not None`; the file's only `skipif`, at
    `test_cli_smoke.py:200-203`, covers "package not installed", not "shim not on PATH") —
    CONFIRMED as stated.
  - **Correction:** `grep -rn getuid tests/` returns **nothing**. The
    `AttributeError: module 'os' has no attribute 'getuid'` is raised in a third-party plugin's
    **teardown**, not in GAIA test code: `pyfakefs/pytest_plugin.py:54 →
    fake_filesystem_unittest.py:1169 tearDown → pyfakefs/helpers.py:109 reset_ids →
    set_uid(os.getuid())`. The three tests themselves *pass* ("4 passed, 3 errors"), and one of them
    (`test_script_without_shebang_returns_none`) is not a `_posix` test at all. A `skipif` there
    would paper over a **pyfakefs-on-Windows incompatibility** (bump/pin pyfakefs) rather than fix a
    POSIX assumption.

- **e. `test_cli_refusal_exit_codes.py` HOME-vs-USERPROFILE — CONFIRMED.**
  `tests/unit/test_cli_refusal_exit_codes.py:26-31` builds
  `env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home()), "PYTHONPATH": ..., "LEMONADE_BASE_URL": ...}`
  — no `USERPROFILE`, no `SYSTEMROOT`, and a POSIX-only `PATH`. Windows `ntpath.expanduser` never
  consults `HOME`. Re-run: **`4 failed in 1.80s`** (every test in the file), each with
  `RuntimeError: Could not determine home directory.` raised inside the subprocess.
  (Path note: the file is `tests/unit/test_cli_refusal_exit_codes.py`, not under `tests/unit/cli/`.)

- **f. `import gaia` crashes when the home dir cannot be resolved (`logger.py:50`) — CONFIRMED with a
  clean probe; this is the strongest claim in I73 and it is a *product* bug, not a test bug.**
  `src/gaia/logger.py:50` is `log_file = Path.home() / ".gaia" / "gaia.log"`, executed from
  `logger.py:281 log_manager = GaiaLogger()` at **module import**. The
  `try/except (PermissionError, OSError)` immediately below guards only the `mkdir`, not
  `Path.home()` itself, which raises `RuntimeError` — an uncaught type. Probe run from `%TEMP%` with
  an env stripped of `USERPROFILE`/`HOME`/`HOMEDRIVE`/`HOMEPATH`:

      python -c "import gaia"   ->  rc=1
        File ".../src/gaia/logger.py", line 281, in <module>   log_manager = GaiaLogger()
        File ".../src/gaia/logger.py", line 50,  in __init__   log_file = Path.home() / ".gaia" / "gaia.log"
        RuntimeError: Could not determine home directory.

  Trigger position: no attacker needed — any Windows process launched with a curated environment
  (scheduled task, service wrapper, CI `env:` block, another tool's `subprocess.run(env=...)`)
  cannot `import gaia` at all. On POSIX `expanduser` falls back to `pwd`, so this is
  Windows-specific. `test_cli_refusal_exit_codes.py` is the in-repo proof that the shape occurs in
  practice.

- **g. `test_memory_discovery.py` reads the developer's real Credential Manager — CONFIRMED, and it
  is a privacy leak, not only a flake.** Re-run: `1 failed, 115 passed, 4 skipped`; the assertion
  message printed into the pytest output is:

      Left contains one more item: {'content': 'Email account: kalin.ovtcharov@gmail.com',
        'category': 'fact', 'context': 'unclassified', 'entity': 'service:gmail', ...}

  A test that stubs one scanner and lets the rest hit the host writes the developer's real email
  address into whatever log the run produces (a CI artifact, a pasted traceback, a `.review` file).

- **Edit to REPORT.md (line 250):** three changes.
  1. Replace "POSIX-only tests without `skipif` (`os.getuid`, `/`-separator asserts)" with
     "POSIX-only tests without `skipif` (`/`-separator asserts; plus `pyfakefs`'s Windows-incompatible
     teardown, `helpers.py:109 set_uid(os.getuid())`, which errors three otherwise-passing cases)".
  2. Split the `import gaia` clause out of its parenthetical into its own sentence — it is a product
     defect and currently reads as a footnote to a test bug:
     "**`import gaia` dies at import when the home dir cannot be resolved** (`logger.py:50`
     `Path.home()` reached from module-scope `logger.py:281`; the `except (PermissionError, OSError)`
     below does not catch `RuntimeError`) — a Windows service, scheduled task, or any
     `subprocess.run(env=...)` with a curated environment cannot import the package at all."
  3. Append to the `test_memory_discovery.py` clause: "— the real address is written into the pytest
     failure output, so any shared log or CI artifact carries it."

- **Severity: keep 🟡 for the bundle, but the `import gaia` sub-claim deserves 🔴 if split out.**
  As written I73 is a grab-bag of test-hygiene issues (correctly 🟡). The `logger.py:50` crash is a
  product-code import failure with a 100% repro and no workaround short of setting an env var —
  that is REVIEW.md's "fires in normal use". Recommend splitting it into its own 🔴 finding
  (proposed **I73a**) and leaving the remaining test-hygiene items as I73 🟡.

- **Dup:** sub-claims (a)…(e) surface as counts inside I72's "unit suite unrunnable" framing; keep
  them separate, but make clear they are *additive* to I72's tallies, not part of them (the
  `tests/unit/eval` 37 F and `tests/unit/installer` 7 F / 3 E are **not** socket-guard failures).
  No I/C duplicate found for the `logger.py` or `load_dotenv` halves.

- **Tracked check:** none found.

---

### I74 CONFIRMED

The most claim-dense finding in the report and — surprisingly — the most accurate. Every number
I could re-derive matched.

- **Sub-claims:**
  - a. **`tests/test_lemonade_client.py` mock classes are never run by CI (`-k Integration` only) —
    CONFIRMED.** The only two workflow invocations are
    `test_gaia_cli_linux.yml:398` (`-k "Integration and not hybrid"`) and
    `test_gaia_cli_windows.yml:312` (`-k "Integration"`). Collection: `75/91 tests collected
    (16 deselected)` under `-k "not Integration"` — so 75 of 91 tests in the file are unreachable
    from CI.
  - b. **"6 of 75 fail today" — CONFIRMED exactly.** Re-run of `-k "not Integration"`:
    `6 failed, 69 passed, 16 deselected in 67.24s`. The six are
    `TestLemonadeClientMock::{test_chat_completions, test_chat_completions_nonstream_includes_auth_header,
    test_get_required_models_for_code, test_streaming_chat_completions, test_streaming_text_completions,
    test_validate_context_size_insufficient}`.
  - c. **"several 'mock' tests open real sockets to `localhost:13305`" — CONFIRMED.**
    `tests/test_lemonade_client.py:44-45` sets `PORT = int(os.environ.get("LEMONADE_PORT", 13305))`;
    run output shows `Created test client with model=Gemma-4-E4B-it-GGUF, host=localhost, port=13305`
    and the 75 "mock" tests take **67 s** — time spent on refused TCP connects, not on mocks.
  - d. **`tests/test_sdk.py` asserts an SDK surface that no longer exists (17 F) — CONFIRMED
    exactly.** Re-run: `17 failed, 66 passed, 1 skipped, 2 xfailed in 77.88s`. Failures span
    `TestAPIComponents::test_create_app_exists` (ImportError), `TestMCPComponents`,
    `TestRAGSDK`, `TestKokoroTTS`, `TestAudioClient`, `TestSDKDocumentation`, etc.
    `grep -rn "test_sdk.py" .github/workflows/` → **no matches**; the file is run by nothing.
  - e. **"one 'unit' test really boots an agent against Lemonade (16 s)" — CONFIRMED.**
    `TestAgentIntegration::test_agent_with_mocked_llm` is among the 17, and the captured stderr is
    `Error: Lemonade server is not running or not accessible … http://localhost:13305/api/v1/health`
    with `WARNING gaia.llm.lemonade_manager:lemonade_manager.py:693`.
  - f. **"43 test files (≈780 test functions) are run by no workflow at all" — CONFIRMED
    (independently re-derived: 40).** I enumerated every `pytest <path>` in `.github/workflows/*.yml`
    (12 named `tests/integration/` selections, 1 `tests/mcp/` file, 7 top-level `tests/*.py`,
    `tests/unit/` as a whole) and then, generously, counted a file as "run" if its basename appears
    *anywhere* in any workflow text. Result: **40 files never referenced**, carrying **635 raw
    `def test_`** (parametrization pushes the collected count higher, so "≈780 collected" is
    plausible). No workflow runs `tests/integration/`, `tests/mcp/`, `tests/stress/`, or
    `tests/installer/` as a directory. The report's 43 and my 40 differ only in how the ambiguous
    cases (e.g. `tests/test_agent_sdk.py`, whose basename collides with the workflow filename
    `test_agent_sdk.yml`) are counted — the claim stands.
  - g. **`tests/integration/test_files_router.py` (76) and `test_chat_ui_integration.py` (84) —
    CONFIRMED exactly** (`76 tests collected`, `84 tests collected`), and neither name appears in any
    workflow.
  - h. **"the 155 memory integration tests" — CONFIRMED exactly.**
    `test_memory_integration.py` + `test_memory_api_integration.py` + `test_memory_eval.py` →
    `155 tests collected`; none referenced by a workflow.
  - i. **"five `test_governed_*.py`" — CONFIRMED.** `test_governed_{agent_workflow, canonical_name,
    real_agent, review_flow, workflow_binding}.py`, none referenced. `test_scheduler_e2e.py` also
    unreferenced — CONFIRMED.
  - j. **"4 of 7 classes in `tests/test_chat_agent.py` are excluded by explicit `::Class`
    selection" — CONFIRMED exactly.** The file defines 7 `Test*` classes (`TestChatAgent`,
    `TestChatAgentEval`, `TestChatAgentTools`, `TestChatAgentSummarization`, `TestChatAgentSessions`,
    `TestChatAgentPathValidation`, `TestChatAgentCodeSupport`); `test_chat_agent.yml:115,139,163`
    selects exactly three. The other `class UserAuth/DataManager/APIClient` hits in that file are
    inside code-fixture strings, not test classes.
  - k. **`src/gaia/audio/tests/*` outside `testpaths` — CONFIRMED.** `pyproject.toml:51`
    `testpaths = ["tests"]`; the three files live under `src/`.
  - l. **`tests/test_sd_model_sweep.py` always ERRORs under `pytest tests/` — CONFIRMED.**
    It *collects* fine (one item, `test_model_combination`), then errors:
    `fixture 'client' not found` — the function signature is
    `def test_model_combination(client, model_id, size, prompt, output_dir)` at
    `test_sd_model_sweep.py:45`, i.e. a helper the module's `__main__` block calls positionally, which
    pytest sees as five undefined fixtures. Run: `1 error in 1.15s`.
  - m. **`tests/test_vlm_integration.py` returns bools and cannot fail — CONFIRMED.** Five collected
    `test_*` functions (`test_vlm_availability`, `test_vlm_loading`,
    `test_image_extraction_from_pdf`, `test_vlm_extraction_on_real_image`,
    `test_vlm_batch_extraction`), each `return True` / `return False` instead of asserting
    (`:37,:43,:60,:63,:78,:128,:178,:181,:188,:203,:275,:278`). Under pytest a `return False` is a
    pass (plus a `PytestReturnNotNoneWarning`).

- **Edit to REPORT.md (line 251):** two small precision fixes, otherwise leave it.
  1. "**6 of 75 fail today**" — add the denominator's provenance so a reader can reproduce it:
     "6 of the 75 non-`Integration` tests (of 91 in the file) fail today".
  2. "43 test files (≈780 test functions)" — soften to "≈40–43 test files (≈780 collected tests)",
     since the count depends on whether a bare basename mention in a workflow counts as "run".
  Everything else in I74 verified verbatim.

- **Severity: keep 🟡.** Nothing here fires for a *user*; it is a coverage/process defect. But it is
  the strongest 🟡 in the section: `tests/integration/test_files_router.py` and
  `test_chat_ui_integration.py` need only `TestClient` + an in-memory DB, so 160 tests that would run
  in seconds are simply not wired up.

- **Dup:** overlaps `.review/_05tmp/part3.md` (the CI reviewer's original 43-file list) and I79–I81's
  CI theme; within REPORT.md, no duplicate ID. Sub-claim (a) also overlaps the "mock validity"
  discussion in source 06 Task 3 (the `test_pull_model` #1655 shape) — worth one clause in I74, since
  a never-run file that also never asserts request payloads is the exact #1655 failure mode CLAUDE.md
  calls out.

- **Tracked check:** #875 ("Deep audit: CI/CD and test coverage gaps + Playwright/Strix Halo E2E
  plan") is the closest open umbrella issue; nothing tracks the specific never-run file list.

---

### I75 CONFIRMED-ADJUST

Real, but one sub-clause is factually wrong and one number is off by one.

- **Sub-claims:**
  - a. **"31 modules import `fastapi` at module level" — CONFIRMED (30 on my count).** Scanning
    `tests/**/*.py` for an unindented `from fastapi… import` / `from starlette… import` /
    `import fastapi` **without** a `pytest.importorskip("fastapi")` in the same file:
    **30 unguarded, 10 guarded**. (13 files contain an `importorskip("fastapi")` at all, matching the
    source doc; 3 of those have no module-level import to guard.) ±1 — not worth an edit.
  - b. **"every hub package test dir has no `importorskip`" — REFUTED as written.** Actual state of
    the six `hub/agents/*/python/tests` trees:

    | tree | files | any `importorskip` | `conftest.py` |
    |---|---|---|---|
    | `email` | 130 | **yes** | **yes** |
    | `gaia` | 14 | **yes** | no |
    | `chat` | 3 | no | no |
    | `connectors-demo` | 1 | no | no |
    | `hello-world` | 1 | no | no |
    | `word-count` | 1 | no | no |

    So it is **4 of 6**, and the two largest trees are the guarded ones. The source doc got this
    right ("`hub/agents/email/python/tests` is only partially guarded"); the report over-generalised.
  - c. **"→ 176 collection errors on a core-only venv" — CONFIRMED BY SOURCE, NOT RE-RUN.** This
    venv has the `[ui]` extra installed and uninstalling `fastapi` would break the checkout for the
    other verifiers sharing it, so I did not reproduce it. Source 06's run #1 recorded
    `176 × No module named 'fastapi'` directly, and the structural precondition (30 unguarded
    module-level imports) is confirmed above, so the number is credible.
  - d. **"`allow_network` marker is registered only in `tests/unit/conftest.py`, so unusable
    elsewhere under `--strict-markers`" — CONFIRMED.** `pyproject.toml:42-50` registers exactly
    seven markers — `slow`, `integration`, `gmail_live`, `real_model`, `real_slm_build`,
    `distributed_seams`, `network` — and **not** `allow_network`; `:55 addopts = "--tb=short
    --strict-markers"`. The only registration is `tests/unit/conftest.py:48-51`
    (`config.addinivalue_line("markers", "allow_network: …")`), which is not loaded for
    `tests/integration`, `tests/mcp`, or the hub trees.
  - e. **"`pytest-timeout` is used by CI but in no extra" — CONFIRMED, and I hit it myself.**
    `grep -rn "pytest-timeout" setup.py pyproject.toml` → **no matches**; 5 workflow files install it
    ad hoc. My very first verification command in this session failed with
    `pytest: error: unrecognized arguments: --timeout=120` — the exact experience of a contributor
    copying a CI command.

- **Edit to REPORT.md (line 252):** replace
  "every hub package test dir has no `importorskip`" with
  "four of the six hub package test dirs (`chat`, `connectors-demo`, `hello-world`, `word-count`)
  have neither a `conftest.py` nor an `importorskip`, so they collection-error instead of skipping
  when the package is not installed".

- **Severity: keep 🟢.** Purely a contributor-ergonomics defect: it makes environmental failures
  indistinguishable from real ones, but no CI lane and no user is affected. Sub-claim (d)
  (`allow_network` unusable outside `tests/unit`) is the one with teeth — it is the marker the *fix
  for I72* would need to reach for, so it should stay visible.

- **Dup:** (c) is the same event stream as I72's error storm but a different root cause (missing
  extra, not the socket guard) — the report already keeps them separate, correctly.

- **Tracked check:** #2927 ("test(ci): network guard skips on no-network but errors on a flaky
  remote") is adjacent (same conftest guard family) but not this. Nothing tracks the
  `pytest-timeout` extra or the hub `importorskip` gap.

---

### I76 CONFIRMED-ADJUST

- **Evidence — read + proved by probe.** Line numbers are right:
  `src/gaia/eval/runner.py:40-43` is
  `if sys.platform == "win32": fcntl = None  # type: ignore[assignment] / else: import fcntl`,
  and `:94` is `def _acquire_eval_lock():` with the no-op at `:107-109`
  (`if fcntl is None: yield; return`). `_LOCK_FILE` is `:77`, `_LOCK_ENV_BYPASS` `:78`.

  Probe on this Windows box (imported the real module, entered the context manager twice, nested):

      platform: win32
      runner.fcntl is None: True
      stderr captured: ''
      stdout captured: ''
      LOCK_FILE: C:\Users\...\Temp\gaia-eval-agent.lock   exists: False

  Two nested acquisitions succeed, nothing is printed on either stream, and the lock file is never
  even created. **Silent no-op: proved.** Contrast with the read-only-`/tmp` branch at `:114-122`,
  which does print `[WARN] Could not create eval lock at … Skipping concurrency guard.` — so the
  codebase already has the warning this path omits.

- **Sub-claims:**
  - a. "`fcntl` is `None` → bare `yield`, no warning" — **CONFIRMED** (probe above).
  - b. "the target platform is Windows" — **CONFIRMED** (product targets Ryzen AI / Windows).
  - c. "CLAUDE.md's serial rule relies on it" — **OVERSTATED.** CLAUDE.md's
    "Run agent evals SERIALLY, never in parallel" section never mentions a lock; its enforcement
    mechanism is a human instruction plus `ps aux | grep "gaia eval" | … | wc -l` (a POSIX command,
    itself unusable on Windows). So the rule does not *rely* on the guard — it is unenforced on
    Windows both ways, which is arguably a stronger statement.
  - d. **Omitted context the report should acknowledge:** the degradation is **deliberate and
    commented**, `runner.py:37-39`:
    "*fcntl is POSIX-only — on Windows the eval lock degrades to a no-op (the Lemonade race the lock
    guards against doesn't happen on a contributor's Windows box, where Lemonade Server isn't
    typically running concurrent evals).*"
    That rationale is the part worth attacking: Windows **is** where Lemonade + Ryzen AI actually
    runs, so the assumption is backwards relative to the product's own targeting. Saying so is a
    better finding than "silent no-op" alone — and it also means this is a *design decision to
    revisit*, not an oversight, which changes how a maintainer will read it.

- **Edit to REPORT.md (line 255):** replace the finding text with:
  > **I76. The eval "one run at a time" guard is a deliberate no-op on Windows, and it warns
  > nobody** (`eval/runner.py:40-43, 94-110` — `fcntl` is `None` → bare `yield`; the sibling
  > read-only-`/tmp` branch at `:114-122` *does* print `[WARN] … Skipping concurrency guard`).
  > The in-code rationale (`:37-39`) assumes Windows boxes don't run concurrent evals — but Windows
  > is the platform where Lemonade/Ryzen AI actually runs, and CLAUDE.md's serial rule has no
  > enforcement there either (its check is a POSIX `ps aux | grep`).

- **Severity: keep 🟡.** It needs two eval runs racing to bite, and `GAIA_EVAL_NO_LOCK=1` shows the
  authors thought about the bypass. Not 🔴 (no normal-use trigger), not 🟢 (when it does bite, the
  output is a *scorecard with bogus failures and no indication anything went wrong* — a wrong answer
  presented as a right one, which is exactly what CLAUDE.md's fail-loudly rule exists to prevent).

- **Fix check:** the report's implied fix (msvcrt/PID file) is right, but the *cheap* half should be
  called out separately: printing the same `[WARN]` the OSError branch already prints is a one-line
  change that makes the degradation visible today, independent of implementing a Windows lock.

- **Dup:** none.

- **Tracked check:** none found (`gh issue list --search "eval baseline"` surfaced #3169, #1511,
  #2960, #2983 — all eval-process issues, none about the lock).

---

### I77 CONFIRMED

- **Evidence — read only; the fixer was NOT executed** (per the brief).
  `src/gaia/eval/runner.py:1329` `def run_fix_iteration(scorecard, run_dir, iteration)`; at
  **`:1368-1369`**:

      claude_cmd = shutil.which("claude") or "claude"
      cmd = [claude_cmd, "-p", prompt, "--dangerously-skip-permissions"]

  and `:1372-1381` `subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
  errors="replace", timeout=600, cwd=str(REPO_ROOT), check=False)`.
  `REPO_ROOT` is `Path(__file__).parent.parent.parent.parent` (`:45`) — the developer's checkout.
  `grep -n "porcelain|git status|git stash|git rev-parse|dirty" src/gaia/eval/runner.py` → **zero
  matches**: there is no clean-tree check, no branch check, and no diff summary anywhere in the file.

- **Sub-claims:**
  - a. "`claude -p … --dangerously-skip-permissions` on the working tree" — **CONFIRMED** (`:1369`).
  - b. "no clean-tree/branch guard" — **CONFIRMED** (grep above).
  - c. "up to 3 iterations" — **CONFIRMED**: `cli.py:2082-2086`
    `--max-fix-iterations, type=int, default=3`; loop at `runner.py:1937` `while iteration <
    max_fix_iterations`.
  - d. "a fixer timeout is swallowed into the fix log and the loop re-evaluates a half-applied tree"
    — **CONFIRMED, and it is worse than 'swallowed'.** `:1404-1410` catches
    `subprocess.TimeoutExpired` and *returns* `{"iteration": …, "error": "Fixer timed out after
    600s", "fixes": []}`; the caller at `:1960-1961` does
    `fix_result = run_fix_iteration(...)` / `fix_history.append(fix_result)` and then falls straight
    through into Phase C (`:1963+`, re-running every previously-failed scenario) **without ever
    inspecting `fix_result` for an `error` key**. A 600 s timeout kills Claude Code mid-edit and the
    next iteration scores the half-applied tree; the return value is never checked at all.
  - e. **"the only 'do NOT commit' rule is prose inside the prompt" — CONFIRMED.**
    `eval/prompts/fixer.md` exists and its RULES are: fix architecture first, then prompts,
    "3. Make minimal, targeted changes -- do NOT rewrite entire files", "4. Do NOT commit changes --
    leave for human review". Nothing constrains *which files* may be touched, and it is a prompt, not
    an enforcement — a model that ignores rule 4 is only stopped by the human noticing.
    (Source 06 listed reading `fixer.md` as an unverified hypothesis; it is now read, and it does
    **not** mitigate the finding.)

- **Edit to REPORT.md (line 256):** tighten the timeout clause, which currently under-states it:
  > a fixer timeout is returned as `{"error": "Fixer timed out after 600s"}` and the caller
  > (`runner.py:1960`) never inspects the result, so the loop re-evaluates a tree Claude Code was
  > killed halfway through editing.
  Also add: "the prompt's `do NOT commit` (`eval/prompts/fixer.md`) is the only restraint, and it
  names no file allowlist."

- **Severity: change 🟡 → 🔴, on the data-loss argument.** REVIEW.md's 🔴 bar is "a bug that fires in
  normal use". `--fix` is a documented, first-class flag (`cli.py:1991` even advertises
  `gaia eval agent --fix --max-fix-iterations 5 --target-pass-rate 0.95` in the epilog), and the
  normal way a developer reaches for it is *while working on the thing that is failing* — i.e. with
  a dirty tree. At that moment the tool grants an unattended agent write access to the whole
  checkout with permissions disabled, three times over, with a kill-mid-edit path and no diff
  summary. There is no undo: uncommitted work overwritten by the fixer is gone, and the developer
  has no record of what changed. That is unrecoverable user-data loss under documented normal use.
  The counter-argument — "the user opted in by typing `--dangerously`-adjacent flags" — does not
  hold, because the *user* never types `--dangerously-skip-permissions`; the runner adds it silently
  behind a flag named `--fix`.

- **Fix check:** the report's fix (refuse on a dirty tree / run in a `git worktree`) is right and
  cheap. The **minimum** viable version is three lines and should be stated separately, because it
  is uncontroversial and shippable today: (1) `git status --porcelain` non-empty → refuse unless
  `--fix-allow-dirty`; (2) `break` the loop when `fix_result.get("error")`; (3) print
  `git diff --stat` after each iteration. A `git worktree` sandbox is the better long-term answer
  but is a bigger change.

- **Dup:** none in REPORT.md. Adjacent theme to the auto-approve hardening the codebase already does
  elsewhere (`gaia/__init__.py` refuses to honour `GAIA_AUTO_APPROVE_TOOLS` from a `.env` — the repo
  clearly cares about exactly this class of risk, which makes the `--fix` path an inconsistency
  worth naming in the finding).

- **Tracked check:** `gh issue list -R amd/gaia --search "dangerously-skip-permissions"` → none.

---

### I78 CONFIRMED-ADJUST

Real and important, but it contains one **false statement**, one **arithmetic error**, and it
**misses a broken CI invocation** that is the strongest evidence for its own thesis.

- **Sub-claims:**
  - a. "`--save-baseline` and single-path `--compare` use `eval/results/baseline.json`
    (`cli.py:3843-3859, 3954-3963`)" — **CONFIRMED**, line numbers correct. Single-path `--compare`:
    `baseline_path = RESULTS_DIR / "baseline.json"`, error+`sys.exit(1)` if absent.
    `--save-baseline`: `baseline_path.write_text(json.dumps(last_scorecard, …))`.
  - b. **"git-ignored" — REFUTED.** `eval/results/.gitignore` reads:

        # Ignore individual eval run directories (runtime artifacts)
        eval-*/
        rerun/

        # Keep baseline reference
        !baseline.json

    and `git ls-files` lists `eval/results/baseline.json` as **tracked**. Its current content is a
    3-scenario `gaia-lite` run judged by `claude-sonnet-4-6`. So `--save-baseline` doesn't write to
    a throwaway — it **overwrites a committed file**, which is *worse* than the report says: a
    developer following the docs silently dirties a tracked artifact.
  - c. "one file for all categories, overwritten by the next `--save-baseline`" — **CONFIRMED**
    (path is a constant; no category in the name).
  - d. **"nothing reads/writes `tests/fixtures/eval_baselines/`" — REFUTED for CI, CONFIRMED for
    `src/`.** `grep -rn eval_baselines src/` → only `eval/sidecar_harness.py:302-328`, a different
    feature (per-agent `<package>/eval_baselines/query_sequences/`). But **two workflows do reference
    the fixture tree**: `test_eval_agent_gemma_consolidation.yml:192`
    (`BASELINE_DIR: tests/fixtures/eval_baselines/gemma-4-e4b-d71cd914`) and `test_eval_rag.yml:74`.
  - e. **MISSED, and it is the best evidence in the finding: `test_eval_rag.yml:74` is a broken
    invocation.** The step runs

        gaia eval agent --category rag_quality --compare tests/fixtures/eval_baselines/gemma-4-e4b-d71cd914/scorecard_rag_quality.json

    But `--compare` is handled at `cli.py:3840-3884`, **before** any eval runs (the comment at
    `:3839` literally says "diff two scorecard files, no eval run needed") and it `sys.exit()`s in
    every branch. With **one** path, the code treats `eval/results/baseline.json` as the *baseline*
    and the committed fixture as the *current* run — i.e. the arguments are backwards — and if
    `eval/results/baseline.json` is missing it prints `[ERROR] No saved baseline found` and exits 1.
    So the "Run eval rag_quality compare" job **never runs an eval at all**; it either fails
    outright or diffs a stale local file against the committed fixture with the roles inverted.
    `--category rag_quality` is parsed and ignored. This should be added to I78 (or split out) —
    it turns "the docs contradict the CLI" into "a CI gate that cannot do what its name says".
  - f. "the default judge is `claude-opus-5` while all seven committed baselines were judged by
    `claude-sonnet-4-6`" — **CONFIRMED exactly.** `eval/config.py:16 DEFAULT_CLAUDE_MODEL =
    "claude-opus-5"`; `grep -ho '"model": "…"' tests/fixtures/eval_baselines/*/scorecard_*.json |
    uniq -c` → `7 "model": "claude-sonnet-4-6"`. Baseline dirs: `gemma-4-e4b-95e4b372`,
    `gemma-4-e4b-d71cd914`, `qwen-3.5-35b-3b51ca92`; files: `scorecard_rag_quality.json` ×3,
    `scorecard_context_retention.json` ×2, `scorecard_tool_selection.json` ×2 = 7.
  - g. "so every documented `--compare` prints the judge-mismatch banner" — **CONFIRMED**;
    `_warn_on_judge_mismatch` (`runner.py:1416-1437`) fires whenever `config.model` differs.
    **Extra:** the banner's own remedy is wrong for the documented workflow — it says
    "Regenerate the baseline under the current judge (`gaia eval agent --save-baseline`)", which
    writes `eval/results/baseline.json`, not the fixture the developer was told to compare against.
    Worth one clause; it closes the loop on the contradiction.
  - h. **"9 of 12 categories have none" — CONFIRMED. "71 of 91 scenarios" — WRONG, it is 74.**
    Per-category YAML counts: `memory` 25, `real_world` 19, `mcp_reliability` 10, `web_system` 6,
    `tool_selection` 5, `context_retention` 4, `adversarial` 3, `error_recovery` 3, `personality` 3,
    `vision` 3, `captured` 2, `rag_quality` 7 (+1 stray top-level `safety_handbook_water.yaml`) = 91.
    Baselined categories cover 7+5+4 = 16 (17 counting the stray as rag_quality), so the
    un-baselineable set is **74** (25+19+10+6+3+3+3+3+2), not 71.
  - i. "`compare_scorecards` lets a scenario that *disappears* pass" — **CONFIRMED.**
    `runner.py:1492-1495`: `if sid in base_map and sid not in curr_map: only_in_baseline.append(sid);
    continue`; `only_in_baseline` is returned (`:1690`) but the CLI's exit-code check
    (`cli.py:3866-3876`) sums only `regressed + score_regressed + time_regressed`.
    A scenario that crashed before judging, or was renamed, is printed under
    `[-] ONLY IN BASELINE … removed or renamed` and exits 0.
  - j. "lists a slower-and-failed scenario as a time regression only" — **CONFIRMED.** The
    `elif` chain at `:1540-1556` puts `elif entry.get("time_regressed")` (`:1542`) **before**
    `elif b_pass and not c_pass: regressed.append(entry)` (`:1547-1548`), so PASS→FAIL is masked by a
    2× slowdown on the same scenario. Exit code is still 2, so nothing merges wrongly — the *report*
    under-states, not the gate. The report says this correctly.

- **Edit to REPORT.md (line 257):** four changes.
  1. Delete "(git-ignored, …)" and replace with "(**a tracked file** — `eval/results/.gitignore` has
     `!baseline.json` — so `--save-baseline` silently dirties committed state; one file for all
     categories".
  2. Replace "nothing reads/writes `tests/fixtures/eval_baselines/`" with "no code in `src/gaia`
     reads or writes `tests/fixtures/eval_baselines/`; only two workflows name it, and
     `test_eval_rag.yml:74` calls it wrongly — `gaia eval agent --category rag_quality --compare
     <fixture>` short-circuits at `cli.py:3840` before any eval runs, treats the committed fixture as
     the *current* scorecard and `eval/results/baseline.json` as the baseline, and exits 1 when that
     file is absent. The 'Run eval rag_quality compare' job therefore never evaluates anything."
  3. "**71 of 91 scenarios**" → "**74 of 91 scenarios**".
  4. Append after the judge-mismatch clause: "— and the banner's own remedy (`--save-baseline`)
     writes to the wrong path, so following it does not fix the mismatch."

- **Severity: change 🟡 → 🔴 for sub-claim (e) if it is split out; keep 🟡 for the rest.**
  As a documentation/plumbing contradiction, 🟡 is right — it wastes a developer's afternoon, it
  doesn't ship a bug to a user. But `test_eval_rag.yml`'s compare job is a **CI gate that cannot
  pass or fail on its stated criterion**, which is the same class as I79/I80 (a green job that
  proves nothing) — and those are already ranked 🔴/🟡 in §3.11. Recommend splitting it out as
  **I78a 🔴** and cross-referencing I79/I80, since it belongs to the "green job, no signal" family
  rather than to the docs-drift family.

- **Fix check:** the report's proposed fix (make `--save-baseline` write
  `tests/fixtures/eval_baselines/<model>-<sha>/scorecard_<category>.json`) is the right shape, but it
  should also (a) make single-path `--compare X` mean "compare *current run* against X", which is
  what every doc and both workflows already assume, and (b) delete or repoint
  `eval/results/baseline.json` so there is one baseline concept, not two.

- **Dup:** partially tracked. **#1511** "AH-X5 — Per-agent eval baselines committed to
  `tests/fixtures/eval_baselines`" is the closest open issue (the fixture-tree half); **#3169**
  "The documented compare command points at `eval/results/latest/`, a path the runner never creates"
  is the same *class* of defect on a neighbouring path and is already marked 🔴 in that issue —
  which is itself an argument for raising this one. **#2960** ("agent-eval gate fails preflight on
  every PR — judge key arrives empty, so no scorecard is trustworthy") is the CI-side sibling of
  sub-claim (e). The report should cite #1511 and #3169.

---

## Summary (I72–I78)

**Verdict counts:** CONFIRMED 3 (I73, I74, I77) · CONFIRMED-ADJUST 4 (I72, I75, I76, I78) ·
OVERSTATED 0 · REFUTED 0 · UNVERIFIABLE 0.
Sub-claim level: 44 verified, **3 refuted** (I72f, I75b, I78b/d), **2 numerically wrong**
(I78h "71"→74; I72d mislabelled location), 1 not re-run (I75c, verified by source + structural
precondition).

This section holds up better than any other part of the report — every re-runnable tally I checked
matched to the test (`33 F / 217 E`, `213 F / 42 E`, `124 E`, `53 F`, `17 F`, `6 of 75`, `37 F`,
`76`, `84`, `155`). The corrections below are precision, not substance — except the two missed
findings.

**Five most important corrections**

1. **I78: "git-ignored" is false and "71 of 91" is wrong.** `eval/results/baseline.json` is a
   **tracked** file (`eval/results/.gitignore` has `!baseline.json`), so `--save-baseline` dirties
   committed state rather than writing a throwaway. The un-baselineable scenario count is **74**,
   not 71.
2. **I78 misses its own best evidence:** `test_eval_rag.yml:74` runs
   `gaia eval agent --category rag_quality --compare <committed fixture>`, which short-circuits at
   `cli.py:3840` **before any eval runs**, inverts baseline/current, and exits 1 when
   `eval/results/baseline.json` is absent. That CI job cannot do what its name says.
3. **I72's "Ubuntu-only CI" is false** — `test_unit.yml:215` also has a **macos-latest** job running
   `pytest tests/unit/`. And Windows CI *does* exist (`test_eval.yml:47`, `test_mcp.yml`,
   `test_security.yml:144` on `windows-latest`; `test_gaia_cli_windows.yml` self-hosted) — it just
   never runs the `tests/unit/` tree. Also: the "53 F" bucket is `tests/unit/test_hub_router.py` +
   `test_email_sidecar_*.py`, **not** `hub/agents/email/python/tests` (which cannot load that
   conftest at all), and it double-counts 15 tests with the "157" daemon bucket.
4. **I73's `os.getuid` attribution is wrong** — the `AttributeError` comes from **pyfakefs's**
   Windows-incompatible teardown (`pyfakefs/helpers.py:109 set_uid(os.getuid())`), not from GAIA
   test code (`grep -rn getuid tests/` → nothing). The three affected tests *pass*; only teardown
   errors. Fix is pinning/bumping pyfakefs, not adding `skipif`.
5. **I75's "every hub package test dir has no `importorskip`" is false** — it is 4 of 6; `email`
   (130 files, has a `conftest.py`) and `gaia` are both guarded.

**Proposed severity changes**

| Finding | Now | Proposed | Argument |
|---|---|---|---|
| **I72** | 🟡 | **🔴** | Every Windows contributor on a clean `main` gets ~640 red tests, 100% repro, disabling the primary local gate on the product's primary platform. "Fires in normal use" is satisfied; the ~5-line fix makes the current ranking hard to justify. |
| **I73 (split)** | 🟡 | **🔴 for the `logger.py:50` half** | `import gaia` raises `RuntimeError: Could not determine home directory.` from module scope whenever `USERPROFILE`/`HOME`/`HOMEDRIVE` are absent — proved with a stripped-env probe. That is a *product* import failure (Windows service, scheduled task, any `subprocess.run(env=…)`), not test hygiene. Split as **I73a**. |
| **I77** | 🟡 | **🔴** | `--fix` is advertised in the CLI epilog and is reached for precisely when the tree is dirty; it then runs `claude -p --dangerously-skip-permissions` over the whole checkout, three times, with a kill-mid-edit path (`:1404`) whose error the caller never inspects (`:1960`). Uncommitted work overwritten this way is unrecoverable, and the user never types the dangerous flag — the runner adds it behind `--fix`. |
| **I78 (split)** | 🟡 | **🔴 for `test_eval_rag.yml:74`** | A CI gate that structurally cannot evaluate its stated criterion belongs with I79/I80's "green job, no signal" family. Split as **I78a**. |
| **I74, I75, I76** | 🟡 / 🟢 / 🟡 | **unchanged** | I74 is a process/coverage defect (no user trigger); I75 is contributor ergonomics; I76 needs two concurrent evals to bite and has a documented bypass. |

**Tracked issues worth citing in the report:** #1511 (per-agent baselines under
`tests/fixtures/eval_baselines`) and #3169 (the documented compare command points at a path the
runner never creates — the same class as I78, already ranked 🔴) for I78; #2960 (agent-eval CI gate
fails preflight) for I78a; #875 (CI/test-coverage deep audit) for I74; #2927 (network-guard conftest
family) as adjacent to I72/I75. Nothing tracks the socket guard, the `pytest-timeout` extra, the
`.gitattributes` `eol=lf` gap, `--dangerously-skip-permissions`, or the `logger.py:50` import crash.

---

## §3.11 CI and release (I79–I86)

# Adversarial verification — I79–I86 (§3.11 CI / release)

Checkout: `211f08c5` (detached). All line numbers below re-derived by re-opening
the files; all run counts re-derived with live `gh run list` on 2026-09-04.

### I79 CONFIRMED
- Evidence: `publish.yml:653/660/667` each `continue-on-error: true` under the three `Download Desktop Installer` steps (648/655/662), and `publish.yml:554-558` is `needs: [validate, post-publish-smoke, publish-npm, build-desktop-installers, build-tui]` with `if: !cancelled() && needs.post-publish-smoke.result == 'success' && needs.publish-npm.result == 'success' && needs.build-desktop-installers.result == 'success'` — `build-tui` is in `needs:` but absent from the `if:`, and `!cancelled()` keeps the job alive on a failed dependency; the release body (`:596-598`) still promises the `.exe`/`.dmg`/`.deb`/`.AppImage`.
- Edit: change the citation `(:612-618, 648-668)` → `(:554-558, 648-667)`. The `if:` block is at 554-558, not 612-618 (612-618 is inside the release-body heredoc).
- Severity: keep 🟡 — user-visible (a release promising a DMG that is not attached) but not a security or data-loss issue; matches REVIEW.md's "would this mislead a user? yes → 🟡".
- Dup: none.

### I80 CONFIRMED
- Evidence: `publish.yml:287-290` — `- name: Run backend tests` / `run: |` / `pip install -e ".[dev]" 2>/dev/null || pip install -e .` / `python -m pytest tests/unit/chat/ui/ -x --tb=short 2>/dev/null || echo "Backend tests skipped (dependencies not available)"`. The step is in job `build-npm` (`:214`); `build-pypi` (`:181`) has `needs: [validate, build-npm]` and `approve-publish` (`:345-350`) hard-requires `needs.build-npm.result == 'success'` — so the wheel build and the whole publish gate ride on a step that cannot fail. Doubly silent: the `||` swallows the exit code *and* `2>/dev/null` hides the collection error that would explain it.
- Edit: cited lines are exact; optionally add "(job `build-npm`, which `build-pypi` and `approve-publish` both depend on)" so the blast radius is visible.
- Severity: change 🟡 → 🔴. REVIEW.md tier 1 names "silent-fallback violations" explicitly, and CLAUDE.md's *No Silent Fallbacks — Fail Loudly* section prohibits exactly `... || echo`. This is the repo's own stated-🔴 category, in the single workflow that ships to PyPI and npm.
- Dup: none.

### I81 CONFIRMED-ADJUST
- Evidence: `gh api repos/pypa/gh-action-pypi-publish/branches/release%2Fv1` → `release/v1 -> dc37677b2e…` while `git/refs/tags/release/v1` → **404** — `release/v1` is a mutable **branch**, confirmed, used at `publish.yml:383` in job `publish-pypi` whose `permissions:` are `id-token: write` + `contents: read` (`:371-373`), i.e. PyPI trusted publishing. `sigstore/gh-action-sigstore-python@v3.5.0` (`:574`) and `softprops/action-gh-release@v3` (`:670`) both sit in `github-release` with `permissions: contents: write, id-token: write` (`:560-562`); `signpath/github-action-submit-signing-request@v2` (`build-installers.yml:536`) sits in `build` with `id-token: write` (`:133-135`); `softprops/action-gh-release@v3` (`build-installers.yml:1273`) sits in `publish-release-assets` with `contents: write` (`:1260-1261`). Full third-party inventory (`grep -rh 'uses:' .github/workflows/*.yml | sort -u`, minus `./` and `actions/`): SHA-pinned = **only** `anthropics/claude-code-action@a874e9e… # v1.0.210` and `dependabot/fetch-metadata@25dd0e3… # v3.1.0`; tag/branch-pinned = `pypa/gh-action-pypi-publish@release/v1`, `sigstore/gh-action-sigstore-python@v3.5.0`, `softprops/action-gh-release@v3`, `signpath/github-action-submit-signing-request@v2`, `codecov/codecov-action@v7`, `FedericoCarboni/setup-ffmpeg@v3`, `golangci/golangci-lint-action@v9`, `astral-sh/setup-uv@v7`, **`github/codeql-action/upload-sarif@v4`** (missed by the report: `claude-security-audit.yml:146,382`, `skill_audit.yml:444` — those jobs hold `security-events: write`).
- Edit: (a) add `github/codeql-action/upload-sarif@v4` to the list; (b) note `publish_agents.yml:143` is currently **unreachable** — its `publish` job carries `if: false && …` (`:123`, publishing paused per #1179), so it is latent, not live; (c) `astral-sh/setup-uv@v7` (`release_agent_email.yml:174`, `release_agent_gaia.yml:383,639`) is in the frozen-binary **build** job, not the OIDC `publish` job (`release_agent_email.yml:435-437`) — it poisons the artifact, one hop upstream of the token; (d) `FedericoCarboni/setup-ffmpeg@v3` and `codecov@v7` are not in the publish/sign path at all (`test_gaia_cli_windows.yml`, `test_unit.yml`) — keep them, but say "plus test-path actions".
- Severity: change 🟡 → 🔴. REVIEW.md tier 1 is "security issues" and 🔴 calibration is "security, breaking changes, …". A mutable **branch** ref executing inside a job that mints a PyPI trusted-publishing OIDC token means one upstream force-push = attacker-controlled code holding an upload credential for `amd-gaia` — the exact tj-actions/changed-files (Mar 2025) shape. The mitigating facts (repo does not control upstream; a manual `approve-publish` environment gate precedes `publish-pypi`) argue for *hardening*, not for downgrading: the approval gate approves the *build result*, not the action code that runs after it.
- Dup: none (distinct from the `claude.yml` prompt-injection 🔴 in `05-ci-release.md:33`, which the report tracks separately).

### I82 CONFIRMED-ADJUST
- Evidence (three sub-claims, one of them stale):
  - **`build_cpp_email.yml` — OVERSTATED/stale.** The always-fail assert is real (`:70-78`, `if [ ! -d "cpp/agents/email" ] … exit 1`; `ls cpp/agents` = `README.md bash health process security-demo vlm wifi` — no `email`), and `merge_group:`/`workflow_dispatch:` (`:40-41`) are indeed unfiltered. But the **cause of the 13 red runs is already fixed at this commit**: `git show c13598c8:.github/workflows/build_cpp_email.yml` shows `paths:` used to include `'.github/workflows/build_cpp_email.yml'` itself, and that line was removed in `793fb4d4` (2026-08-31, #3142). `gh run list -R amd/gaia -w build_cpp_email.yml -L 15` = 13 failure + 2 skipped, all `pull_request`/`push`/`workflow_dispatch`, **latest 2026-08-28 — before the fix**; zero `merge_group` runs. `gh run list -R amd/gaia -L 200 --json event` returns **no `merge_group` events at all** (`pull_request` 144, `push` 12, `pull_request_target` 12, …) — the merge queue is not in use, so the `merge_group:` trigger the report blames has never fired.
  - **`test_gaia_cli.yml` — CONFIRMED exactly.** `on:` is only `workflow_call:`/`workflow_dispatch:` (`:14-15`); `grep -rn test_gaia_cli.yml .github/` outside the file itself returns **nothing** (no caller). `gh run list -w test_gaia_cli.yml -L 15` = 12 `failure` + 3 `cancelled`, newest `2026-07-20T19:15:04Z push failure`.
  - **`release_components.yml` — CONFIRMED with one correction.** `on: push: tags: ['v*']` (`:29-31`) — same trigger as `publish.yml:26`. `gh run list -w release_components.yml -L 20` returns 12 runs: **9 failure, 3 cancelled, 0 success**. `deploy-worker` (`:214-248`) runs `npx wrangler deploy` against production Cloudflare on a `v*` tag under `environment: worker-deploy`, which the file's own comment (`:211-213`) states has **no required reviewers** — confirmed: production Worker redeploy happens with no approval, while `publish.yml`'s `approve-publish` gate is still pending.
- Edit: three changes. (1) Replace "wired to `merge_group` with no path filter — 13 of its last 15 runs red" with "**used to** self-trigger on its own path; that was removed in #3142 (2026-08-31), so the 13 red runs (newest 2026-08-28) are historical. What remains live is the unfiltered `merge_group:`/`workflow_dispatch:` triggers on a job that can only `exit 1` — latent, not currently firing (the repo has run no merge-queue jobs)." (2) Keep `test_gaia_cli.yml` as written. (3) Replace "downloads installers from a release that does not exist yet" with "blocks up to 60 minutes on a bounded, loudly-failing wait for `publish.yml`'s manually-gated release (`:519-546`)" — the wait exists and is the report's one unfair characterization; the real defect is the **unapproved production Worker redeploy** (`:214-248`), which the report already names and should lead with.
- Severity: keep 🟡 for the `test_gaia_cli`/`release_components` halves. Downgrade the `build_cpp_email` half to 🟢 (latent config smell, nothing currently red).
- Dup: none.

### I83 CONFIRMED-ADJUST — RE-CHECKED LIVE, STILL FAILING BUT COUNTS SHIFTED
- Evidence: `gh run view -R amd/gaia 33677587588` → `2026-09-02T20:09:21Z push devlab/dispatch-conversion failure — "Point every self-hosted workflow at the Ryzen Dev Lab pool"` — the cited run is real. **It is not the newest one, and the failure is still live:** the newest failure, run `33710669744` (2026-09-03T04:28Z), ends `anthropic.BadRequestError: Error code: 400 … 'Your credit balance is too low to access the Anthropic API…' (request_id req_011Cefv89daEvuXaQzDNcUQD)` → `drafting eval failed` → `##[error]Process completed with exit code 1`. The report's *mechanism* is confirmed and has NOT been fixed since it was written. `publish.yml:345-350` `approve-publish` still hard-requires `needs.email-eval.result == 'success'`. `gh issue view 1344` → **OPEN**: "eval: route remaining direct-Anthropic-API eval paths through Claude Code (subscription auth)".
- Edit: the run-conclusion breakdown is stale — re-derived `gh run list -w test_email_agent_eval.yml -L 15 --json conclusion` now gives **`{failure: 6, skipped: 9}`**, not `{skipped: 5, failure: 4, action_required: 6}`. Replace it, and cite the newest run `33710669744 (2026-09-03)` alongside `33677587588` so the claim survives the next window shift. Also worth adding: the same log shows the perf gate breaching with `enforce:false` (a *warning*), which is exactly the "eval measured a regression vs eval infra broke" conflation the fix section proposes to separate — good supporting detail.
- Severity: keep 🟡. It blocks releases loudly, does not ship anything wrong to users, and is tracked.
- Dup: none. Tracked: #1344 (open) covers the root cause; the blocked-gate state is still untracked.


### I84 CONFIRMED - severity should rise
- Evidence: all ten named workflows re-checked individually. Every one has a fork-reachable `pull_request:` trigger (`test_agent_sdk.yml:12`, `test_api.yml:16`, `test_embeddings.yml:13`, `test_examples.yml:13`, `test_gaia_cli_windows.yml:14`, `test_lemonade_server.yml:15`, `test_rag.yml:15`, `test_sd.yml:14`, `test_npu_embedder.yml:12`, `build_cpp.yml:17`); every one puts at least one job on a self-hosted Windows box (`test_agent_sdk.yml:46-49`, `test_api.yml:52`, `test_embeddings.yml:43`, `test_examples.yml:96`, `test_gaia_cli_windows.yml:49`, `test_lemonade_server.yml:55`, `test_rag.yml:81`, `test_sd.yml:44`, `build_cpp.yml:213` all resolve to `[self-hosted, Windows, stx|stx-test]`; `test_npu_embedder.yml:53` is `[self-hosted, Windows, devlab-dispatch, strix-halo]`); and `grep -c 'head.repo.full_name|head.repo.fork|github.event.pull_request.head.repo'` returns **0** in all ten - no same-repo gate anywhere. The only job-level `if:` is a draft/`ready_for_ci`-label gate, which is not a security control. `gh api repos/amd/gaia` -> `{"private":false,"visibility":"public"}`. Runner state is persistent, not ephemeral: `test_gaia_cli_windows.yml:91,160,287` re-activate `%GITHUB_WORKSPACE%\.venv`, and `:150` says the Lemonade step "reuses it if already healthy, and warms Gemma-4".
- Edit: (a) the report says "nine" but lists **ten** workflows - fix the count. (b) Name the runner labels (`stx`/`stx-test`, `devlab-dispatch`+`strix-halo`) so a reader can map findings to boxes. (c) Replace "only first-time-contributor approval protects" with the exact attacker position: **any GitHub user who has had one PR merged into `amd/gaia` runs with no approval at all; a first-time contributor needs one maintainer click, and that click covers every subsequent push to the same PR** - so the standard attack is a benign PR, one approval, then a force-push. The PR only has to touch one of the path filters. State the limit too, because the report omits it and a reviewer will ask: these are `pull_request`, not `pull_request_target`, so the fork job gets a read-only `GITHUB_TOKEN` and **no repository secrets**. The payload is therefore not secret exfiltration - it is arbitrary code execution on an AMD-lab Windows host plus **cross-run persistence**: the reused `.venv`, the resident Lemonade install and models, and the runner's own registration credential on disk all survive into later jobs on the same box.
- Severity: change from the report's implicit tier to **change 🟡 -> 🔴**. REVIEW.md tier 1 is "security issues" and its 🔴 calibration leads with "security". This is the configuration GitHub's own docs tell public repositories not to use, replicated across ten workflows with zero gating.
- Dup: none. Related to, but distinct from, the `claude.yml` `contents: write` prompt-injection 🔴 recorded at `.review/05-ci-release.md:33`.

### I85 CONFIRMED-ADJUST - runner half half-stale, Dependabot half fully confirmed
- Evidence (runner): `runner_heartbeat.yml:30-32` matrix is exactly `sjlab-stx-1`, `sjlab-stx-3`, and `monitor_selfhosted_runners.yml:24-35` parses that matrix as its "single committed source". `gh run list -R amd/gaia -w runner_heartbeat.yml -L 15` = `{cancelled: 10, success: 5}` - the count is right, **but every cancellation is 2026-05-24 through 2026-07-19 and the cause is already fixed**: commit `0835a250` (#2306) removed the offline `sjlab-stx-2`, as the file's own comment at `:26-29` explains. The five most recent runs (2026-07-26 .. 2026-08-30) are four success + one cancelled. Consequently **"weekly false Teams alarms" is REFUTED**: `gh run list -w monitor_selfhosted_runners.yml -L 8` is `success` every week since 2026-07-20, and the newest run's steps read `Fail if any runner missing: skipped` / `Send Teams alert via Power Automate: skipped`. What survives is coverage: the repo dispatches to **four** distinct self-hosted pools - `stx`/`stx-test` (9 workflows), `devlab-dispatch`+`strix-halo` (`test_npu_embedder.yml:53`, `test_agent_behavior_e2e.yml:34`), `strix-halo`+`lemonade-eval` (`test_eval_agent_gemma_consolidation.yml:275`), `lemonade-eval` (`test_eval_rag.yml:52`) - and the heartbeat names two boxes. `lemonade-eval` is where the `email-eval` release gate runs (`publish.yml:321`) and it emits no heartbeat, so its going offline raises nothing. One correction to the report's premise: `test_eval_agent_gemma_consolidation.yml:261-262` states the `stx` pool *spans* `sjlab-stx-1`/`sjlab-stx-3`, so the monitored boxes are live members of the pool, not orphans watching nothing.
- Evidence (Dependabot): `.github/dependabot.yml` declares exactly four ecosystems - `pip` `/` (`:13-14`), `npm` `/` (`:29-30`), `npm` `/src/gaia/apps/webui` (`:52-53`), `github-actions` `/` (`:72-73`). Actual manifests in the tree: `tui/go.mod` (**no `gomod` ecosystem at all**) and twelve `package.json` outside `node_modules`, of which `hub/agents/email/npm`, `hub/agents/gaia/npm` (the two *published* sidecars), `website/`, `workers/agent-hub/` (the production Cloudflare Worker), `src/gaia/electron/`, `docs/`, `src/vscode/gaia/`, `tests/electron/`, `hub/agents/email/node/` are all uncovered.
- Edit: (1) delete "10 of 15 heartbeats cancelled, weekly false Teams alarms" - the first is a stale window already fixed by #2306, the second is refuted by the monitor's own step conclusions. (2) Reframe the runner half as: "the heartbeat matrix names 2 boxes out of >=4 self-hosted pools; the `lemonade-eval` runner that hosts the `email-eval` release gate publishes no heartbeat, so its outage raises no alert." (3) Expand the Dependabot list to name `tui/go.mod` (no `gomod` entry), the two published npm sidecars, `website/`, and `workers/agent-hub/`.
- Severity: keep 🟡 for the Dependabot half. The runner half as *currently worded* is 🟢 (stale, already fixed); it earns 🟡 only under the reframed "release-gate runner is unmonitored" wording.
- Dup: none.

### I86 CONFIRMED
- Evidence: `build-installers.yml:334-356`, step `Download Lemonade MSI (Windows)`: builds `URL=".../lemonade-sdk/lemonade/releases/download/v${LEMONADE_VERSION}/lemonade-server-minimal.msi"`, `curl -fsSL --retry 3 --retry-delay 5`, and the **only** post-download check is `SIZE=$(wc -c < ...)` / `if [ "$SIZE" -lt 1048576 ]` - no digest, and the inline comment explicitly declines to pin ("rather than pinning to a specific upstream size that can change between minor releases"). The later `Verify Lemonade MSI embedded in installer (Windows)` step (`:444-463`) greps the `7z l` listing for the **filename only**, and downgrades to `WARNING: 7z could not list installer contents` - i.e. it can pass on a build it never inspected. By contrast uv **is** digest-verified three times in the same file: `:227,233` (linux `UV_SHA256` + `sha256sum -c -`), `:251,259-264` (windows, with an explicit fail-if-placeholder guard), `:291,298` (macOS `shasum -a 256 -c -`), plus `:277-281` which computes the extracted `uv.exe` digest and injects it into `backend-installer.cjs` as `BUNDLED_UV_SHA256`. The unverified MSI then flows into the NSIS installer that `signpath/github-action-submit-signing-request@v2` (`:529-545`, `signing-policy-slug: release-signing`) submits for AMD code signing. `lemonade-version-bump.yml:17-21` runs `cron: "23 17 * * 4"` and auto-opens a PR advancing `LEMONADE_VERSION` in `src/gaia/version.py`, so the pin moves on a schedule with no digest ever recorded anywhere.
- Edit: change the citation `:338-356` -> `:334-356` (the step header is at 334). Add one clause: the "verify embedded" step at `:444-463` is a *filename* check that warns-and-passes when 7z cannot read NSIS solid compression - that is what leaves the `<1MB` size guard as the entire defence.
- Severity: keep 🟡 as a standalone CI hardening gap, **but it must not disagree with I25**. An unverified third-party binary is embedded in an installer that AMD then code-signs, which turns an upstream asset swap into an AMD-signed payload; if I25 is 🔴, this is 🔴.
- Dup: yes - explicitly the CI twin of **I25** (source side). Keep both, but make the cross-reference bidirectional: I25 should point at `build-installers.yml:334-356` as the code path that actually fetches the MSI.

## Summary (I79-I86)

**Counts:** CONFIRMED 4 (I79, I80, I84, I86) - CONFIRMED-ADJUST 4 (I81, I82, I83, I85) - OVERSTATED 0 - REFUTED 0 - UNVERIFIABLE 0. No finding in this section is false; three carry stale run-history numbers and one blames the wrong trigger.

**Most important corrections**

1. **I82's `build_cpp_email` blame is on a trigger that has never fired.** The 13 red runs came from `paths:` including the workflow's own file, removed in `793fb4d4` (#3142, 2026-08-31); the newest red run is 2026-08-28, before the fix. `gh run list -R amd/gaia -L 200 --json event` returns **zero `merge_group` events repo-wide** - the merge queue the report says this is "wired to" is not in use. Rewrite as latent, not live.
2. **I85's runner half is stale and one of its claims is refuted.** The 10 cancellations all pre-date #2306's fix, and the monitor workflow's own step conclusions (`Send Teams alert: skipped`) refute "weekly false Teams alarms". The durable claim - the `lemonade-eval` box that hosts the `email-eval` release gate publishes no heartbeat - needs to become the headline.
3. **I83's run counts have already rotated** to `{failure: 6, skipped: 9}`. The mechanism, however, is confirmed *still live*: the newest run `33710669744` (2026-09-03) dies on the same `credit balance is too low` 400. Cite the newest run alongside `33677587588`, and note #1344 is still OPEN.
4. **I81's headline sub-claim is confirmed hard** - `gh api .../branches/release%2Fv1` resolves while `.../git/refs/tags/release/v1` 404s, so `pypa/gh-action-pypi-publish@release/v1` is a mutable **branch** running in a job with `id-token: write` for PyPI trusted publishing. The report also **misses `github/codeql-action/upload-sarif@v4`** (three call sites, in jobs holding `security-events: write`), and should note `publish_agents.yml:143` is currently unreachable (`if: false &&`, #1179).
5. **I84 undercounts (ten workflows, not nine) and understates the attacker position.** Zero of the ten carry a `head.repo` gate; the correct statement is that any contributor with one merged PR runs unapproved, and a first-timer needs a single maintainer click that then covers every later force-push. Worth stating the limit too - `pull_request` gives no secrets - so the finding lands as host compromise + cross-run persistence rather than credential theft.

**Proposed severity changes**

- **I80 🟡 -> 🔴.** `... || echo "Backend tests skipped"` in `publish.yml:290` is verbatim the "silent-fallback violation" REVIEW.md puts in tier 1 and CLAUDE.md's *No Silent Fallbacks* section prohibits by name - in the one workflow that ships to PyPI and npm, inside the job `build-pypi` and `approve-publish` both depend on.
- **I81 🟡 -> 🔴.** Mutable branch ref executing inside a PyPI trusted-publishing job = one upstream force-push away from an attacker holding an upload credential for `amd-gaia`. REVIEW.md 🔴 = "security".
- **I84 🟡 -> 🔴.** Unauthenticated-ish arbitrary code execution on persistent AMD-lab hardware, ten workflows, no gate.
- **I82 split:** keep 🟡 for `test_gaia_cli.yml` / `release_components.yml`; drop the `build_cpp_email` half to 🟢.
- **I85 split:** keep 🟡 for Dependabot coverage; the runner half is 🟢 as written, 🟡 only if reframed around the unmonitored `lemonade-eval` release-gate box.
- **I86:** keep 🟡 only if I25 is 🟡; the two are the same defect on two sides and must carry the same severity.
- **I79, I83:** keep 🟡, unchanged.

---

## §3.12 Documentation (I87–I93)

# Adversarial verification — I87–I93 (§3.12 Documentation)

Checkout: `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f`, detached HEAD 211f08c5.
Python: `.venv\Scripts\python.exe`. All repo access read-only; probes run from the checkout without writing to it.
Method: every named claim in each finding checked against **both** sides (doc + code), CLI claims re-run
with `--help` (side-effect free), import claims re-run in the venv.

---

### I87 CONFIRMED-ADJUST

Two independent claims, both real; one number is wrong.

**Claim A — v0.23.1 release notes headline `gaia install gaia` / `gaia list`, neither exists.** HOLDS.

| Sub-claim | Verdict | Evidence |
|---|---|---|
| Notes say `gaia install gaia` | holds | `docs/releases/v0.23.1.mdx:8` ("`gaia install gaia` answered HTTP 404"), `:11` ("`gaia install gaia` now downloads the agent"), `:19` ("Use the flagship (`gaia install gaia`)"), `:25` |
| Notes say `gaia list` | holds | `:8` "and `gaia list` hid it entirely" |
| Python `gaia` rejects both | holds | `python -m gaia.cli list --help` → `error: argument action: invalid choice: 'list'`; `gaia install --help` → `usage: cli.py install [-h] … [--lemonade] [--yes] [--silent]` — no positional, so `gaia install gaia` dies in `parser.parse_args()` (`cli.py:3146`) before any handler |
| No *other* `gaia` binary has them either | holds (report doesn't say this, but it's the strongest form) | Go TUI cobra `Use:` strings are only `run`/`status`/`chat`/`version` (`tui/internal/cli/{agents,chat,version}.go`); npm `@amd-gaia/gaia` — whose `bin` really is `gaia` (`package.json:36-38`) — dispatches only `run`/`fetch`/`serve`/`version` (`src/cli.ts:492-498`) |
| Working commands are `gaia hub install` / `gaia hub list` | holds | `cli.py:2691` `gaia hub install <agent_id>` |

**Claim B — `cli.mdx:2683-2900` documents `gaia tui …` commands the CLI rejects.** HOLDS, but **the count "11" is wrong**.

- `awk 'NR>=2683 && NR<=2900' docs/reference/cli.mdx | grep -c "gaia tui"` → **21** (whole file: **23**). The raw source `.review/07-docs.md` recorded `grep -c` → 11, which does not reproduce.
- `python -m gaia.cli tui --help` → `cli.py: error: argument action: invalid choice: 'tui'`. Reproduced verbatim.
- Contradiction is real and in-nav: `docs/guides/terminal-hub.mdx:28` — "`gaia-tui` is the Go terminal binary — **never `gaia`**, which is the Python CLI's"; `tui/Makefile:9-11` — "Built as gaia-tui, never as gaia".
- **Nuance the report omits:** cli.mdx isn't confused about *which* binary — its own Note (`:2689-2692`) explains the TUI drops a leading `tui` word (real: `tui/internal/cli/root.go:173`). The bug is that it spells the binary `gaia` inside a page titled "CLI Reference" whose Info box points at `src/gaia/cli.py`. The note also asserts "`gaia tui status` and `gaia status` are the same command" — `gaia status` is *also* not a Python-CLI subcommand.

**Tracked claim.** Accurate and understated: `gh issue view` confirms **#3219** ("gaia tui is not a subcommand of the CLI this page documents — every gaia tui example errors out on a pip-installed GAIA", OPEN) is a near-verbatim pre-existing report of Claim B; #2709 and #3087 are OPEN and related.

- **Edit:** change "documents 11 `gaia tui …` commands" → "documents **21** `gaia tui …` invocations". Add that #3219 already reports Claim B verbatim, so only Claim A is new. Optionally add "no `gaia`-named binary anywhere (Python CLI, Go TUI, npm `@amd-gaia/gaia`) accepts `install <id>` or `list`" — it closes the obvious "maybe they meant the TUI" objection.
- **Severity:** keep 🟡. Claim A is the sharper half — the one command a patch release exists to advertise does not run — but it fails loudly at argparse, nothing unsafe.
- **Dup:** Claim A is a verbatim duplicate of **C29** (`REPORT.md:140`, which already states "The v0.23.1 notes also tell users to run `gaia install gaia` / `gaia list` — neither exists"). The report cross-refs it, but the sentence is stated twice at two severities. Recommend I87 keep only Claim B + a pointer.

---

### I88 CONFIRMED-ADJUST

All live-state claims **re-verified today (2026-09-04); nothing has changed since the report**.

| Claim | Verdict | Evidence (re-run) |
|---|---|---|
| `@amd-gaia/agent-email` **0.6.0 is published** | holds | `npm view @amd-gaia/agent-email version` → `0.6.0` |
| README doc links pinned to `agent-pkg-email-v0.6.0` | holds, **count wrong** | `grep -o "blob/[A-Za-z0-9._-]*" hub/agents/email/npm/README.md \| uniq -c` → **9** links on 8 lines (`:14,153,166,171,176,177,178×2,179`), plus a 10th in `hub/agents/email/npm/CHANGELOG.md:5`. Report says "seven". |
| That tag **does not exist** | holds | `gh api repos/amd/gaia/git/ref/tags/agent-pkg-email-v0.6.0` → `{"message":"Not Found","status":"404"}`. Control: `…/agent-pkg-gaia-v0.1.1` → `200`, sha `20e08190`. So the method is sound and the 404 is not an auth artifact. |
| Newest email pre-release is v0.5.0 | holds | `gh release list -R amd/gaia --limit 60 \| grep -i email` → only `agent-pkg-email-v0.5.0` (2026-07-19) and `agent-pkg-email-v0.1.0` |
| Package metadata all says 0.6.0 | holds | `hub/agents/email/python/pyproject.toml:7`, `npm/package.json:3`; both CHANGELOGs carry `## [0.6.0] - 2026-08-12` (`npm/CHANGELOG.md:39`, `python/CHANGELOG.md:94`) |
| `@amd-gaia/gaia` CHANGELOG says 0.1.1 "unreleased" | holds | `hub/agents/gaia/npm/CHANGELOG.md:7` → `## [0.1.1] — unreleased` |
| …though it is on npm | holds | `npm view @amd-gaia/gaia version` → `0.1.1` |
| …tagged | holds | `agent-pkg-gaia-v0.1.1` → HTTP 200 |
| …and named in the release notes | holds | `docs/releases/v0.23.1.mdx:8` — "published on the Agent Hub since 0.1.1" |

Nothing here is stale or refuted. The only defect is the undercount.

- **Edit:** "all seven README doc links" → "**all nine** README doc links (plus one in `CHANGELOG.md`)". Optionally note the control check (`agent-pkg-gaia-v0.1.1` → 200) so a reader can see the 404 isn't a permissions artifact.
- **Severity:** keep 🟡 for the email half — the npm README is the only rendered doc integrators see and every deep link in it 404s. The `@amd-gaia/gaia` CHANGELOG half is 🟢 on its own (a wrong word in a changelog heading; misleads a user checking whether a fix is in their build, but nothing breaks). If the report splits these, mark the second 🟢.
- **Dup:** none. (Adjacent to C29's release-readiness theme but a different artifact — npm, not the GitHub release.)

---

### I89 CONFIRMED-ADJUST

Six sub-claims. **Five hold exactly; the headline one (#1) is a misreading and must be rewritten.** Two counts and one line range are wrong.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | CLAUDE.md "says GaiaAgent **not yet landed**" | **OVERSTATED / misread** | `CLAUDE.md:689` actually reads: "**GaiaAgent rename planned (#696)** — not yet landed; **the chat agent class is still `ChatAgent`**". Both halves are *literally true* at 211f08c5: the rename never happened and `class ChatAgent(` is still at `hub/agents/chat/python/gaia_agent_chat/agent.py:176`. `GaiaAgent` is a **new subclass**, not the rename (`hub/agents/gaia/python/gaia_agent/agent.py:211` → `class GaiaAgent(ChatAgent, SkillLibraryToolsMixin, CodeIndexToolsMixin)`). The real defect is different and *sharper*: **#696 was closed `NOT_PLANNED` on 2026-09-02** (`gh issue view 696 --json stateReason` → `NOT_PLANNED`), so the bullet advertises a planned rename that has been cancelled, and its phrasing reads as "GaiaAgent doesn't exist yet" — which the same file's own agent table (`:581`, "**GaiaAgent** \| The flagship") contradicts. |
| 2 | Cites `hub/agents/chat/python/gaia_agent_chat/tools/` — no such dir | holds | `CLAUDE.md:256`; `ls hub/agents/chat/python/gaia_agent_chat/` → `__init__.py agent.py app.py lite_agent.py profiles.py session.py tool_bundles.py` — no `tools/` |
| 3 | Project tree omits `daemon/hub/schedule/sidecar/skills` | holds, **line range wrong** | Tree is at **`CLAUDE.md:483-540`**, not `:455-520` (`:455` is the Testing code block). It omits `daemon/ hub/ schedule/ sidecar/ skills/` **and `factory/`**, plus `cli_agent.py config.py device.py security.py logger.py perf_analysis.py util.py` |
| 4 | CLI list omits `mcp tui`, `config show`, ten `gaia agent` subcommands, `eval {benchmark,sessions,code}`, `lemonade embedded` | holds, **"ten" → eleven** | Probed: `gaia mcp` → `{start,status,stop,test,agent,serve,**tui**,list,tools,test-client}` vs `CLAUDE.md:640`; `gaia config` → `{**show**,get,set}` vs `:650`; `gaia agent` → `{init,version,test,pack,publish,configure,health,status,login,export,import,install,list}` (13) vs `:660` `{export\|import}` → **11** missing, not 10; `gaia eval` → `{agent,benchmark,sessions,code}` vs `:662-665` (only `agent`); `gaia lemonade` → `{embedded}`, zero mentions in CLAUDE.md |
| 5 | "Guides one per feature" lists 9 of 21 | holds | `CLAUDE.md:675` names 9; `ls docs/guides/` → 22 entries, one (`mcp`) is a directory ⇒ **21 `.mdx`** |
| 6 | Points at `docs/reference/eval.mdx`, deprecated-in-v0.18 banner, still in nav, 25 dead `gaia eval -d` examples | holds, all four parts | `CLAUDE.md:662` cites it; `docs/reference/eval.mdx:6-7` `<Warning> **Deprecated:** … removed in v0.18.0`; `docs/docs.json:340` `"reference/eval"`; `grep -c "gaia eval -d"` → **25**; `gaia eval --help` → required positional `{agent,benchmark,sessions,code}`, so every one of them errors |

- **Edit:**
  1. Replace "says GaiaAgent 'not yet landed' (it is `hub/agents/gaia/python/gaia_agent/agent.py:211`…)" with: *"advertises a `ChatAgent`→`GaiaAgent` rename as planned via #696, which was closed `NOT_PLANNED` on 2026-09-02, while a distinct flagship `GaiaAgent` subclass has shipped at `hub/agents/gaia/python/gaia_agent/agent.py:211` — the same file's agent table already calls it the flagship."* As written the report is factually wrong about what CLAUDE.md says, and a maintainer who opens `:689` will bounce it.
  2. `:455-520` → `:483-540`.
  3. "ten `gaia agent` subcommands" → "eleven".
- **Severity:** keep 🟡. Nothing here breaks a user — but CLAUDE.md is *this repo's* agent-steering file, so a stale mixin path and an 11-subcommand omission actively route AI contributors at code that isn't there. That is more than 🟢 "would not mislead".
- **Dup:** none.

---

### I90 CONFIRMED-ADJUST

The staleness is real and larger than the report says; **one named item (image-agent) is refuted** and three citations are imprecise.

**Roadmap half.**

| Claim | Verdict | Evidence |
|---|---|---|
| "Updated April 13, 2026" | holds | `docs/roadmap.mdx:387` — `*Updated: April 13, 2026*` (≈4.7 months before today's 2026-09-04) |
| v0.17.3 "In progress" | holds | `:135,137` — `## v0.17.3 …` / `**Status:** In progress — **Due: April 17, 2026**` |
| "six releases stale" | holds, understated | `## Shipped` ends at `### v0.17.2` (`:76-78`). **v0.18.0 through v0.25.0 are all still listed as future** with due dates April 21 – June 16, 2026 (`:156,162,173,182,200,222,246,270,290,301,325,351`) while `version.py` is 0.23.1 |
| #768 / #746 listed as future | holds | `:143,144` sit inside the v0.17.3 "In progress" table; `gh api` → both **closed** |
| "memory v2" listed as future | **imprecise** | The roadmap has no "memory v2". What exists is `:222` `## v0.20.0 — Agent Memory & Bootstrap — *Due: May 12*`, which shipped. Rename the item. |
| "`gaia schedule`" listed as future | **not supported** | `gaia schedule` appears nowhere in `roadmap.mdx`. The nearest is `:309` "Always-on agent (autonomy engine) … [#634]" — and **#634 is still OPEN**, so citing it as "shipped but listed as future" is wrong. Drop this item. |
| email autonomy listed as future | holds (via section) | `:301` `## v0.23.0 — Autonomous Agent Infrastructure` is in the future block; `gaia email autonomy` exists (`gaia email --help` → `{autonomy}`) |
| Deleted `code`/`sd` agents scheduled | holds | `:235` "Merge CodeAgent views … [#695]" (closed), `:285` "SD agent \| Consolidate SD agent (gaia sd) \| [#771]" (closed); `gaia sd --help` → `invalid choice: 'sd'` |

**Plans half — item-by-item.**

| Plan | Report's claim | Verdict | Evidence |
|---|---|---|---|
| `email-triage-agent.mdx` | "Planning (0% implemented)" | holds | `:11` verbatim; agent ships at 0.6.0 |
| `messaging-integrations-plan.mdx` | "Planning (no implementation)" | holds | `:11` verbatim; `gaia telegram --help` → `{start,stop,status}` |
| `desktop-installer.mdx` | "Planning" | holds | `:9` `**Status:** Planning` |
| `connectors.mdx` | "Target v0.18.x \| implementation underway" | holds | `:19` verbatim |
| `agent-hub.mdx` | "Target Q2 2026 \| Planning" | holds | `:10` verbatim; `gaia hub --help` → `{list,install,uninstall}` |
| `autonomy-engine.mdx` | "Planning (0% implemented)" | holds | `:19` verbatim; `gaia email autonomy` exists |
| `image-agent.mdx` | "says `gaia sd` ships — deleted in #2995" | **REFUTED** | `:9` reads in full: *"**Status:** Partially shipped — a Stable Diffusion agent already ships today (`gaia sd`; source was `hub/agents/sd/python/` — **both since removed**)…"* The page **already documents the removal**. The upstream raw note (`.review/07-docs.md`) quoted only the clause before the parenthesis. |
| 4 orphans not in `docs.json` | holds | `bash-agent`, `email-full-autonomy`, `package-publishing`, `typescript-sdk` all exist on disk, all `grep -c 'plans/<name>"' docs/docs.json` → 0. Totals: 29 `.mdx` on disk vs 25 `"plans/…"` nav entries |

- **Edit:**
  1. Delete the `image-agent` clause entirely — it is false.
  2. "seven `docs/plans/*` status headers" → "**six**".
  3. Drop "`gaia schedule`" from the future-items list (#634 is open, and the string isn't in the roadmap); rename "memory v2" → "v0.20.0 Agent Memory".
  4. Strengthen the roadmap claim to the exactly-checkable form: "`## Shipped` ends at v0.17.2; **v0.18.0–v0.25.0 are all listed as future** with due dates of April–June 2026, while `version.py` reads 0.23.1."
- **Severity:** **change 🟡 → 🟢.** Nothing a reader *runs* fails and no unsafe action is invited — these are status-metadata surfaces (a public roadmap, plan front-matter). Per REVIEW.md, 🟢 = "would not break or mislead a user"; the honest cost here is contributor time (CLAUDE.md sends agents to `docs/plans/`, and "0% implemented" invites re-planning shipped work), not user breakage. Contrast I87/I91, which make a copy-pasted command or import fail outright — those stay 🟡. Keep it in the report; just don't rank it with the executable contradictions.
- **Dup:** none.

---

### I91 CONFIRMED

The strongest finding in this section. Every claim holds; the import half is decisive by execution.

**Part A — broken imports.** Probe (`.venv\Scripts\python.exe -c "…"`, run from the checkout):

```
FAIL  from gaia.testing import temp_database
      -> ImportError: cannot import name 'temp_database' from 'gaia.testing' (src\gaia\testing\__init__.py)
FAIL  from gaia import SilentConsole
      -> ImportError: cannot import name 'SilentConsole' from 'gaia' (src\gaia\__init__.py)
FAIL  from gaia.agents import Agent
      -> ImportError: cannot import name 'Agent' from 'gaia.agents' (src\gaia\agents\__init__.py)
FAIL  from gaia.agents.hello.agent import HelloAgent, HelloAgentConfig
      -> ModuleNotFoundError: No module named 'gaia.agents.hello'
OK    from gaia.testing import MockLLMProvider, MockVLMClient, create_test_agent
```

The control line matters: the *other three* names on the same documented import statement resolve, so this is a real per-name gap, not a broken environment.

| Doc site | Verdict | Line confirmed |
|---|---|---|
| `docs/spec/test-utilities.mdx:12` — "**Import:** `from gaia.testing import … temp_database`" | holds | line reads verbatim as quoted |
| …documented in full at `:272-300`, with a test at `:464` | holds | `:271-272` `@contextmanager` / `def temp_database(schema_file…)`; `:464` `def test_temp_database_creates_and_cleans_up():` |
| `docs/spec/test-utilities.mdx:525` — `from gaia import SilentConsole` | holds | line reads verbatim |
| `docs/spec/llm-client.mdx:1191` — `from gaia.agents import Agent` | holds | inside a ```` ```python With Agent ```` block, followed by `class CustomAgent(Agent):` — i.e. presented as runnable |
| `docs/sdk/patterns.mdx:211,239` — `from gaia.agents.hello.agent import …` | holds | `:211` inside `hello_factory`, `:239` under `# tests/test_hello_agent.py` |

**Part B — eight deleted paths/agents cited by `.claude/**`, AGENTS.md, CONTRIBUTING.md.** All eight verified: the citing line exists **and** the target is gone.

| Citation | Target | On disk |
|---|---|---|
| `.claude/agents/cli-developer.md:20,35,87,108` | `tests/test_cli.py` | MISSING |
| `.claude/agents/mcp-developer.md:37` | `src/gaia/mcp/blender_mcp_server.py` + `blender_mcp_client.py` | both MISSING |
| `.claude/agents/test-engineer.md:40` | `hub/agents/code/python/tests/` | MISSING |
| `.claude/agents/gaia-agent-builder.md:80` | `CodeAgent`, `JiraAgent` as living examples | both deleted (#2995) |
| `.claude/skills/github-issue-response/SKILL.md:112` | `hub/agents/jira/python/` | MISSING |
| `.claude/skills/gaia-executive-presentation/SKILL.md:24`, `gaia-technical-presentation/SKILL.md:22` | `hub/agents/email/python/README.md` | MISSING (README is at `email/npm/README.md`, which EXISTS) |
| `AGENTS.md:157` | `docs/spec/orchestrator.mdx` | MISSING |
| `CONTRIBUTING.md:61` | `tests/unit/test_chat.py` | MISSING |

Weakest of the eight (worth knowing if a maintainer pushes back): the two presentation skills use the path only as an *illustration of slug formation* ("the slug is the source path with `/` replaced by `-`"), not as a file to open. The other six are load-bearing references.

**Part C — `util/check_doc_links.py` does not walk `.claude/`.** Holds. `find_doc_files()` (`:86-99`) globs `docs/**` for `*.mdx`/`*.md` plus exactly two hard-coded READMEs (`README.md`, `cpp/README.md`). `.claude/`, `AGENTS.md`, `CONTRIBUTING.md`, `hub/**` are all outside the walk. Note for the fix: the function returns *files to scan*, so extending it is a one-line list change — the report's "~30 lines" estimate is pessimistic.

- **Edit:** none required. Optionally note the control import (three sibling names on the same line DO resolve), which pre-empts "your env is broken".
- **Severity:** keep 🟡, and rank it **first in §3.12** — it is the only doc finding here where a reader copy-pasting a documented example gets a hard failure on line one, and it has been shipping in `docs/spec` and `docs/sdk`.
- **Dup:** none.

---

### I92 CONFIRMED-ADJUST

Nine sub-claims. **Six hold**, one is **refuted**, one is **overstated**, two line numbers are wrong (one badly). The code-side of the grant/keyring cross-refs belongs to C1/I36/I37's verifier; I checked only the doc side plus the two claims whose code I could reach.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | `security-model.mdx:89-97` "All GAIA services bind exclusively to 127.0.0.1. No public ports" vs the tunnel and `--host 0.0.0.0` | **OVERSTATED** | The sentence is real (`docs/plans/security-model.mdx:91`, table `:93-98`). But the page carries a banner at **`:8-15`**: *"⚠️ **Partially superseded by Agent UI v2** … This doc's **localhost-trust** framing … **no longer holds** under v2"* — i.e. it already self-retracts on exactly this point — and `:19` is `**Status:** Planning`, `:18` `**Date:** 2026-04-01`. Its "Enforcement" block (`:101-105`) is written in `must`/`should` future tense and proposes a `--dangerous-allow-remote` flag that does not exist. This is an unimplemented *plan*, not a doc asserting shipped behaviour. |
| 2 | `gaia api start --host 0.0.0.0` is possible | holds | `gaia api --help` → `--host HOST   Host to bind API server (default: localhost)`, no auth flag anywhere on `gaia api` |
| 3 | `gaia mcp start --host 0.0.0.0` "shown in `cli.mdx:1249` **without `--auth-token`**" | **REFUTED** | `docs/reference/cli.mdx:1245` immediately above reads *"Pass `--auth-token` before exposing it anywhere else:"*, and `:1248` of the same block is `export GAIA_MCP_AUTH_TOKEN=$(python -c "import secrets; …")`. The server honours it: `mcp_bridge.py:505` `auth_token = auth_token or os.environ.get(AUTH_TOKEN_ENV_VAR)`. `:1252` even warns to prefer the env var over the flag because `ps` leaks argv. The doc is *correct here*; delete this parenthetical. (Separately worth knowing: `resolve_bind_host` at `mcp_bridge.py:468-494` still **binds** an unauthenticated `0.0.0.0` and only `logger.warning`s — but that is a code finding, not this doc's error.) |
| 4 | `docs/security/connections.mdx:9` "never writes tokens to plaintext files" vs `instance.json` | holds | `:9` verbatim: "GAIA never writes tokens or API keys to plaintext files. All secrets live **exclusively** in your OS credential store". `src/gaia/daemon/instance.py:31` `_PERSISTED = ("pid","port","**token**","host",…)` written by `write_instance()` at mode 0600. **A stronger instance the report missed:** `src/gaia/llm/lemonade_embedded.py:19` documents `state.json  the running instance: pid, port, **API key**, version` and `:757` reads `state.get("api_key")` back — a literal *API key* in a plaintext JSON file, which is the exact noun the doc disclaims. **Caveat to state fairly:** the sentence sits under `## Credential storage` and the page is scoped to *connector* credentials; it is the absolute wording ("never … exclusively"), not the connector behaviour, that is wrong. |
| 5 | `connections.mdx:37,63` grant-enforcement claims vs C1/I36 | doc side holds, code side delegated | `:37` "An agent that calls `get_credential_sync(…)` without a matching grant receives `AuthRequiredError` … **No token is ever returned to an ungranted agent**"; `:63` "\| Malicious agent requests a credential it wasn't granted \| `get_credential_sync` checks the grants ledger…". Both line numbers exact. Whether the code honours it is C1/I36's verdict. |
| 6 | `connectors.mdx:21-25` keyring refusal vs I37 | doc side holds | `docs/security/connectors.mdx:21-25` verbatim: "Plaintext fallbacks (`keyrings.alt.PlaintextKeyring`, `EncryptedKeyring`) are explicitly **refused** at the entry of **every** save and load". Code side is I37's. |
| 7 | "`:389` lists a `state.json` no code writes" | claim holds, **citation badly wrong** | There is no line 389: `docs/security/connectors.mdx` is **142** lines and `docs/plans/connectors.mdx` is **319**. The real citations are `docs/plans/connectors.mdx:80,81,97,173,246,301` (e.g. `:173` `~/.gaia/connectors/state.json` (mode 0600)`, `:97` `state.py  # ~/.gaia/connectors/state.json atomic store`). The substance holds: `grep -rn "state.json" src/gaia --include=*.py` finds only `electron-install-state.json`, `setup_state.json`, `rate_state.json`, and the Lemonade one — **no `connectors/state.json`, and no `src/gaia/connectors/state.py`**. Note this is again a `docs/plans/` page, not user-facing security doc. |
| 8 | `spec/agent-ui-server.mdx:251` "binds to 0.0.0.0" stale vs `cli.py:858` | holds, **off by one** | The line is **`:250`**: "Note: `gaia chat --ui` binds to `0.0.0.0` for Electron/browser access". `src/gaia/cli.py:858` → `host="127.0.0.1"`. Confirmed stale. |
| 9 | No doc says what a tunnel token grants; no user-facing daemon/custody trust-model page; `SECURITY.md` links to no threat model | holds | Every `tunnel`+`token` hit in `docs/` is either a v0.17.5 release note describing the cookie *mechanism* (`:88,90`) or `sdk/infrastructure/connectors.mdx:117` "gates remote/tunnel access behind a token" — none state the token's scope. No `docs/guides/daemon*`, and `grep -n daemon docs/docs.json` → 0 hits. `SECURITY.md` is 44 lines with zero occurrences of "threat". |

- **Edit:**
  1. **Delete** "(shown in `cli.mdx:1249` without `--auth-token`)" — it is false and it is the most checkable thing in the bullet, so it will discredit the rest.
  2. Reword claim 1: the problem is not that a plan contradicts code but that **an unimplemented, self-superseded plan doc is still the repo's most prominent "Security Model" page** — retitle/archive it rather than "fix" it.
  3. `:389` → `docs/plans/connectors.mdx:97,173`.
  4. `agent-ui-server.mdx:251` → `:250`.
  5. Add the `lemonade_embedded.py:19,757` API-key-in-`state.json` instance to claim 4 — it is a sharper contradiction of "never writes … API keys to plaintext files" than `instance.json`.
- **Severity:** keep 🟡 for the bundle, but note the composition changes after the edits: claims 4, 8 and 9 are genuine shipped-doc-vs-code contradictions; claims 1 and 7 are *plan* documents (🟢 on their own — a `docs/plans/` page in `Status: Planning` is not a promise about shipped behaviour). Do **not** carry the 🔒 marker into I92 — no attacker position is established by any doc claim here; the security weight lives in C1/I36/I37.
- **Dup:** claims 5 and 6 are pure restatements of C1/I36 and I37's doc-contradiction clauses (both already say "Contradicts `docs/security/connections.mdx:63`" / "while `docs/security/connectors.mdx:21-25` says…"). Recommend deleting them from I92 and leaving the cross-reference.

---

### I93 CONFIRMED-ADJUST

**22 individually-named claims checked. 19 hold, 1 is refuted, 2 need adjusting.** Four line numbers are off. For items that cross-reference C4/C7/C20/I48/I53/I55 I verified the **doc side** (which is what I93 asserts) and, where cheap, the code side; the code verdicts belong to those findings' verifiers.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | `shell-tools-mixin.mdx` "only safe, read-only commands" (C4) | holds | `docs/spec/shell-tools-mixin.mdx:21` "Whitelist-based command security (**only safe, read-only commands**)"; also `:30,:42`. Code side = C4. |
| 2 | `filesystem_tools.py:52` docstring (C7) | holds, line exact | `:52` "All path parameters are validated through PathValidator before access." (Bonus: `:54` still names `CodeAgent`, deleted in #2995.) |
| 3 | `signing.py:6-11` "signature travels with the artifact" but nothing re-verifies at load; no `gaia skill verify` | holds | `src/gaia/skills/signing.py:10-11` verbatim. `verify_bundle` (defined `signing.py:351`) has exactly **one** caller: `install.py:242` — install time only. `gaia skill --help` lists `list/info/create/import/export/audit/migrate/search/install/remove/publish/keygen/trust` — **no `verify`**. `loader.py`/`consume.py`/`manager.py` contain no `SIGNATURE.json` or `verify_bundle` reference. |
| 4 | `agent-memory-architecture.md:401,700` + `guides/memory.mdx:457` consolidation (C20) | holds, lines exact | `:401` prune-at-90-days; `:700` "`consolidated_at` prevents re-processing"; `memory.mdx:457` "conversations older than 14 days are distilled into durable notes". Code side = C20. |
| 5 | Extraction timeout documented "3 s", code 8 | holds | Doc: `agent-memory-architecture.md:645` "**LLM timeout (3s)**", `:655` and `:2075` "Timeout: 3s". Code: `src/gaia/agents/base/memory.py:187` "LLM extraction timeout in seconds. **Raised from 3 to 8**"; the enforcing comment at `:1384` still says "(spec: 3s)". Add these cites — the report gives none. |
| 6 | `chat.mdx:294` `add_document -> bool` but returns dict | holds, **with a twist** | `docs/sdk/sdks/chat.mdx:294` `add_document(path: str) -> bool`. `src/gaia/chat/sdk.py:936` is *also* annotated `-> bool` and `:944` says "True if indexing succeeded", but `:949` returns `self.rag.index_document(...)`, and `src/gaia/rag/sdk.py:2552` is `-> Dict[str, Any]`. The doc faithfully copies a **wrong code annotation** — the fix belongs in `sdk.py:936`, not just the doc. |
| 7 | `chat.mdx:314` `create_session(**kwargs)` — nine keys honoured | holds exactly | `sdk.py:1299-1345` contains **9** `config_kwargs.get(...)` calls; `AgentConfig` has **13** fields (probe: `dataclasses.fields`), so `temperature`, `use_local_llm`, `claude_model`, `base_url` are silently dropped — **four**, not the three named in section 4. |
| 8 | `rag.mdx:994` `allowed_paths` promise (C7) | holds, line exact | `:993-995` Warning: "Without `allowed_paths`, RAG can index any file the process can read. In production, always set explicit allowed paths." Code side = C7. |
| 9 | `rag.mdx:205-290` chunk-size guidance (I48) | holds | `:205` `### What is chunk_size?`, table `:213+`, overlap rule `:259-261`, config sample `:274`. Whether the guidance is wrong = I48. |
| 10 | `rag.mdx:285,1082` "cache persists the index" | holds, lines exact | `:285` `cache_dir=".gaia",  # Where to store the index`; `:1082` `cache_dir="./my_rag_cache"  # Index persisted here`; `:1084` "Next run will load existing index instead of re-creating". |
| 11 | Nothing documents the HMAC-signed cache | holds, **understated** | Code ships it: `src/gaia/rag/sdk.py:322-333` `_get_hmac_key`, key at `~/.gaia/cache/hmac.key`. Every `hmac` hit in `docs/` is inside `docs/plans/` — and `docs/plans/security-model.mdx:52` actively lists it as **future** work ("Migrate to fully safe serialization / HMAC (#447 follow-up)"), `:680` likewise. So it isn't merely undocumented; the one doc that mentions it says it hasn't shipped. |
| 12 | `docs/guides/talk.mdx` (I55) | holds | `talk.mdx:79,87` 'Say "exit" or "quit"' vs `src/gaia/audio/audio_client.py:434` `if cleaned_text in ["stop"]:`; `talk.mdx:95` "Natural pauses (>1 second)" vs `audio_client.py:97` `MIN_AUDIO_LENGTH=0.5s`. |
| 13a | Tavily budget documented "per-session" (I53) | holds | `docs/reference/cli.mdx:2945` "a **per-session** credit budget". Enforcement = I53. |
| 13b | "…and a `crawl` the CLI lacks" | **OVERSTATED** | `crawl` **exists** — `src/gaia/web/tavily.py:477 def crawl(`, credit costs `:70-71`, budget check `:492`. The doc line (`docs/connectors/tavily.mdx:78`, "Tavily simply upgrades its quality and adds `extract`/`crawl`") describes *connector capabilities*, not a CLI subcommand, though it sits under a `gaia knowledge …` block so a reader may infer one. `gaia knowledge` really is `{search,extract,usage}`. Reword to "the `crawl` capability has no `gaia knowledge` subcommand" — 🟢 at most. |
| 14 | `vlm.mdx:139-140` `timeline_totals` with no failure caveat | holds, **understated** | `docs/sdk/sdks/vlm.mdx:140` `print(result["aggregated_data"]["timeline_totals"])`. `src/gaia/vlm/structured_extraction.py:230-231` sets `aggregated_data` **only** `if aggregated_timeline is not None`. The documented line raises **`KeyError: 'aggregated_data'`**, not merely "lacks a caveat". |
| 15 | `cpp/README.md` says RAG is Python-only while `chunking.h`/`vector_index.cpp` ship | holds, **self-contradiction** | `cpp/README.md:505` "Audio / RAG / Stable Diffusion — Python-only" and `:493` "without pulling in Python-only features (audio, RAG, …)" — yet `cpp/include/gaia/chunking.h` and `cpp/src/vector_index.cpp` exist **and the same README at `:439-445` documents the C++ splitter and its parity test against `RAGSDK._split_text_into_chunks`**. The file contradicts itself 60 lines apart; that is the sharper framing. |
| 16 | Env-var table names `GAIA_CPP_BASE_URL` but code reads `LEMONADE_BASE_URL` | **CONFIRMED-ADJUST** | Half wrong. `cpp/src/agent.cpp:57-62` reads **`GAIA_BASE_URL` first, then `GAIA_CPP_BASE_URL` as a documented deprecated fallback with a stderr warning** — the documented variable still works. `LEMONADE_BASE_URL` is a *different* knob (`types.h:334`, `lemonade_client.h:39`). The real defect: `GAIA_BASE_URL` appears **nowhere** in `cpp/README.md`, which presents the deprecated name as canonical at `:184,196,201`. |
| 17 | `dev.mdx` has no release-process section — "the only release doc is a Claude skill" | **REFUTED** | `README.md:131-151` is a full `### Release Process` section: the 3-file table (`version.py`, `docs/releases/vX.mdx`, `docs/docs.json` navbar) plus the tag command — the same table `.review/07-docs.md` itself lists under "Checked and fine". `docs/reference/dev.mdx:302-345` also carries "Verifying the Agent Hub publish pipeline". |
| 18 | `CODEOWNERS`/`labeler.yml` cite `.md` doc paths that are all `.mdx` now | holds, "**never fire**" overstated | All 13 targets MISSING: `.github/CODEOWNERS:30` `docs/sdk/security.md`; `.github/labeler.yml:25,30,35,40,45,50,94,109,114,134` (`docs/guides/chat.md`, `docs/sdk/sdks/chat.md`, `docs/guides/mcp.md`, `docs/sdk/sdks/rag.md`, `docs/sdk/sdks/llm.md`, `docs/sdk/sdks/audio.md`, `docs/guides/talk.md`, `docs/sdk/agents/talk.md`, `docs/reference/eval.md`, `docs/reference/cli.md`, `docs/deployment/installer.md`) — **plus `:15` `docs/guides/cpp.mdx`, an `.mdx` that also does not exist**. But each rule carries other globs (`src/gaia/chat/**/*` etc.), so those labels *do* fire on source changes; what breaks is a **docs-only** PR to those pages, which then gets only the generic `documentation` label (`:10`, `docs/**/*`, which works). |
| 19 | `context7.json` advertises Qwen defaults; `publish.yml` re-publishes every release | holds | `context7.json:15` "Default models: Qwen3-0.6B-GGUF (general), Qwen3.5-35B-A3B-GGUF (agents/code), Qwen3-VL-4B-Instruct-GGUF (vision)" vs `DEFAULT_MODEL_NAME = Gemma-4-E4B-it-GGUF`; `.github/workflows/publish.yml:677-686` `refresh-context7:` POSTs `https://context7.com/api/v1/refresh`. |
| 20 | `quickstart.mdx:13` calls Electron the primary install path vs the terminal-hub lead | holds | `:13` `## Recommended: Desktop Installer`, `:15` "The **GAIA Agent UI** desktop app is **the primary install path for end users**" vs `docs/guides/terminal-hub.mdx`, `docs/releases/v0.23.1.mdx:8`, and `hub/agents/gaia/npm/CHANGELOG.md`. |
| 21 | `hub/agents/README.md` promises `<id>/python/README.md` the two products lack | holds exactly | `hub/agents/README.md:45` `└── README.md` inside the mandatory `<id>/python/` tree. Probe: `gaia` NO, `email` NO; `hello-world`/`word-count`/`connectors-demo` HAVE it — only the two shipping products violate it. |
| 22 | `README.md:76` names Qwen3-VL as default VLM; omits macOS | holds, **line is `:77`** | `README.md:77` "Extract text from images with **Qwen3-VL-4B**" vs `src/gaia/vlm/mixin.py:45,55` `default: Gemma-4-E4B-it-GGUF`. `README.md:117` OS row "Windows 11, Linux" vs `docs/guides/install.mdx:13` "**macOS 14+** (Apple Silicon)" and a macOS tab at `:34`. |

- **Edit:**
  1. **Delete item 17's parenthetical** ("the only release doc is a Claude skill") — `README.md:131-151` refutes it, and the same raw source already listed that table as correct. This is the one outright-false claim in I93.
  2. **Rewrite item 16:** "`cpp/README.md:184` documents the deprecated `GAIA_CPP_BASE_URL` as canonical and never mentions `GAIA_BASE_URL`, which `agent.cpp:57-62` actually prefers."
  3. **Soften item 13b** to "the `crawl` capability (`web/tavily.py:477`) has no `gaia knowledge` subcommand".
  4. **Strengthen item 14** to name the failure: `KeyError: 'aggregated_data'`.
  5. **Soften item 18's** "labels never fire" to "the doc globs never match, so a docs-only PR gets no component label".
  6. **Add to item 6** that `sdk.py:936`'s own `-> bool` annotation is wrong, so the fix is a code fix.
  7. Line fixes: `README.md:76` to `:77`; add the missing cites for item 5 (`agent-memory-architecture.md:645,655,2075` / `memory.py:187`) and item 11 (`rag/sdk.py:322-333`, `security-model.mdx:52`).
- **Severity:** keep the bundle 🟡, but it is **not homogeneous** and should not be presented as one finding. Genuinely 🟡 (a reader who follows the doc gets a wrong result or a crash): 3, 5, 6, 7, 11, 12, 14, 15, 16, 19, 22. Genuinely 🟢 (cosmetic/positioning, nothing breaks): 13b, 18, 20, 21. Items 1, 2, 4, 8, 9, 10, 13a are **not independent findings** — they are the doc half of C4/C7/C20/I48/I53 and are already stated there.
- **Dup:** item 7 duplicates the section-4 minor bullet "`create_session(**kwargs)` drops `base_url`, `temperature`, `claude_model`" (which itself undercounts — four keys, `use_local_llm` included). Items 1/2/8 duplicate C4 and C7's own doc clauses; item 4 duplicates C20 ("Contradicts `docs/spec/agent-memory-architecture.md:401,700`"); item 13a duplicates I53; item 12 duplicates I55's last sentence. **Seven of 22 are restatements**, which is why I93 reads longer than its new information warrants.

---

## Missed (candidate, 🟢/🟡 borderline)

**A documented silent fallback that the repo's own rules prohibit.** `docs/reference/cli.mdx:2945-2947` and `docs/connectors/tavily.mdx:76-78` advertise "an **automatic keyless DuckDuckGo fallback** when the `mcp-tavily` connector isn't configured", and `src/gaia/web/tavily.py:16-17` states the same as intended behaviour ("`search` degrades to the keyless DuckDuckGo path instead of failing"). CLAUDE.md's **No Silent Fallbacks — Fail Loudly** section prohibits exactly this shape ("try the other provider" glue). It is a deliberate, documented UX choice rather than a bug, so it belongs as a policy question, not a defect — but §3.12 flags six other doc/code contradictions and misses the one where the *docs and the code agree with each other and disagree with the project's own standard*. Adjacent to I1's "with `PERPLEXITY_API_KEY` set, search_web silently becomes a paid cloud call", which the report already has.

---

## Summary (I87–I93)

**Verdicts:** CONFIRMED 1 (I91) · CONFIRMED-ADJUST 6 (I87, I88, I89, I90, I92, I93) · OVERSTATED 0 · REFUTED 0 · UNVERIFIABLE 0.

No finding in this section is false as a whole. But these seven bullets bundle **~85 individually-named claims**, and **~20 of them are wrong, imprecise, or mis-cited** — including **four that are outright refuted by the file the report cites**. A maintainer who spot-checks any one of those four will discount the section.

### The four refuted sub-claims (delete or rewrite these first)

1. **I90 — `docs/plans/image-agent.mdx` "says `gaia sd` ships".** It does not. `:9` reads "…already ships today (`gaia sd`; source was `hub/agents/sd/python/` — **both since removed**)". The page already documents the deletion; the raw source quoted only the clause before the parenthesis. Drop the clause and change "seven plans" to **six**.
2. **I92 — "`gaia mcp start --host 0.0.0.0` shown in `cli.mdx:1249` *without `--auth-token`*".** `cli.mdx:1245` immediately above says "Pass `--auth-token` before exposing it anywhere else" and `:1248` exports `GAIA_MCP_AUTH_TOKEN`, which `mcp_bridge.py:505` honours. The doc is correct here.
3. **I93 item 17 — "the only release doc is a Claude skill".** `README.md:131-151` is a complete `### Release Process` section with the 3-file table and tag command — the same table `.review/07-docs.md` lists under "Checked and fine".
4. **I89 claim 1 — CLAUDE.md "says GaiaAgent not yet landed".** It says the *`ChatAgent`→`GaiaAgent` rename* has not landed, which is literally true (`class ChatAgent(` is still at `gaia_agent_chat/agent.py:176`). The real, sharper defect: **#696 was closed `NOT_PLANNED` on 2026-09-02**, so CLAUDE.md advertises a cancelled plan while the same file's agent table already calls `GaiaAgent` the flagship.

### Other corrections that matter

- **I92 `:389`** — no such line. `docs/security/connectors.mdx` is 142 lines; `docs/plans/connectors.mdx` is 319. The real `state.json` cites are `docs/plans/connectors.mdx:97,173`. (The substance holds: no `connectors/state.json` and no `connectors/state.py` in code.)
- **I92 claim 1 is a *plan* doc, not a shipped-doc contradiction.** `docs/plans/security-model.mdx` is `Status: Planning`, dated 2026-04-01, and carries a banner at `:8-15` explicitly retracting its localhost-trust framing. The finding should be "an unimplemented, self-superseded plan is still the repo's most prominent Security Model page", not "the docs contradict the implementation". Also drop 🔒 from I92 — no attacker position follows from a doc claim; the security weight is in C1/I36/I37.
- **I93 item 16 is half wrong.** `cpp/src/agent.cpp:57-62` still reads `GAIA_CPP_BASE_URL` (as a warned deprecated fallback); the canonical name is `GAIA_BASE_URL`, which appears nowhere in `cpp/README.md`. `LEMONADE_BASE_URL` is a different knob.
- **Counts:** I87 "11 `gaia tui`" → **21**; I88 "seven README links" → **nine** (+1 in CHANGELOG); I89 "ten `gaia agent` subcommands" → **eleven**; I90 "seven plans" → **six**; I93 `README.md:76` → **:77**; I89 tree `:455-520` → **:483-540**; I92 `agent-ui-server.mdx:251` → **:250**.
- **I93 item 6 needs a code fix, not a doc fix** — `src/gaia/chat/sdk.py:936`'s own `-> bool` annotation is wrong; the doc merely copies it.
- **I93 item 14 is understated** — the documented line raises `KeyError: 'aggregated_data'` (`structured_extraction.py:230-231` sets the key conditionally), not "lacks a caveat".
- **I93 item 18 is overstated** — the labels do fire on source-path changes; what breaks is a docs-only PR, which then gets only the generic `documentation` label.

### Proposed severity changes

| Finding | Now | Proposed | Why |
|---|---|---|---|
| **I91** | 🟡 | **🟡, ranked first** | The only §3.12 finding where a copy-pasted documented example fails outright — four `ImportError`/`ModuleNotFoundError`s reproduced in the venv, with three sibling names on the same line resolving as a control. |
| **I90** | 🟡 | **🟢** | Status metadata only (a public roadmap, plan front-matter). Nothing a reader runs fails and no unsafe action is invited; the cost is contributor time. |
| **I92** | 🟡 🔒 | **🟡, no 🔒** | Two of its nine claims are `docs/plans/` pages (🟢 on their own); no attacker position is established by any doc claim. |
| **I93** | 🟡 | **split** | Genuinely 🟡: items 3, 5, 6, 7, 11, 12, 14, 15, 16, 19, 22. Genuinely 🟢: 13b, 18, 20, 21. Items 1, 2, 4, 8, 9, 10, 13a are the doc half of C4/C7/C20/I48/I53 and are already stated there. |
| I87, I88, I89 | 🟡 | keep 🟡 | Each makes a user's or contributor's next action fail. |

### Duplication to remove

- **I87 claim A is verbatim C29** (`REPORT.md:140` already says "The v0.23.1 notes also tell users to run `gaia install gaia` / `gaia list` — neither exists"). Keep it in one place.
- **I87 claim B is already tracked as #3219**, whose title is a near-verbatim restatement — the report should say so rather than present it as new.
- **Seven of I93's 22 items are restatements** of C4/C7/C20/I48/I53/I55, and item 7 duplicates a §4 minor bullet (which itself undercounts: four keys dropped, not three).

### What is strongest and should survive intact

I91 (four broken documented imports + eight dead `.claude/**` paths + the `check_doc_links.py` gap) is fully confirmed with zero corrections, as is I88 (all live npm/GitHub state re-verified today, unchanged, with a control check proving the 404 is not an auth artifact). I89's claims 2–6 and I93's items 3, 5, 7, 11, 14, 15, 19, 21 are exact and independently reproduced.

---

## §3.13 Hub Worker, website router, install scripts (I94–I99)

# Adversarial verification — I94–I99 (§3.13 Hub Worker / website-router / website / install scripts)

Checkout: `C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-3369977f` @ 211f08c5. Read-only; GET/HEAD only against live hosts.

### I94 CONFIRMED
- Evidence: `grep -rn "etagMatches\|onlyIf\|Durable" workers/agent-hub/src/ wrangler.toml` → **zero hits** (only `index.ts:32 headers.set("etag", obj.httpEtag)`, a GET response header); `storage.ts:211-219 writeAgentManifest` and `:221-225 writeIndex` both plain `bucket.put(...)`; `publish.ts:338` read → `:471 writeAgentManifest` → `:474 rebuildIndex` with 12 awaited R2 calls in between; `release_components.yml:302 terminal-hub needs: [version, worker-check, deploy-worker, tui]` and `:460 agent-ui needs: [version, worker-check, deploy-worker]` — no edge between them, so they run concurrently.
- Edit: none (every line number checks out).
- Severity: keep 🟡 — data-loss window is real but needs a CI race; `POST /reindex` (`index.ts:62-83`) recovers `index.json`.
- Dup: second half ("artifact can never be re-published because `head(key)` 409s while the manifest never lists it") is the **same orphan mechanism as I95**; consider cutting it from I94 and letting I95 own it.

### I95 CONFIRMED-ADJUST
- Evidence: write order `publish.ts:404 put(artifact)` → `:413-455` 8 doc puts → `:462-466` package-files → `:471 writeAgentManifest` → `:474 rebuildIndex`; the inline "already published" test is `:363 Boolean(await env.BUCKET.head(key))`, raising `409 version_exists` at `:365-371`. **Stronger than the report says:** the publisher does not merely *document* 409 as success — `publish_to_r2.py:372-397` GETs the orphaned object (`/agents/<id>/<v>/<filename>`, served straight from R2 by `index.ts:88-96`, no manifest lookup), byte-compares, and prints `"[publish] OK 409 — already published with identical bytes (idempotent no-op)"`. The bytes ARE identical (the put succeeded), so the retry is unconditionally green and the artifact stays absent from `manifest.json`/`index.json`.
- Edit: replace *"(the workflows document 409 as \"already published — success\")"* with *"(`publish_to_r2.py:372-397` reconciles the 409 by re-downloading the stored object and byte-comparing — the orphan's bytes match, so the retry reports `OK 409 … identical bytes` and the job goes green; nothing checks the manifest)"*.
- Severity: keep 🟡 — release-only, needs a transient failure in the tail; recovery is manual R2 deletion.
- Dup: overlaps I94's trailing clause (see above).

### I96 CONFIRMED-ADJUST
- Evidence: regex probe (Node, `%TEMP%`) against `ARTIFACT_FILENAME_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]*$/` — `true gaia-agent.yaml / README.md / CHANGELOG.md / SPEC.md / SKILL.md / EVALUATION.md / CAPABILITY_MATRIX.md / SCORECARD.md / package-files.json / audit.json`, `false ../evil`, `false a/b`; those are exactly the keys `storage.ts:47-81` builds inside the same `versionDir(id, version)` that `artifactKey` (`:43-45`) uses, and `publish.ts:404` puts the artifact before `:413-455` overwrite it (gated on `!versionExists`, i.e. the first publish of a version — the collision case). `grep -rn "reserved" src/` finds no filename guard on either lane.
- Edit: **the cited line is wrong** — `ARTIFACT_FILENAME_RE` is at `multipart.ts:20`, and `multipart.ts` is only 91 lines, so `:154` cannot exist. Change *"`multipart.ts:154 ARTIFACT_FILENAME_RE`"* → *"`multipart.ts:20 ARTIFACT_FILENAME_RE`, enforced at `publish.ts:313` / `skill-publish.ts:110`"*.
- Severity: keep 🟡 — publishers are authenticated and no live artifact uses a reserved name, but it silently breaks the SHA-256 guarantee the installers depend on.
- Dup: none.

### I97 CONFIRMED-ADJUST
- Evidence: real `jsonschema` (Draft7, `.venv`) validation. `audit.ts:246-256` (unaudited) and `:280-291` (ALLOW) objects vs `index.schema.json:248-269` → *"Additional properties are not allowed ('attestation', 'cleared_tiers', 'content_digest', 'manifest_digest' were unexpected)"* for **both** verdict paths. Live `GET https://hub.amd-gaia.ai/agents/<id>/manifest.json` vs `manifest.schema.json`: `terminal-hub` **2 errors** (`'go' is not one of ['python','cpp']` **and** `Additional properties are not allowed ('type' was unexpected)`), `agent-ui` **2 errors** (same two, `typescript`), `gaia` **1 error** (`type`), `email` clean → **3 of the 4 live manifests fail**, not 2. `manifest.ts:25 VALID_LANGUAGES = new Set(["python","cpp","go","typescript"])`; `grep -rn manifest.schema.json test/` is empty. `skill-publish.test.ts:667 Object.keys(entry).filter(...)` — top level only, confirmed.
- Edit: two changes. (1) `manifest.schema.json` is staler than stated — add *"and it has no `type` property under `additionalProperties: false`, so **three of the four live manifests fail** (`terminal-hub`, `agent-ui` on `language` + `type`; `gaia` on `type`)"*. (2) Gate the skill claim: live `index.json` currently validates clean (0 errors) because **no skill has been published yet** — say *"will reject every skill entry the moment the first one is published"* rather than the present tense.
- Severity: keep 🟡 — documentation-contract break, no runtime consumer validates today.
- Dup: none.

### I98 CONFIRMED-ADJUST
- Evidence (LIVE, 2026-09-04 14:11 UTC — matches the report): `curl -sS -I https://amd-gaia.ai/install.sh` → `HTTP/1.1 302 Found` / `Location: https://raw.githubusercontent.com/amd/gaia/main/installer/scripts/install.sh`, `Server: cloudflare`, **no `cf-cache-status` and no Railway/Worker header** — an edge Redirect Rule, which runs before Workers; `/install.ps1` identical (`…/main/installer/scripts/install.ps1`). `workers/website-router/src/index.js` is 151 lines and `grep -n "install\|302\|redirect"` returns **nothing** — only `isDocs()` → Mintlify and "everything else" → Railway (`:107-150`), exactly as `README.md:8` documents. `website/public/` = `favicon.ico gaia-icon.png robots.txt`. Unpinned uv confirmed: `install.sh:172 fetch_file https://astral.sh/uv/install.sh`, `install.ps1:71 irm https://astral.sh/uv/install.ps1 | iex`.
- Edit: add the tracking reference — **`gh issue view 778`: "Installation script 'install.sh' not found" (CLOSED 2026-04-17), reporter got `curl: (22) The requested URL returned error: 404` on the documented one-liner.** The failure mode I98 predicts has already fired in production once. Change *"Tracked: none found"* → *"Tracked: #778 (the same redirect already 404'd once)"*.
- Severity: **keep 🟡.** The 🔴 case is arguable — REVIEW.md:32 reads "security … or a bug that will fire in normal use", and an invisible edge rule that already broke the quickstart once fits the second clause. But the supply-chain framing is weaker than the report implies: the redirect targets **the project's own repo over HTTPS**, so the trust boundary is unchanged from "who can merge to `main`" — the same people who cut releases. It is release-integrity/rollback hygiene, not an exploitable vulnerability. The genuine third-party supply-chain link here is the unpinned `astral.sh/uv` fetch, and that is already I63.
- Dup: the uv clause duplicates **I63** — the report says so; keep the cross-reference, don't double-count.

### I99 CONFIRMED-ADJUST
- Evidence (LIVE, re-probed with real **GET**, not HEAD — HEAD matters because `index.js:139 const cacheable = request.method === "GET"` skips the `cf` block entirely on HEAD): `GET https://amd-gaia.ai/` ×3 and `GET /hub` ×3 → all `CF-Cache-Status: DYNAMIC`, no `Cache-Control`, no `Age`. `x-railway-request-id` differs on **every** `/hub` GET (`riddrz3ATLmHfY7-lt7tkg`, `XWyjT79-R8OuGtoYwUFZXw`, `nRpNuCPZRCi1cLHsCYBc-A`) — proof each page view round-trips to Railway. Assets: `GET /_astro/_id_.DsZcM0tr.css` → `CF-Cache-Status: HIT, Age: 1807, Cache-Control: max-age=14400`; `/_astro/hoisted.DTKnx4bl.js` → `MISS` then `HIT`. Repo says `ASSET_CACHE_TTL = 31536000` (`index.js:17`), `HTML_CACHE_TTL = 60` (`:18`), README:70-80 promises both.
- Edit: **the stated cause is wrong.** The Railway origin sends **no `Cache-Control` at all** — `curl -D- https://website-production-82ab.up.railway.app/hub` and `…/_astro/_id_.DsZcM0tr.css` return only `etag` + `vary: Accept-Encoding` (which is what the router README:70 itself says). So `max-age=14400` is the **zone's default Browser Cache TTL (4 h)**, not an origin header. Replace *"assets are cached only because the Railway origin (`serve`) sends `max-age=14400`"* → *"assets get the zone's default 4 h Browser Cache TTL (`max-age=14400`), not the one-year `ASSET_CACHE_TTL` the Worker asks for — the origin sends no `Cache-Control` at all"*. Also worth adding the `x-railway-request-id` differential as the positive proof of the round-trip.
- Severity: keep 🟡 — performance/cost, no correctness or security impact; the README claim is the concrete defect.
- Dup: none.

## Summary (I94–I99)

**Counts:** CONFIRMED 1 (I94) · CONFIRMED-ADJUST 5 (I95, I96, I97, I98, I99) · OVERSTATED 0 · REFUTED 0 · UNVERIFIABLE 0. Every mechanism in §3.13 holds up; the corrections are to line numbers, causes, and scope — not to the findings themselves.

**Most important corrections (in order):**
1. **I99 names the wrong cause.** The Railway origin sends **no** `Cache-Control` (verified against the origin directly); the `max-age=14400` on assets is the zone's default 4 h Browser Cache TTL. The symptom is confirmed and now has better proof — `x-railway-request-id` changes on every HTML GET. Also note the report's original probe used `curl -sI` (HEAD), and `index.js:139` disables the `cf` cache block on non-GET; I re-ran with real GETs and the result is the same.
2. **I97 understates `manifest.schema.json`.** It is not just the `language` enum: the schema has no `type` property under `additionalProperties: false`, so **three of the four live manifests fail** (`terminal-hub` and `agent-ui` on `language` + `type`; `gaia` on `type`; only `email` validates). Verified with a real Draft-7 `jsonschema` run against the live `GET /agents/<id>/manifest.json`. The skill-audit half is exactly right (`attestation`/`cleared_tiers`/`content_digest`/`manifest_digest` all rejected, on both the `ALLOW` and `unaudited` paths) but is **latent** — the live `index.json` validates clean today because no skill has been published yet; phrase it in the future tense.
3. **I96 cites a line that does not exist.** `ARTIFACT_FILENAME_RE` is `multipart.ts:20`, and the file is only 91 lines long — `:154` is impossible. The claim itself is decisive: a Node probe shows `gaia-agent.yaml`, `README.md`, `SKILL.md`, `package-files.json`, `audit.json` all pass the regex while `a/b` and `../evil` fail, and `grep -rn reserved src/` finds no guard on either lane.
4. **I95 is stronger than written.** The workflows do not merely *document* 409 as success — `publish_to_r2.py:372-397` re-downloads the orphaned object, byte-compares, finds them identical (the `put` succeeded; only the tail failed) and prints `OK 409 … identical bytes`. Nothing consults the manifest, so the retry is unconditionally green.
5. **I98 has a tracking issue the report missed:** #778, *"Installation script 'install.sh' not found"* — a user hit `curl: (22) … 404` on the documented one-liner. The predicted failure has already happened once.

**Proposed severity changes:** none. I98 was argued for 🔴 and I recommend keeping 🟡 — the redirect targets the project's own repo over HTTPS, so the trust boundary is "who can merge to `main`", the same set who cut releases; that is release/rollback hygiene rather than a vulnerability. The real third-party supply-chain link in that finding is the unpinned `astral.sh/uv` fetch, already counted as I63.

**Dedup note:** I94's trailing clause ("…an artifact that can then never be re-published because `head(key)` 409s while the manifest never lists it") is the same orphan mechanism I95 owns. Cut it from I94.

---


# Summary — I1–I99

## Counts per verdict

| Verdict | Count | Section breakdown |
|---|---:|---|
| **CONFIRMED** | 47 | 3.1: 7 · 3.2: 11 · 3.3: 6 · 3.4–3.6: 7 · 3.7–3.8: 7 · 3.9–3.10: 3 · 3.11: 4 · 3.12: 1 · 3.13: 1 |
| **CONFIRMED-ADJUST** | 47 | 3.1: 7 · 3.2: 5 · 3.3: 5 · 3.4–3.6: 7 · 3.7–3.8: 4 · 3.9–3.10: 4 · 3.11: 4 · 3.12: 6 · 3.13: 5 |
| **OVERSTATED** | 5 | I9, I15 (first half), I26, I38, I67 |
| **REFUTED** | 0 | — |
| **UNVERIFIABLE** | 0 | — |

**No 🟡 finding in §3 is false as a whole.** Every mechanism exists in the code at this commit.
That is the headline result and it is worth stating plainly, because the corrections below are
numerous and could otherwise read as a rebuttal. They are not: they are precision fixes to
line numbers, reachability, blast radius, and ranking.

The corrections do cluster, though. At **sub-claim** level the picture is less clean —
roughly **35 individual sub-claims across the 99 are wrong, mis-cited, or refuted by the very
file the report points at**, and they are concentrated in the two most claim-dense sections
(§3.12 alone bundles ~85 named claims with ~20 wrong and 4 refuted). A maintainer who
spot-checks one bad cite discounts the surrounding good ones, so these are worth fixing before
anything else.

## The five most important corrections

1. **The report's supply-chain story is fragmented into invisibility, and its two real 🔴s are
   filed as 🟡.** One defect class — *unverified third-party code executes in a privileged
   position* — is split across four separate 🟡 bullets (I25 elevated Lemonade installer, I63
   packaged `irm | iex` rescue, I86 the same MSI on the CI side, I98 the unpinned `astral.sh`
   fetch in both install one-liners), each of which reads as minor on its own. Meanwhile the
   two findings that clearly clear REVIEW.md's 🔴 bar sit at 🟡: **I81** —
   `pypa/gh-action-pypi-publish@release/v1` is a mutable **branch** (verified: the branch ref
   resolves, `git/refs/tags/release/v1` 404s) running inside a job holding `id-token: write`
   for PyPI trusted publishing, i.e. the tj-actions/changed-files shape — and **I84** — ten
   `pull_request` workflows executing fork code on persistent self-hosted AMD-lab Windows
   runners with **zero** `head.repo` gates and a reused `.venv`. Group the four, escalate the
   two.
2. **I38 is overstated on both halves, and the reachability facts are omitted.** Both
   mechanisms reproduce (probe: scanning a sibling directory deletes `docs/b.txt`; `c++` and
   `report (final)` raise `fts5: syntax error`). But `scan_directory` has **zero callers in
   the entire repo**, and the one caller of `query_files` wraps it in
   `except Exception: logger.debug(… falling back to filesystem)` — *and* routes any query
   containing `(` away from the index entirely, so the report's own headline example never
   reaches the code it blames. As written the finding reads as live index corruption plus a
   live crash; neither reaches a user at this commit.
3. **I43's citations all point past the ends of their files.** `src/gaia/schedule/store.py`
   is 191 lines and is cited at `:314-321`, `:404-408`, `:450-462`; `sinks.py` is 125 lines
   and is cited at `:247`. The finding itself is real and I reproduced it (`Schedule(cron="not
   a cron")` writes cleanly, then `next_fire_time` raises `ValueError` into `list`/`show`/
   `run`/`daemon`) — but no reviewer can check it as written. The Telegram-token clause also
   needs rewording: GAIA never persists that token; there is no `--token` flag. The correct
   line ranges are in the I43 block.
4. **§3.12 is one-third index and one-fifth wrong.** Seven of I93's 22 items are restatements
   of C4/C7/C20/I48/I53/I55, I87's first claim is verbatim C29, and I92 re-points at I36/I37 —
   so the doc section inflates the finding count by roughly nine. Of the claims that *are*
   new, four are refuted by the cited file: `image-agent.mdx` already documents `gaia sd`'s
   deletion; `cli.mdx:1245` *does* tell the user to pass `--auth-token`; `README.md:131-151`
   *is* a full release-process section; and CLAUDE.md says the *rename* has not landed, which
   is true. Drop 🔒 from I92 — no attacker position follows from a doc claim.
5. **Two findings bury a 100 %-reproducible user-facing bug inside a multi-mechanism bullet.**
   I44's "the UI scheduler computes fixed-time schedules in UTC" means *every* "daily at 9am"
   fires at 09:00 UTC for every non-UTC user (`ui/scheduler.py:564-576`, and `compute_next_run`
   is the single next-run source for five call sites). I13's third clause means date-only
   `time_to` silently drops the whole final day — the tool's own docstring example is the
   broken shape. Both should be their own findings; ranked as they are, a maintainer triaging
   by severity will never see them.

## Proposed severity changes

### 🟡 → 🔴 (13, plus 2 splits that arrive at 🔴)

| ID | Why it clears REVIEW.md's 🔴 bar ("security, data loss, breaking API, or a bug that fires in normal use") |
|---|---|
| **I2** | SSRF. `fetch_webpage`/`open_url` are registered in the flagship's default `full` profile and do bare `httpx.get(url, follow_redirects=True)` with only a scheme check — re-opening exactly what `web/client.py` was written to close (loopback, the daemon API, `169.254.169.254`, redirect-to-private). Reached by indirect prompt injection from any indexed document or fetched page. |
| **I4** | `git log --output=<path> --format=<anything>` is a near-arbitrary-content write to an arbitrary path *through a policy the code calls read-only* (probe-verified). Same outcome class as C4, which is already 🔴. |
| **I11** | An `update` op or a `remember` call mints `system`/`profile`/`permission` memory rows that render **first** in every future system prompt — persistent prompt injection plus self-granted autonomy approval, breaking an invariant `memory_store.py:110-118` states verbatim. |
| **I35** | Persistent arbitrary code execution. Probe: `.bashrc`, `.profile`, `.zshrc`, `.gitconfig`, `.config/autostart/*.desktop` and PowerShell `profile.ps1` all return `(True, '')` from `validate_write` while `.ssh` and `.env` are correctly refused; one document attached in `$HOME` makes all of `$HOME` writable; and `_prompt_overwrite` auto-approves in the non-interactive Agent-UI path. C4's outcome reached through the write tool. |
| **I41** | Unauthenticated, remote, no click. Empty allowlist → `_allowed()` returns `True` for anyone; strangers' uploads go straight into the user's global RAG library (`telegram.py:119`) — C9's "persistent prompt injection". The opt-in defence fails because `telegram-adapter.mdx:71` tells the user off-list senders are ignored. |
| **I68** | Fires on normal rendering, no adversary: ```` ```shell ````/`console`/`cmd`/`mermaid`/`jsonc`/`env` fences are unwrapped into prose, and an unbalanced `{` after a `"tool"` key silently deletes **every remaining character of the message**. Probe-verified; no test covers it. |
| **I72** | Every Windows contributor on a clean `main` gets ~640 red tests, 100 % reproducible, disabling the primary local gate on the product's primary platform. ~5-line fix. |
| **I73a** *(split)* | `import gaia` raises `RuntimeError: Could not determine home directory` from module scope when `USERPROFILE`/`HOME`/`HOMEDRIVE` are absent (`logger.py:50`, stripped-env probe). That is a product import failure in any Windows service, scheduled task, or `subprocess.run(env=…)` — not test hygiene. |
| **I77** | `--fix` is advertised in the CLI epilog and is reached for precisely when the tree is dirty; it then runs `claude -p --dangerously-skip-permissions` over the whole checkout, three times, with a kill-mid-edit path whose error the caller never inspects. Uncommitted work overwritten this way is unrecoverable, and the user never types the dangerous flag. |
| **I78a** *(split)* | `test_eval_rag.yml:74` short-circuits at `cli.py:3840` before any eval runs, inverts baseline/current, and exits 1 when `eval/results/baseline.json` is absent. A CI gate that structurally cannot evaluate its stated criterion. |
| **I80** | `pytest … \|\| echo "Backend tests skipped"` in `publish.yml:290` is verbatim the silent-fallback violation CLAUDE.md prohibits by name, in the one workflow that ships to PyPI and npm, inside a job `build-pypi` and `approve-publish` both depend on. |
| **I81** | See correction 1. Mutable **branch** ref inside a PyPI trusted-publishing job. |
| **I84** | See correction 1. Ten workflows, zero gates, persistent lab hardware. Correct attacker position: any contributor with one merged PR runs unapproved; a first-timer needs one maintainer click that then covers every later force-push. (State the limit too — `pull_request` grants no secrets, so the payload is host compromise and cross-run persistence, not credential theft.) |

Two more are 🔴-adjacent and at minimum should be split out and ranked first among 🟡:
**I44's UTC bug** and **I13(c)'s date-only `time_to`** (both in correction 5). **I47** (dropped
embeddings silently misalign chunks and vectors → confidently wrong answers) was argued for 🔴
and rejected, but belongs at the top of 🟡.

### 🟡 → 🟢 (7, plus 4 partial demotions)

| ID | Why |
|---|---|
| **I9** | Not a live hole. Validation runs *before* the confirmation prompt and none of the seven missing binaries is in `ALLOWED_COMMANDS`/`BINARY_POLICIES`, so "always allow" is never offered for them. The stated failure scenario is false at this commit. |
| **I15** (first half) | The failure scenario is fabricated — there is no `gaia memory bootstrap --sources` flag anywhere in the repo and the only in-tree caller passes no `sources`. The second half (`session_registry.delete()` skipping `run_lock`) is confirmed and should carry the bullet. |
| **I62** | Drift hazard only; all four sites currently agree on 30 s, and the constant is in the same file as two of the three strings. |
| **I67** | Dead code. Nothing loads `src/gaia/electron/` — its only entry point points at `src/gaia/apps/jira/webui`, which no longer exists. Ships to nobody; **drop the 🔒**. The stale spec claiming it is the Agent UI shell is the only 🟡 left. |
| **I90** | Status metadata only (a public roadmap, plan front-matter). Nothing a reader runs fails; the cost is contributor time. One of its claims is refuted outright. |
| **I82** (build_cpp_email half) | The 13 red runs came from a self-triggering `paths:` entry removed in #3142 on 2026-08-31; the newest red run predates the fix, and the repo has run **zero** `merge_group` events, so the trigger the report blames has never fired. Latent, not live. Keep 🟡 for the `test_gaia_cli.yml` / `release_components.yml` halves. |
| **I85** (runner half) | The 10 cancellations pre-date #2306's fix and the monitor's own step conclusions refute "weekly false Teams alarms". Keep 🟡 only if reframed around the genuinely unmonitored `lemonade-eval` release-gate box. Dependabot half stays 🟡. |
| **I59, I93, I92** (partial) | I59: the doc contradiction stays 🟡, the CHANGELOG gap and duplicated regex drop to 🟢. I93: items 13b, 18, 20, 21 are 🟢. I92: **drop the 🔒** — two of its nine claims are `docs/plans/` pages and no attacker position follows from a doc claim. |

## Duplicates to merge (the finding count is inflated)

**I-to-C:**

- **I34 ↔ C9** — I34's "also reachable without a tunnel via C9's DNS-rebinding read side" *is*
  C9, which already names `/api/files/preview?path=~/.ssh/id_rsa` as its read-side proof. Cut
  the sentence from I34 and point C9 at I34 instead; as written one exploit is counted twice,
  once at 🔴 and once at 🟡.
- **I36 ↔ C1** — keep both, but sharpen: "(concrete instance of C1)" *understates* it. Tavily
  passes neither `agent_id` nor `required_scopes`, so it stays ungated **even after C1 is
  fixed**. It needs its own fix at the call site.
- **I87 claim A ↔ C29** — verbatim restatement (`gaia install gaia` / `gaia list`). Keep once.
- **I93 items 1, 2, 4, 8, 9, 10, 13a ↔ C4/C7/C20/I48/I53/I55** — seven of 22 items.
- **I42's `--force` / `--health-port` no-ops ↔ C26** — same "accepted and ignored" pattern.

**I-to-I:**

- **I25 ↔ I86** — the same unpinned Lemonade MSI on the installer and CI sides. They must carry
  the same severity; merge into the supply-chain group from correction 1.
- **I63 ↔ I98** — the same unpinned `astral.sh/uv` fetch, two delivery paths. Same group.
- **I94 ↔ I95** — I94's trailing "…an artifact that can then never be re-published" is the
  orphan mechanism I95 owns. Cut it from I94.
- **I31 ↔ I64** — both are the #3307 "failure reported as success" pattern.
- **I92 ↔ I36 / I37** — I92's grant-enforcement and keyring rows re-state them. Keep as
  pointers, not restatements.
- **I56 / I57 / I59's `email.mdx:641`** — not duplicates, but three faces of one story:
  Outlook shipped and was documented as shipped, but the timestamp parser, the query
  translator and the Limitations section were never brought along. One "Outlook parity"
  heading tells it in one line instead of three.

## Where the report is strongest

Worth recording, because the correction list is long. §3.9–3.10 (tests and eval) is the most
solid block in the report: **every re-runnable tally matched to the test** — `33 F / 217 E`,
`213 F / 42 E`, `124 E`, `53 F`, `17 F`, `6 of 75`, `37 F`, and the 76/84/155 uncovered-test
counts. I91 (four documented imports that raise `ImportError`, reproduced in the venv) and I88
(all live npm/GitHub state re-verified today, with a control proving the 404 is not an auth
artifact) survive with zero corrections. My own section's probes all reproduced the report's
stated results exactly.
