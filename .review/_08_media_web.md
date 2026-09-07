# Review 08 — media (audio/talk/sd/vlm/utils) + web

Reviewer scope 08. Checkout: detached HEAD 211f08c5. Python: `.venv\Scripts\python.exe` (3.13.11).

## Scope covered

**Read fully and traced:**
- `src/gaia/web/client.py` (930 lines) — `PinnedIPAdapter`, `WebClient.validate_url` / `_validate_host_ip`, `_request` redirect loop, `_consume_body_capped`, `download`, `_sanitize_filename`, `search_duckduckgo`.
- `src/gaia/agents/tools/browser_tools.py` (322 lines) — `fetch_page`, `search_web`, `download_file` and their PathValidator hooks.
- Empirical probe of the SSRF validator (42 URLs incl. IPv6-mapped, NAT64, 6to4, decimal/hex/octal literals, `localhost.`, userinfo tricks, CGNAT) by importing `WebClient` in the venv — results below.
- Unit test run for the scoped suites (results in *Test gaps*).

**Only partially covered / NOT read (session was cut by a usage limit before these were reached — the integrator should not treat them as reviewed):**
- `src/gaia/web/tavily.py` — not read.
- `src/gaia/audio/*` (audio_client, audio_recorder, kokoro_tts, whisper_asr, README) — not read.
- `src/gaia/talk/{app,sdk}.py` incl. the #3280 RESTART voice command — not read.
- `src/gaia/sd/*`, `src/gaia/vlm/*`, `src/gaia/utils/{file_watcher,parsing}.py` — not read.
- Docs cross-check (`docs/guides/talk.mdx`, `docs/sdk/sdks/audio.mdx`, `docs/sdk/sdks/vlm.mdx`) — not done.
- The test files themselves (`test_web_client_*.py`, `test_fetch_sidecar.py`, talk/audio/sd/vlm unit tests) were executed but not read line-by-line.

## Findings

### 🔴 `download_file` overwrites an existing file in the target directory with server-chosen content, and the post-download "sensitive filename" guard then deletes it
- **Where:** `src/gaia/web/client.py:770-825` (`WebClient.download`) + `src/gaia/agents/tools/browser_tools.py:276-305` (`download_file`)
- **What:** The filename is taken from the remote `Content-Disposition` header (attacker-controlled) when the model does not pass one, and the file is opened with `open(save_path, "wb")` with no existence check. The sensitive-filename guardrail (`is_write_blocked(saved_path)`) only runs *after* the write and reacts by `unlink()`-ing the path — so if the name collided with a pre-existing user file, that file is first clobbered and then deleted.
- **Failure scenario:** Agent is asked (or prompt-injected from a fetched page) to `download_file("https://evil.example/x", save_to="~/Downloads")`. The server replies `Content-Disposition: attachment; filename=report.pdf` — the user's existing `~/Downloads/report.pdf` is silently replaced. If the server instead picks a name the guardrail rejects (e.g. `credentials.json`) and that file already existed, the user's original file is overwritten *and then deleted* by the guard (`Path(saved_path).unlink(missing_ok=True)`), i.e. data loss caused by the security check itself.
- **Evidence:**
  ```python
  # client.py:773-778
  cd = response.headers.get("Content-Disposition", "")
  if "filename=" in cd:
      match = re.search(r'filename[*]?=["\']?([^"\';]+)', cd)
      if match:
          filename = match.group(1)
  ...
  # client.py:813
  with open(save_path, "wb") as f:
  ```
  ```python
  # browser_tools.py:294-301  (runs AFTER the file is written)
  is_blocked, reason = mixin._path_validator.is_write_blocked(saved_path)
  if is_blocked:
      try:
          Path(saved_path).unlink(missing_ok=True)
      except OSError:
          pass
  ```
  Nothing between line 788 (`_sanitize_filename`) and 813 checks `save_path.exists()`; `_sanitize_filename` only strips separators/control chars and Windows device names, it does not prevent collisions with ordinary names.
- **Fix:** (1) Compute `save_path` and run `is_write_blocked(str(save_path))` *before* opening the file; (2) refuse to overwrite (`if save_path.exists(): raise ValueError(...)`) or write to a unique name / a temp file + atomic rename; (3) drop the `except OSError: pass` (silent-fallback violation of CLAUDE.md) — log or re-raise.
- **Confidence:** High
- **Tracked:** none found (see `gh` check note at end of Findings)

### 🟡 SSRF filter allows the 100.64.0.0/10 (CGNAT / Tailscale) range
- **Where:** `src/gaia/web/client.py:44-52` (`_is_blocked_ip`)
- **What:** The block predicate is `is_private or is_loopback or is_link_local or is_reserved or is_multicast`. Python's `ipaddress` classifies `100.64.0.0/10` as neither private nor global, so it passes. That range is the address space of Tailscale/WireGuard mesh networks and carrier NAT — machines reachable *only* because the host is inside a private overlay, which is exactly what an SSRF filter is meant to protect.
- **Failure scenario:** A user running GAIA on a Tailscale-joined laptop asks the agent to fetch a page; a prompt-injected page tells it to `fetch_page("http://100.100.100.100/admin")` (a Tailscale peer / MagicDNS). `validate_url` and `PinnedIPAdapter` both accept it and the request is sent to the private peer.
- **Evidence:** Probe run in the venv (`WebClient().validate_url(...)`):
  ```
  ALLOWED  http://100.64.0.1/
  ALLOWED  http://100.100.100.100/
  100.64.0.1 blocked= False private= False global= False loop= False ll= False res= False
  ```
- **Fix:** add `or not ip.is_global` (also covers future IANA special-purpose ranges) to `_is_blocked_ip`, or add `ip_network("100.64.0.0/10")` to an explicit `BLOCKED_NETWORKS` list; add a case in `tests/unit/test_web_client_edge_cases.py`.
- **Confidence:** High
- **Tracked:** none found

### 🟡 `Content-Length` parsed with a bare `int()` — a malformed header surfaces as a Python parse error to the model
- **Where:** `src/gaia/web/client.py:426-427` (`_request`) and `:763-764` (`download`)
- **What:** `int(content_length)` on a non-numeric `Content-Length` (misconfigured servers send `"unknown"` or comma-joined duplicates like `"123, 123"`) raises `ValueError: invalid literal for int()`. `fetch_page` catches `ValueError` and returns `f"Error: {e}"`, so the model sees the bare Python message with no hint about which server/header caused it. `requests` itself tolerates such headers and the body is already capped by `_consume_body_capped`, so the pre-check is not load-bearing.
- **Failure scenario:** `fetch_page("https://misconfigured.example/")` → tool result is `Error: invalid literal for int() with base 10: '123, 123'`; model retries or gives up.
- **Evidence:**
  ```python
  content_length = response.headers.get("Content-Length")
  if content_length and int(content_length) > self._max_response_size:
  ```
- **Fix:** parse defensively and raise an actionable `ValueError(f"Malformed Content-Length {content_length!r} from {url}")`, or skip the pre-check when unparseable and rely on the streaming cap.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `PinnedIPAdapter` pin cache never expires
- **Where:** `src/gaia/web/client.py:144,147-164` (`_pinned_cache`, `_resolve_first_ip`)
- **What:** Once a `(host, port)` resolves, the IP is reused for every later request for the lifetime of the `WebClient`. In a long-lived process (Agent UI / daemon) DNS changes (CDN failover, host migration) are never picked up until restart.
- **Failure scenario:** Agent UI runs all day; a site fetched in the morning moves IP; every later `fetch_page` on it times out.
- **Evidence:** `if key in self._pinned_cache: return self._pinned_cache[key]` — no TTL, no eviction path anywhere in the file.
- **Fix:** store `(ip, monotonic_ts)` with a 60–300 s TTL, or evict on connection failure.
- **Confidence:** High
- **Tracked:** none found

### 🟢 `except OSError: pass` in `download_file` cleanup (silent fallback)
- **Where:** `src/gaia/agents/tools/browser_tools.py:298-301`
- **What:** A failed deletion of a file the security policy says must not exist is swallowed; the tool still reports "blocked" while the file stays on disk.
- **Failure scenario:** Windows AV holds a lock on the fresh download → `unlink` raises `PermissionError` → blocked file remains in `~/Downloads`, nobody is told.
- **Evidence:** quoted under the 🔴 finding.
- **Fix:** log with the path and say "could not be removed" in the returned error; better, never create the file (see 🔴 fix).
- **Confidence:** High
- **Tracked:** none found

*Issue-tracker check (`gh issue list -R amd/gaia --state open --search ...`):* `"download overwrite"`, `"Content-Disposition"`, `"100.64"`, `"SNI pinned"` → no relevant open issue. `"SSRF"` → only the umbrella #1460 (EPIC: L2 Web Search & Extraction) — none of the above is individually tracked.

## Test gaps

**Pytest run** (`.venv\Scripts\python.exe -m pytest tests/unit/test_talk_config.py tests/unit/test_talk_voice_commands.py tests/unit/test_audio_client_mic_check.py tests/unit/test_audio_recorder_sd.py tests/unit/test_web_client_edge_cases.py tests/unit/test_web_client_ip_pinning.py tests/unit/test_fetch_sidecar.py tests/unit/test_sd_mixin.py tests/unit/test_structured_vlm_extraction.py src/gaia/audio/tests -q -p no:cacheprovider`):

- The nine `tests/unit/*` files: **337 passed in 21.47s**, 0 failures.
- `src/gaia/audio/tests/{test_audio_pipeline,test_mic_simple,test_talk_basic}.py`: **3 collection errors**, exit code 2:
  ```
  src\gaia\audio\tests\test_audio_pipeline.py:15: in <module>
      import sounddevice as sd
  E   ModuleNotFoundError: No module named 'sounddevice'
  ```
  (identical for the other two). Root cause: `sounddevice` is only in the `talk` extra (`setup.py:274-275`), the dev venv doesn't install it, and the files do a bare top-level `import sounddevice` with no `pytest.importorskip("sounddevice")` and no skip marker. They also live under `src/` (outside `testpaths = ["tests"]` in `pyproject.toml:51`), so CI never collects them — they are effectively dead manual scripts masquerading as tests. **Gap:** either move them under `tests/integration/` with `pytest.importorskip` + a hardware marker, or rename them so they stop looking like the test suite.
- No test covers the CGNAT/Tailscale range, a malformed `Content-Length`, or `download()` into a directory where the target filename already exists (grep of `tests/unit/test_web_client_edge_cases.py` for `100.64`, `exists()`, `overwrite` → no hits; every `Content-Length` fixture there is a well-formed integer).
- Not assessed (session cut short): depth of `tests/unit/test_fetch_sidecar.py`, talk/audio/sd/vlm unit tests, and `tests/test_sd_model_sweep.py` / `tests/test_vlm_integration.py`.

## Documentation gaps

- Not cross-checked in this session: `docs/guides/talk.mdx`, `docs/sdk/sdks/audio.mdx`, `docs/sdk/sdks/vlm.mdx` vs. CLI flags / voice-command list / defaults. **The integrator should assign this to another pass.**
- `src/gaia/web/client.py:125-139` documents an HTTPS limitation of the pinned-IP design (SNI is sent as the pinned IP / suppressed, so SNI-vhosted servers "may return the wrong certificate or reject the handshake"). If that is true in practice for CDN-fronted sites, `fetch_page` is documented nowhere user-facing as HTTPS-unreliable — see Hypotheses; if confirmed it must appear in the browser-tools docs.

## Improvement opportunities

- `_is_blocked_ip`: switch to `not ip.is_global` as the primary predicate — one line, closes CGNAT and any future special-purpose block automatically.
- `download()`: derive the filename *before* the network request where possible (URL path) and only consult `Content-Disposition` when the caller gave none, then check existence + `is_write_blocked` before opening; write to `<name>.part` and rename on completion so an aborted/oversize download never leaves a truncated file with the final name (today the oversize path unlinks, but a timeout mid-stream leaves a partial file — `iter_content` raising propagates out of the `with` block without cleanup, `client.py:813-823`).
- `download()` does not rate-limit the redirect-target domain the way `_request` does (`client.py:471-475` vs `:742-758`) — factor the redirect loop into one helper used by both.
- `PinnedIPAdapter._resolve_first_ip` validates only `infos[0]`; `validate_url` validates *all* answers. Consistent, but the adapter could prefer the first *allowed* address rather than failing on the first, which would make dual-stack hosts with a private AAAA record (common on corp networks) fetchable.

## High-impact feature opportunities

- **HTTPS-correct IP pinning.** The docstring itself says the clean fix is a custom `PoolManager`/`HTTPSConnection` that decouples `server_hostname` (SNI) from the connect address. urllib3 2.x's `HTTPSConnection(server_hostname=...)` kwarg makes this ~30 lines. Why it matters: it turns the documented "may fail on CDNs" caveat into a non-issue for the flagship agent's web tools (#1460 epic).
- **Unique-name / overwrite policy for downloads** exposed as a tool parameter (`overwrite: bool = False`) — pairs with the 🔴 fix and gives the model a safe default.

## Checked and fine

- SSRF validator blocks every literal/encoding tried except the CGNAT range: `127.0.0.1`, `localhost`, `localhost.`, `[::1]`, `[::ffff:127.0.0.1]`, `[::ffff:7f00:1]`, `[::ffff:a9fe:a9fe]`, `0.0.0.0`, `[::]`, `169.254.169.254`, `fe80::/fd00::/fc00::`, RFC1918, `[64:ff9b::7f00:1]` (NAT64), `[2002:7f00:1::1]` (6to4), multicast, `240/4`, `255.255.255.255`, `198.18/15`, `192.0.0.8`, `2001:db8::` — all `ValueError`. Decimal/hex/octal/short forms (`2130706433`, `0x7f000001`, `0177.0.0.1`, `127.1`, `0`) fail to resolve on Windows getaddrinfo → blocked; on glibc they resolve to loopback and would be caught by `_is_blocked_ip`. `file:`/`ftp:` schemes blocked; blocked-port list enforced; userinfo tricks (`http://example.com@127.0.0.1/`, `http://127.0.0.1#@example.com/`) blocked; `http://127.0.0.1:80@example.com/` correctly parses host = `example.com`.
- DNS rebinding: `validate_url` resolves + checks all A/AAAA answers, then `PinnedIPAdapter._resolve_first_ip` re-resolves, re-validates the exact IP through the same `_assert_ip_allowed`, and rewrites the URL to the literal so urllib3 does no third lookup. Cache is populated only after validation. Redirects (`_request:446-477`, `download:742-758`) re-run `validate_url` per hop with `allow_redirects=False`, capped at 5 hops, and the prior streamed response is closed before following.
- Response size: `_consume_body_capped` counts *decoded* bytes from `iter_content`, so gzip bombs are bounded regardless of `Content-Length`; `download()` applies the same counting to disk writes and unlinks on overflow.
- `_sanitize_filename`: strips separators/control chars/NULs, blocks leading dot, Windows device names (incl. `CON.txt`), trailing dots/spaces, caps at 200 chars, never empty. Combined with `save_dir / filename` + the `startswith(save_dir + os.sep)` check, traversal out of `save_dir` is not possible.
- `download_file` (browser_tools): directory goes through both `is_path_allowed(prompt_user=True)` and `is_write_blocked` before download; `~/.ssh`, `/etc` etc. are refused up front.
- Talk RESTART voice command (#3280), `src/gaia/talk/sdk.py:248-251`: matches only when the whole utterance, lower-cased and stripped of trailing `.!?`, equals `"restart"` — "restart the server" does **not** trigger it. It only calls `self.clear_history()` and prints a line; it does not re-initialise the audio pipeline, so there is no old-pipeline leak. Semantics are "clear conversation", not "restart" — naming is arguably misleading but behaviour is safe.

## Hypotheses (unverified)

- **HTTPS fetches to SNI-dependent hosts may fail in practice.** `PinnedIPAdapter` rewrites the URL host to the IP; urllib3/CPython do not send SNI for an IP `server_hostname`, so Cloudflare/most CDNs would answer with a default cert (then `assert_hostname` fails → `SSLError`) or reject the handshake. The class docstring (`client.py:125-139`) concedes this. If true, `fetch_page` is broken for a large share of the public web and this would be 🔴. Not confirmed — needs a live `WebClient().get("https://en.wikipedia.org/wiki/AMD")` in the venv; not run before the session was cut.
- `tavily.py` API-key handling (env-only? logged in debug output? sent in URL vs header?) — not read.
- Audio thread/queue lifecycle (unjoined threads, stop flags), Whisper/Kokoro model-download error handling, sample-rate assumptions — not read.
- `vlm/structured_extraction.py` JSON-recovery paths and `sd/mixin.py` output-path handling — not read; both are the kind of code where CLAUDE.md's no-silent-fallback rule is most often violated (regex "salvage" of malformed JSON returning `{}`).
- `utils/file_watcher.py` observer thread stop/join — not read.
