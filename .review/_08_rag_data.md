# 08 — RAG / data layer review (rag SDK, rag tools, code_index, scratchpad, database)

## Scope covered

Read fully and traced end-to-end:
- `src/gaia/rag/sdk.py` (all 3418 lines: HMAC cache, `_safe_open`, `_get_cache_path`, embedder load,
  `_encode_texts`, PDF/PPTX/DOCX/XLSX/CSV/JSON/text extractors, chunking, `index_document`,
  `remove_document`, `reindex_document`, LRU eviction, query/snapshot paths).
- `src/gaia/security.py` — `PathValidator.__init__` / `is_path_allowed` (to confirm the RAG `allowed_paths` contract).
- `tests/unit/rag/test_pdf_extraction_errors.py` (fixture construction; used to build an offline probe).

Grepped / spot-checked only (NOT read line by line — the integrator should not treat these as reviewed):
- `src/gaia/agents/tools/rag_tools.py` (grep for `RAGConfig(`, `allowed_paths`, `cache_dir`).
- `src/gaia/code_index/sdk.py` (grep: `faiss.write_index`/`faiss.read_index` at 965/1021; no pickle).
- `docs/sdk/sdks/rag.mdx` (grep for cache / chunk_size / allowed_paths claims).
- **Not reviewed at all:** `src/gaia/rag/{app,demo,pdf_utils,pptx_utils}.py`, `src/gaia/code_index/parsers.py`,
  `src/gaia/agents/tools/{code_index_tools,scratchpad_tools}.py`, `src/gaia/scratchpad/service.py`,
  `src/gaia/database/{agent,mixin,sql_safety,testing}.py`, `docs/guides/chat.mdx`, `docs/plans/code-index-review.mdx`.
  The SQL-safety bypass probing requested in the brief was **not done** — needs a follow-up pass.

Tests run:
`.venv\Scripts\python.exe -m pytest tests/unit/rag tests/unit/test_rag_tools.py tests/unit/test_rag_index_status.py tests/unit/test_code_index.py tests/unit/test_code_index_mixin.py tests/unit/test_code_index_parsers.py tests/unit/test_code_index_sdk.py tests/unit/test_scratchpad_service.py tests/unit/test_scratchpad_tools_mixin.py tests/unit/test_database_mixin.py -q -p no:cacheprovider`
→ **337 passed, 0 failed, 21.9s** — full log at `.review/_08_rag_pytest.log`.
(`.review/_08_pytest.log` was clobbered by another reviewer's run and contains unrelated `sounddevice` collection errors from `src/gaia/audio/tests` — ignore it.)

Pickle status (#768): `grep -rn "pickle|joblib|allow_pickle" src/gaia/rag src/gaia/agents/tools/rag_tools.py src/gaia/code_index` → **no matches**. The replacement is complete for RAG. The RAG SDK never persists a FAISS index — the on-disk cache holds only `chunks` / `full_text` / `metadata` JSON plus a `.sig`; vectors are recomputed via Lemonade on every load (`sdk.py:2755-2785`). `faiss.read_index` is used only by `code_index/sdk.py:1021` (not reviewed; FAISS's native format is not a deserialization-RCE vector, but that file is unsigned).

## Findings

### [🔴] `allowed_paths` is bypassed for PDF / PPTX / DOCX / XLSX — RAG indexes any binary document the process can read
- **Where:** `src/gaia/rag/sdk.py:452-457` (`_get_cache_path`), `:722` (`_extract_text_from_pdf`), `:1012` (`_extract_text_from_pptx`), `:1601` (`_extract_text_from_xlsx`), `:1737` (`_extract_text_from_docx`)
- **What:** `RAGConfig.allowed_paths` is the documented security boundary ("Without `allowed_paths`, RAG can index any file the process can read. In production, always set explicit allowed paths." — `docs/sdk/sdks/rag.mdx:994`; default `[CWD]`, `security.py:250-254`). It is enforced only through `_safe_open`, which the text/CSV/JSON extractors use. The PDF, PPTX, XLSX and DOCX extractors hand the raw path to their libraries directly, and the one call on the binary path that *does* go through `_safe_open` (`_get_cache_path`) swallows the resulting `PermissionError` because it is an `OSError` subclass.
- **Failure scenario:** `RAGSDK(RAGConfig(allowed_paths=["C:/proj"])).index_document("C:/Users/x/Documents/payroll.pdf")` → the PDF is parsed, chunked, embedded, and its full text is written into `cache_dir` as `*_extracted.md`. The same call on `payroll.txt` → `Access denied`. Any agent/UI that relies on `allowed_paths` to scope a model-driven `index_document` call is exposed for the four most common document types.
- **Evidence:**
  ```python
  # sdk.py:452-457 — PermissionError is an OSError, so the deny is swallowed
  except (OSError, IOError) as e:
      self.log.warning(f"Cannot read file for cache key: {e}")
      file_hash = hashlib.sha256(str(path).encode()).hexdigest()
      return os.path.join(self.config.cache_dir, f"{file_hash}_notfound.json")
  # sdk.py:722 — no _safe_open / is_path_allowed anywhere on the binary path
  reader = PdfReader(pdf_path)
  ```
  Empirical probe (venv, offline; faiss/Lemonade/VLM stubbed exactly as in `test_pdf_extraction_errors.py`), PDF and TXT placed in a temp dir **outside** `allowed_paths`:
  ```
  is_path_allowed(pdf): False
  PDF outside allowed -> success=False pdf_status=empty error='No extractable text in PDF: secret.pdf\nThe file has 1 page(s'
  TXT outside allowed -> success=False error='Access denied: ...\\outside_a42elxre\\secret.txt is '
  ```
  `pdf_status=empty` proves the PDF's pages were opened and read (blank-page fixture); only the TXT was denied.
- **Fix:** Check once at the top of `index_document` (right after the empty-path check, before `os.path.exists`/`getsize` — those also leak existence/size of denied paths): `if not self.path_validator.is_path_allowed(file_path, prompt_user=False): raise PermissionError(...)`. Narrow `_get_cache_path`'s handler to `except (FileNotFoundError, IsADirectoryError)` so a `PermissionError` propagates. Add a unit test that indexes a `.pdf`, `.pptx`, `.docx`, `.xlsx` outside `allowed_paths` and asserts denial (existing tests only cover extraction inside `tmp_path`).
- **Confidence:** High (code trace + executed probe)
- **Tracked:** none found (`gh issue list --search "allowed_paths rag"`, `"rag path validation pdf"`)

### [🟡] Dropped embeddings silently misalign chunks and vectors — retrieval returns the wrong chunk text
- **Where:** `src/gaia/rag/sdk.py:615-637` (`_encode_texts`, one-by-one fallback); consumers `:2296` (`_create_faiss_index`), `:2955-2960` (`index.add` + `file_to_chunk_indices = range(start, start+len(chunks))`)
- **What:** When a batch returns 0 embeddings the code retries each text individually and, if a single text still returns nothing (or raises), logs a warning and **drops that vector** while keeping the chunk. `all_embeddings` then has fewer rows than `chunks`, but every consumer maps FAISS row *i* to `chunks[i]`.
- **Failure scenario:** Lemonade returns empty `data` for one odd chunk (the condition the `MAX_EMBED_CHARS` comment says happens for over-long input). Every chunk after it is off by one: a query matching chunk *k+1*'s vector returns chunk *k*'s text; per-file index ranges point past the end of the per-file FAISS index. If *all* rows of a doc are dropped, `np.array([])` has shape `(0,)` and `_create_faiss_index` raises `IndexError: tuple index out of range` — an unactionable traceback. The `idx < len(file_chunks)` guard at `:3107` hides rather than surfaces the corruption.
- **Evidence:**
  ```python
  # sdk.py:628-637
  else:
      self.log.warning("   ⚠️  Single text (%d chars) returned no embedding, skipping", len(single_text))
  except Exception as e:
      self.log.warning(f"   ⚠️  Single embedding failed: {e}")
  ...
  all_embeddings.extend(batch_embeddings)
  # sdk.py:2955-2960 — assumes one vector per chunk
  self.index.add(file_embeddings.astype("float32"))
  self.file_to_chunk_indices[file_path] = file_chunk_indices
  ```
- **Fix:** Fail loudly (project rule): if any text yields no embedding, raise `RuntimeError("Embedding model returned no vector for chunk N of <file> (len=…) — …")`; at minimum assert `embeddings.shape[0] == len(texts)` before returning. `code_index/sdk.py` already has `_encode_texts_with_sync` returning `(vecs, synced_chunks)` — adopt the same contract here.
- **Confidence:** High on the code path; the trigger (Lemonade returning empty data) is documented in the code's own comments but was not reproduced live.
- **Tracked:** none found

### [🟡] Chunks are embedded on only their first 1200 chars while the default chunk is ~2000 chars — the tail of every full-size chunk is invisible to retrieval
- **Where:** `src/gaia/rag/sdk.py:552-560` (`_encode_texts`), `:92` (`RAGConfig.chunk_size = 500`), `:2118` (`para_tokens = len(para) // 4`)
- **What:** Chunking targets `chunk_size` tokens at 4 chars/token → up to ~2000 chars per chunk (docs recommend 768–1024 tokens = 3–4K chars). `_encode_texts` then truncates every text to 1200 chars before embedding, so a chunk's vector represents roughly its first 60% (30–40% at doc-recommended sizes). The stored/returned chunk text is the full chunk, so the mismatch is invisible.
- **Failure scenario:** A fact in the last 800 chars of a 2000-char chunk never influences that chunk's vector; the query retrieves whatever chunk mentions it *earlier*, or nothing. With `chunk_size=1024` per `docs/sdk/sdks/rag.mdx:896` ("Large chunks preserve argument flow") 70% of each chunk is unembedded. Only an `info`-level "✂️ Truncated N/M chunks" log hints at it.
- **Evidence:**
  ```python
  # sdk.py:552-556
  MAX_EMBED_CHARS = 1200
  ...
  if len(t) > MAX_EMBED_CHARS:
      safe_texts.append(t[:MAX_EMBED_CHARS])
  ```
- **Fix:** Cap `chunk_size` to the embedder window (≈300 tokens for a 512-token model) and warn at `RAGConfig` construction when `chunk_size*4 > MAX_EMBED_CHARS`; or embed several windows per chunk and max-pool at search time. Update the `rag.mdx` chunk-size guidance — the 768/1024 recommendations actively degrade retrieval today.
- **Confidence:** High
- **Tracked:** none found (nearest: #786 VLM blocks split across chunks — different defect)

### [🟡] `remove_document` re-embeds the entire remaining corpus through Lemonade while holding the state lock — every LRU eviction is a full re-index that blocks all queries
- **Where:** `src/gaia/rag/sdk.py:2327` (lock), `:2370-2374` (`_create_vector_index(new_chunks)`); callers `_evict_lru_document:2463`, `_check_memory_limits:2478-2500`, `reindex_document:2433`
- **What:** Removing one document rebuilds the global FAISS index via `_create_vector_index`, which calls `_encode_texts` on *all* remaining chunks (Lemonade round-trips, 25 chunks per call), even though `self.file_embeddings[path]` already holds every file's vectors. It runs entirely inside `with self._state_lock:` (an RLock) and `_snapshot_query_state` needs that lock, so every `query()` stalls until re-embedding finishes.
- **Failure scenario:** 100 docs / 10 000 chunks (the defaults). Indexing doc #101 → `_check_memory_limits` → `_evict_lru_document` → `remove_document` → ~9 900 chunks re-embedded (~400 Lemonade calls) under the lock; concurrent chat turns hang. If Lemonade hiccups mid-way the `except Exception: return False` (`:2414-2416`) leaves the old state in place and the memory limit is never met.
- **Evidence:**
  ```python
  # sdk.py:2372
  new_index, new_chunks = self._create_vector_index(new_chunks)
  # sdk.py:2325-2327 — whole method body is under the lock
  with self._state_lock:
      if file_path not in self.indexed_files:
  ```
- **Fix:** Rebuild from cached vectors: `np.concatenate([self.file_embeddings[p] for p in remaining_in_chunk_order])` → `_create_faiss_index`; no Lemonade call, milliseconds. `file_embeddings` is populated on every index path already (`:2798`, `:2960`).
- **Confidence:** High
- **Tracked:** none found

### [🟡] HMAC mismatch (tampered / corrupt cache) is silently deleted and re-indexed — no user-visible signal
- **Where:** `src/gaia/rag/sdk.py:2833-2843` (`index_document` cache branch); `_verify_and_load_cache:408-411`; `_get_hmac_key:340`
- **What:** `_verify_and_load_cache` correctly uses `hmac.compare_digest` and raises `ValueError("Cache integrity check failed — file may have been tampered with")`. The caller's `except Exception` turns *every* cache failure (missing `.sig`, bad HMAC, bad JSON, `KeyError`) into `log.warning("Cache load failed …, reindexing")`, deletes both cache files and re-indexes; the returned `stats` carries no flag. The tampering #768 introduced HMAC to detect is thus erased without ever being reported, indistinguishable from "cache stale".
- **Failure scenario:** Anyone with write access to `cache_dir` (default is the **relative** `.gaia` under CWD, i.e. any project directory) edits `*.json`; on the next index GAIA quietly overwrites the evidence. Conversely a 0-byte `hmac.key` (`read_bytes()` at `:340` has no length check) makes every cache fail verification forever, each time silently re-indexing.
- **Evidence:**
  ```python
  # sdk.py:2833-2841
  except Exception as e:
      self.log.warning(f"Cache load failed: {e}, reindexing")
      ...
      try: os.remove(cache_path)
      except OSError: pass
  ```
- **Fix:** Catch the integrity `ValueError` separately: log at `error`, set `stats["cache_integrity_failed"] = True`, rename to `*.tampered` instead of deleting. Validate `len(key) == 32` on load and fail loudly otherwise.
- **Confidence:** High
- **Tracked:** none found

### [🟡] Cache key ignores `chunk_size` / `chunk_overlap` / chunking mode — tuning `RAGConfig` silently reuses chunks made with the old settings
- **Where:** `src/gaia/rag/sdk.py:445-448` (`_get_cache_path`); same key at `:3319`
- **What:** The key is `sha256(path)[:16] + "_" + sha256(content)[:32]` and the cached payload is the *chunk list*. Nothing about the chunker (`chunk_size`, `chunk_overlap`, `use_llm_chunking`, VLM availability) is in the key or checked on load, so a user who tunes `chunk_size` per the docs' tuning guide (`rag.mdx:205-290`, `:1040-1082`) and re-indexes gets the old chunks back with `from_cache=True`.
- **Failure scenario:** Index with defaults, then `RAGConfig(chunk_size=256)` → `index_document()` reports success with `num_chunks` unchanged; retrieval does not change; the user concludes the setting does nothing.
- **Evidence:**
  ```python
  # sdk.py:447-448
  path_hash = hashlib.sha256(str(path).encode()).hexdigest()[:16]
  cache_key = f"{path_hash}_{content_hash[:32]}"
  ```
- **Fix:** Fold a chunker fingerprint into the key (`sha256(f"{chunk_size}:{chunk_overlap}:{use_llm_chunking}:v1")[:8]`), or store the config in `metadata` and reject on mismatch with `stats["cache_config_mismatch"]`.
- **Confidence:** High
- **Tracked:** none found

### [🟢] `remove_document` swallows all exceptions and returns `False` (no-silent-fallback violation)
- **Where:** `src/gaia/rag/sdk.py:2414-2416` (`remove_document`)
- **What:** `except Exception as e: self.log.error(...); return False`. Callers (`reindex_document:2433`, `_evict_lru_document:2463`) only see a boolean, so a Lemonade outage during the rebuild is reported as "Failed to remove old version" with the cause only in the log.
- **Failure scenario:** `reindex_document()` on a changed file while Lemonade is restarting → `{"success": False, "error": "Failed to remove old version"}`; the user has no idea it was an embedding failure and retries blindly.
- **Evidence:** `except Exception as e:\n    self.log.error(f"Failed to remove document {file_path}: {e}")\n    return False`
- **Fix:** Re-raise with context (`raise RuntimeError(f"Failed to rebuild index after removing {file_path}: {e}") from e`) or put the cause into the returned dict. With the re-embed removed (finding above) the only remaining failure mode is a numpy/faiss error, which should never be swallowed.
- **Confidence:** High
- **Tracked:** none found

### [🟢] `_llm_based_chunking` applies split positions from a 2000-char preview to a 6000-char segment
- **Where:** `src/gaia/rag/sdk.py:1362-1420`
- **What:** `segment_size = chunk_size * 4 * 3` (6000 chars by default) but the prompt only shows `segment[:2000]`; the LLM's positions are then applied to the full `segment`, so the trailing 4000 chars always end up in one oversized "remaining" chunk. Also `position = segment_end - overlap*4` loops forever if `chunk_overlap*4 >= segment_size` (i.e. `chunk_overlap >= 3*chunk_size`, a misconfiguration but unguarded).
- **Failure scenario:** `use_llm_chunking=True` (opt-in, off by default) → 2/3 of every segment is chunked as one ~4000-char blob, which the 1200-char embed truncation then makes mostly unsearchable.
- **Evidence:** `segment_preview = segment[:2000]` … `chunk = segment[last_pos:split_pos]` … `position = segment_end - (overlap * 4)`
- **Fix:** Send the whole segment (or set `segment_size = 2000`), and guard `if overlap*4 >= segment_size: raise ValueError(...)`.
- **Confidence:** High
- **Tracked:** none found

### [🟢] Encrypted PDFs with an empty user password are refused even though pypdf can open them
- **Where:** `src/gaia/rag/sdk.py:738-752` (`_extract_text_from_pdf`)
- **What:** `if getattr(reader, "is_encrypted", False): raise EncryptedPDFError`. Many "protected" PDFs (owner-password only, e.g. print/copy-restricted reports) decrypt with `reader.decrypt("")`; those are currently bounced with a "remove the password with qpdf" message the user cannot act on because there is no password.
- **Failure scenario:** Index a copy-protected vendor datasheet → `pdf_status=encrypted`, user has no password to remove.
- **Evidence:** `if getattr(reader, "is_encrypted", False):` … `raise EncryptedPDFError(msg)`
- **Fix:** Try `reader.decrypt("")` first (needs `cryptography`/`pycryptodome` for AES); raise only if that fails.
- **Confidence:** Medium (behaviour of pypdf on owner-only PDFs not exercised in this session)
- **Tracked:** none found

## Test gaps

- **No test for the `allowed_paths` boundary on binary documents.** `tests/unit/rag/test_pdf_extraction_errors.py` builds every fixture under `tmp_path` and points `cache_dir` there; the default `allowed_paths=[CWD]` never comes into play, so the 🔴 bypass above is invisible to the suite. A four-line parametrized test (pdf/pptx/docx/xlsx outside `allowed_paths` → denied) would have caught it.
- **`_encode_texts` has no unit test in the RAG suite** (`grep -rn "_encode_texts\|MAX_EMBED" tests/unit tests/test_rag.py` → only `test_code_index_sdk.py` hits). The row-count invariant, the 1200-char truncation, the one-by-one fallback, and the empty-response path are all untested.
- **HMAC cache round-trip is untested** (`grep -rn "_verify_and_load_cache\|hmac" tests/unit tests/test_rag.py` → only a connectors test). Nothing asserts that a tampered `*.json` is rejected, that a missing `.sig` is rejected, or what `index_document` does afterwards.
- **`remove_document` / LRU eviction has no test that asserts the index is rebuilt without re-embedding**, so the re-embed regression can land silently. `test_rag_index_status.py` (7 mock hits) checks status reporting, not the rebuild.
- **Mocks that only prove a call happened:** `tests/unit/test_code_index_sdk.py:78,114,150` patch `_encode_texts_with_sync` and assert it was called; none assert the shape/ordering of the returned vectors against the chunk list. `tests/unit/rag/test_pdf_extraction_errors.py` stubs `VLMClient` and `AgentSDK` entirely (`MagicMock`) — fine for the error-path tests it targets, but it means the happy path (`index_document` → embeddings → FAISS) is not exercised anywhere in the unit tier.
- Chunker tests: none found for `_split_text_into_chunks` overlap arithmetic (`overlap_actual`, `_get_last_n_tokens`) or `_split_into_sentences`. #786 (VLM block splitting) is open precisely because this path is untested.
- Cache invalidation: no test that editing a file (same size, same mtime) produces a new cache key, nor that a changed `RAGConfig.chunk_size` does *not* reuse the cache (it does today — finding above).

## Documentation gaps

- `docs/sdk/sdks/rag.mdx:994` promises `allowed_paths` as the production security control; the code enforces it only for text-like files (🔴 above). Until fixed the doc should say so explicitly.
- `docs/sdk/sdks/rag.mdx:205-290`, `:871-945`, `:1040-1082` recommend `chunk_size` values of 512–1024 tokens with quality rationale ("Large chunks preserve argument flow"). With `MAX_EMBED_CHARS = 1200` every value above ~300 tokens silently degrades retrieval; the guide contradicts the implementation.
- `docs/sdk/sdks/rag.mdx:1082` "`cache_dir="./my_rag_cache"  # Index persisted here`" and `:285` "`# Where to store the index`" — the cache stores chunks and extracted text, **not** the index (`:296` says this correctly two paragraphs later). Also nothing tells the user that changing `chunk_size` does not invalidate the cache.
- Nothing in `rag.mdx` documents the HMAC-signed cache introduced in #768 (where the key lives — `~/.gaia/cache/hmac.key` — that it is per-user, that a cache dir cannot be moved between users, or what happens on a signature failure).
- Not cross-checked (out of budget): `docs/guides/chat.mdx` RAG sections and `docs/plans/code-index-review.mdx`.

## Improvement opportunities

- Default `cache_dir=".gaia"` is **CWD-relative** (`sdk.py:96`) while the HMAC key is under `~/.gaia/cache`; the cache silently forks per working directory and lands inside user repos. Default to `~/.gaia/cache/rag/` — why: predictable location, one cache per user, no stray dirs in projects.
- `_get_hmac_key` swallows `chmod` failures (`:347-349`) and is a no-op on Windows anyway; consider `os.open(..., 0o600)` at creation and log a warning if the resulting mode is wider — why: the key is the only thing standing between "signed" and "unsigned" cache.
- `_check_memory_limits()` is called **outside** the lock on the cache-hit path (`:2819`) but inside it on the fresh-index path (`:2994`); make both consistent — why: `len(self.chunks)` read races with a concurrent indexer and can evict one document too many/few.
- `index_document` returns `{"success": False, "error": …}` for most failures but *raises* `ValueError` for an empty path; the `Raises:` docstring also claims it raises for a missing file, which it does not (`Raises:` at `:2581`, the `return stats` at `:2605-2611`). Pick one contract — why: callers currently have to handle both.
- `_extract_text_from_csv` re-runs the whole parse per encoding attempt and would double-append `Columns:` if a `UnicodeDecodeError` arrived after `text_parts` was populated (it cannot today because decode happens before the loop body, but the structure invites the bug) — hoist decoding out of the parsing loop.
- `stats["total_indexed_files"]` / `total_chunks` are computed before the lock at the top of `index_document` and again at the end; drop the first — why: they are stale by the time the dict is returned.

## High-impact feature opportunities

- **Persist embeddings alongside chunks in the signed cache** (numpy `.npy` or float16 list keyed by `embedding_model`). Today every process start re-embeds every cached document through Lemonade (`:2777-2781`) — for a 100-doc corpus that is minutes of GPU time before the first query. The HMAC machinery already exists; adding a `vectors` file signed with the same key and keyed on `embedding_model` makes cold start near-instant. Roughly a day: `_save_cache`/`_verify_and_load_cache` gain a second payload, `index_document` skips `_encode_texts` on a hit.
- **Incremental removal without index rebuild** — switch `IndexFlatL2` to `IndexIDMap2` over `IndexFlatL2` so `remove_ids` works; combined with the cached-vector rebuild this removes the last O(N) path in `remove_document`. Small change, unlocks safe LRU eviction on large corpora.
- **Embedder-aware chunking** — read the embedder's context length from Lemonade's model info instead of a hard-coded 1200 chars, and size chunks to it. This is the single biggest retrieval-quality lever visible in this file and would also let the docs' tuning guide be honest.

## Checked and fine

- `hmac.compare_digest` is used for the signature comparison (`:408`); signature is over the exact bytes written (`json.dumps(...).encode()` at `:366-372`, read back with `open(cache_path, "rb")` at `:397`), so no canonicalisation mismatch.
- HMAC key: 32 random bytes from `secrets.token_bytes`, per-user (`Path.home()`), created once, cached in memory per instance. Not deterministic, not derived from anything guessable.
- `_safe_open` (`:275-320`): validates against `PathValidator` with `prompt_user=False` (no `input()` hang in servers), uses `O_NOFOLLOW` where available, `fstat` + `S_ISREG` check, closes the fd on any failure. Correct for the paths that use it.
- `_get_cache_path` hashes **content** (SHA-256, streamed) plus the absolute path, so same-size / same-mtime edits do invalidate the cache (mtime resolution is not relied on).
- Encrypted / 0-byte / non-PDF-renamed-`.pdf` handling (#784): 0-byte is caught before extraction (`:2593-2600`, `stats["error"]="File is empty"`); pypdf's `PdfReadError` (covers `EmptyFileError`/`PdfStreamError` for garbage bytes) → `CorruptedPDFError`; `is_encrypted` → `EncryptedPDFError`; all-blank pages → `EmptyPDFError`. Each carries a multi-line remediation message and a stable `status` that `index_document` copies into `stats["pdf_status"]` (`:3034`). Covered by `tests/unit/rag/test_pdf_extraction_errors.py` (13 tests, passing). Nothing is silently indexed as zero chunks.
- Thread safety (#746): every mutation of `chunks` / `chunk_to_file` / `file_to_chunk_indices` / `index` / `indexed_files` / LRU maps that I traced happens under `self._state_lock` (index_document `:2659`, `:2728`, `:2881`, `:2978`; remove_document `:2327`; reindex_document `:2429`; `_retrieve_chunks_from_file` `:3055`; `_snapshot_query_state` `:2306`). Query paths snapshot under the lock and do the embedding/search outside it; `_retrieve_chunks_from_file` deliberately keeps a locally rebuilt per-file index query-local (`:3096-3103`). Double-checked-locking pattern in `index_document` (pre-flight capacity check, then re-check under lock after extraction) is correct. The RLock makes `reindex_document → remove_document → index_document` re-entrant.
- FAISS search results are bounds-checked against the snapshot (`:3190`), so a concurrent shrink cannot index past the chunk list.
- LRU ordering uses a monotonic counter, not `time.time()` (`:187-191`), avoiding the Windows 15 ms clock-resolution tie problem.
- PPTX/DOCX zip-bomb guard (500 MB uncompressed) runs before python-pptx/python-docx parse (`:983`, `:1697`).
- `#768` pickle removal: no `pickle`, `joblib`, or `np.load(allow_pickle=…)` anywhere under `src/gaia/rag`, `rag_tools.py`, or `src/gaia/code_index`.

## Hypotheses (unverified)

- `rag_tools.py` (2051 lines, not read) — whether the mixin passes the agent's `allowed_paths` into `RAGConfig` at all: grep found only a comment at `:1258` ("Validate path with ChatAgent's internal logic (which uses allowed_paths)") and no `RAGConfig(` construction with `allowed_paths=`. If the mixin validates upstream, the 🔴 is mitigated for the agent path but still real for direct SDK users and for any tool that calls `index_document` without pre-validation.
- `code_index/sdk.py` persists a raw FAISS file (`faiss.write_index` at `:965`, `read_index` at `:1021`) with no signature; stale-index detection and what happens when the on-disk index and its metadata disagree were not examined.
- `scratchpad/service.py` and `database/sql_safety.py` were not read; the brief's SQL-injection / deny-list bypass probing (comments, CTEs, `PRAGMA`, `ATTACH`, multi-statement, unicode) still needs to be done.
- `_extract_text_from_pptx` COM fast path (`convert_pptx_to_pdf`) passes `str(Path(pptx_path).resolve())` to PowerPoint — resolve follows symlinks, another place where `_safe_open`'s symlink protection does not apply (consequence of the 🔴 above).
