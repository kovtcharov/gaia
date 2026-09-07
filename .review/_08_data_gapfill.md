# 08 — Data-layer gap-fill review (database, scratchpad, code_index, rag_tools path check, tests)

Follow-up to `.review/_08_rag_data.md` (which owns `src/gaia/rag/sdk.py`). Written incrementally, one
section per module. All probes run with `.venv\Scripts\python.exe` (Python 3.13.11, SQLite 3.50.4) at
commit 211f08c5.

## Scope covered

Read fully and traced end to end:
- `src/gaia/database/sql_safety.py` (378 lines), `mixin.py` (428), `agent.py` (258), `testing.py` (64), `__init__.py`.
- `tests/unit/test_database_mixin.py` (surveyed every test name + the `query_readonly` block; `tests/unit/test_sql_safety.py` and
  `tests/integration/test_database_agent.py` exist but were only located, not read).
- `docs/sdk/mixins/database-mixin.mdx` (grepped for the security claims; matched against code).

(Sections for scratchpad, code_index, rag_tools and the test files follow below as each module is finished.)

## Module 1 — `src/gaia/database/` (SQL guard)

### Verdict: it is a real allow-list, not a regex deny-list — no bypass found

The guard is SQLite's own `set_authorizer` callback (`sql_safety.py:347-361`): every action code except
`SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION`, `SQLITE_RECURSIVE` and five schema-inspection pragmas returns
`SQLITE_DENY`. Because SQLite consults the authorizer at *prepare* time on the parsed statement, comments,
case, string literals and CTE wrapping are irrelevant — there is no text matching anywhere in the guard.
`query_readonly()` (`mixin.py:211-216`) runs the statement with `Connection.execute(sql, params)`, which the
Python driver limits to a single statement, so `;`-chaining is rejected before the second statement is ever
parsed. Denials are classified by the authorizer's own `denied` list, not by matching SQLite's message
(`mixin.py:219-220`), so a table literally named `"not authorized"` cannot be mislabelled (tested at
`test_database_mixin.py:566`).

Every LLM-facing write tool (`db_insert`/`db_update`/`db_delete`, `agent.py:131-220`) validates the table with
`validate_identifier` (fullmatch `[A-Za-z_][A-Za-z0-9_]*`, `sql_safety.py:69`), validates every data key the same
way (`mixin.py:29-30`), and builds the WHERE from a closed operator allow-list with all values bound
(`build_where`, `sql_safety.py:175-285`; AND-only, `OR` unrepresentable). `all_rows` is parsed by an enumerated
spelling table (`coerce_bool`, `:121-147`) so `"no"` cannot authorise a mass delete. The only interpolated
identifier on a read path is `PRAGMA table_info({table})` in `db_schema` (`agent.py:246-248`), validated first.

**Does LLM SQL ever reach the DB unguarded?** No. `db_query` → `query_readonly`; the unrestricted `query()` and
`execute()` (`executescript`, `mixin.py:407`) are Python-API only, and every in-repo user of `DatabaseMixin`
(`daemon/scheduler/clock.py`, `filesystem/index.py`, `scratchpad/service.py`, `web/tavily.py`, `hub/agents/email/*`)
passes literal SQL — see the scratchpad section below for its `create_table` path.

### Executed probe (43 payloads, state diff before/after)

```
comment+multi   SELECT 1 /* */; DROP TABLE x        -> ProgrammingError: You can only execute one statement at a time.
line comment    SELECT 1 -- x\n; DROP TABLE x       -> ProgrammingError (same)
string trick    SELECT 'x'; DROP TABLE x            -> ProgrammingError (same)
NULL byte       SELECT 1\x00; DROP TABLE x          -> ProgrammingError: the query contains a null character
CTE delete      WITH y AS (DELETE FROM t RETURNING *) SELECT * FROM y -> OperationalError: near "DELETE": syntax error  (SQLite CTEs cannot contain DML)
CTE insert      WITH y AS (INSERT ... RETURNING *)  -> OperationalError: syntax error
mixed case      DeLeTe FROM t                       -> PermissionError: Read-only query blocked by SQLite
fullwidth       ｄｅｌｅｔｅ FROM t                  -> OperationalError: near "ｄｅｌｅｔｅ": syntax error (not a keyword to SQLite)
PRAGMA table_info(t)                                -> OK (allow-listed)
PRAGMA writable_schema=1 / journal_mode=OFF / foreign_keys -> PermissionError
SELECT * FROM pragma_table_info('t')                -> OK (table-valued form of an allow-listed pragma)
SELECT * FROM pragma_database_list                  -> PermissionError (tvf of a non-allow-listed pragma is still gated)
ATTACH DATABASE ':memory:' / 'C:/Windows/Temp/evil_probe.db' -> PermissionError (file not created)
DETACH, BEGIN, SAVEPOINT, VACUUM, ANALYZE, ALTER, DROP, CREATE TABLE/TEMP TABLE/VIEW/VIRTUAL TABLE/TRIGGER -> PermissionError
INSERT..SELECT, REPLACE INTO, UPSERT (ON CONFLICT DO UPDATE) -> PermissionError
EXPLAIN DELETE / EXPLAIN QUERY PLAN DELETE          -> PermissionError (authorizer still runs on the inner statement)
UPDATE sqlite_master SET ...                        -> OperationalError: table sqlite_master may not be modified
SELECT load_extension('C:/evil.dll')                -> OperationalError: not authorized (extension loading never enabled)
SELECT writefile(...)/readfile(...)                 -> OperationalError: no such function (no fileio extension)
SELECT sql FROM sqlite_master                       -> OK (schema read; intended)
REINDEX / REINDEX t / REINDEX ix (file DB with an index) -> PermissionError (authorizer action 27 fires per index;
                                                       bare REINDEX on a DB with *no* indexes is a no-op and returns [])
STATE before ([t, x], 2 rows) after ([t, x], 2 rows) UNCHANGED ; evil files exist: False False
```

## Findings (database)

### [🟡] `db_query` has no execution timeout, row cap, or memory cap — one LLM tool call can hang the agent or flood its context
- **Where:** `src/gaia/database/mixin.py:211-216` (`query_readonly`), `src/gaia/database/agent.py:126-128` (`db_query`)
- **What:** The read-only window blocks *writes* but places no bound on *reads*. There is no `set_progress_handler`, no
  `interrupt()`, no `LIMIT` injection and no result-size cap (`grep -rn "set_progress_handler\|interrupt()\|LIMIT\|max_rows" src/gaia/database/` → none). `db_query` returns every row into the tool result that goes back into the LLM prompt.
- **Failure scenario:** A model emits `WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c`
  (a common hallucinated "count" pattern with the bound forgotten) → the agent thread spins forever inside SQLite with the
  authorizer armed; a concurrent writer on the same connection is denied for the duration (`sql_safety.py:337-338`). Or
  `SELECT * FROM big_table` → hundreds of thousands of dicts serialised into the tool result and the context window.
- **Evidence:** executed in the venv:
  ```
  3M-row recursive CTE -> [{'n': 3000000}] in 0.3s, no interruption
  SELECT length(randomblob(200000000)) -> [{'n': 200000000}] (200 MB allocated on demand)
  300k rows returned to caller: len=300000 (no row cap)
  ```
- **Fix:** In `query_readonly`, install `conn.set_progress_handler(_abort, N)` with a wall-clock deadline (e.g. 10 s, tunable)
  inside the window and remove it on exit; in `db_query`, `fetchmany(max_rows + 1)` and annotate truncation
  (`"rows": rows[:max_rows], "truncated": True`) so the model learns to add `LIMIT`. Add tests for both.
- **Confidence:** High (probe above)
- **Tracked:** none found (`gh issue list --search "db_query"`, `"recursive CTE"`, `"query_readonly"` → 0 results)

### [🟢] `db_query` does not type-check `sql`, so a non-string escapes as a raw `TypeError`
- **Where:** `src/gaia/database/agent.py:109-128`
- **What:** If a model passes `sql` as a list/dict, `sqlite3` raises `TypeError` (not `DatabaseError`), which escapes
  `query_readonly`'s `except sqlite3.DatabaseError` unclassified. Not a safety issue — nothing executes — but the
  error is a raw Python `TypeError` rather than the actionable message every other parameter gets.
- **Failure scenario:** `db_query(sql=["SELECT 1"])` → `TypeError: execute() argument 1 must be str, not list`.
- **Fix:** `if not isinstance(sql, str): raise ValueError("db_query: 'sql' must be a string ...")`.
- **Confidence:** High
- **Tracked:** none found

## Checked and fine (database)
- Authorizer covers `SQLITE_ATTACH`, `SQLITE_DETACH`, `SQLITE_PRAGMA` (allow-list), DDL, DML, `TRANSACTION`, `SAVEPOINT`,
  `REINDEX`, `ANALYZE`, `CREATE_VTABLE`; `SQLITE_FUNCTION` is allowed but `load_extension` is disabled at the connection level.
- Table-valued pragma functions (`pragma_xxx`) go through the same `SQLITE_PRAGMA` action → gated by the same allow-list.
- `set_authorizer(None)` pre-3.11 bricking is handled (`_SET_AUTHORIZER_NONE_DETACHES`, `sql_safety.py:311-324`).
- Re-entrant windows keyed by `id(conn)` hold the connection object so ids can't be recycled (`:366-368`); depth-counted disarm.
- `update()` guards `__set_` param collisions (`mixin.py:305-311`); `_require_where` rejects blank fragments so `DELETE FROM t WHERE` can't become a full-table delete by accident.
- `execute()` refuses to run inside `transaction()` (`mixin.py:402-406`) because `executescript` auto-commits.
- Docs (`docs/sdk/mixins/database-mixin.mdx:44,54,79,171,269`) match the code: `query()` unrestricted / `db_query` enforced by SQLite; AND-only conditions; `PermissionError` on block.
- Tests assert real behaviour, not mock calls: `test_query_readonly_blocks_writes` re-queries the table afterwards to prove no mutation (`:524-529`); `test_query_readonly_restores_authorizer` proves the window disarms; `test_query_readonly_works_inside_transaction`; identifier-rejection tests are parametrised over bad names.

## Test gaps (database)
- No test for the table-valued `pragma_*` form (`SELECT * FROM pragma_database_list`) or for `EXPLAIN <write>` — both are
  blocked today (probe above) but nothing pins that.
- No test for CTE/`WITH RECURSIVE`, `REINDEX`, `REPLACE INTO`/UPSERT, or multi-statement `;` payloads against `query_readonly`.
- No timeout/row-cap test (there is nothing to test yet — see the 🟡).

## Module 2 — `src/gaia/scratchpad/service.py` + `src/gaia/agents/tools/scratchpad_tools.py`

Read fully: `service.py` (568 lines), `scratchpad_tools.py` (291), `tests/unit/test_scratchpad_service.py` and
`tests/unit/test_scratchpad_tools_mixin.py` (test names + mock usage), the single instantiation site
`hub/agents/chat/python/gaia_agent_chat/agent.py:344` (`ScratchpadService(db_path=config.scratchpad_db_path)`).

**There is no CSV/XLSX import path in the scratchpad.** The only ingress is `insert_data(table_name, data: JSON string)`
(`scratchpad_tools.py:90-167`): 10 MB payload cap before `json.loads` (`:120-127`), 10 000 rows/call (`:145-150`),
per-row dict check; the service then re-validates every key against `[A-Za-z_][A-Za-z0-9_]*` (`service.py:194-202`)
and binds values via `DatabaseMixin.insert`. No file path is ever read, so `allowed_paths` is not a concern here.

**Identifier handling.** Table names are not quoted — they are *rewritten*: `_sanitize_name` (`service.py:409-415`)
replaces every non-`[A-Za-z0-9_]` char with `_`, prefixes `t_` for a leading digit, truncates to 64, and the result
is prefixed `scratch_` before interpolation. Column DDL for `create_table` is a text guard (`_validate_columns`,
`:441-532`): hard-deny `; -- /* */`, balanced parens, per-column `<ident> <TYPE>` shape check, constraint text passed
through. This matters because `create_table` runs through `DatabaseMixin.execute` → `executescript` (`:157`), which
*would* run a second statement if one got through.

### Executed probe (service API, temp file DB)

```
1  create_table('t"; DROP TABLE x; --', "a TEXT")   -> OK "Table 't___DROP_TABLE_x____' created"  (name rewritten, no error; table scratch_t___DROP_TABLE_x____ exists)
2  create_table("u", "a TEXT); DROP TABLE scratch_t; --")  -> ValueError: contains forbidden token ';'
3  create_table("u2", "a TEXT, b TEXT ) WITHOUT ROWID, (c TEXT")  -> OperationalError: near "(": syntax error  (balanced-paren trick reaches SQLite but cannot form a 2nd statement)
4  create_table("u3", "a TEXT DEFAULT 'x;y'")       -> ValueError: forbidden token ';'  (false positive on a legit literal)
5  create_table("u4", "a TEXT DEFAULT (randomblob(10))")  -> OK  (arbitrary constraint expressions pass through)
6  query_data("SELECT * FROM pragma_database_list")  -> OK [{'seq': 0, 'name': 'main', 'file': 'C:\\...\\s.db'}]   <-- PRAGMA keyword bypassed via table-valued form
7  query_data("SELECT name FROM pragma_table_info('scratch_t')") -> OK
8  query_data("select 1; drop table scratch_t")      -> ValueError: disallowed keyword: DROP
9  query_data("WITH x AS (SELECT 1 v) SELECT * FROM x") -> ValueError: Only SELECT queries are allowed   <-- legit CTE rejected
10 query_data("SELECT * FROM scratch_t -- drop nothing") -> ValueError: disallowed keyword: DROP  <-- comment false positive
11 query_data("SELECT count(*) FROM (WITH RECURSIVE c(x) AS (... x<2000000) SELECT x FROM c)") -> OK [{'c': 2000000}]  (CTE-in-subquery allowed; unbounded = hang)
12 query_data("SELECT sql FROM sqlite_master")       -> OK (reads own schema; harmless, own DB file)
13 query_data("SELECT load_extension('C:/evil.dll')") -> OperationalError: not authorized
14 query_data("SELECT a AS [drop] FROM scratch_t")   -> ValueError: DROP (bracket-quoted identifiers are not stripped)
16 query_data("VALUES(1)") / 17 "EXPLAIN QUERY PLAN SELECT ..." -> rejected (must start with SELECT)
tables after: ['scratch_t', 'scratch_t___DROP_TABLE_x____', 'scratch_u4'] rows: 2   -> no write bypass found
```
No write bypass: `query_data` runs through `DatabaseMixin.query` → single-statement `Connection.execute`, and a
statement that starts with `SELECT` cannot mutate in SQLite. The weaknesses are all on the read side and in
what the guard *rejects*.

## Findings (scratchpad)

### [🟡] `ScratchpadService.query_data` re-implements a weaker text deny-list instead of the parent class's `query_readonly`, so it rejects valid analysis SQL and misses the DoS/pragma cases the authorizer handles
- **Where:** `src/gaia/scratchpad/service.py:252-288` (`query_data`); `src/gaia/agents/tools/scratchpad_tools.py:193,217-222` (tool)
- **What:** `query_data` checks `upper.startswith("SELECT")`, strips quoted literals with a regex, and scans for ten
  keywords, then calls the **unrestricted** `self.query()` (`:288`). `DatabaseMixin.query_readonly` — the SQLite
  authorizer that `db_query` uses — is inherited by this very class and unused. The text scan is both too strict and
  too loose: it refuses `WITH ... SELECT` (probe 9) and any SELECT whose comment or bracket-identifier mentions a
  keyword (probes 10, 14), yet lets `pragma_*` table functions through (probe 6) and imposes no time/row bound
  (probe 11; the tool then formats **every** row into the LLM result, `scratchpad_tools.py:217-222`). CLAUDE.md
  "Code Reuse and Base Classes" is the convention this violates; the mixin's own docstring advertises
  "Supports ... subqueries" but the model's natural CTE spelling is refused with a misleading "Only SELECT queries are allowed".
- **Failure scenario:** (a) Model writes `WITH monthly AS (SELECT strftime('%Y-%m', date) m, SUM(amount) s FROM scratch_tx GROUP BY m) SELECT * FROM monthly` → "Only SELECT queries are allowed" → model retries/loops. (b) Model emits an unbounded recursive CTE inside a subquery, or `SELECT * FROM scratch_big` on a 1 M-row table → agent hangs / 1 M formatted rows pushed into context.
- **Evidence:** probes 6, 9, 10, 11, 14 above; `service.py:288 return self.query(normalized)`; `mixin.py:176` `query_readonly` unused in `service.py` (`grep -n query_readonly src/gaia/scratchpad/service.py` → none).
- **Fix:** Replace the keyword scan with `self.query_readonly(normalized)` (keeps the `scratch_`-only DB file as the isolation boundary; the authorizer already blocks writes/DDL/ATTACH/non-allow-listed pragmas), drop the `startswith("SELECT")` check or widen it to `SELECT|WITH`, and cap rows in the tool (e.g. `LIMIT`/`fetchmany(1001)` + "truncated, add LIMIT" note). Share the timeout fix from the database 🟡.
- **Confidence:** High
- **Tracked:** none found (`gh issue list --search "scratchpad"` → only unrelated #3316/#2313/hub-agent issues)

### [🟢] `_sanitize_name` silently rewrites table names instead of rejecting them, so distinct names collide
- **Where:** `src/gaia/scratchpad/service.py:409-415`
- **What:** `my-table`, `my table`, `my.table` and `my_table` all become `scratch_my_table`; names over 64 chars are
  truncated so two long names can share a table. The tool docstring says "alphanumeric and underscores only" but nothing enforces it — the model is told `Table 't___DROP_TABLE_x____' created` (probe 1) and proceeds.
- **Failure scenario:** Model creates `sales-2025` then `sales_2025` with different columns → second `CREATE TABLE IF NOT EXISTS` is a silent no-op, later `insert_data("sales_2025", ...)` fails on missing columns with a confusing error.
- **Fix:** Use `gaia.database.sql_safety.validate_identifier` (raise with the "bare identifier" hint) like every other identifier in the data layer; keep the `scratch_` prefix.
- **Confidence:** High
- **Tracked:** none found

### [🟢] `get_size_bytes` swallows every exception and returns 0, which silently disables the 100 MB cap
- **Where:** `src/gaia/scratchpad/service.py:378-389`; consumer `:218-225`
- **What:** `except Exception: log.warning(...); return 0` — if `list_tables()` fails (e.g. a scratch table was dropped
  between the listing and the `COUNT(*)`, or the DB is locked), the cap check in `insert_rows` sees 0 and proceeds.
  Straight "no silent fallbacks" violation from CLAUDE.md; the cap it guards is the only thing between an agent and a 20 GB scratchpad (`:212-217`).
- **Fix:** Let the exception propagate (`insert_rows` already surfaces `ValueError` to the tool as `Error: ...`).
- **Confidence:** High
- **Tracked:** none found

## Checked and fine (scratchpad)
- `create_table` DDL guard: `;`, `--`, `/*`, `*/` denied before parsing, parens balanced, per-column type root allow-listed; every attempt to smuggle a second statement into `executescript` failed (probes 2-5). Table-level constraints (`CHECK`, `PRIMARY KEY`, ...) pass through unparsed but cannot terminate the statement.
- `insert_rows` validates every key against the identifier grammar and binds values; per-table row cap and (estimated) size cap enforced before the transaction.
- `drop_table` / `clear_all` only touch `scratch_`-prefixed tables; `TABLE_PREFIX` prevents reaching `sqlite_*`.
- Tool layer: 10 MB JSON and 10 000-row per-call caps; all tool exceptions are translated to `Error: ...` strings at the tool boundary (allowed by CLAUDE.md).
- Rebuild-on-corruption (`_open_or_rebuild`, `:92-123`) did **not** fire when a second process held `BEGIN EXCLUSIVE` on a WAL-mode file (probe: second open succeeded in 0.0 s, tables intact) — WAL readers are not blocked. See Hypotheses for the rollback-journal case.

## Test gaps (scratchpad)
- `tests/unit/test_scratchpad_service.py` runs a real service (good): injection tests `test_insert_rejects_injection_in_key`, `test_query_data_rejects_*`, size-cap tests assert real behaviour.
- `tests/unit/test_scratchpad_tools_mixin.py` sets `self.agent._scratchpad = MagicMock()` for every tool group (`:124,173,282,450,558,654`) — it asserts formatting and passthrough (`assert_called_once_with`), never that the SQL reaching the service is valid. Acceptable given the service tests, but no test covers tool→service end to end (e.g. that a model-style CTE works, or what a 100 k-row result does to the tool output).
- No test pins the `pragma_*` bypass, the CTE rejection, or the comment false positive (all are current behaviour).

## Module 3 — `src/gaia/code_index/{sdk,parsers}.py` + `src/gaia/agents/tools/code_index_tools.py`

Read fully: `sdk.py` (1076 lines), `parsers.py` (436), `code_index_tools.py` (278), `docs/plans/code-index-review.mdx`,
`security.py:387-403` (`is_path_allowed` resolves via `os.path.realpath`), the flagship wiring
`hub/agents/gaia/python/gaia_agent/agent.py:271-273`, and the four `tests/unit/test_code_index*.py` (names + mock usage).
Probes ran with faiss 1.15.0 / numpy 2.5.2 in the venv, Lemonade replaced by a deterministic fake embedder.

**On-disk format.** `faiss.write_index` (native FAISS binary) + `metadata.json` (`sdk.py:965-966`); no pickle anywhere
(`grep -rn pickle src/gaia/code_index` → none). Nothing is signed — unlike the RAG cache's HMAC `.sig` — and
`_load_metadata` (`:985-1003`) validates only "file exists, version == 2"; chunk dicts are trusted (`_dict_to_chunk`, `:1066-1076`).
**Stale-index detection.** Per-file SHA-256 of content (`:218-221`); unchanged files reuse chunks + `faiss.reconstruct` embeddings;
deleted files drop out because only discovered files enter `new_file_hashes`. `search()` never checks staleness — that is by design
(re-index is a separate tool). **Root validation.** `CodeIndexSDK` scopes a `PathValidator` to the resolved repo root (`:160`) and
`_read_file_safe` rejects anything outside it (`:699-702`); `index_codebase` compares `Path.resolve()`d paths (`code_index_tools.py:140-148`)
and refuses the home directory (`:158-168`). **Parser edge cases** (probed): empty file → `[]`; null byte → `[]`; a file whose binary
bytes start after the 8 KB sniff is read via latin-1 and yields 0 chunks (no crash); a latin-1 `.py` decodes (`café` symbol found).

### Executed probe (fake embedder; 3 files alpha/beta/gamma)

```
search alpha (fresh)                                  -> [('alpha.py', 'alpha', 1.0)]
metadata.json chunks rotated by one (same count)      -> search alpha -> [('beta.py', 'beta', 1.0)]     <-- wrong file, no warning
metadata.json truncated to 2 chunks (FAISS has 3)     -> WARNING "Index/metadata mismatch ... cache corrupt, ignoring"
                                                         search alpha -> []   | get_status()["indexed"] -> True
metadata chunks = [{"chunk_type": "code"}] * 3        -> search -> KeyError 'content' ; index_repository -> KeyError 'content'
repo_path given as lower-case vs UPPER-case           -> same cache dir (resolve() canonicalises case on Windows)
index_codebase ratchet (mixin harness, SDK stubbed):
  call 1 repo_path=<root>/projects/a -> {"status": "ok"}   _repo_path now <root>/projects/a
  call 2 repo_path=<root>/projects/b -> {"error": "repo_path must be within <root>\\projects\\a"}
  call 3 repo_path=<root>            -> {"error": "repo_path must be within <root>\\projects\\a"}
```

## Findings (code_index)

### [🟡] Index/metadata consistency is a row-count check only: a desynced cache returns the *wrong* code silently, and a count mismatch makes `search` return `[]` while status says `indexed: true`
- **Where:** `src/gaia/code_index/sdk.py:1031-1037` (`_ensure_index_loaded`), `:439-440` (`search`), `:967-968` (comment in `_save_atomic`), `src/gaia/agents/tools/code_index_tools.py:215-233`
- **What:** The only link between `index.faiss` row *i* and `metadata.json` chunk *i* is position. `_save_atomic` renames the index first and the metadata second and its comment says `_load_metadata` "will detect stale metadata via ntotal check" — but `_load_metadata` does no such check; `_ensure_index_loaded` compares counts, and on mismatch logs a warning and returns `False`, which `search()` turns into `[]` (the tool then emits `[]` — indistinguishable from "no matches"). When the counts happen to agree (a crash between the two renames after an edit that kept the chunk count, or any external edit), every hit maps to the wrong chunk text.
- **Failure scenario:** Re-index after editing one function (same chunk count); process dies between `tmp_index.replace` and `tmp_meta.replace` → next session's `search_code_index("where is retry backoff decided")` returns confident, wrong snippets at score ~1.0. Or the metadata is truncated → every search returns `[]`, `get_index_status` says indexed, the model concludes the codebase does not contain the thing.
- **Evidence:** probe above (`beta.py` returned for `alpha`; `[]` + `indexed: True`). Code: `if index.ntotal != expected: self.log.warning(...); return False` (`:1032-1037`); `if not self._ensure_index_loaded(): return []` (`:439-440`).
- **Fix:** Write a fingerprint into both artefacts (e.g. `sha256(json.dumps(chunks))` stored in `metadata.json` *and* as `index.faiss.sig`, or store `embedding_dim` + per-chunk hash and verify `faiss_index.reconstruct(i)` on a sample), or write both files into a temp directory and swap the directory. On any mismatch raise the same actionable `RuntimeError("...run clear_index() and re-index")` already used for a corrupt FAISS binary (`:1026-1029`) instead of returning `False`. Fix the `_save_atomic` comment.
- **Confidence:** High
- **Tracked:** none found (`gh issue list --search "code index metadata"`, `"index corrupt"` → unrelated)

### [🟡] `index_codebase` ratchets the sandbox root down permanently — on the flagship the model can index exactly one repository per agent lifetime
- **Where:** `src/gaia/agents/tools/code_index_tools.py:140-151` (`index_codebase`), `hub/agents/gaia/python/gaia_agent/agent.py:271-272`
- **What:** The traversal guard compares the requested path against `self._repo_path`, then *overwrites* `self._repo_path` with the narrowed path. The flagship initialises `_repo_path` to `allowed_paths[0]` (the home directory by default) and the tool refuses to index home itself, so the model must narrow on the first call — after which every other repository under home is "outside the root".
- **Failure scenario:** User: "index ~/projects/a" → ok. User: "now index ~/projects/b" → `{"error": "repo_path must be within .../projects/a"}` until the agent process restarts. `clear_code_index` does not reset `_repo_path` either (`:273-275`).
- **Evidence:** probe above (calls 2 and 3 rejected). Code: `self._repo_path = resolved` (`:149`) right after the containment check against `self._repo_path` (`:141`).
- **Fix:** Keep the ceiling separately (`self._code_index_root_ceiling`, set once in `_init_code_index_state`) and check against it; let `_repo_path` be the *current* repo. Add a mixin test: index `a`, then `b`, both succeed.
- **Confidence:** High
- **Tracked:** none found (#870 "multiple repositories in a single namespace" is a feature request, not this regression)

### [🟢] Malformed `metadata.json` crashes with a bare `KeyError` instead of the "cache corrupt" path
- **Where:** `src/gaia/code_index/sdk.py:1062-1076` (`_dict_to_chunk`), callers `:196`, `:486`
- **What:** `_load_metadata` catches only `JSONDecodeError`/`OSError`; a chunk dict missing `content`/`file_path`/... raises `KeyError` from `search()` and `index_repository()`. The tool wraps it as `{"error": "'content'"}` — not actionable, and `index_codebase` cannot self-heal because it crashes before writing.
- **Evidence:** probe: `search -> KeyError 'content'; index_repository -> KeyError 'content'`.
- **Fix:** Validate the chunk schema in `_load_metadata` and treat a violation like a version mismatch (rebuild) or raise the existing "run clear_index()" `RuntimeError`.
- **Confidence:** High
- **Tracked:** none found

### [🟢] `_read_gitignore_patterns` swallows read errors (`except OSError: pass`)
- **Where:** `src/gaia/code_index/sdk.py:692-693`
- **What:** An unreadable `.gitignore` silently yields no patterns, so the walk indexes everything the file was meant to exclude (build output, vendored trees, secrets-by-convention paths). CLAUDE.md "no silent fallbacks".
- **Fix:** `log.warning` with the path and reason at minimum; better, raise — an unreadable file in the repo root is a real problem.
- **Confidence:** High
- **Tracked:** none found

## Checked and fine (code_index)
- No pickle; FAISS native format is not a deserialisation-RCE vector; atomic `Path.replace` on both platforms (`:973-974`).
- `_read_file_safe` goes through `PathValidator.is_path_allowed`, which `realpath`-resolves (`security.py:389,403`), so a symlinked file inside the repo pointing outside is rejected at read time; `os.walk(followlinks=False)`.
- `index_codebase` resolves both sides before the prefix check (plan-doc bug #2 fix is present, `:140-148`) and refuses `Path.home()` (`:158-168`).
- Query encoding failures raise (`_encode_texts`, `:862-869`) — "no matches" is not reachable for a Lemonade outage; model mismatch raises `ValueError` (`:447-454`); corrupt FAISS binary raises `RuntimeError` (`:1020-1029`).
- Files with a dropped chunk are left un-hashed so the next index retries them (`:371-391`); all-embeddings-failed raises (`:357-369`).
- Discovery: `max_files`, `max_walk_entries` and `max_file_size_mb` caps all enforced; sensitive filenames skipped (`_is_sensitive_file`).
- Windows cache-key case normalisation works today (probe) — `Path.resolve()` canonicalises; the plan doc's "known gap" is stale, not a bug.
- `_split_oversized_chunk` computes per-part line ranges (`parsers.py:377-383`) and matching in `index_repository` is keyed on `(file_path, start_line, content)` with each cache row consumed once (`sdk.py:272-288`), so split parts cannot steal part 1's embedding.

## Module 4 — `src/gaia/agents/tools/rag_tools.py`: does the tool layer pre-validate paths before `RAGSDK.index_document`?

**Partly. `index_document` and the auto-index inside `query_specific_file` do; `index_directory` does not.**

```python
# rag_tools.py:1231-1232  — existence is checked BEFORE authorisation (leaks existence of denied paths)
if not os.path.exists(file_path):
    return {"status": "error", "error": f"File not found: {file_path}"}
# rag_tools.py:1258-1266  — pre-validation, but only if the host defines _is_path_allowed
if hasattr(self, "_is_path_allowed"):
    if not self._is_path_allowed(real_file_path):
        return {"status": "error", "error": f"Access denied: {real_file_path} is not in allowed paths"}
result = self.rag.index_document(real_file_path)          # :1268

# rag_tools.py:574-586  — query_specific_file auto-index: same hasattr-guarded check
if hasattr(self, "_is_path_allowed") and not self._is_path_allowed(resolved): ... return error
idx_result = self.rag.index_document(resolved)

# rag_tools.py:1954-2016 — index_directory: NO allowed_paths check on directory_path or on any file
dir_path = Path(directory_path).resolve()
...
files_to_index = [f for f in dir_path.rglob("*") if f.is_file()]     # :2008
for file_path in files_to_index:
    if file_path.suffix.lower() in supported_extensions:               # includes ".pdf"
        result = self.rag.index_document(str(file_path))               # :2016
```
`_is_path_allowed` is defined only on `ChatAgent` (`hub/agents/chat/.../agent.py:1161`, delegating to `self.path_validator`), which
builds `RAGConfig(allowed_paths=config.allowed_paths)` (`:305`). GaiaAgent inherits it. `grep -rn "index_directory" tests/unit/test_rag_tools.py tests/unit/rag` → **no test**.

### [🔴] `index_directory` skips `allowed_paths` entirely, so the RAG SDK's PDF/PPTX/DOCX/XLSX gap is reachable from the model with one tool call
- **Where:** `src/gaia/agents/tools/rag_tools.py:1954-2016` (`index_directory`); pairs with `.review/_08_rag_data.md` 🔴 (`rag/sdk.py:722,1012,1601,1737` bypass `_safe_open`)
- **What:** The single-file tool pre-validates (`:1258-1266`), so for `index_document` the SDK gap is defence-in-depth only (🟡 at product level). `index_directory` performs no validation before handing every `.pdf`/`.txt`/`.csv`/... under an arbitrary directory to `self.rag.index_document`. Text-type files are still stopped inside the SDK by `_safe_open`; the four binary document types are not.
- **Failure scenario:** ChatAgent/GaiaAgent with `allowed_paths=["C:/proj"]`; model (or a prompt-injected document) calls `index_directory("C:/Users/x/Documents", recursive=True)` → every PDF/DOCX/XLSX/PPTX under Documents is parsed, embedded and its full text written to `cache_dir/*_extracted.md`, then answerable via `query_documents`. The `.txt` files in the same tree are denied — so the per-file "failed" count is the only visible symptom.
- **Evidence:** quoted lines above; the SDK side was executed by the RAG reviewer (PDF outside `allowed_paths` → `pdf_status=empty`, i.e. parsed). The tool→SDK chain was traced by code reading, not executed end to end (needs the RAG stack).
- **Fix:** In `index_directory`, run `self._is_path_allowed(str(dir_path))` before listing and `self._is_path_allowed(str(file_path))` per file (rglob can cross into symlinked dirs on older Pythons), returning the same `Access denied` shape as `index_document`; move the `os.path.exists` check in `index_document` after the authorisation check. Land the SDK-level fix from the RAG review as the real boundary. Add `tests/unit/test_rag_tools.py::test_index_directory_outside_allowed_paths_denied`.
- **Confidence:** High (code) / the combined exploit not executed
- **Tracked:** none found (`gh issue list --search "index_directory"` → only #2796 cpp port)

### [🟢] Path validation in the RAG tools is `hasattr`-gated, so a non-ChatAgent host gets no validation at all
- **Where:** `src/gaia/agents/tools/rag_tools.py:575-577, 1259` (`hasattr(self, "_is_path_allowed")`)
- **What:** `class MyAgent(Agent, RAGToolsMixin)` (the documented composition pattern) has no `_is_path_allowed`, so `index_document` silently skips the check and relies on whatever `RAGConfig.allowed_paths` the author set — which, for the four binary types, the SDK ignores. This is the same "mixin expects host attributes nothing sets" pattern as #3316.
- **Fix:** Make the check unconditional via the `PathValidator` the base `Agent`/mixin contract provides (or fail loudly at registration when neither `_is_path_allowed` nor `path_validator` exists).
- **Confidence:** High
- **Tracked:** #3316 (same class of issue; this site is not listed there)

## Module 5 — Tests: what asserts real behaviour vs. "a mock was called"

| File | Real behaviour | Mock-only / weak |
|---|---|---|
| `tests/unit/test_database_mixin.py` (49 tests) | Real in-memory SQLite everywhere; `query_readonly` tests re-query the table to prove no mutation, prove authorizer disarm, nested transaction | — |
| `tests/unit/test_sql_safety.py` | (located, not read) | — |
| `tests/unit/test_scratchpad_service.py` (~45) | Real service on a temp DB: injection-in-key, dangerous-keyword, size cap (`monkeypatch` on the estimate), prefix isolation | none of the query_data false-positive/`pragma_*` cases |
| `tests/unit/test_scratchpad_tools_mixin.py` (~60) | Tool formatting, JSON parsing, caps — real | Every service interaction is `MagicMock()` + `assert_called_once_with` (`:124,173,282,450,558,654`); no tool→service path |
| `tests/unit/test_code_index_sdk.py` (27) | `test_atomic_save_*`, `test_ensure_index_loaded_raises_on_corrupt_faiss`, `test_file_with_dropped_chunk_is_left_unhashed`, walk-budget tests build real files/indices | Embedder always patched (`_embedder`/`_encode_texts`), fine — but **no test for the ntotal-mismatch path or a same-count desync**; `test_search_with_mocked_index` sets `ntotal = 1` by hand |
| `tests/unit/test_code_index_mixin.py` (13) | Traversal guard (`..`, absolute elsewhere, junction) real | `patch` on SDK for index/search; **no test that two different sub-repos can be indexed in one session** |
| `tests/unit/test_code_index_parsers.py` (40), `test_code_index.py` (9) | Pure functions, real | — |
| `tests/unit/test_rag_tools.py` | — | **no `index_directory` test at all** |

## Documentation gaps
- `docs/plans/code-index-review.mdx:155-164` says oversized chunks split at **20,000 chars** with **approximate (whole-range) line numbers**; code splits at `MAX_EMBED_CHARS - 100 = 1100` (`parsers.py:347-357`) with **per-part** line ranges (`:377-383`) and bumped `_CACHE_VERSION` to 2 for it (`sdk.py:118-122`). The doc describes the v1 behaviour.
- Same doc `:201-207, 291-292` lists Windows cache-key case sensitivity as a known open gap; probe shows `Path.resolve()` already canonicalises case (identical cache dir for `c:\...` vs `C:\...`). Stale "still open" claim.
- `sdk.py:967-968` comment: "`_load_metadata` will detect stale metadata via ntotal check" — no such check exists in `_load_metadata`; `_ensure_index_loaded` does a count compare and returns `False` (see 🟡).
- `scratchpad_tools.py:175-176` docstring: "Supports all SQLite functions ... subqueries" — a top-level CTE (`WITH ... SELECT`) is refused with "Only SELECT queries are allowed" (probe 9). Either accept `WITH` or say so.
- `docs/sdk/mixins/database-mixin.mdx` matches the code (checked `:44,54,79,171,269`). No doc mentions that `db_query` has no timeout/row bound.

## Improvement opportunities
- One `readonly_query(conn, sql, params, *, timeout_s, max_rows)` helper in `sql_safety.py` used by both `DatabaseMixin.query_readonly` and `ScratchpadService.query_data` — removes the duplicate text guard and adds the missing bounds in one place.
- Sign `code_index/metadata.json` the way the RAG cache is signed (`rag/sdk.py` HMAC `.sig`): the cache lives in the user's home and its `content` fields go straight into the model context, so a tampered cache is a prompt-injection channel with no integrity check today.
- Validate the metadata schema on load (dataclass-from-dict with required keys) so a corrupt cache always lands on the "run clear_index()" path.
- `search_code_index` could return a cheap `stale: true` per result when the file's current SHA-256 differs from `file_hashes[...]` — the hashes are already in memory; today a search after an edit silently serves old code until the model remembers to re-index.
- `_sanitize_name` → `validate_identifier` (reject, don't rewrite); one identifier grammar across the data layer.

## High-impact feature opportunities
- **Scratchpad file import (`load_table_from_file`)** — there is no CSV/XLSX ingress; the model must transcribe rows through a 10 MB / 10 000-row JSON tool call, which is exactly the bottleneck for the "multi-document financial analysis" the mixin advertises. `RAGSDK` already parses CSV/XLSX (`_extract_text_from_xlsx`), so a tool that streams a validated (allowed_paths-checked) file into a `scratch_` table with header sanitisation is mostly wiring plus the row cap. Ties to #1499 (finance agent).
- **Per-table grants for `DatabaseAgent`** — `agent.py:80-84` admits "no per-table authorization"; a `tables=` allow-list on the constructor, enforced in the authorizer (`SQLITE_READ`/`SQLITE_INSERT` carry the table name in `arg1`) is a small change that turns the mixin into something a hub agent can safely point at a shared DB.
- **Query budget for LLM SQL** (timeout + row cap + truncation flag) across `db_query` and scratchpad `query_data` — turns two hang/flood paths into model-correctable errors.
- **Multi-repo code index** (#870) — the ratchet 🟡 is the immediate blocker; a `repo_path`-keyed SDK map in the mixin would allow the flagship to search several repos without restarting.

## Checked and fine (cross-cutting)
- No LLM-supplied SQL reaches `DatabaseMixin.query`/`execute` unguarded anywhere in `src/` or `hub/` except `ScratchpadService.query_data` (single-statement, SELECT-prefixed — read-only in practice, see 🟡).
- `DatabaseMixin.execute` (`executescript`) is only fed literal DDL or the scratchpad's validated column list.
- `insert_data` JSON cap runs before `json.loads`; `insert_rows` per-row identifier validation happens before any SQL is built.

## Hypotheses (unverified)
- `ScratchpadService._open_or_rebuild` (`service.py:98-120`) deletes the DB file on *any* exception, including `database is locked`. Under WAL the probe could not trigger it (readers aren't blocked). A first-ever open of a rollback-journal file while another process holds a write lock > 5 s (the default busy timeout) may still hit it; on POSIX the unlink would succeed and fork the two processes onto different inodes. Not reproduced.
- `_read_gitignore_patterns` feeds raw lines to `fnmatch` — negations (`!keep.py`), anchored (`/build`) and directory (`dist/`) patterns don't behave like git; probably indexes some ignored trees and skips some kept files. Not measured.
- `parse_python_file` uses `ast.walk`, so a method is chunked twice (inside its class chunk and on its own); index size and duplicate hits in `search` likely inflated for class-heavy code. Not measured.
