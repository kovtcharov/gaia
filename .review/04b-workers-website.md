# 04b — Workers (agent-hub, website-router) + Website + Install scripts

Reviewer scope: `workers/agent-hub/`, `workers/website-router/`, `website/`, and the install
scripts the site serves (`installer/scripts/install.sh`, `installer/scripts/install.ps1`).
Checkout: main @ 211f08c5 (v0.23.1). Read-only. Live probes against `hub.amd-gaia.ai` /
`amd-gaia.ai` were GET/HEAD/OPTIONS only (plus one unauthenticated POST with a bogus token).

## Scope covered

Read in full:
- `workers/agent-hub/src/*.ts` (index, auth, http, multipart, publish, skill-publish, manifest,
  skill-manifest, audit, catalog, storage, types — 3,258 lines), `schemas/*.json`, `wrangler.toml`,
  `Dockerfile`, `docker-entrypoint.sh`, `railway.json`, `package.json`, `vitest.config.ts`, `README.md`.
- `workers/agent-hub/test/`: `fake-r2.ts`, `routes.test.ts`, `publish-by-reference.test.ts` in full;
  `publish.test.ts`, `skill-publish.test.ts`, `audit-binding.test.ts`, `manifest.test.ts`,
  `skill-manifest.test.ts`, `audit-digest-vectors.test.ts` by test-case inventory plus the
  schema-conformance block of `skill-publish.test.ts` (lines 620-735) in full.
- `workers/website-router/src/index.js`, `wrangler.toml`, `README.md`.
- `website/`: `astro.config.mjs`, `railway.json`, `package.json`, `README.md`, `public/robots.txt`,
  `src/data/{catalog,markdown,fileTree}.ts`, `src/scripts/download-target.ts`,
  `src/pages/index.astro`, `src/pages/hub/index.astro`, `src/pages/hub/[id].astro`,
  `src/layouts/Layout.astro`, every component under `src/components/`; test-case inventory of
  `src/data/*.test.ts` and `src/scripts/download-target.test.ts`.
- `installer/scripts/install.sh` (666 lines), `installer/scripts/install.ps1` (461 lines);
  `tests/unit/installer/test_install_scripts_terminal_hub.py` (run).
- CI context: `.github/workflows/{deploy_website,website-ci,agent_hub_worker_ci}.yml` in full;
  the publish jobs of `release_components.yml` and the job graph of `release_agent_{email,gaia}.yml`.
- Consumers cross-checked: `src/gaia/hub/catalog.py`, `hub/agents/email/python/packaging/publish_to_r2.py`
  (grep-level), `src/gaia/apps/webui/src/utils/hubLanes.ts`.

Skipped / not run: `website/src/scripts/{starfield,constellations}.js` and the design CSS
(presentational); the Worker and website vitest suites could NOT be executed (no `node_modules`,
and `npm install` is prohibited for this review) — findings against them are from reading.
Live catalog at review time had 4 entries (agent-ui/app,
email/agent, gaia/agent, terminal-hub/component).

## Findings

### [🔴] The flagship's hub page tells visitors to `pip install gaia-agent-gaia`, which does not exist
- **Where:** `website/src/data/catalog.ts:548-555` (`installMethods`), rendered by
  `website/src/components/InstallMethods.astro`; live at https://amd-gaia.ai/hub/gaia
- **What:** For any `type: agent` entry with `language: python` and no `npm_package`, the site
  adds a "pip" tab with `pip install gaia-agent-<id>`. The flagship `gaia` agent is published
  to the hub only as frozen binaries (`gaia-agent-linux-x64`, `gaia-agent-darwin-*`,
  `gaia-agent-win32-x64.exe`), so the advertised package is fictional.
- **Failure scenario:** A visitor on /hub/gaia clicks the "pip" tab, copies the command, and gets
  `ERROR: No matching distribution found for gaia-agent-gaia`. The same code path would do this
  for every future binary-distributed Python agent. Meanwhile the landing page advertises
  `npm install @amd-gaia/gaia` (which DOES exist, 0.1.1) but the hub page never shows npm because
  `hub/agents/gaia/python/gaia-agent.yaml` declares no `npm_package`, so the two pages
  contradict each other.
- **Evidence:**
  ```
  $ curl -s https://amd-gaia.ai/hub/gaia | grep -o 'data-im-copy="[^"]*"'
  data-im-copy="gaia agent install gaia"
  data-im-copy="pip install gaia-agent-gaia"
  data-im-copy="git clone https://github.com/amd/gaia.git"
  $ curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/gaia-agent-gaia/json
  404
  $ curl -s https://registry.npmjs.org/@amd-gaia%2fgaia | head -c 120
  {"_id":"@amd-gaia/gaia",...,"dist-tags":{"latest":"0.1.1"}
  $ grep -n "npm_package" hub/agents/gaia/python/gaia-agent.yaml   # (no output)
  ```
  `catalog.ts:548`: `if (agent.language === "python") { methods.push({ key: "pip", ... command: \`pip install gaia-agent-${agent.id}\`` — no check that a wheel was ever published.
- **Fix:** The index entry carries no "distribution kind". Either (a) have the Worker emit
  `artifact_kinds` / `has_wheel` in `toIndexEntry` (it already has `latest.artifacts[]`) and gate
  the pip tab on it, or (b) at minimum add `npm_package: "@amd-gaia/gaia"` to the gaia manifest so
  `installMethods` takes the npm branch (no pip tab) and the landing/hub pages agree. Add a
  `catalog.test.ts` case: "a binary-only python agent gets no pip method".
- **Confidence:** High
- **Tracked:** related #3101 (chat agent README leads with a failing PyPI install) — same root
  cause on a different surface; the generated site tab is not tracked.

### [🟡] Catalog writes are unconditional read-modify-write — concurrent publishes can drop an artifact or an entry
- **Where:** `workers/agent-hub/src/storage.ts:211-225` (`writeAgentManifest`, `writeIndex`),
  `src/publish.ts:338,470-474`, `src/catalog.ts:379-429` (`rebuildIndex`)
- **What:** Both `agents/<id>/manifest.json` and `index.json` are produced by read → merge in
  memory → `put()` with no `onlyIf: { etagMatches }` (R2 supports conditional puts) and no
  serialization. Two publishes interleaving lose one of the writes.
- **Failure scenario:** `release_components.yml` runs the `terminal-hub` and `agent-ui` publish
  jobs in parallel (both `needs: [version, worker-check, deploy-worker]`). If job A's
  `rebuildIndex` lists the bucket before job B writes `agents/agent-ui/manifest.json` but
  finishes after B's rebuild, `index.json` is written without agent-ui and the website deploy
  that follows builds from a catalog missing an entry (recoverable only via `POST /reindex`).
  Worse for two publishes of the SAME id (e.g. a re-run matrix): both read `existing`, each
  `upsertVersion`s its own artifact, last write wins — and on the inline lane the dropped artifact
  can never be re-published because `head(key)` now 409s (`version_exists`) while the manifest
  never lists it.
- **Evidence:**
  ```ts
  // storage.ts
  await bucket.put(agentManifestKey(manifest.id), JSON.stringify(manifest, null, 2), {
    httpMetadata: { contentType: "application/json; charset=utf-8" },
  });          // no onlyIf / etag
  ```
  `publish.ts:338 const existing = await readAgentManifest(env.BUCKET, manifest.id);` …
  `publish.ts:471 await writeAgentManifest(env.BUCKET, updated);` with ~10 awaited R2 calls in between.
- **Fix:** Read the manifest with its `etag`, write with `onlyIf: { etagMatches }` (or an
  absent-object guard for first publish) and retry the merge on precondition failure; same for
  `index.json`. Alternatively route publishes through a Durable Object per id. Add a fake-R2
  test that interleaves two publishes.
- **Confidence:** High (code) / Medium (how often the CI window is hit)
- **Tracked:** none found

### [🟡] Inline publish is not atomic: a failure after the artifact `put` leaves a filename that can never be published
- **Where:** `workers/agent-hub/src/publish.ts:361-372, 404-407, 469-474`; same shape in
  `src/skill-publish.ts:157-164, 175-178, 216-219`
- **What:** On the inline lane the artifact object is stored first, then up to 8 doc objects,
  then the manifest, then the whole index rebuild (~40 R2 calls today). If anything after the
  artifact `put` throws (R2 error, a corrupt sibling manifest making `obj.json()` throw in
  `rebuildIndex`, subrequest/CPU limit), the response is 500 but the artifact exists. The
  "already published" check for inline is `Boolean(await env.BUCKET.head(key))`, so every retry
  returns `409 version_exists` although the catalog never listed the artifact.
- **Failure scenario:** A release job's first POST gets a transient 500 from `rebuildIndex`;
  CI re-runs (the workflows document 409 as "already published — success"), the job goes green,
  and the platform binary is silently missing from `manifest.json`/`index.json` and from
  `gaia agent install`. The only recovery is manual R2 deletion.
- **Evidence:**
  ```ts
  const alreadyPublished = byReference
    ? Boolean(existing?.versions[manifest.version]?.artifacts?.some((a) => a.filename === filename))
    : Boolean(await env.BUCKET.head(key));
  ...
  await env.BUCKET.put(key, bytes, {...});        // line 404
  ...                                            // 8 more puts, manifest write, rebuildIndex
  ```
  `catalog.ts:387 const agent = await readAgentManifest(bucket, id);` → `obj.json()` throws on a
  malformed sibling and aborts the whole publish after the artifact was stored.
- **Fix:** Make the manifest the record of truth for BOTH lanes (the by-reference lane already
  does this) and let an inline publish whose object exists but isn't in the manifest proceed
  (re-hash the stored bytes or compare `head.checksums.sha256`); or wrap the tail in a
  `try/catch` that deletes the just-written objects before re-throwing. Either way return
  a 5xx that names the orphaned key.
- **Confidence:** High
- **Tracked:** none found

### [🟡] Artifact filenames can collide with the per-version doc objects and silently corrupt what is served
- **Where:** `workers/agent-hub/src/multipart.ts:154` (`ARTIFACT_FILENAME_RE`),
  `src/publish.ts:349, 404-453`, `src/skill-publish.ts:156-197`
- **What:** The filename rule only forbids separators. `README.md`, `CHANGELOG.md`, `SPEC.md`,
  `SKILL.md`, `EVALUATION.md`, `CAPABILITY_MATRIX.md`, `SCORECARD.md`, `gaia-agent.yaml`,
  `package-files.json` (and `audit.json` on the skill lane) all pass, yet they are the exact keys
  the same request writes right after the artifact.
- **Failure scenario:** First publish of a version with `artifact` named `gaia-agent.yaml` (or a
  skill bundle named `SKILL.md`): the artifact bytes are stored at
  `agents/<id>/<v>/gaia-agent.yaml`, then line 413 overwrites that key with the manifest text.
  The catalog records the artifact's original `sha256`/`size_bytes`; `install.sh` /
  `install.ps1` / `gaia agent install` download the YAML, the checksum mismatches and the install
  aborts ("Checksum mismatch … refusing to install"). The reverse (artifact named `README.md`
  with no `readme` part) makes `rebuildIndex` read the binary as README markdown into
  `index.json`.
- **Evidence:** `multipart.ts:154 export const ARTIFACT_FILENAME_RE = /^[A-Za-z0-9][A-Za-z0-9._+-]*$/;`
  `publish.ts:413 await env.BUCKET.put(rawManifestKey(manifest.id, manifest.version), manifestText, …)`
  where `rawManifestKey` = `${versionDir}gaia-agent.yaml` and `key` = `${versionDir}${filename}`.
- **Fix:** Reject a reserved set of filenames (case-insensitively) in both lanes with a 400
  naming the reserved name; add tests. Publishers are authenticated, so this is a foot-gun rather
  than an attack, but it breaks the "server-side SHA-256 you can trust" guarantee.
- **Confidence:** High
- **Tracked:** none found

### [🟡] `index.schema.json` rejects every published skill entry (audit block) and the conformance test cannot see it
- **Where:** `workers/agent-hub/schemas/index.schema.json:248-269`;
  `src/catalog.ts:366-371` (`toSkillIndexEntry`), `src/audit.ts:244-291` (`SkillAuditRecord`);
  `test/skill-publish.test.ts:627-680`
- **What:** The schema declares `skill_metadata.audit` with `additionalProperties: false` and only
  `verdict/engine/audited_at/findings`, but the Worker stores and emits `attestation`,
  `cleared_tiers`, `content_digest`, `manifest_digest` on every record (including the
  `unaudited` one). README says the schema is "the contract the website Hub pages build from".
  The conformance test only diffs TOP-LEVEL keys of an entry, so nested drift passes.
- **Failure scenario:** Any consumer that validates `index.json` against the shipped schema
  (the README invites this) fails as soon as one skill is published; the website's
  `SkillMetadata.audit` type also lacks the four fields, so the attestation value the Worker
  went out of its way to record is invisible to the site.
- **Evidence:**
  ```json
  "audit": { "type": "object", "additionalProperties": false,
             "required": ["verdict", "engine", "audited_at", "findings"], ... }   // schema
  ```
  ```ts
  return { verdict: "ALLOW", engine: ..., audited_at: ..., findings: ...,
           attestation: "publisher-asserted", cleared_tiers: ..., content_digest: ..., manifest_digest: ... }; // audit.ts:279
  ```
  ```ts
  const undeclared = Object.keys(entry).filter((k) => !declared.has(k));   // top level only, skill-publish.test.ts:667
  ```
- **Fix:** Add the four properties to the schema (and to `website/src/data/catalog.ts`
  `SkillMetadata.audit`), and make the test walk nested `additionalProperties:false` objects
  (or vendor a tiny validator in devDependencies — it does not affect the Worker bundle).
- **Confidence:** High
- **Tracked:** none found

### [🟡] `manifest.schema.json` is stale — it rejects two of the four live manifests
- **Where:** `workers/agent-hub/schemas/manifest.schema.json:34`
- **What:** `"language": { "enum": ["python", "cpp"] }` while `src/manifest.ts:25` accepts
  `python|cpp|go|typescript` and the live `terminal-hub` (go) and `agent-ui` (typescript)
  manifests use the new values. No test reads this schema (`grep manifest.schema.json test/` is
  empty), so nothing catches it.
- **Failure scenario:** Anyone validating `GET /agents/terminal-hub/manifest.json` against the
  published schema gets a false failure; the README advertises the schema as the formal contract.
- **Evidence:** `schemas/manifest.schema.json:34: "language": { "type": "string", "enum": ["python", "cpp"] }`;
  `index.schema.json:76` already lists `["python","cpp","go","typescript","markdown"]`.
- **Fix:** Sync the enum (and audit the rest of the file for `type`, `npm_package`,
  `playground_url`, `artifacts[]`), and add the same "declares every field" test the index schema has.
- **Confidence:** High
- **Tracked:** none found (#2959 is the related "deployed Worker predates VALID_LANGUAGES" incident)

### [🟡] The install one-liners are served by an out-of-repo redirect to GitHub `main`, contradicting the router docs
- **Where:** `workers/website-router/src/index.js:107-150` and `README.md` ("everything else →
  Railway"); `website/public/` (no install scripts); `installer/scripts/install.sh:3`,
  `install.ps1:2`
- **What:** `https://amd-gaia.ai/install.sh` and `/install.ps1` are answered by a Cloudflare
  302 (edge rule, not in this repo) to `raw.githubusercontent.com/amd/gaia/main/installer/scripts/…`.
  The router README and the website README both say every non-`/docs` path is proxied to the
  Astro site, and nothing in the repo owns the rule.
- **Failure scenario:** (1) Users always execute whatever is on `main` at that second — an
  unreleased installer change (or a broken merge) ships to every `curl | sh` immediately, with no
  tag pinning and no rollback story. (2) Whoever rotates the Cloudflare zone or follows the
  router README's rollback ("delete the route — traffic falls back to Mintlify") has no record
  that the rule exists; if it is lost, `/install.sh` becomes a Railway 404 and the documented
  quickstart command fails (`curl -f` exits 22).
- **Evidence:**
  ```
  $ curl -sS -o /dev/null -w "%{http_code} %{redirect_url}\n" https://amd-gaia.ai/install.sh
  302 https://raw.githubusercontent.com/amd/gaia/main/installer/scripts/install.sh
  ```
  `website-router/src/index.js` contains no redirect logic; `website/public/` holds only
  `favicon.ico gaia-icon.png robots.txt`.
- **Fix:** Move the redirect into `website-router` (`/install.sh` → a release-tagged raw URL,
  e.g. `.../v0.23.1/installer/scripts/install.sh`, or serve the scripts from `website/public/`
  copied at build time) so it is versioned, reviewable and pinned to a release; document it in
  both READMEs.
- **Confidence:** High (behaviour) / Medium (that the rule lives only in the dashboard — inferred
  from its absence in the repo)
- **Tracked:** none found

### [🟡] website-router's documented HTML edge cache is not happening in production
- **Where:** `workers/website-router/src/index.js:135-149`, `README.md` "Caching"
- **What:** README: "`/_astro/*` for a year … and 60s for everything else". Live, every HTML
  request returns `cf-cache-status: DYNAMIC`; assets are cached only because the Railway origin
  (`serve`) sends `max-age=14400`, not the promised one-year TTL.
- **Failure scenario:** Every page view round-trips to Railway (the thing the caching was added
  to avoid), and a Railway restart mid-deploy is user-visible instead of absorbed by the edge.
- **Evidence:**
  ```
  $ curl -sI https://amd-gaia.ai/hub                      (twice)
  html 200 cf-cache=DYNAMIC cc=  age=
  html 200 cf-cache=DYNAMIC cc=  age=
  $ curl -sI https://amd-gaia.ai/_astro/_id_.DsZcM0tr.css  (twice)
  asset 200 cf-cache=REVALIDATED cc=max-age=14400
  asset 200 cf-cache=HIT         cc=max-age=14400 age=0
  $ curl -sI https://website-production-82ab.up.railway.app/hub | grep -i cache   # origin sends no Cache-Control for HTML
  ```
- **Fix:** Verify the deployed Worker matches the repo, then either set an explicit
  `cacheTtl` for GET HTML alongside `cacheEverything`, or have the origin emit `Cache-Control`
  (`serve` supports a `serve.json` `headers` block) so the edge has something to honour. Add the
  `cf-cache-status` check to the deploy smoke test so the README claim is verified, not assumed.
- **Confidence:** High (symptom) / Medium (cause)
- **Tracked:** none found

### [🟢] Bearer tokens are looked up on a plain object — prototype keys turn an unauthenticated request into a `500 server_misconfigured`
- **Where:** `workers/agent-hub/src/auth.ts:40,59-70`; `src/index.ts:78` (reindex token)
- **What:** `tokens = JSON.parse(env.PUBLISH_TOKENS)` is a plain object and `tokens[match[1]]`
  is read without `Object.hasOwn`. A token value of `constructor`, `toString`, `__proto__`, …
  resolves to an `Object.prototype` member, skips the 401 branch, and fails the next check with
  a 500 whose message blames the operator's secret. Neither lookup is constant-time
  (`/reindex` uses `!==`); with 256-bit random tokens the timing leak is academic, the wrong
  status code is not.
- **Failure scenario:** Live, right now:
  ```
  $ curl -s -w "\n%{http_code}\n" -X POST -H "Authorization: Bearer constructor" -F manifest=x https://hub.amd-gaia.ai/publish
  {"error":{"code":"server_misconfigured","message":"PUBLISH_TOKENS entry is missing a 'publisher' string."}}
  500
  ```
  Anyone can generate 5xx noise that reads as "the maintainer misconfigured the Worker" in logs
  and alerting; the rest of auth is unaffected (no bypass — a prototype member never has a
  `publisher` string).
- **Evidence:** `auth.ts:59 const record = tokens[match[1]]; if (!record) { throw new HttpError(401, ...) }`.
- **Fix:** `const record = Object.hasOwn(tokens, t) ? tokens[t] : undefined` (or build a
  `Map`), and compare candidates with `crypto.subtle.timingSafeEqual` on equal-length buffers;
  same for `REINDEX_TOKEN`. Add a test for `Bearer constructor` → 401.
- **Confidence:** High
- **Tracked:** none found

### [🟢] Malformed percent-encoding on a download path is a 500 `internal_error`, not a 400
- **Where:** `workers/agent-hub/src/index.ts:91,112-114`
- **What:** `decodeURIComponent(path.slice(1))` throws `URIError` for sequences like `%C0%AF`;
  the catch-all maps every non-`HttpError` to 500 and echoes the message.
- **Failure scenario:** `GET /agents/%C0%AF/manifest.json` → `{"error":{"code":"internal_error","message":"URI malformed"}}` HTTP 500 (verified live) — a client error counted as a server error in Cloudflare analytics/alerts.
- **Fix:** Wrap the decode and throw `HttpError(400, "invalid_path", …)`.
- **Confidence:** High
- **Tracked:** none found

### [🟢] `robots.txt` advertises a sitemap the site never generates
- **Where:** `website/public/robots.txt:4` (`Sitemap: https://amd-gaia.ai/sitemap.xml`); `website/astro.config.mjs` has no `@astrojs/sitemap` integration.
- **Failure scenario:** `curl -o /dev/null -w "%{http_code}" https://amd-gaia.ai/sitemap.xml` → `404` (also `/sitemap-index.xml`); crawlers log a broken sitemap for every visit.
- **Fix:** Add `@astrojs/sitemap` (emits `sitemap-index.xml`; point robots at it) or drop the line.
- **Confidence:** High
- **Tracked:** none found

### [🟢] A malformed scorecard silently drops the eval score (silent-fallback rule)
- **Where:** `workers/agent-hub/src/catalog.ts:121-135` (`parseScorecardScore`)
- **What:** `try { … } catch { return undefined; }` swallows YAML parse errors, and a scorecard
  whose `aggregate.value` is missing/non-numeric also yields `undefined`. The publish still
  returns 201 with no warning, so the hub page quietly loses its score badge.
- **Failure scenario:** A release ships a scorecard with a typo in the front matter; the eval
  badge disappears from /hub/email and nobody is told. CLAUDE.md's "no silent fallbacks" rule
  applies — the comment even says "never throws so a bad scorecard never breaks the catalog build".
- **Fix:** Validate the scorecard at publish time (`400 invalid_scorecard` naming the field),
  keep the rebuild tolerant only for legacy objects, and log the key when it is skipped.
- **Confidence:** High
- **Tracked:** none found

### [🟢] Documentation drift inside `workers/*/README.md` and `website/README.md`
- **Where:** `workers/agent-hub/README.md:219-223,284-285,15-24`; `website/README.md:224-226`
- **What / Evidence:**
  - agent-hub README says the bucket "default `gaia-agent-hub`" and shows
    `wrangler r2 bucket create gaia-agent-hub`; `wrangler.toml:470` binds `bucket_name = "gaia-hub"`.
  - README step 4: "(Optional) Bind the route by uncommenting the `routes` line in `wrangler.toml`" —
    the line is active (`routes = [{ pattern = "hub.amd-gaia.ai", custom_domain = true }]`).
  - The route table omits `POST /reindex` and the `REINDEX_TOKEN` secret it needs (only
    `types.ts` and `index.ts` mention it); the Railway demo entrypoint never materialises it, so
    `/reindex` on the demo is a permanent `500 config_error`.
  - website README: "The Worker's source is being brought into the repo under
    `workers/website-router/`; until that lands, its configuration lives only in Cloudflare" — it landed.
  - `website/src/data/catalog.ts:215-219` hides `agent-ui` with the comment "the Agent UI desktop
    app is no longer maintained", while `release_components.yml` still builds and publishes the
    Agent UI installers on every release and `CLAUDE.md`/`docs/guides/agent-ui.mdx` present it as
    a supported surface. One of the two is wrong; /hub/agent-ui is a 404 today.
- **Fix:** One doc sweep; decide the Agent UI status explicitly and either unhide it or stop
  publishing it.
- **Confidence:** High
- **Tracked:** none found

### [🟢] `install.sh` / `install.ps1` pull the uv installer unpinned and unverified
- **Where:** `installer/scripts/install.sh:172-180`, `installer/scripts/install.ps1:71`
- **What:** Both fetch `https://astral.sh/uv/install.{sh,ps1}` (latest) and execute it with no
  version pin or checksum, while every GAIA artifact the same scripts fetch is SHA-256-verified.
  TLS is the only control on that hop.
- **Failure scenario:** An upstream-compromised or simply broken `latest` uv installer runs on
  every fresh GAIA install worldwide the moment it is published; there is no way to roll back
  from GAIA's side.
- **Fix:** Use Astral's versioned URL (`https://astral.sh/uv/<version>/install.sh`) and bump it
  deliberately; optionally verify the installer's published checksum. Low effort, closes the one
  unverified download in an otherwise careful pipeline.
- **Confidence:** High
- **Tracked:** none found

## Test gaps

Worker (`workers/agent-hub/test`) — the suite is genuinely end-to-end against an in-memory R2
and asserts request *shape* (immutability, checksum, scope, 415/400/413 codes, by-reference
verification against the stored checksum, schema field lists). What it cannot see:
- **Concurrency / conditional writes:** `FakeR2.put` ignores `onlyIf` entirely, so a future
  fix that adds `etagMatches` would pass without proving anything; no test interleaves two
  publishes (finding 2).
- **Partial failure:** no test injects an R2 error after the artifact `put` and asserts a retry
  can still publish (finding 3).
- **Reserved filenames:** no test publishes an artifact named `README.md`/`gaia-agent.yaml` (finding 4).
- **Nested schema conformance:** the "declares every field" test compares top-level keys only;
  `skill_metadata.audit` extras go unnoticed; `manifest.schema.json` is never read by any test (findings 5-6).
- **`list()` paging:** `FakeR2.list` always returns `truncated: false`, so the cursor loops in
  `listAgentIds`/`listSkillNames` are untested.
- **Auth edge cases:** no test for `Bearer constructor`/`__proto__`, trailing-whitespace headers,
  or a `PUBLISH_TOKENS` entry with a non-array `authors` (silently becomes `[]` → every publish 403s
  with "Allowed authors: (none)" — correct but untested).
- **R2 contract vs fake:** the fake models `checksums.sha256` presence the way R2 documents it,
  but nothing in CI ever runs a real by-reference PUT against R2 (`test_publish_pipeline.yml` was
  not in scope); the "R2 only records SHA-256 for single-part uploads" premise is asserted in
  comments, not exercised. One `wrangler dev --remote`/preview-bucket smoke would close it.
- Suites could not be run here (no `node_modules`); CI runs them on `workers/agent-hub/**` changes only.

Website (`website/src/**/*.test.ts`) — good coverage of `download-target` (OS/arch/filename
mapping matches the live terminal-hub artifact names), markdown sanitisation, and lane filtering.
Gaps:
- No test that an install method the site renders corresponds to something that exists; the
  data needed (`artifacts[]` kinds) isn't in `index.json`, which is the root of finding 1.
- `getComponentRelease` (download buttons) is untested; a manifest with only legacy singular
  `artifact` (no `artifacts[]`) throws "publishes no artifacts" — real R2 data always has
  `artifacts[]`, so acceptable, but the installer scripts tolerate both shapes and the site does not.
- `robots.txt`/sitemap and the `HIDDEN_FROM_SITE` behaviour have no test; `website-ci.yml` only
  curls `/`, `/hub`, `/hub/email`.

Install scripts — `tests/unit/installer/test_install_scripts_terminal_hub.py` is solid
(fake hub, checksum mismatch, dash parsing, PATH idempotence). Run here: **27 passed, 15 skipped,
1 failed**:
```
FAILED test_sh_parses_under_dash — dash -n install.sh: 84: Syntax error: word unexpected (expecting "in")
```
Cause: `git ls-files --eol installer/scripts/install.sh` → `i/lf w/crlf` — the Windows checkout
converts the script to CRLF and `dash` rejects it. Not a repo bug, but there is no
`.gitattributes` forcing `*.sh eol=lf`, so anyone who serves/uploads the script from a Windows
checkout ships a file `sh` cannot run; the test should also skip (or normalise) on CRLF checkouts
instead of failing. The 15 skips need a POSIX shell / pwsh — the behavioural half of the suite
never runs on a Windows developer box.

## Documentation gaps

- `workers/agent-hub/README.md`: bucket name, "uncomment routes", missing `/reindex` +
  `REINDEX_TOKEN`, and the "Server-side SHA-256 … computed by the Worker from the bytes it
  received" guarantee should say "or, by reference, the SHA-256 R2 recorded at PUT time" in the
  guarantees list (it is explained only further down).
- `workers/agent-hub/schemas/manifest.schema.json` and `index.schema.json` contradict the code (findings 5-6).
- `workers/website-router/README.md` + `website/README.md`: neither documents the `/install.sh`
  and `/install.ps1` redirect rule or that the scripts are served from `main` (finding 7);
  website README still says the router source has not landed.
- `website/src/data/catalog.ts` comment claims the Agent UI is "no longer maintained"; the
  release pipeline and `docs/guides/agent-ui.mdx` say otherwise (finding "doc drift").
- `docs/quickstart.mdx:111,133` and `InstallCommand.astro` agree byte-for-byte on the one-liners
  (checked) — but neither says the scripts require `python3`/venv to parse the hub manifest
  (`install.sh:277-287` fails the install if no interpreter is available on a box where the venv
  creation failed silently earlier).
- `hub/agents/gaia/python/gaia-agent.yaml` declares no `npm_package` although `@amd-gaia/gaia`
  is published and the landing page advertises it — the manifest is the doc the hub page renders from.

## Improvement opportunities

- **Conditional R2 writes + single writer** (`storage.ts`): `onlyIf: { etagMatches }` on
  manifest/index puts, or a per-id Durable Object — turns the race in finding 2 into a retry.
- **Manifest as the single record of publication for both lanes** (`publish.ts:361`): removes the
  orphaned-artifact dead end (finding 3) and makes idempotent re-runs actually idempotent.
- **Reserved-filename guard** in `multipart.ts`: one `Set` + one test (finding 4).
- **Cache headers on immutable objects** (`index.ts:serveObject`): artifacts are immutable by
  contract yet served with no `Cache-Control` (verified live), so every installer download hits
  the Worker + R2; `public, max-age=31536000, immutable` for `agents/**/<file>` and a short
  `max-age` for `index.json`/`manifest.json` would cut Worker invocations and let clients
  revalidate on the ETag that is already emitted.
- **`400` for `URIError`, `Object.hasOwn`/`Map` for token lookup, `timingSafeEqual`** — three
  one-liners in `index.ts`/`auth.ts`.
- **Expose distribution kind in `index.json`** (`toIndexEntry`): `artifacts: [{filename, size}]`
  or `has_wheel/has_binaries` so the website (and the Agent UI hub panel) can stop inferring
  install methods from `language` — fixes finding 1 structurally.
- **Move the `/install.*` redirect into `website-router`** and pin it to the release tag;
  the router already has a clean place for it (`isDocs`-style predicate).
- **`.gitattributes` with `*.sh text eol=lf`** and a CRLF-aware skip in the dash test, so
  Windows contributors get green tests instead of a false failure.
- **`rebuildIndex` fan-out**: 9 sequential R2 reads per agent + 3 per skill on every publish and
  every `/reindex`; `Promise.all` per agent (or reading the docs only for `latest_version`
  changes) would cut publish latency ~5× and push the subrequest ceiling out (see Hypotheses).
- **Website `getComponentRelease`**: accept the legacy singular `artifact` like the installers do,
  so a hand-repaired manifest never breaks the landing page build.
- **`optionalTextPart` for `artifact_ref_sha256`**: accept upper-case hex (it `.toLowerCase()`s
  the value but the error message says "64 lowercase hex characters") — cosmetic.

## High-impact feature opportunities

1. **Signed audit attestation for skills (#1710)** — `audit.ts` is explicit that the gate is
   replay-proof but not forgery-proof and stores `attestation: "publisher-asserted"`. A CI-held
   signing key (or the Worker running `gaia skill audit` itself on the uploaded bundle) is the one
   thing that makes `security_tier: verified` mean something to a `gaia skill install` user.
   Effort: medium (key management + a `signer-attested` value the schema already anticipates).
2. **`POST /validate` (dry-run publish)** — every release workflow has a `worker-check` job
   because a stale Worker rejects valid manifests (#2959). A no-write endpoint that runs
   `parseManifest`/`parseSkillManifest` + scope + immutability checks would let CI fail in
   seconds before building 100 MB binaries, and give authors `gaia hub validate ./`. Effort: small.
3. **Catalog freshness for clients** — `src/gaia/hub/catalog.py` polls `index.json` on a TTL;
   the Worker already emits an ETag. Honouring `If-None-Match` (304) and adding
   `Cache-Control` would make `gaia hub` / the Agent UI hub panel cheap to poll and let the
   website build-time fetch drop its `?t=` cache-buster. Effort: small.
4. **Per-version pages and download history on the site** — `index.json` is latest-only, but
   `manifest.json` has every version with checksums; `/hub/<id>/<version>` with per-platform
   SHA-256s is what a security-conscious user needs to verify a download by hand, and what the
   email agent's 11 published versions currently have no UI for. Effort: small-medium (Astro
   `getStaticPaths` over versions).
5. **Decide the Agent UI's fate visibly** — it is published every release, hidden on the site,
   and called unmaintained in code. Either a hub page + install method for `type: app`
   (`InstallMethods` already has the download branch) or stop publishing 400 MB of installers per
   release. Effort: decision first, then small.

## Checked and fine

- **Path handling on GET**: WHATWG URL normalisation collapses `..`/`%2e%2e` before the
  `startsWith(prefix) && !includes("..")` guard; R2 keys are flat, so no traversal is possible;
  `/agents/%2e%2e/index.json` resolves to `/index.json` (verified live). `HEAD` works.
- **`/reindex`**: unauthenticated → `401` live; token separate from publish tokens as documented.
- **By-reference verification** (`verifyUploadedArtifact`): size + R2-recorded SHA-256 + size cap;
  refuses objects without a checksum (multipart); no URL is ever fetched — **no SSRF surface**.
  `publish_to_r2.py` uses `put_object` with `ChecksumSHA256` (single-part) as the README requires,
  and the live agent-ui artifacts (118-141 MiB) went through this lane.
- **Key construction**: `id` (`ID_RE`), `version` (`SEMVER_RE`), skill `name` (`SKILL_NAME_RE`)
  and `filename` (`ARTIFACT_FILENAME_RE`) are all validated before they become R2 keys; no
  separators, no traversal. `type: skill` on the agent lane is rejected with a pointer to the
  right route; cross-lane id namespace is enforced both ways.
- **Ownership/scope**: new agents require author scope; existing agents require the same author;
  skills are owned by publisher identity. Error messages don't echo the token.
- **Audit-gate binding** (`assertReportBinding`): gated tiers require `cleared_tiers`, `skill`,
  `version`, `manifest_digest`; digest normalises CRLF and is pinned by cross-language vectors
  (`tests/fixtures/skill_audit_digest/vectors.json`). BLOCK/REVIEW are refused before any write.
- **Version immutability**: per-filename, correct for both lanes; `upsertVersion` keeps
  `published_at`/`publisher` of the first publish and tracks display metadata to `latest`.
- **Markdown renderer** (`website/src/data/markdown.ts`): everything is `escapeHtml`'d before
  markup is emitted; the only `<a>`/`<img>` paths are scheme-allowlisted (`https?:`, `/`, `#`,
  `./`, `../`, `mailto:`; images `https:` or relative) with `"` neutralised; heading ids are
  `[a-z0-9-]`; fence language is sanitised; code-span placeholders use private-use characters so
  numbers in prose are safe. The file-preview modal sets `textContent`, never `innerHTML`; Prism
  reads `textContent`. No XSS found.
- **website-router**: hostname is fixed from config, path/query pass through unchanged, no
  redirects are emitted → no open redirect; `WEBSITE_ORIGIN` is validated by read-back; upstream
  errors surface as labelled 502s (no silent fallback); `/docs`, `/docs/` (308 from Mintlify),
  `/`, `/hub`, `/hub/`, `/hub/{email,gaia,terminal-hub}` all 200 live; the three hub links the
  listing renders all resolve.
- **Install scripts**: SHA-256 verification is mandatory (no hash tool → refuse), checksum
  mismatch is fatal even for the optional flagship sidecar; `curl -f`/`wget` status is honoured;
  no `sudo` anywhere; PATH is appended (sh) / de-duplicated user PATH (ps1); scratch dirs are
  cleaned by one EXIT trap; PS 5.1 is forced to TLS 1.2; the platform→filename mapping
  (`gaia-<os>-<arch>[.exe]`, `gaia-agent-<os>-<arch>` / `gaia-agent-win32-x64.exe`) matches
  every filename in the live `terminal-hub` and `gaia` manifests. `docs/quickstart.mdx` and
  `InstallCommand.astro` carry identical one-liners.
- **download-target.ts**: OS/arch resolution refuses to guess on macOS (offers both builds),
  rejects iPadOS-as-Mac via touch points, and `artifactFileName` produces exactly the six live
  binary names plus the documented installer names; every platform is server-rendered for no-JS.
- **Accessibility basics on the site**: skip link, visible `:focus-visible`, ARIA tablists with
  roving tabindex and live-region announcements, `aria-pressed` on filter pills, `<details>` for
  the file tree, lightbox focus trap/restore, `alt` on images, headings demoted so hub READMEs
  don't produce multiple `<h1>`s.
- **No browser consumer needs CORS today**: `index.json` is fetched at build time (Node), by the
  Python CLI, the Go TUI and the install scripts; the Agent UI reads through its backend
  (`hubLanes.ts` consumes `AgentInfo`), so the absent `Access-Control-Allow-Origin` is not a bug.
- **Fail-loudly config**: missing `PUBLISH_TOKENS`/`REINDEX_TOKEN`/`MAX_ARTIFACT_BYTES`
  misconfig, missing `HUB_CATALOG_URL`, missing `WEBSITE_ORIGIN`, missing `RAILWAY_TOKEN` and
  `GAIA_HUB_PUBLISH_URL` all produce named errors rather than defaults.

## Hypotheses (unverified)

- **Subrequest ceiling on `rebuildIndex`**: each publish costs ~8 own R2 calls + 2 lists +
  9 reads per agent + 3 per skill + 1 write (≈55 today with 4 agents). A Workers Free plan allows
  50 subrequests/request, Paid 1000 — the hub evidently runs Paid, but at ~100 catalog entries
  every publish and `/reindex` will start failing with "Too many subrequests", and by finding 3 it
  will do so *after* the artifact is stored.
- **Worker memory on the inline lane**: `request.formData()` + `file.arrayBuffer()` +
  `new Uint8Array(...)` hold up to three copies of an artifact; the 86.5 MiB
  `gaia-agent-win32-x64.exe` is published inline (below the 90 MiB by-reference threshold) against
  a 128 MiB isolate limit. It works today; a 95 MiB binary may not, and the failure would be the
  orphaned-artifact case.
- **The deployed `website-router` may not be this source** — the DYNAMIC cache status is
  consistent with either the `cf` options not applying to a third-party origin the way the README
  assumes, or an older deploy; `/health`-style build stamping (which the hub Worker has) would
  settle it.
- **`docker-entrypoint.sh` `.dev.vars` quoting**: a `PUBLISH_TOKENS` value containing `#`
  would be truncated by dotenv comment parsing (`printf 'PUBLISH_TOKENS=%s\n'`); tokens are
  usually hex so this is unlikely to bite, and the demo is documented as non-production.
- **`serve` on Railway for `/hub/<id>/`** (trailing slash on a detail page) — not probed; if it
  302s to the bare Railway hostname the visitor would leave `amd-gaia.ai`. `/hub/` itself was 200.

## Summary — top findings

1. 🔴 `/hub/gaia` advertises `pip install gaia-agent-gaia` (PyPI 404) because the site infers install methods from `language`, not from what was published; landing page says `npm install @amd-gaia/gaia` instead — pick one and gate the pip tab on real artifacts.
2. 🟡 Hub catalog writes have no conditional-put/serialisation and the inline publish is non-atomic: parallel release jobs can drop entries, and a failed publish leaves an artifact that 409s forever while never appearing in the catalog.
3. 🟡 Artifact filenames may equal the reserved doc objects (`gaia-agent.yaml`, `README.md`, `SKILL.md`, …) and get overwritten, so the recorded SHA-256 no longer matches the bytes served — installers then refuse the download.
4. 🟡 Both shipped JSON schemas contradict the Worker (skill `audit` block, `language` enum) and the only conformance test checks top-level keys; `install.sh`/`install.ps1` are served by an undocumented out-of-repo redirect to GitHub `main`, and the router's documented HTML edge cache is not happening in production.
5. 🟢 Prototype-key bearer tokens produce a public `500 server_misconfigured`, malformed `%`-paths are 500s, `robots.txt` points at a non-existent sitemap, the uv installer is fetched unpinned, and several README/code statements (bucket name, routes, Agent UI status, `/reindex`) are stale.
