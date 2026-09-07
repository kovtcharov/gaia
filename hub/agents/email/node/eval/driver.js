/**
 * Offline eval driver for the Node email-triage SDK.
 *
 * Reads ONE JSON request from stdin, triages every item through
 * `EmailTriageAgent`, and writes ONE JSON response to stdout. Every log line
 * goes to stderr so stdout stays a clean JSON document.
 *
 * This exists so the committed Python eval harness (`gaia eval benchmark`) can
 * score this SDK without importing it: the harness owns the corpus, the ground
 * truth, and the scoring; this driver owns nothing but "run the agent".
 *
 * Usage:
 *   node eval/driver.js < request.json > response.json
 *
 * Request shape (all fields required unless marked optional):
 *   {
 *     "base_url": "http://localhost:8000/api/v1",
 *     "model": "Gemma-4-E4B-it-GGUF",
 *     "ctx_size": 16384,                  // optional, default 16384
 *     "force_llm_classify": false,        // optional, default false
 *     "collect_stats": true,              // optional, default false
 *     "timeout_ms": 120000,               // optional
 *     "now": "2026-08-19T00:00:00Z",      // optional, pins due-date extraction
 *     "user_context": "…",                // optional
 *     "principal": { "email": "…", "name": "…" },
 *     "aliases": ["…"],                   // optional
 *     "items": [
 *       { "message_id": "…", "from": {"email","name"}, "to": [...], "cc": [...],
 *         "subject": "…", "body": "…", "date": "…" }
 *     ]
 *   }
 *
 * Response shape:
 *   {
 *     "schema_version": "2.2",
 *     "results": [ { "id", "category", "is_spam", "is_phishing", "summary",
 *                    "action_items", "due" } ],
 *     "skipped": [ { "id", "reason" } ],
 *     "errors":  [ { "id", "message" } ],
 *     "usage": { "prompt_tokens", "completion_tokens", "total_tokens" },
 *     "llm_call_count": N,
 *     "stats": [ <verbatim inference-server /stats payload>, … ],
 *     "duration_ms": N
 *   }
 *
 * Fail-loudly: a malformed request, an unreachable server, or a per-item
 * failure is reported explicitly. Per-item failures land in `errors` (so one bad
 * email never voids a 300-email run) and are counted in the exit summary; a
 * request-level problem exits non-zero with a message on stderr.
 */

const { EmailTriageAgent } = require("../core/agent.js");
const { SCHEMA_VERSION } = require("../utils/types.js");

/** Read all of stdin as a UTF-8 string. */
function readStdin() {
  return new Promise((resolve, reject) => {
    let buf = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (chunk) => {
      buf += chunk;
    });
    process.stdin.on("end", () => resolve(buf));
    process.stdin.on("error", reject);
  });
}

/**
 * Validate the request, naming the missing field rather than failing later
 * with an opaque TypeError deep inside the agent.
 *
 * @param {any} req
 * @returns {void}
 */
function validateRequest(req) {
  if (!req || typeof req !== "object") {
    throw new Error("request must be a JSON object");
  }
  for (const field of ["base_url", "model", "principal", "items"]) {
    if (req[field] === undefined || req[field] === null) {
      throw new Error(
        `request is missing required field "${field}". ` +
          "See the header of eval/driver.js for the full request shape."
      );
    }
  }
  if (!Array.isArray(req.items)) {
    throw new Error(`request "items" must be an array, got ${typeof req.items}`);
  }
  if (!req.principal.email) {
    throw new Error('request "principal" must carry an "email"');
  }
}

/**
 * Fetch the inference server's last-request stats.
 *
 * Returns the payload verbatim — the Python harness knows how to read a
 * Lemonade `/stats` body and this driver deliberately does not reinterpret it.
 * A non-OK response or transport error is surfaced as a `stats_error` entry
 * rather than throwing: losing telemetry must not void an otherwise good
 * triage run, but it must be visible in the output.
 *
 * @param {string} baseUrl
 * @param {string} step
 * @returns {Promise<object>}
 */
async function fetchStats(baseUrl, step) {
  const url = `${baseUrl.replace(/\/+$/, "")}/stats`;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      return { stats_error: `GET ${url} -> HTTP ${res.status}`, step };
    }
    const body = await res.json();
    return { ...body, step };
  } catch (err) {
    return { stats_error: `GET ${url} -> ${err.message}`, step };
  }
}

/**
 * Build the agent's per-item input from a driver request item.
 *
 * @param {object} item
 * @param {object} principal
 * @returns {object} an `EmailInput` of kind "single"
 */
function toEmailInput(item, principal) {
  return {
    kind: "single",
    principal,
    message: {
      message_id: item.message_id,
      thread_id: item.thread_id,
      from: item.from,
      to: item.to ?? [],
      cc: item.cc ?? [],
      subject: item.subject ?? "",
      body: item.body ?? "",
      date: item.date,
    },
  };
}

/**
 * Route `console.log` to stderr for the life of the process.
 *
 * `EmailTriageAgent.triageBatch` narrates per-item progress with `console.log`,
 * which lands on stdout — and stdout is this driver's JSON channel. Without
 * this the harness gets `[batch 1/3] ▶ …` prepended to its JSON and fails to
 * parse. Redirect rather than silence: the progress log is genuinely useful on
 * a 300-email run, it just belongs on the log stream.
 */
function routeConsoleToStderr() {
  const write = (...args) =>
    process.stderr.write(
      args
        .map((a) => (typeof a === "string" ? a : JSON.stringify(a)))
        .join(" ") + "\n"
    );
  console.log = write;
  console.info = write;
  console.warn = write;
}

async function main() {
  routeConsoleToStderr();
  const raw = await readStdin();
  if (!raw.trim()) {
    throw new Error(
      "no request on stdin — pipe a JSON request in, e.g. " +
        "`node eval/driver.js < request.json`"
    );
  }

  let req;
  try {
    req = JSON.parse(raw);
  } catch (err) {
    throw new Error(`request on stdin is not valid JSON: ${err.message}`);
  }
  validateRequest(req);

  const collectStats = req.collect_stats ?? false;
  /** @type {object[]} */
  const stats = [];
  let llmCallCount = 0;

  const agent = new EmailTriageAgent({
    baseUrl: req.base_url,
    apiKey: req.api_key,
    model: req.model,
    timeoutMs: req.timeout_ms,
    contextWindowTokens: req.ctx_size ?? 16384,
    forceLlmClassify: req.force_llm_classify ?? false,
    now: req.now,
    userContext: req.user_context,
    aliases: req.aliases ?? [req.principal.email],
    // Serial on purpose: concurrent calls make per-call latency and the
    // server's last-request /stats readings meaningless.
    batchConcurrency: 1,
    debug: req.debug ?? false,
    onCall: async ({ step }) => {
      llmCallCount += 1;
      if (collectStats) stats.push(await fetchStats(req.base_url, step));
    },
  });

  const items = req.items.map((item) => toEmailInput(item, req.principal));
  process.stderr.write(
    `[driver] triaging ${items.length} item(s) via ${req.model} @ ${req.base_url}\n`
  );

  const startedAt = Date.now();
  const batch = await agent.triageBatch({ items, context: req.context });
  const durationMs = Date.now() - startedAt;

  const results = [];
  const skipped = [];
  const errors = [];
  let promptTokens = 0;
  let completionTokens = 0;
  let totalTokens = 0;

  for (const entry of batch.results) {
    const id = req.items[entry.index]?.message_id ?? `index-${entry.index}`;
    if (entry.error) {
      errors.push({ id, message: entry.error.message });
      continue;
    }
    if (entry.skipped) {
      skipped.push({ id, reason: entry.skipped.reason });
      continue;
    }
    const r = entry.result;
    const usage = r.usage ?? {};
    promptTokens += usage.prompt_tokens ?? 0;
    completionTokens += usage.completion_tokens ?? 0;
    totalTokens += usage.total_tokens ?? 0;
    results.push({
      // `id` is what the harness joins on — echo the caller's message_id
      // rather than anything the agent derived.
      id,
      category: r.category,
      is_spam: Boolean(r.is_spam),
      is_phishing: Boolean(r.is_phishing),
      summary: r.summary ?? "",
      action_items: r.action_items ?? [],
      ...(r.due ? { due: r.due } : {}),
    });
  }

  process.stderr.write(
    `[driver] done in ${durationMs}ms — ${results.length} triaged, ` +
      `${skipped.length} skipped, ${errors.length} errored, ` +
      `${llmCallCount} LLM call(s)\n`
  );
  for (const e of errors) {
    process.stderr.write(`[driver] ERROR ${e.id}: ${e.message}\n`);
  }

  process.stdout.write(
    JSON.stringify({
      schema_version: SCHEMA_VERSION,
      results,
      skipped,
      errors,
      usage: {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: totalTokens,
      },
      llm_call_count: llmCallCount,
      stats,
      duration_ms: durationMs,
    })
  );
}

if (require.main === module) {
  main().catch((err) => {
    process.stderr.write(`[driver] FATAL: ${err.stack || err.message}\n`);
    process.exit(1);
  });
}

module.exports = { validateRequest, toEmailInput, fetchStats, main };
