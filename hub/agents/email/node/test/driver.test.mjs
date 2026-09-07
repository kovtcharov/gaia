/**
 * End-to-end test for the offline eval driver.
 *
 * Spawns `node eval/driver.js` against a stub OpenAI-compatible server, so the
 * whole contract the Python harness depends on — stdin request, stdout JSON,
 * per-call `/stats` harvesting, id echo, error isolation — is exercised without
 * a model, a GPU, or a network. If this passes, `gaia eval benchmark --impl
 * node` can only fail for reasons outside the driver.
 */

import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DRIVER = path.join(HERE, "..", "eval", "driver.js");

/** Canned tool arguments per tool name, so any pipeline order is satisfied. */
const TOOL_ARGS = {
  classify_email: {
    category: "NEEDS_RESPONSE",
    reason: "asks a direct question",
    is_spam: false,
    is_phishing: false,
  },
  extract_due_date: { has_deadline: false },
  select_actions: { actions: ["send_reply"] },
  compose_draft: { to: [{ email: "dana@acme.io" }], body: "Sounds good." },
  extract_links: { items: [] },
  extract_calendar_event: { title: "Sync", start: "2026-09-01T15:00:00Z" },
  extract_unsubscribe: { url: "https://example.invalid/u" },
  set_summary: { summary: "Dana asks about the roadmap." },
  extract_actions: { items: [] },
};

let server;
let baseUrl;
let statsHits = 0;
let completionHits = 0;

beforeAll(async () => {
  server = createServer((req, res) => {
    if (req.method === "GET" && req.url.endsWith("/stats")) {
      statsHits += 1;
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          time_to_first_token: 0.25,
          tokens_per_second: 42.5,
          input_tokens: 100,
          output_tokens: 20,
          decode_token_times: [],
        })
      );
      return;
    }
    let body = "";
    req.on("data", (c) => {
      body += c;
    });
    req.on("end", () => {
      completionHits += 1;
      const parsed = JSON.parse(body || "{}");
      const tool = parsed.tools?.[0]?.function?.name;
      const usage = {
        prompt_tokens: 100,
        completion_tokens: 20,
        total_tokens: 120,
      };
      const message = tool
        ? {
            role: "assistant",
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: tool,
                  arguments: JSON.stringify(TOOL_ARGS[tool] ?? {}),
                },
              },
            ],
          }
        : { role: "assistant", content: "Dana asks about the roadmap." };
      res.writeHead(200, { "content-type": "application/json" });
      res.end(
        JSON.stringify({
          id: "chatcmpl-stub",
          choices: [{ index: 0, message, finish_reason: "stop" }],
          usage,
        })
      );
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}/api/v1`;
});

afterAll(async () => {
  await new Promise((resolve) => server.close(resolve));
});

/**
 * Run the driver with `request` on stdin and return its parsed stdout.
 *
 * @param {object} request
 * @returns {Promise<{code: number, out: any, stderr: string}>}
 */
function runDriver(request) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [DRIVER], {
      cwd: path.join(HERE, ".."),
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (c) => {
      stdout += c;
    });
    child.stderr.on("data", (c) => {
      stderr += c;
    });
    child.on("error", reject);
    child.on("close", (code) => {
      resolve({
        code,
        stderr,
        out: stdout.trim() ? JSON.parse(stdout) : null,
      });
    });
    child.stdin.end(JSON.stringify(request));
  });
}

const baseRequest = (items) => ({
  base_url: baseUrl,
  model: "stub-model",
  collect_stats: true,
  now: "2026-08-19T00:00:00Z",
  principal: { email: "me@acme.io", name: "Me" },
  items,
});

const humanEmail = (id) => ({
  message_id: id,
  from: { email: "dana@acme.io", name: "Dana Kim" },
  to: [{ email: "me@acme.io" }],
  subject: "Quick question about the roadmap",
  body: "Do you have five minutes tomorrow to talk about Q4?",
  date: "2026-08-18T09:00:00Z",
});

describe("eval driver", () => {
  it("triages a message and echoes the caller's id", async () => {
    const { code, out } = await runDriver(baseRequest([humanEmail("gid-abc")]));
    expect(code).toBe(0);
    expect(out.results).toHaveLength(1);
    expect(out.results[0].id).toBe("gid-abc");
    expect(out.results[0].category).toBe("NEEDS_RESPONSE");
    expect(out.results[0].is_spam).toBe(false);
    expect(out.results[0].is_phishing).toBe(false);
    expect(typeof out.results[0].summary).toBe("string");
    expect(Array.isArray(out.results[0].action_items)).toBe(true);
  });

  it("harvests one /stats reading per LLM call", async () => {
    statsHits = 0;
    completionHits = 0;
    const { out } = await runDriver(baseRequest([humanEmail("gid-stats")]));
    expect(out.llm_call_count).toBeGreaterThan(0);
    expect(out.stats).toHaveLength(out.llm_call_count);
    expect(out.stats[0].time_to_first_token).toBe(0.25);
    expect(out.stats[0].tokens_per_second).toBe(42.5);
    // Every stats entry is tagged with the step that produced it, so a slow
    // stage is attributable rather than lost in an average.
    expect(out.stats.every((s) => typeof s.step === "string")).toBe(true);
  });

  it("spends far fewer LLM calls on a heuristic-confident promotional email", async () => {
    // The cost story of this SDK: a confident heuristic skips the classifier
    // AND the two-phase action extraction, leaving only the summary. Pinning
    // the ratio catches a regression that quietly triples inference cost.
    const promo = await runDriver(
      baseRequest([
        {
          ...humanEmail("gid-promo"),
          subject: "This week in AI",
          from: { email: "news@letters.example" },
          body: "Lots of news. To stop these, unsubscribe at https://letters.example/u/9",
        },
      ])
    );
    const human = await runDriver(baseRequest([humanEmail("gid-human")]));

    expect(promo.out.results[0].category).toBe("PROMOTIONAL");
    expect(promo.out.llm_call_count).toBe(1);
    expect(promo.out.llm_call_count).toBeLessThan(human.out.llm_call_count);
    expect(promo.out.stats).toHaveLength(1);
  });

  it("forces the LLM classifier when asked", async () => {
    const { out } = await runDriver({
      ...baseRequest([
        {
          ...humanEmail("gid-forced"),
          subject: "This week in AI",
          from: { email: "news@letters.example" },
          body: "Lots of news. To stop these, unsubscribe at https://letters.example/u/9",
        },
      ]),
      force_llm_classify: true,
    });
    expect(out.llm_call_count).toBeGreaterThan(0);
  });

  it("reports totals across a multi-item batch", async () => {
    const { out } = await runDriver(
      baseRequest([humanEmail("a"), humanEmail("b"), humanEmail("c")])
    );
    expect(out.results.map((r) => r.id)).toEqual(["a", "b", "c"]);
    expect(out.usage.total_tokens).toBeGreaterThan(0);
    expect(out.duration_ms).toBeGreaterThanOrEqual(0);
  });

  it("names the missing field when the request is incomplete", async () => {
    const { code, stderr } = await runDriver({ model: "stub-model" });
    expect(code).toBe(1);
    expect(stderr).toContain("base_url");
  });

  it("fails loudly when stdin carries no request", async () => {
    const child = spawn(process.execPath, [DRIVER], {
      cwd: path.join(HERE, ".."),
    });
    let stderr = "";
    child.stderr.on("data", (c) => {
      stderr += c;
    });
    child.stdin.end("");
    const code = await new Promise((resolve) => child.on("close", resolve));
    expect(code).toBe(1);
    expect(stderr).toContain("no request on stdin");
  });
});
