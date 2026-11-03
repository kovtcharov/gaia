// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { beforeEach, describe, expect, it } from "vitest";
import { TokenError, collectEvals } from "../src/evals";
import worker from "../src/index";
import { handleApi } from "../src/api";
import { issueList, parseIssueQuery, type IssueRow } from "../src/issues";
import { loadCore } from "../src/people";
import { memoryD1 } from "./d1";

const baseEnv = { GITHUB_REPO: "amd/gaia", NPM_PACKAGE: "x", PYPI_PACKAGE: "y", ACCESS_TEAM_DOMAIN: "", ACCESS_AUD: "", ADMIN_DEV_NO_AUTH: "1" };
let DB: D1Database;
beforeEach(() => {
  DB = memoryD1();
});
const post = (path: string, action: string, body: unknown, extra: Record<string, string> = {}) =>
  worker.fetch(
    new Request(`http://localhost:8787/admin/api/${path}`, { method: "POST", headers: { "x-admin-action": action, ...extra }, body: JSON.stringify(body) }),
    { ...baseEnv, DB } as never,
  );

const push = (over: Record<string, unknown> = {}) => ({
  run: { id: 77, workflow: "Agent Eval — flagship", event: "push", ref: "v0.25.4-rc2", sha: "a".repeat(40), html_url: "https://github.com/amd/gaia/actions/runs/77" },
  lane: "smoke-1",
  tier: "smoke",
  agent_model: "Gemma-4-E4B-it-GGUF",
  minutes: 12,
  scorecards: [{ category: "gaia_files", scorecard: { scenarios: [{ scenario_id: "one", status: "PASS", overall_score: 9, turns: 3 }, { scenario_id: "two", status: "FAIL", turns: 2 }] } }],
  ...over,
});

describe("POST /api/ingest/evals", () => {
  it("stores a lane as it arrives, marked unsettled so the next collection reads it", async () => {
    expect((await post("ingest/evals", "ingest", push())).status).toBe(200);
    const run = await DB.prepare("SELECT is_tag, conclusion, run_attempt, tier FROM eval_runs WHERE run_id = 77").first<Record<string, unknown>>();
    expect(run).toEqual({ is_tag: 1, conclusion: null, run_attempt: 0, tier: "smoke" });
    const rows = (await DB.prepare("SELECT scenario_id, turns FROM eval_scenarios ORDER BY scenario_id").all()).results;
    expect(rows).toEqual([{ scenario_id: "one", turns: 3 }, { scenario_id: "two", turns: 2 }]);
  });

  it("counts lanes across pushes and replaces a lane that is pushed again", async () => {
    await post("ingest/evals", "ingest", push());
    await post("ingest/evals", "ingest", push({ lane: "smoke-2", scorecards: [{ category: "gaia_web", scorecard: { scenarios: [{ scenario_id: "three", status: "PASS", turns: 1 }] } }] }));
    await post("ingest/evals", "ingest", push({ scorecards: [{ category: "gaia_files", scorecard: { scenarios: [{ scenario_id: "one", status: "FAIL", turns: 1 }] } }] }));
    const n = await DB.prepare("SELECT COUNT(*) AS n FROM eval_scenarios").first<{ n: number }>();
    expect(n?.n).toBe(2);
    expect((await DB.prepare("SELECT lanes FROM eval_runs").first<{ lanes: number }>())?.lanes).toBe(2);
  });

  it("refuses a push without the action header, or with a bad run", async () => {
    expect((await post("ingest/evals", "collect", push())).status).toBe(403);
    expect((await post("ingest/evals", "ingest", push({ run: { id: 1 } }))).status).toBe(400);
    expect((await post("ingest/evals", "ingest", push({ run: { ...push().run, html_url: "https://evil.example/x" } }))).status).toBe(400);
  });
});

describe("collectEvals with a token that lacks Actions read", () => {
  it("stops at the first 403 with one message saying what to do", async () => {
    let downloads = 0;
    const run = (id: number) => ({ id, event: "push", head_branch: "main", head_sha: "b".repeat(40), created_at: "2026-11-01T00:00:00Z", updated_at: "2026-11-01T01:00:00Z", status: "completed", conclusion: "success", html_url: "u", name: "n", run_attempt: 1 });
    const f = async (url: string) => {
      if (url.includes("/workflows/eval_flagship.yml/")) return Response.json({ workflow_runs: [run(1), run(2), run(3)] });
      if (url.includes("/workflows/")) return Response.json({ workflow_runs: [] });
      if (url.endsWith("/zip")) {
        downloads++;
        return new Response("", { status: 403 });
      }
      return Response.json({ artifacts: [{ id: 9, name: "a", size_in_bytes: 10, expired: false }] });
    };
    await expect(collectEvals({ ...baseEnv, DB } as never, f as never, 14, new Date("2026-11-01"))).rejects.toBeInstanceOf(TokenError);
    expect(downloads).toBe(1);
  });
});

describe("core team", () => {
  const body = { "content-type": "application/json" };

  it("adds one person's accounts together and adds them to the page", async () => {
    expect((await post("core/add", "core", { entry: "kovtcharov+kovtcharov-amd" }, body)).status).toBe(200);
    expect(await loadCore(DB, undefined)).toEqual([{ accounts: ["kovtcharov", "kovtcharov-amd"], locked: false }]);
    const html = await (await worker.fetch(new Request("http://localhost:8787/admin/developers"), { ...baseEnv, DB } as never)).text();
    expect(html).toContain('CORE="kovtcharov+kovtcharov-amd"');
  });

  it("refuses a login already listed, including one set by the secret", async () => {
    await post("core/add", "core", { entry: "ann+ann-amd" }, body);
    const res = await post("core/add", "core", { entry: "ann-amd" }, body);
    expect(res.status).toBe(400);
    expect(await res.text()).toContain("already listed");
    expect((await loadCore(DB, "bob")).map((p) => p.locked)).toEqual([true, false]);
  });

  it("refuses anything that is not a login, and writes from another site", async () => {
    expect((await post("core/add", "core", { entry: '"><script>' }, body)).status).toBe(400);
    expect((await post("core/add", "core", { entry: "ann" }, { ...body, origin: "https://evil.example" })).status).toBe(403);
  });

  it("removes an entry", async () => {
    await post("core/add", "core", { entry: "ann" }, body);
    expect((await post("core/remove", "core", { entry: "ann" }, body)).status).toBe(200);
    expect(await loadCore(DB, undefined)).toEqual([]);
  });
});

describe("accounts joined with + count as one developer", () => {
  it("merges them in the contributor list and the commit-author count", async () => {
    await DB.prepare("INSERT INTO commits (sha, day, author, at) VALUES ('1', date('now','-1 day'), 'ann', datetime('now','-1 day')), ('2', date('now','-1 day'), 'ann-amd', datetime('now','-1 day'))").run();
    const url = new URL("http://localhost/admin/api/momentum?range=7&core=ann%2Bann-amd");
    const m = (await (await handleApi(DB, "/api/momentum", url)).json()) as { contributors: { author: string; commits: number }[]; summary: { contributors: number } };
    expect(m.contributors).toEqual([expect.objectContaining({ author: "ann", commits: 2 })]);
    expect(m.summary.contributors).toBe(1);
  });
});

describe("issue list", () => {
  const issue = (number: number, created_at: string, over: Partial<IssueRow> = {}): IssueRow => ({ number, created_at, closed_at: null, updated_at: created_at, assoc: "NONE", author: "carol", title: `#${number}`, comments: 0, labels: "[]", ...over });
  const rows = [
    issue(1, "2026-01-01", { comments: 5 }),
    issue(2, "2026-02-01", { assoc: "MEMBER", author: "ann", comments: 9 }),
    issue(3, "2026-03-01", { closed_at: "2026-03-02" }),
    issue(4, "2026-04-01", { author: "dependabot[bot]" }),
    issue(5, "2026-05-01", { author: "bob-amd", updated_at: "2026-06-01" }),
  ];
  const list = (q: string, core = ["bob-amd"]) => issueList(rows, parseIssueQuery(new URLSearchParams(q)), new Set(core)).issues.map((i) => i.number);

  it("sorts oldest first by default and open only", () => expect(list("")).toEqual([1, 2, 4, 5]));
  it("sorts by comments and by last update", () => {
    expect(list("sort=comments")[0]).toBe(2);
    expect(list("sort=updated")[0]).toBe(5);
    expect(list("sort=newest")[0]).toBe(5);
  });
  it("third parties leave out the core team, listed maintainers and bots", () => expect(list("who=outside")).toEqual([1]));
  it("core team means members and listed maintainers, not bots", () => expect(list("who=core")).toEqual([2, 5]));
  it("finds issues nobody has answered", () => expect(list("noreply=1")).toEqual([4, 5]));
  it("refuses an unknown sort rather than guessing", () => expect(() => list("sort=stars")).toThrow(/sort must be one of/));
});
