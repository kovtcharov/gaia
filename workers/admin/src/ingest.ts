// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Eval results pushed from CI as each lane finishes (util/push_eval_results.py),
 * so a run shows up while it is still going and a run the nightly read never
 * reaches is not lost. A pushed run keeps run_attempt 0 and no conclusion, which
 * makes the next collection read its artifacts and settle it.
 */

import { insertMany, writeAtomic } from "./collector";
import { parseAgentScorecard, parseTasksScorecard, TAG_REF, type ScenarioRow } from "./evals";

export class IngestError extends Error {}

export interface LanePush {
  run: { id: number; workflow: string; event: string; ref: string | null; sha: string; html_url: string };
  lane: string;
  tier?: string | null;
  agent_model?: string | null;
  judge_model?: string | null;
  minutes?: number | null;
  scorecards: { category: string; scorecard: Record<string, unknown> }[];
}

const SHA = /^[0-9a-f]{40}$/;

function validate(b: LanePush): void {
  const r = b?.run;
  if (!r || !Number.isSafeInteger(r.id) || r.id <= 0) throw new IngestError("run.id must be the GitHub Actions run id.");
  if (typeof r.workflow !== "string" || !r.workflow || typeof r.event !== "string" || !r.event) throw new IngestError("run.workflow and run.event are required.");
  if (typeof r.sha !== "string" || !SHA.test(r.sha)) throw new IngestError("run.sha must be a full commit id.");
  if (typeof r.html_url !== "string" || !r.html_url.startsWith("https://github.com/")) throw new IngestError("run.html_url must be a github.com URL.");
  if (typeof b.lane !== "string" || !b.lane) throw new IngestError("lane is required.");
  if (!Array.isArray(b.scorecards)) throw new IngestError("scorecards must be a list of {category, scorecard}.");
}

export async function ingestLane(db: D1Database, body: LanePush): Promise<string> {
  validate(body);
  const rows: ScenarioRow[] = body.scorecards.flatMap(({ category, scorecard }) =>
    Array.isArray(scorecard.tasks) ? parseTasksScorecard(scorecard) : parseAgentScorecard(scorecard, category, body.lane),
  );
  const r = body.run;
  const now = new Date().toISOString();
  const lanes = (await db.prepare("SELECT COUNT(DISTINCT lane) AS n FROM eval_scenarios WHERE run_id = ? AND lane != ?").bind(r.id, body.lane).first<{ n: number }>())?.n ?? 0;
  await writeAtomic(db, [
    db.prepare("DELETE FROM eval_scenarios WHERE run_id = ? AND lane = ?").bind(r.id, body.lane),
    ...insertMany(db, "eval_scenarios",
      ["run_id", "category", "scenario_id", "lane", "status", "score", "elapsed_s", "cost_usd", "tps", "ttft_s", "input_tokens", "output_tokens", "tool_calls", "turns", "root_cause", "kind"],
      rows.map((s) => [r.id, s.category, s.scenario_id, body.lane, s.status, s.score, s.elapsed_s, s.cost_usd, s.tps, s.ttft_s, s.input_tokens, s.output_tokens, s.tool_calls, s.turns, s.root_cause, s.kind])),
    db.prepare(
      `INSERT INTO eval_runs (run_id, workflow, event, ref, is_tag, sha, created_at, conclusion, tier, agent_model, judge_model, minutes, lanes, html_url, collected_at, run_attempt)
       VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, 0)
       ON CONFLICT (run_id) DO UPDATE SET conclusion = NULL, run_attempt = 0, tier = COALESCE(excluded.tier, tier), agent_model = COALESCE(excluded.agent_model, agent_model),
         judge_model = COALESCE(excluded.judge_model, judge_model), minutes = MAX(COALESCE(minutes, 0), COALESCE(excluded.minutes, 0)), lanes = excluded.lanes, collected_at = excluded.collected_at`,
    ).bind(r.id, r.workflow, r.event, r.ref, TAG_REF.test(r.ref ?? "") ? 1 : 0, r.sha, now, body.tier ?? null, body.agent_model ?? null, body.judge_model ?? null, body.minutes ?? null, lanes + 1, r.html_url, now),
  ]);
  return `${rows.length} results for lane ${body.lane} of run ${r.id}`;
}
