// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/** The Developers page's issue list: which issues to read first, by age, activity or who filed them. */

import { isExternal } from "./growth";
import { isBot } from "./momentum";

export interface IssueRow {
  number: number;
  created_at: string;
  closed_at: string | null;
  updated_at: string | null;
  assoc: string;
  author: string | null;
  title: string | null;
  comments: number | null;
  labels: string | null;
}

const SORTS = ["oldest", "newest", "comments", "updated"] as const;
const WHO = ["all", "outside", "core"] as const;
const STATES = ["open", "closed", "all"] as const;
export type IssueQuery = { sort: (typeof SORTS)[number]; who: (typeof WHO)[number]; state: (typeof STATES)[number]; noReply: boolean };

const choose = <T extends string>(v: string | null, ok: readonly T[], name: string): T => {
  if (v === null) return ok[0]!;
  if (!(ok as readonly string[]).includes(v)) throw new RangeError(`${name} must be one of ${ok.join(", ")} (got "${v}").`);
  return v as T;
};

export function parseIssueQuery(p: URLSearchParams): IssueQuery {
  return {
    sort: choose(p.get("sort"), SORTS, "sort"),
    who: choose(p.get("who"), WHO, "who"),
    state: choose(p.get("state"), STATES, "state"),
    noReply: p.get("noreply") === "1",
  };
}

const ORDER: Record<IssueQuery["sort"], (a: IssueRow, b: IssueRow) => number> = {
  oldest: (a, b) => a.created_at.localeCompare(b.created_at),
  newest: (a, b) => b.created_at.localeCompare(a.created_at),
  comments: (a, b) => (b.comments ?? 0) - (a.comments ?? 0) || a.created_at.localeCompare(b.created_at),
  updated: (a, b) => (b.updated_at ?? "").localeCompare(a.updated_at ?? ""),
};

export function issueList(rows: IssueRow[], q: IssueQuery, core: Set<string>, limit = 100) {
  const kept = rows.filter((r) => {
    const outside = isExternal({ ...r, kind: "issue", merged_at: null }, core);
    if (q.who === "outside" && !outside) return false;
    if (q.who === "core" && (outside || isBot(r.author))) return false;
    if (q.state === "open" && r.closed_at) return false;
    if (q.state === "closed" && !r.closed_at) return false;
    return !q.noReply || !r.comments;
  });
  return {
    total: kept.length,
    issues: [...kept].sort(ORDER[q.sort]).slice(0, limit).map((r) => ({
      number: r.number,
      title: r.title,
      author: r.author,
      outside: isExternal({ ...r, kind: "issue", merged_at: null }, core),
      created_at: r.created_at,
      updated_at: r.updated_at,
      closed_at: r.closed_at,
      comments: r.comments,
      labels: r.labels ? (JSON.parse(r.labels) as string[]) : [],
    })),
  };
}
