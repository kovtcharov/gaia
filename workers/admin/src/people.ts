// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * The core team: people edited on the Developers page plus any set in the
 * MAINTAINER_LOGINS secret. Each person may have several accounts, written
 * alice+alice-work, and those accounts count as one developer everywhere.
 */

import { parseMaintainerGroups } from "./page";

export interface CorePerson {
  accounts: string[];
  /** Set by the deploy secret, so it cannot be removed from the page. */
  locked: boolean;
}

export async function loadCore(db: D1Database, secret: string | undefined): Promise<CorePerson[]> {
  const stored = (await db.prepare("SELECT entry FROM core_people ORDER BY entry").all<{ entry: string }>()).results;
  return [
    ...parseMaintainerGroups(secret).map((accounts) => ({ accounts, locked: true })),
    ...stored.map((r) => ({ accounts: parseMaintainerGroups(r.entry)[0]!, locked: false })),
  ];
}

export class CoreError extends Error {}

/** Add one person. A login already listed is refused: one account cannot belong to two people. */
export async function addCore(db: D1Database, secret: string | undefined, raw: string, by: string): Promise<void> {
  let groups: string[][];
  try {
    groups = parseMaintainerGroups(raw);
  } catch (e) {
    throw new CoreError((e as Error).message);
  }
  if (groups.length !== 1) throw new CoreError("Add one person at a time. Join one person's accounts with +, for example alice+alice-work.");
  const accounts = groups[0]!;
  const taken = new Set((await loadCore(db, secret)).flatMap((p) => p.accounts.map((a) => a.toLowerCase())));
  const dup = accounts.filter((a) => taken.has(a.toLowerCase()));
  if (dup.length) throw new CoreError(`${dup.join(", ")} is already listed. Remove that entry first, then add the accounts together.`);
  await db.prepare("INSERT INTO core_people (entry, added_by, added_at) VALUES (?, ?, ?)").bind(accounts.join("+"), by, new Date().toISOString()).run();
}

export async function removeCore(db: D1Database, entry: string): Promise<void> {
  const r = await db.prepare("DELETE FROM core_people WHERE entry = ?").bind(entry).run();
  if (r.meta && (r.meta as { changes?: number }).changes === 0) throw new CoreError(`${entry} is not in the list, or is set by the MAINTAINER_LOGINS secret.`);
}
