// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { describe, expect, it } from "vitest";
import { SOURCES } from "../src/collector";
import { renderEvalsPage } from "../src/evalspage";
import { renderGrowthPage } from "../src/growthpage";
import { renderCiPage, renderDevelopersPage, renderUsersPage } from "../src/page";

/** The client script of a page; a syntax error there leaves the page blank in the browser. */
const scriptOf = (html: string) => html.split("<script>").pop()!.split("</script>")[0]!;

describe("page scripts", () => {
  const pages: [string, string][] = [
    ["users", renderUsersPage()],
    ["growth", renderGrowthPage([["alice", "alice-work"], ["bob"]])],
    ["developers", renderDevelopersPage([{ accounts: ["alice", "alice-work"], locked: false }, { accounts: ["bob"], locked: true }])],
    ["ci", renderCiPage()],
    ["evals", renderEvalsPage()],
  ];

  it.each(pages)("%s page script parses", (_name, html) => {
    expect(() => new Function(scriptOf(html))).not.toThrow();
  });

  it.each(pages)("%s page has no unfilled placeholders", (_name, html) => {
    expect(html).not.toMatch(/__[A-Z_]+__/);
  });

  it.each(pages)("%s page refreshes every collector source through the guarded endpoint", (_name, html) => {
    const script = scriptOf(html);
    expect(html).toContain('id="refresh"');
    expect(script).toContain(`var SOURCES=${JSON.stringify(SOURCES)};`);
    expect(script).toContain("'x-admin-action':'collect'");
  });
});
