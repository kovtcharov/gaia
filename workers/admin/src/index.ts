// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

import { handleApi } from "./api";
import { authenticate, AuthError, type AuthEnv } from "./auth";
import { collectAll, SOURCES, unknownSources, type CollectEnv } from "./collector";
import { renderEvalsPage } from "./evalspage";
import { IngestError, ingestLane, type LanePush } from "./ingest";
import { addCore, CoreError, loadCore, removeCore } from "./people";
import { renderGrowthPage } from "./growthpage";
import { renderCiPage, renderDevelopersPage, renderUsersPage } from "./page";

type Env = CollectEnv & AuthEnv & { MAINTAINER_LOGINS?: string };

const PAGES: Record<string, (env: Env) => string | Promise<string>> = {
  "/": renderUsersPage,
  "/growth": async (env) => renderGrowthPage((await loadCore(env.DB, env.MAINTAINER_LOGINS)).map((p) => p.accounts)),
  "/developers": async (env) => renderDevelopersPage(await loadCore(env.DB, env.MAINTAINER_LOGINS)),
  "/ci": renderCiPage,
  "/evals": renderEvalsPage,
};

const text = (body: string, status: number, type = "text/plain"): Response =>
  new Response(body, { status, headers: { "content-type": `${type}; charset=utf-8`, "cache-control": "no-store" } });

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);
    if (url.pathname !== "/admin" && !url.pathname.startsWith("/admin/")) return text("Not found", 404);
    let user: string;
    try {
      user = await authenticate(req, env);
    } catch (e) {
      if (e instanceof AuthError) return text(e.message, e.status);
      throw e;
    }

    const path = url.pathname.slice("/admin".length) || "/";
    if (req.method === "GET") {
      const page = PAGES[path];
      if (page) return text(await page(env), 200, "text/html");
    }

    // Writes must be deliberate (curl, a script, the page's own buttons), never a cross-site form riding a signed-in browser.
    const deliberate = (action: string): boolean => {
      const origin = req.headers.get("origin");
      const sameSite = (() => {
        if (!origin) return true;
        try {
          return new URL(origin).hostname === url.hostname; // wrangler dev drops the port from Origin
        } catch {
          return false; // "null" (sandboxed frames, file:) or anything unparseable is not this site
        }
      })();
      return req.headers.get("x-admin-action") === action && sameSite;
    };

    if (path === "/api/ingest/evals" && req.method === "POST") {
      if (!deliberate("ingest")) return text("Refused: send the header 'x-admin-action: ingest' (see README).", 403);
      try {
        return text(await ingestLane(env.DB, (await req.json()) as LanePush), 200);
      } catch (e) {
        if (e instanceof IngestError || e instanceof SyntaxError) return text(`Refused: ${e.message}`, 400);
        throw e;
      }
    }

    if ((path === "/api/core/add" || path === "/api/core/remove") && req.method === "POST") {
      if (!deliberate("core")) return text("Refused: send the header 'x-admin-action: core' from the same site (see README).", 403);
      try {
        const { entry } = (await req.json()) as { entry?: unknown };
        if (typeof entry !== "string" || !entry.trim()) throw new CoreError("entry is required, for example alice or alice+alice-work.");
        if (path === "/api/core/add") await addCore(env.DB, env.MAINTAINER_LOGINS, entry, user);
        else await removeCore(env.DB, entry);
      } catch (e) {
        if (e instanceof CoreError || e instanceof SyntaxError) return text(e.message, 400);
        throw e;
      }
      return text("ok", 200);
    }

    if (path === "/api/collect" && req.method === "POST") {
      if (!deliberate("collect")) {
        return text("Refused: send the header 'x-admin-action: collect' from the same site (see README).", 403);
      }
      const days = Number(url.searchParams.get("days") ?? "3");
      const skip = Number(url.searchParams.get("skip") ?? "0");
      if (!Number.isInteger(days) || days < 1 || days > 95 || !Number.isInteger(skip) || skip < 0 || skip > 95) {
        return text("days must be 1-95 and skip 0-95 (whole numbers).", 400);
      }
      const only = url.searchParams.get("only")?.split(",").map((x) => x.trim()).filter(Boolean);
      const unknown = unknownSources(only);
      if (unknown.length) return text(`Unknown source(s): ${unknown.join(", ")}. Known: ${SOURCES.join(", ")}.`, 400);
      try {
        await collectAll(env, { runDays: days, skipDays: skip, fullHistory: url.searchParams.get("full") === "1", only });
      } catch (e) {
        return text(`Collection finished with errors:\n${(e as Error).message}`, 502);
      }
      return text("ok", 200);
    }

    if (req.method === "GET" && path.startsWith("/api/")) return handleApi(env.DB, path, url);
    return text("Not found", 404);
  },

  async scheduled(_event: ScheduledEvent, env: Env): Promise<void> {
    await collectAll(env, { runDays: 3 });
  },
};
