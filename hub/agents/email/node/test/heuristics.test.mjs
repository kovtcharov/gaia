/**
 * Offline unit tests for the zero-LLM parts of the triage pipeline.
 *
 * These cover the code paths that decide whether an email reaches the model at
 * all, so a regression here silently changes both cost and accuracy for every
 * run. No network, no model — they are safe on any runner.
 */

import { describe, it, expect } from "vitest";
import {
  classifyHeuristic,
  extractSigningUrl,
  extractUnsubscribeUrl,
} from "../utils/heuristics.js";
import { rankActions } from "../utils/rank.js";
import { mergeUsage } from "../core/llm.js";

/** @param {string} email @param {string} [name] */
const addr = (email, name) => ({ email, name });

describe("classifyHeuristic", () => {
  it("flags a phishing token as phishing, urgent, and confident", () => {
    const r = classifyHeuristic(
      "Action needed: confirm your password",
      addr("security@paypa1.example"),
      "Click to continue."
    );
    expect(r.is_phishing).toBe(true);
    expect(r.category).toBe("URGENT");
    expect(r.confident).toBe(true);
  });

  it("flags a spam subject token as spam and promotional", () => {
    const r = classifyHeuristic(
      "Congratulations! claim your prize",
      addr("promo@spam.example"),
      "Body text."
    );
    expect(r.is_spam).toBe(true);
    expect(r.category).toBe("PROMOTIONAL");
    expect(r.confident).toBe(true);
  });

  it("scans only the subject for spam tokens, not the body", () => {
    // Guards the cost/accuracy trade-off the pipeline is built on: a body-wide
    // spam scan would misfile legitimate mail that merely quotes a spam phrase.
    const r = classifyHeuristic(
      "Re: quarterly planning",
      addr("dana@acme.io"),
      "Someone emailed me 'claim your prize' — ignore it."
    );
    expect(r.is_spam).toBe(false);
  });

  it("treats an unsubscribe link in the body as confidently promotional", () => {
    const r = classifyHeuristic(
      "This week in AI",
      addr("news@letters.example"),
      "Lots of news. To stop these, unsubscribe at https://letters.example/u/9"
    );
    expect(r.category).toBe("PROMOTIONAL");
    expect(r.confident).toBe(true);
  });

  it("routes an automated sender to FYI without the LLM", () => {
    const r = classifyHeuristic(
      "Your order has shipped",
      addr("auto-confirm@amazon.com"),
      "On its way."
    );
    expect(r.category).toBe("FYI");
    expect(r.confident).toBe(true);
  });

  it("defers to the LLM for an automated sender inside a thread", () => {
    // A reply chain on a transactional address is often a real conversation,
    // so the short-circuit is deliberately disarmed for threads.
    const r = classifyHeuristic(
      "Your order has shipped",
      addr("auto-confirm@amazon.com"),
      "On its way.",
      true
    );
    expect(r.confident).toBe(false);
  });

  it("leaves ordinary human mail for the LLM to classify", () => {
    const r = classifyHeuristic(
      "Quick question about the roadmap",
      addr("dana@acme.io", "Dana Kim"),
      "Do you have five minutes tomorrow?"
    );
    expect(r.confident).toBe(false);
    expect(r.is_spam).toBe(false);
    expect(r.is_phishing).toBe(false);
  });
});

describe("url extraction", () => {
  it("recognises a signing URL and names the service", () => {
    const hit = extractSigningUrl(
      "Please sign: https://na3.docusign.net/signing/abc123"
    );
    expect(hit).toMatchObject({
      url: "https://na3.docusign.net/signing/abc123",
      service: "DocuSign",
    });
  });

  it("returns nothing for a body with no signing link", () => {
    expect(extractSigningUrl("Just a normal email.")).toBeFalsy();
  });

  it("extracts an unsubscribe URL", () => {
    expect(
      extractUnsubscribeUrl("... unsubscribe here https://news.example/u?x=9")
    ).toBe("https://news.example/u?x=9");
  });
});

describe("rankActions", () => {
  it("puts the reply ahead of a link for mail that needs a response", () => {
    const ranked = rankActions(
      [
        { type: "link", description: "View PR", cta: "View", url: "https://x" },
        { type: "send_reply", to: [{ email: "a@b.c" }], body: "ok" },
      ],
      "NEEDS_RESPONSE"
    );
    expect(ranked[0].type).toBe("send_reply");
    expect(ranked[1].type).toBe("link");
  });

  it("does not append auto-delete to mail that needs a response", () => {
    const ranked = rankActions(
      [{ type: "send_reply", to: [{ email: "a@b.c" }], body: "ok" }],
      "URGENT"
    );
    expect(ranked.some((a) => a.type === "delete")).toBe(false);
  });

  it("appends auto-delete for FYI", () => {
    const ranked = rankActions([], "FYI");
    expect(ranked.map((a) => a.type)).toContain("delete");
  });

  it("appends auto-delete for PERSONAL — contradicting the README", () => {
    // README.md says "URGENT, NEEDS_RESPONSE, and PERSONAL emails never get
    // auto-delete", but NO_AUTO_DELETE in rank.js omits PERSONAL. This test
    // pins the ACTUAL behaviour so the contradiction is visible and whichever
    // side is wrong gets fixed deliberately rather than drifting further.
    const ranked = rankActions([], "PERSONAL");
    expect(ranked.map((a) => a.type)).toContain("delete");
  });
});

describe("mergeUsage", () => {
  it("sums token counts across steps", () => {
    const merged = mergeUsage(
      { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 }
    );
    expect(merged.prompt_tokens).toBe(13);
    expect(merged.completion_tokens).toBe(7);
    expect(merged.total_tokens).toBe(20);
  });
});
