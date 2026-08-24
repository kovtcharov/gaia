# GAIA Agent Eval — eval-20260824-104200
**Date:** 2026-08-24T11:25:08.129698+00:00
**Model:** claude-sonnet-4-6

## Summary
- **Total:** 11 scenarios
- **Passed:** 8 ✅
- **Failed:** 3 ❌
- **Blocked:** 0 🚫
- **Timeout:** 0 ⏱
- **Budget exceeded:** 0 💸
- **Infra error:** 0 🔧
- **Skipped (no doc):** 0 ⏭
- **Errored:** 0 ⚠️
- **Pass rate (all):** 73%
- **Pass rate (judged):** 73%
- **Avg score (judged):** 8.8/10

## By Category
| Category | Pass | Fail | Blocked | Infra | Skipped | Avg Score |
|----------|------|------|---------|-------|---------|-----------|
| gaia_core | 8 | 3 | 0 | 0 | 0 | 8.9 |

## Scenarios
- ✅ **core_arithmetic_direct** — PASS (10.0/10)
  - Root cause: null
- ✅ **core_bare_number_followup** — PASS (10.0/10)
  - Root cause: null
- ✅ **core_contradiction_injection** — PASS (10.0/10)
  - Root cause: null
- ✅ **core_false_premise_pushback** — PASS (9.9/10)
  - Root cause: null
- ✅ **core_follow_up_pronoun** — PASS (10.0/10)
  - Root cause: null
- ✅ **core_interruption_recovery** — PASS (10.0/10)
  - Root cause: null
- ❌ **core_long_horizon_history** — FAIL (8.9/10)
  - Root cause: "The agent explicitly acknowledged the planted fact in turn 2 ('Got it — Priya's visit on the 30th') but by turn 14 — after 11 distractor turns — it denied any visiting information was ever shared. The conversation history reaching the LLM was likely truncated, dropping the early messages (turns 1-2, DB message IDs 71-74) from the context window by the time the session reached 30 messages."
- ✅ **core_mango_canary** — PASS (10.0/10)
  - Root cause: null
- ❌ **core_nested_reference** — FAIL (8.5/10)
  - Root cause: "The agent tracked the resolved referent (Wind) across turns 2-3 for the forward reference hops, but failed the elimination-based backward reference in turn 4. When asked about 'the two we haven't dug into,' the agent compared Solar vs. Wind rather than Solar vs. Hydroelectric, suggesting it did not reason over which items from the original list had been covered versus which remained."
- ✅ **core_persona_consistency** — PASS (9.6/10)
  - Root cause: null
- ❌ **core_topic_switch_resume** — FAIL (7.0/10)
  - Root cause: Hard-coded conversation-history window (_MAX_HISTORY_PAIRS = 5) in src/gaia/ui/_chat_helpers.py rebuilds the agent's prompt from only the last 5 exchange pairs per turn. All 24 messages persist in the database but the LLM never sees beyond the 5-pair window, so turn 1 facts ($2,400, 9 days, Chicago) are evicted from context exactly when the user resumes at turn 8 — six turns later. The window slide is confirmed turn-by-turn: at turn 8 the agent believes the chat began at turn 3, at turn 9 at turn 4, at turn 10 at turns 5-6.

## Performance
- **Avg throughput:** 38.7 tok/s
- **Avg TTFT:** 7254ms
- **Total tokens:** 90,710 input → 810 output
- **Scenarios with data:** 2/11
- **Flags:** high_latency, no_stats, token_explosion

**Cost:** $0.5000
