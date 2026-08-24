# Gaia scorecard pipeline — harness-validation proof (Haiku, no Lemonade)

**What this is:** end-to-end proof that the flagship's scorecard pipeline works —
`gaia eval agent --agent-type gaia` → `scorecard.json` →
`packaging/gen_scorecard.py` → a rendered, gate-passing `SCORECARD.md`. It is
**NOT the product baseline**: the agent ran on `claude-haiku-4-5` via the
`GAIA_EVAL_AGENT_PROVIDER=claude` opt-in because this dev box must never start
Lemonade. The product baseline is Gemma-4-E4B-it-GGUF on the self-hosted
runner, captured by the first full `gaia_scorecard_refresh.yml` dispatch
(`rebaseline=true` — the committed card's ctx/model basis changes then).

## The run (2026-08-24, run id `eval-20260824-104200`)

```
backend   python -m gaia.ui.server --port 4200 --host 127.0.0.1
          GAIA_EVAL_AGENT_PROVIDER=claude GAIA_EVAL_CLAUDE_MODEL=claude-haiku-4-5
          GAIA_MEMORY_DISABLED=1 LEMONADE_BASE_URL=http://127.0.0.1:9  (unreachable — guard)
eval      gaia eval agent --agent-type gaia --category gaia_core
          --backend http://127.0.0.1:4200 --model claude-sonnet-4-6 --budget 5.00
          --exclude-tag local_blocked_no_embedder
          (judge/driver `claude -p` rides the Claude Code subscription —
           ANTHROPIC_API_KEY emptied in the eval process so the depleted
           pay-as-you-go key a parent .env injects cannot capture it)
```

Result: **11/11 scenarios judged** (0 infra errors) — 8 PASS / 3 FAIL,
`judged_pass_rate = 0.7273`, avg score 8.85/10. The three FAILs
(`core_long_horizon_history` 8.9, `core_nested_reference` 8.5,
`core_topic_switch_resume` 7.0) are genuine judge verdicts on Haiku — real
signal flowing through the pipeline, deliberately not tuned away.

- `scorecard.json` — the runner's output verbatim (`config.agent_type: "gaia"`).
- `run-summary.md` — the runner's human-readable summary.
- `SCORECARD.md` — the card `gen_scorecard.py` rendered from it (identical to
  the committed `hub/agents/gaia/npm/SCORECARD.md` of this commit; aggregate
  72.73 = 100 × 8/11).

## Gate proof (on this real card)

- `python -m gaia.eval.scorecard_gate --scorecard hub/agents/gaia/npm/SCORECARD.md`
  → **exit 0** (first adoption, presence-only).
- Same gate against a manufactured card with `aggregate.value` lowered to 50.0,
  baselined on this card → **exit 1** (regression blocked).

## No-Lemonade evidence

- `LEMONADE_BASE_URL` pinned to an unreachable port for the whole run; the
  backend logged `Pre-flight skipped: GAIA_EVAL_AGENT_PROVIDER=claude —
  Lemonade is not in use` on every turn.
- Lemonade process count asserted **0** after the run (`tasklist`). A stray
  `LemonadeServer.exe` unrelated to this task (started 2026-08-22, before this
  session) was running idle at session start and exited on its own mid-run;
  nothing in this pipeline ever connected to it.
