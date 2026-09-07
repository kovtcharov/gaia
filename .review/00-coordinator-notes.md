# Coordinator cross-cutting notes (main @ 211f08c5, v0.23.1)

## Size
- src/gaia python: 175,010 lines; hub python: 103,625; tests: 534 files / 215,436 lines
- Go TUI: 59,173 lines; webui TS: 31,151; cpp: 59,652; docs: 199 mdx; workflows: 72
- Open issues: 698; open PRs: 33 (2026-09-02)

## Largest Python files (maintainability hot spots)
- src/gaia/cli.py 8087; src/gaia/agents/base/agent.py 6794; src/gaia/llm/lemonade_client.py 4818
- src/gaia/agents/base/discovery.py 3918; hub/agents/email/.../api_routes.py 3465; src/gaia/rag/sdk.py 3418
- hub/agents/email/.../tools/read_tools.py 3264; memory_store.py 3180; memory.py 3039; ui/_chat_helpers.py 2873

## Silent-swallow handlers (except ...: pass/return None/{}/[]/continue) in src+hub: 76
- system_context.py 20, memory.py 6, console.py 6, lemonade_client.py 5, discovery.py 5, ui/routers/memory.py 3,
  testing/fixtures.py 3, device.py 3, agent.py 3, ui/routers/system.py 2, cli.py 2, memory_store.py 2, + 13 singles
- CLAUDE.md says pre-existing ones (mostly src/gaia/ui) are tech debt; the majority are actually in agents/base, not ui.

## Misc
- TODO/FIXME/XXX/HACK markers across py/go/ts: 13 (low)
- shell=True usage: only in shell_tools.py (Windows cmd resolution, documented) and none elsewhere; no pickle.load / yaml.load in src (only in skills/audit sink list)
- version.py == 0.23.1, matches "Release v0.23.1 (#3054)"; tag v0.23.1 not yet on upstream (only v0.23.0 exists) — check with CI reviewer whether tagging is the manual step.
- npm audit timed out (>5m) on root — network was flaky during this window; retry later.

## npm audit (webui, src/gaia/apps/webui, includes devDependencies) — 2026-09-02
- 6 advisories: 5 high, 1 moderate — @xmldom/xmldom (<=0.8.14, moderate), brace-expansion, fast-uri, js-yaml (4.0.0-4.3.0), tar (<=7.5.20), undici (<=6.27.0 / 7.0.0-7.28.0)
- Dependabot groups exist (agent-ui-dependencies bumped 2026-09-02 in #3156) so these are likely transitive dev-tool deps; whether any ship in the Electron bundle is for the frontends reviewer / a follow-up.
- root package.json audit could not complete (network flaky).
