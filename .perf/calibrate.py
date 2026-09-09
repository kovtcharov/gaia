"""Score distribution + cap/threshold sweep for the flagship. Embeddings only."""

import json

import numpy as np
import tiktoken
from gaia_agent.agent import GaiaAgent, GaiaAgentConfig
from gaia_agent_chat.tool_bundles import FULL_BUNDLES, FULL_CORE_TOOLS

from gaia.agents.base.tool_loader import ToolLoader

enc = tiktoken.get_encoding("cl100k_base")
agent = GaiaAgent(config=GaiaAgentConfig(silent_mode=True))
reg = agent._tools_registry
base = agent.tool_loader

QUERIES = {
    "hey there": set(),
    "list the 3 most recently opened issues in amd/gaia": {"web"},
    "summarize this PDF": {"rag_query"},
    "what's in my Documents folder?": {"file_browse", "file_discovery"},
    "plot the top 5 products by revenue from sales.csv": {"data", "scratchpad_sql"},
    "index this repo and find the function that builds the system prompt": {
        "code_index"
    },
    "take a screenshot and tell me what is on my screen": {"screenshot", "vision"},
    "install the github-triage skill": {"skill_hub"},
    "what is the disk usage on this machine": {"shell"},
}

# Warm the tool-doc embedding cache once, then score by hand.
base.select("warmup", reg)
vecs = base._ensure_tool_embeddings(reg)

print("per-query bundle-level match scores (top 8 tools)")
for q in QUERIES:
    qv = base._embed_fn(q)
    scored = sorted(
        ((float(np.dot(qv, v)), n) for n, v in vecs.items()), reverse=True
    )
    head = ", ".join(f"{n}={s:.2f}" for s, n in scored[:8])
    over = sum(1 for s, _ in scored if s >= 0.20)
    print(f"\n  {q[:60]!r}\n    >=0.20: {over:>2}/{len(scored)}   {head}")


def run(threshold, cap):
    loader = ToolLoader(
        core_tools=FULL_CORE_TOOLS,
        bundles=FULL_BUNDLES,
        embed_fn=base._embed_fn,
        embed_batch_fn=base._embed_batch_fn,
        threshold=threshold,
        max_tools=cap,
    )
    loader._embed_cache = base._embed_cache
    hit = miss = 0
    toks = []
    for q, want in QUERIES.items():
        loader.reset_session()
        sel = set(loader.select(q, reg))
        toks.append(
            len(enc.encode(json.dumps(agent._build_openai_tool_schemas(
                filter_to=sorted(sel)))))
        )
        for bname in want:
            members = next(b.members for b in FULL_BUNDLES if b.name == bname)
            present = {m for m in members if m in reg}
            if present <= sel:
                hit += 1
            else:
                miss += 1
    return hit, miss, sum(toks) / len(toks)


print("\n\nsweep: wanted-bundle fully present / partially cut, mean tools= tokens")
print(f"{'tau':>6}{'cap':>6}{'whole':>8}{'cut':>6}{'mean tok':>10}")
for threshold in (0.18, 0.20, 0.22, 0.25):
    for cap in (18, 22, 26, 30):
        hit, miss, mean = run(threshold, cap)
        print(f"{threshold:>6}{cap:>6}{hit:>8}{miss:>6}{mean:>10,.0f}")
