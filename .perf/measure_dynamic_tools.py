"""Offline before/after for the flagship's per-call tool-prompt cost.

Pure tiktoken over the real renderers. No model call, no Lemonade, no network.

The RECORDED_SELECTIONS below came from the loader's own ``select()`` against
the live nomic embedder (one /embeddings call per query, Lemonade slot pool
"embedding") on 2026-08-18, before Lemonade was taken out of service. Re-derive
them with ``.perf/calibrate.py`` when a backend is available again; the token
arithmetic here is deterministic either way.
"""

import contextlib
import json

import tiktoken
from gaia_agent.agent import GaiaAgentConfig
from gaia_agent_chat.tool_bundles import FULL_CORE_TOOLS

from gaia.agents.base.tools import _TOOL_REGISTRY
from gaia.eval.tool_cost import build_full_agent_skeleton

enc = tiktoken.get_encoding("cl100k_base")


def tok(s):
    return len(enc.encode(s))


@contextlib.contextmanager
def isolated_registry():
    saved = dict(_TOOL_REGISTRY)
    _TOOL_REGISTRY.clear()
    try:
        yield
    finally:
        _TOOL_REGISTRY.clear()
        _TOOL_REGISTRY.update(saved)


with isolated_registry():
    agent = build_full_agent_skeleton()
    reg = agent._tools_registry

    def native(names):
        return tok(json.dumps(agent._build_openai_tool_schemas(filter_to=names)))

    def text(names):
        return tok(agent._format_tools_for_prompt(filter_to=names))

    full_n, full_x = native(None), text(None)
    core = sorted(n for n in FULL_CORE_TOOLS if n in reg)

    print("=" * 82)
    print(f"registry                    : {len(reg)} tools")
    print(f"cap (dynamic_tools_max)     : {GaiaAgentConfig().dynamic_tools_max}")
    print(f"CORE (FULL_CORE_TOOLS)      : {len(core)} tools")
    print("=" * 82)
    print(f"{'':<46}{'native':>8}{'text':>7}{'both':>7}{'saved':>7}")
    print("-" * 82)
    print(f"{'BEFORE - whole registry every call':<46}{full_n:>8,}{full_x:>7,}"
          f"{full_n + full_x:>7,}{0:>7}")

    # Recorded from the live embedder; see module docstring.
    RECORDED_SELECTIONS = {
        "hey there": [],
        "list the 3 most recently opened issues in amd/gaia": [
            "browse_directory", "download_file", "fetch_page", "fetch_webpage",
            "get_file_info", "install_skill", "list_recent_files", "list_skills",
            "list_windows", "notify_desktop", "open_url", "remove_skill",
            "search_documentation", "search_skill_hub", "search_web",
            "text_to_speech",
        ],
        "summarize this PDF": [
            "add_watch_directory", "analyze_data_file", "create_table",
            "drop_table", "dump_document", "evaluate_retrieval", "index_directory",
            "index_document", "insert_data", "list_indexed_documents",
            "list_tables", "query_data", "query_specific_file", "rag_status",
            "search_indexed_chunks", "summarize_document",
        ],
        "what's in my Documents folder?": [
            "add_watch_directory", "bookmark", "browse_directory", "dump_document",
            "evaluate_retrieval", "file_info", "find_files", "get_file_info",
            "index_directory", "index_document", "list_files", "list_recent_files",
            "query_specific_file", "search_indexed_chunks", "summarize_document",
            "tree",
        ],
        "plot the top 5 products by revenue from sales.csv": [
            "add_watch_directory", "analyze_data_file", "create_table",
            "drop_table", "index_directory", "index_document", "insert_data",
            "list_indexed_documents", "list_skills", "list_tables", "list_windows",
            "load_skill", "query_data", "rag_status", "skill_status",
            "unload_skill",
        ],
        "index this repo and find the function that builds the system prompt": [
            "add_watch_directory", "clear_code_index", "download_file",
            "execute_python_file", "fetch_page", "fetch_webpage",
            "get_index_status", "get_system_info", "index_codebase",
            "index_directory", "index_document", "list_indexed_documents",
            "open_url", "rag_status", "run_shell_command", "search_code_index",
        ],
    }

    rows = []
    for q, extra in RECORDED_SELECTIONS.items():
        sel = sorted(set(core) | set(extra))
        missing = [n for n in sel if n not in reg]
        assert not missing, f"recorded selection names no longer registered: {missing}"
        n_, x_ = native(sel), text(sel)
        rows.append((q, sel, n_, x_))
        print(f"{'  ' + q[:43]:<46}{n_:>8,}{x_:>7,}{n_ + x_:>7,}"
              f"{full_n + full_x - n_ - x_:>7,}")

    k = len(rows)
    an = sum(r[2] for r in rows) / k
    ax = sum(r[3] for r in rows) / k
    print("-" * 82)
    print(f"{'AFTER - mean over the 6 queries':<46}{an:>8,.0f}{ax:>7,.0f}"
          f"{an + ax:>7,.0f}{full_n + full_x - an - ax:>7,.0f}")
    print("=" * 82)
    print(
        f"per-LLM-call tool-prompt cost: {full_n + full_x:,} -> "
        f"{an + ax:,.0f} tiktoken(cl100k) tokens "
        f"({100 * (1 - (an + ax) / (full_n + full_x)):.0f}% cut)"
    )
    print("NOT MEASURED: seconds. Needs a Lemonade prefill run on Gemma-4-E4B.")
