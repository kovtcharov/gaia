# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Explicit qualification of a prepared RAG installation; never provision models."""

import argparse
import json
import platform
import sys
from pathlib import Path

from gaia.llm.lemonade_client import LemonadeClient
from gaia.rag.sdk import RAGSDK, RAGConfig

FACT = "The Azure Finch project uses the access phrase marigold satellite. Its shipping day is Thursday."
UPDATED = "The Azure Finch project uses the access phrase cobalt meadow. Its shipping day is Friday."


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rag_phase(state, url, model, revision, checkpoint, phase):
    client = LemonadeClient(base_url=url)
    canonical = model.removeprefix("user.")
    catalog = client.list_models()["data"]
    entries = [
        row for row in catalog if row.get("id") == canonical or row.get("id") == model
    ]
    require(
        len(entries) == 1 and entries[0].get("downloaded") is True,
        "Prepare the exact embedding model before qualification; no download is performed",
    )
    require(
        entries[0].get("checkpoint") == checkpoint,
        "Prepared embedding checkpoint differs from the declared profile",
    )
    require(
        "embeddings" in entries[0].get("labels", []),
        "Declared model is not an embedding model",
    )
    manifest = state / "profile.json"
    if phase == "index":
        state.mkdir(mode=0o700, parents=False, exist_ok=False)
        (state / "facts.txt").write_text(FACT)
        (state / "noise.txt").write_text(
            "The Cedar Lake library opens at nine. It lends books about gardening and sailing."
        )
        data = {
            "schema": 1,
            "model": model,
            "revision": revision,
            "checkpoint": checkpoint,
            "documents": ["facts.txt", "noise.txt"],
            "phase": "prepared",
        }
    else:
        data = json.loads(manifest.read_text())
        require(
            data["schema"] == 1
            and data["model"] == model
            and data["revision"] == revision
            and data["checkpoint"] == checkpoint,
            "Profile changed; use a new state directory and reindex explicitly",
        )
        expected = {
            "restart": "index",
            "delete": "restart",
            "after-delete": "delete",
            "reindex": "after-delete",
        }
        require(data["phase"] == expected[phase], "Run qualification phases in order")
    rag = RAGSDK(
        RAGConfig(
            base_url=url,
            embedding_model=model,
            embedding_revision=revision,
            cache_dir=str(state / "cache"),
            allowed_paths=[str(state.resolve())],
            max_chunks=1,
        )
    )
    for document in data["documents"]:
        result = rag.index_document(str(state / document))
        require(result.get("success") is True, "Indexing failed")
        if phase in {"restart", "after-delete"}:
            require(
                result.get("from_cache") is True,
                "Restart did not reuse the verified document cache",
            )
    if phase == "delete":
        require(
            rag.remove_document(str(state / "facts.txt")), "Document removal failed"
        )
        data["documents"].remove("facts.txt")
        (state / "facts.txt").unlink()
    if phase == "reindex":
        (state / "facts.txt").write_text(UPDATED)
        require(
            rag.reindex_document(str(state / "facts.txt")).get("success") is True,
            "Reindex failed",
        )
        data["documents"].append("facts.txt")
    chunks = rag._search_chunks("What is the access phrase for Azure Finch?")[
        "retrieved_chunks"
    ]
    found = "\n".join(chunks)
    if phase in {"index", "restart"}:
        require("marigold satellite" in found, "Planted fact was not retrieved")
    elif phase in {"delete", "after-delete"}:
        require("marigold satellite" not in found, "Deleted fact remains retrievable")
    else:
        require(
            "cobalt meadow" in found and "marigold satellite" not in found,
            "Reindex returned stale content",
        )
    data["phase"] = phase
    manifest.write_text(json.dumps(data, sort_keys=True))
    return {
        "profile": "rag-retrieval",
        "phase": phase,
        "status": "passed",
        "model": model,
        "checkpoint": checkpoint,
        "revision": revision,
        "client_platform": platform.system() + "/" + platform.machine(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "answer_generation": "not_tested",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=("rag",))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--phase",
        choices=("index", "restart", "delete", "after-delete", "reindex"),
        required=True,
    )
    args = parser.parse_args(argv)
    result = rag_phase(
        args.state, args.url, args.model, args.revision, args.checkpoint, args.phase
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
