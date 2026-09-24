# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Declared embedding profiles, revision separation and fail-closed network policy."""

from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from gaia_agent import profiles
from gaia_agent.supervision.guardian import validate_policy
from gaia_agent.supervision.runtime import PREFIX, DockerRuntime

from gaia.rag.sdk import RAGSDK, RAGConfig


def policy():
    return {
        "deployment": str(uuid4()),
        "image": "sha256:" + "a" * 64,
        "model": "fixture",
        "inference_url": "http://inference/api/v1",
        "workspaces": {str(uuid4()): {"data": "data", "workspace": "workspace"}},
    }


def test_network_profile_refuses_unsupported_host_or_non_internal_network(monkeypatch):
    value = {**policy(), "network_internal": True, "network": "isolated"}
    monkeypatch.setattr("gaia_agent.supervision.guardian.sys.platform", "darwin")
    with pytest.raises(ValueError, match="native Linux"):
        validate_policy(value)
    monkeypatch.setattr("gaia_agent.supervision.guardian.sys.platform", "linux")
    validate_policy(value)
    runtime = DockerRuntime("unix:///explicit.sock")
    call = Mock(return_value="false")
    monkeypatch.setattr(runtime, "call", call)
    with pytest.raises(ValueError, match="does not enforce"):
        runtime.create(
            {"run": str(uuid4())},
            value,
            {"data": "data", "workspace": "workspace"},
            "token",
        )
    assert not any(args.args[0] == "create" for args in call.call_args_list)


def test_internal_address_requires_single_verified_private_network(monkeypatch):
    runtime = DockerRuntime("unix:///explicit.sock")
    container = {
        "State": {"Running": True},
        "Config": {"Labels": {PREFIX + "network-profile": "internal"}},
        "NetworkSettings": {"Networks": {"isolated": {"IPAddress": "172.22.0.3"}}},
    }
    monkeypatch.setattr(runtime, "inspect", lambda *_: container)
    monkeypatch.setattr(runtime, "call", lambda *args: "true")
    assert runtime.start("a" * 64, {}) == "http://172.22.0.3:8080"
    container["NetworkSettings"]["Networks"]["isolated"]["IPAddress"] = "1.1.1.1"
    with pytest.raises(RuntimeError, match="Invalid internal"):
        runtime.start("a" * 64, {})


def test_embedding_revision_separates_query_vectors(monkeypatch):
    rag = RAGSDK.__new__(RAGSDK)
    rag.config = RAGConfig(
        embedding_model="explicit", embedding_revision="revision-two"
    )
    cache = Mock()
    cache.get.return_value = [0.1, 0.2]
    monkeypatch.setattr(rag, "_get_embedding_cache", lambda: cache)
    assert rag._encode_query("question").shape == (1, 2)
    cache.get.assert_called_once_with("explicit", "revision-two", "question")


def test_rag_default_endpoint_respects_environment(monkeypatch):
    monkeypatch.setenv("LEMONADE_BASE_URL", "http://declared-inference:8000/api/v1")
    assert RAGConfig().base_url == "http://declared-inference:8000/api/v1"


def test_unprepared_model_is_refused_before_rag_or_state_creation(
    tmp_path, monkeypatch
):
    client = SimpleNamespace(
        list_models=lambda: {"data": [{"id": "model", "downloaded": False}]}
    )
    monkeypatch.setattr(profiles, "LemonadeClient", lambda **_: client)
    sdk = Mock()
    monkeypatch.setattr(profiles, "RAGSDK", sdk)
    with pytest.raises(RuntimeError, match="Prepare the exact"):
        profiles.rag_phase(
            tmp_path / "state",
            "http://inference",
            "model",
            "revision",
            "checkpoint",
            "index",
        )
    sdk.assert_not_called()
    assert not (tmp_path / "state").exists()


def test_checkpoint_mismatch_is_not_silently_substituted(tmp_path, monkeypatch):
    client = SimpleNamespace(
        list_models=lambda: {
            "data": [
                {
                    "id": "model",
                    "downloaded": True,
                    "checkpoint": "different",
                    "labels": ["embeddings"],
                }
            ]
        }
    )
    monkeypatch.setattr(profiles, "LemonadeClient", lambda **_: client)
    with pytest.raises(RuntimeError, match="checkpoint differs"):
        profiles.rag_phase(
            tmp_path / "state",
            "http://inference",
            "model",
            "revision",
            "expected",
            "index",
        )


@pytest.mark.parametrize("invalid", [True, 42, "", "\n", "x" * 513])
def test_embedding_policy_refuses_invalid_declarations(invalid):
    with pytest.raises(ValueError):
        validate_policy(
            {**policy(), "embedding_model": invalid, "embedding_revision": invalid}
        )


@pytest.mark.parametrize("downloaded", [True, False])
def test_declared_revision_never_invokes_model_download(monkeypatch, downloaded):
    from contextlib import nullcontext

    rag = RAGSDK.__new__(RAGSDK)
    rag.config = RAGConfig(
        embedding_model="user.prepared", embedding_revision="revision"
    )
    rag.log = Mock()
    rag.embedder = None
    rag.llm_client = Mock()
    monkeypatch.setattr(
        "gaia.llm.lemonade_client.MODELS",
        {
            "prepared": SimpleNamespace(
                model_id="user.prepared",
                checkpoint="prepared-checkpoint",
                recipe="llamacpp",
                embedding=True,
            )
        },
    )
    rag.llm_client.list_models.return_value = {
        "data": [{"id": "prepared", "downloaded": downloaded}]
    }
    monkeypatch.setattr(
        "gaia.daemon.broker_client.model_lease", lambda *_args, **_kwargs: nullcontext()
    )
    if downloaded:
        rag._load_embedder()
        assert rag.embedder is rag.llm_client
        rag.llm_client.load_model.assert_called_once()
    else:
        with pytest.raises(RuntimeError, match="not prepared"):
            rag._load_embedder()
        rag.llm_client.load_model.assert_not_called()
    rag.llm_client.ensure_model_downloaded.assert_not_called()
