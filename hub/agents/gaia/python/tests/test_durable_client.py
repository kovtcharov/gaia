# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Durable CLI must inspect full approval details and avoid answer argv."""

from unittest.mock import Mock

import pytest
from gaia_agent.durable import client


def test_remote_plaintext_and_credential_urls_rejected():
    for url in (
        "http://example.com",
        "https://secret@example.com",
        "https://example.com?key=secret",
    ):
        with pytest.raises(ValueError):
            client.ServiceClient(url, "token")


def test_approval_requires_full_argument_review(monkeypatch, capsys):
    calls = []

    def call(_self, method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path == "/capabilities":
            return {"api_version": 1, "durable_runs": True}
        if path.endswith("/interaction"):
            return {
                "id": "interaction",
                "generation": "generation",
                "request": {"arguments": {"path": "complete-path"}},
            }
        return {"accepted": True}

    monkeypatch.setenv("GAIA_GAIA_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(client.ServiceClient, "call", call)
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    args = ["answer", "run", "interaction", "generation", "approve"]
    assert client.main(args) == 1
    assert "complete-path" in capsys.readouterr().out
    assert not any(method == "POST" for method, _, _ in calls)
    monkeypatch.setattr("builtins.input", lambda _prompt: "approve")
    assert client.main(args) == 0
    assert sum(method == "POST" for method, _, _ in calls) == 1


def test_answer_requires_stable_submission_and_hidden_input(monkeypatch):
    post = Mock(return_value={"accepted": True})

    def call(_self, method, path, **kwargs):
        if path == "/capabilities":
            return {"api_version": 1, "durable_runs": True}
        return post(method, path, **kwargs)

    monkeypatch.setenv("GAIA_GAIA_SIDECAR_TOKEN", "token")
    monkeypatch.setattr(client.ServiceClient, "call", call)
    monkeypatch.setattr(client.getpass, "getpass", lambda _prompt: "hidden")
    assert (
        client.main(
            [
                "answer",
                "run",
                "interaction",
                "generation",
                "answered",
                "--submission-id",
                "stable",
            ]
        )
        == 0
    )
    assert post.call_args.kwargs["json"] == {
        "decision": "answered",
        "generation": "generation",
        "submission_id": "stable",
        "response": "hidden",
    }
