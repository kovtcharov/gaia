# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Qualification failures must clean only provably owned executor containers."""

import json
import sys

import pytest
import rag_profile_smoke


@pytest.mark.parametrize("owned", [True, False])
def test_timed_out_profile_cleanup_verifies_ownership(monkeypatch, tmp_path, owned):
    calls = []
    label = None
    container = "a" * 64

    class Runtime:
        def call(self, *args):
            nonlocal label
            calls.append(args)
            if args[:2] == ("image", "inspect"):
                return "sha256:" + "b" * 64
            if args[0] == "run":
                label = args[args.index("--label") + 1].split("=", 1)[1]
                raise TimeoutError("profile timed out")
            if args[0] == "ps":
                assert args[-1] == "label=ai.gaia.qualification=" + label
                return container
            if args[0] == "inspect":
                return json.dumps(
                    [
                        {
                            "Id": container,
                            "Config": {
                                "Labels": {
                                    "ai.gaia.qualification": (
                                        label if owned else "different-owner"
                                    )
                                }
                            },
                        }
                    ]
                )
            return ""

    monkeypatch.setattr(
        rag_profile_smoke, "DockerRuntime", lambda *_args, **_kwargs: Runtime()
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rag_profile_smoke",
            "--endpoint",
            "unix:///test.sock",
            "--embedding-url",
            "http://prepared",
            "--model",
            "model",
            "--revision",
            "revision",
            "--checkpoint",
            "checkpoint",
            "--report",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(TimeoutError if owned else RuntimeError):
        rag_profile_smoke.main()
    removals = [call for call in calls if call[0] == "rm"]
    if owned:
        assert removals == [("rm", "-f", container)]
        assert len([call for call in calls if call[:2] == ("volume", "rm")]) == 2
    else:
        assert not removals
    assert not (tmp_path / "report.json").exists()
