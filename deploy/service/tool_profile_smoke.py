# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Approve exact Python tool calls and test real external I/O and hard cancellation."""

import json
import time
from uuid import uuid4


def exercise(client, url, workspace, fixture_url, runtime, policy):
    outcomes = {}
    if policy.get("network_internal"):
        # A failed destination is not evidence of enforced isolation. Prove the
        # same host/image can reach it over the ordinary Docker bridge first.
        runtime.call(
            "run",
            "--rm",
            "--read-only",
            "--memory=256m",
            "--pids-limit=32",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--network=bridge",
            "--entrypoint=python3",
            policy["image"],
            "-c",
            "import socket; socket.create_connection(('1.1.1.1',443),timeout=5).close()",
        )
        outcomes["egress_positive_control"] = "passed"
    modes = ("http", "authfail", "timeout", "cancel") + (
        ("egress",) if policy.get("network_internal") else ()
    )
    for mode in modes:
        before = client.session.get(fixture_url + "/stats", timeout=3).json()
        session = client.call("POST", "/sessions", json={"workspace_id": workspace})
        run = client.call(
            "POST",
            f"/sessions/{session['id']}/runs",
            json={"prompt": "fixture-profile-" + mode, "max_steps": 3},
            headers={"Idempotency-Key": str(uuid4())},
        )
        path = f"/runs/{run['id']}"
        deadline, approved, cancelled = time.monotonic() + 45, set(), False
        while time.monotonic() < deadline:
            status = client.call("GET", path)
            if status["ended_at"] is not None:
                break
            pending = client.session.get(client.url + path + "/interaction", timeout=3)
            if pending.status_code == 200:
                interaction = pending.json()
                if interaction["id"] not in approved:
                    raw = interaction["request"]
                    assert raw["type"] == "needs_confirmation" and raw.get(
                        "arguments"
                    ), raw
                    assert "run_python" in json.dumps(raw), raw
                    client.call(
                        "POST",
                        path + f"/interactions/{interaction['id']}/answer",
                        json={
                            "generation": interaction["generation"],
                            "decision": "approve",
                        },
                    )
                    approved.add(interaction["id"])
            else:
                assert pending.status_code in {404, 409}, pending.status_code
            if (
                mode == "cancel"
                and not cancelled
                and client.session.get(fixture_url + "/stats", timeout=3).json()[
                    "started"
                ]
            ):
                client.call("POST", path + "/cancel")
                cancelled = True
            time.sleep(0.05)
        else:
            raise TimeoutError("External tool profile did not terminate: " + mode)
        assert not runtime.inventory(policy["deployment"])
        if mode == "cancel":
            assert cancelled and status["state"] == "cancelled", status
        else:
            events = client.call("GET", path + "/events")["events"]
            assert status["state"] == "succeeded", (status, events)
            marker = {
                "http": "authenticated-external-tool",
                "authfail": "expected-http-401",
                "timeout": "expected-timeout",
                "egress": "expected-egress-blocked",
            }[mode]
            results = [
                row["event"] for row in events if row["event"]["type"] == "tool_result"
            ]
            assert results and all(
                event["data"]["success"] for event in results
            ), events
            finals = [row["event"] for row in events if row["event"]["type"] == "final"]
            assert len(finals) == 1 and marker in finals[0]["answer"], events
            if mode in {"http", "authfail", "timeout"}:
                after = client.session.get(fixture_url + "/stats", timeout=3).json()
                assert after[mode] == before[mode] + 1, (before, after)
        outcomes[mode] = "passed"
    print(
        json.dumps({"external_tool_profiles": outcomes, "orphan_executors": 0}),
        flush=True,
    )
