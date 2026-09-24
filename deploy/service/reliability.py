# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Sequential endurance and concurrent admission pressure against real HTTP workers."""

import hashlib
import json
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import gaia_agent
import requests
from gaia_agent.durable.client import ServiceClient
from guardian_smoke_test import wait_for


def exercise(
    url,
    token,
    workspace,
    runtime,
    deployment,
    controller_pid,
    restart,
    tasks,
    saturation_seconds,
    report_path,
    image_digest,
):
    client = ServiceClient(url, token)
    latencies, memory, outcomes = [], [], {}
    started = time.monotonic()

    def rss():
        return (
            int(
                subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(controller_pid())],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3,
                ).stdout.strip()
            )
            * 1024
        )

    def submit(prompt):
        session = client.call("POST", "/sessions", json={"workspace_id": workspace})
        path = f"/sessions/{session['id']}/runs"
        key = str(uuid4())
        run = client.call(
            "POST", path, json={"prompt": prompt}, headers={"Idempotency-Key": key}
        )
        assert (
            client.call(
                "POST", path, json={"prompt": prompt}, headers={"Idempotency-Key": key}
            )["id"]
            == run["id"]
        )
        return run

    for index in range(tasks):
        start = time.monotonic()
        fault = (
            "restart" if index % 100 == 99 else "cancel" if index % 50 == 49 else None
        )
        run = submit("fixture-wait" if fault else "fixture")
        path = f"/runs/{run['id']}"
        if fault:
            wait_for(
                lambda path=path: client.call("GET", path)["state"] == "running", 20
            )
            if fault == "restart":
                restart()
            else:
                client.call("POST", path + "/cancel")
        wait_for(lambda path=path: client.call("GET", path)["ended_at"] is not None, 45)
        terminal = client.call("GET", path)
        expected = (
            "interrupted"
            if fault == "restart"
            else "cancelled" if fault else "succeeded"
        )
        assert terminal["state"] == expected, terminal
        page = client.call("GET", path + "/events")
        assert (
            sum(item["event"]["type"] in {"final", "error"} for item in page["events"])
            == 1
        )
        assert not runtime.inventory(deployment)
        outcomes[expected] = outcomes.get(expected, 0) + 1
        latencies.append(time.monotonic() - start)
        if index % 25 == 0:
            memory.append(rss())
            print(
                json.dumps(
                    {
                        "endurance_completed": index + 1,
                        "requested": tasks,
                        "rss_bytes": memory[-1],
                    }
                ),
                flush=True,
            )
    pressures = {"accepted": 0, "rejected": 0, "errors": 0}
    lock, stop = threading.Lock(), threading.Event()

    def pressure():
        session = requests.Session()
        session.trust_env = False
        session.headers["Authorization"] = "Bearer " + token
        own_session = session.post(
            url + "/v1/gaia/service/sessions",
            json={"workspace_id": workspace},
            timeout=5,
        ).json()["id"]
        try:
            while not stop.is_set():
                response = session.post(
                    url + f"/v1/gaia/service/sessions/{own_session}/runs",
                    json={"prompt": "fixture"},
                    headers={"Idempotency-Key": str(uuid4())},
                    timeout=10,
                )
                if response.status_code == 202:
                    run_id = response.json()["id"]
                    with lock:
                        pressures["accepted"] += 1
                    wait_for(
                        lambda: session.get(
                            url + f"/v1/gaia/service/runs/{run_id}", timeout=5
                        ).json()["ended_at"]
                        is not None,
                        45,
                    )
                    terminal = session.get(
                        url + f"/v1/gaia/service/runs/{run_id}", timeout=5
                    ).json()
                    assert terminal["state"] == "succeeded", terminal
                    page = session.get(
                        url + f"/v1/gaia/service/runs/{run_id}/events", timeout=5
                    ).json()
                    assert (
                        sum(
                            row["event"]["type"] in {"final", "error"}
                            for row in page["events"]
                        )
                        == 1
                    ), page
                elif response.status_code in {409, 503}:
                    with lock:
                        pressures["rejected"] += 1
                else:
                    with lock:
                        pressures["errors"] += 1
                stop.wait(0.05)
        finally:
            session.close()

    control_latencies = []
    if saturation_seconds:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(pressure) for _ in range(8)]
            deadline = time.monotonic() + saturation_seconds
            try:
                while time.monotonic() < deadline:
                    before = time.monotonic()
                    metrics = client.call("GET", "/metrics")
                    control_latencies.append(time.monotonic() - before)
                    assert metrics["store"]["active_runs"] <= 1
                    memory.append(rss())
                    time.sleep(min(5, max(0, deadline - time.monotonic())))
            finally:
                stop.set()
            for future in futures:
                future.result()
        assert (
            pressures["accepted"] > 0
            and pressures["rejected"] > 0
            and pressures["errors"] == 0
        ), pressures
    assert not runtime.inventory(deployment)
    metrics = client.call("GET", "/metrics")
    assert metrics["store"]["active_runs"] == 0
    digest = hashlib.sha256()
    package = Path(gaia_agent.__file__).parent
    for source in sorted(package.rglob("*.py")):
        digest.update(str(source.relative_to(package)).encode())
        digest.update(source.read_bytes())
    result = {
        "schema": 1,
        "controller_source_sha256": digest.hexdigest(),
        "sequential_tasks": tasks,
        "outcomes": outcomes,
        "saturation_seconds": saturation_seconds,
        "pressure": pressures,
        "orphan_executors": 0,
        "active_after": 0,
        "elapsed_seconds": time.monotonic() - started,
        "latency_seconds": {
            "median": statistics.median(latencies) if latencies else None,
            "max": max(latencies, default=None),
        },
        "control_latency_max_seconds": max(control_latencies, default=None),
        "controller_rss_bytes": {
            "first": memory[0] if memory else None,
            "max": max(memory, default=None),
            "last": memory[-1] if memory else None,
        },
        "store": metrics["store"],
        "image": image_digest,
    }
    if memory:
        assert max(memory) - memory[0] < 128 * 1024 * 1024, result
    if control_latencies:
        assert max(control_latencies) < 5, result
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    client.session.close()
