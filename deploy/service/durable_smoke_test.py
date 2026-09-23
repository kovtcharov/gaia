# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Real HTTP, frozen executors, controller crash and coherent clone restore."""

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import requests
from durable_backup import backup, holder, restore, verify
from gaia_agent.durable.client import ServiceClient
from gaia_agent.durable.store import Store
from gaia_agent.supervision.runtime import DockerRuntime
from guardian_smoke_test import wait_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args()
    runtime = DockerRuntime(
        args.endpoint, shutil.which(args.docker) or args.docker, timeout=60
    )
    image = runtime.call("image", "inspect", "gaia-service:test", "--format", "{{.Id}}")
    prefix = "gaia-durable-" + uuid4().hex
    network, inference = prefix + "-net", prefix + "-inference"
    workspace = str(uuid4())
    mapping = {"data": prefix + "-data", "workspace": prefix + "-workspace"}
    clones = {key: value + "-clone" for key, value in mapping.items()}
    processes, containers, volumes = [], [], []
    root = Path(tempfile.mkdtemp(prefix="gd-", dir="/tmp"))
    runtime.call("network", "create", network)
    try:
        runtime.call(
            "run",
            "-d",
            "--name",
            inference,
            "--network",
            network,
            "--mount",
            f"type=bind,src={Path(__file__).with_name('inference_fixture.py').resolve()},dst=/fixture.py,readonly",
            "--entrypoint=python3",
            image,
            "/fixture.py",
        )
        containers.append(inference)
        for name in mapping.values():
            runtime.call("volume", "create", name)
            volumes.append(name)
        secret = root / "inference-key"
        secret.write_text("fixture-inference-key")
        secret.chmod(0o444)
        guardian_token, caller_token = secrets.token_urlsafe(32), secrets.token_urlsafe(
            32
        )
        (root / "guardian-token").write_text(guardian_token)
        (root / "caller-token").write_text(caller_token)
        policy = {
            "deployment": str(uuid4()),
            "image": image,
            "endpoint": args.endpoint,
            "docker": runtime.executable,
            "model": "fireworks.container-fixture",
            "network": network,
            "inference_secret_file": str(secret),
            "inference_url": f"http://{inference}:8099/api/v1",
            "workspaces": {workspace: mapping},
            "slots": 1,
            "lease": 5,
            "lifetime": 90,
        }
        (root / "config.json").write_text(json.dumps(policy))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        guardian_command = [
            sys.executable,
            "-m",
            "gaia_agent.supervision.protocol",
            "--state",
            str(root / "guardian"),
            "--config",
            str(root / "config.json"),
            "--token-file",
            str(root / "guardian-token"),
        ]
        controller_command = [
            sys.executable,
            "-m",
            "gaia_agent.durable.api",
            "--state",
            str(root / "controller"),
            "--guardian-socket",
            str(root / "guardian/guardian.sock"),
            "--guardian-token-file",
            str(root / "guardian-token"),
            "--port",
            str(port),
            "--allowed-host",
            "127.0.0.1",
        ]
        environment = {**os.environ, "GAIA_GAIA_SIDECAR_TOKEN": caller_token}
        with (root / "process.log").open("w+") as log:
            guardian = subprocess.Popen(guardian_command, stdout=log, stderr=log)
            processes.append(guardian)
            wait_for((root / "guardian/guardian-control.sock").exists)
            controller = subprocess.Popen(
                controller_command, env=environment, stdout=log, stderr=log
            )
            processes.append(controller)
            url = f"http://127.0.0.1:{port}"
            client = ServiceClient(url, caller_token)

            def ready():
                try:
                    reply = client.session.get(url + "/ready", timeout=1)
                    return reply.status_code == 200
                except requests.RequestException:
                    return False

            wait_for(ready)
            session = client.call("POST", "/sessions", json={"workspace_id": workspace})
            path = f"/sessions/{session['id']}/runs"
            payload = {"prompt": "fixture-wait"}
            run = client.call(
                "POST", path, json=payload, headers={"Idempotency-Key": "disconnect"}
            )
            with client.session.get(
                url + f"/v1/gaia/service/runs/{run['id']}/stream",
                stream=True,
                timeout=5,
            ) as stream:
                assert next(stream.iter_lines(chunk_size=1)).startswith(b"id:")
            assert (
                client.call(
                    "POST",
                    path,
                    json=payload,
                    headers={"Idempotency-Key": "disconnect"},
                )["id"]
                == run["id"]
            )

            def done(run_id):
                return client.call("GET", f"/runs/{run_id}")["ended_at"] is not None

            wait_for(lambda: done(run["id"]), 45)
            assert client.call("GET", f"/runs/{run['id']}")["state"] == "succeeded"
            page = client.call("GET", f"/runs/{run['id']}/events")
            assert (
                sum(event["event"]["type"] == "final" for event in page["events"]) == 1
            )
            assert (
                "Container inference fixture passed."
                in page["events"][-1]["event"]["answer"]
            ), page["events"][-1]
            cursor = page["events"][-2]["seq"]
            replay = client.call(
                "GET", f"/runs/{run['id']}/events", params={"after": cursor}
            )
            assert len(replay["events"]) == 1
            second = client.call(
                "POST", path, json=payload, headers={"Idempotency-Key": "crash"}
            )
            wait_for(
                lambda: client.call("GET", f"/runs/{second['id']}")["state"]
                == "running",
                20,
            )
            controller.kill()
            controller.wait(timeout=5)
            replacement = subprocess.Popen(
                controller_command, env=environment, stdout=log, stderr=log
            )
            processes.append(replacement)
            wait_for(ready, 20)
            recovered = client.call("GET", f"/runs/{second['id']}")
            assert (
                recovered["state"] == "interrupted" and recovered["outcome_uncertain"]
            )
            assert (
                client.call(
                    "POST", path, json=payload, headers={"Idempotency-Key": "crash"}
                )["id"]
                == second["id"]
            )
            assert not runtime.inventory(policy["deployment"])
            third = client.call(
                "POST",
                path,
                json={"prompt": "fixture"},
                headers={"Idempotency-Key": "after-restart"},
            )
            wait_for(lambda: done(third["id"]), 45)
            assert client.call("GET", f"/runs/{third['id']}")["state"] == "succeeded"
            # CLI exercises the same public contract, without token arguments.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "gaia_agent.durable.client",
                    "--url",
                    url,
                    "--token-file",
                    str(root / "caller-token"),
                    "status",
                    third["id"],
                ],
                check=True,
                timeout=10,
            )
            client.session.close()
            replacement.terminate()
            replacement.wait(timeout=30)
            guardian.terminate()
            guardian.wait(timeout=30)
            with holder(runtime, policy, mapping) as container:
                runtime.call(
                    "run",
                    "--rm",
                    "--network=none",
                    "--user=0:0",
                    "--volumes-from",
                    container,
                    "--entrypoint=/bin/sh",
                    image,
                    "-c",
                    "printf coherent-fixture > /workspace/fact",
                )
            backup(
                root / "backup", root / "controller", root / "guardian", policy, runtime
            )
            verify(root / "backup")
            clone = {
                **policy,
                "deployment": str(uuid4()),
                "workspaces": {workspace: clones},
            }
            volumes.extend(clones.values())
            restore(root / "backup", root / "restored", clone, runtime)
            restored = Store(
                root / "restored", clone["deployment"], clone["workspaces"]
            )
            try:
                assert restored.run(third["id"])["state"] == "succeeded"
                assert restored.run(third["id"])["generation"] != third["generation"]
                assert not restored.active()
            finally:
                restored.close()
            with holder(runtime, clone, clones) as container:
                assert (
                    runtime.call(
                        "run",
                        "--rm",
                        "--network=none",
                        "--volumes-from",
                        container,
                        "--entrypoint=/bin/cat",
                        image,
                        "/workspace/fact",
                    )
                    == "coherent-fixture"
                )
            log.seek(0)
            output = log.read()
            assert (
                caller_token not in output
                and guardian_token not in output
                and "fixture-inference-key" not in output
            )
            print(
                json.dumps(
                    {
                        "disconnect_continues": "passed",
                        "idempotency": "passed",
                        "cursor_replay": "passed",
                        "controller_crash": "passed",
                        "no_redispatch": "passed",
                        "cli": "passed",
                        "database_workspace_clone_restore": "passed",
                        "orphan_containers": 0,
                    }
                )
            )
    except BaseException:
        if (root / "process.log").exists():
            output = (root / "process.log").read_text()
            for value in (
                locals().get("caller_token", ""),
                locals().get("guardian_token", ""),
                "fixture-inference-key",
            ):
                if value:
                    output = output.replace(value, "[redacted]")
            print(output[-6000:], file=sys.stderr)
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        if "policy" in locals():
            for container_id in runtime.inventory(policy["deployment"]):
                records = json.loads((root / "guardian/journal.json").read_text())[
                    "runs"
                ]
                matches = [
                    row
                    for row in records.values()
                    if row.get("container_id") == container_id
                ]
                if len(matches) != 1:
                    raise RuntimeError("Test executor ownership is unproven")
                runtime.stop(container_id, matches[0]["identity"])
                runtime.remove(container_id, matches[0]["identity"])
        for container in containers:
            runtime.call("rm", "-f", container)
        for volume in volumes:
            runtime.call("volume", "rm", volume)
        runtime.call("network", "rm", network)
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
