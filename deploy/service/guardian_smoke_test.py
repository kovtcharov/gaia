# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Exercise the independent guardian against real Docker and frozen GAIA."""

import argparse
import json
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from gaia_agent.supervision.protocol import Client
from gaia_agent.supervision.runtime import DockerRuntime


def wait_for(predicate, timeout=15):
    expires = time.monotonic() + timeout
    while time.monotonic() < expires:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError("Guardian smoke condition did not complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--docker", default="docker")
    args = parser.parse_args()
    args.docker = shutil.which(args.docker) or args.docker
    runtime = DockerRuntime(args.endpoint, args.docker, timeout=15)
    image = runtime.call("image", "inspect", "gaia-service:test", "--format", "{{.Id}}")
    prefix = "gaia-guardian-" + uuid4().hex
    network, inference = prefix + "-net", prefix + "-inference"
    workspaces = {
        str(uuid4()): {
            "data": prefix + "-data-" + str(index),
            "workspace": prefix + "-work-" + str(index),
        }
        for index in range(2)
    }
    volumes, processes, owned = [], [], []
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
        owned.append(inference)
        for mapping in workspaces.values():
            for volume in mapping.values():
                runtime.call("volume", "create", volume)
                volumes.append(volume)
        temporary = tempfile.mkdtemp(prefix="gg-", dir="/tmp")
        if temporary:
            root = Path(temporary)
            secret = root / "inference-key"
            secret.write_text("fixture-inference-key")
            secret.chmod(0o444)
            token = secrets.token_urlsafe(32)
            (root / "token").write_text(token)
            outage = root / "runtime-unavailable"
            proxy = root / "docker-proxy"
            proxy.write_text(
                "#!"
                + sys.executable
                + "\nimport os,sys\nfrom pathlib import Path\n"
                + "if Path("
                + repr(str(outage))
                + ").exists(): sys.exit(75)\n"
                + "os.execv("
                + repr(args.docker)
                + ", ["
                + repr(args.docker)
                + "] + sys.argv[1:])\n"
            )
            proxy.chmod(0o700)
            policy = {
                "deployment": str(uuid4()),
                "image": image,
                "endpoint": args.endpoint,
                "docker": str(proxy),
                "model": "fireworks.container-fixture",
                "network": network,
                "inference_secret_file": str(secret),
                "inference_url": f"http://{inference}:8099/api/v1",
                "workspaces": workspaces,
                "slots": 2,
                "lease": 3,
                "lifetime": 60,
            }
            (root / "config.json").write_text(json.dumps(policy))
            command = [
                sys.executable,
                "-m",
                "gaia_agent.supervision.protocol",
                "--state",
                str(root / "state"),
                "--config",
                str(root / "config.json"),
                "--token-file",
                str(root / "token"),
            ]
            with (root / "guardian.log").open("w+") as log:
                guardian = subprocess.Popen(command, stdout=log, stderr=log)
                processes.append(guardian)
                socket_path = root / "state" / "guardian-control.sock"
                wait_for(lambda: socket_path.exists() or guardian.poll() is not None)
                assert guardian.poll() is None, "Guardian exited during startup"
                client = Client(root / "state" / "guardian.sock", token)
                assert client.call("health")["healthy"]
                runs = []
                for workspace in workspaces:
                    run, generation = str(uuid4()), str(uuid4())
                    reply = client.call(
                        "start", run=run, generation=generation, workspace=workspace
                    )
                    runs.append((run, generation, reply))
                # Independent controller processes prove liveness, then one dies.
                for run, generation, _reply in runs:
                    code = (
                        "import sys,time; from gaia_agent.supervision.protocol import Client; "
                        "c=Client(sys.argv[1],sys.argv[2]); "
                        '\nwhile True:\n c.call("renew",run=sys.argv[3],generation=sys.argv[4]); time.sleep(.3)'
                    )
                    controller = subprocess.Popen(
                        [
                            sys.executable,
                            "-c",
                            code,
                            str(root / "state" / "guardian.sock"),
                            token,
                            run,
                            generation,
                        ]
                    )
                    processes.append(controller)
                run, generation, reply = runs[0]
                headers = {"Authorization": "Bearer " + reply["token"]}

                def ready():
                    try:
                        with urllib.request.urlopen(
                            urllib.request.Request(
                                reply["url"] + "/ready", headers=headers
                            ),
                            timeout=2,
                        ) as response:
                            return response.status == 200
                    except (OSError, TimeoutError):
                        return False

                wait_for(ready)
                request = urllib.request.Request(
                    reply["url"] + "/v1/gaia/query",
                    data=json.dumps(
                        {"query": "fixture", "run_id": str(uuid4()), "context": []}
                    ).encode(),
                    headers={**headers, "Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    events = response.read(4 * 1024 * 1024).decode()
                    assert "Container inference fixture passed." in events
                processes[1].kill()
                processes[1].wait(timeout=5)
                died_at = time.monotonic()
                wait_for(
                    lambda: client.call("status", run=run, generation=generation)[
                        "state"
                    ]
                    == "stopped",
                    8,
                )
                assert time.monotonic() - died_at < 8
                other_run, other_generation, _ = runs[1]
                assert (
                    client.call("status", run=other_run, generation=other_generation)[
                        "state"
                    ]
                    == "active"
                )
                processes[2].kill()
                processes[2].wait(timeout=5)
                # Kill guardian while the second executor is still alive; restart must reconcile.
                guardian.kill()
                guardian.wait(timeout=5)
                replacement = subprocess.Popen(command, stdout=log, stderr=log)
                processes.append(replacement)

                def reconciled():
                    try:
                        return (
                            client.call(
                                "status", run=other_run, generation=other_generation
                            )["state"]
                            == "stopped"
                        )
                    except (OSError, ValueError):
                        return False

                wait_for(reconciled)
                wait_for(lambda: runtime.inventory(policy["deployment"]) == [], 10)
                fault_run, fault_generation = str(uuid4()), str(uuid4())
                client.call(
                    "start",
                    run=fault_run,
                    generation=fault_generation,
                    workspace=next(iter(workspaces)),
                )
                outage.touch()
                wait_for(
                    lambda: client.call(
                        "status", run=fault_run, generation=fault_generation
                    )["state"]
                    == "unknown",
                    10,
                )
                assert not client.call("health")["healthy"]
                outage.unlink()
                wait_for(
                    lambda: client.call(
                        "status", run=fault_run, generation=fault_generation
                    )["state"]
                    == "stopped",
                    10,
                )
                wait_for(lambda: runtime.inventory(policy["deployment"]) == [], 10)
                print(
                    json.dumps(
                        {
                            "real_query": "passed",
                            "controller_death": "passed",
                            "two_workspaces": "passed",
                            "guardian_restart": "passed",
                            "runtime_transport_outage": "passed",
                            "orphan_containers": 0,
                        }
                    )
                )
                replacement.terminate()
                replacement.wait(timeout=20)
                log.seek(0)
                output = log.read()
                assert token not in output and "fixture-inference-key" not in output
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        # Exact pre-recorded deployment only; inspect each ID before removing.
        if "policy" in locals():
            for container_id in runtime.inventory(policy["deployment"]):
                records = json.loads((root / "state" / "journal.json").read_text())[
                    "runs"
                ]
                matches = [
                    record
                    for record in records.values()
                    if record.get("container_id") == container_id
                ]
                if len(matches) != 1:
                    raise RuntimeError(
                        "Cannot prove test executor ownership; refusing cleanup"
                    )
                identity = matches[0]["identity"]
                runtime.stop(container_id, identity)
                runtime.remove(container_id, identity)
        for container in owned:
            runtime.call("rm", "-f", container)
        for volume in volumes:
            runtime.call("volume", "rm", volume)
        runtime.call("network", "rm", network)
        if "temporary" in locals():
            shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
