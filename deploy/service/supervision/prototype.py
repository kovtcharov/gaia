# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""M0 feasibility probe; deliberately not a supported service deployment.

The guardian owns Docker and a durable creation-intent journal. A separate
controller process renews a local lease. Killing the controller cannot kill
the guardian or extend the accepted absolute execution deadline.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from contracts import Identity, deadline


class Runtime:
    """Bounded Docker CLI adapter with explicit runtime endpoint."""

    def __init__(self, endpoint, executable="docker"):
        if not endpoint.startswith("unix://"):
            raise ValueError(
                "Prototype requires an explicit local unix:// Docker endpoint"
            )
        self.endpoint = endpoint
        self.executable = executable

    def call(self, *arguments):
        result = subprocess.run(
            [self.executable, "--host", self.endpoint, *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()

    def inventory(self, identity):
        arguments = ["ps", "-aq", "--no-trunc"]
        for key, value in identity.labels().items():
            arguments += ["--filter", f"label={key}={value}"]
        return self.call(*arguments).splitlines()

    def inspect(self, container_id, identity):
        value = json.loads(self.call("inspect", container_id))[0]
        labels = value["Config"]["Labels"] or {}
        if value["Id"] != container_id or any(
            labels.get(key) != expected for key, expected in identity.labels().items()
        ):
            raise RuntimeError("Runtime identity mismatch; refusing mutation")
        return value

    def stop(self, container_id, identity):
        state = self.inspect(container_id, identity)["State"]
        if state["Running"]:
            self.call("kill", container_id)
        if self.inspect(container_id, identity)["State"]["Running"]:
            raise RuntimeError("Runtime did not confirm executor termination")


def persist(path, record):
    """Durably record intent before creation, then runtime ID before start."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        os.chmod(temporary, 0o600)
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def reconcile(runtime, journal):
    """Find even an executor created before its ID could be committed."""
    record = json.loads(journal.read_text(encoding="utf-8"))
    identity = Identity(**record["identity"])
    inventory = runtime.inventory(identity)
    if record.get("container_id") and record["container_id"] not in inventory:
        raise RuntimeError("Journal/runtime disagreement; admission remains closed")
    if len(inventory) != 1:
        raise RuntimeError(
            "Missing or ambiguous executor identity; admission remains closed"
        )
    container_id = inventory[0]
    runtime.inspect(container_id, identity)
    if not record.get("container_id"):
        record["container_id"] = container_id
        persist(journal, record)
    runtime.stop(container_id, identity)
    return inventory


def guardian(args):
    """Create stopped, record, start, and enforce controller and absolute leases."""
    runtime = Runtime(args.endpoint, args.docker)
    root = Path(args.state)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    journal = root / "intent.json"
    if journal.exists():
        stopped = reconcile(runtime, journal)
        print(json.dumps({"reconciled": stopped}), flush=True)
        return
    identity = Identity(*(str(uuid4()) for _ in range(4)))
    record = {"identity": vars(identity), "container_id": None}
    persist(journal, record)
    labels = []
    for key, value in identity.labels().items():
        labels += ["--label", f"{key}={value}"]
    container_id = runtime.call(
        "create",
        *labels,
        "--restart=no",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=32",
        "--memory=64m",
        "--cpus=0.5",
        "--user=65534:65534",
        "--entrypoint=/bin/sh",
        args.image,
        "-c",
        "sleep 600 & wait",
    )
    if args.orphan:
        print(json.dumps({"orphan_created": container_id}), flush=True)
        return
    record["container_id"] = container_id
    persist(journal, record)
    try:
        if runtime.inspect(container_id, identity)["State"]["Running"]:
            raise RuntimeError("Executor unexpectedly started before dispatch")
        runtime.call("start", container_id)
        if not runtime.inspect(container_id, identity)["State"]["Running"]:
            raise RuntimeError("Executor did not start")
        expires = deadline(time.monotonic(), args.lifetime)
        lease = deadline(time.monotonic(), args.lease)
        previous = None
        print(json.dumps({"started": container_id}), flush=True)
        while time.monotonic() < min(expires, lease):
            heartbeat = root / "heartbeat"
            if heartbeat.exists():
                current = heartbeat.stat().st_mtime_ns
                if current != previous:
                    previous = current
                    lease = deadline(time.monotonic(), args.lease)
            time.sleep(0.05)
    finally:
        runtime.stop(container_id, identity)
    print(json.dumps({"stopped": container_id}), flush=True)


def main():
    """Run the independent guardian or its expendable heartbeat controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("guardian", "controller", "reconcile"))
    parser.add_argument("--state", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image", default="alpine:3.21")
    parser.add_argument("--lease", type=float, default=2)
    parser.add_argument("--lifetime", type=float, default=10)
    parser.add_argument("--orphan", action="store_true")
    args = parser.parse_args()
    deadline(0, args.lease)
    deadline(0, args.lifetime)
    if args.mode == "guardian":
        guardian(args)
    elif args.mode == "reconcile":
        print(
            json.dumps(
                {
                    "reconciled": reconcile(
                        Runtime(args.endpoint, args.docker),
                        Path(args.state) / "intent.json",
                    )
                }
            ),
            flush=True,
        )
    else:
        while True:
            (Path(args.state) / "heartbeat").touch(mode=0o600)
            time.sleep(0.2)


if __name__ == "__main__":
    main()
