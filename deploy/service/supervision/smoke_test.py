# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Exercise M0 against real Docker; never discovers a runtime endpoint."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from contracts import Identity
from prototype import Runtime, reconcile


def wait_record(path, process, timeout=15):
    """Wait for committed container identity, surfacing guardian early exit."""
    expires = time.monotonic() + timeout
    while time.monotonic() < expires:
        if path.exists():
            record = json.loads(path.read_text())
            if record.get("container_id"):
                return record
        if process.poll() is not None:
            raise RuntimeError("Guardian exited before recording executor")
        time.sleep(0.05)
    raise TimeoutError("Guardian did not commit runtime identity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image", default="alpine:3.21")
    args = parser.parse_args()
    runtime = Runtime(args.endpoint, args.docker)
    runtime.call("image", "inspect", args.image)
    prototype = str(Path(__file__).with_name("prototype.py"))
    for scenario in ("controller-death", "absolute-deadline", "orphan"):
        with tempfile.TemporaryDirectory(prefix="gaia-m0-") as temporary:
            root = Path(temporary)
            base = [
                sys.executable,
                prototype,
                "--state",
                temporary,
                "--endpoint",
                args.endpoint,
                "--docker",
                args.docker,
                "--image",
                args.image,
                "--lease",
                "2",
                "--lifetime",
                "60" if scenario == "controller-death" else "4",
            ]
            controller = subprocess.Popen([*base, "controller"])
            guardian = None
            try:
                guardian = subprocess.Popen(
                    [
                        *base,
                        "guardian",
                        *(["--orphan"] if scenario == "orphan" else []),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if scenario != "orphan":
                    record = wait_record(root / "intent.json", guardian)
                    identity = Identity(**record["identity"])
                    expires = time.monotonic() + 10
                    while not runtime.inspect(record["container_id"], identity)[
                        "State"
                    ]["Running"]:
                        if time.monotonic() >= expires:
                            raise TimeoutError("Executor did not start")
                        time.sleep(0.05)
                    processes = runtime.call("top", record["container_id"])
                    if "sleep 600" not in processes:
                        raise AssertionError("Expected descendant process not observed")
                    observed_start = time.monotonic()
                    if scenario == "controller-death":
                        controller.kill()
                        controller.wait(timeout=5)
                stdout, stderr = guardian.communicate(timeout=10)
                if guardian.returncode:
                    raise RuntimeError(stderr)
                if scenario == "orphan":
                    ids = reconcile(runtime, root / "intent.json")
                else:
                    elapsed = time.monotonic() - observed_start
                    assert elapsed < 8, f"Termination exceeded bound: {elapsed:.2f}s"
                    if scenario == "absolute-deadline":
                        assert controller.poll() is None, "Controller must remain live"
                        assert elapsed >= 2, "Absolute deadline fired too early"
                    ids = [record["container_id"]]
                    assert any(
                        json.loads(line).get("stopped") == ids[0]
                        for line in stdout.splitlines()
                    )
                record = json.loads((root / "intent.json").read_text())
                identity = Identity(**record["identity"])
                for container_id in ids:
                    assert not runtime.inspect(container_id, identity)["State"][
                        "Running"
                    ]
                print(
                    json.dumps(
                        {
                            "scenario": scenario,
                            "passed": True,
                            "guardian_output": stdout.splitlines(),
                        }
                    ),
                    flush=True,
                )
            finally:
                if controller.poll() is None:
                    controller.kill()
                controller.wait(timeout=5)
                if guardian is not None and guardian.poll() is None:
                    guardian.kill()
                    guardian.wait(timeout=5)
                journal = root / "intent.json"
                if journal.exists():
                    record = json.loads(journal.read_text())
                    identity = Identity(**record["identity"])
                    for container_id in reconcile(runtime, journal):
                        runtime.inspect(container_id, identity)
                        runtime.call("rm", container_id)


if __name__ == "__main__":
    main()
