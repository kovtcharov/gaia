# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Round-trip stopped worker volumes, verifying data and restored ownership."""

import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from worker_backup import docker


def main():
    image = "gaia-service:test"
    prefix = "gaia-backup-" + uuid4().hex
    source, target = prefix + "-source", prefix + "-target"
    volumes = [
        prefix + suffix
        for suffix in ("-data", "-work", "-restore-data", "-restore-work")
    ]
    containers = []
    created_volumes = []
    try:
        for volume in volumes:
            docker("volume", "create", volume)
            created_volumes.append(volume)
        for name, data, work in (
            (source, volumes[0], volumes[1]),
            (target, volumes[2], volumes[3]),
        ):
            docker(
                "create",
                "--name",
                name,
                "--mount",
                f"type=volume,src={data},dst=/data,volume-nocopy",
                "--mount",
                f"type=volume,src={work},dst=/workspace,volume-nocopy",
                "--entrypoint=/bin/sh",
                image,
                "-c",
                "printf 'backup-state' > /data/fact; printf 'backup-workspace' > /workspace/fact",
            )
            containers.append(name)
        docker(
            "run",
            "--rm",
            "--user=0:0",
            "--volumes-from",
            source,
            "--entrypoint=/bin/sh",
            image,
            "-c",
            "chown 10001:10001 /data /workspace",
        )
        docker("start", "-a", source)
        script = str(Path(__file__).with_name("worker_backup.py"))
        with tempfile.TemporaryDirectory(prefix="gaia-backup-test-") as temporary:
            backup = str(Path(temporary) / "backup")
            for mode, container in (("backup", source), ("restore", target)):
                subprocess.run(
                    [
                        sys.executable,
                        script,
                        mode,
                        "--container",
                        container,
                        "--directory",
                        backup,
                    ],
                    check=True,
                    timeout=120,
                )
            result = docker(
                "run",
                "--rm",
                "--volumes-from",
                target,
                "--entrypoint=/bin/sh",
                image,
                "-c",
                'test "$(cat /data/fact)" = backup-state && '
                'test "$(cat /workspace/fact)" = backup-workspace && '
                "test -w /data/fact && test -w /workspace/fact && echo roundtrip-passed",
            )
            assert result == "roundtrip-passed", result
            print(result)
    finally:
        for container in containers:
            docker("rm", "-f", container)
        for volume in created_volumes:
            docker("volume", "rm", volume)


if __name__ == "__main__":
    main()
