# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Back up or restore both worker volumes while an operator holds them stopped.

This command requires exclusive operator control: no other service may mount
these volumes or start the container during the operation. Archives contain
private workspace and conversation data; protect them as credentials.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def docker(*args):
    """Fail loudly on runtime errors; never serialize container environment."""
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, timeout=300
    ).stdout.strip()


def validate_mounts(mounts):
    """Never copy into a disposable layer or chown an operator bind mount."""
    selected = [
        mount for mount in mounts if mount["Destination"] in {"/data", "/workspace"}
    ]
    if len(selected) != 2 or {m["Destination"] for m in selected} != {
        "/data",
        "/workspace",
    }:
        raise ValueError("Worker requires named volumes at /data and /workspace")
    if any(
        m["Type"] != "volume" or not m.get("Name") or not m.get("RW") for m in selected
    ):
        raise ValueError("Worker backup requires writable named volumes")
    if selected[0]["Name"] == selected[1]["Name"]:
        raise ValueError("State and workspace require distinct volumes")
    if any(m["Destination"].startswith(("/data/", "/workspace/")) for m in mounts):
        raise ValueError("Nested mounts are not supported for backup")


def stopped(container):
    state = json.loads(docker("inspect", "--format", "{{json .State}}", container))
    if state["Running"] or state.get("Restarting") or state.get("Paused"):
        raise ValueError("Stop the worker and hold exclusive volume ownership first")


def inventory(root):
    """Hash regular data, retaining symlink metadata without following targets."""
    records = {}
    for volume in ("data", "workspace"):
        base = root / volume
        if not base.is_dir() or base.is_symlink():
            raise ValueError("Backup must contain data and workspace directories")
        for path in sorted(base.rglob("*")):
            relative = str(path.relative_to(root))
            if path.is_symlink():
                records[relative] = {"symlink": os.readlink(path)}
            elif path.is_file():
                with path.open("rb") as handle:
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                records[relative] = {"sha256": digest, "size": path.stat().st_size}
            elif path.is_dir():
                records[relative] = {"directory": True}
            else:
                raise ValueError("Backup contains unsupported special files")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("backup", "restore", "verify"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--container")
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.mode == "verify":
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["schema"] != 1 or manifest["files"] != inventory(root):
            raise ValueError("Backup integrity verification failed")
        print("Backup integrity verified")
        return
    if not args.container:
        parser.error("--container is required for backup and restore")
    stopped(args.container)
    validate_mounts(
        json.loads(docker("inspect", "--format", "{{json .Mounts}}", args.container))
    )
    if args.mode == "backup":
        root.mkdir(mode=0o700, parents=False, exist_ok=False)
        for volume in ("data", "workspace"):
            docker("cp", f"{args.container}:/{volume}", str(root / volume))
        stopped(args.container)
        image = docker("inspect", "--format", "{{.Image}}", args.container)
        manifest = {"schema": 1, "image": image, "files": inventory(root)}
        (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
        print("Stopped worker backup created and verified")
    else:
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest["schema"] != 1 or manifest["files"] != inventory(root):
            raise ValueError("Backup integrity verification failed")
        # Restores are deliberately restricted to a fresh, never-started target.
        state = json.loads(
            docker("inspect", "--format", "{{json .State}}", args.container)
        )
        if state["Status"] != "created":
            raise ValueError("Restore requires a newly created, never-started worker")
        image = docker("inspect", "--format", "{{.Image}}", args.container)
        if image != manifest["image"]:
            raise ValueError(
                "Restore the backed-up image first, then upgrade separately"
            )
        helper = [
            "run",
            "--rm",
            "--network=none",
            "--user=0:0",
            "--volumes-from",
            args.container,
            "--entrypoint=/bin/sh",
            image,
            "-c",
        ]
        docker(
            *helper,
            'entries=$(find /data /workspace -mindepth 1 -print -quit) && test -z "$entries"',
        )
        for volume in ("data", "workspace"):
            docker("cp", str(root / volume) + "/.", f"{args.container}:/{volume}")
        docker(*helper, "chown -hR 10001:10001 /data /workspace")
        stopped(args.container)
        print("Worker restored; verify volume ownership before starting")


if __name__ == "__main__":
    main()
