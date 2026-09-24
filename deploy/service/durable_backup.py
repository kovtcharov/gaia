# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Offline coherent backup and clone restore; stop controller and guardian first.

The operator must hold exclusive control over configured workspace volumes for
this operation. Credentials and guardian journals are deliberately not copied.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

from gaia_agent.durable.store import SCHEMA_VERSION, Store
from gaia_agent.supervision.guardian import validate_policy
from gaia_agent.supervision.runtime import DockerRuntime
from worker_backup import inventory


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def holder(runtime, policy, mapping):
    container = runtime.call(
        "create",
        "--network=none",
        "--user=0:0",
        "--mount",
        f"type=volume,src={mapping['data']},dst=/data,volume-nocopy",
        "--mount",
        f"type=volume,src={mapping['workspace']},dst=/workspace,volume-nocopy",
        "--entrypoint=/bin/sh",
        policy["image"],
        "-c",
        "true",
    )
    try:
        yield container
    finally:
        runtime.call("rm", container)


def require_idle(runtime, policy):
    if runtime.inventory(policy["deployment"]):
        raise ValueError("Reconcile and remove deployment executors before backup")
    for mapping in policy["workspaces"].values():
        for volume in mapping.values():
            runtime.call("volume", "inspect", volume)
            if runtime.call("ps", "-q", "--filter", "volume=" + volume):
                raise ValueError("Workspace is mounted by a running container")


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] not in (1, SCHEMA_VERSION) or manifest[
        "database_sha256"
    ] != checksum(root / "service.sqlite3"):
        raise ValueError("Database backup integrity failed")
    for workspace, files in manifest["workspaces"].items():
        if str(UUID(workspace)) != workspace:
            raise ValueError("Invalid workspace identity")
        if inventory(root / workspace) != files:
            raise ValueError("Workspace backup integrity failed")
    for artifact, digest in manifest.get("artifacts", {}).items():
        if (
            str(UUID(artifact)) != artifact
            or checksum(root / "artifacts" / artifact) != digest
        ):
            raise ValueError("Artifact backup integrity failed")
    with sqlite3.connect(
        (root / "service.sqlite3").resolve().as_uri() + "?mode=ro", uri=True
    ) as database:
        if manifest["schema"] >= 2:
            expected = {row[0] for row in database.execute("SELECT id FROM artifacts")}
            if not expected.issubset(manifest.get("artifacts", {})):
                raise ValueError("Backup is missing referenced artifacts")
    return manifest


def backup(root, state, guardian_state, policy, runtime):
    if (
        not (state / "service.sqlite3").is_file()
        or not (guardian_state / "owner.lock").is_file()
    ):
        raise ValueError("Backup requires existing controller and guardian state")
    with (guardian_state / "owner.lock").open("a+") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(
            state, policy["deployment"], policy["workspaces"], policy["slots"]
        )
        try:
            require_idle(runtime, policy)
            root.mkdir(mode=0o700, exist_ok=False)
            store.backup(root / "service.sqlite3")
            manifest = {
                "schema": SCHEMA_VERSION,
                "deployment": policy["deployment"],
                "image": policy["image"],
                "workspaces": {},
                "database_sha256": checksum(root / "service.sqlite3"),
            }
            for workspace, mapping in policy["workspaces"].items():
                destination = root / workspace
                destination.mkdir(mode=0o700)
                with holder(runtime, policy, mapping) as container:
                    for volume in ("data", "workspace"):
                        runtime.call(
                            "cp", f"{container}:/{volume}", str(destination / volume)
                        )
                manifest["workspaces"][workspace] = inventory(destination)
            artifacts = state / "artifacts"
            manifest["artifacts"] = {}
            if artifacts.exists():
                if artifacts.is_symlink() or any(
                    path.is_symlink() or not path.is_file()
                    for path in artifacts.iterdir()
                ):
                    raise ValueError("Invalid service artifact directory")
                shutil.copytree(artifacts, root / "artifacts")
                manifest["artifacts"] = {
                    path.name: checksum(path) for path in artifacts.iterdir()
                }
            require_idle(runtime, policy)
            (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
            verify(root)
        finally:
            store.close()


def restore(root, state, policy, runtime):
    manifest = verify(root)
    if policy["deployment"] == manifest["deployment"]:
        raise ValueError("Clone restore requires a new deployment UUID")
    if policy["image"] != manifest["image"] or set(policy["workspaces"]) != set(
        manifest["workspaces"]
    ):
        raise ValueError("Restore requires the backed-up image and workspace handles")
    if state.exists():
        raise ValueError("Restore requires a new controller state directory")
    existing = set(runtime.call("volume", "ls", "--format", "{{.Name}}").splitlines())
    names = {
        name for mapping in policy["workspaces"].values() for name in mapping.values()
    }
    if names & existing:
        raise ValueError("Restore requires new, distinct volume names")
    # A partial restore remains visibly incomplete; discard its new volumes before retrying.
    state.mkdir(mode=0o700, parents=False)
    (state / "RESTORE_INCOMPLETE").touch(mode=0o600)
    for workspace, mapping in policy["workspaces"].items():
        for name in mapping.values():
            runtime.call("volume", "create", name)
        with holder(runtime, policy, mapping) as container:
            for volume in ("data", "workspace"):
                runtime.call(
                    "cp",
                    str(root / workspace / volume) + "/.",
                    f"{container}:/{volume}",
                )
            runtime.call(
                "run",
                "--rm",
                "--network=none",
                "--user=0:0",
                "--volumes-from",
                container,
                "--entrypoint=/bin/sh",
                policy["image"],
                "-c",
                "chown -hR 10001:10001 /data /workspace",
            )
    source = sqlite3.connect(f"file:{root / 'service.sqlite3'}?mode=ro", uri=True)
    target_path = state / "service.sqlite3"
    descriptor = os.open(target_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    target = sqlite3.connect(target_path)
    try:
        source.backup(target)
        if target.execute(
            "SELECT count(*) FROM runs WHERE ended_at IS NULL"
        ).fetchone()[0]:
            raise ValueError("Backup contains active executors")
        target.execute("UPDATE deployment SET id=?", (policy["deployment"],))
        for (run_id,) in target.execute("SELECT id FROM runs").fetchall():
            target.execute(
                "UPDATE runs SET generation=? WHERE id=?", (str(uuid4()), run_id)
            )
        target.commit()
        if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Restored database integrity failed")
    finally:
        source.close()
        target.close()
    if manifest.get("artifacts"):
        shutil.copytree(root / "artifacts", state / "artifacts")
    (state / "RESTORE_INCOMPLETE").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("backup", "restore", "verify"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--guardian-state", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        verify(args.directory)
    else:
        if args.state is None or args.config is None:
            parser.error("--state and --config are required")
        policy = validate_policy(json.loads(args.config.read_text()))
        runtime = DockerRuntime(
            policy["endpoint"], policy.get("docker", "docker"), timeout=300
        )
        if args.mode == "backup":
            if args.guardian_state is None:
                parser.error("--guardian-state is required")
            backup(args.directory, args.state, args.guardian_state, policy, runtime)
        else:
            restore(args.directory, args.state, policy, runtime)
    print("Durable " + args.mode + " verified")


if __name__ == "__main__":
    main()
