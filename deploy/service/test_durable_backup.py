# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Full backups refuse dangling managed artifacts and corrupted data."""

import json
from uuid import uuid4

import pytest
from durable_backup import checksum, verify
from gaia_agent.durable.store import Store


def test_manifest_cannot_hide_missing_referenced_artifact(tmp_path):
    workspace = str(uuid4())
    store = Store(tmp_path / "state", str(uuid4()), [workspace])
    try:
        session = store.create_session(workspace, "")
        artifact = store.put_artifact(session["id"], "private document")
        root = tmp_path / "backup"
        root.mkdir()
        store.backup(root / "service.sqlite3")
        manifest = {
            "schema": 2,
            "database_sha256": checksum(root / "service.sqlite3"),
            "workspaces": {},
            "artifacts": {},
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="missing referenced artifacts"):
            verify(root)
        (root / "artifacts").mkdir()
        path = root / "artifacts" / artifact["id"]
        path.write_text("private document")
        manifest["artifacts"][artifact["id"]] = checksum(path)
        (root / "manifest.json").write_text(json.dumps(manifest))
        verify(root)
        path.write_text("corrupted")
        with pytest.raises(ValueError, match="integrity"):
            verify(root)
    finally:
        store.close()
