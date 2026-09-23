# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Worker backup integrity checks independent of Docker."""

import os

import pytest
from worker_backup import inventory, validate_mounts


def test_inventory_detects_changes_and_preserves_links(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "workspace").mkdir()
    document = tmp_path / "workspace" / "document"
    document.write_text("planted fact")
    (tmp_path / "data" / "link").symlink_to("/opt/embedded")
    before = inventory(tmp_path)
    assert before["data/link"] == {"symlink": "/opt/embedded"}
    document.write_text("changed fact")
    assert inventory(tmp_path) != before


def test_rejects_missing_volume(tmp_path):
    with pytest.raises(ValueError, match="directories"):
        inventory(tmp_path)


def test_rejects_special_files(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "workspace").mkdir()
    os.mkfifo(tmp_path / "data" / "pipe")
    with pytest.raises(ValueError, match="special"):
        inventory(tmp_path)


@pytest.mark.parametrize(
    "mounts",
    [
        [],
        [
            {"Destination": "/data", "Type": "bind", "RW": True},
            {"Destination": "/workspace", "Type": "volume", "Name": "work", "RW": True},
        ],
        [
            {"Destination": "/data", "Type": "volume", "Name": "same", "RW": True},
            {"Destination": "/workspace", "Type": "volume", "Name": "same", "RW": True},
        ],
    ],
)
def test_unsafe_mount_layouts_rejected(mounts):
    with pytest.raises(ValueError):
        validate_mounts(mounts)


def test_distinct_writable_volume_layout():
    validate_mounts(
        [
            {"Destination": "/data", "Type": "volume", "Name": "data", "RW": True},
            {"Destination": "/workspace", "Type": "volume", "Name": "work", "RW": True},
        ]
    )
