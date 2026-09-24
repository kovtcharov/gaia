# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Operations, migration and adversarial data-lifecycle qualification."""

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from gaia_agent.durable.operations import Telemetry, normalize_usage, validate_rates
from gaia_agent.durable.store import Store, StoreError


@pytest.fixture
def store(tmp_path):
    workspace = str(uuid4())
    value = Store(tmp_path / "state", str(uuid4()), [workspace])
    value.workspace = workspace
    yield value
    value.close()


def test_beta_upgrade_and_compatible_rollback_backup(tmp_path):
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    fixture = (
        Path(__file__).resolve().parents[5] / "deploy/service/fixtures/schema-v1.sql"
    )
    with sqlite3.connect(root / "service.sqlite3") as database:
        database.executescript(fixture.read_text())
    store = Store(
        root,
        "11111111-1111-4111-8111-111111111111",
        ["22222222-2222-4222-8222-222222222222"],
    )
    try:
        assert store.db.execute("PRAGMA user_version").fetchone()[0] == 2
        run = store.db.execute("SELECT id FROM runs").fetchone()[0]
        assert store.run(run)["usage"]["status"] == "unavailable"
        assert (
            store.db.execute(
                "SELECT content FROM messages WHERE role='assistant'"
            ).fetchone()[0]
            == "preserved beta answer"
        )
        with sqlite3.connect(root / "before-schema-2.sqlite3") as rollback:
            assert rollback.execute("PRAGMA user_version").fetchone()[0] == 1
            assert "usage" not in {
                row[1] for row in rollback.execute("PRAGMA table_info(runs)")
            }
            restored = sqlite3.connect(tmp_path / "rolled-back.sqlite3")
            try:
                rollback.backup(restored)
                assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                assert (
                    restored.execute("SELECT count(*) FROM messages").fetchone()[0] == 2
                )
            finally:
                restored.close()
    finally:
        store.close()


def test_artifact_cleanup_failure_visible_retryable_and_operator_files_retained(
    store, tmp_path, monkeypatch
):
    session = store.create_session(store.workspace, "")
    artifact = store.put_artifact(session["id"], "private document")
    operator = tmp_path / "operator-file"
    operator.write_text("retain")
    assert (
        store.read_artifact(session["id"], artifact["id"])["content"]
        == "private document"
    )
    job = store.delete_session(session["id"])
    assert job["state"] == "pending" and job["operator_workspace_retained"]
    original = Path.unlink

    def fail(path, *args, **kwargs):
        if path.name == artifact["id"]:
            raise PermissionError("private path must not enter diagnostics")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail)
    store.cleanup_artifacts()
    assert store.cleanup(job["id"])["state"] == "failed"
    assert store.cleanup(job["id"])["error"] == "artifact_delete_failed"
    monkeypatch.setattr(Path, "unlink", original)
    store.cleanup_artifacts()
    assert store.cleanup(job["id"])["state"] == "succeeded"
    assert operator.read_text() == "retain"


def test_adversarial_artifact_path_and_capacity(store):
    session = store.create_session(store.workspace, "")
    with pytest.raises(StoreError, match="artifact_too_large"):
        store.put_artifact(session["id"], "x" * 524289)
    with pytest.raises(StoreError, match="artifact_not_found"):
        store.read_artifact(session["id"], "../../secret")
    for _ in range(64):
        store.put_artifact(session["id"], "")
    with pytest.raises(StoreError, match="artifact_capacity"):
        store.put_artifact(session["id"], "")


def test_usage_unknown_partial_and_versioned_estimate():
    missing = normalize_usage({}, "model", "succeeded")
    assert (
        missing["status"] == "unavailable"
        and missing["input_tokens"] is None
        and missing["estimated_cost"] is None
    )
    partial = normalize_usage({"usage": {"tokens": 20}}, "model", "cancelled")
    assert (
        partial["status"] == "incomplete"
        and partial["output_tokens"] == 20
        and partial["input_tokens"] is None
    )
    rates = validate_rates(
        {
            "version": "operator-1",
            "models": {"model": {"input_per_million": 1, "output_per_million": 2}},
        }
    )
    result = normalize_usage(
        {"usage": {"input_tokens": 100, "output_tokens": 20}},
        "model",
        "succeeded",
        rates,
    )
    assert result["status"] == "reported" and result["estimated_cost"][
        "amount"
    ] == pytest.approx(0.00014)
    assert (
        normalize_usage({"usage": {"input_tokens": True}}, "model", "succeeded")[
            "input_tokens"
        ]
        is None
    )
    with pytest.raises(ValueError):
        validate_rates(
            {
                "version": "bad",
                "models": {
                    "model": {
                        "input_per_million": float("nan"),
                        "output_per_million": 2,
                    }
                },
            }
        )


def test_trace_failure_is_bounded_and_excludes_content(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir(mode=0o700)
    (trace / "events.jsonl").mkdir()
    telemetry = Telemetry(trace)
    run = {
        "id": str(uuid4()),
        "generation": str(uuid4()),
        "state": "succeeded",
        "execution_status": "stopped",
        "outcome_uncertain": False,
        "usage": {"status": "unavailable"},
        "created_at": 1,
        "ended_at": 2,
        "prompt": "secret",
        "nested": {"token": "secret"},
    }
    for _ in range(1000):
        telemetry.terminal(run)
    telemetry.close()
    snapshot = telemetry.snapshot()
    assert any(
        row["name"] == "telemetry_dropped" and row["value"] > 0
        for row in snapshot["counters"]
    )
    assert "secret" not in json.dumps(snapshot) and snapshot["trace_queue_items"] <= 128
    with pytest.raises(ValueError):
        telemetry.increment("outcome", "private-prompt")


def test_thousand_duplicate_admissions_under_contention(store):
    session = store.create_session(store.workspace, "")
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(
            pool.map(
                lambda _: store.admit(
                    session["id"], "same-key", "adversarial replay", 20, {}
                ),
                range(1000),
            )
        )
    assert sum(created for _, created in results) == 1
    assert store.db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert store.db.execute("SELECT count(*) FROM messages").fetchone()[0] == 1


def test_event_flood_preserves_terminal_reserve_and_capacity(store):
    session = store.create_session(store.workspace, "")
    run, _ = store.admit(session["id"], "flood", "hello", 20, {})
    event = {"type": "chunk", "content": "x" * 250000}
    with pytest.raises(StoreError, match="event_storage_limit"):
        for _ in range(100):
            store.append(run["id"], event)
    terminal = store.finish(run["id"], "failed", {"type": "error", "status": 413}, True)
    assert terminal["ended_at"] is not None and not store.active()
    assert terminal["event_bytes"] <= 16 * 1024 * 1024


def test_retention_removes_metadata_without_reusing_live_keys(store):
    session = store.create_session(store.workspace, "")
    run, _ = store.admit(session["id"], "old", "hello", 20, {})
    store.finish(run["id"], "succeeded", {"type": "final", "answer": "old"}, True)
    now = time.time()
    store.maintain(now + 2 * 86400, content_days=0, metadata_days=1)
    with pytest.raises(StoreError, match="run_history_gone"):
        store.admit(session["id"], "old", "hello", 20, {})
    assert store.session(session["id"])["history_status"] == "expired"


def test_orphan_cleanup_progresses_past_retained_entries(store):
    for _ in range(3):
        session = store.create_session(store.workspace, "")
        for _ in range(64):
            store.put_artifact(session["id"], "retained")
    orphan = store.root / "artifacts" / str(uuid4())
    orphan.write_text("uncommitted")
    for _ in range(4):
        store.cleanup_artifacts()
    assert not orphan.exists()
    assert store.db.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 192


def test_artifact_namespace_fsync_precedes_commit(store, monkeypatch):
    session = store.create_session(store.workspace, "")
    ordering = []
    original = store._fsync_directory

    def fsync(path):
        ordering.append(path.name)
        assert store.db.in_transaction
        original(path)

    monkeypatch.setattr(store, "_fsync_directory", fsync)
    artifact = store.put_artifact(session["id"], "content")
    assert ordering == ["state", "artifacts"]
    assert store.read_artifact(session["id"], artifact["id"])["content"] == "content"


def test_stuck_inference_probe_has_one_worker_and_bounded_observation(monkeypatch):
    import threading

    from gaia_agent.durable.operations import InferenceProbe

    probe = InferenceProbe("http://127.0.0.1:1/api/v1", "model")
    release = threading.Event()
    monkeypatch.setattr(
        probe,
        "_perform",
        lambda: (release.wait(5), {"ready": True, "code": "ready"})[1],
    )
    try:
        started = time.monotonic()
        assert not probe.check()["ready"]
        assert time.monotonic() - started < 3.5
        thread = probe.worker
        for _ in range(100):
            assert probe.check()["code"] == "inference_probe_timeout"
            assert probe.worker is thread
    finally:
        release.set()
        probe.worker.join(1)
        probe.close()
