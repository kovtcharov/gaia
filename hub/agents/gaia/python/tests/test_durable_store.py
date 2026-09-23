# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Crash-safe admission, replay, interaction and retention contracts."""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from gaia_agent.durable.store import Store, StoreError


@pytest.fixture
def store(tmp_path):
    workspace = str(uuid4())
    database = Store(tmp_path / "private", str(uuid4()), [workspace])
    database.workspace = workspace
    yield database
    database.close()


def session(store):
    return store.create_session(store.workspace, "test")["id"]


def admit(store, session_id, key="key"):
    return store.admit(session_id, key, "hello", 20, {"model": "fixture"})


def success(store, run):
    return store.finish(
        run["id"], "succeeded", {"type": "final", "answer": "done"}, True
    )


def test_same_key_concurrent_admission_has_one_message_and_dispatch(store):
    session_id = session(store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: admit(store, session_id), range(8)))
    assert sum(created for _, created in results) == 1
    assert len({run["id"] for run, _ in results}) == 1
    assert len(store.messages(session_id)["messages"]) == 1


def test_key_conflict_precedes_capacity(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    replay, created = store.admit(session_id, "key", "hello", 20, {"model": "changed"})
    assert not created and replay["policy"] == {"model": "fixture"}
    with pytest.raises(StoreError, match="idempotency_conflict"):
        store.admit(session_id, "key", "different", 20, {})
    assert store.run(run["id"])["request"]["prompt"] == "hello"


def test_workspace_and_session_locks_are_transactional(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    with pytest.raises(StoreError, match="session_busy"):
        admit(store, session_id, "other")
    with pytest.raises(StoreError, match="workspace_busy"):
        admit(store, session(store), "third")
    success(store, run)
    admit(store, session_id, "fourth")


def test_unconfirmed_stop_cannot_release_capacity(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    with pytest.raises(StoreError, match="termination_unconfirmed"):
        store.finish(run["id"], "failed", {"type": "error"}, False)
    assert store.active()
    assert store.events(run["id"])["latest_seq"] == 1


def test_cancellation_commit_wins_late_success(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    store.started(run["id"])
    store.stopping(run["id"])
    terminal = success(store, run)
    assert terminal["state"] == "cancelled" and terminal["outcome_uncertain"]
    assert success(store, run)["terminal_seq"] == terminal["terminal_seq"]
    assert [m["role"] for m in store.messages(session_id)["messages"]] == ["user"]


def test_event_order_cursor_and_expiry(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    for index in range(4):
        store.append(run["id"], {"type": "status", "index": index})
    success(store, run)
    assert [e["seq"] for e in store.events(run["id"], after=2, limit=2)["events"]] == [
        3,
        4,
    ]
    with pytest.raises(StoreError, match="future_cursor"):
        store.events(run["id"], after=100)
    store.expire_history(time.time() + 1)
    with pytest.raises(StoreError, match="event_history_gone"):
        store.events(run["id"])
    with pytest.raises(StoreError, match="acknowledgement_required"):
        admit(store, session_id, "new")
    store.admit(session_id, "new", "continue", 20, {}, True)


def test_interaction_receipt_is_idempotent_and_arguments_not_persisted(store):
    run, _ = admit(store, session(store))
    store.started(run["id"])
    raw_secret = "private-argument-" + uuid4().hex
    interaction = store.create_interaction(
        run["id"], "needs_confirmation", {"arguments": raw_secret}, time.time() + 60
    )
    receipt, deliver = store.receipt(
        run["id"], interaction["id"], run["generation"], "approve"
    )
    assert deliver and receipt["delivery"] == "unknown"
    _, deliver = store.receipt(
        run["id"], interaction["id"], run["generation"], "approve"
    )
    assert not deliver
    with pytest.raises(StoreError, match="already_decided"):
        store.receipt(run["id"], interaction["id"], run["generation"], "deny")
    for path in store.root.glob("service.sqlite3*"):
        assert raw_secret.encode() not in path.read_bytes()


def test_sensitive_question_receipt_uses_submission_identity(store):
    run, _ = admit(store, session(store))
    store.started(run["id"])
    interaction = store.create_interaction(
        run["id"], "needs_input", {"sensitive": True}, time.time() + 60
    )
    submission = str(uuid4())
    assert store.receipt(
        run["id"], interaction["id"], run["generation"], "answered", submission
    )[1]
    success(store, run)
    assert not store.receipt(
        run["id"], interaction["id"], run["generation"], "answered", submission
    )[1]
    with pytest.raises(StoreError, match="stale_generation"):
        store.receipt(
            run["id"], interaction["id"], str(uuid4()), "answered", submission
        )


def test_deleted_history_keeps_idempotency_tombstone(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    with pytest.raises(StoreError, match="session_busy"):
        store.delete_session(session_id)
    success(store, run)
    cleanup = store.delete_session(session_id)
    assert store.cleanup(cleanup["id"])["state"] == "succeeded"
    with pytest.raises(StoreError) as error:
        admit(store, session_id)
    assert error.value.status == 410
    assert not store.db.execute("SELECT * FROM messages").fetchall()


def test_backup_uses_sqlite_backup_and_rejects_active(store, tmp_path):
    run, _ = admit(store, session(store))
    destination = tmp_path / "backup.sqlite3"
    with pytest.raises(StoreError, match="drain"):
        store.backup(destination)
    success(store, run)
    store.backup(destination)
    with sqlite3.connect(destination) as restored:
        assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert restored.execute("SELECT state FROM runs").fetchone()[0] == "succeeded"


def test_storage_full_rolls_back_entire_admission(store):
    session_id = session(store)
    pages = store.db.execute("PRAGMA page_count").fetchone()[0]
    store.db.execute(f"PRAGMA max_page_count={pages}")
    with pytest.raises(sqlite3.OperationalError):
        store.admit(session_id, "huge", "x" * 65536, 20, {})
    assert not store.active()
    assert not store.messages(session_id)["messages"]


def test_periodic_maintenance_expires_history_and_tombstones(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    success(store, run)
    assert store.maintain(time.time() + 8 * 86400) == 1
    assert store.maintain(time.time() + 8 * 86400) == 0
    assert store.session(session_id)["history_status"] == "expired"
    store.delete_session(session_id)
    store.maintain(time.time() + 31 * 86400)
    assert store.db.execute("SELECT count(*) FROM tombstones").fetchone()[0] == 0


def test_newer_schema_and_incomplete_restore_fail_closed(tmp_path):
    directory = tmp_path / "newer"
    directory.mkdir(mode=0o700)
    with sqlite3.connect(directory / "service.sqlite3") as db:
        db.execute("PRAGMA user_version=99")
    for _ in range(2):
        with pytest.raises(StoreError, match="unsupported_schema"):
            Store(directory, str(uuid4()), [])
    (directory / "RESTORE_INCOMPLETE").touch()
    with pytest.raises(StoreError, match="restore_incomplete"):
        Store(directory, str(uuid4()), [])


def test_terminal_expansion_cannot_strand_stopped_run(store):
    session_id = session(store)
    run, _ = admit(store, session_id)
    store.started(run["id"])
    result = store.finish(
        run["id"], "succeeded", {"type": "final", "answer": "x" * 262120}, True
    )
    assert result["state"] == "failed" and result["execution_status"] == "stopped"
    assert store.events(run["id"])["events"][-1]["event"]["status"] == 413
    assert not store.active()
    admit(store, session_id, "next")
