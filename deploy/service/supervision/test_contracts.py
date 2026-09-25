# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""M0 protocol and recovery fault-injection tests."""

import json
from uuid import uuid4

import pytest
from contracts import Identity, canonical_request, deadline
from prototype import persist, reconcile


def test_canonical_defaults_and_unicode():
    session = str(uuid4())
    assert canonical_request("hello", session) == canonical_request(
        "hello", session, 20
    )
    assert canonical_request("é", session) != canonical_request("é", session)


@pytest.mark.parametrize("steps", [True, 0, 21, 1.0, None])
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        canonical_request("hello", str(uuid4()), steps)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_lease(duration):
    with pytest.raises(ValueError):
        deadline(1, duration)


def test_identity_rejects_noncanonical_uuid():
    with pytest.raises(ValueError):
        Identity("bad", *(str(uuid4()) for _ in range(3)))


class FakeRuntime:
    """Fault-injection runtime checks that recovery binds identity before stop."""

    def __init__(self, journal, ids):
        self.journal = journal
        self.ids = ids
        self.stopped = []

    def inventory(self, identity):
        return self.ids

    def inspect(self, container_id, identity):
        return {"State": {"Running": True}}

    def stop(self, container_id, identity):
        assert json.loads(self.journal.read_text())["container_id"] == container_id
        self.stopped.append(container_id)


def test_orphan_binding_is_durable_before_stop(tmp_path):
    journal = tmp_path / "intent.json"
    identity = Identity(*(str(uuid4()) for _ in range(4)))
    persist(journal, {"identity": vars(identity), "container_id": None})
    runtime = FakeRuntime(journal, ["a" * 64])
    assert reconcile(runtime, journal) == ["a" * 64]
    assert runtime.stopped == ["a" * 64]


@pytest.mark.parametrize("ids", [[], ["a" * 64, "b" * 64]])
def test_ambiguous_identity_never_stops(tmp_path, ids):
    journal = tmp_path / "intent.json"
    persist(
        journal,
        {
            "identity": vars(Identity(*(str(uuid4()) for _ in range(4)))),
            "container_id": None,
        },
    )
    runtime = FakeRuntime(journal, ids)
    with pytest.raises(RuntimeError, match="Missing or ambiguous"):
        reconcile(runtime, journal)
    assert not runtime.stopped


def test_runtime_failure_surfaces_without_modifying_journal(tmp_path):
    journal = tmp_path / "intent.json"
    persist(
        journal,
        {
            "identity": vars(Identity(*(str(uuid4()) for _ in range(4)))),
            "container_id": "a" * 64,
        },
    )
    original = journal.read_bytes()
    runtime = FakeRuntime(journal, ["b" * 64])
    with pytest.raises(RuntimeError, match="disagreement"):
        reconcile(runtime, journal)
    assert journal.read_bytes() == original
    assert not runtime.stopped


def test_runtime_unavailable_never_reports_stopped(tmp_path):
    journal = tmp_path / "intent.json"
    persist(
        journal,
        {
            "identity": vars(Identity(*(str(uuid4()) for _ in range(4)))),
            "container_id": "a" * 64,
        },
    )

    class UnavailableRuntime(FakeRuntime):
        def stop(self, container_id, identity):
            raise TimeoutError("Runtime unavailable")

    runtime = UnavailableRuntime(journal, ["a" * 64])
    with pytest.raises(TimeoutError, match="unavailable"):
        reconcile(runtime, journal)
    assert not runtime.stopped


def test_runtime_inspection_failure_blocks_stop(tmp_path):
    journal = tmp_path / "intent.json"
    persist(
        journal,
        {
            "identity": vars(Identity(*(str(uuid4()) for _ in range(4)))),
            "container_id": None,
        },
    )

    class FailedInspection(FakeRuntime):
        def inspect(self, container_id, identity):
            raise TimeoutError("Inspection unavailable")

    runtime = FailedInspection(journal, ["a" * 64])
    with pytest.raises(TimeoutError, match="Inspection"):
        reconcile(runtime, journal)
    assert json.loads(journal.read_text())["container_id"] is None
    assert not runtime.stopped
