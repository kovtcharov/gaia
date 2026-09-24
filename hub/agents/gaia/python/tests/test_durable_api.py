# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Durable HTTP/controller tests, including disconnect-independent execution."""

import json
import threading
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from gaia_agent import caller_auth
from gaia_agent.durable.api import PREFIX, build_app
from gaia_agent.durable.controller import Controller
from gaia_agent.durable.store import StoreError


class Guardian:
    def __init__(self):
        self.workspace = str(uuid4())
        self.starts = []
        self.cancels = []
        self.caps = {
            "protocol": 2,
            "deployment": str(uuid4()),
            "workspaces": [self.workspace],
            "slots": 1,
            "lease": 10,
            "model": "fixture",
            "lifetime": 300,
        }

    def call(self, operation, **payload):
        if operation == "capabilities":
            return self.caps
        if operation == "fence":
            return {}
        if operation == "start":
            self.starts.append(payload["run"])
            return {
                "url": "http://127.0.0.1:9999",
                "token": "private-worker-credential",
                "deadline": time.time() + 300,
            }
        if operation == "cancel":
            self.cancels.append(payload["run"])
            return {"state": "stopped"}
        if operation == "renew":
            return {}
        raise AssertionError(operation)


class Reply:
    def __init__(self, events=None, result=None):
        self.events = events or []
        self.result = result or {}

    def iter_content(self, chunk_size):
        for event in self.events:
            yield ("data: " + json.dumps(event) + "\n\n").encode()

    def json(self):
        return self.result

    def close(self):
        return


@pytest.fixture
def controller(tmp_path, monkeypatch):
    guardian = Guardian()
    owner = Controller(tmp_path / "private", guardian)

    def worker(_live, method, path, **kwargs):
        return Reply(), Reply(events=[{"type": "final", "answer": "done"}])

    monkeypatch.setattr(owner, "_worker", worker)
    yield owner
    owner.close()
    caller_auth.reset()


def wait_done(controller, run_id):
    expires = time.monotonic() + 3
    while controller.store.run(run_id)["ended_at"] is None:
        assert time.monotonic() < expires
        time.sleep(0.01)
    return controller.store.run(run_id)


def app_client(controller):
    auth = caller_auth.CallerAuthConfig(
        token="test-token",
        allowed_hosts=frozenset({"localhost"}),
        allowed_origin_hosts=frozenset(),
    )
    return TestClient(
        build_app(controller, auth),
        base_url="http://localhost",
        headers={"Authorization": "Bearer test-token"},
    )


def test_http_idempotency_replay_and_durable_events(controller):
    client = app_client(controller)
    session = client.post(
        PREFIX + "/sessions", json={"workspace_id": controller.guardian.workspace}
    ).json()
    url = PREFIX + f"/sessions/{session['id']}/runs"
    response = client.post(
        url, json={"prompt": "hello"}, headers={"Idempotency-Key": "same"}
    )
    assert response.status_code == 202
    run_id = response.json()["id"]
    wait_done(controller, run_id)
    replay = client.post(
        url, json={"prompt": "hello"}, headers={"Idempotency-Key": "same"}
    )
    assert replay.status_code == 200 and replay.json()["id"] == run_id
    assert controller.guardian.starts == [run_id]
    stream = client.get(PREFIX + f"/runs/{run_id}/stream")
    assert '"type":"final"' in stream.text and "id: 2" in stream.text
    assert client.get(PREFIX + f"/runs/{run_id}/events?after=999").status_code == 409
    messages = client.get(PREFIX + f"/sessions/{session['id']}").json()["messages"]
    assert [item["role"] for item in messages] == ["user", "assistant"]


def test_http_auth_host_and_type_bounds(controller):
    client = app_client(controller)
    assert (
        client.get(
            PREFIX + "/capabilities", headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    assert client.get("/health", headers={"Host": "untrusted.example"}).status_code in {
        400,
        403,
    }
    session = client.post(
        PREFIX + "/sessions", json={"workspace_id": controller.guardian.workspace}
    ).json()
    response = client.post(
        PREFIX + f"/sessions/{session['id']}/runs",
        json={"prompt": "hi", "max_steps": True},
        headers={"Idempotency-Key": "key"},
    )
    assert response.status_code == 422


def test_success_is_committed_only_after_guardian_stop(controller, monkeypatch):
    stopped, release = threading.Event(), threading.Event()
    original = controller.guardian.call

    def delayed(operation, **payload):
        if operation == "cancel":
            stopped.set()
            assert release.wait(timeout=3)
        return original(operation, **payload)

    monkeypatch.setattr(controller.guardian, "call", delayed)
    session = controller.store.create_session(controller.guardian.workspace, "")
    run, _ = controller.submit(session["id"], "key", "hello")
    assert stopped.wait(timeout=2)
    try:
        assert controller.store.run(run["id"])["ended_at"] is None
        assert not any(
            item["event"]["type"] == "final"
            for item in controller.store.events(run["id"])["events"]
        )
    finally:
        release.set()
    assert wait_done(controller, run["id"])["state"] == "succeeded"


def test_recovery_interrupts_without_redispatch(tmp_path):
    guardian = Guardian()
    first = Controller(tmp_path / "private", guardian)
    session = first.store.create_session(guardian.workspace, "")
    run, _ = first.store.admit(session["id"], "key", "hello", 20, guardian.caps)
    first.store.dispatching(run["id"])
    first.maintenance_done.set()
    first.maintenance_thread.join(timeout=2)
    first.store.close()
    second = Controller(tmp_path / "private", guardian)
    try:
        recovered = second.store.run(run["id"])
        assert recovered["state"] == "interrupted" and recovered["outcome_uncertain"]
        assert recovered["generation"] != run["generation"]
        assert not guardian.starts
    finally:
        second.close()


def test_sensitive_interaction_is_not_persisted_or_redelivered(controller, monkeypatch):
    ready, answered = threading.Event(), threading.Event()
    secret = "sensitive-answer-" + uuid4().hex
    deliveries = []

    class InteractionReply(Reply):
        def iter_content(self, chunk_size):
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "needs_input",
                        "request_id": "worker-id",
                        "question": "secret?",
                        "sensitive": True,
                    }
                )
                + "\n\n"
            ).encode()
            ready.set()
            assert answered.wait(timeout=3)
            for fragment in (secret[:10], secret[10:]):
                yield (
                    "data: "
                    + json.dumps({"type": "chunk", "content": fragment})
                    + "\n\n"
                ).encode()
            yield (
                "data: " + json.dumps({"type": "final", "answer": secret}) + "\n\n"
            ).encode()

    def worker(live, method, path, **kwargs):
        if path.endswith("/respond"):
            row = controller.store.interaction(live.run["id"], interaction["id"])
            assert row["state"] == "accepted" and row["delivery"] == "unknown"
            deliveries.append(kwargs["json"])
            answered.set()
            return Reply(), Reply(result={"delivered": True})
        return Reply(), InteractionReply() if method == "POST" else Reply()

    monkeypatch.setattr(controller, "_worker", worker)
    session = controller.store.create_session(controller.guardian.workspace, "")
    run, _ = controller.submit(session["id"], "key", "hello")
    assert ready.wait(timeout=2)
    interaction = controller.pending_interaction(run["id"])
    submission = str(uuid4())
    controller.answer(
        run["id"],
        interaction["id"],
        interaction["generation"],
        "answered",
        secret,
        submission,
    )
    wait_done(controller, run["id"])
    receipt = controller.answer(
        run["id"],
        interaction["id"],
        interaction["generation"],
        "answered",
        "different",
        submission,
    )
    assert receipt["replayed"] and len(deliveries) == 1
    events = controller.store.events(run["id"])
    assert "[redacted]" in str(events)
    assert all(
        e["event"].get("content_omitted") == "sensitive_interaction"
        for e in events["events"]
        if e["event"]["type"] == "chunk"
    )
    for path in controller.store.root.glob("service.sqlite3*"):
        assert secret.encode() not in path.read_bytes()


def test_storage_failure_cancels_even_without_cancellation_record(
    controller, monkeypatch
):
    import sqlite3

    from gaia_agent.durable.controller import LiveRun

    session = controller.store.create_session(controller.guardian.workspace, "")
    run, _ = controller.store.admit(session["id"], "failure", "hello", 20, {})
    live = LiveRun(run, epoch=controller.epoch)
    controller.live[run["id"]] = live
    original = controller.store.stopping

    def full(_run):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(controller.store, "stopping", full)
    with pytest.raises(StoreError, match="storage_unavailable"):
        controller.cancel(run["id"])
    assert not controller.healthy and live.done.is_set()
    assert run["id"] in controller.guardian.cancels
    controller.live.pop(run["id"])
    monkeypatch.setattr(controller.store, "stopping", original)
    controller.cancel(run["id"])


def test_failed_receipt_stops_execution_without_delivery(controller, monkeypatch):
    import sqlite3

    session = controller.store.create_session(controller.guardian.workspace, "")
    run, _ = controller.store.admit(session["id"], "receipt", "hello", 20, {})
    controller.store.started(run["id"])
    interaction = controller.store.create_interaction(
        run["id"],
        "needs_confirmation",
        {"arguments": {"path": "/private"}},
        time.time() + 30,
    )
    fences = []
    original = controller.guardian.call

    def guardian(operation, **payload):
        if operation == "fence":
            fences.append(payload["controller_epoch"])
        return original(operation, **payload)

    def full(*_args):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(controller.guardian, "call", guardian)
    monkeypatch.setattr(controller.store, "receipt", full)
    with pytest.raises(StoreError, match="storage_unavailable"):
        controller.answer(run["id"], interaction["id"], run["generation"], "approve")
    assert not controller.healthy and fences and fences[0] != controller.epoch
    assert (
        controller.store.interaction(run["id"], interaction["id"])["state"] == "pending"
    )


def test_drain_fences_even_when_admission_lock_is_stalled(controller):
    acquired, release, fenced = threading.Event(), threading.Event(), threading.Event()
    original = controller.guardian.call

    def guardian(operation, **payload):
        if operation == "fence":
            fenced.set()
        return original(operation, **payload)

    controller.guardian.call = guardian

    def blocked():
        with controller.lock:
            acquired.set()
            assert release.wait(3)

    thread = threading.Thread(target=blocked)
    thread.start()
    assert acquired.wait(1)
    try:
        before = time.monotonic()
        controller.drain()
        assert time.monotonic() - before < 0.2
        assert fenced.wait(1)
    finally:
        release.set()
        thread.join(2)


def test_close_deadline_covers_unrelated_storage_lock(controller):
    controller.drain()
    controller.drain_thread.join(1)
    acquired, release = threading.Event(), threading.Event()

    def storage_operation():
        with controller.store.lock:
            acquired.set()
            assert release.wait(3)

    thread = threading.Thread(target=storage_operation)
    thread.start()
    assert acquired.wait(1)
    try:
        before = time.monotonic()
        with pytest.raises(StoreError, match="storage_close_incomplete"):
            controller.close(timeout=0.1)
        assert time.monotonic() - before < 0.5
    finally:
        release.set()
        thread.join(1)
        controller.store_close_thread.join(1)


@pytest.mark.parametrize(
    "body",
    [
        b'{"prompt":NaN}',
        b'{"prompt":"\\ud800"}',
        b'{"prompt":"canary-private-prompt","max_steps":Infinity}',
        b'{"prompt":"canary-private-prompt","max_steps":-Infinity}',
        b'{"prompt":{"secret":"canary-private-prompt"}}',
    ],
)
def test_adversarial_validation_is_bounded_and_never_reflects_content(controller, body):
    client = app_client(controller)
    session = client.post(
        PREFIX + "/sessions", json={"workspace_id": controller.guardian.workspace}
    ).json()
    response = client.post(
        PREFIX + f"/sessions/{session['id']}/runs",
        content=body,
        headers={"Idempotency-Key": "malformed", "Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "canary-private-prompt" not in response.text
    assert not controller.store.active()
    assert client.get(PREFIX + "/metrics").status_code == 200
