# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Guardian safety and authenticated local protocol tests."""

import json
import threading
import time
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest
from gaia_agent.supervision.guardian import Guardian, GuardianError, validate_policy
from gaia_agent.supervision.protocol import Client, Server
from gaia_agent.supervision.runtime import PREFIX


class Runtime:
    def __init__(self):
        self.containers = {}
        self.fail_stop = False

    def inventory(self, deployment):
        return list(self.containers)

    def create(self, identity, policy, workspace, token):
        container_id = uuid4().hex * 2
        self.containers[container_id] = {"identity": identity, "running": False}
        return container_id

    def start(self, container_id, identity):
        assert self.containers[container_id]["identity"] == identity
        self.containers[container_id]["running"] = True
        return "http://127.0.0.1:8081"

    def stop(self, container_id, identity):
        if self.fail_stop:
            raise TimeoutError("runtime unavailable")
        assert self.containers[container_id]["identity"] == identity
        self.containers[container_id]["running"] = False

    def remove(self, container_id, identity):
        assert not self.containers[container_id]["running"]
        del self.containers[container_id]

    def call(self, operation, container_id):
        assert operation == "inspect"
        record = self.containers[container_id]
        return json.dumps(
            [
                {
                    "Config": {
                        "Labels": {
                            PREFIX + key: value
                            for key, value in record["identity"].items()
                        }
                    }
                }
            ]
        )


@pytest.fixture
def policy():
    return {
        "deployment": str(uuid4()),
        "image": "sha256:" + "a" * 64,
        "endpoint": "unix:///explicit/docker.sock",
        "model": "fixture",
        "inference_url": "http://inference:8099/api/v1",
        "lease": 1,
        "lifetime": 5,
        "workspaces": {str(uuid4()): {"data": "state", "workspace": "files"}},
    }


@pytest.fixture
def guardian(tmp_path, policy):
    root = tmp_path / "private"
    runtime = Runtime()
    owner = Guardian(root, policy, runtime)
    yield owner
    runtime.fail_stop = False
    if not owner.lock_file.closed:
        owner.close()


def launch(guardian):
    run, generation = str(uuid4()), str(uuid4())
    guardian.start(run, generation, next(iter(guardian.policy["workspaces"])))
    return run, generation


def test_exclusive_owner(guardian):
    with pytest.raises(BlockingIOError):
        Guardian(guardian.root, guardian.policy, guardian.runtime)


def test_lease_stops_without_controller(guardian):
    run, generation = launch(guardian)
    expires = time.monotonic() + 3
    while guardian.status(run, generation)["state"] != "stopped":
        assert time.monotonic() < expires
        time.sleep(0.02)
    assert not any(c["running"] for c in guardian.runtime.containers.values())
    with pytest.raises(GuardianError, match="lease_expired"):
        guardian.renew(run, generation)


def test_stale_generation_and_workspace_serialization(guardian):
    run, generation = launch(guardian)
    with pytest.raises(GuardianError, match="stale_generation"):
        guardian.stop(run, str(uuid4()))
    with pytest.raises(GuardianError, match="workspace_busy"):
        launch(guardian)
    guardian.stop(run, generation)
    launch(guardian)


def test_unknown_stop_retains_admission(guardian):
    run, generation = launch(guardian)
    guardian.runtime.fail_stop = True
    with pytest.raises(TimeoutError):
        guardian.stop(run, generation)
    assert guardian.status(run, generation)["state"] == "unknown"
    assert not guardian.healthy
    with pytest.raises(GuardianError, match="admission_closed"):
        launch(guardian)


def test_restart_stops_owned_executor(guardian):
    run, generation = launch(guardian)
    guardian.closing.set()
    guardian.watchdog.join(timeout=2)
    guardian.lock_file.close()
    recovered = Guardian(guardian.root, guardian.policy, guardian.runtime)
    try:
        assert recovered.status(run, generation)["state"] == "stopped"
        assert not any(c["running"] for c in guardian.runtime.containers.values())
    finally:
        recovered.close()


def test_absolute_deadline_cannot_be_extended(guardian):
    run, generation = launch(guardian)
    guardian.records[run]["expires"] = time.time() + 0.25
    expires = time.monotonic() + 2
    while guardian.status(run, generation)["state"] != "stopped":
        try:
            guardian.renew(run, generation)
        except GuardianError as exc:
            assert str(exc) == "lease_expired"
        assert time.monotonic() < expires
        time.sleep(0.02)


def test_ambiguous_create_ack_recovery(guardian):
    run, generation = launch(guardian)
    guardian.records[run]["container_id"] = None
    guardian._persist()
    guardian.closing.set()
    guardian.watchdog.join(timeout=2)
    guardian.lock_file.close()
    recovered = Guardian(guardian.root, guardian.policy, guardian.runtime)
    try:
        assert recovered.status(run, generation)["state"] == "stopped"
        assert recovered.records[run]["container_id"]
    finally:
        recovered.close()


def test_authenticated_protocol(guardian):
    temporary = TemporaryDirectory(prefix="gg-", dir="/tmp")
    path = __import__("pathlib").Path(temporary.name) / "test.sock"
    server = Server(path, guardian, "a" * 32)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(GuardianError, match="unauthorized"):
            Client(path, "wrong", control_path=path).call("health")
        client = Client(path, "a" * 32, control_path=path)
        assert client.call("health") == {"healthy": True, "protocol": 2}
        with pytest.raises(GuardianError, match="invalid_command"):
            client.call("shell", command="unexpected")
        run, generation = str(uuid4()), str(uuid4())
        reply = client.call(
            "start",
            run=run,
            generation=generation,
            workspace=next(iter(guardian.policy["workspaces"])),
        )
        assert reply["token"] and reply["url"]
        assert client.call("stop", run=run, generation=generation)["state"] == "stopped"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
        temporary.cleanup()


def test_shared_writable_volumes_rejected(policy):
    policy["workspaces"][str(uuid4())] = {"data": "state", "workspace": "other"}
    with pytest.raises(ValueError, match="distinct"):
        validate_policy(policy)


def test_output_limits_count_serialized_bytes():
    from gaia_agent.supervision.output import BoundedOutput, OutputLimitError

    cancelled = threading.Event()
    output = BoundedOutput(cancelled.set, max_bytes=30, event_bytes=20)
    output.put({"text": "é"})
    output.get_nowait()
    output.put({"text": "é"})
    with pytest.raises(OutputLimitError):
        output.put({"text": "é"})
    assert cancelled.is_set() and output.exceeded.is_set()


def test_oversized_event_cancels_before_enqueue():
    from gaia_agent.supervision.output import BoundedOutput, OutputLimitError

    cancelled = threading.Event()
    output = BoundedOutput(cancelled.set, event_bytes=20)
    with pytest.raises(OutputLimitError):
        output.put({"text": "x" * 30})
    assert output.empty() and cancelled.is_set()


def test_create_ack_loss_is_reconciled_without_restart(guardian):
    original = guardian.runtime.create

    def lost(*args):
        original(*args)
        raise TimeoutError("create acknowledgement lost")

    guardian.runtime.create = lost
    with pytest.raises(TimeoutError):
        launch(guardian)
    expires = time.monotonic() + 3
    while guardian.runtime.containers:
        assert time.monotonic() < expires
        time.sleep(0.02)
    assert not guardian.healthy
    assert list(guardian.records.values())[0]["state"] == "stopped"


def test_cancel_before_dispatch_never_starts(guardian):
    created = threading.Event()
    release = threading.Event()
    original = guardian.runtime.create
    failures = []
    starts = []

    def delayed(*args):
        result = original(*args)
        created.set()
        assert release.wait(timeout=3)
        return result

    guardian.runtime.create = delayed
    guardian.runtime.start = lambda *_args: starts.append(True)
    run, generation = str(uuid4()), str(uuid4())

    def launch_thread():
        try:
            guardian.start(run, generation, next(iter(guardian.policy["workspaces"])))
        except GuardianError as exc:
            failures.append(str(exc))

    worker = threading.Thread(target=launch_thread)
    worker.start()
    assert created.wait(timeout=2)
    stopper = threading.Thread(target=guardian.stop, args=(run, generation))
    stopper.start()
    expires = time.monotonic() + 2
    while guardian.records[run]["state"] != "stopping":
        assert time.monotonic() < expires
        time.sleep(0.01)
    release.set()
    worker.join(timeout=3)
    stopper.join(timeout=3)
    assert not starts and failures == ["startup_expired"]
    assert guardian.records[run]["state"] == "stopped"


def test_control_socket_rejects_admission(guardian):
    with TemporaryDirectory(prefix="gg-", dir="/tmp") as temporary:
        path = __import__("pathlib").Path(temporary) / "control.sock"
        server = Server(path, guardian, "a" * 32, allow_start=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with pytest.raises(GuardianError, match="admission_socket_required"):
                Client(path, "a" * 32, control_path=path).call(
                    "start",
                    run=str(uuid4()),
                    generation=str(uuid4()),
                    workspace=next(iter(guardian.policy["workspaces"])),
                )
            assert Client(path, "a" * 32, control_path=path).call("health")["healthy"]
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


def test_stalled_journal_cannot_prevent_termination(guardian):
    run, generation = launch(guardian)
    blocked, release = threading.Event(), threading.Event()
    original = guardian._write_snapshot

    def stalled(snapshot):
        blocked.set()
        assert release.wait(timeout=5)
        original(snapshot)

    guardian._write_snapshot = stalled
    try:
        with pytest.raises(GuardianError, match="journal_unavailable"):
            guardian.renew(run, generation)
        assert blocked.is_set()
        expires = time.monotonic() + 2
        while any(item["running"] for item in guardian.runtime.containers.values()):
            assert time.monotonic() < expires
            time.sleep(0.02)
        assert not guardian.healthy
    finally:
        release.set()
        guardian.writer.join(timeout=2)
        guardian._write_snapshot = original


def test_shutdown_attempts_all_executors(guardian):
    guardian.policy["slots"] = 2
    second_workspace = str(uuid4())
    guardian.policy["workspaces"][second_workspace] = {
        "data": "second-data",
        "workspace": "second-work",
    }
    first_run, _ = launch(guardian)
    second_run, generation = str(uuid4()), str(uuid4())
    guardian.start(second_run, generation, second_workspace)
    original = guardian.runtime.stop

    def fail_first(container_id, identity):
        if identity["run"] == first_run:
            raise TimeoutError("first runtime failure")
        return original(container_id, identity)

    guardian.runtime.stop = fail_first
    try:
        with pytest.raises(GuardianError, match="shutdown_incomplete"):
            guardian.close()
        assert guardian.records[second_run]["state"] == "stopped"
        assert not guardian.healthy
    finally:
        guardian.runtime.stop = original


@pytest.mark.parametrize("jump", [-3600, 3600])
def test_clock_discontinuity_stops_before_new_admission(guardian, monkeypatch, jump):
    run, generation = launch(guardian)
    original = time.time
    monkeypatch.setattr(
        "gaia_agent.supervision.guardian.time.time", lambda: original() + jump
    )
    expires = time.monotonic() + 2
    while guardian.status(run, generation)["state"] != "stopped":
        assert time.monotonic() < expires
        time.sleep(0.02)
    assert guardian.runtime.containers == {}


def test_controller_fencing_rejects_delayed_old_dispatch(guardian):
    old_epoch, new_epoch = str(uuid4()), str(uuid4())
    guardian.fence(old_epoch)
    guardian.fence(new_epoch)
    with pytest.raises(GuardianError, match="stale_controller"):
        guardian.start(
            str(uuid4()),
            str(uuid4()),
            next(iter(guardian.policy["workspaces"])),
            old_epoch,
        )
    assert not guardian.runtime.containers
    assert json.loads(guardian.journal.read_text())["controller_epoch"] == new_epoch


def test_pre_dispatch_cancellation_fences_delayed_start(guardian):
    run, generation = str(uuid4()), str(uuid4())
    assert guardian.cancel(run, generation)["state"] == "stopped"
    with pytest.raises(GuardianError, match="already_dispatched"):
        guardian.start(run, generation, next(iter(guardian.policy["workspaces"])))
    assert not guardian.runtime.containers


@pytest.mark.parametrize("failure", ["timeout", "unavailable", "nonzero_exit"])
def test_runtime_failure_diagnostics_never_expose_arguments(monkeypatch, failure):
    import subprocess
    from types import SimpleNamespace

    from gaia_agent.supervision.runtime import DockerOperationError, DockerRuntime

    secret = "private-bearer-value"
    calls = []

    def invoke(arguments, **_kwargs):
        calls.append(arguments)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(arguments, 5, output=secret, stderr=secret)
        if failure == "unavailable":
            raise OSError(secret)
        return SimpleNamespace(returncode=1, stdout=secret, stderr=secret)

    monkeypatch.setattr(subprocess, "run", invoke)
    with pytest.raises(DockerOperationError) as caught:
        DockerRuntime("unix:///tmp/docker.sock").call("create", "--env", secret)
    assert caught.value.operation == "create"
    assert caught.value.reason == failure
    assert secret not in str(caught.value)
    assert caught.value.__suppress_context__ or failure == "nonzero_exit"
    assert len(calls) == 1  # Ambiguous runtime operations must never be retried.
