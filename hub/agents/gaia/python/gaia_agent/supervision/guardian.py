# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Independent execution owner with durable identity and irreversible leases."""

import fcntl
import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from gaia.logger import get_logger

from .runtime import PREFIX, DockerRuntime

LOGGER = get_logger(__name__)


class GuardianError(RuntimeError):
    """A stable non-sensitive protocol error."""


def uuid_value(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Expected canonical UUID")
    return value


def validate_policy(policy):
    uuid_value(policy["deployment"])
    if not re.fullmatch(r"(?:[^\s]+@)?sha256:[a-f0-9]{64}", policy["image"]):
        raise ValueError("Pin executor image by digest")
    for key, default, maximum in (
        ("slots", 1, 8),
        ("lease", 10, 60),
        ("lifetime", 300, 3600),
    ):
        policy.setdefault(key, default)
        if type(policy[key]) is not int or not 1 <= policy[key] <= maximum:
            raise ValueError(f"Invalid {key}")
    upstream = urlsplit(policy["inference_url"])
    if (
        not policy["model"]
        or upstream.scheme not in {"http", "https"}
        or not upstream.hostname
        or upstream.username
        or upstream.password
        or upstream.query
        or upstream.fragment
    ):
        raise ValueError(
            "Explicit model and credential-free HTTP inference endpoint required"
        )
    secret_file = policy.get("inference_secret_file")
    if secret_file and (
        not Path(secret_file).is_absolute()
        or "," in secret_file
        or not Path(secret_file).is_file()
    ):
        raise ValueError("Inference secret must be an existing absolute file")
    if type(policy.get("network_internal", False)) is not bool:
        raise ValueError("network_internal must be a boolean")
    if policy.get("network_internal") and sys.platform != "linux":
        raise ValueError(
            "Internal network profile requires a native Linux controller host"
        )
    if policy.get("network_internal") and not policy.get("network"):
        raise ValueError("Internal network profile requires an explicit network")
    if bool(policy.get("embedding_model")) != bool(policy.get("embedding_revision")):
        raise ValueError("Declare both embedding model and revision")
    for key in ("embedding_model", "embedding_revision"):
        value = policy.get(key)
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 512
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError("Embedding declarations must be bounded nonempty strings")
    volumes = []
    for workspace, mapping in policy["workspaces"].items():
        uuid_value(workspace)
        for key in ("data", "workspace"):
            name = mapping[key]
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", name):
                raise ValueError("Use named workspace volumes")
            volumes.append(name)
    if not volumes or len(volumes) != len(set(volumes)):
        raise ValueError("Every writable volume must have a distinct identity")
    return policy


class Guardian:
    """Own lifecycle mutations; HTTP or controller failure cannot suspend leases."""

    def __init__(self, root, policy, runtime=None):
        self.policy = validate_policy(dict(policy))
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_mode & 0o077:
            raise ValueError("Guardian directory must be private (0700)")
        self.lock_file = (self.root / "owner.lock").open("a+")
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.runtime = runtime or DockerRuntime(
            policy["endpoint"], policy.get("docker", "docker")
        )
        self.lock = threading.RLock()
        self.records = {}
        self.operations = {}
        self.tokens = {}
        self.recovering = set()
        self.healthy = False
        self.closing = threading.Event()
        self.journal = self.root / "journal.json"
        self.writer = None
        self.controller_epoch = None
        if self.journal.exists():
            stored = json.loads(self.journal.read_text())
            if (
                stored["deployment"] != self.policy["deployment"]
                or stored["schema"] != 1
            ):
                raise GuardianError("journal_identity_mismatch")
            self.records = stored["runs"]
            self.controller_epoch = stored.get("controller_epoch")
        self._reconcile()
        self.healthy = True
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def _write_snapshot(self, snapshot):
        temporary = self.root / "journal.next"
        with temporary.open("wb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(snapshot)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.journal)
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _persist(self):
        # At most one writer may exist. A stalled disk cannot indefinitely hold
        # the watchdog state lock or consume unbounded background threads.
        if self.writer is not None and self.writer.is_alive():
            self.healthy = False
            raise GuardianError("journal_unavailable")
        snapshot = json.dumps(
            {
                "schema": 1,
                "deployment": self.policy["deployment"],
                "runs": self.records,
                "controller_epoch": self.controller_epoch,
            }
        ).encode()
        failed = []

        def write():
            try:
                self._write_snapshot(snapshot)
            except Exception as exc:
                failed.append(type(exc).__name__)

        self.writer = threading.Thread(target=write, daemon=True)
        self.writer.start()
        self.writer.join(timeout=0.5)
        if self.writer.is_alive() or failed:
            self.healthy = False
            raise GuardianError("journal_unavailable")

    def _reconcile(self):
        inventory = self.runtime.inventory(self.policy["deployment"])
        matched = set()
        for run, record in self.records.items():
            self.operations[run] = threading.Lock()
            container_id = record.get("container_id")
            if (
                not container_id
                and record["state"] == "stopped"
                and record.get("removed")
            ):
                continue
            if not container_id:
                candidates = []
                for candidate in inventory:
                    # Read labels without mutating any unbound runtime resource.
                    value = json.loads(self.runtime.call("inspect", candidate))[0]
                    labels = value["Config"]["Labels"] or {}
                    if all(
                        labels.get(PREFIX + key) == expected
                        for key, expected in record["identity"].items()
                    ):
                        candidates.append(candidate)
                if len(candidates) != 1:
                    raise GuardianError("ambiguous_executor")
                container_id = record["container_id"] = candidates[0]
                self._persist()
            if container_id not in inventory:
                if record["state"] == "stopped":
                    record["removed"] = True
                    continue
                raise GuardianError("missing_executor")
            self.runtime.stop(container_id, record["identity"])
            matched.add(container_id)
            record["state"] = "stopped"
            record["reason"] = "guardian_restart"
            self._persist()
            self.runtime.remove(container_id, record["identity"])
            record["removed"] = True
        if set(inventory) != matched:
            raise GuardianError("unrecognized_executor")
        self._persist()

    def _get(self, run, generation):
        record = self.records.get(run)
        if record is None:
            raise GuardianError("run_not_found")
        if record["identity"]["generation"] != generation:
            raise GuardianError("stale_generation")
        return record

    def capabilities(self):
        return {
            "protocol": 2,
            "deployment": self.policy["deployment"],
            "workspaces": list(self.policy["workspaces"]),
            "slots": self.policy["slots"],
            "model": self.policy["model"],
            "embedding_model": self.policy.get("embedding_model"),
            "embedding_revision": self.policy.get("embedding_revision"),
            "network_internal": self.policy.get("network_internal", False),
            "lifetime": self.policy["lifetime"],
            "lease": self.policy["lease"],
            "image": self.policy["image"],
        }

    def fence(self, controller_epoch):
        uuid_value(controller_epoch)
        with self.lock:
            self.controller_epoch = controller_epoch
            self._persist()
            pending = [
                (run, record["identity"]["generation"])
                for run, record in self.records.items()
                if record["state"] != "stopped"
            ]
        failed = []
        for run, generation in pending:
            try:
                self.stop(run, generation, "controller_replaced")
            except Exception:
                failed.append(run)
        if failed:
            self.healthy = False
            raise GuardianError("fencing_incomplete")
        return {
            "controller_epoch": controller_epoch,
            "capabilities": self.capabilities(),
        }

    def start(self, run, generation, workspace, controller_epoch=None):
        uuid_value(run)
        uuid_value(generation)
        with self.lock:
            if controller_epoch != self.controller_epoch:
                raise GuardianError("stale_controller")
            if not self.healthy or self.closing.is_set():
                raise GuardianError("admission_closed")
            if len(self.records) >= 10000:
                raise GuardianError("guardian_history_full")
            if run in self.records:
                raise GuardianError("run_already_dispatched")
            if workspace not in self.policy["workspaces"]:
                raise GuardianError("workspace_not_found")
            active = [r for r in self.records.values() if r["state"] != "stopped"]
            if any(r["workspace"] == workspace for r in active):
                raise GuardianError("workspace_busy")
            if len(active) >= self.policy["slots"]:
                raise GuardianError("capacity_exhausted")
            identity = {
                "deployment": self.policy["deployment"],
                "run": run,
                "generation": generation,
                "nonce": str(uuid4()),
            }
            now = time.time()
            record = {
                "identity": identity,
                "workspace": workspace,
                "state": "starting",
                "container_id": None,
                "created": now,
                "expires": now + 60,
                "lease": now + 60,
                "last_clock": now,
                "mono_expires": time.monotonic() + 60,
                "mono_lease": time.monotonic() + 60,
            }
            self.records[run] = record
            operation = self.operations[run] = threading.Lock()
            self.tokens[run] = secrets.token_urlsafe(32)
            self._persist()
        try:
            with operation:
                container_id = self.runtime.create(
                    identity,
                    self.policy,
                    self.policy["workspaces"][workspace],
                    self.tokens[run],
                )
                with self.lock:
                    record["container_id"] = container_id
                    self._persist()
                    if record["state"] != "starting":
                        raise GuardianError("startup_expired")
                    # This persisted transition is the dispatch authorization point.
                    # A later stop wins termination, but cannot undo authorized start.
                    record["state"] = "dispatching"
                    self._persist()
                url = self.runtime.start(container_id, identity)
                with self.lock:
                    if record["state"] != "dispatching":
                        raise GuardianError("startup_expired")
                    now = time.time()
                    record.update(
                        state="active",
                        url=url,
                        expires=now + self.policy["lifetime"],
                        lease=now + self.policy["lease"],
                        last_clock=now,
                        mono_expires=time.monotonic() + self.policy["lifetime"],
                        mono_lease=time.monotonic() + self.policy["lease"],
                    )
                    self._persist()
            return {
                "run": run,
                "generation": generation,
                "url": url,
                "network_internal": self.policy.get("network_internal", False),
                "token": self.tokens[run],
                "deadline": record["expires"],
            }
        except Exception as exc:
            # The intent survives even if create acknowledgement was lost. Do not retry.
            with self.lock:
                if record["state"] != "stopped" and not (
                    isinstance(exc, GuardianError) and str(exc) == "startup_expired"
                ):
                    record["state"] = "unknown"
                    self.healthy = False
            if record.get("container_id"):
                self.stop(run, generation, "startup_failed")
            raise

    def renew(self, run, generation):
        with self.lock:
            record = self._get(run, generation)
            now = time.time()
            if (
                record["state"] != "active"
                or now >= min(record["lease"], record["expires"])
                or time.monotonic() >= min(record["mono_expires"], record["mono_lease"])
                or now < record["last_clock"] - 1
            ):
                raise GuardianError("lease_expired")
            record["lease"] = min(now + self.policy["lease"], record["expires"])
            record["mono_lease"] = min(
                time.monotonic() + self.policy["lease"], record["mono_expires"]
            )
            record["last_clock"] = now
            self._persist()
            return {"lease": record["lease"]}

    def status(self, run, generation):
        with self.lock:
            record = self._get(run, generation)
            return {
                key: record.get(key)
                for key in ("state", "reason", "expires", "workspace")
            }

    def cancel(self, run, generation):
        """Persist a pre-dispatch tombstone so delayed admission cannot revive it."""
        uuid_value(run)
        uuid_value(generation)
        with self.lock:
            if run not in self.records:
                if len(self.records) >= 10000:
                    raise GuardianError("guardian_history_full")
                self.records[run] = {
                    "identity": {
                        "deployment": self.policy["deployment"],
                        "run": run,
                        "generation": generation,
                        "nonce": str(uuid4()),
                    },
                    "state": "stopped",
                    "container_id": None,
                    "workspace": None,
                    "removed": True,
                    "reason": "cancel_before_start",
                }
                self.operations[run] = threading.Lock()
                self._persist()
                return {"state": "stopped"}
        return self.stop(run, generation)

    def stop(self, run, generation, reason="cancelled"):
        with self.lock:
            record = self._get(run, generation)
            if record["state"] == "stopped":
                return {"state": "stopped"}
            record["state"] = "stopping"
            record.setdefault("reason", reason)
        try:
            with self.operations[run]:
                with self.lock:
                    if record["state"] == "stopped":
                        return {"state": "stopped"}
                container_id = record.get("container_id")
                if not container_id:
                    raise GuardianError("executor_identity_unknown")
                self.runtime.stop(container_id, record["identity"])
                with self.lock:
                    record["state"] = "stopped"
                    self.tokens.pop(run, None)
                    self._persist()
                self.runtime.remove(container_id, record["identity"])
                with self.lock:
                    record["removed"] = True
                    self._persist()
            return {"state": "stopped"}
        except Exception:
            with self.lock:
                record["state"] = "unknown"
                self.healthy = False
            raise

    def _stop_background(self, run, generation, reason):
        try:
            self.stop(run, generation, reason)
        except Exception:
            # stop has already marked status unknown and closed admission.
            # A failed runtime is observable through status and health.
            with self.lock:
                self.healthy = False
            LOGGER.error(
                "Guardian termination failed; admission is closed and executor status is unknown"
            )

    def _recover_unknown(self, run):
        """Retry observation, never dispatch, after lost runtime acknowledgement."""
        try:
            with self.lock:
                record = self.records[run]
                identity = record["identity"]
            if not record.get("container_id"):
                candidates = []
                for candidate in self.runtime.inventory(self.policy["deployment"]):
                    value = json.loads(self.runtime.call("inspect", candidate))[0]
                    labels = value["Config"]["Labels"] or {}
                    if all(
                        labels.get(PREFIX + key) == expected
                        for key, expected in identity.items()
                    ):
                        candidates.append(candidate)
                if len(candidates) != 1:
                    raise GuardianError("ambiguous_executor")
                with self.lock:
                    record["container_id"] = candidates[0]
                    self._persist()
            self.stop(run, identity["generation"], "reconciled_unknown")
        except Exception:
            LOGGER.error(
                "Executor recovery remains unconfirmed; admission stays closed"
            )
            self.closing.wait(1)
        finally:
            with self.lock:
                self.recovering.discard(run)

    def _watch(self):
        while not self.closing.wait(0.1):
            with self.lock:
                now = time.time()
                expired = []
                for run, record in self.records.items():
                    if record["state"] in {"starting", "dispatching", "active"} and (
                        now >= min(record["expires"], record["lease"])
                        or time.monotonic()
                        >= min(record["mono_expires"], record["mono_lease"])
                        or now < record["last_clock"] - 1
                        or not self.healthy
                    ):
                        record["state"] = "stopping"
                        record.setdefault("reason", "lease_expired")
                        expired.append(
                            (run, record["identity"]["generation"], "lease_expired")
                        )
                for run, record in self.records.items():
                    if record["state"] == "unknown" and run not in self.recovering:
                        self.recovering.add(run)
                        threading.Thread(
                            target=self._recover_unknown, args=(run,), daemon=True
                        ).start()
                for args in expired:
                    threading.Thread(
                        target=self._stop_background, args=args, daemon=True
                    ).start()

    def close(self):
        self.closing.set()
        self.watchdog.join(timeout=2)
        with self.lock:
            pending = [
                (run, record["identity"]["generation"])
                for run, record in self.records.items()
                if record["state"] != "stopped"
            ]
        failed = []
        for run, generation in pending:
            try:
                self.stop(run, generation, "guardian_shutdown")
            except Exception:
                failed.append(run)
        if self.writer is not None and self.writer.is_alive():
            failed.append("journal_writer")
        if failed:
            self.healthy = False
            raise GuardianError("shutdown_incomplete")
        self.lock_file.close()
