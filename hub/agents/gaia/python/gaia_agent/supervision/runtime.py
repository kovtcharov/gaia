# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Explicit local Docker adapter for owned, independently terminable workers."""

import json
import re
import subprocess

PREFIX = "ai.gaia.execution."


class DockerOperationError(RuntimeError):
    """Safe diagnostics without Docker arguments, output, or mounted secrets."""

    def __init__(self, operation, reason):
        allowed = {
            "create",
            "start",
            "inspect",
            "ps",
            "kill",
            "rm",
            "network",
            "volume",
            "image",
            "run",
        }
        self.operation = operation if operation in allowed else "other"
        self.reason = reason
        super().__init__(
            f"Docker {self.operation}: {reason}; execution status is unknown"
        )


class DockerRuntime:
    """The guardian alone receives the host-administrative runtime endpoint."""

    def __init__(self, endpoint, executable="docker", timeout=5):
        if not endpoint.startswith("unix:///"):
            raise ValueError("An explicit absolute local Docker socket is required")
        self.endpoint, self.executable, self.timeout = endpoint, executable, timeout

    def call(self, *arguments):
        operation = arguments[0] if arguments else "other"
        try:
            result = subprocess.run(
                [self.executable, "--host", self.endpoint, *arguments],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise DockerOperationError(operation, "timeout") from None
        except OSError:
            raise DockerOperationError(operation, "unavailable") from None
        if result.returncode:
            raise DockerOperationError(operation, "nonzero_exit")
        return result.stdout.strip()

    def inventory(self, deployment):
        return self.call(
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            f"label={PREFIX}deployment={deployment}",
        ).splitlines()

    def inspect(self, container_id, identity):
        if not re.fullmatch(r"[a-f0-9]{64}", container_id):
            raise ValueError("Expected exact runtime identity")
        value = json.loads(self.call("inspect", container_id))[0]
        labels = value["Config"]["Labels"] or {}
        if value["Id"] != container_id or any(
            labels.get(PREFIX + key) != expected for key, expected in identity.items()
        ):
            raise RuntimeError("Executor identity mismatch")
        return value

    def create(self, identity, policy, workspace, token):
        labels = []
        for key, value in identity.items():
            labels += ["--label", f"{PREFIX}{key}={value}"]
        args = [
            "create",
            *labels,
            "--restart=no",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=512m",
            "--tmpfs",
            "/run/gaia:rw,nosuid,nodev,size=16m,uid=10001,gid=10001",
            "--log-driver=local",
            "--log-opt=max-size=1m",
            "--log-opt=max-file=2",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=256",
            "--memory=4g",
            "--cpus=2",
            "--publish",
            "127.0.0.1::8080",
            "--mount",
            f"type=volume,src={workspace['data']},dst=/data",
            "--mount",
            f"type=volume,src={workspace['workspace']},dst=/workspace",
            "--env",
            "GAIA_SERVICE_ALLOWED_HOSTS=localhost,127.0.0.1",
            "--env",
            "GAIA_GAIA_SIDECAR_TOKEN=" + token,
            "--env",
            "GAIA_SERVICE_MODEL=" + policy["model"],
            "--env",
            "LEMONADE_BASE_URL=" + policy["inference_url"],
            "--env",
            "GAIA_SERVICE_MAX_CONCURRENT_RUNS=1",
            "--env",
            "GAIA_SERVICE_RUN_TIMEOUT_SECONDS=" + str(policy["lifetime"]),
        ]
        if policy.get("inference_secret_file"):
            args += [
                "--mount",
                f"type=bind,src={policy['inference_secret_file']},dst=/run/inference-secret,readonly",
                "--env",
                "LEMONADE_API_KEY_FILE=/run/inference-secret",
            ]
        if policy.get("network"):
            args += ["--network", policy["network"]]
        return self.call(*args, policy["image"])

    def start(self, container_id, identity):
        self.inspect(container_id, identity)
        self.call("start", container_id)
        value = self.inspect(container_id, identity)
        if not value["State"]["Running"]:
            raise RuntimeError("Executor failed to start")
        ports = value["NetworkSettings"]["Ports"]["8080/tcp"]
        if len(ports) != 1 or ports[0]["HostIp"] != "127.0.0.1":
            raise RuntimeError("Unexpected executor network binding")
        return "http://127.0.0.1:" + ports[0]["HostPort"]

    def stop(self, container_id, identity):
        value = self.inspect(container_id, identity)
        if value["State"]["Running"]:
            self.call("kill", container_id)
        if self.inspect(container_id, identity)["State"]["Running"]:
            raise RuntimeError("Executor termination is unconfirmed")

    def remove(self, container_id, identity):
        if self.inspect(container_id, identity)["State"]["Running"]:
            raise RuntimeError("Refusing removal before confirmed termination")
        self.call("rm", container_id)
