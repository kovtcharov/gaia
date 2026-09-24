# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run real prepared embeddings through five separate frozen container processes."""

import argparse
import json
import shutil
from pathlib import Path
from uuid import uuid4

from gaia_agent.supervision.runtime import DockerRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image", default="gaia-service:test")
    parser.add_argument("--embedding-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    runtime = DockerRuntime(
        args.endpoint, shutil.which(args.docker) or args.docker, timeout=180
    )
    image = runtime.call("image", "inspect", args.image, "--format", "{{.Id}}")
    prefix = "gaia-rag-profile-" + uuid4().hex
    volumes = []
    reports = []
    try:
        for suffix in ("-data", "-workspace"):
            name = prefix + suffix
            runtime.call("volume", "create", name)
            volumes.append(name)
        for phase in ("index", "restart", "delete", "after-delete", "reindex"):
            output = runtime.call(
                "run",
                "--rm",
                "--label",
                "ai.gaia.qualification=" + prefix,
                "--read-only",
                "--memory=4g",
                "--pids-limit=256",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=512m",
                "--tmpfs",
                "/run/gaia:rw,nosuid,nodev,size=16m,uid=10001,gid=10001",
                "--mount",
                f"type=volume,src={volumes[0]},dst=/data",
                "--mount",
                f"type=volume,src={volumes[1]},dst=/workspace",
                "--env",
                "GAIA_MEMORY_DISABLED=1",
                "--env",
                "LEMONADE_BASE_URL=" + args.embedding_url,
                "--entrypoint=gaia-agent",
                image,
                "--validate-profile",
                "rag",
                "--state",
                "/workspace/profile",
                "--url",
                args.embedding_url,
                "--model",
                args.model,
                "--revision",
                args.revision,
                "--checkpoint",
                args.checkpoint,
                "--phase",
                phase,
            )
            report = json.loads(output.splitlines()[-1])
            assert (
                report["status"] == "passed"
                and report["frozen"]
                and report["phase"] == phase
            )
            reports.append(report)
            print(json.dumps(report), flush=True)
        args.report.write_text(
            json.dumps(
                {
                    "image": image,
                    "phases": reports,
                    "downloads": "none_expected_prepared_model_required",
                    "revision_verification": "operator_supplied",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        # A timed-out Docker CLI does not terminate its container. Select only
        # this harness's unique label and verify it before removing any executor.
        for container in runtime.call(
            "ps",
            "-aq",
            "--no-trunc",
            "--filter",
            "label=ai.gaia.qualification=" + prefix,
        ).splitlines():
            metadata = json.loads(runtime.call("inspect", container))[0]
            if (
                metadata["Id"] != container
                or metadata["Config"]["Labels"].get("ai.gaia.qualification") != prefix
            ):
                raise RuntimeError("Qualification executor ownership is unproven")
            runtime.call("rm", "-f", container)
        for volume in volumes:
            runtime.call("volume", "rm", volume)


if __name__ == "__main__":
    main()
