# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Small durable-service CLI: create, run, inspect, reconnect, cancel and answer."""

import argparse
import getpass
import json
import os
import sys
import warnings
from pathlib import Path
from urllib.parse import urlsplit

import requests


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class ServiceClient:
    def __init__(self, url, token):
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Use a credential-free HTTP(S) service URL")
        if parsed.scheme != "https" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("Use HTTPS outside loopback")
        if not token:
            raise ValueError("A caller token is required")
        self.url = url.rstrip("/") + "/v1/gaia/service"
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers["Authorization"] = "Bearer " + token

    def call(self, method, path, **kwargs):
        response = self.session.request(
            method,
            self.url + path,
            timeout=(5, 30),
            allow_redirects=False,
            stream=True,
            **kwargs,
        )
        try:
            if response.status_code not in {200, 201, 202}:
                raise RuntimeError(
                    f"Service request failed: HTTP {response.status_code}"
                )
            raw = bytearray()
            for chunk in response.iter_content(8192):
                if len(raw) + len(chunk) > 2 * 1024 * 1024:
                    raise RuntimeError("Service response exceeded client limit")
                raw.extend(chunk)
            return json.loads(raw)
        finally:
            response.close()

    def tail(self, run_id, after):
        # Bounded polling shares the same persisted cursors as SSE and survives reconnect.
        import time

        while True:
            page = self.call(
                "GET", f"/runs/{run_id}/events", params={"after": after, "limit": 100}
            )
            for event in page["events"]:
                after = event["seq"]
                print(encode(event), flush=True)
            if page["terminal"] and after >= page["latest_seq"]:
                return
            time.sleep(0.5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--token-file", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capabilities")
    for command in ("metrics", "diagnostics", "drain"):
        sub.add_parser(command)
    create = sub.add_parser("create-session")
    create.add_argument("workspace_id")
    create.add_argument("--title", default="")
    run = sub.add_parser("run")
    run.add_argument("session_id")
    run.add_argument("prompt")
    run.add_argument(
        "--idempotency-key",
        required=True,
        help="Reuse exactly this key after a lost response",
    )
    run.add_argument("--max-steps", type=int, default=20)
    run.add_argument("--acknowledge-history-loss", action="store_true")
    for command in ("status", "cancel", "interaction", "tail"):
        child = sub.add_parser(command)
        child.add_argument("run_id")
        if command == "tail":
            child.add_argument("--after", type=int, default=0)
    answer = sub.add_parser("answer")
    answer.add_argument("run_id")
    answer.add_argument("interaction_id")
    answer.add_argument("generation")
    answer.add_argument("decision", choices=("approve", "deny", "answered"))
    answer.add_argument(
        "--submission-id", help="Reuse this ID when retrying a question answer"
    )
    args = parser.parse_args(argv)
    token = (
        args.token_file.read_text().strip()
        if args.token_file
        else os.environ.get("GAIA_GAIA_SIDECAR_TOKEN", "")
    )
    client = ServiceClient(args.url, token)
    try:
        caps = client.call("GET", "/capabilities")
        if caps.get("api_version") != 1 or not caps.get("durable_runs"):
            raise RuntimeError("Unsupported durable service contract")
        if args.command == "capabilities":
            result = caps
        elif args.command in {"metrics", "diagnostics", "drain"}:
            result = client.call(
                "POST" if args.command == "drain" else "GET", "/" + args.command
            )
        elif args.command == "create-session":
            result = client.call(
                "POST",
                "/sessions",
                json={"workspace_id": args.workspace_id, "title": args.title},
            )
        elif args.command == "run":
            result = client.call(
                "POST",
                f"/sessions/{args.session_id}/runs",
                headers={"Idempotency-Key": args.idempotency_key},
                json={
                    "prompt": args.prompt,
                    "max_steps": args.max_steps,
                    "acknowledge_history_loss": args.acknowledge_history_loss,
                },
            )
        elif args.command == "tail":
            client.tail(args.run_id, args.after)
            return 0
        elif args.command == "answer":
            body = {"generation": args.generation, "decision": args.decision}
            if args.decision == "approve":
                pending = client.call("GET", f"/runs/{args.run_id}/interaction")
                if (
                    pending["id"] != args.interaction_id
                    or pending["generation"] != args.generation
                ):
                    raise RuntimeError("Interaction changed; fetch it again")
                print(encode(pending), flush=True)
                if (
                    input("Approve these exact arguments once? Type approve: ")
                    != "approve"
                ):
                    raise RuntimeError("Approval not confirmed")
            if args.decision == "answered":
                if not args.submission_id:
                    parser.error(
                        "Question answers require --submission-id so retries do not redeliver"
                    )
                body["submission_id"] = args.submission_id
                # Never accept sensitive answers in argv or persistent shell history.
                with warnings.catch_warnings():
                    warnings.simplefilter("error", getpass.GetPassWarning)
                    body["response"] = getpass.getpass("Answer: ")
            result = client.call(
                "POST",
                f"/runs/{args.run_id}/interactions/{args.interaction_id}/answer",
                json=body,
            )
        else:
            path = f"/runs/{args.run_id}" + (
                "" if args.command == "status" else "/" + args.command
            )
            result = client.call("POST" if args.command == "cancel" else "GET", path)
        print(encode(result))
        return 0
    except (
        requests.RequestException,
        ValueError,
        RuntimeError,
        getpass.GetPassWarning,
        EOFError,
    ):
        print(
            "Durable service request failed; inspect status before retrying with the same identity.",
            file=sys.stderr,
        )
        return 1
    finally:
        client.session.close()


if __name__ == "__main__":
    raise SystemExit(main())
