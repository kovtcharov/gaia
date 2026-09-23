# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Authenticated, bounded local guardian protocol; no runtime socket forwarding."""

import argparse
import hmac
import http.client
import json
import os
import signal
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from gaia.logger import get_logger

from .guardian import Guardian, GuardianError

LOGGER = get_logger(__name__)
MAX_BODY = 16384


class ClientConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=25):
        super().__init__("localhost", timeout=timeout)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class Client:
    """Controller adapter uses only the guardian's authenticated socket."""

    def __init__(self, path, token, control_path=None):
        self.path, self.token = path, token
        self.control_path = control_path or Path(path).with_name(
            "guardian-control.sock"
        )

    def call(self, operation, **payload):
        connection = ClientConnection(
            self.path if operation == "start" else self.control_path
        )
        try:
            body = json.dumps(payload).encode()
            if len(body) > MAX_BODY:
                raise ValueError("Guardian request too large")
            try:
                connection.request(
                    "POST",
                    "/v1/" + operation,
                    body=body,
                    headers={
                        "Authorization": "Bearer " + self.token,
                        "Content-Type": "application/json",
                    },
                )
            except BrokenPipeError:
                # A header-level rejection can arrive before the small body is
                # written. Read that response; never retry a lifecycle request.
                LOGGER.debug(
                    "Guardian rejected request before body transmission completed"
                )
            response = connection.getresponse()
            raw = response.read(MAX_BODY + 1)
            if len(raw) > MAX_BODY:
                raise GuardianError("invalid_guardian_response")
            result = json.loads(raw)
            if response.status != 200:
                raise GuardianError(result.get("error", "guardian_unavailable"))
            return result
        finally:
            connection.close()


class Handler(BaseHTTPRequestHandler):
    """Fixed routes and schema; no arbitrary Docker command escape hatch."""

    def setup(self):
        self.request.settimeout(5)
        super().setup()

    def log_message(self, *_args):
        # Never log request lines or authorization material.
        return

    def reply(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), "Bearer " + self.server.token
        ):
            self.reply(401, {"error": "unauthorized"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or self.headers.get("Transfer-Encoding"):
            self.reply(400, {"error": "invalid_body_framing"})
            return
        try:
            length = int(lengths[0])
            if not 0 <= length <= MAX_BODY:
                self.reply(413, {"error": "body_too_large"})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("object required")
            guardian = self.server.guardian
            if self.path == "/v1/start" and not self.server.allow_start:
                self.reply(400, {"error": "admission_socket_required"})
                return
            if self.path == "/v1/health" and not payload:
                result = {"healthy": guardian.healthy, "protocol": 2}
            elif self.path == "/v1/capabilities" and not payload:
                result = guardian.capabilities()
            elif self.path == "/v1/fence" and set(payload) == {"controller_epoch"}:
                result = guardian.fence(**payload)
            elif self.path == "/v1/start" and set(payload) in (
                {"run", "generation", "workspace"},
                {"run", "generation", "workspace", "controller_epoch"},
            ):
                result = guardian.start(**payload)
            elif self.path in {
                "/v1/renew",
                "/v1/status",
                "/v1/stop",
                "/v1/cancel",
            } and set(payload) == {"run", "generation"}:
                result = getattr(guardian, self.path.rsplit("/", 1)[1])(**payload)
            else:
                self.reply(400, {"error": "invalid_command"})
                return
            self.reply(200, result)
        except GuardianError as exc:
            self.reply(409, {"error": str(exc)})
        except (ValueError, TypeError, KeyError):
            self.reply(400, {"error": "invalid_request"})
        except Exception:
            LOGGER.error(
                "Guardian command failed; inspect execution status before retrying"
            )
            self.reply(503, {"error": "guardian_unavailable"})


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path, guardian, token, allow_start=True):
        self.guardian, self.token = guardian, token
        self.allow_start = allow_start
        self.connections = threading.BoundedSemaphore(4 if allow_start else 8)
        super().__init__(str(path), Handler)
        os.chmod(path, 0o600)

    def process_request(self, request, client_address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent GAIA executor guardian")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args(argv)
    token = args.token_file.read_text().strip()
    if len(token) < 32 or len(token) > 4096 or any(c.isspace() for c in token):
        parser.error("Use a 32..4096 character guardian credential")
    guardian = Guardian(args.state, json.loads(args.config.read_text()))
    path = args.state / "guardian.sock"
    # The guardian process lock is already held; stale socket belongs to this state root.
    path.unlink(missing_ok=True)
    server = Server(path, guardian, token)
    control_path = args.state / "guardian-control.sock"
    control_path.unlink(missing_ok=True)
    control = Server(control_path, guardian, token, allow_start=False)
    control_thread = threading.Thread(target=control.serve_forever, daemon=True)
    control_thread.start()

    def shutdown(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        server.server_close()
        control.shutdown()
        control_thread.join(timeout=2)
        control.server_close()
        guardian.close()
        control_path.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
