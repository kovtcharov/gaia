# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Real-socket slow uploads, malformed inputs and isolated control capacity."""

import json
import socket
import time
from urllib.parse import urlsplit
from uuid import uuid4


def exercise(client, url, token, workspace):
    prefix = url + "/v1/gaia/service"
    session = client.call("POST", "/sessions", json={"workspace_id": workspace})
    run_url = prefix + f"/sessions/{session['id']}/runs"
    for payload in (
        {"prompt": "x", "max_steps": True},
        {"prompt": "x", "max_steps": 21},
        {"prompt": {"token": "canary-secret"}},
        {"prompt": "x", "bypass_permissions": True},
    ):
        response = client.session.post(
            run_url, json=payload, headers={"Idempotency-Key": str(uuid4())}, timeout=3
        )
        assert response.status_code == 422, response.status_code
    for body in (
        b'{"prompt":NaN}',
        b'{"prompt":"\\ud800"}',
        b'{"prompt":"canary-private-prompt","max_steps":Infinity}',
        b'{"prompt":"canary-private-prompt","max_steps":-Infinity}',
        b'{"prompt":{"secret":"canary-private-prompt"}}',
    ):
        response = client.session.post(
            run_url,
            data=body,
            headers={
                "Idempotency-Key": str(uuid4()),
                "Content-Type": "application/json",
            },
            timeout=3,
        )
        assert response.status_code == 422, response.status_code
        assert "canary-private-prompt" not in response.text
    assert (
        client.session.get(
            prefix + "/metrics",
            headers={"Authorization": "Bearer canary-invalid"},
            timeout=3,
        ).status_code
        == 401
    )
    assert (
        client.session.get(
            prefix + "/metrics",
            headers={"Origin": "https://untrusted.example"},
            timeout=3,
        ).status_code
        == 403
    )
    response = client.session.post(
        run_url,
        data=b'{"prompt":"' + b"x" * 1048576 + b'"}',
        headers={"Idempotency-Key": str(uuid4()), "Content-Type": "application/json"},
        timeout=3,
    )
    assert response.status_code == 413
    parsed = urlsplit(url)
    sockets = []
    try:
        for _ in range(32):
            connection = socket.create_connection(
                (parsed.hostname, parsed.port), timeout=3
            )
            connection.sendall(
                (
                    "POST /v1/gaia/service/sessions HTTP/1.1\r\nHost: "
                    + parsed.netloc
                    + "\r\nAuthorization: Bearer "
                    + token
                    + "\r\nContent-Type: application/json\r\nContent-Length: 1048576\r\n\r\n{"
                ).encode()
            )
            sockets.append(connection)
        time.sleep(0.2)
        before = time.monotonic()
        assert client.call("GET", "/metrics")["store"]["active_runs"] == 0
        elapsed = time.monotonic() - before
        assert elapsed < 2, elapsed
        overload = client.session.post(
            prefix + "/sessions", json={"workspace_id": workspace}, timeout=3
        )
        assert overload.status_code == 503 and overload.headers["Retry-After"] == "1"
    finally:
        for connection in sockets:
            connection.close()
    time.sleep(0.2)
    assert client.call("GET", "/metrics")["store"]["active_runs"] == 0
    print(
        json.dumps(
            {
                "malformed_inputs": "passed",
                "auth_origin": "passed",
                "oversized_upload": "passed",
                "slow_uploads": 32,
                "reserved_control_seconds": elapsed,
            }
        ),
        flush=True,
    )
