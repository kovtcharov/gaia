# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Finite inbound HTTP bodies for the opt-in container service."""

import asyncio
import re
import threading

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Count actual bytes, including chunked requests, before JSON decoding.

    A bounded read also prevents an incomplete upload from occupying a connection
    indefinitely. Query execution and SSE streaming have their own deadlines.
    """

    def __init__(self, app: ASGIApp, max_bytes: int, read_timeout: float = 30) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.read_timeout = read_timeout
        self.readers = {
            "bulk": threading.BoundedSemaphore(32),
            "control": threading.BoundedSemaphore(8),
        }
        self.accounting = threading.Lock()
        self.buffered = {"bulk": 0, "control": 0}
        self.aggregate_limit = 16 * 1024 * 1024

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        control = path in {"/health", "/ready"} or re.fullmatch(
            r"/v1/gaia/query/[0-9a-f-]{36}/(?:cancel|respond|confirm)", path
        )
        control = (
            control
            or path
            in {
                "/v1/gaia/service/diagnostics",
                "/v1/gaia/service/capabilities",
                "/v1/gaia/service/metrics",
                "/v1/gaia/service/drain",
            }
            or re.fullmatch(
                r"/v1/gaia/service/runs/[0-9a-f-]{36}(?:/(?:cancel|interaction|interactions/[0-9a-f-]{36}/answer))?",
                path,
            )
        )
        lane = "control" if control else "bulk"
        if not self.readers[lane].acquire(blocking=False):
            await JSONResponse(
                {"detail": "Too many active uploads."},
                status_code=503,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return
        reservation = {"bytes": 0}
        try:
            await self._handle(scope, receive, send, lane, reservation)
        finally:
            with self.accounting:
                self.buffered[lane] -= reservation["bytes"]
            self.readers[lane].release()

    async def _handle(self, scope, receive, send, lane, reservation):
        lengths = [v for k, v in scope["headers"] if k.lower() == b"content-length"]
        if lengths:
            try:
                length = int(lengths[0])
                if len(lengths) != 1 or length < 0:
                    raise ValueError
            except ValueError:
                await JSONResponse(
                    {"detail": "Invalid Content-Length."}, status_code=400
                )(scope, receive, send)
                return
            if length > self.max_bytes:
                await self._too_large(scope, receive, send)
                return

        async def read_body():
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return None
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > self.max_bytes:
                    raise OverflowError
                with self.accounting:
                    limit = self.aggregate_limit if lane == "bulk" else 256 * 1024
                    if self.buffered[lane] + len(chunk) > limit:
                        raise BufferError
                    self.buffered[lane] += len(chunk)
                    reservation["bytes"] += len(chunk)
                body.extend(chunk)
                if not message.get("more_body", False):
                    return bytes(body)

        try:
            body = await asyncio.wait_for(read_body(), timeout=self.read_timeout)
        except OverflowError:
            await self._too_large(scope, receive, send)
            return
        except BufferError:
            await JSONResponse(
                {"detail": "Aggregate upload budget occupied."},
                status_code=503,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return
        except asyncio.TimeoutError:
            await JSONResponse({"detail": "Request body timed out."}, status_code=408)(
                scope, receive, send
            )
            return
        if body is None:
            return
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            # StreamingResponse must still observe a real client disconnect.
            return await receive()

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, receive: Receive, send: Send) -> None:
        await JSONResponse(
            {"detail": f"Request body exceeds {self.max_bytes} bytes."}, status_code=413
        )(scope, receive, send)
