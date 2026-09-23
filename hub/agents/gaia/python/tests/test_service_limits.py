# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""ASGI request bounds, including unknown-length and slow uploads."""

import asyncio
import json

import pytest
from gaia_agent.service_limits import RequestBodyLimitMiddleware


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")]])
async def test_actual_chunked_bytes_enforced_even_with_small_declared_length(headers):
    messages = iter(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    async def app(*_args):
        pytest.fail("Oversized body reached application")

    await RequestBodyLimitMiddleware(app, 5)(
        {"type": "http", "headers": headers}, receive, send
    )
    assert sent[0]["status"] == 413


@pytest.mark.asyncio
async def test_exact_limit_replays_body_and_preserves_disconnect():
    messages = iter(
        [
            {"type": "http.request", "body": b"ab", "more_body": True},
            {"type": "http.request", "body": b"c", "more_body": False},
            {"type": "http.disconnect"},
        ]
    )

    async def receive():
        return next(messages)

    async def app(_scope, receive, _send):
        assert await receive() == {
            "type": "http.request",
            "body": b"abc",
            "more_body": False,
        }
        assert await receive() == {"type": "http.disconnect"}

    await RequestBodyLimitMiddleware(app, 3)(
        {"type": "http", "headers": []}, receive, None
    )


@pytest.mark.asyncio
async def test_slow_upload_has_finite_deadline():
    sent = []

    async def receive():
        await asyncio.sleep(2)

    async def send(message):
        sent.append(message)

    async def app(*_args):
        pytest.fail("Incomplete body reached application")

    await RequestBodyLimitMiddleware(app, 3, read_timeout=0.01)(
        {"type": "http", "headers": []}, receive, send
    )
    assert sent[0]["status"] == 408
    assert json.loads(sent[1]["body"])["detail"] == "Request body timed out."


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [b"-1", b"invalid"])
async def test_invalid_length_rejected_before_read(value):
    sent = []

    async def unexpected(*_args):
        pytest.fail("Invalid length reached application or body reader")

    async def send(message):
        sent.append(message)

    await RequestBodyLimitMiddleware(unexpected, 3)(
        {"type": "http", "headers": [(b"content-length", value)]}, unexpected, send
    )
    assert sent[0]["status"] == 400


@pytest.mark.asyncio
async def test_bulk_upload_saturation_reserves_control_capacity():
    hold = asyncio.Event()
    entered = asyncio.Event()
    calls = 0

    async def app(_scope, _receive, send):
        await send({"type": "http.response.start", "status": 200})

    async def blocked_receive():
        nonlocal calls
        calls += 1
        if calls == 32:
            entered.set()
        await hold.wait()
        return {"type": "http.request", "body": b"{}"}

    async def discard(_message):
        return

    middleware = RequestBodyLimitMiddleware(app, 100)
    scope = {"type": "http", "headers": [], "path": "/v1/gaia/query"}
    tasks = [
        asyncio.create_task(middleware(scope, blocked_receive, discard))
        for _ in range(32)
    ]
    await asyncio.wait_for(entered.wait(), 2)
    received = []

    async def capture(message):
        received.append(message)

    async def empty():
        return {"type": "http.request", "body": b""}

    try:
        await middleware(scope, empty, capture)
        assert received[-1]["body"]
        assert received[0]["status"] == 503
        received.clear()
        await middleware(
            {
                **scope,
                "path": "/v1/gaia/query/00000000-0000-0000-0000-000000000001/cancel",
            },
            empty,
            capture,
        )
        assert received[0]["status"] == 200
    finally:
        hold.set()
        await asyncio.gather(*tasks)
    assert middleware.buffered == {"bulk": 0, "control": 0}


@pytest.mark.asyncio
async def test_aggregate_budget_held_until_application_finishes():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def app(_scope, receive, _send):
        await receive()
        entered.set()
        await release.wait()

    async def receive():
        return {"type": "http.request", "body": b"12345"}

    async def discard(_message):
        return

    middleware = RequestBodyLimitMiddleware(app, 10)
    middleware.aggregate_limit = 6
    scope = {"type": "http", "headers": []}
    first = asyncio.create_task(middleware(scope, receive, discard))
    await asyncio.wait_for(entered.wait(), 1)
    sent = []

    async def capture(message):
        sent.append(message)

    try:
        await middleware(scope, receive, capture)
        assert sent[0]["status"] == 503
        assert middleware.buffered["bulk"] == 5
    finally:
        release.set()
        await first
    assert middleware.buffered["bulk"] == 0
