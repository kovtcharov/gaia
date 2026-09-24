# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Opt-in durable HTTP API, separate from the legacy connection-bound worker."""

import argparse
import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from gaia_agent import caller_auth
from gaia_agent.service_limits import RequestBodyLimitMiddleware
from gaia_agent.supervision.protocol import Client
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
)

from .controller import Controller
from .store import SCHEMA_VERSION, StoreError, encode

PREFIX = "/v1/gaia/service"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", check_fields=False)
    @classmethod
    def valid_utf8(cls, value):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("Expected valid UTF-8 text") from None
        return value


class SessionRequest(Strict):
    workspace_id: UUID
    title: str = Field(default="", max_length=200)


class RunRequest(Strict):
    prompt: str
    max_steps: StrictInt = Field(default=20, ge=1, le=20)
    acknowledge_history_loss: StrictBool = False


class ArtifactRequest(Strict):
    content: str


class AnswerRequest(Strict):
    generation: UUID
    decision: Literal["approve", "deny", "answered"]
    response: str | None = None
    submission_id: UUID | None = None


def public_run(run):
    return {
        key: run.get(key)
        for key in (
            "id",
            "session_id",
            "workspace_id",
            "state",
            "execution_status",
            "generation",
            "created_at",
            "started_at",
            "ended_at",
            "reason",
            "outcome_uncertain",
            "terminal_seq",
            "seq",
            "earliest_seq",
            "usage",
        )
    }


def build_app(controller, auth):
    if not auth.token or not auth.allowed_hosts:
        raise ValueError(
            "Durable service requires a bearer credential and exact allowed hosts"
        )
    caller_auth.configure(auth)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await asyncio.to_thread(controller.close)

    app = FastAPI(
        title="GAIA durable service", lifespan=lifespan, docs_url=None, redoc_url=None
    )

    @app.middleware("http")
    async def timing(request, call_next):
        started = time.monotonic()
        try:
            return await call_next(request)
        finally:
            controller.telemetry.duration("request", time.monotonic() - started)

    app.add_middleware(caller_auth.HostOriginMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=1048576)

    async def authorized(request: Request):
        if not caller_auth.token_ok(auth, request.headers.get("authorization", "")):
            raise HTTPException(401, "Bearer credential required")

    router = APIRouter(prefix=PREFIX, dependencies=[Depends(authorized)])

    @app.exception_handler(RequestValidationError)
    async def invalid_request(_request, _exc):
        # Framework validation errors reflect input and can themselves fail to
        # encode non-finite JSON numbers. Never echo prompts/answers here.
        return JSONResponse({"error": "invalid_request"}, status_code=422)

    @app.exception_handler(StoreError)
    async def store_error(_request, exc):
        headers = {"Retry-After": "1"} if exc.status == 503 else None
        return JSONResponse(
            {"error": exc.code}, status_code=exc.status, headers=headers
        )

    @app.exception_handler(sqlite3.Error)
    async def storage_error(_request, _exc):
        await asyncio.to_thread(controller.storage_failed)
        return JSONResponse({"error": "storage_unavailable"}, status_code=503)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready():
        available = (
            controller.healthy
            and not controller.draining
            and controller.guardian_ready
            and controller.probe.state["ready"]
        )
        return JSONResponse(
            {"ready": available, "inference": controller.probe.state},
            status_code=200 if available else 503,
        )

    @router.get("/capabilities")
    def capabilities():
        return {
            "api_version": 1,
            "schema_version": SCHEMA_VERSION,
            "durable_runs": True,
            "disconnect_cancels": False,
            "guardian": controller.capabilities,
            "limits": {"prompt_bytes": 65536, "event_bytes": 262144, "page_items": 100},
            "trust_boundary": "single_tenant",
            "storage": "local_sqlite_single_writer",
        }

    @router.post("/sessions", status_code=201)
    def create_session(body: SessionRequest):
        return controller.store.create_session(str(body.workspace_id), body.title)

    @router.get("/sessions/{session_id}")
    def session(session_id: UUID, after: int = 0, limit: int = 100):
        return controller.store.messages(str(session_id), after, limit)

    @router.post("/sessions/{session_id}/runs")
    def create_run(
        session_id: UUID, body: RunRequest, idempotency_key: str = Header(...)
    ):
        run, created = controller.submit(
            str(session_id),
            idempotency_key,
            body.prompt,
            body.max_steps,
            body.acknowledge_history_loss,
        )
        return JSONResponse(public_run(run), status_code=202 if created else 200)

    @router.get("/runs/{run_id}")
    def run(run_id: UUID):
        return public_run(controller.store.run(str(run_id)))

    @router.get("/runs/{run_id}/events")
    def events(run_id: UUID, after: int = 0, limit: int = 100):
        return controller.store.events(str(run_id), after, limit)

    @router.get("/runs/{run_id}/stream")
    async def stream(
        run_id: UUID, after: int = 0, last_event_id: str | None = Header(default=None)
    ):
        if last_event_id is not None:
            try:
                after = int(last_event_id)
            except ValueError:
                raise StoreError("invalid_cursor", 422) from None
        # Validate before sending headers; subsequent retention gaps are explicit SSE.
        await asyncio.to_thread(controller.store.events, str(run_id), after, 100)

        async def generate():
            cursor = after
            last_write = asyncio.get_running_loop().time()
            while True:
                try:
                    page = await asyncio.to_thread(
                        controller.store.events, str(run_id), cursor, 100
                    )
                except StoreError as exc:
                    yield "event: gap\ndata: " + encode({"error": exc.code}) + "\n\n"
                    return
                except sqlite3.Error:
                    await asyncio.to_thread(controller.storage_failed)
                    yield 'event: unavailable\ndata: {"error":"storage_unavailable"}\n\n'
                    return
                for item in page["events"]:
                    cursor = item["seq"]
                    yield f"id: {cursor}\ndata: {encode(item['event'])}\n\n"
                    last_write = asyncio.get_running_loop().time()
                if page["terminal"] and cursor >= page["latest_seq"]:
                    return
                if asyncio.get_running_loop().time() - last_write >= 5:
                    yield ": keepalive\n\n"
                    last_write = asyncio.get_running_loop().time()
                await asyncio.sleep(0.1)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @router.post("/runs/{run_id}/cancel")
    def cancel(run_id: UUID):
        return public_run(controller.cancel(str(run_id)))

    @router.get("/runs/{run_id}/interaction")
    def interaction(run_id: UUID):
        return JSONResponse(
            controller.pending_interaction(str(run_id)),
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/runs/{run_id}/interactions/{interaction_id}/answer")
    def answer(run_id: UUID, interaction_id: UUID, body: AnswerRequest):
        return JSONResponse(
            controller.answer(
                str(run_id),
                str(interaction_id),
                str(body.generation),
                body.decision,
                body.response,
                str(body.submission_id) if body.submission_id else None,
            ),
            headers={"Cache-Control": "no-store"},
        )

    @router.delete("/sessions/{session_id}", status_code=202)
    def delete_session(session_id: UUID):
        result = controller.store.delete_session(str(session_id))
        return {**result, "status_url": PREFIX + "/cleanup/" + result["id"]}

    @router.get("/cleanup/{job_id}")
    def cleanup(job_id: UUID):
        return controller.store.cleanup(str(job_id))

    @router.post("/drain")
    def drain():
        return controller.drain()

    @router.get("/metrics")
    def metrics():
        return {
            **controller.telemetry.snapshot(),
            "store": controller.store.statistics(),
        }

    @router.post("/sessions/{session_id}/artifacts", status_code=201)
    def artifact(session_id: UUID, body: ArtifactRequest):
        return controller.store.put_artifact(str(session_id), body.content)

    @router.get("/sessions/{session_id}/artifacts/{artifact_id}")
    def read_artifact(session_id: UUID, artifact_id: UUID):
        return JSONResponse(
            controller.store.read_artifact(str(session_id), str(artifact_id)),
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/cleanup/{job_id}/retry")
    def retry_cleanup(job_id: UUID):
        controller.store.cleanup(str(job_id))
        controller.store.cleanup_artifacts()
        return controller.store.cleanup(str(job_id))

    @router.get("/diagnostics")
    def diagnostics():
        return {
            "healthy": controller.healthy,
            "draining": controller.draining,
            "store": controller.store.statistics(),
            "model": controller.capabilities["model"],
            "schema_version": SCHEMA_VERSION,
            "telemetry": controller.telemetry.snapshot(),
            "retention": {
                "content_days": controller.content_days,
                "metadata_days": controller.metadata_days,
            },
            "execution_profile": "docker-single-tenant",
            "storage": "local-filesystem-only",
            "inference_readiness": controller.probe.state,
            "guardian_ready": controller.guardian_ready,
            "persistent_volume_quota": "not_enforced",
        }

    app.include_router(router)
    return app


def main(argv=None):
    import uvicorn

    parser = argparse.ArgumentParser(description="GAIA durable single-tenant service")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--guardian-socket", type=Path, required=True)
    parser.add_argument("--guardian-token-file", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--allowed-host", action="append", required=True)
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--rate-table", type=Path)
    parser.add_argument("--content-days", type=int, default=7)
    parser.add_argument("--metadata-days", type=int, default=30)
    parser.add_argument("--inference-probe-url")
    parser.add_argument("--inference-probe-token-file", type=Path)
    args = parser.parse_args(argv)
    if any(
        not host
        or ":" in host
        or "/" in host
        or "*" in host
        or any(c.isspace() for c in host)
        for host in args.allowed_host
    ):
        parser.error("Allowed hosts must be exact hostnames without ports")
    auth = replace(
        caller_auth.config_from_environment(),
        allowed_hosts=frozenset(args.allowed_host),
        allowed_origin_hosts=frozenset(),
    )
    if not auth.token:
        parser.error("Configure the GAIA caller token file or environment variable")
    guardian_token = args.guardian_token_file.read_text().strip()
    controller = Controller(
        args.state,
        Client(args.guardian_socket, guardian_token),
        (auth.token, guardian_token),
        trace_directory=args.trace_directory,
        rates=json.loads(args.rate_table.read_text()) if args.rate_table else None,
        content_days=args.content_days,
        metadata_days=args.metadata_days,
        inference_probe_url=args.inference_probe_url,
        inference_probe_token_file=args.inference_probe_token_file,
    )
    app = build_app(controller, auth)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        timeout_graceful_shutdown=30,
        access_log=False,
    )


if __name__ == "__main__":
    main()
