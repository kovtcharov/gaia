# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""``POST /v1/gaia/query/{run_id}/respond`` — the answer-a-question wire contract.

The seam these tests pin broke once already: the sidecar required ``response``
while the Go TUI (``tui/internal/client/sse.go``) and the email sidecar both send
``value``, so every answer 422'd and the agent stayed parked on its question
until the run timed out — the exact hang ``/respond`` exists to prevent.

The body under test is therefore the LITERAL JSON the shipped client marshals,
not a dict built from the model's own field names: a test that asks the model
what it is called can never catch the two halves disagreeing.

Requests go through the real ``build_app()`` over ``TestClient``, so the strict
(``extra="forbid"``) request model and the route's 404/409 branches are exercised
as the binary serves them.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gaia_agent")

from fastapi.testclient import TestClient  # noqa: E402
from gaia_agent import caller_auth  # noqa: E402
from gaia_agent import server as srv  # noqa: E402

_BASE_URL = "http://127.0.0.1:8141"
_RUN_ID = "9f1c1a7e-4d33-4c4e-9b7a-0a7d4f2b6c81"
_REQUEST_ID = "req-7"


class _RecordingHandler:
    """Stands in for the run's output handler; records what reached the run."""

    def __init__(self, resolves: bool = True) -> None:
        self.resolves = resolves
        self.delivered: list[tuple[str, str]] = []

    def resolve_user_input(self, request_id: str, value: str) -> bool:
        self.delivered.append((request_id, value))
        return self.resolves


@pytest.fixture(autouse=True)
def _no_caller_auth(monkeypatch):
    """These tests are about the body, not the bearer — run with auth off."""
    monkeypatch.delenv(caller_auth.TOKEN_FILE_ENV_VAR, raising=False)
    monkeypatch.delenv(caller_auth.TOKEN_ENV_VAR, raising=False)
    caller_auth.reset()
    yield
    caller_auth.reset()


@pytest.fixture
def client() -> TestClient:
    return TestClient(srv.build_app(), base_url=_BASE_URL)


@pytest.fixture
def parked(request) -> _RecordingHandler:
    """Register a run in the process-local table, as a live /query would."""
    resolves = getattr(request, "param", True)
    handler = _RecordingHandler(resolves=resolves)
    srv._registry.add(srv._QueryRun(_RUN_ID, agent=None, handler=handler))
    yield handler
    srv._registry.remove(_RUN_ID)


def _respond(client: TestClient, body: dict) -> "object":
    return client.post(f"/v1/gaia/query/{_RUN_ID}/respond", json=body)


def test_the_body_the_go_tui_sends_is_accepted(client, parked):
    """The regression: this exact JSON is what ``respondRequest`` marshals."""
    r = _respond(client, {"request_id": _REQUEST_ID, "value": "Inbox"})

    assert r.status_code == 200, r.text
    assert parked.delivered == [(_REQUEST_ID, "Inbox")]
    # The success body the frozen contract pins, and what email returns.
    assert r.json() == {
        "run_id": _RUN_ID,
        "request_id": _REQUEST_ID,
        "accepted": True,
        "status": "ok",
    }


def test_the_deprecated_response_alias_still_delivers(client, parked):
    """A client written against the older gaia-only spelling must not break."""
    r = _respond(client, {"request_id": _REQUEST_ID, "response": "Inbox"})

    assert r.status_code == 200, r.text
    assert parked.delivered == [(_REQUEST_ID, "Inbox")]


def test_both_spellings_agreeing_is_accepted(client, parked):
    """A client hedging across both sidecars is unambiguous, so allow it."""
    r = _respond(
        client, {"request_id": _REQUEST_ID, "value": "Inbox", "response": "Inbox"}
    )

    assert r.status_code == 200, r.text
    assert parked.delivered == [(_REQUEST_ID, "Inbox")]


def test_both_spellings_disagreeing_is_refused(client, parked):
    """Resuming on a guess about which one the user meant would be worse."""
    r = _respond(
        client, {"request_id": _REQUEST_ID, "value": "Inbox", "response": "Archive"}
    )

    assert r.status_code == 422, r.text
    assert parked.delivered == []


def test_an_answer_with_no_text_is_refused(client, parked):
    r = _respond(client, {"request_id": _REQUEST_ID})

    assert r.status_code == 422, r.text
    assert parked.delivered == []


def test_an_empty_answer_is_refused(client, parked):
    """Empty text is not an answer; the run would resume on nothing."""
    r = _respond(client, {"request_id": _REQUEST_ID, "value": ""})

    assert r.status_code == 422, r.text
    assert parked.delivered == []


def test_an_unknown_field_is_refused_not_ignored(client, parked):
    """extra='forbid' is why the name matters: a near-miss is a hard 422."""
    r = _respond(client, {"request_id": _REQUEST_ID, "answer": "Inbox"})

    assert r.status_code == 422, r.text
    assert parked.delivered == []


def test_an_unknown_run_is_a_loud_404(client):
    r = client.post(
        "/v1/gaia/query/2f0e6a4b-1111-4222-8333-444455556666/respond",
        json={"request_id": _REQUEST_ID, "value": "Inbox"},
    )

    assert r.status_code == 404, r.text
    assert "not delivered" in r.json()["detail"]


@pytest.mark.parametrize("parked", [False], indirect=True)
def test_an_answer_for_a_question_no_longer_pending_is_a_409(client, parked):
    r = _respond(client, {"request_id": "stale-req", "value": "Inbox"})

    assert r.status_code == 409, r.text
    assert parked.delivered == [("stale-req", "Inbox")]


def test_value_is_the_canonical_name_across_sidecars():
    """The gaia and email sidecars must not re-diverge on the field name.

    Read from the email package's source rather than importing it: the two
    packages ship independently and the email wheel is not a test dependency
    here, but the name it publishes is still the contract this one matches.
    """
    from pathlib import Path

    email_routes = (
        Path(__file__).resolve().parents[3]
        / "email"
        / "python"
        / "gaia_agent_email"
        / "query_routes.py"
    )
    if not email_routes.is_file():  # pragma: no cover - repo layout changed
        pytest.skip(f"email sidecar source not found at {email_routes}")

    source = email_routes.read_text(encoding="utf-8")
    respond_model = source.split("class QueryRespondRequest", 1)[1].split(
        "class QueryRespondResponse", 1
    )[0]

    assert "    value: str" in respond_model, (
        "the email sidecar no longer takes the answer as 'value'; the two "
        "sidecars must agree, or one Go client cannot speak to both"
    )
    assert "value" in srv.QueryRespondRequest.model_fields
