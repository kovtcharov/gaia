"""The cold-start staging on the stdio transport.

The first message of a session against a cold Lemonade pays the model load
(~minutes) and the system-prompt prefill. These tests pin the contract:
residency is reported on the startup ping, the launch-time warm-up is JOINED
by the first turn (never raced), stages ride the canonical ``status`` type as
an additive ``stage`` field, and a failed load is a loud terminal error —
never a stuck spinner.
"""

import io
import json
import threading

from gaia_agent import stdio


def _events(out):
    return [json.loads(line) for line in out.getvalue().split("\n") if line.strip()]


class _Status:
    def __init__(self, running=True, loaded=()):
        self.running = running
        self.loaded_models = [{"id": m} for m in loaded]


class _ResidencyClient:
    """LemonadeClient stand-in for the residency probe."""

    def __init__(self, status):
        self._status = status

    def get_status(self):
        return self._status

    @staticmethod
    def _find_loaded_entry(status, model):
        for entry in status.loaded_models:
            if entry.get("id") == model:
                return entry
        return None


# ---------------------------------------------------------------------------
# Residency probe
# ---------------------------------------------------------------------------


def test_model_resident_true_when_loaded():
    client = _ResidencyClient(_Status(loaded=["Gemma-4-E4B-it-GGUF"]))
    assert stdio._model_resident(client, "Gemma-4-E4B-it-GGUF") is True


def test_model_resident_false_when_absent():
    client = _ResidencyClient(_Status(loaded=["other-model"]))
    assert stdio._model_resident(client, "Gemma-4-E4B-it-GGUF") is False


def test_model_resident_unknown_when_probe_fails():
    """Indeterminate is None — never guessed either way."""

    class _Broken:
        def get_status(self):
            raise ConnectionError("down")

    assert stdio._model_resident(_Broken(), "any") is None
    assert (
        stdio._model_resident(_ResidencyClient(_Status(running=False)), "any") is None
    )


# ---------------------------------------------------------------------------
# _stage_cold_start: the first turn joins the warm-up and stages the wait
# ---------------------------------------------------------------------------


def _warmup(was_cold=True, error=None, thread=None):
    w = stdio.ModelWarmup(None, "Gemma-4-E4B-it-GGUF")
    w.was_cold = was_cold
    w.error = error
    w._thread = thread
    return w


def test_warm_start_emits_no_stages():
    out = io.StringIO()
    assert stdio._stage_cold_start(_warmup(was_cold=False), out) is True
    assert _events(out) == []


def test_no_warmup_emits_no_stages():
    out = io.StringIO()
    assert stdio._stage_cold_start(None, out) is True
    assert _events(out) == []


def test_cold_start_with_load_in_flight_stages_load_then_prefill():
    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    warmup = _warmup(thread=thread)
    out = io.StringIO()

    # The "load" completes shortly; _stage_cold_start must join it, so the
    # release comes from a timer rather than from before the call.
    threading.Timer(0.05, release.set).start()
    assert stdio._stage_cold_start(warmup, out) is True

    stages = [(e.get("stage"), e["type"]) for e in _events(out)]
    assert stages == [
        (stdio.STAGE_MODEL_LOAD, "status"),
        (stdio.STAGE_PREFILL, "status"),
    ]
    load_msg = _events(out)[0]["message"]
    assert "Gemma-4-E4B-it-GGUF" in load_msg
    assert "first message" in load_msg  # expectation-setting is part of the contract
    assert warmup.consumed is True


def test_cold_start_already_loaded_stages_only_prefill():
    """Warm-up finished before the user typed: no load stage to show."""
    warmup = _warmup()
    out = io.StringIO()

    assert stdio._stage_cold_start(warmup, out) is True

    assert [e.get("stage") for e in _events(out)] == [stdio.STAGE_PREFILL]


def test_stages_are_emitted_once_per_session():
    warmup = _warmup()
    out = io.StringIO()
    stdio._stage_cold_start(warmup, out)

    again = io.StringIO()
    assert stdio._stage_cold_start(warmup, again) is True
    assert _events(again) == []


def test_failed_load_is_a_loud_terminal_error_not_a_stuck_spinner():
    warmup = _warmup(error=RuntimeError("llama-server failed to start"))
    calls = []
    warmup.retry_sync = lambda: calls.append(1) or setattr(  # retry also fails
        warmup, "error", RuntimeError("llama-server failed to start")
    )
    out = io.StringIO()

    assert stdio._stage_cold_start(warmup, out) is False

    events = _events(out)
    assert events[-1]["type"] == "error"
    assert "llama-server failed to start" in events[-1]["detail"]
    assert calls, "a failed background load must be retried in view of the user"
    # The stored error was consumed so the NEXT turn retries fresh.
    assert warmup.error is None or warmup.consumed is False


def test_failed_then_recovered_load_completes_the_stages():
    warmup = _warmup(error=RuntimeError("transient"))
    warmup.retry_sync = lambda: setattr(warmup, "error", None)
    out = io.StringIO()

    assert stdio._stage_cold_start(warmup, out) is True

    assert [e.get("stage") for e in _events(out)] == [
        stdio.STAGE_MODEL_LOAD,
        stdio.STAGE_PREFILL,
    ]


# ---------------------------------------------------------------------------
# run_turn integration: stages precede the agent's own events; a failed load
# never reaches process_query
# ---------------------------------------------------------------------------


class _RecordingAgent:
    def __init__(self):
        self.console = None
        self.queries = []

    def process_query(self, query):
        self.queries.append(query)
        return {"answer": "hi"}


def test_run_turn_writes_stages_before_agent_events():
    agent = _RecordingAgent()
    out = io.StringIO()

    stdio.run_turn(agent, "hello", out, warmup=_warmup())

    types = [(e.get("stage"), e["type"]) for e in _events(out)]
    assert types[0] == (stdio.STAGE_PREFILL, "status")
    assert types[-1] == (None, "final")
    assert agent.queries == ["hello"]


def test_run_turn_stops_before_the_agent_when_the_load_failed():
    warmup = _warmup(error=RuntimeError("boom"))
    warmup.retry_sync = lambda: setattr(warmup, "error", RuntimeError("boom"))
    agent = _RecordingAgent()
    out = io.StringIO()

    stdio.run_turn(agent, "hello", out, warmup=warmup)

    assert agent.queries == [], "a failed load must not stack a second failure"
    terminals = [e for e in _events(out) if e["type"] in ("final", "error")]
    assert len(terminals) == 1 and terminals[0]["type"] == "error"


# ---------------------------------------------------------------------------
# ModelWarmup.start_if_cold: only an explicit "not resident" starts a load
# ---------------------------------------------------------------------------


def test_start_if_cold_skips_when_resident(monkeypatch):
    monkeypatch.setattr(
        stdio,
        "LemonadeClient",
        lambda base_url=None, verbose=True: _ResidencyClient(
            _Status(loaded=["Gemma-4-E4B-it-GGUF"])
        ),
    )
    warmup = stdio.ModelWarmup(None, "Gemma-4-E4B-it-GGUF")
    assert warmup.start_if_cold() is False
    assert warmup.was_cold is False


def test_start_if_cold_skips_when_indeterminate(monkeypatch):
    """Loading on a guess could evict a model another process just placed."""
    monkeypatch.setattr(
        stdio,
        "LemonadeClient",
        lambda base_url=None, verbose=True: _ResidencyClient(_Status(running=False)),
    )
    warmup = stdio.ModelWarmup(None, "Gemma-4-E4B-it-GGUF")
    assert warmup.start_if_cold() is False


def test_start_if_cold_loads_in_the_background(monkeypatch):
    loads = []

    class _Loader(_ResidencyClient):
        def __init__(self, base_url=None, verbose=True):
            super().__init__(_Status(loaded=[]))

        def ensure_model_loaded(self, model):
            loads.append(model)

    monkeypatch.setattr(stdio, "LemonadeClient", _Loader)
    warmup = stdio.ModelWarmup(None, "Gemma-4-E4B-it-GGUF")

    assert warmup.start_if_cold() is True
    warmup.wait()

    assert loads == ["Gemma-4-E4B-it-GGUF"]
    assert warmup.was_cold is True
    assert warmup.error is None


def test_start_if_cold_stores_a_load_failure_for_the_first_turn(monkeypatch):
    class _Failing(_ResidencyClient):
        def __init__(self, base_url=None, verbose=True):
            super().__init__(_Status(loaded=[]))

        def ensure_model_loaded(self, model):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(stdio, "LemonadeClient", _Failing)
    warmup = stdio.ModelWarmup(None, "Gemma-4-E4B-it-GGUF")

    assert warmup.start_if_cold() is True
    warmup.wait()

    assert isinstance(warmup.error, RuntimeError)


# ---------------------------------------------------------------------------
# Startup ping residency field
# ---------------------------------------------------------------------------


def test_lemonade_health_reports_residency_when_asked(monkeypatch):
    class _Healthy(_ResidencyClient):
        base_url = "http://127.0.0.1:13305/api/v1"

        def __init__(self, base_url=None, verbose=True):
            super().__init__(_Status(loaded=["Gemma-4-E4B-it-GGUF"]))

        def health_check(self):
            return {"version": "10.7.0"}

    monkeypatch.setattr(stdio, "LemonadeClient", _Healthy)

    state = stdio._lemonade_health(None, model_id="Gemma-4-E4B-it-GGUF")
    assert state["model_loaded"] is True

    state = stdio._lemonade_health(None, model_id="not-this-one")
    assert state["model_loaded"] is False

    # No model_id (Claude chat backend): residency is not Lemonade's to claim.
    assert "model_loaded" not in stdio._lemonade_health(None)
