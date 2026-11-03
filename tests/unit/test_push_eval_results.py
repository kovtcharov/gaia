# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""util/push_eval_results.py: what a CI lane sends to the admin dashboard."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "util" / "push_eval_results.py"
spec = importlib.util.spec_from_file_location("push_eval_results", SCRIPT)
push_eval_results = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push_eval_results)

ENV = {
    "GITHUB_REPOSITORY": "amd/gaia",
    "GITHUB_RUN_ID": "77",
    "GITHUB_WORKFLOW": "Agent Eval",
    "GITHUB_EVENT_NAME": "push",
    "GITHUB_REF_NAME": "v0.25.4-rc2",
    "GITHUB_SHA": "a" * 40,
}


@pytest.fixture
def results(tmp_path):
    (tmp_path / "gaia_files").mkdir()
    (tmp_path / "gaia_files" / "scorecard.json").write_text(
        json.dumps(
            {
                "config": {"model": "judge-model"},
                "scenarios": [
                    {
                        "scenario_id": "one",
                        "status": "PASS",
                        "overall_score": 9.5,
                        "turns": [{"user": "private transcript"}, {"user": "more"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # PowerShell's UTF8 writers can leave a BOM; the lane files must still parse.
    (tmp_path / "lane-status.json").write_text(
        json.dumps({"lane": "smoke-1", "tier": "smoke", "agent_model": "m"}),
        encoding="utf-8-sig",
    )
    (tmp_path / "lane-timing.json").write_text(
        json.dumps({"total_minutes": 12.5}), encoding="utf-8"
    )
    return tmp_path


def test_payload_names_the_run_and_lane(results):
    p = push_eval_results.build_payload(results, "fallback", ENV)
    assert p["run"]["id"] == 77
    assert p["run"]["html_url"] == "https://github.com/amd/gaia/actions/runs/77"
    assert (p["lane"], p["tier"], p["agent_model"]) == ("smoke-1", "smoke", "m")
    assert p["judge_model"] == "judge-model"
    assert p["minutes"] == 12.5
    assert p["scorecards"][0]["category"] == "gaia_files"


def test_transcripts_are_never_sent(results):
    p = push_eval_results.build_payload(results, "fallback", ENV)
    assert p["scorecards"][0]["scorecard"]["scenarios"][0]["turns"] == 2
    assert "private transcript" not in json.dumps(p)


def test_task_suites_pass_through(tmp_path):
    (tmp_path / "scorecard.json").write_text(
        json.dumps({"suite": "swebench", "tasks": [{"id": "t", "passed": True}]}),
        encoding="utf-8",
    )
    p = push_eval_results.build_payload(tmp_path, "swebench", ENV)
    assert p["lane"] == "swebench"
    assert p["scorecards"][0]["scorecard"]["tasks"][0]["id"] == "t"


def test_no_credentials_skips_without_failing(monkeypatch, capsys):
    monkeypatch.delenv("ADMIN_ACCESS_CLIENT_ID", raising=False)
    monkeypatch.delenv("ADMIN_ACCESS_CLIENT_SECRET", raising=False)
    monkeypatch.setattr("sys.argv", ["push", "--lane", "x"])
    assert push_eval_results.main() == 0
    assert "not pushed" in capsys.readouterr().out


def test_a_refused_push_is_a_failure(results, monkeypatch, capsys):
    monkeypatch.setenv("ADMIN_ACCESS_CLIENT_ID", "id")
    monkeypatch.setenv("ADMIN_ACCESS_CLIENT_SECRET", "secret")
    for key, value in ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        "sys.argv", ["push", "--lane", "x", "--results-dir", str(results)]
    )

    def refuse(*_args):
        raise OSError("connection refused")

    monkeypatch.setattr(push_eval_results, "push", refuse)
    assert push_eval_results.main() == 1
    assert "::warning::" in capsys.readouterr().out
