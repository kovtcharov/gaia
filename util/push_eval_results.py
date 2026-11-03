# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Push one finished eval lane's results to the admin dashboard.

Runs as its own CI step after a lane, so the dashboard shows a run while the
other lanes are still going and does not depend on the nightly artifact read.

Sends only the numbers the dashboard stores. Transcripts stay in the artifact.
Exits 0 with a notice when the credentials are absent (fork pull requests never
get them) and 1 with a CI warning when a push is refused.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "https://amd-gaia.ai/admin/api/ingest/evals"


def trim_scenario(scenario: dict) -> dict:
    """Keep the fields the dashboard reads; a turn list becomes its length."""
    keep = (
        "scenario_id",
        "category",
        "status",
        "overall_score",
        "elapsed_s",
        "root_cause",
        "performance_summary",
        "tool_usage",
        "cost_estimate",
    )
    out = {k: scenario[k] for k in keep if k in scenario}
    turns = scenario.get("turns")
    out["turns"] = len(turns) if isinstance(turns, list) else turns
    return out


def trim_scorecard(scorecard: dict) -> dict:
    if isinstance(scorecard.get("tasks"), list):
        return scorecard
    return {
        "config": scorecard.get("config", {}),
        "scenarios": [trim_scenario(s) for s in scorecard.get("scenarios", [])],
    }


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        raise ValueError(f"could not read {path}: {e}") from e


def build_payload(results_dir: Path, lane: str, env: dict) -> dict:
    """The lane's scorecards plus the GitHub run they belong to."""
    status_file = results_dir / "lane-status.json"
    timing_file = results_dir / "lane-timing.json"
    status = read_json(status_file) if status_file.exists() else {}
    timing = read_json(timing_file) if timing_file.exists() else {}
    scorecards = [
        {"category": p.parent.name, "scorecard": trim_scorecard(read_json(p))}
        for p in sorted(results_dir.glob("**/scorecard.json"))
    ]
    judge = next(
        (
            s["scorecard"]["config"].get("model")
            for s in scorecards
            if s["scorecard"].get("config")
        ),
        None,
    )
    repo = env["GITHUB_REPOSITORY"]
    run_id = int(env["GITHUB_RUN_ID"])
    return {
        "run": {
            "id": run_id,
            "workflow": env["GITHUB_WORKFLOW"],
            "event": env["GITHUB_EVENT_NAME"],
            "ref": env.get("GITHUB_REF_NAME"),
            "sha": env["GITHUB_SHA"],
            "html_url": f"https://github.com/{repo}/actions/runs/{run_id}",
        },
        "lane": status.get("lane") or lane,
        "tier": status.get("tier"),
        "agent_model": status.get("agent_model"),
        "judge_model": judge,
        "minutes": timing.get("total_minutes"),
        "scorecards": scorecards,
    }


def push(url: str, payload: dict, client_id: str, client_secret: str) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-admin-action": "ingest",
            "CF-Access-Client-Id": client_id,
            "CF-Access-Client-Secret": client_secret,
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("eval-out"))
    parser.add_argument("--lane", required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()

    client_id = os.environ.get("ADMIN_ACCESS_CLIENT_ID", "")
    client_secret = os.environ.get("ADMIN_ACCESS_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        print(
            "::notice::Eval results not pushed: ADMIN_ACCESS_CLIENT_ID/SECRET are not set here."
        )
        return 0

    try:
        payload = build_payload(args.results_dir, args.lane, os.environ)
        print(push(args.url, payload, client_id, client_secret))
    except urllib.error.HTTPError as e:
        print(
            f"::warning::Dashboard refused the eval push (HTTP {e.code}): {e.read().decode('utf-8', 'replace')[:300]}"
        )
        return 1
    except (OSError, ValueError, KeyError) as e:
        print(f"::warning::Could not push eval results to {args.url}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
