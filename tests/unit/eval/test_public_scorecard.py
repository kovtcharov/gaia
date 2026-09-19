# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The public scorecard keeps every score and none of the conversation."""

import json

from gaia.eval.scorecard import main, public_scorecard

CARD = {
    "summary": {"passed": 1},
    "scenarios": [
        {
            "scenario_id": "s1",
            "status": "PASS",
            "overall_score": 9.1,
            "error": "claude CLI stderr",
            "turns": [
                {
                    "turn": 1,
                    "user_message": "what does the handbook say?",
                    "agent_response": "15 days of PTO",
                    "agent_tools": ["query_documents"],
                    "scores": {"correctness": 10},
                    "overall_score": 9.1,
                    "pass": True,
                    "reasoning": "matches the fact",
                    "error": "partial output",
                }
            ],
        }
    ],
}


def test_conversation_text_is_removed_and_every_score_kept():
    public = public_scorecard(CARD)
    scenario = public["scenarios"][0]
    turn = scenario["turns"][0]
    assert "error" not in scenario
    for field in ("user_message", "agent_response", "reasoning", "error"):
        assert field not in turn
    assert turn["scores"] == {"correctness": 10} and turn["pass"] is True
    assert scenario["overall_score"] == 9.1 and public["summary"] == {"passed": 1}
    assert CARD["scenarios"][0]["turns"][0]["agent_response"] == "15 days of PTO"


def test_the_command_writes_the_public_copy(tmp_path):
    source, dest = tmp_path / "scorecard.json", tmp_path / "scorecard.public.json"
    source.write_text(json.dumps(CARD), encoding="utf-8")
    assert main([str(source), str(dest)]) == 0
    written = json.loads(dest.read_text(encoding="utf-8"))
    assert written == public_scorecard(CARD)
    assert "15 days of PTO" not in dest.read_text(encoding="utf-8")
