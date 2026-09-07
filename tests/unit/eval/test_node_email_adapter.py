# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Offline tests for the Node email-triage eval adapter.

No Node, no Lemonade, no model: these feed canned driver output through the
adapter and then through the REAL scoring path (``benchmark.build_result``), so
a break in the contract between the two shows up here rather than 40 minutes
into a self-hosted eval run.
"""

import base64
import json

import pytest

from gaia.eval import benchmark
from gaia.eval.node_email_adapter import (
    NodeEmailAgent,
    build_conversation,
    extract_body_text,
    gmail_message_to_driver_item,
)


def _b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def _gmail_message(
    gid: str = "abc123",
    *,
    subject: str = "Quick question",
    sender: str = "Dana Kim <dana@acme.io>",
    body: str = "Do you have five minutes?",
) -> dict:
    return {
        "id": gid,
        "threadId": gid,
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": sender},
                {"name": "To", "value": "me@acme.io"},
                {"name": "Cc", "value": "team@acme.io, lead@acme.io"},
                {"name": "Subject", "value": subject},
                {"name": "Date", "value": "Tue, 18 Aug 2026 09:00:00 +0000"},
            ],
            "body": {"data": _b64(body)},
        },
    }


class TestGmailConversion:
    def test_extracts_plain_text_body(self):
        payload = _gmail_message(body="Hello there")["payload"]
        assert extract_body_text(payload) == "Hello there"

    def test_prefers_plain_text_over_html_in_a_multipart_message(self):
        payload = {
            "mimeType": "multipart/alternative",
            "headers": [],
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64("<p>html</p>")}},
                {"mimeType": "text/plain", "body": {"data": _b64("plain")}},
            ],
        }
        assert extract_body_text(payload) == "plain"

    def test_falls_back_to_html_when_there_is_no_plain_part(self):
        payload = {
            "mimeType": "multipart/alternative",
            "headers": [],
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64("<p>only html</p>")}}
            ],
        }
        assert "only html" in extract_body_text(payload)

    def test_maps_headers_and_addresses_into_the_driver_item(self):
        item = gmail_message_to_driver_item(_gmail_message("gid-1"))
        assert item["message_id"] == "gid-1"
        assert item["from"] == {"email": "dana@acme.io", "name": "Dana Kim"}
        assert item["to"] == [{"email": "me@acme.io"}]
        assert [a["email"] for a in item["cc"]] == ["team@acme.io", "lead@acme.io"]
        assert item["subject"] == "Quick question"
        assert item["body"] == "Do you have five minutes?"

    def test_keys_on_the_gmail_id_not_the_rfc_message_id(self):
        """Ground truth is keyed on the Gmail id — joining on anything else
        silently scores every prediction against the wrong label."""
        msg = _gmail_message("gid-2")
        msg["payload"]["headers"].append(
            {"name": "Message-ID", "value": "<other@mail.example>"}
        )
        assert gmail_message_to_driver_item(msg)["message_id"] == "gid-2"

    def test_malformed_base64_body_fails_loud(self):
        payload = {
            "mimeType": "text/plain",
            "headers": [],
            "body": {"data": "!!!not-base64!!!"},
        }
        with pytest.raises(ValueError, match="base64url"):
            extract_body_text(payload)


class TestBuildConversation:
    def _driver_output(self, **overrides) -> dict:
        out = {
            "schema_version": "2.2",
            "results": [
                {
                    "id": "gid-1",
                    "category": "URGENT",
                    "is_spam": False,
                    "is_phishing": False,
                    "summary": "Contract needs review by Friday.",
                    "action_items": [],
                }
            ],
            "skipped": [],
            "errors": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            "llm_call_count": 2,
            "stats": [
                {
                    "time_to_first_token": 0.5,
                    "tokens_per_second": 30.0,
                    "input_tokens": 60,
                    "output_tokens": 10,
                    "step": "classify",
                },
                {
                    "time_to_first_token": 0.3,
                    "tokens_per_second": 40.0,
                    "input_tokens": 40,
                    "output_tokens": 10,
                    "step": "summarize",
                },
            ],
            "duration_ms": 1200,
        }
        out.update(overrides)
        return out

    def test_emits_a_scoreable_triage_envelope(self):
        conv = build_conversation(self._driver_output())["conversation"]
        tool_msgs = [m for m in conv if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        envelope = json.loads(tool_msgs[0]["content"])
        assert envelope["ok"] is True
        assert envelope["data"]["results"][0]["category"] == "URGENT"

    def test_emits_one_stats_entry_per_llm_call(self):
        conv = build_conversation(self._driver_output())["conversation"]
        stats = [
            m
            for m in conv
            if m["role"] == "system" and m["content"].get("type") == "stats"
        ]
        assert len(stats) == 2

    def test_drops_telemetry_gaps_rather_than_scoring_them_as_zero(self):
        """A failed /stats fetch must not enter the perf average as a 0 reading
        — that would drag TTFT down and read as an improvement."""
        output = self._driver_output(
            stats=[{"stats_error": "GET /stats -> HTTP 500", "step": "classify"}]
        )
        conv = build_conversation(output)["conversation"]
        assert not [
            m
            for m in conv
            if m["role"] == "system" and m["content"].get("type") == "stats"
        ]

    def test_omits_usage_so_tokens_are_never_double_counted(self):
        """build_result ADDS data.usage on top of the per-step totals; the
        adapter reports tokens once, via the stats entries."""
        conv = build_conversation(self._driver_output())["conversation"]
        envelope = json.loads([m for m in conv if m["role"] == "tool"][0]["content"])
        assert "usage" not in envelope["data"]

    def test_missing_results_fails_loud(self):
        with pytest.raises(ValueError, match="no 'results' key"):
            build_conversation({"errors": [{"id": "x", "message": "boom"}]})

    def test_non_dict_output_fails_loud(self):
        with pytest.raises(TypeError):
            build_conversation(["not", "a", "dict"])


class TestScoredByTheRealHarness:
    """The contract that matters: harness scoring of adapter output."""

    def _conversation(self):
        return build_conversation(
            {
                "results": [
                    {
                        "id": "gid-urgent",
                        "category": "URGENT",
                        "is_spam": False,
                        "is_phishing": False,
                        "summary": "s",
                        "action_items": [],
                    },
                    {
                        "id": "gid-promo",
                        "category": "PROMOTIONAL",
                        "is_spam": True,
                        "is_phishing": False,
                        "summary": "s",
                        "action_items": [],
                    },
                    # Wrong on purpose: truth is NEEDS_RESPONSE.
                    {
                        "id": "gid-miss",
                        "category": "FYI",
                        "is_spam": False,
                        "is_phishing": False,
                        "summary": "s",
                        "action_items": [],
                    },
                ],
                "llm_call_count": 3,
                "stats": [
                    {
                        "time_to_first_token": 0.4,
                        "tokens_per_second": 25.0,
                        "input_tokens": 50,
                        "output_tokens": 10,
                    }
                ],
            }
        )

    _GROUND_TRUTH = {
        "_meta": {"fixture": "unit"},
        "gid-urgent": {"category": "URGENT", "is_spam": False, "is_phishing": False},
        "gid-promo": {"category": "PROMOTIONAL", "is_spam": True, "is_phishing": False},
        "gid-miss": {
            "category": "NEEDS_RESPONSE",
            "is_spam": False,
            "is_phishing": False,
        },
    }

    def test_build_result_scores_the_adapter_output(self):
        row = benchmark.build_result(
            self._conversation(),
            run_id="unit-node",
            timestamp="2026-08-19T00:00:00Z",
            model_id="stub-model",
            total_duration_ms=1200,
            ground_truth=self._GROUND_TRUTH,
        )
        assert row["status"] == "PASS"
        # 2 of 3 categories exactly right.
        # The harness rounds reported rates to 4dp.
        assert row["quality"]["category_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
        # The miss is a needs-attention false negative, and it is counted.
        assert row["quality"]["needs_attention"]["fn"] == 1
        assert row["quality"]["spam"]["tp"] == 1

    def test_perf_metrics_come_through_the_stats_entries(self):
        row = benchmark.build_result(
            self._conversation(),
            run_id="unit-node",
            timestamp="2026-08-19T00:00:00Z",
            model_id="stub-model",
            total_duration_ms=1200,
        )
        assert row["avg_time_to_first_token_ms"] == pytest.approx(400.0)
        assert row["avg_tokens_per_second"] == pytest.approx(25.0)
        assert row["total_input_tokens"] == 50
        assert row["total_output_tokens"] == 10

    def test_an_empty_result_set_is_a_fail_not_a_silent_pass(self):
        row = benchmark.build_result(
            build_conversation({"results": [], "stats": []}),
            run_id="unit-node",
            timestamp="2026-08-19T00:00:00Z",
            model_id="stub-model",
            total_duration_ms=10,
        )
        assert row["status"] == "FAIL"


class TestNodeEmailAgentGuards:
    def test_requires_an_explicit_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            NodeEmailAgent(
                gmail_backend=object(),
                model_id="stub",
                base_url="",
            )

    def test_close_db_is_a_safe_no_op(self):
        agent = NodeEmailAgent(
            gmail_backend=object(),
            model_id="stub",
            base_url="http://127.0.0.1:1/api/v1",
            node_dir=".",
        )
        agent.close_db()


class TestImplementationAwareManifests:
    def test_node_manifests_resolve_and_ship_in_report_mode(self):
        """A Node run must never gate on bars nobody has measured yet."""
        quality = benchmark.load_default_quality_thresholds("node")
        perf = benchmark.load_default_perf_thresholds("node")
        assert quality.enforce is False
        assert perf.enforce is False

    def test_python_manifests_are_still_the_default(self):
        assert (
            benchmark.default_quality_thresholds_path().name
            == "quality_gate_thresholds.json"
        )
        assert (
            benchmark.default_quality_thresholds_path("node").name
            == "quality_gate_thresholds.node.json"
        )

    def test_unknown_implementation_fails_loud(self):
        with pytest.raises(ValueError, match="unknown email-triage implementation"):
            benchmark.default_quality_thresholds_path("rust")
