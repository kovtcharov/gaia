# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Base-framework trust ledger tests.

The generic empirical promotion gate (generalized from the email agent's
earn-trust engine): counter increments, the ``min_samples`` floor, the
``threshold`` accuracy gate, the not-enough-evidence case, and the policy
scaffolding (confirm-floor, ledger-proven promotion). Email-specific behavior
stays covered by ``hub/agents/email/python/tests/test_trust.py``.
"""

from __future__ import annotations

import pytest

from gaia.agents.base.trust_ledger import (
    AUTONOMY_LEVELS,
    LEVEL_EARN_TRUST,
    LEVEL_OFF,
    TrustDecision,
    TrustLedger,
    TrustPolicy,
)
from gaia.database.mixin import DatabaseMixin


class _DB(DatabaseMixin):
    pass


@pytest.fixture
def db():
    d = _DB()
    d.init_db(":memory:")
    TrustLedger.init_schema(d)
    return d


# ---------------------------------------------------------------------------
# Counter math
# ---------------------------------------------------------------------------


def test_record_outcome_creates_then_increments(db):
    TrustLedger.record_outcome(
        db, action_type="auto_load", scope="skill:triage", positive=True
    )
    stats = TrustLedger.get_stats(db, action_type="auto_load", scope="skill:triage")
    assert stats == {"positive": 1, "negative": 0, "total": 1, "score": 1.0}

    TrustLedger.record_outcome(
        db, action_type="auto_load", scope="skill:triage", positive=True
    )
    TrustLedger.record_outcome(
        db, action_type="auto_load", scope="skill:triage", positive=False
    )
    stats = TrustLedger.get_stats(db, action_type="auto_load", scope="skill:triage")
    assert stats["positive"] == 2
    assert stats["negative"] == 1
    assert stats["total"] == 3
    assert stats["score"] == pytest.approx(2 / 3)


def test_last_outcome_tracks_polarity(db):
    TrustLedger.record_outcome(db, action_type="a", scope="s", positive=True)
    TrustLedger.record_outcome(db, action_type="a", scope="s", positive=False)
    (row,) = TrustLedger.list_ledger(db)
    assert row["last_outcome"] == "negative"


def test_get_stats_no_evidence_is_zero(db):
    stats = TrustLedger.get_stats(db, action_type="auto_load", scope="skill:unseen")
    assert stats == {"positive": 0, "negative": 0, "total": 0, "score": 0.0}


def test_scopes_and_action_types_are_independent(db):
    TrustLedger.record_outcome(db, action_type="a", scope="s1", positive=True)
    TrustLedger.record_outcome(db, action_type="b", scope="s1", positive=False)
    TrustLedger.record_outcome(db, action_type="a", scope="s2", positive=False)
    assert TrustLedger.get_stats(db, action_type="a", scope="s1")["score"] == 1.0
    assert TrustLedger.get_stats(db, action_type="b", scope="s1")["score"] == 0.0
    assert TrustLedger.get_stats(db, action_type="a", scope="s2")["score"] == 0.0


# ---------------------------------------------------------------------------
# The promotion gate — min_samples floor AND threshold accuracy
# ---------------------------------------------------------------------------


def test_is_trusted_requires_min_samples(db):
    ledger = TrustLedger(min_samples=5, threshold=0.85)
    # 4/4 correct: perfect accuracy but below the sample floor → not trusted.
    for _ in range(4):
        ledger.record_outcome(db, action_type="a", scope="s", positive=True)
    assert not ledger.is_trusted(db, action_type="a", scope="s")
    # 5th correct crosses the floor at 100% → trusted.
    ledger.record_outcome(db, action_type="a", scope="s", positive=True)
    assert ledger.is_trusted(db, action_type="a", scope="s")


def test_is_trusted_requires_threshold_accuracy(db):
    ledger = TrustLedger(min_samples=5, threshold=0.85)
    # 8 correct, 2 wrong → 0.8 accuracy: enough samples but below 0.85.
    for _ in range(8):
        ledger.record_outcome(db, action_type="a", scope="s", positive=True)
    for _ in range(2):
        ledger.record_outcome(db, action_type="a", scope="s", positive=False)
    assert not ledger.is_trusted(db, action_type="a", scope="s")


def test_is_trusted_false_with_no_evidence(db):
    ledger = TrustLedger()
    assert not ledger.is_trusted(db, action_type="a", scope="never-seen")


def test_ledger_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        TrustLedger(min_samples=0)
    with pytest.raises(ValueError):
        TrustLedger(threshold=0.0)
    with pytest.raises(ValueError):
        TrustLedger(threshold=1.5)


# ---------------------------------------------------------------------------
# Table binding
# ---------------------------------------------------------------------------


def test_subclass_binds_its_own_table(db):
    class _MyLedger(TrustLedger):
        TABLE = "my_agent_trust_ledger"

    _MyLedger.init_schema(db)
    _MyLedger.record_outcome(db, action_type="a", scope="s", positive=True)
    # The subclass table has the row; the base table does not.
    assert _MyLedger.get_stats(db, action_type="a", scope="s")["total"] == 1
    assert TrustLedger.get_stats(db, action_type="a", scope="s")["total"] == 0


def test_subclass_rejects_invalid_table_name():
    with pytest.raises(ValueError):

        class _BadLedger(TrustLedger):
            TABLE = "bad table; DROP"


def test_upsert_commits_across_connection_teardown(tmp_path):
    """An outcome recorded on one connection must survive agent teardown —
    the scheduler rebuilds agents between fires against the same on-disk DB."""
    db_path = str(tmp_path / "state.db")
    first = _DB()
    first.init_db(db_path)
    TrustLedger.init_schema(first)
    TrustLedger.record_outcome(first, action_type="a", scope="s", positive=True)
    first.close_db()

    second = _DB()
    second.init_db(db_path)
    assert TrustLedger.get_stats(second, action_type="a", scope="s")["total"] == 1
    second.close_db()


# ---------------------------------------------------------------------------
# TrustPolicy scaffolding
# ---------------------------------------------------------------------------

FLOOR = frozenset({"send_now", "delete_forever"})


def _policy(level=LEVEL_EARN_TRUST, min_samples=3, threshold=0.85):
    ledger = TrustLedger(min_samples=min_samples, threshold=threshold)
    return TrustPolicy(level=level, ledger=ledger, confirm_floor=FLOOR)


def test_policy_rejects_unknown_level():
    with pytest.raises(ValueError):
        TrustPolicy(level="turbo", ledger=TrustLedger(), confirm_floor=FLOOR)


def test_policy_enabled_only_when_not_off():
    assert _policy(LEVEL_OFF).enabled is False
    for level in AUTONOMY_LEVELS:
        if level != LEVEL_OFF:
            assert _policy(level).enabled is True


def test_floor_decision_confirms_floor_tools_only():
    policy = _policy()
    decision = policy.floor_decision("send_now")
    assert isinstance(decision, TrustDecision)
    assert decision.action == "confirm"
    assert decision.confidence == 1.0
    assert policy.floor_decision("archive") is None


def test_trusted_decision_none_until_proven_then_auto(db):
    policy = _policy(min_samples=3, threshold=0.85)
    assert policy.trusted_decision(db, action_type="a", scopes=("scope:x",)) is None
    for _ in range(3):
        TrustLedger.record_outcome(db, action_type="a", scope="scope:x", positive=True)
    decision = policy.trusted_decision(db, action_type="a", scopes=("scope:x",))
    assert decision is not None
    assert decision.action == "auto"
    assert decision.confidence == pytest.approx(1.0)
    assert "3/3" in decision.reason


def test_trusted_decision_skips_empty_scopes(db):
    policy = _policy()
    assert policy.trusted_decision(db, action_type="a", scopes=("", "")) is None


def test_best_score_takes_max_across_scopes(db):
    policy = _policy()
    TrustLedger.record_outcome(db, action_type="a", scope="s1", positive=True)
    TrustLedger.record_outcome(db, action_type="a", scope="s2", positive=True)
    TrustLedger.record_outcome(db, action_type="a", scope="s2", positive=False)
    assert policy.best_score(db, action_type="a", scopes=("", "s1", "s2")) == 1.0
    assert policy.best_score(db, action_type="a", scopes=("s2",)) == pytest.approx(0.5)
    assert policy.best_score(db, action_type="a", scopes=("unseen",)) == 0.0


def test_base_decide_is_domain_specific():
    with pytest.raises(NotImplementedError):
        _policy().decide()
