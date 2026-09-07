# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Earned-trust ledger and policy scaffolding for agent autonomy.

Generalized from the email agent's earn-trust engine (#1483 / #1287) — the
first shipped empirical promotion gate in GAIA. Any agent that wants to widen
its autonomy only on evidence can reuse the same machinery:

- :class:`TrustLedger` — pure functions over a ``DatabaseMixin``-style ``db``
  handle: a per-``(action_type, scope)`` tally of positive vs. negative
  outcomes. ``action_type`` is the agent's own taxonomy key (``archive``,
  ``auto_load_skill``, …); ``scope`` is the granularity trust accrues at
  (a sender, a category, a skill name). A scope becomes trusted only after
  ``min_samples`` outcomes at ``threshold`` accuracy — a counter, not a vibe.

- :class:`TrustPolicy` — scaffolding for the decision layer. It owns the
  autonomy-level validation, the inviolable confirm-floor, and the
  ledger-proven promotion check. ``decide()`` itself is domain-specific:
  subclasses implement it from the helpers here (the email agent's
  ``gaia_agent_email.trust.TrustPolicy`` is the reference implementation).

Storage discipline: the ledger is OPERATIONAL state, so it lives in the
agent's SQLite ``state.db`` via ``DatabaseMixin`` — NOT in MemoryStore, whose
knowledge is subject to LLM consolidation, which must never rewrite an audit
tally. Subclasses bind their own table via the ``TABLE`` class attribute so
each agent's ledger stays in its own namespace.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

from gaia.database.sql_safety import validate_identifier

# ---------------------------------------------------------------------------
# Autonomy levels — the earn-trust gradient
# ---------------------------------------------------------------------------

#: No autonomous activity at all. Default.
LEVEL_OFF = "off"
#: Propose everything, execute nothing autonomously.
LEVEL_SUGGEST = "suggest"
#: Auto-execute only in scopes that have earned trust; propose the rest.
LEVEL_EARN_TRUST = "earn_trust"
#: Auto-execute every auto-eligible action immediately.
LEVEL_FULL = "full"

AUTONOMY_LEVELS = (LEVEL_OFF, LEVEL_SUGGEST, LEVEL_EARN_TRUST, LEVEL_FULL)

AutonomyLevel = Literal["off", "suggest", "earn_trust", "full"]

# Ledger outcome polarity.
OUTCOME_POSITIVE = "positive"
OUTCOME_NEGATIVE = "negative"

Decision = Literal["auto", "draft", "suggest", "confirm"]


@dataclass(frozen=True)
class TrustDecision:
    """A policy's verdict for one candidate action.

    ``action`` is the disposition; ``reason`` is a human-readable rationale
    surfaced wherever the agent explains itself; ``confidence`` is the ledger
    trust score in ``[0, 1]`` (1.0 for the hard-coded floor and for explicit
    user preferences).
    """

    action: Decision
    reason: str
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# TrustLedger — the earned-evidence tally
# ---------------------------------------------------------------------------

_LEDGER_DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {table} (
    action_type  TEXT NOT NULL,
    scope        TEXT NOT NULL,
    positive     INTEGER NOT NULL DEFAULT 0,
    negative     INTEGER NOT NULL DEFAULT 0,
    last_outcome TEXT,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (action_type, scope)
);
"""


class TrustLedger:
    """Pure-function accessors over a per-agent trust-ledger table.

    Every accessor takes a ``DatabaseMixin``-typed ``db`` as its first
    argument and never reaches into an agent class. Instances hold only the
    trust thresholds so a :class:`TrustPolicy` can share one configured
    ledger. Subclasses bind their own table by overriding ``TABLE``
    (validated as a bare SQL identifier at class-definition time).
    """

    #: Table the ledger reads and writes. Override in a subclass to give an
    #: agent its own namespace (e.g. the email agent's ``email_trust_ledger``).
    TABLE: str = "trust_ledger"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        validate_identifier(cls.TABLE, "table name", f"{cls.__name__}.TABLE")

    def __init__(self, *, min_samples: int = 5, threshold: float = 0.85) -> None:
        if min_samples < 1:
            raise ValueError(
                f"{type(self).__name__} min_samples must be >= 1, got {min_samples!r}"
            )
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                f"{type(self).__name__} threshold must be in (0, 1], got {threshold!r}"
            )
        self.min_samples = min_samples
        self.threshold = threshold

    @classmethod
    def ddl(cls) -> str:
        """The ``CREATE TABLE IF NOT EXISTS`` statement for this ledger."""
        return _LEDGER_DDL_TEMPLATE.format(table=cls.TABLE)

    @classmethod
    def init_schema(cls, db) -> None:
        """Create the ledger table if absent. Idempotent."""
        db.execute(cls.ddl())

    @classmethod
    def record_outcome(
        cls,
        db,
        *,
        action_type: str,
        scope: str,
        positive: bool,
        now: Optional[float] = None,
    ) -> None:
        """Increment the positive or negative tally for one scope.

        Upsert: create the row on first sight, otherwise bump the counter.
        A single decision the user accepted / left standing is positive; one
        they rejected / undid / edited is negative.
        """
        ts = time.time() if now is None else now
        outcome = OUTCOME_POSITIVE if positive else OUTCOME_NEGATIVE
        # Atomic upsert (single statement) so two concurrent first-writes to
        # the same (action_type, scope) on one shared on-disk DB can't both
        # INSERT and collide on the PK. Wrapped in a transaction so the write
        # commits (query() alone does not).
        with db.transaction():
            db.query(
                f"INSERT INTO {cls.TABLE} "
                "(action_type, scope, positive, negative, last_outcome, updated_at) "
                "VALUES (:a, :s, :pos, :neg, :outcome, :ts) "
                "ON CONFLICT(action_type, scope) DO UPDATE SET "
                "positive = positive + :pos, negative = negative + :neg, "
                "last_outcome = :outcome, updated_at = :ts",
                {
                    "a": action_type,
                    "s": scope,
                    "pos": 1 if positive else 0,
                    "neg": 0 if positive else 1,
                    "outcome": outcome,
                    "ts": ts,
                },
            )

    @classmethod
    def get_stats(cls, db, *, action_type: str, scope: str) -> Dict[str, Any]:
        """Return ``{positive, negative, total, score}`` for a scope.

        ``score`` is ``positive / total`` (0.0 when there is no evidence yet).
        """
        row = db.query(
            f"SELECT positive, negative FROM {cls.TABLE} "
            "WHERE action_type = :a AND scope = :s",
            {"a": action_type, "s": scope},
            one=True,
        )
        pos = int(row["positive"]) if row else 0
        neg = int(row["negative"]) if row else 0
        total = pos + neg
        score = (pos / total) if total else 0.0
        return {"positive": pos, "negative": neg, "total": total, "score": score}

    def is_trusted(self, db, *, action_type: str, scope: str) -> bool:
        """True when a scope has earned enough correct outcomes to auto-run.

        Requires BOTH a minimum sample count (so a single lucky call can't
        unlock autonomy) AND an accuracy at/above the threshold.
        """
        stats = self.get_stats(db, action_type=action_type, scope=scope)
        return stats["total"] >= self.min_samples and stats["score"] >= self.threshold

    @classmethod
    def list_ledger(cls, db) -> List[Dict[str, Any]]:
        """Every ledger row, most-recently-updated first (for a CLI/UI)."""
        return db.query(
            "SELECT action_type, scope, positive, negative, last_outcome, "
            f"updated_at FROM {cls.TABLE} ORDER BY updated_at DESC"
        )


# ---------------------------------------------------------------------------
# TrustPolicy — decision-layer scaffolding
# ---------------------------------------------------------------------------


class TrustPolicy:
    """Scaffolding for a domain policy that decides candidate actions.

    Owns the pieces every earn-trust policy shares — level validation, the
    inviolable confirm-floor, and the ledger-proven promotion check.
    Subclasses implement :meth:`decide` from these helpers; the floor check
    must run first, and nothing a subclass does may lower it.
    """

    def __init__(
        self,
        *,
        level: str,
        ledger: TrustLedger,
        confirm_floor: frozenset,
    ) -> None:
        if level not in AUTONOMY_LEVELS:
            raise ValueError(
                f"{type(self).__name__} level must be one of "
                f"{AUTONOMY_LEVELS}, got {level!r}"
            )
        self.level = level
        self.ledger = ledger
        self.confirm_floor = frozenset(confirm_floor)

    @property
    def enabled(self) -> bool:
        """False only at :data:`LEVEL_OFF` — no autonomous loop should run."""
        return self.level != LEVEL_OFF

    def floor_decision(self, tool: str) -> Optional[TrustDecision]:
        """``confirm`` when a tool sits on the inviolable floor, else None.

        No level, trust score, or preference ever lowers the floor — call
        this first in every ``decide()`` implementation.
        """
        if tool in self.confirm_floor:
            return TrustDecision(
                "confirm",
                reason=f"{tool} is destructive/irreversible — always requires "
                "your confirmation",
                confidence=1.0,
            )
        return None

    def trusted_decision(
        self, db, *, action_type: str, scopes: Sequence[str]
    ) -> Optional[TrustDecision]:
        """``auto`` for the first ledger-proven scope, else None.

        Empty scope strings are skipped so callers can pass optional scopes
        unconditionally.
        """
        for scope in scopes:
            if not scope:
                continue
            if self.ledger.is_trusted(db, action_type=action_type, scope=scope):
                stats = self.ledger.get_stats(db, action_type=action_type, scope=scope)
                return TrustDecision(
                    "auto",
                    reason=(
                        f"proven on {scope} "
                        f"({stats['positive']}/{stats['total']} correct)"
                    ),
                    confidence=stats["score"],
                )
        return None

    def best_score(self, db, *, action_type: str, scopes: Sequence[str]) -> float:
        """Highest ledger score across the given scopes (0.0 with no evidence).

        Lets a not-yet-trusted proposal carry the current accuracy as its
        confidence, so the user sees how close a scope is to promotion.
        """
        best = 0.0
        for scope in scopes:
            if scope:
                best = max(
                    best,
                    self.ledger.get_stats(db, action_type=action_type, scope=scope)[
                        "score"
                    ],
                )
        return best

    def decide(self, **kwargs: Any) -> TrustDecision:
        """Domain-specific — implement in a subclass.

        The base class cannot know an agent's action taxonomy (what is
        reversible, what is a draft, what guards apply), so it refuses rather
        than guessing a disposition.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.decide() is domain-specific: subclass "
            "gaia.agents.base.trust_ledger.TrustPolicy and implement decide() "
            "using floor_decision()/trusted_decision()/best_score(). See "
            "gaia_agent_email.trust.TrustPolicy for the reference "
            "implementation."
        )
