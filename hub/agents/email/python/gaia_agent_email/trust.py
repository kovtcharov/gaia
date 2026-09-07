# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Earn-trust policy engine for autonomous email handling (#1483 / #1287).

Two cooperating pieces, both built on the base framework's generic earn-trust
machinery (``gaia.agents.base.trust_ledger``, generalized from this module):

- :class:`TrustLedger` — pure functions over the agent's SQLite handle
  (mirrors ``action_store``/``schedule_store``): a per-``(action_type, scope)``
  tally of positive vs. negative outcomes. "scope" is a category
  (``category:PROMOTIONAL``) or a sender (``sender:news@x.com``). Trust is
  *earned* here — a scope becomes trusted only after enough correct decisions.
  The counter math lives in the base class; this subclass only binds the
  ``email_trust_ledger`` table.

- :class:`TrustPolicy` — the decision layer. Given a candidate action it
  returns a :class:`TrustDecision` of ``auto`` | ``draft`` | ``suggest`` |
  ``confirm``. It reads the ledger, the user's explicit preferences, and the
  configured autonomy level. The floor/promotion scaffolding is inherited;
  the email action taxonomy and guards are implemented here.

Inviolable floor (the whole point of "check in on destructive things"):
tools in the agent's ``CONFIRMATION_REQUIRED_TOOLS`` — send, forward, permanent
delete, calendar RSVP, phishing quarantine — ALWAYS resolve to ``confirm``, at
every autonomy level, regardless of how much trust a scope has earned. The
policy layer can widen what runs silently; it can NEVER lower the floor. A
misconfigured level or a fully-trusted sender still cannot cause an unattended
send. ``tests/test_trust.py`` locks this in.

Reversible-first: only reversible actions (archive, label, mark-read, star) are
ever auto-executed, and every one is recorded in ``action_store`` with undo.
Reply composition is ``draft`` — the agent writes the reply but never sends it
unattended; sending stays on the floor.

Storage choice: the accuracy ledger is OPERATIONAL state (like ``action_store``)
so it lives in the agent's ``state.db`` via ``DatabaseMixin``, NOT in
MemoryStore — MemoryStore knowledge is subject to LLM consolidation, which must
never rewrite an audit tally. Spec §6.6 permission *grants* still live in
MemoryStore; this ledger is the evidence those grants are earned from.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Mapping, Optional

# The generic earn-trust machinery (levels, decision dataclass, counter
# ledger, policy scaffolding) lives in the base framework — generalized from
# this module. Names are re-exported here so existing email-agent imports
# (`from gaia_agent_email.trust import LEVEL_OFF, TrustDecision, ...`) keep
# working unchanged.
from gaia.agents.base.trust_ledger import (  # noqa: F401
    AUTONOMY_LEVELS,
    LEVEL_EARN_TRUST,
    LEVEL_FULL,
    LEVEL_OFF,
    LEVEL_SUGGEST,
    OUTCOME_NEGATIVE,
    OUTCOME_POSITIVE,
    AutonomyLevel,
    Decision,
    TrustDecision,
)
from gaia.agents.base.trust_ledger import TrustLedger as BaseTrustLedger
from gaia.agents.base.trust_ledger import TrustPolicy as BaseTrustPolicy

# ---------------------------------------------------------------------------
# Action taxonomy
# ---------------------------------------------------------------------------

#: Reversible mutations that MAY be auto-executed once a scope is trusted.
#: Each is recorded in ``action_store`` with an undo path. ``trash`` is
#: deliberately excluded — a soft-delete is reversible only inside the undo
#: window, so it stays a suggestion until the user opts it up explicitly.
# Names MUST match the ``action_type`` strings ``action_store`` records (so an
# undo of an autonomy action attributes to the same action_type the ledger
# learned under). See organize_tools/delete_tools ``record_action`` calls.
REVERSIBLE_AUTO_ACTIONS = frozenset(
    {
        "archive",
        "add_label",
        "add_star",
        "remove_star",
        "mark_read",
        "mark_unread",
    }
)

#: Reply/forward *composition*. These prepare content but never transmit — the
#: send is a separate floor tool. So drafting is safe to do autonomously even
#: though sending is not.
DRAFT_ACTIONS = frozenset({"draft_reply", "draft_forward"})


# ---------------------------------------------------------------------------
# TrustLedger — the earned-evidence tally
# ---------------------------------------------------------------------------


class TrustLedger(BaseTrustLedger):
    """The email agent's trust ledger, over the ``email_trust_ledger`` table.

    A table-binding subclass of the base framework's
    :class:`~gaia.agents.base.trust_ledger.TrustLedger` — the counter math,
    the ``min_samples``/``threshold`` gate, and the upsert discipline all
    live there. The table name (and therefore the on-disk schema in the
    agent's ``state.db``) is unchanged.
    """

    TABLE = "email_trust_ledger"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EMAIL_TRUST_LEDGER_DDL = TrustLedger.ddl()

# Attribution index: maps each autonomously-executed action back to the
# ``(action_type, sender, category)`` scope it was decided under. When the user
# later undoes that action (a correction), :func:`lookup_autonomy_action`
# recovers the scope so the negative signal lands on the right ledger rows.
# One row per auto-executed action; ``resolved`` guards against a single undo
# being counted twice.
EMAIL_AUTONOMY_ACTIONS_DDL = """
CREATE TABLE IF NOT EXISTS email_autonomy_actions (
    action_id    TEXT PRIMARY KEY,
    action_type  TEXT NOT NULL,
    sender       TEXT,
    category     TEXT,
    created_at   REAL NOT NULL,
    resolved     INTEGER NOT NULL DEFAULT 0
);
"""


# Open-proposal ledger: one row per message the cycle has already proposed an
# action for and that has not yet been resolved. Without this, every heartbeat
# re-proposes the same still-in-inbox message and GoalStore accumulates a
# duplicate pending goal each fire. Keyed ``(message_id, action_type)``.
EMAIL_AUTONOMY_PROPOSALS_DDL = """
CREATE TABLE IF NOT EXISTS email_autonomy_proposals (
    message_id   TEXT NOT NULL,
    action_type  TEXT NOT NULL,
    created_at   REAL NOT NULL,
    resolved     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (message_id, action_type)
);
"""


def init_trust_schema(db) -> None:
    """Create the ledger + attribution tables if absent. Idempotent."""
    db.execute(EMAIL_TRUST_LEDGER_DDL)
    db.execute(EMAIL_AUTONOMY_ACTIONS_DDL)
    db.execute(EMAIL_AUTONOMY_PROPOSALS_DDL)


def has_open_proposal(db, *, message_id: str, action_type: str) -> bool:
    """True when an unresolved proposal already exists for this message+action.

    The re-proposal guard: a message the cycle proposed last fire and that the
    user has not acted on yet must not be proposed again.
    """
    row = db.query(
        "SELECT 1 FROM email_autonomy_proposals WHERE message_id = :m "
        "AND action_type = :a AND resolved = 0",
        {"m": message_id, "a": action_type},
        one=True,
    )
    return row is not None


def record_proposal(
    db,
    *,
    message_id: str,
    action_type: str,
    now: Optional[float] = None,
) -> None:
    """Mark that an action has been proposed for a message. Idempotent."""
    ts = time.time() if now is None else now
    # Wrapped in a transaction so the INSERT commits (query() alone does not).
    # The scheduler rebuilds the agent between fires and closes its connection
    # after each cycle — an uncommitted dedup row would be lost and the next
    # fire would re-propose the same still-in-inbox message.
    with db.transaction():
        db.query(
            "INSERT OR IGNORE INTO email_autonomy_proposals "
            "(message_id, action_type, created_at, resolved) VALUES (:m, :a, :ts, 0)",
            {"m": message_id, "a": action_type, "ts": ts},
        )


def resolve_proposal(db, *, message_id: str, action_type: str) -> None:
    """Clear the open-proposal guard for a message (acted on or superseded)."""
    db.update(
        "email_autonomy_proposals",
        {"resolved": 1},
        "message_id = :m AND action_type = :a",
        {"m": message_id, "a": action_type},
    )


def record_autonomy_action(
    db,
    *,
    action_id: str,
    action_type: str,
    sender: str = "",
    category: str = "",
    now: Optional[float] = None,
) -> None:
    """Index one auto-executed action so a later undo can be attributed.

    Idempotent on ``action_id`` (INSERT OR REPLACE) — re-recording the same
    action id overwrites rather than duplicating.
    """
    ts = time.time() if now is None else now
    # ``db.insert`` commits (query() does not); action_id is a fresh uuid PK so a
    # plain insert is safe — no REPLACE needed. Committing matters for the
    # scheduler's per-run agent, which closes its connection after the cycle: an
    # uncommitted index row would be lost and the later undo never attributed.
    db.insert(
        "email_autonomy_actions",
        {
            "action_id": action_id,
            "action_type": action_type,
            "sender": sender,
            "category": category,
            "created_at": ts,
            "resolved": 0,
        },
    )


def lookup_autonomy_action(db, *, action_id: str) -> Optional[Dict[str, Any]]:
    """Return the unresolved index row for an action id, or None.

    A row already marked ``resolved`` returns None so the same undo can't be
    scored twice.
    """
    return db.query(
        "SELECT action_id, action_type, sender, category FROM "
        "email_autonomy_actions WHERE action_id = :id AND resolved = 0",
        {"id": action_id},
        one=True,
    )


def mark_autonomy_action_resolved(db, *, action_id: str) -> None:
    """Flag an indexed action as resolved (its correction has been counted)."""
    db.update(
        "email_autonomy_actions",
        {"resolved": 1},
        "action_id = :id",
        {"id": action_id},
    )


# ---------------------------------------------------------------------------
# Scope helpers
# ---------------------------------------------------------------------------


def category_scope(category: str) -> str:
    """Scope key for a triage category, e.g. ``category:PROMOTIONAL``."""
    return f"category:{(category or '').strip().upper()}"


def sender_scope(sender: str) -> str:
    """Scope key for a sender address, e.g. ``sender:news@x.com``."""
    return f"sender:{(sender or '').strip().lower()}"


# ---------------------------------------------------------------------------
# Correction capture (#2529) — pure ``db``-over functions so any
# ``DatabaseMixin`` holder can record a trust signal, not only a live
# ``EmailTriageAgent``. ``EmailTriageAgent.record_autonomy_outcome`` /
# ``note_action_undone`` are thin wrappers around these two.
# ---------------------------------------------------------------------------


def record_autonomy_outcome(
    db,
    *,
    action_type: str,
    positive: bool,
    sender: str = "",
    category: str = "",
) -> None:
    """Record one trust signal against both the sender and category scope.

    The outcome is recorded against BOTH scopes (whichever are non-empty) so
    trust accrues at whichever granularity recurs — a specific sender AND its
    category both learn from the same decision.
    """
    for scope in (
        sender_scope(sender) if sender else "",
        category_scope(category) if category else "",
    ):
        if scope:
            TrustLedger.record_outcome(
                db, action_type=action_type, scope=scope, positive=positive
            )


def note_autonomy_undo(db, *, action_id: str) -> bool:
    """Capture a correction: an auto-executed action that was undone.

    Looks up ``action_id`` in the attribution index; if it was recorded as an
    autonomy decision (:func:`record_autonomy_action`), records a negative
    outcome for its scope and marks the index row resolved (so the same undo
    can't be counted twice). Returns True when a correction was captured,
    False when ``action_id`` was not an autonomy action (e.g. the user undid
    something they did manually) — a no-op, not an error.
    """
    row = lookup_autonomy_action(db, action_id=action_id)
    if row is None:
        return False
    record_autonomy_outcome(
        db,
        action_type=row["action_type"],
        positive=False,
        sender=row.get("sender") or "",
        category=row.get("category") or "",
    )
    mark_autonomy_action_resolved(db, action_id=action_id)
    return True


# ---------------------------------------------------------------------------
# Auto-archive safety guard (#2426)
# ---------------------------------------------------------------------------

# Account-security / account-notification senders. A message from one of these
# is never auto-archived unattended — even at ``full`` or on a fully-trusted
# scope — because a mis-categorized security alert silently leaving the inbox
# is the exact failure this guards against. Matched against the bare, lowercased
# address. Deliberately conservative and provider-anchored (known account
# domains + any ``accounts``/``account`` leading domain label + explicit
# ``security``/``account-security`` local-parts) rather than broad (e.g. "any
# no-reply@"), which would drown ``full`` mode in proposals. Downgrading to a
# proposal is the safe failure mode, so bias toward catching a real security
# sender over avoiding a stray proposal.
_SECURITY_SENDER_DOMAINS = frozenset(
    {
        "id.apple.com",
        "accountprotection.microsoft.com",
    }
)
_SECURITY_LEADING_LABELS = frozenset({"accounts", "account"})
_SECURITY_LOCALPART_RE = re.compile(r"^(?:security|account-security)(?:[-.+].*)?$")


def is_security_sender(sender: str) -> bool:
    """True when a sender is an account-security / account-notification address.

    Anchored on a small set of known account domains, on any domain whose
    leading label is ``accounts``/``account`` (``accounts.google.com``,
    ``accounts.anybank.com``), and on a ``security``/``account-security``
    local-part (exactly, or followed by a ``-``/``.``/``+`` separator — so
    ``security@`` and ``security-alert@`` match but ``securityweekly@`` does
    not). Case-insensitive; empty / address-less input is False.
    """
    addr = (sender or "").strip().lower().rstrip(">")
    if "@" not in addr:
        return False
    local, _, domain = addr.partition("@")
    domain = domain.strip()
    if any(domain == d or domain.endswith("." + d) for d in _SECURITY_SENDER_DOMAINS):
        return True
    if domain.split(".", 1)[0] in _SECURITY_LEADING_LABELS:
        return True
    return bool(_SECURITY_LOCALPART_RE.match(local))


# ---------------------------------------------------------------------------
# TrustPolicy — the decision layer
# ---------------------------------------------------------------------------


class TrustPolicy(BaseTrustPolicy):
    """Decide the disposition of a candidate autonomous email action.

    Construct with the configured autonomy level, the ledger, and the agent's
    inviolable confirm-floor set (all validated by the base
    :class:`~gaia.agents.base.trust_ledger.TrustPolicy`). :meth:`decide`
    returns a :class:`TrustDecision` and carries the email-specific steps:
    the draft/reversible taxonomy, the importance / security-sender
    auto-archive guard (#2426), and explicit user preferences.
    """

    def _explicitly_preferred(
        self,
        *,
        action_type: str,
        category: str,
        sender: str,
        preferences: Optional[Mapping[str, Any]],
    ) -> bool:
        """True when the user's own preferences already sanction this action.

        An explicit preference is a direct instruction, so it grants autonomy
        immediately without waiting on the ledger:

        - ``archive`` of a low-priority sender or a category defaulted to
          ``archive`` (from ``preference_tools``).
        """
        if not preferences:
            return False
        if action_type == "archive":
            low = preferences.get("low_priority_senders") or ()
            if sender and sender.strip().lower() in {str(s).lower() for s in low}:
                return True
            defaults = preferences.get("category_defaults") or {}
            cat = (category or "").strip().upper()
            if str(defaults.get(cat, "")).lower() == "archive":
                return True
        return False

    def decide(
        self,
        *,
        tool: str,
        action_type: str,
        category: str = "",
        sender: str = "",
        db: Any = None,
        preferences: Optional[Mapping[str, Any]] = None,
        is_important: bool = False,
    ) -> TrustDecision:
        """Return the disposition for one candidate action.

        ``tool`` is the tool name (checked against the floor); ``action_type``
        is the taxonomy key (``archive``, ``draft_reply``, …). ``is_important``
        is the provider's importance flag (Gmail ``IMPORTANT`` label / Outlook
        high-importance): when set, an ``archive`` candidate is downgraded to a
        proposal rather than auto-executed (#2426), at every autonomy level.
        """
        # 1. Inviolable floor — no level, trust score, or preference lowers it.
        floor = self.floor_decision(tool)
        if floor is not None:
            return floor

        # 2. Loop disabled.
        if self.level == LEVEL_OFF:
            return TrustDecision("suggest", reason="autonomy is off", confidence=0.0)

        # 3. Reply composition is always a draft — safe to write, never to send.
        if action_type in DRAFT_ACTIONS:
            return TrustDecision(
                "draft",
                reason="reply drafted for your review; sending needs confirmation",
                confidence=0.0,
            )

        # 4. Only reversible actions are candidates for auto-execution.
        if action_type not in REVERSIBLE_AUTO_ACTIONS:
            return TrustDecision(
                "suggest",
                reason=f"{action_type} is not an auto-eligible reversible action",
                confidence=0.0,
            )

        # 5. suggest level never auto-executes.
        if self.level == LEVEL_SUGGEST:
            return TrustDecision(
                "suggest", reason="autonomy level is suggest-only", confidence=0.0
            )

        # 5.5 Importance / security-sender guard (#2426): never auto-archive,
        # unattended, a message the provider marked IMPORTANT or one from an
        # account-security / notification sender — a mis-categorized security
        # alert must be PROPOSED, not silently archived. One-directional, like
        # the send/delete confirm-floor: a higher level or a fully-trusted scope
        # can never override it. Scoped to ``archive`` (the visibility-removing
        # action, and the only action the candidate map emits); widen this if
        # the candidate map grows other visibility-affecting actions.
        if action_type == "archive" and (is_important or is_security_sender(sender)):
            why = (
                "provider-flagged IMPORTANT"
                if is_important
                else "account-security / notification sender"
            )
            return TrustDecision(
                "suggest",
                reason=f"{why} — proposed for your review, not auto-archived",
                confidence=0.0,
            )

        # 6. full level auto-executes every reversible action.
        if self.level == LEVEL_FULL:
            return TrustDecision(
                "auto",
                reason="autonomy level is full — reversible action auto-executed",
                confidence=1.0,
            )

        # 7. earn_trust: auto only when explicitly preferred OR ledger-proven.
        scope_sender = sender_scope(sender) if sender else ""
        scope_cat = category_scope(category) if category else ""

        if self._explicitly_preferred(
            action_type=action_type,
            category=category,
            sender=sender,
            preferences=preferences,
        ):
            return TrustDecision(
                "auto",
                reason="you set an explicit preference for this",
                confidence=1.0,
            )

        if db is not None:
            proven = self.trusted_decision(
                db, action_type=action_type, scopes=(scope_sender, scope_cat)
            )
            if proven is not None:
                return proven

        # Not yet trusted — propose and learn from the answer.
        best = 0.0
        if db is not None:
            best = self.best_score(
                db, action_type=action_type, scopes=(scope_sender, scope_cat)
            )
        return TrustDecision(
            "suggest",
            reason="not yet proven for this sender/category — learning from your "
            "choice",
            confidence=best,
        )
