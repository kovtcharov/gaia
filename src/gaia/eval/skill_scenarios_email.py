# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Behaviour scenarios for the six skills bundled with the email agent.

One :class:`~gaia.eval.skill_behavior.SkillScenario` per skill under
``hub/agents/email/python/gaia_agent_email/skills/``, plus the agent factory
they run on. A unit test holds the one-scenario-per-shipped-skill line, so a new
bundled skill without a behaviour scenario is a red test rather than a silently
unvalidated capability.

Every scenario is hermetic. The mailbox is a ``.mbox`` file written into the
repeat's workspace and loaded by ``FakeGmailBackend`` — the documented eval seam
(``EmailAgentConfig(gmail_backend=...)``) that ``gaia.eval.benchmark`` and
``gaia.eval.action_item_quality`` already drive. Nothing here reaches Gmail, a
keyring, or the network.

The evidence discipline is the harness's:

- Every seeded message carries ``context.token`` — eight hex characters minted
  this repeat — in the place the skill has to find it: an action-item sentence,
  an urgent subject line, a booking reference, a calendar block.
- A **read** skill passes only when the token comes *back* out of a tool result
  (``ledger.result_mentions``). The model cannot have produced it from context,
  from cache, or from training data.
- A **write** skill (escalation-routing's star/label, newsletter-digest's
  archive) passes only when the mutation is visible in the fake backend's own
  label state, keyed on the planted message id. The agent's prose is never the
  evidence.

Ordering note: the harness calls ``setup`` before it builds the agent, so
``setup`` owns the backend and stashes it in ``context.state`` for both
:func:`build_email_agent` and the side-effect check to read.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from gaia.eval.skill_behavior import (
    ScenarioContext,
    ScenarioUnavailable,
    SkillScenario,
    ToolLedger,
    Verdict,
    sandbox_home,
    verdict_from_side_effect,
)

#: The bundled skills directory, relative to a repo checkout root.
_SKILLS_SUBPATH = ("hub", "agents", "email", "python", "gaia_agent_email", "skills")

_MAILBOX_OWNER = "user@example.com"

#: ``.invalid`` is reserved (RFC 2606), so a seeded Message-ID can never be
#: confused with — or collide with — real mail.
_SEED_DOMAIN = "gaia-skill-eval.invalid"


# ---------------------------------------------------------------------------
# Skill roots
# ---------------------------------------------------------------------------


def email_skill_roots() -> list[Path]:
    """The directory holding the email agent's bundled skills.

    Resolves from the installed ``gaia_agent_email`` package when present, and
    otherwise from a repo checkout — the wheel is not on PyPI yet (#2240), so
    the checkout is the normal case in CI. Raises ``FileNotFoundError`` naming
    what to fix rather than returning an empty list, which would leave every
    scenario unable to find its skill for a non-obvious reason.
    """
    tried: list[str] = []

    try:
        import gaia_agent_email

        packaged = Path(gaia_agent_email.__file__).resolve().parent / "skills"
        if packaged.is_dir():
            return [packaged]
        tried.append(str(packaged))
    except ImportError:
        pass

    for root in _checkout_roots():
        candidate = root.joinpath(*_SKILLS_SUBPATH)
        if candidate.is_dir():
            return [candidate]
        tried.append(str(candidate))

    raise FileNotFoundError(
        "Could not locate the email agent's bundled skills directory "
        f"({'/'.join(_SKILLS_SUBPATH)}). Run the skill behaviour harness from a "
        "GAIA repo checkout, set GAIA_REPO_ROOT to one, or install the email "
        "agent from source (see gaia.agents.install_hints). Tried: " + ", ".join(tried)
    )


def _checkout_roots() -> list[Path]:
    """Repo-root candidates, most-authoritative first (mirrors fixture_paths)."""
    roots = [Path(__file__).resolve().parents[3]]
    env_root = os.environ.get("GAIA_REPO_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser())
    roots.append(Path.cwd())
    return roots


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def build_email_agent(*, context: ScenarioContext, skill_roots, max_steps: int):
    """The email agent bound to this repeat's fake mailbox.

    ``skill_roots`` is accepted for the harness's builder signature; the agent
    discovers its own bundled ``skills/`` folder through ``SKILL_DIRS``, and the
    harness's ``load_only`` pins exactly one of them afterwards.

    Raises :class:`ScenarioUnavailable` — never returns ``None``, never falls
    back to a live backend — when the agent package is absent or the scenario's
    ``setup`` did not stash a backend.
    """
    try:
        from gaia_agent_email.agent import EmailTriageAgent
        from gaia_agent_email.config import EmailAgentConfig
    except ImportError as exc:
        from gaia.agents.install_hints import agent_not_installed_message

        raise ScenarioUnavailable(
            agent_not_installed_message(
                "Email skill behaviour validation needs the email agent",
                "gaia-agent-email",
                next_step=(
                    "Then re-run `python -m gaia.eval.skill_behavior`. Until it "
                    "imports, all six email skills stay BLOCKED — which is not a "
                    "pass. Source lives in hub/agents/email/python/. Original "
                    f"import error: {exc}"
                ),
            )
        ) from exc

    backend = context.state.get("gmail_backend")
    if backend is None:
        raise ScenarioUnavailable(
            f"Scenario '{context.skill}' did not seed a mailbox: "
            "context.state['gmail_backend'] is unset, so the agent would bind to "
            "the live Gmail connector instead of the fake. Its setup() must call "
            "gaia.eval.skill_scenarios_email._seed_mailbox before the agent is "
            "built — see EMAIL_SKILL_SCENARIOS in that module."
        )

    # Autoload off on a throwaway subclass so the harness's load_only leaves
    # exactly the skill under test loaded; mutating the real class would leak.
    class _HarnessEmailAgent(EmailTriageAgent):
        AUTOLOAD_DECLARED_SKILLS = False

    sandbox_home(context)

    return _HarnessEmailAgent(
        config=EmailAgentConfig(
            max_steps=max_steps,
            silent_mode=True,
            gmail_backend=backend,
            mail_provider="google",
            calendar_backend=context.state.get("calendar_backend"),
            db_path=str(context.home / "email_state.db"),
            memory_db_path=str(context.home / "email_memory.db"),
            # Learned preferences would make repeat N depend on repeat N-1.
            memory_enabled=False,
            output_dir=str(context.workspace),
            use_slm=False,
            start_scheduler=False,
        )
    )


# ---------------------------------------------------------------------------
# Mailbox seeding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Seed:
    """One message to plant. ``key`` names it in ``context.state``."""

    key: str
    subject: str
    sender: str
    body: str
    days_ago: int = 1
    gmail_labels: str = ""
    #: ``key`` of the message this one replies to, for a multi-message thread.
    in_reply_to: str = ""
    extra_headers: Tuple[Tuple[str, str], ...] = ()


def _message_id(context: ScenarioContext, key: str) -> str:
    """Per-repeat unique Message-ID, so ids never repeat across repeats."""
    return f"<{key}-{context.token}@{_SEED_DOMAIN}>"


def _seed_mailbox(context: ScenarioContext, seeds: Sequence[_Seed]) -> Any:
    """Write the ``.mbox``, load a ``FakeGmailBackend``, stash both.

    Populates ``context.state`` with ``gmail_backend``, ``mbox_path``,
    ``message_ids`` (``key -> Gmail message id``) and ``seed_labels`` (the label
    set each message started with, so a check can prove a label was *added*).
    """
    try:
        from tests.fixtures.email.fake_gmail import FakeGmailBackend
    except ImportError as exc:
        raise ScenarioUnavailable(
            "Email skill behaviour validation drives the FakeGmailBackend in "
            "tests/fixtures/email/fake_gmail.py, which ships only in a GAIA repo "
            "checkout. Run the harness from the repo root, or set GAIA_REPO_ROOT "
            f"to one. Original import error: {exc}"
        ) from exc

    mbox_path = Path(context.workspace) / f"{context.skill}-{context.run}.mbox"
    mbox_path.parent.mkdir(parents=True, exist_ok=True)
    mbox_path.write_text(_render_mbox(context, seeds), encoding="utf-8")
    context.state["mbox_path"] = str(mbox_path)

    backend = FakeGmailBackend(mbox_path, user_email=_MAILBOX_OWNER)
    ids = _resolve_ids(context, backend, [s.key for s in seeds])

    context.state["gmail_backend"] = backend
    context.state["message_ids"] = ids
    context.state["seed_labels"] = {
        key: frozenset(backend.get_message(mid).get("labelIds", ()))
        for key, mid in ids.items()
    }
    return backend


def _render_mbox(context: ScenarioContext, seeds: Sequence[_Seed]) -> str:
    """Serialize the seeds as an mbox."""
    now = datetime.now(timezone.utc)
    chunks: list[str] = []
    for seed in seeds:
        msg = EmailMessage()
        msg["Message-ID"] = _message_id(context, seed.key)
        msg["Subject"] = seed.subject
        msg["From"] = seed.sender
        msg["To"] = _MAILBOX_OWNER
        msg["Date"] = format_datetime(now - timedelta(days=seed.days_ago))
        if seed.gmail_labels:
            msg["X-Gmail-Labels"] = seed.gmail_labels
        if seed.in_reply_to:
            parent = _message_id(context, seed.in_reply_to)
            msg["In-Reply-To"] = parent
            msg["References"] = parent
        for name, value in seed.extra_headers:
            msg[name] = value
        msg.set_content(seed.body)
        # ">From " escaping stops a body line being read as the next message.
        rendered = msg.as_string().replace("\nFrom ", "\n>From ")
        chunks.append(
            f"From {seed.key}@{_SEED_DOMAIN} Thu Jan  1 00:00:00 2026\n{rendered}"
        )
    return "\n".join(chunks) + "\n"


def _resolve_ids(
    context: ScenarioContext, backend: Any, keys: Iterable[str]
) -> Dict[str, str]:
    """Map each seed ``key`` to the Gmail id the fake derived for it.

    Matched through the backend's public read surface on the seed's Message-ID
    rather than re-deriving the fake's own id hash, which would rot silently the
    day that derivation changes.
    """
    wanted = {key: _message_id(context, key) for key in keys}
    found: Dict[str, str] = {}
    listing = backend.list_messages(label_ids=["INBOX"], max_results=100)
    for stub in listing.get("messages", []) or []:
        raw = json.dumps(backend.get_message(stub["id"]), default=str)
        for key, header in wanted.items():
            if header in raw:
                found[key] = stub["id"]
    missing = sorted(key for key in wanted if key not in found)
    if missing:
        raise ScenarioUnavailable(
            f"Scenario '{context.skill}' seeded {missing} but FakeGmailBackend "
            f"did not load them from {context.state.get('mbox_path')}. The mbox "
            "is malformed, or the messages landed outside INBOX; inspect that "
            "file, then _render_mbox in gaia/eval/skill_scenarios_email.py."
        )
    return found


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------


def _labels_now(context: ScenarioContext, key: str) -> frozenset:
    """The fake backend's CURRENT label set for a planted message."""
    backend = context.state.get("gmail_backend")
    if backend is None:
        raise ScenarioUnavailable(
            f"Scenario '{context.skill}' has no stashed backend to inspect; its "
            "setup() must call _seed_mailbox."
        )
    return frozenset(
        backend.get_message(context.state["message_ids"][key]).get("labelIds", ())
    )


def _labels_at_seed(context: ScenarioContext, key: str) -> frozenset:
    return frozenset(context.state["seed_labels"][key])


def _token_came_back(
    context: ScenarioContext, ledger: ToolLedger, tools: Sequence[str]
) -> bool:
    """True when the planted token appears in one of ``tools``' RESULTS."""
    return any(ledger.result_mentions(context.token, tool=tool) for tool in tools)


# ---------------------------------------------------------------------------
# action-item-extraction
# ---------------------------------------------------------------------------

_ACTION_ITEM_READ_TOOLS = ("extract_action_items", "get_thread", "summarize_thread")


def _action_item_setup(context: ScenarioContext) -> None:
    token = context.token
    _seed_mailbox(
        context,
        [
            _Seed(
                key="budget-open",
                subject="Q3 budget review",
                sender="priya@example.com",
                body=(
                    "We walked through the Q3 numbers today. Nothing is decided "
                    "yet on the contractor line."
                ),
                days_ago=4,
            ),
            _Seed(
                key="budget-ask",
                subject=f"Re: Q3 budget review (ref GX-{token})",
                sender="priya@example.com",
                in_reply_to="budget-open",
                body=(
                    f"Please send the signed budget sheet ref GX-{token} by "
                    "Friday so finance can close the quarter. I will circulate "
                    "the revised headcount plan on Monday."
                ),
                days_ago=1,
            ),
            _Seed(
                key="noise",
                subject="Coffee machine is fixed",
                sender="facilities@example.com",
                body="The machine on floor 3 is working again.",
                days_ago=2,
            ),
        ],
    )


def _action_item_prompt(context: ScenarioContext) -> str:
    return (
        "What do I still owe people from my inbox, and what am I owed? One "
        f"thread is tagged ref GX-{context.token} — whatever it commits me to "
        "must be in the list, with the owner and the date."
    )


def _action_item_check(
    context: ScenarioContext, reply: str, ledger: ToolLedger
) -> Verdict:
    return verdict_from_side_effect(
        _token_came_back(context, ledger, _ACTION_ITEM_READ_TOOLS), reply
    )


# ---------------------------------------------------------------------------
# escalation-routing
# ---------------------------------------------------------------------------


def _escalation_setup(context: ScenarioContext) -> None:
    token = context.token
    _seed_mailbox(
        context,
        [
            _Seed(
                key="escalation",
                subject=f"Countersignature blocks Monday go-live (ref GX-{token})",
                sender="legal@bigcustomer.example.com",
                body=(
                    "Procurement cannot release the purchase order until the "
                    "amended contract is countersigned. If it is not signed by "
                    "Monday morning the go-live slips a quarter. Quote ref "
                    f"GX-{token} on the signature page."
                ),
                days_ago=1,
            ),
            _Seed(
                key="fyi",
                subject="Weekly engineering notes",
                sender="eng-updates@example.com",
                body="Sprint 42 closed. No action needed.",
                days_ago=2,
            ),
            _Seed(
                key="someone-else",
                subject="Badge reader on floor 2 is broken",
                sender="reception@example.com",
                body="Reporting a broken badge reader; facilities owns this one.",
                days_ago=3,
            ),
        ],
    )


def _escalation_prompt(context: ScenarioContext) -> str:
    return (
        "What in my inbox actually needs me today, and what belongs to someone "
        f"else? The message tagged ref GX-{context.token} is in there. Mark "
        "whatever you route as urgent so the ordering outlives this "
        "conversation."
    )


def _escalation_check(
    context: ScenarioContext, reply: str, ledger: ToolLedger
) -> Verdict:
    before = _labels_at_seed(context, "escalation")
    after = _labels_now(context, "escalation")
    return verdict_from_side_effect(bool(after - before), reply)


# ---------------------------------------------------------------------------
# inbox-triage
# ---------------------------------------------------------------------------

_TRIAGE_READ_TOOLS = ("triage_inbox", "pre_scan_inbox", "get_message")


def _inbox_triage_setup(context: ScenarioContext) -> None:
    token = context.token
    _seed_mailbox(
        context,
        [
            _Seed(
                key="needs-reply",
                subject=f"Can you approve the vendor SOW? (ref GX-{token})",
                sender="dana@example.com",
                body=(
                    "Could you confirm the statement of work before Thursday? "
                    f"The vendor is holding the slot under ref GX-{token}."
                ),
                days_ago=1,
            ),
            _Seed(
                key="needs-decision",
                subject="Two options for the offsite venue",
                sender="ops@example.com",
                body="Venue A is cheaper, venue B is closer. Which one?",
                days_ago=2,
            ),
            _Seed(
                key="fyi",
                subject="Payroll calendar published",
                sender="hr@example.com",
                body="Next year's payroll calendar is on the intranet.",
                days_ago=3,
            ),
            _Seed(
                key="noise",
                subject="50% off standing desks this week only",
                sender="deals@shop.example.com",
                gmail_labels="Promotions",
                body="Limited time offer on every standing desk.",
                days_ago=2,
            ),
        ],
    )


def _inbox_triage_prompt(context: ScenarioContext) -> str:
    return (
        "Triage my inbox and tell me how many things actually need me. The "
        f"message tagged ref GX-{context.token} is somewhere in there — I want "
        "to know which bucket it landed in."
    )


def _inbox_triage_check(
    context: ScenarioContext, reply: str, ledger: ToolLedger
) -> Verdict:
    return verdict_from_side_effect(
        _token_came_back(context, ledger, _TRIAGE_READ_TOOLS), reply
    )


# ---------------------------------------------------------------------------
# meeting-scheduling
# ---------------------------------------------------------------------------

_MEETING_CALENDAR_TOOLS = ("list_calendar_events", "detect_calendar_conflicts")


def _meeting_setup(context: ScenarioContext) -> None:
    try:
        from tests.fixtures.email.fake_gmail import FakeCalendarBackend
    except ImportError as exc:
        raise ScenarioUnavailable(
            "meeting-scheduling needs the FakeCalendarBackend in "
            "tests/fixtures/email/fake_gmail.py, which ships only in a GAIA repo "
            f"checkout. Run the harness from the repo root. Import error: {exc}"
        ) from exc

    token = context.token
    _seed_mailbox(
        context,
        [
            _Seed(
                key="invite",
                subject=f"Design review Thursday 14:00 UTC? (ref GX-{token})",
                sender="marco@example.com",
                body=(
                    "Can we do the design review on Thursday at 14:00 UTC? It "
                    "should take an hour. Let me know if that works for you."
                ),
                days_ago=1,
            ),
            _Seed(
                key="noise",
                subject="Parking permit renewal",
                sender="facilities@example.com",
                body="Permits renew automatically this year.",
                days_ago=3,
            ),
        ],
    )

    slot = (datetime.now(timezone.utc) + timedelta(days=2)).replace(
        hour=14, minute=0, second=0, microsecond=0
    )
    calendar = FakeCalendarBackend()
    calendar.events["evt-planted"] = {
        "id": "evt-planted",
        "summary": f"Board prep — hold GX-{token}",
        "start": {"dateTime": slot.isoformat()},
        "end": {"dateTime": (slot + timedelta(hours=1)).isoformat()},
        "attendees": [{"email": _MAILBOX_OWNER, "responseStatus": "accepted"}],
    }
    context.state["calendar_backend"] = calendar


def _meeting_prompt(context: ScenarioContext) -> str:
    return (
        "Someone emailed asking to meet Thursday afternoon. Check my calendar "
        f"before answering — there is already a block tagged GX-{context.token} "
        "that week — then tell me yes, no, or another time."
    )


def _meeting_check(context: ScenarioContext, reply: str, ledger: ToolLedger) -> Verdict:
    return verdict_from_side_effect(
        _token_came_back(context, ledger, _MEETING_CALENDAR_TOOLS), reply
    )


# ---------------------------------------------------------------------------
# newsletter-digest
# ---------------------------------------------------------------------------

_UNSUBSCRIBE = ("List-Unsubscribe", "<https://example.com/unsubscribe>")


def _newsletter_setup(context: ScenarioContext) -> None:
    token = context.token
    _seed_mailbox(
        context,
        [
            _Seed(
                key="planted-newsletter",
                subject=f"The Weekly Byte — issue GX-{token}",
                sender="news@weeklybyte.example.com",
                gmail_labels="Promotions",
                extra_headers=(_UNSUBSCRIBE,),
                body=(
                    f"Issue GX-{token}: three pieces on local inference, one on "
                    "NPU scheduling, and a reader survey closing Friday."
                ),
                days_ago=1,
            ),
            _Seed(
                key="newsletter-2",
                subject="Dev Digest — April roundup",
                sender="digest@devdigest.example.com",
                gmail_labels="Promotions",
                extra_headers=(_UNSUBSCRIBE,),
                body="Six links about build systems.",
                days_ago=2,
            ),
            _Seed(
                key="newsletter-3",
                subject="Your monthly hosting summary",
                sender="billing@hosting.example.com",
                gmail_labels="Updates",
                extra_headers=(_UNSUBSCRIBE,),
                body="Usage was flat. The invoice follows next week.",
                days_ago=3,
            ),
            _Seed(
                key="real-mail",
                subject="Lunch Thursday?",
                sender="sam@example.com",
                body="Are you around Thursday?",
                days_ago=1,
            ),
        ],
    )


def _newsletter_prompt(context: ScenarioContext) -> str:
    return (
        "Digest the newsletters and bulk mail sitting in my inbox — issue "
        f"GX-{context.token} is one of them — then clear them out of the inbox."
    )


def _newsletter_check(
    context: ScenarioContext, reply: str, ledger: ToolLedger
) -> Verdict:
    before = _labels_at_seed(context, "planted-newsletter")
    after = _labels_now(context, "planted-newsletter")
    archived = "INBOX" in before and "INBOX" not in after
    return verdict_from_side_effect(archived or bool(after - before), reply)


# ---------------------------------------------------------------------------
# travel-itinerary
# ---------------------------------------------------------------------------

_TRAVEL_READ_TOOLS = ("get_message", "summarize_message", "search_messages")


def _travel_setup(context: ScenarioContext) -> None:
    token = context.token
    _seed_mailbox(
        context,
        [
            _Seed(
                key="flight",
                subject=f"Your flight is confirmed — booking GX-{token}",
                sender="noreply@airline.example.com",
                body=(
                    f"Booking reference GX-{token}. AC842 departs Toronto YYZ at "
                    "08:40 on Tue 14 Mar and arrives Austin AUS at 11:55 local "
                    "time. One checked bag is included."
                ),
                days_ago=5,
            ),
            _Seed(
                key="hotel",
                subject="Reservation confirmed: Austin, 14-17 Mar",
                sender="reservations@hotel.example.com",
                body=(
                    "Check-in Tue 14 Mar from 15:00, check-out Fri 17 Mar. "
                    "Confirmation HT-55120."
                ),
                days_ago=5,
            ),
            _Seed(
                key="noise",
                subject="Team retro notes",
                sender="scrum@example.com",
                body="Notes from the retro are in the wiki.",
                days_ago=2,
            ),
        ],
    )


def _travel_prompt(context: ScenarioContext) -> str:
    return (
        "Build me one itinerary for the Austin trip out of whatever "
        f"confirmations are in my mail. Booking GX-{context.token} is one of the "
        "legs — give me each segment in time order and name what is missing."
    )


def _travel_check(context: ScenarioContext, reply: str, ledger: ToolLedger) -> Verdict:
    return verdict_from_side_effect(
        _token_came_back(context, ledger, _TRAVEL_READ_TOOLS), reply
    )


# ---------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------


EMAIL_SKILL_SCENARIOS: Tuple[SkillScenario, ...] = (
    SkillScenario(
        skill="action-item-extraction",
        agent="email",
        setup=_action_item_setup,
        prompt_factory=_action_item_prompt,
        expect_tools=("extract_action_items", "get_thread"),
        side_effect_check=_action_item_check,
    ),
    SkillScenario(
        skill="escalation-routing",
        agent="email",
        setup=_escalation_setup,
        prompt_factory=_escalation_prompt,
        expect_tools=("triage_inbox", "add_star", "label_message"),
        side_effect_check=_escalation_check,
    ),
    SkillScenario(
        skill="inbox-triage",
        agent="email",
        setup=_inbox_triage_setup,
        prompt_factory=_inbox_triage_prompt,
        expect_tools=("triage_inbox", "pre_scan_inbox"),
        side_effect_check=_inbox_triage_check,
    ),
    SkillScenario(
        skill="meeting-scheduling",
        agent="email",
        setup=_meeting_setup,
        prompt_factory=_meeting_prompt,
        expect_tools=("detect_meeting_request", "list_calendar_events"),
        side_effect_check=_meeting_check,
    ),
    SkillScenario(
        skill="newsletter-digest",
        agent="email",
        setup=_newsletter_setup,
        prompt_factory=_newsletter_prompt,
        expect_tools=("search_messages", "archive_message_batch"),
        side_effect_check=_newsletter_check,
    ),
    SkillScenario(
        skill="travel-itinerary",
        agent="email",
        setup=_travel_setup,
        prompt_factory=_travel_prompt,
        expect_tools=("search_messages", "get_message"),
        side_effect_check=_travel_check,
    ),
)
