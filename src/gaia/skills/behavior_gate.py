# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
The pre-publish **behaviour** gate: only validated skills ship.

:mod:`gaia.skills.audit_gate` answers "is this skill safe to run". This one
answers the question nobody was asking before: **does it actually work.** A skill
is a set of instructions that is supposed to make an agent call particular tools.
Nothing in the static audit executes anything, so a skill whose body the model
silently ignores — the #1428 failure class, a confident reply with no tool call
behind it — passes every existing check and reaches users broken.

So publish requires a *record*: proof that
:mod:`gaia.eval.skill_behavior` drove this skill against a real model and watched
its tools leave a real side effect.

**The record is bound to the bytes it was earned on.** Each entry carries the
skill's ``content_digest`` and ``version``, computed by the same
:func:`gaia.skills.audit.content_digest` the audit gate uses. Edit the skill and
the digest moves, the record goes stale, and the gate refuses until the harness
runs again. That is what makes "re-validate on every change" enforceable rather
than a habit — you cannot forget, because forgetting is indistinguishable from
never having done it.

**There is no bypass flag, deliberately.** Not "unvalidated publishes with a
warning", not "experimental is exempt". A missing record is a hard refusal with
an actionable message, because the alternative — treating "no record" as "fine" —
is exactly the silent fallback CLAUDE.md prohibits, and it is how every unvalidated
skill would ship.

Where a record comes from, in resolution order:

1. ``--behavior-report <file>`` — one a CI job or a third-party author produced.
2. ``$GAIA_SKILL_BEHAVIOR_MANIFEST`` — a manifest elsewhere on disk.
3. The manifest shipped with GAIA (:data:`DEFAULT_MANIFEST_PATH`), written by
   ``.github/workflows/test_skill_behavior_e2e.yml`` on the Strix Halo runner.

Produce one with::

    python -m gaia.eval.skill_behavior --skill <name> --output records.json

Check one without publishing::

    python -m gaia.skills.behavior_gate --skill-dir hub/skills/rss-digest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from gaia.logger import get_logger
from gaia.skills.errors import SkillError

log = get_logger(__name__)

#: Manifest schema version. A record written by an older schema is not read as a
#: pass — the evidence rules may have changed underneath it.
MANIFEST_SCHEMA = "gaia-skill-behavior-manifest/1"

#: The manifest shipped inside the wheel.
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "data" / ("skill_behavior_validation.json")
)

#: Points the gate at a manifest built elsewhere (a CI artifact, say).
MANIFEST_ENV = "GAIA_SKILL_BEHAVIOR_MANIFEST"

BEHAVIOR_DOCS_URL = "https://amd-gaia.ai/docs/plans/skill-format"

STATUS_VALIDATED = "validated"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"
STATUS_UNVALIDATED = "unvalidated"
VALID_STATUSES = frozenset(
    {STATUS_VALIDATED, STATUS_FAILED, STATUS_BLOCKED, STATUS_UNVALIDATED}
)

_HOW_TO_VALIDATE = (
    "Run the behaviour harness on a machine with a live Lemonade backend:\n"
    "    python -m gaia.eval.skill_behavior --skill {skill} --output records.json\n"
    "then publish with '--behavior-report records.json', or land the refreshed "
    "manifest from the 'Skill Behavior E2E' workflow "
    "(.github/workflows/test_skill_behavior_e2e.yml)."
)


class SkillBehaviorError(SkillError):
    """Base for every behaviour-gate refusal."""


class SkillNotValidatedError(SkillBehaviorError):
    """No behaviour record exists for this skill, so nothing proves it works."""


class SkillBehaviorStaleError(SkillBehaviorError):
    """A record exists but describes different bytes or a different version."""


class SkillBehaviorFailedError(SkillBehaviorError):
    """The harness ran and the skill did not clear the bar (or could not run)."""


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviorRecord:
    """One skill's validation outcome, bound to the artifact that earned it."""

    skill: str
    status: str
    version: str = ""
    content_digest: str = ""
    harness: str = ""
    validated_at: str = ""
    recorded_at: str = ""
    reason: str = ""
    counts: Dict[str, int] = None  # type: ignore[assignment]
    hard_fail: bool = False

    def __post_init__(self) -> None:
        if self.counts is None:
            object.__setattr__(self, "counts", {})

    @property
    def validated(self) -> bool:
        return self.status == STATUS_VALIDATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "status": self.status,
            "reason": self.reason,
            "version": self.version,
            "content_digest": self.content_digest,
            "harness": self.harness,
            "validated_at": self.validated_at,
            "recorded_at": self.recorded_at,
            "counts": dict(self.counts or {}),
            "hard_fail": self.hard_fail,
        }

    @classmethod
    def from_dict(cls, data: Any, *, where: str) -> "BehaviorRecord":
        """Parse one entry, refusing anything the gate cannot read.

        An unreadable record is an error, never an implied pass — the same rule
        :class:`gaia.skills.audit_gate.AuditReport` applies to audit reports.
        """
        if not isinstance(data, dict):
            raise SkillNotValidatedError(
                f"Behaviour record in {where} is a {type(data).__name__}, not an "
                "object. A record GAIA cannot read is not a record — regenerate "
                "it with 'python -m gaia.eval.skill_behavior'."
            )
        skill = str(data.get("skill") or "").strip()
        if not skill:
            raise SkillNotValidatedError(
                f"Behaviour record in {where} names no skill, so it cannot be "
                "matched to what is being published. Regenerate it with "
                "'python -m gaia.eval.skill_behavior'."
            )
        status = str(data.get("status") or "").strip().lower()
        if status not in VALID_STATUSES:
            raise SkillNotValidatedError(
                f"Behaviour record for '{skill}' in {where} has status "
                f"{status!r}, which is not one of {sorted(VALID_STATUSES)}. An "
                "unrecognised status is refused rather than assumed to pass."
            )
        counts = data.get("counts") or {}
        if not isinstance(counts, dict):
            counts = {}
        return cls(
            skill=skill,
            status=status,
            version=str(data.get("version") or ""),
            content_digest=str(data.get("content_digest") or ""),
            harness=str(data.get("harness") or ""),
            validated_at=str(data.get("validated_at") or ""),
            recorded_at=str(data.get("recorded_at") or ""),
            reason=str(data.get("reason") or ""),
            counts={str(k): int(v) for k, v in counts.items()},
            hard_fail=bool(data.get("hard_fail")),
        )


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def resolve_manifest_path(explicit: Optional[Path] = None) -> Path:
    """The manifest this process should read, by the documented precedence."""
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get(MANIFEST_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_MANIFEST_PATH


def load_manifest(path: Optional[Path] = None) -> Dict[str, BehaviorRecord]:
    """Read a manifest into ``{skill: record}``.

    Raises:
        SkillNotValidatedError: the file is missing, unparseable, or built by a
            schema this GAIA does not understand. Each case names the fix.
    """
    manifest_path = resolve_manifest_path(path)
    if not manifest_path.is_file():
        raise SkillNotValidatedError(
            f"No skill behaviour manifest at {manifest_path}, so no skill can be "
            "shown to have been validated and none may be published. Point "
            f"${MANIFEST_ENV} at a manifest, pass '--behavior-report <file>', or "
            "restore the one shipped with GAIA "
            f"({DEFAULT_MANIFEST_PATH.name}). See {BEHAVIOR_DOCS_URL}"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SkillNotValidatedError(
            f"Could not read the skill behaviour manifest at {manifest_path}: "
            f"{exc}. A manifest GAIA cannot parse is not evidence of validation, "
            "so publish is refused. Regenerate it with "
            "'python -m gaia.eval.skill_behavior --output <file>'."
        ) from exc

    schema = str(raw.get("schema") or "")
    if schema != MANIFEST_SCHEMA:
        raise SkillNotValidatedError(
            f"The behaviour manifest at {manifest_path} declares schema "
            f"{schema!r}, but this GAIA reads {MANIFEST_SCHEMA!r}. The evidence "
            "rules may have changed, so its records are not accepted. Re-run "
            "'python -m gaia.eval.skill_behavior' to produce a current one."
        )

    entries = raw.get("skills") or {}
    if not isinstance(entries, dict):
        raise SkillNotValidatedError(
            f"The behaviour manifest at {manifest_path} has a 'skills' field that "
            "is not an object, so no record can be looked up. Regenerate it."
        )
    return {
        name: BehaviorRecord.from_dict(entry, where=str(manifest_path))
        for name, entry in entries.items()
    }


def load_behavior_report(path: Path) -> BehaviorRecord:
    """Read a single-skill record, or a one-skill manifest, from ``path``."""
    report_path = Path(path)
    if not report_path.is_file():
        raise SkillNotValidatedError(
            f"No behaviour report at {report_path}. Produce one with "
            "'python -m gaia.eval.skill_behavior --skill <name> --output "
            f"{report_path.name}', or drop the flag to use the shipped manifest."
        )
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SkillNotValidatedError(
            f"Could not read the behaviour report at {report_path}: {exc}. "
            "Publish is refused — an unreadable report is not a passing one."
        ) from exc

    if isinstance(raw, dict) and "skills" in raw:
        records = load_manifest(report_path)
        if len(records) != 1:
            raise SkillNotValidatedError(
                f"{report_path} carries {len(records)} records; "
                "'--behavior-report' takes a report for exactly one skill. Use "
                f"${MANIFEST_ENV} to point at a multi-skill manifest instead."
            )
        return next(iter(records.values()))
    return BehaviorRecord.from_dict(raw, where=str(report_path))


def write_manifest(path: Path, results: Iterable[Any]) -> Path:
    """Write ``results`` (``SkillResult`` objects) as a manifest at ``path``.

    Each record is stamped with the skill's live ``content_digest`` and version,
    so the gate can later tell whether the bytes still match what was validated.
    """
    from gaia.eval.skill_scenarios import shipped_skill_dirs
    from gaia.skills.audit.findings import content_digest
    from gaia.skills.format import parse_skill_metadata

    directories = {d.name: d for d in shipped_skill_dirs()}
    skills: Dict[str, Any] = {}
    for result in results:
        directory = directories.get(result.skill)
        if directory is None:
            raise SkillNotValidatedError(
                f"Cannot record a result for '{result.skill}': no shipped skill "
                "directory of that name. A record that does not correspond to a "
                "skill on disk cannot be bound to its bytes."
            )
        skill = parse_skill_metadata(directory)
        skills[result.skill] = result.to_record(
            content_digest=content_digest(directory),
            version=str(skill.version or ""),
        )

    payload = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skills": dict(sorted(skills.items())),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %d behaviour record(s) to %s", len(skills), path)
    return path


#: Stamped on an entry that exists only so the coverage check can see the skill.
UNVALIDATED_REASON = (
    "No behaviour-validation run has produced a verdict for this skill yet. It "
    "is covered by a scenario but has not been driven against a real model, so "
    "it is NOT validated and cannot be published. Run the 'Skill Behavior E2E' "
    "workflow on a Strix Halo runner (or the harness locally) to earn a verdict."
)


def sync_manifest(path: Optional[Path] = None) -> Path:
    """Give every shipped skill an entry, adding missing ones as ``unvalidated``.

    Coverage only. It can add an entry and it can re-stamp a stale digest, but it
    can never write ``validated`` — only a real harness run does that, through
    :func:`write_manifest`. Keeping those two paths separate is what stops "the
    manifest was out of date" from being resolvable by re-stamping it green.
    """
    from gaia.eval.skill_scenarios import shipped_skill_dirs
    from gaia.skills.audit.findings import content_digest
    from gaia.skills.format import parse_skill_metadata

    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    existing: Dict[str, BehaviorRecord] = {}
    if manifest_path.is_file():
        existing = load_manifest(manifest_path)

    from gaia.eval.skill_behavior import HARNESS_VERSION

    entries: Dict[str, Any] = {}
    for directory in shipped_skill_dirs():
        skill = parse_skill_metadata(directory)
        digest = content_digest(directory)
        version = str(skill.version or "")
        previous = existing.get(skill.name)
        if (
            previous is not None
            and previous.content_digest == digest
            and previous.version == version
        ):
            entries[skill.name] = previous.to_dict()
            continue
        entries[skill.name] = BehaviorRecord(
            skill=skill.name,
            status=STATUS_UNVALIDATED,
            version=version,
            content_digest=digest,
            harness=HARNESS_VERSION,
            reason=UNVALIDATED_REASON,
        ).to_dict()

    payload = {
        "schema": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skills": dict(sorted(entries.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("Synced %d skill(s) into %s", len(entries), manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def assert_behavior_validated(
    directory: Path,
    *,
    skill_name: str,
    version: str = "",
    record: Optional[BehaviorRecord] = None,
    manifest_path: Optional[Path] = None,
) -> BehaviorRecord:
    """Refuse unless ``skill_name`` has a current, passing behaviour record.

    Args:
        directory: The skill source folder, digested to check the record is
            still about these bytes.
        skill_name: The skill's name, used to look the record up and named in
            every error.
        version: The version being published. A record earned on a different
            version is stale.
        record: A pre-loaded record (from ``--behavior-report``). When omitted,
            the manifest is consulted.
        manifest_path: Manifest override; see :func:`resolve_manifest_path`.

    Raises:
        SkillNotValidatedError: no record at all.
        SkillBehaviorStaleError: the record does not describe this artifact.
        SkillBehaviorFailedError: the record says failed, blocked, or unvalidated.
    """
    from gaia.skills.audit.findings import content_digest

    if record is None:
        records = load_manifest(manifest_path)
        record = records.get(skill_name)
    if record is None:
        raise SkillNotValidatedError(
            f"Skill '{skill_name}' has no behaviour-validation record, so nothing "
            "shows its instructions actually make an agent run its tools. GAIA "
            "publishes only validated skills.\n"
            + _HOW_TO_VALIDATE.format(skill=skill_name)
            + f"\nSee {BEHAVIOR_DOCS_URL}"
        )

    if record.skill != skill_name:
        raise SkillBehaviorStaleError(
            f"The behaviour record supplied for '{skill_name}' is actually for "
            f"'{record.skill}'. A record from a different skill proves nothing "
            "about this one. Re-run the harness against the skill you are "
            "publishing."
        )

    digest = content_digest(directory)
    if record.content_digest and record.content_digest != digest:
        raise SkillBehaviorStaleError(
            f"Skill '{skill_name}' has changed since it was validated: the "
            f"record was earned on {record.content_digest[:19]}… and this "
            f"directory hashes to {digest[:19]}…. A behaviour verdict belongs to "
            "the bytes that earned it, so it does not carry over to an edit.\n"
            + _HOW_TO_VALIDATE.format(skill=skill_name)
        )
    if version and record.version and record.version != version:
        raise SkillBehaviorStaleError(
            f"Skill '{skill_name}' is being published at version {version}, but "
            f"its behaviour record was earned at {record.version}. Every version "
            "re-earns its verdict — re-run the harness.\n"
            + _HOW_TO_VALIDATE.format(skill=skill_name)
        )

    if record.status == STATUS_BLOCKED:
        raise SkillBehaviorFailedError(
            f"Skill '{skill_name}' could not be behaviour-validated and is "
            f"therefore not validated: {record.reason or '(no reason recorded)'} "
            "A skill whose validation was skipped must never be shipped as if it "
            "had passed. Resolve the blocker — usually a missing fixture or an "
            "uninstalled package — and re-run the harness."
        )
    if record.status == STATUS_FAILED:
        raise SkillBehaviorFailedError(
            f"Skill '{skill_name}' FAILED behaviour validation: "
            f"{record.reason or '(no reason recorded)'} Verdict counts: "
            f"{record.counts or '{}'}. Fix the skill body so the agent actually "
            "calls its tools, then re-run the harness."
            + (
                "\nThis was a FALSE SUCCESS — the agent claimed the work was done "
                "while its tools left no trace. That is the worst failure mode, "
                "not a flaky run."
                if record.hard_fail
                else ""
            )
        )
    if record.status != STATUS_VALIDATED:
        raise SkillBehaviorFailedError(
            f"Skill '{skill_name}' is recorded as '{record.status}', not "
            "'validated'. Only a passing behaviour run clears this gate.\n"
            + _HOW_TO_VALIDATE.format(skill=skill_name)
        )
    return record


def gate_for_publish(
    directory: Path,
    *,
    skill_name: str,
    version: str = "",
    report_path: Optional[Path] = None,
) -> BehaviorRecord:
    """The full behaviour gate: obtain a record and refuse unless it passes."""
    record = load_behavior_report(report_path) if report_path is not None else None
    return assert_behavior_validated(
        directory,
        skill_name=skill_name,
        version=version,
        record=record,
    )


# ---------------------------------------------------------------------------
# CLI — the CI-side check
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python -m gaia.skills.behavior_gate`` — check records without publishing.

    Exit 0 when every named skill has a current, passing record; 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Check that skills have current, passing behaviour-validation "
            "records. This is the gate 'gaia skill publish' applies, runnable on "
            "its own so CI can fail a PR before anything is uploaded."
        )
    )
    parser.add_argument(
        "--skill-dir",
        action="append",
        default=None,
        help="Skill directory to check (repeatable). Default: every shipped skill.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"Manifest to read (default: ${MANIFEST_ENV}, else the shipped one)",
    )
    args = parser.parse_args(argv)

    from gaia.eval.skill_scenarios import shipped_skill_dirs
    from gaia.skills.format import parse_skill_metadata

    if args.skill_dir:
        directories = [Path(d) for d in args.skill_dir]
    else:
        directories = shipped_skill_dirs()

    manifest_path = Path(args.manifest) if args.manifest else None
    failures = 0
    for directory in directories:
        skill = parse_skill_metadata(directory)
        try:
            assert_behavior_validated(
                directory,
                skill_name=skill.name,
                version=str(skill.version or ""),
                manifest_path=manifest_path,
            )
        except SkillBehaviorError as exc:
            failures += 1
            print(f"NOT VALIDATED  {skill.name}\n    {exc}\n", file=sys.stderr)
        else:
            print(f"validated      {skill.name}")

    if failures:
        print(
            f"\n{failures} of {len(directories)} skill(s) are not validated and "
            "cannot be published.",
            file=sys.stderr,
        )
        return 1
    print(f"\nAll {len(directories)} skill(s) have current, passing records.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
