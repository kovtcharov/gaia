# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
The gate that makes "only ship validated skills" a rule rather than an intention.

Every test here is about a way the gate could *fail open*. A gate that refuses a
broken skill is unremarkable; one that quietly lets an unvalidated skill through
because the record was missing, stale, or unparseable is the whole risk. So the
cases below are almost all negative: no record, wrong skill, edited bytes, bumped
version, unreadable file, unknown status, blocked scenario.

The one positive case exists to prove the gate is not simply refusing everything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaia.eval.skill_behavior import HARNESS_VERSION, SkillResult, SkillStatus
from gaia.skills.audit.findings import content_digest
from gaia.skills.behavior_gate import (
    DEFAULT_MANIFEST_PATH,
    MANIFEST_ENV,
    MANIFEST_SCHEMA,
    BehaviorRecord,
    SkillBehaviorFailedError,
    SkillBehaviorStaleError,
    SkillNotValidatedError,
    assert_behavior_validated,
    gate_for_publish,
    load_manifest,
    resolve_manifest_path,
    sync_manifest,
    write_manifest,
)
from gaia.skills.format import parse_skill_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "hub" / "skills" / "rss-digest"


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _record(**overrides) -> dict:
    record = {
        "skill": "rss-digest",
        "status": "validated",
        "reason": "",
        "version": str(parse_skill_metadata(SKILL_DIR).version),
        "content_digest": content_digest(SKILL_DIR),
        "harness": HARNESS_VERSION,
        "validated_at": "2026-08-18T00:00:00+00:00",
        "recorded_at": "2026-08-18T00:00:00+00:00",
        "counts": {"true_success": 3},
        "hard_fail": False,
    }
    record.update(overrides)
    return record


def _manifest(tmp_path: Path, *records: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "generated_at": "2026-08-18T00:00:00+00:00",
                "skills": {r["skill"]: r for r in records},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# The positive case
# ----------------------------------------------------------------------


def test_a_current_passing_record_clears_the_gate(tmp_path):
    manifest = _manifest(tmp_path, _record())
    result = assert_behavior_validated(
        SKILL_DIR,
        skill_name="rss-digest",
        version=str(parse_skill_metadata(SKILL_DIR).version),
        manifest_path=manifest,
    )
    assert result.validated


# ----------------------------------------------------------------------
# Fail-open guards
# ----------------------------------------------------------------------


def test_no_record_at_all_is_refused(tmp_path):
    """The core rule: "we never validated it" must not read as "it's fine"."""
    manifest = _manifest(tmp_path)
    with pytest.raises(SkillNotValidatedError) as excinfo:
        assert_behavior_validated(
            SKILL_DIR, skill_name="rss-digest", manifest_path=manifest
        )
    message = str(excinfo.value)
    assert "rss-digest" in message
    assert (
        "python -m gaia.eval.skill_behavior" in message
    ), "The error must tell the publisher what to run, not just that they lost."


def test_a_missing_manifest_is_refused_rather_than_assumed_empty(tmp_path):
    with pytest.raises(SkillNotValidatedError) as excinfo:
        assert_behavior_validated(
            SKILL_DIR,
            skill_name="rss-digest",
            manifest_path=tmp_path / "nope.json",
        )
    assert MANIFEST_ENV in str(excinfo.value)


def test_an_unparseable_manifest_is_refused(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(SkillNotValidatedError):
        load_manifest(path)


def test_a_manifest_from_a_different_schema_is_refused(tmp_path):
    """Old evidence rules are not evidence under the current ones."""
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema": "gaia-skill-behavior-manifest/0", "skills": {}}),
        encoding="utf-8",
    )
    with pytest.raises(SkillNotValidatedError) as excinfo:
        load_manifest(path)
    assert MANIFEST_SCHEMA in str(excinfo.value)


def test_an_unknown_status_is_refused_not_treated_as_a_pass(tmp_path):
    manifest = _manifest(tmp_path, _record(status="probably-fine"))
    with pytest.raises(SkillNotValidatedError):
        load_manifest(manifest)


@pytest.mark.parametrize("status", ["failed", "blocked", "unvalidated"])
def test_every_non_passing_status_is_refused(tmp_path, status):
    manifest = _manifest(tmp_path, _record(status=status, reason="because."))
    with pytest.raises(SkillBehaviorFailedError) as excinfo:
        assert_behavior_validated(
            SKILL_DIR, skill_name="rss-digest", manifest_path=manifest
        )
    assert "rss-digest" in str(excinfo.value)


def test_a_false_success_failure_is_called_out_as_such(tmp_path):
    manifest = _manifest(
        tmp_path,
        _record(
            status="failed", reason="FALSE SUCCESS in 2 of 3 runs.", hard_fail=True
        ),
    )
    with pytest.raises(SkillBehaviorFailedError) as excinfo:
        assert_behavior_validated(
            SKILL_DIR, skill_name="rss-digest", manifest_path=manifest
        )
    assert "FALSE SUCCESS" in str(excinfo.value)


def test_editing_the_skill_invalidates_its_record(tmp_path):
    """The property that makes re-validation impossible to forget."""
    manifest = _manifest(tmp_path, _record(content_digest="sha256:" + "0" * 64))
    with pytest.raises(SkillBehaviorStaleError) as excinfo:
        assert_behavior_validated(
            SKILL_DIR, skill_name="rss-digest", manifest_path=manifest
        )
    assert "changed since it was validated" in str(excinfo.value)


def test_a_version_bump_re_earns_the_verdict(tmp_path):
    manifest = _manifest(tmp_path, _record(version="0.9.0"))
    with pytest.raises(SkillBehaviorStaleError):
        assert_behavior_validated(
            SKILL_DIR,
            skill_name="rss-digest",
            version="1.0.0",
            manifest_path=manifest,
        )


def test_a_record_for_a_different_skill_proves_nothing(tmp_path):
    record = BehaviorRecord.from_dict(_record(skill="coding"), where="test")
    with pytest.raises(SkillBehaviorStaleError) as excinfo:
        assert_behavior_validated(SKILL_DIR, skill_name="rss-digest", record=record)
    assert "coding" in str(excinfo.value)


# ----------------------------------------------------------------------
# --behavior-report
# ----------------------------------------------------------------------


def test_a_single_skill_report_clears_the_gate(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")
    record = gate_for_publish(SKILL_DIR, skill_name="rss-digest", report_path=path)
    assert record.validated


def test_a_missing_report_file_is_refused(tmp_path):
    with pytest.raises(SkillNotValidatedError):
        gate_for_publish(
            SKILL_DIR, skill_name="rss-digest", report_path=tmp_path / "nope.json"
        )


def test_a_multi_skill_manifest_is_rejected_as_a_single_report(tmp_path):
    """Otherwise the wrong skill's record could be silently picked out of it."""
    manifest = _manifest(tmp_path, _record(), _record(skill="coding"))
    with pytest.raises(SkillNotValidatedError) as excinfo:
        gate_for_publish(SKILL_DIR, skill_name="rss-digest", report_path=manifest)
    assert MANIFEST_ENV in str(excinfo.value)


# ----------------------------------------------------------------------
# Manifest resolution and writing
# ----------------------------------------------------------------------


def test_resolution_order_is_explicit_then_env_then_shipped(tmp_path, monkeypatch):
    monkeypatch.delenv(MANIFEST_ENV, raising=False)
    assert resolve_manifest_path() == DEFAULT_MANIFEST_PATH
    monkeypatch.setenv(MANIFEST_ENV, str(tmp_path / "env.json"))
    assert resolve_manifest_path() == tmp_path / "env.json"
    assert resolve_manifest_path(tmp_path / "explicit.json") == (
        tmp_path / "explicit.json"
    )


def test_sync_cannot_mark_anything_validated(tmp_path):
    """The one thing that must stay impossible without a real run."""
    manifest = sync_manifest(tmp_path / "synced.json")
    records = load_manifest(manifest)
    assert records, "sync produced no records"
    assert not any(r.validated for r in records.values()), (
        "sync_manifest wrote a 'validated' record. Only write_manifest, fed by a "
        "real harness run, may do that — otherwise a red build is fixable by "
        "re-stamping it green."
    )


def test_sync_keeps_an_existing_record_whose_bytes_still_match(tmp_path):
    manifest = _manifest(tmp_path, _record())
    sync_manifest(manifest)
    assert load_manifest(manifest)["rss-digest"].validated


def test_sync_resets_a_record_whose_bytes_moved(tmp_path):
    manifest = _manifest(tmp_path, _record(content_digest="sha256:" + "1" * 64))
    sync_manifest(manifest)
    assert not load_manifest(manifest)["rss-digest"].validated


def test_write_manifest_binds_a_result_to_the_skills_current_bytes(tmp_path):
    path = write_manifest(
        tmp_path / "written.json",
        [SkillResult(skill="rss-digest", status=SkillStatus.validated)],
    )
    record = load_manifest(path)["rss-digest"]
    assert record.validated
    assert record.content_digest == content_digest(SKILL_DIR)
    assert record.harness == HARNESS_VERSION


def test_write_manifest_refuses_a_result_for_a_skill_that_does_not_exist(tmp_path):
    with pytest.raises(SkillNotValidatedError):
        write_manifest(
            tmp_path / "written.json",
            [SkillResult(skill="not-a-skill", status=SkillStatus.validated)],
        )


# ----------------------------------------------------------------------
# The shipped manifest
# ----------------------------------------------------------------------


def test_the_shipped_manifest_covers_every_shipped_skill():
    """A skill absent from the manifest could never be caught as unvalidated."""
    from gaia.eval.skill_scenarios import shipped_skill_dirs

    records = load_manifest(DEFAULT_MANIFEST_PATH)
    missing = sorted({d.name for d in shipped_skill_dirs()} - set(records))
    assert not missing, (
        f"These skills have no entry in the shipped behaviour manifest: {missing}. "
        "Run 'python -c \"from gaia.skills.behavior_gate import sync_manifest; "
        "sync_manifest()\"' and commit the result."
    )


def test_the_shipped_manifest_is_current_for_every_skill():
    """A stale entry would refuse publish for a confusing reason — fail here first."""
    from gaia.eval.skill_scenarios import shipped_skill_dirs

    records = load_manifest(DEFAULT_MANIFEST_PATH)
    stale = []
    for directory in shipped_skill_dirs():
        record = records.get(directory.name)
        if record is None:
            continue
        skill = parse_skill_metadata(directory)
        if record.content_digest != content_digest(directory) or record.version != str(
            skill.version or ""
        ):
            stale.append(directory.name)
    assert not stale, (
        f"These skills changed since their behaviour record was written: {stale}. "
        "Re-run the 'Skill Behavior E2E' workflow to earn a fresh verdict, or "
        "sync_manifest() to reset them to 'unvalidated' — a changed skill is not "
        "a validated one."
    )


def test_publish_refuses_every_skill_the_shipped_manifest_has_not_validated():
    """Proves the policy holds today without needing a live model run.

    Whatever the manifest currently says, the gate's answer must match it exactly:
    validated entries clear, everything else raises. If this ever passes while the
    manifest says 'unvalidated', the gate has failed open.
    """
    from gaia.eval.skill_scenarios import shipped_skill_dirs

    records = load_manifest(DEFAULT_MANIFEST_PATH)
    for directory in shipped_skill_dirs():
        skill = parse_skill_metadata(directory)
        record = records[skill.name]
        if record.validated:
            assert_behavior_validated(
                directory,
                skill_name=skill.name,
                version=str(skill.version or ""),
                manifest_path=DEFAULT_MANIFEST_PATH,
            )
            continue
        with pytest.raises(SkillBehaviorFailedError):
            assert_behavior_validated(
                directory,
                skill_name=skill.name,
                version=str(skill.version or ""),
                manifest_path=DEFAULT_MANIFEST_PATH,
            )
