// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func fixedClock(t time.Time) func() time.Time { return func() time.Time { return t } }

// Two checks in one process must never overlap — the second is refused, not
// queued, because a silent wait inside a CLI command looks like a hang.
func TestGuardRefusesASecondCheckInThisProcess(t *testing.T) {
	dir := t.TempDir()
	now := fixedClock(time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC))

	first, err := Acquire(dir, now)
	if err != nil {
		t.Fatalf("first acquire: %v", err)
	}
	defer func() { _ = first.Release() }()

	_, err = Acquire(dir, now)
	var busy *BusyError
	if !errors.As(err, &busy) {
		t.Fatalf("a second concurrent acquire returned %s, want a BusyError", fmtErr(err))
	}
	if !strings.Contains(err.Error(), "already running") {
		t.Errorf("the refusal does not say what is happening:\n%v", err)
	}
}

// A lock file written by another live process blocks this one, so two
// gaia-tui processes never download over each other.
func TestGuardRefusesAnotherProcessesLock(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)

	record, err := json.Marshal(lockRecord{PID: 4242, StartedAt: FormatTime(now.Add(-2 * time.Minute))})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	mustWrite(t, filepath.Join(dir, LockFileNameOnDisk), record)

	_, err = Acquire(dir, fixedClock(now))
	var busy *BusyError
	if !errors.As(err, &busy) {
		t.Fatalf("a live lock file returned %s, want a BusyError", fmtErr(err))
	}
	if !strings.Contains(err.Error(), "4242") {
		t.Errorf("the refusal does not name the holding pid:\n%v", err)
	}
}

// A lock left behind by a killed run must not block updates forever — but the
// takeover is reported, because it means a previous update did not finish.
func TestGuardReclaimsAnAbandonedLockAndSaysSo(t *testing.T) {
	dir := t.TempDir()
	now := time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC)

	record, err := json.Marshal(lockRecord{PID: 4242, StartedAt: FormatTime(now.Add(-2 * StaleLockAfter))})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	mustWrite(t, filepath.Join(dir, LockFileNameOnDisk), record)

	guard, err := Acquire(dir, fixedClock(now))
	if err != nil {
		t.Fatalf("acquire over an abandoned lock: %v", err)
	}
	defer func() { _ = guard.Release() }()

	if guard.Reclaimed == "" {
		t.Fatal("an abandoned lock was taken over silently")
	}
	if !strings.Contains(guard.Reclaimed, "4242") {
		t.Errorf("the takeover note does not name the abandoning pid: %q", guard.Reclaimed)
	}
}

// Release removes the file so the next run is not made to wait out the stale
// timeout, and a double release is harmless.
func TestGuardReleaseIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	now := fixedClock(time.Now())

	guard, err := Acquire(dir, now)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	if err := guard.Release(); err != nil {
		t.Fatalf("release: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, LockFileNameOnDisk)); !os.IsNotExist(err) {
		t.Errorf("the lock file survived release (%v)", err)
	}
	if err := guard.Release(); err != nil {
		t.Errorf("a second release errored: %v", err)
	}

	next, err := Acquire(dir, now)
	if err != nil {
		t.Fatalf("acquire after release: %v", err)
	}
	_ = next.Release()
}
