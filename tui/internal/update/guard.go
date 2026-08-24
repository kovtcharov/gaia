// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// LockFileNameOnDisk is the cross-process guard inside ~/.gaia.
const LockFileNameOnDisk = "update.lock"

// StaleLockAfter is how long a lock file is honoured before it is treated as
// abandoned. Long enough for the sidecar (>100 MB) to download on a slow link;
// short enough that a killed process does not block updates for a day.
const StaleLockAfter = 30 * time.Minute

// inProcess serializes checks inside one binary. Two Bubble Tea commands and a
// CLI subcommand can all reach the updater; none of them may overlap.
var inProcess sync.Mutex

// BusyError is a refused concurrent check. It names who holds the guard so the
// user can act on it rather than guess.
type BusyError struct {
	// Holder describes what is holding it: "this process" or a pid + file.
	Holder string
}

func (e *BusyError) Error() string {
	return fmt.Sprintf(
		"another GAIA update check is already running (%s), so this one was refused. "+
			"Two checks downloading the same binary would race to replace it. Wait for "+
			"it to finish and re-run.", e.Holder)
}

// lockRecord is what the on-disk guard stores, so a refusal can name the pid.
type lockRecord struct {
	PID       int    `json:"pid"`
	StartedAt string `json:"startedAt"`
}

// Guard is a held update lock. Release it exactly once.
type Guard struct {
	path     string
	released bool
	// Reclaimed is set when a lock abandoned by a dead run was cleared to take
	// this one. Reported, never silent — it means a previous update did not
	// finish.
	Reclaimed string
}

// Acquire takes both guards: the in-process mutex and the on-disk lock file.
//
// It never blocks. A caller that cannot have the guard is told so, because a
// silent wait inside a CLI command looks identical to a hung download.
func Acquire(gaiaDir string, now func() time.Time) (*Guard, error) {
	if !inProcess.TryLock() {
		return nil, &BusyError{Holder: "this process is already checking"}
	}
	guard, err := acquireFileLock(gaiaDir, now)
	if err != nil {
		inProcess.Unlock()
		return nil, err
	}
	return guard, nil
}

func acquireFileLock(gaiaDir string, now func() time.Time) (*Guard, error) {
	if err := os.MkdirAll(gaiaDir, 0o700); err != nil {
		return nil, fmt.Errorf("cannot create %s to hold the update lock: %w", gaiaDir, err)
	}
	path := filepath.Join(gaiaDir, LockFileNameOnDisk)

	guard, err := createLock(path, now)
	if err == nil {
		return guard, nil
	}
	var busy *BusyError
	if !errors.As(err, &busy) {
		return nil, err
	}

	// Held. Only an abandoned lock may be taken, and only once — a second
	// EEXIST means a live process won the race, which is the guard working.
	age, holderPID, ageErr := lockAge(path, now)
	if ageErr != nil {
		return nil, ageErr
	}
	if age < StaleLockAfter {
		return nil, &BusyError{Holder: fmt.Sprintf("pid %d, since %s ago, per %s",
			holderPID, age.Round(time.Second), path)}
	}
	if rmErr := os.Remove(path); rmErr != nil && !errors.Is(rmErr, fs.ErrNotExist) {
		return nil, fmt.Errorf(
			"an abandoned update lock at %s (pid %d, %s old) could not be removed: %w. "+
				"Delete it by hand and re-run", path, holderPID, age.Round(time.Second), rmErr)
	}
	guard, err = createLock(path, now)
	if err != nil {
		return nil, err
	}
	guard.Reclaimed = fmt.Sprintf(
		"took over an update lock abandoned by pid %d %s ago — a previous update did "+
			"not finish. Check that %s is intact",
		holderPID, age.Round(time.Second), path)
	return guard, nil
}

func createLock(path string, now func() time.Time) (*Guard, error) {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600) // #nosec G304 -- path is derived from the user's home dir
	if errors.Is(err, fs.ErrExist) {
		return nil, &BusyError{Holder: "per " + path}
	}
	if err != nil {
		return nil, fmt.Errorf("cannot create the update lock at %s: %w", path, err)
	}
	payload, err := json.Marshal(lockRecord{PID: os.Getpid(), StartedAt: FormatTime(now())})
	if err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return nil, fmt.Errorf("cannot encode the update lock record: %w", err)
	}
	if _, err := f.Write(payload); err != nil {
		_ = f.Close()
		_ = os.Remove(path)
		return nil, fmt.Errorf("cannot write the update lock at %s: %w", path, err)
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(path)
		return nil, fmt.Errorf("cannot close the update lock at %s: %w", path, err)
	}
	return &Guard{path: path}, nil
}

// lockAge reads how long the current holder has held the lock.
//
// A lock file that cannot be read or parsed is treated as infinitely old: it
// cannot identify a live run, and leaving updates blocked forever on an
// unparsable byte is worse than reclaiming it. The reclaim is still reported.
func lockAge(path string, now func() time.Time) (time.Duration, int, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- path is derived from the user's home dir
	if errors.Is(err, fs.ErrNotExist) {
		// Released between the create and this read — treat as free.
		return StaleLockAfter + time.Second, 0, nil
	}
	if err != nil {
		return 0, 0, fmt.Errorf(
			"an update lock at %s exists but cannot be read: %w. Fix its permissions or "+
				"delete it, then re-run", path, err)
	}
	var rec lockRecord
	if err := json.Unmarshal(raw, &rec); err != nil {
		return StaleLockAfter + time.Second, 0, nil
	}
	started, err := time.Parse(time.RFC3339, rec.StartedAt)
	if err != nil {
		return StaleLockAfter + time.Second, rec.PID, nil
	}
	age := now().Sub(started)
	if age < 0 {
		age = 0
	}
	return age, rec.PID, nil
}

// Warnings is what this guard has to say for itself so far.
func (g *Guard) Warnings() []string {
	if g == nil || g.Reclaimed == "" {
		return nil
	}
	return []string{g.Reclaimed}
}

// ReleaseWithWarnings releases and appends any release failure to warnings.
//
// A lock that cannot be removed blocks the next update for StaleLockAfter, so
// it is reported rather than dropped — but it does not invalidate the work that
// just succeeded, which is why it is a warning and not the returned error.
func (g *Guard) ReleaseWithWarnings(warnings []string) []string {
	if err := g.Release(); err != nil {
		return append(warnings, err.Error())
	}
	return warnings
}

// Release drops both guards. Safe to call more than once.
func (g *Guard) Release() error {
	if g == nil || g.released {
		return nil
	}
	g.released = true
	defer inProcess.Unlock()
	if err := os.Remove(g.path); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf(
			"cannot remove the update lock at %s: %w. Delete it by hand or the next "+
				"update waits %s for it to expire", g.path, err, StaleLockAfter)
	}
	return nil
}
