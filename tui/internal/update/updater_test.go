// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// scriptedPrompter answers with a fixed decision and records the plan it saw,
// so a test can assert on exactly what the user would have been told.
type scriptedPrompter struct {
	decision Decision
	err      error
	seen     *Plan
	calls    int
}

func (p *scriptedPrompter) Confirm(plan *Plan) (Decision, error) {
	p.calls++
	p.seen = plan
	return p.decision, p.err
}

func twoReleases() []fakeRelease {
	return []fakeRelease{
		{
			version: "0.2.0", tuiVersion: "0.24.0", sidecarVersion: "0.2.0",
			tuiBody: []byte("tui binary 0.24.0"), sidecarBody: []byte("sidecar binary 0.2.0"),
			publishedAt: "2026-08-20T10:00:00Z",
		},
		{
			version: "0.1.0", tuiVersion: "0.23.0", sidecarVersion: "0.1.0",
			tuiBody: []byte("tui binary 0.23.0"), sidecarBody: []byte("sidecar binary 0.1.0"),
			publishedAt: "2026-07-01T10:00:00Z",
		},
	}
}

// The kill switch has to be honoured on every entry point, and it has to SAY
// so — a user who runs an update command explicitly and gets silence cannot
// tell "disabled" from "up to date".
func TestKillSwitchIsHonouredAndReported(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", map[string]string{EnvDisable: "1"})

	prompter := &scriptedPrompter{decision: DecisionAccept}
	for name, run := range map[string]func() error{
		"check": func() error { _, err := up.Check(context.Background()); return err },
		"install": func() error {
			_, err := up.Install(context.Background(), InstallRequest{Prompter: prompter})
			return err
		},
		"list": func() error { _, _, _, err := up.List(context.Background()); return err },
	} {
		err := run()
		var disabled *DisabledError
		if !errors.As(err, &disabled) {
			t.Fatalf("%s with GAIA_DISABLE_UPDATE=1 returned %s, want a DisabledError", name, fmtErr(err))
		}
		if !strings.Contains(err.Error(), EnvDisable) {
			t.Errorf("%s: the refusal does not name the env var that caused it:\n%v", name, err)
		}
	}
	if feed.hits != 0 {
		t.Errorf("the kill switch let %d artifact download(s) through", feed.hits)
	}
	if prompter.calls != 0 {
		t.Errorf("the kill switch still prompted the user %d time(s)", prompter.calls)
	}
}

// GAIA_DISABLE_UPDATE=0 must NOT disable updates — only "1" does. A typo'd 0
// that silently turned checking off would be indistinguishable from working.
func TestKillSwitchOnlyTriggersOnOne(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", map[string]string{EnvDisable: "0"})

	if _, err := up.Check(context.Background()); err != nil {
		t.Fatalf("check with GAIA_DISABLE_UPDATE=0: %v", err)
	}
}

// With no env var, no config, and a build whose default feed was stripped, the
// updater must say loudly that it checked NOTHING — never "up to date".
func TestNoChannelIsLoud(t *testing.T) {
	original := DefaultFeedURL
	DefaultFeedURL = ""
	t.Cleanup(func() { DefaultFeedURL = original })

	m := newFakeMachine(t, "0.23.0", "0.1.0")
	up := m.updater(t, "0.23.0", nil)

	_, err := up.Check(context.Background())
	var noChannel *NoChannelError
	if !errors.As(err, &noChannel) {
		t.Fatalf("check with no feed configured returned %s, want a NoChannelError", fmtErr(err))
	}
	for _, want := range []string{EnvFeedURL, "feedUrl", ConfigPath(m.gaiaDir), "paused"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the no-channel message never mentions %q, so it is not actionable:\n%v", want, err)
		}
	}
	if strings.Contains(strings.ToLower(err.Error()), "up to date") {
		t.Errorf("the no-channel state must never read as up to date:\n%v", err)
	}
}

// A pin is the user saying "stop moving". An automatic install must refuse and
// name how to resume, not quietly upgrade past it.
func TestPinBlocksAutomaticUpdate(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	if err := up.Pin("0.1.0"); err != nil {
		t.Fatalf("pin: %v", err)
	}
	prompter := &scriptedPrompter{decision: DecisionAccept}
	_, err := up.Install(context.Background(), InstallRequest{Prompter: prompter})

	var pinned *PinnedError
	if !errors.As(err, &pinned) {
		t.Fatalf("install while pinned returned %s, want a PinnedError", fmtErr(err))
	}
	if !strings.Contains(err.Error(), "update unpin") {
		t.Errorf("the pin refusal never says how to resume:\n%v", err)
	}
	if prompter.calls != 0 {
		t.Errorf("a pinned machine still prompted to download (%d time(s))", prompter.calls)
	}
	if feed.hits != 0 {
		t.Errorf("a pinned machine downloaded %d artifact(s)", feed.hits)
	}
	if got := readFile(t, m.tuiPath); got != "tui 0.23.0" {
		t.Errorf("the pinned binary was replaced: %q", got)
	}

	// Unpin resumes, and reports what it cleared.
	previous, err := up.Unpin()
	if err != nil {
		t.Fatalf("unpin: %v", err)
	}
	if previous != "0.1.0" {
		t.Errorf("unpin reported the cleared pin as %q, want 0.1.0", previous)
	}
	if _, err := up.Install(context.Background(), InstallRequest{Prompter: prompter}); err != nil {
		t.Fatalf("install after unpin: %v", err)
	}
	if prompter.calls != 1 {
		t.Errorf("install after unpin prompted %d time(s), want 1", prompter.calls)
	}
}

// An explicit --version is allowed to go BACKWARDS, and pins the machine so the
// next check cannot undo the rollback the user just asked for.
func TestExplicitVersionDowngradesAndPins(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.24.0", "0.2.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.24.0", nil)

	prompter := &scriptedPrompter{decision: DecisionAccept}
	result, err := up.Install(context.Background(), InstallRequest{Version: "0.1.0", Prompter: prompter})
	if err != nil {
		t.Fatalf("install --version 0.1.0: %v", err)
	}
	if !result.Plan.Explicit {
		t.Error("the plan for an explicitly chosen release is not marked as explicit")
	}
	if !IsNewer(result.Plan.CurrentRelease, result.Plan.Release) {
		t.Errorf("this was meant to be a downgrade: %s → %s",
			result.Plan.CurrentRelease, result.Plan.Release)
	}
	if len(result.Installed) != 2 {
		t.Fatalf("installed %v, want both components", result.Installed)
	}
	if got := readFile(t, m.tuiPath); got != "tui binary 0.23.0" {
		t.Errorf("the TUI was not rolled back: %q", got)
	}
	sidecar := readFile(t, m.sidecarDir+string(pathSep)+"gaia-agent.exe")
	if sidecar != "sidecar binary 0.1.0" {
		t.Errorf("the sidecar was not rolled back: %q", sidecar)
	}
	if result.Pinned != "0.1.0" {
		t.Errorf("a rollback did not pin the machine (got %q)", result.Pinned)
	}

	cfg, err := LoadConfig(m.gaiaDir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.PinnedVersion != "0.1.0" {
		t.Errorf("the pin was not persisted to %s (got %q)", ConfigPath(m.gaiaDir), cfg.PinnedVersion)
	}
}

// The whole point of the TUI's flow: a corrupt download is refused, and the
// binary on disk is left exactly as it was.
func TestHashMismatchIsRefusedAndNothingIsReplaced(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	feed.corruptTUI = true
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	prompter := &scriptedPrompter{decision: DecisionAccept}
	_, err := up.Install(context.Background(), InstallRequest{Prompter: prompter})

	var integrity *IntegrityError
	if !errors.As(err, &integrity) {
		t.Fatalf("a tampered download returned %s, want an IntegrityError", fmtErr(err))
	}
	message := err.Error()
	for _, want := range []string{integrity.Expected, integrity.Actual, integrity.Path, "expected", "actual"} {
		if !strings.Contains(message, want) {
			t.Errorf("the integrity refusal never mentions %q:\n%s", want, message)
		}
	}
	if got := readFile(t, m.tuiPath); got != "tui 0.23.0" {
		t.Errorf("a binary that failed its hash check still replaced the old one: %q", got)
	}
}

// A declined prompt downloads nothing at all — that is the promise the prompt
// makes on screen.
func TestDeclineDownloadsNothing(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	prompter := &scriptedPrompter{decision: DecisionDecline}
	result, err := up.Install(context.Background(), InstallRequest{Prompter: prompter})
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if result.Decision != DecisionDecline {
		t.Fatalf("decision %v, want decline", result.Decision)
	}
	if feed.hits != 0 {
		t.Errorf("a declined update downloaded %d artifact(s)", feed.hits)
	}
	if got := readFile(t, m.tuiPath); got != "tui 0.23.0" {
		t.Errorf("a declined update replaced the binary: %q", got)
	}
	cfg, err := LoadConfig(m.gaiaDir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.SkippedVersion != "" {
		t.Errorf("declining recorded a skip (%q) — decline means ask again", cfg.SkippedVersion)
	}
}

// Skip is remembered, so the same version is never offered twice.
func TestSkippedVersionIsNotOfferedAgain(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	skip := &scriptedPrompter{decision: DecisionSkip}
	if _, err := up.Install(context.Background(), InstallRequest{Prompter: skip}); err != nil {
		t.Fatalf("install (skip): %v", err)
	}
	cfg, err := LoadConfig(m.gaiaDir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.SkippedVersion != "0.2.0" {
		t.Fatalf("the skipped version was not persisted (got %q)", cfg.SkippedVersion)
	}

	again := &scriptedPrompter{decision: DecisionAccept}
	result, err := up.Install(context.Background(), InstallRequest{Prompter: again})
	if err != nil {
		t.Fatalf("second install: %v", err)
	}
	if again.calls != 0 {
		t.Errorf("the user was re-asked about a version they skipped (%d prompt(s))", again.calls)
	}
	if !result.AlreadySkipped {
		t.Error("the result does not report that this release was skipped earlier")
	}

	// Asking for it by name is the user overriding their own skip.
	explicit := &scriptedPrompter{decision: DecisionAccept}
	if _, err := up.Install(context.Background(), InstallRequest{Version: "0.2.0", Prompter: explicit}); err != nil {
		t.Fatalf("explicit install of a skipped version: %v", err)
	}
	if explicit.calls != 1 {
		t.Errorf("an explicitly requested version did not prompt (%d call(s))", explicit.calls)
	}
	cfg, err = LoadConfig(m.gaiaDir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.SkippedVersion != "" {
		t.Errorf("installing a skipped version left the skip in place (%q)", cfg.SkippedVersion)
	}
}

// check compares and reports; it must never download.
func TestCheckDownloadsNothing(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	result, err := up.Check(context.Background())
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if result.UpToDate {
		t.Error("check reported up to date while 0.2.0 was published")
	}
	if result.Plan.Release != "0.2.0" {
		t.Errorf("check found release %q, want 0.2.0", result.Plan.Release)
	}
	if feed.hits != 0 {
		t.Errorf("check downloaded %d artifact(s)", feed.hits)
	}
	if got := readFile(t, m.tuiPath); got != "tui 0.23.0" {
		t.Errorf("check replaced the binary: %q", got)
	}
}

// A machine already on the newest release is told so, and is not prompted.
func TestUpToDateDoesNotPrompt(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.24.0", "0.2.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.24.0", nil)

	prompter := &scriptedPrompter{decision: DecisionAccept}
	result, err := up.Install(context.Background(), InstallRequest{Prompter: prompter})
	if err != nil {
		t.Fatalf("install: %v", err)
	}
	if !result.UpToDate {
		t.Errorf("an up-to-date machine was not reported as such: %+v", result.Plan.Components)
	}
	if prompter.calls != 0 {
		t.Errorf("an up-to-date machine prompted %d time(s)", prompter.calls)
	}
}

// A sidecar that is not installed is REPORTED, not silently ignored — the
// updater refreshes what is there and never installs an agent unasked.
func TestMissingSidecarIsReportedNotSilentlyUpdated(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "") // no sidecar installed
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	result, err := up.Check(context.Background())
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	var sidecar *ComponentPlan
	for i := range result.Plan.Components {
		if result.Plan.Components[i].Name == ComponentSidecar {
			sidecar = &result.Plan.Components[i]
		}
	}
	if sidecar == nil {
		t.Fatal("the sidecar is missing from the plan entirely")
	}
	if sidecar.NeedsUpdate {
		t.Error("an uninstalled sidecar was planned for replacement")
	}
	if !strings.Contains(sidecar.Note, "not installed") {
		t.Errorf("an uninstalled sidecar carries no explanation: %q", sidecar.Note)
	}
}

// After an install, what is on disk has to be RECORDED — otherwise the next
// run compares against the stale versions and downloads the same release again.
func TestInstallRecordsWhatLandedOnDisk(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	if _, err := up.Install(context.Background(), InstallRequest{
		Prompter: &scriptedPrompter{decision: DecisionAccept},
	}); err != nil {
		t.Fatalf("install: %v", err)
	}
	afterFirst := feed.hits
	if afterFirst == 0 {
		t.Fatal("the accepted install downloaded nothing")
	}

	// The sidecar's install record is the daemon's source of truth.
	record, note := up.sidecarRecord()
	if record == nil {
		t.Fatalf("the sidecar install record is gone after an update: %s", note)
	}
	if record.Version != "0.2.0" {
		t.Errorf("the sidecar record still says %q after installing 0.2.0", record.Version)
	}

	// A second run must find nothing to do, not re-download the same release.
	again := &scriptedPrompter{decision: DecisionAccept}
	result, err := up.Install(context.Background(), InstallRequest{Prompter: again})
	if err != nil {
		t.Fatalf("second install: %v", err)
	}
	if !result.UpToDate {
		t.Errorf("a machine that just installed 0.2.0 was offered it again: %+v", result.Plan.Components)
	}
	if feed.hits != afterFirst {
		t.Errorf("the same release was downloaded twice (%d → %d artifact fetches)", afterFirst, feed.hits)
	}
	if again.calls != 0 {
		t.Errorf("the user was re-prompted for a release already installed (%d time(s))", again.calls)
	}
}

// The swapped-in TUI is on disk but not yet running. That state has to be
// named, not reported as "already at" the version the user is still running.
func TestPendingRestartIsNamed(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	if _, err := up.Install(context.Background(), InstallRequest{
		Prompter: &scriptedPrompter{decision: DecisionAccept},
	}); err != nil {
		t.Fatalf("install: %v", err)
	}
	result, err := up.Check(context.Background())
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	var tui *ComponentPlan
	for i := range result.Plan.Components {
		if result.Plan.Components[i].Name == ComponentTUI {
			tui = &result.Plan.Components[i]
		}
	}
	if tui == nil {
		t.Fatal("the TUI is missing from the plan")
	}
	if tui.NeedsUpdate {
		t.Fatal("an already-swapped TUI was planned for another download")
	}
	if !strings.Contains(tui.Note, "restart") {
		t.Errorf("the pending-restart state is not explained: %q", tui.Note)
	}
}

// A lock taken over from an unfinished run must reach the CALLER, not just the
// guard — it is the only sign that a previous update died mid-flight.
func TestReclaimedLockSurfacesInTheResult(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "0.1.0")
	m.setFeed(t, feed)

	stale, err := json.Marshal(lockRecord{
		PID:       4242,
		StartedAt: FormatTime(time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC).Add(-2 * StaleLockAfter)),
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	mustMkdir(t, m.gaiaDir)
	mustWrite(t, filepath.Join(m.gaiaDir, LockFileNameOnDisk), stale)

	result, err := m.updater(t, "0.23.0", nil).Check(context.Background())
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	if len(result.Warnings) == 0 {
		t.Fatal("the abandoned lock was taken over without telling the caller")
	}
	if !strings.Contains(strings.Join(result.Warnings, " "), "4242") {
		t.Errorf("the warning does not name the abandoning pid: %v", result.Warnings)
	}
}

// With no sidecar installed there is no on-disk release version, and the last
// check's answer is what is AVAILABLE — reporting it as installed would tell
// the user they already have what they have not installed.
func TestUnknownInstalledReleaseIsNotTheAvailableOne(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	m := newFakeMachine(t, "0.23.0", "") // no sidecar
	m.setFeed(t, feed)
	up := m.updater(t, "0.23.0", nil)

	// The first check persists lastSeenVersion=0.2.0.
	if _, err := up.Check(context.Background()); err != nil {
		t.Fatalf("first check: %v", err)
	}
	result, err := up.Check(context.Background())
	if err != nil {
		t.Fatalf("second check: %v", err)
	}
	if result.Plan.CurrentRelease == result.Plan.Release {
		t.Errorf("a machine with nothing installed was told it already has %s",
			result.Plan.Release)
	}
	if !strings.Contains(result.Plan.Summary(), "you have unknown") {
		t.Errorf("an unknown installed release is not reported as unknown:\n%s",
			result.Plan.Summary())
	}
}
