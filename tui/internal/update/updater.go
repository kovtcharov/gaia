// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// Package update is the GAIA TUI's self-updater.
//
// It mirrors the Agent UI's auto-updater (runtime-resolved channel, a loud
// no-channel state, a concurrency guard, list-then-install-a-chosen-version,
// and a persisted pin that pauses auto-update) and differs in one deliberate
// way: it PROMPTS BEFORE DOWNLOADING. The Agent UI downloads in the background
// and asks only about restarting; a terminal binary that replaces itself
// mid-session has to ask first, so nothing is fetched until the user has seen
// the current version, the available version, and the download size.
//
// Every downloaded file is SHA-256 verified against binaries.lock.json before
// it replaces anything on disk. A mismatch is refused — there is no retry, no
// "use it anyway", and nothing on disk is touched.
package update

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// SidecarAgentID is the flagship agent whose sidecar ships alongside the TUI.
// Its install directory is a cross-repo contract with the daemon and with
// @amd-gaia/gaia — both stage the verified binary at ~/.gaia/agents/gaia.
const SidecarAgentID = "gaia"

// sentinelName is the daemon's install record inside an agent directory.
const sentinelName = ".installed"

// defaultTimeout bounds a whole check-or-install. The sidecar is >100 MB, so
// this is generous; without it a stalled feed hangs the command forever.
const defaultTimeout = 15 * time.Minute

// Options configures an Updater. Every field has a working default except the
// running TUI's version, which only the CLI knows.
type Options struct {
	// GaiaDir is ~/.gaia. Defaults to the user's home.
	GaiaDir string
	// TUIVersion is the version of the running binary.
	TUIVersion string
	// TUIPath is the binary to replace. Defaults to os.Executable().
	TUIPath string
	// SidecarDir holds the flagship agent's binary. Defaults to the directory
	// the running TUI is in when a sidecar sits beside it, else
	// <GaiaDir>/agents/gaia. See resolveSidecarDir.
	SidecarDir string
	// Env reads the environment. Defaults to os.Getenv.
	Env func(string) string
	// Client performs every HTTP call. Defaults to a client with defaultTimeout.
	Client *http.Client
	// GOOS/GOARCH select the platform in the lock. Default to this build's.
	GOOS, GOARCH string
	// Now is the clock, injected for tests.
	Now func() time.Time
}

// Updater resolves a channel, compares versions, and installs a release.
type Updater struct {
	opts Options
}

// New builds an Updater, filling in the defaults.
func New(opts Options) (*Updater, error) {
	if opts.Env == nil {
		opts.Env = os.Getenv
	}
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.GOOS == "" {
		opts.GOOS = runtime.GOOS
	}
	if opts.GOARCH == "" {
		opts.GOARCH = runtime.GOARCH
	}
	if opts.Client == nil {
		opts.Client = &http.Client{Timeout: defaultTimeout}
	}
	if opts.GaiaDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf(
				"cannot locate your home directory, so the update config at %s cannot "+
					"be read: %w", filepath.Join("~", ".gaia", ConfigFileName), err)
		}
		opts.GaiaDir = filepath.Join(home, ".gaia")
	}
	if opts.TUIPath == "" {
		exe, err := os.Executable()
		if err != nil {
			return nil, fmt.Errorf(
				"cannot locate the running gaia-tui binary, so there is nothing to "+
					"replace: %w. Re-run with an absolute path", err)
		}
		// A symlinked launcher must not be replaced by a real binary — resolve
		// to what actually holds the bytes.
		if resolved, err := filepath.EvalSymlinks(exe); err == nil {
			exe = resolved
		}
		opts.TUIPath = exe
	}
	if opts.SidecarDir == "" {
		opts.SidecarDir = resolveSidecarDir(opts.TUIPath, opts.GaiaDir, opts.GOOS)
	}
	return &Updater{opts: opts}, nil
}

// resolveSidecarDir picks the sidecar this machine will actually run.
//
// Two installers put it in different places: the daemon and @amd-gaia/gaia
// stage it at ~/.gaia/agents/gaia, while the one-click installer puts it beside
// gaia-tui and on PATH. The TUI resolves `gaia-agent` with exec.LookPath first
// (catalog.resolveAgentBinary), so with both present the colocated one wins --
// and updating the other would report success while the binary that runs stays
// on the old version.
func resolveSidecarDir(tuiPath, gaiaDir, goos string) string {
	hubDir := filepath.Join(gaiaDir, "agents", SidecarAgentID)
	if tuiPath == "" {
		return hubDir
	}
	beside := filepath.Dir(tuiPath)
	name := "gaia-agent"
	if goos == "windows" {
		name += ".exe"
	}
	if _, err := os.Stat(filepath.Join(beside, name)); err == nil {
		return beside
	}
	return hubDir
}

// GaiaDir is where this updater keeps its state.
func (u *Updater) GaiaDir() string { return u.opts.GaiaDir }

// TUIPath is the binary this updater would replace.
func (u *Updater) TUIPath() string { return u.opts.TUIPath }

// ComponentPlan is one component's part of a release.
type ComponentPlan struct {
	Name        string
	Current     string
	Available   string
	Size        int64
	URL         string
	SHA256      string
	Target      string
	NeedsUpdate bool
	// Note says why a component is not being updated, when that is not obvious.
	Note string
}

// Plan is everything one install would do, assembled before anything is
// downloaded so the prompt can state it in full.
type Plan struct {
	Release               string
	CurrentRelease        string
	Feed                  FeedRef
	Components            []ComponentPlan
	ReplacesRunningBinary bool
	// Explicit is set when the user named a release rather than taking the
	// newest one. That release may be older — naming one is what allows a
	// rollback — so callers must not describe it as "the newest".
	Explicit bool
}

// TotalBytes is the download the prompt quotes.
func (p *Plan) TotalBytes() int64 {
	var total int64
	for _, c := range p.Components {
		if c.NeedsUpdate {
			total += c.Size
		}
	}
	return total
}

// HasWork reports whether anything would actually be downloaded.
func (p *Plan) HasWork() bool {
	for _, c := range p.Components {
		if c.NeedsUpdate {
			return true
		}
	}
	return false
}

// CheckResult is what `update check` reports.
type CheckResult struct {
	Feed FeedRef
	// Plan is nil only when the feed could not be compared at all.
	Plan *Plan
	// UpToDate is true when the newest release is already installed.
	UpToDate bool
	// Pinned is the release auto-update is paused at, if any.
	Pinned string
	// SkippedThisRelease is true when the user already declined this exact one.
	SkippedThisRelease bool
	// Warnings are things that went wrong around the check without invalidating
	// it — an abandoned lock taken over, a lock that could not be released.
	// Surfaced by the caller, never dropped.
	Warnings []string
}

// Check resolves the channel and compares versions. It downloads no binaries.
func (u *Updater) Check(ctx context.Context) (*CheckResult, error) {
	if disabled, value := IsDisabled(u.opts.Env); disabled {
		return nil, &DisabledError{Value: value}
	}
	guard, err := Acquire(u.opts.GaiaDir, u.opts.Now)
	if err != nil {
		return nil, err
	}
	// result is filled in below; the deferred release appends to it, so a lock
	// that could not be removed still reaches the caller.
	var result *CheckResult
	defer func() {
		warnings := guard.ReleaseWithWarnings(guard.Warnings())
		if result != nil {
			result.Warnings = append(result.Warnings, warnings...)
		}
	}()

	cfg, err := LoadConfig(u.opts.GaiaDir)
	if err != nil {
		return nil, err
	}
	ref, err := ResolveFeedRef(u.opts.Env, cfg, u.opts.GaiaDir)
	if err != nil {
		return nil, err
	}
	plan, err := u.plan(ctx, ref, "", cfg)
	if err != nil {
		return nil, err
	}

	cfg.LastCheck = FormatTime(u.opts.Now())
	cfg.LastSeenVersion = plan.Release
	if err := SaveConfig(u.opts.GaiaDir, cfg); err != nil {
		return nil, err
	}

	result = &CheckResult{
		Feed:               ref,
		Plan:               plan,
		UpToDate:           !plan.HasWork(),
		Pinned:             cfg.PinnedVersion,
		SkippedThisRelease: cfg.SkippedVersion != "" && cfg.SkippedVersion == plan.Release,
	}
	return result, nil
}

// PinnedError refuses an automatic update while a pin is in force.
type PinnedError struct{ Version string }

func (e *PinnedError) Error() string {
	return fmt.Sprintf(
		"auto-update is paused: this machine is pinned to GAIA %s, so nothing was "+
			"downloaded.\n\n"+
			"  gaia-tui update unpin                  resume normal updates\n"+
			"  gaia-tui update install --version X    move the pin to another release",
		e.Version)
}

// InstallRequest asks for one release.
type InstallRequest struct {
	// Version is the release to install. Empty means the newest one. A specific
	// version is allowed to be OLDER than what is installed, and sets the pin.
	Version string
	// Prompter asks before anything is downloaded. Required.
	Prompter Prompter
	// Progress, when non-nil, is called during each download.
	Progress func(component string, done, total int64)
}

// InstallResult is what `update install` did.
type InstallResult struct {
	Feed      FeedRef
	Plan      *Plan
	Decision  Decision
	Installed []string
	// Pinned is set when this install pinned the machine to a release.
	Pinned string
	// UpToDate is true when there was nothing to do and no prompt was shown.
	UpToDate bool
	// AlreadySkipped is true when this release was declined earlier, so the
	// prompt was not shown again.
	AlreadySkipped bool
	// Warnings are non-fatal problems around the install. Surfaced, never dropped.
	Warnings []string
}

// Install runs the prompt-then-download flow.
func (u *Updater) Install(ctx context.Context, req InstallRequest) (*InstallResult, error) {
	if req.Prompter == nil {
		return nil, errors.New("install requires a prompter — nothing may be downloaded without asking")
	}
	if disabled, value := IsDisabled(u.opts.Env); disabled {
		return nil, &DisabledError{Value: value}
	}
	guard, err := Acquire(u.opts.GaiaDir, u.opts.Now)
	if err != nil {
		return nil, err
	}
	var result *InstallResult
	defer func() {
		warnings := guard.ReleaseWithWarnings(guard.Warnings())
		if result != nil {
			result.Warnings = append(result.Warnings, warnings...)
		}
	}()

	cfg, err := LoadConfig(u.opts.GaiaDir)
	if err != nil {
		return nil, err
	}
	if req.Version == "" && cfg.PinnedVersion != "" {
		return nil, &PinnedError{Version: cfg.PinnedVersion}
	}
	ref, err := ResolveFeedRef(u.opts.Env, cfg, u.opts.GaiaDir)
	if err != nil {
		return nil, err
	}
	plan, err := u.plan(ctx, ref, req.Version, cfg)
	if err != nil {
		return nil, err
	}

	result = &InstallResult{Feed: ref, Plan: plan}
	if !plan.HasWork() {
		result.UpToDate = true
		if req.Version != "" && cfg.PinnedVersion != plan.Release {
			// Naming a release you are already on is still a request to STAY on
			// it. Skipping the pin here would let the next check move off it.
			cfg.PinnedVersion = plan.Release
			result.Pinned = plan.Release
			if err := SaveConfig(u.opts.GaiaDir, cfg); err != nil {
				return result, err
			}
		}
		return result, nil
	}
	// A version the user already said no to is not offered again. An explicit
	// --version is the user asking for this one, so it clears that.
	if req.Version == "" && cfg.SkippedVersion == plan.Release {
		result.AlreadySkipped = true
		return result, nil
	}

	decision, err := req.Prompter.Confirm(plan)
	result.Decision = decision
	if err != nil {
		return result, err
	}

	switch decision {
	case DecisionSkip:
		cfg.SkippedVersion = plan.Release
		cfg.LastCheck = FormatTime(u.opts.Now())
		cfg.LastSeenVersion = plan.Release
		if err := SaveConfig(u.opts.GaiaDir, cfg); err != nil {
			return result, err
		}
		return result, nil
	case DecisionDecline:
		return result, nil
	}

	installed, err := u.apply(ctx, plan, &cfg, req.Progress)
	result.Installed = installed
	if err != nil {
		// Persist what DID land, so a partial install is not re-downloaded.
		if saveErr := SaveConfig(u.opts.GaiaDir, cfg); saveErr != nil {
			return result, fmt.Errorf("%w (and recording the partial install failed: %v)", err, saveErr)
		}
		return result, err
	}

	cfg.LastCheck = FormatTime(u.opts.Now())
	cfg.LastSeenVersion = plan.Release
	if cfg.SkippedVersion == plan.Release {
		cfg.SkippedVersion = ""
	}
	if req.Version != "" {
		// An explicitly chosen release pins the machine: without this, the next
		// check would offer to undo the rollback the user just asked for.
		cfg.PinnedVersion = plan.Release
		result.Pinned = plan.Release
	}
	if err := SaveConfig(u.opts.GaiaDir, cfg); err != nil {
		return result, fmt.Errorf(
			"the update installed, but recording it failed: %w. Run "+
				"`gaia-tui update status` to check the pin state", err)
	}
	return result, nil
}

// List returns published releases, newest first.
func (u *Updater) List(ctx context.Context) ([]Release, Config, FeedRef, error) {
	if disabled, value := IsDisabled(u.opts.Env); disabled {
		return nil, Config{}, FeedRef{}, &DisabledError{Value: value}
	}
	cfg, err := LoadConfig(u.opts.GaiaDir)
	if err != nil {
		return nil, Config{}, FeedRef{}, err
	}
	ref, err := ResolveFeedRef(u.opts.Env, cfg, u.opts.GaiaDir)
	if err != nil {
		return nil, cfg, FeedRef{}, err
	}
	feed, err := NewFeed(ref, u.opts.Client)
	if err != nil {
		return nil, cfg, ref, err
	}
	releases, err := feed.Versions(ctx)
	if err != nil {
		return nil, cfg, ref, err
	}
	return releases, cfg, ref, nil
}

// Pin pauses auto-update at a release. It does not install anything.
func (u *Updater) Pin(version string) error {
	if strings.TrimSpace(version) == "" {
		return errors.New("pin needs a version, e.g. `gaia-tui update pin 0.1.0`")
	}
	cfg, err := LoadConfig(u.opts.GaiaDir)
	if err != nil {
		return err
	}
	cfg.PinnedVersion = strings.TrimSpace(version)
	return SaveConfig(u.opts.GaiaDir, cfg)
}

// Unpin clears the pin and resumes normal updates.
func (u *Updater) Unpin() (string, error) {
	cfg, err := LoadConfig(u.opts.GaiaDir)
	if err != nil {
		return "", err
	}
	previous := cfg.PinnedVersion
	cfg.PinnedVersion = ""
	if err := SaveConfig(u.opts.GaiaDir, cfg); err != nil {
		return "", err
	}
	return previous, nil
}

// Status is `update status`: everything about the update state, no network.
type Status struct {
	Feed FeedRef
	// FeedErr is the loud no-channel state, or a bad feed kind. Reported rather
	// than rendered as a blank channel line.
	FeedErr        error
	Disabled       bool
	DisableValue   string
	ConfigPath     string
	TUIVersion     string
	TUIPath        string
	SidecarVersion string
	SidecarPath    string
	SidecarNote    string
	Pinned         string
	Skipped        string
	LastCheck      string
	LastSeen       string
	// StaleBackup names a leftover .old from a previous Windows swap.
	StaleBackup string
}

// Status reads local state only.
func (u *Updater) Status() (*Status, error) {
	cfg, err := LoadConfig(u.opts.GaiaDir)
	if err != nil {
		return nil, err
	}
	disabled, value := IsDisabled(u.opts.Env)
	st := &Status{
		Disabled:     disabled,
		DisableValue: value,
		ConfigPath:   ConfigPath(u.opts.GaiaDir),
		TUIVersion:   u.opts.TUIVersion,
		TUIPath:      u.opts.TUIPath,
		Pinned:       cfg.PinnedVersion,
		Skipped:      cfg.SkippedVersion,
		LastCheck:    cfg.LastCheck,
		LastSeen:     cfg.LastSeenVersion,
	}
	ref, refErr := ResolveFeedRef(u.opts.Env, cfg, u.opts.GaiaDir)
	st.Feed, st.FeedErr = ref, refErr

	record, note := u.sidecarRecord()
	if record != nil {
		st.SidecarVersion = record.Version
		st.SidecarPath = filepath.Join(u.opts.SidecarDir, record.Executable)
	}
	st.SidecarNote = note

	if backup := u.opts.TUIPath + BackupSuffix; fileExists(backup) {
		st.StaleBackup = backup
	}
	return st, nil
}

// plan builds the full picture of one release without downloading anything.
func (u *Updater) plan(ctx context.Context, ref FeedRef, version string, cfg Config) (*Plan, error) {
	feed, err := NewFeed(ref, u.opts.Client)
	if err != nil {
		return nil, err
	}
	lock, err := feed.Lock(ctx, version)
	if err != nil {
		return nil, err
	}
	platformKey, err := nodePlatformKey(u.opts.GOOS, u.opts.GOARCH)
	if err != nil {
		return nil, err
	}
	if version != "" && lock.AgentVersion != version {
		return nil, fmt.Errorf(
			"asked the update feed for %s but it served a lock for %s. The feed is "+
				"serving the wrong release — nothing was downloaded", version, lock.AgentVersion)
	}

	explicit := version != ""
	plan := &Plan{
		Release:        lock.AgentVersion,
		CurrentRelease: u.installedRelease(),
		Feed:           ref,
		Explicit:       explicit,
	}

	sidecarRecord, sidecarNote := u.sidecarRecord()

	for _, name := range Components {
		artifact, baseURL, err := lock.Resolve(name, platformKey)
		if err != nil {
			// The frozen sidecar publishes fewer platforms than the cross-compiled
			// TUI, so a missing sidecar build is a reported gap rather than a dead
			// end — the TUI still updates. A missing TUI build IS the dead end.
			if name == ComponentSidecar {
				plan.Components = append(plan.Components, ComponentPlan{Name: name, Note: err.Error()})
				continue
			}
			return nil, err
		}
		comp := lock.Components[name]

		item := ComponentPlan{
			Name:      name,
			Available: comp.ComponentVersion,
			Size:      artifact.Size,
			URL:       strings.TrimRight(baseURL, "/") + "/" + artifact.Filename,
			SHA256:    strings.ToLower(artifact.SHA256),
		}

		switch name {
		case ComponentTUI:
			onDisk, pendingRestart := u.installedTUIVersion(cfg)
			item.Current = onDisk
			item.Target = u.opts.TUIPath
			item.NeedsUpdate = wantsChange(item.Available, item.Current, explicit)
			switch {
			case item.NeedsUpdate:
			case pendingRestart:
				item.Note = onDisk + " is installed and waiting for you to restart this session " +
					"(it is still running " + orUnknown(u.opts.TUIVersion) + ")"
			default:
				item.Note = "already at " + orUnknown(item.Current)
			}
		case ComponentSidecar:
			item.Target = filepath.Join(u.opts.SidecarDir, artifact.Executable)
			if sidecarRecord == nil {
				// Installing the sidecar is the daemon's job; the updater only
				// refreshes one that is already there.
				item.Note = sidecarNote
				break
			}
			item.Current = sidecarRecord.Version
			item.NeedsUpdate = wantsChange(item.Available, item.Current, explicit)
			if !item.NeedsUpdate {
				item.Note = "already at " + orUnknown(item.Current)
			}
		}
		plan.Components = append(plan.Components, item)
	}

	for _, c := range plan.Components {
		if c.NeedsUpdate && c.Target == u.opts.TUIPath {
			plan.ReplacesRunningBinary = true
		}
	}
	return plan, nil
}

// installedTUIVersion is the version of the binary ON DISK, which is the one a
// download would replace.
//
// It is the running version until a swap happens; between that swap and the
// next start they differ, and comparing against the running one would offer —
// and re-download — an update that is already installed.
func (u *Updater) installedTUIVersion(cfg Config) (version string, pendingRestart bool) {
	if cfg.InstalledTUIVersion != "" && cfg.InstalledTUIVersion != u.opts.TUIVersion {
		return cfg.InstalledTUIVersion, true
	}
	return u.opts.TUIVersion, false
}

// wantsChange decides whether a component moves.
//
// A newer version wins on the normal path. An explicitly requested release
// moves to whatever it names — that is what makes rollback possible.
func wantsChange(available, current string, explicit bool) bool {
	if explicit {
		return available != current
	}
	return IsNewer(available, current)
}

// installedRelease is the release the machine is on, when that is knowable.
//
// The sidecar's install record is the only on-disk thing that carries a RELEASE
// version — the TUI's own version is its component version. With no sidecar
// installed the release is genuinely unknown, and is reported that way: the
// last check's answer is the version that is AVAILABLE, and returning it here
// would tell the user they already have what they have not installed.
func (u *Updater) installedRelease() string {
	if record, _ := u.sidecarRecord(); record != nil {
		return record.Version
	}
	return ""
}

// apply downloads and installs each component that needs it.
//
// Downloads are staged next to their targets, verified, and only then swapped
// in. The order is sidecar first: if the TUI swap is the one that fails, the
// user still has a working binary to run the retry from.
func (u *Updater) apply(
	ctx context.Context,
	plan *Plan,
	cfg *Config,
	progress func(string, int64, int64),
) ([]string, error) {
	var installed []string
	for _, comp := range plan.Components {
		if !comp.NeedsUpdate {
			continue
		}
		destDir := filepath.Dir(comp.Target)
		name := filepath.Base(comp.Target)

		var report func(int64, int64)
		if progress != nil {
			report = func(done, total int64) { progress(comp.Name, done, total) }
		}
		staged, err := DownloadVerified(ctx, u.opts.Client, comp.URL, comp.SHA256, destDir, name, report)
		if err != nil {
			return installed, err
		}
		if err := ReplaceBinary(comp.Target, staged); err != nil {
			return installed, err
		}
		// Record what is now on disk BEFORE moving on: a failure in a later
		// component must not leave this one looking un-installed, which would
		// re-download it on the next run.
		switch comp.Name {
		case ComponentTUI:
			cfg.InstalledTUIVersion = comp.Available
		case ComponentSidecar:
			if err := u.writeSidecarSentinel(comp.Available, filepath.Base(comp.Target)); err != nil {
				return installed, err
			}
		}
		installed = append(installed, fmt.Sprintf("%s %s", comp.Name, comp.Available))
	}
	return installed, nil
}

// writeSidecarSentinel updates the daemon's install record after a swap.
//
// The record IS the installed state for the daemon and for `gaia tui list`, so
// a replaced binary with a stale record would report the old version forever —
// and the updater would keep offering the same release.
func (u *Updater) writeSidecarSentinel(version, executable string) error {
	path := filepath.Join(u.opts.SidecarDir, sentinelName)
	fields := map[string]any{}
	raw, err := os.ReadFile(path) // #nosec G304 -- path is derived from the user's home dir
	// A directory the one-click installer owns has no sentinel to begin with;
	// writing a fresh one still serves the reason this record exists, which is
	// that the version on disk must not be reported stale.
	if errors.Is(err, fs.ErrNotExist) {
		raw, err = []byte("{}"), nil
		fields["id"] = SidecarAgentID
	}
	if err != nil {
		return fmt.Errorf(
			"the sidecar was replaced but its install record at %s could not be read "+
				"to update it: %w. Run `gaia tui install %s` to repair the record",
			path, err, SidecarAgentID)
	}
	if err := json.Unmarshal(raw, &fields); err != nil {
		return fmt.Errorf(
			"the sidecar was replaced but its install record at %s is not valid JSON: "+
				"%w. Run `gaia tui install %s` to repair it", path, err, SidecarAgentID)
	}
	// Merged, not rewritten: the daemon stores fields this build does not read.
	fields["version"] = version
	fields["executable"] = executable
	payload, err := json.MarshalIndent(fields, "", "  ")
	if err != nil {
		return fmt.Errorf("cannot encode the sidecar install record for %s: %w", path, err)
	}
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		return fmt.Errorf(
			"the sidecar was replaced but its install record at %s could not be "+
				"written: %w. Run `gaia tui install %s` to repair it",
			path, err, SidecarAgentID)
	}
	return nil
}

// installedRecord is one ~/.gaia/agents/<id>/.installed sentinel, the daemon's
// local record of what is installed.
type installedRecord struct {
	ID         string `json:"id"`
	Version    string `json:"version"`
	Executable string `json:"executable"`
}

// sidecarExecutableOnDisk names the sidecar binary sitting in SidecarDir, or ""
// when there is none. The one-click installer leaves a binary and no sentinel.
func (u *Updater) sidecarExecutableOnDisk() string {
	name := "gaia-agent"
	if u.opts.GOOS == "windows" {
		name += ".exe"
	}
	if _, err := os.Stat(filepath.Join(u.opts.SidecarDir, name)); err == nil {
		return name
	}
	return ""
}

// sidecarRecord reads the flagship sidecar's install record, or explains why
// there is none. A missing sidecar is a normal state, not a failure — the
// updater refreshes what is installed and never installs an agent behind the
// user's back.
func (u *Updater) sidecarRecord() (*installedRecord, string) {
	path := filepath.Join(u.opts.SidecarDir, sentinelName)
	raw, err := os.ReadFile(path) // #nosec G304 -- path is derived from the user's home dir
	if errors.Is(err, fs.ErrNotExist) {
		// Only the daemon's installer writes this sentinel. The one-click
		// installer does not, so the binary itself is the install record there
		// -- reporting "not installed" over a working binary would refuse to
		// update the very copy that runs.
		if exe := u.sidecarExecutableOnDisk(); exe != "" {
			return &installedRecord{ID: SidecarAgentID, Executable: exe}, ""
		}
		return nil, "not installed — `gaia tui install " + SidecarAgentID + "` installs it"
	}
	if err != nil {
		return nil, fmt.Sprintf("cannot be read (%s: %v), so its version is unknown and "+
			"it is left alone", path, err)
	}
	var record installedRecord
	if err := json.Unmarshal(raw, &record); err != nil {
		return nil, fmt.Sprintf("%s is not valid JSON (%v), so its version is unknown "+
			"and it is left alone — reinstall it with `gaia tui install %s`",
			path, err, SidecarAgentID)
	}
	if record.Executable == "" {
		return nil, fmt.Sprintf("%s names no executable, so there is nothing to replace "+
			"— reinstall it with `gaia tui install %s`", path, SidecarAgentID)
	}
	return &record, ""
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
