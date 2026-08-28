package update

import (
	"os"
	"path/filepath"
	"testing"
)

// Two installers put the sidecar in different places: the daemon and
// @amd-gaia/gaia stage it at ~/.gaia/agents/gaia, while the one-click installer
// puts it beside gaia-tui and on PATH. The TUI resolves `gaia-agent` with
// exec.LookPath first, so with both present the colocated one is what runs --
// and updating the other reports success while the running binary stays old.

func writeFile(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte("binary"), 0o755); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
}

func TestSidecarBesideTheTUIWins(t *testing.T) {
	installDir := t.TempDir()
	gaiaDir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(gaiaDir, "agents", SidecarAgentID), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(installDir, "gaia-agent.exe"))
	writeFile(t, filepath.Join(gaiaDir, "agents", SidecarAgentID, "gaia-agent.exe"))

	got := resolveSidecarDir(filepath.Join(installDir, "gaia-tui.exe"), gaiaDir, "windows")
	if got != installDir {
		t.Errorf("resolveSidecarDir = %q, want the colocated install dir %q -- "+
			"updating the hub copy would leave the binary that actually runs on the old version",
			got, installDir)
	}
}

func TestSidecarFallsBackToTheHubInstallRoot(t *testing.T) {
	installDir := t.TempDir() // gaia-tui only, no sidecar beside it
	gaiaDir := t.TempDir()
	hubDir := filepath.Join(gaiaDir, "agents", SidecarAgentID)
	if err := os.MkdirAll(hubDir, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(hubDir, "gaia-agent.exe"))

	got := resolveSidecarDir(filepath.Join(installDir, "gaia-tui.exe"), gaiaDir, "windows")
	if got != hubDir {
		t.Errorf("resolveSidecarDir = %q, want the hub install root %q", got, hubDir)
	}
}

func TestSidecarUsesTheHubRootWhenTheTUIPathIsUnknown(t *testing.T) {
	gaiaDir := t.TempDir()
	want := filepath.Join(gaiaDir, "agents", SidecarAgentID)
	if got := resolveSidecarDir("", gaiaDir, "windows"); got != want {
		t.Errorf("resolveSidecarDir = %q, want %q", got, want)
	}
}

func TestSidecarNameIsNotExeOffWindows(t *testing.T) {
	installDir := t.TempDir()
	gaiaDir := t.TempDir()
	// A unix install has no .exe suffix; looking for one would miss it and
	// silently update the hub copy instead.
	writeFile(t, filepath.Join(installDir, "gaia-agent"))

	got := resolveSidecarDir(filepath.Join(installDir, "gaia-tui"), gaiaDir, "linux")
	if got != installDir {
		t.Errorf("resolveSidecarDir = %q, want the colocated install dir %q", got, installDir)
	}
}

// The daemon's installer writes a .installed sentinel; the one-click installer
// does not. Reading "not installed" off a missing sentinel refused to update the
// very binary that runs.

func TestSidecarBinaryWithNoSentinelCountsAsInstalled(t *testing.T) {
	installDir := t.TempDir()
	writeFile(t, filepath.Join(installDir, "gaia-agent.exe"))

	u, err := New(Options{
		GaiaDir:    t.TempDir(),
		SidecarDir: installDir,
		TUIPath:    filepath.Join(installDir, "gaia-tui.exe"),
		GOOS:       "windows",
		Env:        func(string) string { return "" },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	record, note := u.sidecarRecord()
	if record == nil {
		t.Fatalf("a sidecar binary on disk was reported as absent: %q", note)
	}
	if record.Executable != "gaia-agent.exe" {
		t.Errorf("Executable = %q, want gaia-agent.exe", record.Executable)
	}
	if record.Version != "" {
		t.Errorf("Version = %q, want empty — nothing on disk states it", record.Version)
	}
}

func TestNoSidecarBinaryAndNoSentinelIsStillNotInstalled(t *testing.T) {
	empty := t.TempDir()
	u, err := New(Options{
		GaiaDir:    t.TempDir(),
		SidecarDir: empty,
		TUIPath:    filepath.Join(empty, "gaia-tui.exe"),
		GOOS:       "windows",
		Env:        func(string) string { return "" },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	record, note := u.sidecarRecord()
	if record != nil {
		t.Fatal("an absent sidecar was reported as installed")
	}
	if note == "" {
		t.Error("an absent sidecar produced no explanation")
	}
}
