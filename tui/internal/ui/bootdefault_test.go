package ui

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// We ship one agent, so a bare `gaia-tui` opens it rather than a catalogue of
// mostly-unreleased rows. The hub stays reachable through /hub.

func fakeAgentBinary(t *testing.T, dir string) string {
	t.Helper()
	name := "gaia-agent"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("fake"), 0o755); err != nil {
		t.Fatalf("writing fake agent: %v", err)
	}
	return path
}

func isolate(t *testing.T) {
	t.Helper()
	empty := t.TempDir()
	t.Setenv("HOME", empty)
	t.Setenv("USERPROFILE", empty)
}

func TestABareLaunchOpensTheAgentWhenItsBinaryIsThere(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	fakeAgentBinary(t, dir)
	t.Setenv("PATH", dir)

	if !DefaultAgentIsRunnable("") {
		t.Error("the flagship binary is on PATH but a bare launch would still open the hub")
	}
}

func TestABareLaunchFallsBackToTheHubWithNoAgent(t *testing.T) {
	isolate(t)
	t.Setenv("PATH", t.TempDir()) // nothing resolves

	if DefaultAgentIsRunnable("") {
		t.Error("with no agent binary the hub must be the fallback, not a dead chat view")
	}
}

// --mock drives the hub in its own tests; booting past it would break them.
func TestMockStillOpensTheHub(t *testing.T) {
	isolate(t)
	dir := t.TempDir()
	fakeAgentBinary(t, dir)
	t.Setenv("PATH", dir)

	if DefaultAgentIsRunnable(filepath.Join(dir, "mock")) {
		t.Error("--mock must still land on the hub")
	}
}
