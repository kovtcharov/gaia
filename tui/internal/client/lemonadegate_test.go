package client

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/amd/gaia/tui/internal/catalog"
)

// The flagship builds its embedder during construction, so with no Lemonade it
// exits 1 with the reason in a log file. On screen that was a bare exit code.
// The spawn is refused up front instead, with something to act on.

func noLemonade(t *testing.T) {
	t.Helper()
	// Point the probe at nothing. Any real server on the default ports would
	// otherwise make this test pass for the wrong reason.
	t.Setenv("LEMONADE_BASE_URL", "")
	orig := lemonadePorts
	lemonadePorts = []string{"1"} // port 1: nothing listens there
	t.Cleanup(func() { lemonadePorts = orig })
}

func TestSpawnIsRefusedWhenLemonadeIsRequiredAndAbsent(t *testing.T) {
	noLemonade(t)
	c := NewSubprocessClient("gaia-agent", nil, false)
	c.RequireLemonade(true)

	_, err := c.Send(context.Background(), "hi")
	if err == nil {
		t.Fatal("spawned an agent that cannot start; the user would get a bare exit 1")
	}
	msg := err.Error()
	for _, want := range []string{"Lemonade Server", "Looked on", "Next:", "use-claude"} {
		if !strings.Contains(msg, want) {
			t.Errorf("the refusal must name %q; got:\n%s", want, msg)
		}
	}
	if !strings.Contains(msg, "13305") && !strings.Contains(msg, "localhost:1") {
		t.Errorf("the refusal must name the ports it probed; got:\n%s", msg)
	}
}

// An agent that does not need a model server must still spawn. Making the gate
// unconditional would break every future direct-spawn agent that has no LLM.
func TestSpawnIsNotGatedWhenLemonadeIsNotRequired(t *testing.T) {
	noLemonade(t)
	c := NewSubprocessClient("", nil, false)
	c.RequireLemonade(false)

	_, err := c.Send(context.Background(), "hi")
	if err == nil {
		t.Fatal("expected the empty-path error, not a successful spawn")
	}
	if strings.Contains(err.Error(), "Lemonade") {
		t.Errorf("an agent that needs no model server must not be gated on one; got:\n%s", err)
	}
}

// The requirement is a property of the catalog entry, not of the launch flags.
func TestTheFlagshipDeclaresItNeedsLemonade(t *testing.T) {
	cat := catalog.NewCatalog()
	agent := cat.Get("gaia")
	if agent == nil {
		t.Fatal("the seed catalog no longer carries the flagship entry")
	}
	if !agent.NeedsLemonade {
		t.Error("the flagship builds its embedder at startup; it must declare NeedsLemonade")
	}
}

// "Not installed" and "installed but not running" are different problems, and
// telling someone without Lemonade to start it is a dead end.
func TestRemedyDistinguishesNotInstalledFromNotRunning(t *testing.T) {
	// Nothing on PATH and no install dirs: this machine does not have it.
	t.Setenv("PATH", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Setenv("ProgramFiles", t.TempDir())

	got := lemonadeRemedy()
	if !strings.Contains(got, "not installed") {
		t.Errorf("with nothing installed the remedy must say so, got: %q", got)
	}
	if !strings.Contains(got, "lemonade-server.ai") {
		t.Errorf("the remedy must name where to get it, got: %q", got)
	}
	if strings.Contains(got, "Start it") {
		t.Errorf("must not tell an uninstalled machine to start it, got: %q", got)
	}
}

func TestRemedySaysStartWhenItIsInstalled(t *testing.T) {
	dir := t.TempDir()
	// A stand-in for an installed server, found the way the real one is.
	name := "lemonade-server"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Setenv("ProgramFiles", t.TempDir())

	got := lemonadeRemedy()
	if strings.Contains(got, "not installed") {
		t.Errorf("an installed server must not be reported as missing, got: %q", got)
	}
	if !strings.Contains(got, "Start it") && !strings.Contains(got, "shortcut") {
		t.Errorf("the remedy must say how to start it, got: %q", got)
	}
}
