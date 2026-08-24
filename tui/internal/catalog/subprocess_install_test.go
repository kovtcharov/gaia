package catalog

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// A subprocess agent is spawned as a child: no sidecar spec, no daemon
// supervising it, no hub publication it needs. Its launchability was gated on
// the daemon promoting it or on a ~/.gaia/agents/<id>/.installed sentinel --
// neither of which a direct-spawn agent ever gets. So the one-click installer
// could put gaia-agent on PATH and the flagship would still sit under Coming
// Soon, unlaunchable, beside a binary that runs fine.
//
// These pin both directions: present binary promotes, absent binary does not.

func writeFakeBinary(t *testing.T, dir, name string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte("fake"), 0o755); err != nil {
		t.Fatalf("writing fake binary: %v", err)
	}
	return path
}

// isolateLookups points HOME and the repo probe at empty dirs so the test
// cannot be rescued by a real gaia-agent on the developer's machine.
func isolateLookups(t *testing.T) {
	t.Helper()
	empty := t.TempDir()
	t.Setenv("HOME", empty)
	t.Setenv("USERPROFILE", empty)
}

func TestSubprocessAgentOnPathBecomesLaunchable(t *testing.T) {
	isolateLookups(t)
	binDir := t.TempDir()
	writeFakeBinary(t, binDir, "gaia-agent")
	t.Setenv("PATH", binDir)

	cat := NewCatalog()
	cat.DiscoverBinaries()

	agent := cat.Get("gaia")
	if agent == nil {
		t.Fatal("the seed catalog no longer carries the flagship 'gaia' entry")
	}
	if !agent.Status.IsLaunchable() {
		t.Errorf("gaia resolved to %s but its status is %v (not launchable) -- "+
			"the flagship would show under Coming Soon next to a working binary",
			agent.BinaryPath, agent.Status)
	}
	if agent.NotOfferedReason != "" {
		t.Errorf("gaia is runnable but still carries NotOfferedReason %q, which is now false",
			agent.NotOfferedReason)
	}
}

func TestSubprocessAgentWithNoBinaryStaysComingSoon(t *testing.T) {
	isolateLookups(t)
	// An empty PATH: nothing can resolve, so nothing may claim to be installed.
	t.Setenv("PATH", t.TempDir())

	cat := NewCatalog()
	cat.DiscoverBinaries()

	agent := cat.Get("gaia")
	if agent == nil {
		t.Fatal("the seed catalog no longer carries the flagship 'gaia' entry")
	}
	if agent.Status.IsLaunchable() {
		t.Error("gaia claims to be launchable with no binary on disk -- that is the " +
			"row-that-fails-on-the-first-message bug the seed comment warns about")
	}
}

func TestPromotionDoesNotTouchDaemonAgents(t *testing.T) {
	isolateLookups(t)
	binDir := t.TempDir()
	// Even if something with the right name is on PATH, a daemon agent's
	// lifecycle belongs to the daemon and its status must not be invented here.
	writeFakeBinary(t, binDir, "email")
	t.Setenv("PATH", binDir)

	cat := NewCatalog()
	cat.DiscoverBinaries()

	agent := cat.Get("email")
	if agent == nil {
		t.Fatal("the seed catalog no longer carries the 'email' entry")
	}
	if agent.Transport != TransportDaemon {
		t.Fatalf("email is no longer a daemon agent (transport %v); this test needs rewriting",
			agent.Transport)
	}
	if agent.Status != StatusAvailable {
		t.Errorf("email's status changed to %v; a daemon agent must keep whatever "+
			"the daemon reports, not be promoted by a binary lookup", agent.Status)
	}
}
