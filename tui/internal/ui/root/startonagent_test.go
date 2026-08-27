package root

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/amd/gaia/tui/internal/catalog"
)

// A bare `gaia-tui` opens the agent, but the hub has to stay reachable behind
// it. Building the chat model directly instead of going through the hub's own
// launch path is what broke that: /hub answered "Not launched from hub."

func fakeAgent(t *testing.T, dir string) {
	t.Helper()
	name := "gaia-agent"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte("fake"), 0o755); err != nil {
		t.Fatalf("writing fake agent: %v", err)
	}
}

func TestStartOnAgentOpensChatWithTheHubBehindIt(t *testing.T) {
	empty := t.TempDir()
	t.Setenv("HOME", empty)
	t.Setenv("USERPROFILE", empty)
	dir := t.TempDir()
	fakeAgent(t, dir)
	t.Setenv("PATH", dir)

	cat := catalog.NewCatalog()
	cat.DiscoverBinaries()

	m, ok := NewRootModel(cat, false).StartOnAgent("gaia")
	if !ok {
		t.Fatal("the flagship binary is on PATH but StartOnAgent declined to open it")
	}
	if m.activeView != viewChat {
		t.Errorf("activeView = %v, want the chat view", m.activeView)
	}
	if m.chat == nil {
		t.Fatal("no chat model was built")
	}
	if !m.chat.CanReturnToHub() {
		t.Error("the chat cannot return to the hub; /hub would say \"Not launched from hub\"")
	}
	// Both views must initialise, not just the visible one. Asserting only
	// that Init is non-nil passes even when the hub never loads its catalog,
	// which is exactly the bug this guards: /hub then opens on a list stuck
	// at "loading" that refuses to install anything.
	cmd := m.Init()
	if cmd == nil {
		t.Fatal("Init returned no command; the visible view would never initialise")
	}
	// tea.Batch's command reports its children without running them, so this
	// stays offline -- the hub's own init is what would reach the network.
	batch, ok := cmd().(tea.BatchMsg)
	if !ok {
		t.Fatalf("Init returned a single command (%T); the hub behind the chat never initialises", cmd())
	}
	if len(batch) != 2 {
		t.Errorf("Init scheduled %d commands, want 2 (the chat and the hub behind it)", len(batch))
	}
}

func TestStartOnAgentDeclinesAnUnknownAgent(t *testing.T) {
	cat := catalog.NewCatalog()
	if _, ok := NewRootModel(cat, false).StartOnAgent("no-such-agent"); ok {
		t.Error("an unknown agent must leave the hub as the view, not claim success")
	}
}
