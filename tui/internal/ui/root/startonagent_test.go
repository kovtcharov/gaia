package root

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

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
	if m.Init() == nil {
		t.Error("Init returned no command; the visible view would never initialise")
	}
}

func TestStartOnAgentDeclinesAnUnknownAgent(t *testing.T) {
	cat := catalog.NewCatalog()
	if _, ok := NewRootModel(cat, false).StartOnAgent("no-such-agent"); ok {
		t.Error("an unknown agent must leave the hub as the view, not claim success")
	}
}
