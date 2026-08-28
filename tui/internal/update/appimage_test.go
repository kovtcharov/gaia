package update

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// An AppImage runs from a read-only image, and `--appimage-extract-and-run`
// unpacks it under /tmp and runs from there. Either way, replacing the binaries
// the updater can see changes nothing that survives the process -- so an install
// downloaded 134 MB, reported "Installed 3.0.0", and left the .AppImage
// byte-identical. It has to refuse before downloading.

func appImageUpdater(t *testing.T, env map[string]string, tuiPath string) *Updater {
	t.Helper()
	u, err := New(Options{
		GaiaDir:    t.TempDir(),
		TUIPath:    tuiPath,
		SidecarDir: t.TempDir(),
		GOOS:       "linux",
		Env:        func(k string) string { return env[k] },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return u
}

func TestAppImageIsDetectedFromTheEnvVar(t *testing.T) {
	u := appImageUpdater(t, map[string]string{"APPIMAGE": "/home/me/GAIA.AppImage"}, "/tmp/.mount_x/usr/bin/gaia-tui")
	if got := u.appImagePath(); got != "/home/me/GAIA.AppImage" {
		t.Errorf("appImagePath = %q, want the APPIMAGE path", got)
	}
}

// The extract-and-run mode does not set APPIMAGE; the path is the only signal.
func TestAppImageIsDetectedFromTheExtractionPath(t *testing.T) {
	exe := "/tmp/appimage_extracted_9a1bb8a7/usr/bin/gaia-tui"
	u := appImageUpdater(t, map[string]string{}, exe)
	if got := u.appImagePath(); got == "" {
		t.Error("an extracted AppImage was not detected; an install there silently does nothing")
	}
}

func TestAnOrdinaryInstallIsNotMistakenForAnAppImage(t *testing.T) {
	u := appImageUpdater(t, map[string]string{}, filepath.Join(t.TempDir(), "gaia-tui"))
	if got := u.appImagePath(); got != "" {
		t.Errorf("a normal install was treated as an AppImage (%q); updates would be refused for everyone", got)
	}
}

func TestInstallRefusesInsideAnAppImageBeforeDownloading(t *testing.T) {
	u := appImageUpdater(t, map[string]string{"APPIMAGE": "/home/me/GAIA.AppImage"}, "/tmp/.mount_x/usr/bin/gaia-tui")

	_, err := u.Install(context.Background(), InstallRequest{Prompter: &scriptedPrompter{decision: DecisionAccept}})
	if err == nil {
		t.Fatal("an AppImage install reported success while changing nothing")
	}
	for _, want := range []string{"AppImage", "cannot update itself", "replace that file"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal must mention %q; got:\n%s", want, err)
		}
	}
}
