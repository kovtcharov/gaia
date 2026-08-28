// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// pathSep is the OS separator, spelled out so tests can build expected paths
// without importing filepath at every call site.
const pathSep = os.PathSeparator

func serveBody(t *testing.T, body []byte) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	t.Cleanup(server.Close)
	return server
}

// A download whose hash does not match the lock is refused, and the message
// names expected, actual, and where the rejected bytes are.
func TestDownloadVerifiedRefusesAHashMismatch(t *testing.T) {
	server := serveBody(t, []byte("not what the lock describes"))
	destDir := t.TempDir()

	_, err := DownloadVerified(context.Background(), server.Client(), server.URL,
		sha256Hex([]byte("the real binary")), destDir, "gaia-tui.exe", nil)

	var integrity *IntegrityError
	if !errors.As(err, &integrity) {
		t.Fatalf("a mismatched download returned %s, want an IntegrityError", fmtErr(err))
	}
	if integrity.Expected == integrity.Actual {
		t.Fatal("the error reports the same digest for expected and actual")
	}
	if !strings.Contains(err.Error(), integrity.Expected) || !strings.Contains(err.Error(), integrity.Actual) {
		t.Errorf("the message does not name both digests:\n%v", err)
	}
	if _, statErr := os.Stat(integrity.Path); statErr != nil {
		t.Errorf("the message names %s, but nothing is there: %v", integrity.Path, statErr)
	}
	// Nothing usable is left where the verified binary would have gone.
	if _, statErr := os.Stat(filepath.Join(destDir, "gaia-tui.exe")); !os.IsNotExist(statErr) {
		t.Errorf("a rejected download was left at the target name (%v)", statErr)
	}
}

func TestDownloadVerifiedStagesAMatchingBody(t *testing.T) {
	body := []byte("the real binary")
	server := serveBody(t, body)
	destDir := t.TempDir()

	var lastDone int64
	staged, err := DownloadVerified(context.Background(), server.Client(), server.URL,
		sha256Hex(body), destDir, "gaia-tui.exe", func(done, _ int64) { lastDone = done })
	if err != nil {
		t.Fatalf("download: %v", err)
	}
	if got := readFile(t, staged); got != string(body) {
		t.Errorf("staged %q, want %q", got, body)
	}
	if lastDone != int64(len(body)) {
		t.Errorf("progress reported %d bytes, want %d", lastDone, len(body))
	}
	// Staged, not installed: the swap is a separate, deliberate step.
	if filepath.Base(staged) == "gaia-tui.exe" {
		t.Error("the download was written straight to the target name")
	}
}

// The feed 404ing a withdrawn release must say so, not produce an empty file.
func TestDownloadVerifiedReportsHTTPFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "gone", http.StatusNotFound)
	}))
	t.Cleanup(server.Close)

	_, err := DownloadVerified(context.Background(), server.Client(), server.URL,
		sha256Hex([]byte("x")), t.TempDir(), "gaia-tui.exe", nil)
	if err == nil {
		t.Fatal("a 404 produced no error")
	}
	if !strings.Contains(err.Error(), "404") || !strings.Contains(err.Error(), "update list") {
		t.Errorf("the failure is not actionable:\n%v", err)
	}
}

// The self-replacement pattern: the old binary goes aside, the new one takes
// its place, and the aside copy is cleaned up on the next start.
func TestReplaceBinaryMovesTheOldOneAside(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "gaia-tui.exe")
	staged := filepath.Join(dir, "gaia-tui.exe.download")
	mustWrite(t, target, []byte("old"))
	mustWrite(t, staged, []byte("new"))

	if err := ReplaceBinary(target, staged); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if got := readFile(t, target); got != "new" {
		t.Fatalf("target holds %q, want the new binary", got)
	}
	if _, err := os.Stat(staged); !os.IsNotExist(err) {
		t.Errorf("the staged file survived the swap (%v)", err)
	}

	backup := target + BackupSuffix
	if _, err := os.Stat(backup); err == nil {
		// Windows: the old image is kept aside because it may still be running.
		if got := readFile(t, backup); got != "old" {
			t.Errorf("the backup holds %q, want the old binary", got)
		}
		if err := CleanStaleBackup(target); err != nil {
			t.Fatalf("cleanup: %v", err)
		}
		if _, err := os.Stat(backup); !os.IsNotExist(err) {
			t.Errorf("the stale backup survived cleanup (%v)", err)
		}
	}
	// Cleanup with nothing to clean is not an error.
	if err := CleanStaleBackup(target); err != nil {
		t.Errorf("cleanup with no backup present: %v", err)
	}
}

// A first install has nothing at the target; the swap must still work and must
// not try to restore a backup that was never made.
func TestReplaceBinaryWithNoExistingTarget(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "gaia-agent.exe")
	staged := filepath.Join(dir, "gaia-agent.exe.download")
	mustWrite(t, staged, []byte("fresh"))

	if err := ReplaceBinary(target, staged); err != nil {
		t.Fatalf("replace: %v", err)
	}
	if got := readFile(t, target); got != "fresh" {
		t.Errorf("target holds %q, want the new binary", got)
	}
}

func TestFormatSize(t *testing.T) {
	cases := map[int64]string{
		0:          "unknown size",
		-1:         "unknown size",
		512:        "512 B",
		2048:       "2 KB",
		19_433_984: "18.5 MB",
		2 << 30:    "2.0 GB",
	}
	for in, want := range cases {
		if got := FormatSize(in); got != want {
			t.Errorf("FormatSize(%d) = %q, want %q", in, got, want)
		}
	}
}
