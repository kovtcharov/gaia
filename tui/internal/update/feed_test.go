// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"
	"time"
)

// The channel is resolved at runtime, in a fixed order: env, then config, then
// the built-in default. Each source also decides how its URL is read.
func TestResolveFeedRefPrecedence(t *testing.T) {
	gaiaDir := t.TempDir()

	env := map[string]string{EnvFeedURL: "https://env.test/feed"}
	cfg := Config{FeedURL: "https://config.test/feed", FeedKind: FeedKindNPM}

	ref, err := ResolveFeedRef(func(k string) string { return env[k] }, cfg, gaiaDir)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if ref.URL != "https://env.test/feed" || ref.Source != EnvFeedURL {
		t.Errorf("the environment did not win: %+v", ref)
	}
	if ref.Kind != FeedKindGeneric {
		t.Errorf("a URL set in the environment inherited the config's npm kind: %+v", ref)
	}

	ref, err = ResolveFeedRef(func(string) string { return "" }, cfg, gaiaDir)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if ref.URL != "https://config.test/feed" || ref.Kind != FeedKindNPM {
		t.Errorf("the config feed was not used: %+v", ref)
	}
	if ref.Source != ConfigPath(gaiaDir) {
		t.Errorf("the config source is not named: %+v", ref)
	}

	ref, err = ResolveFeedRef(func(string) string { return "" }, Config{}, gaiaDir)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if ref.URL != DefaultFeedURL || ref.Source != "built-in default" {
		t.Errorf("the built-in default was not used: %+v", ref)
	}
}

// A feed kind nobody can read is refused up front rather than sniffed.
func TestResolveFeedRefRejectsUnknownKind(t *testing.T) {
	gaiaDir := t.TempDir()
	env := map[string]string{EnvFeedURL: "https://env.test/feed", EnvFeedKind: "torrent"}

	_, err := ResolveFeedRef(func(k string) string { return env[k] }, Config{}, gaiaDir)
	if err == nil {
		t.Fatal("an unreadable feed kind was accepted")
	}
	for _, want := range []string{"torrent", FeedKindGeneric, FeedKindNPM} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal never mentions %q:\n%v", want, err)
		}
	}
}

func TestGenericFeedReadsLocksAndVersions(t *testing.T) {
	feed := newFakeFeed(t, twoReleases()...)
	reader, err := NewFeed(feed.ref(), &http.Client{Timeout: 10 * time.Second})
	if err != nil {
		t.Fatalf("NewFeed: %v", err)
	}
	ctx := context.Background()

	lock, err := reader.Lock(ctx, "")
	if err != nil {
		t.Fatalf("latest lock: %v", err)
	}
	if lock.AgentVersion != "0.2.0" {
		t.Errorf("the latest lock is for %q, want 0.2.0", lock.AgentVersion)
	}

	lock, err = reader.Lock(ctx, "0.1.0")
	if err != nil {
		t.Fatalf("versioned lock: %v", err)
	}
	if lock.AgentVersion != "0.1.0" {
		t.Errorf("the versioned lock is for %q, want 0.1.0", lock.AgentVersion)
	}

	releases, err := reader.Versions(ctx)
	if err != nil {
		t.Fatalf("versions: %v", err)
	}
	if len(releases) != 2 || releases[0].Version != "0.2.0" {
		t.Fatalf("versions are not newest-first: %+v", releases)
	}
	if releases[0].PublishedAt.IsZero() {
		t.Error("the publish date was dropped")
	}
}

// The npm reader has to pull the lock out of the published tarball, and verify
// the tarball against the registry's own digest before opening it.
func TestNPMFeedReadsLockFromTarball(t *testing.T) {
	want := minimalLock("0.1.1")
	server := newFakeRegistry(t, want, []string{"0.1.0", "0.1.1"})

	reader, err := NewFeed(FeedRef{URL: server.URL + "/pkg", Kind: FeedKindNPM}, server.Client())
	if err != nil {
		t.Fatalf("NewFeed: %v", err)
	}
	ctx := context.Background()

	lock, err := reader.Lock(ctx, "")
	if err != nil {
		t.Fatalf("lock: %v", err)
	}
	if lock.AgentVersion != "0.1.1" {
		t.Errorf("read agentVersion %q, want 0.1.1", lock.AgentVersion)
	}
	if _, _, err := lock.Resolve(ComponentTUI, "win32-x64"); err != nil {
		t.Errorf("the extracted lock does not resolve the TUI artifact: %v", err)
	}

	releases, err := reader.Versions(ctx)
	if err != nil {
		t.Fatalf("versions: %v", err)
	}
	if len(releases) != 2 || releases[0].Version != "0.1.1" {
		t.Fatalf("versions are not newest-first: %+v", releases)
	}
}

// A lock with an unpublished placeholder hash blocks the download outright —
// an unverifiable binary can never be trusted.
func TestPlaceholderHashBlocksResolve(t *testing.T) {
	lock := minimalLock("0.1.1")
	entry := lock.Components[ComponentTUI].Platforms["win32-x64"]
	entry.SHA256 = "PENDING-replace-with-real-sha256"
	lock.Components[ComponentTUI].Platforms["win32-x64"] = entry
	lock.Source = "test lock"

	_, _, err := lock.Resolve(ComponentTUI, "win32-x64")
	if err == nil {
		t.Fatal("a placeholder sha256 was accepted")
	}
	if !strings.Contains(err.Error(), "PENDING") || !strings.Contains(err.Error(), "Nothing is downloaded") {
		t.Errorf("the refusal is not actionable:\n%v", err)
	}
}

// A schema this build cannot read is refused rather than parsed on a guess.
func TestParseLockRejectsForeignSchema(t *testing.T) {
	_, err := ParseLock([]byte(`{"schemaVersion":"2.0","agentVersion":"1","components":{}}`), "test")
	if err == nil {
		t.Fatal("a schemaVersion 2.0 lock was accepted")
	}
	if !strings.Contains(err.Error(), "3.x") {
		t.Errorf("the refusal does not name the schema this build reads:\n%v", err)
	}
}

// A feed that answers with an HTML error page must be diagnosed as that, not
// reported as "no update available".
func TestParseLockNamesWhatTheFeedActuallyServed(t *testing.T) {
	_, err := ParseLock([]byte("<html>404 Not Found</html>"), "https://feed.test/binaries.lock.json")
	if err == nil {
		t.Fatal("an HTML body parsed as a lock")
	}
	if !strings.Contains(err.Error(), "https://feed.test/binaries.lock.json") {
		t.Errorf("the failure does not name the URL:\n%v", err)
	}
}

// A platform the release does not publish for is a named refusal, so a user on
// unsupported hardware is told why rather than left waiting.
func TestResolveNamesTheMissingPlatform(t *testing.T) {
	lock := minimalLock("0.1.1")
	lock.Source = "test lock"
	_, _, err := lock.Resolve(ComponentTUI, "linux-arm64")
	if err == nil {
		t.Fatal("a platform the lock does not publish for resolved")
	}
	if !strings.Contains(err.Error(), "linux-arm64") || !strings.Contains(err.Error(), "win32-x64") {
		t.Errorf("the refusal names neither the missing platform nor what is published:\n%v", err)
	}
}

func TestNodePlatformKey(t *testing.T) {
	cases := map[[2]string]string{
		{"windows", "amd64"}: "win32-x64",
		{"windows", "arm64"}: "win32-arm64",
		{"darwin", "arm64"}:  "darwin-arm64",
		{"linux", "amd64"}:   "linux-x64",
	}
	for in, want := range cases {
		got, err := nodePlatformKey(in[0], in[1])
		if err != nil {
			t.Fatalf("%v: %v", in, err)
		}
		if got != want {
			t.Errorf("%v produced %q, want %q", in, got, want)
		}
	}
	if _, err := nodePlatformKey("plan9", "amd64"); err == nil {
		t.Error("an unpublished OS resolved to a platform key")
	}
}

func TestIsDisabledOnlyOnOne(t *testing.T) {
	for value, want := range map[string]bool{"1": true, "0": false, "": false, "true": false, " 1 ": true} {
		got, _ := IsDisabled(func(string) string { return value })
		if got != want {
			t.Errorf("GAIA_DISABLE_UPDATE=%q reported disabled=%v, want %v", value, got, want)
		}
	}
}

// A caller must never confuse "nowhere to look" with "nothing to install".
func TestNoChannelErrorIsDistinct(t *testing.T) {
	var err error = &NoChannelError{ConfigPath: "/home/jane/.gaia/update-config.json"}
	var target *NoChannelError
	if !errors.As(err, &target) {
		t.Fatal("NoChannelError is not recoverable with errors.As")
	}
	if !strings.Contains(err.Error(), "checked nothing") {
		t.Errorf("the no-channel message does not say that nothing was checked:\n%v", err)
	}
}
