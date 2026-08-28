// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// A fresh machine has no config, which is a normal state, not a failure.
func TestLoadConfigOnAFreshMachine(t *testing.T) {
	cfg, err := LoadConfig(t.TempDir())
	if err != nil {
		t.Fatalf("a missing config errored: %v", err)
	}
	if cfg != (Config{}) {
		t.Errorf("a missing config produced %+v, want the zero value", cfg)
	}
}

// A corrupt config must NOT read as "unset" — that would silently discard a pin
// and resume updates the user had paused.
func TestLoadConfigRefusesCorruptJSON(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, ConfigPath(dir), []byte("{not json"))

	_, err := LoadConfig(dir)
	if err == nil {
		t.Fatal("a corrupt config was read as unset")
	}
	if !strings.Contains(err.Error(), ConfigPath(dir)) {
		t.Errorf("the failure does not name the file:\n%v", err)
	}
	if !strings.Contains(err.Error(), "pin") {
		t.Errorf("the failure does not explain what is at stake:\n%v", err)
	}
}

func TestSaveConfigRoundTrips(t *testing.T) {
	dir := t.TempDir()
	want := Config{
		FeedURL:       "https://feed.test",
		FeedKind:      FeedKindGeneric,
		PinnedVersion: "0.1.0",
		LastCheck:     "2026-08-24T12:00:00Z",
	}
	if err := SaveConfig(dir, want); err != nil {
		t.Fatalf("save: %v", err)
	}
	got, err := LoadConfig(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got != want {
		t.Errorf("round trip produced %+v, want %+v", got, want)
	}
}

// Clearing a pin deletes the key rather than storing "" — the Agent UI reads
// the same file, and an absent key is unambiguous in both.
func TestSaveConfigDeletesClearedFields(t *testing.T) {
	dir := t.TempDir()
	if err := SaveConfig(dir, Config{PinnedVersion: "0.1.0", FeedURL: "https://feed.test"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if err := SaveConfig(dir, Config{FeedURL: "https://feed.test"}); err != nil {
		t.Fatalf("save: %v", err)
	}

	raw, err := os.ReadFile(ConfigPath(dir))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var stored map[string]any
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if _, present := stored["pinnedVersion"]; present {
		t.Errorf("a cleared pin was stored rather than deleted: %s", raw)
	}
	if stored["feedUrl"] != "https://feed.test" {
		t.Errorf("clearing the pin clobbered the feed URL: %s", raw)
	}
}

// The Agent UI writes this file too, and a newer build may store keys this one
// has never heard of. Dropping them on write would break the other surface.
func TestSaveConfigPreservesUnknownKeys(t *testing.T) {
	dir := t.TempDir()
	mustWrite(t, ConfigPath(dir), []byte(`{"pinnedVersion":"0.1.0","releaseChannel":"beta"}`))

	if err := SaveConfig(dir, Config{PinnedVersion: "0.2.0"}); err != nil {
		t.Fatalf("save: %v", err)
	}
	raw, err := os.ReadFile(ConfigPath(dir))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var stored map[string]any
	if err := json.Unmarshal(raw, &stored); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if stored["releaseChannel"] != "beta" {
		t.Errorf("a key this build does not know was dropped: %s", raw)
	}
	if stored["pinnedVersion"] != "0.2.0" {
		t.Errorf("the pin was not updated: %s", raw)
	}
}

func TestCompareSemver(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"0.2.0", "0.1.0", 1},
		{"0.1.0", "0.2.0", -1},
		{"0.1.0", "0.1.0", 0},
		{"1.0.0", "1.0.0-rc.1", 1},
		{"1.0.0-rc.1", "1.0.0", -1},
		{"v0.24.0", "0.23.0", 1},
		{"0.10.0", "0.9.0", 1},
		{"1.2.3+build", "1.2.3", 0},
		// A "dev" build must still be told a published release exists.
		{"0.1.0", "dev", 1},
	}
	for _, c := range cases {
		if got := CompareSemver(c.a, c.b); got != c.want {
			t.Errorf("CompareSemver(%q, %q) = %d, want %d", c.a, c.b, got, c.want)
		}
	}
}
