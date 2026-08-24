// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"time"
)

// ConfigFileName is the on-disk update state, shared with the Agent UI's
// auto-updater: both read `feedUrl` and `pinnedVersion` out of the same file so
// pinning in one surface is visible in the other.
const ConfigFileName = "update-config.json"

// Config is ~/.gaia/update-config.json.
//
// Every field is optional. Absent and empty mean the same thing — "not set" —
// which is why writes delete a key rather than storing "".
type Config struct {
	// FeedURL is the base URL updates are resolved from. Overridden by
	// GAIA_UPDATE_FEED_URL.
	FeedURL string `json:"feedUrl,omitempty"`
	// FeedKind is how that URL is read: "npm" (a registry package document) or
	// "generic" (a static mirror). Overridden by GAIA_UPDATE_FEED_KIND.
	FeedKind string `json:"feedKind,omitempty"`
	// PinnedVersion pauses auto-update at one release until it is cleared.
	PinnedVersion string `json:"pinnedVersion,omitempty"`
	// SkippedVersion is the release the user declined for good, so the prompt
	// does not re-ask about the same one.
	SkippedVersion string `json:"skippedVersion,omitempty"`
	// LastCheck is the RFC3339 timestamp of the last completed check.
	LastCheck string `json:"lastCheck,omitempty"`
	// LastSeenVersion is the release the last check found available.
	LastSeenVersion string `json:"lastSeenVersion,omitempty"`
	// InstalledTUIVersion is the version of the binary ON DISK. It differs from
	// the running one between a swap and the next start, and without it a second
	// install would re-download a TUI that is already in place.
	InstalledTUIVersion string `json:"installedTuiVersion,omitempty"`
}

// ConfigPath returns the config file inside a ~/.gaia root.
func ConfigPath(gaiaDir string) string { return filepath.Join(gaiaDir, ConfigFileName) }

// LoadConfig reads the update config.
//
// A missing file is the normal fresh-machine state and returns a zero Config.
// A file that exists but cannot be read or parsed is an error naming the path:
// treating it as "unset" would silently discard a pin and let auto-update
// resume against the user's stated wishes.
func LoadConfig(gaiaDir string) (Config, error) {
	path := ConfigPath(gaiaDir)
	raw, err := os.ReadFile(path) // #nosec G304 -- path is derived from the user's home dir
	if errors.Is(err, fs.ErrNotExist) {
		return Config{}, nil
	}
	if err != nil {
		return Config{}, fmt.Errorf(
			"cannot read the update config at %s: %w. Fix the permissions on that "+
				"file, or delete it to start from defaults", path, err)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return Config{}, fmt.Errorf(
			"the update config at %s is not valid JSON: %w. Delete the file to start "+
				"from defaults, or repair it — a version pin stored there is why "+
				"auto-update may be paused", path, err)
	}
	return cfg, nil
}

// SaveConfig writes cfg, preserving any keys this build does not know about.
//
// The Agent UI writes the same file, and a newer one may store fields this
// binary has never heard of. Round-tripping through a struct would drop them,
// so the merge happens over the raw object.
func SaveConfig(gaiaDir string, cfg Config) error {
	path := ConfigPath(gaiaDir)
	merged := map[string]json.RawMessage{}

	existing, err := os.ReadFile(path) // #nosec G304 -- path is derived from the user's home dir
	switch {
	case errors.Is(err, fs.ErrNotExist):
		// First write.
	case err != nil:
		return fmt.Errorf("cannot read the update config at %s before writing it: %w", path, err)
	default:
		if err := json.Unmarshal(existing, &merged); err != nil {
			return fmt.Errorf(
				"the update config at %s is not valid JSON, so it cannot be updated "+
					"without losing what is in it: %w. Delete the file and re-run", path, err)
		}
	}

	fields := []struct {
		key   string
		value string
	}{
		{"feedUrl", cfg.FeedURL},
		{"feedKind", cfg.FeedKind},
		{"pinnedVersion", cfg.PinnedVersion},
		{"skippedVersion", cfg.SkippedVersion},
		{"lastCheck", cfg.LastCheck},
		{"lastSeenVersion", cfg.LastSeenVersion},
		{"installedTuiVersion", cfg.InstalledTUIVersion},
	}
	for _, f := range fields {
		if f.value == "" {
			// Deleted, not stored as "": a `""` pin read back by the Agent UI's
			// updater is falsy there too, but an absent key is unambiguous.
			delete(merged, f.key)
			continue
		}
		encoded, err := json.Marshal(f.value)
		if err != nil {
			return fmt.Errorf("cannot encode %s for the update config: %w", f.key, err)
		}
		merged[f.key] = encoded
	}

	payload, err := json.MarshalIndent(merged, "", "  ")
	if err != nil {
		return fmt.Errorf("cannot encode the update config: %w", err)
	}
	payload = append(payload, '\n')

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return fmt.Errorf("cannot create %s: %w", filepath.Dir(path), err)
	}
	tmp := filepath.Join(filepath.Dir(path), fmt.Sprintf(".%s.%d.tmp", ConfigFileName, os.Getpid()))
	if err := os.Remove(tmp); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("cannot clear the stale temp file %s: %w", tmp, err)
	}
	if err := os.WriteFile(tmp, payload, 0o600); err != nil {
		return fmt.Errorf("cannot write %s: %w", tmp, err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("cannot install %s: %w", path, err)
	}
	return nil
}

// FormatTime renders a timestamp the way the config stores it.
func FormatTime(t time.Time) string { return t.UTC().Format(time.RFC3339) }
