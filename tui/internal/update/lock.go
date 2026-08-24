// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Component names in binaries.lock.json. Both are updatable: the sidecar is the
// agent process the TUI talks to, so a stale one is as broken as a stale TUI.
const (
	ComponentTUI     = "tui"
	ComponentSidecar = "sidecar"
)

// Components is the update order — the sidecar first, so a TUI that swaps
// itself last is never the thing that aborts a half-applied release.
var Components = []string{ComponentSidecar, ComponentTUI}

// LockSchemaMajor is the binaries.lock.json schema this build reads. Anything
// else is refused rather than parsed on a guess.
const LockSchemaMajor = 3

// LockFileName is the manifest's name inside a feed (and inside the npm tarball).
const LockFileName = "binaries.lock.json"

// Artifact is one component+platform entry of binaries.lock.json.
type Artifact struct {
	// Filename is the artifact as published under the component's base URL.
	Filename string `json:"filename"`
	// Executable is the basename it is written as on disk.
	Executable string `json:"executable"`
	// SHA256 is the lowercase hex digest the download must match.
	SHA256 string `json:"sha256"`
	// Size is informational — it is what the prompt quotes before downloading.
	Size int64 `json:"size"`
}

// Component is one lane of binaries.lock.json: where it is published, at what
// version, for which platforms.
type Component struct {
	ComponentVersion string              `json:"componentVersion"`
	BaseURL          string              `json:"baseUrl"`
	Platforms        map[string]Artifact `json:"platforms"`
}

// Lock is binaries.lock.json, schemaVersion 3.x.
type Lock struct {
	SchemaVersion string               `json:"schemaVersion"`
	AgentVersion  string               `json:"agentVersion"`
	Components    map[string]Component `json:"components"`

	// Source is where this lock was read from, for error messages. Not serialized.
	Source string `json:"-"`
}

// ParseLock decodes and validates a lock file. source names where the bytes
// came from so every refusal below can point at it.
func ParseLock(raw []byte, source string) (*Lock, error) {
	var lock Lock
	if err := json.Unmarshal(raw, &lock); err != nil {
		return nil, fmt.Errorf(
			"%s is not valid JSON: %w. The update feed served something that is not a "+
				"%s — check the feed URL with `gaia-tui update status`", source, err, LockFileName)
	}
	lock.Source = source

	major, err := majorVersion(lock.SchemaVersion)
	if err != nil {
		return nil, fmt.Errorf(
			"%s declares schemaVersion %q, which this build cannot read: %w. Expected "+
				"a %d.x lock", source, lock.SchemaVersion, err, LockSchemaMajor)
	}
	if major != LockSchemaMajor {
		return nil, fmt.Errorf(
			"%s declares schemaVersion %q but this build reads %d.x only. Update GAIA "+
				"from https://amd-gaia.ai/docs/guides/install, or point "+
				"GAIA_UPDATE_FEED_URL at a feed that publishes a %d.x lock",
			source, lock.SchemaVersion, LockSchemaMajor, LockSchemaMajor)
	}
	if lock.AgentVersion == "" {
		return nil, fmt.Errorf(
			"%s has no \"agentVersion\", so there is no release version to compare "+
				"against or to pin. The feed is publishing an incomplete lock", source)
	}
	if len(lock.Components) == 0 {
		return nil, fmt.Errorf(
			"%s has no \"components\" map, so there is nothing to download. The feed "+
				"is publishing an incomplete lock", source)
	}
	return &lock, nil
}

// Resolve returns the artifact for one component on one platform.
func (l *Lock) Resolve(component, platformKey string) (Artifact, string, error) {
	comp, ok := l.Components[component]
	if !ok {
		return Artifact{}, "", fmt.Errorf(
			"%s has no %q component (it publishes: %s), so there is nothing to update "+
				"for it", l.Source, component, strings.Join(sortedKeys(l.Components), ", "))
	}
	if comp.BaseURL == "" {
		return Artifact{}, "", fmt.Errorf(
			"%s has no baseUrl for %q, so there is nowhere to download it from. The "+
				"feed is publishing an incomplete lock", l.Source, component)
	}
	entry, ok := comp.Platforms[platformKey]
	if !ok {
		return Artifact{}, "", fmt.Errorf(
			"%s publishes no %q build for this machine (%s). Platforms in the lock: %s. "+
				"Build from source instead — see https://amd-gaia.ai/docs/reference/dev",
			l.Source, component, platformKey, strings.Join(sortedKeys(comp.Platforms), ", "))
	}
	if entry.Filename == "" || entry.Executable == "" {
		return Artifact{}, "", fmt.Errorf(
			"%s: the %s/%s entry is missing filename or executable, so the download "+
				"cannot be named. The feed is publishing an incomplete lock",
			l.Source, component, platformKey)
	}
	if err := validateSHA256(entry.SHA256); err != nil {
		return Artifact{}, "", fmt.Errorf(
			"%s: the %s/%s entry cannot be verified — %w. Nothing is downloaded, "+
				"because a binary whose hash is unknown can never be trusted",
			l.Source, component, platformKey, err)
	}
	return entry, comp.BaseURL, nil
}

var sha256Re = regexp.MustCompile(`^[0-9a-f]{64}$`)

// validateSHA256 rejects both a malformed digest and the "PENDING-…"
// placeholder an unpublished lock carries.
func validateSHA256(digest string) error {
	if digest == "" {
		return fmt.Errorf("it has no sha256")
	}
	lower := strings.ToLower(digest)
	if !sha256Re.MatchString(lower) {
		return fmt.Errorf("its sha256 %q is not a 64-character hex digest "+
			"(an unpublished lock carries a PENDING placeholder here)", digest)
	}
	return nil
}

// nodePlatformKey maps Go's GOOS/GOARCH onto the lock's Node-flavoured keys.
//
// The lock is written by the npm package, so its keys come from
// `process.platform`/`process.arch`: "win32", not "windows"; "x64", not "amd64".
func nodePlatformKey(goos, goarch string) (string, error) {
	osName, ok := map[string]string{
		"windows": "win32",
		"darwin":  "darwin",
		"linux":   "linux",
	}[goos]
	if !ok {
		return "", fmt.Errorf(
			"GAIA publishes no binaries for %s, so there is nothing to update to on "+
				"this machine. Build from source — https://amd-gaia.ai/docs/reference/dev", goos)
	}
	archName, ok := map[string]string{
		"amd64": "x64",
		"arm64": "arm64",
	}[goarch]
	if !ok {
		return "", fmt.Errorf(
			"GAIA publishes no binaries for the %s architecture, so there is nothing "+
				"to update to on this machine. Build from source — "+
				"https://amd-gaia.ai/docs/reference/dev", goarch)
	}
	return osName + "-" + archName, nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func majorVersion(v string) (int, error) {
	head, _, _ := strings.Cut(strings.TrimPrefix(v, "v"), ".")
	if head == "" {
		return 0, fmt.Errorf("it has no leading major number")
	}
	n, err := strconv.Atoi(head)
	if err != nil {
		return 0, fmt.Errorf("%q is not a number", head)
	}
	return n, nil
}
