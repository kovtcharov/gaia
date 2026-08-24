// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// Feed kinds. The kind is how a feed URL is READ, and it is always explicit —
// sniffing it would mean guessing, and a wrong guess reads as "no update
// available" on a feed that has one.
const (
	FeedKindNPM     = "npm"
	FeedKindGeneric = "generic"
)

// Environment overrides, matching the Agent UI's auto-updater where they overlap.
const (
	EnvFeedURL  = "GAIA_UPDATE_FEED_URL"
	EnvFeedKind = "GAIA_UPDATE_FEED_KIND"
	EnvDisable  = "GAIA_DISABLE_UPDATE"
)

// DefaultFeedURL is the built-in channel: the npm package that publishes
// binaries.lock.json for every release. Settable at build time with
// -ldflags "-X .../update.DefaultFeedURL=" — an OEM build that strips it lands
// in the loud no-channel state rather than checking AMD's channel.
var DefaultFeedURL = "https://registry.npmjs.org/@amd-gaia/gaia"

// DefaultFeedKind pairs with DefaultFeedURL.
var DefaultFeedKind = FeedKindNPM

// Release is one published version of the release train.
type Release struct {
	Version     string
	PublishedAt time.Time
}

// Feed reads published releases and their locks from one channel.
type Feed interface {
	// Lock returns binaries.lock.json for version, or for the newest release
	// when version is "".
	Lock(ctx context.Context, version string) (*Lock, error)
	// Versions lists published releases, newest first.
	Versions(ctx context.Context) ([]Release, error)
}

// FeedRef is a resolved channel: what it points at, how to read it, and which
// of the three sources won.
type FeedRef struct {
	URL    string
	Kind   string
	Source string
}

func (r FeedRef) String() string { return fmt.Sprintf("%s (%s, from %s)", r.URL, r.Kind, r.Source) }

// NoChannelError is the loud "no update channel configured" state.
//
// It is a distinct type so no caller can mistake it for "you are up to date".
// An updater with nowhere to look has checked nothing, and says so.
type NoChannelError struct {
	// ConfigPath is the file the user can set a feed in.
	ConfigPath string
}

func (e *NoChannelError) Error() string {
	return "no update channel configured — GAIA checked nothing.\n\n" +
		"Set one of:\n" +
		"  " + EnvFeedURL + "=<base url>   (this shell only)\n" +
		"  {\"feedUrl\": \"<base url>\"} in " + e.ConfigPath + "   (persistent)\n\n" +
		"Add " + EnvFeedKind + "=npm (or \"feedKind\") when the URL is an npm registry " +
		"package; the default for a URL you set yourself is \"generic\", meaning the " +
		"feed serves " + LockFileName + " directly.\n" +
		"Auto-update stays paused until a channel is configured."
}

// DisabledError is the kill switch, reported rather than silently obeyed when
// the user asked for an update explicitly.
type DisabledError struct{ Value string }

func (e *DisabledError) Error() string {
	return fmt.Sprintf(
		"updates are disabled: %s=%s is set in this environment, so nothing was "+
			"checked, downloaded, or replaced. Unset it to re-enable update checks.",
		EnvDisable, e.Value)
}

// IsDisabled reports the kill switch. Only "1" disables — an accidental
// GAIA_DISABLE_UPDATE=0 must not silently turn updates off.
func IsDisabled(env func(string) string) (bool, string) {
	v := strings.TrimSpace(env(EnvDisable))
	return v == "1", v
}

// ResolveFeedRef picks the channel: environment, then config, then the built-in
// default. A source that sets a URL also decides the kind, so a mirror URL from
// the environment never inherits the built-in default's npm reader.
func ResolveFeedRef(env func(string) string, cfg Config, gaiaDir string) (FeedRef, error) {
	if url := strings.TrimSpace(env(EnvFeedURL)); url != "" {
		kind, err := normalizeKind(env(EnvFeedKind), EnvFeedKind)
		if err != nil {
			return FeedRef{}, err
		}
		return FeedRef{URL: url, Kind: kind, Source: EnvFeedURL}, nil
	}
	if url := strings.TrimSpace(cfg.FeedURL); url != "" {
		kind, err := normalizeKind(cfg.FeedKind, `"feedKind" in `+ConfigPath(gaiaDir))
		if err != nil {
			return FeedRef{}, err
		}
		return FeedRef{URL: url, Kind: kind, Source: ConfigPath(gaiaDir)}, nil
	}
	if url := strings.TrimSpace(DefaultFeedURL); url != "" {
		return FeedRef{URL: url, Kind: DefaultFeedKind, Source: "built-in default"}, nil
	}
	return FeedRef{}, &NoChannelError{ConfigPath: ConfigPath(gaiaDir)}
}

// normalizeKind validates an explicitly-set kind. An unset kind is "generic":
// a URL a user points at is a static mirror unless they say otherwise, and the
// choice is printed by `update status` so it is never a hidden assumption.
func normalizeKind(raw, source string) (string, error) {
	switch kind := strings.ToLower(strings.TrimSpace(raw)); kind {
	case "":
		return FeedKindGeneric, nil
	case FeedKindNPM, FeedKindGeneric:
		return kind, nil
	default:
		return "", fmt.Errorf(
			"%s is %q, which is not a feed kind this build understands. Use %q (the "+
				"feed serves %s directly) or %q (the URL is an npm registry package "+
				"document)", source, raw, FeedKindGeneric, LockFileName, FeedKindNPM)
	}
}
