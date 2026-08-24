// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// maxManifestBytes caps every JSON body this package reads. A registry
// packument for a busy package is large; a feed that answers with something
// unbounded is a bug, not a body to buffer.
const maxManifestBytes = 32 << 20

// userAgent identifies the updater in feed access logs.
const userAgent = "gaia-tui-updater"

// NewFeed builds the reader for a resolved channel.
func NewFeed(ref FeedRef, client *http.Client) (Feed, error) {
	base := strings.TrimRight(ref.URL, "/")
	switch ref.Kind {
	case FeedKindGeneric:
		return &genericFeed{base: base, client: client}, nil
	case FeedKindNPM:
		return &npmFeed{base: base, client: client}, nil
	default:
		return nil, fmt.Errorf(
			"feed kind %q is not one this build can read (%q or %q) — reported by "+
				"`gaia-tui update status`", ref.Kind, FeedKindGeneric, FeedKindNPM)
	}
}

// getJSON performs one GET and returns the body, with the URL named in every
// failure so a misconfigured feed is diagnosable from the message alone.
func getJSON(ctx context.Context, client *http.Client, url, what string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("cannot build the request for %s (%s): %w", what, url, err)
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Accept", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf(
			"cannot reach the update feed to read %s (%s): %w. Check your network, "+
				"or point GAIA_UPDATE_FEED_URL at a feed you can reach", what, url, err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"the update feed answered HTTP %d for %s (%s). Confirm the feed URL with "+
				"`gaia-tui update status`", resp.StatusCode, what, url)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxManifestBytes))
	if err != nil {
		return nil, fmt.Errorf("cannot read %s from %s: %w", what, url, err)
	}
	return body, nil
}

// --- generic feed -----------------------------------------------------------

// genericFeed is a static mirror. Its contract is three paths:
//
//	<base>/binaries.lock.json             the newest release's lock
//	<base>/<version>/binaries.lock.json   one specific release's lock
//	<base>/versions.json                  {"versions": [{"version", "publishedAt"}]}
type genericFeed struct {
	base   string
	client *http.Client
}

func (f *genericFeed) lockURL(version string) string {
	if version == "" {
		return f.base + "/" + LockFileName
	}
	return f.base + "/" + version + "/" + LockFileName
}

func (f *genericFeed) Lock(ctx context.Context, version string) (*Lock, error) {
	url := f.lockURL(version)
	label := LockFileName
	if version != "" {
		label = fmt.Sprintf("%s for %s", LockFileName, version)
	}
	body, err := getJSON(ctx, f.client, url, label)
	if err != nil {
		return nil, err
	}
	return ParseLock(body, url)
}

// versionsDoc is <base>/versions.json.
type versionsDoc struct {
	Versions []struct {
		Version     string `json:"version"`
		PublishedAt string `json:"publishedAt"`
	} `json:"versions"`
}

func (f *genericFeed) Versions(ctx context.Context) ([]Release, error) {
	url := f.base + "/versions.json"
	body, err := getJSON(ctx, f.client, url, "the published version list")
	if err != nil {
		return nil, err
	}
	doc, err := decodeJSON[versionsDoc](body, url)
	if err != nil {
		return nil, err
	}
	releases := make([]Release, 0, len(doc.Versions))
	for _, v := range doc.Versions {
		if v.Version == "" {
			continue
		}
		releases = append(releases, Release{Version: v.Version, PublishedAt: parseTime(v.PublishedAt)})
	}
	if len(releases) == 0 {
		return nil, fmt.Errorf(
			"%s lists no versions, so there is nothing to install or roll back to. "+
				"The feed is published but empty", url)
	}
	sortReleases(releases)
	return releases, nil
}

// --- npm feed ---------------------------------------------------------------

// npmFeed reads a registry package document. The lock lives inside the
// published tarball, so reading it means fetching and unpacking that tarball —
// after verifying it against the digest the registry publishes for it.
type npmFeed struct {
	base   string
	client *http.Client
}

type npmVersionDoc struct {
	Version string `json:"version"`
	Dist    struct {
		Tarball   string `json:"tarball"`
		Integrity string `json:"integrity"`
		Shasum    string `json:"shasum"`
	} `json:"dist"`
}

// npmPackument is the whole-package document. Only the version KEYS and the
// publish times are read, so the per-version objects stay undecoded.
type npmPackument struct {
	Versions map[string]json.RawMessage `json:"versions"`
	Time     map[string]string          `json:"time"`
}

func (f *npmFeed) Lock(ctx context.Context, version string) (*Lock, error) {
	target := version
	if target == "" {
		target = "latest"
	}
	url := f.base + "/" + target
	body, err := getJSON(ctx, f.client, url, fmt.Sprintf("the %s package document", target))
	if err != nil {
		return nil, err
	}
	doc, err := decodeJSON[npmVersionDoc](body, url)
	if err != nil {
		return nil, err
	}
	if doc.Dist.Tarball == "" {
		return nil, fmt.Errorf(
			"%s has no dist.tarball, so there is nowhere to read %s from. The registry "+
				"answered a package document this build cannot use", url, LockFileName)
	}

	raw, err := f.fetchTarball(ctx, doc)
	if err != nil {
		return nil, err
	}
	lockBytes, err := extractFromTarGz(raw, "package/"+LockFileName, doc.Dist.Tarball)
	if err != nil {
		return nil, err
	}
	return ParseLock(lockBytes, fmt.Sprintf("%s (from %s)", LockFileName, doc.Dist.Tarball))
}

// fetchTarball downloads the published tarball and verifies it against what the
// registry says it should be. A tarball nothing vouches for is refused: the
// lock inside it is the root of every SHA-256 check that follows.
func (f *npmFeed) fetchTarball(ctx context.Context, doc npmVersionDoc) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, doc.Dist.Tarball, nil)
	if err != nil {
		return nil, fmt.Errorf("cannot build the request for %s: %w", doc.Dist.Tarball, err)
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := f.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf(
			"cannot download the release manifest %s: %w. Check your network and retry",
			doc.Dist.Tarball, err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"the registry answered HTTP %d for the release manifest %s",
			resp.StatusCode, doc.Dist.Tarball)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxManifestBytes))
	if err != nil {
		return nil, fmt.Errorf("cannot read %s: %w", doc.Dist.Tarball, err)
	}
	if err := verifyRegistryDigest(raw, doc.Dist.Integrity, doc.Dist.Shasum, doc.Dist.Tarball); err != nil {
		return nil, err
	}
	return raw, nil
}

func (f *npmFeed) Versions(ctx context.Context) ([]Release, error) {
	body, err := getJSON(ctx, f.client, f.base, "the published version list")
	if err != nil {
		return nil, err
	}
	doc, err := decodeJSON[npmPackument](body, f.base)
	if err != nil {
		return nil, err
	}
	if len(doc.Versions) == 0 {
		return nil, fmt.Errorf(
			"%s lists no published versions, so there is nothing to install or roll "+
				"back to", f.base)
	}
	releases := make([]Release, 0, len(doc.Versions))
	for v := range doc.Versions {
		releases = append(releases, Release{Version: v, PublishedAt: parseTime(doc.Time[v])})
	}
	sortReleases(releases)
	return releases, nil
}

// --- shared helpers ---------------------------------------------------------

func sortReleases(releases []Release) {
	sort.Slice(releases, func(i, j int) bool {
		if c := CompareSemver(releases[i].Version, releases[j].Version); c != 0 {
			return c > 0
		}
		return releases[i].Version > releases[j].Version
	})
}

// parseTime returns the zero time for a timestamp the feed did not publish.
// Callers render that as "-": an invented date is worse than an absent one.
func parseTime(raw string) time.Time {
	if raw == "" {
		return time.Time{}
	}
	t, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return time.Time{}
	}
	return t
}
