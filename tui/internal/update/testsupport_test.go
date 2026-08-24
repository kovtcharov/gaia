// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	urlpath "path"
	"path/filepath"
	"testing"
	"time"
)

// fakeRelease is one release the test feed publishes.
type fakeRelease struct {
	version        string
	tuiVersion     string
	sidecarVersion string
	tuiBody        []byte
	sidecarBody    []byte
	publishedAt    string
}

// fakeFeed is an httptest server speaking the generic feed contract. Nothing in
// these tests touches the live hub.
type fakeFeed struct {
	t        *testing.T
	server   *httptest.Server
	releases []fakeRelease
	// corruptTUI serves a body that does not match the lock's sha256, so the
	// integrity gate can be exercised end to end.
	corruptTUI bool
	// hits counts artifact downloads, to prove nothing was fetched on a decline.
	hits int
}

func newFakeFeed(t *testing.T, releases ...fakeRelease) *fakeFeed {
	t.Helper()
	f := &fakeFeed{t: t, releases: releases}
	mux := http.NewServeMux()
	mux.HandleFunc("/", f.handle)
	f.server = httptest.NewServer(mux)
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeFeed) URL() string { return f.server.URL }

func (f *fakeFeed) ref() FeedRef {
	return FeedRef{URL: f.server.URL, Kind: FeedKindGeneric, Source: "test"}
}

// latest is the newest release, which is the one "" resolves to.
func (f *fakeFeed) latest() fakeRelease { return f.releases[0] }

func (f *fakeFeed) find(version string) (fakeRelease, bool) {
	for _, r := range f.releases {
		if r.version == version {
			return r, true
		}
	}
	return fakeRelease{}, false
}

func (f *fakeFeed) handle(w http.ResponseWriter, r *http.Request) {
	// urlpath, not filepath: these are URL paths, and filepath.Base would split
	// on a backslash when the tests run on Windows.
	switch reqPath := r.URL.Path; {
	case reqPath == "/versions.json":
		type entry struct {
			Version     string `json:"version"`
			PublishedAt string `json:"publishedAt"`
		}
		doc := struct {
			Latest   string  `json:"latest"`
			Versions []entry `json:"versions"`
		}{Latest: f.latest().version}
		for _, rel := range f.releases {
			doc.Versions = append(doc.Versions, entry{rel.version, rel.publishedAt})
		}
		writeJSON(w, doc)

	case reqPath == "/"+LockFileName:
		writeJSON(w, f.lockFor(f.latest()))

	case urlpath.Base(reqPath) == LockFileName:
		version := urlpath.Base(urlpath.Dir(reqPath))
		rel, ok := f.find(version)
		if !ok {
			http.NotFound(w, r)
			return
		}
		writeJSON(w, f.lockFor(rel))

	default:
		version, name := urlpath.Base(urlpath.Dir(reqPath)), urlpath.Base(reqPath)
		rel, ok := f.find(version)
		if !ok {
			http.NotFound(w, r)
			return
		}
		f.hits++
		switch name {
		case "gaia-win-x64.exe":
			body := rel.tuiBody
			if f.corruptTUI {
				body = append(append([]byte(nil), body...), " tampered"...)
			}
			_, _ = w.Write(body)
		case "gaia-agent-win32-x64.exe":
			_, _ = w.Write(rel.sidecarBody)
		default:
			http.NotFound(w, r)
		}
	}
}

// lockFor builds the schemaVersion 3.0 manifest for one release, with the real
// SHA-256 of the bodies this server hands out.
func (f *fakeFeed) lockFor(rel fakeRelease) Lock {
	base := f.server.URL + "/" + rel.version
	return Lock{
		SchemaVersion: "3.0",
		AgentVersion:  rel.version,
		Components: map[string]Component{
			ComponentSidecar: {
				ComponentVersion: rel.sidecarVersion,
				BaseURL:          base,
				Platforms: map[string]Artifact{
					"win32-x64": {
						Filename:   "gaia-agent-win32-x64.exe",
						Executable: "gaia-agent.exe",
						SHA256:     sha256Hex(rel.sidecarBody),
						Size:       int64(len(rel.sidecarBody)),
					},
				},
			},
			ComponentTUI: {
				ComponentVersion: rel.tuiVersion,
				BaseURL:          base,
				Platforms: map[string]Artifact{
					"win32-x64": {
						Filename:   "gaia-win-x64.exe",
						Executable: "gaia-tui.exe",
						SHA256:     sha256Hex(rel.tuiBody),
						Size:       int64(len(rel.tuiBody)),
					},
				},
			},
		},
	}
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// --- machine fixtures --------------------------------------------------------

// fakeMachine is a temp ~/.gaia plus a TUI binary the updater may replace.
type fakeMachine struct {
	gaiaDir    string
	tuiPath    string
	sidecarDir string
}

func newFakeMachine(t *testing.T, tuiVersion, sidecarVersion string) *fakeMachine {
	t.Helper()
	root := t.TempDir()
	m := &fakeMachine{
		gaiaDir:    filepath.Join(root, ".gaia"),
		tuiPath:    filepath.Join(root, "bin", "gaia-tui.exe"),
		sidecarDir: filepath.Join(root, ".gaia", "agents", SidecarAgentID),
	}
	mustMkdir(t, filepath.Dir(m.tuiPath))
	mustWrite(t, m.tuiPath, []byte("tui "+tuiVersion))

	if sidecarVersion != "" {
		mustMkdir(t, m.sidecarDir)
		mustWrite(t, filepath.Join(m.sidecarDir, "gaia-agent.exe"), []byte("sidecar "+sidecarVersion))
		record, err := json.Marshal(installedRecord{
			ID: SidecarAgentID, Version: sidecarVersion, Executable: "gaia-agent.exe",
		})
		if err != nil {
			t.Fatalf("marshal sentinel: %v", err)
		}
		mustWrite(t, filepath.Join(m.sidecarDir, sentinelName), record)
	}
	return m
}

// updater builds an Updater pinned to win32-x64 so the same fixtures exercise
// the same lock entries on every CI runner.
func (m *fakeMachine) updater(t *testing.T, tuiVersion string, env map[string]string) *Updater {
	t.Helper()
	up, err := New(Options{
		GaiaDir:    m.gaiaDir,
		TUIVersion: tuiVersion,
		TUIPath:    m.tuiPath,
		SidecarDir: m.sidecarDir,
		GOOS:       "windows",
		GOARCH:     "amd64",
		Env:        func(k string) string { return env[k] },
		Client:     &http.Client{Timeout: 30 * time.Second},
		Now:        func() time.Time { return time.Date(2026, 8, 24, 12, 0, 0, 0, time.UTC) },
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return up
}

// setFeed points the machine's config at the test server.
func (m *fakeMachine) setFeed(t *testing.T, feed *fakeFeed) {
	t.Helper()
	cfg, err := LoadConfig(m.gaiaDir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	cfg.FeedURL = feed.URL()
	cfg.FeedKind = FeedKindGeneric
	if err := SaveConfig(m.gaiaDir, cfg); err != nil {
		t.Fatalf("SaveConfig: %v", err)
	}
}

func mustMkdir(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
}

func mustWrite(t *testing.T, path string, body []byte) {
	t.Helper()
	if err := os.WriteFile(path, body, 0o700); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return string(raw)
}

// --- npm registry fixture ----------------------------------------------------

// newFakeRegistry serves a package document and a real gzipped tarball with a
// binaries.lock.json inside, so the npm reader is exercised for real.
func newFakeRegistry(t *testing.T, lock Lock, versions []string) *httptest.Server {
	t.Helper()
	lockBytes, err := json.Marshal(lock)
	if err != nil {
		t.Fatalf("marshal lock: %v", err)
	}
	tarball := buildTarGz(t, "package/"+LockFileName, lockBytes)
	sum := sha512.Sum512(tarball)
	integrity := "sha512-" + base64.StdEncoding.EncodeToString(sum[:])

	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/pkg":
			times := map[string]string{}
			versionMap := map[string]map[string]string{}
			for _, v := range versions {
				versionMap[v] = map[string]string{"version": v}
				times[v] = "2026-08-01T00:00:00Z"
			}
			writeJSON(w, map[string]any{"versions": versionMap, "time": times})
		case "/pkg/tarball.tgz":
			_, _ = w.Write(tarball)
		default:
			// /pkg/latest and /pkg/<version> both answer the same document.
			writeJSON(w, map[string]any{
				"version": lock.AgentVersion,
				"dist": map[string]string{
					"tarball":   server.URL + "/pkg/tarball.tgz",
					"integrity": integrity,
				},
			})
		}
	})
	return server
}

func buildTarGz(t *testing.T, name string, body []byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	if err := tw.WriteHeader(&tar.Header{
		Name: name, Mode: 0o644, Size: int64(len(body)), Typeflag: tar.TypeReg,
	}); err != nil {
		t.Fatalf("tar header: %v", err)
	}
	if _, err := tw.Write(body); err != nil {
		t.Fatalf("tar body: %v", err)
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar close: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	return buf.Bytes()
}

func minimalLock(version string) Lock {
	return Lock{
		SchemaVersion: "3.0",
		AgentVersion:  version,
		Components: map[string]Component{
			ComponentTUI: {
				ComponentVersion: version,
				BaseURL:          "https://example.invalid/tui/" + version,
				Platforms: map[string]Artifact{
					"win32-x64": {
						Filename:   "gaia-win-x64.exe",
						Executable: "gaia-tui.exe",
						SHA256:     sha256Hex([]byte(version)),
						Size:       10,
					},
				},
			},
		},
	}
}

func fmtErr(err error) string {
	if err == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%v", err)
}
