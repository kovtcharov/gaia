// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/sha1" // #nosec G505 -- compared against the registry's own shasum field, not used as a security primitive on its own
	"crypto/sha512"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// maxLockBytes caps the manifest extracted from a tarball. The real file is a
// few KB; anything approaching this is a decompression bomb, not a lock.
const maxLockBytes = 4 << 20

// decodeJSON parses one document, naming the source in the failure so a feed
// serving an HTML error page is diagnosed as that rather than as "no update".
func decodeJSON[T any](raw []byte, source string) (T, error) {
	var out T
	if err := json.Unmarshal(raw, &out); err != nil {
		preview := strings.TrimSpace(string(raw))
		if len(preview) > 120 {
			preview = preview[:120] + "…"
		}
		return out, fmt.Errorf(
			"%s did not return the JSON this build expects: %w. It answered: %s",
			source, err, preview)
	}
	return out, nil
}

// verifyRegistryDigest checks a downloaded tarball against the digest its
// registry publishes for it.
//
// `integrity` (sha512) is preferred; `shasum` (sha1) is what older registry
// documents carry. When neither is present the download is REFUSED — the lock
// inside this tarball is the root of trust for every binary hash that follows,
// so an unvouched-for tarball can never be opened.
func verifyRegistryDigest(raw []byte, integrity, shasum, source string) error {
	if integrity != "" {
		return verifyIntegrity(raw, integrity, source)
	}
	if shasum != "" {
		sum := sha1.Sum(raw) // #nosec G401 -- the registry publishes sha1 here; this compares against it
		actual := hex.EncodeToString(sum[:])
		if !strings.EqualFold(actual, shasum) {
			return fmt.Errorf(
				"the release manifest downloaded from %s does not match the registry's "+
					"own shasum:\n  expected %s\n  actual   %s\nRefusing to read it. Retry; "+
					"if it persists, report it at https://github.com/amd/gaia/issues",
				source, shasum, actual)
		}
		return nil
	}
	return fmt.Errorf(
		"the package document for %s carries neither dist.integrity nor dist.shasum, "+
			"so the release manifest cannot be verified. Nothing was downloaded — every "+
			"binary hash this updater checks comes from inside that manifest", source)
}

func verifyIntegrity(raw []byte, integrity, source string) error {
	// A registry may publish several space-separated digests; the first is the
	// one npm itself verifies against.
	fields := strings.Fields(integrity)
	if len(fields) == 0 {
		return fmt.Errorf(
			"the registry published a blank dist.integrity for %s, so the release "+
				"manifest cannot be verified. Nothing was downloaded", source)
	}
	algo, encoded, ok := strings.Cut(fields[0], "-")
	if !ok {
		return fmt.Errorf(
			"the registry published dist.integrity %q for %s in a form this build "+
				"cannot parse (expected \"sha512-<base64>\")", integrity, source)
	}
	if algo != "sha512" {
		return fmt.Errorf(
			"the registry published a %s integrity digest for %s, but this build "+
				"verifies sha512. Nothing was downloaded", algo, source)
	}
	want, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return fmt.Errorf(
			"the registry's dist.integrity for %s is not valid base64: %w", source, err)
	}
	sum := sha512.Sum512(raw)
	if !bytes.Equal(sum[:], want) {
		return fmt.Errorf(
			"the release manifest downloaded from %s does not match the registry's "+
				"own integrity digest:\n  expected %s\n  actual   sha512-%s\nRefusing to "+
				"read it. Retry; if it persists, report it at "+
				"https://github.com/amd/gaia/issues",
			source, integrity, base64.StdEncoding.EncodeToString(sum[:]))
	}
	return nil
}

// extractFromTarGz pulls one named regular file out of a gzipped tar.
func extractFromTarGz(raw []byte, want, source string) ([]byte, error) {
	gz, err := gzip.NewReader(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf(
			"cannot open %s as a gzip archive: %w. The feed served something that is "+
				"not a published package tarball", source, err)
	}
	defer func() { _ = gz.Close() }()

	tr := tar.NewReader(gz)
	for {
		header, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("cannot read %s as a tar archive: %w", source, err)
		}
		if header.Typeflag != tar.TypeReg || header.Name != want {
			continue
		}
		body, err := io.ReadAll(io.LimitReader(tr, maxLockBytes))
		if err != nil {
			return nil, fmt.Errorf("cannot read %s out of %s: %w", want, source, err)
		}
		return body, nil
	}
	return nil, fmt.Errorf(
		"%s does not contain %s, so this release publishes no manifest to verify its "+
			"binaries against. Nothing was downloaded", source, want)
}
