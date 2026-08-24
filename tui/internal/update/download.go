// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// BackupSuffix is what a replaced binary is renamed to.
//
// Windows refuses to overwrite or delete a running .exe, so the swap is
// rename-self-aside then rename-new-in. The aside file survives until the next
// start, which is the only moment nothing holds it open.
const BackupSuffix = ".old"

// stagingSuffix marks a partially-downloaded binary. It is never executable and
// never left behind on a successful install.
const stagingSuffix = ".download"

// rejectedSuffix marks a download that failed its hash check. It is kept, not
// deleted, so the bytes that failed can be inspected — and so the error can
// name a file that actually exists.
const rejectedSuffix = ".rejected"

// IntegrityError is a downloaded artifact whose SHA-256 does not match the lock.
//
// There is no "use it anyway" path and nothing retries on its own: a binary
// that does not match its manifest is refused before it can replace anything.
type IntegrityError struct {
	URL      string
	Path     string
	Expected string
	Actual   string
}

func (e *IntegrityError) Error() string {
	return fmt.Sprintf(
		"SHA-256 mismatch for %s:\n"+
			"  expected  %s\n"+
			"  actual    %s\n"+
			"  saved at  %s\n"+
			"Refusing to install a binary that does not match %s. It was NOT installed "+
			"and the file it would have replaced is untouched. The download may be "+
			"corrupt, truncated, or tampered with — delete the saved file and re-run "+
			"`gaia-tui update install`; if it happens again, report it at "+
			"https://github.com/amd/gaia/issues",
		e.URL, e.Expected, e.Actual, e.Path, LockFileName)
}

// DownloadVerified fetches url into destDir and returns the path of the staged
// file, which is written only after its SHA-256 matches expectedSHA.
//
// progress, when non-nil, is called with the bytes read so far and the expected
// total (0 when the server sends no length).
func DownloadVerified(
	ctx context.Context,
	client *http.Client,
	url, expectedSHA, destDir, name string,
	progress func(done, total int64),
) (string, error) {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", fmt.Errorf("cannot create %s to download into: %w", destDir, err)
	}
	staged := filepath.Join(destDir, name+stagingSuffix)
	if err := os.Remove(staged); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return "", fmt.Errorf(
			"cannot clear the leftover download at %s: %w. Delete it and re-run", staged, err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("cannot build the download request for %s: %w", url, err)
	}
	req.Header.Set("User-Agent", userAgent)

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf(
			"cannot download %s: %w. Check your network and re-run `gaia-tui update "+
				"install`", url, err)
	}
	defer func() {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
		_ = resp.Body.Close()
	}()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf(
			"the update feed answered HTTP %d for %s. The release may have been "+
				"withdrawn — run `gaia-tui update list` to see what is published",
			resp.StatusCode, url)
	}

	// 0700: the file is executable and not yet verified, so nothing else on the
	// machine gets a chance to run it while it is being written.
	f, err := os.OpenFile(staged, os.O_WRONLY|os.O_CREATE|os.O_EXCL|os.O_TRUNC, 0o700) // #nosec G304 -- staged is built from destDir and the lock's executable name
	if err != nil {
		return "", fmt.Errorf("cannot create %s: %w", staged, err)
	}

	digest := sha256.New()
	written, err := io.Copy(io.MultiWriter(f, digest), &progressReader{
		r:        resp.Body,
		total:    resp.ContentLength,
		progress: progress,
	})
	if closeErr := f.Close(); closeErr != nil && err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(staged)
		return "", fmt.Errorf(
			"the download of %s failed after %d bytes: %w. Nothing was replaced — "+
				"re-run `gaia-tui update install`", url, written, err)
	}

	actual := hex.EncodeToString(digest.Sum(nil))
	if !strings.EqualFold(actual, expectedSHA) {
		rejected := filepath.Join(destDir, name+rejectedSuffix)
		if err := os.Rename(staged, rejected); err != nil {
			// Quarantine failed; the staged file is still the one to name.
			rejected = staged
		}
		return "", &IntegrityError{URL: url, Path: rejected, Expected: strings.ToLower(expectedSHA), Actual: actual}
	}
	return staged, nil
}

// progressReader reports download progress without buffering the body.
type progressReader struct {
	r        io.Reader
	total    int64
	done     int64
	progress func(done, total int64)
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.r.Read(b)
	p.done += int64(n)
	if p.progress != nil && n > 0 {
		p.progress(p.done, p.total)
	}
	return n, err
}

// ReplaceBinary installs staged over target.
//
// On Windows a running .exe cannot be overwritten or deleted, so target is
// renamed aside first — a rename of a running image IS permitted — and the new
// file takes its place. If the second rename fails the aside file is put back,
// so the sequence never ends with no binary at target.
func ReplaceBinary(target, staged string) error {
	if runtime.GOOS != "windows" {
		if err := os.Chmod(staged, 0o755); err != nil {
			return fmt.Errorf("cannot make %s executable: %w", staged, err)
		}
		if err := os.Rename(staged, target); err != nil {
			return fmt.Errorf(
				"cannot install the verified download over %s: %w. The old binary is "+
					"untouched and the new one is at %s — move it into place by hand if "+
					"this keeps failing", target, err, staged)
		}
		return nil
	}

	backup := target + BackupSuffix
	// A backup from an earlier update is still on disk when the process that was
	// running from it never exited. Clearing it first is what makes room for
	// this one; if it cannot go, neither can the swap.
	if err := os.Remove(backup); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf(
			"cannot clear the previous backup at %s: %w. Close any other running GAIA "+
				"process, delete that file, then re-run `gaia-tui update install`. "+
				"Nothing was replaced", backup, err)
	}

	movedAside := true
	if err := os.Rename(target, backup); err != nil {
		if !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf(
				"cannot move %s aside to install the update: %w. Nothing was replaced — "+
					"the verified download is at %s", target, err, staged)
		}
		// Nothing at target (a first install, or a manual delete): there is
		// nothing to restore if the next step fails.
		movedAside = false
	}

	if err := os.Rename(staged, target); err != nil {
		restoreNote := ""
		if movedAside {
			if restoreErr := os.Rename(backup, target); restoreErr != nil {
				restoreNote = fmt.Sprintf(
					"\nThe old binary could not be put back either (%v). It is at %s — "+
						"rename it to %s to recover.", restoreErr, backup, filepath.Base(target))
			} else {
				restoreNote = "\nThe old binary was put back, so nothing is missing."
			}
		}
		return fmt.Errorf(
			"cannot install the verified download at %s: %w.%s The new binary is at %s",
			target, err, restoreNote, staged)
	}
	return nil
}

// CleanStaleBackup removes the aside file a previous Windows swap left behind.
//
// Called at start, which is the one moment the replaced image is guaranteed not
// to be running. A backup that is still held open (an older process outliving
// the swap) stays put and is reported, never forced.
func CleanStaleBackup(target string) error {
	backup := target + BackupSuffix
	err := os.Remove(backup)
	switch {
	case err == nil, errors.Is(err, fs.ErrNotExist):
		return nil
	default:
		return fmt.Errorf(
			"the previous update left %s behind and it could not be removed: %w. It is "+
				"harmless; delete it once no other GAIA process is running", backup, err)
	}
}

// FormatSize renders a download size the way a person reads it, matching
// catalog.FormatSize so the two surfaces never disagree about the same number.
func FormatSize(bytes int64) string {
	switch {
	case bytes <= 0:
		return "unknown size"
	case bytes < 1024:
		return fmt.Sprintf("%d B", bytes)
	case bytes < 1024*1024:
		return fmt.Sprintf("%.0f KB", float64(bytes)/1024)
	case bytes < 1024*1024*1024:
		return fmt.Sprintf("%.1f MB", float64(bytes)/(1024*1024))
	default:
		return fmt.Sprintf("%.1f GB", float64(bytes)/(1024*1024*1024))
	}
}
