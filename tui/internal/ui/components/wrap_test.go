// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package components

import (
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/x/ansi"
)

// A token nothing can break on a word boundary: the OAuth consent URL the
// question component's own doc comment uses as its motivating example.
const oauthURL = "https://accounts.google.com/o/oauth2/v2/auth?client_id=1234567890-abcdefghijklmnop.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocalhost%3A8765%2Fcallback&scope=openid+email&access_type=offline"

const longWindowsPath = `C:\Users\someone\AppData\Local\Temp\gaia-cache-0123456789abcdef\artifacts\build-output-2026-08-26.tar.gz`

// The invariant every caller that COUNTS wrapped lines depends on: a wrapper
// that lets one over-long token through hands back a line wider than the
// measure, lipgloss re-wraps it at render time, and the caller's row count no
// longer matches the screen.
func TestWrapLinesNeverExceedsTheLimit(t *testing.T) {
	inputs := []string{
		oauthURL,
		"Open " + oauthURL + " to authorize.",
		longWindowsPath,
		"Delete " + longWindowsPath + " permanently.",
		"      " + oauthURL,
		"      indented prose that also carries " + longWindowsPath + " inline",
		strings.Repeat("x", 300),
		"ordinary prose with nothing unusual in it at all",
	}

	for _, in := range inputs {
		for limit := 8; limit <= 120; limit++ {
			for i, line := range WrapLines(in, limit) {
				if w := ansi.StringWidth(line); w > limit {
					t.Fatalf("limit=%d line %d is %d columns wide: %q", limit, i, w, line)
				}
			}
		}
	}
}

// Splitting mid-token must not lose or reorder any of it — a clipped URL and a
// silently-truncated one are the same bug from the user's side.
func TestWrapLinesLosesNothingWhenItSplitsAToken(t *testing.T) {
	for limit := 8; limit <= 60; limit++ {
		joined := strings.Join(WrapLines(oauthURL, limit), "")
		if joined != oauthURL {
			t.Fatalf("limit=%d: hard split dropped characters\n got %q\nwant %q", limit, joined, oauthURL)
		}
	}
}

// A trailing newline is punctuation, not a paragraph. Agents routinely end a
// summary with one, and rendering it as an extra row opens a gap in the middle
// of a confirmation panel.
func TestWrapTextDropsATrailingNewlineButKeepsInteriorBlanks(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Delete the scratch dir.\n", "Delete the scratch dir."},
		{"Delete it.\n\n", "Delete it."},
		{"line one\n\nline two", "line one\n\nline two"},
		{"", ""},
		{"   ", ""},
	}
	for _, c := range cases {
		if got := WrapText(c.in, 40); got != c.want {
			t.Errorf("WrapText(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Continuations of an indented paragraph stay under it; an option description
// that hangs back to column 0 reads as a new paragraph, not a continuation.
func TestWrapLinesKeepsTheIndentOnContinuations(t *testing.T) {
	got := WrapLines("      alpha beta gamma delta epsilon zeta", 20)
	if len(got) < 2 {
		t.Fatalf("expected the text to wrap, got %q", got)
	}
	for i, line := range got {
		if !strings.HasPrefix(line, "      ") {
			t.Errorf("line %d lost the indent: %q", i, line)
		}
	}
}

// Callers index [0] without a length check.
func TestWrapLinesAlwaysReturnsAtLeastOneLine(t *testing.T) {
	for _, in := range []string{"", "\n", "\n\n", "   "} {
		if got := WrapLines(in, 20); len(got) != 1 || got[0] != "" {
			t.Errorf("WrapLines(%q) = %q, want one empty line", in, got)
		}
	}
}

// A measure narrower than a single double-width rune must still terminate:
// the split loop runs on the UI goroutine, so "cannot make progress" is a hang,
// not a rendering glitch.
func TestWrapLinesTerminatesOnAnUnsplittableRune(t *testing.T) {
	done := make(chan []string, 1)
	go func() { done <- WrapLines("日本語のテキスト", 1) }()
	select {
	case <-done:
	case <-timeoutAfterASecond():
		t.Fatal("WrapLines spun instead of returning")
	}
}

func timeoutAfterASecond() <-chan struct{} {
	ch := make(chan struct{})
	go func() {
		time.Sleep(time.Second)
		close(ch)
	}()
	return ch
}
