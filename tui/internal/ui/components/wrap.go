// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package components

import (
	"strings"

	"github.com/charmbracelet/x/ansi"
)

// WrapLines breaks s to at most limit columns, on word boundaries where it can
// and MID-TOKEN when a single token (a URL, a Windows path, a long flag) is
// wider than the measure. Each paragraph's leading indent is preserved and
// re-applied to its continuations, so an indented block stays a block.
//
// The hard split is the whole point, not a nicety. A wrapper that lets one
// over-long token through returns a line wider than it promised, and every
// caller that COUNTS those lines is then wrong about its own height: lipgloss
// re-wraps the overflow at render time, the block grows rows the caller never
// counted, and any row -> meaning map built from the count (QuestionModel's
// hit-testing, say) silently points at the wrong thing. See TestWrapNeverExceedsLimit
// and question_hittest_test.go.
//
// A trailing newline is not a paragraph: it is trimmed, so "done.\n" wraps to
// one line rather than one line plus a blank. Interior blank lines are kept —
// those are deliberate paragraph breaks. The result always has at least one
// element, so callers can index [0] without a length check.
func WrapLines(s string, limit int) []string {
	if limit < 1 {
		limit = 1
	}

	var out []string
	for _, para := range strings.Split(strings.TrimRight(s, "\n"), "\n") {
		indent := para[:len(para)-len(strings.TrimLeft(para, " "))]
		indentW := ansi.StringWidth(indent)
		// A token measures against what is left after the indent; if the
		// indent alone already eats the measure, keep one usable column so the
		// split below can still make progress.
		avail := limit - indentW
		if avail < 1 {
			avail = 1
		}

		fields := strings.Fields(para)
		if len(fields) == 0 {
			out = append(out, "")
			continue
		}

		line := ""
		for _, word := range fields {
			for ansi.StringWidth(word) > avail {
				if line != "" {
					out = append(out, indent+line)
					line = ""
				}
				head := ansi.Truncate(word, avail, "")
				rest := strings.TrimPrefix(word, head)
				// ansi.Truncate re-emits a reset sequence, so head is not
				// always a literal prefix of word; without this the loop
				// cannot make progress and spins on the UI goroutine.
				if head == "" || rest == word {
					break
				}
				out = append(out, indent+head)
				word = rest
			}
			switch {
			case line == "":
				line = word
			case indentW+ansi.StringWidth(line)+1+ansi.StringWidth(word) <= limit:
				line += " " + word
			default:
				out = append(out, indent+line)
				line = word
			}
		}
		if line != "" {
			out = append(out, indent+line)
		}
	}

	if len(out) == 0 {
		return []string{""}
	}
	return out
}

// WrapText is WrapLines joined back into one newline-separated string, for the
// callers that hand their text straight to a renderer rather than indenting
// each line themselves.
//
// The viewport does NOT soft-wrap: a line longer than the pane is CLIPPED, and
// a clipped message loses its tail — which for an actionable message is exactly
// the part that says what to do. Anything rendered as a bare line rather than
// inside a width-constrained lipgloss block has to come through here.
func WrapText(s string, limit int) string {
	if limit <= 0 {
		return s
	}
	return strings.Join(WrapLines(s, limit), "\n")
}
