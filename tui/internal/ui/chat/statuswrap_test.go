// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package chat

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

// The status lines that carry a remedy put the fix in a URL or a path — the
// tail IS the remedy. The viewport does not soft-wrap, so a status line wider
// than the pane is CLIPPED, and the wrapper used to let an unbreakable token
// through untouched: the URL survived the wrap and then lost its tail on
// screen, which is the half of the line that says what to do.
func TestALongRemedyURLIsWrappedNotClipped(t *testing.T) {
	const url = "https://amd-gaia.ai/docs/reference/troubleshooting#lemonade-server-is-not-reachable-on-this-machine"

	for _, w := range []int{60, 80, 100, 120} {
		m, _ := newTestModel(t)
		m.width, m.height = w, 30
		m.resize()
		msg := Message{
			Role:    RoleStatus,
			Content: "Lemonade Server is not reachable — see " + url,
		}

		pane := m.cardWidth()
		var body strings.Builder
		for _, line := range strings.Split(ansi.Strip(m.renderMessage(&msg, nil)), "\n") {
			if got := ansi.StringWidth(line); got > pane {
				t.Errorf("width=%d: a status line is %d columns for a %d-column pane: %q",
					w, got, pane, line)
			}
			body.WriteString(strings.TrimSpace(line))
		}
		if !strings.Contains(body.String(), url) {
			t.Errorf("width=%d: the remedy URL did not survive wrapping:\n%s", w, body.String())
		}
	}
}
