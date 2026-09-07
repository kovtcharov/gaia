// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package chat

import (
	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"

	"github.com/amd/gaia/tui/internal/event"
)

// needsInputWithALongURL is the shape the bug needs: an option description
// carrying a token no word-boundary wrap can break. QuestionModel's own doc
// comment uses an OAuth consent URL as its example, and that is exactly what a
// "Connect Gmail" option's description contains.
func needsInputWithALongURL() event.CanonicalNeedsInputEvent {
	e := needsInput()
	e.Options[0].Description = "Opens https://accounts.google.com/o/oauth2/v2/auth?client_id=1234567890-abcdefghijklmnop.apps.googleusercontent.com&redirect_uri=http%3A%2F%2Flocalhost%3A8765%2Fcallback in your browser."
	return e
}

// screenCoordOfMarker finds where "[n] " is actually PAINTED — read off the
// rendered frame, deliberately not through questionRowAt, which is the map
// under test. A coordinate derived from the map can never disagree with it.
func screenCoordOfMarker(t *testing.T, m ChatModel, n int) (x, y int) {
	t.Helper()
	marker := fmt.Sprintf("[%d] ", n)
	for row, raw := range strings.Split(m.View(), "\n") {
		if col := strings.Index(ansi.Strip(raw), marker); col >= 0 {
			return col, row
		}
	}
	t.Fatalf("option %d is not on screen:\n%s", n, ansi.Strip(m.View()))
	return 0, 0
}

// The end-to-end shape of the defect: layout() undercounted the panel's rows,
// so questionViewLines (taken from the RENDERED height) and RowAt (computed
// from layout) disagreed, and every row below a wrapped URL was attributed to
// the wrong option. Clicking the row that VISIBLY says "[2] Not now" then
// committed a different option's value — or landed on the free-text input and
// submitted whatever was half-typed there.
func TestClickingTheOptionTheUserCanSeeAnswersWithThatOption(t *testing.T) {
	for _, w := range []int{60, 76, 80, 100, 120} {
		for i, want := range []string{"yes", "no"} {
			c := &respondingClient{}
			m := NewChatModel(c, "email", "", false)
			m.width, m.height = w, 40
			m.resize()
			m.streaming = true
			m = feed(t, m, needsInputWithALongURL())

			x, y := screenCoordOfMarker(t, m, i+1)
			if got := m.questionRowAt(x, y); got != i {
				t.Errorf("width=%d: the row painting option %d maps to row %d", w, i+1, got)
			}

			// Hover, then click — what a real mouse delivers, and the path
			// that turns a mis-mapped row into a committed wrong answer.
			for _, msg := range []tea.MouseMsg{
				{X: x, Y: y, Action: tea.MouseActionMotion},
				{X: x, Y: y, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress},
			} {
				updated, cmd := m.Update(msg)
				m = updated.(ChatModel)
				// Drain the resulting command chain the way Bubble Tea does:
				// the click emits QuestionAnsweredMsg, handling THAT emits the
				// respond call that actually reaches the agent.
				for depth := 0; cmd != nil && depth < 4; depth++ {
					next := cmd()
					if next == nil {
						break
					}
					updated, cmd = m.Update(next)
					m = updated.(ChatModel)
				}
			}

			if len(c.answers) != 1 {
				t.Fatalf("width=%d: clicking option %d produced %d answers", w, i+1, len(c.answers))
			}
			if got := c.answers[0][1]; got != want {
				t.Errorf("width=%d: clicking the row that SHOWS option %d answered %q, want %q",
					w, i+1, got, want)
			}
		}
	}
}
