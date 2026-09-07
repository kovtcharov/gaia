// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package components

import (
	"fmt"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

// pathologicalQuestion is a question whose option descriptions carry tokens no
// word-boundary wrap can break — the OAuth consent URL from QuestionModel's own
// motivating example, and a long Windows path. Both are exactly what an agent
// puts in a "what does choosing this DO?" description.
func pathologicalQuestion() QuestionModel {
	return NewQuestionModel("req-1", "Which mailbox should I connect?", []QuestionOption{
		{Value: "gmail", Label: "Gmail", Description: "Opens " + oauthURL + " in your browser."},
		{Value: "outlook", Label: "Outlook", Description: "Uses Microsoft 365 work or school sign-in."},
		{Value: "local", Label: "Local mbox", Description: "Reads " + longWindowsPath + " from disk."},
	}, true, false)
}

// The defect this pins: layout() wrapped at m.width-4 while View() rendered
// through an m.width panel whose content box is m.width-2, so a token wider
// than the content box got re-wrapped by lipgloss AT RENDER TIME and the panel
// gained rows layout() never counted. RowAt maps a screen row through layout,
// so every row below the first over-long token pointed at the wrong option —
// and a click on the row the cursor was already on committed a DIFFERENT
// option's value to the agent.
func TestQuestionLayoutHeightMatchesRenderedHeight(t *testing.T) {
	for w := 24; w <= 140; w++ {
		m := pathologicalQuestion()
		m.SetWidth(w)

		layout := len(m.layout(m.innerWidth()))
		rendered := lipgloss.Height(m.View())
		if layout != rendered {
			t.Errorf("width=%d: layout counted %d rows, View rendered %d", w, layout, rendered)
		}
	}
}

// The same invariant stated as its cause: nothing layout produces may be wider
// than the panel's content box, or lipgloss re-wraps it behind layout's back.
func TestQuestionNoRenderedRowOverflowsThePanel(t *testing.T) {
	for w := 24; w <= 140; w++ {
		m := pathologicalQuestion()
		m.SetWidth(w)

		// questionPanelStyle is Padding(0, 1): the content box is two columns
		// narrower than the panel.
		content := w - 2
		for i, l := range m.layout(m.innerWidth()) {
			if got := ansi.StringWidth(l.text); got > content {
				t.Errorf("width=%d: layout line %d is %d columns, content box is %d: %q",
					w, i, got, content, ansi.Strip(l.text))
			}
		}
	}
}

// RowAt's stated contract, checked against the pixels rather than against
// layout: "a click can never disagree with what is actually on screen". The
// row a user SEES "[2] Outlook" on must be the row that selects Outlook.
func TestQuestionRowAtAgreesWithWhatIsOnScreen(t *testing.T) {
	for w := 24; w <= 140; w++ {
		m := pathologicalQuestion()
		m.SetWidth(w)

		lines := strings.Split(m.View(), "\n")
		seen := 0
		for row, raw := range lines {
			plain := ansi.Strip(raw)
			for i := range m.options {
				marker := fmt.Sprintf("[%d] ", i+1)
				if !strings.Contains(plain, marker) {
					continue
				}
				seen++
				if got := m.RowAt(row); got != i {
					t.Errorf("width=%d: screen row %d shows %q but RowAt says option %d",
						w, row, strings.TrimSpace(plain), got)
				}
			}
		}
		if seen != len(m.options) {
			t.Errorf("width=%d: only %d of %d option rows were visible on screen",
				w, seen, len(m.options))
		}

		// The free-text row is the one a click must NOT submit, so a shifted
		// index landing on it is how the reported "[2] Outlook clicks the text
		// box" symptom shows up.
		freeText := 0
		for row := range lines {
			if m.RowAt(row) == len(m.options) {
				freeText++
			}
		}
		if freeText == 0 {
			t.Errorf("width=%d: the free-text row is unreachable by mouse", w)
		}
	}
}

// A click on the already-selected row synthesizes Enter (handleQuestionMouse),
// so a row that maps to the wrong option does not merely mis-highlight — it
// commits the wrong VALUE. This drives that end to end.
func TestQuestionClickingTheVisibleRowAnswersWithThatOption(t *testing.T) {
	for w := 24; w <= 140; w++ {
		for want, opt := range pathologicalQuestion().options {
			m := pathologicalQuestion()
			m.SetWidth(w)

			row := -1
			for i, raw := range strings.Split(m.View(), "\n") {
				if strings.Contains(ansi.Strip(raw), fmt.Sprintf("[%d] ", want+1)) {
					row = i
					break
				}
			}
			if row < 0 {
				t.Fatalf("width=%d: option %d never rendered", w, want)
			}

			// Hover, then click the row now under the cursor: WithCursor then
			// the synthesized Enter, exactly what handleQuestionMouse does
			// with RowAt's answer.
			hit := m.RowAt(row)
			if m.IsFreeTextRow(hit) {
				t.Fatalf("width=%d: the row showing %q maps to the free-text input", w, opt.Label)
			}
			m = m.WithCursor(hit)
			_, cmd := m.Update(tea.KeyMsg{Type: tea.KeyEnter})
			if cmd == nil {
				t.Fatalf("width=%d: clicking row %d did not answer", w, row)
			}
			msg, ok := cmd().(QuestionAnsweredMsg)
			if !ok {
				t.Fatalf("width=%d: expected QuestionAnsweredMsg", w)
			}
			if msg.Value != opt.Value {
				t.Errorf("width=%d: clicking the row that SHOWS %q answered %q",
					w, opt.Label, msg.Value)
			}
		}
	}
}
