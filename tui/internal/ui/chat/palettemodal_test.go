// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package chat

import (
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"

	"github.com/amd/gaia/tui/internal/event"
	"github.com/amd/gaia/tui/internal/ui/components"
)

// paletteTitle is the one string only the "/" palette draws.
const paletteTitle = "Slash Commands"

func screen(m ChatModel) string { return ansi.Strip(m.View()) }

// openPaletteMidTurn puts the model in the exact state the bug needs: a turn is
// running (typing mid-turn is advertised) and the user has a slash command
// half-typed, so the palette is open and owns the frame.
func openPaletteMidTurn(t *testing.T) ChatModel {
	t.Helper()
	m, _ := newTestModel(t)
	m.streaming = true
	m = typeInto(t, m, "/mem")
	if !m.paletteShowing() {
		t.Fatal("setup: \"/mem\" should have opened the palette")
	}
	if !strings.Contains(screen(m), paletteTitle) {
		t.Fatal("setup: the palette should be on screen")
	}
	return m
}

// --- a decision surface always wins the frame -------------------------------

// The defect: renderCommandPalette replaces the WHOLE frame, and nothing closed
// the palette when a confirmation arrived mid-turn. The user saw the slash
// command list while y/n/Enter/Esc routed into an unseen confirmation for a
// destructive tool — a permission decision made against a UI they cannot see.
func TestConfirmationArrivingMidTurnTakesTheFrameBackFromThePalette(t *testing.T) {
	m := openPaletteMidTurn(t)
	m = feed(t, m, needsConfirmation(""))

	if m.confirmation == nil {
		t.Fatal("setup: the confirmation should be up")
	}
	if m.paletteShowing() {
		t.Error("the palette must not own input while a confirmation is pending")
	}
	if got := screen(m); strings.Contains(got, paletteTitle) {
		t.Errorf("the palette is drawn OVER the confirmation:\n%s", got)
	}
	if got := screen(m); !strings.Contains(got, "Confirm:") {
		t.Errorf("the confirmation the keyboard is answering is not on screen:\n%s", got)
	}
}

// Same defect on the question path: needs_input parks the turn the same way.
func TestQuestionArrivingMidTurnTakesTheFrameBackFromThePalette(t *testing.T) {
	m := openPaletteMidTurn(t)
	m = feed(t, m, needsInput())

	if m.question == nil {
		t.Fatal("setup: the question should be up")
	}
	if m.paletteShowing() {
		t.Error("the palette must not own input while a question is pending")
	}
	if got := screen(m); strings.Contains(got, paletteTitle) {
		t.Errorf("the palette is drawn OVER the question:\n%s", got)
	}
	if got := screen(m); !strings.Contains(got, "Connect one now?") {
		t.Errorf("the question the keyboard is answering is not on screen:\n%s", got)
	}
}

// --- the keys go where the pixels say ---------------------------------------

// Enter is the palette's "run the selected command" key AND the question's
// "answer". With the palette on screen over an invisible question, Enter
// answered the question. It must reach the surface the user can actually see.
func TestEnterAnswersTheVisibleQuestionNotTheHiddenPalette(t *testing.T) {
	m := openPaletteMidTurn(t)
	m = feed(t, m, needsInput())

	_, cmd := press(t, m, tea.KeyEnter)
	if cmd == nil {
		t.Fatal("Enter did nothing while a question was up")
	}
	msg, ok := cmd().(components.QuestionAnsweredMsg)
	if !ok {
		t.Fatalf("Enter produced %T, want the question's answer", cmd())
	}
	if msg.Value != "yes" {
		t.Errorf("Enter answered %q, want the highlighted option", msg.Value)
	}
}

// Esc DENIES a confirmation. Pressed against a visible palette it means "close
// this list" — so a palette drawn over an unseen confirmation turns a dismissal
// into a denial the user never made. Once the confirmation owns the frame, Esc
// meaning "deny" is at least a decision about something on screen.
func TestEscDeniesTheVisibleConfirmationRatherThanClosingAHiddenPalette(t *testing.T) {
	m := openPaletteMidTurn(t)
	m = feed(t, m, needsConfirmation(""))

	m, _ = press(t, m, tea.KeyEsc)
	if m.confirmation == nil {
		t.Fatal("the confirmation vanished instead of resolving")
	}
	if m.confirmation.State() != components.ConfirmationDenied {
		t.Errorf("state = %v, want Denied", m.confirmation.State())
	}
}

// --- nothing the user typed is thrown away ----------------------------------

// Suppressed, not discarded: the half-typed command lives in the composer,
// which stays visible under the modal, and answering the modal brings the
// palette straight back with the same text.
func TestTheHalfTypedCommandSurvivesTheInterruption(t *testing.T) {
	m := openPaletteMidTurn(t)
	m = feed(t, m, needsConfirmation(""))

	if got := m.input.Value(); got != "/mem" {
		t.Errorf("composer = %q, want the half-typed command intact", got)
	}
	if got := screen(m); !strings.Contains(got, "/mem") {
		t.Errorf("the half-typed command is not visible anywhere:\n%s", got)
	}

	// Deny it — the confirmation clears and the palette comes back.
	updated, _ := m.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune("n")})
	m = updated.(ChatModel)
	m = feed(t, m, event.CanonicalFinalEvent{Type: "final", Answer: "denied"})

	if m.confirmation != nil || m.question != nil {
		t.Fatal("setup: the modal should have cleared")
	}
	if !m.paletteShowing() {
		t.Error("the palette must come back once the decision surface is gone")
	}
	if got := m.input.Value(); got != "/mem" {
		t.Errorf("composer = %q after the interruption, want %q", got, "/mem")
	}
}

// --- the mouse follows the same rule ----------------------------------------

// Mouse routing put the palette AHEAD of the question, so with a question up
// and the palette open a click landed on a palette row the user could not see —
// and clicking the already-selected row RUNS that command.
func TestAClickGoesToTheVisibleQuestionNotTheHiddenPalette(t *testing.T) {
	m := openPaletteMidTurn(t)
	m = feed(t, m, needsInput())

	if m.paletteShowing() {
		t.Fatal("setup: the palette should be suppressed")
	}
	before := m.input.Value()
	updated, _ := m.Update(tea.MouseMsg{
		Action: tea.MouseActionPress, Button: tea.MouseButtonLeft, X: 10, Y: 10,
	})
	m = updated.(ChatModel)

	if m.input.Value() != before {
		t.Errorf("a click ran a hidden palette command: composer went %q -> %q", before, m.input.Value())
	}
	if m.question == nil {
		t.Error("the click resolved the question it was never aimed at")
	}
}
