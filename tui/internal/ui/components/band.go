// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package components

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	"github.com/amd/gaia/tui/internal/ui/theme"
)

// A panel is a filled band naming what it is, with its body hanging underneath.
// No border: a frame costs four columns and two rows, and draws the eye to the
// chrome instead of the words inside it.
//
// The band is a SATURATED fill (theme's *FillBG with OnFill on top), which is
// the only kind that survives ANSI-256 degradation. A subtle near-background
// tint was tried first and abandoned — anything quiet enough to read as subtle
// degrades to invisible on stock terminal themes (Nord, Solarized Light,
// Campbell), and the theme contrast suite measures exactly that.
type PanelKind int

const (
	PanelError PanelKind = iota
	PanelWarn
	PanelInfo
	PanelAccent
)

// panelMeasure caps how wide the body text is set. Prose run to the full width
// of a wide terminal is one long scan line per paragraph.
const panelMeasure = 84

func (k PanelKind) fill() lipgloss.AdaptiveColor {
	switch k {
	case PanelWarn:
		return theme.WarnFillBG
	case PanelInfo:
		return theme.InfoFillBG
	case PanelAccent:
		return theme.AccentFillBG
	default:
		return theme.DangerFillBG
	}
}

// body is the colour the text under the band is painted. It carries the same
// meaning as the band so the two read as one unit, except for Accent/Info where
// ordinary body text is more legible than a coloured paragraph.
func (k PanelKind) body() lipgloss.AdaptiveColor {
	switch k {
	case PanelError:
		return theme.Danger
	case PanelWarn:
		return theme.Warning
	default:
		return theme.Text
	}
}

// Panel renders a borderless titled block. title is shown in the band, upper
// case; an empty title still draws the band so the block is delimited.
func Panel(kind PanelKind, title, body string, width int) string {
	if width < 1 {
		width = 1
	}
	bandStyle := lipgloss.NewStyle().
		Background(kind.fill()).
		Foreground(theme.OnFill).
		Bold(true)
	bodyStyle := lipgloss.NewStyle().Foreground(kind.body())

	lines := []string{bandStyle.Render(bandText(title, width))}

	indent := "  "
	if width < 12 {
		indent = ""
	}
	measure := width - len(indent)
	if measure > panelMeasure {
		measure = panelMeasure
	}
	for _, line := range wrapPanelText(body, measure) {
		lines = append(lines, indent+bodyStyle.Render(line))
	}
	return strings.Join(lines, "\n")
}

// bandText pads the title to exactly width so the fill spans the pane. A fill
// that stops short reads as a ragged highlight, not a section.
func bandText(title string, width int) string {
	t := " " + strings.ToUpper(strings.TrimSpace(title))
	if strings.TrimSpace(title) == "" {
		t = ""
	}
	if ansi.StringWidth(t) > width {
		t = ansi.Truncate(t, width, "…")
	}
	if n := width - ansi.StringWidth(t); n > 0 {
		t += strings.Repeat(" ", n)
	}
	return t
}

// wrapPanelText breaks s to w columns for the panel body. A panel indents each
// line itself, so it needs them separately rather than newline-joined — that is
// the only reason this exists next to WrapText (wrap.go); both wrap the same way.
func wrapPanelText(s string, w int) []string {
	return WrapLines(s, w)
}
