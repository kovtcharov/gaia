// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"bufio"
	"fmt"
	"io"
	"strings"
)

// Decision is the answer to the download prompt.
type Decision int

const (
	// DecisionDecline downloads nothing now, and asks again next time.
	DecisionDecline Decision = iota
	// DecisionAccept downloads and installs.
	DecisionAccept
	// DecisionSkip downloads nothing and remembers this version, so the same
	// one is never offered again.
	DecisionSkip
)

func (d Decision) String() string {
	switch d {
	case DecisionAccept:
		return "accept"
	case DecisionSkip:
		return "skip"
	default:
		return "decline"
	}
}

// Prompter asks the user before anything is downloaded.
type Prompter interface {
	Confirm(plan *Plan) (Decision, error)
}

// NoConsentError is the refusal to proceed when there is no one to ask.
//
// The Agent UI downloads in the background and only asks about restarting. A
// terminal binary that replaces itself mid-session cannot do that, so a
// non-interactive run without an explicit opt-in stops here rather than
// treating silence as a yes.
type NoConsentError struct {
	// Reason names why no one could be asked.
	Reason string
}

func (e *NoConsentError) Error() string {
	return fmt.Sprintf(
		"refusing to download and replace the GAIA binaries without your consent: %s. "+
			"Nothing was downloaded.\n\n"+
			"  gaia-tui update install --yes    accept non-interactively (scripts, CI)\n"+
			"  gaia-tui update check            see what is available, download nothing",
		e.Reason)
}

// TTYPrompter asks on a real terminal.
//
// interactive is the caller's TTY check. When it is false and assumeYes is not
// set, Confirm refuses — it never falls through to a default answer.
type TTYPrompter struct {
	In          io.Reader
	Out         io.Writer
	Interactive bool
	AssumeYes   bool
}

// Confirm renders the plan and reads one answer.
func (p *TTYPrompter) Confirm(plan *Plan) (Decision, error) {
	if p.AssumeYes {
		fmt.Fprint(p.Out, plan.Render())
		fmt.Fprintf(p.Out, "\nAccepted by --yes: downloading %s.\n", FormatSize(plan.TotalBytes()))
		return DecisionAccept, nil
	}
	if !p.Interactive {
		// Print the plan anyway: a script that hits this refusal should still
		// carry, in its log, exactly what it refused.
		fmt.Fprint(p.Out, plan.Render())
		return DecisionDecline, &NoConsentError{Reason: "stdin is not a terminal, so there is no way to ask"}
	}

	fmt.Fprint(p.Out, plan.Render())
	reader := bufio.NewReader(p.In)
	for {
		fmt.Fprintf(p.Out, "\n  [y] download and install   [s] skip %s   [n] not now\nYour answer (y/s/N): ", plan.Release)
		line, err := reader.ReadString('\n')
		if err != nil && strings.TrimSpace(line) == "" {
			return DecisionDecline, &NoConsentError{
				Reason: fmt.Sprintf("the prompt could not be read (%v)", err),
			}
		}
		switch strings.ToLower(strings.TrimSpace(line)) {
		case "y", "yes":
			return DecisionAccept, nil
		case "s", "skip":
			return DecisionSkip, nil
		case "n", "no", "":
			return DecisionDecline, nil
		default:
			fmt.Fprintln(p.Out, "  Please answer y, s, or n.")
		}
	}
}

// Render is the screen a user sees before anything is downloaded: the facts,
// then the promise that answering is what starts the download.
func (p *Plan) Render() string {
	return p.Summary() + "\nNothing is downloaded until you answer.\n"
}

// Summary is the facts alone: what they have, what is available, how big it
// is, and what would be replaced. `update check` shows this without the
// prompt's closing promise, because check never asks anything.
func (p *Plan) Summary() string {
	var b strings.Builder

	fmt.Fprintf(&b, "\nGAIA %s is available (you have %s).\n\n", p.Release, orUnknown(p.CurrentRelease))
	for _, c := range p.Components {
		if !c.NeedsUpdate {
			continue
		}
		fmt.Fprintf(&b, "  %-9s %-9s → %-9s %s\n",
			c.Name, orUnknown(c.Current), c.Available, FormatSize(c.Size))
	}
	fmt.Fprintf(&b, "  %-9s %29s\n", "total", FormatSize(p.TotalBytes()))
	fmt.Fprintf(&b, "\n  channel  %s\n", p.Feed.String())

	b.WriteString("\nThis replaces, on disk:\n")
	for _, c := range p.Components {
		if !c.NeedsUpdate {
			continue
		}
		fmt.Fprintf(&b, "  %s\n", c.Target)
	}
	if p.ReplacesRunningBinary {
		b.WriteString("\nOne of those is the binary running right now. It is swapped only after the\n" +
			"download is SHA-256 verified, and the version you are running keeps working\n" +
			"until you start it again.\n")
	}
	return b.String()
}

func orUnknown(v string) string {
	if strings.TrimSpace(v) == "" {
		return "unknown"
	}
	return v
}
