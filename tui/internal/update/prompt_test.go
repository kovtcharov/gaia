// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package update

import (
	"bytes"
	"errors"
	"strings"
	"testing"
)

func samplePlan() *Plan {
	return &Plan{
		Release:               "0.2.0",
		CurrentRelease:        "0.1.0",
		Feed:                  FeedRef{URL: "https://example.test/feed", Kind: FeedKindGeneric, Source: "built-in default"},
		ReplacesRunningBinary: true,
		Components: []ComponentPlan{
			{Name: ComponentSidecar, Current: "0.1.0", Available: "0.2.0", Size: 90_667_208,
				Target: `C:\Users\jane\.gaia\agents\gaia\gaia-agent.exe`, NeedsUpdate: true},
			{Name: ComponentTUI, Current: "0.23.0", Available: "0.24.0", Size: 19_433_984,
				Target: `C:\Users\jane\.gaia\bin\gaia-tui.exe`, NeedsUpdate: true},
		},
	}
}

// Silence is not consent. A script with no terminal and no --yes must be
// refused, and told exactly which flag opts in.
func TestPromptRefusesWithoutTTYAndWithoutYes(t *testing.T) {
	var out bytes.Buffer
	prompter := &TTYPrompter{In: strings.NewReader(""), Out: &out, Interactive: false}

	decision, err := prompter.Confirm(samplePlan())

	var noConsent *NoConsentError
	if !errors.As(err, &noConsent) {
		t.Fatalf("a non-interactive prompt returned %s, want a NoConsentError", fmtErr(err))
	}
	if decision == DecisionAccept {
		t.Fatal("a non-interactive prompt accepted on the user's behalf")
	}
	if !strings.Contains(err.Error(), "--yes") {
		t.Errorf("the refusal never names the flag that opts in:\n%v", err)
	}
	if !strings.Contains(err.Error(), "Nothing was downloaded") {
		t.Errorf("the refusal does not say that nothing was downloaded:\n%v", err)
	}
	// A CI log has to record what was refused, not just that something was.
	if !strings.Contains(out.String(), "0.2.0") {
		t.Errorf("the refused plan was never printed:\n%s", out.String())
	}
}

// --yes is the explicit non-interactive opt-in, and it still prints what it is
// about to fetch.
func TestPromptAcceptsWithYesFlag(t *testing.T) {
	var out bytes.Buffer
	prompter := &TTYPrompter{In: strings.NewReader(""), Out: &out, Interactive: false, AssumeYes: true}

	decision, err := prompter.Confirm(samplePlan())
	if err != nil {
		t.Fatalf("--yes: %v", err)
	}
	if decision != DecisionAccept {
		t.Fatalf("--yes produced %v, want accept", decision)
	}
	if !strings.Contains(out.String(), "Accepted by --yes") {
		t.Errorf("--yes did not say why it proceeded:\n%s", out.String())
	}
}

func TestPromptAnswers(t *testing.T) {
	cases := map[string]Decision{
		"y\n":    DecisionAccept,
		"yes\n":  DecisionAccept,
		"s\n":    DecisionSkip,
		"skip\n": DecisionSkip,
		"n\n":    DecisionDecline,
		"\n":     DecisionDecline, // bare Enter is the safe default
		"  Y \n": DecisionAccept,
	}
	for input, want := range cases {
		var out bytes.Buffer
		prompter := &TTYPrompter{In: strings.NewReader(input), Out: &out, Interactive: true}
		got, err := prompter.Confirm(samplePlan())
		if err != nil {
			t.Fatalf("answer %q: %v", input, err)
		}
		if got != want {
			t.Errorf("answer %q produced %v, want %v", input, got, want)
		}
	}
}

// An unrecognised answer re-asks rather than picking something.
func TestPromptReAsksOnGarbage(t *testing.T) {
	var out bytes.Buffer
	prompter := &TTYPrompter{In: strings.NewReader("maybe\ny\n"), Out: &out, Interactive: true}
	got, err := prompter.Confirm(samplePlan())
	if err != nil {
		t.Fatalf("confirm: %v", err)
	}
	if got != DecisionAccept {
		t.Fatalf("got %v, want accept after a re-ask", got)
	}
	if !strings.Contains(out.String(), "Please answer y, s, or n.") {
		t.Errorf("a garbage answer was not re-asked:\n%s", out.String())
	}
}

// The prompt is the whole feature — it has to carry the facts a user decides
// on before a byte is fetched.
func TestPromptRendersTheFactsBeforeDownloading(t *testing.T) {
	rendered := samplePlan().Render()
	for _, want := range []string{
		"0.2.0",            // available release
		"0.1.0",            // what they have
		"0.23.0", "0.24.0", // per-component versions
		"86.5 MB", "18.5 MB", "105.0 MB", // sizes, and the total
		`C:\Users\jane\.gaia\bin\gaia-tui.exe`,    // what gets replaced
		"Nothing is downloaded until you answer.", // the promise
		"running right now",                       // the self-replacement warning
	} {
		if !strings.Contains(rendered, want) {
			t.Errorf("the prompt never shows %q:\n%s", want, rendered)
		}
	}
}
