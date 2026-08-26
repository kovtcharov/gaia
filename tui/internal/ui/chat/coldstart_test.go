package chat

// The first message of a session against a cold Lemonade pays a one-time
// model load + prompt prefill (minutes, not seconds). These tests pin the
// staged loading UI: the startup ping's model_loaded field arms it, stage
// events render as distinct closable items with per-stage elapsed time, and
// a completed turn disarms it. No fake progress anywhere — only stages that
// actually happened, each closed by real evidence the next thing started.

import (
	"strings"
	"testing"
	"time"

	"github.com/amd/gaia/tui/internal/event"
)

func boolp(b bool) *bool { return &b }

func startupPing(loaded *bool) event.CanonicalStatusEvent {
	return event.CanonicalStatusEvent{
		Type:         "status",
		ModelID:      "Gemma-4-E4B-it-GGUF",
		ModelDisplay: "Gemma-4-E4B-it-GGUF",
		ModelBackend: "lemonade",
		ModelLoaded:  loaded,
	}
}

func TestStartupPingArmsColdStartOnlyOnExplicitFalse(t *testing.T) {
	cases := []struct {
		name   string
		loaded *bool
		want   bool
	}{
		{"reported cold", boolp(false), true},
		{"reported warm", boolp(true), false},
		{"not reported (older agent)", nil, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m, _ := newTestModel(t)
			m = feed(t, m, startupPing(tc.loaded))
			if m.coldStart != tc.want {
				t.Fatalf("coldStart = %v, want %v", m.coldStart, tc.want)
			}
		})
	}
}

func TestStageEventOpensADistinctActivityItem(t *testing.T) {
	m, _ := newTestModel(t)
	m.streaming = true

	m = feed(t, m, event.CanonicalStatusEvent{
		Type:    "status",
		Message: "Loading Gemma-4-E4B-it-GGUF into memory",
		Stage:   "model_load",
	})

	if len(m.activity) != 1 {
		t.Fatalf("activity = %d items, want 1", len(m.activity))
	}
	item := m.activity[0]
	if item.Kind != "stage" || item.Stage != "model_load" || item.Done {
		t.Fatalf("unexpected item: %+v", item)
	}
	if item.Started.IsZero() {
		t.Fatal("stage item has no start time — elapsed cannot be honest")
	}
}

func TestANewStageClosesThePreviousOneWithItsElapsed(t *testing.T) {
	m, _ := newTestModel(t)
	m.streaming = true

	m = feed(t, m, event.CanonicalStatusEvent{
		Type: "status", Message: "Loading model", Stage: "model_load",
	})
	// Backdate so the recorded elapsed is non-zero and provably real.
	m.activity[0].Started = time.Now().Add(-95 * time.Second)
	m = feed(t, m, event.CanonicalStatusEvent{
		Type: "status", Message: "Prefilling prompt", Stage: "prefill",
	})

	if len(m.activity) != 2 {
		t.Fatalf("activity = %d items, want 2", len(m.activity))
	}
	first := m.activity[0]
	if !first.Done {
		t.Fatal("previous stage not closed by the next one")
	}
	if !strings.Contains(first.Detail, "done — 1:3") { // 1:35 ± a second
		t.Fatalf("stage close detail %q does not carry its elapsed", first.Detail)
	}
	if m.activity[1].Done {
		t.Fatal("the new stage must be the open one")
	}
}

func TestFirstTokenClosesTheOpenStage(t *testing.T) {
	m, _ := newTestModel(t)
	m.streaming = true

	m = feed(t, m,
		event.CanonicalStatusEvent{Type: "status", Message: "Prefilling", Stage: "prefill"},
		event.CanonicalTokenEvent{Type: "token", Delta: "Hello"},
	)

	if !m.activity[0].Done {
		t.Fatal("answer text arrived but the stage is still open")
	}
}

func TestAToolCallClosesTheOpenStage(t *testing.T) {
	m, _ := newTestModel(t)
	m.streaming = true

	m = feed(t, m,
		event.CanonicalStatusEvent{Type: "status", Message: "Prefilling", Stage: "prefill"},
		event.CanonicalToolCallEvent{Type: "tool_call", Tool: "search"},
	)

	if !m.activity[0].Done {
		t.Fatal("a tool ran but the stage is still open")
	}
}

func TestAPlainStatusClosesTheStageAndKeepsItsRecord(t *testing.T) {
	m, _ := newTestModel(t)
	m.streaming = true

	m = feed(t, m,
		event.CanonicalStatusEvent{Type: "status", Message: "Loading model", Stage: "model_load"},
		event.CanonicalStatusEvent{Type: "status", Message: "Working out how to answer"},
	)

	if len(m.activity) != 2 {
		t.Fatalf("activity = %d items, want stage + status", len(m.activity))
	}
	if !m.activity[0].Done {
		t.Fatal("plain status arrived but the stage is still open")
	}
	if m.activity[1].Kind != "status" {
		t.Fatalf("second item kind = %q, want status (never folded into the stage)", m.activity[1].Kind)
	}
}

func TestAFinishedTurnDisarmsColdStart(t *testing.T) {
	m, _ := newTestModel(t)
	m.streaming = true
	m = feed(t, m, startupPing(boolp(false)))
	if !m.coldStart {
		t.Fatal("precondition: coldStart armed")
	}

	m = feed(t, m, event.CanonicalFinalEvent{Type: "final", Answer: "done"})

	if m.coldStart {
		t.Fatal("a completed turn means the model is warm — coldStart must clear")
	}
}

func TestAFailedTurnKeepsColdStartArmed(t *testing.T) {
	m, _ := newTestModel(t)
	m.streaming = true
	m = feed(t, m, startupPing(boolp(false)))

	m = feed(t, m, event.CanonicalErrorEvent{Type: "error", Detail: "load failed"})

	if !m.coldStart {
		t.Fatal("the load never succeeded — the next message still pays it")
	}
}

func TestColdStartIdlePhraseNamesTheModelLoad(t *testing.T) {
	m, _ := newTestModel(t)
	m.coldStart = true

	if got := m.idlePhrase(0); !strings.Contains(got, "model") {
		t.Fatalf("idlePhrase = %q, want it to name the model load", got)
	}
	m.coldStart = false
	if got := m.idlePhrase(0); got != "Getting started" {
		t.Fatalf("warm idlePhrase = %q, want the generic one", got)
	}
}
