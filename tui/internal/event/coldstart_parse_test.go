package event

// The cold-start fields ride the existing `status` type additively: a stage
// event and a model_loaded report must decode, and their absence must decode
// to the same zero values an older agent's events always had.

import "testing"

func TestStatusEventCarriesColdStartFields(t *testing.T) {
	e := ParseCanonicalEvent([]byte(
		`{"type":"status","message":"Loading model","stage":"model_load"}`,
	))
	s, ok := e.(CanonicalStatusEvent)
	if !ok {
		t.Fatalf("expected CanonicalStatusEvent, got %T", e)
	}
	if s.Stage != "model_load" {
		t.Fatalf("Stage = %q, want model_load", s.Stage)
	}
	if s.ModelLoaded != nil {
		t.Fatal("ModelLoaded must stay nil when not sent")
	}
}

func TestStartupPingCarriesModelResidency(t *testing.T) {
	e := ParseCanonicalEvent([]byte(
		`{"type":"status","message":"","model_id":"Gemma-4-E4B-it-GGUF",` +
			`"model_backend":"lemonade","model_loaded":false}`,
	))
	s, ok := e.(CanonicalStatusEvent)
	if !ok {
		t.Fatalf("expected CanonicalStatusEvent, got %T", e)
	}
	if s.ModelLoaded == nil || *s.ModelLoaded {
		t.Fatalf("ModelLoaded = %v, want explicit false", s.ModelLoaded)
	}
	if s.Stage != "" {
		t.Fatalf("Stage = %q, want empty on a ping", s.Stage)
	}
}
