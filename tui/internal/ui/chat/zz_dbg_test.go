package chat

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

func TestDbgCoord(t *testing.T) {
	c := &respondingClient{}
	m := NewChatModel(c, "email", "", false)
	m.width, m.height = 60, 40
	m.resize()
	m.streaming = true
	m = feed(t, m, needsInputWithALongURL())
	x, y := screenCoordOfMarker(t, m, 1)
	t.Logf("marker coord x=%d y=%d rowAt=%d cursor=%d", x, y, m.questionRowAt(x, y), m.question.Cursor())
	t.Logf("contentHeaderRows=%d vpHeight=%d vpY=%d qLine=%d qLines=%d qWidth=%d",
		m.contentHeaderRows(), m.viewport.Height, m.viewport.YOffset, m.questionViewLine, m.questionViewLines, m.questionViewWidth)
	for i, l := range strings.Split(m.View(), "\n") {
		t.Logf("%2d |%s|", i, ansi.Strip(l))
	}
}
