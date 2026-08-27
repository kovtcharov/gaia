package cli

import (
	"testing"

	"github.com/spf13/cobra"
)

// Cobra blocks a Windows console program launched from Explorer, printing
// "This is a command line tool. You need to open cmd.exe and run it from
// there." and waiting for Enter. GAIA is a TUI people double-click, so the
// guard has to be off. Empty string is cobra's documented off switch, and it
// reads like a no-op -- which is exactly why it needs a test holding it down.
func TestMousetrapIsDisabled(t *testing.T) {
	if cobra.MousetrapHelpText != "" {
		t.Errorf("cobra.MousetrapHelpText is %q; double-clicking gaia-tui.exe "+
			"would refuse to start and tell the user to open cmd.exe",
			cobra.MousetrapHelpText)
	}
}
