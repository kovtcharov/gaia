// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/amd/gaia/tui/internal/update"
)

var (
	updateInstallVersion string
	updateAssumeYes      bool
)

// newUpdater builds an updater bound to this binary and its version.
func newUpdater() (*update.Updater, error) {
	return update.New(update.Options{TUIVersion: version})
}

// --- gaia-tui update ---------------------------------------------------------

var updateCmd = &cobra.Command{
	Use:   "update",
	Short: "Check for, install, or roll back GAIA releases",
	Long: "Update the terminal UI and the flagship agent sidecar.\n\n" +
		"Nothing is downloaded until you say yes. `update check` compares versions " +
		"and downloads nothing at all; `update install` shows what it would fetch, " +
		"how big it is, and which files it would replace, then asks. Every download " +
		"is SHA-256 verified against the release manifest before it replaces " +
		"anything — a mismatch is refused outright.\n\n" +
		"Set GAIA_DISABLE_UPDATE=1 to turn all of it off.",
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		return cmd.Help()
	},
}

// --- gaia-tui update check ---------------------------------------------------

var updateCheckCmd = &cobra.Command{
	Use:          "check",
	Short:        "Report whether a newer GAIA release is published",
	Long:         "Resolve the update channel and compare versions. Downloads nothing.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		up, err := newUpdater()
		if err != nil {
			return err
		}
		result, err := up.Check(context.Background())
		if err != nil {
			return err
		}
		out, errW := cmd.OutOrStdout(), cmd.ErrOrStderr()
		printWarnings(errW, result.Warnings)

		if result.UpToDate {
			// The plan summary carries the channel line; with no plan to show,
			// this branch names it itself.
			fmt.Fprintf(out, "Channel   %s\n", result.Feed.String())
			fmt.Fprintf(out, "\nUp to date — the newest published release is %s.\n", result.Plan.Release)
			printComponentNotes(out, result.Plan)
			return nil
		}

		fmt.Fprint(out, result.Plan.Summary())
		switch {
		case result.Pinned != "":
			fmt.Fprintf(out, "\nThis machine is pinned to %s, so nothing will install on its own.\n"+
				"Run `gaia-tui update unpin` to resume updates.\n", result.Pinned)
		case result.SkippedThisRelease:
			fmt.Fprintf(out, "\nYou skipped %s earlier, so you will not be asked about it again.\n"+
				"Install it anyway with `gaia-tui update install --version %s`.\n",
				result.Plan.Release, result.Plan.Release)
		default:
			fmt.Fprintln(out, "\nRun `gaia-tui update install` to be asked about downloading it.")
		}
		printComponentNotes(out, result.Plan)
		return nil
	},
}

// printWarnings surfaces non-fatal problems — an abandoned lock taken over, a
// lock that could not be released. Each one means the next update behaves
// differently than the user expects, so none of them is dropped.
func printWarnings(w io.Writer, warnings []string) {
	for _, warning := range warnings {
		fmt.Fprintf(w, "[!] %s\n", warning)
	}
}

// printComponentNotes surfaces every component the plan is NOT updating and
// why. A component quietly missing from a release reads as "nothing to do".
func printComponentNotes(w io.Writer, plan *update.Plan) {
	for _, c := range plan.Components {
		if c.NeedsUpdate || c.Note == "" {
			continue
		}
		fmt.Fprintf(w, "  %-9s %s\n", c.Name, c.Note)
	}
}

// --- gaia-tui update install -------------------------------------------------

var updateInstallCmd = &cobra.Command{
	Use:   "install",
	Short: "Download and install a release, after asking",
	Long: "Show what would be downloaded — current version, available version, " +
		"download size, and the files that would be replaced — then ask.\n\n" +
		"Answer y to download and install, s to skip this version for good, or n " +
		"to be asked again next time. Nothing is fetched before you answer.\n\n" +
		"--version installs a specific release, including an OLDER one, and pins " +
		"this machine to it so the next check will not undo the rollback. Clear " +
		"that with `gaia-tui update unpin`.\n\n" +
		"--yes is the non-interactive opt-in for scripts. Without it and without a " +
		"terminal to ask on, the command refuses rather than assuming consent.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		up, err := newUpdater()
		if err != nil {
			return err
		}
		out, errW := cmd.OutOrStdout(), cmd.ErrOrStderr()

		prompter := &update.TTYPrompter{
			In:          cmd.InOrStdin(),
			Out:         out,
			Interactive: term.IsTerminal(int(os.Stdin.Fd())),
			AssumeYes:   updateAssumeYes,
		}
		result, err := up.Install(context.Background(), update.InstallRequest{
			Version:  updateInstallVersion,
			Prompter: prompter,
			Progress: progressPrinter(errW),
		})
		if result != nil {
			printWarnings(errW, result.Warnings)
		}
		if err != nil {
			// A release is two binaries. When the second one fails, say what the
			// first one already did — "update failed" alone would leave the user
			// guessing what state the machine is in.
			if result != nil && len(result.Installed) > 0 {
				fmt.Fprintf(errW, "\nInstalled before this failed: %s. The rest of the "+
					"release was not installed; re-run `gaia-tui update install` to finish.\n",
					joinOrNone(result.Installed))
			}
			return err
		}

		switch {
		case result.UpToDate && result.Plan.Explicit:
			// Never "the newest" here — the user may have named an older one.
			fmt.Fprintf(out, "Already on %s — nothing to download.\n", result.Plan.Release)
			if result.Pinned != "" {
				fmt.Fprintf(out, "Pinned to %s — auto-update stays paused until you run "+
					"`gaia-tui update unpin`.\n", result.Pinned)
			}
			printComponentNotes(out, result.Plan)
			return nil
		case result.UpToDate:
			fmt.Fprintf(out, "Up to date — %s is the newest published release, and it is "+
				"what is installed.\n", result.Plan.Release)
			printComponentNotes(out, result.Plan)
			return nil
		case result.AlreadySkipped:
			fmt.Fprintf(out, "%s is available, but you skipped it earlier so you are not "+
				"being asked again.\nInstall it anyway with `gaia-tui update install "+
				"--version %s`.\n", result.Plan.Release, result.Plan.Release)
			return nil
		}

		switch result.Decision {
		case update.DecisionSkip:
			fmt.Fprintf(out, "\nSkipped %s — you will not be asked about this version again.\n"+
				"Install it later with `gaia-tui update install --version %s`.\n",
				result.Plan.Release, result.Plan.Release)
		case update.DecisionDecline:
			fmt.Fprintln(out, "\nDeclined — nothing was downloaded. Run `gaia-tui update install` "+
				"when you want it.")
		case update.DecisionAccept:
			fmt.Fprintf(out, "\nInstalled %s: %s\n", result.Plan.Release,
				joinOrNone(result.Installed))
			if result.Pinned != "" {
				fmt.Fprintf(out, "Pinned to %s — auto-update stays paused until you run "+
					"`gaia-tui update unpin`.\n", result.Pinned)
			}
			if result.Plan.ReplacesRunningBinary {
				fmt.Fprintln(out, "The running session keeps the old binary until you start "+
					"gaia-tui again.")
			}
		}
		return nil
	},
}

// progressPrinter reports download progress on stderr, one line per component,
// so a long download never looks like a hang.
func progressPrinter(w io.Writer) func(string, int64, int64) {
	last := map[string]time.Time{}
	return func(component string, done, total int64) {
		now := time.Now()
		if prev, ok := last[component]; ok && now.Sub(prev) < time.Second && done != total {
			return
		}
		last[component] = now
		if total > 0 {
			fmt.Fprintf(w, "  %s  %s / %s (%.0f%%)\n", component,
				update.FormatSize(done), update.FormatSize(total),
				float64(done)/float64(total)*100)
			return
		}
		fmt.Fprintf(w, "  %s  %s\n", component, update.FormatSize(done))
	}
}

func joinOrNone(items []string) string {
	if len(items) == 0 {
		return "nothing (no component needed replacing)"
	}
	joined := items[0]
	for _, item := range items[1:] {
		joined += ", " + item
	}
	return joined
}

// --- gaia-tui update list ----------------------------------------------------

var updateListCmd = &cobra.Command{
	Use:          "list",
	Short:        "List published releases, newest first",
	Long:         "Show what the update channel publishes, marking what is installed and what is pinned.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		up, err := newUpdater()
		if err != nil {
			return err
		}
		releases, cfg, ref, err := up.List(context.Background())
		if err != nil {
			return err
		}
		st, err := up.Status()
		if err != nil {
			return err
		}

		out := cmd.OutOrStdout()
		fmt.Fprintf(out, "Channel   %s\n\n", ref.String())
		fmt.Fprintf(out, "%-14s %-22s %s\n", "VERSION", "PUBLISHED", "STATE")
		for _, r := range releases {
			published := "-"
			if !r.PublishedAt.IsZero() {
				published = r.PublishedAt.UTC().Format("2006-01-02 15:04 UTC")
			}
			state := ""
			if r.Version == st.SidecarVersion {
				state = "installed"
			}
			if r.Version == cfg.PinnedVersion {
				if state != "" {
					state += ", "
				}
				state += "pinned"
			}
			if r.Version == cfg.SkippedVersion {
				if state != "" {
					state += ", "
				}
				state += "skipped"
			}
			fmt.Fprintf(out, "%-14s %-22s %s\n", r.Version, published, state)
		}
		fmt.Fprintln(out, "\nInstall one with `gaia-tui update install --version <version>` — "+
			"an older version is allowed, and pins this machine to it.")
		return nil
	},
}

// --- gaia-tui update pin / unpin ---------------------------------------------

var updatePinCmd = &cobra.Command{
	Use:   "pin <version>",
	Short: "Pause auto-update at a release",
	Long: "Record a version pin. While it is set, `gaia-tui update install` refuses " +
		"to move to the newest release — it names the pin instead of quietly " +
		"upgrading past it. Installs nothing on its own.",
	Args:         cobra.ExactArgs(1),
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		up, err := newUpdater()
		if err != nil {
			return err
		}
		if err := up.Pin(args[0]); err != nil {
			return err
		}
		fmt.Fprintf(cmd.OutOrStdout(),
			"Pinned to %s — auto-update is paused until you run `gaia-tui update unpin`.\n"+
				"Recorded in %s\n", args[0], update.ConfigPath(up.GaiaDir()))
		return nil
	},
}

var updateUnpinCmd = &cobra.Command{
	Use:          "unpin",
	Short:        "Clear the version pin and resume updates",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		up, err := newUpdater()
		if err != nil {
			return err
		}
		previous, err := up.Unpin()
		if err != nil {
			return err
		}
		out := cmd.OutOrStdout()
		if previous == "" {
			fmt.Fprintln(out, "No version pin was set — updates were already running normally.")
			return nil
		}
		fmt.Fprintf(out, "Cleared the pin on %s — updates resume.\n"+
			"Run `gaia-tui update check` to see what is published.\n", previous)
		return nil
	},
}

// --- gaia-tui update status --------------------------------------------------

var updateStatusCmd = &cobra.Command{
	Use:          "status",
	Short:        "Show the update channel, versions, pin, and kill-switch state",
	Long:         "Report everything about the update state. Reads local files only — no network.",
	Args:         cobra.NoArgs,
	SilenceUsage: true,
	RunE: func(cmd *cobra.Command, args []string) error {
		up, err := newUpdater()
		if err != nil {
			return err
		}
		st, err := up.Status()
		if err != nil {
			return err
		}
		out := cmd.OutOrStdout()

		if st.Disabled {
			fmt.Fprintf(out, "Updates      DISABLED (%s=%s)\n", update.EnvDisable, st.DisableValue)
		} else {
			fmt.Fprintln(out, "Updates      enabled")
		}
		if st.FeedErr != nil {
			fmt.Fprintf(out, "Channel      %v\n", st.FeedErr)
		} else {
			fmt.Fprintf(out, "Channel      %s\n", st.Feed.String())
		}
		fmt.Fprintf(out, "Config       %s\n\n", st.ConfigPath)

		fmt.Fprintf(out, "TUI          %s\n             %s\n", orDash(st.TUIVersion), st.TUIPath)
		if st.SidecarVersion != "" {
			fmt.Fprintf(out, "Sidecar      %s\n             %s\n", st.SidecarVersion, st.SidecarPath)
		} else {
			fmt.Fprintf(out, "Sidecar      %s\n", st.SidecarNote)
		}

		fmt.Fprintf(out, "\nPin          %s\n", orDash(st.Pinned))
		fmt.Fprintf(out, "Skipped      %s\n", orDash(st.Skipped))
		fmt.Fprintf(out, "Last check   %s\n", orDash(st.LastCheck))
		fmt.Fprintf(out, "Last seen    %s\n", orDash(st.LastSeen))
		if st.StaleBackup != "" {
			fmt.Fprintf(out, "\n[!] A previous update left %s behind. It is harmless; delete it "+
				"once no other GAIA process is running.\n", st.StaleBackup)
		}
		return nil
	},
}

func init() {
	updateInstallCmd.Flags().StringVar(&updateInstallVersion, "version", "",
		"install this exact release instead of the newest one. An older version is "+
			"allowed, and pins this machine to it (clear with `update unpin`)")
	updateInstallCmd.Flags().BoolVar(&updateAssumeYes, "yes", false,
		"accept the download without asking — the explicit opt-in for scripts and CI. "+
			"Without it, a non-interactive run refuses rather than assuming consent")

	updateCmd.AddCommand(updateCheckCmd, updateInstallCmd, updateListCmd,
		updatePinCmd, updateUnpinCmd, updateStatusCmd)
	rootCmd.AddCommand(updateCmd)

	// A Windows swap leaves the replaced binary aside because a running .exe
	// cannot be deleted. The next start is the one moment nothing holds it.
	cobra.OnInitialize(sweepStaleUpdateBackup)
}

// sweepStaleUpdateBackup clears the leftover from a previous self-replacement.
//
// A backup that is still held open (an older gaia-tui outliving the swap) is
// left where it is: that is expected on Windows and reporting it on every
// launch would be noise, so it surfaces under --dev and in `update status`.
func sweepStaleUpdateBackup() {
	up, err := update.New(update.Options{TUIVersion: version})
	if err != nil {
		debugLog("update: cannot locate this binary to sweep its backup: %v", err)
		return
	}
	if err := update.CleanStaleBackup(up.TUIPath()); err != nil {
		debugLog("update: %v", err)
	}
}
