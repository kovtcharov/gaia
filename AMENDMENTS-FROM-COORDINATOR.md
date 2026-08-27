
## 8. Boot into the agent, not the hub — and fix the vanishing colour

Two changes from the user, both in the TUI rather than the installer.

### 8a. Drop the hub landing page

We ship one agent. The first screen should be the flagship's chat view, not a
catalogue listing thirteen coming-soon rows. Keep the hub reachable: the command
palette already has `/hub` ("Return to the agent hub"), so that becomes the way
in rather than the default.

- the flagship is subprocess transport with `BinaryPath: gaia-agent`, so on an
  installed machine the binary is right there — boot into it, and fall back to a
  picker only if it is genuinely absent
- this makes §7 items 1-3 mostly moot for the *default* path, but the catalog
  still has to work for `/hub`. Do not leave it broken; just take it off the
  critical path.
- do NOT spawn the agent or reach Lemonade to prove this. Opening the chat view
  and showing the composer is the check.

### 8b. Colour/syntax highlighting sometimes disappears

Intermittent, which points at colour-profile detection rather than the styles.
On Windows the profile depends on the console host and environment —
`WT_SESSION`, `TERM`, `NO_COLOR`, and whether `ENABLE_VIRTUAL_TERMINAL_PROCESSING`
is set on the output handle. From Windows Terminal you get truecolor; through
conhost, a PowerShell wrapper, or a shortcut, lipgloss/termenv can fall back to
the Ascii profile and every style silently flattens.

- find where the profile is resolved
- enable VT processing explicitly on the Windows output handle at startup
- make the launcher prefer Windows Terminal when present
- log the detected profile under `--dev`, so next time this is diagnosable
  instead of "sometimes"

Reproduce both ways, before and after: double-clicked shortcut, and `gaia-tui`
typed into Windows Terminal. Report which host produced which profile.

Everything else stands: no Lemonade, no model load, do not touch the Agent UI
installer, commit to your own branch.
