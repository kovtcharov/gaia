import patch
P = r"C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-d630a360\tui\internal\client\subprocess.go"

pairs = []

# --- struct fields ---
pairs.append(('''\tmu      sync.Mutex
\tproc    *procHandle
\tstdin   io.WriteCloser
\tstdout  *bufio.Scanner
\tstderr  *bytes.Buffer
\tstarted bool
\t// turnDone is closed by the in-flight turn's reader when it exits. nil when
\t// no turn is running.
\tturnDone chan struct{}
}''', '''\tmu      sync.Mutex
\tproc    *procHandle
\tstdin   io.WriteCloser
\tstdout  *bufio.Scanner
\tstderr  *bytes.Buffer
\tstarted bool
\t// turnDone is closed by the in-flight turn's reader when it exits. nil when
\t// no turn is running.
\tturnDone chan struct{}
\t// bypass is the permission mode the SESSION is in, which is not necessarily
\t// the one the child was launched with. A respawn rebuilds argv from this, so
\t// a `/bypass off` typed before a hard cancel cannot come back on by itself.
\tbypass bool
\t// respawned records that the child now backing this client is a REPLACEMENT
\t// for one that was killed. Read and cleared by the next Send, which reports
\t// it: the replacement has no loaded skills, no "always" grants and no prompt
\t// history, and a user who is not told that is reasoning about a session the
\t// agent no longer has.
\trespawned string
}''')) 

# --- constructor records launch bypass ---
pairs.append(('''func NewSubprocessClient(path string, args []string, debug bool) *SubprocessClient {
\treturn &SubprocessClient{
\t\tpath:  path,
\t\targs:  args,
\t\tdebug: debug,
\t}
}''', '''func NewSubprocessClient(path string, args []string, debug bool) *SubprocessClient {
\tc := &SubprocessClient{
\t\tpath:  path,
\t\targs:  args,
\t\tdebug: debug,
\t}
\tc.bypass = c.BypassAtLaunch()
\treturn c
}

// spawnArgs is argv for the NEXT child: the launch arguments with the bypass
// flag forced to match the session's current permission mode.
//
// Respawning from s.args verbatim silently reverted `/bypass off` — the killed
// child had prompts back on, its replacement did not, and the banner that is
// supposed to make unattended mode impossible to miss was gone. Deriving argv
// from the live mode means the flag cannot disagree with it; the control line
// SetBypassPermissions writes stays the mechanism for a LIVE child.
func (s *SubprocessClient) spawnArgs(bypass bool) []string {
\tout := make([]string, 0, len(s.args)+1)
\tfor _, a := range s.args {
\t\tif a == BypassPermissionsFlag {
\t\t\tcontinue
\t\t}
\t\tout = append(out, a)
\t}
\tif bypass {
\t\tout = append(out, BypassPermissionsFlag)
\t}
\treturn out
}'''))

patch.apply(P, pairs)
