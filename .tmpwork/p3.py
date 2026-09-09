import patch
P = r"C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-d630a360\tui\internal\client\subprocess.go"

pairs = []

# turnState gains the respawn notice
pairs.append(('''type turnState struct {
\tstdin    io.WriteCloser
\tscanner  *bufio.Scanner
\tproc     *procHandle
\tstderr   *bytes.Buffer
\tturnDone chan struct{}
}''', '''type turnState struct {
\tstdin    io.WriteCloser
\tscanner  *bufio.Scanner
\tproc     *procHandle
\tstderr   *bytes.Buffer
\tturnDone chan struct{}
\t// notice is non-empty when this turn is the first against a REPLACEMENT
\t// child, and says what the replacement no longer knows.
\tnotice string
}'''))

# startLocked
pairs.append(('''func (s *SubprocessClient) startLocked() (turnState, error) {
\tif s.started {
\t\tdone := make(chan struct{})
\t\ts.turnDone = done
\t\treturn turnState{s.stdin, s.stdout, s.proc, s.stderr, done}, nil
\t}
\tif s.path == "" {
\t\treturn turnState{}, fmt.Errorf("no agent binary was given, so nothing can be launched")
\t}

\tcmd := exec.Command(s.path, s.args...)
\tstderr := &bytes.Buffer{}
\tcmd.Stderr = stderr
''', '''func (s *SubprocessClient) startLocked() (turnState, error) {
\tif s.started {
\t\t// Serialization is a contract, not a hope: two turns sharing one
\t\t// bufio.Scanner means two goroutines reading the same pipe, and the
\t\t// first one to finish closes it under the second ("file already
\t\t// closed"). A caller that got here overlapped its Send calls.
\t\tif s.turnDone != nil {
\t\t\tselect {
\t\t\tcase <-s.turnDone:
\t\t\tdefault:
\t\t\t\treturn turnState{}, fmt.Errorf(
\t\t\t\t\t"the previous message is still running, so this one cannot be sent — " +
\t\t\t\t\t\t"press Esc to stop it first")
\t\t\t}
\t\t}
\t\tdone := make(chan struct{})
\t\ts.turnDone = done
\t\tnotice := s.respawned
\t\ts.respawned = ""
\t\treturn turnState{s.stdin, s.stdout, s.proc, s.stderr, done, notice}, nil
\t}
\tif s.path == "" {
\t\treturn turnState{}, fmt.Errorf("no agent binary was given, so nothing can be launched")
\t}

\tcmd := exec.Command(s.path, s.spawnArgs(s.bypass)...)
\tstderr := &bytes.Buffer{}
\tcmd.Stderr = stderr

\t// Created before Start so a POSIX child is forked straight into its own
\t// process group; on Windows the job is joined immediately after Start.
\tgroup, err := newProcessGroup()
\tif err != nil {
\t\treturn turnState{}, err
\t}
\tgroup.prepare(cmd)
'''))

# stdin pipe err shadowing: `stdinPipe, err := cmd.StdinPipe()` now redeclares err -> change to =
pairs.append(('''\tstdinPipe, err := cmd.StdinPipe()
\tif err != nil {
\t\treturn turnState{}, fmt.Errorf("failed to create stdin pipe: %w", err)
\t}''', '''\tstdinPipe, err := cmd.StdinPipe()
\tif err != nil {
\t\tgroup.close()
\t\treturn turnState{}, fmt.Errorf("failed to create stdin pipe: %w", err)
\t}'''))

pairs.append(('''\tstdoutPipe, err := cmd.StdoutPipe()
\tif err != nil {
\t\treturn turnState{}, fmt.Errorf("failed to create stdout pipe: %w", err)
\t}''', '''\tstdoutPipe, err := cmd.StdoutPipe()
\tif err != nil {
\t\tgroup.close()
\t\treturn turnState{}, fmt.Errorf("failed to create stdout pipe: %w", err)
\t}'''))

pairs.append(('''\tif err := cmd.Start(); err != nil {
\t\treturn turnState{}, fmt.Errorf("failed to start agent %q: %w", s.path, err)
\t}

\tdone := make(chan struct{})
\ts.stdin = stdinPipe
\ts.stdout = scanner
\ts.stderr = stderr
\ts.proc = &procHandle{cmd: cmd}
\ts.started = true
\ts.turnDone = done
\treturn turnState{stdinPipe, scanner, s.proc, stderr, done}, nil
}''', '''\tif err := cmd.Start(); err != nil {
\t\tgroup.close()
\t\treturn turnState{}, fmt.Errorf("failed to start agent %q: %w", s.path, err)
\t}
\t// A grouping failure is fatal, not a warning: without it a later cancel
\t// would kill only the bootloader and leave the real agent running the tool
\t// call the user asked to stop.
\tif err := group.attach(cmd); err != nil {
\t\t_ = cmd.Process.Kill()
\t\t_ = cmd.Wait()
\t\tgroup.close()
\t\treturn turnState{}, fmt.Errorf(
\t\t\t"started agent %q but could not take ownership of its child processes, "+
\t\t\t\t"so a cancelled turn could not be stopped — refusing to run it: %w", s.path, err)
\t}

\tdone := make(chan struct{})
\ts.stdin = stdinPipe
\ts.stdout = scanner
\ts.stderr = stderr
\ts.proc = &procHandle{cmd: cmd, group: group}
\ts.started = true
\ts.turnDone = done
\tnotice := s.respawned
\ts.respawned = ""
\treturn turnState{stdinPipe, scanner, s.proc, stderr, done, notice}, nil
}'''))

patch.apply(P, pairs)
