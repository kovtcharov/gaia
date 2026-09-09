import patch
P = r"C:\Users\14255\Work\gaia\.claudia-worktrees\claudia-task-d630a360\tui\internal\client\subprocess.go"

old_ph = '''// procHandle owns one child process.
//
// Reaping is the READER's job: os/exec forbids calling Wait before all reads
// from a pipe have completed, so a kill from elsewhere must not also reap — it
// would close the stdout pipe under the reader and turn a deliberate kill into a
// spurious "file already closed" read error.
type procHandle struct {
\tcmd      *exec.Cmd
\twaitOnce sync.Once
\tstate    *os.ProcessState
}

// reap waits for the child and returns its final state. Safe to call more than
// once; only the first call waits. Call it only once reads are done.
func (p *procHandle) reap() *os.ProcessState {
\tp.waitOnce.Do(func() {
\t\t_ = p.cmd.Wait()
\t\tp.state = p.cmd.ProcessState
\t})
\treturn p.state
}

// kill signals the child without reaping it.
func (p *procHandle) kill() {
\tif p.cmd.Process != nil {
\t\t_ = p.cmd.Process.Kill()
\t}
}
'''

new_ph = '''// procHandle owns one child process AND every process that child started.
//
// Reaping is the READER's job: os/exec forbids calling Wait before all reads
// from a pipe have completed, so a kill from elsewhere must not also reap — it
// would close the stdout pipe under the reader and turn a deliberate kill into a
// spurious "file already closed" read error.
type procHandle struct {
\tcmd      *exec.Cmd
\tgroup    *processGroup
\twaitOnce sync.Once
\tstate    *os.ProcessState
}

// reap waits for the child and returns its final state. Safe to call more than
// once; only the first call waits. Call it only once reads are done.
func (p *procHandle) reap() *os.ProcessState {
\tp.waitOnce.Do(func() {
\t\t_ = p.cmd.Wait()
\t\tp.state = p.cmd.ProcessState
\t\tif p.group != nil {
\t\t\tp.group.close()
\t\t}
\t})
\treturn p.state
}

// kill terminates the child AND its descendants, without reaping it.
//
// The whole tree, not just cmd.Process: the released agent is a PyInstaller
// one-file binary, so cmd.Process is the bootloader and the interpreter that
// runs the turn is its child, holding both ends of the pipe. Killing the
// bootloader alone left the cancelled tool call running to completion, and the
// surviving child then consumed the user's next message.
//
// Both mechanisms are tried and both failures are returned: a group kill that
// did not work must never be reported as a stopped agent.
func (p *procHandle) kill() error {
\tvar errs []error
\tif p.group != nil {
\t\tif err := p.group.terminate(); err != nil {
\t\t\terrs = append(errs, err)
\t\t}
\t}
\tif p.cmd.Process != nil {
\t\tif err := p.cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
\t\t\terrs = append(errs, fmt.Errorf("could not stop agent process %d: %w", p.cmd.Process.Pid, err))
\t\t}
\t}
\treturn errors.Join(errs...)
}
'''

imports_old = '''\t"encoding/json"
\t"fmt"
'''
imports_new = '''\t"encoding/json"
\t"errors"
\t"fmt"
'''

patch.apply(P, [(imports_old, imports_new), (old_ph, new_ph)])
