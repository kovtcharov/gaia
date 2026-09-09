// Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//go:build windows

package client

import (
	"fmt"
	"os/exec"
	"unsafe"

	"golang.org/x/sys/windows"
)

// processGroup makes every descendant of the agent reachable by one kill.
//
// The shipped agent is a PyInstaller ONE-FILE binary: the process exec.Command
// starts is the bootloader, and the interpreter that runs the turn — and holds
// both ends of the pipe — is its child. Killing the bootloader alone leaves
// that child running the tool call the user just cancelled, and its next write
// arrives on a pipe the host has moved on from.
//
// A Windows job object is the only mechanism that reaches it: a process created
// by a process already in a job joins the same job, so terminating the job
// terminates the whole tree in one call, however deep it goes.
type processGroup struct {
	job windows.Handle
}

// newProcessGroup creates the job the next child will be assigned to.
func newProcessGroup() (*processGroup, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, fmt.Errorf("could not create the job object that stops the agent's child processes: %w", err)
	}
	// KILL_ON_JOB_CLOSE: if this process dies without ever calling terminate,
	// the handle closes with it and Windows reaps the tree. Without it a
	// crashed TUI leaks a running agent holding the model slot.
	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{
		BasicLimitInformation: windows.JOBOBJECT_BASIC_LIMIT_INFORMATION{
			LimitFlags: windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
		},
	}
	if _, err := windows.SetInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		windows.CloseHandle(job)
		return nil, fmt.Errorf("could not configure the agent's job object: %w", err)
	}
	return &processGroup{job: job}, nil
}

// prepare is a no-op on Windows: the grouping is done by assignment after
// Start, not by a spawn attribute.
func (g *processGroup) prepare(*exec.Cmd) {}

// attach puts the freshly started child — and therefore everything it goes on
// to spawn — into the job.
func (g *processGroup) attach(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return fmt.Errorf("cannot group a process that was never started")
	}
	h, err := windows.OpenProcess(
		windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(cmd.Process.Pid))
	if err != nil {
		return fmt.Errorf("could not open agent process %d to group it: %w", cmd.Process.Pid, err)
	}
	defer windows.CloseHandle(h)
	if err := windows.AssignProcessToJobObject(g.job, h); err != nil {
		return fmt.Errorf("could not group agent process %d: %w", cmd.Process.Pid, err)
	}
	return nil
}

// terminate kills every process in the job. Terminating a job whose members
// have all already exited succeeds, so a late call is not an error.
func (g *processGroup) terminate() error {
	if err := windows.TerminateJobObject(g.job, 1); err != nil {
		return fmt.Errorf("could not stop the agent's process tree: %w", err)
	}
	return nil
}

// close releases the job handle. Any surviving member is killed by
// KILL_ON_JOB_CLOSE, which is the intended reap for a client being discarded.
func (g *processGroup) close() {
	windows.CloseHandle(g.job)
}
