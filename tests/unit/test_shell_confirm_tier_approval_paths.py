# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""What a gated shell command needs in order to run, and what it must not.

The companion to ``test_shell_confirm_tier_never_fails_open.py``: that file
pins that nothing runs unapproved, this one pins that everything which *is*
approved still runs, and that the approval cannot be stretched to cover a
second command.

Assertions check the side effect on disk wherever one exists — a refusal that
already deleted the file is the failure both files exist to catch.
"""

import pytest

from gaia.agents.base.approval import _tool_call_approval, call_has_user_approval
from gaia.agents.base.tools import get_tool_metadata
from gaia.agents.tools.shell_tools import TIER_CONFIRM, TIER_REFUSE, ShellToolsMixin

_SHELL_TOOL = "run_shell_command"


class _Console:
    """Answers confirmation prompts, and records what it was asked."""

    blocking_confirmation = True

    def __init__(self, *, approve=True, full_access=False, auto_approve=False):
        self._approve = approve
        self.full_access = full_access
        self.auto_approve_gated_tools = full_access or auto_approve
        self.asked: list = []

    def confirm_tool_execution(self, tool_name, tool_args):
        self.asked.append((tool_name, dict(tool_args)))
        return self._approve

    def confirmation_denied_reason(self, tool_name):
        return f"Tool '{tool_name}' was denied."


class _Host(ShellToolsMixin):
    """A bare tool host: it owns the shell tool but runs no confirmation gate."""

    debug = False

    def __init__(self, console=None):
        super().__init__()
        if console is not None:
            self.console = console
        self.register_shell_tools()


class _GatedAgent(_Host):
    """Enough of ``Agent`` to run ``_execute_tool`` end to end, and no more.

    Borrows the real methods rather than re-implementing them, so the gate
    under test is the one that ships — including ``_call_tool_bounded``, which
    runs the tool body on a worker thread. An approval that does not survive
    that hop refuses every command a user just said yes to.
    """

    from gaia.agents.base.agent import Agent as _Base

    CONFIRMATION_REQUIRED_TOOLS: frozenset = frozenset()
    confirmation_required_tools = _Base.confirmation_required_tools
    _policy_refusal = _Base._policy_refusal
    _call_is_pre_authorized = _Base._call_is_pre_authorized
    _tool_requires_confirmation = _Base._tool_requires_confirmation
    _confirmation_denied_error = _Base._confirmation_denied_error
    _execute_tool = _Base._execute_tool
    _call_tool_bounded = _Base._call_tool_bounded
    _resolve_tool_timeout = _Base._resolve_tool_timeout
    _coerce_tool_args = _Base._coerce_tool_args
    _coerce_scalar = _Base._coerce_scalar
    _COERCIBLE = _Base._COERCIBLE
    _resolve_tool_name = _Base._resolve_tool_name
    _on_tool_invoked = _Base._on_tool_invoked
    _fold_tool_usage = _Base._fold_tool_usage
    current_plan = None
    current_step = 0

    def __init__(self, console):
        super().__init__(console)
        self._tools_registry = {_SHELL_TOOL: get_tool_metadata(_SHELL_TOOL)}

    def run(self, command, cwd):
        return self._execute_tool(
            _SHELL_TOOL, {"command": command, "working_directory": str(cwd)}
        )


def _call_directly(host, command, cwd):
    """Reach the tool without the gate — a script, a test, an embedder."""
    return get_tool_metadata(_SHELL_TOOL)["function"](
        command=command, working_directory=str(cwd)
    )


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "keep.txt").write_text("precious\n")
    return tmp_path


@pytest.fixture
def env_pre_approves(monkeypatch):
    """``GAIA_AUTO_APPROVE_TOOLS=1``, as an unattended run sets it."""
    monkeypatch.setattr(
        "gaia.agents.base.console.auto_approve_env_enabled", lambda: True
    )


# ---------------------------------------------------------------------------
# 1. An approved command still runs
# ---------------------------------------------------------------------------


def test_the_user_approving_the_prompt_runs_the_command(workdir):
    agent = _GatedAgent(_Console(approve=True))

    result = agent.run("rm keep.txt", workdir)

    assert [name for name, _ in agent.console.asked] == [_SHELL_TOOL]
    assert result["status"] == "success", result
    assert not (workdir / "keep.txt").exists(), "approval did not reach the command"


def test_the_user_declining_the_prompt_leaves_the_file_alone(workdir):
    agent = _GatedAgent(_Console(approve=False))

    result = agent.run("rm keep.txt", workdir)

    assert result["status"] == "denied", result
    assert (workdir / "keep.txt").exists()


# ---------------------------------------------------------------------------
# 2. Full access runs a confirmable command unprompted
# ---------------------------------------------------------------------------


def test_full_access_runs_a_confirmable_command_with_no_gate(workdir):
    result = _call_directly(_Host(_Console(full_access=True)), "rm keep.txt", workdir)

    assert result["status"] == "success", result
    assert not (workdir / "keep.txt").exists()


# ---------------------------------------------------------------------------
# 3. A blanket pre-approval answers prompts; it never widens the shell
# ---------------------------------------------------------------------------


def test_the_env_opt_in_refuses_a_confirmable_command(workdir, env_pre_approves):
    result = _call_directly(_Host(), "rm keep.txt", workdir)

    assert result["status"] == "error"
    assert result["tier"] == TIER_REFUSE
    assert "GAIA_AUTO_APPROVE_TOOLS" in result["hint"]
    assert (workdir / "keep.txt").exists()


def test_an_embedders_opt_in_refuses_a_confirmable_command(workdir):
    host = _Host(_Console(auto_approve=True))

    result = _call_directly(host, "rm keep.txt", workdir)

    assert result["status"] == "error"
    assert "auto_approve_gated_tools" in result["hint"]
    assert (workdir / "keep.txt").exists()


def test_the_blanket_message_wins_even_through_the_gate(workdir, env_pre_approves):
    """An unattended console says yes to the prompt; the shell still says no."""
    agent = _GatedAgent(_Console(approve=True, auto_approve=True))

    result = agent.run("rm keep.txt", workdir)

    assert result["status"] == "error"
    assert "auto_approve_gated_tools" in result["hint"]
    assert (workdir / "keep.txt").exists()


# ---------------------------------------------------------------------------
# 4. TIER_REFUSE is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git -c core.pager=evil.sh status",
        "powershell -EncodedCommand aQBlAHgA",
        "ls && rm keep.txt",
        "ls > out.txt",
    ],
)
def test_a_refused_command_stays_refused_under_every_approval(command, workdir):
    for host in (
        _Host(),
        _Host(_Console(full_access=True)),
        _Host(_Console(auto_approve=True)),
    ):
        result = _call_directly(host, command, workdir)
        assert result["status"] == "error", result
        assert result["tier"] == TIER_REFUSE, result

    agent = _GatedAgent(_Console(approve=True))
    result = agent.run(command, workdir)
    assert result["status"] == "error", result
    assert agent.console.asked == [], "a refused command reached the prompt"
    assert (workdir / "keep.txt").exists()
    assert not (workdir / "out.txt").exists()


def test_a_refusal_is_not_something_an_approval_ticket_can_lift(workdir):
    command = "ls && rm keep.txt"
    with _tool_call_approval(_SHELL_TOOL, {"command": command}, granted=True):
        result = _call_directly(_Host(), command, workdir)

    assert result["status"] == "error"
    assert result["tier"] == TIER_REFUSE
    assert (workdir / "keep.txt").exists()


# ---------------------------------------------------------------------------
# 5. The read-only list still runs with no console at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("command", ["ls", "pwd", "git status", "cat keep.txt"])
def test_a_read_only_command_runs_with_no_console(command, workdir):
    result = _call_directly(_Host(), command, workdir)

    assert result["status"] == "success", result


def test_the_read_only_list_still_runs_under_a_blanket_opt_in(
    workdir, env_pre_approves
):
    assert _call_directly(_Host(), "ls", workdir)["status"] == "success"


# ---------------------------------------------------------------------------
# The ticket itself: scoped to one call, spendable once, never inherited
# ---------------------------------------------------------------------------


def test_the_ticket_does_not_cover_a_different_command(workdir):
    with _tool_call_approval(_SHELL_TOOL, {"command": "rm nothing.txt"}, granted=True):
        result = _call_directly(_Host(), "rm keep.txt", workdir)

    assert result["status"] == "error", result
    assert (workdir / "keep.txt").exists(), "one approval covered a second command"


def test_the_ticket_does_not_outlive_the_call_it_was_issued_for(workdir):
    with _tool_call_approval(_SHELL_TOOL, {"command": "rm keep.txt"}, granted=True):
        pass

    result = _call_directly(_Host(), "rm keep.txt", workdir)

    assert result["status"] == "error", result
    assert (workdir / "keep.txt").exists()


def test_a_second_command_in_the_same_turn_is_approved_on_its_own(workdir):
    """The gate runs per call, so the second command raises its own prompt."""
    agent = _GatedAgent(_Console(approve=True))
    (workdir / "also.txt").write_text("x\n")

    agent.run("rm keep.txt", workdir)
    agent.run("rm also.txt", workdir)

    assert [args["command"] for _, args in agent.console.asked] == [
        "rm keep.txt",
        "rm also.txt",
    ]


def test_another_tools_approval_is_not_this_tools_approval(workdir):
    with _tool_call_approval("write_file", {"command": "rm keep.txt"}, granted=True):
        result = _call_directly(_Host(), "rm keep.txt", workdir)

    assert result["status"] == "error", result
    assert (workdir / "keep.txt").exists()


def test_a_declined_gate_leaves_no_ticket_behind():
    with _tool_call_approval(_SHELL_TOOL, {"command": "rm keep.txt"}, granted=False):
        assert call_has_user_approval(_SHELL_TOOL, command="rm keep.txt") is False


def test_an_unapproved_nested_call_cannot_spend_the_outer_approval():
    """A tool body that invokes another tool starts from no approval."""
    with _tool_call_approval(_SHELL_TOOL, {"command": "rm keep.txt"}, granted=True):
        with _tool_call_approval("read_file", {"file_path": "x"}, granted=False):
            assert call_has_user_approval(_SHELL_TOOL, command="rm keep.txt") is False
        assert call_has_user_approval(_SHELL_TOOL, command="rm keep.txt") is True


def test_asking_whether_the_tool_was_approved_at_all_is_refused():
    """Approval is per call. A name-only question has no safe answer."""
    with pytest.raises(ValueError, match="at least one argument"):
        call_has_user_approval(_SHELL_TOOL)


def test_the_pre_prompt_gate_still_lets_a_confirmable_command_ask():
    """The fix must not push CONFIRM-tier commands back into blanket refusal."""
    host = _Host()

    assert host.policy_refusal_for_call(_SHELL_TOOL, {"command": "rm keep.txt"}) is None
    error, _ = host._validate_shell_command("rm keep.txt")
    assert error["tier"] == TIER_CONFIRM
