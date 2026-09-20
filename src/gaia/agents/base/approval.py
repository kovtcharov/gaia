# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Positive evidence that the user approved the call a tool body is running.

A tool that asks "was I approved?" cannot answer it from what it is handed:
the arguments are the model's, and the absence of a blanket-approval flag says
only that no blanket approval exists — not that a person said yes. Something
has to hand the answer down, and it has to be something the model cannot write.

Two callables, asymmetric visibility, mirroring ``gaia.connectors.context``:

- ``_tool_call_approval(...)`` — **PRIVATE**. Only ``Agent._execute_tool``
  calls it, via the explicit private import path, immediately after its
  confirmation gate returns and only around that one dispatch.
- ``call_has_user_approval(...)`` — **PUBLIC**. A tool body may read whether
  the call it is running was approved; it has no way to say that it was.

Fails closed in every direction. The ticket is absent by default, so a tool
reached by any route other than the gate — a unit test, a script, an embedding
application — reads "not approved". It names the tool and the arguments it was
granted for, so a tool body that calls another tool cannot spend the outer
call's approval. And it lives in a ``ContextVar`` set and reset around a single
dispatch, so it cannot outlive the call it belongs to: a worker thread started
with a copied context (``Agent._call_tool_bounded``) inherits its own snapshot,
and the reset in the caller cannot leak into the next call or into a sibling.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

_approved_call_var: ContextVar[Optional[Tuple[str, Dict[str, Any]]]] = ContextVar(
    "gaia_approved_tool_call", default=None
)


@contextmanager
def _tool_call_approval(
    tool_name: str, tool_args: Mapping[str, Any], *, granted: bool
) -> Iterator[None]:
    """Record — for this dispatch only — whether the gate approved this call.

    ``granted=False`` clears the ticket rather than leaving the caller's in
    place, so a tool invoked from inside an approved tool's body starts from no
    approval instead of inheriting one.

    PRIVATE — the agent runtime imports this by its explicit private path. It is
    not re-exported from ``gaia.agents.base``, and nothing the model emits (a
    tool name and a JSON argument object) can reach it.
    """
    ticket = (tool_name, dict(tool_args)) if granted else None
    token = _approved_call_var.set(ticket)
    try:
        yield
    finally:
        _approved_call_var.reset(token)


def call_has_user_approval(tool_name: str, **must_match: Any) -> bool:
    """True when the user approved *this* call, through the agent's gate.

    Args:
        tool_name: The tool whose body is asking.
        must_match: The arguments that identify the call, e.g.
            ``command="rm notes.txt"``. Every one must equal what the gate was
            asked to approve; an approval for a different command is not an
            approval for this one. At least one is required.

    Raises:
        ValueError: If no identifying argument is given. "Some call to this
            tool was approved" is not an answer any caller should act on.
    """
    if not must_match:
        raise ValueError(
            "call_has_user_approval needs at least one argument identifying the "
            "call (e.g. command=...); approval is granted per call, not per tool."
        )
    approved = _approved_call_var.get()
    if approved is None:
        return False
    approved_name, approved_args = approved
    if approved_name != tool_name:
        return False
    return all(approved_args.get(key) == value for key, value in must_match.items())
