# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Versioned, agent-independent contracts for the supervision feasibility probe."""

import hashlib
import json
import math
from dataclasses import dataclass
from uuid import UUID

PROTOCOL_VERSION = 1
LABEL_PREFIX = "ai.gaia.execution."


@dataclass(frozen=True)
class Identity:
    """A journaled identity must match every label before a runtime mutation."""

    deployment: str
    run: str
    generation: str
    nonce: str

    def __post_init__(self):
        for value in (self.deployment, self.run, self.generation, self.nonce):
            if str(UUID(value)) != value:
                raise ValueError("Identity fields must be canonical UUID strings")

    def labels(self):
        """Return the complete immutable ownership labels."""
        return {LABEL_PREFIX + name: value for name, value in vars(self).items()}


def canonical_request(prompt, session_id, max_steps=20):
    """Hash explicit normalized defaults without Unicode normalization/coercion."""
    if not isinstance(prompt, str) or not prompt or len(prompt.encode("utf-8")) > 65536:
        raise ValueError("prompt must contain 1..65536 UTF-8 bytes")
    if not isinstance(session_id, str) or str(UUID(session_id)) != session_id:
        raise ValueError("session_id must be a canonical UUID")
    if type(max_steps) is not int or not 1 <= max_steps <= 20:
        raise ValueError("max_steps must be an integer in 1..20")
    payload = json.dumps(
        {
            "schema": 1,
            "prompt": prompt,
            "session_id": session_id,
            "max_steps": max_steps,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deadline(now, duration):
    """Reject non-finite lease inputs instead of silently disabling expiry."""
    if not math.isfinite(now) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Lease duration must be positive and finite")
    return now + duration
