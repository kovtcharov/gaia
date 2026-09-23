# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Serialized byte accounting before an executor's output enters its queue."""

import json
import queue
import threading


class OutputLimitError(RuntimeError):
    """Output exceeded the service policy; the run must stop."""


class BoundedOutput(queue.Queue):
    """Bound total emitted bytes as well as each event, independent of consumers."""

    def __init__(self, cancel, max_bytes=4 * 1024 * 1024, event_bytes=256 * 1024):
        super().__init__()
        self.cancel = cancel
        self.max_bytes, self.event_bytes = max_bytes, event_bytes
        self.total = 0
        self.accounting = threading.Lock()
        self.exceeded = threading.Event()

    def put(self, item, block=True, timeout=None):
        if item is None:
            return super().put(item, block=block, timeout=timeout)
        if self.exceeded.is_set():
            # The explicit terminal limit error replaces all subsequent output.
            return None
        size = len(
            json.dumps(item, ensure_ascii=False, allow_nan=False).encode("utf-8")
        )
        with self.accounting:
            if (
                self.exceeded.is_set()
                or size > self.event_bytes
                or self.total + size > self.max_bytes
            ):
                self.exceeded.set()
                self.cancel()
                raise OutputLimitError("Service output limit exceeded")
            self.total += size
        return super().put(item, block=block, timeout=timeout)
