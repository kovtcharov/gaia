# Copyright(C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Frozen-binary entrypoint: dispatch to the stdio transport or the REST sidecar.

The frozen executable is named ``gaia-agent``, and that name already means
``gaia_agent.stdio:main`` everywhere else — it is the wheel's console script
(``[project.scripts]`` in ``pyproject.toml``) and it is what the TUI resolves on
PATH for its subprocess transport (``BinaryPath: "gaia-agent"`` in
``tui/internal/catalog/catalog.go``). Freezing ``packaging/server.py`` under that
name gave one name two protocols: the TUI spawned it expecting newline-delimited
JSON and got uvicorn's startup banner, so every turn died with "unreadable event
skipped" and a port-bind attempt.

So the frozen binary dispatches instead of picking a side:

    gaia-agent                       -> stdio JSONL   (the TUI's subprocess transport)
    gaia-agent --serve               -> REST sidecar  (explicit)
    gaia-agent --host H --port P     -> REST sidecar  (what the daemon and the
                                        npm client already pass)

``--host``/``--port`` select HTTP on their own so the existing spawners keep
working byte-for-byte: ``gaia.daemon.sidecars`` and the npm client's
``spawnSidecar`` both build ``["--host", host, "--port", port]`` with no
``--serve``. Anything else is stdio's argv and is forwarded to it untouched —
``--model``, ``--use-claude``, ``--claude-model``, ``--json-events``, ``--dev``,
``--bypass-permissions`` are all load-bearing spellings the Go side emits
literally.

Imports are deferred into each branch on purpose: importing the REST app costs
FastAPI + uvicorn and emits the sidecar's auth banner on stderr, which a stdio
launch should never pay for or print.
"""

from __future__ import annotations

import sys

#: Flags that mean "serve HTTP". ``--serve`` is explicit; the other two are the
#: argv the daemon and the npm client have always sent.
_HTTP_FLAGS = ("--serve", "--host", "--port")

_USAGE = """\
gaia-agent - the GAIA flagship agent, in one of two transports.

  gaia-agent [stdio options]        Newline-delimited JSON on stdin/stdout.
                                    This is the default and what the GAIA
                                    terminal UI spawns.
  gaia-agent --serve [--host H] [--port P]
                                    Serve the /v1/gaia/* REST surface.

Run `gaia-agent --help-stdio` or `gaia-agent --serve --help` for each
transport's own options.
"""


def _wants_http(argv: list) -> bool:
    """True when argv selects the REST sidecar.

    Matches ``--host=127.0.0.1`` as well as ``--host 127.0.0.1``; a bare
    ``--host`` with its value in the next slot is the form both existing
    spawners use.
    """
    for arg in argv:
        if arg in _HTTP_FLAGS or arg.startswith(tuple(f + "=" for f in _HTTP_FLAGS)):
            return True
    return False


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("-h", "--help"):
        sys.stdout.write(_USAGE)
        return 0

    if _wants_http(argv):
        from gaia_agent.server import main as serve_main

        # --serve is this dispatcher's flag, not the server's parser's.
        return serve_main([a for a in argv if a != "--serve"])

    if argv and argv[0] == "--help-stdio":
        argv = ["--help"]

    from gaia_agent.stdio import main as stdio_main

    return stdio_main(argv)


if __name__ == "__main__":
    sys.exit(main())
