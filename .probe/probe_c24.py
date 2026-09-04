"""C24 probe: the port matcher against captured real `netstat -ano` output.

Prints, for a few ports, which rows the CURRENT (substring) matcher selects and
which the FIXED (column-parsing) matcher selects.
"""
import sys
from pathlib import Path

text = Path(sys.argv[1] if len(sys.argv) > 1 else ".probe/netstat_live.txt").read_text(
    encoding="utf-8", errors="replace"
)


def old_match(output, port):
    """cli.py:4446 as shipped — bare substring over the whole netstat line."""
    hits = []
    for line in output.strip().split("\n"):
        if f":{port}" in line:
            parts = line.strip().split()
            try:
                pid = int(parts[-1])
            except (IndexError, ValueError):
                continue
            if pid > 0:
                hits.append((pid, line.strip()))
    return hits


def new_match(output, port):
    try:
        from gaia.cli import _parse_windows_netstat_listeners
    except ImportError:
        return None
    return _parse_windows_netstat_listeners(output, port)


for port in (80, 443, 135):
    old = old_match(text, port)
    print(f"\n=== port {port} ===")
    print(f"  OLD matcher: {len(old)} rows; would taskkill /F PID {old[0][0] if old else None}")
    for pid, line in old[:3]:
        print(f"      {line}")
    new = new_match(text, port)
    if new is None:
        print("  NEW matcher: not present at this commit")
    else:
        print(f"  NEW matcher (LISTENING + exact local port): {new}")
