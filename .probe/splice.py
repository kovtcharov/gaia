from pathlib import Path

p = Path("src/gaia/cli.py")
lines = p.read_text(encoding="utf-8").splitlines(keepends=True)
start = next(
    i for i, l in enumerate(lines) if l.startswith("def kill_process_by_port(port):")
)
end = next(
    i for i in range(start + 1, len(lines)) if lines[i].startswith("def handle_email_command")
)
print("replacing lines", start + 1, "..", end, "->", repr(lines[end][:40]))
new = Path(".probe/new_kill.py").read_text(encoding="utf-8").rstrip("\n") + "\n\n\n"
p.write_text("".join(lines[:start]) + new + "".join(lines[end:]), encoding="utf-8")
print("done")
