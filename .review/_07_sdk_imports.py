import re, glob, importlib, sys, os
files = sorted(glob.glob("docs/**/*.mdx", recursive=True)) + ["README.md","CLAUDE.md","AGENTS.md","CONTRIBUTING.md"] + glob.glob("hub/**/*.md", recursive=True) + glob.glob(".claude/**/*.md", recursive=True)
imp = re.compile(r'^\s*from (gaia(?:\.[\w.]+)?|gaia_agent_\w+(?:\.[\w.]+)?) import ([\w, ()]+)', re.M)
imp2 = re.compile(r'^\s*import (gaia(?:\.[\w.]+)?)', re.M)
seen = {}
for f in files:
    if "node_modules" in f: continue
    txt = open(f, encoding="utf-8", errors="replace").read()
    for m in imp.finditer(txt):
        mod, names = m.group(1), m.group(2)
        for n in re.split(r'[,\s()]+', names):
            if n and n != "as": seen.setdefault((mod, n), set()).add(f)
    for m in imp2.finditer(txt):
        seen.setdefault((m.group(1), None), set()).add(f)
bad = []
for (mod, name), fs in sorted(seen.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        bad.append((mod, name, f"MODULE ERROR: {type(e).__name__}: {str(e)[:80]}", sorted(fs)))
        continue
    if name and not hasattr(m, name):
        try:
            importlib.import_module(mod + "." + name)
        except Exception:
            bad.append((mod, name, "NAME MISSING", sorted(fs)))
print("checked", len(seen), "imports")
for b in bad:
    print(f"{b[0]} :: {b[1]} -> {b[2]}")
    for f in b[3][:6]: print("     ", f)
