import re, os, glob
files = glob.glob(".claude/agents/*.md") + glob.glob(".claude/skills/*/SKILL.md") + ["CLAUDE.md","AGENTS.md","CONTRIBUTING.md","hub/agents/README.md","hub/skills/README.md","REVIEW.md","SECURITY.md"]
pat = re.compile(r'(?<![\w/.-])((?:src|tests|docs|hub|util|scripts|tui|\.claude|\.github|cpp|installer)/[\w./-]+)')
for f in files:
    txt = open(f, encoding="utf-8", errors="replace").read()
    bad = set()
    for m in pat.finditer(txt):
        p = m.group(1).rstrip(".,:;)`'\"")
        p = re.sub(r'[*<>{}]+.*$', '', p)
        if p.endswith("/"): p = p[:-1]
        if "<" in p or "*" in p or not p: continue
        if not os.path.exists(p):
            bad.add(p)
    if bad:
        print(f"{f}:")
        for b in sorted(bad): print("   MISSING:", b)
