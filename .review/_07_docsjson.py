import json, os, re, sys
root = "docs"
nav = json.load(open("docs/docs.json", encoding="utf-8"))
pages = []
def walk(o):
    if isinstance(o, str):
        pages.append(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            if k in ("pages", "groups", "tabs", "navigation", "dropdowns", "anchors", "versions", "languages"):
                walk(v)
            elif k == "page":
                pages.append(v)
    elif isinstance(o, list):
        for x in o: walk(x)
walk(nav.get("navigation"))
print("pages in docs.json:", len(pages), "unique:", len(set(pages)))
missing = [p for p in pages if not (os.path.exists(os.path.join(root, p + ".mdx")) or os.path.exists(os.path.join(root, p + ".md")) or p.startswith("http"))]
print("MISSING files for pages:", missing)
dups = sorted({p for p in pages if pages.count(p) > 1})
print("DUPLICATE pages:", dups)
ondisk = []
for dp, dn, fn in os.walk(root):
    if "node_modules" in dp: continue
    for f in fn:
        if f.endswith(".mdx"):
            rel = os.path.relpath(os.path.join(dp, f), root).replace("\\", "/")[:-4]
            ondisk.append(rel)
orphans = sorted(set(ondisk) - set(pages))
print("ORPHAN .mdx (on disk, not in docs.json):", len(orphans))
for o in orphans: print("  ", o)
# broken relative links + images + anchors
linkre = re.compile(r'\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
hrefre = re.compile(r'href="([^"]+)"')
srcre = re.compile(r'src="([^"]+)"')
broken = []
for p in ondisk:
    path = os.path.join(root, p + ".mdx")
    txt = open(path, encoding="utf-8", errors="replace").read()
    for m in list(linkre.finditer(txt)) + list(hrefre.finditer(txt)) + list(srcre.finditer(txt)):
        l = m.group(1)
        if l.startswith(("http", "mailto:", "#", "{", "$", "tel:")): continue
        target, _, anchor = l.partition("#")
        if not target: continue
        if target.startswith("/"):
            cand = [os.path.join(root, target.lstrip("/") + ".mdx"), os.path.join(root, target.lstrip("/") + ".md"), os.path.join(root, target.lstrip("/")), os.path.join(root, target.lstrip("/"), "index.mdx")]
        else:
            base = os.path.dirname(path)
            cand = [os.path.join(base, target + ".mdx"), os.path.join(base, target + ".md"), os.path.join(base, target), os.path.join(root, target)]
        if not any(os.path.exists(c) for c in cand):
            broken.append((p, l))
print("BROKEN relative links/images:", len(broken))
for b in broken: print("  ", b)
