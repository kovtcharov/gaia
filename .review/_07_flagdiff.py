import json, re
tree = json.load(open(".review/_07_cli_tree.json"))
doc = open("docs/reference/cli.mdx", encoding="utf-8").read()
# For each argparse command, collect documented flags: flags appearing on lines that mention that command in code/table, or within a window after "gaia <cmd>" appearance.
docflags_by_cmd = {}
# Approach: scan doc for occurrences of 'gaia <cmd>' and collect --flags within following 40 lines (until next '## ' heading)
lines = doc.splitlines()
for cmd in tree:
    parts = cmd.split()[1:]
    if not parts: continue
    key = "gaia " + " ".join(parts)
    found = set()
    for i, l in enumerate(lines):
        if re.search(r'(^|[`$ ])' + re.escape(key) + r'(\b|$)', l):
            j = i
            while j < len(lines) and j < i + 60 and not (j > i and lines[j].startswith("## ")):
                for m in re.finditer(r'(--[a-z][a-z0-9-]*)', lines[j]):
                    found.add(m.group(1))
                j += 1
    docflags_by_cmd[cmd] = found
common = {"--base-url","--claude-model","--list-tools","--logging-level","--max-steps","--model","--no-lemonade-check","--show-stats","--stats","--stream","--trace","--use-chatgpt","--use-claude"}
alldoc = set(re.findall(r'(--[a-z][a-z0-9-]*)', doc))
print("### implemented flags never mentioned anywhere in cli.mdx (excluding common inherited set):")
for cmd, flags in tree.items():
    impl = {o for f in flags for o in f["opts"] if o.startswith("--")} - common
    missing = sorted(impl - alldoc)
    if missing: print(f"  {cmd}: {missing}")
print("### flags documented near a command but not implemented on it (candidate stale flags):")
allimpl = {o for flags in tree.values() for f in flags for o in f["opts"] if o.startswith("--")}
for cmd, dflags in docflags_by_cmd.items():
    impl = {o for f in tree[cmd] for o in f["opts"] if o.startswith("--")}
    # only flag those documented flags that exist nowhere in the argparse tree at all
    stale = sorted(d for d in dflags if d not in allimpl)
    if stale: print(f"  {cmd}: {stale}")
