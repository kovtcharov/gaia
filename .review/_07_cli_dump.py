import argparse, sys, json
sys.argv = ["gaia"]
from gaia.cli import build_parser
p = build_parser()
out = {}
def walk(parser, path):
    flags = []
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for name, sub in a.choices.items():
                walk(sub, path + [name])
            continue
        if isinstance(a, argparse._HelpAction): continue
        opts = a.option_strings or [a.dest]
        d = a.default
        if d is argparse.SUPPRESS: d = "SUPPRESS"
        flags.append({"opts": opts, "default": repr(d), "help": (a.help or "")[:140], "choices": list(a.choices) if a.choices else None})
    out[" ".join(path)] = flags
walk(p, ["gaia"])
json.dump(out, open(".review/_07_cli_tree.json","w"), indent=1)
for k in out: print(k, "|", " ".join(sorted(o for f in out[k] for o in f["opts"] if o.startswith("--"))))
