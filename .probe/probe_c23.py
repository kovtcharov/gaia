"""C23 probe: GAIA_HOME pointed at a home dir turns --purge into ~/Documents deletion."""
import os, sys, tempfile, argparse
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="c23probe_"))
fakehome = tmp / "fakehome"
(fakehome / "Documents").mkdir(parents=True)
(fakehome / "Documents" / "thesis.docx").write_text("my life's work")
(fakehome / "venv" / "bin").mkdir(parents=True)
(fakehome / "venv" / "bin" / "python").write_text("#!/bin/sh")

os.environ["GAIA_HOME"] = str(fakehome)
from gaia.installer import uninstall_command as uc

print(f"GAIA_HOME     = {fakehome}")
try:
    print(f"_gaia_home()  = {uc._gaia_home()}")
    print(f"_safe_roots() = {[str(r) for r in uc._safe_roots()]}")
    plan = uc.build_plan(venv=False, purge=True, purge_lemonade=False, purge_models=False)
    roots = [r.resolve(strict=False) for r in uc._safe_roots()]
    for _, p in plan.tiered_paths:
        r = p.resolve(strict=False)
        inside = any(_inside(r, root) for root in roots) if False else any(
            (lambda: (r.is_relative_to(root)))() for root in roots)
        print(f"  {p}  exists={p.exists()}  resolves_to={r}  inside_safe_roots={inside}")
    print("\n--- gaia uninstall --purge --dry-run (via run()) ---")
    ns = argparse.Namespace(venv=False, purge=True, purge_lemonade=False,
                            purge_models=False, purge_hf_cache=False,
                            dry_run=True, yes=True)
    rc = uc.run(ns)
    print(f"exit code = {rc}")
except Exception as exc:
    print(f"\nREFUSED: {type(exc).__name__}: {exc}")
    rc = None
print(f"\nthesis.docx still present: {(fakehome/'Documents'/'thesis.docx').exists()}")
