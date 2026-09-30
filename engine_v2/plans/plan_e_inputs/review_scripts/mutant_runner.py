"""Parallel mutation loop: each mutant in its own scratch copy of `engine_v2/`, the full suite per mutant (2026-09-29d).

usage (repo root):
  python engine_v2/plans/plan_e_inputs/review_scripts/mutant_runner.py MUTANTS.py [--jobs N] [--only ID,ID] [--keep]
                                                                        [--pytest-args "..."] [--out DIR]

MUTANTS.py defines `MUT = [(id, relpath, old, new), ...]`: `relpath` from the repo root (e.g.
"engine_v2/structure/market_structure.py"); `old` must occur EXACTLY ONCE in the working-tree file (else the mutant is
reported BAD — the site moved); it is replaced by `new`. Each mutant gets a fresh copy of the working tree's
`engine_v2/` (without `__pycache__` / `plans`, so a test must not import from `engine_v2.plans`) under a SHORT directory
name (Windows path length: a long mutant id as the folder name broke `copytree` on the first run), runs
`python -m pytest engine_v2/tests -q -p no:cacheprovider` with the OANDA smoke test deselected (it needs `oanda.cfg`)
and PYTHONPATH = that copy, in parallel (default jobs = min(#mutants, cpu - 2)).

Prints one row per mutant — KILLED (n failing tests + the first few) / SURVIVED / BAD — and writes `results.json` into
the output dir (default: the temp dir `mutant_runner/`). Exit 1 if any mutant SURVIVED or is BAD. The trees are deleted
unless `--keep`. The working tree is never modified. Mutants in the same file are independent (each is applied alone).

Example mutants file: `mutants_post_e4.py` (the Post-E·4 rename loop: 12 of mine + the landing review's 3 in-code ones;
all KILLED at `ed99914`). A full-suite mutant costs ~1-2 min wall; 15 in parallel ~3 min on the 32-core box.
"""
import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.getcwd()
SMOKE = "engine_v2/tests/test_smoke.py::test_oanda_history_smoke"


def _load(path):
    spec = importlib.util.spec_from_file_location("mutants", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.MUT)


def _run_one(k, mut, out, pytest_args, keep):
    mid, rel, old, new = mut
    dst = os.path.join(out, f"m{k:02d}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    try:
        shutil.copytree(os.path.join(REPO, "engine_v2"), os.path.join(dst, "engine_v2"),
                        ignore=shutil.ignore_patterns("__pycache__", "plans"))
        p = os.path.join(dst, rel)
        s = open(p, encoding="utf-8").read()
        if s.count(old) != 1:
            return dict(id=mid, verdict="BAD", detail=f"`old` occurs {s.count(old)}x in {rel}")
        open(p, "w", encoding="utf-8", newline="").write(s.replace(old, new))
        cmd = [sys.executable, "-m", "pytest", "engine_v2/tests", "-q", "-p", "no:cacheprovider",
               "--deselect", SMOKE, *pytest_args]
        r = subprocess.run(cmd, cwd=dst, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": dst})
        lines = r.stdout.splitlines()
        fails = sorted({l.split(" - ")[0].split(" ", 1)[1] for l in lines if l.startswith(("FAILED ", "ERROR "))})
        tail = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l), "")
        if fails:
            return dict(id=mid, verdict="KILLED", n=len(fails), fails=fails, tail=tail)
        if r.returncode not in (0,):
            return dict(id=mid, verdict="KILLED", n=0, fails=[], tail=tail or f"pytest exit {r.returncode}")
        return dict(id=mid, verdict="SURVIVED", tail=tail)
    finally:
        if not keep:
            shutil.rmtree(dst, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mutants")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--pytest-args", default="")
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "mutant_runner"))
    a = ap.parse_args()
    assert os.path.isdir(os.path.join(REPO, "engine_v2", "tests")), "run from the repo root"
    muts = _load(a.mutants)
    if a.only:
        want = set(a.only.split(","))
        muts = [m for m in muts if m[0] in want]
    os.makedirs(a.out, exist_ok=True)
    jobs = a.jobs or max(1, min(len(muts), (os.cpu_count() or 4) - 2))
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        res = list(ex.map(lambda km: _run_one(km[0], km[1], a.out, a.pytest_args.split(), a.keep), enumerate(muts)))
    for r in res:
        extra = (f"{r['n']} failing: " + ", ".join(f.split("::")[-1][:60] for f in r["fails"][:4])) if r["verdict"] == "KILLED" \
            else r.get("detail", r.get("tail", ""))
        print(f"{r['verdict']:9s} {r['id']:34s} {extra}")
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    n_bad = sum(r["verdict"] != "KILLED" for r in res)
    print(f"{len(res) - n_bad}/{len(res)} KILLED  (results: {os.path.join(a.out, 'results.json')})")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
