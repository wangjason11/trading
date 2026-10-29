"""PLAN_X X3 — the 5 windows' variant diffs (`x3_variant_diff_<window>.txt`) in one table:

  CSVs md5-different; MS-decision rows (structure events + triggers / subs / unresolved: rows added + removed — any =
  an MS decision moved); fib records differing (keyed) with each difference; the DISTINCT downstream re-asks that
  CHANGED a fib (the proximity views re-feed the same events — deduped by text); walk-back windows reaching the
  moment (the process total, max over the summaries) and walk-back questions examined.

    python engine_v2/plans/plan_x_inputs/x3_summary.py > engine_v2/plans/plan_x_inputs/x3_summary.txt
"""
import glob
import os
import re

here = os.path.dirname(os.path.abspath(__file__))
for p in sorted(glob.glob(os.path.join(here, "x3_variant_diff_*.txt"))):
    w = os.path.basename(p)[len("x3_variant_diff_"):-4]
    t = open(p, encoding="utf-8").read()
    s1 = t.split("== 2.")[0]
    s2 = t.split("== 2.")[1].split("== 3.")[0]
    s3 = t.split("== 3.")[1].split("== 4.")[0]
    s4 = t.split("== 4.")[1]
    md5 = re.search(r"md5-different: (\d+) of (\d+)", s1)
    figs = [l for l in s1.splitlines() if l.startswith("FIG ") and "count deltas {}" not in l]
    ms_rows = sum(int(a) + int(b) for a, b in re.findall(r"removed (\d+), added (\d+)", s2))
    fib = [l.strip() for l in s3.splitlines() if l.startswith("  (") or l.startswith("  only in")]
    changed = sorted({l.split("] ", 1)[-1].strip() for l in s4.splitlines() if "CHANGED" in l and "[fib][reask] sid" in l})
    warn = sum("WARNING walk-back" in l for l in s4.splitlines())
    walk = [(int(m.group(1)), int(m.group(2) or 0)) for m in
            re.finditer(r"walk-back \(process total\) \{'examined': (\d+)(?:, 'window_reaches_moment': (\d+))?\}", s4)]
    ex, reach = (max(x for x, _ in walk), max(y for _, y in walk)) if walk else (0, 0)
    print(f"== {w}")
    print(f"  CSVs md5-different {md5.group(1)} of {md5.group(2)}; charts with a count delta {len(figs)}")
    for l in figs:
        print(f"    {l[:240]}")
    print(f"  MS-decision rows (structure events / triggers / subs / unresolved) added + removed: {ms_rows}")
    print(f"  fib records differing: {len(fib)}")
    for l in fib:
        print(f"    {l[:330]}")
    print(f"  distinct downstream re-asks that CHANGED a fib: {len(changed)}")
    for l in changed:
        print(f"    {l}")
    print(f"  walk-back questions examined (process total) {ex}; windows reaching the moment {reach}; WARNING lines {warn}")
