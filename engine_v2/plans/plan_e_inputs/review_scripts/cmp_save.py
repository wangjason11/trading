"""Cell-level diff of a replay against a saved set (the 24 CSVs + the 3 charts'
figure JSON). Plan E E2 (2026-09-24); used for every E2 `/compare` and E4 variant.

    python engine_v2/plans/plan_e_inputs/review_scripts/cmp_save.py BASE_DIR CUR_DEBUG CUR_CHARTS \
        [--strip key1,key2] [--max N] [--names-from SAVE_DIR]

- BASE_DIR: a folder holding the baseline files (a save folder, or a copy of a
  previous replay's outputs). CUR_DEBUG / CUR_CHARTS: usually artifacts/debug and
  artifacts/charts.
- --names-from: which file names to compare (default: the save named by
  artifacts/commits/LATEST_<branch>) — artifacts/debug also holds stale files.
- --strip: meta keys ignored when a `meta` cell differs (e.g. an additive key, or
  a declared raw reader such as `source_event_idx`); such cells count as
  "strip-equal", everything else as REAL.
- Figures: per trace (name, x, y) and layout["shapes"] via
  `debug/chart_census.load_fig` — count parity cannot see a moved marker.

Windows: keep BASE_DIR short (e.g. %TEMP%/pe/...); the saved file names exceed
MAX_PATH under the long session scratchpad path.
"""
import ast
import csv
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.getcwd())
_argv = sys.argv
sys.argv = [sys.argv[0]]  # chart_census runs a census on import when given paths
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

args = sys.argv[1:]
base, cur_dbg, cur_ch = args[:3]
strip = set(args[args.index("--strip") + 1].split(",")) if "--strip" in args else set()
mx = int(args[args.index("--max") + 1]) if "--max" in args else 40
if "--names-from" in args:
    save = args[args.index("--names-from") + 1]
else:
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    latest = open(f"artifacts/commits/LATEST_{branch}").read().strip()
    save = f"artifacts/commits/{branch}/{latest}"
NAMES = {os.path.basename(x) for x in glob.glob(os.path.join(save, "*.csv")) + glob.glob(os.path.join(save, "*.html"))}


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def norm_meta(v):
    if not strip or not v.startswith("{"):
        return v
    try:
        d = ast.literal_eval(v)
    except Exception:
        return v
    return repr({k: x for k, x in d.items() if k not in strip})


total = 0
for name in sorted(n for n in NAMES if n.endswith(".csv")):
    bp, cp = os.path.join(base, name), os.path.join(cur_dbg, name)
    if not os.path.exists(cp):
        print("MISSING", name)
        continue
    if open(bp, "rb").read() == open(cp, "rb").read():
        continue
    a, b = rows(bp), rows(cp)
    diffs = []
    for i in range(max(len(a), len(b))):
        ra = a[i] if i < len(a) else []
        rb = b[i] if i < len(b) else []
        for j in range(max(len(ra), len(rb))):
            x = ra[j] if j < len(ra) else None
            y = rb[j] if j < len(rb) else None
            if x != y:
                col = a[0][j] if j < len(a[0]) else j
                if x is not None and y is not None and norm_meta(x) == norm_meta(y):
                    diffs.append(("strip-equal", i + 1, col))
                else:
                    diffs.append(("REAL", i + 1, col, x, y))
    real = [d for d in diffs if d[0] == "REAL"]
    total += len(real)
    print(f"DIFF {name}: {len(diffs)} cells ({len(real)} real after strip)")
    for d in real[:mx]:
        print("   ", d)
print("real cells total:", total)

for name in sorted(n for n in NAMES if n.endswith(".html")):
    da, la = load_fig(os.path.join(base, name))
    db, lb = load_fig(os.path.join(cur_ch, name))
    key = lambda d: [(t.get("name"), t.get("x"), t.get("y")) for t in d]  # noqa: E731
    sa, sb = la.get("shapes", []), lb.get("shapes", [])
    same_tr = key(da) == key(db)
    same_sh = json.dumps(sa, sort_keys=True) == json.dumps(sb, sort_keys=True)
    print(f"FIG {name[-30:]}: traces {len(da)}/{len(db)} shapes {len(sa)}/{len(sb)} "
          f"traces_xy_equal={same_tr} shapes_equal={same_sh}")
    if not same_tr:
        for n, (ta, tb) in enumerate((p for p in zip(da, db)
                                      if (p[0].get("name"), p[0].get("x"), p[0].get("y"))
                                      != (p[1].get("name"), p[1].get("x"), p[1].get("y")))):
            print("   trace", ta.get("name"), "|", str(ta.get("x"))[:120], "->", str(tb.get("x"))[:120])
            if n >= 10:
                break
