"""Byte-level save diff: every CSV by md5 + every chart's FULL figure JSON.

    PYTHONPATH=. python engine_v2/plans/plan_e_inputs/review_scripts/cmp_fig_full.py \
        BASE_DIR [CUR_DEBUG] [CUR_CHARTS] [--max N]

`cmp_save.py` compares cells + per-trace (name, x, y) + shapes, and `cmp_hover.py`
hover text; this one compares EVERYTHING in each figure — every trace attribute
(marker, line, opacity, hovertemplate, customdata, visible, legendgroup, ...) and the
whole layout (shapes, annotations, axes, updatemenus). The proof behind a
"byte-identical" claim (2026-09-30: §13.5.e, 24/24 md5 + 0 differing traces +
layout equal). Exit 1 on any CSV / figure difference or a file on one side only.
"""
import glob
import hashlib
import json
import os
import sys

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]  # chart_census runs a census on import when given args
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
max_n = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 5
if "--max" in sys.argv:
    args.remove(str(max_n))
BASE = args[0]
CUR_DEBUG = args[1] if len(args) > 1 else "artifacts/debug"
CUR_CHARTS = args[2] if len(args) > 2 else "artifacts/charts"

bad = False


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def dj(o):
    return json.dumps(o, sort_keys=True, default=str)


base_csv = {os.path.basename(p) for p in glob.glob(f"{BASE}/*.csv")}
cur_csv = {os.path.basename(p) for p in glob.glob(f"{CUR_DEBUG}/*.csv")}
# Only the save's replay prefix matters on the current side (artifacts/debug holds other windows too).
prefixes = {f.split("_sd")[0].replace("_raw.csv", "").replace("_final.csv", "") for f in base_csv}
cur_csv = {f for f in cur_csv if any(f.startswith(p) for p in prefixes)}
common = sorted(base_csv & cur_csv)
diff = [f for f in common if md5(f"{BASE}/{f}") != md5(f"{CUR_DEBUG}/{f}")]
only_b, only_c = sorted(base_csv - cur_csv), sorted(cur_csv - base_csv)
print(f"CSVs: baseline {len(base_csv)}, current {len(cur_csv)}, common {len(common)}, md5-different {len(diff)}")
for f in only_b:
    print("  MISSING (in baseline, not produced):", f)
for f in only_c:
    print("  NEW (no baseline):", f)
for f in diff:
    print("  DIFF", f)
bad |= bool(diff or only_b or only_c)

for bh in sorted(glob.glob(f"{BASE}/*.html")):
    ch = os.path.join(CUR_CHARTS, os.path.basename(bh))
    if not os.path.exists(ch):
        print("MISSING current chart", ch)
        bad = True
        continue
    bd, bl = load_fig(bh)
    cd, cl = load_fig(ch)
    differing = [(k, a, b) for k, (a, b) in enumerate(zip(bd, cd)) if dj(a) != dj(b)]
    layout_eq = dj(bl) == dj(cl)
    print(f"FIG {os.path.basename(bh)}: traces {len(bd)}/{len(cd)} "
          f"differing-traces(all attrs)={len(differing)} layout_equal={layout_eq}")
    for k, a, b in differing[:max_n]:
        keys = sorted(x for x in set(a) | set(b) if dj(a.get(x)) != dj(b.get(x)))
        print(f"   trace {k} name={a.get('name')!r}: attrs differ {keys}")
    if not layout_eq:
        keys = sorted(x for x in set(bl) | set(cl) if dj(bl.get(x)) != dj(cl.get(x)))
        print(f"   layout keys differ: {keys}")
    bad |= bool(differing) or not layout_eq or len(bd) != len(cd)

sys.exit(1 if bad else 0)
