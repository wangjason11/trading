"""Keyed figure check for the M15 fib chart aid (PLAN_H §8.9 F1) vs a baseline save.

    PYTHONPATH=. python engine_v2/plans/plan_h_inputs/fib_aid_check.py BASE_DIR [CUR_CHARTS]   (repo root)

M15 charts: current minus every trace / shape in legend group `m15_fib` == baseline, exactly (all attrs).
H1 chart: baseline minus the hidden fib records' hover traces and their 5 shapes == current, except the
remaining fib hovers' customdata (listed before -> after).
Then lists every M15 fib hover (sub, cycle, BOS -> CTS, status cell) and the legend entry.
"""
import collections
import glob
import json
import os
import sys

sys.argv, _args = [sys.argv[0]], sys.argv[1:]
from engine_v2.debug.chart_census import load_fig  # noqa: E402

BASE = _args[0]
CUR = _args[1] if len(_args) > 1 else "artifacts/charts"
bad = False


def dj(o):
    return json.dumps(o, sort_keys=True, default=str)


for bh in sorted(glob.glob(f"{BASE}/*.html")):
    name = os.path.basename(bh)
    bd, bl = load_fig(bh)
    cd, cl = load_fig(os.path.join(CUR, name))
    bsh, csh = bl.get("shapes", []), cl.get("shapes", [])
    tag = name.split("rk2-5")[-1] or "H1"
    if "_M15_" in name:
        new_tr = [t for t in cd if t.get("legendgroup") == "m15_fib"]
        new_sh = [s for s in csh if s.get("legendgroup") == "m15_fib"]
        rest_tr = [t for t in cd if t.get("legendgroup") != "m15_fib"]
        rest_sh = [s for s in csh if s.get("legendgroup") != "m15_fib"]
        tr_eq = [dj(t) for t in rest_tr] == [dj(t) for t in bd]
        sh_eq = [dj(s) for s in rest_sh] == [dj(s) for s in bsh]
        lay_eq = dj({k: v for k, v in cl.items() if k != "shapes"}) == dj({k: v for k, v in bl.items() if k != "shapes"})
        legend = [(s.get("name"), s.get("showlegend")) for s in new_sh if s.get("showlegend")]
        print(f"{tag}: +{len(new_tr)} fib traces, +{len(new_sh)} fib shapes; rest traces == base {tr_eq}, "
              f"rest shapes == base {sh_eq}, layout(no shapes) == base {lay_eq}; legend entries {legend}")
        bad |= not (tr_eq and sh_eq and lay_eq and len(new_sh) == 5 * len(new_tr) and len(legend) == 1)
        for t in new_tr:
            c = t["customdata"][0]
            print(f"   sub {c[0]} cyc {c[5]}  {c[6]} -> {c[8]}  | {c[12].replace('<br>', ' | ')}")
    else:
        bfib = {t["name"]: t for t in bd if (t.get("name") or "").startswith("fib:")}
        cfib = {t["name"]: t for t in cd if (t.get("name") or "").startswith("fib:")}
        gone = sorted(set(bfib) - set(cfib))
        rest_b = [t for t in bd if t.get("name") not in gone]
        diffs = []
        for tb, tc in zip(rest_b, cd):
            if dj(tb) != dj(tc):
                keys = sorted(k for k in set(tb) | set(tc) if dj(tb.get(k)) != dj(tc.get(k)))
                diffs.append((tb.get("name"), keys))
        bc, cc = collections.Counter(map(dj, bsh)), collections.Counter(map(dj, csh))
        removed, added = bc - cc, cc - bc
        lay_eq = dj({k: v for k, v in cl.items() if k != "shapes"}) == dj({k: v for k, v in bl.items() if k != "shapes"})
        print(f"{tag}: hidden fib traces {gone}; traces {len(bd)} -> {len(cd)}; differing kept traces {diffs}; "
              f"shapes removed {sum(removed.values())} added {sum(added.values())}; layout(no shapes) == base {lay_eq}")
        for s in removed:
            s = json.loads(s)
            print(f"   removed shape {s['type']} x {s['x0']} -> {s['x1']} y {s['y0']:.5f} -> {s['y1']:.5f}")
        for n in sorted(cfib):
            print(f"   {n}: {bfib[n]['customdata'][0][8]!r}\n      -> {cfib[n]['customdata'][0][8]!r}")
        bad |= not (len(rest_b) == len(cd) and all(k == ["customdata"] for _, k in diffs) and not added and lay_eq)
print("RESULT", "FAIL" if bad else "OK")
sys.exit(1 if bad else 0)
