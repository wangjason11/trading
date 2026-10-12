"""PLAN_M — a KEYED diff of a replay against a save (the trial census, PLAN_M §5; read-only).

    PYTHONPATH=. python engine_v2/plans/plan_m_inputs/trial_diff.py BASE_SAVE_DIR CUR_DEBUG [CUR_CHARTS]

1. md5 of every CSV of the save vs CUR (file names from the save — CUR may hold other windows' outputs);
2. fib lifecycle per lens: key (sub_id | None, structure_id, cycle_id, version) → removed / added / the changed
   columns + the changed meta keys (old → new);
3. POIs per lens: key (sub_id | None, structure_id, cycle_id, ic_idx) → removed / added / changed;
4. IC candidates per lens: per fib record (sub_id, structure_id, cycle_id, bos, cts) → the rows that differ;
5. figures (CUR_CHARTS): trace / shape counts and the changed traces by name (full-JSON multisets).
"""
import ast
import glob
import hashlib
import json
import os
import sys
from collections import Counter

import pandas as pd

_argv = sys.argv[:]
sys.argv = [sys.argv[0]]  # chart_census runs a census on import when given args
from engine_v2.debug.chart_census import load_fig  # noqa: E402
sys.argv = _argv

BASE, CUR = sys.argv[1], sys.argv[2]
CHARTS = sys.argv[3] if len(sys.argv) > 3 else None
LENS = {"H1": "*rk2-5_{}.csv", "confluence": "*M15_confluence_{}.csv", "counter": "*M15_counter_{}.csv"}


def one(d, pat):
    got = glob.glob(os.path.join(d, pat))
    assert len(got) == 1, (d, pat, got)
    return got[0]


def cur_path(d, pat):
    return os.path.join(d, os.path.basename(one(BASE, pat)))


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def read(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False)


def meta(s):
    return ast.literal_eval(s) if s else {}


print("== 1. CSVs")
for p in sorted(glob.glob(os.path.join(BASE, "*.csv"))):
    f = os.path.basename(p)
    c = os.path.join(CUR, f)
    if not os.path.exists(c):
        print("  MISSING", f)
    elif md5(c) != md5(p):
        print("  DIFF   ", f)


def keyed(df, keyf):
    out = {}
    for _, r in df.iterrows():
        m = meta(r["meta"])
        k = keyf(r, m)
        assert k not in out, ("duplicate key", k)
        out[k] = (r, m)
    return out


def diff_rows(label, a, b, cols):
    for k in sorted(set(a) - set(b), key=str):
        print(f"  - {label} {k}")
    for k in sorted(set(b) - set(a), key=str):
        print(f"  + {label} {k}")
    for k in sorted(set(a) & set(b), key=str):
        (ra, ma), (rb, mb) = a[k], b[k]
        dc = [f"{c}: {ra[c]} -> {rb[c]}" for c in cols if c != "meta" and ra[c] != rb[c]]
        dm = [f"{kk}: {ma.get(kk, '<absent>')!r} -> {mb.get(kk, '<absent>')!r}"
              for kk in list(dict.fromkeys(list(ma) + list(mb))) if ma.get(kk, "<absent>") != mb.get(kk, "<absent>")]
        order = [kk for kk in ma if kk in mb] != [kk for kk in mb if kk in ma]
        if dc or dm or order:
            print(f"  ~ {label} {k}")
            for x in dc + dm:
                print(f"      {x}")
            if order:
                print("      (meta key ORDER differs)")


for lens, pat in LENS.items():
    print(f"== 2. fib lifecycle {lens}")
    a, b = read(one(BASE, pat.format("fib_lifecycle"))), read(cur_path(CUR, pat.format("fib_lifecycle")))
    kf = lambda r, m: (m.get("sub_id"), r["structure_id"], r["cycle_id"], r["version"])
    diff_rows("fib", keyed(a, kf), keyed(b, kf), list(a.columns))

for lens, pat in LENS.items():
    print(f"== 3. POIs {lens}")
    a, b = read(one(BASE, pat.format("poi_zones"))), read(cur_path(CUR, pat.format("poi_zones")))
    kf = lambda r, m: (m.get("sub_id"), r["structure_id"], r["cycle_id"], r["ic_idx"])
    diff_rows("POI", keyed(a, kf), keyed(b, kf), list(a.columns))

REC = ["sub_id", "structure_id", "cycle_id", "bos_anchor_idx", "cts_anchor_idx"]
for lens, pat in LENS.items():
    a, b = read(one(BASE, pat.format("ic_candidates"))), read(cur_path(CUR, pat.format("ic_candidates")))
    print(f"== 4. IC candidates {lens}: rows {len(a)} -> {len(b)}")
    cols = [c for c in a.columns]
    ga = {k: g for k, g in a.groupby(REC)}
    gb = {k: g for k, g in b.groupby(REC)}
    for k in sorted(set(ga) | set(gb), key=str):
        if k not in gb:
            print(f"  - record {k} ({len(ga[k])} rows)")
            continue
        if k not in ga:
            print(f"  + record {k} ({len(gb[k])} rows)")
            continue
        ra = ga[k].set_index("ic_idx")
        rb = gb[k].set_index("ic_idx")
        if list(ra.index) != list(rb.index):
            print(f"  ~ record {k}: rows {list(ra.index)[:6]}… -> {list(rb.index)[:6]}… ({len(ra)} -> {len(rb)})")
            continue
        changed = Counter()
        for ic in ra.index:
            for c in cols:
                if c in ("ic_idx",) or c not in rb.columns:
                    continue
                if ra.loc[ic, c] != rb.loc[ic, c]:
                    changed[c] += 1
        if changed:
            picks = (list(ra.index[ra["picked"] == "True"]), list(rb.index[rb["picked"] == "True"]))
            print(f"  ~ record {k} ({len(ra)} rows): changed {dict(changed)}; picked {picks[0]} -> {picks[1]}")

if CHARTS:
    for suffix in ("rk2-5.html", "rk2-5_M15_confluence.html", "rk2-5_M15_counter.html"):
        (da, la), (db, lb) = load_fig(one(BASE, "*" + suffix)), load_fig(cur_path(CHARTS, "*" + suffix))
        ka = Counter(json.dumps(t, sort_keys=True, default=str) for t in da)
        kb = Counter(json.dumps(t, sort_keys=True, default=str) for t in db)
        sa = Counter(json.dumps(x, sort_keys=True, default=str) for x in la.get("shapes", []))
        sb = Counter(json.dumps(x, sort_keys=True, default=str) for x in lb.get("shapes", []))
        gone = Counter(json.loads(k).get("name") for k in (ka - kb).elements())
        new = Counter(json.loads(k).get("name") for k in (kb - ka).elements())
        rest = {k: v for k, v in la.items() if k != "shapes"} != {k: v for k, v in lb.items() if k != "shapes"}
        print(f"== 5. figure {suffix}: traces {len(da)} -> {len(db)}, shapes {len(la.get('shapes', []))} -> "
              f"{len(lb.get('shapes', []))}; traces only in the save {dict(gone)}; only now {dict(new)}; shapes "
              f"only in the save {sum((sa - sb).values())}, only now {sum((sb - sa).values())}; layout beyond the "
              f"shapes {'DIFFERS' if rest else 'equal'}")
