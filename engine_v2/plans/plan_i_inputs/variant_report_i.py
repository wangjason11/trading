"""PLAN_I session 2 — a META-AWARE diff of every CSV a variant changed (complements plan_h_inputs/variant_report.py).

    python engine_v2/plans/plan_i_inputs/variant_report_i.py BASE_DIR CUR_DEBUG [--rows N] [--file SUBSTR]

Per changed CSV: row counts; if the non-meta columns are row-for-row equal (same order) -> a FORMAT-ONLY report:
the meta keys added / removed / changed (count of rows, distinct values); else a keyed row diff on the file's
identity columns (fib: path, structure_id, cycle_id, version, the sub (direction, starting_idx) via meta sub_id;
events / levels: positional multiset of the non-meta columns) with the changed columns per kept row.
"""
import ast
import glob
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

BASE, CUR = sys.argv[1], sys.argv[2]
NROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 12
ONLY = sys.argv[sys.argv.index("--file") + 1] if "--file" in sys.argv else None
PFX = "NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def meta(s):
    if not isinstance(s, str):
        return {}
    s2 = re.sub(r"np\.(?:int|float)\d*\(([^)]*)\)", r"\1", s)
    s2 = re.sub(r"np\.(?:True_|False_)", lambda m: "True" if "True" in m.group(0) else "False", s2)
    s2 = re.sub(r"Timestamp\('([^']*)'[^)]*\)", r"'\1'", s2)
    try:
        return ast.literal_eval(s2)
    except Exception:
        return {"_raw": s}


def subkeys(d, sfx):
    p = os.path.join(d, f"{PFX}{sfx}_subs.csv")
    if not os.path.exists(p):
        return {}
    s = pd.read_csv(p)
    return {int(r.sub_id): (int(r.direction), int(r.starting_idx)) for r in s.itertuples()}


def lens_sfx(name):
    for sfx in ("_M15_confluence", "_M15_counter"):
        if name.startswith(PFX + sfx):
            return sfx
    return ""


def fib_key(r, m, sk):
    sub = m.get("sub_id")
    subk = sk.get(int(sub), ("?", sub)) if sub is not None else None
    v = r.get("version")
    v = None if (v is None or (isinstance(v, float) and v != v)) else int(v)
    return (subk, int(r["structure_id"]), int(r["cycle_id"]), v)


def show(x):
    return x if not isinstance(x, float) or x == x else None


for p in sorted(glob.glob(os.path.join(BASE, "*.csv"))):
    name = os.path.basename(p)
    q = os.path.join(CUR, name)
    if ONLY and ONLY not in name:
        continue
    if not os.path.exists(q):
        print(f"\n### MISSING {name}")
        continue
    if md5(p) == md5(q):
        continue
    b, c = pd.read_csv(p, low_memory=False), pd.read_csv(q, low_memory=False)
    short = name[len(PFX):]
    print(f"\n### {short}: rows {len(b)} -> {len(c)}; cols {'same' if list(b.columns) == list(c.columns) else f'{list(b.columns)} -> {list(c.columns)}'}")
    has_meta = "meta" in b.columns and "meta" in c.columns
    nonmeta = [k for k in b.columns if k != "meta" and k in c.columns]
    if len(b) == len(c) and b[nonmeta].astype(str).equals(c[nonmeta].astype(str)):
        if not has_meta:
            print("  non-meta equal, no meta column ?!")
            continue
        added, removed, changed = Counter(), Counter(), Counter()
        vals = defaultdict(Counter)
        for mb, mc in zip(b["meta"], c["meta"]):
            db, dc = meta(mb), meta(mc)
            for k in set(dc) - set(db):
                added[k] += 1
                vals[k][repr(dc[k])] += 1
            for k in set(db) - set(dc):
                removed[k] += 1
            for k in set(db) & set(dc):
                if db[k] != dc[k]:
                    changed[k] += 1
        print(f"  FORMAT-ONLY (non-meta columns row-for-row equal). meta keys: added {dict(added)}; "
              f"removed {dict(removed)}; changed {dict(changed)}")
        for k, cnt in vals.items():
            top = cnt.most_common(8)
            print(f"    {k}: {len(cnt)} distinct -> {top}")
        continue
    if "fib_lifecycle" in name:
        sfx = lens_sfx(name)
        skb, skc = subkeys(BASE, sfx), subkeys(CUR, sfx)
        rb = {fib_key(r, meta(r["meta"]), skb): r for _, r in b.iterrows()}
        rc = {fib_key(r, meta(r["meta"]), skc): r for _, r in c.iterrows()}
        gone, new = sorted(set(rb) - set(rc), key=str), sorted(set(rc) - set(rb), key=str)
        kept = sorted(set(rb) & set(rc), key=str)
        print(f"  fib rows keyed (sub, sid, cycle, version): removed {len(gone)}, added {len(new)}, kept {len(kept)}")
        cols = ["start_idx", "end_idx", "end_reason", "status", "active", "locked", "bos_idx", "cts_idx", "fib_mode"]
        for k in gone[:NROWS]:
            r = rb[k]
            print(f"    - {k} {[show(r[x]) for x in cols]} meta={ {kk: vv for kk, vv in meta(r['meta']).items() if kk in ('via', 'cross_start_cycle', 'activated_at', 'deactivated_by', 'activated_on', 'scenario')} }")
        for k in new[:NROWS]:
            r = rc[k]
            print(f"    + {k} {[show(r[x]) for x in cols]} meta={ {kk: vv for kk, vv in meta(r['meta']).items() if kk in ('via', 'cross_start_cycle', 'activated_at', 'deactivated_by', 'activated_on', 'cross_fib_limit_cycle')} }")
        nchg = 0
        for k in kept:
            x, y = rb[k], rc[k]
            dcols = [col for col in cols if str(x[col]) != str(y[col])]
            mx, my = meta(x["meta"]), meta(y["meta"])
            dk = sorted(kk for kk in set(mx) | set(my) if mx.get(kk, "<absent>") != my.get(kk, "<absent>"))
            if dcols or dk:
                nchg += 1
                if nchg <= NROWS:
                    print(f"    ~ {k} cols {[(col, show(x[col]), show(y[col])) for col in dcols]} meta {dk}")
        print(f"    kept-changed {nchg}")
        continue
    # generic: multiset of non-meta rows
    tb = Counter(tuple(map(str, r)) for r in b[nonmeta].itertuples(index=False))
    tc = Counter(tuple(map(str, r)) for r in c[nonmeta].itertuples(index=False))
    gone, new = tb - tc, tc - tb
    print(f"  non-meta row multiset: removed {sum(gone.values())}, added {sum(new.values())}  (cols {nonmeta})")
    for t in list(gone)[:NROWS]:
        print(f"    - {t}")
    for t in list(new)[:NROWS]:
        print(f"    + {t}")
