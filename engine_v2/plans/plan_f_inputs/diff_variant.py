"""Diff a variant's CSVs against the /compare baseline save.

usage: python diff_variant.py <variant_dir> [--rows N]
Per CSV: IDENTICAL, or row-multiset diff (rows only in baseline / only in variant),
paired by the file's natural key where one exists so changed cells print as old -> new.
"""
import filecmp
import glob
import os
import sys

import pandas as pd

REPO = "C:/Users/wangj/OneDrive/Documents/codingproj/Project Retire/forex_engine_v2"
BASE = f"{REPO}/artifacts/commits/week8-volmom-multitf/20260923_172626_0a4eadc"
var_dir = sys.argv[1]

def L(path):
    return "\\\\?\\" + os.path.abspath(path).replace("/", "\\")

MAXROWS = int(sys.argv[sys.argv.index("--rows") + 1]) if "--rows" in sys.argv else 12

KEYS = {
    "poi_zones": ["structure_id", "cycle_id", "ic_idx"],
    "fib_lifecycle": None,
    "kl_zones": None,
    "structure_events": None,
}

base_csvs = sorted(glob.glob(f"{BASE}/*.csv"))
ident, changed = 0, []
for b in base_csvs:
    name = os.path.basename(b)
    v = L(os.path.join(var_dir, name))
    b = L(b)
    if not os.path.exists(v):
        changed.append((name, "MISSING in variant"))
        continue
    if filecmp.cmp(b, v, shallow=False):
        ident += 1
        continue
    db = pd.read_csv(b, low_memory=False)
    dv = pd.read_csv(v, low_memory=False)
    msg = [f"rows base={len(db)} var={len(dv)}"]
    if list(db.columns) != list(dv.columns):
        msg.append(f"COLUMNS differ: {set(db.columns) ^ set(dv.columns)}")
    else:
        sb = db.astype(str).apply(tuple, axis=1)
        sv = dv.astype(str).apply(tuple, axis=1)
        from collections import Counter
        cb, cv = Counter(sb), Counter(sv)
        only_b = list((cb - cv).elements())
        only_v = list((cv - cb).elements())
        msg.append(f"rows only-in-base={len(only_b)} only-in-var={len(only_v)}")
        cols = list(db.columns)
        # pair by position when counts are equal and row counts equal
        if len(db) == len(dv):
            diffmask = (db.astype(str) != dv.astype(str))
            rows = diffmask.any(axis=1)
            idxs = list(rows[rows].index)
            msg.append(f"positional changed rows={len(idxs)}")
            for r in idxs[:MAXROWS]:
                cc = [c for c in cols if diffmask.at[r, c]]
                keyinfo = {k: db.at[r, k] for k in ("structure_id", "sid", "sub_id", "cycle_id", "ic_idx", "idx", "type", "event_type") if k in cols}
                msg.append(f"   row {r} {keyinfo}: " + "; ".join(
                    f"{c}: {str(db.at[r, c])[:120]} -> {str(dv.at[r, c])[:120]}" for c in cc))
        else:
            for t in only_b[:MAXROWS]:
                msg.append("   - " + str(dict(zip(cols, t)))[:400])
            for t in only_v[:MAXROWS]:
                msg.append("   + " + str(dict(zip(cols, t)))[:400])
    changed.append((name, "\n    ".join(msg)))

print(f"{var_dir}: {ident}/{len(base_csvs)} CSVs byte-identical")
for name, m in changed:
    print(f"  {name.replace('NZD_USD_H1_2025-11-15_2026-01-20_', '')}\n    {m}")
