"""KEYED diff of the WVMI CSVs (Plan G, 2026-09-30) — `cmp_save.py` is positional, so inserted rows and a mid-row
column misreport every later cell.

    python engine_v2/plans/plan_e_inputs/review_scripts/cmp_wvmi_keyed.py BASE_DIR CUR_DIR

For every `*_wvmi.csv` in BASE_DIR (the H1 one + one per lens), the same-named file in CUR_DIR is compared:
- rows keyed by `(sub_id, bos_structure_id, bos_cycle_id)` (sub_id empty on H1 rows); each key must occur ONCE per CSV;
- columns by NAME (added / removed columns listed with their position);
- per common key: every changed column (base -> cur); the `meta` cell diffed per KEY (added / removed / changed) and
  its key ORDER compared;
- added / removed keys, with the added rows' values; the current row order;
- per current row: the `structure_path_id` column == `meta["structure_path_id"]`.
Exit 1 on a duplicate key or a path mismatch. Windows: keep BASE_DIR short (%TEMP%/pg/...).
"""
import ast
import glob
import os
import sys

import pandas as pd

base_dir, cur_dir = sys.argv[1:3]
KEY = ["sub_id", "bos_structure_id", "bos_cycle_id"]
bad = 0


def load(p):
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    return df


def meta(v):
    return ast.literal_eval(v) if v.startswith("{") else {}


def key_of(r):
    return tuple(r[k] for k in KEY)


for bp in sorted(glob.glob(os.path.join(base_dir, "*_wvmi.csv"))):
    name = os.path.basename(bp)
    cp = os.path.join(cur_dir, name)
    short = name.split("rk2-5_")[-1]
    print(f"\n=== {short}")
    b, c = load(bp), load(cp)
    print(f"rows {len(b)} -> {len(c)}")
    bcols, ccols = list(b.columns), list(c.columns)
    for col in ccols:
        if col not in bcols:
            print(f"  + column {col!r} at position {ccols.index(col)} (after {ccols[ccols.index(col) - 1]!r})")
    for col in bcols:
        if col not in ccols:
            print(f"  - column {col!r}")
    common_cols = [x for x in bcols if x in ccols]
    brows = {key_of(r): r for _, r in b.iterrows()}
    crows = {key_of(r): r for _, r in c.iterrows()}
    for label, df, rows in (("base", b, brows), ("cur", c, crows)):
        if len(rows) != len(df):
            print(f"  !! DUPLICATE keys in {label}: {len(df)} rows, {len(rows)} keys")
            bad = 1
    for k, r in crows.items():
        m = meta(r["meta"])
        if r["structure_path_id"] != m.get("structure_path_id"):
            print(f"  !! path mismatch {k}: column {r['structure_path_id']!r} meta {m.get('structure_path_id')!r}")
            bad = 1
    print(f"  current key order: {[k for k in crows]}")
    for k in brows:
        if k not in crows:
            print(f"  - row {k}")
    for k, r in crows.items():
        if k not in brows:
            vals = {col: r[col] for col in ccols if col not in KEY and col != "meta"}
            print(f"  + row {k}: {vals}")
            print(f"      meta {r['meta']}")
    n_same = 0
    for k in brows:
        if k not in crows:
            continue
        br, cr = brows[k], crows[k]
        diffs = [(col, br[col], cr[col]) for col in common_cols if col != "meta" and br[col] != cr[col]]
        bm, cm = meta(br["meta"]), meta(cr["meta"])
        mdiff = []
        for mk in cm:
            if mk not in bm:
                mdiff.append(f"+{mk}={cm[mk]!r}")
            elif bm[mk] != cm[mk]:
                mdiff.append(f"{mk}: {bm[mk]!r} -> {cm[mk]!r}")
        for mk in bm:
            if mk not in cm:
                mdiff.append(f"-{mk}")
        order_same = [x for x in bm if x in cm] == [x for x in cm if x in bm]
        new_cols = {col: cr[col] for col in ccols if col not in bcols}
        if diffs or mdiff or not order_same:
            print(f"  ~ {k}: {diffs} meta {mdiff}{'' if order_same else ' META KEY ORDER CHANGED'} new-cols {new_cols}")
        else:
            n_same += 1
            print(f"  = {k} new-cols {new_cols}")
    print(f"  {n_same} common rows unchanged apart from new columns")

sys.exit(bad)
