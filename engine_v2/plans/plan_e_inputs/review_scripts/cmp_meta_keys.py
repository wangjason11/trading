"""Key-level meta diff of the M15 lens CSVs: BASE vs CUR (same row order).

usage: python cmp_meta_keys.py BASE_DIR CUR_DIR   (Post-E·2, 2026-09-26)
For a change to meta VALUES: cmp_save.py counts changed CELLS (one meta cell can hold several
changed keys: Post-E·2 = 547 cells / 715 keys); this counts KEYS and checks each delta.
Per (csv kind, event type, meta key): changed-value count, and whether every
change is exactly +slice_begin of the row's sub (slice_begin = the subs CSV
starting_idx - 50). Non-meta columns are compared too (any difference printed).
Exit 1 if a change is not +slice_begin or a non-meta column differs.
"""
import ast
import glob
import os
import sys
from collections import Counter

import pandas as pd

base, cur = sys.argv[1], sys.argv[2]
bad = []
per_key = Counter()
cells = Counter()


def load(folder, pat):
    fs = glob.glob(os.path.join(folder, pat))
    assert len(fs) == 1, (folder, pat, fs)
    return pd.read_csv(fs[0])


for lens in ("confluence", "counter"):
    subs = load(base, f"*_M15_{lens}_subs.csv")
    sb = {int(r.sub_id): int(r.starting_idx) - 50 for r in subs.itertuples()}
    for kind in ("structure_events", "kl_zones", "poi_zones", "fib_lifecycle", "wvmi",
                 "subs", "triggers"):
        a, b = load(base, f"*_M15_{lens}_{kind}.csv"), load(cur, f"*_M15_{lens}_{kind}.csv")
        assert list(a.columns) == list(b.columns), (lens, kind, "header")
        assert len(a) == len(b), (lens, kind, "rows", len(a), len(b))
        for col in a.columns:
            if col == "meta":
                continue
            diff = ~((a[col] == b[col]) | (a[col].isna() & b[col].isna()))
            if diff.any():
                bad.append((lens, kind, col, int(diff.sum())))
        if "meta" not in a.columns:
            continue
        for i in range(len(a)):
            ma = ast.literal_eval(a.at[i, "meta"]) if isinstance(a.at[i, "meta"], str) else {}
            mb = ast.literal_eval(b.at[i, "meta"]) if isinstance(b.at[i, "meta"], str) else {}
            et = a.at[i, "type"] if kind == "structure_events" else "-"
            if set(ma) != set(mb):
                bad.append((lens, kind, i, "keyset", sorted(set(ma) ^ set(mb))))
                continue
            changed = False
            for k in ma:
                if ma[k] == mb[k]:
                    continue
                changed = True
                per_key[(kind, et, k, lens)] += 1
                off = sb.get(ma.get("sub_id"))
                if not (isinstance(ma[k], int) and isinstance(mb[k], int) and mb[k] - ma[k] == off):
                    bad.append((lens, kind, i, k, ma[k], mb[k], off))
            if changed:
                cells[(kind, lens)] += 1

ua = load(base, "*_M15_unresolved_triggers.csv")
ub = load(cur, "*_M15_unresolved_triggers.csv")
if not ua.equals(ub):
    bad.append(("unresolved", "differs"))

rows = {}
for (kind, et, k, lens), n in per_key.items():
    rows.setdefault((kind, et, k), {})[lens] = n
print(f"{'kind':16} {'type':22} {'key':20} conf  ctr")
for key in sorted(rows):
    print(f"{key[0]:16} {key[1]:22} {key[2]:20} {rows[key].get('confluence', 0):4} {rows[key].get('counter', 0):4}")
print("meta CELLS changed:", dict(sorted(cells.items())), "total", sum(cells.values()))
print("keys changed total:", sum(per_key.values()))
print("PROBLEMS:", bad[:20] if bad else "none")
sys.exit(1 if bad else 0)
