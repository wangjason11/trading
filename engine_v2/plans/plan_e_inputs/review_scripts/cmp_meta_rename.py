"""Key-RENAME diff of every CSV in two replay folders: BASE vs CUR (same row order).

usage (repo root): python cmp_meta_rename.py BASE_DIR CUR_DIR old=new [old=new ...]   (2026-09-29d)

For a meta-key rename (LANDMINES "Event Contract Rules" rule 3): `cmp_save.py` counts changed CELLS and
`cmp_meta_keys.py` checks VALUE shifts; this checks that CUR == BASE with ONLY the listed keys renamed:
  - a CSV without a `meta` column must be byte-identical;
  - a CSV with one: same header, same rows, every non-meta column equal, and each row's meta equal to the BASE
    meta with the listed keys renamed IN PLACE (same values, same key order — the emitters rename inside the
    dict literal); no old key may remain in CUR;
  - per (CSV, event type / level kind, old key) the renamed-key count, and the changed-cell count per CSV.
Exit 1 on any other difference. Files present on one side only are reported (and fail).
"""
import ast
import filecmp
import glob
import os
import sys
from collections import Counter

import pandas as pd

base, cur, *pairs = sys.argv[1:]
RENAME = dict(p.split("=", 1) for p in pairs)
assert RENAME, "give at least one old=new"
bad, per_key, cells = [], Counter(), Counter()


def _short(f):
    n = os.path.basename(f)
    return n.split("rk2-5_")[-1] if "rk2-5_" in n else n.split("2026-01-20_")[-1]


def _meta(s):
    return ast.literal_eval(s) if isinstance(s, str) and s.startswith("{") else None


bfiles = {os.path.basename(f): f for f in glob.glob(os.path.join(base, "*.csv"))}
cfiles = {os.path.basename(f): f for f in glob.glob(os.path.join(cur, "*.csv"))}
for n in sorted(set(bfiles) ^ set(cfiles)):
    bad.append(f"only in {'BASE' if n in bfiles else 'CUR'}: {n}")

for n in sorted(set(bfiles) & set(cfiles)):
    fb, fc = bfiles[n], cfiles[n]
    A, B = pd.read_csv(fb, low_memory=False), pd.read_csv(fc, low_memory=False)
    if "meta" not in A.columns:
        if not filecmp.cmp(fb, fc, shallow=False):
            bad.append(f"{_short(n)}: no meta column and not byte-identical")
        continue
    if list(A.columns) != list(B.columns) or len(A) != len(B):
        bad.append(f"{_short(n)}: header / row count differs")
        continue
    other = [c for c in A.columns if c != "meta"]
    neq = ~((A[other] == B[other]) | (A[other].isna() & B[other].isna())).all(axis=1)
    if neq.any():
        bad.append(f"{_short(n)}: {int(neq.sum())} rows differ outside meta (first row {int(neq.idxmax())})")
    kind = "type" if "type" in A.columns else ("kind" if "kind" in A.columns else None)
    for i in range(len(A)):
        ma, mb = _meta(A.at[i, "meta"]), _meta(B.at[i, "meta"])
        if ma is None or mb is None:
            if str(A.at[i, "meta"]) != str(B.at[i, "meta"]):
                bad.append(f"{_short(n)} row {i}: unparsed meta differs")
            continue
        renamed = [(RENAME.get(k, k), v) for k, v in ma.items()]
        if renamed != list(mb.items()):
            bad.append(f"{_short(n)} row {i}: meta is not BASE with the keys renamed")
            continue
        left = [k for k in mb if k in RENAME]
        if left:
            bad.append(f"{_short(n)} row {i}: old key(s) still present {left}")
        hit = [k for k in ma if k in RENAME]
        for k in hit:
            per_key[(_short(n), A.at[i, kind] if kind else "-", k)] += 1
        cells[_short(n)] += int(bool(hit))

print("renamed keys per (CSV, type, old key):")
for (f, t, k), v in sorted(per_key.items()):
    print(f"  {f:45s} {str(t):22s} {k:10s} -> {RENAME[k]:24s} {v:4d}")
print(f"cells: {dict(cells)}  total cells {sum(cells.values())}  total keys {sum(per_key.values())}")
if bad:
    print("PROBLEMS:")
    for b in bad[:40]:
        print("  " + b)
    sys.exit(1)
print("OK: every other cell identical; CSVs without meta byte-identical")
