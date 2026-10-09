"""PLAN_I S6 (2026-10-09, session 7): CUR == BASE with ONLY the `fib_mode` column dropped and the `fib_mode` meta key
deleted (a rule-3 migration; the key was "cross_cycle" on every fib record since S4).

usage (repo root): python engine_v2/plans/plan_i_inputs/cmp_s6_fib_mode.py BASE_DIR CUR_DIR

- every BASE CSV must exist in CUR (CUR's other files are ignored — artifacts/debug also holds stale outputs);
- a CSV without a `fib_mode` column: md5-identical;
- a CSV with one: CUR's header == BASE's without `fib_mode`; per row, every other non-meta cell TEXT-identical, the
  dropped cell == "cross_cycle", and CUR's meta text == `str()` of BASE's meta with the top-level key deleted (its
  value "cross_cycle"; key order kept); "fib_mode" appears nowhere in CUR's file.
Raw text via the csv module (no pandas coercion). Exit 1 on any other difference.
"""
import ast
import csv
import filecmp
import glob
import os
import sys

base, cur = sys.argv[1:3]
KEY = "fib_mode"
bad, report = [], []


def _rows(f):
    with open(f, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def _short(n):
    return n.split("rk2-5")[-1] if "rk2-5" in n else n.split("2026-01-20_")[-1]


for fb in sorted(glob.glob(os.path.join(base, "*.csv"))):
    n = os.path.basename(fb)
    fc = os.path.join(cur, n)
    if not os.path.exists(fc):
        bad.append(f"MISSING in CUR: {n}")
        continue
    A, B = _rows(fb), _rows(fc)
    if KEY not in A[0]:
        if not filecmp.cmp(fb, fc, shallow=False):
            bad.append(f"{_short(n)}: no {KEY} column in BASE and not byte-identical")
        continue
    ci, mi = A[0].index(KEY), A[0].index("meta")
    if B[0] != [h for h in A[0] if h != KEY] or len(A) != len(B):
        bad.append(f"{_short(n)}: header is not BASE's without {KEY}, or the row count differs")
        continue
    mi_c = B[0].index("meta")
    n_col = n_key = 0
    for i, (ra, rb) in enumerate(zip(A[1:], B[1:])):
        if ra[ci] != "cross_cycle":
            bad.append(f"{_short(n)} row {i}: BASE {KEY} cell {ra[ci]!r}")
        n_col += 1
        rest_a = [x for j, x in enumerate(ra) if j not in (ci, mi)]
        rest_b = [x for j, x in enumerate(rb) if j != mi_c]
        if rest_a != rest_b:
            bad.append(f"{_short(n)} row {i}: a non-meta cell differs (text)")
            continue
        ma = ast.literal_eval(ra[mi])
        if str(ma) != ra[mi]:
            bad.append(f"{_short(n)} row {i}: BASE meta does not round-trip through str()")
            continue
        if ma.get(KEY) != "cross_cycle":
            bad.append(f"{_short(n)} row {i}: BASE meta {KEY}={ma.get(KEY)!r}")
            continue
        del ma[KEY]
        n_key += 1
        if str(ma) != rb[mi_c]:
            bad.append(f"{_short(n)} row {i}: CUR meta is not BASE's minus {KEY}")
    with open(fc, encoding="utf-8") as fh:
        if KEY in fh.read():
            bad.append(f"{_short(n)}: '{KEY}' still appears in CUR")
    report.append(f"{_short(n)}: {len(A) - 1} rows - {n_col} column cells dropped, {n_key} meta keys deleted")

print("\n".join(report) if report else f"no CSV with a {KEY} column in BASE")
print("\n".join(bad) if bad else "OK: every other CSV md5-identical; the fib CSVs == BASE minus the column and the key")
sys.exit(1 if bad else 0)
