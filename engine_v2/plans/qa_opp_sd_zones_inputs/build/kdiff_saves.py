"""Keyed diff of two replays' M15 lens CSVs, by SUB KEY = (direction, starting_idx) — `sub_id`s renumber whenever a
sub is added or removed (start order), so a positional diff of the lens CSVs is mostly renumbering (the `/compare`
skill's PLAN_R step 2 entry).

    python kdiff_saves.py DIR_A DIR_B [--rows]

DIR_* = a save folder or `artifacts/debug` (the files named NZD_USD_H1_..._M15_{lens}_*.csv). Each side's sub_id → key
map comes from ITS OWN `_M15_{lens}_subs.csv` (both lenses). Every sub-id-valued column (and meta key) is remapped to
the key, `seq` is dropped, and each key's rows are compared as a multiset. Prints, per lens CSV: the keys only in A,
only in B, and the keys whose rows differ (with --rows: each differing row's columns). Rows without a sub id are keyed
None."""
import ast
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

LENSES = ("counter", "confluence")
SUB_COLS = ("sub_id", "ended_by_sub_id", "cross_fib_limit_from_sub_id", "replaced_by_sub_id", "parent_sub_id")


def prefix(d):
    hits = glob.glob(os.path.join(d, "*_M15_counter_subs.csv"))
    assert len(hits) == 1, (d, hits)
    return hits[0][: -len("_M15_counter_subs.csv")]


def key_map(p):
    m = {}
    for lens in LENSES:
        with open(f"{p}_M15_{lens}_subs.csv", newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                k = (int(float(r["direction"])), int(float(r["starting_idx"])))
                sid = int(float(r["sub_id"]))
                assert m.get(sid, k) == k, (p, sid, m.get(sid), k)
                m[sid] = k
    return m


def _remap(v, m):
    if v in ("", "None", "nan", None):
        return v
    try:
        return str(m[int(float(v))])
    except (ValueError, KeyError):
        return v


def _num(v):
    # "3589" == "3589.0": a column's dtype turns float when the lens has an open (NaN) row — not a value change
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return "nan" if f != f else repr(f)


def load(path, m):
    out = defaultdict(Counter)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta = {}
            if row.get("meta", "").startswith("{"):
                meta = ast.literal_eval(row["meta"])
            sid = row.get("sub_id") or meta.get("sub_id")
            key = m.get(int(float(sid))) if sid not in (None, "", "None", "nan") else None
            flat = {k: _num(v) for k, v in row.items() if k not in ("meta", "seq")}
            for k, v in meta.items():
                flat["meta." + k] = repr(v)
            for c in list(flat):
                if c.split(".")[-1] in SUB_COLS:
                    flat[c] = _remap(flat[c].strip("'") if isinstance(flat[c], str) else flat[c], m)
            out[key][tuple(sorted(flat.items()))] += 1
    return out


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    rows = "--rows" in sys.argv
    pa, pb = prefix(a_dir), prefix(b_dir)
    ma, mb = key_map(pa), key_map(pb)
    print(f"A {a_dir}: {len(set(ma.values()))} subs; B {b_dir}: {len(set(mb.values()))} subs")
    print(f"  only in A: {sorted(set(ma.values()) - set(mb.values()))}; only in B: "
          f"{sorted(set(mb.values()) - set(ma.values()))}")
    tails = sorted({os.path.basename(p)[len(os.path.basename(pa)):] for p in glob.glob(pa + "_M15_*.csv")})
    for tail in tails:
        if "unresolved" in tail:
            continue
        fa, fb = pa + tail, pb + tail
        if not os.path.exists(fb):
            print(f"{tail}: MISSING in B")
            continue
        a, b = load(fa, ma), load(fb, mb)
        only_a = sorted((k for k in a if k not in b), key=str)
        only_b = sorted((k for k in b if k not in a), key=str)
        diff = sorted((k for k in a if k in b and a[k] != b[k]), key=str)
        if not (only_a or only_b or diff):
            print(f"{tail}: identical by key ({sum(sum(c.values()) for c in a.values())} rows)")
            continue
        print(f"{tail}: only A {only_a}  only B {only_b}  differ {diff}")
        if rows:
            for k in diff:
                xa, xb = a[k] - b[k], b[k] - a[k]
                print(f"   {k}: {sum(xa.values())} rows only in A, {sum(xb.values())} only in B")
                for r in list(xa)[:6]:
                    print("     A", dict(r))
                for r in list(xb)[:6]:
                    print("     B", dict(r))


if __name__ == "__main__":
    main()
