"""PLAN_I build: a fib-lifecycle CSV diff keyed by (sub_id | structure_id, cycle_id, version) — columns by name, the
meta per KEY (added / removed / changed keys), rows present on one side only.

    python engine_v2/plans/plan_i_inputs/fib_meta_diff.py BASE_DIR CUR_DIR [--ignore-meta k1,k2]

Each `*_fib_lifecycle.csv` of BASE_DIR is matched by filename in CUR_DIR. `--ignore-meta` drops keys from both metas
before comparing (e.g. to show that a stage differs from a variant ONLY by a label)."""
import ast
import glob
import os
import sys

import pandas as pd

args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE, CUR = args[0], args[1]
IGN = set(sys.argv[sys.argv.index("--ignore-meta") + 1].split(",")) if "--ignore-meta" in sys.argv else set()


def rows(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for _, r in df.iterrows():
        meta = ast.literal_eval(r["meta"]) if r["meta"] else {}
        for k in IGN:
            meta.pop(k, None)
        key = (meta.get("sub_id", r["structure_id"]), r["cycle_id"], r["version"])
        assert key not in out, (path, key)
        out[key] = ({c: r[c] for c in df.columns if c != "meta"}, meta)
    return out


total = 0
for bp in sorted(glob.glob(os.path.join(BASE, "*_fib_lifecycle.csv"))):
    name = os.path.basename(bp)
    cp = os.path.join(CUR, name)
    if not os.path.exists(cp):
        print(f"{name}: MISSING in CUR"); total += 1; continue
    a, b = rows(bp), rows(cp)
    lines = []
    for key in sorted(set(a) | set(b), key=lambda k: tuple(str(x) for x in k)):
        if key not in b:
            lines.append(f"  - ROW REMOVED {key}: {a[key][0]} {a[key][1]}"); continue
        if key not in a:
            lines.append(f"  + ROW ADDED   {key}: {b[key][0]} {b[key][1]}"); continue
        (ca, ma), (cb, mb) = a[key], b[key]
        cols = [f"{c}: {ca.get(c)!r} -> {cb.get(c)!r}" for c in sorted(set(ca) | set(cb)) if ca.get(c) != cb.get(c)]
        mk = [f"meta.{k}: {ma.get(k, '<absent>')!r} -> {mb.get(k, '<absent>')!r}"
              for k in sorted(set(ma) | set(mb)) if ma.get(k, "<absent>") != mb.get(k, "<absent>")]
        if cols or mk:
            lines.append(f"  ~ {key}: " + "; ".join(cols + mk))
    print(f"{name}: {len(lines)} differing rows (BASE {len(a)} rows, CUR {len(b)})")
    for ln in lines:
        print(ln)
    total += len(lines)
sys.exit(1 if total else 0)
