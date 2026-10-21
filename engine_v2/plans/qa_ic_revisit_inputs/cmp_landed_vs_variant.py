"""PLAN_S step 3: the landed replay vs the behaviour-only variant (`make_variant_step1b.py`) — every CSV md5-equal
except the two step-3 save-format additions, which must equal the variant once removed: `*_ic_candidates.csv` minus
its five Step 1B columns, `*_poi_zones.csv` with the six Step 1B meta keys stripped from the `meta` cell.

    python engine_v2/plans/qa_ic_revisit_inputs/cmp_landed_vs_variant.py VARIANT_DEBUG LANDED_DEBUG
"""
import ast
import glob
import hashlib
import os
import sys

import pandas as pd

VAR, CUR = sys.argv[1], sys.argv[2]
NEW_COLS = ["prior_imbalance_first_c2_idx", "prior_imbalance_unfilled", "prior_imbalance_outcome",
            "prior_imbalance_passer_count", "ic_candidate_count"]
NEW_KEYS = ["prior_imbalance_outcome", "prior_imbalance_first_c2_idx", "prior_imbalance_passer_count",
            "ic_candidate_count", "prior_imbalance_outranked_ic_idx", "prior_imbalance_outranked_ip_rank"]


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


bad = 0
for vp in sorted(glob.glob(os.path.join(VAR, "*.csv"))):
    name = os.path.basename(vp)
    cp = os.path.join(CUR, name)
    if not os.path.exists(cp):
        print("MISSING", name)
        bad += 1
        continue
    if md5(vp) == md5(cp):
        continue
    v = pd.read_csv(vp, dtype=str, keep_default_na=False)
    c = pd.read_csv(cp, dtype=str, keep_default_na=False)
    if name.endswith("_ic_candidates.csv"):
        assert all(k in c.columns for k in NEW_COLS) and not any(k in v.columns for k in NEW_COLS), name
        same = c.drop(columns=NEW_COLS).equals(v)
        print(f"{'OK  ' if same else 'DIFF'} {name}: equal without the 5 new columns" if same else f"DIFF {name}")
    elif name.endswith("_poi_zones.csv"):
        def strip(s):
            m = ast.literal_eval(s)
            assert all(k in m for k in NEW_KEYS), sorted(set(NEW_KEYS) - set(m))
            return {k: x for k, x in m.items() if k not in NEW_KEYS}
        same = list(v.columns) == list(c.columns) and len(v) == len(c) and v.drop(columns=["meta"]).equals(
            c.drop(columns=["meta"])) and all(ast.literal_eval(a) == strip(b) for a, b in zip(v["meta"], c["meta"]))
        print(f"{'OK  ' if same else 'DIFF'} {name}: equal with the 6 new meta keys stripped")
    else:
        same = False
        print("DIFF", name, "(not a step-3 file)")
    bad += not same
print("RESULT:", "landed == variant + the step-3 additions" if bad == 0 else f"{bad} file(s) differ")
