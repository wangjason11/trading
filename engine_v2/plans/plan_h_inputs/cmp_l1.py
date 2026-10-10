"""PLAN_L landing 1 — the keyed compare of the IC-candidate CSVs (`*_ip_candidates.csv` in a save before L1 →
`*_ic_candidates.csv` from L1 on), PLAN_L §3.3. A positional diff cannot follow the inserted rows / columns.

    python engine_v2/plans/plan_h_inputs/cmp_l1.py BASE_SAVE_DIR CUR_DEBUG_DIR

Checks, per lens (H1, M15 confluence, M15 counter):
  1. every baseline row is in the current CSV, equal on every baseline column (key = the record + `ic_idx`; the
     `IP: none` row keyed by its record);
  2. the current CSV adds exactly the two columns `has_unfilled_imbalance_after`, `is_ic_candidate`, right after
     `immediate_imbalance_first_c2_idx`;
  3. on every old CANDIDATE row: `has_unfilled_imbalance_after` True and `is_ic_candidate` == (`touches_zone` and
     `ic_ref_reason` empty); on every `IP: none` row both new columns empty;
  4. every NEW row: `has_unfilled_imbalance_after` False, `is_ic_candidate` False, `picked` False,
     `immediate_imbalance_first_c2_idx` empty, the pattern columns filled (`ip_rank` or `reason`);
  5. the per-lens new-row counts == the prediction (H1 +5, confluence +41, counter +31; 2026-10-09).
Prints RESULT OK / FAIL.
"""
import glob
import os
import sys

import pandas as pd

BASE, CUR = sys.argv[1], sys.argv[2]
LENSES = {"H1": ("*rk2-5_ip_candidates.csv", "*rk2-5_ic_candidates.csv", 5),
          "confluence": ("*M15_confluence_ip_candidates.csv", "*M15_confluence_ic_candidates.csv", 41),
          "counter": ("*M15_counter_ip_candidates.csv", "*M15_counter_ic_candidates.csv", 31)}
REC = ["structure_path_id", "sub_id", "structure_id", "cycle_id", "bos_anchor_idx", "cts_anchor_idx"]
NEW_COLS = ["has_unfilled_imbalance_after", "is_ic_candidate"]
ok = True


def fail(msg):
    global ok
    ok = False
    print("  FAIL", msg)


def norm(v):
    return "" if pd.isna(v) else str(v)


for lens, (pat_old, pat_new, want_new) in LENSES.items():
    a = pd.read_csv(glob.glob(os.path.join(BASE, pat_old))[0], dtype=str, keep_default_na=False)
    b = pd.read_csv(glob.glob(os.path.join(CUR, pat_new))[0], dtype=str, keep_default_na=False)
    print(f"== {lens}: rows {len(a)} -> {len(b)}")
    ca, cb = list(a.columns), list(b.columns)
    k = ca.index("immediate_imbalance_first_c2_idx")
    if cb != ca[:k + 1] + NEW_COLS + ca[k + 1:]:
        fail(f"columns: {cb}")
    key = lambda df: df[REC + ["ic_idx"]].astype(str).agg("|".join, axis=1)
    a["_k"], b["_k"] = key(a), key(b)
    if a["_k"].duplicated().any() or b["_k"].duplicated().any():
        fail("duplicate keys")
    bi = b.set_index("_k")
    for _, r in a.iterrows():
        if r["_k"] not in bi.index:
            fail(f"baseline row missing: {r['_k']}")
            continue
        c = bi.loc[r["_k"]]
        diff = [col for col in ca if col != "_k" and norm(r[col]) != norm(c[col])]
        if diff:
            fail(f"row {r['_k']} differs on {diff}")
        if r["ic_idx"] == "":
            if c["has_unfilled_imbalance_after"] != "" or c["is_ic_candidate"] != "":
                fail(f"none row {r['_k']}: new columns not empty")
        else:
            if c["has_unfilled_imbalance_after"] != "True":
                fail(f"old row {r['_k']}: has_unfilled_imbalance_after {c['has_unfilled_imbalance_after']}")
            want = str(r["touches_zone"] == "True" and r["ic_ref_reason"] == "")
            if c["is_ic_candidate"] != want:
                fail(f"old row {r['_k']}: is_ic_candidate {c['is_ic_candidate']} != {want}")
    new = b[~b["_k"].isin(set(a["_k"]))]
    for _, r in new.iterrows():
        bad = [(col, r[col]) for col, w in (("has_unfilled_imbalance_after", "False"), ("is_ic_candidate", "False"),
                                            ("picked", "False"), ("immediate_imbalance_first_c2_idx", ""))
               if r[col] != w]
        if bad or (r["ip_rank"] == "" and r["reason"] == ""):
            fail(f"new row {r['_k']}: {bad or 'no pattern columns'}")
    print(f"   new rows {len(new)} (predicted {want_new})")
    if len(new) != want_new:
        fail(f"{lens}: {len(new)} new rows, predicted {want_new}")

print("RESULT", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
