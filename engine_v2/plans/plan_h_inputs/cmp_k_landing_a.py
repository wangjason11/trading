"""PLAN_K landing A's check: the current replay vs a pre-A save (S_S6) == the prediction (PLAN_K §5).

Usage (repo root): python engine_v2/plans/plan_h_inputs/cmp_k_landing_a.py BASE_SAVE [CUR_DEBUG]

1. every CSV md5-equal, except the 3 structure-events, 3 fib-lifecycle and 3 ip-candidates CSVs;
2. structure events: the same rows; the ONLY meta delta = `pattern_anchor_idx` added on STATE_CHANGED(to=reversal),
   each value == the REVERSAL_CANDIDATE anchor whose apply is that reversal (on the reference window);
3. fib lifecycle: the same rows; meta minus the 4 `ic_ref_*` keys == the base meta (order included); the `ic_ref_*`
   values printed per record (vs the §5 table);
4. ip candidates: minus the `ic_ref_reason` column == the base; the reason counts per record printed.
Exit 1 on any unexpected delta.
"""
import ast
import hashlib
import sys
from pathlib import Path

import pandas as pd

BASE = Path(sys.argv[1])
CUR = Path(sys.argv[2] if len(sys.argv) > 2 else "artifacts/debug")
IC_KEYS = ("ic_ref_price", "ic_ref_source", "ic_ref_break_pattern_anchor_idx", "ic_ref_break_confirmed_idx")
bad = []


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def meta(s):
    return ast.literal_eval(s) if isinstance(s, str) and s else {}


base_csvs = sorted(p.name for p in BASE.glob("*.csv"))
cur_csvs = sorted(p.name for p in CUR.glob("*.csv"))
if base_csvs != cur_csvs:
    bad.append(f"file lists differ: only base {sorted(set(base_csvs) - set(cur_csvs))}, "
               f"only cur {sorted(set(cur_csvs) - set(base_csvs))}")
changed = [n for n in base_csvs if n in cur_csvs and md5(BASE / n) != md5(CUR / n)]
expected_families = ("_structure_events.csv", "_fib_lifecycle.csv", "_ip_candidates.csv")
print(f"{len(base_csvs)} CSVs; md5-equal {len(base_csvs) - len(changed)}; changed {len(changed)}:")
for n in changed:
    print("   ", n)
    if not n.endswith(expected_families):
        bad.append(f"unexpected change: {n}")

for n in [c for c in changed if c.endswith("_structure_events.csv")]:
    b, c = pd.read_csv(BASE / n), pd.read_csv(CUR / n)
    if len(b) != len(c) or list(b.columns) != list(c.columns):
        bad.append(f"{n}: shape / columns differ")
        continue
    stamped = []
    for i in range(len(b)):
        rb, rc = b.iloc[i], c.iloc[i]
        same_cols = all((rb[k] == rc[k]) or (pd.isna(rb[k]) and pd.isna(rc[k])) for k in b.columns if k != "meta")
        mb, mc = meta(rb["meta"]), meta(rc["meta"])
        if not same_cols:
            bad.append(f"{n} row {i}: a non-meta column differs")
        if mb == mc:
            continue
        extra = {k: v for k, v in mc.items() if k not in mb}
        if (rc["type"] == "STATE_CHANGED" and mc.get("to") == "reversal" and list(extra) == ["pattern_anchor_idx"]
                and {k: v for k, v in mc.items() if k != "pattern_anchor_idx"} == mb):
            stamped.append((int(rc["idx"]), mc.get("sub_id"), int(extra["pattern_anchor_idx"])))
        else:
            bad.append(f"{n} row {i}: unexpected meta delta {extra} / {mb} -> {mc}")
    # each stamp == the REVERSAL_CANDIDATE anchor whose apply is that reversal
    for idx, sub, anchor in stamped:
        cands = [meta(r["meta"]) for _, r in c.iterrows() if r["type"] == "REVERSAL_CANDIDATE"
                 and meta(r["meta"]).get("apply_idx") == idx and meta(r["meta"]).get("sub_id") == sub]
        ok = any(m["pattern_anchor_idx"] == anchor for m in cands)
        if not ok:
            bad.append(f"{n}: reversal @{idx} sub {sub} stamped {anchor}, no REVERSAL_CANDIDATE with that anchor")
    print(f"  {n.split('rk2-5')[-1]}: STATE_CHANGED(to=reversal) stamped (idx, sub, pattern_anchor_idx): {stamped}")

for n in [c for c in changed if c.endswith("_fib_lifecycle.csv")]:
    b, c = pd.read_csv(BASE / n), pd.read_csv(CUR / n)
    if len(b) != len(c) or list(b.columns) != list(c.columns):
        bad.append(f"{n}: shape / columns differ")
        continue
    print(f"  {n.split('rk2-5')[-1]}:")
    for i in range(len(b)):
        rb, rc = b.iloc[i], c.iloc[i]
        mb, mc = meta(rb["meta"]), meta(rc["meta"])
        stripped = {k: v for k, v in mc.items() if k not in IC_KEYS}
        if stripped != mb or list(stripped) != list(mb):
            bad.append(f"{n} row {i}: meta minus ic_ref_* != base")
        for k in b.columns:
            if k != "meta" and not ((rb[k] == rc[k]) or (pd.isna(rb[k]) and pd.isna(rc[k]))):
                bad.append(f"{n} row {i}: column {k} differs")
        ic = {k: mc[k] for k in IC_KEYS if k in mc}
        sub = mc.get("sub_id", "-")
        print(f"     sub {sub} sid {rc['structure_id']} cyc {rc['cycle_id']} v{rc['version']} "
              f"{rc['bos_idx']}->{rc['cts_idx']}: {ic or 'IC ref: none'}")

for n in [c for c in changed if c.endswith("_ip_candidates.csv")]:
    b, c = pd.read_csv(BASE / n), pd.read_csv(CUR / n)
    if "ic_ref_reason" not in c.columns or list(c.drop(columns=["ic_ref_reason"]).columns) != list(b.columns):
        bad.append(f"{n}: columns are not base + ic_ref_reason")
        continue
    if not c.drop(columns=["ic_ref_reason"]).equals(b):
        bad.append(f"{n}: rows differ beyond ic_ref_reason")
    if "sub_id" not in c:
        c["sub_id"] = -1
    cand = c[c["ic_idx"].notna()]
    g = cand.groupby([cand["sub_id"].fillna(-1).astype(int), "structure_id", "cycle_id", "bos_anchor_idx",
                      "cts_anchor_idx"])
    print(f"  {n.split('rk2-5')[-1]} (record: candidates -> pass | past_reference | not_before_break):")
    for key, grp in g:
        r = grp["ic_ref_reason"]
        print(f"     {key}: {len(grp)} -> {int(r.isna().sum())} | {int((r == 'ic_past_reference').sum())} | "
              f"{int((r == 'ic_not_before_break').sum())}")

print("\nRESULT:", "OK" if not bad else "UNEXPECTED")
for x in bad:
    print("  -", x)
sys.exit(1 if bad else 0)
