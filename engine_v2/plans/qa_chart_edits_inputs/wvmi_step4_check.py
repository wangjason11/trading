"""Step 4: the replay's three *_wvmi_lps.csv vs the prediction (step4_pred.json from step4_predict.py), row by row,
plus the cross-checks: the last row of each record = its _wvmi.csv lp_idx / pullback_momentum, every non-collapsed
record with an LP has rows, no collapsed record has rows.

Reads the CURRENT `artifacts/debug` (run after a replay). Written for step 4 (2026-10-26; was a scratchpad script):
0 problems on the compared and the save replay. Reusable as a no-change check of the per-LP CSVs (e.g. step 5).

    PYTHONIOENCODING=utf-8 python engine_v2/plans/qa_chart_edits_inputs/wvmi_step4_check.py PRED_JSON
"""
import json, sys
import pandas as pd

P = json.load(open(sys.argv[1]))
D = "artifacts/debug/NZD_USD_H1_2025-11-15_2026-01-20_sd-1_eps0p0001_rk2-5"
FILES = {"H1": (D + "_wvmi_lps.csv", D + "_wvmi.csv"),
         "counter": (D + "_M15_counter_wvmi_lps.csv", D + "_M15_counter_wvmi.csv"),
         "confluence": (D + "_M15_confluence_wvmi_lps.csv", D + "_M15_confluence_wvmi.csv")}

bad = 0
for lens, (lps_p, w_p) in FILES.items():
    lps = pd.read_csv(lps_p)
    w = pd.read_csv(w_p)
    pred = [r for rec in P for r in rec["rows"] if r["lens"] == lens]
    print(f"{lens}: rows {len(lps)} (predicted {len(pred)}) columns {list(lps.columns)}" if lens == "H1"
          else f"{lens}: rows {len(lps)} (predicted {len(pred)})")
    if len(lps) != len(pred):
        bad += 1
    for (_, a), b in zip(lps.iterrows(), pred):
        key_a = (int(a.lp_idx), int(a.as_of_idx), a.lp_status, a.lp_status_reason,
                 None if pd.isna(a.lp_status_idx) else int(a.lp_status_idx),
                 None if pd.isna(a.relative_dir) else a.relative_dir, a.zone_side,
                 None if pd.isna(a.sub_id) else int(a.sub_id))
        key_b = (b["lp_idx"], b["as_of_idx"], b["lp_status"], b["lp_status_reason"], b["lp_status_idx"],
                 b["relative_dir"], b["side"], b["sub_id"])
        if key_a != key_b or abs(a.pullback_momentum - b["pm"]) > 1e-12 or abs(a.breakout_momentum - b["bm"]) > 1e-12:
            bad += 1
            print("  MISMATCH", key_a, key_b, a.pullback_momentum, b["pm"])
    # cross-checks against _wvmi.csv
    idk = ["bos_structure_id", "bos_cycle_id"] + ([] if lens == "H1" else ["sub_id"])
    for _, r in w.iterrows():
        m = lps
        for k in idk:
            m = m[m[k] == r[k]]
        if r.cycle_collapsed:
            if len(m):
                bad += 1; print("  collapsed record has rows", dict(r[idk]))
            continue
        if pd.isna(r.lp_idx):
            if len(m):
                bad += 1; print("  no-LP record has rows", dict(r[idk]))
            continue
        last = m.iloc[-1]
        if int(last.lp_idx) != int(r.lp_idx) or abs(last.pullback_momentum - r.pullback_momentum) > 1e-12:
            bad += 1; print("  last row != record", dict(r[idk]), last.lp_idx, r.lp_idx)
        if (last.lp_status == "locked") != bool(r.lp_locked):
            bad += 1; print("  locked flag mismatch", dict(r[idk]))
print("problems:", bad)
