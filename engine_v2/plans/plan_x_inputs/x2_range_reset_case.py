"""PLAN_X X2, sweep 1.8: one sub's events around a range that is RESET before the moment its RANGE_STARTED is confirmed,
in EMISSION order (the CSV's row order) with each event's moment, + the candles.

    python engine_v2/plans/plan_x_inputs/x2_range_reset_case.py SAVE_DIR LENS SUB_ID LO HI
"""
import ast
import glob
import sys

import pandas as pd

save, lens, sub_id, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ev = pd.read_csv(glob.glob(f"{save}/*_M15_{lens}_structure_events.csv")[0])
ev["m"] = ev["meta"].map(ast.literal_eval)
ev = ev[ev["m"].map(lambda m: m.get("sub_id")) == sub_id]
keep = ["confirmed_at", "range_anchor_idx", "cycle_id", "reason", "from", "to", "via", "pat", "confirmation_method",
        "cts_anchor_idx", "bos_anchor_idx", "lo", "hi"]
print(f"== {lens} sub {sub_id}: events with idx or moment in [{lo}, {hi}], emission order ==")
for row_no, r in ev.iterrows():
    m = r["m"]
    mom = m.get("confirmed_at", r["idx"])
    if lo <= int(r["idx"]) <= hi or (mom is not None and lo <= int(mom) <= hi):
        print(f"  row {row_no:5d} idx {int(r['idx']):5d} {r['type']:<22} " + ", ".join(f"{k}={m[k]}" for k in keep if k in m))
c = pd.read_csv(glob.glob(f"{save}/*_M15_{lens}_final.csv")[0]) if glob.glob(f"{save}/*_M15_{lens}_final.csv") else None
if c is None:
    print("(no M15 final CSV in the save)")
