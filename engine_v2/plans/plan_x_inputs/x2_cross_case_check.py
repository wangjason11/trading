"""PLAN_X X2, sweep 2.1: a window's sub — its cycle events and fib records — to judge a cross-fib exposure (a target
cycle's own gap that forms one candle after the decision). Run on a window replay's artifacts/debug folder.

    python engine_v2/plans/plan_x_inputs/x2_cross_case_check.py DEBUG_DIR LENS SUB_ID LO HI
"""
import ast
import glob
import sys

import pandas as pd

d, lens, sub_id, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
subs = pd.read_csv(glob.glob(f"{d}/*_M15_{lens}_subs.csv")[0])
print(subs[["sub_id", "direction", "starting_idx", "start_idx", "end_idx", "end_reason"]].to_string(index=False))
ev = pd.read_csv(glob.glob(f"{d}/*_M15_{lens}_structure_events.csv")[0])
ev["m"] = ev["meta"].map(ast.literal_eval)
ev = ev[ev["m"].map(lambda m: m.get("sub_id")) == sub_id]
print(f"== sub {sub_id}: CTS_ESTABLISHED / CTS_CONFIRMED / BOS_CONFIRMED / CTS_THRESHOLD_UPDATED in [{lo}, {hi}]")
for _, r in ev.iterrows():
    if r["type"] in ("CTS_ESTABLISHED", "CTS_CONFIRMED", "BOS_CONFIRMED", "CTS_THRESHOLD_UPDATED") and lo <= int(r["idx"]) <= hi:
        m = r["m"]
        print(f"  {int(r['idx'])} {r['type']} cycle={m.get('cycle_id')} method={m.get('confirmation_method')} price={r['price']}")
fl = pd.read_csv(glob.glob(f"{d}/*_M15_{lens}_fib_lifecycle.csv")[0])
print("== the lens fib records (the LIVE window only)")
print(fl[["structure_id", "cycle_id", "version", "start_idx", "end_idx", "end_reason", "active", "locked", "bos_idx",
          "cts_idx"]].to_string(index=False))
