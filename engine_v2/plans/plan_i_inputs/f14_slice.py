"""PLAN_I §9.2 F16 (session 4, 2026-10-08): the real-candle example (M15 5515 173 11 -1 / M15 5528 470 0 -1).
Run MS on an EXACT real slice (no jitter): python f14_slice.py TF OFF LENGTH START SD [LO HI]
Prints the F14 summary, every CTS_CONFIRMED whose CTS anchor lies after its idx, and the events in [LO, HI] with
UTC times (slice idx i -> the CSV row OFF + i)."""
import importlib.util, json, os, sys

import pandas as pd

import engine_v2.structure.event_fields as ef

spec = importlib.util.spec_from_file_location(
    "rts", "engine_v2/plans/plan_e_inputs/review_scripts/random_tail_search.py")
rts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rts)

tf, off, length, start, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
lo, hi = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (None, None)
src = pd.read_csv(os.path.join(os.environ["TEMP"], "pi", "f14", f"NZD_USD_{tf}_2025.csv"))
sl = src.iloc[off:off + length].reset_index(drop=True)
rows = sl[["o", "h", "l", "c"]].to_dict("records")
df = rts._prepare_df(rows)
ms = rts._make_market_structure(df, cross_fib_limit=rts._NO_P, struct_direction=sd, start_idx=start, structure_id=0,
                                timeframe=tf, pip_size=rts._pip_size_from_pair(df))
rts._ms_run(ms)   # PLAN_I S5: MS's own order tripwire (recorded, not raised) — the run is complete
print("MS order tripwire:", getattr(ms, "_rts_tripwire", None))
f = rts._f14(ms.events)
print("f14:", json.dumps(f))
for e in ms.events:
    if e.type == "CTS_CONFIRMED" and ef.cts_anchor_idx(e) > int(e.idx):
        print("CONFIRMED anchor after idx:", int(e.idx), sl["time"][int(e.idx)], "anchor", ef.cts_anchor_idx(e),
              sl["time"][ef.cts_anchor_idx(e)], e.price, e.meta.get("via"), "cycle", e.meta["cycle_id"])
if lo is not None:
    for e in ms.events:
        if lo <= int(e.idx) <= hi:
            extra = {k: v for k, v in e.meta.items() if k in ("cts_anchor_idx", "confirm_idx", "start_idx", "via", "to",
                                                             "cycle_id", "effective_idx", "pattern_anchor_idx")}
            print(f"  {e.type:22s} idx={int(e.idx):4d} {sl['time'][int(e.idx)]}  price={e.price}  {extra}")
    for i in range(lo, hi + 1):
        r = sl.iloc[i]
        print(f"  candle {i:4d} {r['time']}  o={r['o']} h={r['h']} l={r['l']} c={r['c']}  "
              f"{df['candle_type'].iloc[i]}")
