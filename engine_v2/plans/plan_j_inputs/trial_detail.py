"""PLAN_J: one random-tail trial in detail, through the tree first on PYTHONPATH.

    PYTHONPATH=TREE python trial_detail.py waves|real SEED T [LO HI]

Regenerates the trial (`random_tail_search.wave_trials` / `real_trials` — every draw before the run, so every tree sees
the same candles), runs MS like the search (no reference price; the order tripwire recorded, not raised) and prints
the tripwire, the f14 record, every CTS_ESTABLISHED / BOS_CONFIRMED / CTS_CONFIRMED with its anchors, and — with
LO HI — every event and candle in [LO, HI] (REAL: with the source CSV's UTC time).
"""
import contextlib
import importlib.util
import io
import json
import os
import sys

import engine_v2
import engine_v2.structure.event_fields as ef

tree = os.path.dirname(os.path.dirname(engine_v2.__file__))
spec = importlib.util.spec_from_file_location(
    "rts", os.path.join(tree, "engine_v2", "plans", "plan_e_inputs", "review_scripts", "random_tail_search.py"))
rts = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rts)

stream, seed, t_want = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (None, None)
times = None
if stream == "waves":
    for t, sd, tf, base, rows, nw in rts.wave_trials(seed):
        if t == t_want:
            start = 0
            break
else:
    for t, sd, tf, off, start, length, jit, rows in rts.real_trials(seed):
        if t == t_want:
            import pandas as pd
            d = os.environ.get("RTS_REAL_DIR") or os.path.join(os.environ["TEMP"], "pi", "f14")
            times = pd.read_csv(os.path.join(d, f"NZD_USD_{tf}_2025.csv"))["time"].iloc[off:off + length].tolist()
            print(f"REAL off {off} len {length} start {start} jit {jit}")
            break
T = (lambda i: "") if times is None else (lambda i: " " + str(times[int(i)])[5:16])
df = rts._prepare_df(rows)
ms = rts._make_market_structure(df, cross_fib_limit=rts._NO_P, struct_direction=sd, start_idx=start, structure_id=0,
                                timeframe=tf, pip_size=rts._pip_size_from_pair(df))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rts._ms_run(ms)
print(f"{stream} seed {seed} trial {t_want}: sd {sd} tf {tf} n {len(rows)}; MS tripwire: {getattr(ms, '_rts_tripwire', None)}")
print("f14:", json.dumps(rts._f14(ms.events)))
pj = [l for l in buf.getvalue().splitlines() if l.startswith("[PJ") or l.startswith("WARNING [ms] cts_frozen")]
for l in pj:
    print("  log:", l)
for e in ms.events:
    if e.type in ("CTS_ESTABLISHED", "BOS_CONFIRMED", "CTS_CONFIRMED"):
        a = ef.cts_anchor_idx(e) if e.type != "BOS_CONFIRMED" else ef.bos_anchor_idx(e)
        print(f"  {e.type:16s} cycle {e.meta.get('cycle_id')} moment {int(e.idx)}{T(e.idx)} anchor {a}{T(a)} "
              f"price {e.price} {e.meta.get('via', '') or e.meta.get('confirmation_method', '')}")
if lo is not None:
    for e in ms.events:
        if lo <= int(e.idx) <= hi:
            extra = {k: v for k, v in e.meta.items() if k in ("cts_anchor_idx", "bos_anchor_idx", "confirm_idx",
                                                             "start_idx", "via", "to", "cycle_id", "cts_frozen",
                                                             "pattern_anchor_idx", "confirmation_method")}
            print(f"  {e.type:22s} idx={int(e.idx):4d}{T(e.idx)} price={e.price} {extra}")
    for i in range(lo, min(hi, len(rows) - 1) + 1):
        r = rows[i]
        print(f"  candle {i:4d}{T(i)} o={r['o']} h={r['h']} l={r['l']} c={r['c']} {df['candle_type'].iloc[i]} "
              f"dir={int(df['direction'].iloc[i])}")
