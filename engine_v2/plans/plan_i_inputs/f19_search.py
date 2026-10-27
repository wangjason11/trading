"""PLAN_I F19 search on EXACT real NZD_USD candles (no jitter): a fib that outlives its cycle because the next
cycle got no fib. python f19_search.py SEED N
For each slice: MS from the slice's sd-side extreme in its first 1-29 candles; the downstream tracker fed like the
orchestrator (processing_order_key; BOS per cycle at its BOS_CONFIRMED); finalize with events (no floor / cap);
flag a fib (0, c) whose end_idx > the establishment moment of cycle c+1."""
import json, os, sys

import numpy as np
import pandas as pd

import engine_v2.structure.event_fields as ef
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.tests._cross_fib_limits import main_limits
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.zones.fib_tracker import FibTracker, FibTrackerConfig

seed, n = int(sys.argv[1]), int(sys.argv[2])
D = os.path.join(os.environ["TEMP"], "pi", "f14")
SRC = {tf: pd.read_csv(os.path.join(D, f"NZD_USD_{tf}_2025.csv")) for tf in ("H1", "M15")}
rng = np.random.RandomState(seed)
hits = 0
for t in range(n):
    sd = 1 if rng.rand() < 0.5 else -1
    tf = "H1" if rng.rand() < 0.4 else "M15"
    src = SRC[tf]
    length = rng.randint(150, 900)
    off = rng.randint(0, len(src) - length)
    w = rng.randint(1, 30)
    sl = src.iloc[off:off + length].reset_index(drop=True)
    rows = sl[["o", "h", "l", "c"]].to_dict("records")
    ext = [r["l"] for r in rows[:w]] if sd == 1 else [-r["h"] for r in rows[:w]]
    start = int(np.argmin(ext))
    try:
        df = _prepare_df(rows)
        ms = _make_market_structure(df, cycle0_proximity_blocked=True, cross_fib_limit=CrossFibLimit(None, "none"), struct_direction=sd,
                                    start_idx=start, structure_id=0, timeframe=tf, pip_size=_pip_size_from_pair(df))
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            ms.run()   # PLAN_I S5: an F16 / F17 slice raises MS's order tripwire -> an "err" row
        events = ms.events
        est_moment = {int(e.meta["cycle_id"]): int(e.idx) for e in events if e.type == "CTS_ESTABLISHED"}
        trk = FibTracker(FibTrackerConfig(fib_levels=[30.0, 50.0, 61.8, 80.0], fill_threshold=0.70),
                         cross_fib_limits=main_limits(events))
        bos = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for e in sorted(events, key=ef.processing_order_key):
                k = (int(e.meta.get("structure_id", 0)), int(e.meta.get("cycle_id", 0)))
                if e.type == "BOS_CONFIRMED":
                    bos[k] = (ef.bos_anchor_idx(e), e.price)
                elif e.type == "CTS_ESTABLISHED":
                    if k not in bos:
                        raise RuntimeError("F17 skip")
                    trk.on_cts_established(e, df, bos[k][0], bos[k][1])
                elif e.type == "CTS_UPDATED":
                    trk.on_cts_updated(e, df)
                elif e.type == "CTS_CONFIRMED":
                    trk.on_cts_confirmed(e)
                elif e.type == "CTS_THRESHOLD_UPDATED":
                    trk.on_cts_threshold_updated(e, df)
            trk._finalize_lifecycle_fields(events)
    except Exception as x:  # noqa: BLE001
        print(json.dumps({"t": t, "err": f"{type(x).__name__}: {x}"[:200]}), flush=True)
        continue
    fib_cycles = sorted({f.cycle_id for f in trk.get_all_fibs()})
    for f in trk.get_all_fibs():
        c = f.cycle_id
        nxt = est_moment.get(c + 1)
        if nxt is not None and f.end_idx is not None and f.end_idx > nxt:
            hits += 1
            print(json.dumps({"t": t, "tf": tf, "sd": sd, "off": off, "len": length, "start": start, "cycle": c,
                              "end_idx": f.end_idx, "end_reason": f.end_reason, "next_est": nxt,
                              "est": est_moment, "fib_cycles": fib_cycles,
                              "key_meta_version": f.meta.get("version"), "time_off": str(sl["time"][0])}),
                  flush=True)
print(json.dumps({"done": n, "hits": hits}), flush=True)
