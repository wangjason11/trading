"""F16 on the exact real slice, the current FibTracker: processing vs emission order.
python f16_orders.py TF OFF LEN START SD"""
import contextlib, io, os, sys

import pandas as pd

import engine_v2.structure.event_fields as ef
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.tests._cross_fib_limits import main_limits, unreached_for
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.zones.fib_tracker import FibEventOrderError, FibTracker, FibTrackerConfig

tf, off, length, start, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
src = pd.read_csv(os.path.join(os.environ["TEMP"], "pi", "f14", f"NZD_USD_{tf}_2025.csv"))
sl = src.iloc[off:off + length].reset_index(drop=True)
T = lambda i: str(sl["time"][int(i)])[5:16]
df = _prepare_df(sl[["o", "h", "l", "c"]].to_dict("records"))
ms = _make_market_structure(df, cycle0_proximity_blocked=True, cross_fib_limit=CrossFibLimit(None, "none"), struct_direction=sd, start_idx=start,
                            structure_id=0, timeframe=tf, pip_size=_pip_size_from_pair(df))
with contextlib.redirect_stdout(io.StringIO()):
    try:
        ms.run()
    except FibEventOrderError as e:   # PLAN_I S5: MS's own order tripwire — the run is complete
        print("MS order tripwire:", str(e)[:240])
ev = ms.events


def feed(order, limits):
    trk = FibTracker(FibTrackerConfig(fib_levels=[30.0, 50.0, 61.8, 80.0], fill_threshold=0.70),
                     cross_fib_limits=limits)
    bos = {e.meta["cycle_id"]: (ef.bos_anchor_idx(e), e.price) for e in ev if e.type == "BOS_CONFIRMED"}
    seq = []
    with contextlib.redirect_stdout(io.StringIO()):
        for e in order:
            c = int(e.meta.get("cycle_id", 0))
            if e.type == "CTS_ESTABLISHED":
                trk.on_cts_established(e, df, *bos[c])
            elif e.type == "CTS_UPDATED":
                trk.on_cts_updated(e, df)
            elif e.type == "CTS_CONFIRMED":
                trk.on_cts_confirmed(e)
            elif e.type == "CTS_THRESHOLD_UPDATED":
                trk.on_cts_threshold_updated(e, df)
            else:
                continue
            seq.append((e.type, c, ef.stamped_idx(e)))
    return trk, seq


for name, order in (("processing", sorted(ev, key=ef.processing_order_key)), ("emission", list(ev))):
    for lname, lim in (("main(no P)", main_limits(ev)), ("unreached", unreached_for(ev))):
        trk, seq = feed(order, lim)
        recs = [(f.cycle_id, f.meta.get("version"), f.cts_idx, T(f.cts_idx), f.cts_price, f.active, f.locked,
                 f.meta.get("locked_at")) for f in trk.get_all_fibs() if f.cycle_id == 0]
        print(f"{name:10s} {lname:11s} cycle-0 fib(s): {recs}")
    print("   fib-handled sequence around 21..25:", [s for s in seq if 21 <= s[2] <= 25])
conf = [e for e in ev if e.type == "CTS_CONFIRMED"][0]
print("CTS_CONFIRMED idx", int(conf.idx), T(conf.idx), "anchor", ef.cts_anchor_idx(conf), T(ef.cts_anchor_idx(conf)),
      "price", conf.price)
