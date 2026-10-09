"""python f19_detail.py TF OFF LEN START SD — the F19 hit's cycles + fib records with UTC times."""
import contextlib, io, os, sys

import pandas as pd

import engine_v2.structure.event_fields as ef
from engine_v2.structure.structure_engine import _make_market_structure, _pip_size_from_pair
from engine_v2.tests.test_unified_probe import _prepare_df
from engine_v2.tests._cross_fib_limits import main_limits
from engine_v2.zones.cross_cycle_fib import CrossFibLimit
from engine_v2.zones.fib_tracker import FibEventOrderError, FibTracker, FibTrackerConfig

tf, off, length, start, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
src = pd.read_csv(os.path.join(os.environ["TEMP"], "pi", "f14", f"NZD_USD_{tf}_2025.csv"))
sl = src.iloc[off:off + length].reset_index(drop=True)
T = lambda i: str(sl["time"][int(i)])[5:16]
df = _prepare_df(sl[["o", "h", "l", "c"]].to_dict("records"))
ms = _make_market_structure(df, cross_fib_limit=CrossFibLimit(None, "none"), struct_direction=sd, start_idx=start,
                            structure_id=0, timeframe=tf, pip_size=_pip_size_from_pair(df))
with contextlib.redirect_stdout(io.StringIO()):
    try:
        ms.run()
    except FibEventOrderError as e:   # PLAN_I S5: MS's own order tripwire — the run is complete
        print("MS order tripwire:", str(e)[:240])
ev = ms.events
for e in ev:
    if e.type in ("CTS_ESTABLISHED", "BOS_CONFIRMED", "CTS_CONFIRMED") or (e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"):
        a = ef.cts_anchor_idx(e) if e.type in ("CTS_ESTABLISHED", "CTS_CONFIRMED") else (
            ef.bos_anchor_idx(e) if e.type == "BOS_CONFIRMED" else None)
        print(f"{e.type:16s} cyc {e.meta.get('cycle_id')} moment {int(e.idx)} ({T(e.idx)}) anchor {a}"
              f"{'' if a is None else ' (' + T(a) + ')'} price {e.price} {e.meta.get('via', '')}")
trk = FibTracker(FibTrackerConfig(fib_levels=[30.0, 50.0, 61.8, 80.0], fill_threshold=0.70),
                 cross_fib_limits=main_limits(ev))
bos = {}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    for e in sorted(ev, key=ef.processing_order_key):
        k = (int(e.meta.get("structure_id", 0)), int(e.meta.get("cycle_id", 0)))
        if e.type == "BOS_CONFIRMED":
            bos[k] = (ef.bos_anchor_idx(e), e.price)
        elif e.type == "CTS_ESTABLISHED":
            trk.on_cts_established(e, df, bos[k][0], bos[k][1])
        elif e.type == "CTS_UPDATED":
            trk.on_cts_updated(e, df)
        elif e.type == "CTS_CONFIRMED":
            trk.on_cts_confirmed(e)
        elif e.type == "CTS_THRESHOLD_UPDATED":
            trk.on_cts_threshold_updated(e, df)
    trk._finalize_lifecycle_fields(ev)
print("\n".join(l for l in buf.getvalue().splitlines() if "NO FIB" in l or "ACTIVATED" in l or "LOCKED" in l
                or "DEACTIVATED" in l)[:3000])
for f in trk.get_all_fibs():
    print(f"fib sid {f.structure_id} cyc {f.cycle_id} BOS {f.bos_idx} ({T(f.bos_idx)}) {f.bos_price} -> CTS {f.cts_idx} "
          f"({T(f.cts_idx)}) {f.cts_price} active={f.active} locked={f.locked} life=[{f.start_idx}, {f.end_idx}] "
          f"{f.end_reason} status={f.status} start={T(f.start_idx) if f.start_idx is not None else None} "
          f"end={T(f.end_idx) if f.end_idx is not None else None} via={f.meta.get('via')}")
print("last candle", len(sl) - 1, T(len(sl) - 1))
