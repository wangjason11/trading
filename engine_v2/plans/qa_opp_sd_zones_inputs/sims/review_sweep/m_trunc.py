"""Cold review of PLAN_R §5 (read-only): is a sub zone, read at its usable-from moment, the same as in the natural-end
view when the FRAME is truncated at that moment (no candle after the moment exists)? BOS zone at BOS_CONFIRMED,
POI at CTS_CONFIRMED. Also the KL base-pattern indices vs the BOS moment. Writes nothing to the repo."""
import sys, os, contextlib, io
from types import SimpleNamespace
sys.argv = [sys.argv[0]]
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
from engine_v2.multitf.pooled_structure_build import project_to_window
from engine_v2.structure import event_fields as ef

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
with contextlib.redirect_stdout(io.StringIO()):
    res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
pool = res.meta["sub_pool"]

def zones(down):
    kl = {int(z.meta["cycle_id"]): (round(float(z.meta["inner"]), 6), z.meta.get("anchor_idx"), z.meta.get("base_idx"),
                                    z.meta.get("base_pattern"))
          for z in down["kl_zones"] if z.source_kind == "BOS"}
    poi = {}
    for p in down["poi_zones"]:
        poi.setdefault(int(p.meta["cycle_id"]), []).append((round(float(p.top), 6), round(float(p.bottom), 6), int(p.ic_idx)))
    return kl, poi

for sub in pool.all():
    if sub.geometry is None:
        continue
    bounded, sb = sub.geometry
    sb = int(sb)
    cfl = {0: sub.cross_fib_limit.localized(sb)}
    d = int(sub.direction)
    with contextlib.redirect_stdout(io.StringIO()):
        vkl, vpoi = zones(project_to_window(bounded, floor=None, cap=None, cap_reason=None, direction=d,
                                            timeframe="M15", cross_fib_limits=cfl))
    mom = {}
    for e in bounded.events:
        if e.type in ("BOS_CONFIRMED", "CTS_CONFIRMED"):
            mom[(int(e.meta.get("cycle_id", 0)), e.type)] = ef.event_moment(e)
    for (c, typ), m in sorted(mom.items()):
        tdf = bounded.df.iloc[: m + 1].copy()
        tdf.attrs = dict(bounded.df.attrs)
        tb = SimpleNamespace(df=tdf, events=bounded.events)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tkl, tpoi = zones(project_to_window(tb, floor=None, cap=m, cap_reason="probe", direction=d,
                                                    timeframe="M15", cross_fib_limits=cfl))
        except Exception as exc:
            print(f"[trunc] pool#{sub.pool_idx} c{c} {typ}@{m + sb}: ERROR {exc!r}"[:300])
            continue
        if typ == "BOS_CONFIRMED":
            a, b = vkl.get(c), tkl.get(c)
            same = (a and a[0]) == (b and b[0])
            anc = a[1] if a else None
            print(f"[trunc] pool#{sub.pool_idx} c{c} BOS@{m + sb} view={a and a[0]} trunc={b and b[0]} "
                  f"{'SAME' if same else 'DIFF'} anchor={anc + sb if anc is not None else None} "
                  f"base={a[2] + sb if a and a[2] is not None else None} pattern={a and a[3]} "
                  f"anchor+5-moment={(anc + 5 - m) if anc is not None else None}")
        else:
            a, b = vpoi.get(c), tpoi.get(c)
            print(f"[trunc] pool#{sub.pool_idx} c{c} POI@{m + sb} view={a} trunc={b} {'SAME' if a == b else 'DIFF'}")
