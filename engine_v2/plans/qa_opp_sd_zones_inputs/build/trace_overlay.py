"""The step-2 firing chain on the engine: per (0,1) / (1,2) candle, the reference changes, gaps, passes, gates and
triggers — in m4_sim's log shape, for a line-by-line comparison with build/m4_run.log. Also dumps every used / overlay
sub zone (for the embedded real-collaborator test). Read-only."""
import contextlib
import io
import pickle
import sys

sys.argv = [sys.argv[0]]
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
ov = res.meta["proximity_overlay"]
pool = res.meta["sub_pool"]


def lab(r):
    if r is None:
        return "None"
    if r.source == "parent":
        return f"P.{r.zone_kind}({r.inner})"
    s = pool.get_by_pool_idx(r.sub_pool_idx)
    return f"({s.direction}, {s.starting_idx})c{r.sub_cycle_id}.{r.zone_kind}({r.inner})"


passes = {(k, h): keys for k, h, keys in ov.pass_log}
for key in sorted(ov.scans):
    scan = ov.scans[key]
    trig = {t.idx: t for t in scan.triggers}
    print(f"=== {key} scan [{scan.scan_start}, {scan.scan_end}] ===")
    prev = None
    for k, h, sd, opp, gap, narrow, gate in ov.ref_log:
        if k != key:
            continue
        state = (lab(sd), lab(opp), gap, narrow)
        if state != prev:
            print(f"  {h}: sd={lab(sd)} opp={lab(opp)} gap={gap}{' NARROW' if narrow else ''}")
            prev = state
        if gate is not None:
            print(f"  {h}: {scan.expected_dir if False else ''}BLOCKED gate={gate}")
        if h in trig:
            t = trig[h]
            print(f"  {h}: TRIGGER {t.direction} {t.zone_kind} {t.trigger_inner} meta={t.meta}")
        if (key, h) in passes:
            print(f"  {h}: close passes {passes[(key, h)]}")
    print(f"  final locked={scan.locked} counters={scan.counters}")

views = {}
for sub in pool.all():
    if sub.pool_idx in ov._views:
        views[(sub.direction, sub.starting_idx)] = [(z.cycle_id, z.kind, z.inner, z.usable_at, z.outer_steps)
                                                    for z in ov._views[sub.pool_idx]]
print("=== the views (key -> zones) ===")
for k, zs in sorted(views.items()):
    print(" ", k, zs)
pickle.dump(views, open(sys.argv[0].replace("trace_overlay.py", "views.pkl"), "wb"))
