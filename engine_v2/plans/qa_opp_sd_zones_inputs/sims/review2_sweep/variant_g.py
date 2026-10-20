"""R-G on the real data: the orchestrator's _make_proximity_triggers with `<= h` for var 3 (patched source)."""
import contextlib, io, sys, traceback, inspect, textwrap, types
sys.argv = [sys.argv[0]]
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
src = inspect.getsource(orch._run_multi_tf_dual)
old = "            if int(v3.trigger_event_idx) == int(h):"
assert src.count(old) == 1
src = src.replace(old, "            if int(v3.trigger_event_idx) <= int(h):")
ns = orch.__dict__
exec(compile(textwrap.dedent(src), orch.__file__, "exec"), ns)
df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
    print("VARIANT G ran:", {k: [t.idx for t in v] for k, v in res.df.attrs["zone_proximity_triggers"].items()},
          len(res.meta["subsequent_confluence_triggers"]))
except Exception:
    print("VARIANT G CRASHED:", traceback.format_exc()[-600:])
