"""PLAN_M — run the pipeline of a patched TREE on its CONFIG window WITHOUT the exports (no CSV beyond the raw
frame, no chart): the census only needs the `[census]` lines the patch prints.

    cd WORKDIR && python REPO/engine_v2/plans/plan_m_inputs/census_run.py TREE > run.log 2>&1

WORKDIR holds a copy of `oanda.cfg` (a SHORT %TEMP% dir — MAX_PATH). TREE/engine_v2 is put first on sys.path
(asserted).
"""
import os
import sys
import time

tree = os.path.abspath(sys.argv[1])
sys.path.insert(0, tree)
import engine_v2.run_replay as rr  # noqa: E402
assert os.path.normcase(rr.__file__).startswith(os.path.normcase(tree)), rr.__file__

os.makedirs("artifacts/debug", exist_ok=True)
t0 = time.perf_counter()
df, effective_start, _decision = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
print(f"[census_run] window {effective_start} -> {rr.CONFIG.end}: {len(df)} H1 candles")
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
print(f"[census_run] done in {time.perf_counter() - t0:.1f} s")
