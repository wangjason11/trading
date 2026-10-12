"""PLAN_M — run the pipeline of a patched TREE on its CONFIG window WITHOUT the exports (no CSV beyond the raw
frame, no chart): the census only needs the `[census]` lines the patch prints.

    cd WORKDIR && python REPO/engine_v2/plans/plan_m_inputs/census_run.py TREE [--export] > run.log 2>&1

`--export` (PLAN_M landing 2) also prints one `[census] export {json}` line per mirrored sub fib record: the limit
keys the engine exported (as of the sub's end) and its IC reference's source / break — `census_l2_report.py` reads them.

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
if "--export" in sys.argv:
    # PLAN_M landing 2: what the engine EXPORTED per sub fib record (the mirrored lens records — the CSV's rows):
    # the limit keys (as of the sub's end) + the IC reference's break (entity-absolute after the mirror).
    import json
    for lens, ldf in sorted((res.meta.get("sub_lens_dfs") or {}).items()):
        for f in ldf.attrs.get("fib_states", []):
            m = f.meta
            if m.get("sub_id") is None:
                continue
            print("[census] export " + json.dumps({
                "lens": lens, "sub_id": int(m["sub_id"]), "cycle": int(f.cycle_id), "version": m.get("version"),
                "cross": bool(m.get("cross_cycle")), "via": m.get("via"),
                "P": m.get("cross_fib_limit_price"), "src": m.get("cross_fib_limit_source"),
                "M": m.get("cross_fib_limit_cycle"), "P_start": m.get("cross_fib_limit_price_at_start"),
                "moves": m.get("cross_fib_limit_move_count"), "ic_src": m.get("ic_ref_source"),
                "ic_brk": m.get("ic_ref_break_confirmed_idx"),
            }, default=str))
print(f"[census_run] done in {time.perf_counter() - t0:.1f} s")
