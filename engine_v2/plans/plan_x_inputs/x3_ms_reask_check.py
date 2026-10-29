"""PLAN_X X3 — did MarketStructure's EMBEDDED tracker run the re-ask in the variant (it logs nothing: quiet)? Run
in-process inside a VARIANT scratch tree (counting wrappers only — every wrapper returns the original's value):

    cd TREE && PYTHONIOENCODING=utf-8 PYTHONPATH=. python <repo>/engine_v2/plans/plan_x_inputs/x3_ms_reask_check.py START END OUT.txt

Counts, per tracker kind (MS-embedded = quiet / downstream): the re-asks run and CHANGED by kind (`reask_stats` deltas
per `flush_reasks` call); MS's POI refreshes triggered by a re-ask (`_flush_fib_reasks` → `_refresh_poi_inners_for_cycle`)
and those whose inner differs from the snapshot before it — each listed (frame, sid, cycle, candle, before -> after).
"""
import dataclasses
import io
import sys
from collections import Counter
from contextlib import redirect_stdout
from datetime import datetime, timezone

import engine_v2.run_replay as rr
from engine_v2.structure import market_structure as msm
from engine_v2.zones import fib_tracker as ftm

start, end, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
assert hasattr(ftm.FibTracker, "flush_reasks"), "not a variant tree"
C = Counter()
REFRESH: list = []

_orig_flush = ftm.FibTracker.flush_reasks


def flush(self, upto, df):
    before = Counter(self.reask_stats)
    r = _orig_flush(self, upto, df)
    who = "ms_embedded" if self._quiet else "downstream"
    for k, v in (Counter(self.reask_stats) - before).items():
        if k.startswith(("ran_", "CHANGED_", "dropped")):
            C[f"{who}.{k}"] += v
    return r


ftm.FibTracker.flush_reasks = flush

_orig_ms_flush = msm.MarketStructure._flush_fib_reasks
_orig_refresh = msm.MarketStructure._refresh_poi_inners_for_cycle
_in_flush = [False]


def ms_flush(self, i):
    _in_flush[0] = True
    try:
        return _orig_ms_flush(self, i)
    finally:
        _in_flush[0] = False


def refresh(self, moment):
    if not _in_flush[0]:
        return _orig_refresh(self, moment)
    st = self.state
    before = list(st.poi_inners_for_cycle or [])
    r = _orig_refresh(self, moment)
    after = list(st.poi_inners_for_cycle or [])
    C["ms_poi_refresh_by_reask"] += 1
    if before != after:
        C["ms_poi_refresh_inner_changed"] += 1
        REFRESH.append(f"frame-from {str(self.df['time'].iloc[0])[:16]} n={len(self.df)} sid={st.structure_id} "
                       f"cycle={st.cts_cycle_id} candle={int(moment)}: {before} -> {after}")
    return r


msm.MarketStructure._flush_fib_reasks = ms_flush
msm.MarketStructure._refresh_poi_inners_for_cycle = refresh


def main() -> None:
    rr.CONFIG = dataclasses.replace(
        rr.CONFIG,
        start=datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc),
        end=datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc),
    )
    with redirect_stdout(io.StringIO()):
        df, _, _ = rr.fetch_history_with_auto_extend(
            pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
            end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
        )
        df.attrs["pair"] = rr.CONFIG.pair
        seeds = rr.fetch_floor_seeds(rr.CONFIG.pair, rr.CONFIG.timeframe, df["time"].iloc[0], rr.CONFIG.lower_timeframes)
        rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)
    out = [f"window {start} -> {end}: H1 {len(df)} candles", f"counts: {dict(sorted(C.items()))}",
           "MS POI refreshes by a re-ask whose inner changed:"] + [f"  {l}" for l in REFRESH]
    open(out_path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
