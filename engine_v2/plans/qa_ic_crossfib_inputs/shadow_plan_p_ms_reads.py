"""Read-only SHADOW for PLAN_P (2026-10-13; plan review lens A finding 4): how often does MS's in-flight POI read
(`poi_inner_for_fib`, the sd-proximity CTS confirmation's POI inner) take a Step 1A (80–100 fallback) IC? The drawn
picks are in the IC CSV; MS's reads run `select_ic` on the AS-OF representative fib at each CTS_ESTABLISHED /
CTS_UPDATED and are exported nowhere. No engine edit: the wrapper re-runs `select_ic` with the read's own inputs and
logs the band. Run on the PLAN_P code (the working tree).

    python <this> [OUT_LOG]      (repo root; ~1 min)
"""
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.structure.structure_engine as se  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402

LINES, STATS, SEEN = [], Counter(), set()
_real = se.poi_inner_for_fib


def _read(df, fib_state, fill_threshold, *, tracker, evaluated_at):
    inner = _real(df, fib_state, fill_threshold, tracker=tracker, evaluated_at=evaluated_at)
    if fib_state is None:
        return inner
    ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))
    sel = pz.select_ic(df, fib_state, pz.POIConfig(fill_threshold=fill_threshold), ic_reference=ref)
    tf = df.attrs.get("timeframe")
    STATS[f"{tf} reads"] += 1
    if sel.ic_zone_band == pz.IC_ZONE_FALLBACK:
        STATS[f"{tf} reads Step 1A ran"] += 1
        if sel.pick is not None:
            STATS[f"{tf} reads Step 1A pick"] += 1
            key = (tf, id(tracker), int(fib_state.structure_id), int(fib_state.cycle_id), int(fib_state.bos_idx),
                   int(sel.pick.ic_idx))
            if key not in SEEN:
                SEEN.add(key)
                t = str(df.loc[int(sel.pick.ic_idx), "time"])[5:16]
                LINES.append(f"[STEP1A-READ] {tf} sid{fib_state.structure_id} cycle {fib_state.cycle_id} fib "
                             f"{fib_state.bos_idx}..{fib_state.cts_idx} at {evaluated_at}: IC {sel.pick.ic_idx} ({t}) "
                             f"inner {inner}")
    return inner


se.poi_inner_for_fib = _read

t0 = time.perf_counter()
df, effective_start, _d = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
LINES.append(f"[STATS] {dict(sorted(STATS.items()))}")
LINES.append(f"[done] {time.perf_counter() - t0:.1f} s")
text = "\n".join(LINES)
if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
print(text)
