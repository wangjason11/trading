"""Read-only: the IC-identification inputs of ONE sub's drawn POI records (select_ic with classify_all=True inside
the sub's projection), entity-absolute.   python <this> SUB_ID CYCLES(comma) LO HI   (repo root)"""
import sys
import time

sys.path.insert(0, ".")
TARGET = int(sys.argv[1])
CYCLES = {int(x) for x in sys.argv[2].split(",")}
LO, HI = int(sys.argv[3]), int(sys.argv[4])
import numpy as np  # noqa: E402
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402
from engine_v2.features.candles_v2 import agreed_direction  # noqa: E402
from engine_v2.patterns.imbalance import iter_unfilled_imbalances  # noqa: E402

CTX = []
_real_rsp = edm.render_sub_projection


def _rsp(sub, *a, **k):
    CTX.append((int(sub.sub_id), int(sub.geometry[1])))
    try:
        return _real_rsp(sub, *a, **k)
    finally:
        CTX.pop()


edm.render_sub_projection = _rsp
DUMPED = {"df": False}
_real_sel = pz.select_ic


def _sel(df, fib_state, config, *, ic_reference, classify_all=False):
    out = _real_sel(df, fib_state, config, ic_reference=ic_reference, classify_all=classify_all)
    c = CTX[-1] if CTX else None
    if c is None or c[0] != TARGET or not classify_all or int(fib_state.cycle_id) not in CYCLES:
        return out
    b = c[1]
    sd = int(fib_state.struct_direction)
    f = fib_state.fib
    lo, hi = pz.fib_zone_bounds(f.anchor_high, f.anchor_low, sd, config)
    print(f"[IC] sub {TARGET} cycle {fib_state.cycle_id} cross={fib_state.meta.get('cross_cycle')} "
          f"bos {fib_state.bos_idx + b} @{fib_state.bos_price} cts {fib_state.cts_idx + b} @{fib_state.cts_price} "
          f"anchor_high {f.anchor_high} anchor_low {f.anchor_low} zone [{lo:.6f}, {hi:.6f}] "
          f"ref price={ic_reference.price} src={ic_reference.source} "
          f"break={None if ic_reference.break_pattern_anchor_idx is None else ic_reference.break_pattern_anchor_idx + b}"
          f" cap={config.ip_leg_cap_by_tf[df.attrs['timeframe']]} imm_window={config.immediate_imbalance_candles}")
    print(f"[IC]   pick={None if out.pick is None else out.pick.ic_idx + b} ip_outcome={out.ip_outcome} "
          f"tiebreak={out.tiebreak} ranking={out.ranking} no_ic={out.no_ic_reason} "
          f"candidates={[x.ic_idx + b for x in out.ic_candidates]}")
    for cc in out.ic_candidates:
        p = cc.pattern
        print(f"[IC]   cand {cc.ic_idx + b}: rank={p.ip_rank} reason={p.reason} counts={p.ip_leg_candle_counts} "
              f"detail={p.detail} imm={None if cc.immediate_imbalance_first_c2_idx is None else cc.immediate_imbalance_first_c2_idx + b}")
    # unfilled sd imbalances after each candidate, fill judged through the CTS anchor (condition 3's test)
    for cc in out.ic_candidates:
        ins = list(iter_unfilled_imbalances(df, cc.ic_idx + 1, fib_state.cts_idx, fib_state.cts_idx, 0.70,
                                            direction=sd, evaluated_at=None))
        print(f"[IC]   cond3 unfilled after {cc.ic_idx + b}: {[(i.start_idx + b, i.end_idx + b) for i in ins]}")
    if not DUMPED["df"]:
        DUMPED["df"] = True
        ag = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
        print(f"[ICDF] idx time o h l c candle_type direction vol_dir agreed prior_maru_max_len_as0 body_pips")
        for i in range(LO - b, HI - b + 1):
            r = df.iloc[i]
            print(f"[ICDF] {i + b} {r['time'] if 'time' in df.columns else ''} {r['o']:.5f} {r['h']:.5f} {r['l']:.5f} "
                  f"{r['c']:.5f} {r['candle_type']} {int(r['direction'])} {int(r['vol_dir'])} {int(ag[i])} "
                  f"{float(r['prior_maru_max_len_as0']):.5f} {abs(r['c'] - r['o']) / 0.0001:.1f}")
        for inst in df.attrs.get("imbalances", []):
            if inst.direction == sd and LO - b <= inst.start_idx <= HI - b:
                print(f"[ICIMB] c2 {inst.start_idx + b}..{inst.end_idx + b} gap [{inst.gap_bottom:.5f}, "
                      f"{inst.gap_top:.5f}] filled_by_2842={inst.is_filled(df, 2842 - b)} "
                      f"filled_by_3047={inst.is_filled(df, 3047 - b)}")
    return out


pz.select_ic = _sel

t0 = time.perf_counter()
df, effective_start, _d = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
print(f"[trace] done in {time.perf_counter() - t0:.1f} s")
