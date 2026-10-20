"""Shadow run for PLAN_R step 2 / 0a: today's pipeline with (0,1)'s proximity list replaced by the round-4 rules'
chain (748 sd:POI, 820 opp_sd on sub_3 c2 POI 0.58096, 872 sd:BOS 0.57494); (1,2) and every other cycle as S_Q.
argv[1] == "fix0a": also bound the sibling-CTS fallback window's lo to the parent cycle's M15 floor."""
import sys, time
sys.argv, VARIANT = [sys.argv[0]], (sys.argv[1] if len(sys.argv) > 1 else "base")
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
import engine_v2.multitf.entity_df_mutation as edm
from engine_v2.zones.zone_proximity import ZoneProximityTrigger, check_zone_proximity as _orig

def patched(**kw):
    out = _orig(**kw)
    lst = out[(0, 1)]
    assert [t.idx for t in lst] == [748], [t.idx for t in lst]
    mk = lambda d, i, inner, k: ZoneProximityTrigger(structure_id=0, cycle_id=1, direction=d, idx=i,
        trigger_inner=inner, zone_kind=k, proximity_pips=8, pip_size=0.0001, timeframe="H1", meta={"shadow": True})
    out[(0, 1)] = [lst[0], mk("opp_sd", 820, 0.58096, "POI"), mk("sd", 872, 0.57494, "BOS")]
    print("[shadow] (0,1) triggers:", [(t.idx, t.direction, t.zone_kind, t.trigger_inner) for t in out[(0, 1)]])
    if VARIANT == "full":
        l12 = out[(1, 2)]
        assert [t.idx for t in l12] == [935, 954, 1035], [t.idx for t in l12]
        mk2 = lambda d, i, inner, k: ZoneProximityTrigger(structure_id=1, cycle_id=2, direction=d, idx=i,
            trigger_inner=inner, zone_kind=k, proximity_pips=8, pip_size=0.0001, timeframe="H1", meta={"shadow": True})
        out[(1, 2)] = [l12[0], mk2("opp_sd", 954, 0.57306, "POI"), l12[2],
                       mk2("opp_sd", 1045, 0.57728, "BOS"), mk2("sd", 1046, 0.57786, "POI")]
        print("[shadow] (1,2) triggers:", [(t.idx, t.direction, t.zone_kind, t.trigger_inner) for t in out[(1, 2)]])
    return out
orch.check_zone_proximity = patched

if VARIANT == "fix0a":
    FLOOR = {(0, 0): 463, (0, 1): 2611, (1, 2): 3611}   # S_Q records' parent_floor_idx (M15)
    _orig_win = edm._sibling_cts_idx_window
    def win(trigger, parent_df, m15_df, hi):
        lo, hi2 = _orig_win(trigger, parent_df, m15_df, hi)
        fl = FLOOR[(int(trigger.parent_sid), int(trigger.parent_cycle_id))]
        print(f"[shadow] window {trigger.use_case} ({trigger.parent_sid},{trigger.parent_cycle_id}) lo {lo} -> {max(lo, fl)}")
        return max(lo, fl), hi2
    edm._sibling_cts_idx_window = win

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
t0 = time.perf_counter()
res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
print(f"[shadow] variant={VARIANT} pipeline {time.perf_counter()-t0:.1f}s")

print("[dump] keys", [k for k in res.meta if "result" in k], {k: [r.meta.get("sub_id") for r in (res.meta.get(k) or []) if hasattr(r, "meta")] for k in res.meta if "result" in k})
for name in ("lower_tf_results", "first_confluence_results"):
    for r in (res.meta.get(name) or []):
        sid = r.meta.get("sub_id"); sb = int(r.meta.get("slice_begin", 0))
        if sid not in (4, 5, 9, 10):
            continue
        evs = [(e.type, int(e.idx) + sb, e.meta.get("cycle_id"), e.meta.get("confirmed_at"), round(float(e.price), 5) if e.price is not None else None)
               for e in r.events if e.type in ("BOS_CONFIRMED", "CTS_CONFIRMED", "CTS_ESTABLISHED", "REVERSAL_CONFIRMED")]
        print(f"[dump] {name} sub_id={sid} slice_begin={sb} window=[{r.meta.get('start_idx')},{r.meta.get('end_idx')}] events={evs}")
        for z in r.kl_zones:
            print(f"[dump]   KL {z.source_kind} c{z.meta.get('cycle_id')} inner={z.meta.get('inner')} outer={z.meta.get('outer')} conf={z.meta.get('confirmed_idx')}")
        for z in r.poi_zones:
            print(f"[dump]   POI c{z.meta.get("cycle_id")} top={z.top} bottom={z.bottom} ic={z.ic_idx}")
