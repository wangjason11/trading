"""Review: run the H1-only pipeline, capture check_zone_proximity's inputs + output, then simulate PLAN_R step 1
(parent ladders, signed half-up gap, counters, pass test) independently. Prints only."""
import os, math, pickle
import pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__))
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
from engine_v2.zones import zone_proximity as zp
from engine_v2.structure import event_fields as ef

CAP = {}
_orig = orch.check_zone_proximity
def cap(**kw):
    out = _orig(**kw)
    CAP.update(kw); CAP["out"] = out
    return out
orch.check_zone_proximity = cap

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
res = orch.run_pipeline(df, lower_timeframes=())
d = CAP["df"]; evs = CAP["sorted_events"]; kl = CAP["kl_zones"]; poi = CAP["poi_zones"]; pip = CAP["pip_size"]
print("pip", pip, "tf", CAP["timeframe"])
print("TODAY triggers:", {k: [(t.idx, t.direction, t.zone_kind, t.trigger_inner) for t in v] for k, v in sorted(CAP["out"].items())})

# scan windows, as in check_zone_proximity
cts_conf = {}
for ev in evs:
    if ev.type == "CTS_CONFIRMED":
        cts_conf[(ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))] = ev
bosc = {}
for ev in evs:
    if ev.type == "BOS_CONFIRMED":
        bosc[(ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))] = ef.event_moment(ev)
rev = zp.compute_reversal_idx_by_sid(evs)
print("reversal by sid", rev)
cycles = {}
for key, cev in sorted(cts_conf.items()):
    sid, cyc = key; sd = int(cev.meta.get("struct_direction", 0))
    s0 = ef.event_moment(cev); e0 = len(d) - 1
    if (sid, cyc + 1) in bosc: e0 = min(e0, bosc[(sid, cyc + 1)] - 1)
    if sid in rev: e0 = min(e0, rev[sid] - 1)
    bz = zp._find_kl_zone_for_cycle(kl, sid, cyc, "BOS"); cz = zp._find_kl_zone_for_cycle(kl, sid, cyc, "CTS")
    pz = [p for p in poi if p.meta.get("structure_id") == sid and p.meta.get("cycle_id") == cyc]
    cycles[key] = dict(sd=sd, lo=s0, hi=e0, bos=bz, cts=cz, pois=pz, method=cev.meta.get("confirmation_method"))
    print(f"cycle {key} sd={sd} scan [{s0},{e0}] cts_method={cev.meta.get('confirmation_method')} "
          f"BOS inner {bz.meta['inner']} outer {bz.meta['outer']} steps {[(s.get('expanded_idx', s.get('base_idx')), s['top'], s['bottom']) for s in bz.meta['bounds_steps']]}")
    print(f"     CTS inner {cz.meta['inner']} outer {cz.meta['outer']} steps {[(s.get('expanded_idx', s.get('base_idx')), s['top'], s['bottom']) for s in cz.meta['bounds_steps']]}")
    for p in pz:
        print(f"     POI top {p.top} bottom {p.bottom} side {p.side} ic {p.ic_idx}")
pickle.dump({k: dict(sd=v["sd"], lo=v["lo"], hi=v["hi"], method=v["method"],
                     bos=(v["bos"].meta["inner"], v["bos"].meta["outer"], v["bos"].side, v["bos"].meta["bounds_steps"]),
                     cts=(v["cts"].meta["inner"], v["cts"].meta["outer"], v["cts"].side, v["cts"].meta["bounds_steps"]),
                     pois=[(p.top, p.bottom, p.side) for p in v["pois"]]) for k, v in cycles.items()},
            open(os.path.join(OUT, "h1_cycles.pkl"), "wb"))
# threshold timeline gaps (Rules 1-3) at scan start
for key, c in cycles.items():
    tl = zp._build_cycle_threshold_timeline(evs, *key)
    rc = rb = None
    for e in tl:
        if ef.event_moment(e) <= c["lo"]:
            rc, rb = zp._apply_threshold_event(e, rc, rb)
    print("threshold gap at scan start", key, None if rc is None or rb is None else round(abs(rc - rb) / pip, 1))
