"""Cold review of PLAN_R §5 (read-only): measures on TODAY's pipeline (S_Q code):
 (1) H1->M15 tick alignment: LOH(h-1)+1 == first M15 of hour h; hours with no M15 bars.
 (2) per pooled sub: the "natural-end view" project_to_window(cap=None) vs the real projection (floor=start, cap=end)
     -- BOS zone inners, POI inners per cycle; zones in the view whose moment is past the sub's end.
 (3) POI pick stability over the reading moment: for every cycle POI, the pick at caps from its CTS_CONFIRMED
     moment to the edge (full capped projections, every 4 M15).
 (4) cost: seconds per project_to_window(cap=None) and per capped projection.
Writes nothing to the repo."""
import sys, os, time, pickle
from collections import defaultdict
sys.argv = [sys.argv[0]]
OUT = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
import engine_v2.multitf.data_bridge as db
from engine_v2.multitf.entity_df_mutation import _map_parent_idx_to_m15_hour_end as LOH
from engine_v2.multitf.pooled_structure_build import project_to_window
from engine_v2.structure import event_fields as ef

CAP = {}
_orig_prep = db.prepare_lower_tf_data
def _prep(raw):
    m = _orig_prep(raw)
    CAP["m15"] = m
    return m
db.prepare_lower_tf_data = _prep

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
t0 = time.perf_counter()
res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
print(f"[m] pipeline {time.perf_counter()-t0:.1f}s")
m15 = CAP["m15"]
h1 = res.df
pool = res.meta["sub_pool"]
tables = res.meta["parent_tables"]

# ---------- (1) alignment
h1t = pd.to_datetime(h1["time"], utc=True)
m15t = pd.to_datetime(m15["time"], utc=True)
first_of_hour = {}
cnt = defaultdict(int)
hour_of_m15 = m15t.dt.floor("h")
h1_pos = {t: i for i, t in zip(h1.index, h1t)}
for j, ht in zip(m15.index, hour_of_m15):
    i = h1_pos.get(ht)
    if i is None:
        cnt["m15_bar_in_hour_not_in_h1"] += 1
        continue
    first_of_hour.setdefault(i, j)
bad_tick, empty_hours, loh_back = [], [], []
lohs = {}
for h in h1.index:
    lohs[h] = LOH(int(h), h1, m15)
for h in list(h1.index)[1:]:
    if h not in first_of_hour:
        empty_hours.append(h)
        continue
    tick = lohs[h - 1] + 1 if lohs.get(h - 1) is not None else None
    if tick != first_of_hour[h]:
        bad_tick.append((h, tick, first_of_hour[h]))
    if lohs[h] is not None and tick is not None and lohs[h] < tick:
        loh_back.append((h, tick, lohs[h]))
print(f"[align] H1 rows={len(h1)} M15 rows={len(m15)} m15 bars whose hour is not an H1 row={cnt['m15_bar_in_hour_not_in_h1']}")
print(f"[align] H1 hours with NO M15 bar: {len(empty_hours)} {empty_hours[:20]}")
print(f"[align] LOH(h-1)+1 != first M15 of hour h: {len(bad_tick)} {bad_tick[:20]}")
print(f"[align] LOH(h) < LOH(h-1)+1 (FIRE would be queued in the past): {len(loh_back)} {loh_back[:20]}")
print(f"[align] last H1 idx {h1.index[-1]} LOH={lohs[h1.index[-1]]} m15 last idx {m15.index[-1]}; "
      f"last H1 hour bars={sum(1 for x in hour_of_m15 if x == h1t.iloc[-1])}")
# the scan windows' extent in H1 (main triggers) for context
for k, v in res.meta["zone_proximity_triggers"].items():
    print(f"[align] cycle {k} triggers {[ (t.idx, t.direction, t.zone_kind) for t in v]} collapsed={tables.is_collapsed(*k)}")

# ---------- (2)/(3)/(4)
def zones_of(down, sb):
    kl = {}
    for z in down["kl_zones"]:
        if z.source_kind != "BOS":
            continue
        kl[int(z.meta["cycle_id"])] = (round(float(z.meta["inner"]), 6), z.meta.get("confirmed_idx"))
    poi = defaultdict(list)
    for p in down["poi_zones"]:
        poi[int(p.meta["cycle_id"])].append((round(float(p.top), 6), round(float(p.bottom), 6), int(p.ic_idx) + sb,
                                              p.meta.get("ic_zone_band")))
    return kl, dict(poi)

summary = []
for sub in pool.all():
    if sub.geometry is None:
        continue
    bounded, sb = sub.geometry
    sb = int(sb)
    lim = sub.cross_fib_limit
    cfl = {0: lim.localized(sb)}
    d = int(sub.direction)
    evs = bounded.events
    moments = defaultdict(dict)
    for e in evs:
        if e.type in ("BOS_CONFIRMED", "CTS_CONFIRMED", "CTS_ESTABLISHED"):
            c = int(e.meta.get("cycle_id", 0))
            moments[c][e.type] = ef.event_moment(e) + sb
    t1 = time.perf_counter()
    view = project_to_window(bounded, floor=None, cap=None, cap_reason=None, direction=d, timeframe="M15",
                             log_prefix=f"VIEW{sub.pool_idx}", cross_fib_limits=cfl)
    t_view = time.perf_counter() - t1
    vkl, vpoi = zones_of(view, sb)
    line = (f"[sub] pool#{sub.pool_idx} sub_id={sub.sub_id} key=({d},{sub.starting_idx}) start={sub.start_idx} "
            f"end={sub.end_idx}/{sub.end_reason} nat_rev={sub.natural_reversal_idx} edge={sb + len(bounded.df) - 1} "
            f"lim={lim.source} P={lim.price} moves={len(lim.price_moves) if lim.price_moves else 0} view_s={t_view:.2f}")
    print(line)
    print(f"[sub]   moments(abs) {dict(moments)}")
    print(f"[sub]   VIEW kl={vkl} poi={vpoi}")
    if sub.start_idx is not None:
        cap_local = (int(sub.end_idx) - sb) if sub.end_idx is not None else None
        t1 = time.perf_counter()
        real = project_to_window(bounded, floor=int(sub.start_idx) - sb, cap=cap_local, cap_reason=sub.end_reason,
                                 direction=d, timeframe="M15", log_prefix=f"REAL{sub.pool_idx}", cross_fib_limits=cfl)
        t_real = time.perf_counter() - t1
        rkl, rpoi = zones_of(real, sb)
        print(f"[sub]   REAL kl={rkl} poi={rpoi} real_s={t_real:.2f}")
        for c in sorted(set(vkl) | set(rkl)):
            a, b = vkl.get(c), rkl.get(c)
            if (a and a[0]) != (b and b[0]):
                print(f"[diff] pool#{sub.pool_idx} BOS c{c} view={a} real={b}")
        for c in sorted(set(vpoi) | set(rpoi)):
            a = [x[:3] for x in vpoi.get(c, [])]; b = [x[:3] for x in rpoi.get(c, [])]
            if a != b:
                print(f"[diff] pool#{sub.pool_idx} POI c{c} view={vpoi.get(c)} real={rpoi.get(c)}")
        if sub.end_idx is not None:
            for c, mm in sorted(moments.items()):
                for typ, mo in mm.items():
                    if mo > int(sub.end_idx) and typ in ("BOS_CONFIRMED", "CTS_CONFIRMED"):
                        print(f"[leak] pool#{sub.pool_idx} c{c} {typ} moment {mo} > sub end {sub.end_idx} "
                              f"(in VIEW: bos={c in vkl} poi={c in vpoi}; in REAL: bos={c in rkl} poi={c in rpoi})")
    # (3) POI pick vs reading moment: caps from each cycle's CTS_CONFIRMED moment, every 4 M15, to the edge
    edge = sb + len(bounded.df) - 1
    cts_conf = sorted((c, mm["CTS_CONFIRMED"]) for c, mm in moments.items() if "CTS_CONFIRMED" in mm)
    if not cts_conf:
        continue
    first = min(m for _, m in cts_conf)
    caps = list(range(first, edge + 1, 4)) + [edge]
    hist = defaultdict(list)  # cycle -> [(cap, poi tuple)]
    t1 = time.perf_counter()
    import contextlib, io
    for cap in caps:
        with contextlib.redirect_stdout(io.StringIO()):
            dn = project_to_window(bounded, floor=None, cap=cap - sb, cap_reason="probe", direction=d,
                                   timeframe="M15", log_prefix="", cross_fib_limits=cfl)
        _, pp = zones_of(dn, sb)
        for c, mo in cts_conf:
            if cap >= mo:
                val = tuple(x[:3] for x in pp.get(c, []))
                if not hist[c] or hist[c][-1][1] != val:
                    hist[c].append((cap, val))
    t_caps = time.perf_counter() - t1
    print(f"[pick] pool#{sub.pool_idx} {len(caps)} capped projections in {t_caps:.1f}s ({t_caps/len(caps):.3f}s each)")
    for c, h in sorted(hist.items()):
        tag = "STABLE" if len(h) == 1 else "CHANGES"
        print(f"[pick] pool#{sub.pool_idx} c{c} CTS_CONFIRMED={dict(cts_conf)[c]} {tag}: {h}")
        summary.append((sub.pool_idx, c, tag, h))
print("[summary] POI pick changes after CTS_CONFIRMED:", [(p, c) for p, c, tag, _ in summary if tag == "CHANGES"])
pickle.dump({"h1": h1[["time", "o", "h", "l", "c"]].copy(), "m15": m15[["time", "o", "h", "l", "c"]].copy()},
            open(os.path.join(OUT, "frames_sweep.pkl"), "wb"))
