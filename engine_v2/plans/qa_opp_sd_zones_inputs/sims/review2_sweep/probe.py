"""Read-only landing-review probe (PLAN_R step 2, the sweep integration). Runs run_pipeline (M15) once in-process with
instrumentation; writes nothing in the repo."""
import contextlib
import io
import sys
import time
from collections import defaultdict

sys.argv = [sys.argv[0]]
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
import engine_v2.multitf.proximity_overlay as po
import engine_v2.multitf.lifecycle_sweep as ls
from engine_v2.zones import zone_proximity as zp

viol = []
reads = defaultdict(int)

# (1) causality of every read: the sub's processed end / start never past the tick's known moment
_orig_usable = po.ProximityOverlay._usable


def _usable(self, sub, known):
    reads["usable"] += 1
    if sub.end_idx is not None and int(sub.end_idx) > known:
        viol.append(("end_after_known", sub.pool_idx, sub.end_idx, known))
    if sub.start_idx is None or int(sub.start_idx) > known:
        viol.append(("start_after_known", sub.pool_idx, sub.start_idx, known))
    return _orig_usable(self, sub, known)


po.ProximityOverlay._usable = _usable

# the sweep state at each tick: every heap entry is at (idx, phase) >= (t, 5); FIRE pushes land in the future
_orig_tick = ls._Sweep._phase5_proximity_tick
tick_state = []


def _tick(self, payload, t):
    if self._heap:
        i, p = self._heap[0][0], self._heap[0][1]
        if (i, p) < (t, ls._PHASE_PROXIMITY):
            viol.append(("heap_before_tick", (i, p), t))
    # every record with start <= t is non-pending: its RECORD_START ran (active or ended or zero-length)
    for r in self.pool.all_records():
        if r.start_idx <= t and not r.is_zero_length and r.seq not in self._active and r.trigger_end_idx is None:
            viol.append(("started_record_not_active", r.identity, r.start_idx, t))
    return _orig_tick(self, payload, t)


ls._Sweep._phase5_proximity_tick = _tick

# (2) every overlay rung ever set
all_rungs = {}
_orig_set = zp.CycleProximityScan.set_overlay


def _set_overlay(self, side, rungs):
    for r in rungs:
        all_rungs.setdefault(r.key, ((self.sid, self.cycle_id), side, r))
    return _orig_set(self, side, rungs)


zp.CycleProximityScan.set_overlay = _set_overlay

# (3) the view's projections, kept for the outer comparison
proj = {}
_orig_q = po._quiet_project
qtime = [0.0, 0]


def _q(project, sub, cap, log):
    t0 = time.perf_counter()
    d = _orig_q(project, sub, cap, log)
    qtime[0] += time.perf_counter() - t0
    qtime[1] += 1
    proj[(sub.pool_idx, cap)] = d
    return d


po._quiet_project = _q

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
log = buf.getvalue()
open(sys.argv[0].replace("probe.py", "probe_run.log"), "w", encoding="utf-8").write(log)
ov = res.meta["proximity_overlay"]
pool = res.meta["sub_pool"]
print("violations:", viol[:20], len(viol))
print("reads:", dict(reads), "view projections:", qtime[1], f"{qtime[0]:.1f}s")

# (4) every overlay rung (not only the used ones) vs the sub's REAL projection
results = {int(r.meta["sub_id"]): r for r in (res.meta["lower_tf_results"] + res.meta["first_confluence_results"])}
bad = []
for key, (cyc, side, r) in sorted(all_rungs.items(), key=lambda kv: str(kv[0])):
    sub = pool.get_by_pool_idx(r.sub_pool_idx)
    rp = results.get(sub.sub_id)
    if rp is None:
        bad.append(("no projection", key))
        continue
    if r.zone_kind == "BOS":
        inn = [float(z.meta["inner"]) for z in rp.kl_zones if z.source_kind == "BOS" and int(z.meta["cycle_id"]) == r.sub_cycle_id]
    else:
        inn = [float(p.top) if p.side == "buy" else float(p.bottom) for p in rp.poi_zones if int(p.meta["cycle_id"]) == r.sub_cycle_id]
    if inn != [r.inner]:
        bad.append(("inner", cyc, side, key, r.inner, inn))
print("overlay rungs:", len(all_rungs), "mismatching the real projection:", bad)

# (5) the BOS outer steps: the natural-end INIT outer vs the capped projection's; the view's outer vs the real
#     projection's at every M15 moment up to the sub's end (or the edge)
out_bad = []
for pidx, zones in ov._views.items():
    sub = pool.get_by_pool_idx(pidx)
    sb = int(sub.geometry[1])
    rp = results.get(sub.sub_id)
    for z in zones:
        if z.kind != "BOS":
            continue
        capped = [k for k in proj[(pidx, z.usable_at)]["kl_zones"] if k.source_kind == "BOS" and int(k.meta["cycle_id"]) == z.cycle_id][0]
        edge = "bottom" if capped.side == "buy" else "top"
        cs = capped.meta["bounds_steps"]
        cap_outer_at_m = float(cs[-1][edge])
        if abs(cap_outer_at_m - z.outer_at(z.usable_at)) > 1e-12 or abs(float(cs[0][edge]) - z.outer_steps[0][1]) > 1e-12:
            out_bad.append(("capped_vs_view_at_m", pidx, z.cycle_id, cap_outer_at_m, z.outer_at(z.usable_at), [(s["event"], s.get("expanded_idx"), s[edge]) for s in cs]))
        early = [s for s in z.outer_steps[1:] if s[0] < z.usable_at]
        if early:
            out_bad.append(("expansion_before_usable", pidx, z.cycle_id, z.usable_at, early))
        if rp is None:
            continue
        rk = [k for k in rp.kl_zones if k.source_kind == "BOS" and int(k.meta["cycle_id"]) == z.cycle_id]
        if not rk:
            if sub.end_idx is None or z.usable_at <= sub.end_idx:
                out_bad.append(("zone_missing_in_real", pidx, z.cycle_id, z.usable_at, sub.start_idx, sub.end_idx))
            continue
        rs = rk[0].meta["bounds_steps"]
        hi = int(sub.end_idx) if sub.end_idx is not None else sb + len(sub.geometry[0].df) - 1
        for t in range(z.usable_at, hi + 1):
            real = float(rs[0][edge])
            for s in rs[1:]:
                if int(s["expanded_idx"]) + sb <= t:
                    real = float(s[edge])
            if abs(real - z.outer_at(t)) > 1e-12:
                out_bad.append(("real_vs_view_outer", pidx, z.cycle_id, t, real, z.outer_at(t)))
                break
print("outer checks:", out_bad)

# (6) the POI outer of the view vs the real projection
poi_bad = []
for pidx, zones in ov._views.items():
    sub = pool.get_by_pool_idx(pidx)
    rp = results.get(sub.sub_id)
    if rp is None:
        continue
    for z in zones:
        if z.kind != "POI" or (sub.end_idx is not None and z.usable_at > sub.end_idx):
            continue
        pz = [p for p in rp.poi_zones if int(p.meta["cycle_id"]) == z.cycle_id]
        got = [((float(p.top), float(p.bottom)) if p.side == "buy" else (float(p.bottom), float(p.top))) for p in pz]
        if got != [(z.inner, z.outer_steps[0][1])]:
            poi_bad.append((pidx, z.cycle_id, z.usable_at, (z.inner, z.outer_steps[0][1]), got, sub.start_idx, sub.end_idx))
print("poi checks (view vs real, zones usable by the end):", poi_bad)

# (7) the final lists' consistency: detectors re-run on the final zpt vs attrs
from engine_v2.multitf.subsequent_confluence_trigger import detect_subsequent_confluence_triggers
from engine_v2.multitf.subsequent_counter_trigger import detect_subsequent_counter_triggers
zpt = res.df.attrs["zone_proximity_triggers"]
ev = res.meta.get("structure_events") or res.df.attrs["structure_events"]
from engine_v2.structure import event_fields as ef
sorted_events = sorted(ev, key=lambda e: (e.idx, e.type))
v3 = detect_subsequent_confluence_triggers(sorted_events, zpt, res.df, parent_tf="H1")
v4 = detect_subsequent_counter_triggers(sorted_events, zpt, res.df, parent_tf="H1")
a3 = res.df.attrs["subsequent_confluence_triggers"]
a4 = res.df.attrs["subsequent_counter_triggers"]
print("var3 re-derived", [(t.parent_sid, t.parent_cycle_id, t.trigger_event_idx, t.input_idx) for t in v3])
print("var3 attrs     ", [(t.parent_sid, t.parent_cycle_id, t.trigger_event_idx, t.input_idx) for t in a3])
print("var4 re-derived", [(t.parent_sid, t.parent_cycle_id, t.trigger_event_idx, t.input_idx) for t in v4])
print("var4 attrs     ", [(t.parent_sid, t.parent_cycle_id, t.trigger_event_idx, t.input_idx) for t in a4])
print("meta is attrs:", res.meta["zone_proximity_triggers"] is zpt, res.meta["subsequent_confluence_triggers"] is a3)
print("zpt:", {k: [(t.idx, t.direction, t.zone_kind, t.meta.get("prox_ref_sub_id")) for t in v] for k, v in zpt.items()})
for line in log.splitlines():
    if "[prox_view]" in line or "prox_overlay" in line or "no_m15_hour" in line:
        print("LOG:", line)
