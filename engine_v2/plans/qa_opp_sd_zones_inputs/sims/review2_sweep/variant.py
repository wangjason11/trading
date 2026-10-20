"""Run the pipeline (M15) with one surviving mutant monkeypatched in-process; print the H1 triggers + pool summary.
Variant from env RV2_VARIANT: A (view cap not localized), B (view without cross-fib limit), D (record start max),
E (refresh before close)."""
import contextlib
import io
import os
import sys
import traceback

sys.argv = [sys.argv[0]]
V = os.environ["RV2_VARIANT"]
import engine_v2.run_replay as rr
from engine_v2.config import CONFIG
import engine_v2.pipeline.orchestrator as orch
import engine_v2.multitf.pooled_structure_build as psb
import engine_v2.multitf.proximity_overlay as po

_ptw = psb.project_to_window


def _ptw_mut(bounded, **kw):
    if kw.get("cap_reason") == "proximity_view":
        if V == "A" and kw.get("cap") is not None:
            # undo the localization: the caller passed cap_abs - sb; the mutant passes cap_abs
            kw["cap"] = kw["cap"] + int(_CUR_SB[0])
        if V == "B":
            kw["cross_fib_limits"] = None
    return _ptw(bounded, **kw)


_CUR_SB = [0]
_orig_bsv = po.build_sub_zone_view


def _bsv(sub, project, log=print):
    _CUR_SB[0] = int(sub.geometry[1])
    return _orig_bsv(sub, project, log)


if V in ("A", "B"):
    psb.project_to_window = _ptw_mut
    po.build_sub_zone_view = _bsv
if V == "D":
    def _rs(self, sub, key, known):
        starts = [r.start_idx for lens in (po.LENS_CONFLUENCE, po.LENS_COUNTER)
                  for r in self.pool.records_for(lens, key[0], key[1])
                  if r.pool_idx == sub.pool_idx and not r.is_zero_length and r.start_idx <= known]
        return max(starts) if starts else None
    po.ProximityOverlay._record_start = _rs
if V == "E":
    _orig_on_tick = po.ProximityOverlay.on_tick

    def _on_tick(self, payload, idx):
        key, h = payload
        self._refresh(key, h, self.loh(h - 1))      # an extra refresh BEFORE the close (the mutant's order)
        return _orig_on_tick(self, payload, idx)
    po.ProximityOverlay.on_tick = _on_tick

df, eff, _ = rr.fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
    start=rr._floor_to_day_start(CONFIG.start), end=CONFIG.end, lookback_days=183, min_history=50,
    extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)
except Exception:
    print("VARIANT", V, "CRASHED:")
    print(traceback.format_exc()[-1500:])
    sys.exit(0)
zpt = res.df.attrs["zone_proximity_triggers"]
print("VARIANT", V)
print("zpt:", {k: [(t.idx, t.direction, t.zone_kind, round(float(t.trigger_inner), 5), t.meta.get("prox_ref_sub_id"),
                    t.meta.get("prox_ref_usable_idx")) for t in v] for k, v in zpt.items()})
pool = res.meta["sub_pool"]
print("pool:", sorted((s.direction, s.starting_idx, s.start_idx, s.end_idx, s.end_reason, len(s.records)) for s in pool.all()))
ov = res.meta["proximity_overlay"]
print("passes:", ov.pass_log)
print("views:", {(pool.get_by_pool_idx(p).direction, pool.get_by_pool_idx(p).starting_idx):
                 [(z.cycle_id, z.kind, z.inner, z.usable_at, z.outer_steps) for z in zs] for p, zs in ov._views.items()})
