"""Shadow for the WVMI pass (2026-09-30): per rendered sub (swept or not) and the main, what each GATE option yields.

Import before the replay (the `runpy` recipe, like `wvmi_shadow.py`); behaviour-neutral (it only re-runs a private
`WVMITracker` on the same inputs). Writes WVMI_GATE_SHADOW_OUT (default: the temp dir). Per sub: lenses, start / end
(entity-absolute), every cycle's lifecycle `(start, end, reason)` from `compute_cycle_lifecycle` with the sub's floor /
cap (the KL / POI / fib table), every CTS_CONFIRMED / BOS_CONFIRMED moment, and the records of:
  cur   — what the current sweep produced (only swept subs; LP search to the frame end = the sub end, inclusive);
  none  — no gate: every CTS_CONFIRMED of every rendered sub, LP search as now (frame end, inclusive);
  life  — no gate, lifecycle-bounded: the LP search stops at the cycle's `end - 1` (the main's rule since
          2026-09-28) — record existence unchanged (a pre-start / collapsed-cycle record is flagged, not dropped).
Main: the same three over the H1 events (cur = the first-sd-prox gate).
"""
import json
import os
import atexit
import tempfile

import engine_v2.pipeline.orchestrator as orch
from engine_v2.structure import event_fields as ef
from engine_v2.zones.structure_lifecycle import compute_cycle_lifecycle, compute_reversal_idx_by_sid
from engine_v2.zones.wvmi import WVMITracker

OUT = os.environ.get("WVMI_GATE_SHADOW_OUT") or os.path.join(tempfile.gettempdir(), "wvmi_gate_shadow.json")
subs, main = [], []


def _run(events, df, wave_candles, kl_zones, cycle_end_by_key=None):
    t = WVMITracker(structure_path_id="shadow", cycle_end_by_key=cycle_end_by_key)
    evs = sorted(events, key=ef.processing_order_key)
    for ev in evs:
        if ev.type == "CTS_CONFIRMED":
            t.on_cts_confirmed(ev, df, wave_candles, kl_zones)
    for ev in evs:
        if ev.type == "BOS_CONFIRMED":
            t.on_bos_confirmed(ev, df, wave_candles)
    t.update_temporary_lp(df, kl_zones)
    return t.get_records()


def _recs(recs, sb):
    def a(v):
        return None if v is None else int(v) + sb
    return [dict(sid=r.bos_structure_id, cyc=r.bos_cycle_id, fb=a(r.fb_idx), lb=a(r.lb_idx), fp=a(r.fp_idx),
                 lp=a(r.lp_idx), locked=r.lp_locked, by=r.locked_by_cycle_id,
                 bo=None if r.breakout_momentum is None else round(r.breakout_momentum, 4),
                 pb=None if r.pullback_momentum is None else round(r.pullback_momentum, 4))
            for r in sorted(recs, key=lambda r: (r.bos_structure_id, r.bos_cycle_id))]


def _cycles(events, floor, cap, reason, sb):
    table = compute_cycle_lifecycle(events, compute_reversal_idx_by_sid(events), floor, cap, reason)
    moments = {}
    for ev in events:
        if ev.type in ("CTS_ESTABLISHED", "CTS_CONFIRMED", "BOS_CONFIRMED"):
            k = (int(ev.meta["structure_id"]), int(ev.meta["cycle_id"]))
            moments.setdefault(k, {})[ev.type] = int(ef.event_moment(ev)) + sb
    rows = []
    for k in sorted(set(table) | set(moments)):
        s, e, why = table.get(k, (None, None, None))
        rows.append(dict(sid=k[0], cyc=k[1], start=None if s is None else s + sb, end=None if e is None else e + sb,
                         end_reason=why, **{t.lower(): v for t, v in moments.get(k, {}).items()}))
    return table, rows


_orig_assign = orch._assign_sub_wvmi_per_sub


def assign(sub_results, streams_by_lens, **kw):
    out = _orig_assign(sub_results, streams_by_lens, **kw)
    for res in sub_results:
        m = res.meta
        sb = int(m["slice_begin"])
        floor = int(m["start_idx"]) - sb
        cap = None if m["end_idx"] is None else int(m["end_idx"]) - sb
        table, rows = _cycles(res.events, floor, cap, m["end_reason"], sb)
        ends = {k: e for k, (_s, e, _r) in table.items() if e is not None}
        subs.append(dict(
            sub_id=m["sub_id"], lenses=list(m["lenses"]), started_by=m["started_by"], start=m["start_idx"],
            end=m["end_idx"], m15_end=m["m15_end_idx"], end_reason=m["end_reason"], frame_last=len(res.df) - 1 + sb,
            cycles=rows,
            cur=_recs(res.wvmi_records or [], sb),
            none=_recs(_run(res.events, res.df, res.wave_candles, res.kl_zones), sb),
            life=_recs(_run(res.events, res.df, res.wave_candles, res.kl_zones, ends), sb)))
    return out


orch._assign_sub_wvmi_per_sub = assign

_orig_down = orch._run_downstream_pipeline


def down(df, events, *a, **kw):
    out = _orig_down(df, events, *a, **kw)
    if not kw.get("skip_wvmi") and kw.get("structure_path_id", "H1.main") == "H1.main":
        floor, cap, why = kw.get("lifecycle_floor"), kw.get("lifecycle_cap"), kw.get("cap_reason", "lifecycle_end")
        table, rows = _cycles(events, floor, cap, why, 0)
        ends = {k: e for k, (_s, e, _r) in table.items() if e is not None}
        main.append(dict(df_last=len(df) - 1, cycles=rows, cur=_recs(out["wvmi_records"], 0),
                         none=_recs(_run(events, df, out["wave_candles"], out["kl_zones"]), 0),
                         life=_recs(_run(events, df, out["wave_candles"], out["kl_zones"], ends), 0)))
    return out


orch._run_downstream_pipeline = down
atexit.register(lambda: json.dump(dict(subs=subs, main=main), open(OUT, "w"), indent=1, default=str))
