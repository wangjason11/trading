"""PLAN_W 9c measurement — the sub CTS KL zones (the CONFIG window, the replay's fetch / floor seeds / lower TFs).

MODE `base` (the code as is), `cts_export` (question 1: every sub projection RETURNS its CTS KL zones too —
`pooled_structure_build.project_to_window` called with `source_kinds=("BOS", "CTS")`) or `derived_ref` (question 2's
variant: a sibling read whose winner is a CTS_CONFIRMED returns the DERIVED CTS KL zone as of the read — `hi` — instead
of the ad-hoc one; the reversal handoff unchanged — its as-of zone = the ad-hoc one on every read of the window). Dumped to OUT.json: each
M15 lens's KL zones, POI zones, IC rows, events, sub records and WVMI records (to show what else moves) — diffed by
`w9c_diff.py`.

In BOTH modes, question 2 — the probe's reference zone: the two pool-path readers that build a sub probe's reference
from a CTS event (`entity_df_mutation._build_sibling_cts_ref_zone_from_pool` — first_counter / subsequent_* — and
`_resolve_reversal_start` — the reversal handoff) are wrapped. Each call is recorded with the ad-hoc zone it returns
(today: `kl_zones=[]`) and, when the winning event is a CTS_CONFIRMED (the only type the derived-zone branch reads),
the DERIVED CTS KL zone of the winner's (sub, cycle): from the winner's sub projected with its CTS zones (the same
`project_to_window`) capped AS OF the read moment (the sibling window's `hi` / the reversal candle — causal) and at its
NATURAL end (the whole geometry — look-ahead), each as the probe's (outer, inner), plus the as-of zone's INIT step.
The H1 main's reversal reference (`structure_engine`, which already passes its derived zones) is recorded too: the
zone it got vs the ad-hoc one.

    PYTHONPATH=. python engine_v2/plans/plan_w_inputs/w9c_measure.py MODE OUT.json   (from the repo root)
"""
import contextlib
import dataclasses
import io
import json
import sys

import engine_v2.run_replay as rr
from engine_v2.multitf import entity_df_mutation as edm
from engine_v2.multitf import pooled_structure_build as pbs
from engine_v2.structure import event_fields as ef
from engine_v2.structure import reference_zone as rz
from engine_v2.structure import structure_engine as se

mode, out_path = sys.argv[1], sys.argv[2]
assert mode in ("base", "cts_export", "derived_ref")
_orig_ptw = pbs.project_to_window
if mode == "cts_export":
    def _ptw_cts(bounded, **kw):
        kw["source_kinds"] = ("BOS", "CTS")
        return _orig_ptw(bounded, **kw)
    pbs.project_to_window = _ptw_cts

_TYPE_ORDER = {"CTS_CONFIRMED": 2, "CTS_UPDATED": 1, "CTS_ESTABLISHED": 0}
REFS = []


def _ref(z):
    return None if z is None else {"outer": round(float(z.outer), 6), "inner": round(float(z.inner), 6),
                                   "side": z.side, "source": z.source, "anchor_idx": int(z.anchor_idx)}


def _derived(sub, cycle_id, cap_abs, probe_direction):
    """The winner's sub's CTS KL zone of `cycle_id`, as the probe's (outer, inner): as of `cap_abs` (entity-absolute;
    None = the natural end) + the zone's INIT step."""
    bounded, sb = sub.geometry
    sb = int(sb)
    with contextlib.redirect_stdout(io.StringIO()):
        down = _orig_ptw(bounded, floor=None, cap=None if cap_abs is None else int(cap_abs) - sb,
                         cap_reason="probe_ref_view", direction=int(sub.direction), timeframe="M15",
                         source_kinds=("BOS", "CTS"), log_prefix=f"REFVIEW{sub.pool_idx}",
                         cross_fib_limits={0: sub.cross_fib_limit.localized(sb)})
    zs = [z for z in down["kl_zones"] if z.source_kind == "CTS" and int(z.meta.get("structure_id", 0)) == 0
          and int(z.meta["cycle_id"]) == int(cycle_id)]
    if not zs:
        return None
    assert len(zs) == 1, f"pool#{sub.pool_idx} c{cycle_id}: {len(zs)} CTS zones"
    z = zs[0]
    init = (z.meta.get("bounds_steps") or [{}])[0]
    out = {"now": _ref(rz._zone_to_reference(z, probe_direction, source="cts_confirmed", anchor_idx=0)),
           "zone": rz._zone_to_reference(z, probe_direction, source="cts_confirmed", anchor_idx=0),
           "n_steps": len(z.meta.get("bounds_steps") or []),
           "base_idx": None if z.meta.get("base_idx") is None else int(z.meta["base_idx"]) + sb,
           "base_pattern": z.meta.get("base_pattern")}
    if init.get("top") is not None:
        zi = dataclasses.replace(z, top=float(init["top"]), bottom=float(init["bottom"]))
        out["init"] = _ref(rz._zone_to_reference(zi, probe_direction, source="cts_confirmed", anchor_idx=0))
    return out


def _winner(cands):
    """The builder's pick: the most recent MOMENT, ties CONFIRMED > UPDATED > ESTABLISHED (`reference_zone.py`)."""
    return max(cands, key=lambda c: (ef.event_moment(c[0]), _TYPE_ORDER[c[0].type]))


_orig_sib = edm._build_sibling_cts_ref_zone_from_pool


def sibling(pool, other_lens, parent_sid, parent_cycle_id, probe_direction, idx_window, m15_df):
    got = _orig_sib(pool, other_lens, parent_sid, parent_cycle_id, probe_direction, idx_window, m15_df)
    lo, hi = int(idx_window[0]), int(idx_window[1])
    cands = []      # (event with its MOMENT shifted to entity-absolute, sub) — the reader's candidate set
    for r in pool.records_for(other_lens, parent_sid, parent_cycle_id):
        if not (r.direction == -int(probe_direction) and not r.is_zero_length and r.start_idx <= hi
                and (r.trigger_end_idx is None or r.trigger_end_idx >= lo)):
            continue
        sub = pool.get_by_pool_idx(r.pool_idx)
        if sub.geometry is None:
            continue
        bounded, sb = sub.geometry
        r_lo = max(lo, int(r.start_idx))
        r_hi = min(hi, int(r.trigger_end_idx) if r.trigger_end_idx is not None else hi)
        for ev in bounded.events:
            if ev.type in rz._CTS_EVENT_TYPES and int(ev.meta.get("structure_id", 0)) == 0 \
                    and r_lo <= ef.event_moment(ev) + int(sb) <= r_hi:
                cands.append((ev, sub))
    rec = {"path": "sibling", "lens_read": other_lens, "parent": [int(parent_sid), int(parent_cycle_id)],
           "probe_direction": int(probe_direction), "window": [lo, hi], "ad_hoc": _ref(got)}
    if cands:
        ev, sub = _winner(cands)
        sb = int(sub.geometry[1])
        rec.update(winner_type=ev.type, winner_moment=int(ef.event_moment(ev)) + sb,
                   winner_anchor=int(ef.cts_anchor_idx(ev)) + sb, winner_cycle=int(ev.meta["cycle_id"]),
                   winner_sub_pool_idx=int(sub.pool_idx), winner_sub_start=int(sub.start_idx))
        assert got is None or rec["winner_anchor"] == got.anchor_idx, (rec, got)
        if ev.type == "CTS_CONFIRMED":
            rec["derived_as_of"] = _derived(sub, int(ev.meta["cycle_id"]), hi, int(probe_direction))
            rec["derived_natural_end"] = _derived(sub, int(ev.meta["cycle_id"]), None, int(probe_direction))
            if mode == "derived_ref" and rec["derived_as_of"] is not None:
                z = rec["derived_as_of"]["zone"]
                got = rz.ReferenceZone(outer=z.outer, inner=z.inner, side=z.side, source="cts_confirmed",
                                       anchor_idx=rec["winner_anchor"])
                rec["returned"] = _ref(got)
    for k in ("derived_as_of", "derived_natural_end"):
        if rec.get(k):
            rec[k].pop("zone", None)
    REFS.append(rec)
    return got


_orig_rev = edm._resolve_reversal_start


def reversal(pool, sub, reversal_idx, probe_direction, m15_df, **kw):
    bounded, sb = sub.geometry
    sb = int(sb)
    got = rz.build_reference_zone_from_cts_event(bounded.events, [], bounded.df, sid=0,
                                                 probe_direction=int(probe_direction), idx_window=None)
    rec = {"path": "reversal", "sub_pool_idx": int(sub.pool_idx), "sub_start": int(sub.start_idx),
           "reversal_idx": int(reversal_idx), "probe_direction": int(probe_direction), "ad_hoc": _ref(got)}
    cands = [(ev, sub) for ev in bounded.events
             if ev.type in rz._CTS_EVENT_TYPES and int(ev.meta.get("structure_id", 0)) == 0]
    if cands:
        ev, _ = _winner(cands)
        rec.update(winner_type=ev.type, winner_moment=int(ef.event_moment(ev)) + sb,
                   winner_anchor=int(ef.cts_anchor_idx(ev)) + sb, winner_cycle=int(ev.meta["cycle_id"]))
        if ev.type == "CTS_CONFIRMED":
            rec["derived_as_of"] = _derived(sub, int(ev.meta["cycle_id"]), int(reversal_idx), int(probe_direction))
            rec["derived_natural_end"] = _derived(sub, int(ev.meta["cycle_id"]), None, int(probe_direction))
    for k in ("derived_as_of", "derived_natural_end"):
        if rec.get(k):
            rec[k].pop("zone", None)
    REFS.append(rec)
    return _orig_rev(pool, sub, reversal_idx, probe_direction, m15_df, **kw)


edm._build_sibling_cts_ref_zone_from_pool = sibling
edm._resolve_reversal_start = reversal

_orig_se_build = se.build_reference_zone_from_cts_event


def main_ref(events, kl_zones, df, sid, probe_direction, *, idx_window=None):
    got = _orig_se_build(events, kl_zones, df, sid, probe_direction, idx_window=idx_window)
    adhoc = _orig_se_build(events, [], df, sid, probe_direction, idx_window=idx_window)
    REFS.append({"path": "main_reversal", "sid": int(sid), "probe_direction": int(probe_direction),
                 "got (derived when CONFIRMED)": _ref(got), "ad_hoc": _ref(adhoc),
                 "n_cts_zones_passed": sum(1 for z in kl_zones if z.source_kind == "CTS"), "n_frame": len(df)})
    return got


se.build_reference_zone_from_cts_event = main_ref

df, effective_start, _ = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
seeds = rr.fetch_floor_seeds(rr.CONFIG.pair, rr.CONFIG.timeframe, df["time"].iloc[0], rr.CONFIG.lower_timeframes)
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes, floor_seeds=seeds)


def _j(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if dataclasses.is_dataclass(v):
        return _j(dataclasses.asdict(v))
    return str(v)


def frame(d):
    a = d.attrs
    return {"kl_zones": [_j({"kind": z.source_kind, "side": z.side, "top": z.top, "bottom": z.bottom,
                             "start_time": str(z.start_time), "end_time": str(z.end_time), "meta": z.meta})
                         for z in a.get("kl_zones", []) or []],
            "poi_zones": [_j({"ic_idx": z.ic_idx, "top": z.top, "bottom": z.bottom, "meta": z.meta})
                          for z in a.get("poi_zones", []) or []],
            "ic_candidates": [_j(r) for r in a.get("ic_candidates", []) or []],
            "events": [_j({"type": e.type, "idx": e.idx, "meta": e.meta}) for e in a.get("events", []) or []],
            "sids": [_j(s) for s in a.get("sids", []) or []],
            "wvmi": [_j(r) for r in a.get("wvmi", []) or []],
            "wave_candles": [_j(w) for w in a.get("wave_candles", []) or []]}


reg = res.meta["registry"]
dump = {"mode": mode, "refs": REFS}
for path in ("H1.main >> M15.counter", "H1.main >> M15.confluence"):
    dump[path] = frame(reg.get(path).df)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dump, f)
print("wrote", out_path, len(REFS), "reference reads")
