"""As-of temporary LP sequences per WVMI record (the engine's own LP rule replayed candle by candle).

Wraps orchestrator._compute_wvmi_records: for every record, the CTS_CONFIRMED moment c, the BOS_{n+1}
CONFIRMED moment b (lock), the search bound, and for t in [c, end]:
  rule_extreme(t) = _find_temporary_lp(frame, fp, zone, fp+1, min(t, bound))   (the engine rule, as of t)
  rule_latest(t)  = the latest qualified candle in [fp+1, min(t, bound)]
  rule_furthest(t) = the qualified candle closing furthest in the pullback direction in [fp+1, min(t, bound)]
                    (ties to the earlier) - differs from rule_extreme only once a close passes the zone's outer edge
  rule_r4          = the user's round-4 rules (2026-10-26): the FIRST temporary LP by how the CTS confirmed
                    (pullback pattern: the deepest qualified close among the pattern's candles; sd proximity at p:
                    the deepest prior qualified close beyond p's close, else p if qualified, else the first later
                    qualified candle closing beyond p's close), then a new LP at each qualified candle closing
                    further than the current one. The LP search still starts at FP + 1 (as the engine's).
The pullback pattern's candles come from a measurement-only meta key `_m_pat_off` stamped on each pullback
CTS_CONFIRMED (offsets from the moment, so they survive a sub's index remap). With `gate` set (the main), the
records an UNGATED run would add are captured too (`extra=True`). Matched to the lens / main records after the run
for sub_id + absolute coordinates. A record can match several captures (a sub is projected more than once): keep
the capture whose lp_idx / lp_locked / fp_idx / cycle_collapsed equal the saved `_wvmi.csv` row. Written for "WVMI
back on the chart" round 2, 2026-10-25; rule_r4 / extra / end_cand round 4, 2026-10-26 (runs the pipeline, ~1 min;
writes nothing to artifacts/). Step 3 (2026-10-26): `new` / `new_to_bound` = `_rule_new`, the decided rule (rule_r4 +
the fallback + candidates bounded by end - 1; as of the lock, and to end - 1), `lock_lp` = BOS_{n+1}'s last wave candle
(the lock warning), `pat_anchor`; the `extra` branch is dead since step 2 (no `gate` kwarg — every H1 record is a
regular capture). On step 3's code the engine's own LP (`lp_final`) = `new_to_bound`'s last on an unlocked record.

    PYTHONIOENCODING=utf-8 PYTHONPATH=. python engine_v2/plans/qa_chart_edits_inputs/wvmi_asof.py OUT_JSON
"""
import json
import sys

import engine_v2.pipeline.orchestrator as orch
import engine_v2.zones.wvmi as wv
from engine_v2.common.types import PatternEvent
from engine_v2.features.candles_v2 import agreed_direction
from engine_v2.structure.market_structure import MarketStructure

CAP = []
_orig = orch._compute_wvmi_records
_orig_emit = MarketStructure._emit_cts_confirmed_once


def _emit(self, idx, meta=None, confirmation_method="pullback"):
    n = len(self.events)
    _orig_emit(self, idx, meta=meta, confirmation_method=confirmation_method)
    if len(self.events) > n and confirmation_method == "pullback":
        pev = sys._getframe(1).f_locals.get("ev")
        if isinstance(pev, PatternEvent):
            conf = "" if pev.confirmation_idx is None else str(int(idx) - int(pev.confirmation_idx))
            self.events[-1].meta["_m_pat_off"] = (f"{int(idx) - int(pev.start_idx)},{int(idx) - int(pev.end_idx)},"
                                                  f"{conf},{pev.status.name}")


MarketStructure._emit_cts_confirmed_once = _emit


def _closest_to_outer(frame, fp, zone, start, end):
    """The engine's LP rule until step 3 (`wvmi._find_temporary_lp`, removed 2026-10-26): the qualified close nearest
    the BOS zone's `meta["outer"]`, ties to the earlier — kept here so `ext` replays it on any code."""
    best, best_d = None, float("inf")
    for i in range(start, end + 1):
        if i in frame.index and _qualified(frame, fp, i):
            d = abs(float(frame.loc[i, "c"]) - float(zone.meta["outer"]))
            if d < best_d:
                best, best_d = i, d
    return best


def _qualified(frame, fp, i):
    fd = int(frame.loc[fp, "direction"])
    if int(frame.loc[i, "direction"]) != fd:
        return False
    return int(agreed_direction(fd, int(frame.loc[i, "vol_dir"]))) == fd


def _rule_r4(frame, fp, pb_dir, c, cts_meta, last_t):
    """The user's round-4 first-LP rules + the further-close updates: ([(as_of_t, lp)], info)."""
    C = lambda i: float(frame.loc[i, "c"])
    q = lambda i: i >= fp + 1 and _qualified(frame, fp, i)
    further = lambda a, b: a < b if pb_dir == -1 else a > b

    def deepest(idxs):
        best = None
        for i in idxs:
            if best is None or further(C(i), C(best)):
                best = i
        return best

    info = {"method": cts_meta.get("confirmation_method")}
    first = None
    if info["method"] == "sd_zone_proximity":
        p, pc = c, C(c)
        prior = [i for i in range(fp + 1, p) if q(i) and further(C(i), pc)]
        prior_q = [i for i in range(fp + 1, p) if q(i)]
        info.update(p=p, p_qualified=q(p), n_prior_deeper=len(prior), n_prior_qualified=len(prior_q),
                    deepest_prior_qualified=deepest(prior_q))
        if prior:
            first, info["route"] = (p, deepest(prior)), "prior"
        elif q(p):
            first, info["route"] = (p, p), "self"
        else:
            info["route"] = "wait"
            for t in range(p + 1, last_t + 1):
                if q(t) and further(C(t), pc):
                    first = (t, t)
                    break
    else:
        so, eo, co, status = cts_meta["_m_pat_off"].split(",")
        s, e = c - int(so), c - int(eo)
        conf = None if co == "" else c - int(co)
        last = max(e, conf if conf is not None else e)
        pat = list(range(s, last + 1))
        cands = [i for i in pat if q(i)]
        pre = [i for i in range(fp + 1, s) if q(i)]
        info.update(pat_start=s, pat_end=e, pat_conf=conf, pat_status=status, fp_in_pat=s <= fp,
                    n_pat=len(pat), n_pat_qualified=len(cands), pat_qualified=cands)
        if cands:
            first, info["route"] = (c, deepest(cands)), "pattern"
            pre_deeper = [i for i in pre if further(C(i), C(first[1]))]
            info["pre_pattern_deeper"] = deepest(pre_deeper)
        else:
            info["route"] = "no qualified pattern candle"
    seq = []
    if first is not None:
        seq.append(first)
        cur = first[1]
        for t in range(first[0] + 1, last_t + 1):
            if q(t) and further(C(t), C(cur)):
                cur = t
                seq.append((t, t))
    return seq, info


def _rule_new(frame, fp, pb_dir, c, cts_meta, last_t, bound):
    """Step 3's rule as decided (rounds 4-7): rule_r4 + the pattern route's FALLBACK (no qualified pattern candle
    after FP -> the first later qualified candle closing beyond FP's close), the pattern span = [anchor, moment c],
    and every candidate bounded to [FP + 1, bound] (bound = the cycle's end - 1). Written 2026-10-26 (step 3)."""
    C = lambda i: float(frame.loc[i, "c"])
    q = lambda i: fp + 1 <= i <= bound and _qualified(frame, fp, i)
    further = lambda a, b: a < b if pb_dir == -1 else a > b

    def deepest(idxs):
        best = None
        for i in idxs:
            if best is None or further(C(i), C(best)):
                best = i
        return best

    def wait(ref_close):
        for t in range(c + 1, last_t + 1):
            if q(t) and further(C(t), ref_close):
                return (t, t)
        return None

    if cts_meta.get("confirmation_method") == "sd_zone_proximity":
        prior = [i for i in range(fp + 1, c) if q(i) and further(C(i), C(c))]
        if prior:
            first, route = (c, deepest(prior)), "prior"
        elif q(c):
            first, route = (c, c), "self"
        else:
            first, route = wait(C(c)), "wait"
    else:
        so = int(cts_meta["_m_pat_off"].split(",")[0])
        cands = [i for i in range(c - so, c + 1) if q(i)]
        if cands:
            first, route = (c, deepest(cands)), "pattern"
        else:
            first, route = wait(C(fp)), "fallback"
    seq = []
    if first is not None:
        seq.append(first)
        cur = first[1]
        for t in range(first[0] + 1, last_t + 1):
            if q(t) and further(C(t), C(cur)):
                cur = t
                seq.append((t, t))
    return seq, route


def _capture(recs, df, events, sorted_events, kl_zones, kw, extra, wave_candle_results=()):
    cap = kw.get("lifecycle_cap")
    frame = df.iloc[:cap + 1] if cap is not None else df
    life = orch.compute_cycle_lifecycle(events, orch.compute_reversal_idx_by_sid(events), kw.get("lifecycle_floor"),
                                        cap, kw.get("cap_reason"))
    for r in recs:
        key = (r.bos_structure_id, r.bos_cycle_id)
        cts = [e for e in sorted_events if e.type == "CTS_CONFIRMED" and (e.meta.get("structure_id", 0), e.meta.get("cycle_id", 0)) == key]
        bos = [e for e in sorted_events if e.type == "BOS_CONFIRMED" and (e.meta.get("structure_id", 0), e.meta.get("cycle_id", 0)) == (key[0], key[1] + 1)]
        c = int(cts[0].idx)
        via = cts[0].meta.get("via")
        b = int(bos[0].idx) if bos else None
        start, end, reason = life[key]
        bound = (len(frame) - 1) if end is None else min(len(frame) - 1, int(end) - 1)
        zone = wv._find_bos_zone(kl_zones, *key)
        pb_dir = -1 if r.zone_side == "buy" else 1
        last_t = bound if b is None else min(bound, b - 1)
        ext, lat, fur = [], [], []
        prev_e = prev_l = prev_f = "unset"
        for t in range(c, last_t + 1):
            e = _closest_to_outer(frame, r.fp_idx, zone, r.fp_idx + 1, min(t, bound))
            q = [i for i in range(r.fp_idx + 1, min(t, bound) + 1) if _qualified(frame, r.fp_idx, i)]
            l = q[-1] if q else None
            f = None
            for i in q:
                ci = float(frame.loc[i, "c"])
                if f is None or (ci < float(frame.loc[f, "c"]) if pb_dir == -1 else ci > float(frame.loc[f, "c"])):
                    f = i
            if f != prev_f:
                fur.append((t, f))
                prev_f = f
            if e != prev_e:
                ext.append((t, e))
                prev_e = e
            if l != prev_l:
                lat.append((t, l))
                prev_l = l
        r4, r4_info = _rule_r4(frame, r.fp_idx, pb_dir, c, cts[0].meta, last_t)
        # step 3: the decided rule, as of the lock (last_t) and to the cycle's end - 1 only (the engine's pre-lock
        # search), + the lock's LP (BOS_{n+1}'s last wave candle) for the warning count
        new, new_route = _rule_new(frame, r.fp_idx, pb_dir, c, cts[0].meta, last_t, bound)
        new_to_bound, _ = _rule_new(frame, r.fp_idx, pb_dir, c, cts[0].meta, bound, bound)
        lock_wc = wv._find_wave_candle(list(wave_candle_results), key[0], key[1] + 1, "BOS") if b is not None else None
        lock_lp = None if lock_wc is None else lock_wc.last_wave_candle_idx
        pat_anchor = None
        if cts[0].meta.get("confirmation_method") != "sd_zone_proximity":
            so, eo, co, _st = cts[0].meta["_m_pat_off"].split(",")
            pat_anchor = c - int(so)
            pat_last = max(c - int(eo), c - int(co) if co != "" else c - int(eo))
        end_cand = None
        if end is not None and int(end) <= len(frame) - 1 and r4:
            ei, li = int(end), r4[-1][1]
            ce, cl = float(frame.loc[ei, "c"]), float(frame.loc[li, "c"])
            end_cand = dict(e=ei, qualified=_qualified(frame, r.fp_idx, ei),
                            further=(ce < cl if pb_dir == -1 else ce > cl))

        def pm(i):
            if i is None:
                return None
            return float(frame.loc[i, "volume"]) * wv._compute_last_wave_weight(frame, i, pb_dir) / r.fp_volume
        CAP.append(dict(path=kw.get("structure_path_id"), extra=extra, sid=key[0], cyc=key[1], side=r.zone_side,
                        bm=r.breakout_momentum, fb=r.fb_idx, lb=r.lb_idx, fp=r.fp_idx, lp_final=r.lp_idx,
                        locked=r.lp_locked, collapsed=r.cycle_collapsed, c=c, via=via, b=b, bound=bound,
                        life=(start, end, reason),
                        ext=[(t, i, pm(i)) for t, i in ext], lat=[(t, i, pm(i)) for t, i in lat],
                        fur=[(t, i, pm(i)) for t, i in fur], r4=[(t, i, pm(i)) for t, i in r4], r4_info=r4_info,
                        end_cand=end_cand,
                        new=[(t, i, pm(i)) for t, i in new], new_route=new_route,
                        new_to_bound=[(t, i, pm(i)) for t, i in new_to_bound], lock_lp=lock_lp,
                        lock_lp_in_frame=lock_lp is not None and lock_lp in frame.index,
                        pat_anchor=pat_anchor,
                        pat_last_is_c=None if pat_anchor is None else pat_last == c,
                        n_qual_total=len([i for i in range(r.fp_idx + 1, bound + 1) if _qualified(frame, r.fp_idx, i)])))


def wrapped(df, events, sorted_events, wave_candle_results, kl_zones, **kw):
    recs = _orig(df, events, sorted_events, wave_candle_results, kl_zones, **kw)
    _capture(recs, df, events, sorted_events, kl_zones, kw, extra=False, wave_candle_results=wave_candle_results)
    if kw.get("gate") is not None:
        have = {(r.bos_structure_id, r.bos_cycle_id) for r in recs}
        more = _orig(df, events, sorted_events, wave_candle_results, kl_zones, **{**kw, "gate": None})
        _capture([r for r in more if (r.bos_structure_id, r.bos_cycle_id) not in have], df, events, sorted_events,
                 kl_zones, kw, extra=True)
    return recs


orch._compute_wvmi_records = wrapped

from engine_v2.config import CONFIG  # noqa: E402
from engine_v2.run_replay import fetch_history_with_auto_extend, _floor_to_day_start  # noqa: E402

df, _, _ = fetch_history_with_auto_extend(pair=CONFIG.pair, timeframe=CONFIG.timeframe,
                                          start=_floor_to_day_start(CONFIG.start), end=CONFIG.end,
                                          lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8)
df.attrs["pair"] = CONFIG.pair
res = orch.run_pipeline(df, lower_timeframes=CONFIG.lower_timeframes)

# match each captured record to a main / lens record (absolute coords + sub_id)
targets = [("H1", None, r) for r in res.meta.get("wvmi", [])]
for lens, ldf in (res.meta.get("sub_lens_dfs") or {}).items():
    targets += [(lens, ldf, r) for r in ldf.attrs["wvmi"]]
out = []
for cap in CAP:
    if cap["extra"]:
        if cap["path"] == "H1.main":
            out.append(dict(cap, lens="H1", sub_id=None, off=0))
        continue
    for lens, _ldf, r in targets:
        if r.bos_cycle_id == cap["cyc"] and abs((r.breakout_momentum or -1) - cap["bm"]) < 1e-12 and r.zone_side == cap["side"]:
            off = r.fp_idx - cap["fp"]
            out.append(dict(cap, lens=lens, sub_id=r.meta.get("sub_id"), off=off))
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
print("captured", len(CAP), "matched rows", len(out))
