"""Read-only SHADOW of the user's Step 1B (2026-10-20 write-up, `user_writeup_20261020.md`) on today's structure —
no engine edit. Step 1B = a PREFERENCE after Step 1 / Step 1A: among the IC candidates, those whose CLOSEST PRIOR
imbalance is still unfilled AS OF the IC go on (exactly one → the final IC); none → every candidate goes on.

Per drawn POI record (every entity: the H1 main + every sub projection), every IC candidate's closest prior imbalance
under the readings of the write-up's open terms, and the pick under each flow:

  readings of "the closest prior imbalance unfilled as of the IC" (P = primary):
    P      sd direction, its c2 run at or after the fib's BOS anchor and ending before the IC (its last c3 may BE the
           IC), unfilled = NOT committed-filled (two-stroke `is_filled(check_to_idx=IC)`, the IC candle included)
    ARM    as P, but unfilled = not ARMED by the IC (stroke 1 alone: no candle in (end_idx, IC] retraced >= thr)
    ALL    as P, no lower bound (the closest prior sd gap anywhere, also before the BOS anchor)
    C3     as P, the gap must be complete before the IC (its last c3 < IC — a gap whose c3 IS the IC excluded)
    ANY    as P, either direction
  flows:
    F1     as written: >= 2 pass → Steps 2-3 on them; exactly 1 → it is final; 0 → Steps 2-3 on all
    F2     as F1, but a single passer still goes through Steps 2-3 (a lone pinbar with no pattern → no IC)
    F3     as F1, but when Steps 2-3 on the passers give no IC (all pinbar), Steps 2-3 run on all candidates
    F4     F2 + F3: the passers (even one) go through Steps 2-3; no IC from them → Steps 2-3 on all candidates
    INC/EXC (P only): "as of the IC" = after the IC candle closes (INC, = P) vs the state the IC candle opens on
           (EXC: fills counted through IC − 1); LEG: fills counted through the end of the IC's pullback run (the
           last of the consecutive agreed −sd candles from the IC on)
  thresholds: 0.70 (today) and 0.65 (the user's change 2 — IC SIDE ONLY here: the fib records, crosses and MS
  structure are today's; the full 0.65 cascade needs a variant replay).

And MS's in-flight read (`structure_engine.poi_inner_for_fib`): how many reads would return a different inner under
P/F1 at 0.70 — the cascade risk (MS reads the same IC; GOTCHAS "An IC / POI Rule Change Can Move MS Outcomes").

    python engine_v2/plans/qa_ic_revisit_inputs/shadow_step1b.py OUT_LOG      (repo root; ~1-3 min)
"""
import sys
import time
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.pipeline.orchestrator as orch  # noqa: E402
import engine_v2.structure.structure_engine as se  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402

READINGS = ("P", "ARM", "ALL", "C3", "ANY", "EXC", "LEG")
FLOWS = ("F1", "F2", "F3", "F4")
# Candle tables for the decision examples: (entity, timeframe) → [(lo, hi)] entity-absolute, inclusive.
TABLES = {("H1main", "H1"): [(168, 184), (853, 866)], ("sub2", "M15"): [(2560, 2580)],
          ("sub3", "M15"): [(2922, 2932)], ("sub0", "M15"): [(1130, 1143), (1203, 1217)]}
DONE_TABLES = set()
THRESHOLDS = (0.70, 0.65)
LINES = []
CTX = []
STATS = Counter()
MS_DIFF = defaultdict(Counter)
SEEN = set()


def out(s):
    LINES.append(s)


_real_rsp = edm.render_sub_projection


def _rsp(sub, *a, **k):
    CTX.append((int(sub.sub_id), int(sub.geometry[1])))
    try:
        return _real_rsp(sub, *a, **k)
    finally:
        CTX.pop()


edm.render_sub_projection = _rsp


def armed_idx(df, inst, upto, thr):
    """Stroke 1 alone: the first candle in (end_idx, upto] retracing >= thr into the gap (None = not armed)."""
    for i in range(inst.end_idx + 1, upto + 1):
        if i not in df.index:
            continue
        if inst.direction == 1 and float(df.loc[i, "l"]) <= inst.gap_top - inst.gap_size * thr:
            return i
        if inst.direction == -1 and float(df.loc[i, "h"]) >= inst.gap_bottom + inst.gap_size * thr:
            return i
    return None


def committed_idx(df, inst, upto, thr):
    """Two-stroke: the stroke-2 candle in [armed, upto] (None = not committed-filled by `upto`)."""
    a = armed_idx(df, inst, upto, thr)
    if a is None:
        return None
    for i in range(a, upto + 1):
        if i not in df.index:
            continue
        c = float(df.loc[i, "c"])
        if (inst.direction == 1 and c >= inst.gap_top) or (inst.direction == -1 and c <= inst.gap_bottom):
            return i
    return None


def closest_prior(df, ic, fs, reading):
    sd = int(fs.struct_direction)
    best = None
    for inst in df.attrs.get("imbalances", []):
        if reading != "ANY" and inst.direction != sd:
            continue
        if inst.end_idx >= ic:                       # its c2 run must end before the IC
            continue
        if reading == "C3" and inst.end_idx + 1 >= ic:
            continue
        if reading == "EXC" and inst.end_idx + 1 >= ic:   # the gap must exist when the IC candle opens
            continue
        if reading != "ALL" and inst.start_idx < fs.bos_idx:
            if inst.end_idx >= fs.bos_idx:
                STATS["straddles_bos_anchor"] += 1
            continue
        if best is None or inst.end_idx > best.end_idx:
            best = inst
    return best


def passes_1b(df, ic, fs, reading, thr):
    inst = closest_prior(df, ic, fs, reading)
    if inst is None:
        return False, None, None, None
    if inst.gap_size <= 0:
        return False, inst, None, None          # `is_filled` treats a degenerate gap as filled
    if reading == "ARM":
        a = armed_idx(df, inst, ic, thr)
        return a is None, inst, a, None
    upto = ic - 1 if reading == "EXC" else ic
    if reading == "LEG":
        from engine_v2.features.candles_v2 import agreed_direction
        ag = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
        pos = df.index.get_loc(ic)
        while pos + 1 < len(df) and int(ag[pos + 1]) == -int(fs.struct_direction):
            pos += 1
        upto = int(df.index[pos])
    a = armed_idx(df, inst, upto, thr)
    c = committed_idx(df, inst, upto, thr)
    assert (c is not None) == inst.is_filled(df, upto, thr), (ic, inst)
    return c is None, inst, a, c


def steps_2_3(df, fs, cfg, cands):
    p, tb = pz.select_institutional_pattern(df, fs, cfg, cands)
    if p is not None:
        return int(p.ic_idx), f"pattern {p.ip_rank}" + (f" tb={tb[0]}" if tb else "")
    p, rk = pz.apply_ranking_rules(df, fs, cfg, cands)
    if p is None:
        return None, "all_ic_candidates_pinbar"
    return int(p.ic_idx), "ranked" + (f" {rk[0]}" if rk else " (only)")


def pick_1b(df, fs, cfg, sel, reading, thr, flow):
    cands = list(sel.ic_candidates)
    if not cands:
        return None, "no_ic_candidate", 0
    passers = [c for c in cands if passes_1b(df, c.ic_idx, fs, reading, thr)[0]]
    if len(passers) == 1 and flow not in ("F2", "F4"):
        return int(passers[0].ic_idx), "1B single", 1
    pool = passers if passers else cands
    pick, why = steps_2_3(df, fs, cfg, pool)
    if pick is None and flow in ("F3", "F4") and passers:
        pick, why = steps_2_3(df, fs, cfg, cands)
        why = "F3 fallback " + why
    return pick, ("1B " if passers else "no-1B ") + why, len(passers)


_real_select_ic = pz.select_ic
_real_sir = orch.select_ic_records


def _sir(df, events, fib_tracker, config, *, lifecycle_cap=None):
    recs = _real_sir(df, events, fib_tracker, config, lifecycle_cap=lifecycle_cap)
    c = CTX[-1] if CTX else None
    ent = "H1main" if c is None else f"sub{c[0]}"
    b = 0 if c is None else c[1]
    tf = df.attrs.get("timeframe")

    def A(i):
        return None if i is None else int(i) + b

    if (ent, tf) in TABLES and (ent, tf) not in DONE_TABLES:
        DONE_TABLES.add((ent, tf))
        from engine_v2.features.candles_v2 import agreed_direction
        agreed = pd.Series(agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy()), index=df.index)
        last = int(df.index.max())
        for lo, hi in TABLES[(ent, tf)]:
            out(f"[TABLE] {ent} {tf} {lo}..{hi} (entity-absolute; agreed = candle/volume agreed direction)")
            out("   idx  time(UTC)         open     high     low      close    dir vol agr type")
            for i in range(lo - b, hi - b + 1):
                if i not in df.index:
                    continue
                rw = df.loc[i]
                out(f"   {A(i):>5} {str(rw['time'])[:16]} {rw['o']:.5f} {rw['h']:.5f} {rw['l']:.5f} {rw['c']:.5f} "
                    f"{int(rw['direction']):+d} {int(rw['vol_dir']):+d} {int(agreed.loc[i]):+d} {rw['candle_type']}")
            for inst in df.attrs.get("imbalances", []):
                if inst.end_idx < lo - b or inst.start_idx > hi - b:
                    continue
                fills = "; ".join(f"thr {t}: armed {A(armed_idx(df, inst, last, t))} committed "
                                  f"{A(committed_idx(df, inst, last, t))}" for t in THRESHOLDS)
                out(f"   gap c2 {A(inst.start_idx)}..{A(inst.end_idx)} dir {inst.direction:+d} "
                    f"[{inst.gap_bottom:.5f}, {inst.gap_top:.5f}] — {fills}")
    for r in recs:
        fs = r.fib_state
        m = fs.meta
        key = (ent, int(fs.structure_id), int(fs.cycle_id), m.get("version"), A(fs.bos_idx), A(fs.cts_idx))
        if key in SEEN:
            continue
        SEEN.add(key)
        ref = fib_tracker.ic_reference(fs, df, evaluated_at=lifecycle_cap)
        kind = f"cross {m.get('cross_start_cycle')}->{fs.cycle_id}" if m.get("cross_cycle") else f"single {fs.cycle_id}"
        real = None if r.selection.pick is None else int(r.selection.pick.ic_idx)
        sd = int(fs.struct_direction)
        out(f"[REC] {ent} {tf} sid{fs.structure_id} {kind} sd {sd:+d} bos {A(fs.bos_idx)} cts {A(fs.cts_idx)} "
            f"band {r.selection.ic_zone_band} today: IC {A(real)} "
            f"({r.selection.no_ic_reason or r.selection.ip_outcome})")
        STATS["records"] += 1
        for thr in THRESHOLDS:
            cfg = pz.POIConfig(fill_threshold=thr)
            sel = r.selection if thr == 0.70 else _real_select_ic(df, fs, cfg, ic_reference=ref, classify_all=True)
            if thr == 0.70:
                assert sel.pick is None or int(sel.pick.ic_idx) == real
            cands = list(sel.ic_candidates)
            if thr != 0.70:
                base = None if sel.pick is None else int(sel.pick.ic_idx)
                out(f"   thr {thr}: Step 1{'A' if sel.ic_zone_band != '61.8-80' else ''} candidates "
                    f"{[A(x.ic_idx) for x in cands]} -> today's flow IC {A(base)} "
                    f"({sel.no_ic_reason or sel.ip_outcome}){'  MOVED vs 0.70' if base != real else ''}")
                STATS[f"thr{thr}_today_flow_moved"] += base != real
                STATS[f"thr{thr}_today_flow_no_ic"] += base is None
            else:
                base = real
            for cc in cands:
                parts = []
                for rd in READINGS:
                    ok, inst, a, cm = passes_1b(df, cc.ic_idx, fs, rd, thr)
                    if rd == "P" or ok != passes_1b(df, cc.ic_idx, fs, "P", thr)[0]:
                        if inst is None:
                            parts.append(f"{rd}: none -> FAIL")
                        else:
                            parts.append(f"{rd}: gap {A(inst.start_idx)}..{A(inst.end_idx)} dir {inst.direction:+d}"
                                         f" armed {A(a)} committed {A(cm)} -> {'PASS' if ok else 'FAIL'}")
                    STATS[f"thr{thr}_{rd}_cand_{'pass' if ok else 'fail'}"] += 1
                    if inst is None:
                        STATS[f"thr{thr}_{rd}_cand_no_prior_gap"] += 1
                    elif rd == "P" and inst.end_idx + 1 == cc.ic_idx:
                        STATS[f"thr{thr}_P_cand_gap_c3_is_ic"] += 1
                pt = cc.pattern.ip_rank if cc.pattern is not None and cc.pattern.rank is not None else "-"
                out(f"   thr {thr} cand {A(cc.ic_idx)} {df.loc[cc.ic_idx, 'candle_type']} ip {pt} | "
                    + " | ".join(parts))
            for rd in READINGS:
                for fl in FLOWS:
                    if rd != "P" and fl != "F1":
                        continue
                    pick, why, n = pick_1b(df, fs, cfg, sel, rd, thr, fl)
                    moved = pick != base
                    STATS[f"thr{thr}_{rd}_{fl}_moved"] += moved
                    STATS[f"thr{thr}_{rd}_{fl}_no_ic"] += pick is None
                    STATS[f"thr{thr}_{rd}_{fl}_single_1b"] += why == "1B single"
                    if rd == "P" or moved:
                        out(f"   thr {thr} {rd}/{fl}: passers {n} -> IC {A(pick)} {why}"
                            f"{'  MOVED' if moved else ''}")
    return recs


orch.select_ic_records = _sir

_real_pif = se.poi_inner_for_fib


def _pif(df, fib_state, fill_threshold, *, tracker, evaluated_at):
    res = _real_pif(df, fib_state, fill_threshold, tracker=tracker, evaluated_at=evaluated_at)
    if fib_state is None:
        return res
    tf = df.attrs.get("timeframe")
    MS_DIFF[tf]["reads"] += 1
    ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))
    cfg = pz.POIConfig(fill_threshold=fill_threshold)
    sel = _real_select_ic(df, fib_state, cfg, ic_reference=ref)
    sd = int(fib_state.struct_direction)
    for fl in FLOWS:
        pick, _why, _n = pick_1b(df, fib_state, cfg, sel, "P", fill_threshold, fl)
        inner = None if pick is None else float(df.loc[pick, "h" if sd == 1 else "l"])
        MS_DIFF[tf][f"P_{fl}_inner_differs"] += inner != res
    return res


se.poi_inner_for_fib = _pif

t0 = time.perf_counter()
df, effective_start, _d = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
out(f"[STATS] {dict(sorted(STATS.items()))}")
out(f"[MS] reads + reads whose inner would differ (P, thr 0.70), per TF: { {k: dict(v) for k, v in MS_DIFF.items()} }")
out(f"[trace] done in {time.perf_counter() - t0:.1f} s")
text = "\n".join(LINES)
if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
print(text[-6000:])
