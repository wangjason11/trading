"""Read-only SHADOW of the user's three IC changes (2026-10-13 write-up, `user_writeup_20261013.md`) on today's
structure — no engine edit. Per drawn POI record (every entity: the H1 main + every sub projection) the pick under:

  V0   today (asserted == the real `select_ic` pick)
  V1   change 1: IC condition 3 = an unfilled sd imbalance IMMEDIATELY after (first c2 in IC+1 … IC+2, fill through
       the CTS anchor — `immediate_unfilled_imbalance_first_c2_idx`); the tiebreak / ranking immediate rule is then a
       no-op (every candidate passes it), so the existing rule functions run unchanged
  V2   change 2: the CTS reference = CTS_{T−1} (T = the fib's target cycle) for a cross too — `ic_reference` with
       start cycle = T (bound = cycle T's establishing pattern anchor)
  V12  changes 1 + 2
  V123 changes 1 + 2 + 3: no candidate in 61.8–80 → the same conditions with the 80–100 zone (Step 1A)

and (MS's in-flight read, `structure_engine.poi_inner_for_fib`) how many reads would return a different inner — the
cascade risk (GOTCHAS "An IC / POI Rule Change Can Move MS Outcomes"); the shadow cannot show the cascade itself.

    python <this> [OUT_LOG]      (repo root; ~2–4 min)
"""
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.pipeline.orchestrator as orch  # noqa: E402
import engine_v2.structure.structure_engine as se  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402
from engine_v2.zones.cross_cycle_fib import ICReference  # noqa: E402

VARIANTS = {"V1": (True, False, False), "V2": (False, True, False), "V12": (True, True, False),
            "V123": (True, True, True)}
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


def ref_v2(tracker, fs, df, evaluated_at):
    """Change 2's reference: CTS_{T−1} whatever the start cycle (a single is unchanged)."""
    if not fs.meta.get("cross_cycle"):
        return tracker.ic_reference(fs, df, evaluated_at=evaluated_at)
    sid, t = int(fs.structure_id), int(fs.cycle_id)
    if (sid, t) in tracker._est_pattern_anchor and (sid, t) in tracker._bos_moment_by_cycle:
        return tracker.ic_reference(replace(fs, meta={**fs.meta, "cross_cycle": False}), df,
                                    evaluated_at=evaluated_at)
    # target not yet established (a pre-established cross): CTS_{T−1}, no known break yet, vs the structure reference
    STATS["v2_pre_established_ref"] += 1
    struct = tracker.ic_reference(replace(fs, meta={**fs.meta, "cross_start_cycle": 0}), df,
                                  evaluated_at=evaluated_at)
    prev = float(tracker._cts_by_cycle[(sid, t - 1)][1])
    sd = int(fs.struct_direction)
    if struct.price is None or (prev < struct.price if sd == 1 else prev > struct.price):
        return ICReference(prev, "prev_cycle_cts")
    return struct


def pick_under(df, fs, ft, ref, c1, c3, classify_all):
    cfg = pz.POIConfig(fill_threshold=ft)

    def cands_of(sel):
        res = []
        for c in sel.checks:
            if c.ic_ref_reason is not None or not c.touches_zone:
                continue
            if c1:
                # new condition 3 ⊂ old: an unclassified check failed the old one → fails the new one
                if c.pattern is None or c.immediate_imbalance_first_c2_idx is None:
                    continue
            elif not c.has_unfilled_imbalance_after:
                continue
            res.append(c)
        return res

    sel = _real_select_ic(df, fs, cfg, ic_reference=ref, classify_all=classify_all)
    cands, zone = cands_of(sel), "61.8-80"
    if not cands and c3:
        cfg_d = pz.POIConfig(fill_threshold=ft, ic_fib_min=80.0, ic_fib_max=100.0)
        cands, zone = cands_of(_real_select_ic(df, fs, cfg_d, ic_reference=ref, classify_all=classify_all)), "80-100"
    if not cands:
        return {"pick": None, "why": "no_ic_candidate", "zone": zone, "cands": []}
    ids = [c.ic_idx for c in cands]
    p, tb = pz.select_institutional_pattern(df, fs, cfg, cands)
    if p is not None:
        return {"pick": int(p.ic_idx), "why": f"pattern {p.ip_rank}" + (f" tb={tb[0]}" if tb else ""),
                "zone": zone, "cands": ids}
    p, rk = pz.apply_ranking_rules(df, fs, cfg, cands)
    if p is None:
        return {"pick": None, "why": "all_ic_candidates_pinbar", "zone": zone, "cands": ids}
    return {"pick": int(p.ic_idx), "why": "no_pattern" + (f" rank={rk[0]}" if rk else " (only candidate)"),
            "zone": zone, "cands": ids}


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

    for r in recs:
        fs = r.fib_state
        m = fs.meta
        key = (ent, int(fs.structure_id), int(fs.cycle_id), m.get("version"), A(fs.bos_idx), A(fs.cts_idx))
        if key in SEEN:
            continue
        SEEN.add(key)
        ref0 = fib_tracker.ic_reference(fs, df, evaluated_at=lifecycle_cap)
        base = pick_under(df, fs, config.fill_threshold, ref0, False, False, True)
        real = None if r.selection.pick is None else int(r.selection.pick.ic_idx)
        assert base["pick"] == real, (key, base, real)
        kind = f"cross {m.get('cross_start_cycle')}->{fs.cycle_id} v{m.get('version')}" if m.get("cross_cycle") \
            else f"single {fs.cycle_id} ({m.get('via')})"
        out(f"[REC] {ent} {tf} sid{fs.structure_id} {kind} bos {A(fs.bos_idx)} cts {A(fs.cts_idx)} "
            f"locked={fs.locked} ref0={ref0.price}/{ref0.source}/brk {A(ref0.break_pattern_anchor_idx)}")
        out(f"      V0   pick={A(base['pick'])} {base['why']} cands={[A(i) for i in base['cands']]}")
        STATS["records"] += 1
        if base["pick"] is None:
            STATS["V0_no_ic"] += 1
        ref2 = ref_v2(fib_tracker, fs, df, lifecycle_cap)
        for name, (c1, c2, c3) in VARIANTS.items():
            ref = ref2 if c2 else ref0
            v = pick_under(df, fs, config.fill_threshold, ref, c1, c3, True)
            moved = v["pick"] != base["pick"]
            STATS[f"{name}_moved"] += moved
            STATS[f"{name}_no_ic"] += v["pick"] is None
            STATS[f"{name}_deep_zone"] += (v["zone"] == "80-100" and v["pick"] is not None)
            if "most_recent_before_imbalance" in v["why"]:
                STATS[f"{name}_decided_by_most_recent_before_imbalance"] += 1
            extra = f" ref={ref.price}/{ref.source}/brk {A(ref.break_pattern_anchor_idx)}" if c2 and ref != ref0 else ""
            out(f"      {name:<4} pick={A(v['pick'])} {'MOVED ' if moved else ''}{v['why']} zone={v['zone']} "
                f"cands={[A(i) for i in v['cands']]}{extra}")
    return recs


orch.select_ic_records = _sir

_real_pif = se.poi_inner_for_fib


def _pif(df, fib_state, fill_threshold, *, tracker, evaluated_at):
    res = _real_pif(df, fib_state, fill_threshold, tracker=tracker, evaluated_at=evaluated_at)
    if fib_state is None:
        return res
    tf = df.attrs.get("timeframe")
    STATS[f"ms_reads_{tf}"] += 1
    ref0 = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))
    ref2 = ref_v2(tracker, fib_state, df, int(evaluated_at))
    sd = int(fib_state.struct_direction)
    for name, (c1, c2, c3) in VARIANTS.items():
        v = pick_under(df, fib_state, fill_threshold, ref2 if c2 else ref0, c1, c3, False)
        inner = None if v["pick"] is None else float(df.loc[v["pick"], "h" if sd == 1 else "l"])
        MS_DIFF[tf][name] += inner != res
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
out(f"[MS] reads whose inner would differ, per TF: { {k: dict(v) for k, v in MS_DIFF.items()} }")
out(f"[trace] done in {time.perf_counter() - t0:.1f} s")
text = "\n".join(LINES)
if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
print(text[-3000:])
