"""Read-only SHADOW on HEAD (no engine edit; PLAN_S step 3 prep, 2026-10-21): Step 1B as decided (qa_findings.md
round 5) at the engine's LANDED fill rule (0.65 + the two strokes on separate candles — steps 1–2):

  per IC candidate: its closest prior imbalance = the sd instance whose c2 run starts at / after the fib's BOS anchor
  and ends before the IC, the one ending latest; it PASSES when that gap is not filled through IC − 1
  (`ImbalanceInstance.is_filled(df, IC − 1)`); no prior gap -> fails.
  flow: passers -> Steps 2–3; no IC from them -> Steps 2–3 on all the candidates; no passer -> all the candidates.

Prints, per drawn record (H1 main + every unique sub, entity-absolute indices): today's pick and the 1B pick with its
outcome; a candle table for every record whose pick moves (every IC candidate, its closest prior gap, the strokes
through IC − 1). Also counts (first order): the proximity overlay's capped views whose pick moves, MS's in-flight
reads whose inner differs (`structure_engine.poi_inner_for_fib` wrapped), straddling gaps that would be closer,
degenerate closest gaps (gap_size <= 0 — `is_filled` treats them as filled).

    python engine_v2/plans/qa_ic_revisit_inputs/shadow_step1b_final.py OUT_LOG      (repo root)
"""
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.pipeline.orchestrator as orch  # noqa: E402
import engine_v2.structure.structure_engine as se  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402

PIP = 0.0001
LINES = []
CTX = []
STATS = Counter()
SEEN_REC = set()


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


def closest_prior(df, ic, fs):
    sd = int(fs.struct_direction)
    best, straddler = None, None
    for inst in df.attrs.get("imbalances", []):
        if inst.direction != sd or inst.end_idx >= ic:
            continue
        if inst.start_idx < fs.bos_idx:
            if inst.end_idx >= fs.bos_idx:
                straddler = inst
            continue
        if best is None or inst.end_idx > best.end_idx:
            best = inst
    if straddler is not None and (best is None or straddler.end_idx > best.end_idx):
        STATS["straddling gap that would be the closest"] += 1
    return best


def strokes(df, inst, upto, thr):
    """(armed_idx, committed_idx) through `upto`, the landed rule (stroke 2 strictly after the arming candle)."""
    armed = None
    for i in range(inst.end_idx + 1, upto + 1):
        if armed is not None:
            c = float(df.loc[i, "c"])
            if (inst.direction == 1 and c >= inst.gap_top) or (inst.direction == -1 and c <= inst.gap_bottom):
                return armed, i
        elif inst.direction == 1 and float(df.loc[i, "l"]) <= inst.gap_top - inst.gap_size * thr:
            armed = i
        elif inst.direction == -1 and float(df.loc[i, "h"]) >= inst.gap_bottom + inst.gap_size * thr:
            armed = i
    return armed, None


def passes(df, ic, fs, thr):
    inst = closest_prior(df, ic, fs)
    if inst is None:
        return False, None
    ok = not inst.is_filled(df, ic - 1, thr)
    if inst.gap_size > 0:
        assert ok == (strokes(df, inst, ic - 1, thr)[1] is None), (ic, inst)
    else:
        STATS["degenerate closest prior gap (gap_size <= 0)"] += 1
    return ok, inst


def steps_2_3(df, fs, cfg, cands):
    p, _tb = pz.select_institutional_pattern(df, fs, cfg, cands)
    if p is not None:
        return int(p.ic_idx), f"pattern {p.ip_rank}"
    p, rk = pz.apply_ranking_rules(df, fs, cfg, cands)
    if p is None:
        return None, "all_ic_candidates_pinbar"
    return int(p.ic_idx), "ranked " + (rk[0] if rk else "(only)")


def pick_1b(df, fs, cfg, cands, thr):
    if not cands:
        return None, "no_ic_candidate", 0
    passers = [c for c in cands if passes(df, int(c.ic_idx), fs, thr)[0]]
    if passers:
        pick, why = steps_2_3(df, fs, cfg, passers)
        if pick is not None:
            return pick, "preferred: " + why, len(passers)
        pick, why = steps_2_3(df, fs, cfg, cands)
        return pick, "passers gave no IC -> all: " + why, len(passers)
    pick, why = steps_2_3(df, fs, cfg, cands)
    return pick, "no passer -> all: " + why, 0


def candle_table(df, fs, cands, b, thr):
    out("     | cand | time (UTC)       | type   |       o |       h |       l |       c | closest prior gap (c2) "
        "| bottom..top (pips) | armed | committed <= IC-1 | 1B")
    for cc in cands:
        ic = int(cc.ic_idx)
        r = df.loc[ic]
        ok, inst = passes(df, ic, fs, thr)
        if inst is None:
            g, bt, a, cm = "none", "", "", ""
        else:
            a, cm = strokes(df, inst, ic - 1, thr)
            g = f"{inst.start_idx + b}" + ("" if inst.start_idx == inst.end_idx else f"..{inst.end_idx + b}")
            bt = f"{inst.gap_bottom:.5f}..{inst.gap_top:.5f} ({inst.gap_size / PIP:.1f})"
            a = "-" if a is None else a + b
            cm = "-" if cm is None else cm + b
        out(f"     | {ic + b} | {str(r['time'])[:16]} | {r['candle_type']:<6} | {r['o']:.5f} | {r['h']:.5f} | "
            f"{r['l']:.5f} | {r['c']:.5f} | {g} | {bt} | {a} | {cm} | {'PASS' if ok else 'fail'}")


_real_sir = orch.select_ic_records


def _sir(df, events, fib_tracker, config, *, lifecycle_cap=None):
    recs = _real_sir(df, events, fib_tracker, config, lifecycle_cap=lifecycle_cap)
    c = CTX[-1] if CTX else None
    tf = df.attrs.get("timeframe")
    thr = config.fill_threshold
    view = c is None and tf != "H1"   # a capped proximity-overlay view (view-local indices)
    ent = "H1main" if c is None else f"sub{c[0]}"
    b = 0 if c is None else c[1]

    def A(i):
        return None if i is None else int(i) + b

    for r in recs:
        fs = r.fib_state
        cands = list(r.selection.ic_candidates)
        real = None if r.selection.pick is None else int(r.selection.pick.ic_idx)
        pick, why, n = pick_1b(df, fs, config, cands, thr)
        assert (pick is None) == (real is None), (ent, fs.structure_id, fs.cycle_id, real, pick)
        if view:
            STATS["capped-view records"] += 1
            STATS["capped-view records whose pick moves"] += pick != real
            continue
        m = fs.meta
        key = (ent, int(fs.structure_id), int(fs.cycle_id), m.get("version"), A(fs.bos_idx), A(fs.cts_idx))
        if key in SEEN_REC:
            continue
        SEEN_REC.add(key)
        kind = f"cross {m.get('cross_start_cycle')}->{fs.cycle_id}" if m.get("cross_cycle") else f"single {fs.cycle_id}"
        STATS["drawn records"] += 1
        STATS["drawn records with no IC"] += real is None
        STATS["drawn picks moved"] += pick != real
        STATS[f"outcome: {why.split(':')[0].split(' ->')[0]}"] += 1
        STATS["IC candidates"] += len(cands)
        STATS["IC candidates passing 1B"] += sum(passes(df, int(cc.ic_idx), fs, thr)[0] for cc in cands)
        out(f"[REC] {ent} {tf} sid{fs.structure_id} {kind} sd {int(fs.struct_direction):+d} bos {A(fs.bos_idx)} "
            f"cts {A(fs.cts_idx)} | candidates {len(cands)}, passers {n} | today IC {A(real)} | 1B IC {A(pick)} "
            f"({why}){'  MOVED' if pick != real else ''}")
        if pick != real:
            candle_table(df, fs, cands, b, thr)
    return recs


orch.select_ic_records = _sir

_real_inner = se.poi_inner_for_fib


def _inner(df, fib_state, fill_threshold, *, tracker, evaluated_at):
    got = _real_inner(df, fib_state, fill_threshold, tracker=tracker, evaluated_at=evaluated_at)
    STATS["MS in-flight reads"] += 1
    if fib_state is None:
        return got
    cfg = pz.POIConfig(fill_threshold=fill_threshold)
    ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))
    sel = pz.select_ic(df, fib_state, cfg, ic_reference=ref)
    pick = pick_1b(df, fib_state, cfg, list(sel.ic_candidates), fill_threshold)[0]
    alt = None if pick is None else (
        float(df.loc[pick, "h"]) if int(fib_state.struct_direction) == 1 else float(df.loc[pick, "l"]))
    STATS["MS in-flight reads whose inner differs (first order)"] += alt != got
    if alt != got:
        # CTX is pushed only inside `render_sub_projection`: an M15 read outside it is a sub GEOMETRY MS run, whose
        # indices are slice-local (no offset known here) — labelled so, never "H1main" (plan review lens B, 2026-10-21)
        c = CTX[-1] if CTX else None
        b = 0 if c is None else c[1]
        tf = df.attrs.get("timeframe")
        who = (f"sub{c[0]}" if c is not None else "H1main" if tf == "H1" else "sub-geometry (slice-local idx)")
        out(f"[MS] {who} {tf} sid{fib_state.structure_id} "
            f"cycle {fib_state.cycle_id} fib {fib_state.bos_idx + b}..{fib_state.cts_idx + b} at moment "
            f"{int(evaluated_at) + b} (frame len {len(df)}): today IC {None if sel.pick is None else int(sel.pick.ic_idx) + b} "
            f"inner {got} -> 1B IC {None if pick is None else pick + b} inner {alt}")
    return got


se.poi_inner_for_fib = _inner

t0 = time.perf_counter()
df, effective_start, _d = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
for k in sorted(STATS):
    out(f"[STATS] {k}: {STATS[k]}")
out(f"[trace] done in {time.perf_counter() - t0:.1f} s")
text = "\n".join(LINES)
if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
print(text[-12000:])
