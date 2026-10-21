"""Read-only SHADOW (no engine edit) for the user's round-2 answers (2026-10-21): "the IC can never actually fill the
gap, but only arm it since it is an opposite direction candle" and "filled requires armed and ANOTHER candle to close
beyond". The engine's two-stroke rule (`ImbalanceInstance.is_filled`, IMBALANCE_FILL_SEMANTICS "The predicate") lets
stroke 2 fire on the arming candle itself (`idx >= armed_idx`) and counts a close AT the outer edge (`<=` / `>=`).

Fill-rule variants measured:
    E    the engine today: stroke 2 in [armed, upto], close at or past the outer edge
    S    stroke 2 strictly after the arming candle (the user's "another candle"), close at or past the edge
    B    stroke 2 in [armed, upto], close strictly past the edge (the literal "close beyond")
    SB   S + B
    EN   E, but a candidate with NO prior sd gap in the window passes 1B (the write-up is silent; default: it fails)
    X    E, but 1B reads the gap's fill state as of the candle BEFORE the IC (a gap whose c3 is the IC still counts)
         — the engine's rule unchanged, the IC cannot disqualify itself (1B-only; census unaffected)
    SX   S + X: the user's round-5 decision (2026-10-21) — separate candles engine-wide, 1B reads the gap as of
         the candle right before the IC

1. Every imbalance instance (H1 main df, every sub projection df — M15 deduped by entity-absolute key): how many
   commit at all, how many commit on the arming candle, how many on a close exactly AT the edge, and how many
   instances' committed-fill candle moves under S / B / SB (the scale of a GLOBAL rule change: every fill reader —
   fib activation, IC condition 3, POI activation — would see it; the cascade itself needs a variant replay).
2. Step 1B (reading P: sd gaps, c2 run from the fib's BOS anchor, ending before the IC, fills counted THROUGH the IC;
   flow F4 = the user's round-2 flow) on the 27 drawn records under each variant, at 0.70 and 0.65 (IC side only):
   the candidates whose closest prior gap is committed ON the IC candle, and the picks.

    python engine_v2/plans/qa_ic_revisit_inputs/shadow_fill_rule.py OUT_LOG      (repo root)
"""
import sys
import time
from collections import Counter

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.pipeline.orchestrator as orch  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402

VARIANTS = ("E", "S", "B", "SB", "EN", "X", "SX")  # EN / X / SX: see the docstring
THRESHOLDS = (0.70, 0.65)
EPS = 5e-7  # prices are 5-decimal; "exactly at the edge" = within half a 0.1-pip step
LINES = []
CTX = []
STATS = Counter()
SEEN_INST = set()
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


def strokes(df, inst, upto, thr, variant):
    """(armed_idx, committed_idx) through `upto` under a fill-rule variant (None = not reached)."""
    armed = None
    for i in range(inst.end_idx + 1, upto + 1):
        if i not in df.index:
            continue
        if inst.direction == 1 and float(df.loc[i, "l"]) <= inst.gap_top - inst.gap_size * thr:
            armed = i
            break
        if inst.direction == -1 and float(df.loc[i, "h"]) >= inst.gap_bottom + inst.gap_size * thr:
            armed = i
            break
    if armed is None:
        return None, None
    first = armed + 1 if "S" in variant else armed
    for i in range(first, upto + 1):
        if i not in df.index:
            continue
        c = float(df.loc[i, "c"])
        if "B" in variant:
            hit = c > inst.gap_top + EPS if inst.direction == 1 else c < inst.gap_bottom - EPS
        else:
            hit = c >= inst.gap_top if inst.direction == 1 else c <= inst.gap_bottom
        if hit:
            return armed, i
    return armed, None


def closest_prior(df, ic, fs):
    sd = int(fs.struct_direction)
    best = None
    for inst in df.attrs.get("imbalances", []):
        if inst.direction != sd or inst.end_idx >= ic or inst.start_idx < fs.bos_idx:
            continue
        if best is None or inst.end_idx > best.end_idx:
            best = inst
    return best


def passes(df, ic, fs, thr, variant):
    inst = closest_prior(df, ic, fs)
    if inst is None:
        return variant == "EN", inst, None, None
    if inst.gap_size <= 0:
        return False, inst, None, None
    upto = ic - 1 if "X" in variant else ic
    a, c = strokes(df, inst, upto, thr, variant)
    if variant in ("E", "X"):
        assert (c is not None) == inst.is_filled(df, upto, thr), (ic, inst)
    return c is None, inst, a, c


def steps_2_3(df, fs, cfg, cands):
    p, _tb = pz.select_institutional_pattern(df, fs, cfg, cands)
    if p is not None:
        return int(p.ic_idx), f"pattern {p.ip_rank}"
    p, rk = pz.apply_ranking_rules(df, fs, cfg, cands)
    if p is None:
        return None, "all_ic_candidates_pinbar"
    return int(p.ic_idx), "ranked" + (f" {rk[0]}" if rk else " (only)")


def pick_f4(df, fs, cfg, cands, thr, variant):
    if not cands:
        return None, "no_ic_candidate", 0
    passers = [c for c in cands if passes(df, c.ic_idx, fs, thr, variant)[0]]
    pick, why = steps_2_3(df, fs, cfg, passers if passers else cands)
    if pick is None and passers:
        pick, why = steps_2_3(df, fs, cfg, cands)
        why = "fallback " + why
    return pick, ("1B " if passers else "no-1B ") + why, len(passers)


def census(df, ent, b):
    last = int(df.index.max())
    tf = df.attrs.get("timeframe")
    for inst in df.attrs.get("imbalances", []):
        key = (tf, inst.start_idx + b, inst.end_idx + b, inst.direction)
        if key in SEEN_INST:
            continue
        SEEN_INST.add(key)
        for thr in THRESHOLDS:
            res = {v: strokes(df, inst, last, thr, v) for v in VARIANTS}
            a, c = res["E"]
            STATS[f"{tf} thr{thr} instances"] += 1
            if c is None:
                continue
            STATS[f"{tf} thr{thr} committed (E)"] += 1
            if a == c:
                STATS[f"{tf} thr{thr} committed on the arming candle (E)"] += 1
            edge = inst.gap_top if inst.direction == 1 else inst.gap_bottom
            if abs(float(df.loc[c, "c"]) - edge) <= EPS:
                STATS[f"{tf} thr{thr} committed by a close AT the edge (E)"] += 1
            for v in ("S", "B", "SB"):
                if res[v][1] != c:
                    STATS[f"{tf} thr{thr} commit candle moves under {v}"] += 1
                    if res[v][1] is None:
                        STATS[f"{tf} thr{thr} never committed under {v}"] += 1
            if thr == 0.70 and a == c and len([k for k in STATS if k.startswith("example")]) < 12:
                rw = df.loc[c]
                STATS[f"example {ent} {tf} gap c2 {inst.start_idx + b}..{inst.end_idx + b} dir {inst.direction:+d} "
                      f"[{inst.gap_bottom:.5f},{inst.gap_top:.5f}] armed+committed {c + b} "
                      f"o {rw['o']:.5f} h {rw['h']:.5f} l {rw['l']:.5f} c {rw['c']:.5f} "
                      f"dir {int(rw['direction']):+d} {rw['candle_type']}"] += 1


_real_sir = orch.select_ic_records


def _sir(df, events, fib_tracker, config, *, lifecycle_cap=None):
    recs = _real_sir(df, events, fib_tracker, config, lifecycle_cap=lifecycle_cap)
    c = CTX[-1] if CTX else None
    tf = df.attrs.get("timeframe")
    if c is None and tf != "H1":
        return recs  # a capped proximity-overlay view (view-local indices) — not a drawn record
    ent = "H1main" if c is None else f"sub{c[0]}"
    b = 0 if c is None else c[1]
    census(df, ent, b)

    def A(i):
        return None if i is None else int(i) + b

    for r in recs:
        fs = r.fib_state
        m = fs.meta
        key = (ent, int(fs.structure_id), int(fs.cycle_id), m.get("version"), A(fs.bos_idx), A(fs.cts_idx))
        if key in SEEN_REC:
            continue
        SEEN_REC.add(key)
        kind = f"cross {m.get('cross_start_cycle')}->{fs.cycle_id}" if m.get("cross_cycle") else f"single {fs.cycle_id}"
        real = None if r.selection.pick is None else int(r.selection.pick.ic_idx)
        ref = fib_tracker.ic_reference(fs, df, evaluated_at=lifecycle_cap)
        out(f"[REC] {ent} {tf} sid{fs.structure_id} {kind} sd {int(fs.struct_direction):+d} bos {A(fs.bos_idx)} "
            f"cts {A(fs.cts_idx)} today: IC {A(real)}")
        STATS["drawn records"] += 1
        for thr in THRESHOLDS:
            cfg = pz.POIConfig(fill_threshold=thr)
            sel = r.selection if thr == 0.70 else pz.select_ic(df, fs, cfg, ic_reference=ref, classify_all=True)
            cands = list(sel.ic_candidates)
            for cc in cands:
                ic = int(cc.ic_idx)
                parts = []
                for v in VARIANTS:
                    ok, inst, a, cm = passes(df, ic, fs, thr, v)
                    if v == "E" and cm == ic:
                        STATS[f"thr{thr} cand: closest prior gap committed ON the IC (E)"] += 1
                        STATS[f"thr{thr} cand: ... and armed on the IC too (E)"] += a == ic
                    parts.append(f"{v} {'PASS' if ok else 'FAIL'}" + ("" if inst is None else f" a{A(a)} c{A(cm)}"))
                gap = closest_prior(df, ic, fs)
                STATS[f"thr{thr} cand (drawn records)"] += 1
                STATS[f"thr{thr} cand with NO prior sd gap in the window"] += gap is None
                if len({p.split()[1] for p in parts}) > 1 or any(f" c{A(ic)}" in p for p in parts):
                    out(f"   thr {thr} cand {A(ic)} {df.loc[ic, 'candle_type']} gap "
                        f"{None if gap is None else (A(gap.start_idx), A(gap.end_idx))} | " + " | ".join(parts))
            for v in VARIANTS:
                pick, why, n = pick_f4(df, fs, cfg, cands, thr, v)
                STATS[f"thr{thr} F4/{v} moved vs today"] += pick != real
                STATS[f"thr{thr} F4/{v} no IC"] += pick is None
                e_pick = pick_f4(df, fs, cfg, cands, thr, "E")[0]
                if pick != real or pick != e_pick:
                    out(f"   thr {thr} F4/{v}: passers {n} -> IC {A(pick)} {why}"
                        f"{'  MOVED vs today' if pick != real else ''}{'  DIFFERS from E' if pick != e_pick else ''}")
    return recs


orch.select_ic_records = _sir

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
print(text[-9000:])
