"""Read-only SHADOW (no engine edit) for the user's round-4 proposal (2026-10-21): "require IC to be both opposite in
candle direction and opposite in volume direction of structure direction". Today IC condition 1 = the AGREED direction
is −sd (`agreed_direction`, GLOSSARY "Agreed direction"): candle direction ±1 with volume direction the same OR 0 →
the candle's direction; candle direction 0 (a doji) with volume direction ±1 → the volume direction. So two candle
kinds pass today that the literal proposal would drop:
    VOL0   candle direction −sd, volume direction 0 (close == the previous close)
    DOJI   candle direction 0, volume direction −sd (a volume-only doji)
Variants (condition 1's row set only — the institutional pattern still reads the unchanged agreed array):
    STRICT  both dropped (the proposal, literally)
    NOVOL0  VOL0 dropped only
    NODOJI  DOJI dropped only
Per drawn record: the Step 1 rows / IC candidates / picks of each kind, and the pick under each variant (Step 1A
included — it runs when a variant leaves Step 1 with no candidate). Plus MS's in-flight read (`poi_inner_for_fib`):
reads whose inner would differ under STRICT (first order).

    python engine_v2/plans/qa_ic_revisit_inputs/shadow_ic_direction.py OUT_LOG      (repo root)
"""
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.pipeline.orchestrator as orch  # noqa: E402
import engine_v2.structure.structure_engine as se  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402

VARIANTS = ("STRICT", "NOVOL0", "NODOJI")
MODE = ["today"]
LINES = []
CTX = []
STATS = Counter()
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

_real_cic = pz.check_ic_conditions


def _cic(df, fib_state, config, *, ic_reference, agreed):
    if MODE[0] != "today":
        d = df["direction"].to_numpy().astype(int)
        v = df["vol_dir"].to_numpy().astype(int)
        keep = np.ones(len(d), dtype=bool)
        if MODE[0] in ("STRICT", "NODOJI"):
            keep &= d != 0
        if MODE[0] in ("STRICT", "NOVOL0"):
            keep &= v != 0
        agreed = np.where(keep, agreed, 2)
    return _real_cic(df, fib_state, config, ic_reference=ic_reference, agreed=agreed)


pz.check_ic_conditions = _cic


def kind(df, i):
    d, v = int(df.loc[i, "direction"]), int(df.loc[i, "vol_dir"])
    return "DOJI" if d == 0 else ("VOL0" if v == 0 else "both")


def select(df, fs, ref, mode, classify_all=True):
    MODE[0] = mode
    try:
        return pz.select_ic(df, fs, pz.POIConfig(), ic_reference=ref, classify_all=classify_all)
    finally:
        MODE[0] = "today"


_real_sir = orch.select_ic_records


def _sir(df, events, fib_tracker, config, *, lifecycle_cap=None):
    recs = _real_sir(df, events, fib_tracker, config, lifecycle_cap=lifecycle_cap)
    c = CTX[-1] if CTX else None
    tf = df.attrs.get("timeframe")
    if c is None and tf != "H1":
        return recs  # a capped proximity-overlay view — not a drawn record
    ent = "H1main" if c is None else f"sub{c[0]}"
    b = 0 if c is None else c[1]

    def A(i):
        return None if i is None else int(i) + b

    for r in recs:
        fs = r.fib_state
        m = fs.meta
        key = (ent, int(fs.structure_id), int(fs.cycle_id), m.get("version"), A(fs.bos_idx), A(fs.cts_idx))
        if key in SEEN:
            continue
        SEEN.add(key)
        STATS["drawn records"] += 1
        ref = fib_tracker.ic_reference(fs, df, evaluated_at=lifecycle_cap)
        sel = r.selection
        real = None if sel.pick is None else int(sel.pick.ic_idx)
        assert (None if select(df, fs, ref, "today").pick is None else int(select(df, fs, ref, "today").pick.ic_idx)) == real
        for ch in sel.checks:
            k = kind(df, ch.ic_idx)
            STATS[f"Step-1 rows: {k}"] += 1
            if ch.is_ic_candidate:
                STATS[f"IC candidates: {k}"] += 1
                if k != "both":
                    out(f"   {ent} sid{fs.structure_id} c{fs.cycle_id} candidate {A(ch.ic_idx)} {k} "
                        f"dir {int(df.loc[ch.ic_idx, 'direction']):+d} vol {int(df.loc[ch.ic_idx, 'vol_dir']):+d} "
                        f"{df.loc[ch.ic_idx, 'candle_type']}{'  PICKED' if ch.ic_idx == real else ''}")
        if real is not None:
            STATS[f"picks: {kind(df, real)}"] += 1
        for v in VARIANTS:
            s2 = select(df, fs, ref, v)
            p2 = None if s2.pick is None else int(s2.pick.ic_idx)
            STATS[f"{v}: picks moved"] += p2 != real
            STATS[f"{v}: no IC"] += p2 is None
            if p2 != real or s2.ic_zone_band != sel.ic_zone_band:
                out(f"   {v}: {ent} sid{fs.structure_id} c{fs.cycle_id} IC {A(real)} -> {A(p2)} band {sel.ic_zone_band} -> "
                    f"{s2.ic_zone_band} ({s2.no_ic_reason or s2.ip_outcome})")
    return recs


orch.select_ic_records = _sir

_real_pif = se.poi_inner_for_fib


def _pif(df, fib_state, fill_threshold, *, tracker, evaluated_at):
    res = _real_pif(df, fib_state, fill_threshold, tracker=tracker, evaluated_at=evaluated_at)
    if fib_state is None:
        return res
    tf = df.attrs.get("timeframe")
    STATS[f"MS {tf} reads"] += 1
    ref = tracker.ic_reference(fib_state, df, evaluated_at=int(evaluated_at))
    s2 = select(df, fib_state, ref, "STRICT", classify_all=False)
    sd = int(fib_state.struct_direction)
    inner = None if s2.pick is None else float(df.loc[int(s2.pick.ic_idx), "h" if sd == 1 else "l"])
    STATS[f"MS {tf} reads whose inner differs under STRICT"] += inner != res
    return res


se.poi_inner_for_fib = _pif

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
print(text[-8000:])
