"""Read-only SHADOW (2026-10-13): the POI activation sweep's imbalance condition under change 1. Run on the
working tree as it stands (the combined TRIAL: changes 1 + 2 + 3). Per drawn POI, the activation history:

  NOW   the sweep as built — condition 3 = ANY sd imbalance formed by t that overlaps (ic_idx, t] and is not
        committed-filled (`_compute_poi_activation_history`: the old IC condition 3, re-checked per candle)
  IMM   the same sweep with only the IC's IMMEDIATE gap counted (the sd instance whose first c2 is in IC+1 … IC+2 —
        change 1's Step 1 condition 3): the POI drops `imbalance_filled` once that gap commit-fills

Only the H1 POIs feed anything downstream (`orchestrator` → `check_zone_proximity` → `poi_active_as_of` → the
counter / confluence triggers); MS's in-flight read ignores activation; sub POIs' histories are chart-only.

    python <this> [OUT_LOG]      (repo root; ~1 min)
"""
import sys
import time

sys.path.insert(0, ".")
OUT = sys.argv[1] if len(sys.argv) > 1 else None
import engine_v2.run_replay as rr  # noqa: E402
import engine_v2.multitf.entity_df_mutation as edm  # noqa: E402
import engine_v2.zones.poi_zones as pz  # noqa: E402

LINES = []
CTX = []
SEEN = set()


def out(s):
    LINES.append(s)


_real_rsp = edm.render_sub_projection


def _rsp(sub, *a, **k):
    CTX.append(int(sub.sub_id))
    try:
        return _real_rsp(sub, *a, **k)
    finally:
        CTX.pop()


edm.render_sub_projection = _rsp
_real_hist = pz._compute_poi_activation_history


def tm(df, i):
    return str(df.loc[i, "time"])[:16] if "time" in df.columns and i in df.index else "?"


def fmt(df, h):
    return " ".join(f"{'A' if e['active'] else 'D'}@{e['idx']}({tm(df, e['idx'])[5:]}){'' if e['active'] else ':' + e['reason']}"
                    for e in h) or "never active"


def _hist(df, **kw):
    now = _real_hist(df, **kw)
    ic, sd = int(kw["ic_idx"]), int(kw["sd"])
    imm = [inst for inst in kw["imbalances"]
           if inst.direction == sd and inst.gap_size > 0 and ic + 1 <= inst.start_idx <= ic + 2]
    assert len(imm) == 1, (ic, [(i.start_idx, i.end_idx) for i in imm])
    alt = _real_hist(df, **{**kw, "imbalances": imm})
    ent = f"sub {CTX[-1]}" if CTX else "H1"
    key = (ent, ic, int(kw["cts_established_idx"]))
    if key not in SEEN:
        SEEN.add(key)
        g = imm[0]
        armed, fill = kw["fill_idx_cache"].get(id(g), (None, None))
        out(f"[{ent}] IC {ic} ({tm(df, ic)}) gap {g.start_idx}..{g.end_idx} formed {g.formed_at} armed {armed} "
            f"commit-filled {fill} ({tm(df, fill) if fill is not None else '-'}) scan_end {kw['scan_end']} "
            f"{'SAME' if now == alt else 'DIFFERS'}")
        out(f"    NOW  {fmt(df, now)}")
        if now != alt:
            out(f"    IMM  {fmt(df, alt)}")
    return now


pz._compute_poi_activation_history = _hist

t0 = time.perf_counter()
df, effective_start, _d = rr.fetch_history_with_auto_extend(
    pair=rr.CONFIG.pair, timeframe=rr.CONFIG.timeframe, start=rr._floor_to_day_start(rr.CONFIG.start),
    end=rr.CONFIG.end, lookback_days=183, min_history=50, extend_days=4, max_extend_iters=8,
)
df.attrs["pair"] = rr.CONFIG.pair
res = rr.run_pipeline(df, lower_timeframes=rr.CONFIG.lower_timeframes)
out(f"[done] {len(SEEN)} POIs, {sum('DIFFERS' in s for s in LINES)} differ, {time.perf_counter() - t0:.1f} s")
text = "\n".join(LINES)
if OUT:
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
print(text)
