"""One-off diagnostic for var3/var4 trigger over-firing investigation.

Runs the H1 main pipeline on the configured pair/range, then dumps three
CSVs describing the zone-proximity-trigger output:

  /tmp/zpt_per_trigger.csv    — one row per trigger in each cycle's
                                alternating list, with candle prices,
                                trigger_inner, gap_to_inner_pips,
                                candles_since_prior_trigger.

  /tmp/zpt_var3_windows.csv   — one row per var-3 (subsequent_confluence)
                                window: prior_sd_idx, this_cts_idx,
                                window_span, apex_idx (toward parent BOS),
                                apex_to_bos_inner_pips.

  /tmp/zpt_var4_windows.csv   — one row per var-4 (subsequent_counter)
                                Λ/V window: prior_sd_idx, cts_idx,
                                this_sd_idx, window_span, apex_idx
                                (toward parent CTS), apex_to_cts_inner_pips.

  /tmp/zpt_per_cycle.csv      — one row per parent (sid, cycle_id) with
                                BOS/CTS inner prices, BOS-CTS gap, var3
                                count, var4 count, max alternating-list
                                index reached.

Cross-reference the var3/var4 window CSVs against the H1 chart to decide
the tightening lever (A: min window span / B: min apex excursion /
C: POI-inclusion gate). Delete this file after the investigation.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from engine_v2.config import CONFIG
from engine_v2.pipeline.orchestrator import run_pipeline
from engine_v2.run_replay import fetch_history_with_auto_extend, _floor_to_day_start
from engine_v2.structure.structure_engine import _pip_size_from_pair


OUT_PER_TRIGGER = Path("/tmp/zpt_per_trigger.csv")
OUT_VAR3 = Path("/tmp/zpt_var3_windows.csv")
OUT_VAR4 = Path("/tmp/zpt_var4_windows.csv")
OUT_PER_CYCLE = Path("/tmp/zpt_per_cycle.csv")


def _bos_inner_for_cycle(kl_zones, sid: int, cycle_id: int):
    for z in kl_zones:
        if (z.meta.get("structure_id") == sid
                and z.meta.get("cycle_id") == cycle_id
                and z.source_kind == "BOS"):
            return z.meta.get("inner")
    return None


def _cts_inner_for_cycle(kl_zones, sid: int, cycle_id: int):
    for z in kl_zones:
        if (z.meta.get("structure_id") == sid
                and z.meta.get("cycle_id") == cycle_id
                and z.source_kind == "CTS"):
            return z.meta.get("inner")
    return None


def _struct_dir_for_cycle(events, sid: int, cycle_id: int) -> int:
    for ev in events:
        if (ev.type == "CTS_CONFIRMED"
                and ev.meta.get("structure_id") == sid
                and ev.meta.get("cycle_id") == cycle_id):
            return int(ev.meta.get("struct_direction", 0))
    return 0


def _gap_to_inner_pips(direction: str, sd: int, candle_low: float,
                        candle_high: float, trigger_inner: float,
                        pip_size: float) -> float:
    """Signed pips by which the candle penetrates / falls short of inner.

    Positive = beyond inner (deeper into zone). Negative = still outside.
    For sd buy (sd=+1, expect approach from above): penetration =
    trigger_inner - candle_low.
    For sd sell (sd=-1, expect approach from below): penetration =
    candle_high - trigger_inner.
    For opp_sd buy (CTS sits above): penetration = candle_high - inner.
    For opp_sd sell (CTS sits below): penetration = inner - candle_low.
    """
    if direction == "sd" and sd == 1:
        return (trigger_inner - candle_low) / pip_size
    if direction == "sd" and sd == -1:
        return (candle_high - trigger_inner) / pip_size
    if direction == "opp_sd" and sd == 1:
        return (candle_high - trigger_inner) / pip_size
    if direction == "opp_sd" and sd == -1:
        return (trigger_inner - candle_low) / pip_size
    return float("nan")


def main() -> None:
    df, effective_start, _ = fetch_history_with_auto_extend(
        pair=CONFIG.pair,
        timeframe=CONFIG.timeframe,
        start=_floor_to_day_start(CONFIG.start),
        end=CONFIG.end,
        lookback_days=183,
        min_history=50,
        extend_days=4,
        max_extend_iters=8,
    )
    df.attrs["pair"] = CONFIG.pair
    res = run_pipeline(df, lower_timeframes=())  # H1 only — we only care
                                                 # about parent triggers.
    h1_df = res.df
    pip_size = _pip_size_from_pair(h1_df)
    zpt: Dict[Tuple[int, int], List[Any]] = res.meta.get(
        "zone_proximity_triggers", {}
    )
    kl_zones = res.meta.get("kl_zones", [])
    events = res.meta.get("structure_events") or h1_df.attrs.get(
        "structure_events", []
    )

    OUT_PER_TRIGGER.parent.mkdir(parents=True, exist_ok=True)

    per_trigger_rows: List[Dict[str, Any]] = []
    var3_rows: List[Dict[str, Any]] = []
    var4_rows: List[Dict[str, Any]] = []
    per_cycle_rows: List[Dict[str, Any]] = []

    for (sid, cycle_id) in sorted(zpt.keys()):
        trigs = zpt[(sid, cycle_id)]
        sd = _struct_dir_for_cycle(events, sid, cycle_id)
        bos_inner = _bos_inner_for_cycle(kl_zones, sid, cycle_id)
        cts_inner = _cts_inner_for_cycle(kl_zones, sid, cycle_id)
        bos_cts_gap_pips = (
            abs(cts_inner - bos_inner) / pip_size
            if (bos_inner is not None and cts_inner is not None)
            else float("nan")
        )

        var3_count = 0
        var4_count = 0
        prior_idx = None
        for i, t in enumerate(trigs):
            row_idx = int(t.idx)
            row = h1_df.loc[row_idx]
            cl = float(row["l"])
            ch = float(row["h"])
            gap_in = _gap_to_inner_pips(
                t.direction, sd, cl, ch, float(t.trigger_inner), pip_size,
            )
            since_prior = (row_idx - prior_idx) if prior_idx is not None else None
            per_trigger_rows.append({
                "sid": sid,
                "cycle_id": cycle_id,
                "parent_sd": sd,
                "seq_in_cycle": i,
                "idx": row_idx,
                "direction": t.direction,
                "zone_kind": t.zone_kind,
                "trigger_inner": float(t.trigger_inner),
                "candle_low": cl,
                "candle_high": ch,
                "gap_to_inner_pips": round(gap_in, 2),
                "candles_since_prior": since_prior,
                "proximity_pips": int(t.proximity_pips),
                "bos_inner": bos_inner,
                "cts_inner": cts_inner,
                "bos_cts_gap_pips": round(bos_cts_gap_pips, 2)
                if bos_cts_gap_pips == bos_cts_gap_pips else None,
            })
            prior_idx = row_idx

            # var-3 window: opp_sd at index ≥ 1 (immediately preceded by sd)
            if i >= 1 and t.direction == "opp_sd":
                var3_count += 1
                prior_sd = trigs[i - 1]
                start = int(prior_sd.idx)
                end = int(t.idx)
                window = h1_df.loc[start:end]
                if not window.empty:
                    if sd == 1:
                        # toward BOS = lowest low (BOS sits below for buy)
                        ap_target = window["l"].astype(float).min()
                        ap_match = window[window["l"].astype(float) == ap_target]
                        ap_idx = int(ap_match.index[0])
                        ap_price = float(ap_target)
                        ap_to_bos = (
                            (ap_price - float(bos_inner)) / pip_size
                            if bos_inner is not None else float("nan")
                        )
                    else:
                        ap_target = window["h"].astype(float).max()
                        ap_match = window[window["h"].astype(float) == ap_target]
                        ap_idx = int(ap_match.index[0])
                        ap_price = float(ap_target)
                        ap_to_bos = (
                            (float(bos_inner) - ap_price) / pip_size
                            if bos_inner is not None else float("nan")
                        )
                else:
                    ap_idx, ap_price, ap_to_bos = end, float("nan"), float("nan")

                # apex_to_cts: how far did the cts trigger candle reach
                # toward CTS inner (positive = penetration; negative = short)
                if sd == 1:
                    cts_pen_pips = (ch - float(cts_inner)) / pip_size if cts_inner is not None else float("nan")
                else:
                    cts_pen_pips = (float(cts_inner) - cl) / pip_size if cts_inner is not None else float("nan")

                var3_rows.append({
                    "sid": sid,
                    "cycle_id": cycle_id,
                    "parent_sd": sd,
                    "seq_in_cycle": i,
                    "prior_sd_idx": start,
                    "prior_sd_zone_kind": prior_sd.zone_kind,
                    "this_cts_idx": end,
                    "window_span": end - start,
                    "apex_idx_toward_bos": ap_idx,
                    "apex_price": ap_price,
                    "apex_to_bos_inner_pips": round(ap_to_bos, 2)
                    if ap_to_bos == ap_to_bos else None,
                    "cts_trigger_penetration_pips": round(cts_pen_pips, 2)
                    if cts_pen_pips == cts_pen_pips else None,
                    "bos_inner": bos_inner,
                    "cts_inner": cts_inner,
                    "bos_cts_gap_pips": round(bos_cts_gap_pips, 2)
                    if bos_cts_gap_pips == bos_cts_gap_pips else None,
                })

            # var-4 window: sd at index ≥ 2 (the [-2, -1, this] = [sd, opp, sd])
            if i >= 2 and t.direction == "sd":
                var4_count += 1
                prior_sd = trigs[i - 2]
                cts_t = trigs[i - 1]
                start = int(prior_sd.idx)
                end = int(t.idx)
                window = h1_df.loc[start:end]
                if not window.empty:
                    if sd == 1:
                        # toward CTS = highest high (Λ apex)
                        ap_target = window["h"].astype(float).max()
                        ap_match = window[window["h"].astype(float) == ap_target]
                        ap_idx = int(ap_match.index[0])
                        ap_price = float(ap_target)
                        ap_to_cts = (
                            (ap_price - float(cts_inner)) / pip_size
                            if cts_inner is not None else float("nan")
                        )
                    else:
                        ap_target = window["l"].astype(float).min()
                        ap_match = window[window["l"].astype(float) == ap_target]
                        ap_idx = int(ap_match.index[0])
                        ap_price = float(ap_target)
                        ap_to_cts = (
                            (float(cts_inner) - ap_price) / pip_size
                            if cts_inner is not None else float("nan")
                        )
                else:
                    ap_idx, ap_price, ap_to_cts = end, float("nan"), float("nan")

                # this sd-trigger penetration into closest sd-inner
                if sd == 1:
                    sd_pen_pips = (float(t.trigger_inner) - cl) / pip_size
                else:
                    sd_pen_pips = (ch - float(t.trigger_inner)) / pip_size

                var4_rows.append({
                    "sid": sid,
                    "cycle_id": cycle_id,
                    "parent_sd": sd,
                    "seq_in_cycle": i,
                    "prior_sd_idx": start,
                    "prior_sd_zone_kind": prior_sd.zone_kind,
                    "cts_idx": int(cts_t.idx),
                    "this_sd_idx": end,
                    "this_sd_zone_kind": t.zone_kind,
                    "window_span_total": end - start,
                    "window_span_post_cts": end - int(cts_t.idx),
                    "apex_idx_toward_cts": ap_idx,
                    "apex_price": ap_price,
                    "apex_to_cts_inner_pips": round(ap_to_cts, 2)
                    if ap_to_cts == ap_to_cts else None,
                    "this_sd_trigger_penetration_pips": round(sd_pen_pips, 2),
                    "this_sd_trigger_inner": float(t.trigger_inner),
                    "bos_inner": bos_inner,
                    "cts_inner": cts_inner,
                    "bos_cts_gap_pips": round(bos_cts_gap_pips, 2)
                    if bos_cts_gap_pips == bos_cts_gap_pips else None,
                })

        per_cycle_rows.append({
            "sid": sid,
            "cycle_id": cycle_id,
            "parent_sd": sd,
            "n_triggers_alternating": len(trigs),
            "n_var3": var3_count,
            "n_var4": var4_count,
            "bos_inner": bos_inner,
            "cts_inner": cts_inner,
            "bos_cts_gap_pips": round(bos_cts_gap_pips, 2)
            if bos_cts_gap_pips == bos_cts_gap_pips else None,
        })

    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("(empty)\n", encoding="utf-8")
            print(f"[diag] {path}: empty")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[diag] {path}: {len(rows)} rows")

    _write_csv(OUT_PER_TRIGGER, per_trigger_rows)
    _write_csv(OUT_VAR3, var3_rows)
    _write_csv(OUT_VAR4, var4_rows)
    _write_csv(OUT_PER_CYCLE, per_cycle_rows)

    # Brief stdout summary
    print("\n=== Per-cycle counts ===")
    for r in per_cycle_rows:
        print(
            f"sid={r['sid']} cycle={r['cycle_id']} sd={r['parent_sd']:+d} "
            f"alt_list={r['n_triggers_alternating']:2d} "
            f"var3={r['n_var3']:2d} var4={r['n_var4']:2d} "
            f"BOS-CTS gap={r['bos_cts_gap_pips']}p"
        )


if __name__ == "__main__":
    main()
