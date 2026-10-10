from __future__ import annotations
from pathlib import Path
import pandas as pd
from engine_v2.common.types import KLZone


def export_poi_zones(zones: list, path: str | Path) -> None:
    """Export POI zones (one row per POI = the IC picked for its fib record — a
    pattern's IC, its rank in `ip_rank`, or a ranked one with `ip_rank` empty
    (PLAN_L L2); `ip_outcome`, the legs, the tiebreak and the ranking in `meta`).

    POI zones are otherwise chart-only — this is the inspectable surface for
    `/compare` + debugging (mirrors `export_kl_zones`). Key meta fields are
    flattened out; the full meta is kept in the `meta` column. Columns are
    explicit so an empty list still writes a stable header.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in zones:
        m = getattr(z, "meta", {}) or {}
        rows.append({
            "ic_idx": getattr(z, "ic_idx", None),
            "start_time": getattr(z, "start_time", None),
            "end_time": getattr(z, "end_time", None),
            "side": getattr(z, "side", None),
            "top": getattr(z, "top", None),
            "bottom": getattr(z, "bottom", None),
            "structure_id": m.get("structure_id"),
            "cycle_id": m.get("cycle_id"),
            "parent_sid": m.get("parent_sid"),
            "parent_cycle_id": m.get("parent_cycle_id"),
            "confirmed_idx": m.get("confirmed_idx"),
            "ip_rank": m["ip_rank"],
            "status": m.get("status"),
            "meta": m,
        })
    cols = [
        "ic_idx", "start_time", "end_time", "side", "top", "bottom",
        "structure_id", "cycle_id", "parent_sid", "parent_cycle_id",
        "confirmed_idx", "ip_rank", "status", "meta",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


IC_CANDIDATE_COLUMNS = [
    "structure_path_id", "sub_id", "parent_sid", "parent_cycle_id", "structure_id", "cycle_id",
    "bos_anchor_idx", "cts_anchor_idx", "sd",
    "ic_idx", "candle_type", "agreed_dir", "touches_zone", "ic_ref_reason", "immediate_imbalance_first_c2_idx",
    "has_unfilled_imbalance_after", "is_ic_candidate",
    "ip_outcome", "ip_rank", "reason",
    "pattern_anchor_idx", "opp_leg_first_idx", "pattern_end_idx", "pattern_confirmed_idx", "ip_leg_candle_counts",
    "legs",
    "initial_type", "initial_ratio", "initial_body_pips",
    "opp_type", "opp_ratio", "opp_body_pips",
    "continuation_type", "continuation_ratio", "continuation_body_pips",
    "picked", "ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count", "ip_tb_imbalance_first_c2_idx",
    "ranking_rule", "ranking_runner_up_ic_idx", "ranking_in_play_count", "ranking_imbalance_first_c2_idx",
]
_IP_DETAIL_COLUMNS = [
    "initial_type", "initial_ratio", "initial_body_pips",
    "opp_type", "opp_ratio", "opp_body_pips",
    "continuation_type", "continuation_ratio", "continuation_body_pips",
]


def _ip_legs(ic_idx, counts):
    """`a..b|c..ic|ic+1..d` from the IC and the "initial-opp-continuation" counts."""
    if ic_idx is None or not counts:
        return None
    n_init, n_opp, n_cont = (int(x) for x in counts.split("-"))
    opp_first = ic_idx - n_opp + 1
    return f"{opp_first - n_init}..{opp_first - 1}|{opp_first}..{ic_idx}|{ic_idx + 1}..{ic_idx + n_cont}"


def export_ic_candidates(records: list, path: str | Path) -> None:
    """Export the IC selection (`poi_zones.ICCandidateRecord`; PLAN_H §8.3, PLAN_L
    L1): one row per Step 1 check of each gated fib record — every agreed-opposite
    candle of the fib's window with its condition outcomes (`touches_zone`,
    `ic_ref_reason`, `has_unfilled_imbalance_after`, `is_ic_candidate`) and its
    pattern (classified for every row; only an IC candidate is ever picked) and
    `ip_outcome` (Step 2's outcome, every row of the record) — + one no-IC row
    (`ic_idx` empty, `reason` `no_ic_candidate` / `all_ic_candidates_pinbar`) per
    record without a pick; the tiebreak / ranking fields on the picked row
    (PLAN_L L2). `legs` = `initial first..last|opp first..last|continuation
    first..last`, located from `ic_idx` + `ip_leg_candle_counts` (the legs are
    contiguous around the IC) — the pattern's, or the first failing variant's
    when `ip_rank` is empty; the stacks' type / big ratio / body pips per leg.
    Columns are explicit (an empty list writes the header); indices
    entity-absolute."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        m = r.meta or {}
        row = {
            "structure_path_id": m.get("structure_path_id"),
            "sub_id": m.get("sub_id"),
            "parent_sid": m.get("parent_sid"),
            "parent_cycle_id": m.get("parent_cycle_id"),
            "structure_id": r.structure_id,
            "cycle_id": r.cycle_id,
            "bos_anchor_idx": r.bos_anchor_idx,
            "cts_anchor_idx": r.cts_anchor_idx,
            "sd": r.struct_direction,
            "ic_idx": r.ic_idx,
            "candle_type": r.candle_type,
            "agreed_dir": r.agreed_dir,
            "touches_zone": r.touches_zone,
            "ic_ref_reason": r.ic_ref_reason,
            "immediate_imbalance_first_c2_idx": r.immediate_imbalance_first_c2_idx,
            "has_unfilled_imbalance_after": r.has_unfilled_imbalance_after,
            "is_ic_candidate": r.is_ic_candidate,
            "ip_outcome": r.ip_outcome,
            "ip_rank": r.ip_rank,
            "reason": r.reason,
            "pattern_anchor_idx": r.pattern_anchor_idx,
            "opp_leg_first_idx": r.opp_leg_first_idx,
            "pattern_end_idx": r.pattern_end_idx,
            "pattern_confirmed_idx": r.pattern_confirmed_idx,
            "ip_leg_candle_counts": r.ip_leg_candle_counts,
            "picked": r.picked,
            "ip_tiebreak": r.ip_tiebreak,
            "ip_tb_runner_up_ic_idx": r.ip_tb_runner_up_ic_idx,
            "ip_tied_count": r.ip_tied_count,
            "ip_tb_imbalance_first_c2_idx": r.ip_tb_imbalance_first_c2_idx,
            "ranking_rule": r.ranking_rule,
            "ranking_runner_up_ic_idx": r.ranking_runner_up_ic_idx,
            "ranking_in_play_count": r.ranking_in_play_count,
            "ranking_imbalance_first_c2_idx": r.ranking_imbalance_first_c2_idx,
        }
        row["legs"] = _ip_legs(r.ic_idx, r.ip_leg_candle_counts)
        for k in _IP_DETAIL_COLUMNS:
            row[k] = r.detail.get(k)
        rows.append(row)
    # dtype=object keeps ints as ints next to None (no 607.0)
    pd.DataFrame(rows, columns=IC_CANDIDATE_COLUMNS, dtype=object).to_csv(path, index=False)


def export_kl_zones(zones: list[KLZone], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "start_time": z.start_time,
        "end_time": z.end_time,
        "side": z.side,
        "top": z.top,
        "bottom": z.bottom,
        "source_kind": z.source_kind,
        "source_time": z.source_time,
        "source_price": z.source_price,
        "strength": z.strength,
        "meta": z.meta,
    } for z in zones])
    df.to_csv(path, index=False)
