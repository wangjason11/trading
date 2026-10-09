from __future__ import annotations
from pathlib import Path
import pandas as pd
from engine_v2.common.types import KLZone


def export_poi_zones(zones: list, path: str | Path) -> None:
    """Export POI zones (one row per POI = the institutional-pattern pick of its
    fib record; its rank in `ip_rank`, the legs / tiebreak in `meta` — PLAN_H B).

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


IP_CANDIDATE_COLUMNS = [
    "structure_path_id", "sub_id", "parent_sid", "parent_cycle_id", "structure_id", "cycle_id",
    "bos_anchor_idx", "cts_anchor_idx", "sd",
    "ic_idx", "candle_type", "agreed_dir", "touches_zone", "ic_ref_reason", "ip_rank", "reason",
    "pattern_anchor_idx", "opp_leg_first_idx", "pattern_end_idx", "pattern_confirmed_idx", "ip_leg_candle_counts",
    "legs",
    "initial_type", "initial_ratio", "initial_body_pips",
    "opp_type", "opp_ratio", "opp_body_pips",
    "continuation_type", "continuation_ratio", "continuation_body_pips",
    "picked", "ip_tiebreak", "ip_tb_runner_up_ic_idx", "ip_tied_count", "ip_tb_imbalance_first_c2_idx",
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


def export_ip_candidates(records: list, path: str | Path) -> None:
    """Export the institutional-pattern selection (`poi_zones.IPCandidateRecord`,
    PLAN_H §8.3 landing A — computed, not yet drawn): one row per §3.1 candidate
    of each gated fib record (`touches_zone` False rows are classified for the
    record only, never picked) + one `IP: none` row (`ic_idx` empty) per record
    without a pick. `legs` = `initial first..last|opp first..last|continuation
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
        }
        row["legs"] = _ip_legs(r.ic_idx, r.ip_leg_candle_counts)
        for k in _IP_DETAIL_COLUMNS:
            row[k] = r.detail.get(k)
        rows.append(row)
    # dtype=object keeps ints as ints next to None (no 607.0)
    pd.DataFrame(rows, columns=IP_CANDIDATE_COLUMNS, dtype=object).to_csv(path, index=False)


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
