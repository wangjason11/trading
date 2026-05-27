"""
Wave Volume Momentum Indicator (WVMI) — Week 8 Part 2.

Measures BOS zone strength by tracking volume momentum across wave cycles.
Triggered on-demand when CTS_n is confirmed, providing both a locked breakout
momentum and a shifting pullback momentum that locks when BOS_n+1 forms.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from engine_v2.common.types import KLZone, WVMIRecord
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.poi_zones import POIZone
from engine_v2.zones.structure_lifecycle import (
    compute_cycle_lifecycle,
    compute_reversal_idx_by_sid,
    compute_struct_start_by_sid,
)
from engine_v2.zones.wave_candles import WaveCandleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_last_wave_weight(df: pd.DataFrame, idx: int, wave_dir: int) -> float:
    """
    Weight for a last-wave candle (LB or LP).

    1.0 if (is_big_normal_as0 AND ctype in (maru, normal))
         OR (is_big_maru_as0 AND pinbar AND pinbar_dir != wave_dir)
    0.5 if round(body_pct * 100) <= 10
    0.7 otherwise
    """
    if idx not in df.index:
        return 0.7

    ctype = str(df.loc[idx, "candle_type"])
    is_big_normal = int(df.loc[idx, "is_big_normal_as0"]) == 1 if "is_big_normal_as0" in df.columns else False
    is_big_maru = int(df.loc[idx, "is_big_maru_as0"]) == 1 if "is_big_maru_as0" in df.columns else False

    if is_big_normal and ctype in ("maru", "normal"):
        return 1.0
    if is_big_maru and ctype == "pinbar":
        pdir = int(df.loc[idx, "pinbar_dir"]) if "pinbar_dir" in df.columns else 0
        if pdir != wave_dir:
            return 1.0

    body_pct = float(df.loc[idx, "body_pct"]) if "body_pct" in df.columns else 0.5
    if round(body_pct * 100) <= 10:
        return 0.5

    return 0.7


def _find_temporary_lp(
    df: pd.DataFrame,
    fp_idx: int,
    bos_zone: KLZone,
    search_start: int,
    search_end: int,
) -> Optional[int]:
    """
    Find qualified candle whose close is closest to BOS outer bound.

    Qualified = same direction & vol_dir as FP candle (or vol_dir == 0).
    Selection = min distance from close to zone outer.
    """
    if fp_idx not in df.index:
        return None

    fp_dir = int(df.loc[fp_idx, "direction"])
    fp_vol_dir = int(df.loc[fp_idx, "vol_dir"]) if "vol_dir" in df.columns else 0

    outer = bos_zone.meta.get("outer")
    if outer is None:
        return None
    outer = float(outer)

    best_idx = None
    best_dist = float("inf")

    for i in range(search_start, search_end + 1):
        if i not in df.index:
            continue

        # Must match FP direction
        if int(df.loc[i, "direction"]) != fp_dir:
            continue

        # Must match FP vol_dir (or candle's vol_dir == 0)
        candle_vol_dir = int(df.loc[i, "vol_dir"]) if "vol_dir" in df.columns else 0
        if candle_vol_dir != fp_dir and candle_vol_dir != 0:
            continue

        c = float(df.loc[i, "c"])
        d = abs(c - outer)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


def _find_wave_candle(
    wave_candles: List[WaveCandleResult],
    sid: int,
    cycle_id: int,
    source_kind: str,
) -> Optional[WaveCandleResult]:
    """Lookup wave candle result by (sid, cycle_id, source_kind)."""
    for wc in wave_candles:
        if wc.structure_id == sid and wc.cycle_id == cycle_id and wc.source_kind == source_kind:
            return wc
    return None


def _find_bos_zone(kl_zones: List[KLZone], sid: int, cycle_id: int) -> Optional[KLZone]:
    """Lookup BOS KL zone by (sid, cycle_id)."""
    for z in kl_zones:
        if (z.source_kind == "BOS"
                and z.meta.get("structure_id") == sid
                and z.meta.get("cycle_id") == cycle_id):
            return z
    return None


def _assign_direction_labels(
    zone_side: str,
    breakout: Optional[float],
    pullback: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Map breakout/pullback to buy/sell based on zone side.

    Buy zone: buy_momentum = breakout, sell_momentum = pullback
    Sell zone: buy_momentum = pullback, sell_momentum = breakout
    """
    def _r(v: Optional[float]) -> Optional[float]:
        return round(v, 2) if v is not None else None
    if zone_side == "buy":
        return _r(breakout), _r(pullback)
    else:
        return _r(pullback), _r(breakout)


# ---------------------------------------------------------------------------
# (Removed `check_proximity_activation` — replaced by the more general
#  `check_zone_proximity` in `zones/zone_proximity.py`. Orchestrator now
#  builds the WVMI gate dict from the first sd zone-proximity trigger
#  per cycle.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# WVMITracker
# ---------------------------------------------------------------------------

class WVMITracker:
    """
    Tracks Wave Volume Momentum Indicators across market structure cycles.

    Two orthogonal axes per record (WVMI_SPEC):
      - Computation (`lp_status`, mirrors FibTracker's create/lock):
          1. created at CTS_n confirmation — breakout locked, pullback shifting
          2. updated as candles arrive — temp LP shifts to closer qualified candle
          3. locked at BOS_n+1 confirmation — LP finalizes from BOS_n+1 wave candles
      - Lifecycle (`status` + `start_idx`/`end_idx`/`end_reason`): tier-1
        start/end, stamped once by `_finalize_lifecycle_fields` at end of sweep.

    One tracker per entity (Part 4 §9). `structure_path_id` is stamped on
    every record this tracker emits.
    """

    def __init__(self, structure_path_id: Optional[str] = None):
        self._records: Dict[tuple, WVMIRecord] = {}  # key: (sid, cycle_id)
        self._structure_path_id = structure_path_id
        # Creation idx per (sid, cycle_id) = CTS_n CONFIRMED event idx, captured
        # at on_cts_confirmed. The lifecycle `start_idx` source (clamped to the
        # struct floor in _finalize_lifecycle_fields). WVMI_SPEC "Fields (scalar)".
        self._creation_idx: Dict[tuple, int] = {}

    def on_cts_confirmed(
        self,
        cts_event: StructureEvent,
        df: pd.DataFrame,
        wave_candles: List[WaveCandleResult],
        kl_zones: List[KLZone],
    ) -> Optional[WVMIRecord]:
        """
        CTS_n confirmed -> create WVMI for BOS_n.

        Breakout momentum is locked at creation.
        Pullback momentum starts shifting (temporary LP).
        """
        sid = cts_event.meta.get("structure_id", 0)
        cycle_id = cts_event.meta.get("cycle_id", 0)

        # Find BOS_n zone
        bos_zone = _find_bos_zone(kl_zones, sid, cycle_id)
        if bos_zone is None:
            return None

        zone_side = bos_zone.side

        # Wave direction: buy zone → breakout=+1, pullback=-1; sell → breakout=-1, pullback=+1
        if zone_side == "buy":
            bo_wave_dir = 1
            pb_wave_dir = -1
        else:
            bo_wave_dir = -1
            pb_wave_dir = 1

        # Find BOS_n wave candle (first breakout = first_wave_candle_idx)
        bos_wc = _find_wave_candle(wave_candles, sid, cycle_id, "BOS")
        if bos_wc is None or bos_wc.first_wave_candle_idx is None:
            return None
        fb_idx = bos_wc.first_wave_candle_idx

        # Find CTS_n wave candle (last breakout = last_wave_candle_idx, first pullback = first_wave_candle_idx)
        cts_wc = _find_wave_candle(wave_candles, sid, cycle_id, "CTS")
        if cts_wc is None or cts_wc.last_wave_candle_idx is None or cts_wc.first_wave_candle_idx is None:
            return None
        lb_idx = cts_wc.last_wave_candle_idx
        fp_idx = cts_wc.first_wave_candle_idx

        # Get volumes
        fb_vol = float(df.loc[fb_idx, "volume"]) if fb_idx in df.index else None
        lb_vol = float(df.loc[lb_idx, "volume"]) if lb_idx in df.index else None
        fp_vol = float(df.loc[fp_idx, "volume"]) if fp_idx in df.index else None

        # Guard: missing volume or zero FB volume (division by zero)
        if fb_vol is None or lb_vol is None or fp_vol is None:
            return None
        if fb_vol == 0 or fp_vol == 0:
            return None

        # Compute LB weight
        lb_weight = _compute_last_wave_weight(df, lb_idx, bo_wave_dir)

        # Breakout momentum (LOCKED)
        breakout_momentum = (lb_vol * lb_weight) / fb_vol

        # Find temporary LP
        search_start = fp_idx + 1
        search_end = len(df) - 1
        lp_idx = _find_temporary_lp(df, fp_idx, bos_zone, search_start, search_end)

        # Compute pullback momentum if temp LP exists
        lp_vol = None
        lp_weight = 1.0
        pullback_momentum = None
        if lp_idx is not None and lp_idx in df.index:
            lp_vol = float(df.loc[lp_idx, "volume"])
            lp_weight = _compute_last_wave_weight(df, lp_idx, pb_wave_dir)
            if lp_vol is not None and fp_vol > 0:
                pullback_momentum = (lp_vol * lp_weight) / fp_vol

        # Assign buy/sell labels
        buy_mom, sell_mom = _assign_direction_labels(zone_side, breakout_momentum, pullback_momentum)

        record = WVMIRecord(
            bos_structure_id=sid,
            bos_cycle_id=cycle_id,
            zone_side=zone_side,
            structure_path_id=self._structure_path_id,
            fb_idx=fb_idx,
            lb_idx=lb_idx,
            fp_idx=fp_idx,
            lp_idx=lp_idx,
            fb_volume=fb_vol,
            lb_volume=lb_vol,
            fp_volume=fp_vol,
            lp_volume=lp_vol,
            lb_weight=lb_weight,
            lp_weight=lp_weight,
            breakout_momentum=breakout_momentum,
            pullback_momentum=pullback_momentum,
            buy_momentum=buy_mom,
            sell_momentum=sell_mom,
            lp_status="created",
            lp_locked=False,
        )

        key = (sid, cycle_id)
        self._records[key] = record
        # Creation idx = the CTS_n CONFIRMED event idx (== confirmed_at, the
        # confirmation candle — when the record is born and breakout momentum
        # locks). Sticky per cycle; the lifecycle start_idx source.
        self._creation_idx.setdefault(key, int(cts_event.idx))
        print(f"[wvmi] CREATED sid={sid} cycle={cycle_id}: bo_mom={breakout_momentum:.4f} pb_mom={pullback_momentum if pullback_momentum is not None else 'N/A'} lp_idx={lp_idx}")
        return record

    def update_temporary_lp(self, df: pd.DataFrame, kl_zones: List[KLZone]) -> List[WVMIRecord]:
        """Re-scan all non-locked records, update temp LP closest to outer."""
        updated = []
        for key, rec in self._records.items():
            if rec.lp_locked:
                continue
            if rec.fp_idx is None:
                continue

            sid, cycle_id = key
            bos_zone = _find_bos_zone(kl_zones, sid, cycle_id)
            if bos_zone is None:
                continue

            # Determine pullback wave direction
            pb_wave_dir = -1 if rec.zone_side == "buy" else 1

            search_start = rec.fp_idx + 1
            search_end = len(df) - 1
            new_lp_idx = _find_temporary_lp(df, rec.fp_idx, bos_zone, search_start, search_end)

            if new_lp_idx == rec.lp_idx:
                continue

            # Update record
            rec.lp_idx = new_lp_idx
            if new_lp_idx is not None and new_lp_idx in df.index:
                rec.lp_volume = float(df.loc[new_lp_idx, "volume"])
                rec.lp_weight = _compute_last_wave_weight(df, new_lp_idx, pb_wave_dir)
                if rec.fp_volume and rec.fp_volume > 0:
                    rec.pullback_momentum = (rec.lp_volume * rec.lp_weight) / rec.fp_volume
                else:
                    rec.pullback_momentum = None
            else:
                rec.lp_volume = None
                rec.lp_weight = 1.0
                rec.pullback_momentum = None

            rec.buy_momentum, rec.sell_momentum = _assign_direction_labels(
                rec.zone_side, rec.breakout_momentum, rec.pullback_momentum
            )
            rec.lp_status = "updated"
            updated.append(rec)

        return updated

    def on_bos_confirmed(
        self,
        bos_event: StructureEvent,
        df: pd.DataFrame,
        wave_candles: List[WaveCandleResult],
    ) -> List[WVMIRecord]:
        """
        BOS_n+1 confirmed -> lock BOS_n's WVMI pullback momentum.

        Replace temp LP with official LP from BOS_n+1 wave candles (last_wave_candle_idx).
        """
        sid = bos_event.meta.get("structure_id", 0)
        bos_cycle_id = bos_event.meta.get("cycle_id", 0)

        # BOS with cycle_id=N+1 locks WVMI for cycle_id=N
        prev_cycle_id = bos_cycle_id - 1
        if prev_cycle_id < 0:
            return []

        locked = []
        key = (sid, prev_cycle_id)
        rec = self._records.get(key)
        if rec is None or rec.lp_locked:
            return locked

        # Determine pullback wave direction
        pb_wave_dir = -1 if rec.zone_side == "buy" else 1

        # Try to get LP from BOS_n+1 wave candle (last_wave_candle_idx)
        bos_wc = _find_wave_candle(wave_candles, sid, bos_cycle_id, "BOS")
        if bos_wc is not None and bos_wc.last_wave_candle_idx is not None:
            lp_idx = bos_wc.last_wave_candle_idx
            rec.lp_idx = lp_idx
            if lp_idx in df.index:
                rec.lp_volume = float(df.loc[lp_idx, "volume"])
                rec.lp_weight = _compute_last_wave_weight(df, lp_idx, pb_wave_dir)
                if rec.fp_volume and rec.fp_volume > 0:
                    rec.pullback_momentum = (rec.lp_volume * rec.lp_weight) / rec.fp_volume
                else:
                    rec.pullback_momentum = None
            else:
                rec.lp_volume = None
                rec.lp_weight = 1.0
                rec.pullback_momentum = None
        # else: lock with existing temp LP as final

        rec.buy_momentum, rec.sell_momentum = _assign_direction_labels(
            rec.zone_side, rec.breakout_momentum, rec.pullback_momentum
        )
        rec.lp_locked = True
        rec.lp_status = "locked"
        rec.locked_by_cycle_id = bos_cycle_id
        locked.append(rec)
        print(f"[wvmi] LOCKED sid={sid} cycle={prev_cycle_id}: pb_mom={rec.pullback_momentum if rec.pullback_momentum is not None else 'N/A'} lp_idx={rec.lp_idx} (by BOS cycle={bos_cycle_id})")

        return locked

    def _finalize_lifecycle_fields(
        self,
        events: List[StructureEvent],
        lifecycle_floor: Optional[int] = None,
        lifecycle_cap: Optional[int] = None,
        cap_reason: str = "lifecycle_end",
    ) -> None:
        """Stamp the scalar lifecycle axes onto every WVMI record.

        Mirrors `FibTracker._finalize_lifecycle_fields` (FIB_LIFECYCLE_SPEC §15.6)
        but TIER-1 — WVMI is created-once / locked-once with NO reversible
        condition axis (WVMI_SPEC "Tier-1: NO active/inactive axis"), so there is
        no `active` bool and no `activation_history`. Per record (keyed by its
        cycle identity `(sid, cycle_id)`):

          - `start_idx` = the creation idx (CTS_n CONFIRMED, captured in
            `_creation_idx`) CLAMPED UP to the structure/parent floor
            (`compute_struct_start_by_sid`). Like fib, the clamp is to the
            STRUCTURE floor (not the full cycle-start) — it only raises a start
            that precedes the parent floor (the sub B1 gap); main is a no-op
            (CTS_CONFIRMED always follows the structure's first event).
          - `end_idx`/`end_reason` = INHERITED from
            `compute_cycle_lifecycle[(sid,cycle)]` — the cycle pass-through end
            (next-cycle clamped start / reversal / sub cap). Data-end is NOT a
            terminator (PART4 §5): an open last / single-cycle WVMI gets
            `end_idx=None` → stays `active` to the edge.
          - `status` (derived): collapsed (clamped start ≥ end) or no creation
            idx → "inactive"; `end_idx` set → "ended"; else "active".

        The computation axis (`lp_status`/`lp_locked`/`locked_by_cycle_id`) is
        untouched — orthogonal (a record can be `ended` yet not `lp_locked`).

        Coordinate space follows `events`: the main tracker is finalized with the
        entity-absolute H1 events (floor/cap None); a sub tracker is finalized
        with the sub's slice-local events + slice-local floor/cap, and
        `persist_facade_wvmi_to_entity_df` then shifts `start_idx`/`end_idx` to
        entity-absolute alongside the wave-candle idx fields.

        Idempotent; call once after the create/lock/update sweep.
        """
        rev = compute_reversal_idx_by_sid(events)
        struct_floor = compute_struct_start_by_sid(events, rev, lifecycle_floor)
        cycle_life = compute_cycle_lifecycle(
            events, rev, lifecycle_floor, lifecycle_cap, cap_reason,
        )

        for (sid, cycle_id), rec in self._records.items():
            start_idx = self._creation_idx.get((sid, cycle_id))
            if start_idx is not None:
                sfloor = struct_floor.get(sid)
                if sfloor is not None and int(sfloor) > start_idx:
                    start_idx = int(sfloor)

            life = cycle_life.get((sid, cycle_id))
            end_idx = life[1] if life is not None else None
            end_reason = life[2] if life is not None else None

            # Collapse (mirrors KL/POI/fib): a clamped start at/after the cycle
            # end means the cycle had no live window → inactive.
            if start_idx is not None and end_idx is not None and start_idx >= end_idx:
                start_idx = None

            if start_idx is None:
                status = "inactive"
            elif end_idx is not None:
                status = "ended"
            else:
                status = "active"

            rec.start_idx = start_idx
            rec.end_idx = end_idx
            rec.end_reason = end_reason
            rec.status = status

    def get_records(self) -> List[WVMIRecord]:
        """Return all WVMI records."""
        return list(self._records.values())
