"""
Wave Candle Identification for Volume Momentum (Week 8 Part 2 foundation).

For each KL zone at a BOS/CTS anchor, identifies:
- last_wave_candle_idx: final candle of the ending wave
- first_wave_candle_idx: initial candle of the starting wave

These feed into downstream volume momentum logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaveCandleResult:
    structure_id: int
    cycle_id: int
    source_kind: Literal["BOS", "CTS"]
    zone_side: Literal["buy", "sell"]
    last_wave_candle_idx: Optional[int]
    first_wave_candle_idx: Optional[int]
    meta: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Zone geometry helpers
# ---------------------------------------------------------------------------

def _is_qualified(df: pd.DataFrame, idx: int, required_dir: int) -> bool:
    """Candle direction == required_dir."""
    if idx not in df.index:
        return False
    return int(df.loc[idx, "direction"]) == required_dir


def _touches_zone(df: pd.DataFrame, idx: int, zone: KLZone) -> bool:
    """Buy zone: low <= zone.top; Sell zone: high >= zone.bottom."""
    if idx not in df.index:
        return False
    if zone.side == "buy":
        return float(df.loc[idx, "l"]) <= zone.top
    else:
        return float(df.loc[idx, "h"]) >= zone.bottom


def _closes_within_zone(df: pd.DataFrame, idx: int, zone: KLZone) -> bool:
    """zone.bottom <= close <= zone.top."""
    if idx not in df.index:
        return False
    c = float(df.loc[idx, "c"])
    return zone.bottom <= c <= zone.top


def _close_distance_to_outer(df: pd.DataFrame, idx: int, zone: KLZone) -> float:
    """abs(close - outer). outer = zone.meta["outer"]."""
    if idx not in df.index:
        return float("inf")
    outer = zone.meta.get("outer")
    if outer is None:
        return float("inf")
    return abs(float(df.loc[idx, "c"]) - float(outer))


def _wick_enters_zone(df: pd.DataFrame, idx: int, zone: KLZone) -> bool:
    """Check if the candle's wick enters the zone (same as _touches_zone)."""
    return _touches_zone(df, idx, zone)


def _is_compound_first_wave(df: pd.DataFrame, idx: int, wave_dir: int) -> bool:
    """
    Compound first-wave candle condition:
    (is_big_normal_as0 AND type in (maru, normal)) OR
    (is_big_maru_as0 AND type == pinbar AND pinbar_dir == wave_dir)
    """
    if idx not in df.index:
        return False

    ctype = str(df.loc[idx, "candle_type"])
    is_big_normal = int(df.loc[idx, "is_big_normal_as0"]) == 1 if "is_big_normal_as0" in df.columns else False
    is_big_maru = int(df.loc[idx, "is_big_maru_as0"]) == 1 if "is_big_maru_as0" in df.columns else False

    if is_big_normal and ctype in ("maru", "normal"):
        return True
    if is_big_maru and ctype == "pinbar":
        pdir = int(df.loc[idx, "pinbar_dir"]) if "pinbar_dir" in df.columns else 0
        return pdir == wave_dir
    return False


# ---------------------------------------------------------------------------
# Event query helpers
# ---------------------------------------------------------------------------

def _find_cts_established(events: List[StructureEvent], sid: int, cycle_id: int) -> Optional[StructureEvent]:
    """Return CTS_ESTABLISHED event for (sid, cycle_id)."""
    for ev in events:
        if ev.type == "CTS_ESTABLISHED":
            if ev.meta.get("structure_id") == sid and ev.meta.get("cycle_id") == cycle_id:
                return ev
    return None


def _find_cts_updated_events(events: List[StructureEvent], sid: int, cycle_id: int) -> List[StructureEvent]:
    """Return sorted list of CTS_UPDATED events for (sid, cycle_id)."""
    result = []
    for ev in events:
        if ev.type == "CTS_UPDATED":
            if ev.meta.get("structure_id") == sid and ev.meta.get("cycle_id") == cycle_id:
                result.append(ev)
    return sorted(result, key=lambda e: e.idx)


def _find_cts_confirmed(events: List[StructureEvent], sid: int, cycle_id: int) -> Optional[StructureEvent]:
    """Return CTS_CONFIRMED event for (sid, cycle_id)."""
    for ev in events:
        if ev.type == "CTS_CONFIRMED":
            if ev.meta.get("structure_id") == sid and ev.meta.get("cycle_id") == cycle_id:
                return ev
    return None


def _find_last_pullback_idx(events: List[StructureEvent], sid: int, before_idx: int) -> Optional[int]:
    """Last STATE_CHANGED to='pullback' for this sid, before before_idx."""
    best = None
    for ev in events:
        if ev.type != "STATE_CHANGED":
            continue
        if ev.meta.get("structure_id") != sid:
            continue
        if ev.meta.get("to") != "pullback":
            continue
        if ev.idx < before_idx:
            if best is None or ev.idx > best:
                best = ev.idx
    return best


def _find_prior_cts_anchor(events: List[StructureEvent], sid: int, cycle_id: int) -> Optional[int]:
    """CTS_CONFIRMED.meta['cts_anchor_idx'] for cycle_id-1."""
    if cycle_id <= 0:
        return None
    prev_confirmed = _find_cts_confirmed(events, sid, cycle_id - 1)
    if prev_confirmed is None:
        return None
    return prev_confirmed.meta.get("cts_anchor_idx")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def identify_wave_candles(
    anchor_idx: int,
    anchor_type: Literal["BOS", "CTS"],
    zone: KLZone,
    events: List[StructureEvent],
    structure_id: int,
    struct_direction: int,
    df: pd.DataFrame,
) -> Optional[WaveCandleResult]:
    """
    Identify wave boundary candles for a KL zone.

    Parameters
    ----------
    anchor_idx : int
        The anchor candle index of the zone base pattern.
    anchor_type : "BOS" or "CTS"
        Which type of zone this is.
    zone : KLZone
        The KL zone to find wave candles for.
    events : list of StructureEvent
        All structure events.
    structure_id : int
        Structure id for this zone.
    struct_direction : int
        Structure direction (+1 bullish, -1 bearish).
    df : pd.DataFrame
        OHLC dataframe with candle features.

    Returns
    -------
    WaveCandleResult or None
    """
    cycle_id = zone.meta.get("cycle_id", 0)

    if anchor_type == "BOS":
        return _bos_wave_candles(anchor_idx, zone, events, structure_id, cycle_id, struct_direction, df)
    else:
        return _cts_wave_candles(anchor_idx, zone, events, structure_id, cycle_id, struct_direction, df)


# ---------------------------------------------------------------------------
# BOS wave candles
# ---------------------------------------------------------------------------

def _bos_wave_candles(
    anchor_idx: int,
    zone: KLZone,
    events: List[StructureEvent],
    sid: int,
    cycle_id: int,
    sd: int,
    df: pd.DataFrame,
) -> Optional[WaveCandleResult]:
    """
    BOS zone wave candles.
    - Last pullback candle: opposite direction to zone side
    - First breakout candle: same direction as zone side
    """
    base_pattern = zone.meta.get("base_pattern", "")

    # Direction mapping
    # Buy zone (sd=+1): last pullback dir = -1, first breakout dir = +1
    # Sell zone (sd=-1): last pullback dir = +1, first breakout dir = -1
    if zone.side == "buy":
        pb_dir = -1
        bo_dir = 1
    else:
        pb_dir = 1
        bo_dir = -1

    # --- Last pullback candle ---
    last_pb_idx = None

    if base_pattern == "base inside bar":
        last_pb_idx = _bos_bib_last_pullback(anchor_idx, zone, events, sid, cycle_id, pb_dir, df)
    else:
        last_pb_idx = _bos_non_bib_last_pullback(anchor_idx, zone, pb_dir, df)

    # --- First breakout candle ---
    first_bo_idx = None
    if last_pb_idx is not None:
        # Scan forward from last_pb_idx+1
        # Upper bound: next CTS confirmed or end of structure
        cts_conf = _find_cts_confirmed(events, sid, cycle_id)
        upper = int(cts_conf.meta.get("confirmed_at", len(df) - 1)) if cts_conf else len(df) - 1
        upper = min(upper, len(df) - 1)

        for i in range(last_pb_idx + 1, upper + 1):
            if i not in df.index:
                continue
            if int(df.loc[i, "direction"]) != bo_dir:
                continue
            if _is_compound_first_wave(df, i, bo_dir):
                first_bo_idx = i
                break

    return WaveCandleResult(
        structure_id=sid,
        cycle_id=cycle_id,
        source_kind="BOS",
        zone_side=zone.side,
        last_wave_candle_idx=last_pb_idx,
        first_wave_candle_idx=first_bo_idx,
        meta={
            "base_pattern": base_pattern,
            "anchor_idx": anchor_idx,
        },
    )


def _bos_bib_last_pullback(
    anchor_idx: int,
    zone: KLZone,
    events: List[StructureEvent],
    sid: int,
    cycle_id: int,
    pb_dir: int,
    df: pd.DataFrame,
) -> Optional[int]:
    """
    BOS + base inside bar: last pullback candle.
    1. Forward search: (anchor_idx+1, first_breakout_anchor).
       Accept only if closer to outer than BOS candle's close.
    2. Backward fallback: [most_recent_pb_idx, anchor_idx].
    """
    # Forward search
    cts_est = _find_cts_established(events, sid, cycle_id)
    if cts_est is not None:
        first_bo_anchor = cts_est.meta.get("anchor_idx")
        if first_bo_anchor is not None:
            first_bo_anchor = int(first_bo_anchor)
            # BOS candle's close distance to outer
            bos_dist = _close_distance_to_outer(df, anchor_idx, zone)

            best_idx = None
            best_dist = float("inf")
            for i in range(anchor_idx + 1, first_bo_anchor):
                if i not in df.index:
                    continue
                if not _is_qualified(df, i, pb_dir):
                    continue
                d = _close_distance_to_outer(df, i, zone)
                if d < best_dist:
                    best_dist = d
                    best_idx = i

            # Accept only if closer to outer than BOS candle's close
            if best_idx is not None and best_dist < bos_dist:
                return best_idx

    # Backward fallback
    pb_state_idx = _find_last_pullback_idx(events, sid, anchor_idx)
    if pb_state_idx is None:
        # Use prior CTS anchor
        pb_state_idx = _find_prior_cts_anchor(events, sid, cycle_id)
    if pb_state_idx is None:
        # Cycle 0: no prior pullback/CTS, cap at 5 candles back
        # Cycle 1+: wider 10-candle guard
        lookback = 5 if cycle_id == 0 else 20
        pb_state_idx = max(0, anchor_idx - lookback)

    best_idx = None
    best_dist = float("inf")
    for i in range(int(pb_state_idx), anchor_idx + 1):
        if i not in df.index:
            continue
        if not _is_qualified(df, i, pb_dir):
            continue
        d = _close_distance_to_outer(df, i, zone)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


def _bos_non_bib_last_pullback(
    anchor_idx: int,
    zone: KLZone,
    pb_dir: int,
    df: pd.DataFrame,
) -> Optional[int]:
    """
    BOS + non-BIB: window [anchor_idx-5, anchor_idx+5].
    Among qualified candles that touch zone, find closest to outer.
    """
    start = max(0, anchor_idx - 5)
    end = min(len(df) - 1, anchor_idx + 5)

    best_idx = None
    best_dist = float("inf")
    for i in range(start, end + 1):
        if i not in df.index:
            continue
        if not _is_qualified(df, i, pb_dir):
            continue
        if not _touches_zone(df, i, zone):
            continue
        d = _close_distance_to_outer(df, i, zone)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx


# ---------------------------------------------------------------------------
# CTS wave candles
# ---------------------------------------------------------------------------

def _cts_wave_candles(
    anchor_idx: int,
    zone: KLZone,
    events: List[StructureEvent],
    sid: int,
    cycle_id: int,
    sd: int,
    df: pd.DataFrame,
) -> Optional[WaveCandleResult]:
    """
    CTS zone wave candles.
    - Last breakout candle: opposite direction to zone side
    - First pullback candle: same direction as zone side
    """
    base_pattern = zone.meta.get("base_pattern", "")

    # Direction mapping
    # Sell zone (sd=+1): last breakout dir = +1, first pullback dir = -1
    # Buy zone (sd=-1): last breakout dir = -1, first pullback dir = +1
    if zone.side == "sell":
        bo_dir = 1   # last breakout (bullish) -> opposite to sell
        pb_dir = -1  # first pullback (bearish) -> same as sell
    else:
        bo_dir = -1  # last breakout (bearish) -> opposite to buy
        pb_dir = 1   # first pullback (bullish) -> same as buy

    cts_anchor_idx = zone.meta.get("anchor_idx", anchor_idx)
    if cts_anchor_idx is not None:
        cts_anchor_idx = int(cts_anchor_idx)

    # --- Last breakout candle ---
    last_bo_idx = None

    if base_pattern == "base inside bar":
        last_bo_idx = _cts_bib_last_breakout(cts_anchor_idx, zone, events, sid, cycle_id, bo_dir, df)
    else:
        last_bo_idx = _cts_non_bib_last_breakout(cts_anchor_idx, zone, bo_dir, df)

    # --- First pullback candle ---
    first_pb_idx = None
    if last_bo_idx is not None:
        # Same compound condition, first match after last_bo_idx
        cts_conf = _find_cts_confirmed(events, sid, cycle_id)
        upper = len(df) - 1
        if cts_conf:
            conf_at = cts_conf.meta.get("confirmed_at")
            if conf_at is not None:
                upper = min(int(conf_at) + 10, len(df) - 1)

        for i in range(last_bo_idx + 1, upper + 1):
            if i not in df.index:
                continue
            if int(df.loc[i, "direction"]) != pb_dir:
                continue
            if _is_compound_first_wave(df, i, pb_dir):
                first_pb_idx = i
                break

    return WaveCandleResult(
        structure_id=sid,
        cycle_id=cycle_id,
        source_kind="CTS",
        zone_side=zone.side,
        last_wave_candle_idx=last_bo_idx,
        first_wave_candle_idx=first_pb_idx,
        meta={
            "base_pattern": base_pattern,
            "anchor_idx": cts_anchor_idx,
        },
    )


def _cts_bib_last_breakout(
    cts_anchor_idx: int,
    zone: KLZone,
    events: List[StructureEvent],
    sid: int,
    cycle_id: int,
    bo_dir: int,
    df: pd.DataFrame,
) -> Optional[int]:
    """
    CTS + base inside bar: last breakout candle.

    Step 1: Event walk — collect CTS_ESTABLISHED + CTS_UPDATED, sorted by ev.idx.
    For each event:
      (a) Check ev.idx: qualified AND wick enters zone AND closes within zone → done
      (b) If not: scan between current ev.idx and next CTS_UPDATED
    Step 2: Fallback — scan before/after cts_anchor_idx
    """
    cts_est = _find_cts_established(events, sid, cycle_id)
    cts_updates = _find_cts_updated_events(events, sid, cycle_id)
    cts_conf = _find_cts_confirmed(events, sid, cycle_id)

    confirmed_at = None
    if cts_conf:
        confirmed_at = cts_conf.meta.get("confirmed_at")
        if confirmed_at is not None:
            confirmed_at = int(confirmed_at)

    # Build ordered event list: CTS_ESTABLISHED first, then CTS_UPDATEDs
    ordered_events: List[StructureEvent] = []
    if cts_est is not None:
        ordered_events.append(cts_est)
    ordered_events.extend(cts_updates)
    ordered_events.sort(key=lambda e: e.idx)

    # Step 1: Event walk
    for ei, ev in enumerate(ordered_events):
        ev_idx = int(ev.idx)

        # (a) Direct check on event candle
        if _is_qualified(df, ev_idx, bo_dir) and _wick_enters_zone(df, ev_idx, zone) and _closes_within_zone(df, ev_idx, zone):
            return ev_idx

        # (b) Scan between current ev and next event
        if ei + 1 < len(ordered_events):
            next_ev_idx = int(ordered_events[ei + 1].idx)
        elif confirmed_at is not None:
            next_ev_idx = confirmed_at
        else:
            continue

        for i in range(ev_idx + 1, next_ev_idx):
            if i not in df.index:
                continue
            if _is_qualified(df, i, bo_dir) and _closes_within_zone(df, i, zone):
                return i

    # Step 2: Fallback
    # (a) Before cts_anchor_idx: earliest qualified candle closing within zone
    start = max(0, cts_anchor_idx - 10)
    for i in range(start, cts_anchor_idx):
        if i not in df.index:
            continue
        if _is_qualified(df, i, bo_dir) and _closes_within_zone(df, i, zone):
            return i

    # (b) From cts_anchor_idx forward to confirmed_at
    upper = confirmed_at if confirmed_at is not None else min(cts_anchor_idx + 10, len(df) - 1)
    for i in range(cts_anchor_idx, upper + 1):
        if i not in df.index:
            continue
        if _is_qualified(df, i, bo_dir) and _closes_within_zone(df, i, zone):
            return i

    return None


def _cts_non_bib_last_breakout(
    cts_anchor_idx: int,
    zone: KLZone,
    bo_dir: int,
    df: pd.DataFrame,
) -> Optional[int]:
    """
    CTS + non-BIB: window [cts_anchor_idx-5, cts_anchor_idx+5].
    Qualified candle touching zone, closest to outer.
    """
    start = max(0, cts_anchor_idx - 5)
    end = min(len(df) - 1, cts_anchor_idx + 5)

    best_idx = None
    best_dist = float("inf")
    for i in range(start, end + 1):
        if i not in df.index:
            continue
        if not _is_qualified(df, i, bo_dir):
            continue
        if not _touches_zone(df, i, zone):
            continue
        d = _close_distance_to_outer(df, i, zone)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return best_idx
