# engine_v2/zones/zone_proximity.py
"""
Zone proximity trigger detection.

Per (sid, cycle_id) with a CTS_CONFIRMED event, walk the scan window and
log each candle that comes within `proximity_pips` of the relevant zone
inner bound. Triggers alternate: the first must be sd-direction (BOS/POI),
then opp_sd (CTS zone), then sd, etc. — capturing each leg of a V/lambda
movement within a structure cycle.

Replaces the old `check_proximity_activation` in `wvmi.py`. The single
"activated cycles" gate is no longer the unit of interest — instead this
returns the full alternating list of trigger candles per cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.poi_zones import POIZone


# ---------------------------------------------------------------------------
# Default thresholds per timeframe
# ---------------------------------------------------------------------------
# Proximity threshold in pips. Caller may override per-call. Same values
# are used regardless of pair (pip_size handles pair scaling separately).
DEFAULT_PROXIMITY_PIPS: Dict[str, int] = {
    "H1": 9,
    "M15": 6,
    "M5": 3,
}

# Probe reset threshold in pips — used by the BOS_0 / Exception 2 reach-back
# probe to decide whether price has returned to the prior zone. Per Part 4
# spec §4.4, must be strictly less than DEFAULT_PROXIMITY_PIPS on the same
# TF so probe-reset and proximity-trigger semantics never overlap.
DEFAULT_PROBE_RESET_PIPS: Dict[str, int] = {
    "H1": 3,
    "M15": 2,
    "M5": 1,
}

for _tf in DEFAULT_PROBE_RESET_PIPS:
    assert DEFAULT_PROXIMITY_PIPS[_tf] > DEFAULT_PROBE_RESET_PIPS[_tf], (
        f"{_tf}: probe reset {DEFAULT_PROBE_RESET_PIPS[_tf]} pips must be "
        f"strictly less than proximity trigger {DEFAULT_PROXIMITY_PIPS[_tf]} pips"
    )
del _tf


@dataclass(frozen=True)
class ZoneProximityTrigger:
    """One trigger candle within a cycle's scan window.

    Direction "sd" = price came within threshold of an sd-direction zone
    (BOS or POI). "opp_sd" = price came within threshold of the CTS zone
    (the only opposite-direction zone, by definition).
    """
    structure_id: int
    cycle_id: int
    direction: Literal["sd", "opp_sd"]
    idx: int                                  # the trigger candle index
    trigger_inner: float                       # the inner price used as trigger level
    zone_kind: Literal["BOS", "CTS", "POI"]   # which zone matched
    proximity_pips: int
    pip_size: float
    timeframe: str
    meta: Dict[str, Any] = field(default_factory=dict)


def _find_kl_zone_for_cycle(
    kl_zones: List[KLZone],
    sid: int,
    cycle_id: int,
    source_kind: Literal["BOS", "CTS"],
) -> Optional[KLZone]:
    """Find the KL zone for (sid, cycle_id) of a specific source_kind."""
    for z in kl_zones:
        if (z.meta.get("structure_id") == sid
                and z.meta.get("cycle_id") == cycle_id
                and z.source_kind == source_kind):
            return z
    return None


def check_zone_proximity(
    df: pd.DataFrame,
    sorted_events: List[StructureEvent],
    kl_zones: List[KLZone],
    poi_zones: List[POIZone],
    pip_size: float,
    timeframe: str,
    proximity_pips: Optional[int] = None,
) -> Dict[Tuple[int, int], List[ZoneProximityTrigger]]:
    """Find zone proximity trigger candles per (sid, cycle_id).

    Per cycle, walk the scan window and log alternating sd/opp_sd triggers:
    1. First trigger must be sd (price within threshold of BOS or active POI)
    2. After an sd trigger, only opp_sd is eligible (CTS zone)
    3. After an opp_sd trigger, only sd again — alternating

    Each "first match wins" applies per slot (i.e., once an sd trigger is
    logged, no more sd triggers fire until an opp_sd has been logged).

    Scan window per (sid, cycle_id):
      start = CTS_CONFIRMED.meta["confirmed_at"] (the confirmation candle)
      end   = min(next_BOS_CONFIRMED.confirmed_at - 1,
                  REVERSAL_CANDIDATE.apply_idx - 1,
                  end_of_df)

    Parameters
    ----------
    df : DataFrame
        OHLC data with structure columns.
    sorted_events : list of StructureEvent
        All structure events for the run, sorted by (idx, type).
    kl_zones : list of KLZone
        Both BOS- and CTS-derived zones (sd and opp_sd respectively).
    poi_zones : list of POIZone
        All POI zones (always sd direction by construction — Fib spans
        BOS→CTS so any POI within is in the structure direction).
    pip_size : float
        Pip size for the pair (typically 0.0001 / 0.01 depending on quote).
    timeframe : str
        "H1", "M15", "M5", etc. — used to look up default proximity_pips.
    proximity_pips : int, optional
        Threshold in pips. If None, looked up from DEFAULT_PROXIMITY_PIPS
        by `timeframe` (default fallback 20 for unknown timeframes).

    Returns
    -------
    Dict[(sid, cycle_id), List[ZoneProximityTrigger]]
        Ordered list of triggers per cycle. Empty list / missing key when
        no triggers fired in the scan window.
    """
    if proximity_pips is None:
        proximity_pips = DEFAULT_PROXIMITY_PIPS.get(timeframe, 20)
    threshold = proximity_pips * pip_size

    # Lookups
    cts_conf_by_key: Dict[tuple, StructureEvent] = {}
    for ev in sorted_events:
        if ev.type == "CTS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            cts_conf_by_key[key] = ev

    bos_conf_idx_by_key: Dict[tuple, int] = {}
    for ev in sorted_events:
        if ev.type == "BOS_CONFIRMED":
            key = (ev.meta.get("structure_id", 0), ev.meta.get("cycle_id", 0))
            # confirmed_at = confirmation candle, not BOS extreme
            bos_conf_idx_by_key[key] = int(ev.meta.get("confirmed_at", ev.idx))

    reversal_idx_by_sid: Dict[int, int] = {}
    for ev in sorted_events:
        if ev.type == "REVERSAL_CANDIDATE":
            sid = ev.meta.get("structure_id", 0)
            apply_idx = ev.meta.get("apply_idx")
            if apply_idx is not None:
                reversal_idx_by_sid[sid] = apply_idx

    triggers_by_cycle: Dict[Tuple[int, int], List[ZoneProximityTrigger]] = {}

    for key, cts_ev in cts_conf_by_key.items():
        sid, cycle_id = key
        sd = int(cts_ev.meta.get("struct_direction", 0))
        if sd == 0:
            continue

        bos_zone = _find_kl_zone_for_cycle(kl_zones, sid, cycle_id, "BOS")
        if bos_zone is None:
            continue
        bos_inner = bos_zone.meta.get("inner")
        if bos_inner is None:
            continue

        cts_zone = _find_kl_zone_for_cycle(kl_zones, sid, cycle_id, "CTS")
        cts_inner = cts_zone.meta.get("inner") if cts_zone is not None else None

        cycle_poi_zones = [
            pz for pz in poi_zones
            if pz.meta.get("structure_id") == sid
            and pz.meta.get("cycle_id") == cycle_id
        ]

        # Scan window
        scan_start = int(cts_ev.meta.get("confirmed_at", cts_ev.idx))
        scan_end = len(df) - 1
        next_bos_idx = bos_conf_idx_by_key.get((sid, cycle_id + 1))
        if next_bos_idx is not None:
            scan_end = min(scan_end, next_bos_idx - 1)
        reversal_idx = reversal_idx_by_sid.get(sid)
        if reversal_idx is not None:
            scan_end = min(scan_end, reversal_idx - 1)

        if scan_start > scan_end:
            continue

        triggers: List[ZoneProximityTrigger] = []
        expected_dir: Literal["sd", "opp_sd"] = "sd"

        # Candle-direction filter. A trigger must be a candle moving INTO
        # the target zone — its body direction must oppose the zone's side:
        #   sd zone (side = sd):     required candle direction = -sd
        #   opp_sd zone (side = -sd): required candle direction = +sd
        # Doji (direction == 0) and same-side candles are filtered out;
        # this rejects wick-only touches that close opposite the approach
        # (e.g., bullish pinbars at a buy zone — wick probes but body
        # rejects).
        has_dir = "direction" in df.columns
        required_dir_sd = -int(sd)
        required_dir_opp_sd = int(sd)

        for i in range(scan_start, scan_end + 1):
            if i not in df.index:
                continue

            if expected_dir == "sd":
                # Candle-direction filter for sd trigger
                if has_dir and int(df.loc[i, "direction"]) != required_dir_sd:
                    continue

                # Build sd-direction inners: BOS (always active) + active POIs
                sd_inners: List[Tuple[float, str]] = [
                    (float(bos_inner), "BOS")
                ]
                for pz in cycle_poi_zones:
                    confirmed_idx = pz.meta.get("confirmed_idx")
                    end_idx_pz = pz.meta.get("end_idx")
                    if confirmed_idx is None or i < confirmed_idx:
                        continue
                    if end_idx_pz is not None and i > end_idx_pz:
                        continue
                    if sd == 1:
                        sd_inners.append((float(pz.top), "POI"))
                    else:
                        sd_inners.append((float(pz.bottom), "POI"))

                # Closest-to-current-price inner: max for buy (sd=+1), min for sell (sd=-1)
                if sd == 1:
                    trigger_inner, zone_kind = max(sd_inners, key=lambda t: t[0])
                    candle_low = float(df.loc[i, "l"])
                    if candle_low <= trigger_inner + threshold:
                        triggers.append(ZoneProximityTrigger(
                            structure_id=sid, cycle_id=cycle_id,
                            direction="sd", idx=i,
                            trigger_inner=trigger_inner, zone_kind=zone_kind,
                            proximity_pips=proximity_pips, pip_size=pip_size,
                            timeframe=timeframe,
                        ))
                        expected_dir = "opp_sd"
                else:
                    trigger_inner, zone_kind = min(sd_inners, key=lambda t: t[0])
                    candle_high = float(df.loc[i, "h"])
                    if candle_high >= trigger_inner - threshold:
                        triggers.append(ZoneProximityTrigger(
                            structure_id=sid, cycle_id=cycle_id,
                            direction="sd", idx=i,
                            trigger_inner=trigger_inner, zone_kind=zone_kind,
                            proximity_pips=proximity_pips, pip_size=pip_size,
                            timeframe=timeframe,
                        ))
                        expected_dir = "opp_sd"

            else:  # expected_dir == "opp_sd"
                if cts_inner is None:
                    # No opp_sd zone available; nothing more can trigger
                    # for this cycle. Break to avoid wasted scanning.
                    break

                # Candle-direction filter for opp_sd trigger
                if has_dir and int(df.loc[i, "direction"]) != required_dir_opp_sd:
                    continue

                trigger_inner = float(cts_inner)
                if sd == 1:
                    # CTS zone is sell-side, sits below outer top.
                    # Approach from below: candle high reaches near inner.
                    candle_high = float(df.loc[i, "h"])
                    if candle_high >= trigger_inner - threshold:
                        triggers.append(ZoneProximityTrigger(
                            structure_id=sid, cycle_id=cycle_id,
                            direction="opp_sd", idx=i,
                            trigger_inner=trigger_inner, zone_kind="CTS",
                            proximity_pips=proximity_pips, pip_size=pip_size,
                            timeframe=timeframe,
                        ))
                        expected_dir = "sd"
                else:
                    # CTS zone is buy-side, sits above outer bottom.
                    # Approach from above: candle low reaches near inner.
                    candle_low = float(df.loc[i, "l"])
                    if candle_low <= trigger_inner + threshold:
                        triggers.append(ZoneProximityTrigger(
                            structure_id=sid, cycle_id=cycle_id,
                            direction="opp_sd", idx=i,
                            trigger_inner=trigger_inner, zone_kind="CTS",
                            proximity_pips=proximity_pips, pip_size=pip_size,
                            timeframe=timeframe,
                        ))
                        expected_dir = "sd"

        if triggers:
            triggers_by_cycle[(sid, cycle_id)] = triggers

    return triggers_by_cycle
