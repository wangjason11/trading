"""Reference-zone construction for the unified identify-start + probe.

Per the Phase 1 design (2026-05-29, see project memory
`project_unified_identify_start_probe.md`), every probe consults a
`reference_zone` derived from the relevant prior/sibling sid's most recent
CTS event. The construction rule is uniform across `subsequent_*` and
`reversal` triggers:

    walk events reverse-idx, take the most recent of
    {CTS_CONFIRMED, CTS_UPDATED, CTS_ESTABLISHED}
        - CTS_CONFIRMED → use the existing KLZone derived from the event
        - CTS_UPDATED / CTS_ESTABLISHED → build ad hoc from the extreme
          candle, mirroring the BOS_0 ad-hoc shape Scenario 3 already
          uses (`identify_base_pattern` + `zone_thresholds(bos=False)`)

`first_*` triggers skip this helper — their reference zones are the
parent's BOS / CTS zones, which are guaranteed to exist by trigger
prerequisite.

The helper is intentionally TF-agnostic; the caller decides which `df`
and `events` belong to which TF.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd

from engine_v2.common.types import KLZone
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.kl_zones_v1 import identify_base_pattern, zone_thresholds


# CTS event types consulted by the reverse-idx walk, in spec priority order
# (most-recent wins regardless of type — type only matters for the
# CONFIRMED-vs-ad-hoc branch).
_CTS_EVENT_TYPES = frozenset({"CTS_CONFIRMED", "CTS_UPDATED", "CTS_ESTABLISHED"})


@dataclass(frozen=True)
class ReferenceZone:
    """A probe's reference zone — the price target the probe checks
    retraces against.

    `outer` / `inner` are semantic (outer = away from struct direction;
    inner = near the body). For sd=+1 the zone sits ABOVE price (sell
    side), so outer=top, inner=bottom. For sd=-1 the zone sits BELOW
    price (buy side), so outer=bottom, inner=top.

    `source` records provenance for debug/attribution. `source_event_idx`
    is the idx of the CTS event the zone was derived from (the CTS extreme
    for UPDATED/ESTABLISHED; `cts_anchor_idx` for CONFIRMED), NOT the
    confirmation candle.
    """
    outer: float
    inner: float
    side: Literal["buy", "sell"]
    source: Literal["cts_confirmed", "cts_updated", "cts_established"]
    source_event_idx: int


def _extreme_idx_for_cts_event(ev: StructureEvent) -> int:
    """Return the CTS extreme candle's idx for a given CTS-type event.

    For CTS_CONFIRMED: `ev.idx` is the confirmation candle (later); the
    extreme is `meta["cts_anchor_idx"]` (earlier). For CTS_UPDATED /
    CTS_ESTABLISHED: `ev.idx` IS the extreme by construction.
    """
    if ev.type == "CTS_CONFIRMED":
        return int(ev.meta.get("cts_anchor_idx", ev.idx))
    return int(ev.idx)


def _find_existing_cts_kl_zone(
    kl_zones: List[KLZone],
    sid: int,
    cycle_id: int,
) -> Optional[KLZone]:
    """Look up the CTS-derived KLZone for (sid, cycle_id), if one exists."""
    for z in kl_zones:
        if (
            z.source_kind == "CTS"
            and z.meta.get("structure_id") == sid
            and z.meta.get("cycle_id") == cycle_id
        ):
            return z
    return None


def _zone_to_reference(
    z: KLZone,
    direction: int,
    source: str,
    source_event_idx: int,
) -> ReferenceZone:
    """Convert a KLZone's geographic (top/bottom) to a probe's semantic
    (outer/inner) per `direction`.

    For sd=+1: CTS is a sell zone above price → outer=top, inner=bottom.
    For sd=-1: CTS is a buy zone below price → outer=bottom, inner=top.
    """
    if direction == 1:
        return ReferenceZone(
            outer=float(z.top),
            inner=float(z.bottom),
            side="sell",
            source=source,  # type: ignore[arg-type]
            source_event_idx=int(source_event_idx),
        )
    return ReferenceZone(
        outer=float(z.bottom),
        inner=float(z.top),
        side="buy",
        source=source,  # type: ignore[arg-type]
        source_event_idx=int(source_event_idx),
    )


def _derive_cts_zone_ad_hoc(
    df: pd.DataFrame,
    extreme_idx: int,
    direction: int,
) -> Optional[Tuple[float, float, Literal["buy", "sell"]]]:
    """Derive an ad-hoc CTS zone (outer, inner, side) from the extreme
    candle alone. Mirrors `derive_kl_zones_v1`'s per-event derivation for
    `CTS_*` events (anchor=extreme, `identify_base_pattern(..., bos=False)`,
    `zone_thresholds(..., bos=False)`).

    Returns None if the base pattern can't be identified or `extreme_idx`
    is out of df bounds — the caller's probe will skip this iteration.
    """
    if extreme_idx not in df.index:
        return None
    try:
        zone_pattern, base_idx = identify_base_pattern(
            df,
            anchor_idx=int(extreme_idx),
            struct_direction=int(direction),
            bos=False,
        )
        if base_idx is None or base_idx not in df.index:
            return None
        outer, inner = zone_thresholds(
            df,
            base_idx=int(base_idx),
            struct_direction=int(direction),
            zone_pattern=zone_pattern,
            bos=False,
        )
    except Exception:
        return None
    if outer is None or inner is None:
        return None
    side: Literal["buy", "sell"] = "sell" if direction == 1 else "buy"
    return (float(outer), float(inner), side)


def build_reference_zone_from_cts_event(
    events: List[StructureEvent],
    kl_zones: List[KLZone],
    df: pd.DataFrame,
    sid: int,
    direction: int,
    *,
    idx_window: Optional[Tuple[int, int]] = None,
) -> Optional[ReferenceZone]:
    """Find the reference zone for a probe that anchors against `sid`'s
    most recent CTS event.

    Walks `events` reverse-idx for `structure_id == sid`, taking the
    most recent of {CTS_CONFIRMED, CTS_UPDATED, CTS_ESTABLISHED}. Returns:

    - CTS_CONFIRMED → the existing KLZone for that cycle, wrapped as a
      `ReferenceZone` (outer/inner derived from sd). If for any reason the
      kl_zone is missing (race/slice/derivation skip), falls back to the
      ad-hoc derivation so the probe can still run.
    - CTS_UPDATED / CTS_ESTABLISHED → ad-hoc derivation from the extreme
      candle (= `ev.idx` for these types).

    Parameters
    ----------
    events : list of StructureEvent
        Source events. Filtered internally by `structure_id == sid` and
        `type in _CTS_EVENT_TYPES`.
    kl_zones : list of KLZone
        Derived KL zones (for the CONFIRMED-zone lookup).
    df : DataFrame
        OHLC + candle-feature df on the same TF as the events. Used for
        the ad-hoc derivation.
    sid : int
        The structure_id whose events anchor the reference.
    direction : int
        The sid's struct_direction (+1 or -1). Determines side mapping.
    idx_window : (int, int), optional
        Inclusive `(min_idx, max_idx)` filter on event idx. Used by
        `subsequent_*` callers to restrict the walk to events within a
        time-mapped sibling window. Reversal callers leave this None.

    Returns
    -------
    ReferenceZone or None
        None when no qualifying CTS event exists for `sid` (extremely
        rare — caller should fall back to its variant's per-spec fallback,
        e.g. window-extreme-toward-parent-zone for subsequent_*).
    """
    candidates: List[StructureEvent] = []
    for ev in events:
        if ev.type not in _CTS_EVENT_TYPES:
            continue
        if ev.meta.get("structure_id") != sid:
            continue
        if idx_window is not None:
            lo, hi = idx_window
            if ev.idx < lo or ev.idx > hi:
                continue
        candidates.append(ev)

    if not candidates:
        return None

    # Most-recent wins regardless of type. Tie-break on type order
    # (CONFIRMED > UPDATED > ESTABLISHED) is irrelevant in practice because
    # the engine never emits two CTS-type events at the same idx for the
    # same cycle — but make the order deterministic anyway for safety.
    _TYPE_ORDER = {
        "CTS_CONFIRMED": 2,
        "CTS_UPDATED": 1,
        "CTS_ESTABLISHED": 0,
    }
    candidates.sort(
        key=lambda e: (int(e.idx), _TYPE_ORDER[e.type]),
        reverse=True,
    )
    ev = candidates[0]
    extreme_idx = _extreme_idx_for_cts_event(ev)

    if ev.type == "CTS_CONFIRMED":
        cycle_id = ev.meta.get("cycle_id")
        if cycle_id is not None:
            existing = _find_existing_cts_kl_zone(kl_zones, sid, int(cycle_id))
            if existing is not None:
                return _zone_to_reference(
                    existing,
                    direction,
                    source="cts_confirmed",
                    source_event_idx=extreme_idx,
                )
        # CONFIRMED but no derived zone — fall through to ad-hoc.

    derived = _derive_cts_zone_ad_hoc(df, extreme_idx, direction)
    if derived is None:
        return None
    outer, inner, side = derived

    if ev.type == "CTS_CONFIRMED":
        source: Literal["cts_confirmed", "cts_updated", "cts_established"] = "cts_confirmed"
    elif ev.type == "CTS_UPDATED":
        source = "cts_updated"
    else:
        source = "cts_established"

    return ReferenceZone(
        outer=outer,
        inner=inner,
        side=side,
        source=source,
        source_event_idx=int(extreme_idx),
    )
