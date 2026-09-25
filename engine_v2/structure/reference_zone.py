"""Reference-zone construction for the unified identify-start + probe.

Per the Phase 1 design (2026-05-29, see project memory
`project_unified_identify_start_probe.md`), every probe consults a
`reference_zone` derived from the relevant prior/sibling sid's most recent
CTS event. The construction rule is uniform across `subsequent_*` and
`reversal` triggers:

    take the event with the most recent MOMENT (Plan E E3b) among
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
from engine_v2.structure import event_fields as ef
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.zones.kl_zones_v1 import identify_base_pattern, zone_thresholds


# CTS event types consulted by the most-recent-moment pick, in spec priority order
# (most-recent wins regardless of type — type only matters for the
# CONFIRMED-vs-ad-hoc branch).
_CTS_EVENT_TYPES = frozenset({"CTS_CONFIRMED", "CTS_UPDATED", "CTS_ESTABLISHED"})


@dataclass(frozen=True)
class ReferenceZone:
    """A probe's reference zone — the price target the probe checks
    retraces against.

    Outer/inner semantics are keyed off the **probe's direction** (not the
    source zone's parent direction):

    - **Probe direction +1** (uptrend): structure body grows UP from
      `input_idx`; reference zone sits BELOW body; retraces of the
      uptrend go DOWN and approach the zone from above. So
      ``inner = z.top`` (body-facing side of zone-below-body) and
      ``outer = z.bottom`` (far side). ``side="buy"`` (zone acts as
      support).
    - **Probe direction -1** (downtrend): body grows DOWN; reference is
      ABOVE body; retraces approach from below. ``inner = z.bottom``,
      ``outer = z.top``. ``side="sell"`` (zone acts as resistance).

    Numerically: ``inner > outer`` for +1 probes; ``inner < outer`` for
    -1 probes.

    `source` records provenance for debug/attribution. `anchor_idx`
    is the parent-frame idx of the structural event the zone was derived
    from:

    - ``cts_confirmed`` → ``cts_anchor_idx`` (the CTS extreme, NOT the
      later confirmation candle)
    - ``cts_updated`` / ``cts_established`` → the CTS anchor
      ``ef.cts_anchor_idx(ev)`` (the CTS extreme; not an EST's ``ev.idx``,
      the moment since Plan E E4a)
    """
    outer: float
    inner: float
    side: Literal["buy", "sell"]
    source: Literal[
        "cts_confirmed",
        "cts_updated",
        "cts_established",
        "ad_hoc_bos_0",
    ]
    anchor_idx: int


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
    probe_direction: int,
    source: str,
    anchor_idx: int,
) -> ReferenceZone:
    """Convert a KLZone's geographic (top/bottom) to a probe's semantic
    (outer/inner), keyed off the **probe's direction** (= the new
    structure's lower_sd, NOT the source zone's parent direction).

    See ReferenceZone docstring for the full rule. In short:

    - ``probe_direction == +1``: body above zone → ``inner = z.top``,
      ``outer = z.bottom``, ``side = "buy"``.
    - ``probe_direction == -1``: body below zone → ``inner = z.bottom``,
      ``outer = z.top``, ``side = "sell"``.

    This rule is universal across all reference-zone sources (parent BOS,
    parent CTS, sibling CTS, prior-sid CTS): regardless of how the source
    zone relates to its own parent's direction, ``inner`` is always the
    side closest to the body of the structure being probed.
    """
    if probe_direction == 1:
        return ReferenceZone(
            outer=float(z.bottom),
            inner=float(z.top),
            side="buy",
            source=source,  # type: ignore[arg-type]
            anchor_idx=int(anchor_idx),
        )
    return ReferenceZone(
        outer=float(z.top),
        inner=float(z.bottom),
        side="sell",
        source=source,  # type: ignore[arg-type]
        anchor_idx=int(anchor_idx),
    )


def _derive_zone_ad_hoc(
    df: pd.DataFrame,
    anchor_idx: int,
    direction: int,
    *,
    bos: bool,
) -> Optional[Tuple[float, float, Literal["buy", "sell"]]]:
    """Derive an ad-hoc zone (outer, inner, side) from a single anchor
    candle. Mirrors `derive_kl_zones_v1`'s per-event derivation:

    - ``bos=False`` (CTS-style): `identify_base_pattern(..., bos=False)` +
      `zone_thresholds(..., bos=False)`. Used by reversal /
      subsequent_* fallback paths whose source event is CTS_UPDATED /
      CTS_ESTABLISHED (no derived KL zone yet).
    - ``bos=True`` (BOS-style): `identify_base_pattern(..., bos=True)` +
      `zone_thresholds(..., bos=True)`. Used by first_confluence (the
      sub's "first BOS the new structure would form" — anchored at the
      M15 input_idx, which is the parent BOS extreme price-mapped to M15).

    `direction` is the SOURCE structure's direction in both cases (= the
    "side" assignment below in geographic terms — sd=+1 ⇒ structure
    rising ⇒ zone-of-its-own-extreme on the +1 side ⇒ "sell" geographic
    label for that zone). The caller's probe then re-interprets via
    `_zone_to_reference(z, probe_direction)`.

    Returns None if the base pattern can't be identified or `anchor_idx`
    is out of df bounds — the caller's probe will skip / treat as
    missing.
    """
    if anchor_idx not in df.index:
        return None
    try:
        zone_pattern, base_idx = identify_base_pattern(
            df,
            anchor_idx=int(anchor_idx),
            struct_direction=int(direction),
            bos=bool(bos),
        )
        if base_idx is None or base_idx not in df.index:
            return None
        outer, inner = zone_thresholds(
            df,
            base_idx=int(base_idx),
            struct_direction=int(direction),
            zone_pattern=zone_pattern,
            bos=bool(bos),
        )
    except Exception:
        return None
    if outer is None or inner is None:
        return None
    side: Literal["buy", "sell"] = "sell" if direction == 1 else "buy"
    return (float(outer), float(inner), side)


def build_ad_hoc_bos0_reference_zone(
    df: pd.DataFrame,
    anchor_idx: int,
    probe_direction: int,
) -> Optional[ReferenceZone]:
    """Ad-hoc BOS_0 reference zone anchored at `anchor_idx`, keyed to
    `probe_direction`.

    The BOS_0 belongs to the structure being PROBED (its own direction ==
    `probe_direction`), so the ad-hoc derivation passes `probe_direction`
    as the source-structure direction, then re-keys the geographic
    (top/bottom) to the probe's outer/inner via the universal rule
    (`inner = z.top` for +1, `z.bottom` for -1).

    Two callers share this one helper (single source of truth):
      - `first_confluence`'s CONSTANT reference zone (anchor = the M15
        input_idx) — via `_build_first_confluence_ref_zone`.
      - the unified probe's MOVING BOS_0 threshold after a retrace-reset
        (anchor = the new `current_start`) — per the "two zones" design
        (`project_true_first_breakout_cycle0.md`: iter 2+ = a fresh
        `bos=True` BOS_0 at the moved start).

    Returns None when the base pattern can't be derived from `anchor_idx`
    (the probe caller then falls back to the candle's own extreme).
    """
    derived = _derive_zone_ad_hoc(df, anchor_idx, probe_direction, bos=True)
    if derived is None:
        return None
    outer_geo, inner_geo, _side_geo = derived
    z_top = max(outer_geo, inner_geo)
    z_bottom = min(outer_geo, inner_geo)
    if probe_direction == 1:
        ref_outer, ref_inner, side = z_bottom, z_top, "buy"
    else:
        ref_outer, ref_inner, side = z_top, z_bottom, "sell"
    return ReferenceZone(
        outer=float(ref_outer),
        inner=float(ref_inner),
        side=side,  # type: ignore[arg-type]
        source="ad_hoc_bos_0",
        anchor_idx=int(anchor_idx),
    )


# Back-compat alias for the CTS-only callers that still exist in
# `build_reference_zone_from_cts_event`. Kept thin (one-line forward)
# so future readers see it's the same machinery.
def _derive_cts_zone_ad_hoc(
    df: pd.DataFrame,
    cts_anchor_idx: int,
    direction: int,
) -> Optional[Tuple[float, float, Literal["buy", "sell"]]]:
    return _derive_zone_ad_hoc(df, cts_anchor_idx, direction, bos=False)


def build_reference_zone_from_cts_event(
    events: List[StructureEvent],
    kl_zones: List[KLZone],
    df: pd.DataFrame,
    sid: int,
    probe_direction: int,
    *,
    idx_window: Optional[Tuple[int, int]] = None,
) -> Optional[ReferenceZone]:
    """Find the reference zone for a probe that anchors against `sid`'s
    most recent CTS event.

    Picks, among `events` with `structure_id == sid`, the most recent MOMENT
    (`ef.event_moment`, Plan E E3b; ties CONFIRMED > UPDATED > ESTABLISHED) of
    {CTS_CONFIRMED, CTS_UPDATED, CTS_ESTABLISHED}. Returns:

    - CTS_CONFIRMED → the existing KLZone for that cycle, wrapped as a
      `ReferenceZone` (outer/inner derived from `probe_direction`). If for
      any reason the kl_zone is missing (race/slice/derivation skip),
      falls back to the ad-hoc derivation so the probe can still run.
    - CTS_UPDATED / CTS_ESTABLISHED → ad-hoc derivation from the CTS anchor
      (`ef.cts_anchor_idx`).

    For ad-hoc derivation, `sid`'s `struct_direction` is reconstructed as
    `-probe_direction` (callers of this helper run a probe in the OPPOSITE
    direction to the source sid — that's the reversal / subsequent_*
    semantic: the new sub flips from the prior sid).

    NOTE: this `-probe_direction` reconstruction ASSUMES every CTS in `events`
    came from a structure in the `-probe_direction` direction. Reversal callers
    satisfy this by construction (the only prior sub is the one being reversed).
    Sibling callers (`_build_sibling_cts_ref_zone`) must PRE-FILTER `events` to
    that direction, because a sibling can reverse before the trigger fires and a
    post-reversal CTS would otherwise be picked as "most recent" and mis-sided
    here (direction qualification, 2026-06-15).

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
    probe_direction : int
        The PROBE's direction (= the new sub's lower_sd, +1 or -1). The
        source sid's struct_direction is the OPPOSITE (`-probe_direction`).
        Determines the resulting ReferenceZone's outer/inner/side per
        `_zone_to_reference`.
    idx_window : (int, int), optional
        Inclusive `(min_idx, max_idx)` filter on the event's MOMENT
        (`ef.event_moment`, Plan E E3b). Used by
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
            # A TIME filter (the sibling window): the event's MOMENT (Plan E E3b).
            ev_moment_idx = ef.event_moment(ev)
            if ev_moment_idx < lo or ev_moment_idx > hi:
                continue
        candidates.append(ev)

    if not candidates:
        return None

    # Most-recent MOMENT wins regardless of type (Plan E E3b; hazard H5). Ties
    # break on type order (CONFIRMED > UPDATED > ESTABLISHED). A tie is real: a
    # cycle's CTS_ESTABLISHED and a CTS_UPDATED / CTS_CONFIRMED can share a
    # moment candle (the stamped idx of an EST is its anchor, so the old
    # "never at the same idx" claim held only for stamped indices).
    _TYPE_ORDER = {
        "CTS_CONFIRMED": 2,
        "CTS_UPDATED": 1,
        "CTS_ESTABLISHED": 0,
    }
    candidates.sort(
        key=lambda e: (ef.event_moment(e), _TYPE_ORDER[e.type]),
        reverse=True,
    )
    ev = candidates[0]
    # The winner's CTS ANCHOR (a location): the ad-hoc zone base and the probe
    # input (`anchor_idx` → the pool key).
    cts_anchor_idx = ef.cts_anchor_idx(ev)

    # Source sid's struct_direction is the OPPOSITE of probe_direction
    # (reversal / subsequent_* semantic — the probe runs in the new sub's
    # direction, which is flipped from the source sid).
    source_sd = -probe_direction

    if ev.type == "CTS_CONFIRMED":
        cycle_id = ev.meta.get("cycle_id")
        if cycle_id is not None:
            existing = _find_existing_cts_kl_zone(kl_zones, sid, int(cycle_id))
            if existing is not None:
                return _zone_to_reference(
                    existing,
                    probe_direction,
                    source="cts_confirmed",
                    anchor_idx=cts_anchor_idx,
                )
        # CONFIRMED but no derived zone — fall through to ad-hoc.

    derived = _derive_cts_zone_ad_hoc(df, cts_anchor_idx, source_sd)
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
        anchor_idx=int(cts_anchor_idx),
    )
