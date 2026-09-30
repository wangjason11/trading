from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS, StructureLevel
from engine_v2.structure.market_structure import MarketStructure, StructureEvent
from engine_v2.structure.identify_start import identify_start_scenario_1
from engine_v2.structure.reference_zone import (
    build_ad_hoc_bos0_reference_zone,
    build_reference_zone_from_cts_event,
)
from engine_v2.zones.kl_zones_v1 import (
    compute_bos_inner_from_event,
    derive_kl_zones_v1,
)
from engine_v2.zones.poi_zones import compute_poi_inners_for_cycle
from engine_v2.zones.zone_proximity import (
    DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS,
    DEFAULT_PROXIMITY_PIPS,
)


def _proximity_pips(timeframe: str) -> int:
    """Look up the zone-proximity threshold (in pips) for a timeframe."""
    return DEFAULT_PROXIMITY_PIPS.get(timeframe, DEFAULT_PROXIMITY_PIPS["H1"])


def _min_gap_pips(timeframe: str) -> int:
    """Look up the narrow-cycle gap threshold (in pips) for a timeframe.

    Used by the dual CTS proximity confirmation Rule 1 gate
    (`MarketStructure._apply_pattern_at_apply_idx`) and by Rules 2/3 in
    the post-facto scan (`zones/zone_proximity.check_zone_proximity`).
    """
    return DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS.get(
        timeframe, DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS["H1"]
    )


def _make_market_structure(
    df: pd.DataFrame,
    struct_direction: int,
    *,
    timeframe: str = "H1",
    **kwargs,
) -> MarketStructure:
    """Construct MarketStructure with proximity resolvers + TF-defaulted
    proximity_pips wired in. Per Part 4 §13.5.b, MarketStructure does not
    import from zones/; resolvers are passed at construction so the
    structure→zones import inversion stays gone.

    Parity carve-out (matches Step 1 baseline): probe-only call sites
    historically constructed MarketStructure without `timeframe`, falling
    back to the H1 default of `DEFAULT_PROXIMITY_PIPS["H1"]` proximity_pips
    regardless of the outer caller's TF. The one such site left —
    `unified_probe`'s iterative (Phase 2) run — still calls this helper
    without `timeframe`, preserving the H1 default. Spec §4.4 explicitly
    listed only the probe-RESET threshold for TF-aware lookup;
    proximity_pips for probes is a deferred cleanup.
    """
    kwargs.setdefault("proximity_pips", _proximity_pips(timeframe))
    kwargs.setdefault("min_gap_pips", _min_gap_pips(timeframe))
    kwargs.setdefault("bos_inner_resolver", compute_bos_inner_from_event)
    kwargs.setdefault("poi_inners_resolver", compute_poi_inners_for_cycle)
    return MarketStructure(
        df,
        struct_direction=struct_direction,
        timeframe=timeframe,
        **kwargs,
    )


@dataclass
class StructureEngineResult:
    df: pd.DataFrame
    levels: List[StructureLevel]
    events: List[StructureEvent]
    struct_direction: int
    notes: str = ""


@dataclass
class BoundedStructureResult:
    """Result of a bounded SINGLE-structure run (Part 4 §5, REVISED 2026-05-25).

    Unlike StructureEngineResult (multi-structure), this represents exactly
    one directional structure (structure_id=0). `reversal_idx` is the
    boundary to the next sub sid — the candle where the run hit its first
    internal reversal — or None if no reversal occurred inside
    `[start_idx, end_idx]` (bounded out by end_idx, or no reversal at all).
    """
    df: pd.DataFrame
    levels: List[StructureLevel]
    events: List[StructureEvent]
    struct_direction: int
    start_idx: int
    end_idx: Optional[int]
    reversal_idx: Optional[int]   # reversal pattern apply idx; None if no reversal in bounds
    notes: str = ""


def compute_structure(df: pd.DataFrame, *, timeframe: str = "H1") -> StructureEngineResult:
    """
    Adapter boundary: df(with patterns/features) -> MarketStructure outputs:
      - df2 (with structure columns)
      - ms_events (StructureEvent list)
      - levels (StructureLevel list)

    Multi-structure flow:
      1. Identify initial start candle and direction (Scenario 1)
      2. Run MarketStructure for structure_id=0
      3. On reversal, identify the next start with `unified_probe` (the Step 4
         reversal handoff, 2026-06-20 — it replaced Scenario 2 + Exception 1 +
         Exception 2; see the handoff comment in the loop below)
      4. Run MarketStructure for structure_id=1, etc.
      5. Repeat until no more reversals or guard limit reached
    """
    _validate_input(df)

    pip_size = _pip_size_from_pair(df)

    # --- Scenario 1 (initial start identification) ---
    input_idx = int(df.index.max())
    d0 = identify_start_scenario_1(df, input_idx=input_idx, lookback_days=183, min_history=50)

    df2 = df
    all_events: List[StructureEvent] = []
    all_levels: List[StructureLevel] = []

    start_idx = int(d0.start_idx)
    struct_direction = int(d0.struct_direction)
    structure_id = 0

    # Cycle-0 true-first-breakout scan mode (project_true_first_breakout_cycle0.md).
    # A structure's CTS_0 is established only by the first true breakout past the
    # BOS_0 inner. MS's pre-CTS_0 scan mode (`enforce_cts0_new_extreme`) gates on
    # that inner threshold (`cur_bos0_inner`), which is carried per-sid through the
    # loop below:
    #   - sid=0  (trading open): the ad-hoc bos=True BOS_0 anchored at the
    #     Scenario-1 start, built here. Single-shot: no probe, no retrace-resets
    #     (current_start is fixed at the Scenario-1 start).
    #   - sid>=1 (reversals, Step 4): the BOS_0 inner returned by the reversal
    #     unified_probe inside the loop, mirroring the sub reversal path.
    # `.inner` mirrors the sub probe's `bos0_inner = reference_zone.inner`
    # (unified_probe.py) — single source of truth, NOT a re-derivation.
    #
    # None => the ad-hoc base can't be derived. In a full backtest all history
    # exists, so None can only be the DEGENERATE case -> Option A: skip scan mode +
    # warn (sid=0 establishes as before). The INSUFFICIENT-HISTORY cause (a needed
    # neighbor candle past the live edge) cannot occur in backtest; if this warn
    # ever fires at a live edge, the wait-for-candles logic hooks in here
    # (memory/project_ad_hoc_zone_wait_for_candles).
    bos0_ref = build_ad_hoc_bos0_reference_zone(df, start_idx, struct_direction)
    cur_bos0_inner = bos0_ref.inner if bos0_ref is not None else None
    if cur_bos0_inner is None:
        print(
            f"[structure_engine][warn] main sid=0 cycle-0 ad-hoc BOS_0 zone could "
            f"not be built at start_idx={start_idx} (sd={struct_direction}); "
            f"skipping pre-CTS_0 scan mode (Option A) — sid=0 establishes as before. "
            f"If this fires at a live edge, the wait-for-candles logic hooks in here "
            f"(memory/project_ad_hoc_zone_wait_for_candles)."
        )

    # Run multiple structure segments until no more reversals (or we hit end)
    max_structures_guard = 20  # safety guard against infinite loops
    for loop_iter in range(max_structures_guard):
        # Pre-CTS_0 scan mode runs for EVERY sid whose BOS_0 inner was derivable:
        # sid=0 from the Scenario-1 ad-hoc above; sid>=1 from the reversal
        # unified_probe below (Step 4). None -> scan off for that sid (cycle-0
        # establishes the ordinary way), matching the sub-parity convention
        # `enforce_cts0_new_extreme=(bos0_inner is not None)`.
        _use_cts0_scan = (cur_bos0_inner is not None)
        ms = _make_market_structure(
            df2, struct_direction=struct_direction, start_idx=start_idx,
            structure_id=structure_id, timeframe=timeframe, pip_size=pip_size,
            enforce_cts0_new_extreme=_use_cts0_scan,
            bos0_inner=cur_bos0_inner if _use_cts0_scan else None,
        )
        ms.debug = True
        df2, ms_events, levels = ms.run()

        all_events.extend(ms_events)
        all_levels.extend(levels)

        # Find reversal idx for THIS structure_id (MarketStructure stops when it
        # hits reversal). The mask matches exactly ONE candle — the reversal
        # apply candle — because the run sets REVERSAL state at a single candle
        # then breaks (every back-fill stops at the terminal candle and
        # `_set_state` asserts nothing leaves REVERSAL — MARKET_STRUCTURE_SPEC
        # "Reversal inside a back-fill", 2026-09-28), and the terminal forward-stamp
        # touches market_state only (not structure_id). So .min() == .max() by
        # construction; we WARN (not crash) if that invariant is ever violated and
        # proceed with the apply idx. This single value supersedes the old reversal_start/confirmed pair
        # (vestigial names for the same candle — unification cleanup).
        rev_mask = (df2["market_state"].astype(str).str.lower() == "reversal") & (df2["structure_id"].astype(int) == structure_id)
        if not rev_mask.any():
            break

        reversal_apply_idx = int(df2.loc[rev_mask].index.min())
        _rev_max = int(df2.loc[rev_mask].index.max())
        if _rev_max != reversal_apply_idx:
            print(
                f"[structure_engine][warn] reversal mask spans >1 candle for "
                f"sid={structure_id} (min={reversal_apply_idx} max={_rev_max}); "
                f"proceeding with apply idx (.min())."
            )

        # --- Reversal handoff (Step 4): unified_probe replaces Scenario 2 +
        # Exception 1 + Exception 2 (project_unified_identify_start_probe.md /
        # project_true_first_breakout_cycle0.md). Mirrors the sub reversal path
        # (entity_df_mutation._resolve_reversal_start): reference = the prior sid's most
        # recent {CONF/UPD/EST} CTS; the probe runs in the flipped direction over
        # [the prior CTS anchor, reversal apply idx] and hands back a DECISION
        # (start + BOS_0 inner) — NOT events. The reversal structure is produced
        # by a fresh unbounded scan-from-start MS run in the next loop iteration.
        # `unified_probe` is imported locally: it imports _make_market_structure
        # from this module, so a top-level import would be circular.
        from engine_v2.structure.unified_probe import unified_probe

        probe_sd = -struct_direction
        # Prior sid's CTS zones for the CONFIRMED-zone reference lookup. Same
        # derivation the (now-removed) Exception-2 path used via
        # _get_last_cts_zone_bounds; build_reference_zone_from_cts_event falls
        # back to an ad-hoc CTS derivation when no confirmed zone exists.
        kl_zones = derive_kl_zones_v1(df2, all_events, struct_direction=struct_direction)
        ref_zone = build_reference_zone_from_cts_event(
            all_events,
            kl_zones,
            df2,
            sid=structure_id,
            probe_direction=probe_sd,
            idx_window=None,
        )
        if ref_zone is None:
            # Degenerate: a reversal with zero CTS events for the prior structure
            # (should not occur — a structure must establish a cycle to reverse).
            # Stop adding structures rather than guess a start (sub-parity).
            print(
                f"[structure_engine][warn] reversal reference zone unavailable "
                f"for sid={structure_id} — no CTS event for prior structure; "
                f"stopping (no reversal-born sid)."
            )
            break

        probe_input_idx = int(ref_zone.anchor_idx)
        if probe_input_idx >= reversal_apply_idx:
            # No forward scan window — degenerate (sub-parity).
            print(
                f"[structure_engine][warn] degenerate reversal probe window for "
                f"sid={structure_id} (input={probe_input_idx} "
                f"end={reversal_apply_idx}) — stopping (no reversal-born sid)."
            )
            break

        probe = unified_probe(
            df2.copy(),
            input_idx=probe_input_idx,
            direction=probe_sd,
            reference_zone=ref_zone,
            probe_end_idx=reversal_apply_idx,
            timeframe=timeframe,
            enable_phase2=False,
        )
        print(
            f"[structure_engine] unified_probe (reversal): sid={structure_id} "
            f"input={probe_input_idx} end={reversal_apply_idx} "
            f"ref={ref_zone.source} -> start={probe.starting_idx} "
            f"status={probe.status} cond={probe.finalize_condition} "
            f"iter={probe.iterations}"
        )
        if probe.status == "pending":
            # Only possible with end_idx=None; we always pass a defined end_idx,
            # so this is dormant. Guard for symmetry with the sub path.
            print(
                f"[structure_engine][warn] reversal unified_probe did not "
                f"finalize for sid={structure_id}; stopping (no reversal-born sid)."
            )
            break

        next_start_idx = int(probe.starting_idx)
        # Advance to the reversal-born structure. Its cycle-0 CTS_0 is
        # established by the next iteration's scan-from-start MS run gated on
        # this BOS_0 inner (a price -> slice-invariant; None -> scan off,
        # sub-parity).
        start_idx = next_start_idx
        struct_direction = probe_sd
        structure_id = structure_id + 1
        cur_bos0_inner = probe.bos0_inner

    notes = (
        f"MarketStructure v1: initial_start={d0.start_idx} initial_sd={d0.struct_direction} "
        f"structures={structure_id + 1} events={len(all_events)} levels={len(all_levels)} "
        f"start_reason={d0.reason}"
    )

    _validate_output(df2)

    return StructureEngineResult(
        df=df2,
        levels=all_levels,
        events=all_events,
        struct_direction=struct_direction,
        notes=notes,
    )


def compute_bounded_structure(
    df: pd.DataFrame,
    start_idx: int,
    struct_direction: int,
    *,
    timeframe: str = "H1",
    end_idx: Optional[int] = None,
    enforce_cts0_new_extreme: bool = False,
    bos0_inner: Optional[float] = None,
) -> BoundedStructureResult:
    """Run a bounded SINGLE directional structure and stop at its first reversal.

    Part 4 §5 (REVISED 2026-05-25) — the bounded single-structure primitive.
    Each subordinate sid is a SINGLE directional structure whose first
    reversal is the boundary to the next sid. This runs exactly one
    MarketStructure (structure_id=0) over ``[start_idx, end_idx]`` and reports
    where it reversed.

    Unlike ``compute_structure``, this does NOT roll past the reversal into
    further structure_ids, and does NOT run the reversal-handoff probe — those
    are the multi-structure concerns this primitive deliberately omits. The stop-at-first-reversal behaviour is
    already intrinsic to ``MarketStructure.run()`` (its loop breaks on
    ``MarketState.REVERSAL``); this wrapper just wires the proximity/zone
    resolvers (via ``_make_market_structure``) and extracts the reversal idx.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with candle features/patterns already applied.
    start_idx : int
        Structural anchor to run from (already validated by the upstream
        probe / identify_start — no internal Scenario 1).
    struct_direction : int
        +1 or -1. The single structure's direction for its whole run.
    timeframe : str
        TF for proximity-pips lookup (H1/M15/M5).
    end_idx : int, optional
        Inclusive upper bound on the run (parent-cycle-end or next subsequent
        trigger). The run is identical to an unbounded run on the frame
        truncated at ``end_idx`` (nothing past it is read — Plan A §2, with
        the ``attrs["imbalances"]`` residual noted there); no event carries an
        ``idx`` past it (asserted in ``MarketStructure.run()``).

    Returns
    -------
    BoundedStructureResult
        ``reversal_idx`` = the first reversal-marked candle (the reversal
        pattern's apply idx), or None if no reversal occurred within
        ``[start_idx, end_idx]``.
    """
    _validate_input(df)

    df2 = df  # MarketStructure copies internally; caller's df is untouched.
    pip_size = _pip_size_from_pair(df2)

    ms = _make_market_structure(
        df2,
        struct_direction=struct_direction,
        start_idx=start_idx,
        structure_id=0,
        timeframe=timeframe,
        pip_size=pip_size,
        end_idx=end_idx,
        enforce_cts0_new_extreme=enforce_cts0_new_extreme,
        bos0_inner=bos0_inner,
    )
    ms.debug = True
    df2, events, levels = ms.run()

    # Reversal detection: the first structure_id=0 row marked "reversal".
    # A single run only ever writes structure_id=0; the structure_id==0
    # filter also excludes any rows the terminal stamping marked "reversal"
    # past end_idx (those were never written, so keep structure_id=-1).
    rev_mask = (
        (df2["market_state"].astype(str).str.lower() == "reversal")
        & (df2["structure_id"].astype(int) == 0)
    )
    reversal_idx = int(df2.loc[rev_mask].index.min()) if rev_mask.any() else None

    _validate_output(df2)

    notes = (
        f"BoundedStructure: start={start_idx} sd={struct_direction} "
        f"end_idx={end_idx} reversal_idx={reversal_idx} "
        f"events={len(events)} levels={len(levels)}"
    )

    return BoundedStructureResult(
        df=df2,
        levels=levels,
        events=events,
        struct_direction=struct_direction,
        start_idx=start_idx,
        end_idx=end_idx,
        reversal_idx=reversal_idx,
        notes=notes,
    )


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[structure_engine] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[structure_engine] Input df is empty")


def _validate_output(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[structure_engine] Output df missing required columns: {missing}")


# ---------------------------------------------------------------------
# Exception 2 Probe Helpers
# ---------------------------------------------------------------------

def _pip_size_from_pair(df: pd.DataFrame) -> float:
    """Get pip size (0.01 for JPY pairs, 0.0001 otherwise)."""
    pair = df.attrs.get("pair", "")
    return 0.01 if "JPY" in str(pair).upper() else 0.0001


def _get_last_cts_zone_bounds(
    df: pd.DataFrame,
    events: List[StructureEvent],
    sid: int,
    sd: int,
) -> Optional[Tuple[float, float, str]]:
    """
    Get (outer, inner, zone_side) bounds of last CTS zone for given structure_id.
    For uptrend (sd=+1): CTS zone is sell zone, outer=top, inner=bottom
    For downtrend (sd=-1): CTS zone is buy zone, outer=bottom, inner=top

    Returns: (outer, inner, zone_side) or None if no CTS zone found.
    """
    zones = derive_kl_zones_v1(df, events, struct_direction=sd)
    cts_zones = [
        z for z in zones
        if z.source_kind == "CTS" and z.meta.get("structure_id") == sid
    ]
    if not cts_zones:
        return None

    # Sort by confirmed_idx to get the last one
    cts_zones.sort(key=lambda z: z.meta.get("confirmed_idx", -1))
    last = cts_zones[-1]

    # For sell zone: outer=top, inner=bottom
    # For buy zone: outer=bottom, inner=top
    if last.side == "sell":
        return (float(last.top), float(last.bottom), "sell")
    else:
        return (float(last.bottom), float(last.top), "buy")
