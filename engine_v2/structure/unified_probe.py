"""Unified identify-start + probe primitive (Phase 1).

Replaces the four asymmetric paths (Scenario 1, Scenario 2 + Exc1 + Exc2
H1-main reversal, Scenario 3 iterative for subs, Scenario 2 + Exc1 sub
reversal) with a single primitive whose contract is:

    "Given a candidate start `input_idx` and a price reference (the
    `reference_zone`), produce a validated structural anchor for a new
    structure running in `direction` from that anchor, bounded above by
    `end_idx`."

Design notes — full spec lives in
`memory/project_unified_identify_start_probe.md`. Highlights:

- Runs on the structure's own TF; no post-probe cross-TF mapping.
- Per iteration evaluates EXACTLY ONE candidate — the single most extreme
  retrace candle in the scan window (lowest low for `direction=+1`,
  highest high for `direction=-1`).
- Two-condition reset (BOTH must hold to restart from the candidate):
    1. Wick extreme within X pips of `reference_zone.inner`
       (X = `DEFAULT_PROBE_RESET_PIPS[timeframe]`).
    2. Toward-zone wick size ≤ Y pips
       (Y = `DEFAULT_PROBE_RESET_WICK[timeframe]`). Wick is body-bounded
       (`body_top = max(o,c)`, `body_bottom = min(o,c)`); wick side is
       opposite to `direction` (the side that points toward the zone
       outer).
- Max 10 iterations. The `reference_zone` is unchanged across iterations
  (the probe walks forward in time, not outward in reference).
- Scan window: `[first_CTS_ESTABLISHED.idx + 1, exc_upper]` where
  `exc_upper = end_idx` whenever defined (LANDMINES: "Probe `end_idx` Is
  the Supreme Bound"); fallback `cts_est[1].idx` only in live mode.

This module sits unused in Phase 1 — no caller migrates until Session 2+.
The primitive's existence + tests gate the per-trigger migrations that
follow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import pandas as pd

from engine_v2.common.types import REQUIRED_CANDLE_COLS
from engine_v2.structure.market_structure import StructureEvent
from engine_v2.structure.reference_zone import ReferenceZone

# `_make_market_structure` is the canonical MarketStructure constructor —
# it wires proximity resolvers + TF-keyed proximity/min-gap pips. Reusing
# it (rather than constructing MarketStructure directly) keeps the wiring
# in one place and avoids the spec §13.5.b inversion regression.
from engine_v2.structure.structure_engine import (
    _make_market_structure,
    _pip_size_from_pair,
)
from engine_v2.zones.zone_proximity import (
    DEFAULT_PROBE_RESET_PIPS,
    DEFAULT_PROBE_RESET_WICK,
)


FinalizeCondition = Literal[
    "no_retrace",            # ≥1 CTS_EST + candidate fails 2-condition reset → finalized
    "reversal_in_probe",     # reversal before 2nd CTS_EST → finalized
    "end_idx_reached",       # end_idx reached without CTS_EST → finalized at caller bound
    "no_cts_pending",        # no CTS_EST + end_idx None → pending (live mode only)
    "one_cts_pending",       # 1 CTS_EST + end_idx None → pending (can't bound check window)
    "max_iterations",        # all 10 iterations triggered reset → pending
]


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one `unified_probe` invocation.

    `start_idx` is the validated/finalized start when `status="finalized"`,
    or the best current candidate when `status="pending"` (caller may
    re-invoke once more data lands).

    `original_ref_zone` is the same `ReferenceZone` passed in — the probe
    never refines its reference. Returned for caller convenience /
    debug attribution.
    """
    start_idx: int
    status: Literal["finalized", "pending"]
    iterations: int
    original_ref_zone: ReferenceZone
    finalize_condition: FinalizeCondition
    notes: str = ""


def _select_extreme_retrace_candidate(
    df: pd.DataFrame,
    lo: int,
    hi: int,
    direction: int,
) -> Optional[int]:
    """Pick the single most-extreme retrace candle in `[lo, hi]`.

    For `direction=+1` (uptrend probe, retrace is downward): candle with
    minimum `l`. For `direction=-1`: candle with maximum `h`. Tie-break
    earliest (`.idxmin()` / `.idxmax()` return first occurrence).

    Returns None when the window is empty or out-of-bounds.
    """
    if lo > hi:
        return None
    seg = df.loc[lo:hi]
    if seg.empty:
        return None
    if direction == 1:
        return int(seg["l"].astype(float).idxmin())
    return int(seg["h"].astype(float).idxmax())


def _evaluate_reset_conditions(
    df: pd.DataFrame,
    candidate_idx: int,
    reference_zone: ReferenceZone,
    direction: int,
    reset_tol: float,
    wick_cap: float,
) -> bool:
    """Return True iff BOTH reset conditions hold on `candidate_idx`.

    Condition 1: wick extreme within `reset_tol` of `reference_zone.inner`.
        sd=+1: candidate `l` ≤ inner + reset_tol
        sd=-1: candidate `h` ≥ inner - reset_tol
    Condition 2: toward-zone wick ≤ `wick_cap` (body-bounded).
        sd=+1: body_bottom - l ≤ wick_cap   (lower wick)
        sd=-1: h - body_top ≤ wick_cap      (upper wick)
    """
    if candidate_idx not in df.index:
        return False
    o = float(df.loc[candidate_idx, "o"])
    h = float(df.loc[candidate_idx, "h"])
    low = float(df.loc[candidate_idx, "l"])
    c = float(df.loc[candidate_idx, "c"])
    body_top = max(o, c)
    body_bottom = min(o, c)
    inner = reference_zone.inner

    if direction == 1:
        cond1 = low <= inner + reset_tol
        cond2 = (body_bottom - low) <= wick_cap
    else:
        cond1 = h >= inner - reset_tol
        cond2 = (h - body_top) <= wick_cap
    return bool(cond1 and cond2)


def _collect_cts_established(
    events: List[StructureEvent],
    structure_id: int,
) -> List[StructureEvent]:
    """Return CTS_ESTABLISHED events for the given `structure_id`, sorted
    by idx ascending."""
    out = [
        ev for ev in events
        if ev.type == "CTS_ESTABLISHED"
        and ev.meta.get("structure_id") == structure_id
    ]
    out.sort(key=lambda e: int(e.idx))
    return out


def _has_reversal(df: pd.DataFrame, structure_id: int) -> bool:
    """Detect whether `structure_id` reached the reversal state inside
    `df`. Mirrors the predicate `compute_structure_scenario_3` uses."""
    rev_mask = (
        (df["market_state"].astype(str).str.lower() == "reversal")
        & (df["structure_id"].astype(int) == structure_id)
    )
    return bool(rev_mask.any())


def _validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_CANDLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[unified_probe] Missing required columns: {missing}")
    if df.empty:
        raise ValueError("[unified_probe] Input df is empty")


def unified_probe(
    df: pd.DataFrame,
    input_idx: int,
    direction: int,
    reference_zone: ReferenceZone,
    end_idx: Optional[int],
    timeframe: str,
    *,
    max_iterations: int = 10,
) -> ProbeResult:
    """Iteratively validate / refine `input_idx` against `reference_zone`.

    See module docstring for the full design. Per iteration:

    1. Run a fresh `MarketStructure` (sid=0) on `df.copy()` from the
       current candidate start through `end_idx`.
    2. Inspect outcomes:
       - Reversal before 2nd CTS_EST → finalize (`reversal_in_probe`).
       - 0 CTS_EST + `end_idx is None` → pending (`no_cts_pending`).
       - 0 CTS_EST + `end_idx` set → finalize at caller's bound
         (`end_idx_reached`).
       - 1 CTS_EST + `end_idx is None` → pending (`one_cts_pending`).
       - ≥1 CTS_EST + bounded check window → evaluate retrace.
    3. For the bounded check window `[cts_est[0].idx + 1, exc_upper]`,
       pick the single most-extreme retrace candle and evaluate both
       reset conditions. If both hold: restart from that candle. If either
       fails: finalize (`no_retrace`).

    Bounded `end_idx` parameter is the **supreme** upper bound per
    LANDMINES — internal rules like "2 CTS_EST seen" never narrow it.

    Parameters
    ----------
    df : DataFrame
        Candle features + patterns applied. Pair derived from `df.attrs`.
    input_idx : int
        Initial candidate start idx.
    direction : int
        +1 (uptrend probe) or -1 (downtrend probe).
    reference_zone : ReferenceZone
        Price target for the proximity check. Held constant across
        iterations.
    end_idx : int, optional
        Inclusive upper bound on the probe's MarketStructure run AND the
        retrace-check window. When None, pending paths become reachable.
    timeframe : str
        TF for threshold lookup (H1 / M15 / M5).
    max_iterations : int
        Cap on the iterative loop (default 10).
    """
    _validate_input(df)
    if direction not in (1, -1):
        raise ValueError(f"[unified_probe] direction must be ±1, got {direction}")

    pip_size = _pip_size_from_pair(df)
    reset_pips = DEFAULT_PROBE_RESET_PIPS.get(
        timeframe, DEFAULT_PROBE_RESET_PIPS["H1"]
    )
    wick_pips = DEFAULT_PROBE_RESET_WICK.get(
        timeframe, DEFAULT_PROBE_RESET_WICK["H1"]
    )
    reset_tol = float(reset_pips) * pip_size
    wick_cap = float(wick_pips) * pip_size

    current_start = int(input_idx)
    iteration = 0

    # Final fallthrough condition if the loop exhausts max_iterations
    # without finalizing — reported as pending so the caller can decide.
    final_condition: FinalizeCondition = "max_iterations"
    final_status: Literal["finalized", "pending"] = "pending"

    for iteration in range(1, max_iterations + 1):
        df_probe = df.copy()
        ms = _make_market_structure(
            df_probe,
            struct_direction=direction,
            start_idx=current_start,
            structure_id=0,
            end_idx=end_idx,
        )
        ms.debug = True
        df_probe, probe_events, _probe_levels = ms.run()

        cts_est = _collect_cts_established(probe_events, structure_id=0)
        n_cts = len(cts_est)
        has_rev = _has_reversal(df_probe, structure_id=0)

        # --- Reversal before 2nd CTS_EST → finalize ---
        if has_rev and n_cts < 2:
            final_condition = "reversal_in_probe"
            final_status = "finalized"
            break

        # --- 0 CTS_EST: bounded vs pending fork ---
        if n_cts == 0:
            if end_idx is not None:
                final_condition = "end_idx_reached"
                final_status = "finalized"
            else:
                final_condition = "no_cts_pending"
                final_status = "pending"
            break

        # --- Determine check window upper bound (end_idx supreme) ---
        if end_idx is not None:
            exc_upper = int(end_idx)
        elif n_cts >= 2:
            exc_upper = int(cts_est[1].idx)
        else:
            # 1 CTS_EST, no end_idx → can't bound the check window
            final_condition = "one_cts_pending"
            final_status = "pending"
            break

        check_lo = int(cts_est[0].idx) + 1
        candidate_idx = _select_extreme_retrace_candidate(
            df_probe, check_lo, exc_upper, direction,
        )

        if candidate_idx is None:
            # Empty window → no candle to reset against. Finalize.
            final_condition = "no_retrace"
            final_status = "finalized"
            break

        passes = _evaluate_reset_conditions(
            df_probe, candidate_idx, reference_zone, direction,
            reset_tol, wick_cap,
        )

        if not passes:
            # Either condition failed → no qualifying retrace → finalize.
            final_condition = "no_retrace"
            final_status = "finalized"
            break

        # Both conditions hold → restart from this candidate.
        print(
            f"[unified_probe] reset triggered: iter={iteration} "
            f"candidate={candidate_idx} (was {current_start})"
        )
        current_start = int(candidate_idx)
        # Continue loop with fresh probe from the new start.

    notes = (
        f"unified_probe: start={current_start} direction={direction} "
        f"iter={iteration} status={final_status} condition={final_condition} "
        f"end_idx={end_idx} timeframe={timeframe}"
    )

    return ProbeResult(
        start_idx=current_start,
        status=final_status,
        iterations=iteration,
        original_ref_zone=reference_zone,
        finalize_condition=final_condition,
        notes=notes,
    )
