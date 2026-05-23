# engine_v2/zones/fib_tracker.py
"""
Fibonacci level tracking for POI Zones.

Fib levels are drawn from BOS to CTS within each market structure cycle.
This module handles activation, updates, and lifecycle management.

Scenario Logic (sid 1+ only):
- Scenario 1: CTS_0 idx >= reversal_confirmed_idx → cycle 0 Fib unlocked
- Scenario 2: Cross-cycle Fib (BOS_0 → CTS_1) when all conditions met
- Scenario 3: Normal cycle 1 Fib when cross-cycle conditions fail
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any

import pandas as pd

from engine_v2.features.fibonacci import (
    FibRetracement,
    create_fib_retracement,
    DEFAULT_FIB_LEVELS,
)
from engine_v2.patterns.imbalance import has_unfilled_imbalance, get_unfilled_imbalances
from engine_v2.structure.market_structure import StructureEvent


@dataclass(frozen=True)
class FibState:
    """
    State of a Fibonacci retracement for a single cycle.

    Immutable - updates create new instances.
    """
    # Identity
    structure_id: int
    cycle_id: int
    struct_direction: int  # +1 bullish, -1 bearish

    # Anchors
    bos_idx: int
    bos_price: float
    cts_idx: int
    cts_price: float

    # State
    active: bool = True
    locked: bool = False  # True after CTS_CONFIRMED

    # The actual Fib retracement (computed from anchors)
    fib: Optional[FibRetracement] = None

    # Metadata for logging/debugging
    meta: Dict[str, Any] = field(default_factory=dict)

    # History of CTS anchor changes (for logging)
    cts_history: tuple = field(default_factory=tuple)


@dataclass
class FibTrackerConfig:
    """Configuration for FibTracker."""
    fib_levels: List[float] = field(default_factory=lambda: DEFAULT_FIB_LEVELS.copy())
    fill_threshold: float = 0.70  # 70% = filled


def select_fib_anchor_for_cycle(
    df: pd.DataFrame,
    sid: int,
    cycle_id: int,
    bos_idx: int,
    bos_price: float,
    cts_idx: int,
    cts_price: float,
    c0_data: Optional[Dict[str, Any]],
    fill_threshold: float = 0.70,
    struct_direction: int = 0,
) -> tuple:
    """Pick the Fib anchor for a cycle. Pure function — no FibTracker state.

    Returns ``(anchor_bos_idx, anchor_bos_price, anchor_cts_idx,
    anchor_cts_price, scenario_label)``. Cross-cycle returns are signalled
    by an anchor_bos_idx that differs from the input bos_idx.

    Decision tree (mirrors FibTracker._handle_cycle1_scenarios for cycle 1):

    - ``sid == 0`` or ``cycle_id != 1`` or ``c0_data is None`` →
      intra-cycle, label ``"intra"``.
    - ``c0_data["scenario1"] is True`` → intra-cycle (cycle 1 stays normal),
      label ``"scenario_1"``.
    - Otherwise evaluate Scenario 2 conditions over the imbalance set:

        cond1 — cycle 1 has unfilled sd-direction imbalance in [BOS_1, CTS_1]
                as of cts_idx
        cond2 — cycle 0 has unfilled sd-direction imbalance (cached on
                ``c0_data``; caller computed with the same direction filter)
        cond3 — BOS_1 has not filled cycle 0's sd-direction imbalances

      All three true → cross-cycle, label ``"scenario_2_cross"`` (anchor
      becomes BOS_0 → CTS_1). Else → intra-cycle, label ``"scenario_3"``.

    ``struct_direction`` is the sd of the structure being evaluated (+1 / -1).
    Threads into cond1 / cond3 as the imbalance direction filter — counter-
    direction imbalances never become POIs, so they shouldn't influence the
    Scenario 2 decision. cond2 inherits this filter through whatever upstream
    computed ``c0_data["has_unfilled"]``. ``struct_direction == 0`` (the
    default) preserves legacy permissive behaviour if a caller hasn't
    migrated yet.

    Approximation (intentional): Scenario 1 REVERT — FibTracker reverts
    Scenario 1 from TRUE to FALSE if BOS_1 touches the prev structure's BOS
    zone outer, then evaluates Scenario 2/3. This utility cannot detect
    that itself because ``prev_bos_outer`` / ``prev_sd`` aren't inputs. A
    caller that tracks the revert state should flip ``c0_data["scenario1"]``
    from True to False/None BEFORE invoking this function and let the
    Scenario 2/3 path take over. MarketStructure's in-flight POI snapshot
    intentionally leaves ``scenario1`` as None (it doesn't track Scenario 1
    at all), so the utility always falls into the Scenario 2/3 evaluation
    path for cycle 1 — matching FibTracker's behavior whenever Scenario 1
    is False (the common case).
    """
    if sid < 1 or cycle_id != 1 or c0_data is None:
        return (int(bos_idx), float(bos_price), int(cts_idx), float(cts_price), "intra")

    if c0_data.get("scenario1") is True:
        return (int(bos_idx), float(bos_price), int(cts_idx), float(cts_price), "scenario_1")

    sd_filter = int(struct_direction) if struct_direction in (1, -1) else None
    c1_lo = int(min(bos_idx, cts_idx))
    c1_hi = int(max(bos_idx, cts_idx))
    cond1 = has_unfilled_imbalance(
        df, c1_lo, c1_hi, int(cts_idx), fill_threshold, direction=sd_filter,
    )
    cond2 = bool(c0_data.get("has_unfilled", False))

    c0_bos_idx = int(c0_data["bos_idx"])
    c0_cts_idx = int(c0_data["cts_idx"])
    c0_lo = min(c0_bos_idx, c0_cts_idx)
    c0_hi = max(c0_bos_idx, c0_cts_idx)
    cond3 = has_unfilled_imbalance(
        df, c0_lo, c0_hi, int(bos_idx), fill_threshold, direction=sd_filter,
    )

    if cond1 and cond2 and cond3:
        return (
            c0_bos_idx,
            float(c0_data["bos_price"]),
            int(cts_idx),
            float(cts_price),
            "scenario_2_cross",
        )
    return (int(bos_idx), float(bos_price), int(cts_idx), float(cts_price), "scenario_3")


class FibTracker:
    """
    Tracks Fibonacci levels across market structure cycles.

    Lifecycle:
    1. CTS_ESTABLISHED + unfilled imbalance → Fib activated
    2. CTS moves to new extreme → anchor 2 updates
    3. All imbalances filled → Fib deactivates
    4. CTS_CONFIRMED → Fib locked (stops updating)

    Scenario Logic (sid 1+ only):
    - Scenario 1 checked at each CTS_0 ESTABLISHED/UPDATED
    - If CTS_0 idx >= rv_idx → Scenario 1 TRUE (permanent), cycle 0 Fib unlocked
    - If still FALSE at CTS_0 CONFIRMED → Scenario 1 FALSE (permanent)
    - Scenario 2/3 determined at CTS_1 ESTABLISHED (only when Scenario 1 is FALSE)

    Only 1 active Fib per structure at a time (new cycle obsoletes previous).
    """

    def __init__(self, config: Optional[FibTrackerConfig] = None, fib_mode: str = "h1"):
        self.config = config or FibTrackerConfig()
        self.fib_mode = fib_mode  # "h1" (default) or "cross_cycle"
        # "cross_cycle" enables generalized cross-cycle fibs + pre_established
        # phase support. Used by all subordinate-structure pipelines (counter
        # and confluence, M15 today, deeper TFs in the future). The legacy
        # name was "m15_reverse" before subordinate structures generalized
        # beyond counter/reverse direction.

        # Fib states per key. Two key shapes coexist in this dict:
        #   - (sid, cycle_id)                          → single fib
        #   - (sid, cycle_id, "cross", version)        → cross fib (cross_cycle only)
        # `_fibs.values()` yields all fibs for charting iteration.
        self._fibs: Dict[tuple, FibState] = {}

        # Track which cycle is "current" per structure (for obsolescence)
        # {structure_id: cycle_id}
        self._current_cycle: Dict[int, int] = {}

        # Track cross-cycle Fib eligibility (H1 post-reversal, sid 1+ only)
        # {structure_id: {"cycle0": {...}, "normal_cycle1": FibState, "cross_cycle": FibState}}
        self._cross_cycle_data: Dict[int, Dict] = {}

        # Track Scenario 1 resolution per structure (H1 sid 1+ only)
        # {structure_id: True/False/None}
        # None = undetermined, True = CTS_0 >= rv_idx, False = resolved at CTS_0 CONFIRMED
        self._scenario1: Dict[int, Optional[bool]] = {}

        # ------------------------------------------------------------------
        # M15 reverse mode: cross-fib state
        # ------------------------------------------------------------------
        # Per (sid, cycle_id): phase ∈ {"pre_established", "established", "confirmed"}.
        # Lifecycle: implicit → pre_established (on CTS_{n-1} CONFIRMED)
        #            → established (on CTS_n ESTABLISHED)
        #            → confirmed (on CTS_n CONFIRMED).
        # Cycle 0 has no pre_established phase.
        self._m15_phase: Dict[tuple, str] = {}

        # Per sid: set of cycle_ids whose [BOS_k, CTS_k] imbalances are all filled.
        # Once dead, a cycle stays dead (filling is permanent). Populated lazily
        # during walk-backward in cross-fib checks.
        self._dead_cycles: Dict[int, set] = {}

        # Per (sid, cycle_id): idx where CTS_k CONFIRMED fired. Used to compute
        # the prospective BOS for the NEXT cycle's pre-established phase.
        self._cts_confirmed_idx: Dict[tuple, int] = {}

        # Per (sid, cycle_id): (bos_idx, bos_price) captured on CTS_ESTABLISHED.
        # Used by the cross-fib walk-backward to look up each prior cycle's BOS.
        self._bos_by_cycle: Dict[tuple, tuple] = {}

        # Per (sid, cycle_id): (cts_idx, cts_price) captured on CTS_CONFIRMED
        # (the locked/final CTS). Only cycles with CONFIRMED CTS are eligible
        # for dead-cycle evaluation.
        self._cts_by_cycle: Dict[tuple, tuple] = {}

        # Per (sid, cycle_id): highest cross-fib version currently stored.
        # Cross keys are (sid, cycle_id, "cross", version). Extension replaces
        # in-place (same version); shrink bumps version by 1 and creates a
        # new entry, marking the old one inactive.
        self._cross_version: Dict[tuple, int] = {}

    def on_cts_established(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        bos_idx: int,
        bos_price: float,
        reversal_confirmed_idx: Optional[int] = None,
        prev_bos_outer: Optional[float] = None,
        prev_sd: Optional[int] = None,
    ) -> Optional[FibState]:
        """
        Handle CTS_ESTABLISHED event - potentially activate a new Fib.

        Parameters
        ----------
        event : StructureEvent
            The CTS_ESTABLISHED event
        df : DataFrame
            OHLC data with imbalance columns
        bos_idx : int
            Index of the confirmed BOS (anchor 1)
        bos_price : float
            Price of the confirmed BOS
        reversal_confirmed_idx : int, optional
            Index where the reversal from previous structure was confirmed.
            Required for sid 1+ Scenario 1/2/3 logic.
        prev_bos_outer : float, optional
            The max expanded outer threshold of sid N-1's last BOS zone.
            Used for Scenario 1 revert check at CTS_1 ESTABLISHED.
        prev_sd : int, optional
            Direction of previous structure (+1 bullish, -1 bearish).
            Used for Scenario 1 revert check at CTS_1 ESTABLISHED.

        Returns
        -------
        FibState or None
            New FibState if activated, None otherwise
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))
        sd = int(event.meta.get("struct_direction", 0))
        cts_idx = int(event.idx)
        cts_price = float(event.price) if event.price else 0.0

        # Get CTS price from event or df
        if cts_price == 0.0 and cts_idx in df.index:
            if sd == 1:
                cts_price = float(df.loc[cts_idx, "h"])
            else:
                cts_price = float(df.loc[cts_idx, "l"])

        # Check for unfilled imbalance between BOS and CTS. sd-direction
        # filter: Fibs only ever produce sd-direction POIs, so counter-
        # direction imbalances in the BOS->CTS swing don't justify activation.
        start_idx = min(bos_idx, cts_idx)
        end_idx = max(bos_idx, cts_idx)
        has_unfilled = has_unfilled_imbalance(
            df, start_idx, end_idx, cts_idx, self.config.fill_threshold,
            direction=sd,
        )

        # Populate BOS lookup (used by cross-fib walk-backward in cross_cycle)
        self._bos_by_cycle[(sid, cycle_id)] = (bos_idx, bos_price)

        # ============================================================
        # BRANCH: fib_mode determines logic
        # ============================================================
        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_established(
                sid, cycle_id, sd, bos_idx, bos_price, cts_idx, cts_price, has_unfilled, df
            )

        if sid == 0:
            return self._handle_sid0_cts_established(
                sid, cycle_id, sd, bos_idx, bos_price, cts_idx, cts_price, has_unfilled, df
            )
        else:
            return self._handle_sid1plus_cts_established(
                sid, cycle_id, sd, bos_idx, bos_price, cts_idx, cts_price,
                has_unfilled, df, reversal_confirmed_idx, prev_bos_outer, prev_sd
            )

    def _handle_cross_cycle_cts_established(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """Handle CTS_ESTABLISHED for M15 reverse structure.

        - Cycle 0: single-fib only (no cross possible, no pre-established phase).
        - Cycle ≥1: phase flips pre_established → established. Run cross-fib
          check with anchor = CTS_n. If cross fails, fall back to single.
        """
        # Flip phase to established
        self._m15_phase[(sid, cycle_id)] = "established"

        if cycle_id == 0:
            # Cycle 0: no cross possible. Simple single-fib activation.
            if not has_unfilled:
                print(f"[fib] cross_cycle sid={sid} cycle=0 NO FIB: no unfilled imbalance")
                return None
            print(f"[fib] cross_cycle sid={sid} cycle=0 ACTIVATED (single, no cross)")
            return self._activate_fib(
                sid=sid,
                cycle_id=0,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_idx, "fib_mode": "cross_cycle"},
            )

        # Cycle ≥1: run cross-fib check with CTS_n as anchor
        # own_imb_start = BOS_n (just-confirmed BOS for this cycle)
        print(f"[fib] cross_cycle sid={sid} cycle={cycle_id} CTS_ESTABLISHED: "
              f"running cross check (anchor=CTS_n)")
        self._m15_cross_check(
            sid=sid,
            target_cycle=cycle_id,
            df=df,
            sd=sd,
            current_candle=cts_idx,
            anchor_idx=cts_idx,
            anchor_price=cts_price,
            own_imb_start=bos_idx,
        )
        # Return the currently active fib for this cycle (cross or single)
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            return latest[1]
        return self._fibs.get((sid, cycle_id))

    def _handle_sid0_cts_established(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """
        Handle CTS_ESTABLISHED for sid=0 (no prior reversal).

        Simple flow:
        - Cycle 0: No Fib
        - Cycle 1+: Normal Fib (if unfilled imbalance)
        - No cross-cycle logic
        """
        if cycle_id == 0:
            # sid=0, cycle=0: No Fib - just store data for reference
            print(f"[fib] sid=0 cycle=0 NO FIB (simple flow): BOS idx={bos_idx} -> CTS idx={cts_idx}")
            return None

        # sid=0, cycle 1+: Normal Fib activation
        if not has_unfilled:
            print(f"[fib] sid=0 cycle={cycle_id} NOT activated (simple flow): no unfilled imbalance")
            return None

        print(f"[fib] sid=0 cycle={cycle_id} ACTIVATED (simple flow)")
        return self._activate_fib(
            sid=sid,
            cycle_id=cycle_id,
            sd=sd,
            bos_idx=bos_idx,
            bos_price=bos_price,
            cts_idx=cts_idx,
            cts_price=cts_price,
            meta={"activated_at": cts_idx, "flow": "simple"},
        )

    def _handle_sid1plus_cts_established(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
        prev_bos_outer: Optional[float] = None,
        prev_sd: Optional[int] = None,
    ) -> Optional[FibState]:
        """
        Handle CTS_ESTABLISHED for sid=1+ (post-reversal).

        Scenario logic:
        - Cycle 0: Check Scenario 1 (CTS_0 idx >= rv_idx)
          - If TRUE → cycle 0 Fib unlocked (can activate if unfilled imbalance)
          - If undetermined → store data, no Fib yet
        - Cycle 1: If Scenario 1 is TRUE, check revert condition; if FALSE check Scenario 2/3
        - Cycle 2+: Normal Fib
        """
        # --- Cycle 0: Scenario 1 check ---
        if cycle_id == 0:
            return self._handle_cycle0_scenario1(
                sid, sd, bos_idx, bos_price, cts_idx, cts_price,
                has_unfilled, reversal_confirmed_idx
            )

        # --- Cycle 1: Depends on Scenario 1 resolution ---
        if cycle_id == 1:
            return self._handle_cycle1_scenarios(
                sid, sd, bos_idx, bos_price, cts_idx, cts_price,
                has_unfilled, df, reversal_confirmed_idx,
                prev_bos_outer, prev_sd
            )

        # --- Cycle 2+: Normal Fib ---
        if not has_unfilled:
            print(f"[fib] sid={sid} cycle={cycle_id} NOT activated: no unfilled imbalance")
            return None

        return self._activate_fib(
            sid=sid,
            cycle_id=cycle_id,
            sd=sd,
            bos_idx=bos_idx,
            bos_price=bos_price,
            cts_idx=cts_idx,
            cts_price=cts_price,
            meta={"activated_at": cts_idx},
        )

    def _handle_cycle0_scenario1(
        self,
        sid: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        reversal_confirmed_idx: Optional[int],
    ) -> Optional[FibState]:
        """
        Handle cycle 0 CTS_ESTABLISHED for sid 1+ - check Scenario 1.

        Scenario 1: CTS_0 idx >= reversal_confirmed_idx
        - If TRUE → cycle 0 Fib unlocked (permanent)
        - If FALSE/undetermined → store data for cross-cycle check
        """
        # Initialize cross-cycle data storage
        if sid not in self._cross_cycle_data:
            self._cross_cycle_data[sid] = {}

        # Store cycle 0 data
        self._cross_cycle_data[sid]["cycle0"] = {
            "bos_idx": bos_idx,
            "bos_price": bos_price,
            "cts_idx": cts_idx,
            "cts_price": cts_price,
            "struct_direction": sd,
            "has_unfilled": has_unfilled,
            "locked": False,
        }

        # Check Scenario 1: CTS_0 idx >= rv_idx
        if reversal_confirmed_idx is not None and cts_idx >= reversal_confirmed_idx:
            # Scenario 1 TRUE (permanent) - cycle 0 Fib unlocked
            self._scenario1[sid] = True
            print(f"[fib] sid={sid} cycle=0 Scenario 1 TRUE at idx={cts_idx} (CTS >= rv_idx={reversal_confirmed_idx})")

            if has_unfilled:
                return self._activate_fib(
                    sid=sid,
                    cycle_id=0,
                    sd=sd,
                    bos_idx=bos_idx,
                    bos_price=bos_price,
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={"activated_at": cts_idx, "scenario1": True},
                )
            else:
                print(f"[fib] sid={sid} cycle=0 NOT activated: Scenario 1 TRUE but no unfilled imbalance")
                return None
        else:
            # Scenario 1 undetermined - store data, no Fib yet
            if sid not in self._scenario1:
                self._scenario1[sid] = None  # Undetermined
            print(f"[fib] sid={sid} cycle=0 STORED for cross-cycle check: BOS idx={bos_idx} -> CTS idx={cts_idx}, has_unfilled={has_unfilled} (Scenario 1 undetermined)")
            return None

    def _handle_cycle1_scenarios(
        self,
        sid: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        has_unfilled: bool,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
        prev_bos_outer: Optional[float] = None,
        prev_sd: Optional[int] = None,
    ) -> Optional[FibState]:
        """
        Handle cycle 1 CTS_ESTABLISHED for sid 1+.

        If Scenario 1 is TRUE → check revert condition first
          - If BOS_1 touches prev BOS zone outer → revert to FALSE, deactivate cycle 0 Fib
          - If no touch → stays TRUE, normal cycle 1 Fib
        If Scenario 1 is FALSE → check Scenario 2/3
        """
        scenario1 = self._scenario1.get(sid)

        # NEW: If Scenario 1 was TRUE, check revert condition
        if scenario1 is True and prev_bos_outer is not None and prev_sd is not None:
            if self._should_revert_scenario1(bos_price, prev_bos_outer, prev_sd):
                # Revert Scenario 1 to FALSE
                self._scenario1[sid] = False
                # Deactivate cycle 0 Fib if it exists
                self._deactivate_cycle0_fib(sid)
                scenario1 = False
                print(f"[fib] sid={sid} Scenario 1 REVERTED to FALSE (BOS_1={bos_price:.5f} touched prev BOS zone outer={prev_bos_outer:.5f})")
            else:
                print(f"[fib] sid={sid} Scenario 1 stays TRUE (BOS_1={bos_price:.5f} did NOT touch prev BOS zone outer={prev_bos_outer:.5f})")

        # Scenario 1 TRUE: Normal cycle 1 Fib
        if scenario1 is True:
            if not has_unfilled:
                print(f"[fib] sid={sid} cycle=1 NOT activated (Scenario 1): no unfilled imbalance")
                return None

            print(f"[fib] sid={sid} cycle=1 ACTIVATED (Scenario 1 TRUE, normal flow)")
            return self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_idx, "scenario1": True},
            )

        # Scenario 1 FALSE: Check Scenario 2/3
        # (scenario1 is False or None - if None, it was resolved FALSE at CTS_0 CONFIRMED)
        if sid not in self._cross_cycle_data or "cycle0" not in self._cross_cycle_data[sid]:
            # No cycle 0 data - fallback to normal
            if not has_unfilled:
                print(f"[fib] sid={sid} cycle=1 NOT activated: no cycle 0 data, no unfilled imbalance")
                return None

            return self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_idx},
            )

        c0 = self._cross_cycle_data[sid]["cycle0"]

        # Route cross vs intra anchor selection through the shared utility so
        # MarketStructure's in-flight POI resolver and this downstream layer
        # agree on Scenario 2 (see select_fib_anchor_for_cycle for the
        # decision contract). `scenario1` here is the post-revert value
        # (Scenario 1 TRUE was handled above); pass it explicitly so the
        # utility doesn't have to re-derive Scenario 1 from inputs.
        c0_for_utility = dict(c0)
        c0_for_utility["scenario1"] = scenario1

        anchor_bos_idx, anchor_bos_price, anchor_cts_idx, anchor_cts_price, label = (
            select_fib_anchor_for_cycle(
                df,
                sid,
                1,
                bos_idx,
                bos_price,
                cts_idx,
                cts_price,
                c0_for_utility,
                self.config.fill_threshold,
                struct_direction=sd,
            )
        )
        print(f"[fib] sid={sid} cycle=1 anchor decision: label={label} "
              f"(has_unfilled={has_unfilled})")

        if label == "scenario_2_cross":
            # Scenario 2: Cross-cycle Fib
            print(f"[fib] sid={sid} cycle=1 Scenario 2: CROSS-CYCLE ACTIVATED: "
                  f"BOS_0 idx={anchor_bos_idx} -> CTS_1 idx={anchor_cts_idx}")

            # Create normal cycle 1 Fib for fallback (FibTracker tracks
            # this separately for its own bookkeeping; not the utility's
            # responsibility)
            normal_fib = FibState(
                structure_id=sid,
                cycle_id=1,
                struct_direction=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                active=True,
                locked=False,
                fib=self._create_fib_retracement(sd, bos_idx, bos_price, cts_idx, cts_price, sid, 1),
                meta={"activated_at": cts_idx, "scenario": 2, "role": "fallback"},
                cts_history=((cts_idx, cts_price),),
            )
            self._cross_cycle_data[sid]["normal_cycle1"] = normal_fib
            print(f"[fib] sid={sid} cycle=1 NORMAL computed (fallback): "
                  f"BOS idx={bos_idx} -> CTS idx={cts_idx}")

            # Activate the cross-cycle Fib using utility-chosen anchors
            cross_fib = self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=anchor_bos_idx,
                bos_price=anchor_bos_price,
                cts_idx=anchor_cts_idx,
                cts_price=anchor_cts_price,
                meta={
                    "cross_cycle": True,
                    "scenario": 2,
                    "activated_at": cts_idx,
                    "cycle1_bos_idx": bos_idx,
                },
            )
            self._cross_cycle_data[sid]["cross_cycle"] = cross_fib
            return cross_fib

        if has_unfilled:
            # Scenario 3: Normal cycle 1 Fib (utility selected intra anchors)
            print(f"[fib] sid={sid} cycle=1 Scenario 3: NORMAL ACTIVATED: "
                  f"BOS idx={anchor_bos_idx} -> CTS idx={anchor_cts_idx}")
            return self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=anchor_bos_idx,
                bos_price=anchor_bos_price,
                cts_idx=anchor_cts_idx,
                cts_price=anchor_cts_price,
                meta={"activated_at": cts_idx, "scenario": 3},
            )

        # No unfilled imbalance in cycle 1
        print(f"[fib] sid={sid} cycle=1 NOT activated: no unfilled imbalance in cycle 1")
        return None

    def _create_fib_retracement(
        self,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        sid: int,
        cycle_id: int,
    ) -> FibRetracement:
        """Helper to create a FibRetracement from anchors."""
        if sd == 1:  # Bullish
            anchor_high = cts_price
            anchor_low = bos_price
            anchor_high_idx = cts_idx
            anchor_low_idx = bos_idx
        else:  # Bearish
            anchor_high = bos_price
            anchor_low = cts_price
            anchor_high_idx = bos_idx
            anchor_low_idx = cts_idx

        return create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            anchor_high_idx=anchor_high_idx,
            anchor_low_idx=anchor_low_idx,
            meta={"structure_id": sid, "cycle_id": cycle_id},
        )

    def _activate_fib(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        bos_idx: int,
        bos_price: float,
        cts_idx: int,
        cts_price: float,
        meta: Optional[Dict] = None,
    ) -> FibState:
        """Internal helper to create and store a FibState."""
        if sd == 1:  # Bullish
            anchor_high = cts_price
            anchor_low = bos_price
            anchor_high_idx = cts_idx
            anchor_low_idx = bos_idx
        else:  # Bearish
            anchor_high = bos_price
            anchor_low = cts_price
            anchor_high_idx = bos_idx
            anchor_low_idx = cts_idx

        fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            anchor_high_idx=anchor_high_idx,
            anchor_low_idx=anchor_low_idx,
            meta={"structure_id": sid, "cycle_id": cycle_id},
        )

        state = FibState(
            structure_id=sid,
            cycle_id=cycle_id,
            struct_direction=sd,
            bos_idx=bos_idx,
            bos_price=bos_price,
            cts_idx=cts_idx,
            cts_price=cts_price,
            active=True,
            locked=False,
            fib=fib,
            meta=meta or {},
            cts_history=((cts_idx, cts_price),),
        )

        # Mark previous cycle's Fib as obsolete (if exists)
        prev_cycle = self._current_cycle.get(sid)
        if prev_cycle is not None and prev_cycle != cycle_id:
            prev_key = (sid, prev_cycle)
            if prev_key in self._fibs:
                old_fib = self._fibs[prev_key]
                obsolete = replace(old_fib, active=False, meta={**old_fib.meta, "obsolete_reason": "new_cycle"})
                self._fibs[prev_key] = obsolete

        key = (sid, cycle_id)
        self._fibs[key] = state
        self._current_cycle[sid] = cycle_id

        print(f"[fib] sid={sid} cycle={cycle_id} ACTIVATED: BOS idx={bos_idx} price={bos_price:.5f} -> CTS idx={cts_idx} price={cts_price:.5f}")

        return state

    def on_cts_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int] = None,
    ) -> Optional[FibState]:
        """
        Handle CTS_UPDATED event - update CTS anchor if Fib is active.

        Parameters
        ----------
        event : StructureEvent
            The CTS_UPDATED event
        df : DataFrame
            OHLC data
        reversal_confirmed_idx : int, optional
            Index where the reversal was confirmed. Required for sid 1+ Scenario 1 check.

        Returns
        -------
        FibState or None
            Updated FibState, or None if no active Fib for this cycle
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))
        sd = int(event.meta.get("struct_direction", 0))
        cts_idx = int(event.idx)
        cts_price = float(event.price) if event.price else 0.0

        # Get CTS price from df if not in event
        if cts_price == 0.0 and cts_idx in df.index:
            if sd == 1:
                cts_price = float(df.loc[cts_idx, "h"])
            else:
                cts_price = float(df.loc[cts_idx, "l"])

        # ============================================================
        # BRANCH: fib_mode → cross_cycle / sid=0 / sid≥1
        # ============================================================
        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_updated(sid, cycle_id, sd, cts_idx, cts_price, df)

        if sid == 0:
            return self._handle_sid0_cts_updated(sid, cycle_id, cts_idx, cts_price, df)
        else:
            return self._handle_sid1plus_cts_updated(
                sid, cycle_id, sd, cts_idx, cts_price, df, reversal_confirmed_idx
            )

    def _handle_cross_cycle_cts_updated(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED for cross_cycle. Re-run cross check with the
        extended CTS anchor. Cycle 0 uses simple single-fib update."""
        if cycle_id == 0:
            # Cycle 0: no cross logic. Update single fib via shared helper.
            key = (sid, 0)
            if key not in self._fibs:
                return None
            return self._update_fib_cts(key, cts_idx, cts_price, df)

        # Cycle ≥1: should be in established phase. Re-run cross check.
        phase = self._m15_phase.get((sid, cycle_id))
        if phase != "established":
            # Shouldn't happen — CTS_UPDATED implies ESTABLISHED has already fired
            print(f"[fib][cross_cycle][warn] CTS_UPDATED for sid={sid} cycle={cycle_id} "
                  f"in unexpected phase={phase}")
            return None

        bos = self._bos_by_cycle.get((sid, cycle_id))
        if bos is None:
            return None

        self._m15_cross_check(
            sid=sid,
            target_cycle=cycle_id,
            df=df,
            sd=sd,
            current_candle=cts_idx,
            anchor_idx=cts_idx,
            anchor_price=cts_price,
            own_imb_start=bos[0],
        )
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            return latest[1]
        return self._fibs.get((sid, cycle_id))

    def _handle_sid0_cts_updated(
        self,
        sid: int,
        cycle_id: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED for sid=0 (simple flow)."""
        if cycle_id == 0:
            # sid=0, cycle=0: No Fib
            return None

        # sid=0, cycle 1+: Update normal Fib if exists
        key = (sid, cycle_id)
        if key not in self._fibs:
            return None

        return self._update_fib_cts(key, cts_idx, cts_price, df)

    def _handle_sid1plus_cts_updated(
        self,
        sid: int,
        cycle_id: int,
        sd: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
    ) -> Optional[FibState]:
        """Handle CTS_UPDATED for sid=1+ (post-reversal)."""
        # --- Cycle 0: Scenario 1 re-check ---
        if cycle_id == 0:
            return self._handle_cycle0_cts_updated(
                sid, sd, cts_idx, cts_price, df, reversal_confirmed_idx
            )

        # --- Cycle 1: Update cross-cycle or normal Fib ---
        if cycle_id == 1 and sid in self._cross_cycle_data and "cross_cycle" in self._cross_cycle_data[sid]:
            return self._update_cycle1_fibs(sid, cts_idx, cts_price, df)

        # --- Cycle 1+ normal Fib update ---
        key = (sid, cycle_id)
        if key not in self._fibs:
            return None

        return self._update_fib_cts(key, cts_idx, cts_price, df)

    def _handle_cycle0_cts_updated(
        self,
        sid: int,
        sd: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
        reversal_confirmed_idx: Optional[int],
    ) -> Optional[FibState]:
        """
        Handle cycle 0 CTS_UPDATED for sid 1+ - re-check Scenario 1.

        If Scenario 1 is already TRUE, update the cycle 0 Fib.
        If Scenario 1 is still undetermined, check again.
        """
        scenario1 = self._scenario1.get(sid)

        # Update stored cycle 0 data
        if sid in self._cross_cycle_data and "cycle0" in self._cross_cycle_data[sid]:
            c0 = self._cross_cycle_data[sid]["cycle0"]
            if not c0.get("locked", False) and cts_idx > c0["cts_idx"]:
                c0["cts_idx"] = cts_idx
                c0["cts_price"] = cts_price
                # Re-check unfilled imbalance. sd-direction filter mirrors
                # the activation-time filter at on_cts_established so the
                # cycle-0 snapshot stays direction-consistent across updates.
                start_idx = min(c0["bos_idx"], cts_idx)
                end_idx = max(c0["bos_idx"], cts_idx)
                c0["has_unfilled"] = has_unfilled_imbalance(
                    df, start_idx, end_idx, cts_idx, self.config.fill_threshold,
                    direction=sd,
                )

        # If Scenario 1 is already TRUE, update the Fib
        if scenario1 is True:
            key = (sid, 0)
            if key in self._fibs:
                return self._update_fib_cts(key, cts_idx, cts_price, df)
            # No Fib but Scenario 1 is TRUE - check if we can activate now
            c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
            if c0.get("has_unfilled", False):
                print(f"[fib] sid={sid} cycle=0 ACTIVATED on update (Scenario 1 TRUE, unfilled imbalance found)")
                return self._activate_fib(
                    sid=sid,
                    cycle_id=0,
                    sd=sd,
                    bos_idx=c0["bos_idx"],
                    bos_price=c0["bos_price"],
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={"activated_at": cts_idx, "scenario1": True, "activated_on": "update"},
                )
            return None

        # Scenario 1 undetermined - check again
        if reversal_confirmed_idx is not None and cts_idx >= reversal_confirmed_idx:
            # Scenario 1 becomes TRUE
            self._scenario1[sid] = True
            print(f"[fib] sid={sid} cycle=0 Scenario 1 TRUE at idx={cts_idx} (CTS >= rv_idx={reversal_confirmed_idx}) on update")

            c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
            if c0.get("has_unfilled", False):
                return self._activate_fib(
                    sid=sid,
                    cycle_id=0,
                    sd=sd,
                    bos_idx=c0["bos_idx"],
                    bos_price=c0["bos_price"],
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={"activated_at": cts_idx, "scenario1": True, "activated_on": "update"},
                )
            else:
                print(f"[fib] sid={sid} cycle=0 NOT activated: Scenario 1 TRUE but no unfilled imbalance")

        return None

    def _update_fib_cts(
        self,
        key: tuple,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """
        Internal helper to update a Fib's CTS anchor and check imbalance conditions.

        Used for both normal Fibs and cross-cycle Fibs.
        """
        state = self._fibs[key]
        if state.locked:
            return state

        # Only update if CTS moved to new extreme
        if cts_idx <= state.cts_idx:
            return state

        sid = state.structure_id
        cycle_id = state.cycle_id
        sd = state.struct_direction

        if sd == 1:
            anchor_high = cts_price
            anchor_low = state.bos_price
            anchor_high_idx = cts_idx
            anchor_low_idx = state.bos_idx
        else:
            anchor_high = state.bos_price
            anchor_low = cts_price
            anchor_high_idx = state.bos_idx
            anchor_low_idx = cts_idx

        new_fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            anchor_high_idx=anchor_high_idx,
            anchor_low_idx=anchor_low_idx,
            meta={"structure_id": sid, "cycle_id": cycle_id},
        )

        new_history = state.cts_history + ((cts_idx, cts_price),)
        new_state = replace(
            state,
            cts_idx=cts_idx,
            cts_price=cts_price,
            fib=new_fib,
            cts_history=new_history,
        )

        is_cross_cycle = state.meta.get("cross_cycle", False)
        label = "cross-cycle" if is_cross_cycle else f"cycle={cycle_id}"
        print(f"[fib] sid={sid} {label} UPDATED: CTS idx={cts_idx} price={cts_price:.5f}")

        # Check unfilled imbalance condition - can reactivate or deactivate
        if is_cross_cycle:
            # Cross-cycle Fib: check BOTH cycles' imbalance conditions.
            # sd-direction filter throughout (counter-direction imbalances
            # never produce POIs, so they don't influence cross-cycle eligibility).
            # Condition 1: Cycle 0 has unfilled imbalance (in cycle 0's locked range)
            c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
            c0_bos_idx = c0.get("bos_idx", new_state.bos_idx)
            c0_cts_idx = c0.get("cts_idx", new_state.bos_idx)  # Locked CTS_0
            c0_start = min(c0_bos_idx, c0_cts_idx)
            c0_end = max(c0_bos_idx, c0_cts_idx)
            cond1 = has_unfilled_imbalance(
                df, c0_start, c0_end, c0_cts_idx, self.config.fill_threshold,
                direction=sd,
            )

            # Condition 2: Cycle 1 has unfilled imbalance (BOS_1 to current CTS_1)
            cycle1_bos_idx = new_state.meta.get("cycle1_bos_idx", cts_idx)
            c1_start = min(cycle1_bos_idx, cts_idx)
            c1_end = max(cycle1_bos_idx, cts_idx)
            cond2 = has_unfilled_imbalance(
                df, c1_start, c1_end, cts_idx, self.config.fill_threshold,
                direction=sd,
            )

            # Condition 3: Cycle 1's BOS doesn't fill cycle 0's imbalances (static check)
            cond3 = has_unfilled_imbalance(
                df, c0_start, c0_end, cycle1_bos_idx, self.config.fill_threshold,
                direction=sd,
            )

            has_unfilled = cond1 and cond2 and cond3
            print(f"[fib] sid={sid} cross-cycle check: cond1={cond1} cond2={cond2} cond3={cond3}")
        else:
            # Normal Fib: check its own range. sd-direction filter.
            start_idx = min(new_state.bos_idx, new_state.cts_idx)
            end_idx = max(new_state.bos_idx, new_state.cts_idx)
            has_unfilled = has_unfilled_imbalance(
                df, start_idx, end_idx, cts_idx, self.config.fill_threshold,
                direction=sd,
            )

        if has_unfilled and not new_state.active:
            # Reactivate - unfilled imbalances now exist in expanded range
            new_state = replace(new_state, active=True, meta={**new_state.meta, "reactivated_at": cts_idx})
            print(f"[fib] sid={sid} {label} REACTIVATED: unfilled imbalance found at idx={cts_idx}")
        elif not has_unfilled and new_state.active:
            # Deactivate - all imbalances filled
            new_state = replace(new_state, active=False, meta={**new_state.meta, "deactivated_at": cts_idx, "reason": "all_imbalances_filled"})
            print(f"[fib] sid={sid} {label} DEACTIVATED: all imbalances filled at idx={cts_idx}")

        self._fibs[key] = new_state
        return new_state

    def _update_cycle1_fibs(
        self,
        sid: int,
        cts_idx: int,
        cts_price: float,
        df: pd.DataFrame,
    ) -> Optional[FibState]:
        """
        Handle cycle 1 CTS update when cross-cycle Fib exists.

        Updates both cross-cycle and normal_cycle1, determines which is active,
        and stores the appropriate one in _fibs[(sid, 1)].
        """
        cross_data = self._cross_cycle_data[sid]
        cross_fib = cross_data.get("cross_cycle")
        normal_fib = cross_data.get("normal_cycle1")

        if cross_fib is None:
            return None

        if cross_fib.locked:
            return cross_fib

        # Only update if CTS moved to new extreme
        if cts_idx <= cross_fib.cts_idx:
            return self._fibs.get((sid, 1))

        sd = cross_fib.struct_direction

        # --- Update cross-cycle Fib ---
        new_cross_fib = self._create_updated_fib_state(cross_fib, cts_idx, cts_price, sd, sid)

        # --- Update normal_cycle1 Fib ---
        new_normal_fib = None
        if normal_fib and not normal_fib.locked:
            new_normal_fib = self._create_updated_fib_state(normal_fib, cts_idx, cts_price, sd, sid)

        print(f"[fib] sid={sid} cross-cycle UPDATED: CTS idx={cts_idx} price={cts_price:.5f}")

        # --- Check cross-cycle conditions --- sd-direction filter throughout.
        c0 = cross_data.get("cycle0", {})
        c0_bos_idx = c0.get("bos_idx", new_cross_fib.bos_idx)
        c0_cts_idx = c0.get("cts_idx", new_cross_fib.bos_idx)
        c0_start = min(c0_bos_idx, c0_cts_idx)
        c0_end = max(c0_bos_idx, c0_cts_idx)
        cond1 = has_unfilled_imbalance(
            df, c0_start, c0_end, c0_cts_idx, self.config.fill_threshold,
            direction=sd,
        )

        cycle1_bos_idx = new_cross_fib.meta.get("cycle1_bos_idx", cts_idx)
        c1_start = min(cycle1_bos_idx, cts_idx)
        c1_end = max(cycle1_bos_idx, cts_idx)
        cond2 = has_unfilled_imbalance(
            df, c1_start, c1_end, cts_idx, self.config.fill_threshold,
            direction=sd,
        )

        cond3 = has_unfilled_imbalance(
            df, c0_start, c0_end, cycle1_bos_idx, self.config.fill_threshold,
            direction=sd,
        )

        cross_active = cond1 and cond2 and cond3
        print(f"[fib] sid={sid} cross-cycle check: cond1={cond1} cond2={cond2} cond3={cond3}")

        # --- Determine active state ---
        if cross_active and not new_cross_fib.active:
            new_cross_fib = replace(new_cross_fib, active=True, meta={**new_cross_fib.meta, "reactivated_at": cts_idx})
            print(f"[fib] sid={sid} cross-cycle REACTIVATED")
        elif not cross_active and new_cross_fib.active:
            new_cross_fib = replace(new_cross_fib, active=False, meta={**new_cross_fib.meta, "deactivated_at": cts_idx})
            print(f"[fib] sid={sid} cross-cycle DEACTIVATED")

        cross_data["cross_cycle"] = new_cross_fib

        # --- Check normal_cycle1 conditions --- sd-direction filter.
        if new_normal_fib:
            normal_start = min(new_normal_fib.bos_idx, cts_idx)
            normal_end = max(new_normal_fib.bos_idx, cts_idx)
            normal_has_unfilled = has_unfilled_imbalance(
                df, normal_start, normal_end, cts_idx, self.config.fill_threshold,
                direction=sd,
            )

            if normal_has_unfilled and not new_normal_fib.active:
                new_normal_fib = replace(new_normal_fib, active=True, meta={**new_normal_fib.meta, "reactivated_at": cts_idx})
                print(f"[fib] sid={sid} normal cycle=1 REACTIVATED")
            elif not normal_has_unfilled and new_normal_fib.active:
                new_normal_fib = replace(new_normal_fib, active=False, meta={**new_normal_fib.meta, "deactivated_at": cts_idx})
                print(f"[fib] sid={sid} normal cycle=1 DEACTIVATED")

            cross_data["normal_cycle1"] = new_normal_fib

        # --- Decide which Fib to use for _fibs[(sid, 1)] ---
        key = (sid, 1)
        if new_cross_fib.active:
            self._fibs[key] = new_cross_fib
            return new_cross_fib
        elif new_normal_fib and new_normal_fib.active:
            self._fibs[key] = new_normal_fib
            print(f"[fib] sid={sid} FALLBACK to normal cycle=1")
            return new_normal_fib
        else:
            # Both deactivated - keep cross-cycle in _fibs but inactive
            self._fibs[key] = new_cross_fib
            return new_cross_fib

    def _create_updated_fib_state(
        self,
        state: FibState,
        cts_idx: int,
        cts_price: float,
        sd: int,
        sid: int,
    ) -> FibState:
        """Helper to create updated FibState with new CTS."""
        new_fib = self._create_fib_retracement(
            sd, state.bos_idx, state.bos_price, cts_idx, cts_price, sid, state.cycle_id
        )
        new_history = state.cts_history + ((cts_idx, cts_price),)
        return replace(
            state,
            cts_idx=cts_idx,
            cts_price=cts_price,
            fib=new_fib,
            cts_history=new_history,
        )

    def on_cts_confirmed(self, event: StructureEvent) -> Optional[FibState]:
        """
        Handle CTS_CONFIRMED event - lock the Fib.

        For sid 1+ cycle 0: Also resolves Scenario 1 as FALSE if still undetermined.

        Parameters
        ----------
        event : StructureEvent
            The CTS_CONFIRMED event

        Returns
        -------
        FibState or None
            Locked FibState, or None if no active Fib
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))

        # Populate CTS lookup (used by cross-fib walk-backward for dead-cycle
        # evaluation). CTS_anchor_idx is the CTS extreme; ev.idx is the
        # confirmation candle.
        cts_anchor_idx = int(event.meta.get("cts_anchor_idx", event.idx))
        cts_price_val = float(event.price) if event.price else 0.0
        self._cts_by_cycle[(sid, cycle_id)] = (cts_anchor_idx, cts_price_val)
        # Record confirmation candle (used to bound prospective-BOS search for n+1)
        self._cts_confirmed_idx[(sid, cycle_id)] = int(event.idx)

        # ============================================================
        # BRANCH: fib_mode → cross_cycle / sid=0 / sid≥1
        # ============================================================
        if self.fib_mode == "cross_cycle":
            return self._handle_cross_cycle_cts_confirmed(sid, cycle_id, event)

        if sid == 0:
            return self._handle_sid0_cts_confirmed(sid, cycle_id, event)
        else:
            return self._handle_sid1plus_cts_confirmed(sid, cycle_id, event)

    def _handle_cross_cycle_cts_confirmed(
        self,
        sid: int,
        cycle_id: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """Handle CTS_CONFIRMED for cross_cycle: lock active fib (cross or
        single), flip phase to 'confirmed', set up pre_established for n+1."""
        # Lock active cross fib if present
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active and not latest[1].locked:
            key, state = latest
            locked = replace(
                state, locked=True,
                meta={**state.meta, "locked_at": event.idx},
            )
            self._fibs[key] = locked
            v = state.meta.get("version", 0)
            print(f"[fib] cross_cycle sid={sid} cycle={cycle_id} CROSS v{v} LOCKED")

        # Lock active single fib if present
        single_key = (sid, cycle_id)
        if single_key in self._fibs:
            state = self._fibs[single_key]
            if state.active and not state.locked:
                locked = replace(
                    state, locked=True,
                    meta={**state.meta, "locked_at": event.idx},
                )
                self._fibs[single_key] = locked
                print(f"[fib] cross_cycle sid={sid} cycle={cycle_id} SINGLE LOCKED")

        # Phase transitions
        self._m15_phase[(sid, cycle_id)] = "confirmed"
        self._m15_phase[(sid, cycle_id + 1)] = "pre_established"

        # Return whichever is currently active (preferring cross)
        if latest is not None and latest[1].locked:
            return self._fibs.get(latest[0])
        return self._fibs.get(single_key)

    def _handle_sid0_cts_confirmed(
        self,
        sid: int,
        cycle_id: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """Handle CTS_CONFIRMED for sid=0 (simple flow)."""
        if cycle_id == 0:
            # sid=0, cycle=0: No Fib to lock
            print(f"[fib] sid=0 cycle=0 CONFIRMED (no Fib, simple flow)")
            return None

        # sid=0, cycle 1+: Lock normal Fib
        key = (sid, cycle_id)
        if key not in self._fibs:
            return None

        state = self._fibs[key]
        if state.locked:
            return state

        locked_state = replace(state, locked=True, meta={**state.meta, "locked_at": event.idx})
        self._fibs[key] = locked_state
        print(f"[fib] sid=0 cycle={cycle_id} LOCKED (simple flow): CTS idx={state.cts_idx}")
        return locked_state

    def _handle_sid1plus_cts_confirmed(
        self,
        sid: int,
        cycle_id: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """Handle CTS_CONFIRMED for sid=1+ (post-reversal)."""
        # --- Cycle 0: Resolve Scenario 1 and lock ---
        if cycle_id == 0:
            return self._handle_cycle0_cts_confirmed(sid, event)

        key = (sid, cycle_id)

        # For cycle 1 CTS_CONFIRMED, lock both cross-cycle and normal_cycle1 if they exist
        if cycle_id == 1 and sid in self._cross_cycle_data:
            cross_data = self._cross_cycle_data[sid]

            # Lock cross-cycle Fib
            if "cross_cycle" in cross_data:
                cross_fib = cross_data["cross_cycle"]
                if not cross_fib.locked:
                    locked_cross = replace(cross_fib, locked=True, meta={**cross_fib.meta, "locked_at": event.idx})
                    cross_data["cross_cycle"] = locked_cross
                    print(f"[fib] sid={sid} cross-cycle LOCKED: CTS idx={cross_fib.cts_idx}")

            # Lock normal_cycle1 Fib
            if "normal_cycle1" in cross_data:
                normal_fib = cross_data["normal_cycle1"]
                if not normal_fib.locked:
                    locked_normal = replace(normal_fib, locked=True, meta={**normal_fib.meta, "locked_at": event.idx})
                    cross_data["normal_cycle1"] = locked_normal
                    print(f"[fib] sid={sid} normal cycle=1 LOCKED: CTS idx={normal_fib.cts_idx}")

        if key not in self._fibs:
            return None

        state = self._fibs[key]
        if state.locked:
            return state

        locked_state = replace(state, locked=True, meta={**state.meta, "locked_at": event.idx})
        self._fibs[key] = locked_state

        is_cross_cycle = state.meta.get("cross_cycle", False)
        label = "cross-cycle" if is_cross_cycle else f"cycle={cycle_id}"
        print(f"[fib] sid={sid} {label} LOCKED: CTS idx={state.cts_idx} price={state.cts_price:.5f}")

        return locked_state

    def _handle_cycle0_cts_confirmed(
        self,
        sid: int,
        event: StructureEvent,
    ) -> Optional[FibState]:
        """
        Handle cycle 0 CTS_CONFIRMED for sid 1+.

        Resolves Scenario 1 as FALSE if still undetermined.
        Locks cycle 0 data and any cycle 0 Fib.
        """
        # Resolve Scenario 1 if still undetermined
        scenario1 = self._scenario1.get(sid)
        if scenario1 is None:
            # Scenario 1 never became TRUE → resolve as FALSE (permanent)
            self._scenario1[sid] = False
            c0_cts = self._cross_cycle_data.get(sid, {}).get("cycle0", {}).get("cts_idx", "?")
            print(f"[fib] sid={sid} cycle=0 Scenario 1 FALSE (resolved at CONFIRMED): CTS idx={c0_cts} never reached rv_idx")

        # Lock cycle 0 data in _cross_cycle_data
        if sid in self._cross_cycle_data and "cycle0" in self._cross_cycle_data[sid]:
            c0 = self._cross_cycle_data[sid]["cycle0"]
            c0["locked"] = True
            print(f"[fib] sid={sid} cycle=0 LOCKED in cross-cycle data: CTS idx={c0['cts_idx']}")

        # Lock cycle 0 Fib if it exists (Scenario 1 was TRUE)
        key = (sid, 0)
        if key in self._fibs:
            state = self._fibs[key]
            if not state.locked:
                locked_state = replace(state, locked=True, meta={**state.meta, "locked_at": event.idx})
                self._fibs[key] = locked_state
                print(f"[fib] sid={sid} cycle=0 Fib LOCKED: CTS idx={state.cts_idx}")
                return locked_state
            return state

        return None

    def _should_revert_scenario1(
        self,
        bos1_price: float,
        prev_bos_outer: float,
        prev_sd: int,
    ) -> bool:
        """
        Check if BOS_1 level price touches/crosses into the prev structure's last BOS zone.

        "Touch" means price reaches the outer edge OR goes beyond it into the zone.

        For bullish prev structure (sd=1): BOS zone is buy, outer is bottom, zone is ABOVE outer
          → BOS_1 touches if bos1_price >= outer (price at or above outer edge)
        For bearish prev structure (sd=-1): BOS zone is sell, outer is top, zone is BELOW outer
          → BOS_1 touches if bos1_price <= outer (price at or below outer edge)
        """
        if prev_sd == 1:  # Prev was bullish, buy zone sits above outer (bottom)
            return bos1_price >= prev_bos_outer
        else:  # Prev was bearish, sell zone sits below outer (top)
            return bos1_price <= prev_bos_outer

    def _deactivate_cycle0_fib(self, sid: int) -> None:
        """Deactivate cycle 0 Fib when Scenario 1 reverts to FALSE."""
        key = (sid, 0)
        if key in self._fibs:
            state = self._fibs[key]
            if state.active:
                self._fibs[key] = replace(
                    state,
                    active=False,
                    meta={**state.meta, "deactivated_by": "scenario1_revert"}
                )
                print(f"[fib] sid={sid} cycle=0 DEACTIVATED (Scenario 1 reverted)")

    # ------------------------------------------------------------------
    # M15 reverse mode helpers
    # ------------------------------------------------------------------

    def _find_prospective_bos(
        self,
        df: pd.DataFrame,
        cts_n_confirmed_idx: int,
        current_candle: int,
        sd: int,
    ) -> Optional[tuple]:
        """Find the running-extreme pullback candle after CTS_n CONFIRMED.

        For sd=+1: argmin low in [cts_n_confirmed_idx + 1, current_candle].
        For sd=-1: argmax high in same range.

        This is the "prospective BOS_{n+1}" — the candle that will become
        BOS_{n+1} if and when a new breakout establishes CTS_{n+1}. Used as
        the start anchor for cycle n+1's pre-established imbalance range.

        Returns (idx, price) or None if window is empty.
        """
        start = cts_n_confirmed_idx + 1
        end = current_candle + 1
        if start >= end or start not in df.index:
            return None
        window = df.iloc[start:end]
        if sd == 1:
            rel = int(window["l"].values.argmin())
            return (start + rel, float(window["l"].values[rel]))
        else:
            rel = int(window["h"].values.argmax())
            return (start + rel, float(window["h"].values[rel]))

    def _running_extreme_anchor(
        self,
        df: pd.DataFrame,
        cts_n_confirmed_idx: int,
        current_candle: int,
        sd: int,
    ) -> Optional[tuple]:
        """Find the running extreme past CTS_n (used as the CTS anchor during
        pre-established phase).

        For sd=+1: argmax high in [cts_n_confirmed_idx + 1, current_candle].
        For sd=-1: argmin low in same range.

        Returns (idx, price) or None if window is empty.
        """
        start = cts_n_confirmed_idx + 1
        end = current_candle + 1
        if start >= end or start not in df.index:
            return None
        window = df.iloc[start:end]
        if sd == 1:
            rel = int(window["h"].values.argmax())
            return (start + rel, float(window["h"].values[rel]))
        else:
            rel = int(window["l"].values.argmin())
            return (start + rel, float(window["l"].values[rel]))

    def _get_latest_cross(self, sid: int, cycle_id: int) -> Optional[tuple]:
        """Return (key, FibState) for the highest-version cross fib at
        (sid, cycle_id), or None."""
        version = self._cross_version.get((sid, cycle_id))
        if version is None:
            return None
        key = (sid, cycle_id, "cross", version)
        fib = self._fibs.get(key)
        if fib is None:
            return None
        return (key, fib)

    def _obsolete_prev_cycle_all_fibs(self, sid: int, new_cycle_id: int) -> None:
        """Mark every fib for cycle < new_cycle_id (single + all cross versions)
        as inactive via obsolete_reason='new_cycle'. Skips already-inactive
        fibs."""
        for key in list(self._fibs.keys()):
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            if key[0] != sid:
                continue
            k_cycle = key[1]
            if k_cycle >= new_cycle_id:
                continue
            state = self._fibs[key]
            if not state.active:
                continue
            self._fibs[key] = replace(
                state,
                active=False,
                meta={**state.meta, "obsolete_reason": "new_cycle"},
            )

    # ------------------------------------------------------------------
    # M15 reverse: CTS_THRESHOLD_UPDATED dispatch (pre-established phase)
    # ------------------------------------------------------------------

    def on_cts_threshold_updated(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
    ) -> None:
        """Dispatch CTS_THRESHOLD_UPDATED for cross_cycle mode.

        Triggers a cross-fib check for the target cycle (= event.cycle_id + 1)
        ONLY when that target is in pre-established phase. In all other cases
        (h1 mode, or target not pre-established), this is a no-op.

        The anchor is the running extreme past CTS_n; own_imb_start is the
        prospective BOS_{n+1} (the deepest pullback since CTS_n CONFIRMED).
        """
        if self.fib_mode != "cross_cycle":
            return

        sid = int(event.meta.get("structure_id", 0))
        source_cycle = int(event.meta.get("cycle_id", 0))
        target_cycle = source_cycle + 1

        phase = self._m15_phase.get((sid, target_cycle))
        if phase != "pre_established":
            # Not in pre-established for the expected target: skip.
            return

        sd = int(event.meta.get("struct_direction", 0))
        if sd == 0:
            return
        current_candle = int(event.idx)

        # Lookup CTS_n confirmation candle to bound the prospective-BOS search
        cts_n_conf = self._cts_confirmed_idx.get((sid, source_cycle))
        if cts_n_conf is None:
            return

        anchor = self._running_extreme_anchor(df, cts_n_conf, current_candle, sd)
        if anchor is None:
            return
        anchor_idx, anchor_price = anchor

        prospective_bos = self._find_prospective_bos(df, cts_n_conf, current_candle, sd)
        if prospective_bos is None:
            return
        own_imb_start = prospective_bos[0]

        self._m15_cross_check(
            sid=sid,
            target_cycle=target_cycle,
            df=df,
            sd=sd,
            current_candle=current_candle,
            anchor_idx=anchor_idx,
            anchor_price=anchor_price,
            own_imb_start=own_imb_start,
        )

    # ------------------------------------------------------------------
    # M15 reverse cross-fib check + state transitions
    # ------------------------------------------------------------------

    def _m15_cross_check(
        self,
        sid: int,
        target_cycle: int,
        df: pd.DataFrame,
        sd: int,
        current_candle: int,
        anchor_idx: int,
        anchor_price: float,
        own_imb_start: int,
    ) -> None:
        """Central cross-fib check routine for cross_cycle mode.

        1. Check target cycle's own imbalance (range [own_imb_start, current_candle]).
           If none: deactivate any active cross; no fallback in pre-established.
        2. Walk backward from target-1 down to 0 using _dead_cycles cache.
           For each live cycle k: check [BOS_k, CTS_k] imbalance with fill to
           current_candle (Interpretation B). If all filled, mark dead and stop.
        3. Determine earliest_x (smallest cycle in contiguous run ending at target).
        4. Apply state transition:
           - earliest_x == target: cross fails; deactivate active cross;
             activate single (established phase only).
           - earliest_x < target, no active cross: create new cross (v0).
           - earliest_x < target, active cross same x: extend anchor in place.
           - earliest_x < target, active cross earlier x: shrink (deactivate old,
             create new with version+1).
        """
        phase = self._m15_phase.get((sid, target_cycle))

        # Step 1: target cycle's own imbalance. sd-direction filter — only
        # imbalances that could ever produce sd POIs count.
        own_has = has_unfilled_imbalance(
            df,
            min(own_imb_start, current_candle),
            max(own_imb_start, current_candle),
            current_candle,
            self.config.fill_threshold,
            direction=sd,
        )

        if not own_has:
            self._deactivate_active_cross(sid, target_cycle, current_candle, "own_imb_filled")
            # Pre-established: no fallback. Established: existing single fib
            # (if any) is managed by _update_fib_cts elsewhere — nothing to do here.
            return

        # Step 2: walk backward, collect earliest eligible x
        if sid not in self._dead_cycles:
            self._dead_cycles[sid] = set()

        earliest_x = target_cycle
        k = target_cycle - 1
        while k >= 0:
            if k in self._dead_cycles[sid]:
                break
            bos_k = self._bos_by_cycle.get((sid, k))
            cts_k = self._cts_by_cycle.get((sid, k))
            if bos_k is None or cts_k is None:
                # No data for this cycle — walk stops
                break
            range_start = min(bos_k[0], cts_k[0])
            range_end = max(bos_k[0], cts_k[0])
            has_unf = has_unfilled_imbalance(
                df,
                range_start,
                range_end,
                current_candle,
                self.config.fill_threshold,
                direction=sd,
            )
            if not has_unf:
                self._dead_cycles[sid].add(k)
                break
            earliest_x = k
            k -= 1

        # Step 3: act on result
        active_cross = self._get_latest_cross(sid, target_cycle)

        if earliest_x == target_cycle:
            # No prior cycle eligible → cross fails
            if active_cross is not None and active_cross[1].active:
                self._deactivate_cross(active_cross[0], current_candle, "cross_failed")
            if phase == "pre_established":
                return
            # Established / confirmed: fall back to single fib
            self._activate_or_update_single_m15(
                sid, target_cycle, sd, anchor_idx, anchor_price, df, current_candle
            )
            return

        # Cross eligible (earliest_x < target_cycle)
        bos_x = self._bos_by_cycle.get((sid, earliest_x))
        if bos_x is None:
            return  # safety guard (shouldn't happen)

        if active_cross is None or not active_cross[1].active:
            self._m15_create_cross(
                sid, target_cycle, sd, bos_x, anchor_idx, anchor_price,
                earliest_x, current_candle
            )
            return

        # Compare earliest_x to existing cross's start cycle
        active_x = int(active_cross[1].meta.get("cross_start_cycle", earliest_x))
        if earliest_x == active_x:
            # Same start — update anchor in place (extension)
            self._m15_extend_cross_anchor(
                active_cross[0], active_cross[1], anchor_idx, anchor_price
            )
        elif earliest_x > active_x:
            # Shrink — deactivate old version, create new
            self._deactivate_cross(active_cross[0], current_candle, "cross_shortened")
            self._m15_create_cross(
                sid, target_cycle, sd, bos_x, anchor_idx, anchor_price,
                earliest_x, current_candle
            )
        else:
            # earliest_x < active_x — shouldn't happen (dead cycles monotonic)
            print(f"[fib][cross_cycle][warn] unexpected earliest_x={earliest_x} "
                  f"< active_x={active_x} for sid={sid} cycle={target_cycle}")

    def _m15_create_cross(
        self,
        sid: int,
        target_cycle: int,
        sd: int,
        bos_x: tuple,
        anchor_idx: int,
        anchor_price: float,
        earliest_x: int,
        current_candle: int,
    ) -> None:
        """Create a new cross fib at (sid, target_cycle, 'cross', version)."""
        bos_x_idx, bos_x_price = bos_x
        version = self._cross_version.get((sid, target_cycle), -1) + 1
        key = (sid, target_cycle, "cross", version)

        if sd == 1:
            anchor_high = anchor_price
            anchor_low = bos_x_price
            anchor_high_idx = anchor_idx
            anchor_low_idx = bos_x_idx
        else:
            anchor_high = bos_x_price
            anchor_low = anchor_price
            anchor_high_idx = bos_x_idx
            anchor_low_idx = anchor_idx

        fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            anchor_high_idx=anchor_high_idx,
            anchor_low_idx=anchor_low_idx,
            meta={"structure_id": sid, "cycle_id": target_cycle},
        )

        state = FibState(
            structure_id=sid,
            cycle_id=target_cycle,
            struct_direction=sd,
            bos_idx=bos_x_idx,
            bos_price=bos_x_price,
            cts_idx=anchor_idx,
            cts_price=anchor_price,
            active=True,
            locked=False,
            fib=fib,
            meta={
                "fib_mode": "cross_cycle",
                "cross_cycle": True,
                "cross_start_cycle": earliest_x,
                "version": version,
                "activated_at": current_candle,
            },
            cts_history=((anchor_idx, anchor_price),),
        )
        self._fibs[key] = state
        self._cross_version[(sid, target_cycle)] = version
        self._current_cycle[sid] = target_cycle

        # Obsolete all prior-cycle fibs (single + cross versions)
        self._obsolete_prev_cycle_all_fibs(sid, target_cycle)

        print(f"[fib] cross_cycle sid={sid} CROSS ({earliest_x}->{target_cycle}) "
              f"v{version} ACTIVATED: bos_x_idx={bos_x_idx} -> anchor_idx={anchor_idx}")

    def _m15_extend_cross_anchor(
        self,
        key: tuple,
        state: FibState,
        new_anchor_idx: int,
        new_anchor_price: float,
    ) -> None:
        """Extend a cross fib's CTS anchor in place (same version)."""
        if new_anchor_idx <= state.cts_idx:
            return  # anchor not advancing

        sd = state.struct_direction
        if sd == 1:
            anchor_high = new_anchor_price
            anchor_low = state.bos_price
            anchor_high_idx = new_anchor_idx
            anchor_low_idx = state.bos_idx
        else:
            anchor_high = state.bos_price
            anchor_low = new_anchor_price
            anchor_high_idx = state.bos_idx
            anchor_low_idx = new_anchor_idx

        new_fib = create_fib_retracement(
            anchor_high=anchor_high,
            anchor_low=anchor_low,
            direction=sd,
            levels=self.config.fib_levels,
            anchor_high_idx=anchor_high_idx,
            anchor_low_idx=anchor_low_idx,
            meta={"structure_id": state.structure_id, "cycle_id": state.cycle_id},
        )
        new_history = state.cts_history + ((new_anchor_idx, new_anchor_price),)
        self._fibs[key] = replace(
            state,
            cts_idx=new_anchor_idx,
            cts_price=new_anchor_price,
            fib=new_fib,
            cts_history=new_history,
        )
        v = state.meta.get("version", 0)
        print(f"[fib] cross_cycle sid={state.structure_id} CROSS v{v} EXTENDED: "
              f"cts idx={new_anchor_idx}")

    def _deactivate_cross(self, key: tuple, current_candle: int, reason: str) -> None:
        """Mark a cross fib inactive."""
        state = self._fibs.get(key)
        if state is None or not state.active:
            return
        self._fibs[key] = replace(
            state,
            active=False,
            meta={
                **state.meta,
                "deactivated_by": reason,
                "deactivated_at": current_candle,
            },
        )
        v = state.meta.get("version", 0)
        print(f"[fib] cross_cycle sid={state.structure_id} cycle={state.cycle_id} "
              f"CROSS v{v} DEACTIVATED: {reason}")

    def _deactivate_active_cross(
        self,
        sid: int,
        cycle_id: int,
        current_candle: int,
        reason: str,
    ) -> None:
        latest = self._get_latest_cross(sid, cycle_id)
        if latest is not None and latest[1].active:
            self._deactivate_cross(latest[0], current_candle, reason)

    def _activate_or_update_single_m15(
        self,
        sid: int,
        target_cycle: int,
        sd: int,
        anchor_idx: int,
        anchor_price: float,
        df: pd.DataFrame,
        current_candle: int,
    ) -> None:
        """Activate or update the single fib for target_cycle when cross fails
        (established phase fallback). BOS is the confirmed BOS_{target_cycle}."""
        bos = self._bos_by_cycle.get((sid, target_cycle))
        if bos is None:
            return  # shouldn't happen in established phase
        key = (sid, target_cycle)
        existing = self._fibs.get(key)
        if existing is None or not existing.active:
            # Activate new single fib
            self._activate_fib(
                sid=sid,
                cycle_id=target_cycle,
                sd=sd,
                bos_idx=bos[0],
                bos_price=bos[1],
                cts_idx=anchor_idx,
                cts_price=anchor_price,
                meta={
                    "fib_mode": "cross_cycle",
                    "via": "cross_failed",
                    "activated_at": current_candle,
                },
            )
        else:
            # Extend existing single fib anchor + re-check imbalance
            self._update_fib_cts(key, anchor_idx, anchor_price, df)

    def get_active_fib(self, structure_id: int) -> Optional[FibState]:
        """Get the current active Fib for a structure.

        Prefers the highest-version active cross fib over the single-fib entry
        for the current cycle.
        """
        cycle_id = self._current_cycle.get(structure_id)
        if cycle_id is None:
            return None
        # Prefer active cross fib if present
        latest = self._get_latest_cross(structure_id, cycle_id)
        if latest is not None and latest[1].active:
            return latest[1]
        state = self._fibs.get((structure_id, cycle_id))
        if state and state.active:
            return state
        return None

    def get_all_fibs(self) -> List[FibState]:
        """Get all Fib states (including historical)."""
        return list(self._fibs.values())

    def get_fibs_for_charting(self) -> List[FibState]:
        """
        Get Fibs for charting - current state per (structure_id, cycle_id).

        Excludes fibs that were invalidated (e.g., deactivated due to Scenario 1 revert).
        These fibs should not be shown on the chart at all.
        """
        return [
            fib for fib in self._fibs.values()
            if fib.meta.get("deactivated_by") != "scenario1_revert"
        ]
