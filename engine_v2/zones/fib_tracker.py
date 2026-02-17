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
        self.fib_mode = fib_mode  # "h1" (default) or "m15_reverse"

        # Current Fib state per (structure_id, cycle_id) - always up-to-date
        # {(structure_id, cycle_id): FibState}
        self._fibs: Dict[tuple, FibState] = {}

        # Track which cycle is "current" per structure (for obsolescence)
        # {structure_id: cycle_id}
        self._current_cycle: Dict[int, int] = {}

        # Track cross-cycle Fib eligibility (post-reversal, sid 1+ only)
        # {structure_id: {"cycle0": {...}, "normal_cycle1": FibState, "cross_cycle": FibState}}
        self._cross_cycle_data: Dict[int, Dict] = {}

        # Track Scenario 1 resolution per structure (sid 1+ only)
        # {structure_id: True/False/None}
        # None = undetermined, True = CTS_0 >= rv_idx, False = resolved at CTS_0 CONFIRMED
        self._scenario1: Dict[int, Optional[bool]] = {}

        # M15 reverse mode: track active cross-cycle Fib per structure
        # When a cross-cycle Fib starts, subsequent cycles don't get their own Fibs
        self._m15_active_cross_cycle: Dict[int, bool] = {}

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

        # Check for unfilled imbalance between BOS and CTS
        start_idx = min(bos_idx, cts_idx)
        end_idx = max(bos_idx, cts_idx)
        has_unfilled = has_unfilled_imbalance(df, start_idx, end_idx, cts_idx, self.config.fill_threshold)

        # ============================================================
        # BRANCH: fib_mode determines logic
        # ============================================================
        if self.fib_mode == "m15_reverse":
            return self._handle_m15_reverse_cts_established(
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

    def _handle_m15_reverse_cts_established(
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
        Handle CTS_ESTABLISHED for M15 reverse structure (imbalance-gated cross-cycle).

        Rules:
        1. Walk cycles sequentially
        2. If active cross-cycle Fib exists from a prior cycle, skip this cycle
        3. No unfilled imbalance -> NO Fib for this cycle
        4. Has unfilled imbalance -> cross-cycle Fib from this cycle's BOS
        """
        # If a cross-cycle Fib is already active for this structure, skip
        if self._m15_active_cross_cycle.get(sid, False):
            print(f"[fib] m15_reverse sid={sid} cycle={cycle_id} SKIPPED: cross-cycle Fib already active")
            return None

        if not has_unfilled:
            print(f"[fib] m15_reverse sid={sid} cycle={cycle_id} NO FIB: no unfilled imbalance")
            return None

        # Has unfilled imbalance -> activate cross-cycle Fib
        print(f"[fib] m15_reverse sid={sid} cycle={cycle_id} CROSS-CYCLE ACTIVATED")
        self._m15_active_cross_cycle[sid] = True

        return self._activate_fib(
            sid=sid,
            cycle_id=cycle_id,
            sd=sd,
            bos_idx=bos_idx,
            bos_price=bos_price,
            cts_idx=cts_idx,
            cts_price=cts_price,
            meta={
                "activated_at": cts_idx,
                "cross_cycle": True,
                "fib_mode": "m15_reverse",
            },
        )

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

        # Check Scenario 2 conditions
        cond1 = has_unfilled  # Cycle 1 has unfilled imbalance
        cond2 = c0.get("has_unfilled", False)  # Cycle 0 has unfilled imbalance

        # Cond 3: BOS_1 doesn't fill cycle 0's imbalances
        c0_start = min(c0["bos_idx"], c0["cts_idx"])
        c0_end = max(c0["bos_idx"], c0["cts_idx"])
        cond3 = has_unfilled_imbalance(df, c0_start, c0_end, bos_idx, self.config.fill_threshold)

        print(f"[fib] sid={sid} cycle=1 Scenario 2 check: cond1={cond1} cond2={cond2} cond3={cond3}")

        if cond1 and cond2 and cond3:
            # Scenario 2: Cross-cycle Fib
            print(f"[fib] sid={sid} cycle=1 Scenario 2: CROSS-CYCLE ACTIVATED: BOS_0 idx={c0['bos_idx']} -> CTS_1 idx={cts_idx}")

            # Create normal cycle 1 Fib for fallback
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
            print(f"[fib] sid={sid} cycle=1 NORMAL computed (fallback): BOS idx={bos_idx} -> CTS idx={cts_idx}")

            # Create cross-cycle Fib: BOS_0 -> CTS_1
            cross_fib = self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=c0["bos_idx"],
                bos_price=c0["bos_price"],
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={
                    "cross_cycle": True,
                    "scenario": 2,
                    "activated_at": cts_idx,
                    "cycle1_bos_idx": bos_idx,
                },
            )
            self._cross_cycle_data[sid]["cross_cycle"] = cross_fib
            return cross_fib
        elif cond1:
            # Scenario 3: Normal cycle 1 Fib (cross-cycle conditions not met)
            print(f"[fib] sid={sid} cycle=1 Scenario 3: NORMAL ACTIVATED: BOS idx={bos_idx} -> CTS idx={cts_idx}")
            return self._activate_fib(
                sid=sid,
                cycle_id=1,
                sd=sd,
                bos_idx=bos_idx,
                bos_price=bos_price,
                cts_idx=cts_idx,
                cts_price=cts_price,
                meta={"activated_at": cts_idx, "scenario": 3},
            )
        else:
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
        # BRANCH: sid = 0 vs sid = 1+
        # ============================================================
        if sid == 0:
            return self._handle_sid0_cts_updated(sid, cycle_id, cts_idx, cts_price, df)
        else:
            return self._handle_sid1plus_cts_updated(
                sid, cycle_id, sd, cts_idx, cts_price, df, reversal_confirmed_idx
            )

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
                # Re-check unfilled imbalance
                start_idx = min(c0["bos_idx"], cts_idx)
                end_idx = max(c0["bos_idx"], cts_idx)
                c0["has_unfilled"] = has_unfilled_imbalance(df, start_idx, end_idx, cts_idx, self.config.fill_threshold)

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
            # Cross-cycle Fib: check BOTH cycles' imbalance conditions
            # Condition 1: Cycle 0 has unfilled imbalance (in cycle 0's locked range)
            c0 = self._cross_cycle_data.get(sid, {}).get("cycle0", {})
            c0_bos_idx = c0.get("bos_idx", new_state.bos_idx)
            c0_cts_idx = c0.get("cts_idx", new_state.bos_idx)  # Locked CTS_0
            c0_start = min(c0_bos_idx, c0_cts_idx)
            c0_end = max(c0_bos_idx, c0_cts_idx)
            cond1 = has_unfilled_imbalance(df, c0_start, c0_end, c0_cts_idx, self.config.fill_threshold)

            # Condition 2: Cycle 1 has unfilled imbalance (BOS_1 to current CTS_1)
            cycle1_bos_idx = new_state.meta.get("cycle1_bos_idx", cts_idx)
            c1_start = min(cycle1_bos_idx, cts_idx)
            c1_end = max(cycle1_bos_idx, cts_idx)
            cond2 = has_unfilled_imbalance(df, c1_start, c1_end, cts_idx, self.config.fill_threshold)

            # Condition 3: Cycle 1's BOS doesn't fill cycle 0's imbalances (static check)
            cond3 = has_unfilled_imbalance(df, c0_start, c0_end, cycle1_bos_idx, self.config.fill_threshold)

            has_unfilled = cond1 and cond2 and cond3
            print(f"[fib] sid={sid} cross-cycle check: cond1={cond1} cond2={cond2} cond3={cond3}")
        else:
            # Normal Fib: check its own range
            start_idx = min(new_state.bos_idx, new_state.cts_idx)
            end_idx = max(new_state.bos_idx, new_state.cts_idx)
            has_unfilled = has_unfilled_imbalance(df, start_idx, end_idx, cts_idx, self.config.fill_threshold)

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

        # --- Check cross-cycle conditions ---
        c0 = cross_data.get("cycle0", {})
        c0_bos_idx = c0.get("bos_idx", new_cross_fib.bos_idx)
        c0_cts_idx = c0.get("cts_idx", new_cross_fib.bos_idx)
        c0_start = min(c0_bos_idx, c0_cts_idx)
        c0_end = max(c0_bos_idx, c0_cts_idx)
        cond1 = has_unfilled_imbalance(df, c0_start, c0_end, c0_cts_idx, self.config.fill_threshold)

        cycle1_bos_idx = new_cross_fib.meta.get("cycle1_bos_idx", cts_idx)
        c1_start = min(cycle1_bos_idx, cts_idx)
        c1_end = max(cycle1_bos_idx, cts_idx)
        cond2 = has_unfilled_imbalance(df, c1_start, c1_end, cts_idx, self.config.fill_threshold)

        cond3 = has_unfilled_imbalance(df, c0_start, c0_end, cycle1_bos_idx, self.config.fill_threshold)

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

        # --- Check normal_cycle1 conditions ---
        if new_normal_fib:
            normal_start = min(new_normal_fib.bos_idx, cts_idx)
            normal_end = max(new_normal_fib.bos_idx, cts_idx)
            normal_has_unfilled = has_unfilled_imbalance(df, normal_start, normal_end, cts_idx, self.config.fill_threshold)

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

        # ============================================================
        # BRANCH: sid = 0 vs sid = 1+
        # ============================================================
        if sid == 0:
            return self._handle_sid0_cts_confirmed(sid, cycle_id, event)
        else:
            return self._handle_sid1plus_cts_confirmed(sid, cycle_id, event)

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

    def get_active_fib(self, structure_id: int) -> Optional[FibState]:
        """Get the current active Fib for a structure."""
        cycle_id = self._current_cycle.get(structure_id)
        if cycle_id is None:
            return None
        key = (structure_id, cycle_id)
        state = self._fibs.get(key)
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
