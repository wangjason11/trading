# engine_v2/zones/fib_tracker.py
"""
Fibonacci level tracking for POI Zones.

Fib levels are drawn from BOS to CTS within each market structure cycle.
This module handles activation, updates, and lifecycle management.
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

    Only 1 active Fib per structure at a time (new cycle obsoletes previous).
    """

    def __init__(self, config: Optional[FibTrackerConfig] = None):
        self.config = config or FibTrackerConfig()

        # Current Fib state per (structure_id, cycle_id) - always up-to-date
        # {(structure_id, cycle_id): FibState}
        self._fibs: Dict[tuple, FibState] = {}

        # Track which cycle is "current" per structure (for obsolescence)
        # {structure_id: cycle_id}
        self._current_cycle: Dict[int, int] = {}

        # Track cross-cycle Fib eligibility (post-reversal)
        # {structure_id: {"cycle0": {...}, "normal_cycle1": FibState, "cross_cycle": FibState}}
        self._cross_cycle_data: Dict[int, Dict] = {}

    def on_cts_established(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        bos_idx: int,
        bos_price: float,
        reversal_confirmed_idx: Optional[int] = None,
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
            Only needed for cycle 1 cross-cycle check.

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

        # --- Cycle 0: Store data for potential cross-cycle Fib, but don't activate ---
        if cycle_id == 0:
            if sid not in self._cross_cycle_data:
                self._cross_cycle_data[sid] = {}

            self._cross_cycle_data[sid]["cycle0"] = {
                "bos_idx": bos_idx,
                "bos_price": bos_price,
                "cts_idx": cts_idx,
                "cts_price": cts_price,
                "struct_direction": sd,
                "has_unfilled": has_unfilled,
                "locked": False,
            }
            print(f"[fib] sid={sid} cycle=0 STORED for cross-cycle check: BOS idx={bos_idx} -> CTS idx={cts_idx}, has_unfilled={has_unfilled}")
            return None

        # --- Cycle 1: Check cross-cycle exception first ---
        if cycle_id == 1 and sid in self._cross_cycle_data and "cycle0" in self._cross_cycle_data[sid]:
            c0 = self._cross_cycle_data[sid]["cycle0"]

            # Check all 4 conditions for cross-cycle Fib
            cond1 = c0.get("has_unfilled", False)  # Cycle 0 has unfilled imbalance
            cond2 = has_unfilled  # Cycle 1 has unfilled imbalance

            # Cond 3: Cycle 1's BOS doesn't fill cycle 0's imbalances
            c0_start = min(c0["bos_idx"], c0["cts_idx"])
            c0_end = max(c0["bos_idx"], c0["cts_idx"])
            # Check if any of cycle 0's imbalances remain unfilled after cycle 1's BOS
            cond3 = has_unfilled_imbalance(df, c0_start, c0_end, bos_idx, self.config.fill_threshold)

            # Cond 4: Cycle 0's locked CTS idx < reversal_confirmed_idx
            c0_cts_idx = c0["cts_idx"]
            cond4 = reversal_confirmed_idx is not None and c0_cts_idx < reversal_confirmed_idx

            print(f"[fib] sid={sid} cycle=1 cross-cycle check: cond1={cond1} cond2={cond2} cond3={cond3} cond4={cond4} (c0_cts={c0_cts_idx} < rv_idx={reversal_confirmed_idx})")

            if cond1 and cond2 and cond3 and cond4:
                # Cross-cycle conditions met - create both cross-cycle and normal Fibs
                # Normal cycle 1 Fib (for fallback when cross-cycle deactivates)
                if has_unfilled:
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
                        meta={"activated_at": cts_idx},
                        cts_history=((cts_idx, cts_price),),
                    )
                    self._cross_cycle_data[sid]["normal_cycle1"] = normal_fib
                    print(f"[fib] sid={sid} cycle=1 NORMAL computed (fallback): BOS idx={bos_idx} -> CTS idx={cts_idx}")

                # Create cross-cycle Fib: BOS_0 -> CTS_1 (stored as cycle 1)
                print(f"[fib] sid={sid} CROSS-CYCLE FIB: BOS_0 idx={c0['bos_idx']} -> CTS_1 idx={cts_idx}")
                cross_fib = self._activate_fib(
                    sid=sid,
                    cycle_id=1,  # Cross-cycle replaces cycle 1
                    sd=sd,
                    bos_idx=c0["bos_idx"],
                    bos_price=c0["bos_price"],
                    cts_idx=cts_idx,
                    cts_price=cts_price,
                    meta={
                        "cross_cycle": True,
                        "activated_at": cts_idx,
                        # Store cycle 1's BOS for deactivation/reactivation checks
                        "cycle1_bos_idx": bos_idx,
                    },
                )
                self._cross_cycle_data[sid]["cross_cycle"] = cross_fib
                return cross_fib

        # --- Normal activation for cycle 1+ (no cross-cycle) ---
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
    ) -> Optional[FibState]:
        """
        Handle CTS_UPDATED event - update CTS anchor if Fib is active.

        Parameters
        ----------
        event : StructureEvent
            The CTS_UPDATED event
        df : DataFrame
            OHLC data

        Returns
        -------
        FibState or None
            Updated FibState, or None if no active Fib for this cycle
        """
        sid = int(event.meta.get("structure_id", 0))
        cycle_id = int(event.meta.get("cycle_id", 0))
        cts_idx = int(event.idx)
        cts_price = float(event.price) if event.price else 0.0

        # Update cycle 0 data in _cross_cycle_data if applicable (before cross-cycle Fib is created)
        if cycle_id == 0 and sid in self._cross_cycle_data and "cycle0" in self._cross_cycle_data[sid]:
            c0 = self._cross_cycle_data[sid]["cycle0"]
            if not c0.get("locked", False) and cts_idx > c0["cts_idx"]:
                c0["cts_idx"] = cts_idx
                c0["cts_price"] = cts_price
                # Re-check unfilled imbalance
                start_idx = min(c0["bos_idx"], cts_idx)
                end_idx = max(c0["bos_idx"], cts_idx)
                c0["has_unfilled"] = has_unfilled_imbalance(df, start_idx, end_idx, cts_idx, self.config.fill_threshold)
            return None

        # For cycle 1 updates, check if there's a cross-cycle Fib (now stored as cycle 1)
        key = (sid, cycle_id)

        # Check if we have cross-cycle data for this structure
        if cycle_id == 1 and sid in self._cross_cycle_data and "cross_cycle" in self._cross_cycle_data[sid]:
            # We have a cross-cycle Fib - update it and also update normal_cycle1
            return self._update_cycle1_fibs(sid, cts_idx, cts_price, df)

        if key not in self._fibs:
            return None

        return self._update_fib_cts(key, cts_idx, cts_price, df)

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

        # Lock cycle 0 data in _cross_cycle_data if applicable
        if cycle_id == 0 and sid in self._cross_cycle_data and "cycle0" in self._cross_cycle_data[sid]:
            c0 = self._cross_cycle_data[sid]["cycle0"]
            c0["locked"] = True
            print(f"[fib] sid={sid} cycle=0 LOCKED in cross-cycle data: CTS idx={c0['cts_idx']}")
            return None

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

        Returns all FibStates from _fibs (always up-to-date).
        """
        return list(self._fibs.values())
