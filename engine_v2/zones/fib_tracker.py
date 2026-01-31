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
        # {structure_id: {"cycle0_fib": FibState, ...}}
        self._cross_cycle_data: Dict[int, Dict] = {}

    def on_cts_established(
        self,
        event: StructureEvent,
        df: pd.DataFrame,
        bos_idx: int,
        bos_price: float,
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

        # Skip cycle 0 - only activate Fibs from cycle 1 onwards
        if cycle_id == 0:
            return None

        # Get CTS price from event or df
        if cts_price == 0.0 and cts_idx in df.index:
            if sd == 1:
                cts_price = float(df.loc[cts_idx, "h"])
            else:
                cts_price = float(df.loc[cts_idx, "l"])

        # Check for unfilled imbalance between BOS and CTS
        start_idx = min(bos_idx, cts_idx)
        end_idx = max(bos_idx, cts_idx)

        if not has_unfilled_imbalance(df, start_idx, end_idx, cts_idx, self.config.fill_threshold):
            print(f"[fib] sid={sid} cycle={cycle_id} NOT activated: no unfilled imbalance")
            return None

        # Create Fib retracement
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

        # Create FibState
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
            meta={"activated_at": cts_idx},
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

        # Store the new Fib
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
        key = (sid, cycle_id)

        if key not in self._fibs:
            return None

        state = self._fibs[key]
        if state.locked:
            return state

        cts_idx = int(event.idx)
        cts_price = float(event.price) if event.price else 0.0

        # Only update if CTS moved to new extreme
        if cts_idx <= state.cts_idx:
            return state

        # CTS moved - update anchor 2
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

        print(f"[fib] sid={sid} cycle={cycle_id} UPDATED: CTS idx={cts_idx} price={cts_price:.5f}")

        # Check unfilled imbalance condition - can reactivate or deactivate
        start_idx = min(new_state.bos_idx, new_state.cts_idx)
        end_idx = max(new_state.bos_idx, new_state.cts_idx)

        has_unfilled = has_unfilled_imbalance(df, start_idx, end_idx, cts_idx, self.config.fill_threshold)

        if has_unfilled and not new_state.active:
            # Reactivate - unfilled imbalances now exist in expanded range
            new_state = replace(new_state, active=True, meta={**new_state.meta, "reactivated_at": cts_idx})
            print(f"[fib] sid={sid} cycle={cycle_id} REACTIVATED: unfilled imbalance found at idx={cts_idx}")
        elif not has_unfilled and new_state.active:
            # Deactivate - all imbalances filled
            new_state = replace(new_state, active=False, meta={**new_state.meta, "deactivated_at": cts_idx, "reason": "all_imbalances_filled"})
            print(f"[fib] sid={sid} cycle={cycle_id} DEACTIVATED: all imbalances filled at idx={cts_idx}")

        self._fibs[key] = new_state
        return new_state

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
        key = (sid, cycle_id)

        if key not in self._fibs:
            return None

        state = self._fibs[key]
        if state.locked:
            return state

        locked_state = replace(state, locked=True, meta={**state.meta, "locked_at": event.idx})
        self._fibs[key] = locked_state

        print(f"[fib] sid={sid} cycle={cycle_id} LOCKED: CTS idx={state.cts_idx} price={state.cts_price:.5f}")

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
