# engine_v2/zones/cross_cycle_fib.py
"""Shared cross-cycle Fib eligibility decision (CROSS_CYCLE_FIB_SPEC.md §4, §7) + the cross-fib limit's
`CrossFibLimit` / `beyond` (§3).

ONE pure routine deciding whether a cross-cycle Fib is eligible at a given target cycle, and from which earliest
cycle it spans: the body of ``FibTracker._cross_check`` steps 1–3 (own-imbalance test + dead-cycle backward walk +
``earliest_x``); the storage / version transitions stay in FibTracker — its only caller (every structure since PLAN_I
S4; MarketStructure reads the same decision through its embedded tracker since S5). The ``"snapshot"`` fill-as-of
policy of the retired H1-main Scenario-2 selector was deleted with it in PLAN_I S5 (CROSS_CYCLE_FIB_SPEC.md Appendix
A, old §3.1 / §11a).

Kept dependency-free of ``zones`` (imports only ``patterns.imbalance``) so
``fib_tracker`` can import it without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional

import pandas as pd

from engine_v2.patterns.imbalance import has_unfilled_imbalance


# --- The cross-fib LIMIT (PLAN_I §1.2 / §4 N-a / §9.3 D1) --------------------------------------------------

# `cross_fib_limit_source` values (§4 N-a + R6-3 (vi)): the rule that produced `P`.
CROSS_FIB_LIMIT_SOURCES = frozenset({
    "none",                     # main start trading; a sub's first_confluence on parent cycle 0; no previous sub
    "reversal_bos_frozen",      # main or sub reversal: the applied reversal's frozen BOS barrier
    "parent_cts_threshold",     # first_confluence on parent cycle c >= 1: the parent's (S, c) `cts_frozen`
    "prev_sub_cts_threshold",   # x−1 same direction: its most recent confirmed cycle's `cts_threshold` (R2-1)
    "prev_sub_bos_threshold",   # x−1 opposite: its `bos_threshold` (R2-1)
    "prev_sub_starting_price",  # x−1 opposite, no `bos_threshold` yet: its starting candle's extreme (R2-2)
    "prev_sub_latest_cts",      # x−1 same, no confirmed CTS yet: its latest CTS anchor price (R2-b / R3-5)
    "prev_sub_no_cts",          # x−1 same, no CTS anchor at all: no `P` (R6-3 (vi))
})
_NO_PRICE_SOURCES = frozenset({"none", "prev_sub_no_cts"})


@dataclass(frozen=True)
class CrossFibLimit:
    """A structure's cross-fib LIMIT: the reference price `P` its STARTING trigger fixes once, at the
    structure's start (R2-4), and the rule that produced it.

    - `price` (`cross_fib_limit_price`): `P`; None = no reference → no cross on any cycle (every cycle single);
    - `source` (`cross_fib_limit_source`, one of `CROSS_FIB_LIMIT_SOURCES`);
    - `from_pool_idx` (sweep-time) / `from_sub_id` (the final, start-numbered id, set once the sweep has
      numbered the subs — PLAN_I D2 / D4): the sub that supplied `P` (x−1, or the reversing sub), else None;
    - `started_by`: main only (`start_trading` | `reversal`, §4 N-c); a sub's comes from its mirror;
    - `ref_anchor_price` (PLAN_K A8, RX1-2): the ANCHOR price behind `P` — the IC reference's price side for the
      structure's cycles ≤ `M` (`P` is a threshold that may have moved; the IC reference never uses a threshold):
      the reversing structure's most recent `BOS_CONFIRMED` price, the parent cycle c−1's `CTS_CONFIRMED` price,
      x−1's `BOS_CONFIRMED` / confirmed `CTS_CONFIRMED` price, or `P` itself where `P` already is an anchor price
      (`prev_sub_latest_cts`, `prev_sub_starting_price`); None ⟺ `price` None;
    - `break_pattern_anchor_idx` / `break_confirmed_idx` (PLAN_K A8, RX1-1): a `reversal_bos_frozen` structure's
      REALISED reversal pattern — its first candle and its apply (the `STATE_CHANGED(to=reversal)` moment), in the
      coordinates of the frame the limit is used on (main: the H1 frame; a sub: entity-absolute at the pool,
      `localized` to the sub's slice at each entry point — PLAN_K §3.5); set ⟺ `reversal_bos_frozen`. Every other
      source's break comes from the cross-limit breakout watch (`patterns.cross_limit_watch`).

    `M` (`cross_fib_limit_cycle`) is not here — it follows the structure's cycles (`FibTracker`).
    """
    price: Optional[float]
    source: str
    from_pool_idx: Optional[int] = None
    from_sub_id: Optional[int] = None
    started_by: Optional[str] = None
    ref_anchor_price: Optional[float] = None
    break_pattern_anchor_idx: Optional[int] = None
    break_confirmed_idx: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.source in CROSS_FIB_LIMIT_SOURCES, f"unknown cross_fib_limit_source {self.source!r}"
        assert (self.price is None) == (self.source in _NO_PRICE_SOURCES), (
            f"cross_fib_limit price {self.price!r} does not fit source {self.source!r}")
        assert (self.ref_anchor_price is None) == (self.price is None), (
            f"cross_fib_limit ref_anchor_price {self.ref_anchor_price!r} does not fit price {self.price!r} "
            f"(source {self.source!r}; PLAN_K A8)")
        has_break = self.break_pattern_anchor_idx is not None
        assert has_break == (self.break_confirmed_idx is not None), (
            f"cross_fib_limit break fields set apart: {self.break_pattern_anchor_idx!r} / {self.break_confirmed_idx!r}")
        assert has_break == (self.source == "reversal_bos_frozen"), (
            f"cross_fib_limit break {self.break_pattern_anchor_idx!r} does not fit source {self.source!r} (PLAN_K A8)")
        if has_break:
            assert int(self.break_pattern_anchor_idx) <= int(self.break_confirmed_idx), (
                f"cross_fib_limit break anchor {self.break_pattern_anchor_idx} after its apply {self.break_confirmed_idx}")

    def localized(self, offset: int) -> "CrossFibLimit":
        """This limit with its break indices moved by −`offset` (PLAN_K §3.5): the pool computes a sub's limit in
        entity-absolute M15 indices, while its geometry MS and its projection run on the sub's slice starting at
        `offset` (`slice_begin`). Prices / identity unchanged; a limit without a break is returned as is."""
        if self.break_pattern_anchor_idx is None:
            return self
        return replace(self, break_pattern_anchor_idx=int(self.break_pattern_anchor_idx) - int(offset),
                       break_confirmed_idx=int(self.break_confirmed_idx) - int(offset))


# The IC reference's `source` values (PLAN_K §1, the fib meta `ic_ref_source`; PLAN_H RX1).
IC_REF_SOURCES = frozenset({
    "prev_cycle_cts",       # cycle a−1's CTS_CONFIRMED price (the fib's BOS in cycle a ≥ 1)
    "prev_structure_bos",   # a reversal start: the reversing structure's most recent BOS_CONFIRMED price
    "cross_limit_anchor",   # a non-reversal start with a cross limit: the anchor behind P
})


@dataclass(frozen=True)
class ICReference:
    """A fib record's IC reference (PLAN_K §3.1; PLAN_H §8.9 RX1): the price an IC candidate must lie entirely on the
    near side of (sd +1: below), and the first candle of the pattern that BROKE that price — the IC must come before
    it. Built ONLY by `FibTracker.ic_reference` (both consumers hand it to `poi_zones.select_ic`); `NONE` = no
    reference (the condition passes, RX1-B). A reference without a known break checks the price side only
    (RX1-2a)."""
    price: Optional[float]
    source: Optional[str]
    break_pattern_anchor_idx: Optional[int] = None
    break_confirmed_idx: Optional[int] = None

    def __post_init__(self) -> None:
        assert (self.price is None) == (self.source is None), (
            f"ICReference price {self.price!r} / source {self.source!r} set apart")
        assert self.source is None or self.source in IC_REF_SOURCES, f"unknown IC reference source {self.source!r}"
        has_break = self.break_pattern_anchor_idx is not None
        assert has_break == (self.break_confirmed_idx is not None), (
            f"ICReference break fields set apart: {self.break_pattern_anchor_idx!r} / {self.break_confirmed_idx!r}")
        assert not has_break or self.price is not None, "ICReference: a break without a reference price"


ICReference.NONE = ICReference(None, None)


def beyond(price: float, ref: float, sd: int) -> bool:
    """`price` STRICTLY past `ref` in the structure's direction (Q2 / §4 N-b): sd = +1 `price > ref`, sd = −1
    `price < ref` — reaching the price is not beyond."""
    assert int(sd) in (1, -1), f"beyond: sd must be +1 / -1, got {sd!r}"
    return float(price) > float(ref) if int(sd) == 1 else float(price) < float(ref)


@dataclass(frozen=True)
class CrossEligibility:
    """Pure cross-cycle eligibility decision for a single ``target_cycle``.

    Storage-agnostic: callers map this onto their own version / named-slot
    model. ``earliest_x == target_cycle`` means no eligible prior cycle (no
    cross). The resolved anchors describe the cross ``BOS_earliest_x → CTS-side
    anchor`` (CTS_n, or the running extreme past CTS_n when pre-established);
    they mirror the CTS-side inputs when ``crosses`` is False
    (so the fields are always populated, never garbage).
    """

    own_has: bool        # target cycle's own imbalance is unfilled (step-1 gate)
    earliest_x: int      # smallest live cycle in the contiguous run ending at
                         # target; == target_cycle ⇒ no eligible prior → no cross
    crosses: bool        # own_has AND earliest_x < target_cycle
    bos_idx: int
    bos_price: float
    cts_idx: int
    cts_price: float


def resolve_cross_cycle_eligibility(
    df: pd.DataFrame,
    target_cycle: int,
    sd: int,
    own_window_end_idx: int,
    own_imb_start: int,
    anchor_idx: int,
    anchor_price: float,
    bos_by_cycle: Dict[int, tuple],
    cts_by_cycle: Dict[int, tuple],
    dead_cycles: set,
    fill_threshold: float = 0.70,
    *,
    evaluated_at: Optional[int],
    fill_horizon_idx: int,
) -> CrossEligibility:
    """Decide cross-cycle eligibility at ``target_cycle`` (pure).

    Reproduces ``FibTracker._cross_check`` steps 1–3 exactly:

    1. **Own-imbalance test** — sd-direction unfilled imbalance over
       ``[own_imb_start, own_window_end_idx]`` as of ``fill_horizon_idx``. None ⇒
       ``own_has=False`` (caller deactivates any active cross; no cross).
    2. **Dead-cycle backward walk** ``k = target-1 … 0``, stopping at the first
       cycle already in ``dead_cycles`` / missing data / dead. A cycle is *live*
       when ``[BOS_k, CTS_k]`` holds an unfilled sd imbalance as of
       ``fill_horizon_idx`` (the current candle); a dead cycle is added to
       ``dead_cycles`` (mutated in place — the caller passes its sid-scoped
       cache so the walk's memo persists, exactly as today).
    3. ``earliest_x`` = smallest live cycle in the contiguous run ending at
       target; ``crosses`` iff ``own_has and earliest_x < target_cycle``.

    Parameters
    ----------
    bos_by_cycle, cts_by_cycle : dict
        ``{cycle_id: (idx, price)}`` for this structure (caller slices to sid).
        Only cycles ``0 … target_cycle-1`` are consulted by the walk.
    dead_cycles : set
        Mutable, sid-scoped: FibTracker's cache (``_dead_cycles[sid]``).
    own_window_end_idx, own_imb_start : int
        The target cycle's own-test window ``[own_imb_start, own_window_end_idx]``
        (LOCATIONS: the cycle's BOS and its CTS anchor / running extreme).
    fill_horizon_idx : int
        Keyword-only, REQUIRED: the fill horizon of the own test and of the
        walk (a TIME; Plan E E2b split it from the window end; the handled
        event's moment since Plan E E3a).
    evaluated_at : int or None
        Keyword-only, REQUIRED: the MOMENT the decision is taken (Plan F;
        IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule"). Every
        imbalance question below counts only instances formed by then. The
        step-1 own test is the one it can change (its window ends AT
        ``own_window_end_idx``); the prior-cycle walks end at ``CTS_k`` < the moment
        (already bounded) and get it for uniformity. ``None`` = no cut (no
        production caller since PLAN_I S5 retired MS's resolver). FibTracker passes
        the same value as ``fill_horizon_idx`` on every path since Plan E E3a.

    Returns
    -------
    CrossEligibility
    """
    direction = sd if sd in (1, -1) else None

    # Step 1: target cycle's own imbalance.
    own_has = has_unfilled_imbalance(
        df,
        min(own_imb_start, own_window_end_idx),
        max(own_imb_start, own_window_end_idx),
        fill_horizon_idx,
        fill_threshold,
        direction=direction,
        evaluated_at=evaluated_at,
    )
    if not own_has:
        return CrossEligibility(
            own_has=False,
            earliest_x=target_cycle,
            crosses=False,
            bos_idx=int(anchor_idx),
            bos_price=float(anchor_price),
            cts_idx=int(anchor_idx),
            cts_price=float(anchor_price),
        )

    # Step 2: walk backward, collect earliest live cycle.
    earliest_x = target_cycle
    k = target_cycle - 1
    while k >= 0:
        if k in dead_cycles:
            break
        bos_k = bos_by_cycle.get(k)
        cts_k = cts_by_cycle.get(k)
        if bos_k is None or cts_k is None:
            break
        range_start = min(bos_k[0], cts_k[0])
        range_end = max(bos_k[0], cts_k[0])
        live = has_unfilled_imbalance(
            df, range_start, range_end, fill_horizon_idx, fill_threshold,
            direction=direction, evaluated_at=evaluated_at,
        )
        if not live:
            dead_cycles.add(k)
            break
        earliest_x = k
        k -= 1

    # Step 3: assemble decision.
    if earliest_x < target_cycle:
        bos_x = bos_by_cycle[earliest_x]
        return CrossEligibility(
            own_has=True,
            earliest_x=earliest_x,
            crosses=True,
            bos_idx=int(bos_x[0]),
            bos_price=float(bos_x[1]),
            cts_idx=int(anchor_idx),
            cts_price=float(anchor_price),
        )
    return CrossEligibility(
        own_has=True,
        earliest_x=target_cycle,
        crosses=False,
        bos_idx=int(anchor_idx),
        bos_price=float(anchor_price),
        cts_idx=int(anchor_idx),
        cts_price=float(anchor_price),
    )
