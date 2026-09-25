# engine_v2/zones/cross_cycle_fib.py
"""Shared cross-cycle Fib eligibility decision (CROSS_CYCLE_FIB_SPEC.md §3, §9).

ONE pure routine both fib modes call to decide whether a cross-cycle Fib is
eligible at a given target cycle, and from which earliest cycle it spans. The
storage / version-transition application stays in the caller (subordinate
`cross_cycle` mode keeps its versioned ``_fibs`` machinery; H1-main keeps its
named-slot ``_cross_cycle_data`` until §11a-ii migrates it).

This module is the §11a "extract" — it is the body of the subordinate
``FibTracker._m15_cross_check`` steps 1–3 (own-imbalance test + dead-cycle
backward walk + ``earliest_x``), generalized so the H1-main single-step
Scenario-2 decision (cond1/cond2/cond3 at ``target=1``) routes through it too,
via ``select_fib_anchor_for_cycle`` (the thin wrapper in ``fib_tracker.py``).

The only mode difference is the *fill-as-of* point for prior-cycle liveness
(CROSS_CYCLE_FIB_SPEC.md §3.1): subs check "current"; main checks the frozen
snapshot (cond2 @CTS_0 cached + cond3 @BOS_1). Parameterizing it is what keeps
§11a byte-identical for both modes.

Kept dependency-free of ``zones`` (imports only ``patterns.imbalance``) so
``fib_tracker`` can import it without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from engine_v2.patterns.imbalance import has_unfilled_imbalance


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
    fill_as_of: str = "current",
    prior_cached_liveness: Optional[Dict[int, bool]] = None,
    target_ceiling: Optional[int] = None,
    *,
    evaluated_at: Optional[int],
    fill_horizon_idx: int,
    snapshot_horizon_idx: Optional[int],
) -> CrossEligibility:
    """Decide cross-cycle eligibility at ``target_cycle`` (pure).

    Reproduces ``FibTracker._m15_cross_check`` steps 1–3 exactly:

    1. **Own-imbalance test** — sd-direction unfilled imbalance over
       ``[own_imb_start, own_window_end_idx]`` as of ``fill_horizon_idx``. None ⇒
       ``own_has=False`` (caller deactivates any active cross; no cross).
    2. **Dead-cycle backward walk** ``k = target-1 … 0``, stopping at the first
       cycle already in ``dead_cycles`` / missing data / dead. A cycle is *live*
       per the ``fill_as_of`` policy (below); a dead cycle is added to
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
        Mutable, sid-scoped. Pass the real cache (subs) or a throwaway set
        (the single-step main wrapper, which never caches).
    fill_as_of : {"current", "snapshot"}
        Per-cycle liveness fill point (CROSS_CYCLE_FIB_SPEC.md §3.1):

        - ``"current"`` (subs) — each prior cycle's ``[BOS_k, CTS_k]`` checked
          as of ``fill_horizon_idx``.
        - ``"snapshot"`` (H1-main single step) — checked as of ``snapshot_horizon_idx``
          (= cond3 @BOS_target) AND-ed with ``prior_cached_liveness[k]``
          (= cond2, the cached cycle-0 @CTS_0 liveness). Defined only for the
          ``target=1`` single step in §11a; deeper-walk snapshot semantics are
          an §11b open item.
    prior_cached_liveness : dict, optional
        ``{cycle_id: bool}`` used only by the snapshot policy. A missing cycle
        defaults to ``True`` (no extra constraint).
    target_ceiling : int, optional
        Reserved for §11b (the main-only ``M`` cap). **Ignored in §11a.**
    own_window_end_idx, own_imb_start : int
        The target cycle's own-test window ``[own_imb_start, own_window_end_idx]``
        (LOCATIONS: the cycle's BOS and its CTS anchor / running extreme).
    fill_horizon_idx : int
        Keyword-only, REQUIRED: the fill horizon of the own test and of the
        ``"current"`` walk (a TIME; Plan E E2b split it from the window end; the handled
        event's moment since Plan E E3a).
    snapshot_horizon_idx : int or None
        Keyword-only, REQUIRED: the ``"snapshot"`` walk's fill horizon (cond3,
        "has BOS_target filled the prior cycle?"); None only with
        ``fill_as_of="current"``.
    evaluated_at : int or None
        Keyword-only, REQUIRED: the MOMENT the decision is taken (Plan F;
        IMBALANCE_FILL_SEMANTICS.md "Knowability — the c3 rule"). Every
        imbalance question below counts only instances formed by then. The
        step-1 own test is the one it can change (its window ends AT
        ``own_window_end_idx``); the prior-cycle walks end at ``CTS_k`` < the moment
        (already bounded) and get it for uniformity. ``prior_cached_liveness``
        (cond2) is a cached value judged at its use, so it is not re-cut here.
        ``None`` = no cut: the unchanged MS in-flight resolver. FibTracker passes
        the same value as ``fill_horizon_idx`` on every path since Plan E E3a;
        they stay two parameters because the MS in-flight resolver passes a
        horizon (its refresh moment) with no cut.

    Returns
    -------
    CrossEligibility
    """
    direction = sd if sd in (1, -1) else None
    if fill_as_of == "snapshot" and snapshot_horizon_idx is None:
        raise ValueError("fill_as_of='snapshot' needs snapshot_horizon_idx")

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
        if fill_as_of == "snapshot":
            live = has_unfilled_imbalance(
                df, range_start, range_end, snapshot_horizon_idx, fill_threshold,
                direction=direction, evaluated_at=evaluated_at,
            )
            if prior_cached_liveness is not None:
                live = live and bool(prior_cached_liveness.get(k, True))
        else:  # "current"
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
