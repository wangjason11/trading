"""Cross-fib limits for tests (PLAN_I S3 / S4: a FibTracker consulted for a cycle >= 1 of a structure with no limit
entry raises `CrossFibLimitError` — the limit is REQUIRED, D1).

`unreached(sd)` is a reference price `P` no CTS anchor can go beyond (+inf for a buy structure, -inf for a sell one):
`M` is never set, so every cross the pre-limit rules allow stays allowed — the limit's no-op. The tests that predate
the limit (census S3, `plans/plan_i_inputs/census/census_S3.txt`) keep their intent with it; the limit's own rules are
pinned in `test_plan_i_s3_cross_fib_limit_applied.py` with real prices. A helper that hard-wires `unreached(1)`
(test_cross_cycle_fib `_make_tracker`, test_e3a_mutation_pins `_t`, test_imbalance_c3_knowability `_tracker`) serves
BUY fixtures only: a sell fixture through it would get M = 0 and every cycle single — use `unreached(-1)` or
`unreached_for(events)` there.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable

from engine_v2.zones.cross_cycle_fib import CrossFibLimit


def unreached(sd: int) -> CrossFibLimit:
    """... and its anchor (`ref_anchor_price`, PLAN_K A8) is the same ±inf: the IC reference's structure side never
    binds (every candle lies strictly on the near side of ±inf; no candle closes past it, so no watch break)."""
    assert int(sd) in (1, -1), sd
    p = math.inf if int(sd) == 1 else -math.inf
    return CrossFibLimit(p, "parent_cts_threshold", ref_anchor_price=p)


def main_limits(events: Iterable) -> Dict[int, CrossFibLimit]:
    """The H1 main's limits, exactly as production derives them (`compute_main_cross_fib_limits`): sid 0 none
    (start_trading), sid N the realised reversal's `bos_frozen`. The replacement of the retired `fib_mode="h1"`
    for a test of a main-style run (PLAN_I S4)."""
    from engine_v2.zones.structure_lifecycle import compute_main_cross_fib_limits
    return compute_main_cross_fib_limits(list(events))


def unreached_for(events: Iterable) -> Dict[int, CrossFibLimit]:
    """`{sid: unreached(sd)}` for every structure in `events` (its direction read from the events' meta)."""
    out: Dict[int, CrossFibLimit] = {}
    for e in events:
        sd = e.meta.get("struct_direction")
        if sd in (1, -1):
            out.setdefault(int(e.meta.get("structure_id", 0)), unreached(sd))
    return out


# A break "far past every fixture candle": a reversal limit's IC-reference bound that never binds (PLAN_K A8).
FAR = 10 ** 6


def limit(price, source: str, **kw) -> CrossFibLimit:
    """A test limit whose IC-reference fields do not bind unless the test sets them (PLAN_K A8 / A20 a): the anchor
    behind `P` = `P` itself, and — a `reversal_bos_frozen` source — a break at `FAR`. For the fixtures that pin the
    cross-fib limit's own rules; the IC reference's rules are pinned with real values elsewhere."""
    if price is not None:
        kw.setdefault("ref_anchor_price", price)
    if source == "reversal_bos_frozen":
        kw.setdefault("break_pattern_anchor_idx", FAR)
        kw.setdefault("break_confirmed_idx", FAR)
    return CrossFibLimit(price, source, **kw)


class _NoRefTracker:
    """A stand-in tracker for a direct `poi_zones.poi_inner_for_fib` call on a hand-built FibState (PLAN_K A4 / A20
    d): no IC reference (`ICReference.NONE` — the condition passes)."""

    def ic_reference(self, fib_state, df, *, evaluated_at):
        from engine_v2.zones.cross_cycle_fib import ICReference
        return ICReference.NONE


NO_REF_TRACKER = _NoRefTracker()


# No reference price (PLAN_I S5 / D5): every cycle >= 1 a single — what the probe passes, and the default for a
# test's own `compute_bounded_structure` / `_make_market_structure` run (MarketStructure's embedded tracker needs a
# limit; the argument is REQUIRED).
NO_P = CrossFibLimit(None, "none")
