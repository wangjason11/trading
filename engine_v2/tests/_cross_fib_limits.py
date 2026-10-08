"""Cross-fib limits for tests of the cross machinery itself (PLAN_I S3: a `cross_cycle` tracker consulted for a
cycle >= 1 of a structure with no limit entry raises `CrossFibLimitError` — the limit is REQUIRED, D1).

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
    assert int(sd) in (1, -1), sd
    return CrossFibLimit(math.inf if int(sd) == 1 else -math.inf, "parent_cts_threshold")


def unreached_for(events: Iterable) -> Dict[int, CrossFibLimit]:
    """`{sid: unreached(sd)}` for every structure in `events` (its direction read from the events' meta)."""
    out: Dict[int, CrossFibLimit] = {}
    for e in events:
        sd = e.meta.get("struct_direction")
        if sd in (1, -1):
            out.setdefault(int(e.meta.get("structure_id", 0)), unreached(sd))
    return out
