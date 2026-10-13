"""PLAN_O — the IC reference's CTS = cycle T−1's for every fib record (the user's change 2, 2026-10-13).

The rule's flips live where they were pinned (`test_plan_k_landing_a.py`: the builder, the pre-established read, the
choice, T > M, the hover; `test_plan_m_landing_1.py`: a cross 0 -> 1 of an unfixed target). Pinned here:

- the establishment pattern anchor and moment are read together — both (T established) or neither (pre-established);
  either half alone fails the assert (`AssertionError`, not a later `TypeError`);
- a pre-established cross at the data edge, through the REAL stamp (`stamp_ic_references`, `lifecycle_cap=None`):
  its meta carries CTS_{T−1} with no `ic_ref_break_*` keys — the price side only (the only reader that can see a
  pre-established target, PLAN_O §2).
"""
from __future__ import annotations

import pytest

from engine_v2.tests._cross_fib_limits import NO_P
from engine_v2.tests.test_plan_i_s3_cross_fib_limit_applied import _df, _m
from engine_v2.tests.test_plan_k_landing_a import _tracker_with
from engine_v2.tests.test_plan_m_landing_1 import _live, _pre_est


@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("drop", ["_est_pattern_anchor", "_bos_moment_by_cycle"])
def test_the_establishment_anchor_and_moment_are_read_together(sd, drop):
    t, df = _tracker_with(NO_P, sd)
    del getattr(t, drop)[(0, 2)]
    with pytest.raises(AssertionError, match="establishment pattern anchor"):
        t.ic_reference(t._fibs[(0, 2)], df, evaluated_at=None)


@pytest.mark.parametrize("sd", [1, -1])
def test_a_pre_established_cross_at_the_data_edge_is_stamped_with_the_price_side_only(sd):
    """Cycle 0 (CTS 1.2) confirmed at 25 → cycle 1 pre-established; `P` 9.0 never reached (M None): the cross 0 -> 1
    is created at the first threshold update and never established. CTS_0 (1.2) is further opposite than the anchor
    behind `P` → `prev_cycle_cts`, no bound."""
    t, seen = _pre_est(_live(9.0, [], sd), sd, updates=(30, 31))
    assert seen == [True, True]
    assert (0, 1) not in t._est_pattern_anchor
    t.stamp_ic_references(_df(40, []), None)
    _key, fib = t._get_latest_cross(0, 1)
    assert (fib.cycle_id, fib.meta["cross_cycle"], fib.meta["cross_start_cycle"]) == (1, True, 0)
    assert (fib.meta["ic_ref_price"], fib.meta["ic_ref_source"]) == (_m(1.2, sd), "prev_cycle_cts")
    assert "ic_ref_break_pattern_anchor_idx" not in fib.meta and "ic_ref_break_confirmed_idx" not in fib.meta
