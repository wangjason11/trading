"""PLAN_I S3b (R6-1b, R8-3): the single's two latent quirks, both 0 on the reference window, pinned on constructed
streams (the S3 pin file's flat frame — its own mirror at the pivot 3.0, so sd = -1 is the exact mirror).

- F5: the `cross_failed` single re-checks its own window on every update. When cycle T's own window is filled at a
  CTS_UPDATED the cross check returns early; the existing single used to stay active on its older CTS with every
  imbalance in its window filled — now it extends and deactivates, as every single does.
- F15 / F18: a fib activated (single) or created (cross) AFTER its cycle's establishment ends the previous cycle at
  that establishment, not at the late activation — a cycle's fib cannot outlive its cycle. The pre-established cross
  (created BEFORE the establishment — Option A) still ends the previous cycle at its creation.
"""
from __future__ import annotations

import pytest

from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
from engine_v2.tests.test_plan_i_s3_cross_fib_limit_applied import (
    _conf, _df, _est, _ev, _gap, _kind, _lim, _m, _pre_established, _q, _tracker,
)


def _upd(idx, price, cyc, sd):
    return _ev("CTS_UPDATED", idx, price, cyc, sd, via=CTS_UPDATED_RAW_VIA)


def _dip(df, sd, frm, level):
    """Price retraces to `level` from candle `frm` on (a low for a buy structure, a high for a sell one)."""
    if sd == 1:
        df.loc[frm:, "l"] = level
    else:
        df.loc[frm:, "h"] = _m(level, -1)


@pytest.mark.parametrize("sd", [1, -1])
def test_f5_the_cross_failed_single_deactivates_when_its_own_window_fills(sd):
    """Within the limit (P unreached). Cycle 0's gap (1.44-1.48, ABOVE cycle 1's) is filled by a dip to 1.43 from 22,
    before cycle 1 exists → cycle 0 dead → at cycle 1's establishment (40) the cross fails → the `cross_failed` single
    (BOS 30 → CTS 40, own gap 35). A dip to 1.19 from 45 fills the own gap; the CTS_UPDATED to 48 finds the own window
    [30, 48] filled → the single extends to 48 and DEACTIVATES there (it used to stay active at CTS 40)."""
    t = _tracker(_lim(9.0, sd))
    df = _df(80, [_gap(15, sd, 1.48, 1.44), _gap(35, sd, 1.30, 1.20)])
    _dip(df, sd, 22, 1.43)
    _q(t.on_cts_established, _est(20, 1.15, 0, sd), df, bos_idx=10, bos_price=_m(1.0, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, sd, 20))
    _q(t.on_cts_established, _est(40, 1.35, 1, sd), df, bos_idx=30, bos_price=_m(1.2, sd))
    assert _kind(t, 1) == "single cross_failed"
    _dip(df, sd, 45, 1.19)
    _q(t.on_cts_updated, _upd(48, 1.37, 1, sd), df)
    s = t._fibs[(0, 1)]
    assert (s.active, s.cts_idx, s.cts_price) == (False, 48, _m(1.37, sd))
    assert s.meta["deactivated_at"] == 48 and s.meta["reason"] == "all_imbalances_filled"
    assert s.meta["via"] == "cross_failed"


@pytest.mark.parametrize("sd", [1, -1])
def test_f15_a_late_activated_single_ends_the_previous_cycle_at_its_establishment(sd):
    """Past the limit (no P): cycle 1 established at 40 with no gap in [30, 40] → no fib; a CTS_UPDATED to 48 finds the
    gap 44 → the single activates at 48 — cycle 0's fib ends at 40 (cycle 1's establishment), not 48."""
    t = _tracker(_lim(None, sd))
    df = _df(80, [_gap(15, sd, 1.10, 1.00), _gap(44, sd, 1.30, 1.20)])
    _q(t.on_cts_established, _est(20, 1.15, 0, sd), df, bos_idx=10, bos_price=_m(1.0, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, sd, 20))
    _q(t.on_cts_established, _est(40, 1.35, 1, sd), df, bos_idx=30, bos_price=_m(1.2, sd))
    assert (0, 0) not in t._terminal
    _q(t.on_cts_updated, _upd(48, 1.40, 1, sd), df)
    assert _kind(t, 1) == "single cross_fib_limit" and t._fibs[(0, 1)].meta["activated_at"] == 48
    assert t._terminal[(0, 0)] == (40, "new_cycle")
    assert t._first_active[(0, 1)] == 48                              # the late fib's own start stays its activation


@pytest.mark.parametrize("sd", [1, -1])
def test_f18_a_late_created_cross_ends_the_previous_cycle_at_its_establishment(sd):
    """Within the limit: the same shape, but cycle 0 is alive → the update at 48 creates the cross 0→1 there —
    cycle 0's fib ends at 40, not 48."""
    t = _tracker(_lim(9.0, sd))
    df = _df(80, [_gap(15, sd, 1.10, 1.00), _gap(44, sd, 1.30, 1.20)])
    _q(t.on_cts_established, _est(20, 1.15, 0, sd), df, bos_idx=10, bos_price=_m(1.0, sd))
    _q(t.on_cts_confirmed, _conf(25, 1.15, 0, sd, 20))
    _q(t.on_cts_established, _est(40, 1.35, 1, sd), df, bos_idx=30, bos_price=_m(1.2, sd))
    assert _kind(t, 1) is None
    _q(t.on_cts_updated, _upd(48, 1.40, 1, sd), df)
    assert _kind(t, 1) == "cross 0->1" and t._get_latest_cross(0, 1)[1].meta["activated_at"] == 48
    assert t._terminal[(0, 0)] == (40, "new_cycle")


@pytest.mark.parametrize("sd", [1, -1])
def test_the_pre_established_cross_still_ends_the_previous_cycle_at_its_creation(sd):
    """Option A unchanged: the cross for cycle 1 created by the threshold update at 30 (before cycle 1 exists — no
    establishment moment yet) ends cycle 0 at 30; cycle 1's establishment at 35 then keeps the cross (extended) and
    never moves that end."""
    t = _pre_established(9.0, sd)
    assert t._terminal[(0, 0)] == (30, "new_cycle") and (0, 1) not in t._bos_moment_by_cycle
    df = _df(40, [_gap(15, sd, 1.10, 1.00), _gap(29, sd, 1.40, 1.30)])
    _q(t.on_cts_established, _est(35, 1.55, 1, sd), df, bos_idx=27, bos_price=_m(1.45, sd))
    assert _kind(t, 1) == "cross 0->1" and t._get_latest_cross(0, 1)[1].meta["activated_at"] == 30
    assert t._terminal[(0, 0)] == (30, "new_cycle")
