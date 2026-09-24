"""Isolation validation for MS pre-CTS_0 scan-from-start mode (edit #3).

The scan-from-start pivot (2026-06-07) replaced seed-and-resume: MS, when
`enforce_cts0_new_extreme=True`, re-runs the SHARED
`find_true_first_breakout` routine from `start_idx` against the handed
`bos0_inner` and establishes cycle 0 at the located winner via its normal
cycle-0 path. The old "highest-risk seed byte-match" item is therefore
replaced by THIS tripwire: MS's cycle-0 establishment must agree, by
construction, with a direct call to the routine.
"""
from __future__ import annotations

import pytest

from engine_v2.structure import event_fields as ef
from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.structure.structure_engine import _make_market_structure
from engine_v2.structure.true_first_breakout import find_true_first_breakout
from engine_v2.tests.test_unified_probe import (
    _make_downtrend_data,
    _make_uptrend_data,
    _prepare_df,
)


def _first_cts_established(events, structure_id=0, cycle_id=0):
    for ev in events:
        if (
            ev.type == "CTS_ESTABLISHED"
            and ev.meta.get("structure_id") == structure_id
            and ev.meta.get("cycle_id") == cycle_id
        ):
            return ev
    return None


@pytest.mark.parametrize(
    "make_data, direction, bos0_inner",
    [
        (_make_uptrend_data, 1, 0.5550),   # well below the 0.60+ uptrend
        (_make_downtrend_data, -1, 0.8050),  # well above the downtrend body
    ],
)
def test_ms_scan_mode_agrees_with_shared_routine(make_data, direction, bos0_inner):
    df = _prepare_df(make_data(n=80))
    bp = BreakoutPatterns(df)

    # Direct routine call = the "probe" view.
    tfb = find_true_first_breakout(
        bp, current_start=0, upper_idx=len(df) - 1,
        direction=direction, bos0_inner=bos0_inner,
    )
    assert tfb is not None, "test data must contain a true first breakout"

    # MS in scan-from-start mode = the canonical structure run.
    ms = _make_market_structure(
        df,
        struct_direction=direction,
        start_idx=0,
        structure_id=0,
        end_idx=None,
        timeframe="H1",
        enforce_cts0_new_extreme=True,
        bos0_inner=bos0_inner,
    )
    ms.debug = True
    _df2, events, _levels = ms.run()

    cts0 = _first_cts_established(events)
    assert cts0 is not None, "MS scan mode must establish cycle-0 CTS"

    # Agreement BY CONSTRUCTION: the CTS_ESTABLISHED event is anchored at
    # the full-pattern extreme (idx + price), and its confirmed_at is the
    # apply/confirm idx — exactly the routine's extreme_idx / extreme_price
    # / est_idx.
    assert ef.cts_anchor_idx(cts0) == int(tfb.extreme_idx)
    assert float(cts0.price) == pytest.approx(float(tfb.extreme_price))
    assert int(cts0.meta["confirmed_at"]) == int(tfb.est_idx)


def test_scan_mode_requires_bos0_inner():
    df = _prepare_df(_make_uptrend_data(n=20))
    with pytest.raises(ValueError):
        _make_market_structure(
            df,
            struct_direction=1,
            start_idx=0,
            structure_id=0,
            timeframe="H1",
            enforce_cts0_new_extreme=True,
            # bos0_inner omitted on purpose
        )


def test_no_breakout_in_window_establishes_no_cycle0():
    # bos0_inner ABOVE every close → check_break (close past inner) never
    # passes → routine returns None → MS establishes no cycle 0.
    df = _prepare_df(_make_uptrend_data(n=40))
    high_inner = float(df["c"].astype(float).max()) + 1.0
    ms = _make_market_structure(
        df,
        struct_direction=1,
        start_idx=0,
        structure_id=0,
        end_idx=None,
        timeframe="H1",
        enforce_cts0_new_extreme=True,
        bos0_inner=high_inner,
    )
    ms.debug = True
    _df2, events, _levels = ms.run()
    assert _first_cts_established(events) is None
