"""BOS selection guards (2026-09-28; zones-audit latent items (c) and (e)).

(e) `_emit_bos_confirmed` asserts `bos_anchor_idx <= idx`: the anchor is a location already SEEN at the moment. Only
`_select_bos_on_breakout`'s swapped window (`window_start > breakout_apply_idx` then) could hand it a later one, and the
anchor-keyed processing order would then run the cycle's CTS_ESTABLISHED before its BOS_CONFIRMED (FibTracker needs
the BOS first). Reference window: 19 cycle >= 1 selections, all on the pullback window, 0 swapped, 0 anchors after
the apply (shadow over every MS run, 2026-09-28). Since PLAN_J J3 (F17 (ii)) the window ENDS at the new cycle's CTS
anchor (<= the apply), so the BOS never lies after its CTS either; a CTS anchor before the window start (the F17 (i)
shape) raises since 2026-10-09 (unreached since PLAN_J J2; until then the window was swapped).
(c) `_select_bos_on_breakout` with neither a pullback nor a proximity confirmation raises (user decision): it is
unreachable — a cycle >= 1 breakout needs the CTS CONFIRMED and both confirmation paths set `cts_confirmed_idx` —
and its old silent fallback, `_initial_bos_before_first_cts(breakout_apply_idx)`, returned the STRUCTURE's BOS_0.
"""
from __future__ import annotations

import pytest

from engine_v2.structure.market_structure import MarketStructure
from engine_v2.tests.test_unified_probe import _make_multicycle_data, _prepare_df


def _ms():
    return MarketStructure(_prepare_df(_make_multicycle_data()), 1)


@pytest.mark.illegal_event_contract  # the refused event is never built; keep the validator out of the way
@pytest.mark.skipif(not __debug__, reason="the emitter's check is an assert (stripped under -O)")
def test_emit_bos_confirmed_refuses_an_anchor_after_its_moment():
    ms = _ms()
    with pytest.raises(AssertionError, match="anchor 11 after its moment 10"):
        ms._emit_bos_confirmed(10, 1.0, bos_anchor_idx=11, meta={"confirmed_at": 10})
    assert ms.events == []
    ms._emit_bos_confirmed(10, 1.0, bos_anchor_idx=10, meta={"confirmed_at": 10})    # anchor == moment: fine
    ms._emit_bos_confirmed(12, 1.0, bos_anchor_idx=7, meta={"confirmed_at": 12})     # the usual lag
    assert [(e.idx, e.meta["bos_anchor_idx"]) for e in ms.events] == [(10, 10), (12, 7)]


def test_bos_selection_without_any_confirmation_raises():
    ms = _ms()
    st = ms.state
    assert not st.pullback_fired_for_cycle and st.last_pullback_pat_apply_idx is None and st.cts_confirmed_idx is None
    with pytest.raises(AssertionError, match="without a pullback or a proximity confirmation"):
        ms._select_bos_on_breakout(10, 10)


def test_bos_selection_on_the_pullback_window():
    """Positive control: the pullback window [7, 10] of the multicycle fixture -> its lowest low, at 7 (the
    pullback's own candle, l .6088 — the cycle-1 BOS anchor of the unbounded run), never after the apply 10."""
    ms = _ms()
    ms.state.pullback_fired_for_cycle = True
    ms.state.last_pullback_pat_apply_idx = 7
    anchor, price = ms._select_bos_on_breakout(10, 10)
    assert anchor == 7 and price == pytest.approx(0.6088)
    assert price == min(float(ms._l[k]) for k in range(7, 11))


def test_the_bos_window_ends_at_the_cts_anchor():
    """PLAN_J J3: a candle after the new CTS anchor never becomes the BOS — the pullback window [7, apply 10] with the
    CTS anchor at 9 and candle 10 made the deepest (l .6000): the BOS is 7 (the window [7, 9]); with the CTS anchor
    at the apply the same candle 10 wins (the window [7, 10])."""
    ms = _ms()
    ms.state.pullback_fired_for_cycle = True
    ms.state.last_pullback_pat_apply_idx = 7
    ms._l[10] = 0.6000
    assert ms._select_bos_on_breakout(10, 9) == (7, pytest.approx(0.6088))
    assert ms._select_bos_on_breakout(10, 10) == (10, pytest.approx(0.6000))


def test_a_cts_anchor_before_the_window_start_raises():
    """The F17 (i) shape (a cycle established on the previous cycle's CTS candle, before its confirmation): the CTS
    anchor 5 precedes the window start 7 — it raises (unreached since PLAN_J J2: 0 in the 72,000-trial corpus). Until
    2026-10-09 the window was swapped to [5, 7]."""
    ms = _ms()
    ms.state.pullback_fired_for_cycle = True
    ms.state.last_pullback_pat_apply_idx = 7
    with pytest.raises(AssertionError, match="the new CTS anchor 5 precedes the BOS window start 7"):
        ms._select_bos_on_breakout(10, 5)


def test_a_cts_anchor_at_the_window_start_is_a_one_candle_window():
    """The boundary of the raise: the CTS anchor 7 == the window start 7 is the window [7, 7] — the BOS is candle 7
    (l .6088), even with an earlier candle 5 made deeper (l .6000)."""
    ms = _ms()
    ms.state.pullback_fired_for_cycle = True
    ms.state.last_pullback_pat_apply_idx = 7
    ms._l[5] = 0.6000
    assert ms._select_bos_on_breakout(10, 7) == (7, pytest.approx(0.6088))
