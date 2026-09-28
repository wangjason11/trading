"""BOS selection guards (2026-09-28; zones-audit latent items (c) and (e)).

(e) `_emit_bos_confirmed` asserts `bos_anchor_idx <= idx`: the anchor is a location already SEEN at the moment. Only
`_select_bos_on_breakout`'s swapped window (`window_start > breakout_apply_idx`) could hand it a later one, and the
anchor-keyed processing order would then run the cycle's CTS_ESTABLISHED before its BOS_CONFIRMED (FibTracker needs
the BOS first). Reference window: 19 cycle >= 1 selections, all on the pullback window, 0 swapped, 0 anchors after
the apply (shadow over every MS run, 2026-09-28).
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
        ms._select_bos_on_breakout(10)


def test_bos_selection_on_the_pullback_window():
    """Positive control: the pullback window [7, 10] of the multicycle fixture -> its lowest low, at 7 (the
    pullback's own candle, l .6088 — the cycle-1 BOS anchor of the unbounded run), never after the apply 10."""
    ms = _ms()
    ms.state.pullback_fired_for_cycle = True
    ms.state.last_pullback_pat_apply_idx = 7
    anchor, price = ms._select_bos_on_breakout(10)
    assert anchor == 7 and price == pytest.approx(0.6088)
    assert price == min(float(ms._l[k]) for k in range(7, 11))
