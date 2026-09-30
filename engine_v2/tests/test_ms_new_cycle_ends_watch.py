"""A new cycle ends an open reversal watch (user decision 2026-09-29; MARKET_STRUCTURE_SPEC "A new cycle ends an
open watch"; GOTCHAS "A Cycle Cannot Be Established Inside an Open Reversal Watch").

A watch freezes the CURRENT cycle's BOS when a candle closes beyond it; a reversal pattern confirming inside the
window reverses the structure. When a breakout establishes a NEW cycle while the watch is open, the market has made
a new extreme instead of reversing, and the new cycle's BOS_CONFIRMED supersedes the barrier the watch froze. The
watch ends there: its pending reversal (confirming only after the establishing candle — not knowable yet) is dropped
and a later close beyond the NEW BOS opens a new watch. The new cycle's BOS_CONFIRMED carries
`ended_watch_pattern_anchor_idx` (the ended watch's close-break candle = its REVERSAL_WATCH_START idx).

Before: the old watch stayed open across the establishment — its pending reversal applied on the superseded barrier
(invariant 4 then raised "bos_threshold changed during reversal watch" and ended the replay: a live crash path), or
its own expiry rewound the new cycle away (the common shape, `_make_double_rewind_data`). Measured before the change:
replay 0 in-watch cycles (byte-identical); suite 38 (2 crash = the old raise-pin, sd +-1); random tails 0 crashes in
48k but ~38% of trials with an in-watch cycle, all rewound away by the old watch's expiry; 32 crashes in 12k tails
started from the pin's open watch.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.structure.market_structure import MarketStructure
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _R, _make_multicycle_data, _prepare_df


# The real-rows fixture the invariant-4 landing review built (sd=+1): `_make_double_rewind_data()[:6]` + 5 candles.
#   0-2  BOS_0 .5998 (l0); cycle 0 established at 2, CTS_0 .6042.
#   4    pullback -> CTS_0 confirmed; closes .5982 < BOS_0 -> watch 4 (frozen .5998, expires 9); its pattern (4-5)
#        confirms on a bearish close <= min(l4, l5) = .5978 — candle 8 (pending apply 8).
#   6-7  breakout, new high .6092 -> cycle 1 established at 7, BOS_1 = l5 .5978 — INSIDE watch 4: it ends here.
#   8    closes .5960: below BOS_1 -> watch 8 (frozen .5978); its pattern (8-9) confirms at 10.
# Until 2026-09-29: REVERSAL @8 on the superseded .5998, then invariant 4 raised at row 7.
def _in_watch_cycle_rows() -> list[dict]:
    return list(_make_double_rewind_data()[:6]) + [
        _R(0.59900, 0.60620, 0.59880, 0.60600),   # 6
        _R(0.60600, 0.60920, 0.60580, 0.60900),   # 7
        _R(0.60900, 0.60920, 0.59580, 0.59600),   # 8
        _R(0.59600, 0.59620, 0.59300, 0.59320),   # 9
        _R(0.59320, 0.59340, 0.59100, 0.59120),   # 10
    ]


def _run(rows, sd):
    rows = rows if sd == 1 else _noreg._mirror(rows)
    with redirect_stdout(io.StringIO()):
        return compute_bounded_structure(_prepare_df(rows), 0, sd)   # debug_invariants on (the default)


def _p(price, sd):
    return price if sd == 1 else round(_noreg._MIRROR - price, 5)


def _est(res):
    return [(e.meta["cycle_id"], e.idx) for e in res.events if e.type == "CTS_ESTABLISHED"]


def _bos_confirmed(res):
    return [(e.idx, round(e.price, 5), e.meta.get("ended_watch_pattern_anchor_idx"))
            for e in res.events if e.type == "BOS_CONFIRMED"]


def _watches(res):
    return [(e.idx, round(e.price, 5), e.meta["expires_idx"]) for e in res.events if e.type == "REVERSAL_WATCH_START"]


def _reversal_frozen(res):
    return [round(e.meta["bos_frozen"], 5) for e in res.events
            if e.type == "STATE_CHANGED" and e.meta.get("to") == "reversal"]


@pytest.mark.parametrize("sd", [1, -1])
def test_a_new_cycle_ends_the_open_watch_and_a_new_bos_break_reverses(sd):
    """The pin + 2 more bearish candles (so the new watch's pattern can apply before the data edge): watch 4 ends at
    7 (traced on the BOS_CONFIRMED), its candidate (4 -> 8) never applies, candle 8 opens watch 8 on BOS_1 and the
    structure reverses at 10 on BOS_1 — not at 8 on the superseded BOS_0."""
    res = _run(_in_watch_cycle_rows() + [_R(0.59120, 0.59140, 0.58900, 0.58920),
                                         _R(0.58920, 0.58950, 0.58700, 0.58720)], sd)
    assert _est(res) == [(0, 2), (1, 7)]
    assert _bos_confirmed(res) == [(2, _p(0.5998, sd), None), (7, _p(0.5978, sd), 4)]
    assert [(e.idx, e.meta["apply_idx"]) for e in res.events if e.type == "REVERSAL_CANDIDATE"] == [(4, 8), (8, 10)]
    assert _watches(res) == [(4, _p(0.5998, sd), 9), (8, _p(0.5978, sd), 12)]
    df = res.df
    assert [int(df.loc[i, "reversal_watch_active"]) for i in (6, 7, 8)] == [1, 0, 1]   # watch 4 | ended | watch 8
    assert res.reversal_idx == 10 and _reversal_frozen(res) == [_p(0.5978, sd)]


@pytest.mark.parametrize("sd", [1, -1])
def test_the_pinned_rows_run_clean(sd):
    """The 11 pinned rows (the old raise-pin): no invariant error. The new watch 8 meets the data edge (expires 10 ==
    its pending apply): the reversal applies AT the edge on the NEW BOS (F3b, 2026-09-29; before: a false break)."""
    res = _run(_in_watch_cycle_rows(), sd)
    assert _bos_confirmed(res)[-1] == (7, _p(0.5978, sd), 4)
    assert res.reversal_idx == 10
    rev = [e for e in res.events if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal"]
    assert [round(float(e.meta["bos_frozen"]), 5) for e in rev] == [_p(0.5978, sd)]


@pytest.mark.parametrize("sd", [1, -1])
def test_the_old_pattern_cannot_reverse_above_the_new_bos(sd):
    """Why the old watch must end: the old pattern's confirm threshold min(l4, l5) = .5978 can sit ABOVE the new BOS.
    Candle 6's wick makes BOS_1 = .5970; candle 8 closes .5976 — it confirms the old pattern but does NOT close below
    the current BOS. Before: REVERSAL @8 (on .5998) though no candle closed below .5970 (price then rallies to .6048).
    Now: no watch after 4, no reversal."""
    rows = list(_make_double_rewind_data()[:6]) + [
        _R(0.59900, 0.60620, 0.59700, 0.60600),   # 6: the deeper wick .5970 -> BOS_1
        _R(0.60600, 0.60920, 0.60580, 0.60900),   # 7: cycle 1 established
        _R(0.60900, 0.60920, 0.59740, 0.59760),   # 8: closes .5976 <= .5978 (old threshold), > .5970 (BOS_1)
        _R(0.59760, 0.59780, 0.59740, 0.59770),   # 9
        _R(0.59770, 0.60100, 0.59760, 0.60080),   # 10
        _R(0.60080, 0.60300, 0.60060, 0.60280),   # 11
        _R(0.60280, 0.60500, 0.60260, 0.60480),   # 12
    ]
    res = _run(rows, sd)
    assert _bos_confirmed(res) == [(2, _p(0.5998, sd), None), (7, _p(0.5970, sd), 4)]
    assert _watches(res) == [(4, _p(0.5998, sd), 9)]
    assert res.reversal_idx is None


def test_the_double_rewind_fixture_keeps_its_new_high_as_a_cycle():
    """`_make_double_rewind_data` (sd +1), the common shape in random tails: cycle 1 (new high .6066) is established
    at 8 inside watch 4, whose pattern confirms ON the expiry 9. Before: the expiry won, rewound to 5, and the .6066
    high became neither a cycle nor a CTS update (cycles at 2 / 12). Now cycle 1 stands at 8, ending watch 4."""
    res = _run(_make_double_rewind_data(), 1)
    assert _est(res) == [(0, 2), (1, 8), (2, 12)]
    assert [round(e.price, 5) for e in res.events if e.type == "CTS_ESTABLISHED"] == [0.6042, 0.6066, 0.6092]
    assert [x[2] for x in _bos_confirmed(res)] == [None, 4, None]
    assert res.reversal_idx == 17


# --- unit level: the boundary of `_end_watch_superseded_by_new_cycle` --------------------------------------------

def _ms_with_watch(*, pending_apply):
    ms = MarketStructure(_prepare_df(_make_multicycle_data()), 1)
    st = ms.state
    st.reversal_watch_active, st.reversal_watch_start_idx = True, 4
    st.reversal_bos_th_frozen, st.reversal_watch_expires_idx = 0.6, 9
    st.pending_reversal_ev, st.pending_reversal_pattern_anchor_idx = object(), 4
    st.pending_reversal_apply_idx = pending_apply
    return ms


def test_a_pending_after_the_establishing_candle_is_dropped():
    ms = _ms_with_watch(pending_apply=8)
    assert ms._end_watch_superseded_by_new_cycle(7, {"k": 1}) == {"k": 1, "ended_watch_pattern_anchor_idx": 4}
    st = ms.state
    assert not st.reversal_watch_active and st.reversal_watch_start_idx is None
    assert st.pending_reversal_apply_idx is None and st.pending_reversal_ev is None


def test_a_pending_on_the_establishing_candle_is_dropped_too():
    """No equal-apply exception (user decision 2026-09-29, landing review of 2285232): on that candle the new BOS is
    at or beyond its close, so that reversal could never have broken the current BOS."""
    ms = _ms_with_watch(pending_apply=7)
    assert ms._end_watch_superseded_by_new_cycle(7, {"k": 1}) == {"k": 1, "ended_watch_pattern_anchor_idx": 4}
    assert not ms.state.reversal_watch_active and ms.state.pending_reversal_apply_idx is None


# The equal apply on real rows (the landing review of 2285232 built both; sd +1). Only a `one_maru_opposite`
# breakout applies on a candle that can also confirm the old bearish pattern: its small OPPOSITE candle. That candle
# closes in the top ~35% of the breakout maru, which needs >= 30% of its body beyond the range high — so, without a
# price gap, the maru must be ~20x the range-high-to-threshold distance (here ~1,280 pips). Both rows use gaps.
#   6 / 8  a gapped bull maru .5800 -> .6150 (new high .6152; BOS_1 = its low .5798);
#   7 / 9  a gapped small bear candle closing .5977 <= min(l4, l5) = .5978: it completes the breakout AND confirms
#          watch 4's pattern (candidate apply 7 / 9 == the establishing candle).
# Until the decision: at 7 the cycle was established and then reversed on the superseded .5998 (a close far above
# BOS_1); at 9 == the watch's expiry the expiry won and rewound to 5 — the .6152 high became neither a cycle nor a
# CTS update. Now the new cycle ends the watch on both: no reversal, cycle 1 stands.
_GAP_MARU = _R(0.58000, 0.61520, 0.57980, 0.61500)
_GAP_BEAR = _R(0.59900, 0.59920, 0.59760, 0.59770)
_BEAR_FILL = [_R(0.59750, 0.59800, 0.59500, 0.59550), _R(0.59550, 0.59600, 0.59400, 0.59450)]


@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("apply_at", [7, 9])   # 9 == the watch's expiry
def test_an_equal_apply_ends_the_watch_too(sd, apply_at):
    pre = [] if apply_at == 7 else [_R(0.59900, 0.59960, 0.59880, 0.59940), _R(0.59940, 0.59990, 0.59920, 0.59970)]
    rows = list(_make_double_rewind_data()[:6]) + pre + [_GAP_MARU, _GAP_BEAR] + _BEAR_FILL
    res = _run(rows, sd)
    assert [(e.idx, e.meta["apply_idx"]) for e in res.events if e.type == "REVERSAL_CANDIDATE"] == [(4, apply_at)]
    assert _est(res) == [(0, 2), (1, apply_at)]
    assert _bos_confirmed(res) == [(2, _p(0.5998, sd), None), (apply_at, _p(0.5798, sd), 4)]
    assert _watches(res) == [(4, _p(0.5998, sd), 9)]
    assert res.reversal_idx is None


def test_no_open_watch_leaves_the_meta_alone():
    ms = MarketStructure(_prepare_df(_make_multicycle_data()), 1)
    assert ms._end_watch_superseded_by_new_cycle(7, {"k": 1}) == {"k": 1}
