"""A reversal whose pattern confirms ON its watch's expiry candle applies — on every path (F3b; user decision
2026-09-29, option I "E inclusive").

MARKET_STRUCTURE_SPEC "A reversal confirming on E applies": a close-break at anchor A opens a watch expiring at
E = min(A + range_max_k, effective_end) with a pending reversal applying at p <= E (no pending -> `rv_anchor_failed`
clears the watch at once). A + 5 is also the LAST candle a reversal pattern anchored at A may confirm on (2-candle
patterns: 4 candles after their end; `continuous`: 3), and E is inclusive for the pattern rules, the scheduler (only
`apply > E` is dropped) and the later-anchor cap. But the per-candle step ran the expiry BEFORE the pending apply, so
the same kind of pattern confirming on E was
  P1  applied when A was its own step anchor (the pattern won step A; the H1 main reversal @902 of the reference
      window is one),
  P2  applied when a later anchor's reversal won at E (`test_ms_expiry_stop` boundary pin),
  P3  DISCARDED as a false break when the pattern never won a step — A was processed inside another step, or A's
      own step chose an earlier-applying winner — so it lived only as the pending: the expiry rewound to A + 1 and
      re-ran those candles with a barrier decided at E.
Now the pending applies first (`_replay_step_no_patterns`): P3 reverses at E like P1 and P2. At the data edge the
result is prefix-stable (before, the edge false break turned into a reversal at the old edge candle as soon as one more
candle arrived). No expiry fires any more: every clear of the pending also ends the watch.

Measured before (review_scripts/reversal_shadow.py + random_tail_search.py F3b counters, scratch variants): reference
window 0 expiries and 3 of its 5 reversals already P1 at E -> byte-identical; suite 30 expiries; 42k random tails per
tree: HEAD 813 P3 discards vs 1,179 P1 + 36 P2 applied, the rule changes exactly the expiry trials, 0 errors, 0
expiries left. "E exclusive" instead would have removed the H1 main reversal (sid 0 never reversing in the window).
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

from engine_v2.structure.market_structure import MarketStructure
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests import test_ms_cts_update_no_regress as _noreg
from engine_v2.tests.test_ms_stop_after_cts import _make_double_rewind_data
from engine_v2.tests.test_unified_probe import _R, _prepare_df


# Candle 8 of `_make_double_rewind_data` completes a breakout (cycle 1 @8) inside the watch opened at 4 — which ENDS that
# watch (MARKET_STRUCTURE_SPEC "A new cycle ends an open watch"). This candle closes back at .6040, below CTS_0 .6042.
_C8 = _R(0.60540, 0.60560, 0.60380, 0.60400)


def _no_cycle_at_8(rows: list[dict]) -> list[dict]:
    rows = list(rows)
    rows[8] = _C8
    return rows


# P3 — sd=+1 (the F3 fixture, persisted 2026-09-28; candle 8 = `_C8`). 0-2 cycle 0 established at 2 (CTS .6042,
# BOS_0 = l0 .5998).
#   4   bear maru closing .5982: step 3's pullback winner applies here, and 4 close-breaks BOS_0 -> watch A=4, E=9, frozen
#       .5998. 4 is NOT a step anchor, so its reversal `one_maru_opposite` 4-5 (confirmation: a bearish normal/maru
#       closing <= min(l4, l5) = .5978) exists only as the pending.
#   5-8 no confirmation (5-7 bullish; 8 closes .6040).
#   9   bear maru closing .5962: confirms ON E (its 4th and last look-ahead candle). Before: the expiry ran first -> false
#       break, BOS := l4 .5980, rewind to 5, then REVERSAL @10 on the re-run's .5978. Now: REVERSAL @9 on .5998.
def _p3_rows() -> list[dict]:
    return _no_cycle_at_8(_make_double_rewind_data()[:9]) + [
        _R(0.60640, 0.60660, 0.59600, 0.59620),   # 9
        _R(0.59700, 0.59720, 0.58800, 0.58820),   # 10
        _R(0.58820, 0.58840, 0.58300, 0.58320),   # 11
        _R(0.58320, 0.58400, 0.58250, 0.58380),   # 12
        _R(0.58380, 0.58420, 0.58300, 0.58350),   # 13
    ]


# P1 — sd=+1 (random tails seed 11 trial 11254: `_make_double_rewind_data()[:10]` + 8). Cycle 1 established at 8 (BOS_1
# .5978; it ends watch 4). 9 (normal, closes .5970) close-breaks BOS_1 -> watch A=9, E=14, and 9 IS its step's anchor:
# the step's winner is its own `continuous` 9-11 (confirmation: a bearish normal/maru closing <= min(l9..l11) = .5918).
# 12 closes .59194 (short of it), 13 is bullish, 14 closes .58208 -> confirms ON E. Applied before and now: REVERSAL @14.
def _p1_rows() -> list[dict]:
    return list(_make_double_rewind_data()[:10]) + [
        _R(0.59677, 0.60097, 0.59519, 0.59559),   # 10
        _R(0.59533, 0.59575, 0.59187, 0.59268),   # 11
        _R(0.59271, 0.59317, 0.59148, 0.59194),   # 12
        _R(0.59212, 0.59289, 0.59170, 0.59264),   # 13
        _R(0.59276, 0.59293, 0.58183, 0.58208),   # 14
        _R(0.58222, 0.58264, 0.58090, 0.58152),   # 15
        _R(0.58142, 0.58571, 0.58008, 0.58048),   # 16
        _R(0.58050, 0.58242, 0.57778, 0.57836),   # 17
    ]


# P3, native sd=-1 (the invariant-4 review fixture, pinned in `test_ms_invariant_watch_identity` until F3b). Cycle 1
# established at 8 (BOS .60207); 9 (bull maru closing .60259) close-breaks it -> watch A=9, E=14, frozen .60207. 9 IS
# its step's anchor, but that step's winner is a pullback (`one_maru_continuous` 9-10, applying at 10 — earlier), so
# the reversal against .60207 lives only as the pending; it confirms ON 14. Before: expiry at 14 -> false break, BOS := h9 .60278, rewind to 10 -> 10
# opened a SECOND watch on .60278 whose reversal applied at 14 (back-to-back watches). Now: one watch, REVERSAL @14 on
# .60207.
def _p3_review_rows() -> list[dict]:
    return [
        _R(0.59775, 0.60095, 0.59755, 0.60075), _R(0.60066, 0.60076, 0.59846, 0.59866),   # 0, 1
        _R(0.59786, 0.59796, 0.59566, 0.59586), _R(0.59589, 0.59859, 0.59579, 0.59839),   # 2, 3
        _R(0.59774, 0.60124, 0.59764, 0.60104), _R(0.60167, 0.60207, 0.60027, 0.60087),   # 4, 5
        _R(0.60048, 0.60068, 0.59728, 0.59748), _R(0.59739, 0.59759, 0.59379, 0.59399),   # 6, 7
        _R(0.59511, 0.59531, 0.59391, 0.59411), _R(0.59345, 0.60278, 0.59327, 0.60259),   # 8, 9
        _R(0.60254, 0.60618, 0.60190, 0.60556), _R(0.60528, 0.60645, 0.60476, 0.60601),   # 10, 11
        _R(0.60573, 0.60990, 0.60464, 0.60535), _R(0.60545, 0.60687, 0.60072, 0.60614),   # 12, 13
        _R(0.60615, 0.60999, 0.60438, 0.60900), _R(0.60926, 0.61057, 0.60866, 0.61013),   # 14, 15
        _R(0.60994, 0.61163, 0.60477, 0.60591), _R(0.60584, 0.60586, 0.60053, 0.60078),   # 16, 17
        _R(0.60051, 0.60139, 0.59587, 0.59642), _R(0.59619, 0.59695, 0.59510, 0.59575),   # 18, 19
        _R(0.59591, 0.59787, 0.59210, 0.59257), _R(0.59259, 0.59386, 0.58801, 0.59356),   # 20, 21
        _R(0.59330, 0.59339, 0.58666, 0.58677), _R(0.58679, 0.59119, 0.58498, 0.59025),   # 22, 23
        _R(0.59026, 0.59034, 0.58558, 0.58564),                                           # 24
    ]


def _rows(fn, sd: int, native_sd: int = 1) -> list[dict]:
    return fn() if sd == native_sd else _noreg._mirror(fn())


def _price(p: float, sd: int, native_sd: int = 1) -> float:
    return p if sd == native_sd else round(_noreg._MIRROR - p, 5)


@pytest.fixture
def trace(monkeypatch):
    """Class-level taps: step anchors, expiries fired, rewinds, and each reversal apply's path (winner / pending)."""
    log = {"anchors": [], "winners": [], "expiries": [], "rewinds": [], "applied": []}
    o_step, o_exp = MarketStructure._step_anchor, MarketStructure._maybe_expire_reversal_watch
    o_best = MarketStructure._best_bopb_pattern_at_anchor
    o_rw, o_pend = MarketStructure._rewind_to, MarketStructure._maybe_apply_pending_reversal
    o_apply = MarketStructure._apply_pattern_at_apply_idx

    def step(self, i):
        log["anchors"].append(int(i))
        return o_step(self, i)

    def best(self, *, i, breakout_th, pullback_th, D):
        w = o_best(self, i=i, breakout_th=breakout_th, pullback_th=pullback_th, D=D)
        if w is not None:
            log["winners"].append((int(i), w[2], int(w[1])))
        return w

    def exp(self, i):
        st = self.state
        if st.reversal_watch_active and int(i) >= int(st.reversal_watch_expires_idx):
            log["expiries"].append(int(i))
        return o_exp(self, i)

    def rw(self, jump_to, *, seed=None):
        log["rewinds"].append(int(jump_to))
        return o_rw(self, jump_to, seed=seed)

    def pend(self, i):
        self._f3b_path = "pending"
        try:
            return o_pend(self, i)
        finally:
            self._f3b_path = "winner"

    def apply(self, ev, apply_idx, kind):
        if kind == "reversal" and self.state.reversal_watch_active:
            st = self.state
            log["applied"].append((getattr(self, "_f3b_path", "winner"), int(apply_idx),
                                   int(st.reversal_watch_start_idx), int(st.reversal_watch_expires_idx)))
        return o_apply(self, ev, apply_idx, kind)

    monkeypatch.setattr(MarketStructure, "_step_anchor", step)
    monkeypatch.setattr(MarketStructure, "_best_bopb_pattern_at_anchor", best)
    monkeypatch.setattr(MarketStructure, "_maybe_expire_reversal_watch", exp)
    monkeypatch.setattr(MarketStructure, "_rewind_to", rw)
    monkeypatch.setattr(MarketStructure, "_maybe_apply_pending_reversal", pend)
    monkeypatch.setattr(MarketStructure, "_apply_pattern_at_apply_idx", apply)
    return log


def _run(rows, sd: int, end_idx=None):
    with redirect_stdout(io.StringIO()):
        return compute_bounded_structure(_prepare_df(rows), 0, sd, end_idx=end_idx)


def _reversals(res):
    return [(int(e.idx), e.meta["pat"], round(float(e.meta["bos_frozen"]), 5))
            for e in res.events if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal"]


def _watches(res):
    return [(int(e.idx), int(e.meta["expires_idx"])) for e in res.events if e.type == "REVERSAL_WATCH_START"]


@pytest.mark.parametrize("sd", [1, -1])
def test_p3_a_pending_reversal_confirming_on_the_expiry_applies(sd, trace):
    res = _run(_rows(_p3_rows, sd), sd)
    assert 4 not in trace["anchors"]                      # the close-break candle ran inside step 3
    assert _watches(res) == [(4, 9)]
    assert trace["applied"] == [("pending", 9, 4, 9)]     # the pending, ON its watch's expiry candle
    assert trace["expiries"] == [] and trace["rewinds"] == []
    assert _reversals(res) == [(9, "one_maru_opposite", _price(0.5998, sd))]   # before: @10 on .5978


@pytest.mark.parametrize("sd", [1, -1])
def test_p3_one_watch_where_an_expiry_made_two(sd, trace):
    res = _run(_rows(_p3_review_rows, sd, native_sd=-1), sd)
    assert (9, "pullback", 10) in trace["winners"]         # the close-break candle's step chose an earlier winner
    assert trace["applied"] == [("pending", 14, 9, 14)]
    assert trace["expiries"] == [] and trace["rewinds"] == []
    assert _watches(res)[-1] == (9, 14)                   # before: a second watch (10, 15) after the expiry's rewind
    assert [r[0] for r in _reversals(res)] == [14]
    assert _reversals(res)[0][2] == _price(0.60207, sd, native_sd=-1)          # before: .60278


@pytest.mark.parametrize("sd", [1, -1])
def test_p1_the_close_break_candles_own_winner_on_the_expiry_applies(sd, trace):
    """The other path to the same boundary (unchanged by F3b): the close-break candle is its step's anchor."""
    res = _run(_rows(_p1_rows, sd), sd)
    assert (9, "reversal", 14) in trace["winners"]         # its own pattern wins step 9
    assert _watches(res)[-1] == (9, 14)
    assert trace["applied"] == [("winner", 14, 9, 14)]
    assert trace["expiries"] == [] and trace["rewinds"] == []
    assert _reversals(res) == [(14, "continuous", _price(0.5978, sd))]


@pytest.mark.parametrize("sd", [1, -1])
def test_a_reversal_on_the_expiry_at_the_data_edge_is_prefix_stable(sd):
    """Bounded at E = 9 the watch window ends at the edge (LANDMINES L4): the reversal applies there, and every later
    bound keeps it (before: no reversal at B = 9, then REVERSAL @10 from B = 10 on — the edge false break was rewritten
    as soon as the next candle arrived)."""
    for B in (9, 10, 11, 13):
        res = _run(_rows(_p3_rows, sd), sd, end_idx=B)
        assert [r[0] for r in _reversals(res)] == [9], f"B={B}"
