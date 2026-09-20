"""Plan A property test — a bounded MarketStructure run reads nothing past its bound.

Definition under test (`plans/PLAN_A_ms_bounds_leak.md` §2; LANDMINES "Bounded MS Runs
Must Not Read Past `end_idx`"): for EVERY bound B,

    MarketStructure(df, start, end_idx=B).run()
        ==  MarketStructure(df[:B+1] (features recomputed), start, end_idx=None).run()

on (a) the whole event list (type, idx, price, full meta), (b) the 27 MS output columns
over rows [0, B] (rows < start included — rewinds write them), (c) `reversal_idx`.

Checked over every bound of every fixture so it cannot pass by fixture luck. Parametrised
over sd=±1 and over plain vs pre-CTS_0 scan mode (the production Phase-2 caller).

At the pre-fix base (Stage 3.2a tree, revert save `5a658dc`) this test FAILS — measured:
reversing fixture sd=+1: 26 of 57 bounds (both modes) = 22 bounds with a bounded-only event
past B (the plan's "22 of 57", first at B=3 where the bounded run emits `CTS_UPDATED@4` — a
pattern anchored at i<=B applied at i+2>B, leak site L1 `D = min(i + range_max_k, n-1)`) plus
4 bounds (B=52..55) that differ ONLY in `expires_idx` on `REVERSAL_WATCH_START`/`REVERSAL_CANDIDATE`
(L4 — the field the full-meta signature exists to catch); reversing sd=-1: 7 (plain) / 3 (scan);
uptrend sd=+1: 38 of 76 (both modes); uptrend sd=-1: 0 (no structure forms); the hand-written
§5.2 fixtures below (L1..L4) and the multi-cycle fixture (9 of 20 bounds) fail at the base too.

Module layout: property test over `_FIXTURES` -> per-mechanism tests (L1..L4, the fixed
behaviour at the exact bound) -> negative test for the post-run assert.

Runtime: ~0.08 s per bound (features + two runs); whole matrix ~35 s on a quiet machine.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine_v2.structure.market_structure import (
    _OUT_FLOAT_NAN,
    _OUT_FLOAT_NEG1,
    _OUT_INT_NEG1,
    _OUT_INT_ZERO,
    _OUT_OBJ_EMPTY,
)
from engine_v2.structure.structure_engine import compute_bounded_structure
from engine_v2.tests.test_bounded_structure import (
    _make_reversing_data,
    _make_uptrend_data,
    _prepare_df,
)
from engine_v2.tests.test_unified_probe import _make_multicycle_data

OUT_COLS = list(_OUT_INT_NEG1 + _OUT_INT_ZERO + _OUT_FLOAT_NEG1 + _OUT_FLOAT_NAN + _OUT_OBJ_EMPTY)
assert len(OUT_COLS) == 27

_START = 0


# ---------------------------------------------------------------------------
# Hand-written boundary fixtures (Plan A §5.2). Each pins ONE leak mechanism at
# an exact bound so a regression names its site; each is also fed to the
# property test below. All sd=+1, start_idx=0, NZD_USD, classified as H1 by
# `_prepare_df` (2.2-pip body floor => bodies < 0.00022 are forced `pinbar`).
# Rows are 5-dp literals; the layout comments are the construction recipe.
# ---------------------------------------------------------------------------

def _R(o, h, l, c):
    return {"o": round(o, 5), "h": round(h, 5), "l": round(l, 5), "c": round(c, 5)}


def _make_l1_pattern_past_bound():
    """L1 — a breakout pattern anchored at i=5 whose apply candle is i+2=7.

    0      big down maru (len 20 pips); l[0] = 0.59820 is the pre-leg low -> BOS_0.
    1..4   small DOWN candles (normal/maru alternating) above l[0]: no +1 anchor, the
           marus seed the prior-maru pool for candle 5.
    5      big UP maru, range 30 pips (ratio 1.5 -> is_big_maru_as0) = one_maru_opposite c0.
    6      down NORMAL, len 15 pips = 0.5*len5 -> c1_len_check fails -> FAIL_NEEDS_CONFIRM.
    7      up maru closing 3 pips above max(h5, h6) -> _price_confirmation lands at 7.
    8..13  staircase-up 1-pip-body pinbars, each above the previous high: no range label.
           (Anchor 7 does carry a one_maru_continuous FAIL_NEEDS_CONFIRM — it stays unconfirmed
           only because 9..12 are pinbars, which _price_confirmation ignores; keep them pinbars.)
    Base: bounded(B=5 or 6) applied the CONFIRMED pattern at 7 > B (CTS_ESTABLISHED /
    BOS_CONFIRMED confirmed_at=7, STATE_CHANGED@7). Fixed == truncated: no events until B=7.
    Scan mode does not leak here (find_true_first_breakout drops est 7 > hi) — the plan's
    Phase-1 equivalence claim, live.
    """
    return [
        _R(0.60000, 0.60020, 0.59820, 0.59840),   # 0
        _R(0.59900, 0.59920, 0.59845, 0.59860),   # 1
        _R(0.59910, 0.59925, 0.59835, 0.59850),   # 2
        _R(0.59900, 0.59930, 0.59840, 0.59860),   # 3
        _R(0.59910, 0.59925, 0.59835, 0.59850),   # 4
        _R(0.59850, 0.60140, 0.59840, 0.60105),   # 5 = i (omo c0)
        _R(0.60085, 0.60130, 0.59980, 0.60018),   # 6 (omo c1, fails c1_len_check)
        _R(0.60028, 0.60188, 0.60010, 0.60170),   # 7 = i+2 (confirmation candle)
        _R(0.60198, 0.60248, 0.60188, 0.60208),   # 8
        _R(0.60258, 0.60308, 0.60248, 0.60268),   # 9
        _R(0.60318, 0.60368, 0.60308, 0.60328),   # 10
        _R(0.60378, 0.60428, 0.60368, 0.60388),   # 11
        _R(0.60438, 0.60488, 0.60428, 0.60448),   # 12
        _R(0.60498, 0.60548, 0.60488, 0.60508),   # 13
    ]


def _make_l2_range_at_bound():
    """L2 — range candidate i=5 whose FIRST confirming close is at i+3=8.

    0      bearish maru (reference maru so candle 1 is is_big_maru_as0).
    1      bullish big maru = one_maru_continuous c0; 2: small bullish normal, l > mid(1)
           -> omc SUCCESS, apply 2: CTS_0=(1, .60310), BOS_0=(0, .60000), state BREAKOUT.
    3..4   bullish normals (dir == sd, not pinbar) -> can never be range candidates.
    5 = i  bearish normal (dir != sd -> qualifies); [l_i, h_i] = [.60210, .60292].
    6      bullish normal inside candle i (irrelevant to the label: min lookahead k=2).
    7      close .60300 > h_i (outside -> no confirm at i+2) but <= CTS .60310 (no CTS_UPDATED).
    8      bearish normal, close .60268 inside [l_i, h_i] -> is_range_confirm_idx[5] = 8.
    9..13  tiny bearish pinbars inside [l_i, CTS]; no maru after candle 1 -> no later
           pattern; all lows > BOS_0 -> no watch.
    Base: bounded(B=7) stamped RANGE_STARTED@8 + STATE_CHANGED->range@8 (the label's
    confirm_idx, never compared to the bound). Fixed == truncated: nothing until B=8.
    """
    return [
        _R(0.60200, 0.60210, 0.60000, 0.60010),   # 0
        _R(0.60010, 0.60310, 0.60000, 0.60300),   # 1
        _R(0.60250, 0.60295, 0.60240, 0.60280),   # 2
        _R(0.60230, 0.60290, 0.60220, 0.60270),   # 3
        _R(0.60240, 0.60300, 0.60225, 0.60285),   # 4
        _R(0.60285, 0.60292, 0.60210, 0.60245),   # 5 = i
        _R(0.60250, 0.60288, 0.60235, 0.60275),   # 6
        _R(0.60275, 0.60306, 0.60262, 0.60300),   # 7 = i+2
        _R(0.60300, 0.60308, 0.60240, 0.60268),   # 8 = i+3
        _R(0.60265, 0.60295, 0.60230, 0.60262),   # 9
        _R(0.60262, 0.60290, 0.60225, 0.60258),   # 10
        _R(0.60258, 0.60288, 0.60228, 0.60255),   # 11
        _R(0.60255, 0.60285, 0.60222, 0.60250),   # 12
        _R(0.60250, 0.60280, 0.60220, 0.60248),   # 13
    ]


def _make_l3_priority_preemption():
    """L3 — at anchor i=5 a 2-candle `one_maru_continuous` SUCCEEDs at i+1 AND a
    3-candle `continuous` (variant 1) SUCCEEDs at i+2; `detect_best_for_anchor`
    prefers the continuous.

    0..4   five small DOWN marus (prior-maru pool; direction -1 -> state stays NONE).
    5      c0: big UP maru (44 pips, body_pct .91) -> is_big_maru_as0 / as0; mid 1.0080.
    6      c1: 1-pip UP body (pinbar by body_pct 0.08, also under the H1 floor), l=1.0088 >
           c0.mid -> omc SUCCESS at 6; the pinbar continuous variant 1 needs; h6 < h5.
    7      c2: big UP maru closing 1.0140 > max(h5, h6) -> continuous v1 SUCCESS at 7.
    8..9   filler UP normals closing above h7 (no range label). Anchor 7 would carry an omc
           SUCCESS applying at 8, but it is consumed by the continuous apply at 7 (next anchor 8).
    Base: bounded(B=6) returned the continuous (priority 1) and applied it at 7 > B.
    Fixed == truncated: at B=6 the continuous does not exist (needs idx+2), so the
    2-candle pattern establishes CTS_0 AT the bound (confirmed_at=6) — an L1-only fix
    would establish nothing here. B>=7: continuous wins (confirmed_at=7), unchanged.
    """
    return [
        _R(1.01000, 1.01010, 1.00910, 1.00920),   # 0
        _R(1.00920, 1.00930, 1.00830, 1.00840),   # 1
        _R(1.00840, 1.00850, 1.00750, 1.00760),   # 2
        _R(1.00760, 1.00770, 1.00670, 1.00680),   # 3
        _R(1.00680, 1.00690, 1.00590, 1.00600),   # 4
        _R(1.00600, 1.01020, 1.00580, 1.01000),   # 5 = i (c0)
        _R(1.00940, 1.01000, 1.00880, 1.00950),   # 6 (c1)
        _R(1.00950, 1.01420, 1.00930, 1.01400),   # 7 (c2)
        _R(1.01450, 1.01500, 1.01430, 1.01480),   # 8
        _R(1.01500, 1.01550, 1.01480, 1.01530),   # 9
    ]


def _make_l4_watch_at_bound():
    """L4 — a close-break of `bos_threshold` at i=4 whose reversal pattern applies at
    i+k = 6 (k=2).

    0      down maru -> BOS_0 low 0.59980.
    1..2   up marus: one_maru_continuous(1) SUCCESS, apply 2 -> CTS_0 = h[2] = 0.60620.
    3      big down maru closing ABOVE BOS -> one_maru_continuous(3, -1) pullback, apply 4
           (CTS_CONFIRMED@4, range created).
    4 = i  down maru closing BELOW BOS (the pullback's apply candle -> the watch starts in
           the per-candle step); one_maru_opposite(4, -1, bos) = FAIL_NEEDS_CONFIRM.
    5      up normal (defeats cand2; not a pinbar).
    6      down maru, c6 <= min(l4, l5) -> confirms the reversal at 6 = i+k.
    7      up maru with len7 >= len6 (no -1 pattern anchors at 6); 8..11 up filler so that
           the unbounded expires_idx = min(i+5, n-1) = 9.
    Base: `expires_idx = min(i+5, n-1) = 9` at every bound, so at B=4/5 the run ended with an
    open watch + a pending reversal scheduled from candle 6 > B, and at B=6 the pending apply
    ran before the (never-reached) expiry -> a reversal at 6 on a run bounded at 6.
    Fixed == truncated: B=i+k-1 -> the anchor fails at once (`rv_anchor_failed`, no candidate);
    B=i+k -> `expires_idx == B`, the pending apply is discarded as a false break (expiry runs
    before the apply in the per-candle step), NO reversal; B>i+k -> reversal at 6 as before.
    """
    return [
        _R(0.60300, 0.60320, 0.59980, 0.60000),   # 0
        _R(0.60000, 0.60420, 0.59990, 0.60400),   # 1
        _R(0.60400, 0.60620, 0.60380, 0.60600),   # 2
        _R(0.60600, 0.60610, 0.60260, 0.60280),   # 3
        _R(0.60280, 0.60290, 0.59800, 0.59820),   # 4 = i (close-break)
        _R(0.59830, 0.60080, 0.59810, 0.60000),   # 5
        _R(0.60000, 0.60010, 0.59680, 0.59700),   # 6 = i+k (confirms the reversal)
        _R(0.59700, 0.60070, 0.59690, 0.60050),   # 7
        _R(0.60050, 0.60270, 0.60040, 0.60250),   # 8
        _R(0.60250, 0.60470, 0.60240, 0.60450),   # 9 = i+5
        _R(0.60450, 0.60620, 0.60440, 0.60600),   # 10
        _R(0.60600, 0.60720, 0.60590, 0.60700),   # 11
    ]


def _make_l5_bos_inner_at_bound():
    """L5 — the cycle-1 BOS inner is derived by the resolver from candles past the bound
    and consumed by the proximity confirmation at a candle inside it. Prices = 0.6000 +
    pips.

    0..5   bullish special-marus stepping +10p: double_maru@0 CONFIRMED@2 -> CTS_0=(2, .6031),
           BOS_0=(0, .5991); raw CTS_UPDATEDs to 6.
    6..7   two bearish marus: one_maru_continuous(-1) pullback applies @7 -> CTS_0 CONFIRMED.
    8 = X  BOS_1 extreme: a big bearish forced-pinbar (2p body, 28p upper wick), range [12, 40]p.
    9      huge bullish maru (double_maru c0, closes above CTS_0); 10: tall bullish normal
           (dm alt path) -> breakout applies @10 = a1: CTS_1=(10, .6121), BOS_1=(8, .6012),
           |CTS_1-BOS_1| = 109p >= the 50p gap gate.
    11 = j bearish normal, low .6039: within 8 pips of the inside-bar inner (.6033) but NOT
           of the alternative inner (.6015, the 1-candle 'no base big tail down' path the
           frame takes while the inside bars are not yet visible).
    12,13  inside bars of X (fully inside [.6012, .6040]) = bos_idx+4, bos_idx+5 -> on the
           full frame identify_base_pattern returns ('base inside bar', 8), inner .6033.
    14,15  tail bearish normals (no range candidate, lows > BOS_1).
    Base: bounded(B=11 or 12) derived the inner from candles 12-13 > B and confirmed CTS_1 via
    sd_zone_proximity at 11 (trigger_inner .6033), creating a range. Fixed == truncated: the
    resolver sees the frame cut at B -> inner .6015 -> no hit; from B=13 (bos_idx+5) both
    frames contain the inside bars and the confirmation at 11 is back.
    """
    P0, PIP = 0.60000, 0.0001

    def r(o, h, l, c):
        return _R(P0 + o * PIP, P0 + h * PIP, P0 + l * PIP, P0 + c * PIP)

    rows = []
    for k in range(6):
        p = 10 * k
        rows.append(r(p, p + 11, p - 9, p + 10))     # 0..5
    rows.append(r(61, 62, 40, 44))                    # 6
    rows.append(r(44, 45, 14, 16))                    # 7 (pullback apply)
    rows.append(r(15, 40, 12, 13))                    # 8 = X = bos_idx
    rows.append(r(15, 85, 14, 84))                    # 9
    rows.append(r(84, 121, 36, 121))                  # 10 = a1, CTS_1
    rows.append(r(119, 120, 39, 70))                  # 11 = j
    rows.append(r(24, 36, 24, 27))                    # 12 inside bar (bos+4)
    rows.append(r(30, 34, 21, 33))                    # 13 inside bar (bos+5)
    rows.append(r(23, 25, 18, 20))                    # 14
    rows.append(r(19, 21, 14, 16))                    # 15
    return rows


_FIXTURES = {
    "reversing": _make_reversing_data,
    "uptrend": _make_uptrend_data,
    "l1_pattern_past_bound": _make_l1_pattern_past_bound,
    "l2_range_at_bound": _make_l2_range_at_bound,
    "l3_priority_preemption": _make_l3_priority_preemption,
    "l4_watch_at_bound": _make_l4_watch_at_bound,
    "l5_bos_inner_at_bound": _make_l5_bos_inner_at_bound,
    # 4 CTS cycles (Plan A §5.3 / Plan B): at the base 9 of 20 bounds leaked (L1 at
    # every breakout anchor; L2 at the pullback candles 6/11/16).
    "multicycle": _make_multicycle_data,
}


def _events(raw, B, sd=1, **kw):
    """Bounded run at B -> list of (type, idx, meta)."""
    res = compute_bounded_structure(_prepare_df(raw), _START, sd, end_idx=B, **kw)
    return [(ev.type, int(ev.idx), ev.meta) for ev in res.events]


def _of_type(evs, t):
    return [e for e in evs if e[0] == t]


def _ev_sig(events):
    """Whole-event signature: type, idx, price, and the FULL meta (so `expires_idx` on the
    reversal-watch events — the field the L4 decision makes equal — is compared too)."""
    return sorted(
        (
            ev.type,
            int(ev.idx),
            None if ev.price is None else round(float(ev.price), 6),
            json.dumps(ev.meta, sort_keys=True, default=str),
        )
        for ev in events
    )


def _mode_kwargs(mode: str, full: pd.DataFrame, sd: int) -> dict:
    if mode == "plain":
        return {}
    # Production Phase-2 caller: pre-CTS_0 scan mode against a handed BOS_0 inner.
    inner = float(full["l"].iloc[_START]) if sd == 1 else float(full["h"].iloc[_START])
    return {"enforce_cts0_new_extreme": True, "bos0_inner": inner}


def _check_all_bounds(raw, sd: int, mode: str):
    full = _prepare_df(raw)
    kw = _mode_kwargs(mode, full, sd)
    failures = []
    for B in range(_START + 3, len(raw) - 1):
        bounded = compute_bounded_structure(full, _START, sd, end_idx=B, **kw)
        truncated = compute_bounded_structure(_prepare_df(raw[: B + 1]), _START, sd, end_idx=None, **kw)

        problems = []
        if _ev_sig(bounded.events) != _ev_sig(truncated.events):
            b_only = sorted(set(_ev_sig(bounded.events)) - set(_ev_sig(truncated.events)))
            t_only = sorted(set(_ev_sig(truncated.events)) - set(_ev_sig(bounded.events)))
            problems.append(
                f"events differ: bounded-only={[(t, i) for t, i, _, _ in b_only]} "
                f"truncated-only={[(t, i) for t, i, _, _ in t_only]}"
            )
        try:
            pd.testing.assert_frame_equal(
                bounded.df.iloc[0 : B + 1][OUT_COLS].reset_index(drop=True),
                truncated.df.iloc[0 : B + 1][OUT_COLS].reset_index(drop=True),
                check_dtype=False,
            )
        except AssertionError as exc:  # pragma: no cover - message only
            problems.append("df output cols differ: " + str(exc).splitlines()[0])
        if bounded.reversal_idx != truncated.reversal_idx:
            problems.append(f"reversal_idx {bounded.reversal_idx} != {truncated.reversal_idx}")
        if problems:
            failures.append((B, problems))
    return failures, len(range(_START + 3, len(raw) - 1))


@pytest.mark.parametrize("mode", ["plain", "scan"])
@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("fixture_name", sorted(_FIXTURES))
def test_bounded_run_equals_truncated_run(fixture_name, sd, mode):
    raw = _FIXTURES[fixture_name]()
    failures, n_bounds = _check_all_bounds(raw, sd, mode)
    assert not failures, (
        f"{fixture_name} sd={sd} mode={mode}: bounded != truncated on "
        f"{len(failures)}/{n_bounds} bounds; first at B={failures[0][0]}: {failures[0][1]}"
    )


# ---------------------------------------------------------------------------
# Mechanism tests (Plan A §5.2) — the fixed behaviour at the exact bound. Each
# expectation was verified live against the truncated-twin oracle at the base,
# where the bounded run showed the leak named in the fixture docstring.
# ---------------------------------------------------------------------------

def test_l1_pattern_applying_past_the_bound_is_not_applied():
    raw = _make_l1_pattern_past_bound()
    i = 5
    for B in (i, i + 1):
        assert _events(raw, B) == [], f"B={B}: nothing is knowable before the apply candle"
    est = _of_type(_events(raw, i + 2), "CTS_ESTABLISHED")
    assert len(est) == 1
    assert est[0][2]["confirmed_at"] == i + 2 and est[0][2]["pattern_name"] == "one_maru_opposite"


def test_l2_range_confirmed_past_the_bound_is_not_started():
    raw = _make_l2_range_at_bound()
    i = 5
    evs = _events(raw, i + 2)
    assert _of_type(evs, "RANGE_STARTED") == [], "confirming close at i+3 > B=i+2"
    assert all(e[2].get("to") != "range" for e in _of_type(evs, "STATE_CHANGED"))
    for B in (i + 3, i + 4, i + 5):
        rs = _of_type(_events(raw, B), "RANGE_STARTED")
        assert [e[1] for e in rs] == [i + 3], f"B={B}"
        assert rs[0][2]["start_idx"] == i and rs[0][2]["confirm_idx"] == i + 3


def test_l3_two_candle_pattern_wins_when_continuous_needs_the_future():
    raw = _make_l3_priority_preemption()
    i = 5
    assert _events(raw, i) == []
    est = _of_type(_events(raw, i + 1), "CTS_ESTABLISHED")
    assert len(est) == 1, "an L1-only fix would drop the continuous and establish nothing"
    assert est[0][2]["pattern_name"] == "one_maru_continuous" and est[0][2]["confirmed_at"] == i + 1
    for B in (i + 2, i + 3):
        est = _of_type(_events(raw, B), "CTS_ESTABLISHED")
        assert len(est) == 1
        assert est[0][2]["pattern_name"] == "continuous" and est[0][2]["confirmed_at"] == i + 2


def test_l4_reversal_watch_clamps_at_the_bound():
    """Plan A §2 L4 (option a): the watch window ends at the data edge."""
    raw = _make_l4_watch_at_bound()
    i, k = 4, 2

    def run(B):
        res = compute_bounded_structure(_prepare_df(raw), _START, 1, end_idx=B)
        evs = [(ev.type, int(ev.idx), ev.meta) for ev in res.events]
        return res, evs

    # B = i+k-1: the reversal pattern would apply past the edge -> never scheduled; the
    # anchor fails at once (rv_anchor_failed), no candidate, no reversal.
    res, evs = run(i + k - 1)
    assert res.reversal_idx is None
    assert _of_type(evs, "REVERSAL_CANDIDATE") == []
    ws = _of_type(evs, "REVERSAL_WATCH_START")
    assert [(e[1], e[2]["expires_idx"]) for e in ws] == [(i, i + k - 1)]
    assert any(e[1] == i and e[2]["reason"] == "rv_anchor_failed" for e in _of_type(evs, "BOS_THRESHOLD_UPDATED"))

    # B = i+k: scheduled (apply == expires_idx == B) but discarded as a false break —
    # expiry precedes the pending apply in the per-candle step. No reversal at the edge.
    res, evs = run(i + k)
    assert res.reversal_idx is None
    assert all(e[2].get("to") != "reversal" for e in _of_type(evs, "STATE_CHANGED"))
    cands = _of_type(evs, "REVERSAL_CANDIDATE")
    assert [(e[1], e[2]["apply_idx"], e[2]["expires_idx"]) for e in cands] == [(i, i + k, i + k)]
    assert all(e[1] <= i + k for e in evs)

    # B > i+k: the reversal applies at i+k as before; only the expires_idx meta clamps.
    for B in (i + k + 1, i + 5, i + 6):
        res, evs = run(B)
        assert res.reversal_idx == i + k, f"B={B}"
        ws = _of_type(evs, "REVERSAL_WATCH_START")
        assert [(e[1], e[2]["expires_idx"]) for e in ws] == [(i, min(i + 5, B))], f"B={B}"


def test_l5_bos_inner_is_derived_from_candles_inside_the_bound():
    raw = _make_l5_bos_inner_at_bound()
    bos_idx = 8

    def cycle1_conf(B):
        return [e for e in _of_type(_events(raw, B), "CTS_CONFIRMED") if e[2]["cycle_id"] == 1]

    # bos_idx+2: cycle 1 just established (apply 10 == CTS_1 idx) — nothing can consume the
    # inner yet; bos_idx+3 / +4: the inside bars at 12-13 are past the bound, the resolver's
    # inner is .6015 and candle 11 (low .6039) is not within 8 pips of it -> no confirmation.
    for B in (bos_idx + 2, bos_idx + 3, bos_idx + 4):
        evs = _events(raw, B)
        assert [e for e in _of_type(evs, "CTS_ESTABLISHED") if e[2]["cycle_id"] == 1], f"B={B}"
        assert cycle1_conf(B) == [], f"B={B}: inner must not come from candles > B"
        assert all(e[2].get("reason") != "proximity_created_range" for e in _of_type(evs, "RANGE_STARTED"))

    # bos_idx+5: both inside bars are inside the bound -> ('base inside bar', 8), inner .6033,
    # candle 11 hits -> CTS_CONFIRMED@11 via sd_zone_proximity, as on the full frame.
    for B in (bos_idx + 5, bos_idx + 6):
        conf = cycle1_conf(B)
        assert [(e[1], e[2]["confirmation_method"]) for e in conf] == [(11, "sd_zone_proximity")], f"B={B}"
        assert abs(conf[0][2]["trigger_inner"] - 0.6033) < 1e-9


# ---------------------------------------------------------------------------
# Negative test for the post-run assert: re-create the L2 leak and expect the
# assert (not a silent leak).
# ---------------------------------------------------------------------------

def _raw_label_range_test(self, i):
    """Pre-Plan-A `_is_range_candle_given_confirm`: the raw label, never compared
    to the bound (leak site L2)."""
    if i is None:
        return (False, None)
    if int(self.df.iloc[i].get("is_range", 0)) != 1:
        return (False, None)
    confirm_idx = int(self.df.iloc[i].get("is_range_confirm_idx", -1))
    if confirm_idx < 0:
        return (False, None)
    candle_i = str(self.df.iloc[i].get("candle_type", ""))
    dir_i = int(self.df.iloc[i].get("direction", 0))
    return ((candle_i == "pinbar" and dir_i == self.struct_direction) or (dir_i != self.struct_direction),
            confirm_idx)


@pytest.mark.parametrize("raw, B", [
    (_make_l2_range_at_bound(), 7),      # confirm_idx 8 > B
    (_make_reversing_data(), 15),        # base run stamped RANGE_STARTED@18 at B=15
])
def test_post_run_assert_catches_a_reintroduced_l2_leak(monkeypatch, raw, B):
    """With the L2 gate removed, a range candle whose confirming close is past the
    bound gets `RANGE_STARTED` stamped there -> `run()` must raise, not leak."""
    from engine_v2.structure.market_structure import MarketStructure

    monkeypatch.setattr(MarketStructure, "_is_range_candle_given_confirm", _raw_label_range_test)
    with pytest.raises(AssertionError, match=r"event past effective_end"):
        compute_bounded_structure(_prepare_df(raw), _START, 1, end_idx=B)
