"""PLAN_K landing A — the IC reference condition (item 1's rule experiment RX1), computed and exported, the filter OFF.

The rule is the user's (PLAN_H §8.9 RX1 + §6 RX1-*); PLAN_K §1 reads it, §6 lists these tests. Pinned here:

- the condition (`poi_zones.ic_reference_reason`): the price side strict in both directions, the bound strict, no
  reference → passes, a reference without a known break → the price side only, the price side reported first;
- the cross-limit breakout watch (`patterns.cross_limit_watch.find_cross_limit_break`) on HAND-BUILT candles (the
  detector reads only `structure_patterns._ROW_FIELDS`, so each fixture sets the candle features it needs and asserts
  the detector's answer first): a close past `P` opens a watch, a wick does not; `P` is the break threshold; re-arm;
  the earliest confirmation wins, a tie → the earlier anchor (K-Q1); MS's detection convention (K-Q6 — the cold
  review's constructed anchor-flip case does NOT flip); the detector never reads past `end_idx`;
- the reference (`FibTracker.ic_reference`): `prev_cycle_cts` (the start cycle's CTS_{a−1}, cycle a's establishment
  pattern; a cross from a ≥ 1; a pre-established cross), `prev_structure_bos` (known for the whole run, K-Q7),
  `cross_limit_anchor` (the watch from BOS_0, on `P`, as of the moment, to `end_idx`), the choice (further opposite;
  EQUAL → the structure reference and ITS bound, K-Q5), `T > M` drops the structure reference, no limit entry → no
  structure reference and no raise (§9 C03);
- the anchor behind `P` for the two prev-sub sources not pinned in `test_plan_i_s2_cross_fib_limit.py`;
- the stamp (meta keys before the limit keys; the cap; re-stamped in place), the exports (CSV column, hover lines on
  the H1 and a lens chart), the selection (reported, the pick unchanged; the switch's ON path);
- the plumbing: `ICReference` / `localized`; C01 (an error inside the reference during a sub geometry surfaces as
  `ICSelectionError`); the C04 spies (the geometry's limit is slice-local; the read's moment at both refresh paths);
  D2 with the reference (both consumers hand the same value); the projection round trip; `STATE_CHANGED(to=reversal)`
  `pattern_anchor_idx` on the winner AND the pending path.

Record: PLAN_K §11.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import math

import pandas as pd
import pytest

import engine_v2.structure.event_fields as ef
import engine_v2.structure.structure_engine as se
import engine_v2.zones.poi_zones as poi_zones
from engine_v2.common.types import PatternStatus
from engine_v2.patterns.cross_limit_watch import find_cross_limit_break
from engine_v2.patterns.structure_patterns import BreakoutPatterns
from engine_v2.tests import test_ms_reversal_on_expiry as RV
from engine_v2.tests._cross_fib_limits import NO_P
from engine_v2.tests._event_factory import make_cts_established
from engine_v2.tests.test_ms_reversal_on_expiry import trace  # noqa: F401  (fixture)
from engine_v2.tests.test_plan_i_s3_cross_fib_limit_applied import _CYCLES, _conf, _df, _gap, _m, _q
from engine_v2.tests.test_plan_i_s5_ms_reads_the_fib import embedded  # noqa: F401  (fixture)
from engine_v2.tests.test_render_sub_projection import geometry, m15_df  # noqa: F401  (fixtures)
from engine_v2.zones.cross_cycle_fib import CrossFibLimit, ICReference
from engine_v2.zones.fib_tracker import FibState, FibTracker, FibTrackerConfig
from engine_v2.zones.poi_zones import (
    IC_REF_NOT_BEFORE_BREAK as NOT_BEFORE,
    IC_REF_PAST_REFERENCE as PAST,
    ICSelectionError,
    POIConfig,
    ic_reference_reason,
)

_IC_KEYS = ["ic_ref_price", "ic_ref_source", "ic_ref_break_pattern_anchor_idx", "ic_ref_break_confirmed_idx"]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# =====================================================================================================================
# The condition — `ic_reference_reason`
# =====================================================================================================================

def _hl(*rows):
    return pd.DataFrame({"h": [h for h, _ in rows], "l": [l for _, l in rows]})


@pytest.mark.parametrize("sd", [1, -1])
def test_the_price_side_is_strict_and_reads_the_near_extreme(sd):
    """sd +1: the candle's HIGH strictly below the reference; sd −1: its LOW strictly above. Touching the reference
    fails; the other extreme is never read (the far side sits beyond the reference on purpose)."""
    ref = ICReference(1.2, "prev_cycle_cts")
    if sd == 1:
        df = _hl((1.19999, 0.5), (1.2, 0.5), (1.20001, 0.5))
    else:
        df = _hl((1.9, 1.20001), (1.9, 1.2), (1.9, 1.19999))
    assert [ic_reference_reason(df, i, ref, sd) for i in range(3)] == [None, PAST, PAST]


def test_the_bound_is_strict_and_the_price_side_is_reported_first():
    df = _hl(*[(1.0, 0.9)] * 8)                                  # every candle on the near side of 1.2
    ref = ICReference(1.2, "prev_structure_bos", 5, 7)
    assert [ic_reference_reason(df, i, ref, 1) for i in (3, 4, 5, 6)] == [None, None, NOT_BEFORE, NOT_BEFORE]
    df.loc[6, "h"] = 1.3                                         # both sides fail -> the price side (K-Q2)
    assert ic_reference_reason(df, 6, ref, 1) == PAST


def test_no_reference_passes_and_a_reference_without_a_break_checks_the_price_only():
    df = _hl((1.5, 1.4), (1.0, 0.9))
    assert ic_reference_reason(df, 0, ICReference.NONE, 1) is None          # RX1-B: no reference -> passes
    ref = ICReference(1.2, "cross_limit_anchor")                             # RX1-2a: no known break
    assert ic_reference_reason(df, 1, ref, 1) is None
    assert ic_reference_reason(df, 0, ref, 1) == PAST


# =====================================================================================================================
# The cross-limit breakout watch — hand-built candles (sd +1, P = 1.10 unless stated)
# =====================================================================================================================

P = 1.10
_FLAGS = ("is_big_normal_as0", "is_big_normal_as1", "is_big_normal_as2", "is_big_maru_as0", "is_big_maru_as1")


def _k(o, h, l, c, ctype, **flags):
    """One candle with the features the detector reads (`_ROW_FIELDS`); direction from the body."""
    assert set(flags) <= set(_FLAGS), flags
    row = dict(o=o, h=h, l=l, c=c, candle_type=ctype, direction=(c > o) - (c < o),
               candle_len=round(h - l, 6), body_len=round(abs(c - o), 6), mid_price=(h + l) / 2)
    row.update({f: int(flags.get(f, 0)) for f in _FLAGS})
    return row


_LOW = _k(1.050, 1.060, 1.040, 1.045, "pinbar")                 # bearish filler, closes far below P
# A double_maru SUCCESS (anchor = the first, applies at the second): big c0, c1 closing higher and >= 0.7 x c0's length;
# c1's low under c0's mid so no one_maru_continuous rides along.
_DM0 = _k(1.090, 1.120, 1.085, 1.115, "maru", is_big_normal_as0=1)
_DM1 = _k(1.115, 1.140, 1.100, 1.135, "maru")


def _frame(*rows):
    return pd.DataFrame(list(rows))


def _detect(df, a, thr=P, end_idx=None):
    return BreakoutPatterns(df, end_idx=end_idx).detect_best_for_anchor(a, 1, thr)


def test_a_close_past_p_opens_a_watch_anchored_at_that_candle():
    df = _frame(_LOW, _LOW, _DM0, _DM1, _LOW, _LOW)
    ev = _detect(df, 2)
    assert (ev.name, ev.status, ev.start_idx, ev.end_idx) == ("double_maru", PatternStatus.SUCCESS, 2, 3)
    assert find_cross_limit_break(df, P, 1, 0) == (2, 3)
    assert find_cross_limit_break(_frame(_LOW, _LOW, _LOW, _LOW), P, 1, 0) is None          # never -> None


def test_a_wick_past_p_does_not_open_a_watch():
    """Candle 1's HIGH is past P, its close is not: no watch — although the pair is a valid pattern against a threshold
    below that close (1.09). Candle 2 closes past P but has no pattern (its next candle is a bearish filler)."""
    wick = _k(1.060, 1.120, 1.055, 1.095, "maru", is_big_normal_as0=1)
    df = _frame(_LOW, wick, _k(1.095, 1.150, 1.090, 1.125, "maru"), _LOW, _LOW)
    assert _detect(df, 1, 1.09).name == "double_maru" and _detect(df, 2) is None
    assert find_cross_limit_break(df, P, 1, 0) is None
    assert find_cross_limit_break(df, 1.09, 1, 0) == (1, 2)


def test_p_is_the_break_threshold_not_the_anchor_behind_it():
    """A pattern closing between the anchor behind P (1.08) and P (1.10) does not break P (sub 7's 4178, PLAN_H RX1)."""
    df = _frame(_LOW, _k(1.070, 1.100, 1.065, 1.095, "maru", is_big_normal_as0=1),
                _k(1.095, 1.120, 1.075, 1.105, "maru"), _LOW, _LOW)
    assert _detect(df, 1, 1.08).name == "double_maru"
    assert find_cross_limit_break(df, 1.08, 1, 0) == (1, 2)
    assert find_cross_limit_break(df, P, 1, 0) is None


def test_every_later_close_past_p_re_arms():
    """W3: P never moves — a close past it with no pattern (1) does not end the watching; a later one (4) breaks."""
    df = _frame(_LOW, _k(1.090, 1.120, 1.085, 1.115, "maru"), _LOW, _LOW, _DM0, _DM1, _LOW)
    assert _detect(df, 1) is None
    assert find_cross_limit_break(df, P, 1, 0) == (4, 5)


def _earliest_frame():
    """Two concurrent watches: anchor 1's double_maru needs confirmation (c0 not big) and confirms only at 6 (candle 6
    closes past max(h1, h2) = 1.20); anchor 2's double_maru SUCCEEDS at 3. The earliest confirmation wins (K-Q1) —
    not the first anchor whose pattern confirms."""
    return _frame(
        _LOW,
        _k(1.090, 1.120, 1.085, 1.115, "maru"),                       # 1
        _k(1.115, 1.200, 1.100, 1.130, "maru", is_big_normal_as0=1),  # 2
        _k(1.130, 1.196, 1.125, 1.150, "maru"),                       # 3 (closes under 1.20: no confirmation of 1)
        _LOW, _LOW,                                                   # 4, 5
        _k(1.150, 1.215, 1.140, 1.210, "maru"),                       # 6 confirms anchor 1
        _LOW,
    )


def test_the_earliest_confirmation_wins_over_the_first_anchor():
    df = _earliest_frame()
    e1, e2 = _detect(df, 1), _detect(df, 2)
    assert (e1.name, e1.status, e1.confirmation_idx) == ("double_maru", PatternStatus.CONFIRMED, 6)
    assert (e2.name, e2.status, e2.end_idx) == ("double_maru", PatternStatus.SUCCESS, 3)
    assert find_cross_limit_break(df, P, 1, 0) == (2, 3)
    # flip-free: the as-of answer only ever adds the break
    assert [find_cross_limit_break(df, P, 1, 0, known_at=t) for t in (2, 3, 6, None)] == [None] + [(2, 3)] * 3


def _flip_frame():
    """The cold review's constructed case (PLAN_K §9 C02 / K-Q6), a = 1:
    a    double_maru needing confirmation (c0 not big) -> confirmed at a+3;
    a+1  big maru: on the FULL frame `continuous` (maru, pinbar, maru) SUCCEEDS at a+3 — checked first; on a frame cut
         at a+2 it cannot be seen and `one_maru_continuous` SUCCEEDS at a+2;
    a+2  bull pinbar (closes past P, no pattern of its own); a+3 the continuous' big maru; a+4 a big bearish maru."""
    return _frame(
        _LOW,
        _k(1.090, 1.120, 1.085, 1.115, "maru"),                                              # 1 = a
        _k(1.115, 1.140, 1.100, 1.135, "maru", is_big_normal_as0=1, is_big_maru_as0=1),      # 2
        _k(1.135, 1.145, 1.130, 1.138, "pinbar"),                                            # 3
        _k(1.138, 1.170, 1.136, 1.165, "maru", is_big_normal_as2=1),                         # 4
        _k(1.165, 1.166, 1.120, 1.122, "maru", is_big_maru_as1=1),                           # 5
        _LOW, _LOW,
    )


def test_the_watch_detects_by_ms_convention_and_the_anchor_flip_does_not_flip():
    df = _flip_frame()
    a = 1
    e0, e1 = _detect(df, a), _detect(df, a + 1)
    assert (e0.name, e0.status, e0.confirmation_idx) == ("double_maru", PatternStatus.CONFIRMED, a + 3)
    assert (e1.name, e1.status, e1.end_idx) == ("continuous", PatternStatus.SUCCESS, a + 3)
    cut = _detect(df, a + 1, end_idx=a + 2)
    assert (cut.name, cut.status, cut.end_idx) == ("one_maru_continuous", PatternStatus.SUCCESS, a + 2)
    assert _detect(df, a, end_idx=a + 2) is None
    # MS's convention (K-Q6): the full frame, a break counting once confirmed — none as of a+2 ...
    assert find_cross_limit_break(df, P, 1, 0, known_at=a + 2) is None
    # ... and from a+3 on the final answer: the tie at a+3 goes to the earlier anchor (K-Q1)
    assert [find_cross_limit_break(df, P, 1, 0, known_at=t) for t in (a + 3, a + 6, None)] == [(a, a + 3)] * 3
    # the rejected reading (a frame CUT at the moment) would say a+1 as of a+2, then flip to a
    assert find_cross_limit_break(df, P, 1, 0, end_idx=a + 2) == (a + 1, a + 2)


def test_the_break_threshold_reaches_the_detector():
    """`P` is the detector's break threshold, not only the close gate: candle 1 closes past P with < 30 % of its body
    beyond it, so its one_maru_continuous fails `body_check` and needs a confirmation that never comes -> no break;
    against no threshold the same pair SUCCEEDS."""
    df = _frame(_LOW, _k(1.050, 1.104, 1.048, 1.102, "maru", is_big_maru_as0=1),
                _k(1.100, 1.106, 1.095, 1.101, "pinbar"), _LOW, _LOW, _LOW, _LOW, _LOW)
    assert _detect(df, 1) is None
    ev = BreakoutPatterns(df).detect_best_for_anchor(1, 1, None)
    assert (ev.name, ev.status, ev.end_idx) == ("one_maru_continuous", PatternStatus.SUCCESS, 2)
    assert find_cross_limit_break(df, P, 1, 0) is None


def test_the_detector_never_reads_past_end_idx():
    df = _frame(_LOW, _LOW, _DM0, _DM1, _LOW)
    assert find_cross_limit_break(df, P, 1, 0, end_idx=2) is None          # the pair's second candle is past the end
    assert find_cross_limit_break(df, P, 1, 0, end_idx=3) == (2, 3)
    assert find_cross_limit_break(df, P, 1, 0, end_idx=1) is None
    assert find_cross_limit_break(df, P, 1, 3) is None                      # start after the anchor
    assert find_cross_limit_break(df, P, 1, 9) is None                      # start past the frame


# =====================================================================================================================
# The reference — `FibTracker.ic_reference` (the s3 flat frame; sd −1 = the price mirror at 3.0)
#   cycle 0: BOS 10, CTS 1.15 (anchor = moment 20, confirmed 25); cycle 1: BOS 30, CTS 1.35 (40, confirmed 45);
#   cycle 2: BOS 50, CTS 1.55 (60, open). Each cycle's establishment PATTERN anchor (`_PAT`) differs from its CTS anchor.
# =====================================================================================================================

_PAT = {0: 18, 1: 38, 2: 58}


def _tracker_with(lim, sd, upto=2, df=None):
    t = FibTracker(FibTrackerConfig(fill_threshold=0.70), cross_fib_limits={0: lim})
    if df is None:
        df = _df(80, [_gap(g[0], sd, g[1], g[2]) for *_, g in _CYCLES])
    for c, (bos, bos_p, cts, cts_p, conf, _g) in enumerate(_CYCLES[:upto + 1]):
        est = make_cts_established(cts_anchor_idx=cts, confirmed_at=cts, price=_m(cts_p, sd), structure_id=0,
                                   cycle_id=c, struct_direction=sd, pattern_anchor_idx=_PAT[c])
        _q(t.on_cts_established, est, df, bos_idx=bos, bos_price=_m(bos_p, sd))
        if conf is not None and c < upto:
            _q(t.on_cts_confirmed, _conf(conf, cts_p, c, sd, cts))
    return t, df


def _probe(cyc, sd, start=None):
    """A fib record of cycle `cyc` — a single, or a cross from `start` (`ic_reference` reads only the identity and
    the cross keys)."""
    meta = {} if start is None else {"cross_cycle": True, "cross_start_cycle": start}
    return FibState(structure_id=0, cycle_id=cyc, struct_direction=sd, bos_idx=0, bos_price=0.0, cts_idx=0,
                    cts_price=0.0, meta=meta)


def _rev(p, anchor, sd, brk=(7, 9)):
    return CrossFibLimit(_m(p, sd), "reversal_bos_frozen", ref_anchor_price=_m(anchor, sd),
                         break_pattern_anchor_idx=brk[0], break_confirmed_idx=brk[1])


@pytest.mark.parametrize("sd", [1, -1])
def test_a_fib_from_cycle_a_takes_cts_a_minus_1_and_cycle_a_s_establishment_pattern(sd):
    t, df = _tracker_with(NO_P, sd)
    ref = lambda f, at=None: t.ic_reference(f, df, evaluated_at=at)  # noqa: E731
    assert ref(t._fibs[(0, 0)]) == ICReference.NONE                                         # cycle 0, no P
    assert ref(t._fibs[(0, 1)]) == ICReference(_m(1.15, sd), "prev_cycle_cts", 38, 40)       # the real single
    assert ref(t._fibs[(0, 2)]) == ICReference(_m(1.35, sd), "prev_cycle_cts", 58, 60)
    # a cross 1 -> 2: its START cycle's reference (CTS_0, cycle 1's pattern), not the target's
    assert ref(_probe(2, sd, start=1)) == ICReference(_m(1.15, sd), "prev_cycle_cts", 38, 40)
    # as of a moment: the break is known from cycle a's establishment moment (inclusive)
    assert ref(_probe(2, sd), 59) == ICReference(_m(1.35, sd), "prev_cycle_cts")
    assert ref(_probe(2, sd), 60) == ICReference(_m(1.35, sd), "prev_cycle_cts", 58, 60)


@pytest.mark.parametrize("sd", [1, -1])
def test_a_pre_established_cross_reads_only_its_start_cycle(sd):
    """A cross 1 -> 2 before cycle 2 is established (`a < T`, the pre-established phase): nothing of cycle 2 is read."""
    t, df = _tracker_with(NO_P, sd, upto=1)
    assert (0, 2) not in t._est_pattern_anchor
    assert t.ic_reference(_probe(2, sd, start=1), df, evaluated_at=47) == ICReference(
        _m(1.15, sd), "prev_cycle_cts", 38, 40)


@pytest.mark.parametrize("sd", [1, -1])
def test_a_reversal_starts_reference_and_its_bound_hold_for_the_whole_run(sd):
    """`prev_structure_bos`: the anchor behind P and the realised reversal pattern — known from the run's START
    (K-Q7), unlike the watch's break."""
    t, df = _tracker_with(_rev(9.0, 1.05, sd), sd, upto=0)
    want = ICReference(_m(1.05, sd), "prev_structure_bos", 7, 9)
    assert [t.ic_reference(t._fibs[(0, 0)], df, evaluated_at=at) for at in (None, 3, 50)] == [want] * 3


@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("anchor, want", [
    (1.10, "structure"),      # further opposite (sd +1 lower) -> the structure's
    (1.20, "cts"),            # the previous CTS lies further opposite (RX1-3, the gap case)
    (1.15, "structure"),      # EQUAL prices -> the STRUCTURE reference and ITS bound (K-Q5)
])
def test_both_references_take_the_further_opposite_and_equal_takes_the_structure(sd, anchor, want):
    t, df = _tracker_with(_rev(9.0, anchor, sd), sd)                    # P never reached: M None
    got = t.ic_reference(_probe(2, sd, start=1), df, evaluated_at=None)
    if want == "structure":
        assert got == ICReference(_m(anchor, sd), "prev_structure_bos", 7, 9)
    else:
        assert got == ICReference(_m(1.15, sd), "prev_cycle_cts", 38, 40)


@pytest.mark.parametrize("sd", [1, -1])
def test_the_structure_reference_applies_while_t_is_at_most_m(sd):
    """P = 1.30: CTS_1 (1.35) is the first anchor beyond -> M = 1. Cycle 1 (T = M) keeps the structure reference
    (1.00, further opposite than CTS_0 1.15); cycle 2 (T > M) drops it -> CTS_1."""
    t, df = _tracker_with(_rev(1.30, 1.00, sd), sd)
    assert t.cross_fib_limit_cycle(0) == 1
    assert t.ic_reference(_probe(1, sd), df, evaluated_at=None) == ICReference(_m(1.00, sd), "prev_structure_bos", 7, 9)
    assert t.ic_reference(t._fibs[(0, 2)], df, evaluated_at=None) == ICReference(
        _m(1.35, sd), "prev_cycle_cts", 58, 60)
    # `T` is the TARGET cycle, not the cross's start: a cross 1 -> 2 drops it too (unreachable in production — a cross
    # never targets past `M`, and `M` cannot move below an existing target — pinned as the rule's definition)
    assert t.ic_reference(_probe(2, sd, start=1), df, evaluated_at=None) == ICReference(
        _m(1.15, sd), "prev_cycle_cts", 38, 40)


@pytest.mark.parametrize("sd", [1, -1])
def test_no_limit_entry_means_no_structure_reference_and_no_raise(sd):
    """§9 C03: the reference evaluates `M`'s test WITHOUT `_cross_allowed` (which raises without an entry)."""
    t, df = _tracker_with(NO_P, sd)
    t._limits.clear()
    assert t.ic_reference(t._fibs[(0, 0)], df, evaluated_at=None) == ICReference.NONE
    assert t.ic_reference(t._fibs[(0, 1)], df, evaluated_at=None) == ICReference(_m(1.15, sd), "prev_cycle_cts", 38, 40)


# --- the watch inside the reference: a non-reversal start (anchor 1.05 behind P 1.12) on the s3 frame -------------

def _mk(o, h, l, c, ctype, sd, **flags):
    if sd == -1:
        o, h, l, c = _m(o, -1), _m(l, -1), _m(h, -1), _m(c, -1)
    return _k(o, h, l, c, ctype, **flags)


def _watch_frame(sd, *, pair_at=(12,), between=False):
    """The s3 frame + candle features: a double_maru breaking P = 1.12 at each `pair_at` (anchor, anchor + 1) — or,
    `between`, one closing between the anchor 1.05 and P. Every other candle has no type (no pattern)."""
    df = _df(80, [_gap(g[0], sd, g[1], g[2]) for *_, g in _CYCLES])
    for col, v in (("candle_type", ""), ("candle_len", 0.002), ("body_len", 0.0), ("mid_price", 1.5)):
        df[col] = v
    for f in _FLAGS:
        df[f] = 0
    if between:
        pair = (_mk(1.080, 1.105, 1.075, 1.100, "maru", sd, is_big_normal_as0=1),
                _mk(1.100, 1.118, 1.090, 1.115, "maru", sd))
    else:
        pair = (_mk(1.110, 1.140, 1.105, 1.135, "maru", sd, is_big_normal_as0=1),
                _mk(1.135, 1.160, 1.120, 1.155, "maru", sd))
    for a in pair_at:
        for j, row in enumerate(pair):
            for col, v in row.items():
                df.loc[a + j, col] = v
    return df


def _xlim(sd):
    return CrossFibLimit(_m(1.12, sd), "parent_cts_threshold", ref_anchor_price=_m(1.05, sd))


@pytest.mark.parametrize("sd", [1, -1])
def test_a_cross_limit_anchor_reference_breaks_where_the_watch_finds_p_broken(sd):
    df = _watch_frame(sd)
    t, _ = _tracker_with(_xlim(sd), sd, upto=0, df=df)
    f0 = t._fibs[(0, 0)]
    ref = lambda **kw: t.ic_reference(f0, df, **kw)  # noqa: E731
    assert ref(evaluated_at=None) == ICReference(_m(1.05, sd), "cross_limit_anchor", 12, 13)
    assert ref(evaluated_at=12) == ICReference(_m(1.05, sd), "cross_limit_anchor")          # not confirmed yet
    assert ref(evaluated_at=13) == ref(evaluated_at=None)


def _flip_watch_frame(sd):
    """The s3 frame + `_flip_frame`'s candles 1–5 at 12–16 (a = 12; P = 1.10): on the WHOLE frame anchor 12's
    double_maru confirms at 15 and anchor 13's `continuous` succeeds at 15; on a frame CUT at 14, anchor 13's
    `one_maru_continuous` succeeds at 14."""
    df = _watch_frame(sd, pair_at=())
    flip = _flip_frame()
    for j in range(1, 6):
        r = flip.iloc[j]
        flags = {f: 1 for f in _FLAGS if int(r[f])}
        for col, v in _mk(*(float(r[k]) for k in ("o", "h", "l", "c")), str(r["candle_type"]), sd, **flags).items():
            df.loc[11 + j, col] = v
    return df


@pytest.mark.parametrize("sd", [1, -1])
def test_the_final_reference_is_the_as_of_reference_at_the_cap_on_the_whole_frame(sd):
    """PLAN_K §11.2 LA1 (RX1-K6): the FINAL reference (the stamp, the drawn selection) = the reference as of the
    structure's lifecycle cap with the watch's detector on the WHOLE frame — MS's convention — never a frame cut at
    the cap. Cap 14: the cut frame would export (13, 14); as of 14 on the whole frame nothing has confirmed."""
    df = _flip_watch_frame(sd)
    p = _m(P, sd)
    assert find_cross_limit_break(df, p, sd, 10, end_idx=14) == (13, 14)        # the rejected reading (the fixture)
    assert find_cross_limit_break(df, p, sd, 10, known_at=14) is None
    assert find_cross_limit_break(df, p, sd, 10) == (12, 15)                    # the tie at 15 -> the earlier anchor
    t, _ = _tracker_with(CrossFibLimit(p, "parent_cts_threshold", ref_anchor_price=_m(1.05, sd)), sd, upto=0, df=df)
    t.stamp_ic_references(df, lifecycle_cap=14)
    m = t._fibs[(0, 0)].meta
    assert (m["ic_ref_price"], m["ic_ref_source"]) == (_m(1.05, sd), "cross_limit_anchor")
    assert "ic_ref_break_pattern_anchor_idx" not in m and "ic_ref_break_confirmed_idx" not in m
    t.stamp_ic_references(df, lifecycle_cap=15)
    m = t._fibs[(0, 0)].meta
    assert (m["ic_ref_break_pattern_anchor_idx"], m["ic_ref_break_confirmed_idx"]) == (12, 15)


@pytest.mark.parametrize("sd", [1, -1])
def test_the_watch_starts_at_the_structures_bos_0_anchor(sd):
    """From BOS_0 (10): a pattern at 3-4 (before the structure) is never the break; the one at 12-13 (before the
    cycle's establishment moment 20) is — the sub 4 shape (its break 3588 lies before its lifecycle start 3621)."""
    df = _watch_frame(sd, pair_at=(3, 12))
    t, _ = _tracker_with(_xlim(sd), sd, upto=0, df=df)
    assert t._bos_by_cycle[(0, 0)][0] == 10
    assert t.ic_reference(t._fibs[(0, 0)], df, evaluated_at=None).break_pattern_anchor_idx == 12


@pytest.mark.parametrize("sd", [1, -1])
def test_the_watch_breaks_p_not_the_anchor_behind_it(sd):
    df = _watch_frame(sd, between=True)
    t, _ = _tracker_with(_xlim(sd), sd, upto=0, df=df)
    assert find_cross_limit_break(df, _m(1.05, sd), sd, 10) == (12, 13)                  # it would break the anchor
    assert t.ic_reference(t._fibs[(0, 0)], df, evaluated_at=None) == ICReference(_m(1.05, sd), "cross_limit_anchor")


# =====================================================================================================================
# The anchor behind P — the two prev-sub sources `test_plan_i_s2_cross_fib_limit.py` pins price-only
# =====================================================================================================================

def test_the_anchor_behind_a_prev_sub_cts_threshold_is_its_confirmed_price():
    """`prev_sub_cts_threshold`: the confirmed cycle's CTS_CONFIRMED price, never its moved threshold; a later
    confirmed cycle replaces both. `prev_sub_latest_cts`: P itself (already an anchor price)."""
    import engine_v2.multitf.cross_fib_limit as cfl
    from engine_v2.tests.test_plan_i_s2_cross_fib_limit import _SAME_DIR_EVENTS, _cts, _geo_sub
    prev = _geo_sub(5, -1, 100, _SAME_DIR_EVENTS, slice_begin=100, start_idx=105)
    got = {s: cfl.read_prev_sub(prev, s, -1, log=lambda m: None) for s in (118, 117, 114, 111)}
    assert {s: (g.price, g.source, g.ref_anchor_price) for s, g in got.items()} == {
        118: (0.5785, "prev_sub_cts_threshold", 0.579),     # the threshold moved; the anchor did not
        117: (0.579, "prev_sub_cts_threshold", 0.579),
        114: (0.579, "prev_sub_latest_cts", 0.579),
        111: (0.580, "prev_sub_latest_cts", 0.580),
    }
    two = _geo_sub(5, -1, 100, [
        _cts("CTS_ESTABLISHED", 8, 0.580, 0, moment=10), _cts("CTS_CONFIRMED", 15, 0.579, 0, anchor=8),
        _cts("CTS_THRESHOLD_UPDATED", 18, 0.5785, 0), _cts("CTS_ESTABLISHED", 19, 0.577, 1, moment=20),
        _cts("CTS_CONFIRMED", 25, 0.576, 1, anchor=19), _cts("CTS_THRESHOLD_UPDATED", 28, 0.5765, 1),
    ], slice_begin=100, start_idx=105)
    assert [(g.price, g.ref_anchor_price) for g in (cfl.read_prev_sub(two, s, -1) for s in (124, 126, 130))] == [
        (0.5785, 0.579), (0.576, 0.576), (0.5765, 0.576)]


# =====================================================================================================================
# The stamp, the exports, the selection
# =====================================================================================================================

@pytest.mark.parametrize("sd", [1, -1])
def test_the_stamp_writes_the_final_reference_and_re_stamps_in_place(sd):
    df = _watch_frame(sd)
    t, _ = _tracker_with(_xlim(sd), sd, upto=0, df=df)
    t.stamp_ic_references(df)
    m = t._fibs[(0, 0)].meta
    assert {k: m[k] for k in _IC_KEYS} == dict(zip(_IC_KEYS, (_m(1.05, sd), "cross_limit_anchor", 12, 13)))
    t.stamp_ic_references(df, lifecycle_cap=12)                 # the structure ended before the break confirmed
    m = t._fibs[(0, 0)].meta
    assert (m["ic_ref_price"], m["ic_ref_source"]) == (_m(1.05, sd), "cross_limit_anchor")
    assert "ic_ref_break_pattern_anchor_idx" not in m and "ic_ref_break_confirmed_idx" not in m
    t2, df2 = _tracker_with(NO_P, sd, upto=0)
    t2.stamp_ic_references(df2)
    assert not set(_IC_KEYS) & set(t2._fibs[(0, 0)].meta)       # no reference -> no key


def _multicycle_pipeline():
    from engine_v2.tests.test_plan_h_landing_b import _pipeline
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data
    return _pipeline(_make_ip_multicycle_data)


def test_a_pipeline_record_carries_its_reference_before_the_limit_keys():
    """The H1-main-style run (sid 0: no P): cycle 0 has no reference; cycle k >= 1 references CTS_{k−1}'s
    CTS_CONFIRMED price, bounded by cycle k's CTS_ESTABLISHED pattern — the 4 keys right before the limit keys (the 4
    `cross_fib_limit_*` + main's `started_by`, pinned last by `test_plan_i_s2_cross_fib_limit`)."""
    res, out, _ = _multicycle_pipeline()
    conf = {e.meta["cycle_id"]: e for e in res.events if e.type == "CTS_CONFIRMED"}
    est = {e.meta["cycle_id"]: e for e in res.events if e.type == "CTS_ESTABLISHED"}
    limit_keys = ["cross_fib_limit_price", "cross_fib_limit_source", "cross_fib_limit_cycle",
                  "cross_fib_limit_from_sub_id", "started_by"]
    assert all(list(f.meta)[-len(limit_keys):] == limit_keys for f in out["fib_states"])
    seen = 0
    for f in out["fib_states"]:
        keys = list(f.meta)
        if f.cycle_id == 0:
            assert not set(_IC_KEYS) & set(keys)
            continue
        seen += 1
        assert keys[-len(limit_keys) - 4:-len(limit_keys)] == _IC_KEYS
        k = f.cycle_id
        assert [f.meta[x] for x in _IC_KEYS] == [float(conf[k - 1].price), "prev_cycle_cts",
                                                 ef.pattern_anchor_idx(est[k]), ef.event_moment(est[k])]
    assert seen


def test_the_reason_is_reported_and_a_cut_candidate_is_never_picked():
    """`test_plan_h_landing_b`'s M1 frame: candidates 32 (2A.i, the pick without a reference; high 1.13) and 36 (no
    pattern; high 1.125). A reference at 1.128 cuts 32 only: reported, and since landing B (the filter on) 32 is no
    IC candidate and 36 has no pattern -> no pick; 32 is still classified for the CSV."""
    from engine_v2.tests.test_plan_h_landing_b import _fib, _m1_frame
    df = _m1_frame()
    fs = poi_zones.FibState(structure_id=0, cycle_id=1, struct_direction=1, bos_idx=30, bos_price=1.0, cts_idx=40,
                            cts_price=1.35, active=True, locked=False, fib=_fib(1.35, 1.0))
    ref = ICReference(1.128, "prev_cycle_cts")
    sel = poi_zones.select_ic(df, fs, POIConfig(), ic_reference=ref, classify_all=True)
    assert {c.ic_idx: c.ic_ref_reason for c in sel.checks} == {32: PAST, 36: None}
    assert sel.pick is None and next(c for c in sel.checks if c.ic_idx == 32).pattern.rank is not None
    assert poi_zones.select_ic(df, fs, POIConfig(), ic_reference=ref).pick is None
    assert poi_zones.select_ic(df, fs, POIConfig(), ic_reference=ICReference.NONE).pick.ic_idx == 32


def test_the_csv_carries_the_reason_after_the_touch(tmp_path):
    from engine_v2.debug.export_zones import IC_CANDIDATE_COLUMNS, export_ic_candidates
    from engine_v2.tests.test_fib_chart_render import _ip
    i = IC_CANDIDATE_COLUMNS.index("touches_zone")
    assert IC_CANDIDATE_COLUMNS[i + 1] == "ic_ref_reason"
    recs = [dataclasses.replace(_ip(ic=14, rank="2A.i"), ic_ref_reason=PAST), _ip(reason="no_pattern")]
    out = tmp_path / "ip.csv"
    export_ic_candidates(recs, out)
    got = pd.read_csv(out)
    assert list(got.columns) == IC_CANDIDATE_COLUMNS
    assert got["ic_ref_reason"].tolist()[0] == PAST and pd.isna(got["ic_ref_reason"].tolist()[1])   # IP: none row


def test_every_exported_reason_is_the_condition_on_the_records_reference(monkeypatch):
    """`derive_ic_candidates` hands each candidate's own outcome through (A5) — recomputed from the tracker's final
    reference for its record; the `IP: none` rows carry none. The fixture's real references cut nothing, so the
    builder is replaced by one cutting every candidate of the odd cycles (sd +1: no high lies below −inf)."""
    def by_cycle(self, f, df, *, evaluated_at):
        return ICReference(-math.inf, "prev_cycle_cts") if f.cycle_id % 2 else ICReference.NONE

    monkeypatch.setattr(FibTracker, "ic_reference", by_cycle)
    res, out, _ = _multicycle_pipeline()
    fibs = {(f.structure_id, f.cycle_id, f.bos_idx, f.cts_idx): f for f in out["fib_states"]}
    rows = [r for r in out["ic_candidates"] if r.ic_idx is not None]
    assert {r.ic_ref_reason for r in rows} == {PAST, None}
    for r in rows:
        f = fibs[(r.structure_id, r.cycle_id, r.bos_anchor_idx, r.cts_anchor_idx)]
        assert r.ic_ref_reason == ic_reference_reason(res.df, r.ic_idx, by_cycle(None, f, res.df, evaluated_at=None),
                                                      r.struct_direction)
    assert all(r.ic_ref_reason is None for r in out["ic_candidates"] if r.ic_idx is None)


def test_the_h1_chart_renders_the_ic_reference_line(tmp_path, monkeypatch):
    from engine_v2.tests.test_fib_chart_render import _render_h1
    res, out, _ = _multicycle_pipeline()
    df = res.df.copy()
    df.attrs.update(structure_events=res.events, fib_states=out["fib_states"], poi_zones=out["poi_zones"],
                    ic_candidates=out["ic_candidates"], kl_zones=list(out["kl_zones"]))
    fig = _render_h1(df, monkeypatch, tmp_path)
    cells = {tr.name: tr.customdata[0][8].split("<br>") for tr in fig.data if (tr.name or "").startswith("fib:")}
    conf0 = next(e for e in res.events if e.type == "CTS_CONFIRMED" and e.meta["cycle_id"] == 0)
    est1 = next(e for e in res.events if e.type == "CTS_ESTABLISHED" and e.meta["cycle_id"] == 1)
    assert cells["fib:sid0:c0"][-2] == "IC ref: none"
    assert cells["fib:sid0:c1"][-2] == (f"IC ref {float(conf0.price):.5f} (prev_cycle_cts, cycle 0) · "
                                       f"break {ef.pattern_anchor_idx(est1)}")
    assert cells["fib:sid0:c1"][-1].startswith("limit")


def test_the_hover_names_the_cross_start_cycles_previous_cts():
    from types import SimpleNamespace
    from engine_v2.charting._zone_render import fib_hover_ic_ref_text
    meta = {"cross_cycle": True, "cross_start_cycle": 1, "ic_ref_price": 0.6, "ic_ref_source": "prev_cycle_cts",
            "ic_ref_break_pattern_anchor_idx": 38, "ic_ref_break_confirmed_idx": 40, "cross_fib_limit_source": "none"}
    assert fib_hover_ic_ref_text(SimpleNamespace(meta=meta, cycle_id=2)) == (
        "IC ref 0.60000 (prev_cycle_cts, cycle 0) · break 38")          # a cross 1 -> 2: CTS_0, not CTS_1


def test_every_retrospective_reference_reads_to_the_lifecycle_cap(monkeypatch):
    """A3 / §3.4: the stamp, the drawn selection and the CSV's selection (+ both `ic_records=None` fallbacks) build the
    FINAL reference as of the structure's lifecycle cap (`evaluated_at=lifecycle_cap`, PLAN_K §11.2 LA1) — the drawn
    pick and the exported meta are one value. (Behaviour at a cap:
    `test_the_final_reference_is_the_as_of_reference_at_the_cap_on_the_whole_frame`.)"""
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.tests._cross_fib_limits import main_limits
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data, _prepare_df
    res = _quiet(se.compute_bounded_structure, _prepare_df(_make_ip_multicycle_data()), 0, 1, cross_fib_limit=NO_P)
    ends = []
    real = FibTracker.ic_reference

    def spy(self, f, df, *, evaluated_at):
        ends.append(evaluated_at)                                      # no MS run below: final reads only
        return real(self, f, df, evaluated_at=evaluated_at)

    monkeypatch.setattr(FibTracker, "ic_reference", spy)
    cap = len(res.df) - 2
    out = _quiet(_run_downstream_pipeline, res.df, res.events, 1, wvmi="off",
                 cross_fib_limits=main_limits(res.events), lifecycle_cap=cap)
    assert ends and set(ends) == {cap}
    ends.clear()
    t = out["fib_tracker"]
    _quiet(poi_zones.derive_poi_zones, res.df, res.events, t, POIConfig(), lifecycle_cap=cap)
    _quiet(poi_zones.derive_ic_candidates, res.df, res.events, t, POIConfig(), structure_path_id="H1.main",
           lifecycle_cap=cap)
    assert ends and set(ends) == {cap}


def test_a_lens_chart_renders_a_reference_with_no_known_break(tmp_path, monkeypatch):
    from engine_v2.tests import test_fib_chart_render as FCR
    rec = FCR._M15_FIBS["s6c1"]
    meta = {**rec.meta, "ic_ref_price": 0.601, "ic_ref_source": "cross_limit_anchor",
            "cross_fib_limit_price": 0.6012, "cross_fib_limit_source": "prev_sub_bos_threshold",
            "cross_fib_limit_cycle": None, "cross_fib_limit_from_sub_id": 5}
    monkeypatch.setitem(FCR._M15_FIBS, "s6c1", dataclasses.replace(rec, meta=meta))
    fig = FCR._render_m15(monkeypatch, tmp_path)
    cell, = [tr.customdata[0][12] for tr in fig.data if tr.name == "M15 fib" and tr.customdata[0][0] == 6]
    assert cell.split("<br>")[-2:] == ["IC ref 0.60100 (cross_limit_anchor) · break: none",
                                       "limit 0.60120 · prev_sub_bos_threshold (sub 5) · M=none"]


# =====================================================================================================================
# The plumbing
# =====================================================================================================================

def test_ic_reference_validates_its_fields():
    assert ICReference.NONE == ICReference(None, None)
    ICReference(0.57, "cross_limit_anchor")
    ICReference(0.57, "prev_structure_bos", 5, 6)
    for kw in (dict(price=0.57, source=None), dict(price=None, source="prev_cycle_cts"),
               dict(price=0.57, source="nonsense"),
               dict(price=0.57, source="prev_cycle_cts", break_pattern_anchor_idx=5),
               dict(price=None, source=None, break_pattern_anchor_idx=5, break_confirmed_idx=6)):
        with pytest.raises(AssertionError):
            ICReference(**kw)


def test_localized_moves_only_the_break_indices():
    lim = CrossFibLimit(0.58156, "reversal_bos_frozen", from_pool_idx=2, ref_anchor_price=0.58198,
                        break_pattern_anchor_idx=137, break_confirmed_idx=140)
    assert lim.localized(100) == dataclasses.replace(lim, break_pattern_anchor_idx=37, break_confirmed_idx=40)
    plain = CrossFibLimit(0.58105, "prev_sub_bos_threshold", from_pool_idx=4, ref_anchor_price=0.58105)
    assert plain.localized(100) is plain


def test_the_reversal_event_key_reads_only_on_a_reversal():
    from engine_v2.structure.market_structure import StructureEvent
    rev = StructureEvent(idx=9, category="STRUCTURE", type="STATE_CHANGED", price=None,
                         meta={"from": "range", "to": "reversal", "pattern_anchor_idx": 7})
    assert ef.pattern_anchor_idx(rev) == 7
    other = StructureEvent(idx=9, category="STRUCTURE", type="STATE_CHANGED", price=None,
                           meta={"from": "range", "to": "breakout", "pattern_anchor_idx": 7})
    with pytest.raises(ValueError):
        ef.pattern_anchor_idx(other)


def test_an_error_inside_the_reference_during_a_sub_geometry_surfaces(monkeypatch):
    """§9 C01: MS builds the reference INSIDE `poi_inner_for_fib`'s re-raising scope — a ValueError there arrives as
    `ICSelectionError` (a RuntimeError) instead of reaching `_build_geometry`'s `except (ValueError, IndexError)`,
    which would drop the whole sub."""
    from engine_v2.tests.test_plan_i_s5_ms_reads_the_fib import _geometry, _reversing
    df, inner = _reversing()
    calls = []
    real = FibTracker.ic_reference
    monkeypatch.setattr(FibTracker, "ic_reference", lambda self, *a, **k: calls.append(1) or real(self, *a, **k))
    assert _geometry(df, inner) is not None and calls, "the fixture's geometry reads a fib's reference"

    def boom(self, *a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(FibTracker, "ic_reference", boom)
    with pytest.raises(ICSelectionError) as exc:
        _geometry(df, inner)
    assert isinstance(exc.value.__cause__, ValueError)


def test_the_geometry_hands_its_tracker_the_slice_local_limit(embedded):  # noqa: F811
    """§3.5 / §9 C04: the pool's limit is entity-absolute; the geometry MS runs on the slice [start − 50, cap] — its
    embedded tracker holds the break moved by `slice_begin` (60 − 50 = 10)."""
    from engine_v2.multitf import entity_df_mutation as edm
    from engine_v2.tests.test_render_sub_projection import _make_flat_prefix, _make_reversing_data, _prepare_df
    df = _prepare_df(_make_flat_prefix(60) + _make_reversing_data())
    lim = CrossFibLimit(math.inf, "reversal_bos_frozen", ref_anchor_price=math.inf, break_pattern_anchor_idx=100,
                        break_confirmed_idx=105)
    _quiet(edm._build_geometry, df, sd=1, start_abs=60, run_cap_abs=len(df) - 1, bos0_inner=None,
           timeframe="M15", cross_fib_limit=lim)
    t, = embedded
    assert t._limits == {0: dataclasses.replace(lim, break_pattern_anchor_idx=90, break_confirmed_idx=95)}


@pytest.mark.parametrize("fixture", ["regress", "anchor_before_moment"])
def test_the_read_takes_the_fed_events_moment_on_both_paths(monkeypatch, fixture):
    """§9 C04 / C06: every in-flight read hands the tracker and the moment of the event just fed — the pattern path's
    apply (CTS_ESTABLISHED / a pattern CTS_UPDATED) and the raw path's re-stepped candle (a raw CTS_UPDATED) — and the
    read builds the fib's reference AT that moment. `anchor_before_moment`: cycle 1's CTS anchor 9 precedes its
    moment 10 (`test_unified_probe._make_second_cts_moment_after_anchor_data`) — the anchor is not the moment."""
    from engine_v2.structure.event_fields import CTS_UPDATED_RAW_VIA
    from engine_v2.tests import test_ms_cts_update_no_regress as NR
    from engine_v2.tests.test_unified_probe import _make_second_cts_moment_after_anchor_data, _prepare_df
    seen, refs = [], []
    real, real_ref = se.poi_inner_for_fib, FibTracker.ic_reference

    def spy(df, fib, ft, **kw):
        last = kw["tracker"]._handled[-1]
        anchor = ef.cts_anchor_idx(last) if last.type == "CTS_ESTABLISHED" else None
        seen.append((kw["evaluated_at"], ef.event_moment(last), last.type, last.meta.get("via"), anchor, fib))
        return real(df, fib, ft, **kw)

    def ref_spy(self, f, df, *, evaluated_at):
        refs.append(evaluated_at)
        return real_ref(self, f, df, evaluated_at=evaluated_at)

    monkeypatch.setattr(se, "poi_inner_for_fib", spy)
    monkeypatch.setattr(FibTracker, "ic_reference", ref_spy)
    if fixture == "regress":
        NR._run(NR._regress_rows(), 1)
    else:
        _quiet(se.compute_bounded_structure, _prepare_df(_make_second_cts_moment_after_anchor_data()), 0, 1,
               cross_fib_limit=NO_P)
    assert seen and all(at == moment for at, moment, *_ in seen)
    assert refs == [at for at, *_, fib in seen if fib is not None] and refs
    if fixture == "regress":
        kinds = {(typ, via == CTS_UPDATED_RAW_VIA) for _at, _mo, typ, via, _a, _f in seen}
        assert ("CTS_ESTABLISHED", False) in kinds and ("CTS_UPDATED", True) in kinds
    else:
        assert (10, 10, "CTS_ESTABLISHED", "one_maru_opposite", 9) in [s[:5] for s in seen]


def test_both_consumers_hand_the_same_reference_on_the_final_fib(monkeypatch):
    """D2 with the reference (PLAN_K §6): the drawn record's selection (`classify_all=True`) and MS's last read on
    the same anchors receive the same `ICReference` — one builder, two moments, one value on the final fib."""
    got = []
    real = poi_zones.select_ic

    def spy(df, fs, cfg, **kw):
        got.append(((fs.structure_id, fs.cycle_id, fs.bos_idx, fs.cts_idx), kw.get("classify_all", False),
                    kw["ic_reference"]))
        return real(df, fs, cfg, **kw)

    monkeypatch.setattr(poi_zones, "select_ic", spy)
    _multicycle_pipeline()
    drawn = [(k, r) for k, cm, r in got if cm]
    compared = 0
    for key, ref in drawn:
        ms = [r for k, cm, r in got if not cm and k == key]
        if ms:
            assert ms[-1] == ref, key
            compared += ref.price is not None
    assert compared == 3            # cycles 1-3 (prev_cycle_cts); cycle 0 has none


def test_the_projection_localizes_the_limit_and_the_mirror_restores_the_break(geometry, m15_df):  # noqa: F811
    """§3.5 round trip (slice_begin 5): the sub's entity-absolute break 58 / 59 is slice-local 53 / 54 in the
    projection's records and entity-absolute again in the lens df's mirrored records."""
    from engine_v2.tests.test_render_sub_projection import _START, _lens_dfs, _render, _set_lifecycle, _two_lens_records
    pool, sub = geometry
    sub.cross_fib_limit = CrossFibLimit(math.inf, "reversal_bos_frozen", ref_anchor_price=0.6, from_pool_idx=0,
                                        from_sub_id=3, break_pattern_anchor_idx=58, break_confirmed_idx=59)
    _two_lens_records(pool, sub, m15_df)
    _set_lifecycle(sub, _START, None, None)
    lens_dfs = _lens_dfs(m15_df)
    res = _render(sub, m15_df, lens_dfs)
    local = [f for f in res.fib_states if f.cycle_id == 0]
    assert local and {(f.meta["ic_ref_source"], f.meta["ic_ref_break_pattern_anchor_idx"],
                       f.meta["ic_ref_break_confirmed_idx"]) for f in local} == {("prev_structure_bos", 53, 54)}
    for lens_df in lens_dfs.values():
        mirrored = [f for f in lens_df.attrs["fib_states"] if f.cycle_id == 0]
        assert mirrored and {(f.meta["ic_ref_break_pattern_anchor_idx"], f.meta["ic_ref_break_confirmed_idx"])
                             for f in mirrored} == {(58, 59)}


# =====================================================================================================================
# The reversal event's realised pattern (A13) — the winner AND the pending path
# =====================================================================================================================

@pytest.mark.parametrize("sd", [1, -1])
@pytest.mark.parametrize("rows, path, anchor", [
    (RV._p1_rows, "winner", 9),            # the close-break candle's own winner
    (RV._p2_rows, "winner", 8),            # a later anchor's winner on the expiry
    (RV._later_winner_rows, "winner", 7),  # a later anchor's winner BEFORE the pending (no candidate at the apply)
    (RV._p3_rows, "pending", 4),           # the pending reversal (the watch's candidate)
])
def test_the_reversal_event_carries_its_realised_patterns_first_candle(sd, trace, rows, path, anchor):  # noqa: F811
    res = RV._run(RV._rows(rows, sd), sd)
    rev, = [e for e in res.events if e.type == "STATE_CHANGED" and e.meta["to"] == "reversal"]
    assert [p for p, *_ in trace["applied"]] == [path]
    assert rev.meta["pattern_anchor_idx"] == anchor == ef.pattern_anchor_idx(rev)
    if path == "pending":
        cand, = [e for e in res.events if e.type == "REVERSAL_CANDIDATE" and e.meta["apply_idx"] == int(rev.idx)]
        assert cand.meta["pattern_anchor_idx"] == anchor
