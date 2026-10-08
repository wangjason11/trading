"""Week 9 Part 1 item 1, landing A — the institutional-pattern (IP) IC selector
(`plans/PLAN_H_institutional_pattern_poi.md` §1 the reading, §6 the decisions,
§8 the plan). Landing A computes and EXPORTS the selection; nothing draws it yet.

Each test pins ONE rule and names the mutant it kills. Structural rules run on
`leg_variants` over a bare agreed-direction array (U = sd, D = opp, Z = a true
doji, S = a split candle); ranking rules on synthetic candle frames classified by
the real `apply_candle_classification` (H1 params); selection rules through
`select_ip_ic` with a hand-built fib and hand-built imbalance instances.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io

import numpy as np
import pandas as pd
import pytest

from engine_v2.tests._cross_fib_limits import main_limits

from engine_v2.common.types import ImbalanceInstance
from engine_v2.features.candle_classifier import (
    apply_candle_classification,
    candle_params_for_timeframe,
)
from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import (
    AGREED_SPLIT,
    agreed_direction,
    big_ratio_flags,
    compute_candle_features,
)
from engine_v2.features.fibonacci import DEFAULT_FIB_LEVELS, create_fib_retracement
from engine_v2.patterns.imbalance import has_unfilled_imbalance, iter_unfilled_imbalances
from engine_v2.patterns.institutional_pattern import (
    MULTI_OPP_RANK,
    SINGLE_OPP_RANK,
    classify_ics,
    classify_ip,
    leg_variants,
)
from engine_v2.zones.fib_tracker import FibState
from engine_v2.zones.poi_zones import (
    IPCandidateRecord,
    POIConfig,
    derive_ip_candidates,
    fib_zone_bounds,
    find_ic_candidates,
    ic_touches_zone,
    select_ic_variants,
    select_ip_ic,
    select_ip_records,
)

U, D, Z, S = 1, -1, 0, AGREED_SPLIT
PIP = 0.0001
H1 = candle_params_for_timeframe("H1")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

def _variants(seq, ic, *, sd=1, bos=0, cts=None, cap=3):
    a = np.array(seq)
    return leg_variants(a, ic, sd, bos_anchor_idx=bos, cts_anchor_idx=len(seq) - 1 if cts is None else cts, cap=cap)


_SHAPE = {"M": (0.9, 0.05, 0.05), "N": (0.5, 0.25, 0.25), "P": (0.2, 0.7, 0.1)}


def _ohlc(o, kind, d, r_pips):
    """A candle opening at `o`: M maru (body 90 % of the range), N normal (50 %),
    P pinbar (20 %, long open-side wick), Z true doji (o == c)."""
    r = r_pips * PIP
    if kind == "Z":
        return o, o + r / 2, o - r / 2, o
    body, open_wick, close_wick = _SHAPE[kind]
    c = o + d * body * r
    if d == 1:
        return o, c + close_wick * r, o - open_wick * r, c
    return o, o + open_wick * r, c - close_wick * r, c


_PREFIX = tuple(("M", U if k % 2 == 0 else D, 10) for k in range(6))


def _frame(specs, *, prefix=_PREFIX, start=0.6000, timeframe="H1"):
    """Candles from specs (kind, dir, range pips[, gap pips]) or ("raw", o, h, l, c); each opens at the previous
    close + gap, so a gapless candle's volume direction equals its candle direction. `prefix` = the default 6
    marus of 10 pips (U/D alternating, last D) → the big-ratio reference at the pattern is 10 pips."""
    rows, c_prev = [], start
    for sp in list(prefix) + list(specs):
        if sp[0] == "raw":
            o, h, l, c = sp[1:]
        else:
            kind, d, r = sp[:3]
            gap = sp[3] if len(sp) > 3 else 0.0
            o, h, l, c = _ohlc(c_prev + gap * PIP, kind, d, r)
        rows.append({"o": o, "h": h, "l": l, "c": c})
        c_prev = c
    df = pd.DataFrame(rows)
    df.insert(0, "time", pd.date_range("2026-01-05", periods=len(df), freq="h", tz="UTC"))
    df["volume"] = 1.0
    df.attrs["pair"] = "NZD_USD"
    return apply_candle_classification(df, timeframe=timeframe).df


N_PREFIX = 6


def _ip(specs, ic, *, prefix=None, bos=N_PREFIX, cts=None, params=H1, sd=1, cap=3):
    """The IP of the IC at pattern position `ic` (0 = the first spec)."""
    df = _frame(specs) if prefix is None else _frame(specs, prefix=prefix)
    n_pre = len(df) - len(specs)
    agreed = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
    return classify_ip(
        df, n_pre + ic, sd, bos_anchor_idx=(n_pre if bos == N_PREFIX else bos),
        cts_anchor_idx=len(df) - 1 if cts is None else cts, agreed=agreed, cap=cap, params=params,
    ), n_pre


def _fib_over(df, bos, cts, zone_lo, zone_hi):
    """A buy fib (sd +1) on [bos, cts] whose 61.8–80 zone is exactly [zone_lo, zone_hi]."""
    r = (zone_hi - zone_lo) / 0.182
    cts_price = zone_hi + 0.618 * r
    bos_price = cts_price - r
    fib = create_fib_retracement(anchor_high=cts_price, anchor_low=bos_price, direction=1, levels=DEFAULT_FIB_LEVELS)
    return FibState(structure_id=0, cycle_id=0, struct_direction=1, bos_idx=bos, bos_price=bos_price, cts_idx=cts,
                    cts_price=cts_price, active=True, locked=False, fib=fib)


def _far_gap(start, end=None):
    """A buy imbalance instance far below every price — never filled."""
    return ImbalanceInstance(start_idx=start, end_idx=start if end is None else end, direction=1,
                             gap_top=0.5000, gap_bottom=0.4990, gap_size=0.0010)


def _filled_gap(df, start, fill_idx):
    """A buy instance at `start` that candle `fill_idx` two-stroke-fills (its low arms stroke 1, its close ≥ the
    gap top is stroke 2)."""
    top = float(df.loc[fill_idx, "c"])
    bottom = float(df.loc[fill_idx, "l"]) - 0.00001
    inst = ImbalanceInstance(start_idx=start, end_idx=start, direction=1, gap_top=top, gap_bottom=bottom,
                             gap_size=top - bottom)
    assert inst.is_filled(df, fill_idx, 0.70)
    return inst


def _select(df, bos, cts, imbalances, *, zone=(0.5, 0.7), **kw):
    df.attrs["imbalances"] = imbalances
    return select_ip_ic(df, _fib_over(df, bos, cts, *zone), POIConfig(), **kw)


# ---------------------------------------------------------------------------
# the agreed direction (V1a) + the frame's params
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("direction, vol_dir, agreed", [
    (1, 1, 1), (1, 0, 1), (-1, -1, -1), (-1, 0, -1),       # agreed / one of them 0
    (1, -1, AGREED_SPLIT), (-1, 1, AGREED_SPLIT),          # opposite → a split candle
    (0, 1, 1), (0, -1, -1),                                # a volume-only doji
    (0, 0, 0),                                             # a true doji
])
def test_agreed_direction_table(direction, vol_dir, agreed):
    assert int(agreed_direction(direction, vol_dir)) == agreed
    assert list(agreed_direction(np.array([direction] * 3), np.array([vol_dir] * 3))) == [agreed] * 3


def test_qualified_rule_reads_the_agreed_direction_unchanged():
    """`wave_candles._is_qualified` now reads `agreed_direction` — equal to the old formula on the whole grid
    (incl. a frame without `vol_dir`; byte-identity of wave candles / WVMI on the window: landing A's `/compare`)."""
    from engine_v2.zones.wave_candles import _is_qualified
    grid = [(d, v) for d in (-1, 0, 1) for v in (-1, 0, 1)]
    df = pd.DataFrame({"direction": [d for d, _ in grid], "vol_dir": [v for _, v in grid]})
    no_vol = df[["direction"]]
    for i, (d, v) in enumerate(grid):
        for req in (-1, 0, 1):
            assert _is_qualified(df, i, req) == (d == req and v in (req, 0)), (d, v, req)
            assert _is_qualified(no_vol, i, req) == (d == req), (d, req)


def test_candle_params_for_timeframe_is_the_one_floor_table():
    assert [candle_params_for_timeframe(tf).pinbar_body_pip_floor for tf in ("H1", "M15", "M5", "D1")] == \
        [2.2, 1.0, 0.5, 0.0]


def test_classification_stamps_the_frame_timeframe_and_it_survives_slicing():
    df = _frame([("M", U, 10)])
    assert df.attrs["timeframe"] == "H1"
    m15 = _frame([("M", U, 10)], timeframe="M15")
    assert m15.attrs["timeframe"] == "M15"
    # the sub-slice shape (iloc → reset_index → compute_imbalance) and a lens copy keep it
    from engine_v2.patterns.imbalance import compute_imbalance
    sub = compute_imbalance(m15.iloc[2:].reset_index(drop=True))
    assert sub.attrs["timeframe"] == "M15" and m15.copy().attrs["timeframe"] == "M15"


def test_prepare_lower_tf_data_stamps_m15():
    from engine_v2.multitf.data_bridge import prepare_lower_tf_data
    raw = _frame([("M", U, 10)])[["time", "o", "h", "l", "c", "volume"]].copy()
    raw.attrs = {"pair": "NZD_USD"}
    assert prepare_lower_tf_data(raw).attrs["timeframe"] == "M15"


def test_select_ip_ic_fails_loudly_on_an_unstamped_frame():
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10)])
    del df.attrs["timeframe"]
    with pytest.raises(KeyError):
        _select(df, 6, 8, [_far_gap(8)])


def test_unknown_timeframe_cap_raises():
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10)])
    df.attrs["timeframe"] = "M5"        # no M5 cap until the M5 lens (PLAN_H §8.1)
    with pytest.raises(KeyError):
        _select(df, 6, 8, [_far_gap(8)])


def test_big_ratio_flags_is_classify_big_flags_per_candle():
    df = _frame([("M", U, 7), ("N", D, 4), ("M", U, 12), ("P", D, 3), ("M", U, 5)])
    for i in range(len(df)):
        got = big_ratio_flags(float(df.loc[i, "candle_len"]), float(df.loc[i, "prior_maru_max_len_as0"]), H1)
        if got is None:
            assert df.loc[i, "big_ratio_as0"] == 0.0 and not df.loc[i, "is_big_maru_as0"]
            continue
        assert (got[0], bool(got[1]), bool(got[2])) == (
            df.loc[i, "big_ratio_as0"], bool(df.loc[i, "is_big_maru_as0"]), bool(df.loc[i, "is_big_normal_as0"]))


def test_big_ratio_is_rounded_before_the_thresholds():
    assert big_ratio_flags(0.000649, 0.001, H1) == (0.65, True, True)      # 0.649 → 0.65 ≥ 0.65 (big maru)
    assert big_ratio_flags(0.000644, 0.001, H1) == (0.64, False, True)
    assert big_ratio_flags(0.0005, 0.0, H1) is None                        # no prior maru → no flag


def test_iter_unfilled_imbalances_is_has_unfilled_imbalance():
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 10)])
    insts = [_far_gap(7), _filled_gap(df, 8, 10), _far_gap(9, 10),
             ImbalanceInstance(start_idx=8, end_idx=8, direction=-1, gap_top=0.70, gap_bottom=0.69, gap_size=0.01)]
    df.attrs["imbalances"] = insts
    seen = 0
    for lo in range(6, 11):
        for hi in range(lo, 11):
            for direction in (None, 1, -1):
                for at in (None, 8, 10):
                    got = list(iter_unfilled_imbalances(df, lo, hi, 10, 0.70, direction=direction, evaluated_at=at))
                    assert has_unfilled_imbalance(df, lo, hi, 10, 0.70, direction=direction,
                                                  evaluated_at=at) == bool(got)
                    seen += bool(got)
    assert seen > 10
    assert list(iter_unfilled_imbalances(df, 7, 10, 10, direction=1, evaluated_at=None)) == [insts[0], insts[2]]


# ---------------------------------------------------------------------------
# legs (structure only — `leg_variants`)
# ---------------------------------------------------------------------------

def test_ic_must_be_the_last_opp_candle():
    assert _variants([U, D, D, U], 1)[3] == "ic_not_last_opp"
    assert _variants([U, D, Z, D, U], 1)[3] == "ic_not_last_opp"            # a doji then opp
    assert _variants([U, D, D, U], 2)[0]                                     # the last one is


def test_no_continuation_after_a_split_or_only_dojis_to_the_cts():
    assert _variants([U, D, S, U], 1)[3] == "no_continuation"
    assert _variants([U, D, Z, Z], 1)[3] == "no_continuation"


def test_a_split_candle_breaks_the_initial_run():
    variants = _variants([U, S, U, D, U], 3)[0]
    assert {v.initial for v in variants} == {(2,)}


def test_doji_after_the_ic_joins_the_continuation_and_makes_it_multi():
    (v,) = _variants([U, D, Z, U, D], 1)[0]
    assert v.continuation == (2, 3)


def test_caps_initial_most_recent_continuation_first():
    variants, _, confirmed, _ = _variants([U, U, U, U, D, U, U, U, U], 4)
    main = [v for v in variants if not v.only_3d]
    assert [v.initial for v in main] == [(1, 2, 3)] and main[0].continuation == (5, 6, 7)
    assert confirmed == 7                                   # the run reached the cap: its last candle


@pytest.mark.parametrize("seq, cts, cont, confirmed", [
    ([U, D, U, U, D], None, (2, 3), 4),                     # shorter than the cap → the next candle
    ([U, D, U, U, U], 3, (2, 3), 4),                        # cut at the CTS anchor (a member) → the next candle
    ([U, D, U, Z, D], None, (2,), 4),                       # E1-trimmed end doji, below the cap → after the doji
    ([U, D, U, U, Z, U], None, (2, 3), 4),                  # E1 at the cap (3464–3466): the doji closes the leg
])
def test_continuation_cut_trim_and_pattern_confirmed_idx(seq, cts, cont, confirmed):
    variants, _, got_confirmed, _ = _variants(seq, 1, cts=cts)
    assert {v.continuation for v in variants} == {cont}
    assert got_confirmed == confirmed


def test_a_true_doji_inside_the_initial_run_never_breaks_it():
    assert [v.initial for v in _variants([U, Z, U, D, U], 3)[0] if not v.only_3d] == [(0, 1, 2)]


def test_initial_cut_at_the_bos_anchor_which_is_a_member():
    variants = _variants([U, U, U, D, U], 3, bos=1)[0]
    assert [v.initial for v in variants if not v.only_3d] == [(1, 2)]
    assert _variants([U, D, D, U], 2, bos=1)[3] == "initial_before_bos_anchor"


def test_no_initial_candle_before_the_opp_run():
    assert _variants([S, D, U], 1)[3] == "no_initial_candle"
    assert _variants([D, U], 0)[3] == "no_initial_candle"


def test_e1_trims_a_doji_off_the_initial_start():
    assert {v.initial for v in _variants([Z, U, D, U], 2)[0]} == {(1,)}
    # ↑ ↑ ⊖ ↑ ↑ · opp → ↑ ↑ (the cap window [⊖ ↑ ↑], then the trim — E1-confirm)
    assert [v.initial for v in _variants([U, U, Z, U, U, D, U], 5)[0] if not v.only_3d] == [(3, 4)]


def test_a_continuation_of_only_dojis_within_the_cap_fails():
    assert _variants([U, D, Z, Z, Z, U], 1)[3] == "continuation_no_sd_candle"


def test_the_opp_run_is_never_cut_and_an_inner_doji_stays_in_it():
    for v in _variants([U, D, D, D, U], 3)[0]:
        assert v.opp == (1, 2, 3)
    for v in _variants([U, D, Z, D, U], 3)[0]:
        assert v.opp == (1, 2, 3)                           # P6c: the doji makes the opp leg multi


def test_junction_doji_is_tried_in_each_leg_and_never_a_3d_initial():
    variants, rule_ab, _, _ = _variants([U, Z, D, U], 2)
    assert sorted((v.placement, v.initial, v.opp, v.only_3d) for v in variants) == [
        (0, (0,), (1, 2), False), (1, (0, 1), (2,), False)]       # no cap-1 (1,) — a doji
    assert rule_ab == 1


def test_junction_at_the_cap_slides_the_initial_window():
    variants, rule_ab, _, _ = _variants([U, U, U, Z, D, U], 4)
    assert {v.initial for v in variants if v.placement == 1 and not v.only_3d} == {(1, 2, 3)}
    assert rule_ab == 0


def test_two_junction_dojis_give_every_split_and_the_initial_keeps_an_sd_candle():
    variants = _variants([U, Z, Z, D, U], 3)[0]
    assert sorted({(v.placement, v.opp) for v in variants if not v.only_3d}) == [
        (0, (1, 2, 3)), (1, (2, 3)), (2, (3,))]
    # 1 initial candle + 3 junction dojis at cap 3: placing all 3 leaves no sd candle in the window
    assert {v.placement for v in _variants([U, Z, Z, Z, D, U], 4)[0]} == {0, 1, 2}


def test_d3c_cap1_initial_is_offered_next_to_the_cap3():
    variants = _variants([U, U, D, U], 2)[0]
    assert sorted((v.initial, v.only_3d) for v in variants) == [((0, 1), False), ((1,), True)]


# ---------------------------------------------------------------------------
# ranks (candle frames)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("specs, ic, rank", [
    ([("M", U, 10), ("M", D, 6), ("M", U, 10)], 1, "1A.i"),
    ([("M", U, 5), ("M", D, 4), ("M", U, 10)], 1, "1B.i"),       # a maru initial, not big (0.5 < 0.65)
    ([("M", U, 10), ("N", D, 6), ("M", U, 10)], 1, "2A.i"),
    ([("N", U, 10), ("M", D, 6), ("M", U, 10)], 1, "2B.i"),
    ([("N", U, 10), ("N", D, 6), ("M", U, 10)], 1, "2C.i"),
    ([("M", U, 10), ("P", D, 6), ("M", U, 10)], 1, "3A.i"),
    ([("N", U, 10), ("P", D, 6), ("M", U, 10)], 1, "3B.i"),
    ([("M", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 10)], 2, "3C.i"),
    ([("N", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 10)], 2, "3D.i"),
    ([("N", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 6), ("M", U, 6)], 2, "3D.ii"),
    ([("M", U, 6), ("M", U, 6), ("M", D, 6), ("M", U, 10)], 2, "1A.ii"),
    ([("M", U, 10), ("M", D, 6), ("M", U, 6), ("M", U, 6)], 1, "1A.iii"),
    ([("M", U, 6), ("M", U, 6), ("M", D, 6), ("M", U, 6), ("M", U, 6)], 2, "1A.iv"),
    ([("M", U, 10), ("M", D, 6), ("Z", 0, 2), ("M", U, 10)], 1, "1A.iii"),   # a doji after the IC: multi
    ([("M", U, 10), ("P", D, 3), ("P", D, 3), ("M", U, 10)], 2, "3C.i"),     # P8a: a pinbar-type opp stack
])
def test_rank_table(specs, ic, rank):
    res, _ = _ip(specs, ic)
    assert res.ip_rank == rank, (res.reason, res.detail)


def test_rank_table_constants_cover_every_listed_rank():
    assert sorted(set(SINGLE_OPP_RANK.values()) | set(MULTI_OPP_RANK.values())) == [
        "1A", "1B", "2A", "2B", "2C", "3A", "3B", "3C", "3D"]


@pytest.mark.parametrize("specs, ic, reason", [
    ([("P", U, 10), ("M", D, 6), ("M", U, 10)], 1, "initial_fails"),            # N2: a big pinbar is not big_normal
    ([("N", U, 4.8), ("M", D, 6), ("M", U, 10)], 1, "initial_fails"),          # a normal under 0.5: unlisted
    ([("M", U, 10), ("M", D, 6), ("N", U, 20)], 1, "continuation_not_big_maru"),  # N2: big normal ≠ big maru
    ([("M", U, 10), ("M", D, 6), ("M", U, 5)], 1, "continuation_not_big_maru"),   # a maru under 0.65
    ([("P", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 10)], 2, "initial_fails_multi_opp"),
])
def test_no_pattern(specs, ic, reason):
    res, _ = _ip(specs, ic)
    assert res.ip_rank is None and res.reason == reason


def test_continuation_body_must_be_strictly_greater_float_exact():
    """Equal bodies (|0.60040 − 0.60100| on both legs, exact in float) fail; one tenth of a pip more passes."""
    ic = ("raw", 0.60100, 0.60105, 0.60035, 0.60040)
    equal, _ = _ip([("M", U, 10), ic, ("raw", 0.60040, 0.60105, 0.60035, 0.60100)], 1)
    assert equal.reason == "continuation_body_not_greater"
    more, _ = _ip([("M", U, 10), ic, ("raw", 0.60040, 0.60106, 0.60035, 0.60101)], 1)
    assert more.ip_rank == "1A.i"


def test_stacks_use_the_frame_pip_floor():
    """Two tiny up marus stack to a 90 % body of 2.16 pips — under H1's 2.2-pip floor → a pinbar (not big_maru).
    Without the floor the same stack is a big maru and the pattern passes (the floor is what fails it)."""
    small = tuple(("M", U if k % 2 == 0 else D, 2.5) for k in range(6))
    specs = [("M", U, 2.5), ("P", D, 1), ("M", U, 1.2), ("M", U, 1.2)]
    res, _ = _ip(specs, 1, prefix=small)
    assert res.reason == "continuation_not_big_maru" and res.detail["continuation_type"] == "pinbar"
    no_floor, _ = _ip(specs, 1, prefix=small, params=CandleParams())
    assert no_floor.ip_rank == "3A.iii"


def test_big_ratio_reference_is_at_the_pattern_anchor():
    """A 20-pip initial maru: the continuation (8 pips) is big against the 10-pip marus before the PATTERN's
    first candle (0.8), not against a reference that includes the initial leg (0.4)."""
    res, n = _ip([("M", U, 20), ("N", D, 6), ("M", U, 8)], 1)
    assert res.ip_rank == "2A.i" and res.pattern_anchor_idx == n


def test_d3c_reference_moves_with_the_cap1_anchor():
    """The cap-3 initial (5, 6) is measured against the 20-pip maru (ratio 0.4 → the continuation fails); the D3c
    cap-1 initial (6,) against the five 5-pip marus before IT → a big normal and a big continuation → 3D.i."""
    prefix = (("M", D, 20), ("M", U, 5), ("M", D, 5), ("M", U, 5), ("M", D, 5))
    res, n = _ip([("M", U, 5), ("N", U, 5), ("M", D, 3), ("M", D, 3), ("M", U, 8)], 3, prefix=prefix)
    assert res.ip_rank == "3D.i" and res.pattern_anchor_idx == n + 1


def test_d3c_3c_beats_3d_and_3d_when_the_stack_fails():
    both, n = _ip([("M", U, 10), ("N", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 10)], 3)
    assert both.ip_rank == "3C.ii" and both.pattern_anchor_idx == n
    only_3d, n = _ip([("P", U, 10), ("N", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 10)], 3)
    assert only_3d.ip_rank == "3D.i" and only_3d.pattern_anchor_idx == n + 1


def test_cap1_only_for_3d_a_2a_ii_stays_ii():
    res, n = _ip([("M", U, 6), ("M", U, 6), ("N", D, 6), ("M", U, 10)], 2)
    assert res.ip_rank == "2A.ii" and res.ip_leg_candle_counts == "2-1-1"


def test_junction_best_placement_wins_over_rule_ab():
    """At the cap (rule A/B → the doji to opp = 3C.ii) the doji in the initial leg slides the window and gives
    1A.ii — the better pattern wins (J1)."""
    res, n = _ip([("M", U, 6), ("M", U, 6), ("M", U, 6), ("Z", 0, 2), ("M", D, 6), ("M", U, 10)], 4)
    assert res.ip_rank == "1A.ii" and res.ip_leg_candle_counts == "3-1-1" and res.pattern_anchor_idx == n + 1


@pytest.mark.parametrize("specs, ic, counts", [
    # below the cap: rule A/B gives the doji to the initial leg
    ([("M", U, 6), ("M", U, 6), ("Z", 0, 2), ("M", D, 3), ("M", D, 3), ("M", U, 10)], 4, "3-2-1"),
    # at the cap: to the opp leg
    ([("M", U, 6), ("M", U, 6), ("M", U, 6), ("Z", 0, 2), ("M", D, 3), ("M", D, 3), ("M", U, 10)], 5, "3-3-1"),
])
def test_full_tie_rule_ab(specs, ic, counts):
    res, _ = _ip(specs, ic)
    assert res.ip_rank == "3C.ii" and res.ip_leg_candle_counts == counts


def test_classify_ics_batches_and_keeps_order():
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10), ("N", D, 6), ("M", U, 10), ("P", D, 6), ("M", U, 10)])
    agreed = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
    got = classify_ics(df, [11, 7, 9], 1, bos_anchor_idx=6, cts_anchor_idx=12, agreed=agreed, cap=3, params=H1)
    assert [(r.ic_idx, r.ip_rank) for r in got] == [(11, "3A.i"), (7, "1A.i"), (9, "2A.i")]


# ---------------------------------------------------------------------------
# selection (select_ip_ic)
# ---------------------------------------------------------------------------

def test_fib_zone_bounds_is_the_fib_price_formula():
    df = _frame([("M", U, 10)])
    fib = _fib_over(df, 0, 6, 0.6, 0.61)
    lo, hi = fib_zone_bounds(fib.fib.anchor_high, fib.fib.anchor_low, 1, POIConfig())
    assert (lo, hi) == (min(fib.fib.price_at_pct(61.8), fib.fib.price_at_pct(80.0)),
                        max(fib.fib.price_at_pct(61.8), fib.fib.price_at_pct(80.0)))
    assert lo == pytest.approx(0.6) and hi == pytest.approx(0.61)


@pytest.mark.parametrize("h, l, touches", [
    (0.60, 0.59, True),      # h == zone_lo
    (0.62, 0.61, True),      # l == zone_hi
    (0.5999, 0.59, False),
    (0.62, 0.6101, False),
    (0.605, 0.604, True),
])
def test_touch_is_inclusive_at_both_edges(h, l, touches):
    assert ic_touches_zone(h, l, 0.60, 0.61) is touches


def test_best_rank_wins_over_three_candidates():
    df = _frame([("M", U, 10), ("P", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 10), ("N", D, 6), ("M", U, 10)])
    sel = _select(df, 6, 12, [_far_gap(12)])
    assert [(c.ip.ic_idx, c.ip.ip_rank) for c in sel.candidates] == [(7, "3A.i"), (9, "1A.i"), (11, "2A.i")]
    assert sel.pick.ic_idx == 9 and sel.tiebreak is None


def test_shape_breaks_a_rank_tie_before_any_tiebreaker():
    """1A.i (the OLDEST) beats 1A.iii and 1A.ii: the order is the rank letter, then the shape — a selector that
    compared the letter only would tie all three and TB2 would take the most recent."""
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 6), ("M", U, 6), ("M", D, 6),
                 ("M", U, 10)])
    sel = _select(df, 6, 13, [_far_gap(13)])
    assert [(c.ip.ic_idx, c.ip.ip_rank) for c in sel.candidates] == [(7, "1A.i"), (9, "1A.iii"), (12, "1A.ii")]
    assert sel.pick.ic_idx == 7 and sel.tiebreak is None


def test_touch_filters_the_pick_and_misses_are_listed_only_on_request():
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10), ("N", D, 6), ("M", U, 10)])
    lo = float(df.loc[7, "h"]) + 0.00001             # a zone touching IC 9 (2A) but not IC 7 (1A, lower)
    hi = float(df.loc[9, "h"])
    assert float(df.loc[9, "l"]) <= hi
    sel = _select(df, 6, 10, [_far_gap(10)], zone=(lo, hi))
    assert [c.ip.ic_idx for c in sel.candidates] == [9] and sel.pick.ic_idx == 9
    full = _select(df, 6, 10, [_far_gap(10)], zone=(lo, hi), classify_misses=True)
    assert [(c.ip.ic_idx, c.touches_zone, c.ip.ip_rank) for c in full.candidates] == [
        (7, False, "1A.i"), (9, True, "2A.i")]
    assert full.pick.ic_idx == 9


def test_volume_only_doji_is_a_candidate_and_a_split_candle_is_not():
    # 7: a volume-only doji (gaps 3 pips down, o == c); 9: a split candle (a down candle closing above 8's close)
    df = _frame([("M", U, 10), ("Z", 0, 2, -3), ("M", U, 10), ("M", D, 3, 5), ("M", U, 10)])
    assert (int(df.loc[7, "direction"]), int(df.loc[7, "vol_dir"])) == (0, -1)
    assert (int(df.loc[9, "direction"]), int(df.loc[9, "vol_dir"])) == (-1, 1)
    df.attrs["imbalances"] = [_far_gap(10)]
    fib = _fib_over(df, 6, 10, 0.5, 0.7)
    assert find_ic_candidates(df, fib, POIConfig()) == [9]            # the V path: candle direction
    sel = select_ip_ic(df, fib, POIConfig(), classify_misses=True)
    assert [c.ip.ic_idx for c in sel.candidates] == [7]               # VD: the agreed direction
    assert sel.pick.ic_idx == 7 and sel.pick.ip_rank == "3A.i"        # a doji IC is a pinbar


def _tb2_frame():
    # ICs 7, 9, 11 — all 1A.i with maru ICs (a full rank + TB1 tie)
    return _frame([("M", U, 10), ("M", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 10)])


def test_tb2a_most_recent_tied_ic_before_the_reference_ic_may_be_c1():
    df = _tb2_frame()
    sel = _select(df, 6, 12, [_far_gap(10), _far_gap(12)])     # reference c2 10: IC 9 is its c1
    assert [c.ip.ip_rank for c in sel.candidates] == ["1A.i"] * 3
    assert sel.pick.ic_idx == 9 and sel.tiebreak == ("TB2", 11, 3, 10)


def test_tb2a_excludes_ics_at_or_after_the_reference_first_c2():
    df = _tb2_frame()
    sel = _select(df, 6, 12, [_far_gap(11), _far_gap(12)])
    assert sel.pick.ic_idx == 9 and sel.tiebreak == ("TB2", 11, 3, 11)


def test_tb2a_skips_a_filled_gap_as_the_reference():
    df = _tb2_frame()
    sel = _select(df, 6, 12, [_filled_gap(df, 8, 10), _far_gap(10), _far_gap(12)])
    assert sel.pick.ic_idx == 9 and sel.tiebreak[3] == 10
    unfilled = _select(_tb2_frame(), 6, 12, [_far_gap(8), _far_gap(10), _far_gap(12)])
    assert unfilled.pick.ic_idx == 7 and unfilled.tiebreak[3] == 8


def test_tb2a_reference_never_straddles_the_oldest_tied_ic():
    """Structural (an IC is never an sd c2): a hand-built instance that would straddle it fails loudly."""
    df = _tb2_frame()
    with pytest.raises(AssertionError, match="TB2 reference"):
        _select(df, 6, 12, [_far_gap(7, 8), _far_gap(12)])


def test_tb1_ic_type_maru_over_normal_over_pinbar():
    """Three 3C.i patterns (the IC type does not change a multi-opp rank): the maru IC wins although it is the
    OLDEST; the runner-up = the most recent IC of the best LOSING type — the normal (14), not the pinbar (11)."""
    df = _frame([("M", U, 10), ("M", D, 3), ("M", D, 3),
                 ("M", U, 10), ("M", D, 3), ("P", D, 5),
                 ("M", U, 10), ("M", D, 3), ("N", D, 5), ("M", U, 10)])
    sel = _select(df, 6, 15, [_far_gap(15)])
    picks = [(c.ip.ic_idx, c.ip.ip_rank) for c in sel.candidates if c.ip.ip_rank]
    assert picks == [(8, "3C.i"), (11, "3C.i"), (14, "3C.i")]
    assert [str(df.loc[i, "candle_type"]) for i, _ in picks] == ["maru", "pinbar", "normal"]
    assert sel.pick.ic_idx == 8 and sel.tiebreak == ("TB1", 14, 3, None)


def test_v_path_characterization():
    """The kept V30 / V60 / V90 path (X1) on a fixture: the candidate list and the variant dict (no direct test
    pinned them before landing A)."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10), _far_gap(12)]
    fib = _fib_over(df, 6, 12, float(df.loc[9, "l"]), float(df.loc[9, "h"]))
    cands = find_ic_candidates(df, fib, POIConfig())
    assert cands == [7, 9, 11]
    # V30 = the MOST RECENT candidate overlapping >= 30 % (11); V60 / V90 the most recent at those (9)
    assert select_ic_variants(cands, df, fib, POIConfig()) == {9: ["V60", "V90"], 11: ["V30"]}


# ---------------------------------------------------------------------------
# the records, the gate, the export, the mirror
# ---------------------------------------------------------------------------

class _Tracker:
    def __init__(self, fibs):
        self.fibs = fibs

    def get_fibs_for_charting(self):
        return list(self.fibs)


def _est(sid, cycle):
    from engine_v2.tests._event_factory import make_cts_established
    return make_cts_established(cts_anchor_idx=12, confirmed_at=12, price=0.61, structure_id=sid, cycle_id=cycle)


def test_records_gate_none_rows_and_tiebreak_on_the_pick():
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10), _far_gap(12)]
    live = _fib_over(df, 6, 12, 0.5, 0.7)
    no_touch = dataclasses.replace(_fib_over(df, 6, 12, 0.70, 0.71), cycle_id=1)  # a zone above every IC
    ended = dataclasses.replace(live, cycle_id=2, end_idx=12)                    # active but ended → gated out
    no_est = dataclasses.replace(live, cycle_id=3)                               # no CTS_ESTABLISHED → skipped
    recs = derive_ip_candidates(df, [_est(0, 0), _est(0, 1), _est(0, 2)], _Tracker([live, no_touch, ended, no_est]),
                                POIConfig(), structure_path_id="H1.main")
    assert recs, "the gate kept the live records"
    assert [(r.cycle_id, r.ic_idx, r.touches_zone, r.picked, r.reason) for r in recs] == [
        (0, 7, True, False, None), (0, 9, True, True, None), (0, 11, True, False, None),
        (1, 7, False, False, None), (1, 9, False, False, None), (1, 11, False, False, None),   # misses: never picked
        (1, None, None, False, "no_pattern")]
    pick = recs[1]
    assert (pick.ip_tiebreak, pick.ip_tb_runner_up_ic_idx, pick.ip_tied_count, pick.ip_tb_imbalance_first_c2_idx) \
        == ("TB2", 11, 3, 10)
    assert all(r.ip_tiebreak is None for r in recs if not r.picked)
    assert pick.meta == {"structure_path_id": "H1.main"} and pick.pattern_confirmed_idx == 11   # 10 + the next candle


def test_export_header_none_row_and_legs(tmp_path):
    from engine_v2.debug.export_zones import IP_CANDIDATE_COLUMNS, export_ip_candidates
    empty = tmp_path / "empty.csv"
    export_ip_candidates([], empty)
    assert list(pd.read_csv(empty).columns) == IP_CANDIDATE_COLUMNS
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10), _far_gap(12)]
    live = _fib_over(df, 6, 12, 0.5, 0.7)
    recs = derive_ip_candidates(df, [_est(0, 0), _est(0, 1)],
                                _Tracker([live, dataclasses.replace(_fib_over(df, 6, 12, 0.70, 0.71), cycle_id=1)]),
                                POIConfig(), structure_path_id="H1.main")
    out = tmp_path / "ip.csv"
    export_ip_candidates(recs, out)
    got = pd.read_csv(out)
    assert len(got) == 7 and list(got.cycle_id) == [0, 0, 0, 1, 1, 1, 1]
    assert got.ic_idx.isna().tolist() == [False] * 6 + [True]                        # the IP: none row last
    assert got.loc[6, "reason"] == "no_pattern" and not got.loc[3:5, "touches_zone"].any()
    assert got.loc[1, "legs"] == "8..8|9..9|10..10" and got.loc[1, "ip_rank"] == "1A.i"
    assert bool(got.loc[1, "picked"]) and got.loc[1, "ip_tiebreak"] == "TB2"


def test_mirror_shifts_every_index_field_once_per_lens():
    from engine_v2.multitf import entity_df_mutation as edm
    from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
    idx_fields = [f.name for f in dataclasses.fields(IPCandidateRecord) if f.name.endswith("_idx")]
    assert sorted(idx_fields) == sorted(edm._IP_CANDIDATE_IDX_FIELDS)
    vals = {f: 10 + k for k, f in enumerate(idx_fields)}
    rec = IPCandidateRecord(structure_id=0, cycle_id=1, struct_direction=1, candle_type="maru", agreed_dir=-1,
                            touches_zone=True, ip_rank="1A.i", reason=None, ip_leg_candle_counts="1-1-1",
                            picked=True, ip_tiebreak="TB2", ip_tied_count=2, meta={"structure_path_id": "src"},
                            **vals)
    none_row = dataclasses.replace(rec, ic_idx=None, pattern_anchor_idx=None, picked=False)
    trig = MultiTFTrigger(parent_tf="H1", parent_sid=0, parent_cycle_id=1, parent_sd=1, use_case="first_counter",
                          lower_tf="M15", lower_sd=-1)
    res = LowerTFResult(trigger=trig, df=pd.DataFrame({"o": [0.0]}), events=[], kl_zones=[], wave_candles=[],
                        fib_states=[], poi_zones=[], wvmi_records=[], prev_bos_lines=[], status="finalized",
                        meta={"slice_begin": 100, "sub_id": 7}, ip_candidates=[rec, none_row])
    lenses = {"H1.main >> M15.counter": pd.DataFrame({"o": [0.0] * 200}),
              "H1.main >> M15.confluence": pd.DataFrame({"o": [0.0] * 200})}
    for path, lens_df in lenses.items():
        edm.mirror_lower_tf_result_to_entity_df(lens_df, res, structure_path_id=path)
    for path, lens_df in lenses.items():
        got, got_none = lens_df.attrs["ip_candidates"]
        assert {f: getattr(got, f) for f in idx_fields} == {f: v + 100 for f, v in vals.items()}
        assert got_none.ic_idx is None and got_none.pattern_anchor_idx is None and got_none.pattern_end_idx == \
            vals["pattern_end_idx"] + 100
        assert got.meta["structure_path_id"] == path and got.meta["sub_id"] == 7
    assert rec.ic_idx == vals["ic_idx"] and rec.meta == {"structure_path_id": "src"}     # the source untouched
    a, b = (d.attrs["ip_candidates"][0] for d in lenses.values())
    assert a is not b and a.meta is not b.meta


def test_downstream_pipeline_reads_the_poi_config_defaults(monkeypatch):
    """PLAN_H §8.1 one parameter source: the selection, the drawn POIs and the debug CSV all get `POIConfig()`
    (the re-typed literals are gone; the leg cap is the default table) — and (landing B, §8.5) the selection is
    computed ONCE: the SAME `select_ip_records` list is handed to both readers."""
    import engine_v2.pipeline.orchestrator as orch
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data, _prepare_df
    seen = []
    real_sel, real_poi, real_ip = orch.select_ip_records, orch.derive_poi_zones, orch.derive_ip_candidates

    def sel(*a, **k):
        out = real_sel(*a, **k)
        seen.append(("sel", a[3], out))
        return out

    monkeypatch.setattr(orch, "select_ip_records", sel)
    monkeypatch.setattr(orch, "derive_poi_zones",
                        lambda *a, **k: seen.append(("poi", k["config"], k["ip_records"])) or real_poi(*a, **k))
    monkeypatch.setattr(orch, "derive_ip_candidates",
                        lambda *a, **k: seen.append(("ip", a[3], k["ip_records"])) or real_ip(*a, **k))
    df = _prepare_df(_make_ip_multicycle_data())
    res = compute_bounded_structure(df, 0, +1)
    with contextlib.redirect_stdout(io.StringIO()):
        out = orch._run_downstream_pipeline(res.df, res.events, res.struct_direction, wvmi="off",
                                            cross_fib_limits=main_limits(res.events))
    assert out["poi_zones"], "the fixture draws a POI"
    assert [k for k, _, _ in seen] == ["sel", "poi", "ip"]
    assert all(cfg == POIConfig() for _, cfg, _ in seen)
    assert seen[1][2] is seen[0][2] and seen[2][2] is seen[0][2]


def test_downstream_pipeline_returns_the_selection_of_its_own_fibs():
    """The pipeline call site (the real collaborator): `_run_downstream_pipeline` hands back exactly
    `derive_ip_candidates` over its own fib tracker — one or more rows per gated record — and (landing B, D2)
    draws exactly the picked rows: one POI per picked row, the same IC / rank / legs / tiebreak."""
    from engine_v2.pipeline.orchestrator import _run_downstream_pipeline
    from engine_v2.structure.structure_engine import compute_bounded_structure
    from engine_v2.tests.test_unified_probe import _make_ip_multicycle_data, _prepare_df
    df = _prepare_df(_make_ip_multicycle_data())
    res = compute_bounded_structure(df, 0, +1)
    with contextlib.redirect_stdout(io.StringIO()):
        out = _run_downstream_pipeline(res.df, res.events, res.struct_direction, wvmi="off",
                                       cross_fib_limits=main_limits(res.events))
    assert out["ip_candidates"], "the fixture has gated fib records"
    again = derive_ip_candidates(res.df, res.events, out["fib_tracker"], POIConfig(), structure_path_id="H1.main")
    assert [dataclasses.astuple(r) for r in out["ip_candidates"]] == [dataclasses.astuple(r) for r in again]
    assert {r.meta["structure_path_id"] for r in out["ip_candidates"]} == {"H1.main"}
    picked = [r for r in out["ip_candidates"] if r.picked]
    assert picked, "the fixture has a pick"
    # cycles 1 / 3: no pattern; cycle 0 (main's single since PLAN_I S4, Q6): no candidates — no POI for any
    assert [(r.cycle_id, r.reason) for r in out["ip_candidates"] if r.ic_idx is None] == [
        (0, "no_candidates"), (1, "no_pattern"), (3, "no_pattern")]
    drawn = [(z.meta["structure_id"], z.meta["cycle_id"], z.ic_idx, z.meta["ip_rank"], z.meta["ip_leg_candle_counts"],
              z.meta["pattern_anchor_idx"], z.meta["opp_leg_first_idx"], z.meta["pattern_end_idx"],
              z.meta["pattern_confirmed_idx"], z.meta["ip_tiebreak"]) for z in out["poi_zones"]]
    assert drawn == [(r.structure_id, r.cycle_id, r.ic_idx, r.ip_rank, r.ip_leg_candle_counts, r.pattern_anchor_idx,
                      r.opp_leg_first_idx, r.pattern_end_idx, r.pattern_confirmed_idx, r.ip_tiebreak)
                     for r in picked]

# ---------------------------------------------------------------------------
# the landing review's test-gap lens (2026-10-06): each kills a mutant the first suite let survive
# (`plans/plan_h_inputs/mutants_plan_h_a.py`, ids N*)
# ---------------------------------------------------------------------------

# N01 — rule A/B with 2 junction dojis below the cap: fill the initial leg UP TO the cap (1 doji), the rest to opp
def test_rule_ab_fills_the_initial_leg_only_up_to_the_cap():
    _, rule_ab, _, _ = _variants([U, U, Z, Z, D, U], 4)
    assert rule_ab == 1


# N05 — a continuation run of EXACTLY cap candles is known complete at its last candle
def test_pattern_confirmed_when_the_run_is_exactly_the_cap():
    variants, _, confirmed, _ = _variants([U, D, U, U, U, D], 1)
    assert {v.continuation for v in variants} == {(2, 3, 4)} and confirmed == 4


# N10 / N11 — a SELL pattern: down stacks have their max high on the FIRST candle, min low on the LAST
def test_sell_direction_stacks():
    res, n = _ip([("M", D, 6), ("M", D, 6), ("M", U, 6), ("M", D, 10)], 2, sd=-1)
    assert res.ip_rank == "1A.ii", (res.reason, res.detail)
    res, n = _ip([("M", D, 10), ("M", U, 6), ("M", D, 6), ("M", D, 6)], 1, sd=-1)
    assert res.ip_rank == "1A.iii", (res.reason, res.detail)


# N13 — a stack is classified with the frame's PAIR (JPY pip 0.01): the floor test on a USD_JPY frame
def _frame_jpy(specs, prefix):
    rows, c_prev = [], 60.0
    for kind, d, r in list(prefix) + list(specs):
        o, h, l, c = _ohlc(c_prev / 100.0, kind, d, r)
        o, h, l, c = o * 100, h * 100, l * 100, c * 100
        rows.append({"o": o, "h": h, "l": l, "c": c})
        c_prev = c
    df = pd.DataFrame(rows)
    df.insert(0, "time", pd.date_range("2026-01-05", periods=len(df), freq="h", tz="UTC"))
    df["volume"] = 1.0
    df.attrs["pair"] = "USD_JPY"
    return apply_candle_classification(df, timeframe="H1").df


def test_jpy_stack_uses_the_frame_pair_pip():
    small = tuple(("M", U if k % 2 == 0 else D, 2.5) for k in range(6))
    df = _frame_jpy([("M", U, 2.5), ("P", D, 1), ("M", U, 1.2), ("M", U, 1.2)], small)
    agreed = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
    res = classify_ip(df, 7, 1, bos_anchor_idx=6, cts_anchor_idx=9, agreed=agreed, cap=3, params=H1)
    assert res.reason == "continuation_not_big_maru" and res.detail["continuation_type"] == "pinbar"


# N14 — a no-pattern candidate reports its FIRST failing variant (the cap-3 initial), not the D3c cap-1's
def test_first_failing_variant_is_reported():
    res, _ = _ip([("N", U, 2), ("N", U, 2), ("M", D, 6), ("M", U, 10)], 2)
    assert res.ip_rank is None
    assert (res.reason, res.ip_leg_candle_counts) == ("initial_fails", "2-1-1")


# N15 — TB1 runs over the RANK TIE only: a worse-ranked pattern with a maru IC never joins it
def test_tb1_group_is_the_rank_tie():
    df = _frame([("M", U, 10), ("M", D, 3), ("M", D, 3),          # 6..8: IC 8 maru, 3C.i
                 ("M", U, 10), ("M", D, 3), ("P", D, 5),          # 9..11: IC 11 pinbar, 3C.i
                 ("M", U, 10), ("M", D, 3, 5),                    # 12 cont; 13 a split candle (breaks the run)
                 ("N", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 10)])  # 14..17: IC 16 maru, 3D.i
    assert (int(df.loc[13, "direction"]), int(df.loc[13, "vol_dir"])) == (-1, 1)
    sel = _select(df, 6, 17, [_far_gap(17)])
    ranks = {c.ip.ic_idx: c.ip.ip_rank for c in sel.candidates}
    assert ranks[8] == "3C.i" and ranks[11] == "3C.i" and ranks[16] == "3D.i", ranks
    assert sel.pick.ic_idx == 8 and sel.tiebreak == ("TB1", 11, 2, None)


# N16 — the TB2 reference may be the gap whose c2 IS the CTS anchor (fill / formed test as of no moment)
def test_tb2_reference_may_sit_at_the_cts_anchor():
    sel = _select(_tb2_frame(), 6, 12, [_far_gap(12)])
    assert sel.pick.ic_idx == 11 and sel.tiebreak == ("TB2", 9, 3, 12)


# N17 — TB2's fill horizon is the CTS anchor: a gap filled only AFTER the CTS is still the reference
def test_tb2_fill_horizon_is_the_cts_anchor():
    specs = [("M", U, 10), ("M", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 10),
             ("M", D, 40), ("M", U, 40)]
    df = _frame(specs)
    c11 = float(df.loc[11, "c"])
    gap = ImbalanceInstance(start_idx=10, end_idx=10, direction=1, gap_top=c11 - 5 * PIP,
                            gap_bottom=c11 - 8 * PIP, gap_size=3 * PIP)
    assert not gap.is_filled(df, 12, 0.70) and gap.is_filled(df, 14, 0.70)
    sel = _select(df, 6, 12, [gap, _far_gap(12)])
    assert sel.pick.ic_idx == 9 and sel.tiebreak == ("TB2", 11, 3, 10)


# N19 — after TB1 removes a type, TB2's tied count is the TB1 survivors
def test_tb2_count_is_the_tb1_survivors():
    df = _frame([("M", U, 10), ("M", D, 3), ("M", D, 3),
                 ("M", U, 10), ("M", D, 3), ("P", D, 5),
                 ("M", U, 10), ("M", D, 3), ("M", D, 3), ("M", U, 10)])
    sel = _select(df, 6, 15, [_far_gap(15)])
    assert [str(df.loc[i, "candle_type"]) for i in (8, 11, 14)] == ["maru", "pinbar", "maru"]
    assert sel.pick.ic_idx == 14 and sel.tiebreak == ("TB2", 8, 2, 15)


# N20 — a gated record with no §3.1 candidate gets the `no_candidates` none row
def test_no_candidates_row():
    df = _tb2_frame()
    df.attrs["imbalances"] = []
    recs = derive_ip_candidates(df, [_est(0, 0)], _Tracker([_fib_over(df, 6, 12, 0.5, 0.7)]), POIConfig(),
                                structure_path_id="H1.main")
    assert [(r.ic_idx, r.picked, r.reason) for r in recs] == [(None, False, "no_candidates")]


# N22 — every gated record is its own selection (two records of one cycle → both)
def test_two_gated_records_of_one_cycle_are_both_selected():
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(10), _far_gap(12)]
    live = _fib_over(df, 6, 12, 0.5, 0.7)
    other = dataclasses.replace(live, bos_idx=8, active=False, locked=True)
    recs = derive_ip_candidates(df, [_est(0, 0)], _Tracker([live, other]), POIConfig(), structure_path_id="H1.main")
    assert sorted({r.bos_anchor_idx for r in recs}) == [6, 8]


# N23 / N24 / N25 — the record's index fields and agreed_dir, on a volume-only doji IC
def test_record_index_fields_and_agreed_dir():
    df = _frame([("M", U, 10), ("Z", 0, 2, -3), ("M", U, 10), ("M", D, 3, 5), ("M", U, 10)])
    df.attrs["imbalances"] = [_far_gap(10)]
    recs = derive_ip_candidates(df, [_est(0, 0)], _Tracker([_fib_over(df, 6, 10, 0.5, 0.7)]), POIConfig(),
                                structure_path_id="H1.main")
    (pick,) = [r for r in recs if r.picked]
    assert (pick.ic_idx, pick.bos_anchor_idx, pick.cts_anchor_idx) == (7, 6, 10)
    assert (pick.pattern_anchor_idx, pick.opp_leg_first_idx, pick.pattern_end_idx) == (6, 7, 8)
    assert (pick.candle_type, pick.agreed_dir) == ("pinbar", -1)


# N26 / N27 / N28 / N30 — the CSV equals the records, column by column, as RAW text
def test_export_round_trip_raw_text(tmp_path):
    from engine_v2.debug.export_zones import export_ip_candidates
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10), ("M", D, 6), ("M", U, 6), ("M", U, 6), ("M", D, 6),
                 ("M", U, 10)])
    df.attrs["imbalances"] = [_far_gap(13)]
    live = _fib_over(df, 6, 13, 0.5, 0.7)
    miss = dataclasses.replace(_fib_over(df, 6, 13, 0.70, 0.71), cycle_id=1)
    recs = derive_ip_candidates(df, [_est(0, 0), _est(0, 1)], _Tracker([live, miss]), POIConfig(),
                                structure_path_id="H1.main")
    assert len(recs) >= 3 and any(r.ic_idx is None for r in recs)
    out = tmp_path / "ip.csv"
    export_ip_candidates(recs, out)
    lines = out.read_text().splitlines()
    head = lines[0].split(",")
    rows = [dict(zip(head, l.split(","))) for l in lines[1:]]
    assert [r["touches_zone"] for r in rows] == [str(x.touches_zone) if x.touches_zone is not None else "" for x in recs]
    assert [r["picked"] for r in rows] == [str(x.picked) for x in recs]
    assert [r["ic_idx"] for r in rows] == ["" if x.ic_idx is None else str(x.ic_idx) for x in recs]  # no "7.0"
    by_ic = {(r["cycle_id"], r["ic_idx"]): r for r in rows}
    assert by_ic[("0", "9")]["ip_leg_candle_counts"] == "1-1-2" and by_ic[("0", "9")]["legs"] == "8..8|9..9|10..11"
    assert by_ic[("0", "12")]["legs"] == "10..11|12..12|13..13"


# N31 — two subs mirrored into ONE lens df: both kept, in order
def test_mirror_extends_across_subs_of_one_lens():
    from engine_v2.multitf import entity_df_mutation as edm
    from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
    from engine_v2.zones.poi_zones import IPCandidateRecord
    trig = MultiTFTrigger(parent_tf="H1", parent_sid=0, parent_cycle_id=1, parent_sd=1, use_case="first_counter",
                          lower_tf="M15", lower_sd=-1)
    lens = pd.DataFrame({"o": [0.0] * 300})
    for sub_id, begin in ((7, 100), (8, 150), (9, 200)):
        rec = IPCandidateRecord(structure_id=0, cycle_id=1, struct_direction=1, bos_anchor_idx=1, cts_anchor_idx=5,
                                ic_idx=3, candle_type="maru", agreed_dir=-1, touches_zone=True, ip_rank="1A.i",
                                reason=None, pattern_anchor_idx=2, opp_leg_first_idx=3, pattern_end_idx=4,
                                pattern_confirmed_idx=5, ip_leg_candle_counts="1-1-1", picked=True,
                                meta={"structure_path_id": "src"})
        res = LowerTFResult(trigger=trig, df=pd.DataFrame({"o": [0.0]}), events=[], kl_zones=[], wave_candles=[],
                            fib_states=[], poi_zones=[], wvmi_records=[], prev_bos_lines=[], status="finalized",
                            meta={"slice_begin": begin, "sub_id": sub_id}, ip_candidates=[rec])
        edm.mirror_lower_tf_result_to_entity_df(lens, res, structure_path_id="H1.main >> M15.counter")
    assert [(r.meta["sub_id"], r.ic_idx) for r in lens.attrs["ip_candidates"]] == [(7, 103), (8, 153), (9, 203)]


# N32 — select_ip_ic classifies stacks with the FRAME's timeframe params (M15 floor 1.0, not H1's 2.2)
def test_select_uses_the_frame_timeframe_params():
    small = tuple(("M", U if k % 2 == 0 else D, 2.5) for k in range(6))
    df = _frame([("M", U, 2.5), ("P", D, 1), ("M", U, 1.2), ("M", U, 1.2)], prefix=small, timeframe="M15")
    sel = _select(df, 6, 9, [_far_gap(9)])
    assert sel.pick is not None and sel.pick.ic_idx == 7 and sel.pick.ip_rank == "3A.iii"


# N33 — the zone of a SELL fib is measured up from the low
def test_sell_zone_bounds():
    lo, hi = fib_zone_bounds(1.1, 1.0, -1, POIConfig())
    assert lo == pytest.approx(1.0618) and hi == pytest.approx(1.08)
    lo, hi = fib_zone_bounds(1.1, 1.0, 1, POIConfig())
    assert lo == pytest.approx(1.02) and hi == pytest.approx(1.0382)


def test_pattern_confirmed_past_the_frame_when_the_data_ends_below_the_cap():
    """R1: a continuation that runs into the frame's last candle below the cap is not known complete yet —
    `pattern_confirmed_idx` = one past the frame (a POI on it would never activate)."""
    assert _variants([U, D, U], 1)[2] == 3


def test_a_mirrored_record_exports_entity_absolute_legs(tmp_path):
    """§8.3a deviation 3: `legs` is located from the SHIFTED `ic_idx` + the counts — a mirrored sub record (slice
    begin 100) exports entity-absolute legs (the first replay had a stored slice-local legs string)."""
    from engine_v2.debug.export_zones import export_ip_candidates
    from engine_v2.multitf import entity_df_mutation as edm
    from engine_v2.multitf.types import LowerTFResult, MultiTFTrigger
    trig = MultiTFTrigger(parent_tf="H1", parent_sid=0, parent_cycle_id=1, parent_sd=1, use_case="first_counter",
                          lower_tf="M15", lower_sd=-1)
    rec = IPCandidateRecord(structure_id=0, cycle_id=1, struct_direction=1, bos_anchor_idx=1, cts_anchor_idx=5,
                            ic_idx=3, candle_type="maru", agreed_dir=-1, touches_zone=True, ip_rank="1A.i",
                            reason=None, pattern_anchor_idx=2, opp_leg_first_idx=3, pattern_end_idx=4,
                            pattern_confirmed_idx=5, ip_leg_candle_counts="1-1-1", picked=True,
                            meta={"structure_path_id": "src"})
    lens = pd.DataFrame({"o": [0.0] * 200})
    edm.mirror_lower_tf_result_to_entity_df(
        lens, LowerTFResult(trigger=trig, df=pd.DataFrame({"o": [0.0]}), events=[], kl_zones=[], wave_candles=[],
                            fib_states=[], poi_zones=[], wvmi_records=[], prev_bos_lines=[], status="finalized",
                            meta={"slice_begin": 100, "sub_id": 7}, ip_candidates=[rec]),
        structure_path_id="H1.main >> M15.counter")
    out = tmp_path / "ip.csv"
    export_ip_candidates(lens.attrs["ip_candidates"], out)
    assert pd.read_csv(out).loc[0, "legs"] == "102..102|103..103|104..104"


# ---------------------------------------------------------------------------
# the landing review's code lens (2026-10-06) — the fail-loud reads (F* mutants)
# ---------------------------------------------------------------------------

def test_a_non_contiguous_index_fails_loudly():
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10)])
    bad = df.set_axis([0, 1, 1] + list(range(3, len(df))))       # first 0, last n-1, monotonic — not contiguous
    agreed = agreed_direction(bad["direction"].to_numpy(), bad["vol_dir"].to_numpy())
    with pytest.raises(AssertionError, match="0-based contiguous"):
        classify_ip(bad, 7, 1, bos_anchor_idx=6, cts_anchor_idx=8, agreed=agreed, cap=3, params=H1)


def test_a_frame_without_its_pair_fails_loudly():
    """The stacks' pip size is the frame's own (P2): no silent non-JPY default."""
    df = _frame([("M", U, 10), ("M", D, 6), ("M", U, 10)])
    del df.attrs["pair"]
    agreed = agreed_direction(df["direction"].to_numpy(), df["vol_dir"].to_numpy())
    with pytest.raises(KeyError):
        classify_ip(df, 7, 1, bos_anchor_idx=6, cts_anchor_idx=8, agreed=agreed, cap=3, params=H1)


def test_a_locked_fib_without_its_cts_established_fails_loudly():
    """As `derive_poi_zones`: a locked fib always has its CTS_ESTABLISHED (the event contract)."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(12)]
    locked = dataclasses.replace(_fib_over(df, 6, 12, 0.5, 0.7), active=False, locked=True)
    with pytest.raises(AssertionError, match="has no CTS_ESTABLISHED"):
        derive_ip_candidates(df, [], _Tracker([locked]), POIConfig(), structure_path_id="H1.main")


def test_the_locked_fib_assert_runs_before_the_selection():
    """`select_ip_records` checks the event contract BEFORE it selects (PLAN_H §8.5a deviation 2): on an unstamped
    frame the selection would raise KeyError first (landing review G02)."""
    df = _tb2_frame()
    df.attrs["imbalances"] = [_far_gap(12)]
    locked = dataclasses.replace(_fib_over(df, 6, 12, 0.5, 0.7), active=False, locked=True)
    del df.attrs["timeframe"]
    with pytest.raises(AssertionError, match="has no CTS_ESTABLISHED"):
        select_ip_records(df, [], _Tracker([locked]), POIConfig())
