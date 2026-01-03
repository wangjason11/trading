import pandas as pd
import pytest

from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import (
    compute_candle_metrics,
    classify_candles,
    compute_candle_features,
)

def _df(rows):
    # rows: list of dicts with time,o,h,l,c
    return pd.DataFrame(rows)

@pytest.fixture
def params():
    return CandleParams(
        maru=0.7,
        pinbar=0.5,
        pinbar_distance=0.5,
        big_maru_threshold=0.7,
        big_normal_threshold=0.5,
        lookback=5,
        special_maru=0.5,
        special_maru_distance=0.1,
    )

def test_metrics_body_and_len(params):
    df = _df([{"time":"2025-01-01","o":1.0,"h":2.0,"l":0.0,"c":1.5}])
    out = compute_candle_metrics(df)
    assert out.loc[0, "body_len"] == pytest.approx(0.5)
    assert out.loc[0, "candle_len"] == pytest.approx(2.0)
    assert out.loc[0, "body_pct"] == pytest.approx(0.25)

def test_direction_up_down_flat():
    df = _df([
        {"time":"t1","o":1.0,"h":1.2,"l":0.8,"c":1.1},  # up
        {"time":"t2","o":1.0,"h":1.2,"l":0.8,"c":0.9},  # down
        {"time":"t3","o":1.0,"h":1.2,"l":0.8,"c":1.0},  # flat
    ])
    out = compute_candle_metrics(df)
    assert int(out.loc[0, "direction"]) == 1
    assert int(out.loc[1, "direction"]) == -1
    assert int(out.loc[2, "direction"]) == 0

def test_maru_when_body_pct_gte_threshold(params):
    # candle_len=10, body=7 => 0.7 => maru (boundary)
    df = _df([{"time":"t","o":1.0,"h":11.0,"l":1.0,"c":8.0}])
    out = compute_candle_metrics(df)
    out = classify_candles(out, params)
    assert out.loc[0, "candle_type"] == "maru"

def test_not_maru_when_body_pct_below_threshold(params):
    # candle_len=10, body=6.9 => 0.69 => not maru
    df = _df([{"time":"t","o":1.0,"h":11.0,"l":1.0,"c":7.9}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] != "maru"

def test_pinbar_when_body_pct_lte_threshold(params):
    # candle_len=10, body=5 => 0.5 => pinbar (boundary)
    df = _df([{"time":"t","o":1.0,"h":11.0,"l":1.0,"c":6.0}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "pinbar"

def test_maru_wins_if_overlap(params):
    # If maru threshold <= pinbar threshold overlap could happen;
    # here it won't with current params, but ensure logic order is maru after pinbar.
    p = CandleParams(**{**params.__dict__, "maru": 0.5, "pinbar": 0.5})
    df = _df([{"time":"t","o":1.0,"h":11.0,"l":1.0,"c":6.0}])  # body=5 len=10 => 0.5
    out = classify_candles(compute_candle_metrics(df), p)
    assert out.loc[0, "candle_type"] == "maru"

def test_pinbar_dir_up(params):
    # Make pinbar with body near top: small upper wick, large lower wick
    # o=9.5 c=10.0 h=10.1 l=0.0 => len=10.1, upper=0.1, lower=9.5
    df = _df([{"time":"t","o":9.5,"h":10.1,"l":0.0,"c":10.0}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "pinbar"
    assert int(out.loc[0, "pinbar_dir"]) == 1

def test_pinbar_dir_down(params):
    # body near bottom: small lower wick, large upper wick
    # o=0.5 c=0.0 h=10.1 l=-0.1 => len=10.2, upper=9.6, lower=0.1
    df = _df([{"time":"t","o":0.5,"h":10.1,"l":-0.1,"c":0.0}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "pinbar"
    assert int(out.loc[0, "pinbar_dir"]) == -1

def test_pinbar_dir_none_if_not_meeting_distance(params):
    # Make a pinbar body but wicks not asymmetric enough to pass distance gates
    df = _df([{"time":"t","o":4.0,"h":10.0,"l":0.0,"c":6.0}])  # len=10 body=2 => 0.2 pinbar
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "pinbar"
    assert int(out.loc[0, "pinbar_dir"]) == 0

def test_special_maru_flag_true(params):
    # body_pct >= 0.5 and one wick <= 0.1*len
    df = _df([{"time":"t","o":1.0,"h":11.0,"l":1.0,"c":6.5}])  # len=10, body=5.5 => 0.55; upper=4.5 lower=0
    out = classify_candles(compute_candle_metrics(df), params)
    assert bool(out.loc[0, "is_special_maru"]) is True

def test_zero_length_candle_safe(params):
    # h==l, avoid div-by-zero; body_pct should be 0
    df = _df([{"time":"t","o":1.0,"h":1.0,"l":1.0,"c":1.0}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "body_pct"] == pytest.approx(0.0)
    # Should classify as pinbar because body_pct <= 0.5
    assert out.loc[0, "candle_type"] == "pinbar"

def test_big_flags_use_prior_marus_only(params):
    """
    prior max is computed from previous MARU candles only.
    Create history where the largest candle is NORMAL, but maru history is smaller.
    Ensure big_normal compares to maru max, not overall max.
    """
    # idx0 maru len=10 (body 7)
    # idx1 normal len=100 (body 60) - huge but NOT maru
    # idx2 normal len=8 (body 4) -> big_normal ratio uses prior maru max=10, so 8/10=0.8 >= 0.5 => True
    df = _df([
        {"time":"t0","o":0.0,"h":10.0,"l":0.0,"c":7.0},     # maru: len=10 body=7 => 0.7
        {"time":"t1","o":0.0,"h":100.0,"l":0.0,"c":60.0},   # normal: len=100 body=60 => 0.6
        {"time":"t2","o":0.0,"h":8.0,"l":0.0,"c":4.1},      # normal: len=8 body=4 => 0.5 pinbar? (boundary)
    ])
    out = compute_candle_features(df, params, anchor_shifts=(0,))
    assert out.loc[0, "candle_type"] == "maru"
    assert out.loc[1, "candle_type"] == "normal"  # not maru since 0.6 < 0.7
    # prior maru max before idx2 is 10, so ratio=8/10=0.8 => big_normal should be True (ctype normal allowed)
    assert float(out.loc[2, "prior_maru_max_len_as0"]) == pytest.approx(10.0)
    assert bool(out.loc[2, "is_big_normal_as0"]) is True


def test_big_maru_threshold_boundary(params):
    """
    If ratio == big_maru_threshold, should be True.
    """
    p = CandleParams(**{**params.__dict__, "big_maru_threshold": 0.7, "lookback": 1})

    df = _df([
        {"time":"t0","o":0.0,"h":10.0,"l":0.0,"c":7.0},  # maru len=10
        {"time":"t1","o":0.0,"h":7.0,"l":0.0,"c":4.9},   # maru? len=7 body=4.9 => 0.7 => maru
    ])
    out = compute_candle_features(df, p, anchor_shifts=(0,))
    assert out.loc[1, "candle_type"] == "maru"
    # prior maru max len=10, current len=7 => ratio=0.7 => boundary True
    assert bool(out.loc[1, "is_big_maru_as0"]) is True


def test_anchor_shift_changes_reference_set(params):
    """
    Demonstrate that anchor_shift=1 excludes the immediately previous maru (i-1)
    by making i-1 a huge maru. Then:
      - as0 should see that huge maru and prior max is huge => not big
      - as1 should *exclude* it, see older smaller maru => becomes big
    """
    p = CandleParams(**{**params.__dict__, "lookback": 1, "big_normal_threshold": 0.5})

    # idx0 maru len=10
    # idx1 maru len=100 (huge)
    # idx2 normal len=40
    # For idx2:
    #   as0 prior maru max before idx2 = 100 => 40/100=0.4 => NOT big_normal
    #   as1 prior maru max before idx2-1 (cutoff idx1) = 10 => 40/10=4.0 => big_normal
    df = _df([
        {"time":"t0","o":0.0,"h":10.0,"l":0.0,"c":7.0},      # maru
        {"time":"t1","o":0.0,"h":100.0,"l":0.0,"c":70.0},    # maru
        {"time":"t2","o":0.0,"h":40.0,"l":0.0,"c":25.0},     # normal (0.5 pinbar boundary? but type doesn't matter for big_normal if normal)
    ])
    out = compute_candle_features(df, p, anchor_shifts=(0, 1))
    assert out.loc[2, "candle_type"] in ("normal", "pinbar", "maru")

    assert float(out.loc[2, "prior_maru_max_len_as0"]) == pytest.approx(100.0)
    assert bool(out.loc[2, "is_big_normal_as0"]) is False

    assert float(out.loc[2, "prior_maru_max_len_as1"]) == pytest.approx(10.0)
    assert bool(out.loc[2, "is_big_normal_as1"]) is True


def test_anchor_shift_2_for_third_candle_in_pattern(params):
    """
    anchor_shift=2 should reference marus strictly before (i-2).
    This matches the requirement for evaluating candle idx+2 relative to pattern start idx.
    """
    p = CandleParams(**{**params.__dict__, "lookback": 1, "big_normal_threshold": 0.5})

    # idx0 maru len=10
    # idx1 maru len=50
    # idx2 normal len=20
    # For idx2:
    #   as2 cutoff = idx0 (since i-2=0) => prior marus before idx0 = none => max=0
    #   as1 cutoff = idx1 => prior max=10
    #   as0 cutoff = idx2 => prior max=50
    df = _df([
        {"time":"t0","o":0.0,"h":10.0,"l":0.0,"c":7.0},     # maru len 10
        {"time":"t1","o":0.0,"h":50.0,"l":0.0,"c":35.0},    # maru len 50
        {"time":"t2","o":0.0,"h":20.0,"l":0.0,"c":10.0},    # normal
    ])
    out = compute_candle_features(df, p, anchor_shifts=(0, 1, 2))

    assert float(out.loc[2, "prior_maru_max_len_as0"]) == pytest.approx(50.0)
    assert float(out.loc[2, "prior_maru_max_len_as1"]) == pytest.approx(10.0)
    assert float(out.loc[2, "prior_maru_max_len_as2"]) == pytest.approx(0.0)

    # big flags should be False when prior max is 0
    assert bool(out.loc[2, "is_big_normal_as2"]) is False

def test_pinbar_dir_up_boundary_distance(params):
    """
    Up-pinbar: upper_wick <= dist and lower_wick >= dist.
    Test boundary equality on upper_wick == dist.
    """
    # Choose len=10, dist=5
    # Make upper_wick exactly 5, lower_wick >= 5, body small enough to be pinbar
    # o=9.0 c=9.5 => body=0.5
    # h=14.5 => upper_wick = 14.5 - 9.5 = 5.0
    # l=-0.5 => lower_wick = 9.0 - (-0.5) = 9.5
    df = _df([{"time":"t","o":9.0,"h":14.5,"l":-0.5,"c":9.5}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "pinbar"
    assert int(out.loc[0, "pinbar_dir"]) == 1


def test_pinbar_dir_down_boundary_distance(params):
    """
    Down-pinbar: lower_wick <= dist and upper_wick >= dist.
    Test boundary equality on lower_wick == dist.
    """
    # len=10, dist=5
    # o=0.5 c=0.0 => body=0.5
    # l=-5.0 => lower_wick = min(o,c)-l = 0.0 - (-5.0)=5.0 boundary
    # h=15.0 => upper_wick = 15.0 - max(o,c)=15.0 - 0.5=14.5
    df = _df([{"time":"t","o":0.5,"h":15.0,"l":-5.0,"c":0.0}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "pinbar"
    assert int(out.loc[0, "pinbar_dir"]) == -1


def test_pinbar_dir_not_set_if_both_wicks_small(params):
    """
    If both wicks are <= dist, should not assign a pinbar_dir (ambiguous body location).
    """
    # len=10, dist=5
    # Make both upper and lower <=5
    df = _df([{"time":"t","o":4.0,"h":9.9,"l":-0.1,"c":5.0}])  # len=8, body=1 => pinbar; upper=4, lower=3
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "pinbar"
    assert int(out.loc[0, "pinbar_dir"]) == 0


def test_special_maru_flag_false_when_body_pct_below(params):
    df = _df([{"time":"t","o":0.0,"h":10.0,"l":0.0,"c":4.9}])  # body_pct=0.49
    out = classify_candles(compute_candle_metrics(df), params)
    assert bool(out.loc[0, "is_special_maru"]) is False


def test_special_maru_flag_false_when_body_not_near_extreme(params):
    """
    body_pct meets special_maru but neither wick is <= special_maru_distance * len.
    """
    # len=10, body=6 => 0.6 meets special_maru=0.5
    # but wicks both 2 => special_maru_distance*len = 1.0, so neither wick <= 1.0
    df = _df([{"time":"t","o":2.0,"h":10.0,"l":0.0,"c":8.0}])  # upper=2, lower=2
    out = classify_candles(compute_candle_metrics(df), params)
    assert bool(out.loc[0, "is_special_maru"]) is False


def test_maru_not_pinbar_when_body_pct_between(params):
    """
    body_pct between pinbar and maru thresholds => normal
    """
    # len=10, body=6 => 0.6 -> normal (pinbar=0.5, maru=0.7)
    df = _df([{"time":"t","o":0.0,"h":10.0,"l":0.0,"c":6.0}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "normal"


def test_maru_classification_ignores_wicks(params):
    """
    maru is purely body_pct threshold; ensure a candle with large wick but body_pct >= maru still classifies as maru.
    """
    # len=10, body=7 -> 0.7 maru; choose o=0,c=7,h=10,l=0 -> big upper wick exists (3)
    df = _df([{"time":"t","o":0.0,"h":10.0,"l":0.0,"c":7.0}])
    out = classify_candles(compute_candle_metrics(df), params)
    assert out.loc[0, "candle_type"] == "maru"


def test_big_normal_does_not_apply_to_pinbar(params):
    """
    Even if length ratio is huge, pinbar candles should not be big_normal.
    """
    p = CandleParams(**{**params.__dict__, "lookback": 1, "big_normal_threshold": 0.5})
    df = _df([
        {"time":"t0","o":0.0,"h":10.0,"l":0.0,"c":7.0},   # maru len 10
        # pinbar: len=100 body=10 => 0.1 pinbar
        {"time":"t1","o":0.0,"h":100.0,"l":0.0,"c":10.0},
    ])
    out = compute_candle_features(df, p, anchor_shifts=(0,))
    assert out.loc[1, "candle_type"] == "pinbar"
    assert bool(out.loc[1, "is_big_normal_as0"]) is False


def test_big_flags_multiple_shifts_exist(params):
    """
    Ensure compute_candle_features creates all requested shift columns.
    """
    df = _df([
        {"time":"t0","o":0.0,"h":10.0,"l":0.0,"c":7.0},
        {"time":"t1","o":0.0,"h":12.0,"l":0.0,"c":6.5},
        {"time":"t2","o":0.0,"h":8.0,"l":0.0,"c":4.1},
    ])
    out = compute_candle_features(df, params, anchor_shifts=(0, 1, 2))
    for s in (0, 1, 2):
        assert f"is_big_maru_as{s}" in out.columns
        assert f"is_big_normal_as{s}" in out.columns
        assert f"prior_maru_max_len_as{s}" in out.columns
