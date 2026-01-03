import pandas as pd
import pytest

from engine_v2.features.candle_params import CandleParams
from engine_v2.features.candles_v2 import (
    compute_candle_metrics,
    classify_candles,
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
