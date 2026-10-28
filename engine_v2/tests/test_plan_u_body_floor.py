"""PLAN_U — the pinbar body floor as a per-candle input (U1, 2026-10-28).

Pinned here: the frame-level wiring (`apply_candle_classification` → `compute_candle_features` → the floor on
every row + the three columns), and the candle hover's body-floor line on the H1 and M15 charts (CHARTING_SPEC §15).
The classifier's step 4 itself: `test_candles_v2.py`; the stacked legs: `test_institutional_pattern.py`.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine_v2.charting.export_plotly import _candle_floor_hover
from engine_v2.features.candle_classifier import apply_candle_classification
from engine_v2.tests import test_chart_controls as CC
from engine_v2.tests import test_wvmi_lp_marks as W


def _raw(rows, pair="NZD_USD"):
    df = pd.DataFrame(rows)
    df.insert(0, "time", pd.date_range("2026-01-05", periods=len(df), freq="h", tz="UTC"))
    df["volume"] = 1.0
    df.attrs["pair"] = pair
    return df


_TINY_MARU = {"o": 0.60000, "h": 0.60022, "l": 0.59999, "c": 0.60020}    # a 2.0-pip body, ~87 % of the range
_BIG_MARU = {"o": 0.60000, "h": 0.60052, "l": 0.59999, "c": 0.60050}     # a 5.0-pip body


@pytest.mark.parametrize("tf", ["H1", "M15"])
def test_a_frame_without_a_window_has_no_floor_on_any_timeframe(tf):
    """Since U2 the floor is the rolling one, the same rule on every TF (no per-TF table): a 2-candle frame without
    a seed has no window → no floor, the body-% types (the rolling floor: `test_plan_u_rolling_floor.py`)."""
    out = apply_candle_classification(_raw([_TINY_MARU, _BIG_MARU]), timeframe=tf).df
    assert out["pinbar_body_floor"].isna().all()
    assert list(out["candle_type"]) == list(out["candle_type_before_floor"]) == ["maru", "maru"]
    assert not out["is_floor_pinbar"].any()


def test_an_explicit_body_floor_reaches_every_candle():
    """The `body_floor` seam of `apply_candle_classification` (PLAN_U §8 A5): one floor per row."""
    out = apply_candle_classification(_raw([_TINY_MARU, _BIG_MARU]), body_floor=[0.00022, 0.00022]).df
    assert list(out["pinbar_body_floor"]) == pytest.approx([0.00022, 0.00022])
    assert list(out["candle_type"]) == ["pinbar", "maru"]
    assert list(out["is_floor_pinbar"]) == [True, False]


def test_candle_floor_hover_lines():
    dfx = pd.DataFrame({"pinbar_body_floor": [0.00022, 0.000283, float("nan")],
                        "is_floor_pinbar": [False, True, False],
                        "candle_type_before_floor": ["maru", "normal", "pinbar"]})
    assert _candle_floor_hover(dfx) == ["pinbar_body_floor=0.000220",
                                        "pinbar_body_floor=0.000283 | candle_type_before_floor=normal",
                                        "pinbar_body_floor=NA (no floor)"]
    assert _candle_floor_hover(pd.DataFrame({"x": [1]})) == ["pinbar_body_floor=NA (no floor)"]


_OHLCV = W._ohlcv


def _with_floor_columns(n, freq):
    """The chart tests' frame + the floor columns: candle 0 no floor, 2 a pinbar by the floor (was normal)."""
    df = _OHLCV(n, freq)
    df["pinbar_body_floor"] = [float("nan")] + [0.00022] * (n - 1)
    df["is_floor_pinbar"] = [i == 2 for i in range(n)]
    df["candle_type_before_floor"] = "normal"
    return df


def _candle_rows(fig):
    rows = {}
    for t in fig.data:
        if t.type == "candlestick":
            for cd in t.customdata:
                rows[int(cd[0])] = (t.hovertemplate, cd)
    return rows


@pytest.mark.parametrize("chart,slot", [("H1", 12), ("M15", 14)])
def test_the_candle_hover_shows_the_floor_line(monkeypatch, tmp_path, chart, slot):
    monkeypatch.setattr(W, "_ohlcv", _with_floor_columns)
    fig, _ = (CC._render_h1 if chart == "H1" else CC._render_m15)(monkeypatch, tmp_path)
    rows = _candle_rows(fig)
    template, cd = rows[2]
    assert f"candle_type=%{{customdata[3]}}<br>%{{customdata[{slot}]}}<br>" in template
    assert cd[slot] == "pinbar_body_floor=0.000220 | candle_type_before_floor=normal"
    assert rows[1][1][slot] == "pinbar_body_floor=0.000220"
    assert rows[0][1][slot] == "pinbar_body_floor=NA (no floor)"
