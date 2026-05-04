"""Unit tests for subsequent_confluence pipeline wrapper."""
from __future__ import annotations

import pandas as pd

from engine_v2.multitf.subsequent_confluence_pipeline import _to_multi_tf_trigger
from engine_v2.multitf.types import SubsequentConfluenceTrigger


def _h1_df() -> pd.DataFrame:
    rows = []
    for i in range(60):
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "o": 0.6010, "h": 0.6020, "l": 0.5990, "c": 0.6010, "volume": 100,
        })
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    df.loc[42, "l"] = 0.5950   # bullish parent: input is the lowest low
    df.loc[42, "h"] = 0.6080   # bearish parent would pick highest high
    return df


def test_to_multi_tf_trigger_bullish_parent_uses_low():
    h1 = _h1_df()
    trig = SubsequentConfluenceTrigger(
        parent_tf="H1", parent_sid=0, parent_cycle_id=1, parent_sd=1,
        input_idx=42, end_idx=50, trigger_event_idx=50,
        lifecycle_end_idx=200,
        meta={"prior_sd_trigger_idx": 30, "sequence_index_in_cycle": 1},
    )
    out = _to_multi_tf_trigger(trig, h1)
    assert out.use_case == "subsequent_confluence"
    assert out.lower_sd == 1                    # confluence with bullish parent
    assert out.lower_tf == "M15"
    assert out.parent_cycle_id == 1
    assert out.start_price == 0.5950            # bullish → lowest low
    assert out.lifecycle_end_idx == 200
    assert out.meta["probe_input_idx"] == 42
    assert out.meta["probe_end_idx"] == 50
    assert out.meta["prior_sd_trigger_idx"] == 30


def test_to_multi_tf_trigger_bearish_parent_uses_high():
    h1 = _h1_df()
    trig = SubsequentConfluenceTrigger(
        parent_tf="H1", parent_sid=1, parent_cycle_id=2, parent_sd=-1,
        input_idx=42, end_idx=55, trigger_event_idx=55,
        lifecycle_end_idx=None,
    )
    out = _to_multi_tf_trigger(trig, h1)
    assert out.lower_sd == -1
    assert out.start_price == 0.6080            # bearish → highest high
    assert out.lifecycle_end_idx is None
