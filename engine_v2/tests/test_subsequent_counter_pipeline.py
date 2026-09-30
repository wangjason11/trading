"""Unit tests for subsequent_counter pipeline wrapper."""
from __future__ import annotations

import pandas as pd

from engine_v2.multitf.subsequent_counter_pipeline import to_multi_tf_trigger
from engine_v2.multitf.types import SubsequentCounterTrigger


def _h1_df() -> pd.DataFrame:
    rows = []
    for i in range(60):
        rows.append({
            "time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            "o": 0.6010, "h": 0.6020, "l": 0.5990, "c": 0.6010, "volume": 100,
        })
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    df.loc[42, "h"] = 0.6080  # bullish parent: input is the highest high (Λ apex)
    df.loc[42, "l"] = 0.5950  # bearish parent: input is the lowest low (V trough)
    return df


def test_to_multi_tf_trigger_bullish_parent():
    h1 = _h1_df()
    trig = SubsequentCounterTrigger(
        parent_tf="H1", parent_sid=0, parent_cycle_id=1, parent_sd=1,
        input_idx=42, trigger_event_idx=70,
        meta={
            "prior_sd_trigger_idx": 30,
            "prior_cts_prox_idx": 50,
            "sequence_index_in_cycle": 2,
        },
    )
    out = to_multi_tf_trigger(trig, h1)
    assert out.use_case == "subsequent_counter"
    assert out.lower_sd == -1                   # counter to bullish parent
    assert out.lower_tf == "M15"
    assert out.parent_cycle_id == 1
    assert out.meta["parent_input_idx"] == 42
    assert "probe_input_idx" not in out.meta    # renamed, no alias (PLAN_E §9.2)
    assert "probe_end_idx" not in out.meta     # unread; deleted in Plan E E1b
    assert out.meta["prior_sd_trigger_idx"] == 30
    assert out.meta["prior_cts_prox_idx"] == 50


def test_to_multi_tf_trigger_bearish_parent():
    h1 = _h1_df()
    trig = SubsequentCounterTrigger(
        parent_tf="H1", parent_sid=1, parent_cycle_id=2, parent_sd=-1,
        input_idx=42, trigger_event_idx=80,
    )
    out = to_multi_tf_trigger(trig, h1)
    assert out.lower_sd == 1                    # counter to bearish parent
    assert out.meta["parent_input_idx"] == 42
    assert "probe_input_idx" not in out.meta    # renamed, no alias (PLAN_E §9.2)
