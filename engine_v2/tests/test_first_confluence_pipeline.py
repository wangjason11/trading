"""Unit tests for the first_confluence trigger → MultiTFTrigger translator.

Pending-skip semantics moved to the orchestrator in §13.5.c.ii (the
wrapper `run_first_confluence_pipeline` was deleted with
`run_lower_tf_pipeline` since the entity-direct compute primitive
`apply_trigger_to_entity_df` consumes a `MultiTFTrigger` directly).
"""
from __future__ import annotations

import pandas as pd

from engine_v2.multitf.first_confluence_pipeline import to_multi_tf_trigger
from engine_v2.multitf.types import FirstConfluenceTrigger


def _h1_df() -> pd.DataFrame:
    """Minimal H1 df with 5 candles around idx 50."""
    rows = [
        {"time": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
         "o": 0.6000, "h": 0.6020, "l": 0.5980, "c": 0.6010, "volume": 100}
        for i in range(60)
    ]
    df = pd.DataFrame(rows)
    df.attrs["pair"] = "NZD_USD"
    # Override BOS extreme candle (idx 50) so the price reflects a clear high/low
    df.loc[50, "h"] = 0.6100
    df.loc[50, "l"] = 0.5900
    return df


def test_to_multi_tf_trigger_bullish_parent_uses_high():
    h1_df = _h1_df()
    trig = FirstConfluenceTrigger(
        parent_tf="H1", parent_sid=0, parent_cycle_id=1, parent_sd=1,
        input_idx=50, end_idx=58, trigger_event_idx=52,
        lifecycle_end_idx=200, status="finalized",
        meta={"bos_price": 0.6100},
    )
    out = to_multi_tf_trigger(trig, h1_df)
    assert out.use_case == "first_confluence"
    assert out.lower_sd == 1                      # confluence = same as parent
    assert out.lower_tf == "M15"
    assert out.parent_sid == 0
    assert out.parent_cycle_id == 1
    assert out.parent_sd == 1
    assert out.start_price == 0.6100              # bullish parent → BOS high
    assert out.lifecycle_end_idx == 200
    assert out.meta["probe_input_idx"] == 50
    assert out.meta["probe_end_idx"] == 58


def test_to_multi_tf_trigger_bearish_parent_uses_low():
    h1_df = _h1_df()
    trig = FirstConfluenceTrigger(
        parent_tf="H1", parent_sid=1, parent_cycle_id=2, parent_sd=-1,
        input_idx=50, end_idx=58, trigger_event_idx=52,
        lifecycle_end_idx=None, status="finalized",
    )
    out = to_multi_tf_trigger(trig, h1_df)
    assert out.lower_sd == -1                     # confluence with bearish parent
    assert out.start_price == 0.5900              # bearish parent → BOS low
    assert out.lifecycle_end_idx is None
