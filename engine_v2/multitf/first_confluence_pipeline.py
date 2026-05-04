"""`first_confluence` (var 1) sub pipeline.

Per spec §4.3.2:
  Probe sd:        +parent_sd  (confluence — same direction as parent)
  Probe input_idx: parent BOS extreme idx (= BOS_CONFIRMED.ev.idx)
  Probe end_idx:   parent CTS_CONFIRMED idx in the same parent cycle

Mapping (§4.3.1, unified rule `mapping_sd = -sub_sd`): for confluence
sub `lower_sd = +parent_sd`, so `mapping_sd = -parent_sd` — the M15
candle within the H1 BOS hour with the lowest low (bullish parent) or
highest high (bearish parent).

This module thinly wraps `run_lower_tf_pipeline` by translating the
`FirstConfluenceTrigger` into a generic `MultiTFTrigger`. The probe and
M15 build re-use the existing machinery.

Pending triggers (parent CTS not yet confirmed) are skipped per spec
§4.3.2 / §14: "no first_confluence sub is built until parent
CTS_CONFIRMED resolves end_idx."
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from engine_v2.multitf.lower_tf_pipeline import run_lower_tf_pipeline
from engine_v2.multitf.types import (
    FirstConfluenceTrigger,
    LowerTFResult,
    MultiTFTrigger,
)


def _to_multi_tf_trigger(
    trig: FirstConfluenceTrigger,
    parent_df: pd.DataFrame,
) -> MultiTFTrigger:
    """Translate a FirstConfluenceTrigger into a generic MultiTFTrigger."""
    bos_idx = int(trig.input_idx)
    if trig.parent_sd == 1:
        start_price = float(parent_df.loc[bos_idx, "h"])
    else:
        start_price = float(parent_df.loc[bos_idx, "l"])
    start_time = pd.to_datetime(parent_df.loc[bos_idx, "time"], utc=True)

    return MultiTFTrigger(
        parent_tf=trig.parent_tf,
        parent_sid=trig.parent_sid,
        parent_cycle_id=trig.parent_cycle_id,
        parent_sd=trig.parent_sd,
        use_case="first_confluence",
        lower_tf="M15",
        lower_sd=trig.parent_sd,        # confluence: same direction as parent
        start_time=start_time,
        start_price=start_price,
        lifecycle_end_idx=trig.lifecycle_end_idx,
        meta={
            "probe_input_idx": trig.input_idx,
            "probe_end_idx": trig.end_idx,
            "trigger_event_idx": trig.trigger_event_idx,
            "bos_price": trig.meta.get("bos_price"),
        },
    )


def run_first_confluence_pipeline(
    trigger: FirstConfluenceTrigger,
    m15_df_prepared: pd.DataFrame,
    h1_df: pd.DataFrame,
) -> Optional[LowerTFResult]:
    """Build a `first_confluence` sub for one parent BOS_CONFIRMED.

    Returns None if the trigger is pending or any probe / structure step
    fails (caught upstream by `run_lower_tf_pipeline`).
    """
    if trigger.status != "finalized" or trigger.end_idx is None:
        print(f"[first_confluence] SKIP pending: "
              f"sid={trigger.parent_sid} cycle={trigger.parent_cycle_id}")
        return None

    multi_tf = _to_multi_tf_trigger(trigger, h1_df)
    return run_lower_tf_pipeline(multi_tf, m15_df_prepared, h1_df)
