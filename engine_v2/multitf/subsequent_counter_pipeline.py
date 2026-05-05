"""`subsequent_counter` (var 4) sub pipeline.

Per spec §4.3.5 / §4.3.7:
  Probe sd:        -parent_sd (counter — opposite direction to parent)
  Probe input_idx: parent-TF window extreme toward parent CTS (Λ apex /
                   V trough)
  Probe end_idx:   the sd-prox trigger candle

Mapping (§4.3.1, unified rule `mapping_sd = -sub_sd`): for counter sub
`lower_sd = -parent_sd`, so `mapping_sd = +parent_sd` — same as
first_counter.

This module thinly wraps `run_lower_tf_pipeline` by translating the
`SubsequentCounterTrigger` into a generic `MultiTFTrigger`.

**Carve-out (3d.iv, intentional):** spec §6.1 says a new var 4 sid should
overwrite the previous open counter sub sid's df rows in place, with the
previous sid's events/zones marked
`deactivated_by="overwritten_by_sid_N+1"`. Implementing in-place
overwrite touches entity df mutation infrastructure that doesn't exist
yet. For 3d.iv we instead build var 4 results as independent
`LowerTFResult` entries appended to the M15.counter entity. Practical
effect: M15.counter has var 2 sids (one per parent cycle that produced a
first_counter) plus a single var 4 sid per parent cycle (last var 4 only,
mirroring the var 3 carve-out from 3d). Both carve-outs (var 3 + var 4
last-per-cycle) get removed together when §6.1 in-place overwrite lands.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from engine_v2.multitf.lower_tf_pipeline import run_lower_tf_pipeline
from engine_v2.multitf.types import (
    LowerTFResult,
    MultiTFTrigger,
    SubsequentCounterTrigger,
)


def _to_multi_tf_trigger(
    trig: SubsequentCounterTrigger,
    parent_df: pd.DataFrame,
) -> MultiTFTrigger:
    """Translate a SubsequentCounterTrigger into a generic MultiTFTrigger."""
    input_idx = int(trig.input_idx)
    if trig.parent_sd == 1:
        # Bullish parent: input_idx is the highest-high candle (Λ apex toward CTS)
        start_price = float(parent_df.loc[input_idx, "h"])
    else:
        # Bearish parent: input_idx is the lowest-low candle (V trough toward CTS)
        start_price = float(parent_df.loc[input_idx, "l"])
    start_time = pd.to_datetime(parent_df.loc[input_idx, "time"], utc=True)

    return MultiTFTrigger(
        parent_tf=trig.parent_tf,
        parent_sid=trig.parent_sid,
        parent_cycle_id=trig.parent_cycle_id,
        parent_sd=trig.parent_sd,
        use_case="subsequent_counter",
        lower_tf="M15",
        lower_sd=-trig.parent_sd,       # counter: opposite direction to parent
        start_time=start_time,
        start_price=start_price,
        lifecycle_end_idx=trig.lifecycle_end_idx,
        meta={
            "probe_input_idx": trig.input_idx,
            "probe_end_idx": trig.end_idx,
            "trigger_event_idx": trig.trigger_event_idx,
            "prior_sd_trigger_idx": trig.meta.get("prior_sd_trigger_idx"),
            "prior_cts_prox_idx": trig.meta.get("prior_cts_prox_idx"),
            "sequence_index_in_cycle": trig.meta.get("sequence_index_in_cycle"),
        },
    )


def run_subsequent_counter_pipeline(
    trigger: SubsequentCounterTrigger,
    m15_df_prepared: pd.DataFrame,
    h1_df: pd.DataFrame,
) -> Optional[LowerTFResult]:
    """Build a `subsequent_counter` sub from one var 4 trigger.

    Returns None if any probe / structure step fails (caught upstream by
    `run_lower_tf_pipeline`).
    """
    multi_tf = _to_multi_tf_trigger(trigger, h1_df)
    return run_lower_tf_pipeline(multi_tf, m15_df_prepared, h1_df)
