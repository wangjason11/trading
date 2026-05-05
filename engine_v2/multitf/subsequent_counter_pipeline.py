"""`subsequent_counter` (var 4) typed-trigger → MultiTFTrigger translator.

Per spec §4.3.5 / §4.3.7:
  Probe sd:        -parent_sd (counter — opposite direction to parent)
  Probe input_idx: parent-TF window extreme toward parent CTS (Λ apex /
                   V trough)
  Probe end_idx:   the sd-prox trigger candle

Mapping (§4.3.1, unified rule `mapping_sd = -sub_sd`): for counter sub
`lower_sd = -parent_sd`, so `mapping_sd = +parent_sd` — same as
first_counter.

§13.5.c.ii: `run_subsequent_counter_pipeline` deleted with
`run_lower_tf_pipeline`. The orchestrator now translates and applies via
`apply_trigger_to_entity_df`. Spec §6.1 in-place overwrite fires for
real; the `var4_last_per_cycle` carve-out remains until §13.5.d.
"""
from __future__ import annotations

import pandas as pd

from engine_v2.multitf.types import (
    MultiTFTrigger,
    SubsequentCounterTrigger,
)


def to_multi_tf_trigger(
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


