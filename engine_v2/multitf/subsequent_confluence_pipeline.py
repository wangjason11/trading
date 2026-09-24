"""`subsequent_confluence` (var 3) typed-trigger → MultiTFTrigger translator.

Per spec §4.3.4 / §4.3.7:
  Probe sd:        +parent_sd (confluence)
  Probe input_idx: parent-TF window extreme toward parent BOS
  Probe end_idx:   the CTS-prox trigger candle

Mapping (§4.3.1, unified rule `mapping_sd = -sub_sd`): for confluence
sub `lower_sd = +parent_sd`, so `mapping_sd = -parent_sd` — same as
first_confluence.

§13.5.c.ii: `run_subsequent_confluence_pipeline` deleted with
`run_lower_tf_pipeline`. The orchestrator now translates and applies via
`apply_trigger_to_entity_df`. Spec §6.1 in-place overwrite semantics now
fire for real (cascade tags previous sid on new build); the
`var3_last_per_cycle` carve-out remains until §13.5.d.
"""
from __future__ import annotations

import pandas as pd

from engine_v2.multitf.types import (
    MultiTFTrigger,
    SubsequentConfluenceTrigger,
)


def to_multi_tf_trigger(
    trig: SubsequentConfluenceTrigger,
    parent_df: pd.DataFrame,
) -> MultiTFTrigger:
    """Translate a SubsequentConfluenceTrigger into a generic MultiTFTrigger."""
    return MultiTFTrigger(
        parent_tf=trig.parent_tf,
        parent_sid=trig.parent_sid,
        parent_cycle_id=trig.parent_cycle_id,
        parent_sd=trig.parent_sd,
        use_case="subsequent_confluence",
        lower_tf="M15",
        lower_sd=trig.parent_sd,        # confluence: same direction as parent
        meta={
            "probe_input_idx": trig.input_idx,
            "trigger_event_idx": trig.trigger_event_idx,
            "prior_sd_trigger_idx": trig.meta.get("prior_sd_trigger_idx"),
            "sequence_index_in_cycle": trig.meta.get("sequence_index_in_cycle"),
        },
    )


