"""`subsequent_counter` (var 4) typed-trigger → MultiTFTrigger translator.

Per spec §4.3.5 / §4.3.7:
  Probe sd:        -parent_sd (counter — opposite direction to parent)
  Parent input:    meta `parent_input_idx` (H1) = the parent-TF window
                   extreme toward parent CTS (Λ apex / V trough) —
                   informational: the resolver co-sources its M15 input
                   from the sibling CTS (§4.3.4)
  Probe end_idx:   the sd-prox trigger candle

Mapping (§4.3.1): none — the H1 input is never mapped; the resolver
co-sources its M15 input from the sibling confluence CTS. Only the §4.3.4
fallback uses the `-sub_sd` side: the M15 extreme on the `-lower_sd`
(= `+parent_sd`) side of the sibling-read window `[lo, hi]`.

§13.5.c.ii: `run_subsequent_counter_pipeline` deleted with
`run_lower_tf_pipeline`. The per-trigger apply path
(`apply_trigger_to_entity_df`) and the `var4_last_per_cycle` carve-out are
gone too (the 2026-05-25 lifecycle redesign, then the sub-structure pool —
PART4 §17): every trigger feeds the pool sweep.
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
    return MultiTFTrigger(
        parent_tf=trig.parent_tf,
        parent_sid=trig.parent_sid,
        parent_cycle_id=trig.parent_cycle_id,
        parent_sd=trig.parent_sd,
        use_case="subsequent_counter",
        lower_tf="M15",
        lower_sd=-trig.parent_sd,       # counter: opposite direction to parent
        meta={
            "parent_input_idx": trig.input_idx,
            "trigger_event_idx": trig.trigger_event_idx,
            "prior_sd_trigger_idx": trig.meta.get("prior_sd_trigger_idx"),
            "prior_cts_prox_idx": trig.meta.get("prior_cts_prox_idx"),
            "sequence_index_in_cycle": trig.meta.get("sequence_index_in_cycle"),
        },
    )


