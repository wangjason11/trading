"""`first_confluence` (var 1) typed-trigger → MultiTFTrigger translator.

Per spec §4.3.2:
  Probe sd:        +parent_sd  (confluence — same direction as parent)
  Parent input:    meta `parent_input_idx` (H1) = the parent BOS anchor
                   (BOS_CONFIRMED.meta["bos_anchor_idx"]); the resolver
                   price-maps it into the M15 probe input (PLAN_E §9.2)
  Probe end:       meta `parent_cts_anchor_idx` (H1) = the confirmed CTS's
                   anchor (`cts_anchor_idx`) in the same parent cycle (resolved
                   once parent CTS_CONFIRMED fires — not the confirmation
                   candle); the resolver price-maps it into the M15 probe bound
                   `probe_end_idx` (Plan E Post-E·3 renamed the H1 key)

Mapping (§4.3.1, unified rule `mapping_sd = -sub_sd`): for confluence
sub `lower_sd = +parent_sd`, so `mapping_sd = -parent_sd` — the M15
candle within the H1 BOS hour with the lowest low (bullish parent) or
highest high (bearish parent).

§13.5.c.ii: the wrapper `run_first_confluence_pipeline` was deleted with
`run_lower_tf_pipeline`. The orchestrator now translates each typed
trigger via `to_multi_tf_trigger` and feeds the resulting
`MultiTFTrigger` to `apply_trigger_to_entity_df`. Pending-skip semantics
for §4.3.2 / §14 moved to the orchestrator (it filters
`status=="finalized"` before sorting + applying).
"""
from __future__ import annotations

import pandas as pd

from engine_v2.multitf.types import FirstConfluenceTrigger, MultiTFTrigger


def to_multi_tf_trigger(
    trig: FirstConfluenceTrigger,
    parent_df: pd.DataFrame,
) -> MultiTFTrigger:
    """Translate a FirstConfluenceTrigger into a generic MultiTFTrigger.
    (`parent_df` is unused since the unread `start_time` / `start_price` were
    deleted in Plan E E1b; the translator signature is shared by all three.)"""
    return MultiTFTrigger(
        parent_tf=trig.parent_tf,
        parent_sid=trig.parent_sid,
        parent_cycle_id=trig.parent_cycle_id,
        parent_sd=trig.parent_sd,
        use_case="first_confluence",
        lower_tf="M15",
        lower_sd=trig.parent_sd,        # confluence: same direction as parent
        meta={
            "parent_input_idx": trig.input_idx,
            "parent_cts_anchor_idx": trig.parent_cts_anchor_idx,
            "trigger_event_idx": trig.trigger_event_idx,
            "bos_price": trig.meta.get("bos_price"),
        },
    )


