"""`first_confluence` (var 1) typed-trigger → MultiTFTrigger translator.

Per spec §4.3.2:
  Probe sd:        +parent_sd  (confluence — same direction as parent)
  Probe input_idx: parent BOS extreme idx (= BOS_CONFIRMED.ev.idx)
  Probe end_idx:   the confirmed CTS's extreme idx (`cts_anchor_idx`) in the
                   same parent cycle (resolved once parent CTS_CONFIRMED fires —
                   not the confirmation candle)

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


