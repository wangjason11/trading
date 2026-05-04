"""`subsequent_confluence` (var 3) sub pipeline.

Per spec §4.3.4 / §4.3.7:
  Probe sd:        +parent_sd (confluence)
  Probe input_idx: parent-TF window extreme toward parent BOS
  Probe end_idx:   the CTS-prox trigger candle

Mapping (§4.3.1, unified rule `mapping_sd = -sub_sd`): for confluence
sub `lower_sd = +parent_sd`, so `mapping_sd = -parent_sd` — same as
first_confluence.

This module thinly wraps `run_lower_tf_pipeline` by translating the
`SubsequentConfluenceTrigger` into a generic `MultiTFTrigger`.

**Carve-out (3d, intentional):** spec §6.1 says a new var 3 sid should
overwrite the previous open confluence sub sid's df rows in place, with
the previous sid's events/zones marked `deactivated_by="overwritten_by_sid_N+1"`.
Implementing in-place overwrite touches entity df mutation infrastructure
that doesn't exist yet. For 3d we instead build var 3 results as
independent `LowerTFResult` entries appended to the M15.confluence
entity. Practical effect: M15.confluence has var 1 sids 0..N plus var 3
sids N+1..M, none overwriting each other. Overwrite semantics get a
later substep.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from engine_v2.multitf.lower_tf_pipeline import run_lower_tf_pipeline
from engine_v2.multitf.types import (
    LowerTFResult,
    MultiTFTrigger,
    SubsequentConfluenceTrigger,
)


def _to_multi_tf_trigger(
    trig: SubsequentConfluenceTrigger,
    parent_df: pd.DataFrame,
) -> MultiTFTrigger:
    """Translate a SubsequentConfluenceTrigger into a generic MultiTFTrigger."""
    input_idx = int(trig.input_idx)
    if trig.parent_sd == 1:
        # Bullish parent: input_idx is the lowest-low candle (extreme toward BOS)
        start_price = float(parent_df.loc[input_idx, "l"])
    else:
        start_price = float(parent_df.loc[input_idx, "h"])
    start_time = pd.to_datetime(parent_df.loc[input_idx, "time"], utc=True)

    return MultiTFTrigger(
        parent_tf=trig.parent_tf,
        parent_sid=trig.parent_sid,
        parent_cycle_id=trig.parent_cycle_id,
        parent_sd=trig.parent_sd,
        use_case="subsequent_confluence",
        lower_tf="M15",
        lower_sd=trig.parent_sd,        # confluence: same direction as parent
        start_time=start_time,
        start_price=start_price,
        lifecycle_end_idx=trig.lifecycle_end_idx,
        meta={
            "probe_input_idx": trig.input_idx,
            "probe_end_idx": trig.end_idx,
            "trigger_event_idx": trig.trigger_event_idx,
            "prior_sd_trigger_idx": trig.meta.get("prior_sd_trigger_idx"),
            "sequence_index_in_cycle": trig.meta.get("sequence_index_in_cycle"),
        },
    )


def run_subsequent_confluence_pipeline(
    trigger: SubsequentConfluenceTrigger,
    m15_df_prepared: pd.DataFrame,
    h1_df: pd.DataFrame,
) -> Optional[LowerTFResult]:
    """Build a `subsequent_confluence` sub from one var 3 trigger.

    Returns None if any probe / structure step fails (caught upstream by
    `run_lower_tf_pipeline`).
    """
    multi_tf = _to_multi_tf_trigger(trigger, h1_df)
    return run_lower_tf_pipeline(multi_tf, m15_df_prepared, h1_df)
