# GLOSSARY.md — Domain Terminology Reference

> Definitions for domain-specific terms used throughout the codebase. Refer here when encountering unfamiliar terminology.

---

## Structure Terms

| Term | Definition |
|------|------------|
| **struct_direction** | Regime direction: +1 for uptrend, -1 for downtrend |
| **CTS** | Continue the Structure — continuation swing confirmed by pullback |
| **BOS** | Break of Structure — break level confirmed by breakout |
| **range** | Consolidation period bounded by (range_hi, range_lo) |
| **reversal watch** | Monitoring window after close-break of BOS threshold |
| **structure_id** | Regime identifier; increments on reversal (sid 0, sid 1, etc.) |
| **cts_cycle_id** | Cycle counter within a structure_id (resets on reversal) |

---

## Zone Terms

| Term | Definition |
|------|------------|
| **base_pattern** | Candle pattern at zone anchor (pinbar, star, long-tail, etc.) |
| **base inside bar** | Pattern where anchor candle has ≥2 neighboring candles entirely within its range |
| **base_idx** | Index of the zone anchor candle |
| **anchor_idx** | Index of the BOS/CTS candle used for pattern identification |
| **bounds_steps** | History of zone boundary changes (INIT → expansions) |
| **outer threshold** | Zone boundary price (before top/bottom conversion) |
| **inner threshold** | Zone boundary price closer to current price |
| **KL zone** | Key Level zone — derived from CTS/BOS confirmation events |

---

## Timing Terms

| Term | Definition |
|------|------------|
| **apply_idx** | Candle where pattern effect is applied (end_idx or confirmation_idx) |
| **confirmed_idx** | Candle where zone becomes confirmed for charting |
| **confirmed_at** | Candle index in event meta where level was confirmed |
| **start_idx** | Logical start of a range or structure (may differ from confirm_idx) |
| **confirm_idx** | Candle index when event was emitted (may differ from start_idx) |
| **too_early** | Flag when identify_start finds extreme before min_history |

---

## Event Terms

| Term | Definition |
|------|------------|
| **event.idx** | Candle index where event was emitted |
| **event.meta** | Dictionary of event-specific metadata |
| **STATE_CHANGED** | Transition between market states (e.g., cts → bos) |
| **CTS_CONFIRMED** | Continue-the-structure level confirmed |
| **BOS_CONFIRMED** | Break-of-structure level confirmed |
| **RANGE_STARTED** | New consolidation range detected |
| **RANGE_UPDATED** | Range bounds expanded |
| **RANGE_RESET** | Range closed (breakout or reversal) |

---

## Wave Candle Terms

| Term | Definition |
|------|------------|
| **wave candle** | Candle at the boundary between two waves (ending wave → starting wave) at a KL zone |
| **last wave candle** | Final candle of the ending wave (last pullback for BOS, last breakout for CTS) |
| **first wave candle** | Initial candle of the starting wave (first breakout for BOS, first pullback for CTS) |
| **qualified candle** | Candle with correct direction AND vol_dir matching (vol_dir == candle dir or vol_dir == 0) |
| **compound first-wave** | First wave candle condition: (big_normal & (maru OR normal)) OR (big_maru & pinbar & pinbar_dir == wave_dir) |
| **BIB (base inside bar)** | Zone base pattern requiring special forward/backward search logic for wave candles |

---

## WVMI Terms

| Term | Definition |
|------|------------|
| **WVMI** | Wave Volume Momentum Indicator — measures BOS zone strength via volume ratios of wave candle pairs |
| **breakout momentum** | `(LB_vol * LB_weight) / FB_vol` — locked at CTS_n confirmation |
| **pullback momentum** | `(LP_vol * LP_weight) / FP_vol` — shifts until BOS_n+1 locks it |
| **last wave weight** | Weight (0.5/0.7/1.0) applied to LB/LP volume based on candle type and size |
| **temporary LP** | Last Pullback candle that shifts to qualified candle closest to outer bound; finalizes on BOS_n+1 |
| **buy_momentum / sell_momentum** | Direction-labeled wrappers: buy zone → buy=breakout, sell=pullback; sell zone → reversed |
| **vol_dir** | Volume direction column: +1 (buying pressure), -1 (selling pressure), 0 (neutral/no signal) |

---

## Scenario 3 Terms

| Term | Definition |
|------|------------|
| **Scenario 3** | Arbitrary-start structure analysis with iterative BOS_0 probe validation |
| **BOS_0 zone** | First BOS zone from the initial CTS_ESTABLISHED event; used for exception evaluation |
| **Phase 1** | Iterative probing: validate start_idx by checking if price reaches BOS_0 zone inner bound |
| **Phase 2** | Multi-structure continuation from finalized Phase 1 (same logic as `compute_structure`) |
| **finalized** | Scenario 3 status: probe validated, full structure analysis complete |
| **pending** | Scenario 3 status: insufficient data (no CTS_EST captured, or 1 CTS_EST + no `end_idx` to bound the exception check) |
| **pip_tolerance** | Distance threshold for exception evaluation near zone inner bound (per-TF defaults in `zones/zone_proximity.py::DEFAULT_PROBE_RESET_PIPS`: H1=3, M15=2.5, M5=2) |
| **subordinate probe** | `compute_structure_scenario_3` invocation via `multitf/lower_tf_pipeline.py::_run_subordinate_probe`. Validates a lower-TF sub structure's start by running a bounded Phase-1 probe on the parent TF. Direction is `trigger.lower_sd` — same as parent (`+parent_sd`) for confluence variants (var 1 / var 3), opposite (`-parent_sd`) for counter variants (var 2 / var 4). The legacy name "H1 reverse probe" applies only to counter direction; the function itself is direction-agnostic. |

---

## Pattern Terms

| Term | Definition |
|------|------------|
| **pinbar** | Single candle with long wick rejecting price level |
| **star** | Small-bodied candle indicating indecision |
| **long-tail** | Candle with extended tail showing price rejection |
| **maru** | Strong-bodied candle (marubozu-like) |
| **breakout pattern** | Multi-candle pattern signaling structure break |

---

## Imbalance Terms

| Term | Definition |
|------|------------|
| **imbalance candle** | A c2 (middle) candle whose neighbors form an FVG and whose direction matches the gap direction. Flagged per-candle via `is_imbalance`. |
| **imbalance instance** | One imbalance candle, or a run of consecutive same-direction imbalance candles, merged with combined gap bounds. Stored in `df.attrs["imbalances"]`. |
| **FVG** | Fair Value Gap — 3-candle gap pattern where c1 and c3 wicks don't overlap |
| **gap_top / gap_bottom** | Merged bounds of an imbalance instance (first c1.high to last c3.low for bullish; reversed for bearish) |
| **fill check** | An instance is "filled" when price retraces ≥70% into its merged gap within `(end_idx, check_to_idx]` |

---

## Pipeline Terms

| Term | Definition |
|------|------------|
| **orchestrator** | Central pipeline coordinator (pipeline/orchestrator.py) |
| **feature** | Computed column added to dataframe (e.g., swing detection) |
| **base features** | Foundation features required by market structure |
| **charting** | Final pipeline stage that generates visualizations |

---

## Zone Proximity Terms

| Term | Definition |
|------|------------|
| **zone proximity trigger** | A candle where price wicks within `proximity_pips` of a zone's inner bound. Computed by `check_zone_proximity` in `zones/zone_proximity.py`. |
| **sd zone** | A zone in the structure direction. For sd=+1: buy-side zones (BOS KL + POI). For sd=-1: sell-side (BOS KL + POI). POI zones are always sd by Fib construction (Fib spans BOS→CTS). |
| **opp_sd zone** | A zone opposite to the structure direction. For sd=+1: sell-side (CTS KL only). For sd=-1: buy-side (CTS KL only). There are no opp_sd POIs. |
| **triggered_by_event_idx** | The candle index of the first sd zone proximity trigger per cycle (Part 4 §8.7 attribution schema). Stored in WVMIRecord.meta. The corresponding MultiTFTrigger.meta uses `probe_end_idx`. |
| **V / lambda movement** | Alternating retracement pattern within a structure cycle: price moves toward sd zone, bounces, moves toward opp_sd zone, bounces back, etc. The zone-proximity state machine captures each leg. |
| **proximity_pips** | The pip threshold for zone proximity detection. Defaults: H1=9, M15=6, M5=3. Caller-overridable per call. |
| **narrow-gap cycle** | A cycle where `|cts_threshold − bos_threshold|` is below the per-TF `min_gap_pips` threshold (H1=50, M15=30, M5=15). In narrow cycles: sd-proximity cannot confirm CTS (Rule 1); proximity scan only fires after pullback CTS_confirmed (Rule 2); ≤1 sd + ≤1 opp_sd trigger total (Rule 3). |
| **wide-gap cycle** | A cycle where the gap is ≥ `min_gap_pips`. Default proximity behavior applies (sd can confirm CTS, unlimited alternation). |
| **mid-cycle crossing** | The transition narrow → wide that can happen at most once per cycle (gap is monotonically non-decreasing). After crossing, narrow-mode caps lift and alternation continues seamlessly from the cap state at the crossing candle. |
| **min_gap_pips** | The per-TF narrow-cycle threshold (H1=50, M15=30, M5=15) in `DEFAULT_MIN_GAP_FOR_REPEATED_PROXIMITY_PIPS`. Invariant: `min_gap_pips[tf] > proximity_pips[tf]`. |

---

## CTS Confirmation Terms

| Term | Definition |
|------|------------|
| **confirmation_method** | How a CTS got confirmed — either `"pullback"` (valid pullback pattern fired) or `"sd_zone_proximity"` (price wick reached within proximity threshold of the BOS inner before any pullback). Stored on `CTS_CONFIRMED.meta` and propagated to the CTS KL zone. |
| **CTS_RECONFIRMED** | Event emitted at the pullback idx when CTS was originally confirmed via sd zone proximity AND a valid pullback later fires. The original `CTS_CONFIRMED` event stays at the proximity idx (append-only). The CTS zone meta gets upgraded to `"pullback"` with `pb_reconfirm_idx` recorded. |
| **pb_reconfirm_idx** | Idx of the pullback that re-affirmed a proximity-confirmed CTS. Logged on the CTS KL zone meta. Only present when `confirmation_method` was originally `"sd_zone_proximity"` and a pullback fired afterward. |
| **proximity-confirmed CTS** | Shorthand for a CTS whose `confirmation_method == "sd_zone_proximity"`. Such cycles can complete WITHOUT a valid pullback pattern firing (the cycle still progresses to next BOS via breakout pattern; BOS_n+1 = max retracement in `[cts_confirmed_idx, breakout_idx]`). |

---

## Sub-Structure Pool Terms (model decided 2026-09-19; canonical: `memory/project_sub_structure_pool_architecture.md`)

| Term | Definition |
|------|------------|
| **unique sub** | One M15 (sub-TF) market structure, identified by `(parent_path, sub_TF, direction, starting_idx)` with ABSOLUTE `direction`; `sub_id` is its global monotonic id. Computed once (geometry to the data edge), has ONE lifecycle `[start_idx, end_idx]`, **spans parent cycles and parent sids**, and is what we trade on. |
| **TriggerRecord** | A triggered *instance* of a unique sub: one trigger that resolved to it. Identity `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)`. Has its OWN parent-bound lifecycle; feeds the unique sub's lifecycle but does not equal it. `sub_id` is never None on a record. |
| **trigger_sub_sid** | Per-`(lens, parent_sid, parent_cycle_id)` counter, +1 each time a trigger in that scope resolves to a *new* unique sub (two triggers resolving to the same sub share one record). Replaces the old per-parent-cycle `sub_sid`. |
| **sub_id** | Global unique-sub id. Stamped on every structural artifact (events, KL/POI/fib/WVMI meta, `SidRecord`). Replaces `sub_sid` on those objects (hard rename). |
| **lens** | Which chart a record attributes its sub to: `confluence` / `counter`. Named triggers map by use case; a `reversal` record inherits the lens of the record whose sub reversed (sticky-per-chart). A unique sub is drawn on every lens it has a non-zero-length record on, over the SUB's window. |
| **relative_dir** | `confluence` iff `direction == parent_sd` of the record's parent cycle, else `counter`. A *semantic* label distinct from `lens` — a sub can be `relative_dir=counter` yet render on the confluence chart (reversal-stickiness). On the unique sub it is a step function over record handovers. |
| **trigger_idx** | Candle the trigger fired (last-of-hour map of the parent candle; native M15 for `reversal`). Historical, not a lifecycle value. |
| **probe_finalize_idx** | `ProbeResult.finalize_idx` as-is — when the probe's decision became final. Native M15 or a mapped parent value depending on `finalize_condition` (GOTCHAS). Historical. |
| **start_idx (record)** | `max(probe_finalize_idx, parent-cycle floor)` — the TRUE real-time lifecycle start. `start_idx − probe_finalize_idx` measures how retroactive the structure was. |
| **trigger_end_idx / end_idx (record)** | `trigger_end_idx` = first of {own reversal, same-direction replacement, parent end} (or the sub's frozen end on a post-end re-trigger); `end_idx = max(trigger_end_idx, start_idx)`. Set once, never updated. |
| **same-direction replacement** | A new record in the same `(lens, parent_sid, parent_cycle_id, sub_TF, direction)` mapping to a DIFFERENT unique sub ends the active record at the new record's `start_idx`. Same-lens only; ≤1 active record per `(lens, parent, direction)`. |
| **degenerate parent cycle** | Parent cycle whose lifecycle floor ≥ lifecycle end (zero/negative length; caused by the reversal handoff on a retroactive parent). Triggers inside it go to the unresolved-trigger log; nothing is built. |
| **zero-length record** | A `TriggerRecord` with `end_idx == start_idx` (post-end re-trigger, or own reversal before the floored start). Has a `sub_id`; participates in nothing but the log. |
| **unresolved-trigger log** | Separate table of triggers that never became records: `reason ∈ {pending, degenerate_parent_cycle, probe_failed}`. No `sub_id`, no `trigger_sub_sid`. |
| **`cts_anchor_idx` (parent)** | On `CTS_CONFIRMED.meta`: the H1 candle holding the parent CTS **extreme** at confirmation (migrates via `CTS_UPDATED`; ≠ `CTS_ESTABLISHED.idx`). `first_confluence`'s probe bound, price-mapped to M15. |
| **`cts0_anchor` (probe)** | Inside `unified_probe` Phase 2: the probe's OWN M15 `CTS_0_CONFIRMED.cts_anchor_idx`, which caps the retrace window at `cts0_anchor − 1`. A different object from the parent's `cts_anchor_idx` despite the name. |
