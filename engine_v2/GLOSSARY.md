# GLOSSARY.md — Domain Terminology Reference

> Definitions for domain-specific terms used throughout the codebase. Refer here when encountering unfamiliar terminology.

---

## Structure Terms

| Term | Definition |
|------|------------|
| **struct_direction** | Regime direction: +1 for uptrend, -1 for downtrend |
| **CTS** | Continue the Structure — continuation swing established by a breakout pattern and confirmed by a pullback (or by sd-zone proximity — see "CTS Confirmation Terms") |
| **BOS** | Break of Structure — break level confirmed by breakout |
| **range** | Consolidation period bounded by (range_hi, range_lo) |
| **reversal watch** | Monitoring window after close-break of BOS threshold |
| **structure_id** | Regime identifier; increments on reversal (sid 0, sid 1, etc.) |
| **cts_cycle_id** | Cycle counter within a structure_id (resets on reversal) |

---

## Naming Standard (candle indices) — CANONICAL, adopted 2026-09-23

Every name that holds a candle index says WHICH kind of candle it is. The test, in the user's words:
**"Does it serve as an endpoint in our structure? Or are we simply trying to record an extreme price?"**
A candle can be both an anchor and an extreme — the name follows the PURPOSE of the log, never the
coincidence.

| Kind | Spelling | Meaning |
|------|----------|---------|
| **Moment** | past participle `*_established_idx`, `*_confirmed_idx`; `*_at` stamps (`confirmed_at`, `ImbalanceInstance.formed_at`) and parameters (`evaluated_at`); `apply_idx` | the candle at which something became knowable / took effect — real-time; the ONLY kind a timing or lifecycle read may use |
| **Pattern anchor** | `pattern_anchor_idx` (first candle), `pattern_end_idx` (last candle) | *Candle-pattern realm* (breakout / pullback / reversal patterns; the MS scan candle; the reversal close-break candle): "anchor" ALWAYS means the pattern's FIRST candle |
| **Market-structure anchor** | `*_anchor_idx` (`cts_anchor_idx`, `bos_anchor_idx`, KL-zone / fib anchors, a structure's start) | *Market-structure realm* (BOS/CTS of a cycle, zones, fibs, structures, subs): an ENDPOINT — start or end — of a structure element. Always a location, never a moment |
| **Extreme** | `*_extreme_idx` / `*_extreme_price`; `pattern_extreme_idx` / `pattern_extreme_price` | a recorded price extreme that is NOT serving as an endpoint at that site: window / retrace searches, running extremes, "is this a new extreme" checks, lower-TF price-mapping, candle anatomy, and the extreme reached INSIDE a pattern (from its first candle through its apply candle) |
| **Bare element idx** | `bos_idx`, `cts_idx` on a record that pairs them with a price (FibState / fib_lifecycle.csv, POI meta, final.csv, cycle-0 dicts) | that element's anchor |

Status: many existing names do not follow this yet — they move in staged, one-cause-per-`/compare`
migrations. Frozen event names are bridged by `ARCHITECTURE.md` "`ev.idx` convention". Older prose that
calls `CTS_ESTABLISHED.idx` / `BOS_CONFIRMED.idx`, `CTS_CONFIRMED.meta["cts_anchor_idx"]` or a KL-zone
`meta["anchor_idx"]` "the CTS/BOS **extreme** (at confirmation)" means the CTS/BOS **anchor**. Known
mismatch still in code: the fib meta `activated_at` currently holds the CTS anchor, not a moment (fixed in the
fib-timing stage — `FIB_LIFECYCLE_SPEC.md` §15.3).

---

## Zone Terms

| Term | Definition |
|------|------------|
| **base_pattern** | Candle pattern at zone anchor (pinbar, star, long-tail, etc.) |
| **base inside bar** | Pattern where anchor candle has ≥2 neighboring candles entirely within its range |
| **base_idx** | The FIRST candle of the zone's base pattern (its pattern-realm anchor; where the KL rectangle starts). A different field from the zone's market-structure `anchor_idx`: equal to it for inside-bar / 1-candle / catch-all bases, `anchor_idx − 1` for star bases and some 2-candle bases (equal on 31 of 39 zones on the reference window). |
| **anchor_idx** | The bare key exists in BOTH realms of the Naming Standard (pattern realm: a pattern's FIRST candle; market-structure realm: an ENDPOINT of a structure element) — never compare it across event / zone types. Which field is in which realm: `ARCHITECTURE.md` "`ev.idx` convention", "Anchor has two realms". |
| **bounds_steps** | History of zone boundary changes (INIT → expansions) |
| **outer threshold** | Zone boundary price (before top/bottom conversion) |
| **inner threshold** | Zone boundary price closer to current price |
| **KL zone** | Key Level zone — derived from CTS/BOS confirmation events |

---

## Timing Terms

| Term | Definition |
|------|------------|
| **apply_idx** | Candle where pattern effect is applied (`end_idx` on SUCCESS, `confirmation_idx` on CONFIRMED). On `REVERSAL_CANDIDATE.meta` it is the SCHEDULED apply — a prediction that can expire. |
| **confirmed_idx** | Candle where zone becomes confirmed for charting |
| **confirmed_at** | The MOMENT field in event meta — the candle at which the event became knowable. `CTS_ESTABLISHED` / `BOS_CONFIRMED`: the breakout pattern's apply candle = the cycle's CTS-established **moment** (one value for both events of a cycle, by construction); their `ev.idx` is the CTS / BOS anchor (a price location), which can precede it. `CTS_CONFIRMED` / `CTS_RECONFIRMED`: the confirmation candle, == `ev.idx`. Absent on every `CTS_UPDATED`. Canonical per-event table: `ARCHITECTURE.md` "`ev.idx` convention". |
| **moment** | The candle at which an event / level becomes knowable in real time (vs `ev.idx`, which for the structural events is a historical price location). Every timing / lifecycle read uses the moment, never `ev.idx`. For a cycle: the **CTS-established moment** = `CTS_ESTABLISHED.meta["confirmed_at"]` (see "cts_moment / CTS-established moment" under "Sub-Structure Pool Terms"). |
| **event_moment** | `structure/market_structure.event_moment(ev)` (Plan F, 2026-09-24) — an event's moment in code: `CTS_ESTABLISHED` → `meta["confirmed_at"]`; `CTS_UPDATED` → `ev.idx` on the raw path, `None` on the pattern path (no moment recorded); `CTS_THRESHOLD_UPDATED` → `ev.idx`; any other type raises `ValueError`. FibTracker asks its imbalance questions at it (see `evaluated_at` under "Imbalance Terms"). Canonical per-event table: `ARCHITECTURE.md` "`ev.idx` convention". |
| **start_idx** | Logical start of a range or structure (may differ from confirm_idx). On the sub-structure pool objects (`TriggerRecord`, `PooledStructure`, sub `SidRecord`) and on KL/POI/fib lifecycle meta, `start_idx` is a REAL-TIME lifecycle value — never confuse it with `starting_idx` (the historical structural anchor / pool key; see "Sub-Structure Pool Terms"). |
| **confirm_idx** | Candle index when event was emitted (may differ from start_idx) |
| **too_early** | Flag when identify_start finds extreme before min_history |

---

## Event Terms

| Term | Definition |
|------|------------|
| **event.idx** | Candle index stamped on the event — event-specific, NOT uniformly "when it became knowable": a price location (the CTS / BOS **anchor**) for `CTS_ESTABLISHED`, `BOS_CONFIRMED` and pattern-path `CTS_UPDATED` (their moment is `meta["confirmed_at"]`, or not recorded for `CTS_UPDATED`); the moment for the others. Canonical per-event table: `ARCHITECTURE.md` "`ev.idx` convention". |
| **event.meta** | Dictionary of event-specific metadata |
| **STATE_CHANGED** | Transition between market states (e.g., cts → bos) |
| **CTS_ESTABLISHED** | A new CTS cycle established by a breakout pattern (emitted together with that cycle's `BOS_CONFIRMED`). `idx` = the CTS anchor — the pattern's extreme candle, adopted as the cycle's CTS; `meta["confirmed_at"]` = the apply candle (the moment); `meta["anchor_idx"]` = the breakout pattern's FIRST candle (the pattern-realm anchor) — not necessarily the CTS anchor (the two coincide when the first candle holds the extreme; never on the reference window, 0 of 34). |
| **CTS_UPDATED** | The current (unconfirmed) CTS point was (re)stamped. Raw path (`meta["via"] == CTS_UPDATED_RAW_VIA`, `"replay_raw"`, the constant in `structure/market_structure.py`): a new wick extreme; `idx` = the processed candle = the moment. Pattern path (`via` = a pattern name — emitted on every breakout while the cycle is unconfirmed): `idx` = the pattern's extreme candle, stamped as the new CTS anchor; the apply candle is not recorded (no `confirmed_at`). It can duplicate a raw-path `CTS_UPDATED` already emitted at the same idx/price during the back-fill (1 of 37 pattern-path rows on the reference window — see `ARCHITECTURE.md` "`ev.idx` convention"). |
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
| **pending** | Scenario 3 / `unified_probe` status: insufficient data (no CTS_EST captured, or 1 CTS_EST + no `probe_end_idx` to bound the exception check — the probe's search bound, renamed from `end_idx` by Plan C 2026-09-20). A `first_confluence` trigger whose probe is pending becomes an `UnresolvedTrigger(reason="pending")`. |
| **pip_tolerance** | Distance threshold for exception evaluation near zone inner bound (per-TF defaults in `zones/zone_proximity.py::DEFAULT_PROBE_RESET_PIPS`: H1=3, M15=2.5, M5=2) |
| **subordinate probe** | The `structure/unified_probe.unified_probe` run that resolves a sub structure's `starting_idx` on the sub's own (M15) frame, called through `multitf/entity_df_mutation._resolve_trigger_m15_start` (`first_confluence` → own ad-hoc BOS_0 reference, Phase 1 + Phase 2; `first_counter` / `subsequent_confluence` / `subsequent_counter` → the sibling lens's CTS reference read from the pool, Phase 1 only) or `_resolve_reversal_start` (the reversal handoff over the reversing sub's own geometry), always behind the probe cache (`_probe_with_cache`). Direction is `trigger.lower_sd` — same as parent (`+parent_sd`) for confluence variants (var 1 / var 3), opposite (`-parent_sd`) for counter variants (var 2 / var 4). Historical: the legacy parent-TF probe (`compute_structure_scenario_3` via `multitf/lower_tf_pipeline.py::_run_subordinate_probe`, "H1 reverse probe") was retired; `lower_tf_pipeline.py` was deleted by Plan C (2026-09-20). |

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
| **imbalance candle** | A c2 (middle) candle whose neighbors form an FVG and whose direction matches the gap direction. Flagged per-candle via `is_imbalance`. The flag sits on c2, but the gap exists only once c3 has closed (see `formed_at`). |
| **imbalance instance** | One imbalance candle, or a run of consecutive same-direction imbalance candles, merged with combined gap bounds; `start_idx` / `end_idx` = its first / last c2. Stored in `df.attrs["imbalances"]`. |
| **FVG** | Fair Value Gap — 3-candle gap pattern where c1 and c3 wicks don't overlap |
| **gap_top / gap_bottom** | Merged bounds of an imbalance instance (first c1.high to last c3.low for bullish; reversed for bearish) |
| **formed_at** | `ImbalanceInstance.formed_at` = `start_idx + 1` — the close of the instance's FIRST c3, the MOMENT its gap exists (Plan F rule R1, 2026-09-24). A merged run keeps growing until `end_idx + 1`, but an unfilled gap already exists from `formed_at`. The POI activation sweep enters an instance at `max(formed_at, first_active)`. Canonical: `IMBALANCE_FILL_SEMANTICS.md` "Knowability — the c3 rule". |
| **formed prefix** | At a moment `K >= formed_at`: the c2s whose c3 has closed, `[start_idx, min(end_idx, K − 1)]` — the only part of a merged run tested against a window (`ImbalanceInstance.overlaps_formed_prefix`). |
| **evaluated_at** | The MOMENT an imbalance question is asked — keyword-only and REQUIRED on `has_unfilled_imbalance`, `resolve_cross_cycle_eligibility` and `select_fib_anchor_for_cycle`; only instances formed by then count (formed prefix). Not `check_to_idx`, the FILL HORIZON, which can be an anchor preceding the moment. `None` = an explicit "no cut" each caller justifies (retrospective IC identification, the MS in-flight resolver, the uncut cycle-0 caches — FibTracker's and its MS mirror —, a pattern-path `CTS_UPDATED`, which records no moment). FibTracker passes `event_moment` of the handled event. |
| **fill check** | `ImbalanceInstance.is_filled(df, check_to_idx)` — two-stroke (since 2026-05-23): filled once, within `(end_idx, check_to_idx]`, stroke 1 fires (a candle retraces ≥70% into the merged gap) AND stroke 2 fires at or after it (a close past the gap's outer edge in the instance's direction: bullish `close >= gap_top`, bearish `close <= gap_bottom`). `check_to_idx` is the fill horizon, not the moment. Canonical: `IMBALANCE_FILL_SEMANTICS.md` "The predicate". |

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

## Sub-Structure Pool Terms (model decided 2026-09-19; landed by Plan C 2026-09-20; canonical spec `PART4_REFACTOR_SPEC.md §17` rev 2; code `multitf/sub_structure_pool.py`, `multitf/lifecycle_sweep.py`, `multitf/parent_tables.py`, `multitf/entity_df_mutation.py`)

All idxs are entity-absolute M15 ints unless stated; every candle index is
named `*_idx` (never `*_dt`). **Vocabulary principle (§17 header):** lifecycle
fields (`start_idx`, `end_idx`, `trigger_end_idx`) are REAL-TIME — what was
tradeable when; pattern/element fields (`starting_idx`, `trigger_idx`,
`probe_finalize_idx`, every anchor) are HISTORICAL — where things logically
sit. Historical fields are inputs when constructing a lifecycle value, never
used as one.

| Term | Definition |
|------|------------|
| **unique sub** (`PooledStructure`) | One M15 (sub-TF) market structure, identified by `StructureKey = (parent_path, sub_tf, direction, starting_idx)` with ABSOLUTE `direction`; `sub_id` is its global monotonic creation index (assigned by `SubStructurePool.get_or_create`, only after a successful MS run). Computed once (geometry to the data edge), has ONE real-time lifecycle `[start_idx, end_idx]` aggregated from its records, **spans parent cycles and parent sids**, and is what we trade on. Fields: `key`, `sub_id`, `geometry`, `natural_reversal_idx`, `bos0_inner`, `records`, `start_idx`, `end_idx`, `end_reason`, `relative_dir_segments`; accessors `live_records()`, `lenses()`, `relative_dir_at(idx)`. |
| **parent_path** | The `structure_path_id` of the parent entity a sub hangs under — the constant `"H1.main"` today. Part of the pool key so an M5-under-counter can never collide with an M5-under-confluence. |
| **starting_idx** | HISTORICAL. The sub's structural anchor = the probe's finalized `ProbeResult.starting_idx` (renamed from `ProbeResult.start_idx` by Plan C): where the M15 MS run starts. Part of the pool key; denormalised onto every record; exported as `creation_event_idx` on the sub `SidRecord`. Never a lifecycle value. |
| **geometry** | `(bounded, slice_begin)` on a `PooledStructure`: the natural-end `compute_bounded_structure` run with run cap = the DATA EDGE (`len(m15) - 1`), built by `entity_df_mutation.build_or_get_geometry` (MS first; pool entry only on success — a failed build consumes no `sub_id`). `bounded.events` / `bounded.df` are SLICE-LOCAL (the slice starts 50 candles before `starting_idx`); add `slice_begin` for entity-absolute. Shared across records — never mutated in place. |
| **natural_reversal_idx** | The sub's own reversal (entity-absolute `bounded.reversal_idx + slice_begin`), known at build time because geometry runs to the data edge; None if the structure never reverses. Feeds the record end condition `reversal` and the successor spawn. |
| **bos0_inner** | The first probe's BOS_0 threshold (`ProbeResult.bos0_inner`) handed to MS as the cycle-0 breakout gate; stored on the sub, NOT part of the key. A later probe reaching the same key with a different inner is `WARNING [sweep] bos0_inner mismatch`-logged (keep the first), never raised. NOT the probe cache's tripwire comparand — that is `ProbeCacheEntry.ref_inner` (the reference inner the probe ran against; `bos0_inner` is the final iteration's threshold and moves on a reset). |
| **TriggerRecord** | A triggered *instance* of a unique sub: one trigger (or one absorbed group of triggers) that resolved to it. Identity `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)`; FK `sub_id` (never None). Has its OWN parent-bound real-time lifecycle (`start_idx`, `trigger_end_idx`, `end_idx`, `end_reason`, `ended_by_sub_id`); feeds the unique sub's lifecycle but does not equal it. Provenance (historical): `trigger_type`, `trigger_idx`, `probe_finalize_idx`, `probe_finalize_condition`, `validated_parent_idx`, `starting_idx`, `direction`, `sub_tf`, `relative_dir`, `parent_floor_idx`; bookkeeping: `seq`, `extra_trigger_idxs`, `source_trigger` (internal, never exported). Not frozen — the sweep writes the lifecycle fields in place, write-once. |
| **trigger_sub_sid** | Per-`(lens, parent_sid, parent_cycle_id)` counter starting at **0**, consumed (+1) each time a trigger in that scope resolves to a *new* unique sub (`SubStructurePool.next_trigger_sub_sid`). CREATION-ordered, not `start_idx`-ordered (an FC record is created at its `trigger_idx` and may start hundreds of candles later). Replaces the old per-parent-cycle `sub_sid`; lives on records only. |
| **sub_id** | Global unique-sub id (creation order). The chart identity for subs. Stamped on every structural artifact of a sub (event / KL / POI / fib / wave-candle / WVMI meta, `SidRecord.sub_id`, CSV columns); hard rename from `sub_sid` — `sub_sid` survives ONLY on main-entity `SidRecord`s where `sub_sid = structure_id` (sub `SidRecord`s carry `sub_sid = None`). |
| **trigger_type** | `first_confluence` \| `subsequent_confluence` \| `first_counter` \| `subsequent_counter` \| `reversal` — the trigger that produced the record. |
| **lens** | Which chart a record attributes its sub to: `"confluence"` / `"counter"` (`LENS_CONFLUENCE` / `LENS_COUNTER`). Named triggers map by use case (`resolve_lens`: `first_confluence` / `subsequent_confluence` → confluence; `first_counter` / `subsequent_counter` → counter); a `reversal` record inherits the lens of the record whose sub reversed (sticky-per-chart; `resolve_lens("reversal", reversed_from_lens=...)`). A unique sub is drawn on every lens it has a non-zero-length record on (`PooledStructure.lenses()`), over the SUB's window. The `"confluence" in sub_path_id` substring test is retired. |
| **relative_dir** | `"confluence"` iff `direction == parent_sd` of the record's parent **sid** (`ParentTables.parent_sd[S]` = `CTS_ESTABLISHED.meta["struct_direction"]`, one direction per sid), else `"counter"`. A *semantic* label distinct from `lens` — a sub can be `relative_dir=counter` yet render on the confluence chart (reversal-stickiness; e.g. `1797/−1` under H1 sid 0 (`+1`)). |
| **relative_dir_segments** | On the unique sub: the step function `[(from_idx, relative_dir), ...]` over `[start_idx, end_idx]` — at each record start/end, the active record with the latest `start_idx` gives the value; carried forward when none is active (`lifecycle_sweep._relative_dir_segments`). Read at a candle via `relative_dir_at(idx)`; the chart hover shows it per dot. |
| **trigger_idx** | HISTORICAL. The candle the trigger fired: `LOH(trigger_event_idx)` for the four H1 types (`first_counter`'s event idx is `WVMIRecord.meta["triggered_by_event_idx"]`); the native M15 reversal idx `R` for `reversal`. One of the three terms of the record's `start_idx`; also the sibling read's `hi`. |
| **trigger_event_idx** | The H1 candle of the parent trigger event (the detectors' field), before the LOH map; the sweep's `TRIGGER_FIRE` order key `(lens_rank, type_rank, trigger_event_idx)`. |
| **LOH** | `entity_df_mutation._map_parent_idx_to_m15_hour_end(parent_idx, h1_df, m15_df)`: the LAST M15 candle of an H1 candle's hour — the mapper for EVERY timing / lifecycle value (`trigger_idx`, `floor_m15`, `end_m15`, the WVMI trigger candle). `LOH(h) = 4h + 3` holds on the reference window as a cross-check only (the map is by timestamp). The price-extreme mapper `data_bridge.map_candle_to_lower_tf` is used only for the `first_confluence` probe's structural inputs (`probe_input_idx`, `probe_end_idx`). Never unify the two. |
| **probe_finalize_idx** | HISTORICAL. When THIS record's probe finalized, where "this record's probe" is the run keyed by `(direction, initial input)`: probe **ran** → its own `ProbeResult.finalize_idx`, raw (may precede `trigger_idx`, e.g. FC(0,1) 2608 < 2611); probe **skipped** (probe-cache hit) → the cached `finalize_idx`, inherited raw. A different-input probe that converges on a known structure keeps its own finalize. Native M15 or a mapped parent value depending on `finalize_condition` (`ProbeResult` docstring table). Non-FC, non-reversal types: `finalize_idx == trigger_idx` by construction (asserted, cache hits exempt). |
| **probe_finalize_condition** | `ProbeResult.finalize_condition` of that probe (`no_retrace`, `end_idx_reached`, `second_cts_reached`, `reversal_in_probe`, ...) — diagnostic copy on the record. |
| **validated_parent_idx** | The candle that seeded the probe: the H1 BOS anchor for `first_confluence`; the sibling CTS anchor (M15) for the sibling types; None for reversal-born records. Exported as `validated_h1_start` on the projection meta / `validated_parent_start` on the sub `SidRecord.meta`. |
| **parent_floor_idx** | `ParentTables.floor_m15[(S,C)] = LOH(max(struct_start[S], cts_moment[(S,C)]))` — the parent cycle's CLAMPED lifecycle-start on the CTS-established **moment**. Diagnostic copy on the record of the floor that was applied. |
| **start_idx (record)** | REAL-TIME. `max(probe_finalize_idx, trigger_idx, parent_floor_idx)` — the record EXISTS from here and nothing earlier. All three terms are load-bearing: `probe_finalize_idx` when the probe finished after the trigger (FC(0,0): trigger 463 → start 1020); `trigger_idx` when the structure was already known before this trigger fired (a cache-hit record inheriting an earlier finalize); `parent_floor_idx` when the parent structure was not alive yet. Historical fields are never adjusted; only `start_idx` is. |
| **trigger_end_idx / end_idx (record)** | `trigger_end_idx` = the first end condition to fire: own `reversal` (`natural_reversal_idx`), `same_dir_replacement`, `parent_end` (`ParentTables.end_m15[(S,C)]`) — or the sub's frozen end on a post-end re-trigger; `end_idx = max(trigger_end_idx, start_idx)`; None while open. Set once, never updated. At an equal idx the priority is `_END_REASON_PRIORITY`: `reversal` (0) > `parent_end` (1) > `same_dir_replacement` (2). `ended_by_sub_id` = the replacing sub for `same_dir_replacement`. |
| **end_reason** | `"reversal"` \| `"same_dir_replacement"` \| `"parent_end"` \| `None` — on records, on unique subs, on sub `SidRecord`s, and (as `cap_reason`) on every KL/POI/fib of a sub capped by the sub's window. The pre-pool `"lifecycle_end"` is gone; `next_cycle` is internal to `compute_cycle_lifecycle` (a cycle-owned zone/fib may carry it; a sub/record never does). |
| **same-direction replacement** | A new record in the same `(lens, parent_sid, parent_cycle_id, sub_tf, direction)` mapping to a DIFFERENT unique sub ends the active (incumbent) record at the new record's TRUE `start_idx` (not its `trigger_idx` or finalize). Same-lens only — a counter record never ends a confluence record. Invariant: ≤1 active record per `(lens, parent_sid, parent_cycle_id, direction)` (`SubStructurePool.active_record` asserts it). |
| **parent_end** | The record's OWN parent cycle ending: `end_h1[(S,C)] = floor_h1[(S,C+1)]` (the next cycle's CLAMPED start) if that cycle exists, else `rev_by_sid[S]` (the sid's `STATE_CHANGED→reversal` idx — NOT `REVERSAL_CANDIDATE.apply_idx`, a prediction that can expire), else None (open); LOH-mapped to `end_m15`. One helper (`build_parent_tables`) replaces the retired `parent_end_lookup`, the detectors' `lifecycle_end_idx` and `_find_m15_lifecycle_end`. |
| **active record** | `SubStructurePool.active_record(lens, S, C, direction, at_idx)`: the ≤1 record with `TriggerRecord.is_active_at(at_idx)` = `not is_zero_length and start_idx <= at_idx and (trigger_end_idx is None or trigger_end_idx > at_idx)` — the half-open interval `[start_idx, trigger_end_idx)` on `trigger_end_idx`, NOT `end_idx`. |
| **live at reversal** | `TriggerRecord.is_live_at_reversal(R)`: `not is_zero_length and start_idx <= R and (trigger_end_idx is None or trigger_end_idx >= R)` — closed on the right (a record whose parent ends AT `R` is still live at `R`). The successor-spawn test. |
| **zero-length record** | `TriggerRecord.is_zero_length = trigger_end_idx is not None and trigger_end_idx <= start_idx` (defined on `trigger_end_idx`, not `end_idx`, so it is already decidable before `end_idx` is written). Sources: a post-end re-trigger (the sub had already ended before this record's `start_idx` → frozen with the sub's `end_idx` / `end_reason`, linked), or a structure whose own reversal / parent end lands at or before the record's floored start. Has a `sub_id`; participates in nothing (not the sub's start, not `max_start`, not an end candidate, not an incumbent, not lens membership, not rendering) — logged only (`_triggers.csv`, `†` in the chart hover). |
| **start_idx / end_idx (unique sub)** | REAL-TIME. `start_idx` = the FIRST non-zero-length record's `start_idx`, set once. `end_idx`: at each candle `t` where a live record started or ended, over the live records that EXIST at `t` (`start_idx <= t`): `max_start = max(start_idx)`; candidates = non-null record `end_idx`s strictly `> max_start`; `end_idx = min(candidates)` (equal-idx ties by `_END_REASON_PRIORITY`, then `seq`); set once, frozen; `end_reason` = the winning record's. The strict `>` with start-before-end is what keeps a sub continuous across a same-candle handover. |
| **the sweep** (`run_lifecycle_sweep`) | The sub-structure driver (`multitf/lifecycle_sweep.py`): a priority-queue sweep over MOMENTS, provably identical to iterating candle-by-candle. Phases at one idx, each to completion: 0 `TRIGGER_FIRE` / `REVERSAL_SPAWN` (resolve → record) → 1 `RECORD_START` → 2 `SUB_START` → 3 `RECORD_END` → 4 `SUB_END`. `REVERSAL_SPAWN` sorts before `TRIGGER_FIRE` at the same idx; `TRIGGER_FIRE` orders by `(lens_rank confluence=0/counter=1, type_rank first_*=0/subsequent_*=1/reversal=2, trigger_event_idx)`; every other moment by record `seq`. The heap is idx-monotone (asserted). Probe, geometry build and reversal handoff are INJECTED callables (`resolve_start`, `build_geometry`, `resolve_reversal`, `synth_reversal_trigger`). Replaces `_ChainCursor` / `build_two_entity_parent_cycle` / `build_parent_cycle_chain` / `build_one_sid`. |
| **SweepTrigger** | The sweep's input: one lens-tagged, LOH-mapped H1 trigger (or a sweep-synthesised `reversal`): `lens`, `parent_sid`, `parent_cycle_id`, `trigger_type`, `trigger_idx`, `direction` (= `lower_sd`), `trigger_event_idx`, `source` (the `MultiTFTrigger`), `pending` (a `first_confluence` with `status != "finalized"`), `probe_input_idx`. |
| **ResolvedStart / ProbeFailure** | A resolver's return: `ResolvedStart(starting_idx, validated_parent_idx, bos0_inner, finalize_idx, finalize_condition, probe_input_idx, cache_hit)` on success; `ProbeFailure(detail, probe_input_idx)` (or None) → `UnresolvedTrigger(reason="probe_failed")`. |
| **successor spawn** | A sub's `natural_reversal_idx = R` (queued as `REVERSAL_SPAWN` when its geometry is built, only if `R > the current idx`) spawns the opposite-direction successor iff `R` falls inside a live record's window (`is_live_at_reversal(R)`): one `reversal` `SweepTrigger` per live record (`lens = r.lens`, the record's parent, `trigger_idx = R`, `direction = −sub.direction`, `source = _synth_reversal_trigger(r.source_trigger, sd, R)`), probed by `_resolve_reversal_start` over the reversing sub's own geometry; successors from two lenses dedup into ONE sub with a record per lens. No live record at `R` → the reversal is inert (no successor). Successor `start_idx = R`. |
| **absorbed trigger** (`extra_trigger_idxs`) | A later trigger in the same `(lens, parent_sid, parent_cycle_id)` that resolves to the SAME unique sub is appended to the existing record's `extra_trigger_idxs` — no new record, `trigger_sub_sid` not consumed. |
| **seq** | Global record creation counter (`SubStructurePool.next_seq`) — the sweep's deterministic tie-break and the `_triggers.csv` row order; never exposed as identity. |
| **ParentTables** (`build_parent_tables`) | Static per-parent-cycle tables computed ONCE from the H1 event stream: `rev_by_sid`, `struct_start`, `cts_moment[(S,C)] = CTS_ESTABLISHED.meta["confirmed_at"]` (last-seen), `parent_sd[S]`, `floor_h1`, `end_h1`, `floor_m15`, `end_m15`, `degenerate`; accessors `floor(S,C)`, `end(S,C)`, `is_degenerate(S,C)`, `has_cycle(S,C)`, `cycles()`. Asserts: every `CTS_ESTABLISHED` carries `confirmed_at`; every cycle with a `BOS_CONFIRMED` has a `CTS_ESTABLISHED` and `BOS_CONFIRMED.confirmed_at == CTS_ESTABLISHED.confirmed_at`; every LOH map succeeds. Logs one `[parent_tables]` line per cycle. |
| **cts_moment / CTS-established moment** | `CTS_ESTABLISHED.meta["confirmed_at"]` — the apply candle at which the cycle was established (== `BOS_CONFIRMED.meta["confirmed_at"]`, definitional). The canonical cycle lifecycle-START for main, sub cycles and the parent tables alike (`compute_cycle_lifecycle`, Plan C). NOT `CTS_ESTABLISHED.idx`, which is the CTS **anchor** (the pattern's extreme candle; historical; can precede the moment: M15 1223 vs 1224, 2828 vs 2829). |
| **degenerate parent cycle** | `ParentTables.degenerate[(S,C)] = end_m15 is not None and floor_m15 >= end_m15` — zero or inverted length, caused by the reversal handoff on a retroactive parent (H1 sid 1's `struct_start = reversal(sid 0) = 902` puts cycles (1,0) and (1,1) entirely before their floor 3611 on the reference window). Any trigger inside it → `UnresolvedTrigger(reason="degenerate_parent_cycle")`: no probe, no MS, no `sub_id`; one `WARNING [parent_tables] degenerate parent cycle` per cycle. |
| **UnresolvedTrigger / unresolved-trigger log** | Frozen row for a trigger that never became a record: `lens`, `parent_sid`, `parent_cycle_id`, `trigger_type`, `trigger_idx`, `direction`, `probe_input_idx` (whatever was known — H1 for the four named types, M15 for `reversal`), `reason ∈ UNRESOLVED_REASONS = {pending, degenerate_parent_cycle, probe_failed, geometry_failed}`, `detail` (free text, e.g. `floor 3611 >= end 3611`). No `sub_id`, no `trigger_sub_sid`. Stored pool-wide on `SubStructurePool.unresolved`, mirrored to every lens df's `attrs["unresolved_triggers"]`, exported once as `*_M15_unresolved_triggers.csv`; each logged `[sweep] UNRESOLVED (skipping) reason=…` (the `/compare` log grep keys on "skipping"). |
| **probe cache / ProbeCacheEntry** | `SubStructurePool` cache keyed `(parent_path, sub_tf, direction, initial_input_idx)` — "same probe" = same direction + same initial input. Value `ProbeCacheEntry(starting_idx, finalize_idx, finalize_condition, bos0_inner, probe_end_idx)`. **First probe to finalize for a key is the truth for every later probe of that key, (a) regardless of its `probe_end_idx` and (b) regardless of its reference zone** (accepted approximation, 2026-09-19). A hit returns the entry and SKIPS `unified_probe`: `[probe_cache] hit` (same `probe_end_idx`) / `APPROX hit` (different bound); `REF-ZONE DIFFERS` logged when the hitting trigger's reference `inner` is not `isclose` to the cached entry's `ref_inner` (the reference the cached probe ran against — not its final `bos0_inner`). `record_probe` is first-write-wins and asserts a second write for the same key is equal. Reversal-handoff probes key on the ENTITY-ABSOLUTE input (`_resolve_reversal_start` shifts by `slice_begin`), so they share keys with later H1 triggers. Measured on the first Plan C replay (2026-09-20): three hits on the reference window (FC(0,1) input 2365 ← the sub-1 reversal probe; `first_counter`(0,1) input 2609 ← the sub-2 reversal probe; the sub-6 reversal at 4200, input 4000 ← `subsequent_counter`(1,2)), none changing a lifecycle value; accepted as designed (user decision 2026-09-20) — the reversal handoff's input is part of the key space by definition. |
| **probe_end_idx** | The probe's SEARCH BOUND — the inclusive upper edge of `unified_probe`'s window (a compute bound like the run cap, unrelated to lifecycle `end_idx`; renamed from `end_idx` by Plan C on `unified_probe`, `ProbeResult`, `FirstConfluenceTrigger.probe_end_idx`, `MultiTFTrigger.meta["probe_end_idx"]`). `first_confluence`: the parent's confirmed-CTS EXTREME (`cts_anchor_idx`) price-mapped to M15 (a price bound — changing it moves `starting_idx`, the pool key); sibling types: `hi` = the trigger's `trigger_idx`; reversal: the reversal candle. |
| **starting_idx vs start_idx** | `starting_idx` = HISTORICAL structural anchor (pool key; `ProbeResult.starting_idx`; sub `SidRecord.creation_event_idx`; `LowerTFResult.meta["m15_start_idx"]`). `start_idx` = REAL-TIME lifecycle start (record / unique sub / sub `SidRecord.start_idx` / `LowerTFResult.meta["start_idx"]` / the zones' `lifecycle_floor`). The old `start_trigger_idx` was split into `trigger_idx` / `probe_finalize_idx` / `start_idx`. |
| **`cts_anchor_idx` (parent)** | On `CTS_CONFIRMED.meta`: the H1 candle of the parent's CTS **anchor** at confirmation (= the current CTS point when the confirmation fires, so it migrates via `CTS_UPDATED`). It equals `CTS_ESTABLISHED.idx` unless a `CTS_UPDATED` moved the anchor first (only 7 of 30 rows on the reference window — measured in `ARCHITECTURE.md` "`ev.idx` convention") — so never assume either way. `first_confluence`'s `probe_end_idx`, price-mapped to M15. |
| **`cts0_anchor` (probe)** | Inside `unified_probe` Phase 2: the probe's OWN M15 `CTS_0_CONFIRMED.cts_anchor_idx`, which caps the retrace window at `cts0_anchor − 1`. A different object from the parent's `cts_anchor_idx` despite the name. |
| **sibling read** (`_build_sibling_cts_ref_zone_from_pool`) | The sibling-referencing probes' reference: "the most recent qualifying CTS of the opposite-direction sub on the OTHER lens in this parent cycle" as a POOL query — candidates = `pool.records_for(other_lens, S, C)` with `direction == −probe_direction`, non-zero-length, live somewhere in `[lo, hi]` (`start_idx <= hi` and `trigger_end_idx is None or >= lo`); each record's CTS events (`_CTS_EVENT_TYPES`) are taken from its sub's geometry shifted by `slice_begin` and CLIPPED to the record's own live window ∩ `[lo, hi]`; `hi` = the reading trigger's `trigger_idx` (the only thing keeping the read causal — never widen); the reference zone is built ad hoc from the winning event with `kl_zones=[]` and `df` = the shared M15 frame. Replaces the per-trigger scratch entity dfs. |
| **projection** (`render_sub_projection` / `project_to_window`) | ONE downstream derivation per unique sub over the sub's window: `project_to_window(bounded, floor = start_idx − slice_begin, cap = end_idx − slice_begin or None, cap_reason = end_reason, ...)` clips events by knowable-at (`knowable_at_idx`) at the cap and runs `_run_downstream_pipeline` with `lifecycle_floor` / `lifecycle_cap` / `cap_reason`; the resulting `LowerTFResult` (`trigger` = the first live record's `source_trigger`; `meta` = `sub_id`, `m15_start_idx` (= `starting_idx`), `start_idx`, `end_idx`, `m15_end_idx` (= `end_idx` or the geometry edge), `end_reason`, `natural_reversal_idx`, `slice_begin`, `lenses`, `relative_dir_segments`, `n_records`, `first_record`, `validated_h1_start`, ...) is mirrored into every lens df in `sub.lenses()` by `mirror_lower_tf_result_to_entity_df`. |
| **lens df** | One of the two M15 entity dfs (`H1.main >> M15.confluence`, `H1.main >> M15.counter`): copies of the ONE shared M15 feature frame populated by the mirror — views for the chart/export readers, not separate structural storage. Each carries `attrs["events" / "kl_zones" / "poi_zones" / "fib_states" / "wave_candles" / "wvmi" / "prev_bos_lines"]` (attributed by `sub_id`), `attrs["sids"]`, `attrs["triggers"]`, `attrs["unresolved_triggers"]`. |
| **sub `SidRecord`** | One row per unique sub rendered on a lens (`build_sid_records_for_subordinate`): `sub_id`, `sub_sid = None`, `starting_sd = direction`, `creation_event_idx = starting_idx`, `start_idx`, `end_event_idx = end_idx` (None while open), `end_reason`, `parent_sid` / `parent_cycle_id` None, `lenses`, `relative_dir_segments`, `meta = {natural_reversal_idx, n_records, first_record, slice_begin, validated_parent_start}`. Main rows are unchanged (`sub_sid = structure_id`, `sub_id = None`). |
| **`_subs.csv` / `_triggers.csv` / `_unresolved_triggers.csv`** | The three pool tables (`debug/export_sub_tables.export_sub_tables`), written BEFORE the M15 chart loop in `run_replay.py` inside their own try/except: per lens `*_M15_{lens}_subs.csv` (one row per sub on the lens: `sub_id, direction, starting_idx, start_idx, end_idx, end_reason, natural_reversal_idx, lenses, relative_dir_segments, n_records, first_record_*`), per lens `*_M15_{lens}_triggers.csv` (every `TriggerRecord` field except `source_trigger`, plus `is_zero_length`, `seq` order), and one pool-wide `*_M15_unresolved_triggers.csv`. `*_M15_{lens}_sids.csv` is GONE. |
