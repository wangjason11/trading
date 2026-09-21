# Architecture & System Design (through Week 8)

This doc explains the “shape” of the system so a new engineer can extend it without breaking project invariants.

---

## Design goals

1. **Explainable decisions**
   - Every trade-relevant claim must be backed by (a) dataframe columns, and (b) emitted events.
2. **Deterministic replay**
   - Given the same input candles, the full pipeline should produce identical outputs.
3. **Event-driven composition**
   - Each stage produces explicit outputs that downstream stages consume.
4. **Visualization-first**
   - The canonical debugging tool is the replay chart.

---

## Execution model

### 1) Batch / Replay (today)
- We simulate “live timing” by:
  - Computing features/patterns on the full df
  - Emitting structure events at the candle index where they would have become known
  - Using rewinds only when thresholds are known after a lookahead window

The MarketStructure engine is explicitly sequential and uses internal rewind/replay logic when it must evaluate ranges with corrected thresholds.【fileciteturn1file0】

### 2) Live (future)
- Same logic should be usable incrementally:
  - Candle features per new candle
  - Pattern detection per new candle
  - Market structure update per new candle
  - Zones updated by structure events (no additional rewinds/waits)

#### Stateless function + caller-managed state pattern (live-mode prep)

When adding components that may run repeatedly as new candles arrive,
prefer **stateless functions with explicit pending/finalized status** over
stateful tracker classes that internalize iteration state. Callers manage
the small amount of state they need (e.g., the current best `start_idx`)
externally and re-invoke the function per new candle.

**Example (already in place):** `compute_structure_scenario_3` Phase 1 —
Condition 4 splits on `end_idx`:
- `end_idx` defined → `finalized` (caller's bound is a real terminal)
- `end_idx is None` → `pending` (more candles may resolve later)

A live caller re-invokes the probe with the same or advanced `start_idx`
each new candle, and uses status to decide whether to start downstream
work. No tracker class needed. New similar features should follow this
pattern unless there's a strong reason to encapsulate state in a class.

---

## Data model contracts

### The dataframe is the shared “truth”
Each stage:
- Adds well-scoped columns (avoid overwriting unrelated columns)
- Optionally writes debug columns (suffix `_debug` recommended)
- Leaves earlier columns intact

### Event contracts
Downstream components must rely on events over inference.

#### PatternEvent
Produced by structure patterns:
- `name`: continuous / double_maru / one_maru_continuous / one_maru_opposite
- `status`: SUCCESS / CONFIRMED / FAIL_NEEDS_CONFIRM
- `start_idx`, `end_idx`, `confirmation_idx` (if confirmed)
- `confirmation_threshold` (for confirmation lookahead)
- `break_threshold_used` (range or BOS/CTS thresholds)

Pattern priority rules are defined in BreakoutPatterns.【fileciteturn2file7】

#### StructureEvent
Produced by MarketStructure:
- `category`: STRUCTURE / RANGE / (etc)
- `type` examples:
  - `CTS_ESTABLISHED`, `CTS_CONFIRMED`, `CTS_UPDATED`
  - `BOS_CONFIRMED`
  - `RANGE_STARTED`, `RANGE_UPDATED`, `RANGE_RESET`
  - threshold events such as `CTS_THRESHOLD_UPDATED` (used for zones)

**`ev.idx` convention — IMPORTANT:**
- Most events: `ev.idx` = confirmation/apply candle (when the event is known)
- **BOS_CONFIRMED exception:** `ev.idx` = BOS extreme candle (the level location), NOT when it was confirmed. Use `ev.meta["confirmed_at"]` for timing boundaries (scan windows, lifecycle ends).
- **CTS_ESTABLISHED exception (second extreme-not-apply exception; Plan C, 2026-09-20):** `ev.idx` = the CTS **extreme** candle inside the breakout pattern span (`market_structure._emit_cts_established(cts_idx, ...)`), NOT the candle at which the cycle was established. The establishing moment is `ev.meta["confirmed_at"]` (the pattern's `apply_idx`), and it equals the same cycle's `BOS_CONFIRMED.meta["confirmed_at"]` **by definition** (both are stamped from the one `apply_idx` in the same emission block; `multitf/parent_tables.py` asserts this identity). The extreme can precede the moment (on the reference window three M15 sub cycles: `.idx` 1223 vs `confirmed_at` 1224, 2828 vs 2829 ×2; equal on all five H1 cycles only by coincidence). Every **timing / lifecycle** read of a cycle start — `zones/structure_lifecycle.compute_cycle_lifecycle`, `multitf/parent_tables.build_parent_tables` (`cts_moment`), the Plan-B early stop's finalize — uses `meta["confirmed_at"]`; `ev.idx` is a historical anchor (pattern/element definition). A `CTS_ESTABLISHED` without `meta["confirmed_at"]` makes both readers raise (`AssertionError`), never fall back to the extreme.

The engine maintains a stable downstream interface by converting structure events into StructureLevels (CTS/BOS list).【fileciteturn1file14】

#### KLZone
Produced by `derive_kl_zones_v1` from structure events (not structure levels).【fileciteturn2file4】

---

## Lifecycle state convention (active / inactive / ended)

A long-lived object (POI zone, Fib, KL zone, structure cycle) has **two
orthogonal axes of state** that must be kept distinct in its representation:

| Axis | Nature | Reversible? | Determined by |
|---|---|---|---|
| **active / inactive** | Condition-state — "are all my required conditions true right now?" | **Yes** — flips back and forth as conditions change | A defined set of activation conditions evaluated at any candle `t` |
| **ended (terminal)** | Time-based irrelevance | **No** — once ended, stays ended | A defined set of end conditions (next cycle, reversal, the owning sub's window end — `parent_end` / `same_dir_replacement` — etc.) |

Once ended, "active" is undefined / always False. Before ended, the object
flips between active and inactive based on its activation conditions.

**Two tiers — not every lifecycle object has the condition (active/inactive)
axis.** It applies ONLY to objects with a genuinely **reversible condition** —
today just **POI** and **Fib**, whose condition is the unfilled-imbalance state
(flips as imbalances form / commit-fill). Objects with **no reversible
condition** — **structure cycles, structures, KL zones, WVMI** — are **tier-1**:
their "active" simply means **started-and-not-ended**, fully derivable from
`start_idx`/`end_idx`. They need no stored `active` flag and no
`activation_history`; `status` derives from start/end alone. So when adding a new
lifecycle object, ask **"does it have a reversible condition?"** — if not, give it
only `start_idx` + `end_idx`/`end_reason` + derived `status` (do NOT add an
`active` axis just for symmetry; it would be redundant).

`activation_history` (the per-flip list) is itself only **load-bearing for POI**,
whose chart fill + proximity gate walk it per-candle. **Fib dropped it** for
scalar `start_idx` logging (`FIB_LIFECYCLE_SPEC.md §15`, 2026-05-27 — no consumer
walked a fib history). **KL** carries a degenerate single-entry history purely as
a chart-fill convenience. Tier-1 objects carry none.

Recommended representation for a new lifecycle object (full form — trim per tier):

| Field | Type | Meaning |
|---|---|---|
| `end_idx` | `Optional[int]` | Terminal idx if known; `None` if not yet ended |
| `end_reason` | `Optional[str]` | Why it ended. Cycle-owned zones/fibs: `"next_cycle"` \| `"reversal"` \| the sub's `cap_reason` \| `"obsolete:<reason>"` \| ... Sub-structure pool objects (`TriggerRecord`, unique sub) and everything capped by a sub's window: `"reversal"` \| `"same_dir_replacement"` \| `"parent_end"` \| `None` (Plan C, 2026-09-20 — `"lifecycle_end"` is no longer emitted by any production path; `next_cycle` stays internal to `compute_cycle_lifecycle`) |
| `activation_history` | `list[{idx, active, reason}]` | All activate/deactivate flips within `[first_active, end_idx)`, driven by the object's activation conditions |
| `status` (derived) | `"active"` \| `"inactive"` \| `"ended"` | Computed from `end_idx` + `activation_history` at any candle `t` |

Derivation at candle `t`:
```python
if end_idx is not None and t >= end_idx:
    status = "ended"
elif current_state_per_activation_history(t) is True:
    status = "active"
else:
    status = "inactive"
```

**Following this convention:** `POIZone` (tier-2, full `activation_history`),
`KLZone` (tier-1, degenerate single-entry history), and `FibState` (tier-2,
scalar — see below); `WVMI` designed (tier-1, see below). POI/KL store `end_idx` +
`end_reason` + `activation_history` + derived 3-state `status` in `meta`. `KLZone` joined the convention in the Phase 3 unified-lifecycle pass
(2026-05-25): it computes no end of its own and inherits its owning cycle's
resolved end (reversal / next-cycle CTS-established / parent-cycle-end for
subs) — see `zones/KL_ZONES_SPEC.md` "Lifecycle". KL has no reversible
condition-state, so its `activation_history` is the single interval
`[{idx: confirmed_idx, active: True, reason: "confirmed"}]` and `status` is
`"active"` until `end_idx`, then `"ended"`.

POI activation conditions
(per-candle, in `zones/poi_zones.py::_compute_poi_activation_history`):
  1. `cts_at(t) >= ic_idx` — IC lies within the fib's bounds at t
     (the fib's `cts_idx` only grows via `CTS_UPDATED`).
  2. `ic_idx <= t` — IC candle exists.
  3. `has_unfilled_imbalance(df, ic_idx + 1, t, check_to_idx=t,
     direction=sd)` — sd-direction imbalance overlaps `(ic_idx, t]`
     that is not yet *committed-filled* per the two-stroke state machine
     (stroke 1 = 70% retrace; stroke 2 = close past gap outer in
     instance direction). See `IMBALANCE_FILL_SEMANTICS.md` for the full
     predicate definition. Flips as imbalances form / commit-fill.
  5. Variant ≥ V30 — IC candle overlaps the 61.8-80% fib zone
     (computed from `bos_price` and the time-varying `cts_price_at(t)`)
     by at least 30%. Variants can downgrade (V90 → V60 → V30) or
     vanish entirely as `cts_price` extends and the zone slides.
Scenario conditions (#4: per-scenario idx/price constraints) are gated
at IC identification and not re-checked per candle (once the
constraining event fires, the comparison is fixed).

**Activation floor (both POI and KL).** A zone's first-active is clamped to
its owning cycle's lifecycle-start: `first_active = max(<zone-specific
confirm idx>, cycle_lifecycle_start)`, where `cycle_lifecycle_start =
max(CTS_n ESTABLISHED MOMENT (meta["confirmed_at"]), structure lifecycle-start[,
lifecycle_floor])` — `zones/structure_lifecycle.compute_cycle_lifecycle`
(Plan C, 2026-09-20: the canonical cycle-start is the established **moment**,
never `CTS_ESTABLISHED.idx`, the extreme — see the `ev.idx` convention above).
`lifecycle_floor` is `None` for main; for a sub it is the unique sub's
real-time `start_idx` (slice-local, passed by
`multitf/entity_df_mutation.render_sub_projection`) — the pre-Plan-C
"parent sid / parent_cycle_id floors" (plan B1, 2026-05-27) are now folded
into that one value, because the record's `start_idx` already contains the
parent floor (`parent_floor_idx`). This prevents a zone from activating before
its structure is alive — e.g. a post-reversal cycle-0 zone whose
`CTS_ESTABLISHED` precedes the reversal confirmation. See
`PART4_REFACTOR_SPEC.md §5` / `§17.4` (starting_idx vs lifecycle-start + the
clamp). NB the per-zone `<zone-specific confirm idx>` term: for KL it is the
zone's own `confirmed_idx` (a BOS zone's `confirmed_idx` IS
`BOS_CONFIRMED.meta["confirmed_at"]` = the moment) clamped to the per-sid
structure start (`compute_struct_start_by_sid`); for POI
`_compute_poi_activation_history` takes `cts_established_idx =
CTS_ESTABLISHED.idx` (the **extreme**) as its cycle term — flagged 2026-09-20
as a ≤1-candle inconsistency with the moment rule on cycles where the extreme
precedes the moment; not changed by Plan C.

**`FibState` — NOW following (2026-05-27).** Tier-2 (it has the reversible
imbalance condition). Sessions 1 & 2 separated the overloaded `active` into
condition-only `active` + terminal `end_idx`/`end_reason` + derived `status`; the
cross-fib versioning is handled by making the **cycle the lifecycle identity**
with versions an internal sub-axis. The **§15 simplification** then DROPPED
`activation_history` for a sticky scalar `start_idx` (no fib consumer walked the
history) and wired fib onto the shared `compute_cycle_lifecycle` (clamp start to
the structure floor; cycle-end as an earliest-wins terminal candidate). Canonical:
`zones/FIB_LIFECYCLE_SPEC.md` (§15 is authoritative); `memory/project_fib_lifecycle_design.md`.

**`WVMI` — DESIGNED, impl pending (2026-05-27).** Tier-1 (created-once/locked-once,
no reversible condition): scalar `start_idx` (= creation `CTS_n` CONFIRMED, clamped)
+ inherited `end_idx` + derived `status`; NO active/inactive, NO `activation_history`;
its existing `created/updated/locked` computation axis is renamed `lp_status` so
`status` is the lifecycle label. Canonical: `zones/WVMI_SPEC.md` "Lifecycle
convention"; `memory/project_wvmi_lifecycle_deferred.md`.

So after WVMI lands, **every** lifecycle-like object will be on the convention
(`project_lifecycle_convention_klzone_fibstate.md` tracked the original migration).

**Sub-structure pool objects — `TriggerRecord` + unique sub (`PooledStructure`)
are tier-1 (Plan C, landed 2026-09-20; canonical: `PART4_REFACTOR_SPEC.md §17`,
code: `multitf/sub_structure_pool.py`, driver: `multitf/lifecycle_sweep.py`).**
Both have no reversible condition: "active" means started-and-not-ended, fully
derivable from `start_idx` / `end_idx`; no stored `active` flag, no
`activation_history`. Their lifecycle fields are **real-time** (what was
tradeable when); their pattern/element fields (`starting_idx`, `trigger_idx`,
`probe_finalize_idx`, every anchor) are **historical** and are only ever
*inputs* to a lifecycle value, never used *as* one in an is-active-at-`t` test.

| Object | `start_idx` | end |
|---|---|---|
| `TriggerRecord` (a triggered instance of a unique sub; identity `(lens, parent_sid, parent_cycle_id, trigger_sub_sid)`, FK `sub_id`) | `max(probe_finalize_idx, trigger_idx, parent_floor_idx)` — the record EXISTS from here and nothing earlier. `parent_floor_idx` = `LOH(max(struct_start[S], cts_moment[(S,C)]))`, the parent cycle's clamped lifecycle-start on the CTS-established **moment** (`multitf/parent_tables.py`). | `trigger_end_idx` = the first end condition to fire (own reversal / `same_dir_replacement` at the replacing record's true `start_idx`, same lens only / `parent_end` = `end_m15[(S,C)]`), or the sub's frozen end on a post-end re-trigger; `end_idx = max(trigger_end_idx, start_idx)`; `end_reason ∈ {reversal, same_dir_replacement, parent_end, None}`; `ended_by_sub_id` names the replacing sub. `is_zero_length = trigger_end_idx is not None and trigger_end_idx <= start_idx` — such a record participates in nothing (logged only). |
| unique sub (`PooledStructure`; identity `(parent_path, sub_tf, direction, starting_idx)`, `sub_id` global creation index; NOT bound to a parent — spans parent cycles and sids) | the **first live (non-zero-length) record's** `start_idx`; set once (sweep phase 2 `SUB_START`). | re-evaluated at every candle `t` where one of its live records started or ended: over the live records that **exist at `t`** (`start_idx <= t`), `max_start = max(start_idx)`; candidates = non-null record `end_idx`s **strictly `> max_start`**; `end_idx = min(candidates)`, ties broken by `_END_REASON_PRIORITY` (`reversal > parent_end > same_dir_replacement`). Set once, frozen. `end_reason` = the winning record's. A sub with no live record has no `start_idx` and is logged, not rendered. |

`status` for both derives from `start_idx` / `end_idx` alone. The sub's window
`[start_idx, end_idx or data edge]` is the ONE lifecycle every lens draws
(§16.5 rev 2 ownership, `charting/CHARTING_SPEC.md` "M15 Dedicated Chart")
and the `lifecycle_floor` / `lifecycle_cap` / `cap_reason` that
`render_sub_projection` hands the zones layer (so every KL/POI/fib on a sub
inherits the sub's `end_reason` vocabulary as its cap reason).

**Deliberately NOT lifecycle objects:** structure events
(`df.attrs["structure_events"]`) and candle / structure patterns. These are
**immutable, append-only historical facts** (`PART4_REFACTOR_SPEC.md §7`) —
"what the algorithm believed at the time." They are never mutated and carry
no `active`/`ended` state. Their only time-varying property is **currency**
("which sid/cycle owns candle *t* for display"), and that is a *query-time
derivation* (the §16.5 most-recent-sid chart filter on H1; on the M15 charts
`export_m15_chart._compute_owner_by_idx_dir` — owner per `(candle, direction)`
over each unique sub's real-time window `[start_idx, end_idx or edge]`), not
state stored on the event. This is a settled decision, not a deferral: giving
an immutable fact a mutable lifecycle would contradict the append-only event
contract.

**Why decouple:** mixing "condition-state" and "terminal-state" into one
field (today's `active=False`) loses information. A consumer can't tell
whether a fib's `active=False` means "imbalances temporarily filled, could
come back" vs "this cycle is permanently done." Different consumers care
about different distinctions — charting wants to know "currently-tradeable
vs historical"; debug tooling wants to know "why did this go inactive?".
The 3-state status carries enough to answer both without overloading.

---

## Module boundaries

### Pipeline / Orchestration
`run_pipeline(df)` owns ordering and returns a single bundle for replay and future live use:
- df (enriched)
- pattern events
- structure levels
- meta (including zones)

Ordering is intentionally locked for Week 6: base features must be computed **before** structure so zone resolution is stable.【fileciteturn2file1】

### Wave Candles (`zones/wave_candles.py`)
Identifies boundary candles between consecutive waves at each KL zone. For each zone, produces a `WaveCandleResult` with `last_wave_candle_idx` (end of prior wave) and `first_wave_candle_idx` (start of new wave). BIB zones use an event-driven multi-step search; non-BIB zones use a ±5 candle window. Results stored in `df.attrs["wave_candles"]`. See `WAVE_CANDLES_SPEC.md`.

### WVMI (`zones/wvmi.py`)
Measures BOS zone strength via volume ratios of wave candle pairs. Runs **after POI zones** because it depends on POI zone inner bounds for its activation gate. Lifecycle:
0. **Gated by first sd zone-proximity trigger** — `check_zone_proximity()` (in `zones/zone_proximity.py`) scans candles from CTS_CONFIRMED to zone deactivation (next BOS or reversal). It produces a list of alternating sd / opp_sd trigger candles per cycle. The orchestrator uses only the first sd trigger as the WVMI gate (preserves pre-refactor behavior). Threshold defaults: H1 = 9 pips, M15 = 6 pips, M5 = 3 pips (caller-overridable).
1. **Created** at CTS_n confirmation (only if activated) — breakout momentum locked from FB/LB volumes
2. **Updated** each candle — temporary LP shifts to closest qualified candle near outer bound
3. **Locked** at BOS_n+1 confirmation — LP finalizes, pullback momentum locked

Results stored in `df.attrs["wvmi"]` (list of `WVMIRecord`). See `WVMI_SPEC.md`.

### Scenario 3 (`structure/structure_engine.py`)
Arbitrary-start structure analysis with iterative BOS_0 probe. Phase 1 validates/refines `start_idx` by checking if price reaches the BOS_0 zone inner bound (within configurable pip tolerance: H1=10, M15=3, M5=1). Phase 2 continues multi-structure analysis from the finalized probe using the same logic as `compute_structure`. Returns `Scenario3Result` with status always "finalized" (probe accepts current start when bound is reached).

Parameters: `end_idx` bounds the probe window (passed to MarketStructure); `run_continuation=False` skips Phase 2 for probe-only use (e.g. H1 reverse probe that only needs the validated `start_idx`).

### Structure From Start (`structure/structure_engine.py`)
`compute_structure_from_start()` runs multi-structure analysis from a known start without Scenario 1 identification or Scenario 3 probes. Same Exception 1/2 handling on reversals as `compute_structure()`. Used for lower-TF structures where the start has been pre-validated by a higher-TF probe.

### Multi-TF Analysis (`multitf/`) — the sub-structure pool (Plan C, landed 2026-09-20)
Subordinate lower-TF (M15) structures triggered by H1 events. Canonical spec:
`PART4_REFACTOR_SPEC.md §17` (rev 2). The pre-pool description (UC1 "H1 reverse
probe" → `compute_structure_from_start` → `lower_tf_pipeline.py`, one build per
trigger, lifecycle bounded by the parent cycle) is history: `lower_tf_pipeline.py`
is deleted, and the Phase-1 two-entity cadence chain (`_ChainCursor` /
`build_two_entity_parent_cycle` / `build_parent_cycle_chain` / `build_one_sid`)
was replaced by the sweep below.

**Flow (`pipeline/orchestrator._run_multi_tf_dual`):**
1. Detect the five trigger types on H1: `first_confluence` (`first_confluence_pipeline`),
   `first_counter` (`uc1_trigger.detect_uc1_triggers`), `subsequent_confluence`,
   `subsequent_counter` (their `*_pipeline.to_multi_tf_trigger`); `reversal` is
   synthesised by the sweep from a sub's own natural reversal. Each H1 trigger is
   lens-tagged (`sub_structure_pool.resolve_lens(use_case)`: `*_confluence` →
   confluence, `*_counter` → counter; a `reversal` inherits the spawning record's
   lens) and timing-mapped: `trigger_idx = LOH(trigger_event_idx)`
   (`entity_df_mutation._map_parent_idx_to_m15_hour_end`, last M15 candle of the
   H1 hour — the mapper for EVERY timing value; the price-extreme mapper
   `data_bridge.map_candle_to_lower_tf` is used only for the `first_confluence`
   probe's structural inputs; never unify them).
2. Prepare ONE shared M15 feature frame (`prepare_lower_tf_data`, once) plus two
   **lens dfs** (copies; views for the chart/export readers).
3. `multitf/parent_tables.build_parent_tables(sorted_events, h1_df, m15)` — the
   static parent tables from the H1 events: `cts_moment[(S,C)] =
   CTS_ESTABLISHED.meta["confirmed_at"]`, `floor_h1 = max(struct_start[S],
   cts_moment)`, `end_h1 = floor_h1[(S,C+1)]` else `rev_by_sid[S]` else None,
   `floor_m15` / `end_m15` = `LOH(...)`, `degenerate = end_m15 is not None and
   floor_m15 >= end_m15`. Asserts (never degrades): every LOH map succeeds;
   `BOS_CONFIRMED(S,C).confirmed_at == CTS_ESTABLISHED(S,C).confirmed_at`.
4. `multitf/lifecycle_sweep.run_lifecycle_sweep` — a priority-queue sweep over
   moments, phases at one idx in the order `TRIGGER_FIRE`/`REVERSAL_SPAWN` (0) →
   `RECORD_START` (1) → `SUB_START` (2) → `RECORD_END` (3) → `SUB_END` (4), each
   to completion; start-before-end at the same idx is load-bearing (a same-candle
   handover keeps a sub continuous). Phase 0 resolves a trigger: degenerate
   parent cycle → `UnresolvedTrigger(reason="degenerate_parent_cycle")` (no probe,
   no MS, no `sub_id`); else probe via the injected resolver
   (`_resolve_trigger_m15_start` for the four H1 types — `first_confluence` on its
   own ad-hoc BOS_0 with Phase 2, the sibling types on the OTHER lens's most
   recent qualifying CTS read **from the pool** (`_build_sibling_cts_ref_zone_from_pool`,
   records clipped to their own live window ∩ `[lo, hi]`, `hi` = the reading
   trigger's `trigger_idx`); `_resolve_reversal_start` for a reversal), every
   probe behind the probe cache (`_probe_with_cache`, key `(parent_path, sub_tf,
   direction, initial_input_idx)`; first probe to finalize is the truth for the
   key; a hit skips `unified_probe`, logs `[probe_cache] hit|APPROX hit` and
   `REF-ZONE DIFFERS` when the hitting reference inner ≠ the cached `bos0_inner`);
   then `build_or_get_geometry` (MS first with run cap = the DATA EDGE, pool entry
   only on success); then the `TriggerRecord` (`start_idx = max(finalize,
   trigger_idx, floor)`), with a same-`(lens, S, C)` re-trigger of the same sub
   absorbed into the existing record (`extra_trigger_idxs`).
5. ONE projection per unique sub (`entity_df_mutation.render_sub_projection` →
   `pooled_structure_build.project_to_window` with `floor = sub.start_idx`,
   `cap = sub.end_idx`, `cap_reason = sub.end_reason`, slice-local) mirrored into
   every lens df in `sub.lenses()` by `mirror_lower_tf_result_to_entity_df`, in
   `start_idx` order (later-live wins overlapping structure columns).
6. Sub WVMI, one sweep per unique sub (`_assign_sub_wvmi_per_sub`, §17.10
   minimal — not settled; `zones/WVMI_SPEC.md` "Sub entities").
7. Per lens df: `attrs["sids"]` (one `SidRecord` per unique sub on that lens,
   `sub_id` set, `sub_sid = None`), `attrs["triggers"]` (that lens's
   `TriggerRecord`s incl. zero-length), `attrs["unresolved_triggers"]`
   (pool-wide); registry registration; `meta["sub_pool"]`, `meta["parent_tables"]`.

**Key design decisions:**
- Identity of a unique sub = `(parent_path, sub_tf, direction, starting_idx)`
  with ABSOLUTE `direction` (`StructureKey`); `sub_id` = global creation index.
  Confluence vs counter lives on the record only: `relative_dir` (semantic:
  `direction == parent_sd` of the parent sid) vs `lens` (which chart) — they
  legitimately differ (reversal-stickiness).
- A record cannot outlive its parent cycle (`parent_end`); a unique sub can
  (it inherits parent bounds only through the aggregation rule above).
- Geometry is cap-free (one MS run to the data edge, slice-local events/df +
  `slice_begin`; the M15 slice keeps the 50-candle lookback and re-runs
  `compute_imbalance` + the `is_range_*` re-derivation after `reset_index`).
- KL zones are BOS-only for subs (`source_kinds=["BOS"]`); Fib uses
  `fib_mode="cross_cycle"`.
- Attribution stamped on every mirrored event/zone/POI/fib/wave-candle/WVMI
  record: `structure_path_id`, `timeframe`, `parent_tf`, **`sub_id`** (the
  identity) + informational `use_case`, `parent_sid`, `parent_cycle_id`,
  `started_by` from the sub's FIRST live record. No consumer may use the
  informational three for identity. `sub_sid` survives only on main-entity
  `SidRecord`s (`sub_sid = structure_id`); `trigger_sub_sid` lives on records.
- Exports (decoupled from the chart loop, `debug/export_sub_tables.py`, written
  BEFORE the M15 charts in `run_replay.py`): `*_M15_{lens}_subs.csv`,
  `*_M15_{lens}_triggers.csv`, `*_M15_unresolved_triggers.csv`; the old
  `*_M15_{lens}_sids.csv` is gone.
- Lifecycle values are real-time; `starting_idx` / `trigger_idx` /
  `probe_finalize_idx` are historical. The probe's search bound is
  `probe_end_idx` (a compute bound, unrelated to lifecycle `end_idx`); the
  probe's output anchor is `ProbeResult.starting_idx`.

### Charting
Charting reads from:
- dataframe columns
- `df.attrs["kl_zones"]`
- `df.attrs["wave_candles"]`
- `df.attrs["poi_zones"]`
- `df.attrs["fib_states"]`
- `df.attrs["wvmi"]`
- `df.attrs["prev_bos_lines"]`
- `df.attrs["structure_events"]`
It should not mutate algorithm state.

---

## Debug & QA invariants

### Structure invariants
The MarketStructure engine runs lightweight df-level invariant checks:
- range_lo <= range_hi while active
- CTS_CONFIRMED rows coherent with stage/phase
- BOS_CONFIRMED rows coherent
- reversal is terminal (once reversal appears, it never leaves reversal)【fileciteturn1file11】

### Zone visualization invariants
- Chart shows zones for most recent structure_id
- Within that structure, active zones are most recent buy and sell
- Deterministic draw ordering (inactive under active; older under newer)【fileciteturn2file0】

---

## Branching + versioning rules (process)

- **One branch per week** (e.g., `week6-kl-zones`) branched off `main`.
- Short-lived day/topic branches allowed.
- Merge to `main` only when that week’s Definition of Done is satisfied.
- Keep replay outputs for “golden” scenarios to detect regressions.

(These are project-level agreements; treat them as hard guardrails.)

