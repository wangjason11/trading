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
| **ended (terminal)** | Time-based irrelevance | **No** — once ended, stays ended | A defined set of end conditions (next cycle, reversal, lifecycle_end, etc.) |

Once ended, "active" is undefined / always False. Before ended, the object
flips between active and inactive based on its activation conditions.

Recommended representation for any new lifecycle object:

| Field | Type | Meaning |
|---|---|---|
| `end_idx` | `Optional[int]` | Terminal idx if known; `None` if not yet ended |
| `end_reason` | `Optional[str]` | Why it ended (`"next_cycle"`, `"reversal"`, `"lifecycle_end"`, `"obsolete:<reason>"`, ...) |
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

**Currently following this convention:** `POIZone` and `KLZone` (both store
`end_idx` + `end_reason` + `activation_history` + derived 3-state `status` in
`meta`). `KLZone` joined the convention in the Phase 3 unified-lifecycle pass
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
max(CTS_n ESTABLISHED idx, structure lifecycle-start[, parent sid
lifecycle-start])`. This prevents a zone from activating before its
structure is alive — e.g. a post-reversal cycle-0 zone whose
`CTS_ESTABLISHED` precedes the reversal confirmation. See
`PART4_REFACTOR_SPEC.md §5` (starting_idx vs lifecycle-start + the clamp).

**Not yet following:** `FibState` (the `active` flag does double duty —
condition-state for imbalance check AND terminal flag for `new_cycle` /
`scenario1_revert` / `cross_failed` / `lifecycle_end`). The convention
*applies in principle* — FibState genuinely has both axes (imbalance-fill is
reversible condition-state; the four `deactivated_by` reasons are terminal) —
and the migration is now **fully DESIGNED** (2026-05-27, not yet implemented):
the cross-fib versioning (`cross_shortened` / `cross_failed` spawning new
`version`s) that complicated the `activation_history` model is resolved by making
the **cycle the lifecycle identity** with versions as an internal sub-axis, plus a
tracker-level `(sid, cycle)` history projection that unifies the two storage
subsystems without refactoring them. It still spans ~500 lines across
`fib_tracker.py`, POI lifecycle, and chart/debug consumers, so it remains its own
staged build. Canonical design: `zones/FIB_LIFECYCLE_SPEC.md`; tracked in
`memory/project_lifecycle_convention_klzone_fibstate.md` +
`memory/project_fib_lifecycle_design.md`.

**Deliberately NOT lifecycle objects:** structure events
(`df.attrs["structure_events"]`) and candle / structure patterns. These are
**immutable, append-only historical facts** (`PART4_REFACTOR_SPEC.md §7`) —
"what the algorithm believed at the time." They are never mutated and carry
no `active`/`ended` state. Their only time-varying property is **currency**
("which sid/cycle owns candle *t* for display"), and that is a *query-time
derivation* (`owner_by_idx` / the §16.5 most-recent-sid chart filter), not
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

### Multi-TF Analysis (`multitf/`)
Subordinate lower-TF structures triggered by higher-TF events. Foundation supports UC1 (15M reverse structure from H1 CTS).

**UC1 flow:** H1 `CTS_CONFIRMED` + WVMI activation → detect trigger (`uc1_trigger.py`) → fetch/prepare M15 data (`data_bridge.py`) → H1 reverse probe → map to M15 → plain structure + downstream pipeline (`lower_tf_pipeline.py`).

**H1 reverse probe:** Runs `compute_structure_scenario_3()` on H1 data with `end_idx` set to the first sd zone-proximity trigger candle (`triggered_by_event_idx` from the cycle's WVMI record meta) and `run_continuation=False` to find a validated start candle for M15. Probe window is [cts_idx, triggered_by_event_idx]. The H1 probe results (events, levels) are discarded — only `start_idx` is used.

**Key design decisions:**
- M15 structure uses opposite direction to H1 (`lower_sd = -1 * h1_sd`)
- H1 reverse probe validates start; validated H1 candle mapped to M15 extreme match
- M15 runs `compute_structure_from_start()` (no probes) — start is pre-validated by H1 probe
- M15 slice includes 50-candle lookback buffer for neighbor-dependent calculations
- KL zones are BOS-only (`source_kinds=["BOS"]`), Fib uses imbalance-gated cross-cycle mode (`fib_mode="cross_cycle"`)
- Lifecycle bounded by parent H1 cycle (ends at next BOS or reversal)
- All events/zones carry attribution: `timeframe`, `use_case`, `parent_tf`, `parent_sid`, `parent_cycle_id`
- Chart renders M15 zones as dashed rectangles with lower opacity

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

