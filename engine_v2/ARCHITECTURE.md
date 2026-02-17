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
0. **Activated** — `check_proximity_activation()` scans candles from CTS_CONFIRMED+1 to zone deactivation (next BOS or reversal). Only cycles where price approaches within 20 pips of the closest active zone inner bound (KL or POI) proceed.
1. **Created** at CTS_n confirmation (only if activated) — breakout momentum locked from FB/LB volumes
2. **Updated** each candle — temporary LP shifts to closest qualified candle near outer bound
3. **Locked** at BOS_n+1 confirmation — LP finalizes, pullback momentum locked

Results stored in `df.attrs["wvmi"]` (list of `WVMIRecord`). See `WVMI_SPEC.md`.

### Scenario 3 (`structure/structure_engine.py`)
Arbitrary-start structure analysis with iterative BOS_0 probe. Phase 1 validates/refines `start_idx` by checking if price reaches the BOS_0 zone inner bound (within configurable pip tolerance: H1=10, M15=3, M5=1). Phase 2 continues multi-structure analysis from the finalized probe using the same logic as `compute_structure`. Returns `Scenario3Result` with status "finalized" or "pending".

### Multi-TF Analysis (`multitf/`)
Subordinate lower-TF structures triggered by higher-TF events. Foundation supports UC1 (15M reverse structure from H1 CTS).

**UC1 flow:** H1 `CTS_CONFIRMED` + WVMI activation → detect trigger (`uc1_trigger.py`) → fetch/prepare M15 data (`data_bridge.py`) → run Scenario 3 + downstream pipeline on M15 slice (`lower_tf_pipeline.py`).

**Key design decisions:**
- M15 structure uses opposite direction to H1 (`lower_sd = -1 * h1_sd`)
- Start mapped from H1 CTS extreme candle (not confirmation candle) to M15 extreme match
- M15 slice includes 50-candle lookback buffer for neighbor-dependent calculations
- KL zones are BOS-only (`source_kinds=["BOS"]`), Fib uses imbalance-gated mode (`fib_mode="m15_reverse"`)
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

