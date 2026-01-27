# Architecture & System Design (through Week 6)

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

### Charting
Charting reads from:
- dataframe columns
- `df.attrs["kl_zones"]`
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

