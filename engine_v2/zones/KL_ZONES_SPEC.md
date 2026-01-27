# KL Zones v1 Spec (through Week 6)

Implementation: `engine_v2/zones/kl_zones_v1.py`.【fileciteturn2file10】

KL zones are event-driven rectangles derived from **market structure confirmation events**.
They are intended to be visually validated (and later traded) as “key levels”/supply-demand style zones.

---

## Canonical semantics (authoritative)

This is copied from the canonical spec block in the module:

### Identifiers
- `structure_id`: market structure unit id (directional regime). Starts at 0. Increments on reversal.
- `cts_cycle_id`: internal CTS/BOS cycle id within a structure. Starts at 0.

### StructureEvent indexing
- `ev.idx`: the *level index* (where the BOS/CTS level is anchored; often an earlier extreme).
- `ev.meta["confirmed_at"]`: candle index where that level was confirmed (breakout/pullback timing).

### Zone indexing
- `meta["base_idx"]`: anchor candle of the zone base pattern (where rectangle begins).
- `meta["source_event_idx"]`: the StructureEvent level index used to derive the zone (ev.idx).
- `meta["confirmed_idx"]`: candle index where the zone becomes confirmed for charting:
  - BOS-derived: confirmed_idx = `ev.meta["confirmed_at"]` (breakout candle)
  - CTS-derived: confirmed_idx = `ev.idx` (pullback candle)

### Chart rules
- Show zones for the most recent `structure_id`.
- Within that structure, the most recent buy and sell zones have higher opacity (`active=True`).【fileciteturn2file2】

---

## Pipeline placement

KL zones are computed after structure, but **base features are computed before structure**:
- candle features → patterns → base features → structure → zones【fileciteturn2file1】

---

## Base features

`compute_base_features(df, length_threshold=0.7)` ports a legacy base-pattern classification and produces:
- `base_pattern` (categorical: "no base", "no base 1st big", "no base 2nd big", star variants, long-tail variants, etc.)
- base OHLC-derived helper columns:
  - `base_low`, `base_high`
  - `base_min_close_open`, `base_max_close_open`
  - `mid_price` etc.

It includes the corrected “no base 1st big” vs “no base 2nd big” length-threshold condition.【fileciteturn2file10】

---

## Creating a zone from a StructureEvent

`derive_kl_zones_v1(df, events, struct_direction)` iterates structure events in order:
- For each eligible event (CTS_CONFIRMED / BOS_CONFIRMED), create a zone:
  1) Determine `source_event_idx = ev.idx`
  2) Determine `confirmed_idx`:
     - `confirmed_idx = ev.meta["confirmed_at"]` when present else source_event_idx
  3) Determine base source:
     - BOS: base_source_idx = source_event_idx
     - CTS: base_source_idx = ev.meta["cts_anchor_idx"] (fallback to source_event_idx)
  4) Resolve (base_idx, base_pattern) via `resolve_base_idx_and_pattern(...)`
  5) Compute thresholds via `zone_thresholds(...)`
  6) Map side based on struct_direction + event type
  7) Produce `KLZone` with meta, including bounds_steps list initialized with INIT segment【fileciteturn2file6】

### Side mapping (locked)
- If sd=+1:
  - BOS → buy zone
  - CTS → sell zone
- If sd=-1:
  - BOS → sell zone
  - CTS → buy zone【fileciteturn2file6】

---

## Zone thresholds (outer/inner)

`zone_thresholds(...)` returns (outer, inner), then the engine converts to (top/bottom) for charting.

### Pinbar-specific inner threshold
`find_pinbar_threshold` chooses the neighbor open/close closest to the correct extreme reference,
where the reference depends on BOS/CTS and struct_direction:

- BOS, sd=+1 → reference = LOW
- CTS, sd=+1 → reference = HIGH
- BOS, sd=-1 → reference = HIGH
- CTS, sd=-1 → reference = LOW【fileciteturn2file12】

Other base_pattern mappings use `mid_price`, `base_min_close_open`, `base_max_close_open`, or a generalized `find_base_threshold(...)` fallback.【fileciteturn2file8】

---

## Zone expansion

Zones maintain `meta["bounds_steps"]`:
- Each step has:
  - `start_idx`: where the segment begins (INIT = base_idx; expansions begin at event idx where expansion happens)
  - `top`, `bottom`
  - `event` (INIT / CTS_THRESHOLD_UPDATED / BOS_THRESHOLD_UPDATED / etc)
  - optional `price`

When later threshold update events imply bounds extension:
- The zone is replaced with updated top/bottom and an appended bounds_steps entry, and meta flags `expanded` and `expanded_last_*` are set.【fileciteturn2file6】

> Note: The current system ties zone expansion to emitted threshold-update events (e.g., CTS threshold updates that come from range sync). This is deliberate to avoid incorrect expansions based on unrelated values.

---

## Active / inactive zones

The engine maintains:
- 1 active buy + 1 active sell zone per structure_id (the most recent of each)
- Older zones become inactive but still visualized under active zones.
Charting uses opacity to convey active vs inactive.【fileciteturn2file0】

