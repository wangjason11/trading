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

KL zones are computed after structure, with **base patterns identified on-demand** during zone creation:
- candle features → patterns → imbalance → structure → zones (base patterns identified here)【fileciteturn2file1】

---

## Base Pattern Identification (Structure-Aware)

Base patterns are now identified **on-demand** when a BOS/CTS event is received, using the anchor_idx and struct_direction context. Pattern identification is performed by `identify_base_pattern()`.

### Pattern Check Order
1. **Inside bar pattern** (new): Check if anchor candle has ≥2 candles within its range (5 left + 5 right)
2. **2-candle patterns**: "no base", "no base 1st big", "no base 2nd big", "no base long tails up/down"
3. **1-candle patterns**: "no base big tail up", "no base big tail down" (pinbar)
4. **3-candle patterns**: "no base star", "no base star 1st big", "no base star 2nd big"
5. **Default**: "base" (fallback)

### 2-Candle Positioning Logic
| Event | Condition | idx1 (1st candle) | idx2 (2nd candle) |
|-------|-----------|-------------------|-------------------|
| BOS | BOS dir == struct_dir | anchor - 1 | anchor |
| BOS | BOS dir != struct_dir | anchor | anchor + 1 |
| CTS | CTS dir == struct_dir | anchor | anchor + 1 |
| CTS | CTS dir != struct_dir | anchor - 1 | anchor |

### 3-Candle (Star) Pre-conditions
- BOS: candle 3 direction must equal struct_direction
- CTS: candle 1 direction must equal struct_direction

### base_idx by Pattern Type
| Pattern Type | base_idx |
|--------------|----------|
| Inside bar | anchor |
| 1-candle (pinbar) | anchor |
| 2-candle | idx1 (1st candle of pattern) |
| 3-candle (star) | anchor - 1 |
| "base" (catchall) | anchor |

### Feature Computation
Base window features (`base_low`, `base_high`, etc.) are computed on-the-fly via `compute_base_window_features()` based on pattern type and base_idx.【fileciteturn2file10】

---

## Creating a zone from a StructureEvent

`derive_kl_zones_v1(df, events, struct_direction)` iterates structure events in order:
- For each eligible event (CTS_CONFIRMED / BOS_CONFIRMED), create a zone:
  1) Determine `source_event_idx = ev.idx`
  2) Determine `confirmed_idx`:
     - `confirmed_idx = ev.meta["confirmed_at"]` when present else source_event_idx
  3) Determine anchor_idx:
     - BOS: anchor_idx = source_event_idx
     - CTS: anchor_idx = ev.meta["cts_anchor_idx"] (fallback to source_event_idx)
  4) Identify (base_pattern, base_idx) via `identify_base_pattern(df, anchor_idx, struct_direction, bos=...)`
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
- Charting uses opacity to convey active vs inactive.

### CTS zone early ending

CTS zones end at `CTS_ESTABLISHED` (not at the next `CTS_CONFIRMED`). When a new CTS is established, the previous CTS zone's `end_time` is set to the CTS_ESTABLISHED candle's time, and its `active` flag is set to `False` with `deactivated_by = "cts_established"`.

This means there can be periods with **no active CTS zone** — between CTS_ESTABLISHED (old zone ends) and CTS_CONFIRMED (new zone created). BOS zones are unaffected; they still end when replaced by a new BOS zone of the same side.【fileciteturn2file0】

