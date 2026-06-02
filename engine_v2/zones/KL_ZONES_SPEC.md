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

### `find_base_threshold` inner (for `base` / `base inside bar`) — inner-edge rule (2026-06-01)

The outer of a 1-candle base is the base candle's own extreme (`base_high` for
BOS sd=−1 / CTS sd=+1; `base_low` for BOS sd=+1 / CTS sd=−1 — the `use_low_ref`
split). `find_base_threshold` derives the **inner** from the ±5 neighbours
(pre + post pooled, ranked by price — no positional precedence) using each
neighbour's **inner-edge body point** — the body extreme on the side facing the
zone interior:

- outer = `base_high` → inner-edge = `min(o,c)` (bottom of body = `o` of a
  bullish candle, `c` of a bearish one)
- outer = `base_low` → inner-edge = `max(o,c)` (top of body = `c` bullish, `o` bearish)

Rule: **drop neighbours whose inner-edge point is beyond the outer** (on its far
side), then set the inner at the **2nd inner-edge point closest to the outer**
(two neighbours back it). Only that single point is tested — *not* both `o` and
`c` — so a neighbour whose far-side body extreme pokes past the outer still
qualifies on its near side. Fallback: the single closest qualifying neighbour,
else NaN.

This guarantees the base extreme stays the **outermost** edge, the inner sits
**within** it (so the zone contains the base; inner never beyond the base
candle's own high/low), and it avoids degenerate too-narrow zones. Prior code
tested both `o` and `c` with no inner-side bound, so a neighbour body beyond the
outer could become the inner → an **inverted** zone sitting outside the base
(see GOTCHAS "Base/inside-bar zone inner could land beyond the outer").

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

## Lifecycle: zones inherit their cycle's end (Unified model — REVISED 2026-05-26, Phase 3)

> **This section supersedes the prior "Active / inactive zones" + "CTS zone
> early ending" behavior.** Before Phase 3, KL zones computed their own end
> via three scattered mechanisms (CTS_ESTABLISHED early-end, same-side
> replacement, reversal/lifecycle caps). Those are removed. A KL zone now
> computes **no end of its own** — it inherits the resolved end of the cycle
> that owns it, exactly as POI zones already do
> (`zones/poi_zones.py`). This is the KL half of the pass-through lifecycle
> model in `PART4_REFACTOR_SPEC.md §5` and the active/inactive/ended
> convention in `ARCHITECTURE.md`.

### Cycle ownership

Every KL zone belongs to a `(structure_id, cycle_id)`, read from
`meta["structure_id"]` and `meta["cycle_id"]` (= `cts_cycle_id` at the
zone's `confirmed_idx`). Within one structure, cycle *n* owns exactly two
zones: the **BOS_n** zone (buy if `sd=+1`, sell if `sd=-1`) and the
**CTS_n** zone (the opposite side).

### End resolution (pure inheritance — identical to POI)

A cycle ends at the first of:
1. **reversal** of its structure, or
2. **next cycle starts** — the `(structure_id, cycle_id+1)` `CTS_ESTABLISHED` idx.

```python
end_idx = None; end_reason = None
if sid has a reversal:                end_idx, end_reason = reversal_idx[sid], "reversal"
if (sid, cycle_id+1) CTS_ESTABLISHED exists and < end_idx (or end_idx is None):
                                      end_idx, end_reason = next_cts_est_idx, "next_cycle"
# else: end_idx = None  → zone extends to end of data
```

Both the BOS_n and CTS_n zone of cycle *n* inherit this **same** `end_idx` /
`end_reason`. `end_time = df.time[end_idx]` (or `None`). For subordinate
(lower-TF) structures the owning structure also ends at parent-cycle-end;
that propagation is applied as a cap with `end_reason="lifecycle_end"` in
`multitf/entity_df_mutation.py::build_one_sid` (the structure-end →
open-cycle-end → zone-end pass-through for subs).

### End-side change: BOS ends align to the cycle boundary (CTS-established extreme)

The cycle boundary is the next cycle's `CTS_ESTABLISHED` **`ev.idx`** — the
CTS *extreme* candle (the same idx POI uses: `cts_established_by_key[next].idx`).
Both the BOS_n and CTS_n zone of cycle *n* now end there.

| Zone | Old end | New end (cycle-end) | Change |
|------|---------|---------------------|--------|
| **CTS_n** | next CTS established `ev.idx` (early-end) | next CTS established `ev.idx` | **none** — already this idx |
| **BOS_n** | next BOS's `confirmed_at` (breakout, same-side replace) | next CTS established `ev.idx` (extreme) | **≤1 candle earlier** — see below |
| reversal-capped | reversal idx | reversal idx | label only (`end_reason` replaces `deactivated_by`) |
| last / open zone | `None` / reversal | `None` / reversal | none |

The BOS shift size depends on whether the breakout candle is itself the CTS
extreme: a breakout emits `BOS_{n+1} CONFIRMED` (`confirmed_at` = breakout
`apply_idx`) and `CTS_{n+1} ESTABLISHED` (`ev.idx` = the breakout-window
extreme) together. When the breakout candle *is* the extreme they coincide
(`confirmed_at == ev.idx`) and the BOS end is unchanged; otherwise `ev.idx`
is 1 candle earlier. **Empirically:** all 10 H1 zones in baseline `e0b70dd`
had `confirmed_at == ev.idx`, so H1 `end_time` is byte-identical; on the M15
subs (BOS-only zones) exactly one BOS zone per sub shifts 1 candle earlier
(e.g. counter cyc1: 09:15→09:00). This is the intended unification — BOS
ends now match where CTS/POI of the same cycle end, removing the old
1-candle inconsistency between a cycle's BOS and CTS zone ends.

Aside from that ≤1-candle BOS alignment, the end side is a *representation*
change (`active`→`status`, `deactivated_by`→`end_reason`, +`end_idx`). (See
the start-side clamp below for the other behavioral change.)

### Lifecycle-start clamp (start-side behavior change — REVISED 2026-05-26)

A KL zone's first-active is clamped to its cycle's lifecycle-start
(`PART4_REFACTOR_SPEC.md §5`): `first_active = max(confirmed_idx,
cycle_lifecycle_start)`, where `cycle_lifecycle_start = max(CTS_n
ESTABLISHED idx, structure lifecycle-start[, parent sid lifecycle-start[,
parent_cycle_id lifecycle-start]])`. (The `parent_cycle_id` floor is sub-only and
**implemented 2026-05-27 — plan B1**; canonical model + mapping in
`PART4_REFACTOR_SPEC.md §5`.)

- The CTS_n zone's `confirmed_idx` (its pullback candle) is always after the
  cycle start, so it is rarely clamped.
- The BOS_n zone's `confirmed_idx` (the breakout) equals the cycle's
  CTS-established for *normal* cycles, but for **post-reversal cycle 0** the
  probe can place the structure's anchor historically so cycle-0
  CTS-established precedes the reversal confirmation. Then the BOS (and CTS)
  zone's first-active snaps forward to the structure lifecycle-start. (Live
  instance: sid 1 cycle 0 zones at idx 703 → clamp to the sid-0 reversal at
  710.)
- The zone **rectangle is still drawn from `base_idx`** (historical anchor);
  only first-active / `confirmed_idx` / `status` move. If the clamped
  first-active lands at/after `end_idx`, the zone never becomes active
  (`status` never `"active"`; kept in the list as history).

### State convention fields (in `meta`, mirroring POI)

KLZone follows the active/inactive/ended convention
(`ARCHITECTURE.md`). The fields live in `meta` (KLZone is a frozen
dataclass; POI stores the same set in `meta`):

| Field | Meaning |
|-------|---------|
| `end_idx` | Terminal idx if the cycle ended, else `None` |
| `end_reason` | `"reversal"` \| `"next_cycle"` \| `"lifecycle_end"` (subs) \| `None` |
| `activation_history` | KL has no condition-state flips, so it is the single interval `[{"idx": clamped_first_active, "active": True, "reason": "confirmed"}]`, or `[]` if the clamped first-active is at/after `end_idx` (collapsed — never active) |
| `status` (derived) | `"active"` (clamped-confirmed, not ended) \| `"ended"` (`t >= end_idx`) \| `"inactive"` (before clamped first-active, or collapsed) |

`status` is derived once at end-of-data (like POI). The retired
`meta["active"]` boolean and `meta["deactivated_by"]` string are gone —
consumers read `meta["status"]`. `confirmed_idx` in meta is the **clamped**
first-active idx (`max(raw confirmed_idx, cycle lifecycle-start)`).

### Charting (rectangle extent unchanged; confirm marker may move)

The chart still shows the most-recent buy + sell zone of the most-recent
structure at full opacity, and tiering keys on `meta["status"]=="active"`
instead of the old `meta["active"] AND end_time is None`. Rectangle extents
are unchanged (`base_idx` → `end_time`, both timing-invariant). The only
visible shift is for **post-reversal cycle-0 zones**, whose confirm marker /
active-start moves forward to the clamped lifecycle-start.

### Prior model (historical, pre-Phase-3)

For reference, the removed behavior was:

CTS zones ended at `CTS_ESTABLISHED` (not at the next `CTS_CONFIRMED`). When a new CTS is established, the previous CTS zone's `end_time` is set to the CTS_ESTABLISHED candle's time, and its `active` flag is set to `False` with `deactivated_by = "cts_established"`.

This means there can be periods with **no active CTS zone** — between CTS_ESTABLISHED (old zone ends) and CTS_CONFIRMED (new zone created). BOS zones are unaffected; they still end when replaced by a new BOS zone of the same side.【fileciteturn2file0】

