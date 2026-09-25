# KL Zones v1 Spec (through Week 6)

Implementation: `engine_v2/zones/kl_zones_v1.py`.【fileciteturn2file10】

KL zones are event-driven rectangles derived from **market structure confirmation events**.
They are intended to be visually validated (and later traded) as “key levels”/supply-demand style zones.

---

## Canonical semantics (authoritative)

Originally copied from the canonical spec block in the module docstring (`kl_zones_v1.py`). The
indexing bullets below were corrected 2026-09-22 against `ARCHITECTURE.md` "`ev.idx` convention"
(the canonical per-event field table); the module docstring still carries the older "level
index" wording for both events (a `.py` follow-up).

### Identifiers
- `structure_id`: market structure unit id (directional regime). Starts at 0. Increments on reversal.
- `cts_cycle_id`: internal CTS/BOS cycle id within a structure. Starts at 0.

### StructureEvent indexing (the two zone-creating events)
`ev.idx` is **event-specific** — it is not uniformly "the level" nor "when it is known":
- `BOS_CONFIRMED.idx` = the BOS **extreme** (the level; `<= confirmed_at` on the normal path —
  34/34 rows on the reference window; not asserted in code).
  `meta["confirmed_at"]` = the breakout's apply candle — the cycle's CTS-established moment.
- `CTS_CONFIRMED.idx` = the **confirmation candle** (the pullback apply candle, or the
  sd-proximity candle) and `== meta["confirmed_at"]`; it is NOT the CTS level. The level — the
  current CTS extreme at confirmation — is `meta["cts_anchor_idx"]`.

### Zone indexing
- `meta["base_idx"]`: FIRST candle of the zone base pattern (where the rectangle begins) — at or before
  the zone's `anchor_idx` (see "base_idx by Pattern Type"); a different field from the anchor.
- `meta["source_event_idx"]`: the source event's RAW `ev.idx` — BOS: the BOS extreme; CTS: the confirmation candle (so it equals the CTS zone's raw
  `confirmed_idx`). Write-only (no reader; a declared raw reader, deleted in Plan E E4b-pre). Slice-local
  (not shifted by `slice_begin`) in the M15 lens CSVs — it is not in `_ZONE_META_IDX_KEYS`.
- `meta["anchor_idx"]`: the candle base-pattern identification starts from — BOS:
  `BOS_CONFIRMED.meta["bos_anchor_idx"]` (the BOS anchor, via `event_fields.bos_anchor_idx`; Plan E E2c); CTS: `CTS_CONFIRMED.meta["cts_anchor_idx"]` (the CTS
  anchor at confirmation). A market-structure-realm anchor — **not** `CTS_ESTABLISHED.meta
  ["pattern_anchor_idx"]` (the breakout pattern's first candle, a pattern-realm anchor); see GLOSSARY
  "Naming Standard" / ARCHITECTURE.md "Anchor has two realms".
- `meta["confirmed_idx"]`: candle index where the zone becomes confirmed for charting — raw value
  `ev.meta["confirmed_at"]` for both kinds (fallback `ev.idx`), then clamped up to the structure
  lifecycle-start (see "Lifecycle-start clamp" below):
  - BOS-derived: the breakout's apply candle (the moment)
  - CTS-derived: the confirmation candle (`== ev.idx`)

### Chart rules
Canonical in `charting/CHARTING_SPEC.md`: KL zone opacity is 3-tier — `meta["status"]=="active"` → active; else the
most recent `structure_id` → recent_inactive; else prior_inactive. (The old `meta["active"]` flag was retired in
Phase 3, 2026-05-26, when KL adopted the active/inactive/ended convention.)

---

## Pipeline placement

KL zones are computed after structure, with **base patterns identified on-demand** during zone creation:
- candle features → patterns → imbalance → structure → zones (base patterns identified here)【fileciteturn2file1】

---

## Base Pattern Identification (Structure-Aware)

Base patterns are now identified **on-demand** when a BOS/CTS event is received, using the zone's anchor_idx (the BOS / CTS extreme — "Zone indexing" above) and struct_direction context. Pattern identification is performed by `identify_base_pattern()`.

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
  3) Determine anchor_idx (stored as the zone's `meta["anchor_idx"]` — a different field from
     `CTS_ESTABLISHED.meta["pattern_anchor_idx"]`, the breakout pattern's first candle):
     - BOS: anchor_idx = source_event_idx (the BOS extreme)
     - CTS: anchor_idx = ev.meta["cts_anchor_idx"] (the CTS extreme at confirmation; fallback to
       source_event_idx)
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

**Pinbar inner-side bound (2026-06-20, follow-up (c)).** Before the
closest-neighbour pick, candidates **beyond the outer** are dropped (outer=LOW
→ keep `>= ref`; outer=HIGH → keep `<= ref`), so the inner can never sit beyond
the base candle's own extreme (no inverted zone). If every i±1 neighbour O/C is
beyond the outer (or no neighbour exists), it falls back to the base candle's
own body point closest to `ref` (always within the outer). This is the **only**
change vs the original rule and is **inert in the normal case** (closest
neighbour already within the outer) → byte-identical.

**Pinbar deliberately keeps the tight closest-neighbour rule — NOT the
`find_base_threshold` inner-edge rule.** Delegating pinbar to that rule was
tried and reverted: its ±5 pooling + 2nd-closest widening + body-bottom-only
selection *over-widened* pinbar zones (that rule is for `base` / `inside bar`).
Pinbar zones key off the pinbar's own body/tail and must stay tight; only the
inner-side bound is borrowed. The two neighbour-based inner derivations are
intentionally NOT unified.

**2-candle and star families need no bound — inversion-safe by construction:**
their inner is always an `o`/`c`/`mid_price` of a candle *within* the pattern
window, and `compute_base_window_features` sets `base_low/base_high` = min/max
over that *whole* window, so `inner ∈ [base_low, base_high]` ⊆ outer always.
(Pinned by `tests/test_kl_zone_thresholds.py`.)

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

A cycle ends at the first of (`zones/structure_lifecycle.compute_cycle_lifecycle`):
1. **reversal** of its structure, or
2. **next cycle starts** — the `(structure_id, cycle_id+1)` cycle's **clamped
   lifecycle-start** = `max(CTS_ESTABLISHED.meta["confirmed_at"], struct_start[,
   lifecycle_floor])` — the CTS-established **moment**, not the extreme
   (Plan C, 2026-09-20; see "End-side change" below), or
3. **the sub's window end** (subs only) — `lifecycle_cap`.

```python
end_idx = None; end_reason = None
if sid has a reversal:                end_idx, end_reason = reversal_idx[sid], "reversal"
if (sid, cycle_id+1) exists and its clamped start < end_idx (or end_idx is None):
                                      end_idx, end_reason = next_cycle_clamped_start, "next_cycle"
if lifecycle_cap is not None and lifecycle_cap < end_idx (or end_idx is None):
                                      end_idx, end_reason = lifecycle_cap, cap_reason
# else: end_idx = None  → zone extends to end of data
```

Both the BOS_n and CTS_n zone of cycle *n* inherit this **same** `end_idx` /
`end_reason`. `end_time = df.time[end_idx]` (or `None`). For subordinate
(lower-TF) structures the cap is the **unique sub's real-time `end_idx`**
(slice-local) with `cap_reason` = the sub's `end_reason`, passed as
`lifecycle_cap` / `cap_reason` by
`multitf/entity_df_mutation.py::render_sub_projection` →
`pooled_structure_build.project_to_window` → `_run_downstream_pipeline` →
`derive_kl_zones_v1` (Plan C, 2026-09-20 — before Plan C the cap was applied in
the now-deleted `build_one_sid` with `end_reason="lifecycle_end"`). `cap =
None` for an open sub (the data edge is not a lifecycle terminator) and for
main.

### End-side change: BOS ends align to the cycle boundary (CTS-established moment)

> **CORRECTED 2026-09-19 — the boundary is the MOMENT, not the extreme. LANDED by
> Plan C 2026-09-20.** A cycle's lifecycle begins when it is *established* —
> `CTS_ESTABLISHED.meta["confirmed_at"]` (the apply candle, == `BOS_CONFIRMED.meta["confirmed_at"]`
> by definition). `CTS_ESTABLISHED.ev.idx` is the CTS **extreme**, a historical anchor like
> `BOS_CONFIRMED.ev.idx`; it can precede the moment. Extreme == moment is the COMMON case, not a
> coincidence (31 of 34 `CTS_ESTABLISHED` CSV rows on the reference window, all five H1 cycles;
> the bound, the lagging rows and why they lag: canonical: `ARCHITECTURE.md` "`ev.idx` convention").
> `structure_lifecycle.compute_cycle_lifecycle` now reads the moment — for main, every sub cycle and
> the sub-structure parent tables (`multitf/parent_tables.py`) alike — and **raises** (`AssertionError`)
> on a `CTS_ESTABLISHED` without `meta["confirmed_at"]` rather than falling back to the extreme.
> Measured on the first Plan C replay (2026-09-20): H1 byte-identical; ONE visible sub shift (sub
> `454/+1`'s cycle-1 end 1223→1224); the two predicted 2828→2829 shifts are masked by the identical
> record floor 2829 on both lenses (same value before and after). Lifecycle values are
> real-time; anchors are historical — never mix. (Historical: from Phase 3 (2026-05-26) until Plan C
> the code used the extreme, as the earlier revision of this section described.)

The cycle boundary is the next cycle's **clamped lifecycle-start** —
`max(CTS_{n+1} ESTABLISHED.meta["confirmed_at"], struct_start[, lifecycle_floor])`,
the CTS-established *moment* (the same value POI inherits through
`compute_cycle_lifecycle`). Both the BOS_n and CTS_n zone of cycle *n* end there.

| Zone | Old end (pre-Phase-3) | End (cycle-end, Plan C) | Change |
|------|---------|---------------------|--------|
| **CTS_n** | next CTS established `ev.idx` (early-end) | next cycle's clamped start (moment) | none where extreme == moment (the common case); later by the extreme→moment lag where the extreme precedes the moment (see the note above) |
| **BOS_n** | next BOS's `confirmed_at` (breakout, same-side replace) | next cycle's clamped start (moment) | **none** by definition when the next cycle's start is unclamped — `BOS_{n+1}.confirmed_at == CTS_{n+1}.confirmed_at` (the Phase-3 shift onto the extreme — that same extreme→moment lag — is undone by Plan C); later only when `struct_start` / `lifecycle_floor` clamps the next cycle's start |
| reversal-capped | reversal idx | reversal idx | label only (`end_reason` replaces `deactivated_by`) |
| sub-window-capped | — | the sub's `end_idx` | `end_reason` = the sub's `end_reason` |
| last / open zone | `None` / reversal | `None` / reversal | none |

Why the two definitions can differ: a breakout emits `BOS_{n+1} CONFIRMED`
(`confirmed_at` = breakout `apply_idx`) and `CTS_{n+1} ESTABLISHED`
(`ev.idx` = the breakout-window extreme, `meta["confirmed_at"]` = the same
`apply_idx`) together. When the apply candle *is* the extreme (the common case),
extreme and moment coincide; otherwise the extreme is earlier (bound and lag
figures: `ARCHITECTURE.md` "`ev.idx` convention"). **Empirically
(history):** all 10 H1 zones in baseline `e0b70dd` had `confirmed_at == ev.idx`,
so the Phase-3 extreme rule left H1 `end_time` byte-identical; on the M15 subs
(BOS-only zones) exactly one BOS zone per sub shifted 1 candle earlier under the
extreme rule (e.g. counter cyc1: 09:15→09:00). Plan C's moment rule brings a
cycle's BOS-zone end back onto the breakout candle that established the next
cycle, while keeping the unification that a cycle's BOS zone, CTS zone and POI
all end at the SAME idx.

Aside from that boundary alignment (the extreme→moment lag above), the end side is a *representation*
change (`active`→`status`, `deactivated_by`→`end_reason`, +`end_idx`). (See
the start-side clamp below for the other behavioral change.)

### Lifecycle-start clamp (start-side behavior change — REVISED 2026-05-26; floor terms REVISED by Plan C 2026-09-20)

A KL zone's first-active is clamped to its cycle's lifecycle-start
(`PART4_REFACTOR_SPEC.md §5` / `§17.4`): `first_active = max(confirmed_idx,
cycle_lifecycle_start)`, where `cycle_lifecycle_start = max(CTS_n ESTABLISHED
MOMENT (meta["confirmed_at"]), structure lifecycle-start[, lifecycle_floor])`.
`structure lifecycle-start` is `compute_struct_start_by_sid` (the structure's first
CTS_ESTABLISHED moment since Plan E E3f — before it the first anchor, BOS_0's — / the
reversal handoff); `lifecycle_floor` is `None` for main and, for a sub, the
**unique sub's real-time `start_idx`** (slice-local, from
`render_sub_projection`) — which already contains the parent-cycle floor
(`TriggerRecord.parent_floor_idx = LOH(max(struct_start[S], cts_moment[(S,C)]))`)
because the record's `start_idx = max(probe_finalize_idx, trigger_idx,
parent_floor_idx)`. (Historical: plan B1, 2026-05-27, wrote the sub floor as
separate "parent sid / parent_cycle_id lifecycle-start" terms; Plan C folds
them into the one `lifecycle_floor` value.) Implementation note: in
`derive_kl_zones_v1` the per-zone clamp is `confirmed_idx = max(raw
confirmed_idx, struct_start_by_sid[sid])`; the `CTS_n` moment term is
satisfied by construction for the BOS_n zone (its raw `confirmed_idx` IS
`BOS_CONFIRMED.meta["confirmed_at"]` = the moment) and for the CTS_n zone (its
`CTS_CONFIRMED` confirmation-candle `confirmed_idx` is later).

- The CTS_n zone's `confirmed_idx` (its confirmation candle — pullback or
  sd-proximity) is at or after the cycle start (a pullback confirmation is strictly later; a
  proximity confirmation can fire on the apply candle itself when the extreme precedes it), so it
  is rarely clamped.
- The BOS_n zone's `confirmed_idx` (the breakout) equals the cycle's
  CTS-established moment for *normal* cycles, but for **post-reversal cycle 0** the
  probe can place the structure's anchor historically so the cycle-0
  CTS-established moment precedes the reversal confirmation. Then the BOS (and CTS)
  zone's first-active snaps forward to the structure lifecycle-start. (Live
  instance: sid 1 cycle 0 zones at idx 703 → clamp to the sid-0 reversal at
  902 on the 2025-11-15 window — 710 on the older 2025-12-01 window; here the
  clamped start 902 equals the cycle end 902, so sid 1 cycles 0–1 collapse to
  `status="inactive"`.)
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
| `end_reason` | `"reversal"` \| `"next_cycle"` \| the sub's `cap_reason` — `"reversal"` \| `"same_dir_replacement"` \| `"parent_end"` — when the sub's `end_idx` capped the cycle (subs only; Plan C 2026-09-20 — `"lifecycle_end"` is no longer emitted) \| `None` |
| `activation_history` | KL has no condition-state flips, so it is the single interval `[{"idx": clamped_first_active, "active": True, "reason": "confirmed"}]`, or `[]` if the clamped first-active is at/after `end_idx` (collapsed — never active) |
| `status` (derived) | `"active"` (clamped-confirmed, not ended) \| `"ended"` (`t >= end_idx`) \| `"inactive"` (before clamped first-active, or collapsed). A COLLAPSED zone (`"inactive"` with clamped `confirmed_idx >= end_idx` — the cycle's whole span precedes its structure's lifecycle start: a sub's forming-phase cycles, H1's retroactive post-reversal cycles) is kept in the data/CSVs but NOT drawn on any chart since 2026-09-20 (`charting/_zone_render.is_collapsed_cycle_zone`). |

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

CTS zones ended at `CTS_ESTABLISHED` (not at the next `CTS_CONFIRMED`). When a new CTS is established, the previous CTS zone's `end_time` is set to the time of the CTS_ESTABLISHED `ev.idx` candle (the new CTS extreme — not the establishing moment), and its `active` flag is set to `False` with `deactivated_by = "cts_established"`.

This means there can be periods with **no active CTS zone** — between CTS_ESTABLISHED (old zone ends) and CTS_CONFIRMED (new zone created). BOS zones are unaffected; they still end when replaced by a new BOS zone of the same side.【fileciteturn2file0】

