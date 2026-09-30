# POI Zones Specification

> Point of Interest (POI) Zones using Fibonacci retracement and Institutional Candle identification.

---

## Overview

POI Zones are derived from:
1. **Imbalance Candle Pattern** — Identifies large price movements (standard FVG)
2. **Fibonacci Retracement Levels** — Drawn between BOS and CTS anchor points
3. **Institutional Candle (IC)** — Located within 61.8%-80% Fib bounds; its high/low define the zone

**3 Variants** are created based on how much of the IC must be within the Fib zone.

---

## 1. Imbalance Candle Pattern

### Two concepts

- **Imbalance candle** — a c2 (middle) candle whose neighbors form an FVG
  *and* whose own direction matches the gap direction. Flagged per-candle via
  the `is_imbalance` dataframe column. Charting consumes this.
- **Imbalance instance** — one imbalance candle or a run of consecutive
  same-direction imbalance candles merged into a single entity with merged
  gap bounds. Stored as `ImbalanceInstance` objects in
  `df.attrs["imbalances"]`. Fib/POI logic consumes this.

A single-candle instance has `start_idx == end_idx`.

### Detection Logic (per-candle)
- **Bullish imbalance candle:** `c1.high < c3.low` AND `c2.direction == +1`
- **Bearish imbalance candle:** `c1.low > c3.high` AND `c2.direction == -1`
- The c2-direction requirement prevents counter-direction candles from being
  flagged (e.g., a gap with a bearish c2 is not a bullish imbalance).

### Merging Rule
Consecutive imbalance candles with the **same direction** are merged into a
single `ImbalanceInstance`. A direction break or a non-imbalance candle ends
the run.

### Merged Gap Bounds
- **Bullish:** `gap_bottom = df[start_idx-1].h` (first c1.high),
  `gap_top = df[end_idx+1].l` (last c3.low)
- **Bearish:** `gap_bottom = df[end_idx+1].h` (last c3.high),
  `gap_top = df[start_idx-1].l` (first c1.low)

### Implementation
- DF column `is_imbalance` (0/1) — every imbalance candle gets the flag
- `df.attrs["imbalances"]` — list of `ImbalanceInstance` with merged bounds
- Computed once in pipeline before market structure (`compute_imbalance`)
- Lower-TF (M15) pipelines re-run `compute_imbalance` after slicing because
  `df.attrs["imbalances"]` indices don't survive `reset_index(drop=True)`

### Role in POI Zones
- An unfilled imbalance must exist **overlapping the Fib's BOS→CTS span** for
  the Fib — and so its POIs — to exist (FibTracker's `has_unfilled_imbalance` check)
- Imbalance must be **after the IC candle** (between IC and the break) —
  POI IC validation passes `direction=sd` so only structure-direction
  imbalances qualify
- **"Exists" means FORMED (Plan F, 2026-09-24).** An instance exists from its
  first c3, `ImbalanceInstance.formed_at = start_idx + 1` — never from its c2.
  A question asked at a moment counts only instances formed by then, and only
  their **formed prefix** (`overlaps_formed_prefix`); the POI activation sweep
  enters an instance at `formed_at` (§4 "Lifecycle"). Canonical:
  `IMBALANCE_FILL_SEMANTICS.md` "Knowability — the c3 rule".

### All Fib + scenario + POI imbalance checks are sd-direction strict
Every consumer of `has_unfilled_imbalance` — Fib activation, Scenario 2
cond1/cond2/cond3, MarketStructure's in-flight cycle-0 snapshot, the cross-
cycle dead-cycle walks, AND POI IC validation — passes `direction=sd`. The
rationale is uniform: **Fibs only ever produce sd-direction POIs by
construction** (POIs are sd-direction — see §4 "POIs are always sd-direction"),
so a counter-direction imbalance inside a structural span cannot influence
any downstream POI and shouldn't drive the Fib's lifecycle either.

The pre-2026-05-23 design left Fib activation permissive on the grounds that
the BOS→CTS span is structurally directional. That reasoning still holds —
counter-direction imbalances in the span are geometrically unusual — but the
extra permissiveness was disconnecting cause (any-direction unfilled imbalance)
from effect (sd-direction POIs). The strict filter aligns the question
("should this Fib be drawn?") with what it actually affects downstream
("the sd-direction POI set").

The more load-bearing distinction across call sites is the primitive's **two
as-ofs** (Plan F, 2026-09-24):

- **`check_to_idx` — the fill horizon** (`is_filled` scans `(end_idx,
  check_to_idx]`). Fib lifecycle and scenario checks pass varying horizons: the
  handled event's moment (FibTracker, since Plan E E3a / E3a′), a
  fixed reference event's moment (BOS_1 for cond3, since E3a′), or the cross-cycle routine's
  `current_candle`. POI IC validation passes `check_to_idx = end_idx = cts_idx`,
  the fib's CTS.
- **`evaluated_at` — the moment the question is asked** (keyword-only,
  REQUIRED): only instances formed by then count. FibTracker passes the handled
  CTS event's moment (`event_fields.event_moment`); POI IC validation passes
  `evaluated_at=None`, an explicit "no cut" (§3.1 condition 3) — as do
  FibTracker's two cycle-0 cache writes and MarketStructure's in-flight reads
  (IMBALANCE_FILL_SEMANTICS call-site matrix). The cut is keyed on the moment,
  never on `check_to_idx`.

### Fill Check (per instance) — two-stroke state machine

Filled iff BOTH strokes fire by `check_to_idx`:

- **Stroke 1 — armed (70% retrace).** First candle in `(end_idx, check_to_idx]`:
  - Bullish: `low <= gap_top - gap_size * 0.70`
  - Bearish: `high >= gap_bottom + gap_size * 0.70`
- **Stroke 2 — confirmed (close past gap outer).** First candle at idx
  `>= armed_idx`:
  - Bullish: `close >= gap_top`
  - Bearish: `close <= gap_bottom`

Both strokes latch monotonically (stroke 1 then stroke 2; neither un-latches).
Scan starts at `end_idx + 1` — the **last c3**, which cannot arm its own gap
(its wick IS the gap edge; stroke 1 needs 70% of the gap past it); the
instance's c2s are excluded.

See `engine_v2/IMBALANCE_FILL_SEMANTICS.md` for the canonical reference —
state machine details, edge cases, consumer call-site matrix, and the
rationale for the two-stroke definition.

---

## 2. Fibonacci Retracement Levels

### Levels
```python
FIB_LEVELS = [30, 50, 61.8, 80]  # percentages
```

### Anchor Points
- **Anchor 1 (BOS):** the BOS anchor (`ef.bos_anchor_idx`) & its price — **LOCKED** once established
- **Anchor 2 (CTS):** idx & price of CTS — **UPDATES** as CTS moves to new extreme
- **Only draw Fib if unfilled imbalance exists between the anchor points**

### Calculation
For bullish swing (retracement from high):
```python
fib_price = anchor_high - (anchor_high - anchor_low) * (level_pct / 100)
```

For bearish swing (retracement from low):
```python
fib_price = anchor_low + (anchor_high - anchor_low) * (level_pct / 100)
```

### FibTracker Lifecycle

```
CTS_ESTABLISHED (cycle 1+)
    ↓
Check: unfilled imbalance between BOS and CTS, FORMED by the event's moment?
    ↓ (yes)
Fib ACTIVATED: BOS idx/price → CTS idx/price
    ↓
On CTS_UPDATED:
  - Anchor 2 (CTS) UPDATES to new extreme
  - Re-check imbalance condition (formed by the update's moment) → can DEACTIVATE or REACTIVATE
    ↓
CTS_CONFIRMED
    ↓
Fib LOCKED (anchor 2 stops updating)
```

**Key behaviors:**
- **For sid=0:** Cycle 0 never has its own Fib — only stored for cross-cycle check
- **For sid 1+:** See Scenario Logic below (cycle 0 may have Fib in Scenario 1)
- **Deactivation/Reactivation:** Fib can toggle active state based on imbalance conditions at each CTS update
- **Each check is asked at the handled event's MOMENT** (Plan F, 2026-09-24),
  `event_fields.event_moment(ev)` — per-event values in ARCHITECTURE
  "`ev.idx` convention" (a pattern-path CTS_UPDATED: its apply candle, since Plan E E3·0).
  FibTracker re-asks only at CTS events, so a
  gap whose c3 closes after an event counts only if a later event re-asks — none
  does for an H1 cycle ≥ 1 fib that failed at EST (activation is one-shot there),
  so such a fib is dropped, not delayed (IMBALANCE_FILL_SEMANTICS "Decided at the
  event")
- **Obsolescence:** When new cycle forms, previous cycle's Fib becomes obsolete

### Scenario Logic (Post-Reversal, sid 1+)

For structures after a reversal, Fib activation follows a 3-scenario system:

#### Scenario 1: Normal Cycle 0 Fib
**Condition:** the CTS_0 event's moment >= reversal_confirmed_idx (the EST
`confirmed_at` / the update's moment — Plan E E3a; before it the CTS_0 anchor idx;
`reversal_confirmed_idx` = the realised reversal of the previous sid,
`STATE_CHANGED(to=reversal).idx` — the same value as the zone end below — since
2026-09-27; before, the last `REVERSAL_CANDIDATE`'s scheduled apply)

**Behavior:**
- Cycle 0 gets normal Fib (if unfilled imbalance)
- Cycle 1 gets normal Fib (if unfilled imbalance)
- Skip Scenario 2/3 checks

**Revert Condition (checked at CTS_1 ESTABLISHED):**
- If BOS_1 price touches/crosses into prev structure's last BOS zone → revert to FALSE
- Deactivate cycle 0 Fib, proceed with Scenario 2/3
- "Touch" means: BOS_1 >= zone outer (for buy zone) or BOS_1 <= zone outer (for sell zone)

#### Scenario 2: Cross-Cycle Fib
**Condition:** Scenario 1 is FALSE AND all 3 conditions met:
1. cond1: Cycle 1 has unfilled imbalance
2. cond2: Cycle 0 has unfilled imbalance
3. cond3: BOS_1 doesn't fill cycle 0's imbalances

As-ofs (Plan F): FibTracker asks cond1 at the CTS_1 ESTABLISHED moment (formed
gaps only); cond2 reads the cycle-0 liveness cache, stored UNCUT (every gap in
`[BOS_0, CTS_0]` has formed by CTS_1 ESTABLISHED); cond3's window ends before any
moment. MarketStructure's in-flight resolver asks all three uncut, so the layers
agree on cond2 / cond3 and can differ on cond1 — the accepted M1 divergence
(IMBALANCE_FILL_SEMANTICS "Knowability — the c3 rule"; LANDMINES "Scenario 2
anchor agreement").

**Behavior:**
- No cycle 0 Fib
- Cycle 1 gets cross-cycle Fib: BOS_0 → CTS_1
- Normal Fib stored as fallback

#### Scenario 3: Normal Cycle 1 Fib
**Condition:** Scenario 1 is FALSE AND Scenario 2 conditions not met

**Behavior:**
- No cycle 0 Fib
- Cycle 1 gets normal Fib (if unfilled imbalance)

### Cross-Cycle Mode (`fib_mode="cross_cycle"`)

> Formerly named `m15_reverse`. Renamed when subordinate structures grew
> beyond M15 + counter direction to include confluence variants (same
> direction as parent) and potentially deeper TFs.

Subordinate-structure pipelines use a Fib model that generalizes the
cross-cycle concept to **any** cycle (not just cycle 1) AND allows cross
fibs to form **before** a cycle's CTS is established. Used by both counter
and confluence subordinate variants regardless of direction.

#### Phase state per (sid, cycle_id)
- `pre_established` — cycle's CTS not yet established. Only cross fib checks
  run (single fib requires CTS).
- `established` — CTS_n ESTABLISHED fired. Cross-first-then-single dispatch.
- `confirmed` — CTS_n CONFIRMED fired. Fib locked. Phase for cycle n+1 flips
  to pre_established.

Cycle 0 has no pre_established phase (there's no CTS_-1 to trigger it).

#### Cross-fib condition (generalized)

For target cycle n+1, walk backward from cycle n to 0, skipping cycles in
the `_dead_cycles` cache (permanently-filled cycles). For each live cycle
k, check whether its own swing `[BOS_k, CTS_k]` still has unfilled
imbalance with fill-check extended to `current_candle` (Interpretation B — the
fill horizon: the handled event's moment since Plan E E3a; CROSS_CYCLE_FIB_SPEC).

The earliest contiguous cycle `x` (where all cycles `x..n` are live) becomes
the cross fib's BOS anchor. If no prior cycle qualifies, cross fails and —
in established phase — single fib is activated as fallback.

#### Anchor by phase
- **pre_established** (cycle n+1): anchor = running extreme past CTS_n;
  own imbalance range = `[prospective_BOS_n+1, current_candle]` where
  `prospective_BOS_n+1` is the deepest pullback since CTS_n CONFIRMED
- **established** (cycle n+1): anchor = CTS_n+1; own range = `[BOS_n+1, CTS_n+1]`

The own-imbalance test counts only gaps FORMED by the event's moment
(`evaluated_at`, Plan F). In pre_established the moment IS `current_candle`, so a
gap whose c2 is `current_candle` does not count yet. Reference window: the cross
fib that counter sub 5 used to PRE-CREATE at 3806 for a cycle 1 it never
establishes (sub 5 only ever establishes cycle 0), and that fib's IC 3654 twin
POI, no longer exist — the only gap in its own-imbalance window was the instance
(3806, 3806), formed at 3807, and no later event re-checks it (§4 field list).

#### Triggers
- pre_established: `CTS_THRESHOLD_UPDATED` events for cycle n drive re-checks
  (natural trigger — fires on every new running extreme past CTS_n; its `ev.idx`
  is the processing candle = the moment)
- established: `CTS_UPDATED` events for cycle n+1

#### State transitions (cross fib)
- Extension (same earliest cycle `x`, new anchor): replace in place, same version
- Shrink (`x` moves forward because a cycle went dead): deactivate old version
  with `deactivated_by="cross_shortened"`, create new with `version+1`
- Cross fails (earliest_x == target_cycle): deactivate with
  `deactivated_by="cross_failed"`, fall back to single (established only)
- Own imbalance filled: deactivate cross with `deactivated_by="own_imb_filled"`
- Lifecycle end (the sub's window ended — `reversal` / `same_dir_replacement` /
  `parent_end`): still-active unlocked fibs get a terminal `end_idx` /
  `end_reason` = the sub's `end_reason` via `compute_cycle_lifecycle`'s
  `lifecycle_cap` / `cap_reason` (FIB_LIFECYCLE_SPEC §15.6). Historical: the
  pre-Phase-3 form was `deactivated_by="lifecycle_end"`; `"lifecycle_end"` is
  no longer emitted (Plan C, 2026-09-20)
- CTS_n+1 CONFIRMED: lock active fib (`locked=True`)

#### Storage — versioned keys
Cross fibs live in `_fibs` under key `(sid, cycle_id, "cross", version)`
alongside single fibs at `(sid, cycle_id)`. Old versions are preserved as
inactive snapshots so the chart can render them with faded styling.

### Prev BOS Line (Visualization Helper)

After each reversal, a black horizontal line shows the Scenario 1 revert threshold:
- **Start idx:** the previous structure's last BOS anchor (`ef.bos_anchor_idx`)
- **End idx:** the CTS anchor of the new sid's earliest-moment CTS_ESTABLISHED / CTS_UPDATED whose moment is
  at or after reversal_confirmed_idx (Plan E E3d; the END is a location, Q6)
- **Price:** Last BOS price of previous structure

### Unfilled vs Filled Imbalance

- **FVG gap** = distance between candle 1 wick and candle 3 wick
- The gap exists once its c3 closes (`formed_at`, §1 "Role in POI Zones")
- Check candles in **`(end_idx, check_to_idx]`** — from the last c3
- **Filled** requires TWO strokes within the scan range (see §1
  "Fill Check" and `IMBALANCE_FILL_SEMANTICS.md` for the canonical
  definition):
  - Stroke 1: price retraces ≥ 70% into the gap (low past 70% for
    bullish, high past 70% for bearish) → instance is **armed**
  - Stroke 2: at a later candle (or the same one), close passes the
    gap outer in the imbalance's direction (≥ gap_top bullish,
    ≤ gap_bottom bearish) → instance is **confirmed-filled**
- **Unfilled:** stroke 1 hasn't fired yet, OR stroke 1 fired but stroke 2
  hasn't yet — both cases keep the instance "in play". Only a FORMED instance
  can be in play: one whose first c3 has not closed by the moment of the
  question is not an imbalance yet (neither filled nor unfilled)

---

## 3. Institutional Candle (IC) Identification

IC identification is a two-step process:
1. **IC Candidates** — Candles that meet base + scenario conditions
2. **IC Variants** — From candidates, select by overlap threshold (V30/V60/V90)

No IC candidates → No IC variants → No POI zones.

### 3.1 IC Candidate Base Conditions (ALL required)

1. **Within Fib bounds (inclusive):** `BOS_idx <= candle_idx <= CTS_idx`
2. **Opposite direction of struct_direction:** `candle.direction == -struct_direction`
3. **Unfilled imbalance after:** At least 1 unfilled imbalance (matching sd) in range `(candidate_idx, CTS_idx]`,
   fill horizon `CTS_idx` (the fib's CTS). **No knowability cut** (`evaluated_at=None`, Plan F): ICs are identified
   retrospectively on the final fib (and, in MarketStructure's in-flight snapshot, read only at candles after the
   CTS); WHEN the POI can go live is the activation sweep's job, which counts only formed gaps (§4 "Lifecycle")

If Fib is deactivated, there are no bounds → no candidates.

### 3.2 IC Candidate Scenario Conditions

Additional conditions based on which Fib scenario applies:

**Scenario 1: cycle 0 normal fib, cycle 1+ normal fib**

| Cycle | Idx Constraint | Price Constraint |
|-------|----------------|------------------|
| 0 | `idx < reversal_confirmed_idx` | Entire candle below (sd=+1) or above (sd=-1) prev structure's last BOS price |
| 1+ | `idx < CTS_N_established_idx` | Entire candle below (sd=+1) or above (sd=-1) CTS_N-1 price |

**Scenario 2: no cycle 0 fib, cycle 1 cross-cycle fib, cycle 2+ normal fib**

| Cycle | Sub-condition | Idx Constraint | Price Constraint |
|-------|---------------|----------------|------------------|
| 1 (cross) | CTS_0 < reversal_idx | `idx < CTS_1_established_idx` | Entire candle below/above CTS_0 price |
| 1 (cross) | CTS_0 >= reversal_idx | `idx < reversal_confirmed_idx` | Entire candle below/above prev structure's last BOS price |
| 2+ | — | `idx < CTS_N_established_idx` | Entire candle below/above CTS_N-1 price |

**Scenario 3: no cycle 0 fib, cycle 1+ normal fib**

| Cycle | Idx Constraint | Price Constraint |
|-------|----------------|------------------|
| 1+ | `idx < CTS_N_established_idx` | Entire candle below (sd=+1) or above (sd=-1) CTS_N-1 price |

**Price Constraint Definition:**
- sd=+1 (bullish): Entire candle (HIGH) must be BELOW reference price
- sd=-1 (bearish): Entire candle (LOW) must be ABOVE reference price

### 3.3 IC Variants

From IC candidates, find the **most recent candle** meeting each overlap threshold:

| Variant | Min Overlap | Description |
|---------|-------------|-------------|
| **V30** | 30% | Most lenient |
| **V60** | 60% | Middle |
| **V90** | 90% | Most stringent |

**Overlap Calculation:** What % of candle falls within 61.8%-80% Fib zone.

```python
def calculate_candle_overlap_pct(candle_high, candle_low, fib_zone_top, fib_zone_bottom):
    candle_range = candle_high - candle_low
    if candle_range <= 0:
        return 0.0

    overlap_top = min(candle_high, fib_zone_top)
    overlap_bottom = max(candle_low, fib_zone_bottom)
    overlap = max(0, overlap_top - overlap_bottom)

    return overlap / candle_range
```

**Selection:** For each variant, scan candidates from most recent (highest idx) and pick first that meets threshold.

**Storage:** Group by unique IC candle, store qualifying versions as metadata.
- Example: IC at idx 150 with 95% overlap → `versions: ["V30", "V60", "V90"]`
- Example: IC at idx 150 (40%), IC at idx 140 (95%) → Two ICs, idx 150 has `["V30"]`, idx 140 has `["V60", "V90"]`

---

## 4. POI Zone Construction

### POIs are always sd-direction (invariant)

A POI zone is always in the **structure direction** by construction:
- Fib levels span `BOS → CTS` within a cycle, so the 61.8%–80% retrace zone
  sits entirely within that span
- IC candidates must be opposite-direction candles (`candle_dir == -sd`)
  located inside that span, with an sd-direction unfilled imbalance after them
- The resulting POI zone's `side` mirrors the structure direction:
  `"buy"` if `sd == +1`, `"sell"` if `sd == -1`

**Consequence:** there are no opp_sd POIs anywhere in the system. The zone
proximity logic relies on this invariant — opp_sd zone proximity triggers
only consider the CTS KL zone (the single opposite-direction zone per
cycle); no POIs participate.

If you ever introduce a feature that produces an opposite-direction POI,
you'll need to revisit `check_zone_proximity` and any code that currently
assumes "POI ⟹ sd direction."

### Zone Boundaries
- **Top:** IC candle high
- **Bottom:** IC candle low
- **One zone per unique IC** (not per variant)

### Zone Data Fields
- `side`: "buy" if sd=+1, "sell" if sd=-1
- `structure_id`, `struct_direction`, `cycle_id`
- `ic_idx`: Index of the IC candle (rectangle start)
- `confirmed_idx`: **LAST** activate idx (lossy scalar — `poi_zones.py` overwrites it on every activation and never clears it on deactivation). A POI can flap active/inactive within a cycle, so this scalar does NOT bound the active interval. To ask "is the POI active as of candle X?" use `zones/poi_lifecycle.py::poi_active_as_of` (walks `activation_history`), never `confirmed_idx`. See GOTCHAS "POI `confirmed_idx` Is a Lossy Scalar".
- `top`, `bottom`: IC high/low
- `versions`: List of qualifying variants ["V30", "V60", "V90"]
- `status`: "active" | "inactive" | "ended" (+ "disappeared" — a reserved terminal-invalidation status; see Zone States). Charting note (2026-09-20): a POI whose CYCLE collapsed (its BOS KL zone is `inactive` with clamped `confirmed_idx >= end_idx`) is not drawn on any chart (`charting/_zone_render.collapsed_cycles` / `is_poi_of_collapsed_cycle`); an `"inactive"` POI whose cycle is live (never met its activation conditions) is still drawn as an outline. POI fills are side-tinted since 2026-09-21 (buy gold-lime, sell amber — CHARTING_SPEC §7).
- `end_idx` / `end_reason`: terminal axis inherited from the owning cycle (`compute_cycle_lifecycle`): `"reversal"` | `"next_cycle"` (the next cycle's clamped lifecycle-start = the CTS-established **moment** `meta["confirmed_at"]`, Plan C 2026-09-20) | for subs the sub's `cap_reason` — `"reversal"` | `"same_dir_replacement"` | `"parent_end"` — when the unique sub's `end_idx` capped the cycle (`"lifecycle_end"` is no longer emitted) | `None`
- `activation_history`: per-candle activate/deactivate flips (condition axis)
- `current_versions`: the variants live at the last activate event (empty while the POI is deactivated) — vs
  `versions` = the peak variants ever achieved
- `cts_established_idx`: the owning cycle's CTS-established **moment** (`CTS_ESTABLISHED.meta["confirmed_at"]`)
  — the activation floor's cycle term (see "Activation floor" below). **Meaning changed by Plan D (2026-09-23):**
  saves before it hold the CTS anchor (then `CTS_ESTABLISHED.idx`) under this key. **A cycle with no
  `CTS_ESTABLISHED` builds NO POI (user decision 2026-09-29, zones-audit "fallback POI" option N)** — always a moment
  since then. Such a cycle's fib is the pre-established cross FibTracker pre-creates for a next cycle that never
  establishes (subordinate-only; FIB_LIFECYCLE_SPEC §6): a reversal or the sub's lifecycle cap ends it (§15.4
  candidates 1 / 3) and an ended, unlocked fib builds no POIs, so only a still-LIVE one (an open-ended sub, at the data
  edge) reached this case — and its POIs could never activate (the floor's cycle term is the establishment moment; the
  sweep's progression comes only from the cycle's own CTS events). Its POIs now appear once the cycle establishes,
  exactly as for a pre-established cross that does establish (their floor is that moment either way). A LOCKED fib
  without `CTS_ESTABLISHED` breaks the event contract (a fib locks at its cycle's `CTS_CONFIRMED`) and fails loudly.
  Before: the fallback `fib_state.cts_idx` — the fib's CTS anchor, NOT a moment (the GLOSSARY "Naming Standard"
  exception of PLAN_D §7.3) — the running extreme near the data edge, so such a POI was drawn inactive and never
  activated, then vanished when the sub was capped or reversed (a repaint). Measured before: reference window 0
  (byte-identical); suite 2 (the open-ended fixture of `tests/test_fib_never_established_cap.py` + a constructed
  locked case). Rejected: keep the inert POI with the key set to None (visible at the live edge, but it vanishes when
  the sub closes; the key's meaning widened); activate pre-established POIs early from the fib's pre-creation (a design
  change — it would move 10 window fibs that were pre-established and later established). No live case on the
  reference window since Plan F (2026-09-24): the only one was the IC 3654
  twin POI on the cross fib counter sub 5 PRE-CREATED at 3806 for a cycle 1 it never establishes (value 3806, never
  activated; sub 5 was capped, so the 2026-09-28 cap would now drop it too); Plan F no longer creates that fib (§2
  "Anchor by phase"). The sub 5 cycle-0 IC 3654 POI is a different row and unchanged.
  Rebased to entity-absolute on sub POIs by the mirror (`entity_df_mutation._ZONE_META_IDX_KEYS`). Readers: none
  that decide anything (the env-gated `POI_LIFECYCLE_DEBUG` print only).
- `bos_idx` / `cts_idx`: the owning fib's `bos_idx` / `cts_idx` (its BOS / CTS anchors) copied when the POI is built —
  the same values as the fib CSV's own columns. Rebased to entity-absolute on sub POIs by the mirror since Post-E·2
  (2026-09-26; slice-local in the M15 POI CSVs before — e.g. confluence sub 0 cycle 0 IC 678: 50 / 380, now 454 / 784).
  Readers: none that decide anything (the same debug print).
- `end_time`: When zone ends (None = extends to chart end)

### Zone States

Per the lifecycle convention (ARCHITECTURE.md "Lifecycle state convention"),
`status` is **derived** from the two orthogonal axes stored on the zone:
`end_idx`/`end_reason` (terminal, irreversible) + `activation_history`
(condition, reversible). POI currently produces `{active, inactive, ended}`
(`poi_zones.py` status derivation); `disappeared` is a **reserved** terminal-
invalidation status the chart still filters but POI does **not** currently
produce (the planned FibState lifecycle work will produce it for
`scenario1_revert` — see `zones/FIB_LIFECYCLE_SPEC.md`).

| Status | Meaning | Charting |
|--------|---------|----------|
| `active` | Condition holds at end-of-data (IC qualifies + unfilled imbalance, per the activation conditions) | Rendered fully (bright tier) |
| `inactive` | Alive but condition currently off (reversible — e.g. imbalance filled; can reactivate) | Rendered faded |
| `ended` | Terminal — cycle/structure ended (`reversal` \| `next_cycle` (next cycle's clamped start on the CTS-established moment) \| the sub's window end: `reversal` / `same_dir_replacement` / `parent_end`). Irreversible | Rendered faded |
| `disappeared` | Terminal **+ suppressed** (invalidation). Reserved; produced only by the planned FibState work (`scenario1_revert`), not by POI today | NOT rendered (kept in list for history) |

### Zone End Time (Priority Order)
1. **Reversal:** `end_time = reversal_confirmed_idx` (all zones end immediately)
2. **New CTS:** Cycle N zones end at cycle N+1's clamped lifecycle-start = `max(CTS_{N+1} ESTABLISHED.meta["confirmed_at"], struct_start[, lifecycle_floor])` — the CTS-established **moment**, not the CTS anchor (`meta["cts_anchor_idx"]`, `CTS_ESTABLISHED.idx` until Plan E E4a; Plan C, 2026-09-20)
3. **Sub window end (subs only):** the unique sub's `end_idx` (`lifecycle_cap`), tagged with the sub's `end_reason`
4. **No event:** Zone remains active (`end_time = None`)
`end_idx` is `min(...)` of the applicable candidates (earliest wins; reversal wins an equal-idx tie) — the cycle table from `zones/structure_lifecycle.compute_cycle_lifecycle`, shared with KL and fib.

### Lifecycle
- Zone activates the first time IC qualifies; **every** activate/deactivate flip is recorded in `activation_history` (`[{"idx", "active", ...}]`)
- Zone can deactivate if IC no longer qualifies on subsequent candles, then re-activate later — a cycle can flap multiple times
- The imbalance condition (an sd imbalance in `(ic_idx, t]` that has formed and is not committed-filled) is swept by
  events (`_compute_poi_activation_history`): each instance **enters** the unfilled set at
  `max(inst.formed_at, first_active)` — its first c3 — and **leaves** at its cached stroke-2 candle
  `confirmed_fill_idx` (`_compute_fill_idx_cache`). Identity: `has_unfilled_imbalance(df, ic_idx+1, t,
  check_to_idx=t, direction=sd, evaluated_at=t)`. Plan F (2026-09-24) moved the enter from the c2 (`inst.start_idx`,
  one candle before the gap existed) to `formed_at`: on the reference window H1 sid 1 cyc 2 IC 865 and IC 860 each
  re-activate at 954 and 998 (were 953 and 997), and M15 sub 7 IC 4048 re-activates at 4119 (was 4118)
- `confirmed_idx` collapses to the **LAST** activate idx (NOT the first). Per-candle activation lives in `activation_history`; query it via `zones/poi_lifecycle.py` (`poi_active_as_of`, `poi_confirmed_idx_as_of`), never the scalar. Consumers that treat `[confirmed_idx, end_idx]` as one active span are blind to earlier active stretches (this caused the sid1-cyc2 proximity-trigger loss — see GOTCHAS)

### Activation floor — cycle lifecycle-start clamp (REVISED 2026-05-26; floor terms REVISED by Plan C 2026-09-20; cycle term moved to the MOMENT by Plan D 2026-09-23)

POI activation is gated to start no earlier than the owning cycle's
lifecycle-start. The activation scan begins at
(`poi_zones._compute_poi_activation_history`):

```
first_active = max(cts_established_idx, ic_idx, lifecycle_floor_idx)
cts_established_idx = CTS_ESTABLISHED(sid, cycle).meta["confirmed_at"]   # the cycle's established MOMENT
```

where `lifecycle_floor_idx = struct_start_by_sid[sid]` =
`compute_struct_start_by_sid(events, reversal_idx_by_sid, lifecycle_floor)` —
the structure lifecycle-start (the first CTS_ESTABLISHED moment since Plan E E3f / the reversal handoff), raised to
`lifecycle_floor`: `None` for main; for a sub the **unique sub's real-time
`start_idx`** (slice-local, passed by `render_sub_projection` →
`project_to_window`), which already contains the parent-cycle floor
(`TriggerRecord.parent_floor_idx = LOH(max(struct_start[S], cts_moment[(S,C)]))`
on the CTS-established **moment**) because the record's `start_idx =
max(probe_finalize_idx, trigger_idx, parent_floor_idx)`. (Historical: plan B1,
2026-05-27, wrote the sub floor as separate "parent sid / parent_cycle_id
lifecycle-start" terms; Plan C folds them into the one `lifecycle_floor`.)
Before B1 the floor was just `max(cts_established_idx, ic_idx)`; the added
term clamps **post-reversal cycle-0 POIs** (and sub analogues) whose
`CTS_ESTABLISHED` precedes the structure's lifecycle-start (e.g. probe placed
the anchor historically).

`max(cts_established_idx, lifecycle_floor_idx)` equals the cycle's clamped lifecycle-start
(`compute_cycle_lifecycle`) whenever the cycle has a `CTS_ESTABLISHED`; with none, the cycle term falls back to
`fib_state.cts_idx` (see the field list). The lookup indexes `meta["confirmed_at"]` directly — no fallback to
`ev.idx`; `compute_cycle_lifecycle` asserts the key first for every event carrying `structure_id`/`cycle_id`.

> **Resolved (Plan D, 2026-09-23).** Until Plan D the cycle term was `CTS_ESTABLISHED.idx` — then the CTS
> **anchor** (the breakout pattern's extreme candle, retro-stamped; the idx is the moment since Plan E E4a), not the moment the cycle became knowable —
> while the canonical cycle lifecycle-start and every cycle END were already on the moment
> (`compute_cycle_lifecycle`, Plan C). On a cycle whose anchor precedes its moment a POI could go live
> `confirmed_at − idx` candles (≤ 5) before its cycle existed, overlap the previous cycle's POI on those candles,
> and activate on a cycle collapsed at its moment. On the reference window 3 CSV rows = 2 unique M15 cycles lag by 1
> (sub 0 cyc 2 1223/1224; sub 3 cyc 1 2828/2829, masked by the sub's floor). Measured at landing (PLAN_D §9; exactly the §4 prediction,
> as predicted): M15 confluence sub 0 cyc 2 IC 678 activation 1223 → 1224, plus 3 `cts_established_idx` meta cells
> re-valued to the moment; the other 22 CSVs byte-identical; chart counts unchanged. Pinned by
> `tests/test_poi_activation_moment.py`.

Any activate/deactivate flips before `first_active` are discarded; if the
POI would have been active earlier, its first active snaps to
`first_active`. If `first_active >= end_idx` the POI never activates
(`activation_history = []`, `status` never `"active"`), kept in the list as
history.

---

## 5. Charting

### Toggle
```python
cfg = {
    "zones": {"POI": True},
    "fib": {"lines": True},
    "imbalance": {"highlight": True},  # Candle color highlighting
}
```

### Visual Elements
1. **POI Zone Rectangle** — Yellow fill
   - Starts at `ic_idx` (the IC candle)
   - Ends at `end_time` (or chart end if None)
   - Horizontal lines at top/bottom bounds
2. **Confirm Line(s)** — one darker vertical line per activate event in `activation_history` (legacy fallback: a single line at `confirmed_idx` for zones predating the history)
3. **Fibonacci Lines** — Dotted lines at 0%/100% anchors, rectangle at 61.8-80%
4. **Imbalance Candle Highlighting** — Entire candle (body + wicks) colored distinctly. Marks each flagged c2
   (`is_imbalance`) — the pattern's location, not the moment its gap exists (CHARTING_SPEC §6):
   - Bullish imbalance: Lime Green `rgba(50, 205, 50, 0.8)`
   - Bearish imbalance: Amber Yellow `rgba(235, 190, 0, 0.8)`

### Zone Rendering Rules
- `active` zones: Full opacity
- `inactive` / `ended` zones: Faded opacity (3-tier by status + sid-recency)
- `disappeared` zones: NOT rendered (terminal-invalidation; filtered out — reserved, not produced by POI today)

### Hover Data
- Side, structure_id, cycle_id
- IC_idx, confirmed_idx
- Versions (V30, V60, V90)
- Top/bottom bounds

### Style Keys (style_registry.py)
```python
"zone.poi.buy"         # Buy-side POI zone fill (Yellow)
"zone.poi.sell"        # Sell-side POI zone fill (Yellow)
"zone.poi.hover_line"  # Invisible hover hitbox
"fib.line"             # Fibonacci level lines
"imbalance.bullish"    # Bullish imbalance candle rgba
"imbalance.bearish"    # Bearish imbalance candle rgba
```

---

## Data Flow

```
Fib active for cycle
    ↓
Find IC candidates (base + scenario conditions)
    ↓
Select IC variants (30%/60%/90% overlap with 61.8-80% Fib zone)
    ↓
Create POI zones (one per unique IC, bounds = IC high/low)
    ↓
Track lifecycle (active ⇄ inactive condition flips; → ended at terminal. disappeared = reserved terminal-invalidation, not produced by POI today)
    ↓
Charting renders zones + Fib lines + imbalance highlighting
```

---

## Files

| File | Purpose |
|------|---------|
| `patterns/imbalance.py` | Imbalance (FVG) pattern detection + fill checking |
| `features/fibonacci.py` | Fib level calculation |
| `zones/fib_tracker.py` | FibTracker lifecycle (activation/update/lock) |
| `zones/poi_zones.py` | POI zone derivation (3 variants) |
| `charting/export_plotly.py` | Rendering (Fib lines + zones) |
| `charting/style_registry.py` | Visual styles |

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Imbalance pattern (columns) | Done |
| Imbalance fill checking | Done |
| Imbalance candle highlighting | Done |
| Fibonacci dataclass | Done |
| FibTracker (activation/update/lock) | Done |
| Cross-cycle Fib exception | Done |
| Fib charting (0%/100% lines + 61.8-80% rect) | Done |
| IC candidate identification | Done |
| IC variant selection (V30/V60/V90) | Done |
| POI zone creation | Done |
| POI zone lifecycle management | Done |
| POI zone charting (yellow rectangles) | Done |

**Status:** Complete. Ready for Week 7 Part 2 (Volume Patterns & Indicators).
