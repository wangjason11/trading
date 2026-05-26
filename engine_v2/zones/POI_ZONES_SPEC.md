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
- Imbalance instance must exist **overlapping the Fib anchor points** for POI
  zone creation (via `has_unfilled_imbalance`)
- Imbalance must be **after the IC candle** (between IC and the break) —
  POI IC validation passes `direction=sd` so only structure-direction
  imbalances qualify

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

The other axis the primitive exposes — `check_to_idx` — is the more
load-bearing distinction across call sites: fib lifecycle and scenario
checks pass varying "as-of" idx (the fib's current `cts_idx`, a fixed
reference event idx, or the current candle in live evaluation), while POI
IC validation always passes `check_to_idx = end_idx = cts_idx` because the
question is asked once at the cycle's current state.

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
Scan starts at `end_idx + 1` — the instance's own candles (including the
last c3 that defines the gap) are excluded.

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
- **Anchor 1 (BOS):** idx & price of confirmed BOS — **LOCKED** once established
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
Check: unfilled imbalance between BOS and CTS?
    ↓ (yes)
Fib ACTIVATED: BOS idx/price → CTS idx/price
    ↓
On CTS_UPDATED:
  - Anchor 2 (CTS) UPDATES to new extreme
  - Re-check imbalance condition → can DEACTIVATE or REACTIVATE
    ↓
CTS_CONFIRMED
    ↓
Fib LOCKED (anchor 2 stops updating)
```

**Key behaviors:**
- **For sid=0:** Cycle 0 never has its own Fib — only stored for cross-cycle check
- **For sid 1+:** See Scenario Logic below (cycle 0 may have Fib in Scenario 1)
- **Deactivation/Reactivation:** Fib can toggle active state based on imbalance conditions at each CTS update
- **Obsolescence:** When new cycle forms, previous cycle's Fib becomes obsolete

### Scenario Logic (Post-Reversal, sid 1+)

For structures after a reversal, Fib activation follows a 3-scenario system:

#### Scenario 1: Normal Cycle 0 Fib
**Condition:** CTS_0 idx >= reversal_confirmed_idx

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
imbalance with fill-check extended to the current candle (Interpretation B).

The earliest contiguous cycle `x` (where all cycles `x..n` are live) becomes
the cross fib's BOS anchor. If no prior cycle qualifies, cross fails and —
in established phase — single fib is activated as fallback.

#### Anchor by phase
- **pre_established** (cycle n+1): anchor = running extreme past CTS_n;
  own imbalance range = `[prospective_BOS_n+1, current_candle]` where
  `prospective_BOS_n+1` is the deepest pullback since CTS_n CONFIRMED
- **established** (cycle n+1): anchor = CTS_n+1; own range = `[BOS_n+1, CTS_n+1]`

#### Triggers
- pre_established: `CTS_THRESHOLD_UPDATED` events for cycle n drive re-checks
  (natural trigger — fires on every new running extreme past CTS_n)
- established: `CTS_UPDATED` events for cycle n+1

#### State transitions (cross fib)
- Extension (same earliest cycle `x`, new anchor): replace in place, same version
- Shrink (`x` moves forward because a cycle went dead): deactivate old version
  with `deactivated_by="cross_shortened"`, create new with `version+1`
- Cross fails (earliest_x == target_cycle): deactivate with
  `deactivated_by="cross_failed"`, fall back to single (established only)
- Own imbalance filled: deactivate cross with `deactivated_by="own_imb_filled"`
- Lifecycle end (parent H1 cycle ended): still-active unlocked fibs get
  `deactivated_by="lifecycle_end"`
- CTS_n+1 CONFIRMED: lock active fib (`locked=True`)

#### Storage — versioned keys
Cross fibs live in `_fibs` under key `(sid, cycle_id, "cross", version)`
alongside single fibs at `(sid, cycle_id)`. Old versions are preserved as
inactive snapshots so the chart can render them with faded styling.

### Prev BOS Line (Visualization Helper)

After each reversal, a black horizontal line shows the Scenario 1 revert threshold:
- **Start idx:** Last BOS idx of previous structure
- **End idx:** Earliest CTS event at or after reversal_confirmed_idx
- **Price:** Last BOS price of previous structure

### Unfilled vs Filled Imbalance

- **FVG gap** = distance between candle 1 wick and candle 3 wick
- Check candles from **imbalance_idx+1 to check_to_idx**
- **Filled** requires TWO strokes within the scan range (see §1
  "Fill Check" and `IMBALANCE_FILL_SEMANTICS.md` for the canonical
  definition):
  - Stroke 1: price retraces ≥ 70% into the gap (low past 70% for
    bullish, high past 70% for bearish) → instance is **armed**
  - Stroke 2: at a later candle (or the same one), close passes the
    gap outer in the imbalance's direction (≥ gap_top bullish,
    ≤ gap_bottom bearish) → instance is **confirmed-filled**
- **Unfilled:** stroke 1 hasn't fired yet, OR stroke 1 fired but stroke 2
  hasn't yet — both cases keep the instance "in play"

---

## 3. Institutional Candle (IC) Identification

IC identification is a two-step process:
1. **IC Candidates** — Candles that meet base + scenario conditions
2. **IC Variants** — From candidates, select by overlap threshold (V30/V60/V90)

No IC candidates → No IC variants → No POI zones.

### 3.1 IC Candidate Base Conditions (ALL required)

1. **Within Fib bounds (inclusive):** `BOS_idx <= candle_idx <= CTS_idx`
2. **Opposite direction of struct_direction:** `candle.direction == -struct_direction`
3. **Unfilled imbalance after:** At least 1 unfilled imbalance (matching sd) in range `(candidate_idx, CTS_idx]`

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
- `status`: "active" | "inactive" | "disappeared"
- `end_time`: When zone ends (None = extends to chart end)

### Zone States

| Status | Meaning | Charting |
|--------|---------|----------|
| `active` | Zone currently valid | Rendered fully |
| `inactive` | Valid but superseded by newer zone | Rendered faded |
| `disappeared` | IC no longer qualifies | NOT rendered (kept in list for history) |

### Zone End Time (Priority Order)
1. **Reversal:** `end_time = reversal_confirmed_idx` (all zones end immediately)
2. **New CTS:** Cycle N zones end when CTS_N+1 established
3. **No event:** Zone remains active (`end_time = None`)

### Lifecycle
- Zone activates the first time IC qualifies; **every** activate/deactivate flip is recorded in `activation_history` (`[{"idx", "active", ...}]`)
- Zone can deactivate if IC no longer qualifies on subsequent candles, then re-activate later — a cycle can flap multiple times
- `confirmed_idx` collapses to the **LAST** activate idx (NOT the first). Per-candle activation lives in `activation_history`; query it via `zones/poi_lifecycle.py` (`poi_active_as_of`, `poi_confirmed_idx_as_of`), never the scalar. Consumers that treat `[confirmed_idx, end_idx]` as one active span are blind to earlier active stretches (this caused the sid1-cyc2 proximity-trigger loss — see GOTCHAS)

### Activation floor — cycle lifecycle-start clamp (REVISED 2026-05-26)

POI activation is gated to start no earlier than the owning cycle's
lifecycle-start. The activation scan begins at:

```
first_active = max(cts_established_idx, ic_idx, cycle_lifecycle_start)
```

where `cycle_lifecycle_start = max(CTS_n ESTABLISHED idx, structure
lifecycle-start[, parent sid lifecycle-start])` per the unified model
(`PART4_REFACTOR_SPEC.md §5`). Before this change the floor was just
`max(cts_established_idx, ic_idx)`; the added term clamps **post-reversal
cycle-0 POIs** (and sub analogues) whose `CTS_ESTABLISHED` precedes the
structure's lifecycle-start (e.g. probe placed the anchor historically).
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
4. **Imbalance Candle Highlighting** — Entire candle (body + wicks) colored distinctly:
   - Bullish imbalance: Lime Green `rgba(50, 205, 50, 0.8)`
   - Bearish imbalance: Amber Yellow `rgba(235, 190, 0, 0.8)`

### Zone Rendering Rules
- `active` zones: Full opacity
- `inactive` zones: Faded opacity
- `disappeared` zones: NOT rendered

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
Track lifecycle (active → inactive → disappeared)
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
