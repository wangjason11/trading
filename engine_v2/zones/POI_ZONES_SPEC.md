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

### Core Definition
Standard FVG (Fair Value Gap) — 3-candle pattern where a gap exists between candle 1's wick and candle 3's wick.

### Detection Logic
- **Bullish imbalance:** `candle_1.high < candle_3.low`
- **Bearish imbalance:** `candle_1.low > candle_3.high`
- Direction determined by candle 2's direction
- **No alterations** from standard FVG logic

### Implementation
- Stored as **columns** (not events): `is_imbalance`, `imbalance_gap_size`
- Middle candle (candle 2) gets the flag
- Computed in pipeline before base features

### Role in POI Zones
- Imbalance must exist **between the Fib anchor points** for POI zone to be created
- Imbalance must be **after the IC candle** (between IC and the break)
- Use `has_imbalance_in_range(df, start_idx, end_idx)` to check

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

### Prev BOS Line (Visualization Helper)

After each reversal, a black horizontal line shows the Scenario 1 revert threshold:
- **Start idx:** Last BOS idx of previous structure
- **End idx:** Earliest CTS event at or after reversal_confirmed_idx
- **Price:** Last BOS price of previous structure

### Unfilled vs Filled Imbalance

- **FVG gap** = distance between candle 1 wick and candle 3 wick
- Check candles from **imbalance_idx+1 to check_to_idx**
- **Unfilled:** Price retraced **<70%** of FVG gap
- **Filled:** Price retraced **≥70%** of FVG gap

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

### Zone Boundaries
- **Top:** IC candle high
- **Bottom:** IC candle low
- **One zone per unique IC** (not per variant)

### Zone Data Fields
- `side`: "buy" if sd=+1, "sell" if sd=-1
- `structure_id`, `struct_direction`, `cycle_id`
- `ic_idx`: Index of the IC candle (rectangle start)
- `confirmed_idx`: First activation idx (always > ic_idx, vertical line here)
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
- Zone activates first time IC is found (`confirmed_idx` recorded)
- Zone can disappear if IC no longer qualifies on subsequent candles
- Zone re-appears if IC qualifies again (but `confirmed_idx` stays as first activation)

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
2. **Confirm Line** — Darker vertical line at `confirmed_idx`
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
