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
- From **BOS to CTS** (or vice versa)
- Can be same cycle OR different cycles (but **same structure_id**)
- **Only draw Fib if imbalance exists between the anchor points**

### Calculation
For bullish swing (retracement from high):
```python
fib_price = anchor_high - (anchor_high - anchor_low) * (level_pct / 100)
```

For bearish swing (retracement from low):
```python
fib_price = anchor_low + (anchor_high - anchor_low) * (level_pct / 100)
```

---

## 3. Institutional Candle (IC) Identification

### Common Rules (All 3 Variants)
1. IC candle must be **prior to** price breaking BOS or CTS
2. There must be **imbalance after** the IC candle (between IC and the break)
3. IC candle must overlap with **61.8% - 80%** Fib zone

### 3 Variants

| Variant | IC Overlap Requirement | Description |
|---------|------------------------|-------------|
| **POI-v1** | At least **40%** of full candle in Fib zone | Most permissive |
| **POI-v2** | At least **70%** of full candle in Fib zone | Moderate |
| **POI-v3** | **100%** of full candle in Fib zone | Strictest |

### Overlap Calculation
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

### IC Selection Logic
- Search **backward** from the most recent:
  - CTS_ESTABLISHED, or
  - BOS_CONFIRMED, or
  - bos_threshold/cts_threshold update
- Find the **most recent candle** meeting the variant's overlap requirement

---

## 4. POI Zone Construction

### Zone Boundaries
- **Top:** IC high
- **Bottom:** IC low

### Lifecycle

```
CTS_ESTABLISHED
    ↓
Create pending POI zones (all 3 variants)
    ↓
Re-evaluate on each candle (IC may change)
    ↓
CTS_CONFIRMED
    ↓
Finalize zones (stop updating)
```

| Stage | Description |
|-------|-------------|
| **Pending** | Zone created after CTS_ESTABLISHED, can still update |
| **Finalized** | Zone stops updating after CTS_CONFIRMED |
| **Active** | Zone ready for trade consideration |
| **Mitigated** | Price touches zone but doesn't break through |
| **Broken** | Price closes beyond zone boundary |

### Activation Condition
Zone activates when:
- BOS or CTS confirmed, AND
- Imbalance exists between anchor points

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
1. **POI Zone Rectangles** — All 3 variants shown simultaneously with same formatting
2. **Fibonacci Lines** — Horizontal lines at 30, 50, 61.8, 80 levels
3. **Imbalance Candle Highlighting** — Entire candle (body + wicks) colored distinctly:
   - Bullish imbalance: Lime Green `rgba(50, 205, 50, 0.8)`
   - Bearish imbalance: Gold `rgba(255, 215, 0, 0.8)`

### Style Keys (style_registry.py)
```python
"zone.poi.buy"         # Buy-side POI zone fill
"zone.poi.sell"        # Sell-side POI zone fill
"zone.poi.hover_line"  # Invisible hover hitbox
"fib.line"             # Fibonacci level lines
"imbalance.bullish"    # Bullish imbalance candle rgba
"imbalance.bearish"    # Bearish imbalance candle rgba
```

---

## Data Flow

```
BOS/CTS events (with same sid)
    ↓
Check: imbalance exists between anchors?
    ↓ (yes)
Calculate Fib levels (30, 50, 61.8, 80)
    ↓
CTS_ESTABLISHED triggers IC search
    ↓
Search backward for IC in 61.8-80% zone
    ↓
Create POI Zones (v1, v2, v3 based on overlap %)
    ↓
Re-evaluate on each candle until CTS_CONFIRMED
    ↓
Charting renders zones + Fib lines + imbalance highlighting
```

---

## Files

| File | Purpose |
|------|---------|
| `patterns/imbalance.py` | Imbalance (FVG) pattern detection |
| `features/fibonacci.py` | Fib level calculation |
| `zones/poi_zones.py` | POI zone derivation (3 variants) |
| `charting/export_plotly.py` | Rendering |
| `charting/style_registry.py` | Visual styles |

---

## Implementation Status

| Component | Status |
|-----------|--------|
| Imbalance pattern (columns) | Done |
| Imbalance candle highlighting | Done |
| Fibonacci dataclass | Done |
| POI zones skeleton | Done |
| 3-variant IC overlap logic | Pending |
| Event-driven lifecycle | Pending |
| IC backward search | Pending |

**Next:** Implement 3-variant IC identification with proper overlap calculation.
