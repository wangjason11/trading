# Wave Candles Spec — Week 8

Primary file: `zones/wave_candles.py`

---

## Overview

Wave candles are boundary candles between two consecutive waves at a KL zone. For each zone, the system identifies:
- **Last wave candle**: final candle of the ending wave
- **First wave candle**: initial candle of the starting wave

These feed into WVMI (volume momentum) downstream.

---

## Zone Type Matrix

| Zone Type | Last Wave Candle | First Wave Candle |
|-----------|-----------------|-------------------|
| **BOS** | Last Pullback (dir opposite to zone side) | First Breakout (dir same as zone side) |
| **CTS** | Last Breakout (dir opposite to zone side) | First Pullback (dir same as zone side) |

Direction mapping:
- Buy zone (sd=+1): pullback dir = -1, breakout dir = +1
- Sell zone (sd=-1): pullback dir = +1, breakout dir = -1

---

## Qualification Criteria

### Qualified Candle (`_is_qualified`)
Both must be true:
1. `candle direction == required_dir`
2. `vol_dir == required_dir OR vol_dir == 0`

**Rationale:** vol_dir == 0 (neutral volume) is allowed because volume data may be incomplete or neutral during legitimate wave transitions.

### Compound First-Wave (`_is_compound_first_wave`)
Match ONE of:
1. `is_big_normal_as0 == 1 AND candle_type in (maru, normal)` — strong directional candle with size
2. `is_big_maru_as0 == 1 AND candle_type == pinbar AND pinbar_dir == wave_dir` — large candle with pinbar rejection in wave direction

**Rationale:** First wave candles must show conviction — they're the initiating move of the new wave. Weak or indecisive candles don't qualify.

---

## BOS Wave Candles

### Last Pullback (BIB path)

Two-step search:

**Step 1 — Forward search:** `(anchor_idx+1, first_CTS_anchor)`
- Find qualified candle closest to zone outer bound
- Accept ONLY if closer to outer than the BOS candle's own close distance
- Rationale: candidate must be a better "pullback into zone" than the BOS candle itself

**Step 2 — Backward fallback:** `[pullback_start, anchor_idx]`
- `pullback_start` priority: last STATE_CHANGED to='pullback' → prior CTS anchor → lookback window
- Lookback window: **15 candles** for cycle 0, **50 candles** for cycle 1+
- Pick qualified candle closest to zone outer bound

**Why different lookbacks:** Cycle 0 has limited history (structure just started). Cycle 1+ has more price action and the prior pullback may be significantly earlier.

### Last Pullback (Non-BIB path)

Simple window: `[anchor_idx-5, anchor_idx+5]`
- Qualified candle that touches zone AND closest to outer bound

### First Breakout

Scan forward from `last_pb_idx + 1` to next CTS_CONFIRMED (or end of data).
- First qualified candle matching compound first-wave condition.

---

## CTS Wave Candles

### Last Breakout (BIB path)

Three-step event walk:

**Step 1 — Event walk:** Iterate CTS_ESTABLISHED + CTS_UPDATED events in order.
For each event:
- (a) Direct check: event candle is qualified AND wick enters zone AND closes within zone → done
- (b) Gap scan: scan between current event idx and next event idx for qualified candle closing within zone

**Step 2 — Fallback before anchor:** `[cts_anchor_idx - 10, cts_anchor_idx)`
- First qualified candle closing within zone

**Step 3 — Fallback from anchor forward:** `[cts_anchor_idx, confirmed_at]`
- First qualified candle closing within zone

**Rationale for event walk:** CTS zones can shift via CTS_UPDATED events. The event walk checks each snapshot of the zone to find the breakout candle at the right zone position.

### Last Breakout (Non-BIB path)

Same as BOS non-BIB: window `[cts_anchor_idx-5, cts_anchor_idx+5]`, qualified + touches zone + closest to outer.

### First Pullback

Scan forward from `last_bo_idx + 1` to `CTS_CONFIRMED + 10 candles` (or end of data).
- First qualified candle matching compound first-wave condition.

---

## Zone Geometry

- **outer threshold**: `zone.meta["outer"]` — the far boundary of the zone
  - Buy zone: outer = bottom (zone sits above outer)
  - Sell zone: outer = top (zone sits below outer)
- **touches zone**: Buy → `low <= zone.top`; Sell → `high >= zone.bottom`
- **closes within zone**: `zone.bottom <= close <= zone.top`
- **close distance to outer**: `abs(close - outer)` — selection metric for "closest to outer"

---

## Edge Cases

| Case | Result |
|------|--------|
| No qualified candle found for last wave | `last_wave_candle_idx = None` |
| Last wave candle found but no first wave candle matches compound condition | `first_wave_candle_idx = None` |
| BIB forward search finds candidate but it's farther from outer than BOS candle | Rejected; falls through to backward fallback |
| Zone has no events (missing CTS_ESTABLISHED) | BIB event walk yields nothing; fallback ±10 window used |
| Candle with vol_dir == 0 | Qualifies (neutral volume is acceptable) |
| Candle with direction == 0 (neutral) | Does NOT qualify for any wave candle role |

---

## Data Flow

```
KL Zones → identify_wave_candles() per zone → WaveCandleResult list
  → df.attrs["wave_candles"]
  → WVMI (volume momentum)
  → Charting (vertical lines + hover overlay)
```

---

## Output

`WaveCandleResult` (frozen dataclass):
- `structure_id`, `cycle_id`, `source_kind` (BOS/CTS), `zone_side`
- `last_wave_candle_idx`: Optional[int]
- `first_wave_candle_idx`: Optional[int]
- `meta`: dict with `base_pattern` and `anchor_idx`
