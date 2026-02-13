# WVMI Spec — Week 8 Part 2

Primary file: `zones/wvmi.py`
Type definition: `common/types.py` (`WVMIRecord`)

---

## Overview

Wave Volume Momentum Indicator (WVMI) measures BOS zone strength by tracking volume momentum across wave cycles. Each record captures volume ratios between paired wave candles (first/last for breakout and pullback waves).

---

## Lifecycle (mirrors FibTracker)

### 1. Created — CTS_n Confirmed

Triggered by `on_cts_confirmed()`. Derives 4 wave candle indices from BOS_n and CTS_n:

| Role | Source | Wave Candle Field |
|------|--------|-------------------|
| FB (First Breakout) | BOS_n wave candle | `first_wave_candle_idx` |
| LB (Last Breakout) | CTS_n wave candle | `last_wave_candle_idx` |
| FP (First Pullback) | CTS_n wave candle | `first_wave_candle_idx` |
| LP (Last Pullback) | Temporary — qualified candle closest to outer | Shifts until locked |

**Breakout momentum is LOCKED at creation:**
```
breakout_momentum = (LB_vol * LB_weight) / FB_vol
```

**Pullback momentum starts SHIFTING:**
```
pullback_momentum = (LP_vol * LP_weight) / FP_vol   # recomputed each candle
```

### 2. Updated — Each Candle

`update_temporary_lp()` re-scans all non-locked records:
- Finds qualified candle (FP direction + vol_dir match) between FP and end of data
- Picks candle whose close is closest to BOS zone outer bound
- Updates LP idx, volume, weight, and pullback_momentum

### 3. Locked — BOS_n+1 Confirmed

`on_bos_confirmed()` with cycle_id=N+1 locks WVMI for cycle_id=N:
- Replaces temp LP with official LP from BOS_n+1's `last_wave_candle_idx`
- Finalizes pullback_momentum
- Sets `lp_locked=True`, `status="locked"`

---

## Weight Computation (`_compute_last_wave_weight`)

Applied to **last candles only** (LB and LP). First candles (FB, FP) always use weight 1.0.

| Condition | Weight | Rationale |
|-----------|--------|-----------|
| `is_big_normal_as0 AND ctype in (maru, normal)` | **1.0** | Strong directional candle with significant size |
| `is_big_maru_as0 AND pinbar AND pinbar_dir != wave_dir` | **1.0** | Rejection pinbar with large body — strong signal |
| `round(body_pct * 100) <= 10` | **0.5** | Doji-like / indecisive — weak signal |
| Everything else | **0.7** | Default — moderate confidence |

---

## Direction Labels

`buy_momentum` and `sell_momentum` map breakout/pullback to trading direction:

| Zone Side | buy_momentum | sell_momentum |
|-----------|-------------|---------------|
| Buy | breakout_momentum | pullback_momentum |
| Sell | pullback_momentum | breakout_momentum |

**Rationale:** For a buy zone, breakout momentum reflects buying strength. For a sell zone, the pullback wave is in the buy direction, so it maps to buy_momentum.

---

## Formulas

```
breakout_momentum = (LB_volume * LB_weight) / FB_volume
pullback_momentum = (LP_volume * LP_weight) / FP_volume
```

- Momentum > 1.0: last wave candle has more volume than first → increasing conviction
- Momentum < 1.0: volume fading → weakening wave
- Momentum = N/A: missing candle or zero denominator

---

## Temporary LP Selection (`_find_temporary_lp`)

1. Search range: `[FP_idx + 1, end_of_data]`
2. Qualification: same direction AND vol_dir as FP candle (or vol_dir == 0)
3. Selection: candle whose `close` is closest to BOS zone outer bound

**Outer bound:**
- Buy zone: outer = bottom (zone sits above)
- Sell zone: outer = top (zone sits below)

"Closest to outer" means the pullback has retraced most deeply toward the zone — a stronger pullback signal.

---

## Guard Rails

| Condition | Behavior |
|-----------|----------|
| BOS zone not found | Returns None (no record created) |
| BOS or CTS wave candle missing required indices | Returns None |
| FB_volume == 0 or FP_volume == 0 | Returns None (division by zero) |
| Temp LP not found | Record created with `lp_idx=None`, `pullback_momentum=None` |
| Momentum is None | Charting shows "N/A" in hover |

---

## Scenario 3 Integration

- `add_scenario3_record()`: stores WVMI from Scenario 3 probe with `source="scenario3"`
- `discard_scenario3()`: removes record when probe is discarded
- Key: `(sid, cycle_id, "scenario3")` — separate from main WVMI `(sid, cycle_id, "main")`
- On `on_bos_confirmed()`, both "main" and "scenario3" records are checked for locking

---

## Pipeline Integration

```
wave_candles → WVMITracker:
  1. for CTS_CONFIRMED events → on_cts_confirmed()    [create]
  2. for BOS_CONFIRMED events → on_bos_confirmed()    [lock]
  3. update_temporary_lp()                             [shift]
  → df.attrs["wvmi"] = wvmi_tracker.get_records()
```

---

## WVMIRecord Fields

| Field | Type | Description |
|-------|------|-------------|
| `bos_structure_id` | int | Structure ID of the BOS zone |
| `bos_cycle_id` | int | Cycle ID of the BOS zone |
| `zone_side` | "buy"/"sell" | BOS zone side |
| `source` | "main"/"scenario3" | Origin of the record |
| `fb_idx`, `lb_idx`, `fp_idx`, `lp_idx` | Optional[int] | Wave candle indices |
| `fb_volume`, `lb_volume`, `fp_volume`, `lp_volume` | Optional[float] | Raw volumes |
| `lb_weight`, `lp_weight` | float | Last candle weights (default 1.0) |
| `breakout_momentum`, `pullback_momentum` | Optional[float] | Computed ratios |
| `buy_momentum`, `sell_momentum` | Optional[float] | Direction-labeled wrappers |
| `status` | str | "created"/"updated"/"locked" |
| `lp_locked` | bool | Whether LP is finalized |
| `locked_by_cycle_id` | Optional[int] | BOS cycle that locked this record |
