# WVMI Spec — Week 8 Part 2

Primary file: `zones/wvmi.py`
Type definition: `common/types.py` (`WVMIRecord`)

---

## Overview

Wave Volume Momentum Indicator (WVMI) measures BOS zone strength by tracking volume momentum across wave cycles. Each record captures volume ratios between paired wave candles (first/last for breakout and pullback waves).

---

## Price Proximity Activation

WVMI records are only created for cycles where price actually approaches the zone. This prevents eagerly computing momentum for zones that price never retraces to.

### Scan Window

For each `(sid, cycle_id)` with a `CTS_CONFIRMED` event:

- **Start:** `CTS_CONFIRMED confirmed_at + 1` — avoids trivial activation during the pullback that formed the CTS (uses `ev.meta["confirmed_at"]`, not `ev.idx`)
- **End:** earliest of:
  - `BOS_CONFIRMED confirmed_at - 1` for `(sid, cycle_id + 1)` — next cycle's BOS confirmation deactivates current zones (uses `ev.meta["confirmed_at"]`, not `ev.idx` which is the BOS extreme)
  - `REVERSAL_CANDIDATE apply_idx - 1` for sid — structure ends
  - End of data

If the scan window is empty (start > end), the cycle is skipped.

### Trigger Zone Selection

At each candle in the scan window, the trigger level is built from **active** zones only:

1. BOS KL zone `inner` bound (active throughout the scan window)
2. POI zones for `(sid, cycle_id)` where `confirmed_idx <= candle_idx` and (`end_idx` is None or `end_idx >= candle_idx`)
3. Inner bounds: `top` for buy zones, `bottom` for sell zones
4. **Trigger inner:** highest inner for buy zones (closest to CTS above), lowest inner for sell zones (closest to CTS below)

### Proximity Check

- **Buy zone:** activated when `candle_low <= trigger_inner + 20 * pip_size`
- **Sell zone:** activated when `candle_high >= trigger_inner - 20 * pip_size`
- Activation stops at the first qualifying candle

### Edge Cases

| Case | Behavior |
|------|----------|
| No active POI zones at candle | Only BOS KL zone inner bound used as trigger |
| No BOS KL zone | No activation possible → no WVMI |
| BOS zone has no `inner` AND no active POI zones | Candle skipped; later candles may have active POI zones |
| CTS_CONFIRMED but empty scan window | Skipped (next BOS at same idx or earlier) |
| CTS_CONFIRMED but price never approaches | Not activated → no WVMI |

### Activation Metadata

When activated, the `WVMIRecord.meta` dict is populated with:
- `activation_idx` — first candle that triggered activation
- `trigger_inner` — the inner price used as trigger level
- `proximity_pips` — configured pip threshold (default 20)

---

## Lifecycle (mirrors FibTracker)

### 0. Activated — Price Approaches Zone

`check_proximity_activation()` determines which `(sid, cycle_id)` pairs are eligible for WVMI creation. Only activated cycles proceed to step 1.

### 1. Created — CTS_n Confirmed

Triggered by `on_cts_confirmed()` **only if the cycle was activated**. Derives 4 wave candle indices from BOS_n and CTS_n:

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

**Rounding:** `buy_momentum` and `sell_momentum` are rounded to 2 decimal places for display/logging.

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

WVMI runs **after POI zones** (needs POI zone inner bounds for proximity gate):

```
wave_candles → Fib tracking → POI zones → WVMI:
  0. check_proximity_activation()                      [gate]
  1. for CTS_CONFIRMED events (activated only) →
       on_cts_confirmed() + meta update                [create]
  2. for BOS_CONFIRMED events → on_bos_confirmed()     [lock]
  3. update_temporary_lp()                              [shift]
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
