# WVMI Spec — Week 8 Part 2

Primary file: `zones/wvmi.py`
Type definition: `common/types.py` (`WVMIRecord`)

---

## Overview

Wave Volume Momentum Indicator (WVMI) measures BOS zone strength by tracking volume momentum across wave cycles. Each record captures volume ratios between paired wave candles (first/last for breakout and pullback waves).

---

## Zone Proximity Trigger (gate for WVMI creation today)

WVMI records are only created for cycles where price actually approaches a
zone. The proximity-trigger gate is now produced by the more general
`check_zone_proximity()` in `zones/zone_proximity.py` — see
[POI_ZONES_SPEC](POI_ZONES_SPEC.md) and `zone_proximity.py` for the full
spec including alternating sd/opp_sd triggers per cycle.

For the WVMI gate specifically, the orchestrator extracts only the
**first sd-direction trigger** per cycle (preserving the pre-refactor
behavior). When that exists, the corresponding `WVMIRecord` is created
and its meta is populated with:

- `proximity_trigger_idx` — first sd trigger candle index
- `trigger_inner` — the inner price used as the trigger level
- `proximity_pips` — threshold in pips (default 20 for H1, 10 for M15,
  5 for M5; configurable per call)

Cycles without an sd trigger get no WVMI record (current backward-compat
behavior). This will change when WVMI is rewired off the gate in a
later refactor.

---

## Lifecycle (mirrors FibTracker)

### 0. Gate — Price Approaches Zone

`check_zone_proximity()` (in `zones/zone_proximity.py`) determines which
`(sid, cycle_id)` pairs have an sd-direction proximity trigger. Only those
cycles proceed to step 1.

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

## Entity Attribution (Part 4 §8.7)

- `WVMITracker.__init__(structure_path_id=...)` stamps every record with the
  owning entity's path (e.g., `"H1.main"`, `"H1.main >> M15.confluence"`).
- `WVMIRecord._records` is keyed by `(sid, cycle_id)` — one tracker per
  entity, so source-based disambiguation isn't needed.
- `WVMIRecord.source` (`Literal["main", "scenario3"]`) is retained as a
  back-compat field; always `"main"` today and slated for removal in
  Part 4 migration plan §13.5.
- `add_scenario3_record` / `discard_scenario3` were removed in Part 4
  Step 3c — they were dead code (production never invoked them).

---

## Pipeline Integration

WVMI runs **after POI zones** (needs POI zone inner bounds for proximity gate):

```
wave_candles → Fib tracking → POI zones → WVMI:
  0. check_zone_proximity() → first sd trigger          [gate]
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
| `bos_structure_id` | int | Structure ID of the BOS zone (entity-local) |
| `bos_cycle_id` | int | Cycle ID of the BOS zone (entity-local) |
| `zone_side` | "buy"/"sell" | BOS zone side |
| `source` | "main"/"scenario3" | Deprecated — always "main" since Part 4 Step 3c. Removed in §13.5. |
| `structure_path_id` | Optional[str] | Owning entity's path (e.g., `"H1.main"`). New in Part 4 Step 3c. |
| `fb_idx`, `lb_idx`, `fp_idx`, `lp_idx` | Optional[int] | Wave candle indices |
| `fb_volume`, `lb_volume`, `fp_volume`, `lp_volume` | Optional[float] | Raw volumes |
| `lb_weight`, `lp_weight` | float | Last candle weights (default 1.0) |
| `breakout_momentum`, `pullback_momentum` | Optional[float] | Computed ratios |
| `buy_momentum`, `sell_momentum` | Optional[float] | Direction-labeled wrappers |
| `status` | str | "created"/"updated"/"locked" |
| `lp_locked` | bool | Whether LP is finalized |
| `locked_by_cycle_id` | Optional[int] | BOS cycle that locked this record |
