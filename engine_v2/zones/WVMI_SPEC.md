# WVMI Spec — Week 8 Part 2

Primary file: `zones/wvmi.py`
Type definition: `common/types.py` (`WVMIRecord`)

---

## Overview

Wave Volume Momentum Indicator (WVMI) measures BOS zone strength by tracking volume momentum across wave cycles. Each record captures volume ratios between paired wave candles (first/last for breakout and pullback waves).

---

## WVMI Gating

Gating differs by entity. **Main**: entity-local zone proximity. **Sub**:
parent-event-driven (Part 4 §8.3 / §8.4).

### Main entity (`H1.main`)

WVMI records are gated by `check_zone_proximity()` in
`zones/zone_proximity.py` (alternating sd/opp_sd triggers per cycle — see
[POI_ZONES_SPEC](POI_ZONES_SPEC.md)). The orchestrator extracts only the
**first sd-direction trigger** per cycle as the gate. When that exists,
the cycle's `CTS_CONFIRMED` produces a `WVMIRecord` whose meta carries
the §8.7 attribution schema:

- `triggered_by_event_idx` — first sd trigger candle index
- `triggered_by_event_type = "ZONE_PROXIMITY_TRIGGER"`
- `structure_path_id`
- `trigger_inner`, `proximity_pips`

Cycles without an sd trigger get no main WVMI record.

#### Narrow-cycle implications (Rule 1/2/3)

Per `zones/zone_proximity.py`'s narrow-gap rules:

- **Narrow cycle (gap < `min_gap_pips`) without pullback CTS_confirmed:**
  no proximity triggers (Rule 2) → no WVMI record. Mirrors the silent
  no-WVMI behavior of cycle 0 today.
- **Narrow cycle with pullback CTS_confirmed:** triggers can fire from
  the pullback idx forward, capped at ≤1 sd + ≤1 opp_sd (Rule 3). The
  first sd trigger, if it occurs, gates a WVMI record whose
  `triggered_by_event_idx` is the post-pullback sd wick idx (NOT the
  pullback idx).
- **Mid-cycle crossing (narrow → wide):** the cap lifts at the
  crossing; alternation continues unchanged. WVMI record references
  the first sd trigger regardless of which mode it fired in.

Attribution remains consistent: `triggered_by_event_idx` always points
at the first sd-direction trigger candle, never at the pullback
confirmation candle.

### Sub entities (`H1.main >> M15.counter`, `H1.main >> M15.confluence`)

Sub WVMI is **parent-event-driven and trigger-centric** (§8.3 / §8.4 / §8.5).
`_run_downstream_pipeline` runs with `skip_wvmi=True` for subs (the
entity-local proximity gate is bypassed). After a parent cycle's sub sid chain
is built (`build_parent_cycle_chain`), the orchestrator runs a per-cycle pass
(`_assign_trigger_centric_sub_wvmi`):

1. Build the cycle's **parent trigger stream** (parent-df idxs):
   - **confluence** (`_confluence_trigger_stream`): main's first sd-prox after
     CTS **plus each var 4** (`subsequent_counter`) in the cycle — the
     sd-prox-class events.
   - **counter** (`_counter_trigger_stream`): **each var 3**
     (`subsequent_confluence`) in the cycle — the CTS-prox-class events.
2. For each trigger (time-ordered), map its parent idx to M15 and find the sub
   sid whose **active window `[start_trigger_idx, m15_end_idx]`** (lifecycle-start
   → effective end) contains it. Sweep that sid once via
   `compute_parent_driven_sub_wvmi`, stamping the trigger.

**There is no `use_case` special-casing.** A sid — bootstrap (var1/var2),
subsequent (var3/var4), or **reversal-born** — gets WVMI **iff a parent trigger
of the entity's class lands inside its active window**; a sid no trigger lands
on gets none. This is the correct reading of §8.3's "whichever sid is active —
created by var1, var3, or sub internal reversal": a reversal sid is covered when
a *later* parent trigger falls in its window, **not** by self-triggering on the
reversal (a sub reversal is neither an sd-prox nor a CTS-prox parent event). See
GOTCHAS "Sub WVMI is Trigger-Centric, Not Sid-Centric".

A sid touched by multiple triggers is swept once (continuous tracker — the sweep
already covers all the sid's cycles); the earliest (initiating) trigger wins
attribution. Re-trigger semantics (a var 4 landing on an already-swept confluence
sid) are deferred to the forthcoming WVMI lifecycle spec.

Records share main's `WVMITracker.on_cts_confirmed` / `on_bos_confirmed` /
`update_temporary_lp` lifecycle, plus §8.7 attribution merged into
`record.meta`: `triggered_by_event_idx` (**parent-df coords — never
translated**), `triggered_by_event_type`, `parent_path_id`.

---

## Computation lifecycle: created / updated / locked (the LP-finalization axis)

> **NOTE (2026-05-27):** this `created`/`updated`/`locked` axis is the
> **computation** state — it tracks LP (Last Pullback) finalization, NOT an
> active/inactive lifecycle. When the convention lifecycle below is implemented,
> this field is **renamed `status` → `lp_status`** so the convention's derived
> `status` (`active`/`ended`/`inactive`) can take the `status` name (mirrors
> fib: `status` = lifecycle, a separate field carries computation). `lp_locked`
> / `locked_by_cycle_id` are unchanged. See "Lifecycle convention" below.

### 0. Gate — see "WVMI Gating" above

Different per entity. Only gated cycles (sub: gated sids) proceed to
step 1.

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
- Sets `lp_locked=True`, `lp_status="locked"`

A record only locks when its cycle's **successor** BOS forms. A single-cycle
structure (no N+1 BOS) never locks — it stays `created`/`updated` forever with a
temp LP scanned to end-of-data. Under the lifecycle convention below that record
is correctly **`active`** (open), not a defect.

---

## Lifecycle convention (active/ended + start/end) — AGREED 2026-05-27, IMPL PENDING

> Brings `WVMIRecord` onto the lifecycle convention (ARCHITECTURE "Lifecycle state
> convention"), the LAST element off it. Design agreed 2026-05-27 (full discussion
> in `memory/project_wvmi_lifecycle_deferred.md`); mirrors the FibState scalar
> model (`FIB_LIFECYCLE_SPEC.md §15`). **Not yet implemented.**

### Tier-1: NO active/inactive axis

WVMI is created once and locked once — **no reversible condition flips** (unlike
POI/Fib, whose `active` tracks the unfilled-imbalance condition). So, like KL
zones / cycles / structures, WVMI is **tier-1**: "active" just means
"started and not ended." We do **not** store an `active` bool or an
`activation_history`; `status` is **derived** from `start_idx`/`end_idx`.

### Fields (scalar)

| Field | Meaning |
|---|---|
| `start_idx` | **creation idx = `CTS_n` CONFIRMED** (the WVMI's "own start" — when the record is born and breakout momentum locks), clamped to the structure/parent floor. Decision: creation idx, not FB/CTS-established — it's the honest birth (breakout can't be computed before LB = CTS wave candle is known) and entity-local for subs. |
| `end_idx`, `end_reason` | **inherited** from `compute_cycle_lifecycle[(sid,cycle)]` — the cycle pass-through end (next-cycle clamped start / reversal / sub cap). **Data-end is NOT a terminator** (PART4 §5): an open last / single-cycle WVMI gets `end_idx=None` → stays `active` to the edge. |
| `status` (derived) | `start_idx None`/collapsed → `inactive`; `end_idx` set & ≤ last candle → `ended`; else `active`. |
| `lp_status` (renamed from `status`) | computation axis: `created`/`updated`/`locked` (above). Orthogonal — a record can be `ended` yet not `lp_locked` (reversal-ended), or `active` with a shifting temp LP. |
| `lp_locked`, `locked_by_cycle_id` | unchanged. |

**Uniform main + sub.** Subs inherit the same cycle end via the helper (the sub's
`compute_cycle_lifecycle`, incl. the `cap_open` data-edge rule). Gating stays
trigger-centric as today. The **implementation wrinkle**: sub WVMI is computed
trigger-centrically after the sub is built (`_assign_trigger_centric_sub_wvmi` →
`compute_parent_driven_sub_wvmi`), so the `(sid,cycle)` lifecycle table must be
threaded into that path (a fib-style finalize). Main WVMI finalizes in
`_run_downstream_pipeline` next to the KL/POI/fib derivations.

### Relationship to `lp_locked`

Lock fires at `BOS_{n+1}` CONFIRMED, and in this engine
`BOS_{n+1}.confirmed_at == CTS_{n+1}.established` = cycle n's end. So for a
multi-cycle structure `lp_locked` ≈ `end_idx` (coincide). For the open/last cycle,
neither fires → `active`, temp LP shifting (correct).

### Chart rendering intent — DEFERRED to the chart-wide pass (NOT implemented now)

The lifecycle *data* must carry enough for this; the *rendering* lands later.
WVMI renders FB/LB/FP/LP as markers on the wave-candle lines, **per-component**
(not a single show/hide gate):

- **FB, LB, FP** — shown once created (start idx); permanent (locked at creation),
  independent of active/ended/locked.
- **LP** — shown iff `status=="active"` (the shifting temp) **or** `lp_locked`
  (the official LP). So **reversal-ended-before-lock → FB/LB/FP shown, no LP**;
  **cycle+1-ended → all four (LP locked)**; **open/df-end → FB/LB/FP + temp LP**.
- **inactive/collapsed** — nothing shown.

This **diverges from fib** (fib vanishes an ended-unlocked record): WVMI keeps the
breakout markers because the breakout leg is a fact locked at creation. Opacity/
tiering deferred to the same chart-wide pass.

### Implementation validation

Data-only change (charting deferred): the WVMI CSV (`debug/export_wvmi.py`) gains
`start_idx`/`end_idx`/`end_reason`/`status` + the `status`→`lp_status` rename;
the chart is byte-identical. Validate via the WVMI CSV (main + both subs),
enumerating per-`(sid,cycle)`: `start_idx` = creation, `end_idx` = cycle end
(None at the open edge), `status` derived correctly, single/open-cycle records
`active`-not-`ended`.

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
- `add_scenario3_record` / `discard_scenario3` were removed in Part 4
  Step 3c — they were dead code (production never invoked them).

---

## Pipeline Integration

WVMI runs **after POI zones** (the main-entity gate needs POI inner bounds):

```
# Main (H1.main) — entity-local gate
wave_candles → Fib tracking → POI zones → WVMI:
  0. check_zone_proximity() → first sd trigger          [gate]
  1. for CTS_CONFIRMED events (gated only) →
       on_cts_confirmed() + meta update                [create]
  2. for BOS_CONFIRMED events → on_bos_confirmed()     [lock]
  3. update_temporary_lp()                              [shift]
  → df.attrs["wvmi"] = wvmi_tracker.get_records()
```

```
# Sub (M15.counter / M15.confluence) — parent-event gate
# `_run_downstream_pipeline(..., skip_wvmi=True)` short-circuits the gate above.
# Orchestrator computes sub WVMI after sub LowerTFResult is built:
for each sub LowerTFResult:
    if parent trigger fired in this result's parent cycle:
        compute_parent_driven_sub_wvmi(result, sub_path_id, parent_trigger)
        → result.wvmi_records = [...]
```

---

## WVMIRecord Fields

| Field | Type | Description |
|-------|------|-------------|
| `bos_structure_id` | int | Structure ID of the BOS zone (entity-local) |
| `bos_cycle_id` | int | Cycle ID of the BOS zone (entity-local) |
| `zone_side` | "buy"/"sell" | BOS zone side |
| `structure_path_id` | Optional[str] | Owning entity's path (e.g., `"H1.main"`). |
| `fb_idx`, `lb_idx`, `fp_idx`, `lp_idx` | Optional[int] | Wave candle indices |
| `fb_volume`, `lb_volume`, `fp_volume`, `lp_volume` | Optional[float] | Raw volumes |
| `lb_weight`, `lp_weight` | float | Last candle weights (default 1.0) |
| `breakout_momentum`, `pullback_momentum` | Optional[float] | Computed ratios |
| `buy_momentum`, `sell_momentum` | Optional[float] | Direction-labeled wrappers |
| `status` | str | "created"/"updated"/"locked" |
| `lp_locked` | bool | Whether LP is finalized |
| `locked_by_cycle_id` | Optional[int] | BOS cycle that locked this record |
