# WVMI Spec — Week 8 Part 2

Primary file: `zones/wvmi.py`
Type definition: `common/types.py` (`WVMIRecord`)

---

## Overview

Wave Volume Momentum Indicator (WVMI) measures BOS zone strength by tracking volume momentum across wave cycles. Each record captures volume ratios between paired wave candles (first/last for breakout and pullback waves).

---

## WVMI Gating

Gating differs by entity. **Main**: entity-local zone proximity (the first sd
trigger per cycle). **Sub**: none — every CTS_CONFIRMED of a rendered unique sub
gets a record, computed inside the sub's projection like its zones (Plan G,
2026-09-30); the parent triggers are per-lens ATTRIBUTION, never a gate.
`_run_downstream_pipeline(wvmi=...)` selects it: `"first_sd_prox"` (the main),
`"none"` (a sub, `project_to_window`), `"off"` (tests); any other value raises.

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

> **Plan G (landed 2026-09-30, `plans/PLAN_G_wvmi_unique_sub.md`) — WVMI on the unique sub, like zones.**
> User direction (2026-09-28): "once we have the pooled sub MS completed, we would actually move WVMI directly to
> unique subs as opposed to existing on lens / sub records level. Conceptually, this is the same as zones. Since
> unique subs are where trading decisions will be made and governed by life cycle, it would make sense WVMI lives
> here as well." It replaces the Plan C §17.10 minimal implementation (one trigger-GATED sweep per unique sub,
> persisted into every lens with the sweeping lens's path; kept below as dated history).

1. **Computed inside the projection.** `project_to_window` (ONE call per unique sub, PART4 §17.9) runs
   `_run_downstream_pipeline(..., wvmi="none")` over the knowable-at-clipped events with the sub's
   `lifecycle_floor` / `lifecycle_cap` / `cap_reason` — the same inputs as its KL / POI / fib. Every CTS_CONFIRMED
   is offered to the tracker (no gate; zone proximity is never run for a sub). The ONE tracker helper
   (`orchestrator._compute_wvmi_records`, shared with the main) uses:
   - the frame **`df.iloc[:cap + 1]`** when a cap is set (the natural-end frame of the pool's geometry would let
     FP (<= CTS_CONFIRMED moment + 10), LB (<= CTS anchor + 5) or a lock LP (<= BOS_{n+1} anchor + 5) read candles
     past the sub's end — a look-ahead; 0 on the reference window);
   - the `compute_cycle_lifecycle` table for the temp-LP bound (**`end − 1`**, like the main — "Temporary LP
     Selection") and for **`cycle_collapsed`** (`start >= end`, see "WVMIRecord Fields"); a capped record without a
     table end fails an assert (unreachable: a kept CTS_CONFIRMED implies its CTS_ESTABLISHED);
   - a lock LP outside the frame falls back to the temp LP (Plan G Q10, "Locked" below).
   The projection's records (`LowerTFResult.wvmi_records`) are slice-local, carry the projection's
   `structure_path_id` (the first live record's lens) and meta `{}`.
2. **Persisted like zones** — the mirror (`mirror_lower_tf_result_to_entity_df` step 8) deep-copies each record
   into EVERY lens df the sub is on, shifts FB / LB / FP / LP to entity-absolute, stamps the §17.9 attribution
   (`sub_id` + informational `parent_sid` / `parent_cycle_id` / `use_case` / `started_by`) and sets the copy's
   FIELD `structure_path_id` to THAT lens's path (= the meta's — before Plan G the field kept the sweeping lens's
   path, so a dual-lens sub's counter rows said `M15.confluence` in the column and `M15.counter` in the meta). A
   dual-lens sub's records are on both lens `_wvmi.csv` files, like its zones.
3. **Trigger metadata per lens** (`orchestrator._stamp_sub_wvmi_trigger_meta`, after every projection is mirrored).
   Each lens reads its §8.5 WVMI-class parent trigger stream over ALL parent cycles
   (`_wvmi_trigger_streams_by_lens` — the one place the lens -> stream mapping is made):
   - **confluence**: the sd-prox class — the main's first sd-prox per cycle (`ZONE_PROXIMITY_TRIGGER`) + each var 4
     (`SUBSEQUENT_COUNTER_TRIGGER`);
   - **counter**: the CTS-prox class — each var 3 (`SUBSEQUENT_CONFLUENCE_TRIGGER`).
   Each entry is LOH-mapped once (`_map_parent_idx_to_m15_hour_end`), sorted by `(m15_idx, parent_idx)`; per sub
   on the lens, the FIRST entry inside the sub's real-time window **`[meta["start_idx"], meta["m15_end_idx"]]`**
   (`m15_end_idx` = the sub's end, or the geometry's data edge for an open sub) is the lens's trigger. Every
   record of that lens df, joined to its sub on `meta["sub_id"]`, gets as its FIRST meta keys
   `triggered_by_event_idx` (**parent-df coords — never translated**) / `triggered_by_event_type` — both None when
   no entry lands in the window — and `parent_path_id` = `"H1.main"` (always: the parent entity). On a sub row
   `triggered_by_*` means **"the lens's first WVMI-class trigger inside the sub's window"** — attribution, stamped
   retroactively (a declared meaning change, LANDMINES "Event Contract Rules" rule 3; the main's rows keep the
   gate meaning).
4. **run.log**: each projection prints `[wvmi] total=N, locked=M` in its own block; the summary
   `[multi_tf:dual] sub wvmi acted=… records=… by_started_by=… by_lens=…` counts per UNIQUE sub from the
   projections' own records (a dual-lens sub once; `by_lens` adds its count to each of its lenses).

Reference window at Plan G (vs the Post-E·5 save): 18 sub records on 8 subs (before: 8 on the 5 subs a trigger
reached — subs 0–2 had none); the confluence `_wvmi.csv` 7 -> 17 rows, the counter one 6 rows with sub 3's trigger
H1 871 `SUBSEQUENT_CONFLUENCE_TRIGGER` (was the confluence lens's 710) and sub 7's None; 4 collapsed-cycle records
(sub 0 c0, sub 1 c0 + c1, sub 3 c0); record values unchanged (the `end − 1` bound changed 0 values here).

**Dated history — Plan C (2026-09-20 → 2026-09-30), superseded by Plan G.** One trigger-gated sweep per unique sub
(`_assign_sub_wvmi_per_sub`): the union of the two lens streams restricted to the sub's lenses; the FIRST entry in
the window swept the sub once (`multitf/sub_wvmi.compute_parent_driven_sub_wvmi`, deleted), its lens deciding the
records' `structure_path_id`; no entry -> no WVMI; the records persisted into every lens df
(`persist_facade_wvmi_to_entity_df`, deleted). The pre-pool text (per-parent-cycle pass
`_assign_trigger_centric_sub_wvmi`, window `[start_trigger_idx, m15_end_idx]`) was superseded by Plan C.

**Measured consequence on the reference window (first Plan C replay,
2026-09-20):** a sub on both lenses now carries its WVMI rows on both lens
CSVs — sub `2639/−1` (`sub_id` 3; confluence record reversal-born from
`2365/+1`, counter record from `first_counter`) has rows on the counter
`_wvmi.csv` as well as the confluence one; sub `4027/+1` (`sub_id` 7;
`subsequent_counter` on counter, reversal-born on confluence) has rows on the
confluence `_wvmi.csv` too. Before Plan C each per-trigger sid was swept on its
own lens only.

## Computation lifecycle: created / updated / locked (`status`)

> **NOTE (2026-05-27):** `WVMIRecord.status` ∈ `{created, updated, locked}` tracks
> LP (Last Pullback) finalization — the **computation** state. This is WVMI's only
> status field. (A separate active/ended *lifecycle* layer was briefly added then
> **removed** 2026-05-27 — see "Lifecycle" below; lifecycle belongs to the cycle /
> wave candles, not to WVMI.)

### 0. Gate — see "WVMI Gating" above

The main only: only its gated cycles proceed to step 1. A sub is ungated —
every CTS_CONFIRMED of its projection proceeds (Plan G).

### 1. Created — CTS_n Confirmed

Triggered by `on_cts_confirmed()` for every CTS_CONFIRMED that passes step 0. Derives 4 wave candle indices from BOS_n and CTS_n:

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
- Finds qualified candle (FP direction + vol_dir match) between FP and the search end
  (see "Temporary LP Selection": the cycle's last live candle, else the data end)
- Picks candle whose close is closest to BOS zone outer bound
- Updates LP idx, volume, weight, and pullback_momentum

### 3. Locked — BOS_n+1 Confirmed

`on_bos_confirmed()` with cycle_id=N+1 locks WVMI for cycle_id=N:
- Replaces temp LP with official LP from BOS_n+1's `last_wave_candle_idx` — when
  BOS_n+1 has no last wave candle, OR that candle lies outside the tracker's frame
  (past a capped sub's end, or past the data edge — Plan G Q10; before it the idx
  was kept with volume / pullback None), the record locks with its existing temp LP
  as final (`on_bos_confirmed`), i.e. the bounded one ("Temporary LP Selection")
- Finalizes pullback_momentum
- Sets `lp_locked=True`, `status="locked"`

A record only locks when its cycle's **successor** BOS forms. A cycle that ends
otherwise (a reversal; a sub's cap) or is still open never locks — it stays
`created`/`updated` with a temp LP searched up to its cycle's end − 1 (open: the
data end) — the main and, since Plan G, every sub. A cycle whose end comes at or
before FP + 1 gets no temp LP (`lp_idx=None`). Details: "Temporary LP Selection".

---

## Lifecycle — REMOVED 2026-05-27 (lifecycle is a cycle/wave-candle property, not WVMI's)

> **DECISION 2026-05-27.** A WVMI-record lifecycle (`status`→`lp_status` rename +
> scalar `start_idx`/`end_idx`/`end_reason` + derived `status`) was briefly added
> (commit `68b862b`) and then **REMOVED**. Rationale: after reframing, **WVMI is a
> derived momentum calc; the lifecycle belongs to the CYCLE / wave candles**, which
> already have it via `compute_cycle_lifecycle` (for ALL cycles — WVMI records exist
> only for proximity-gated cycles, so a WVMI-record lifecycle can't even serve the
> "chart all wave candles" goal). The WVMI-record lifecycle had **no consumer** (the
> chart gates on the cycle/wave-candle lifecycle) — the same "no-consumer → drop it"
> call as the fib `activation_history` (`FIB_LIFECYCLE_SPEC.md §15`). `WVMIRecord`
> reverts to its pre-`68b862b` shape: `status` ∈ `{created, updated, locked}` only
> (the computation axis above), no scalar lifecycle fields, no `_finalize`.
>
> **Wave-candle chart rendering** (which is what motivated all this) is specified in
> **`WAVE_CANDLES_SPEC.md` "Chart Rendering & Lifecycle"** — gated by the cycle
> lifecycle (hide collapsed/never-active cycles), drawn for ALL structures like
> zones, with Phase-1 simplifications (no momentum hover, uniform opacity).

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

1. Search range: `[FP_idx + 1, search_end]`, at creation AND on every update
   (`WVMITracker._lp_search_end`). `search_end` = the cycle's lifecycle end − 1
   (`compute_cycle_lifecycle`, half-open `[start, end)` — the table KL / POI read;
   `_compute_wvmi_records` passes it as `cycle_end_by_key`), the frame end for an
   open cycle — the main and every sub projection (a sub's cap ends every cycle,
   so a capped sub record always has an end). Since 2026-09-28 on the main: the
   search ran to the data end, so a cycle ended by a reversal took its LP from the
   structure that superseded it — reference window H1 (0,1) (reversal 902) had LP
   988 (pullback_momentum 0.904) → now 896 (0.759); `tests/test_wvmi_lp_bound.py`.
   Since Plan G (Q4) on subs: before it the sub sweep passed no ends and searched
   to its frame end INCLUSIVE (the sub's end candle could be picked); the `end − 1`
   bound changed 0 values on the reference window (18/18 records identical).
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
| Temp LP not found (incl. an empty bounded search: the cycle ended at or before FP + 1) | Record created with `lp_idx=None`, `pullback_momentum=None` |
| BOS_n+1 has no `last_wave_candle_idx`, or it lies outside the tracker's frame (Plan G Q10) | Locks with the existing (bounded) temp LP as final |
| FB / LB / FP outside the tracker's frame (past a capped sub's end) | Returns None (no record) — the frame is `df.iloc[:cap + 1]` |
| Momentum is None | Charting shows "N/A" in hover |

---

## Entity Attribution (Part 4 §8.7)

- `WVMITracker.__init__(structure_path_id=...)` stamps every record with the
  owning entity's path (e.g., `"H1.main"`; a sub projection's first live record's
  lens path). On a lens df the mirror sets each sub record copy's FIELD to THAT
  lens's path (= its meta's; Plan G G3) — the projection's own record keeps its.
- `WVMITracker._records` is keyed by `(sid, cycle_id)` — one tracker per
  entity (per sub projection), so source-based disambiguation isn't needed.
- `add_scenario3_record` / `discard_scenario3` were removed in Part 4
  Step 3c — they were dead code (production never invoked them).

---

## Pipeline Integration

WVMI runs **after POI zones** (the main-entity gate needs POI inner bounds), through ONE helper,
`orchestrator._compute_wvmi_records`, for the main and every sub projection (Plan G G1):

```
# Main (H1.main) — _run_downstream_pipeline(wvmi="first_sd_prox"), the default
wave_candles → Fib tracking → POI zones → WVMI:
  0. check_zone_proximity() → first sd trigger per cycle   [gate: _first_sd_prox_gate]
# Sub projection — project_to_window → _run_downstream_pipeline(wvmi="none"): no gate, no proximity
# Both (_compute_wvmi_records):
  life = compute_cycle_lifecycle(events, reversals, floor, cap, reason)
     cycle_end_by_key = its ends                            [temp-LP bound: end - 1]
  frame = df.iloc[:cap + 1] if cap is not None else df     [no read past a sub's end]
  1. for CTS_CONFIRMED events (main: gated only) →
       on_cts_confirmed(frame) + the gate's meta (main)    [create]
  2. for BOS_CONFIRMED events → on_bos_confirmed(frame)     [lock; an LP outside the frame → the temp LP]
  3. update_temporary_lp(frame)                              [shift]
  4. rec.cycle_collapsed = end is not None and start >= end [the cycle's window is empty]
  → main: df.attrs["wvmi"]; sub: LowerTFResult.wvmi_records (slice-local)
```

```
# Sub records onto the lens dfs (_run_multi_tf_dual)
step 5: render_sub_projection → mirror step 8: one deep copy per lens the sub is on,
        FB/LB/FP/LP + slice_begin, field + meta structure_path_id = that lens's path
step 6: _stamp_sub_wvmi_trigger_meta(results_by_lens, _wvmi_trigger_streams_by_lens(...))
        per lens: the first LOH-mapped WVMI-class trigger in [start_idx, m15_end_idx]
        → meta = {triggered_by_event_idx | None, triggered_by_event_type | None,
                  parent_path_id "H1.main", **attribution}
        summary: _count_sub_wvmi(all_results) — per unique sub
```

---

## WVMIRecord Fields

| Field | Type | Description |
|-------|------|-------------|
| `bos_structure_id` | int | Structure ID of the BOS zone (entity-local) |
| `bos_cycle_id` | int | Cycle ID of the BOS zone (entity-local) |
| `zone_side` | "buy"/"sell" | BOS zone side |
| `structure_path_id` | Optional[str] | Owning entity's path (e.g., `"H1.main"`); on a lens df's sub record copy, that lens's path (== `meta["structure_path_id"]`) |
| `fb_idx`, `lb_idx`, `fp_idx`, `lp_idx` | Optional[int] | Wave candle indices |
| `fb_volume`, `lb_volume`, `fp_volume`, `lp_volume` | Optional[float] | Raw volumes |
| `lb_weight`, `lp_weight` | float | Last candle weights (default 1.0) |
| `breakout_momentum`, `pullback_momentum` | Optional[float] | Computed ratios |
| `buy_momentum`, `sell_momentum` | Optional[float] | Direction-labeled wrappers |
| `status` | str | "created"/"updated"/"locked" (LP-finalization computation state) |
| `lp_locked` | bool | Whether LP is finalized |
| `locked_by_cycle_id` | Optional[int] | BOS cycle that locked this record |
| `cycle_collapsed` | bool | The record's CYCLE has an empty lifecycle window (`start >= end` in `compute_cycle_lifecycle`, e.g. a cycle confirmed before a sub's `start_idx`): exported (the column right after `lp_locked`), inert — like a collapsed cycle's zones (`_zone_render.collapsed_cycles`). A flag only, no lifecycle fields (see "Lifecycle — REMOVED"). Main and subs (Plan G G2). A record created ON its cycle's end candle (CTS_CONFIRMED moment == the cap, kept by the inclusive knowable-at clip) is not flagged — documented edge |
