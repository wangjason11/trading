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

> **Status (Plan C, landed 2026-09-20): this is the MINIMAL implementation of the
> deferred WVMI design under the sub-structure pool (`PART4_REFACTOR_SPEC.md
> §17.10`) — the user's stated lean ("WVMI is a property of the unique sub → one
> sweep"), implemented so the code runs. It is NOT settled: the deferred WVMI
> pass may return to per-(sub, lens) sweeps. Do not build on it as final.
> REVISIT (user decision 2026-09-20 — "accept for now, revisit later"): the
> persist-into-every-lens rule puts a sub's records on BOTH lens CSVs/charts
> (measured: sub `2639/−1`'s three records also on counter, sub `4027/+1`'s
> two also on confluence); the WVMI pass decides sub-owned vs lens-owned.**
> The pre-pool text (per-parent-cycle pass `_assign_trigger_centric_sub_wvmi`
> after `build_parent_cycle_chain`, window `[start_trigger_idx, m15_end_idx]`
> with "`start_trigger_idx` IS the sub's lifecycle-start") is superseded by the
> rules below: the window is now the unique sub's real-time `start_idx`, and
> `start_trigger_idx` was split into `trigger_idx` / `probe_finalize_idx` /
> `start_idx` (GLOSSARY "Sub-Structure Pool Terms").

Sub WVMI is **parent-event-driven and trigger-centric** (§8.3 / §8.4 / §8.5),
computed **once per unique sub**. `_run_downstream_pipeline` runs with
`skip_wvmi=True` for subs (the entity-local proximity gate is bypassed;
`project_to_window` returns `wvmi_records=[]`). After every sub projection has
been mirrored into its lens dfs, the orchestrator runs ONE pass over the unique
subs (`pipeline/orchestrator._assign_sub_wvmi_per_sub(sub_results,
streams_by_lens, ...)`):

1. Build the **parent trigger streams** (parent-df idxs) over **all** parent
   cycles (`ParentTables.cycles()` — a unique sub spans parent cycles):
   - **confluence** (`_confluence_trigger_stream`): main's first sd-prox after
     CTS **plus each var 4** (`subsequent_counter`) in the cycle — the
     sd-prox-class events (`event_type` `ZONE_PROXIMITY_TRIGGER` /
     `SUBSEQUENT_COUNTER_TRIGGER`).
   - **counter** (`_counter_trigger_stream`): **each var 3**
     (`subsequent_confluence`) in the cycle — the CTS-prox-class events
     (`SUBSEQUENT_CONFLUENCE_TRIGGER`).
   Each stream entry is LOH-mapped once (`_map_parent_idx_to_m15_hour_end`) and
   tagged with its lens; the union is sorted by `(m15_idx, parent_idx)`.
2. For each sub projection (`LowerTFResult`, in the sub `start_idx` order the
   orchestrator renders them): the window is the sub's real-time lifecycle
   **`[meta["start_idx"], meta["m15_end_idx"]]`** — `start_idx` = the unique
   sub's `start_idx` (the first live record's
   `max(probe_finalize_idx, trigger_idx, parent_floor_idx)`); `m15_end_idx` =
   the sub's `end_idx`, or the geometry's data edge for an open sub. The stream
   is the union of the two lens streams **restricted to the sub's lenses**
   (`meta["lenses"]`, a confluence entry counts only for a sub on the
   confluence lens, a counter entry only for a sub on the counter lens). The
   **first** entry (by `m15_idx`) with `start_idx <= m15_idx <= m15_end_idx`
   sweeps the sub once via `compute_parent_driven_sub_wvmi(result,
   sub_path_id=lens_paths[lens], parent_trigger=ParentTrigger(idx=parent_idx,
   event_type, parent_path_id="H1.main"))` — **the sweeping trigger's lens
   decides the records' `structure_path_id`**. No entry inside the window → no
   WVMI for that sub.
3. **Dedup key = `sub_id`** (a sub is swept at most once, whatever number of
   lenses or triggers touch it). The records are written to
   `result.wvmi_records` and **persisted into EVERY lens df the sub is on**
   (`persist_facade_wvmi_to_entity_df(lens_dfs[l], result,
   structure_path_id=lens_paths[l])` for each `l` in the sub's lenses), each
   stamped with the §17.9 attribution (`sub_id` + informational `parent_sid` /
   `parent_cycle_id` / `use_case` / `started_by` from the sub's first record).

**Measured consequence on the reference window (first Plan C replay,
2026-09-20):** a sub on both lenses now carries its WVMI rows on both lens
CSVs — sub `2639/−1` (`sub_id` 3; confluence record reversal-born from
`2365/+1`, counter record from `first_counter`) has rows on the counter
`_wvmi.csv` as well as the confluence one; sub `4027/+1` (`sub_id` 7;
`subsequent_counter` on counter, reversal-born on confluence) has rows on the
confluence `_wvmi.csv` too. Before Plan C each per-trigger sid was swept on its
own lens only.

**There is no `use_case` special-casing.** A unique sub — first-born
(var1/var2), subsequent (var3/var4), or **reversal-born** — gets WVMI **iff a
parent trigger of one of its lenses' classes lands inside its window**; a sub
no trigger lands on gets none. This is the correct reading of §8.3's "whichever
sid is active — created by var1, var3, or sub internal reversal": a
reversal-born sub is covered when a *later* parent trigger falls in its window,
**not** by self-triggering on the reversal (a sub reversal is neither an sd-prox
nor a CTS-prox parent event). See GOTCHAS "Sub WVMI is Trigger-Centric, Not
Sid-Centric".

A sub touched by multiple triggers is swept once (continuous tracker — the sweep
already covers all the sub's internal cycles); the earliest (initiating) trigger
wins attribution. Re-trigger semantics (a var 4 landing on an already-swept
confluence sub) are deferred to the WVMI pass.

Records share main's `WVMITracker.on_cts_confirmed` / `on_bos_confirmed` /
`update_temporary_lp` lifecycle, plus §8.7 attribution merged into
`record.meta`: `triggered_by_event_idx` (**parent-df coords — never
translated**), `triggered_by_event_type`, `parent_path_id`; and the sub
identity **`sub_id`** (renamed from `sub_sid` by Plan C — the `_wvmi.csv`
exporter's column is `sub_id`; on the reference window this is the H1
`_wvmi.csv`'s only diff vs the Plan B save: header-only, the shared exporter's
column rename).

---

## Computation lifecycle: created / updated / locked (`status`)

> **NOTE (2026-05-27):** `WVMIRecord.status` ∈ `{created, updated, locked}` tracks
> LP (Last Pullback) finalization — the **computation** state. This is WVMI's only
> status field. (A separate active/ended *lifecycle* layer was briefly added then
> **removed** 2026-05-27 — see "Lifecycle" below; lifecycle belongs to the cycle /
> wave candles, not to WVMI.)

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
- Finds qualified candle (FP direction + vol_dir match) between FP and the search end
  (see "Temporary LP Selection": the cycle's last live candle, else the data end)
- Picks candle whose close is closest to BOS zone outer bound
- Updates LP idx, volume, weight, and pullback_momentum

### 3. Locked — BOS_n+1 Confirmed

`on_bos_confirmed()` with cycle_id=N+1 locks WVMI for cycle_id=N:
- Replaces temp LP with official LP from BOS_n+1's `last_wave_candle_idx`
- Finalizes pullback_momentum
- Sets `lp_locked=True`, `status="locked"`

A record only locks when its cycle's **successor** BOS forms. A cycle that ends
otherwise (a reversal; a sub's cap) or is still open never locks — it stays
`created`/`updated` with a temp LP searched up to its cycle's last live candle
(open: the data end; "Temporary LP Selection").

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
   (`WVMITracker._lp_search_end`). **Main:** `search_end` = the cycle's lifecycle
   end − 1 (`compute_cycle_lifecycle`, half-open `[start, end)` — the table KL / POI
   read; the orchestrator passes it as `cycle_end_by_key`), the data end for an open
   cycle. Since 2026-09-28: the search ran to the data end, so a cycle ended by a
   reversal took its LP from the structure that superseded it — reference window H1
   (0,1) (reversal 902) had LP 988 (pullback_momentum 0.904) → now 896 (0.759);
   `tests/test_wvmi_lp_bound.py`. **Sub sweep:** no ends passed — the projection's
   frame already stops at the sub's `end_idx` (measured 5/5), so a sub LP stays in
   the sub; that frame end is inclusive (the end candle itself can be picked) — the
   one-candle difference from main's half-open bound is left to the deferred WVMI
   plan (WVMI moves into the per-sub projection).
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
# Sub (M15.counter / M15.confluence) — parent-event gate, ONE sweep per unique sub
# (§17.10 minimal, Plan C 2026-09-20 — not settled)
# `_run_downstream_pipeline(..., skip_wvmi=True)` short-circuits the gate above.
# Orchestrator (`_assign_sub_wvmi_per_sub`) runs after every sub projection is mirrored:
mapped = LOH-map(confluence stream ∪ counter stream over ALL parent cycles), sorted
for each unique sub projection (start_idx order):
    window = [meta["start_idx"], meta["m15_end_idx"]]      # real-time lifecycle
    hit = first mapped entry whose lens ∈ meta["lenses"] and start <= m15_idx <= end
    if hit and sub_id not yet swept:
        compute_parent_driven_sub_wvmi(result, sub_path_id=lens_paths[hit.lens], parent_trigger)
        → result.wvmi_records = [...]
        persist_facade_wvmi_to_entity_df(lens_df, result) for EVERY lens the sub is on
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
| `status` | str | "created"/"updated"/"locked" (LP-finalization computation state) |
| `lp_locked` | bool | Whether LP is finalized |
| `locked_by_cycle_id` | Optional[int] | BOS cycle that locked this record |
