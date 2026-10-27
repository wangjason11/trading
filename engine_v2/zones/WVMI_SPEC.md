# WVMI Spec — Week 8 Part 2

Primary file: `zones/wvmi.py`
Type definition: `common/types.py` (`WVMIRecord`)

---

## Overview

Wave Volume Momentum Indicator (WVMI) measures BOS zone strength by tracking volume momentum across wave cycles. Each record captures volume ratios between paired wave candles (first/last for breakout and pullback waves).

---

## WVMI Records and Trigger Attribution (no gate)

**No entity gates a record:** every CTS_CONFIRMED is offered to the tracker — a rendered unique sub's since Plan G
(2026-09-30, computed inside the sub's projection like its zones), the main's since 2026-10-26 ("WVMI back on the
chart" step 2; the user, round 4: "I'm leaning toward updating the gate to being CTS confirmed rather than
proximity", decided round 5 — a record is created at its CTS_CONFIRMED, i.e. no gate; until then a main cycle got a
record only when its first zone-proximity trigger was sd). The triggers are ATTRIBUTION, stamped after the tracker: the main's by
`orchestrator._stamp_main_wvmi_attribution`, a sub's per lens (`_stamp_sub_wvmi_trigger_meta`, below).
`_run_downstream_pipeline(wvmi=...)` selects the mode: `"first_sd_prox"` (the main — zone proximity runs and the
records are attributed; the name kept by the user's choice 2026-10-26, it names what the main reads), `"none"` (a
sub, `project_to_window`), `"off"` (tests); any other value raises.

### Main entity (`H1.main`)

`check_zone_proximity()` in `zones/zone_proximity.py` still runs on the main (alternating sd/opp_sd triggers per
cycle — see [POI_ZONES_SPEC](POI_ZONES_SPEC.md); `first_counter`, the `subsequent_*` triggers and this attribution
read it). Every record's meta carries the §8.7 attribution schema, as its FIRST keys, from its cycle's **first
zone-proximity trigger when that is sd** (`zone_proximity.first_sd_proximity_triggers`, the ONE reading shared with
`first_counter` — `multitf/uc1_trigger.detect_uc1_triggers`, which reads the proximity lists directly since
2026-10-26, "WVMI back on the chart" step 1):

- `triggered_by_event_idx` — the first sd trigger's candle index
- `triggered_by_event_type = "ZONE_PROXIMITY_TRIGGER"`
- `structure_path_id`
- `trigger_inner`, `proximity_pips`

A cycle whose first trigger is opp_sd, or that has none, still gets its record, with the same keys and None values
(`structure_path_id` kept): `debug/export_wvmi.py` reads `triggered_by_*` strictly (an empty cell = no trigger).
On the reference window: 5 H1 records, 3 attributed (sid 0 c1 748, sid 1 c1 820, sid 1 c2 935) and 2 not (sid 0 c0,
sid 1 c0 — before step 2 these two had no record).

#### Narrow-cycle implications (Rule 1/2/3)

Per `zones/zone_proximity.py`'s narrow-gap rules (they shape the attribution only — the record exists either way):

- **Narrow cycle (gap < `min_gap_pips`) without pullback CTS_confirmed:**
  no proximity triggers (Rule 2) → an unattributed record.
- **Narrow cycle with pullback CTS_confirmed:** triggers can fire from
  the pullback idx forward, capped at ≤1 sd + ≤1 opp_sd (Rule 3). The
  first sd trigger, if it occurs, is the record's
  `triggered_by_event_idx`: the post-pullback sd wick idx (NOT the
  pullback idx).
- **Mid-cycle crossing (narrow → wide):** the cap lifts at the
  crossing; alternation continues unchanged. The record references
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
     Selection") and for **`cycle_collapsed`** (`start >= end`, see "WVMIRecord Fields"); a record without a table
     row fails an assert, capped or not (unreachable: a record needs its cycle's CTS wave candle, so its
     CTS_ESTABLISHED, which the clip keeps with its CTS_CONFIRMED; under a cap every row has an end);
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
   - **counter**: the opp_sd-prox class — each var 3 (`SUBSEQUENT_CONFLUENCE_TRIGGER`).
   Each entry is LOH-mapped once (`_map_parent_idx_to_m15_hour_end`), sorted by `(m15_idx, parent_idx)`; per sub
   on the lens, the FIRST entry inside the sub's real-time window **`[meta["start_idx"], meta["m15_end_idx"]]`**
   (`m15_end_idx` = the sub's end, or the geometry's data edge for an open sub) is the lens's trigger. Every
   record of that lens df, joined to its sub on `meta["sub_id"]`, gets as its FIRST meta keys
   `triggered_by_event_idx` (**parent-df coords — never translated**) / `triggered_by_event_type` — both None when
   no entry lands in the window — and `parent_path_id` = `"H1.main"` (always: the parent entity). On a sub row
   `triggered_by_*` means **"the lens's first WVMI-class trigger inside the sub's window"** — attribution, stamped
   retroactively (a declared meaning change, LANDMINES "Event Contract Rules" rule 3; the main's rows kept the
   gate meaning until 2026-10-26, when they became attribution too — "the cycle's first zone-proximity trigger,
   when sd"). The exporter reads both keys strictly (`meta[...]`): a record reaching a CSV unstamped raises
   instead of exporting as "no trigger"; a `sub_id` twice on one lens fails an assert (the join key).
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

### 0. No gate — see "WVMI Records and Trigger Attribution" above

Every CTS_CONFIRMED proceeds to step 1, the main's (since 2026-10-26; gated on its first sd zone-proximity trigger
before) and a sub projection's (Plan G).

### 1. Created — CTS_n Confirmed

Triggered by `on_cts_confirmed()` for every CTS_CONFIRMED. Derives 4 wave candle indices from BOS_n and CTS_n:

| Role | Source | Wave Candle Field |
|------|--------|-------------------|
| FB (First Breakout) | BOS_n wave candle | `first_wave_candle_idx` |
| LB (Last Breakout) | CTS_n wave candle | `last_wave_candle_idx` |
| FP (First Pullback) | CTS_n wave candle | `first_wave_candle_idx` |
| LP (Last Pullback) | Temporary — the furthest qualified close in the pullback direction ("Temporary LP Selection") | Moves until locked |

**Breakout momentum is LOCKED at creation:**
```
breakout_momentum = (LB_vol * LB_weight) / FB_vol
```

**Pullback momentum starts SHIFTING:**
```
pullback_momentum = (LP_vol * LP_weight) / FP_vol   # recomputed each candle
```

### 2. Updated — Each Candle

`update_temporary_lp(df)` re-scans all non-locked records:
- The temp LP = the last of the record's furthest-close sequence up to the search end
  ("Temporary LP Selection": the cycle's last live candle, else the data end); the
  tracker keeps each record's CTS_CONFIRMED for its first-LP route. No zone is read.
- Updates LP idx, volume, weight, and pullback_momentum; `status="updated"` only when
  the LP moved (the replay creates and re-scans on the same frame, so its unlocked
  records stay `created`)

### 3. Locked — BOS_n+1 Confirmed

`on_bos_confirmed()` with cycle_id=N+1 locks WVMI for cycle_id=N:
- Replaces temp LP with official LP from BOS_n+1's `last_wave_candle_idx` — when
  BOS_n+1 has no last wave candle, OR that candle lies outside the tracker's frame
  (past a capped sub's end, or past the data edge — Plan G Q10; before it the idx
  was kept with volume / pullback None), the record locks with its existing temp LP
  as final (`on_bos_confirmed`), i.e. the bounded one ("Temporary LP Selection")
- Prints `[wvmi] WARNING <structure_path_id> sid=… cycle=…: the locked LP … != the last
  temporary LP …` when the swap changes the LP (2026-10-26; the user, round 5: keep the
  swap, warn on a difference) — a log line only, slice-local indices on a sub
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

## Temporary LP Selection (`_temporary_lp_sequence`) — the furthest-close rule

The temporary LP follows the pullback: a new temporary LP at each qualified candle that
closes FURTHER in the pullback direction than the current one (lower for a buy zone's
bearish pullback, higher for a sell zone's bullish one); an equal close does not move
it. The record's `lp_idx` = the last one; `_temporary_lp_sequence` returns them all as
`[(as_of_idx, lp_idx), …]`. Decided by the user 2026-10-26 ("WVMI back on the chart"
rounds 4–7, step 3 — `plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`);
it replaced the closest-to-outer rule (`_find_temporary_lp`: the qualified close nearest
the BOS zone's `meta["outer"]`), which stopped following a pullback once it closed past
the zone's outer edge — 6 records on the reference window, all ended by a reversal
(e.g. H1 sid 0 cycle 1: LP 896 → 901, pullback momentum 0.759 → 0.478). The BOS zone is
no longer read for the LP (a record still needs one — "Guard Rails").

1. **Candidates**: a qualified candle (GLOSSARY "Qualified candle": the pullback
   direction AND the volume direction agrees or is 0 — `wave_candles._is_qualified`,
   the one helper every wave-candle role uses) in `[FP_idx + 1, search_end]`, at
   creation AND on every re-scan — never FP, on every route below
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
   bound changed 0 values on the reference window (18/18 records identical). The
   candle that ENDS a cycle (a reversal's, a replacement's, the next cycle's BOS) is
   never an LP (the user, round 5) — on the reference window every reversal-ended
   cycle's ending candle is qualified and closes further (H1 902; confluence 1940,
   2470, 2829, 3589, 4200), so the bound decides those records' final LP.
2. **The first temporary LP, by how the CTS confirmed** (p = the CTS_CONFIRMED moment,
   `ef.event_moment`):
   - **pullback pattern** (`confirmation_method == "pullback"`): the furthest close
     among the pattern's candles `[pattern_anchor_idx, p]` — the pullback pattern's
     first candle (`meta["pattern_anchor_idx"]`, read by `ef.pattern_anchor_idx`)
     through its confirmation candle, in-between candles included — as of p.
     **Fallback** (no qualified pattern candle after FP — the pattern's only qualified
     candle is FP): the first later candidate closing beyond **FP's close**, not the
     confirmation candle's (the user, round 6: that candle closes away from the
     pullback direction). Only the fallback is held to FP's close (round 7).
   - **sd zone proximity** (`"sd_zone_proximity"`): the furthest close in
     `[FP + 1, p − 1]` that closes beyond p's close, as of p; else p itself if it is a
     candidate (qualified, and not FP); else the first later candidate closing beyond
     p's close. The pattern route looks only at the pattern's own candles, the
     proximity route back to FP + 1 — intended (round 5).
   - Until the first LP exists (the fallback or the proximity route still waiting)
     the record has none (`lp_idx` None, "Guard Rails").
3. **Then** each candidate after the current LP's as-of candle that closes further
   than the current LP is the next temporary LP, as of itself.
4. **The lock** ("3. Locked") replaces the last temporary LP with BOS_{n+1}'s last
   wave candle and warns when they differ. Reference window: 1 record — the collapsed
   confluence sub 1 cycle 0, last temporary LP 1866 vs locked 1837 — printed once per
   tracker run that locks it: 10 lines (the confluence lens + 9 proximity-view
   projections of the pool sub, `[prox_view] pool#1 cap=…`); every visible locked
   record's lock LP = its last temporary LP.

Reference window (2026-10-26): the pattern route on 28 of the 30 records, the
proximity candle itself on 2 (H1 sid 1 cycle 2; the collapsed confluence sub 1 cycle
1); no record reaches the fallback. Measured with
`plans/qa_chart_edits_inputs/wvmi_asof.py` (`rule_new`).

---

## Per-LP rows (`WVMIRecord.lp_rows`, `*_wvmi_lps.csv`)

Every LP a record has had is kept — one row per temporary LP and the locked one, one dot each on the chart
("WVMI back on the chart" step 4, 2026-10-26; the user's rounds 3 / 8 / 10 / 11 —
`plans/qa_chart_edits_inputs/user_spec_20261024_wvmi_on_chart.md`). A row (`common/types.WVMILPRow`): `lp_idx`,
`as_of_idx` (the candle it became the LP: its own, or the CTS_CONFIRMED moment for a first LP picked from earlier
candles — "Temporary LP Selection" 2 —, or the lock for a swapped-in locked LP), `lp_status` / `lp_status_reason` /
`lp_status_idx` (what / why / when — the hover joins them as `lp_status=<status> (<reason> at <idx>)`), and
`pullback_momentum` with THAT LP (`_lp_pullback_momentum`: LP_vol × LP_weight / FP_vol, the record's formula). The
tracker builds them (`WVMITracker._set_lp_rows`) from the record's furthest-close sequence on every create /
re-scan / lock, so the last row is always the record's `lp_idx` / `pullback_momentum`:

| `lp_status` | `lp_status_reason` | `lp_status_idx` | Which row |
|---|---|---|---|
| `temporary` | `replaced by a further close` | the next LP's as-of candle | every LP but the last |
| `temporary` | `replaced by the locked LP` | the lock (BOS_{n+1}'s moment) | the last temporary LP when the locked LP is another candle (the `[wvmi] WARNING` case) |
| `locked` | `cycle <N+1>'s BOS confirmed` | the lock | the locked LP — the last temporary LP itself when they agree (no extra row), else its own row as of the lock; with no LP before the lock, its only row |
| `open` | `still moving at the data end` | None | the last LP of a cycle with no end |
| `unlocked` | `cycle ended by a reversal` / `sub replaced` / `sub ended` | the cycle's lifecycle end | the last LP of a cycle that ended without a lock — its `compute_cycle_lifecycle` `end_reason` (`reversal` / `same_dir_replacement` / `parent_end`; `_compute_wvmi_records` passes it as `cycle_end_reason_by_key`; any other reason prints raw, `cycle ended (<reason>)`) |

A lock that keeps the temporary LP (BOS_{n+1} has no last wave candle, or it lies outside the frame — "3. Locked")
turns the last temporary row `locked`. A record without an LP has no rows. On a lens df the mirror shifts each row's
`lp_idx` / `as_of_idx` / `lp_status_idx` by `slice_begin` with the record's FB / LB / FP / LP.

**The CSV** (`debug/export_wvmi.export_wvmi_lps`): `*_wvmi_lps.csv` (H1) and `*_M15_{counter,confluence}_wvmi_lps.csv`,
the rows of the records of `_wvmi.csv` in its order, then LP order. Columns (round 8): `structure_path_id`,
`parent_sid`, `parent_cycle_id`, `sub_id`, `bos_structure_id`, `bos_cycle_id` (named as in `_wvmi.csv`),
`relative_dir`, `zone_side`, `lp_idx`, `as_of_idx`, `lp_status`, `lp_status_reason`, `lp_status_idx`,
`breakout_momentum` (the record's), `pullback_momentum` (the row's) — full precision, integer columns nullable `Int64`.
- **One row per dot: a COLLAPSED record (`cycle_collapsed`) gets none** (the user, round 10: "for now collapsed records
  don't need to get per-LP rows. I may change my mind later" — register §G); it keeps its rows on the record and its
  `_wvmi.csv` row. `cycle_collapsed` is set after the tracker, so the export filters.
- **`relative_dir`** (M15) = the sub's at the LP candle — the §17.3 step function over THIS lens's `SidRecord`
  (`multitf/types.sub_relative_dir_at`, the dot hover's reader); before the sub's `start_idx` it is
  `NA (before start_idx)` (the user, round 11: a sub cycle can confirm its CTS while the sub is still forming —
  reference window sub 8 cycle 0, LPs 4102 / 4129 before its start 4143). Empty on an H1 row. A sub row whose
  `SidRecord` is not on the lens fails loudly.

Reference window (2026-10-26, S_lp_furthest's records): H1 57 rows, counter 74, confluence 185 — `temporary` 54 / 68 /
171, `locked` 1 / 3 / 7, `unlocked` 1 / 3 / 7, `open` 1 / 0 / 0; no `replaced by the locked LP` row (the one record
whose locked LP differs is collapsed).

---

## Guard Rails

| Condition | Behavior |
|-----------|----------|
| BOS zone not found | Returns None (no record created) |
| BOS or CTS wave candle missing required indices | Returns None |
| FB_volume == 0 or FP_volume == 0 | Returns None (division by zero) |
| Temp LP not found (incl. an empty bounded search: the cycle ended at or before FP + 1; a first LP still waited for) | Record created with `lp_idx=None`, `pullback_momentum=None` |
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

WVMI runs **after POI zones** (the main's attribution reads the zone-proximity triggers, which need POI inner
bounds — a gate until 2026-10-26), through ONE helper, `orchestrator._compute_wvmi_records`, for the main and every
sub projection (Plan G G1), ungated for both:

```
# Main (H1.main) — _run_downstream_pipeline(wvmi="first_sd_prox"), the default
wave_candles → Fib tracking → POI zones → WVMI:
  0. check_zone_proximity() → the cycles' trigger lists     [first_counter, subsequent_*, the attribution]
# Sub projection — project_to_window → _run_downstream_pipeline(wvmi="none"): no proximity
# Both (_compute_wvmi_records):
  life = compute_cycle_lifecycle(events, reversals, floor, cap, reason)
     cycle_end_by_key = its ends                            [temp-LP bound: end - 1]
     cycle_end_reason_by_key = its end_reasons              [an unlocked record's last per-LP row]
  frame = df.iloc[:cap + 1] if cap is not None else df     [no read past a sub's end]
  1. for CTS_CONFIRMED events → on_cts_confirmed(frame)     [create; meta {}; the per-LP rows]
  2. for BOS_CONFIRMED events → on_bos_confirmed(frame)     [lock; an LP outside the frame → the temp LP; rows rebuilt]
  3. update_temporary_lp(frame)                              [shift; rows rebuilt]
  4. rec.cycle_collapsed = end is not None and start >= end [the cycle's window is empty]
# Main only: _stamp_main_wvmi_attribution(records, the trigger lists) [each cycle's first sd trigger, or None]
  → main: df.attrs["wvmi"]; sub: LowerTFResult.wvmi_records (slice-local)
```

```
# Sub records onto the lens dfs (_run_multi_tf_dual)
step 5: render_sub_projection → mirror step 8: one deep copy per lens the sub is on,
        FB/LB/FP/LP + each per-LP row's lp_idx / as_of_idx / lp_status_idx + slice_begin,
        field + meta structure_path_id = that lens's path
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
| `lp_rows` | List[WVMILPRow] | Every LP the record has had, in order — the last = `lp_idx` ("Per-LP rows"); empty without an LP (2026-10-26) |

**CSV export** (`debug/export_wvmi.py`, one `_wvmi.csv` per entity / lens): the fields above + the flattened attribution
(`parent_sid`, `parent_cycle_id`, `sub_id`, `started_by`, `triggered_by_event_idx` / `_type`, `parent_path_id`) + the
full `meta` text; `cycle_collapsed` right after `lp_locked`. Every integer column is written as pandas nullable
`Int64` (`_INT_COLUMNS`; the user's decision 2026-09-30): an int column holding a None would otherwise print every
int as a float (`locked_by_cycle_id` `1.0`, `triggered_by_event_idx` `710.0`); a None is an empty cell. The trigger
keys are read strictly (`meta[...]`). The per-LP rows (`lp_rows`) go to their own CSV, `*_wvmi_lps.csv` — "Per-LP
rows" above.

