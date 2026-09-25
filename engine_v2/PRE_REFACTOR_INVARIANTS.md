# Pre-Refactor Invariants

> Snapshot of currently-correct behaviors and architectural decisions as of
> the end of Week 8 Part 3 / before Part 4 (the major pipeline/strategy/
> multi-TF refactor). Use this as a regression-check reference during the
> refactor: each item is either preserved (the refactor must not break it)
> or intentionally changed (the refactor explicitly redesigns it).

This file is meant to be **deleted after the refactor lands** — it's
transitional. The behaviors listed here are documented in their respective
SPEC.md files; this is a consolidated inventory, not a replacement spec.

---

## Pipeline ordering (LOCKED today; refactor may restructure)

```
candle features → structure patterns → imbalance → market structure
  → KL zones → wave candles → Fib tracking → POI zones → WVMI
  → multi-TF (UC1) → charting
```

**Today's invariants:**
- Base features must run before market structure
- WVMI must run after POI zones (depends on POI inner bounds for proximity gate)
- Imbalance instances live in `df.attrs["imbalances"]`; columns also flag `is_imbalance=1` per candle
- KL zones live in `df.attrs["kl_zones"]`; POI zones in `df.attrs["poi_zones"]`; etc.

**Refactor watch:**
- The "WVMI tied to specific market structures" goal will likely change
  pipeline ordering and the WVMI gate (currently uses first sd zone proximity
  trigger as the gate)
- "Strategy" layer will be added — placement TBD

---

## MarketStructure state machine

**Today's invariants:**
- Sequential / event-driven; no skipping; uses rewind+replay only when range
  threshold corrections require it
- `cts_phase` ∈ {NONE, EST_OR_UPD, CONFIRMED, FALSE_BREAK}
- `cts_cycle_id` increments at each `CTS_ESTABLISHED` (the new cycle's CTS)
- Reversal is terminal — once entered, never leaves
- Range invariant: `range_lo <= range_hi` while active
- Pullback never resets range; breakout deactivates range

**Dual CTS confirmation paths (Stage 1 + Stage 2 done):**
- After CTS_ESTABLISHED, the engine watches for both pullback pattern AND
  sd zone proximity (BOS inner + active POI inners). First match confirms.
- `CTS_CONFIRMED.meta["confirmation_method"]` ∈ {`"pullback"`, `"sd_zone_proximity"`}
- `CTS_RECONFIRMED` event fires at pullback idx if proximity confirmed first
- KL zone meta upgrades on CTS_RECONFIRMED (`pb_reconfirm_idx` recorded)
- Range under proximity-only confirmation seeded with proximity wick (Option B)
- `BOS_n+1` derivation:
  - With pullback in cycle n: pullback's range deepest extreme (existing)
  - Without pullback (proximity-only): max retracement in `[cts_n_confirmed_idx, breakout_idx]`
- POI inner snapshot refreshed at CTS_ESTABLISHED + each CTS_UPDATED
  (per-CTS-update, not per-candle — deliberate performance trade-off)

**Refactor watch:**
- The dual confirmation logic currently lives inside MarketStructure with
  a backward dependency on `zones/` modules (via `proximity_helpers.py`).
  Refactor may invert / clean up this dependency direction.

---

## Event contracts

**`ev.idx` semantics — index conventions per event type (CRITICAL):** the
canonical per-event field table (every field, the bound, the measured
frequencies) is `ARCHITECTURE.md` "`ev.idx` convention" — read it there. Summary
only (corrected 2026-09-22; see the note below):

| Event | `ev.idx` means | Notes |
|---|---|---|
| `CTS_ESTABLISHED` | **the moment** = `meta["confirmed_at"]`, the pattern's apply candle (Plan E E4a, 2026-09-25; before it the CTS extreme — retro-stamped) | the anchor (the **CTS extreme** in the breakout pattern span) = `meta["cts_anchor_idx"]` (Plan E E2a); `ev.price` = the anchor's price; `meta["pattern_anchor_idx"]` = the breakout pattern's FIRST candle — **not necessarily the extreme** (it is when the first candle holds it; 0 of 34 on the reference window), never a timing value |
| `CTS_UPDATED` | raw path (`meta["via"] == CTS_UPDATED_RAW_VIA`, i.e. `"replay_raw"`): the processed candle; pattern path (`via` = a pattern name): **the moment**, the apply candle `confirmed_at` (Plan E E4c, 2026-09-25; before it the pattern-span extreme) | no `pattern_anchor_idx` on any `CTS_UPDATED`; a pattern-path update records its apply candle as `confirmed_at` (Plan E E3·0 — `event_moment(ev)` returns it) and its anchor (the span extreme) as `cts_anchor_idx` (E4c); the raw path has neither |
| `CTS_THRESHOLD_UPDATED` | the processing candle (`_sync_thresholds_from_range(i)`) — a moment | `event_fields.event_moment(ev)` (Plan F; moved + extended by Plan E E2a) resolves the moment of every CTS / BOS event in one place |
| `CTS_CONFIRMED` | confirmation candle | `confirmed_at` mirrors `ev.idx`; `meta["cts_anchor_idx"]` = the CURRENT CTS extreme at confirmation (== the cycle's `CTS_ESTABLISHED.meta["cts_anchor_idx"]` only if no `CTS_UPDATED` moved it); `meta["confirmation_method"]` |
| `CTS_RECONFIRMED` (new) | pullback confirmation candle | only fires after proximity-confirmed CTS; `confirmed_at` mirrors `ev.idx` |
| `BOS_CONFIRMED` | **the moment** = `meta["confirmed_at"]`, the same apply candle as the cycle's `CTS_ESTABLISHED` (Plan E E4b, 2026-09-25; before it the **BOS extreme candle**) | the anchor (the BOS extreme) = `meta["bos_anchor_idx"]` (Plan E E2a); `ev.price` = the anchor's price |
| `REVERSAL_CANDIDATE` | reversal-pattern anchor candle (= `meta["pattern_anchor_idx"]`) | `meta["apply_idx"]` = the SCHEDULED apply — a prediction that can expire (the applied reversal is `STATE_CHANGED(to=reversal)`) |
| `RANGE_*` | varies — see GOTCHAS for sort-order rules | |

**Event ordering invariant:** `sorted_events` uses `event_fields.processing_order_key`
= the pre-E4 `(idx, type)` pinned on the anchors (alphabetical type tiebreak; Plan E
E2b) — the E4 flips reorder nothing. Mode C M15 phase gate depends on this — see LANDMINES.

> **Correction note (Plan C 2026-09-20; rewritten 2026-09-22).** The pre-Part-4
> snapshot of this table was wrong on two rows: it gave `CTS_ESTABLISHED` and
> `CTS_UPDATED` `ev.idx` as "confirmation candle", and said of `CTS_ESTABLISHED`
> "extreme is in `meta["anchor_idx"]` for pattern context". **That sentence is
> RETRACTED:** `CTS_ESTABLISHED.meta["anchor_idx"]` (renamed `pattern_anchor_idx`, Plan E E1) is the breakout pattern's
> first candle (`ev.start_idx`, the MS loop anchor) — on the reference window it
> is never the extreme (0 of 34 `CTS_ESTABLISHED` CSV rows); the extreme is
> `ev.idx` itself (until Plan E E4a, 2026-09-25, moved `ev.idx` to the moment;
> the extreme is `meta["cts_anchor_idx"]` since Plan E E2a). The 2026-09-20 note fixed the `ev.idx` column but left that
> sentence standing, and it is the probable origin of a wrong `memory/_INBOX.md`
> line ("anchor_idx is the extreme", 2026-09-22). Extreme == moment is the
> COMMON case, not a coincidence (31 of 34 rows); the 3 lagging rows are 2 unique
> M15 cycles (sub 3 is mirrored into both lenses) — 1223 vs 1224 and 2828 vs
> 2829, lag 1, empirical and not a bound (the bound is `pattern_anchor_idx <= cts_anchor_idx <=
> confirmed_at <= pattern_anchor_idx + 5`). `CTS_ESTABLISHED` was one of the
> extreme-located events until Plan E E4a (with `BOS_CONFIRMED` until E4b and pattern-path
> `CTS_UPDATED` until E4c — none since; ARCHITECTURE.md "`ev.idx` convention"); every lifecycle reader
> (`zones/structure_lifecycle.compute_cycle_lifecycle`,
> `multitf/parent_tables.py`) uses `meta["confirmed_at"]`.

**Event append-only contract:** Events are never modified after emission.
Augmentations (e.g., the imbalance instance refactor's "rewrite events")
must replace events in the list, not mutate event objects.

---

## Zones

**KL zones (`kl_zones_v1`):**
- Stored in `df.attrs["kl_zones"]` as `KLZone` dataclass
- `meta["confirmation_method"]` (CTS zones only): "pullback" / "sd_zone_proximity"
- CTS zone may upgrade method on CTS_RECONFIRMED; `pb_reconfirm_idx` recorded
- Bounds steps: list of (idx, top, bottom, event) for chart rendering of expansions
- Active = the most recent of each side per structure

**POI zones (`poi_zones`):**
- Stored in `df.attrs["poi_zones"]` as `POIZone` dataclass
- **Always sd-direction by Fib construction** (Fib spans BOS→CTS, IC candidates
  inside that span are necessarily sd). No opp_sd POIs exist.
- IC variants V30/V60/V90 stored on each POI

**FibTracker:**
- Two modes: `"h1"` (Scenario 1/2/3) and `"cross_cycle"` (cross-fib state
  machine + phase model with pre_established / established / confirmed).
  `cross_cycle` was previously named `m15_reverse`; renamed when subordinate
  structures generalized beyond M15 + counter direction.
- Cross fibs stored under `_fibs[(sid, cycle_id, "cross", version)]`; old
  versions kept as inactive snapshots for chart history
- `_dead_cycles` cache tracks permanently-filled cycles (Interpretation B)

**Refactor watch:**
- The `_run_downstream_pipeline` shared helper between H1 and M15 has many
  parameters (`source_kinds`, `fib_mode`, etc.) — refactor may create separate
  pipelines per mode

---

## Zone proximity triggers

**Today's invariants:**
- `check_zone_proximity` (in `zones/zone_proximity.py`) returns alternating
  sd/opp_sd trigger candles per cycle
- sd zones = BOS KL + POI; opp_sd zone = CTS KL only
- First trigger must be sd; alternation enforced by state machine
- Scan starts AT CTS_CONFIRMED candle (not +1)
- Default thresholds: H1=9, M15=6, M5=3 pips; caller-overridable
- Single sd trigger per cycle from this function gates WVMI record creation
  (backward compat — proximity_trigger_idx in WVMIRecord.meta)
- `MarketStructure` ALSO does its own internal proximity check (different
  scope: per candle during pre-confirmation, BOS + POI inners) for
  CTS confirmation. Both coexist today.

**Refactor watch:**
- WVMI is currently gated by the first sd zone proximity trigger. Refactor
  will rewire WVMI to be tied to specific market structures, not "ad hoc."
- The two proximity checks (`check_zone_proximity` for general use,
  MarketStructure inline for CTS confirmation) may be unified

---

## Multi-TF (UC1)

**Today's invariants:**
- UC1 trigger: H1 CTS_CONFIRMED + first sd zone proximity trigger
- M15 lower-TF pipeline runs `compute_structure_from_start` from a validated
  start (H1 reverse Scenario 3 probe → mapped to M15)
- KL zones BOS-only on M15 (`source_kinds=["BOS"]`)
- Fib mode `"cross_cycle"` on M15 subs
- Lifecycle bounded by parent H1 cycle (`lifecycle_end_idx`)
- Open-ended zones / POIs / fibs capped at lifecycle end with `deactivated_by="lifecycle_end"`
- M15 events / zones carry attribution: `timeframe`, `use_case`, `parent_tf`, `parent_sid`, `parent_cycle_id`
- M15 slice includes 50-candle lookback buffer; `compute_imbalance` re-run after slice

> **Superseded note (Plan C, 2026-09-20) — the four bullets above are the
> pre-refactor snapshot and are intentionally left as written.** Under the
> sub-structure pool (`PART4_REFACTOR_SPEC.md §17`): a *TriggerRecord* is
> bounded by its parent cycle (`parent_end` from `multitf/parent_tables.py`;
> the detectors' `lifecycle_end_idx` field is RETIRED — deleted in Plan E E1b), while the
> *unique sub* it points at is NOT parent-bound — its window is aggregated from
> its records and spans parent cycles/sids; zones / POIs / fibs are capped at
> the sub's `end_idx` with `end_reason` = the sub's `end_reason`
> (`reversal` \| `same_dir_replacement` \| `parent_end`) — `deactivated_by` and
> `"lifecycle_end"` no longer exist on those objects; attribution gains
> **`sub_id`** (the identity) with `use_case` / `parent_sid` /
> `parent_cycle_id` informational only; the 50-candle lookback slice +
> `compute_imbalance` re-run are preserved (`entity_df_mutation._build_geometry`).

**Refactor watch:**
- UC1 trigger detection itself will likely change (mentioned during
  Scenario 3 Condition 4 split work)
- Multi-TF will become more general — multiple use cases beyond UC1, with
  different trigger conditions

---

## Probes (`compute_structure*` family)

**Today:**
- `compute_structure` — Scenario 1 (auto-identify) + Exception 2 per reversal
- `compute_structure_from_start` — caller-provided start + Exception 2 per reversal
- `compute_structure_scenario_3` — Phase 1 BOS_0 probe (+ optional Phase 2 continuation)
  - Phase 1 Condition 4 split: `end_idx` defined → finalized; `end_idx=None` → pending
  - Phase 2 currently never invoked in production (only in tests)
- All three share Exception 2 mechanics: bounded probe, `end_idx`, max 10 iters,
  exception triggered → discard + outer-loop unbounded run; no exception → keep first probe
- All three accept `timeframe` for proximity threshold lookup

**Refactor watch:**
- The three-function structure may consolidate
- `_run_h1_reverse_probe` may change scope (fewer/more events as triggering
  conditions)

---

## Imbalance instance model

**Today's invariants:**
- Per-candle `is_imbalance` flag for charting — it marks the **c2** (the
  pattern's middle candle): a rendering location, not the moment the gap exists
- `df.attrs["imbalances"]` list of `ImbalanceInstance` for Fib/POI logic
- Detection requires c2 direction matches FVG direction
- Consecutive same-direction imbalance candles merge into one instance
- Merged bounds: first c1 to last c3
- An instance's `start_idx` / `end_idx` are its first / last c2 (corrected
  2026-09-24: this bullet used to read "`imb_idx` references 'last candle of
  run' (representative)" — no such field; `imb_idx` was the c2 argument of the
  pre-instance helpers, removed in `9ecce9d`)
- **An instance exists from its first c3** (Plan F, 2026-09-24):
  `ImbalanceInstance.formed_at = start_idx + 1`. A question asked at a moment
  counts only instances formed by then, and only their formed prefix
  (`has_unfilled_imbalance(..., evaluated_at=)` — keyword-only and required;
  `None` = an explicit no-cut). Canonical: `IMBALANCE_FILL_SEMANTICS.md`
  "Knowability — the c3 rule"
- M15 lower-TF re-runs `compute_imbalance` after slicing

**Refactor watch:**
- Live trading mode will need incremental detection; current is one-pass
  (a live engine creates each instance at `formed_at` and grows it while the
  run continues)

---

## Charting

**Today:**
- All styling in `style_registry.py`
- H1 chart: `export_plotly.py`; M15 chart: `export_m15_chart.py`
- M15 chart styling: royalblue M15 elements, black H1 overlay, dashed M15 wave candles, solid H1 overlay
- Open-ended elements rendered to chart end unless capped
- Chart is the primary debugger

**Refactor watch:**
- Likely additions for multi-TF / strategy visualization

---

## Behavioral checkpoints (replay outputs)

Last clean replay outputs: `artifacts/commits/20260504_130926_c1cfd00/`
- 5 H1 WVMI records (3 locked), 5 first_counter triggers, 5 first_counter results
- M15 chart traces: 334, shapes: 256
- H1 chart traces: 149
- Use `/compare` against this snapshot during refactor to detect regressions

---

## What's intentionally being changed by the refactor (per user)

These are not invariants to preserve — they're the explicit goals:
- Pipeline / strategy / multi-TF restructuring
- WVMI tied to specific market structures (not ad hoc)
- Different market structures across different TFs unified under one model
- UC1 probe will get major changes (deferred from earlier sessions)

---

**Delete this file after the refactor lands and the new architecture's
docs are consolidated.**
