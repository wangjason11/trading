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

**`ev.idx` semantics — index conventions per event type (CRITICAL):**

| Event | `ev.idx` means | Notes |
|---|---|---|
| `CTS_ESTABLISHED` | confirmation candle | extreme is in `meta["anchor_idx"]` for pattern context |
| `CTS_UPDATED` | confirmation candle | |
| `CTS_CONFIRMED` | confirmation candle | extreme in `meta["cts_anchor_idx"]`; `confirmed_at` mirrors `ev.idx`; **new:** `meta["confirmation_method"]` |
| `CTS_RECONFIRMED` (new) | pullback confirmation candle | only fires after proximity-confirmed CTS |
| `BOS_CONFIRMED` | **BOS extreme candle** (NOT confirmation) | `meta["confirmed_at"]` = confirmation candle |
| `REVERSAL_CANDIDATE` | anchor candle | `meta["apply_idx"]` = effective application |
| `RANGE_*` | varies — see GOTCHAS for sort-order rules | |

**Event ordering invariant:** `sorted_events` uses `(idx, type)` (alphabetical
type tiebreak). Mode C M15 phase gate depends on this — see LANDMINES.

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
- Three modes: `"h1"` (Scenario 1/2/3), `"m15_reverse"` (cross-fib state machine),
  and the M15 reverse phase model (pre_established / established / confirmed)
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
- Default thresholds: H1=20, M15=10, M5=5 pips; caller-overridable
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
- Fib mode `"m15_reverse"` on M15
- Lifecycle bounded by parent H1 cycle (`lifecycle_end_idx`)
- Open-ended zones / POIs / fibs capped at lifecycle end with `deactivated_by="lifecycle_end"`
- M15 events / zones carry attribution: `timeframe`, `use_case`, `parent_tf`, `parent_sid`, `parent_cycle_id`
- M15 slice includes 50-candle lookback buffer; `compute_imbalance` re-run after slice

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
- Per-candle `is_imbalance` flag for charting
- `df.attrs["imbalances"]` list of `ImbalanceInstance` for Fib/POI logic
- Detection requires c2 direction matches FVG direction
- Consecutive same-direction imbalance candles merge into one instance
- Merged bounds: first c1 to last c3
- `imb_idx` references "last candle of run" (representative)
- M15 lower-TF re-runs `compute_imbalance` after slicing

**Refactor watch:**
- Live trading mode will need incremental detection; current is one-pass

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

Last clean replay outputs: `artifacts/commits/20260428_182602_b5b76be/`
- 6 H1 WVMI records, 6 UC1 triggers, 5 successful UC1 results
- M15 chart traces: 334
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
